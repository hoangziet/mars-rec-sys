"""
pipeline/metrics.py
===================
Full-sort evaluation metrics shared for all 7 models.

Protocol:
    - Rank target item against the full item catalog.
    - Items in the user's training history are masked to -inf before ranking.
    - Report Recall@K and NDCG@K for K in {10, 20}.
"""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm


def _should_use_tqdm() -> bool:
    if os.environ.get("DISABLE_TQDM") == "1":
        return False
    return sys.stderr.isatty()


def _progress(iterable, desc: str):
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        disable=not _should_use_tqdm(),
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{desc:<18} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_metrics_from_ranks(ranks: list[int], k_list: tuple = (10, 20)) -> dict:
    """Compute Recall@K and NDCG@K from a list of 1-indexed target ranks.

    Parameters
    ----------
    ranks:
        List of 1-indexed ranks (1 = perfect, catalog_size = worst).
    k_list:
        Cutoff values to evaluate at.
    """
    results = {}
    for k in k_list:
        hrs = [1.0 if r <= k else 0.0 for r in ranks]
        ndcgs = [1.0 / np.log2(r + 1) if r <= k else 0.0 for r in ranks]
        results[f"Recall@{k}"] = float(np.mean(hrs))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs))
    return results


def _ranks_from_logits(
    logits: torch.Tensor,
    history_mask: torch.Tensor,
    target: torch.Tensor,
) -> list[int]:
    """Mask seen items, then compute conservative 1-indexed target ranks.

    Parameters
    ----------
    logits:       (B, n_items+1)
    history_mask: (B, n_items+1) bool — True = mask to -inf
    target:       (B,) item indices

    Tie handling is conservative: items with the same score as the target are
    counted ahead of it. This avoids optimistic metrics for heuristic models
    (e.g. popularity, item-CF) where many items often share identical scores.
    """
    logits = logits.masked_fill(history_mask, float("-inf"))
    target_scores = logits.gather(1, target.unsqueeze(1))   # (B, 1)
    ranks = (logits >= target_scores).sum(dim=1)            # (B,) 1-indexed
    return ranks.cpu().tolist()


# ---------------------------------------------------------------------------
# Evaluate functions by model type
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sequential(
    model,
    eval_loader,
    device,
    k_list: tuple = (10, 20),
) -> dict:
    """Full-sort evaluation for SASRec, gSASRec, GRU4Rec.

    Model signature: ``model(input_seq) -> (B, n_items+1)`` logits.
    """
    model.eval()
    all_ranks: list[int] = []

    for batch in _progress(eval_loader, desc="eval sequential"):
        input_seq    = batch["input_seq"].to(device)     # (B, L)
        history_mask = batch["history_mask"].to(device)  # (B, n_items+1)
        target       = batch["target"].to(device)        # (B,)

        logits = model(input_seq)                        # (B, n_items+1)
        all_ranks.extend(_ranks_from_logits(logits, history_mask, target))

    return compute_metrics_from_ranks(all_ranks, k_list)


@torch.no_grad()
def evaluate_bert4rec(
    model,
    eval_loader,
    device,
    k_list: tuple = (10, 20),
) -> dict:
    """Full-sort evaluation for BERT4Rec.

    Appends [MASK] to the end of each input sequence so the model predicts at
    the last position, matching the training objective.
    """
    model.eval()
    all_ranks: list[int] = []
    mask_token = model.mask_token

    for batch in _progress(eval_loader, desc="eval bert4rec"):
        input_seq    = batch["input_seq"].to(device)     # (B, L)
        history_mask = batch["history_mask"].to(device)  # (B, n_items+1)
        target       = batch["target"].to(device)        # (B,)

        # Shift left by 1 and append [MASK] at position -1.
        mask_col  = torch.full(
            (input_seq.size(0), 1), mask_token, dtype=torch.long, device=device
        )
        input_seq = torch.cat([input_seq[:, 1:], mask_col], dim=1)

        logits      = model(input_seq)             # (B, L, vocab_size=n_items+2)
        # Slice to n_items+1 to exclude the mask_token column (index n_items+1).
        # history_mask has shape (B, n_items+1) so dimensions must match.
        last_logits = logits[:, -1, :model.n_items + 1]   # (B, n_items+1)

        all_ranks.extend(_ranks_from_logits(last_logits, history_mask, target))

    return compute_metrics_from_ranks(all_ranks, k_list)


@torch.no_grad()
def evaluate_bprmf(
    model,
    eval_loader,
    device,
    k_list: tuple = (10, 20),
) -> dict:
    """Full-sort evaluation for BPR-MF.

    Scores all items via ``user_emb @ all_item_emb.T``.
    """
    model.eval()
    all_ranks: list[int] = []

    n_items      = model.n_items
    all_item_ids = torch.arange(n_items + 1, device=device)
    all_item_emb = model.item_embedding(all_item_ids)  # (n_items+1, D)

    for batch in _progress(eval_loader, desc="eval bprmf"):
        users        = batch["user"].to(device)          # (B,)
        history_mask = batch["history_mask"].to(device)  # (B, n_items+1)
        target       = batch["target"].to(device)        # (B,)

        user_emb = model.user_embedding(users)           # (B, D)
        logits   = user_emb @ all_item_emb.T             # (B, n_items+1)

        all_ranks.extend(_ranks_from_logits(logits, history_mask, target))

    return compute_metrics_from_ranks(all_ranks, k_list)


def evaluate_popularity(
    item_scores,
    eval_loader,
    k_list: tuple = (10, 20),
) -> dict:
    """Full-sort evaluation for the Popularity baseline.

    Parameters
    ----------
    item_scores:
        dict {item_idx: count} or ndarray indexed by item id.
    """
    use_mapping = hasattr(item_scores, "get")
    if use_mapping:
        max_id      = max(item_scores.keys()) + 1
        scores_base = np.zeros(max_id, dtype=np.float32)
        for k, v in item_scores.items():
            scores_base[k] = float(v)
    else:
        scores_base = np.array(item_scores, dtype=np.float32)

    all_ranks: list[int] = []

    for batch in eval_loader:
        history_mask = batch["history_mask"].numpy()  # (B, n_items+1)
        target       = batch["target"].numpy()        # (B,)

        for i in range(len(target)):
            scores = scores_base.copy()
            scores[history_mask[i]] = -np.inf
            tgt_score = scores[target[i]]
            rank = int((scores >= tgt_score).sum())
            all_ranks.append(rank)

    return compute_metrics_from_ranks(all_ranks, k_list)


def evaluate_itemcf(
    sim_matrix: dict,
    user_history: dict,
    eval_loader,
    k_list: tuple = (10, 20),
) -> dict:
    """Full-sort evaluation for Item-based CF.

    score(user, item) = sum of sim(item, h) for h in user history.
    """
    all_ranks: list[int] = []

    for batch in eval_loader:
        users        = batch["user"].numpy()          # (B,)
        history_mask = batch["history_mask"].numpy()  # (B, n_items+1)
        target       = batch["target"].numpy()        # (B,)
        n_vocab      = history_mask.shape[1]

        for i, uid in enumerate(users):
            history = user_history.get(int(uid), [])
            scores  = np.array(
                [
                    sum(sim_matrix.get(item_id, {}).get(h, 0.0) for h in history)
                    for item_id in range(n_vocab)
                ],
                dtype=np.float32,
            )
            scores[history_mask[i]] = -np.inf
            tgt_score = scores[target[i]]
            rank = int((scores >= tgt_score).sum())
            all_ranks.append(rank)

    return compute_metrics_from_ranks(all_ranks, k_list)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_results(model_name: str, results: dict, phase: str = "Test") -> None:
    print(f"\n{'─' * 45}")
    print(f"  {model_name} | {phase}")
    print(f"{'─' * 45}")
    for metric, value in results.items():
        print(f"  {metric:<12} : {value:.4f}")
    print(f"{'─' * 45}")


def compare_models(results_dict: dict, k_list: tuple = (10, 20)) -> None:
    """Print a comparison table for all models."""
    metrics = [f"Recall@{k}" for k in k_list] + [f"NDCG@{k}" for k in k_list]
    col_w   = 12
    header  = f"{'Model':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)

    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))

    for model_name, res in results_dict.items():
        row = f"{model_name:<20}" + "".join(
            f"{res.get(m, 0.0):>{col_w}.4f}" for m in metrics
        )
        print(row)

    print("═" * len(header))
    print("\n  * Best per metric:")
    for m in metrics:
        best_model = max(results_dict, key=lambda x: results_dict[x].get(m, 0))
        best_val   = results_dict[best_model].get(m, 0)
        print(f"    {m:<12}: {best_model} ({best_val:.4f})")
