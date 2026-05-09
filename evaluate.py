"""
evaluate.py
===========
Evaluation metrics shared for all 7 models.

Metrics:
    - HR@K  (Hit Rate) : is the target item in top-K
    - NDCG@K           : rank-weighted score of target in top-K

Standard protocol:
    - 1 positive target + 99 random negatives = 100 candidates
    - Model scores 100 candidates -> rank -> compute HR and NDCG
"""

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def hit_rate_at_k(scores: np.ndarray, k: int) -> float:
    """HR@K: 1.0 if target (index 0) is in top-K ranked candidates."""
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if 0 in top_k else 0.0


def ndcg_at_k(scores: np.ndarray, k: int) -> float:
    """NDCG@K: discounted gain if target (index 0) appears in top-K."""
    top_k = np.argsort(scores)[::-1][:k]
    if 0 in top_k:
        rank = np.where(top_k == 0)[0][0] + 1  # 1-based rank
        return 1.0 / np.log2(rank + 1)
    return 0.0


def compute_metrics(all_scores: list[np.ndarray], k_list: tuple = (10, 20)) -> dict:
    """Aggregate HR@K and NDCG@K over the full evaluation set.

    Parameters
    ----------
    all_scores:
        List of 1-D arrays of length n_candidates.
        Index 0 of each array is always the positive target.
    k_list:
        Cutoff values to evaluate at.

    Returns
    -------
    dict with keys HR@K and NDCG@K for each K in k_list.
    """
    results = {}
    for k in k_list:
        hrs = [hit_rate_at_k(s, k) for s in all_scores]
        ndcgs = [ndcg_at_k(s, k) for s in all_scores]
        results[f"HR@{k}"] = round(float(np.mean(hrs)), 4)
        results[f"NDCG@{k}"] = round(float(np.mean(ndcgs)), 4)
    return results


# ---------------------------------------------------------------------------
# Evaluate functions by model type
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sequential(model, eval_loader, device, k_list: tuple = (10, 20)) -> dict:
    """Evaluation for SASRec, gSASRec, GRU4Rec.

    Model signature: model(input_seq) -> (B, n_items+1) score tensor.
    Candidates are scored by gathering from the full vocab output.
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
        input_seq = batch["input_seq"].to(device)  # (B, L)
        candidates = batch["candidates"].to(device)  # (B, 100)

        logits = model(input_seq)  # (B, n_items+1)

        for i in range(input_seq.size(0)):
            cand_scores = logits[i][candidates[i]]  # (100,)
            all_scores.append(cand_scores.cpu().numpy())

    return compute_metrics(all_scores, k_list)


@torch.no_grad()
def evaluate_bert4rec(model, eval_loader, device, k_list: tuple = (10, 20)) -> dict:
    """Evaluation for BERT4Rec.

    Input sequence already has mask_token appended at the last position
    (handled by MaskedSequenceDataset with is_train=False).
    Prediction is taken from the last position output.
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating BERT4Rec", leave=False):
        input_seq = batch["input_seq"].to(device)  # (B, L)
        candidates = batch["candidates"].to(device)  # (B, 100)

        logits = model(input_seq)  # (B, L, vocab)
        last_logits = logits[:, -1, :]  # (B, vocab)

        for i in range(input_seq.size(0)):
            cand_scores = last_logits[i][candidates[i]]
            all_scores.append(cand_scores.cpu().numpy())

    return compute_metrics(all_scores, k_list)


@torch.no_grad()
def evaluate_bprmf(model, eval_loader, device, k_list: tuple = (10, 20)) -> dict:
    """Evaluation for BPR-MF.

    Score = dot product of user embedding and candidate item embeddings.
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating BPR-MF", leave=False):
        users = batch["user"].to(device)  # (B,)
        candidates = batch["candidates"].to(device)  # (B, 100)

        user_emb = model.user_embedding(users)  # (B, D)
        item_emb = model.item_embedding(candidates)  # (B, 100, D)

        scores = torch.bmm(
            item_emb,  # (B, 100, D)
            user_emb.unsqueeze(-1),  # (B, D, 1)
        ).squeeze(-1)  # (B, 100)

        for i in range(users.size(0)):
            all_scores.append(scores[i].cpu().numpy())

    return compute_metrics(all_scores, k_list)


def evaluate_popularity(
    item_scores,
    eval_loader,
    k_list: tuple = (10, 20),
) -> dict:
    """Evaluation for Popularity baseline.

    Parameters
    ----------
    item_scores:
        dict {item_idx: count} or ndarray indexed by item id.
    """
    all_scores = []
    use_mapping = hasattr(item_scores, "get")

    for batch in eval_loader:
        candidates = batch["candidates"].numpy()  # (B, 100)
        for i in range(candidates.shape[0]):
            scores = np.array(
                [
                    item_scores.get(int(c), 0)
                    if use_mapping
                    else (item_scores[int(c)] if 0 <= int(c) < len(item_scores) else 0)
                    for c in candidates[i]
                ],
                dtype=np.float32,
            )
            all_scores.append(scores)

    return compute_metrics(all_scores, k_list)


def evaluate_itemcf(
    sim_matrix: dict,
    user_history: dict,
    eval_loader,
    k_list: tuple = (10, 20),
) -> dict:
    """Evaluation for Item-based CF.

    score(user, item) = sum of sim(item, h) for h in user history.

    Parameters
    ----------
    sim_matrix:
        Nested dict {item_i: {item_j: similarity}}.
    user_history:
        dict {user_idx: list[item_idx]}.
    """
    all_scores = []

    for batch in eval_loader:
        users = batch["user"].numpy()  # (B,)
        candidates = batch["candidates"].numpy()  # (B, 100)

        for i, uid in enumerate(users):
            history = user_history.get(int(uid), [])
            scores = [
                sum(sim_matrix.get(int(c), {}).get(h, 0.0) for h in history)
                for c in candidates[i]
            ]
            all_scores.append(np.array(scores, dtype=np.float32))

    return compute_metrics(all_scores, k_list)


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
    metrics = [f"HR@{k}" for k in k_list] + [f"NDCG@{k}" for k in k_list]
    col_w = 12
    header = f"{'Model':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)

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
        best_val = results_dict[best_model].get(m, 0)
        print(f"    {m:<12}: {best_model} ({best_val:.4f})")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("-- Smoke test metric functions --")
    np.random.seed(42)
    n_users, n_candidates = 5, 100

    scores_good = []
    for _ in range(n_users):
        s = np.random.randn(n_candidates)
        s[0] = 999.0
        scores_good.append(s)
    r = compute_metrics(scores_good)
    print("Perfect model:", r)
    assert r["HR@10"] == 1.0 and r["NDCG@10"] == 1.0

    scores_bad = []
    for _ in range(n_users):
        s = np.random.randn(n_candidates)
        s[0] = -999.0
        scores_bad.append(s)
    r = compute_metrics(scores_bad)
    print("Worst model  :", r)
    assert r["HR@10"] == 0.0 and r["NDCG@10"] == 0.0

    print("evaluate.py smoke test passed!")
