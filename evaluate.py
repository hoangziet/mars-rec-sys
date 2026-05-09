"""
evaluate.py
===========
Evaluation metrics shared for all 7 models.

Metrics:
  - HR@K   (Hit Rate)  : is the target item in top-K
  - NDCG@K             : calculates the rank of target in top-K

Standard protocol:
  - 1 positive target + 99 random negatives = 100 candidates
  - Model scores 100 candidates -> ranks -> calculates HR and NDCG
"""

import numpy as np
import torch
from tqdm import tqdm



# Core metric functions

def hit_rate_at_k(scores, k):
    """
    HR@K: is target (index 0) in top-K.

    Args:
        scores : np.array shape [n_candidates] — score of each candidate
                 candidates[0] is always the positive target
        k      : cutoff
    Returns:
        1.0 if target is in top-K, 0.0 otherwise
    """
    top_k_indices = np.argsort(scores)[::-1][:k]
    return 1.0 if 0 in top_k_indices else 0.0


def ndcg_at_k(scores, k):
    """
    NDCG@K: if target is in top-K, calculate 1 / log2(rank + 1).
    Rank starts from 1.

    Args:
        scores : np.array shape [n_candidates]
        k      : cutoff
    Returns:
        NDCG score (0.0 -> 1.0)
    """
    top_k_indices = np.argsort(scores)[::-1][:k]
    if 0 in top_k_indices:
        rank = np.where(top_k_indices == 0)[0][0] + 1   # rank starts from 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def compute_metrics(all_scores, k_list=(10, 20)):
    """
    Calculate HR@K and NDCG@K for the entire eval set.

    Args:
        all_scores : list of np.array, each element is the score of 100 candidates
                     candidates[0] is always the positive target
        k_list     : list of K to compute, default [10, 20]

    Returns:
        dict {"HR@10": float, "NDCG@10": float, "HR@20": float, "NDCG@20": float}
    """
    results = {}
    for k in k_list:
        hrs   = [hit_rate_at_k(s, k) for s in all_scores]
        ndcgs = [ndcg_at_k(s, k)     for s in all_scores]
        results[f"HR@{k}"]   = round(float(np.mean(hrs)),   4)
        results[f"NDCG@{k}"] = round(float(np.mean(ndcgs)), 4)
    return results



# Evaluate functions by model type

@torch.no_grad()
def evaluate_sequential(model, eval_loader, device, k_list=(10, 20)):
    """
    Used for: GRU4Rec, SASRec, gSASRec, BERT4Rec.

    Model takes input_seq -> returns scores for all items.
    We only take the score of 100 candidates.

    Args:
        model       : trained PyTorch model
        eval_loader : DataLoader from get_eval_loader()
        device      : "cuda" or "cpu"
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
        input_seq  = batch["input_seq"].to(device)     # [B, max_len]
        candidates = batch["candidates"].to(device)     # [B, 100]
        mask       = batch["mask"].to(device)           # [B, max_len]

        # Forward — model returns logits for the entire item vocab
        # Shape: [B, n_items+1] or model returns embedding -> calculates score separately
        logits = model(input_seq, mask=mask)             # [B, n_items+1]

        # Get scores of 100 candidates for each sample in batch
        for i in range(input_seq.size(0)):
            cand_scores = logits[i][candidates[i]]       # [100]
            all_scores.append(cand_scores.cpu().numpy())

    return compute_metrics(all_scores, k_list)


@torch.no_grad()
def evaluate_bert4rec(model, eval_loader, device, k_list=(10, 20)):
    """
    BERT4Rec predicts at the last MASK position.
    Input sequence already has mask_token at the end (MaskedSequenceDataset is_train=False).
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating BERT4Rec", leave=False):
        input_seq  = batch["input_seq"].to(device)      # [B, max_len]
        candidates = batch["candidates"].to(device)      # [B, 100]

        # Model returns logits at each position: [B, max_len, n_items+2]
        logits = model(input_seq)                        # [B, max_len, vocab]

        # Get output at the last position (mask position)
        last_logits = logits[:, -1, :]                   # [B, vocab]

        for i in range(input_seq.size(0)):
            cand_scores = last_logits[i][candidates[i]]  # [100]
            all_scores.append(cand_scores.cpu().numpy())

    return compute_metrics(all_scores, k_list)


@torch.no_grad()
def evaluate_bprmf(model, eval_loader, device, k_list=(10, 20)):
    """
    BPR-MF calculates score via dot product: user_emb · item_emb.
    """
    model.eval()
    all_scores = []

    for batch in tqdm(eval_loader, desc="Evaluating BPR-MF", leave=False):
        users      = batch["user"].to(device)           # [B]
        candidates = batch["candidates"].to(device)     # [B, 100]

        # Get user embedding
        user_emb = model.user_embedding(users)           # [B, dim]

        # Get item embeddings for 100 candidates
        item_emb = model.item_embedding(candidates)      # [B, 100, dim]

        # Score = dot product
        scores = torch.bmm(
            item_emb,                                    # [B, 100, dim]
            user_emb.unsqueeze(-1)                       # [B, dim, 1]
        ).squeeze(-1)                                    # [B, 100]

        for i in range(users.size(0)):
            all_scores.append(scores[i].cpu().numpy())

    return compute_metrics(all_scores, k_list)


def evaluate_popularity(item_scores, eval_loader, k_list=(10, 20)):
    """
    Popularity does not need model forward.
    item_scores: dict {item_idx: count} or ndarray indexed by item id.

    For each user, rank 100 candidates by popularity.
    """
    all_scores = []
    use_mapping = hasattr(item_scores, "get")

    for batch in eval_loader:
        candidates = batch["candidates"].numpy()         # [B, 100]

        for i in range(candidates.shape[0]):
            scores = np.array([
                item_scores.get(int(c), 0)
                if use_mapping else (item_scores[int(c)] if 0 <= int(c) < len(item_scores) else 0)
                for c in candidates[i]
            ], dtype=np.float32)
            all_scores.append(scores)

    return compute_metrics(all_scores, k_list)


def evaluate_itemcf(sim_matrix, user_history, eval_loader, k_list=(10, 20)):
    """
    Item-based CF: score(user, item) = sum of sim(item, item_j)
    with item_j in user's history.

    sim_matrix   : dict hoặc sparse matrix {item_i: {item_j: sim}}
    user_history : dict {user_idx: [list of item_idx]}
    """
    all_scores = []

    for batch in eval_loader:
        users      = batch["user"].numpy()               # [B]
        candidates = batch["candidates"].numpy()         # [B, 100]

        for i, uid in enumerate(users):
            history = user_history.get(int(uid), [])
            scores  = []
            for c in candidates[i]:
                c = int(c)
                # Score = sum of similarities with all items in history
                sc = sum(
                    sim_matrix.get(c, {}).get(h, 0.0)
                    for h in history
                )
                scores.append(sc)
            all_scores.append(np.array(scores, dtype=np.float32))

    return compute_metrics(all_scores, k_list)



# Pretty print results

def print_results(model_name, results, phase="Test"):
    """Print formatted results to console."""
    print(f"\n{'─'*45}")
    print(f"  {model_name} | {phase}")
    print(f"{'─'*45}")
    for metric, value in results.items():
        print(f"  {metric:<12} : {value:.4f}")
    print(f"{'─'*45}")


def compare_models(results_dict, k_list=(10, 20)):
    """
    Print comparison table for all models.

    Args:
        results_dict: {"ModelName": {"HR@10": ..., "NDCG@10": ..., ...}}
    """
    metrics = [f"HR@{k}" for k in k_list] + [f"NDCG@{k}" for k in k_list]

    # Header
    col_w = 12
    header = f"{'Model':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "═"*len(header))
    print(header)
    print("═"*len(header))

    # Rows
    for model_name, res in results_dict.items():
        row = f"{model_name:<20}"
        for m in metrics:
            row += f"{res.get(m, 0.0):>{col_w}.4f}"
        print(row)

    print("═"*len(header))

    # Highlight best
    print("\n  * Best values per metric:")
    for m in metrics:
        best_model = max(results_dict, key=lambda x: results_dict[x].get(m, 0))
        best_val   = results_dict[best_model].get(m, 0)
        print(f"    {m:<12}: {best_model} ({best_val:.4f})")



# Quick test

if __name__ == "__main__":
    print("-- Test metric functions --")

    # Mock scores for 5 users, 100 candidates each
    np.random.seed(42)
    n_users     = 5
    n_candidate = 100

    # Case 1: target (index 0) always have highest score -> HR = 1.0
    scores_good = []
    for _ in range(n_users):
        s    = np.random.randn(n_candidate)
        s[0] = 999.0     # highest score for target
        scores_good.append(s)

    r = compute_metrics(scores_good)
    print("Perfect model:", r)
    assert r["HR@10"] == 1.0 and r["NDCG@10"] == 1.0

    # Case 2: target is always at the last position -> HR = 0.0
    scores_bad = []
    for _ in range(n_users):
        s    = np.random.randn(n_candidate)
        s[0] = -999.0    # lowest score for target
        scores_bad.append(s)

    r = compute_metrics(scores_bad)
    print("Worst model  :", r)
    assert r["HR@10"] == 0.0 and r["NDCG@10"] == 0.0

    # Test compare_models
    fake_results = {
        "Popularity"  : {"HR@10": 0.12, "NDCG@10": 0.07, "HR@20": 0.18, "NDCG@20": 0.09},
        "Item-CF"     : {"HR@10": 0.15, "NDCG@10": 0.09, "HR@20": 0.22, "NDCG@20": 0.11},
        "BPR-MF"      : {"HR@10": 0.18, "NDCG@10": 0.11, "HR@20": 0.26, "NDCG@20": 0.13},
        "GRU4Rec"     : {"HR@10": 0.22, "NDCG@10": 0.14, "HR@20": 0.30, "NDCG@20": 0.16},
        "BERT4Rec"    : {"HR@10": 0.24, "NDCG@10": 0.15, "HR@20": 0.32, "NDCG@20": 0.17},
        "SASRec"      : {"HR@10": 0.26, "NDCG@10": 0.17, "HR@20": 0.34, "NDCG@20": 0.19},
        "gSASRec"     : {"HR@10": 0.29, "NDCG@10": 0.19, "HR@20": 0.37, "NDCG@20": 0.21},
    }
    compare_models(fake_results)
    print("\nevaluate.py test passed!")