"""
itemcf.py
=========
Item-based Collaborative Filtering.

Idea:
  1. Calculate similarity between items based on user co-occurrence
  2. Score(user, item) = sum of sim(item, item_j) with item_j in user history
  3. Recommend top-K items with highest score

Similarity metric: Cosine similarity on user-item matrix (sparse).
"""

import ast
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

from evaluate import compute_metrics, print_results


class ItemCFRecommender:
    """
    Item-based Collaborative Filtering.

    Attributes:
        sim_matrix   : sparse matrix [n_items+1, n_items+1] — item-item similarity
        user_history : dict {user_idx: list of item_idx}
        n_items      : total number of items
    """

    def __init__(self, top_k_sim=20):
        """
        Args:
            top_k_sim: Only keep top-K similar items for each item
                       → reduce memory, increase inference speed
        """
        self.top_k_sim    = top_k_sim
        self.sim_matrix   = None    # dict {item_i: {item_j: sim}}
        self.user_history = {}
        self.n_items      = 0

    def fit(self, interactions_csv, stats_path="data/processed/dataset_stats.json"):
        """
        Calculate item-item cosine similarity from user-item matrix.

        Step 1: Build user-item matrix (sparse)
        Step 2: Calculate item-item cosine similarity
        Step 3: For each item, only keep top_k_sim similar items

        Args:
            interactions_csv: path to interactions.csv
            stats_path       : path to dataset_stats.json
        """
        with open(stats_path) as f:
            stats = json.load(f)
        self.n_items = stats["n_items"]
        n_users      = stats["n_users"]

        df = pd.read_csv(interactions_csv)

        # Build user 
        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            iid = int(row["item_idx"])
            if uid not in self.user_history:
                self.user_history[uid] = []
            self.user_history[uid].append(iid)

        print(f"Loaded {len(df):,} interactions | {n_users:,} users | {self.n_items:,} items")

        # Build sparse user-item matrix
        # Shape: [n_users+1, n_items+1], value = 1 if user has seen the item
        rows = df["user_idx"].tolist()
        cols = df["item_idx"].tolist()
        data = [1.0] * len(rows)

        user_item = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users + 1, self.n_items + 1),
            dtype=np.float32
        )
        print(f"   User-item matrix: {user_item.shape}, nnz={user_item.nnz:,}")

        # Calculate item-item cosine similarity
        # item_item = (user_item.T @ user_item) → [n_items+1, n_items+1]
        # Cosine = normalize each item vector before multiplying

        # Normalize: divide each item vector by its norm
        item_user = user_item.T.tocsr()           # [n_items+1, n_users+1]
        norms      = np.sqrt(item_user.power(2).sum(axis=1)).A1  # [n_items+1]
        norms[norms == 0] = 1e-10                 # avoid division by zero

        # Normalize item vectors
        norm_diag   = sp.diags(1.0 / norms)
        item_normed = norm_diag @ item_user       # [n_items+1, n_users+1]

        # Cosine similarity matrix: [n_items+1, n_items+1]
        print("   Calculating cosine similarity matrix...")
        sim_sparse = item_normed @ item_normed.T  # sparse matrix mult
        sim_sparse = sim_sparse.tocsr()

        # Only keep top_k_sim similar items per item
        print(f"   Keeping top-{self.top_k_sim} similar items per item...")
        self.sim_matrix = {}

        for i in tqdm(range(1, self.n_items + 1), desc="Building sim dict"):
            row    = sim_sparse.getrow(i).toarray().flatten()
            row[i] = 0.0               # remove self-similarity

            # Get top_k_sim indices with highest similarity
            top_idx = np.argsort(row)[::-1][:self.top_k_sim]
            self.sim_matrix[i] = {
                int(j): float(row[j])
                for j in top_idx if row[j] > 0
            }

        n_pairs = sum(len(v) for v in self.sim_matrix.values())
        print(f"Fit complete: {n_pairs:,} item-item similarity pairs")

    # Score
    def score_candidates(self, user_idx, candidates):
        """
        Calculate score for list of candidates based on user history.

        score(item) = sum của sim(item, item_j) with item_j in history

        Args:
            user_idx  : int
            candidates: list of item_idx [n_candidates]
        Returns:
            np.array [n_candidates]
        """
        history = self.user_history.get(user_idx, [])
        scores  = []

        for c in candidates:
            c   = int(c)
            sim = self.sim_matrix.get(c, {})
            sc  = sum(sim.get(h, 0.0) for h in history)
            scores.append(sc)

        return np.array(scores, dtype=np.float32)

    # Evaluate
    def evaluate(self, eval_csv, k_list=(10, 20), num_neg=99):
        """
        Evaluate on val.csv or test.csv.

        Args:
            eval_csv: path to val.csv hoặc test.csv
        """
        df         = pd.read_csv(eval_csv)
        all_scores = []
        all_items  = list(range(1, self.n_items + 1))

        # Build user history from eval file to avoid duplicate negatives
        eval_history = {}
        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            seq = _parse_seq(row["train_seq"])
            tgt = int(row["target"])
            eval_history[uid] = set(seq) | {tgt}

        for _, row in df.iterrows():
            uid  = int(row["user_idx"])
            tgt  = int(row["target"])
            seen = eval_history.get(uid, set())

            # Sample num_neg negatives
            neg_pool  = [i for i in all_items if i not in seen]
            neg_items = np.random.choice(
                neg_pool,
                size=min(num_neg, len(neg_pool)),
                replace=False
            ).tolist()

            candidates = [tgt] + neg_items
            scores     = self.score_candidates(uid, candidates)
            all_scores.append(scores)

        return compute_metrics(all_scores, k_list)

    # Recommend
    def recommend(self, user_idx, top_k=10):
        """
        Returns top-K items with highest score for user,
        filtering out items user has seen.
        """
        history  = set(self.user_history.get(user_idx, []))
        all_items = [i for i in range(1, self.n_items + 1) if i not in history]

        scores = self.score_candidates(user_idx, all_items)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [all_items[i] for i in top_idx]

    # Save / Load
    def save(self, save_dir="data/processed"):
        save_dir = Path(save_dir)

        # Save sim_matrix
        sim_path = save_dir / "itemcf_sim.json"
        with open(sim_path, "w") as f:
            json.dump({str(k): v for k, v in self.sim_matrix.items()}, f)

        # Save user_history
        hist_path = save_dir / "itemcf_history.json"
        with open(hist_path, "w") as f:
            json.dump({str(k): v for k, v in self.user_history.items()}, f)

        print(f"Model saved → {save_dir}/itemcf_*.json")

    def load(self, save_dir="data/processed"):
        save_dir = Path(save_dir)

        with open(save_dir / "itemcf_sim.json") as f:
            raw = json.load(f)
        self.sim_matrix = {
            int(k): {int(j): v for j, v in neighbors.items()}
            for k, neighbors in raw.items()
        }

        with open(save_dir / "itemcf_history.json") as f:
            raw = json.load(f)
        self.user_history = {int(k): v for k, v in raw.items()}

        self.n_items = max(self.sim_matrix.keys())
        print(f"Model loaded <- {save_dir}/itemcf_*.json")


# Helper 
def _parse_seq(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s))



# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--top_k_sim",  default=20,  type=int,
                        help="Number of similar items to keep per item")
    parser.add_argument("--num_neg",    default=99,  type=int)
    parser.add_argument("--seed",       default=42,  type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)

    print("\n" + "="*45)
    print("  Item-based Collaborative Filtering")
    print("="*45)

    model = ItemCFRecommender(top_k_sim=args.top_k_sim)
    model.fit(
        data_dir / "interactions.csv",
        stats_path=data_dir / "dataset_stats.json"
    )

    print("\nEvaluating on Val set...")
    val_results = model.evaluate(data_dir / "val.csv",
                                  k_list=(10, 20), num_neg=args.num_neg)
    print_results("Item-CF", val_results, phase="Val")

    print("\nEvaluating on Test set...")
    test_results = model.evaluate(data_dir / "test.csv",
                                   k_list=(10, 20), num_neg=args.num_neg)
    print_results("Item-CF", test_results, phase="Test")

    model.save(data_dir)
    print("\nDone!")