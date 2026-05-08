"""
itemcf.py
=========
Item-based Collaborative Filtering.

Algorithm:
    1. Build user-item co-occurrence matrix
    2. Compute item-item cosine similarity
    3. Score(user, item) = sum(sim(item, history_item)) for each history_item
"""

import ast
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

from evaluate import compute_metrics


class ItemCFRecommender:
    def __init__(self, top_k_sim=20):
        self.top_k_sim = top_k_sim
        self.sim_matrix = None
        self.user_history = {}
        self.n_items = 0

    def fit(self, interactions_csv, stats_path="data/processed/dataset_stats.json"):
        with open(stats_path) as f:
            stats = json.load(f)
        self.n_items = stats["n_items"]
        n_users = stats["n_users"]

        df = pd.read_csv(interactions_csv)

        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            iid = int(row["item_idx"])
            if uid not in self.user_history:
                self.user_history[uid] = []
            self.user_history[uid].append(iid)

        print(f"Loaded {len(df):,} interactions | {n_users:,} users | {self.n_items:,} items")

        rows = df["user_idx"].tolist()
        cols = df["item_idx"].tolist()
        data = [1.0] * len(rows)

        user_item = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users + 1, self.n_items + 1),
            dtype=np.float32
        )
        print(f"   User-item matrix: {user_item.shape}, nnz={user_item.nnz:,}")

        item_user = user_item.T.tocsr()
        norms = np.sqrt(item_user.power(2).sum(axis=1)).A1
        norms[norms == 0] = 1e-10

        norm_diag = sp.diags(1.0 / norms)
        item_normed = norm_diag @ item_user

        print("   Calculating cosine similarity matrix...")
        sim_sparse = (item_normed @ item_normed.T).tocsr()

        print(f"   Keeping top-{self.top_k_sim} similar items per item...")
        self.sim_matrix = {}

        for i in tqdm(range(1, self.n_items + 1), desc="Building sim dict"):
            row = sim_sparse.getrow(i).toarray().flatten()
            row[i] = 0.0
            top_idx = np.argsort(row)[::-1][:self.top_k_sim]
            self.sim_matrix[i] = {
                int(j): float(row[j])
                for j in top_idx if row[j] > 0
            }

        n_pairs = sum(len(v) for v in self.sim_matrix.values())
        print(f"Fit complete: {n_pairs:,} item-item similarity pairs")

    def score_candidates(self, user_idx, candidates):
        history = self.user_history.get(user_idx, [])
        scores = []
        for c in candidates:
            c = int(c)
            sim = self.sim_matrix.get(c, {})
            sc = sum(sim.get(h, 0.0) for h in history)
            scores.append(sc)
        return np.array(scores, dtype=np.float32)

    def evaluate(self, eval_csv, k_list=(10, 20), num_neg=99):
        df = pd.read_csv(eval_csv)
        all_scores = []
        all_items = list(range(1, self.n_items + 1))

        eval_history = {}
        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            seq = _parse_seq(row["train_seq"])
            tgt = int(row["target"])
            eval_history[uid] = set(seq) | {tgt}

        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            tgt = int(row["target"])
            seen = eval_history.get(uid, set())

            neg_pool = [i for i in all_items if i not in seen]
            neg_items = np.random.choice(
                neg_pool,
                size=min(num_neg, len(neg_pool)),
                replace=False
            ).tolist()

            candidates = [tgt] + neg_items
            scores = self.score_candidates(uid, candidates)
            all_scores.append(scores)

        return compute_metrics(all_scores, k_list)

    def recommend(self, user_idx, top_k=10):
        history = set(self.user_history.get(user_idx, []))
        all_items = [i for i in range(1, self.n_items + 1) if i not in history]
        scores = self.score_candidates(user_idx, all_items)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [all_items[i] for i in top_idx]

    def save(self, save_dir="data/processed"):
        save_dir = Path(save_dir)
        with open(save_dir / "itemcf_sim.json", "w") as f:
            json.dump({str(k): v for k, v in self.sim_matrix.items()}, f)
        with open(save_dir / "itemcf_history.json", "w") as f:
            json.dump({str(k): v for k, v in self.user_history.items()}, f)

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


def _parse_seq(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s))
