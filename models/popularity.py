"""
popularity.py
=============
Popularity-based recommender — simplest baseline.

Algorithm:
    Count item frequency in training data → recommend top-K most popular items.
"""

import ast
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from evaluate import compute_metrics


class PopularityRecommender:
    def __init__(self):
        self.item_counts = {}
        self.sorted_items = []

    def fit(self, interactions_csv):
        df = pd.read_csv(interactions_csv)
        self.item_counts = Counter(df["item_idx"].tolist())
        self.sorted_items = [item for item, _ in self.item_counts.most_common()]
        print(f"Fit complete: {len(self.item_counts):,} items")
        print(f"   Top 5: {self.sorted_items[:5]}")

    def score_candidates(self, candidates):
        return np.array([
            self.item_counts.get(int(c), 0)
            for c in candidates
        ], dtype=np.float32)

    def evaluate(self, eval_csv, k_list=(10, 20), num_neg=99):
        df = pd.read_csv(eval_csv)
        all_scores = []

        user_history = {}
        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            seq = _parse_seq(row["train_seq"])
            tgt = int(row["target"])
            user_history[uid] = set(seq) | {tgt}

        all_item_ids = list(self.item_counts.keys())

        for _, row in df.iterrows():
            uid = int(row["user_idx"])
            tgt = int(row["target"])
            seen = user_history.get(uid, set())

            neg_pool = [i for i in all_item_ids if i not in seen]
            neg_items = np.random.choice(
                neg_pool,
                size=min(num_neg, len(neg_pool)),
                replace=False
            ).tolist()

            candidates = [tgt] + neg_items
            scores = self.score_candidates(candidates)
            all_scores.append(scores)

        return compute_metrics(all_scores, k_list)

    def recommend(self, user_history=None, top_k=10):
        if user_history is None:
            return self.sorted_items[:top_k]
        recs = []
        for item in self.sorted_items:
            if item not in user_history:
                recs.append(item)
            if len(recs) == top_k:
                break
        return recs

    def save(self, path="data/processed/popularity_model.json"):
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self.item_counts.items()}, f)

    def load(self, path="data/processed/popularity_model.json"):
        with open(path) as f:
            raw = json.load(f)
        self.item_counts = {int(k): v for k, v in raw.items()}
        self.sorted_items = [
            item for item, _ in Counter(self.item_counts).most_common()
        ]


def _parse_seq(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s))
