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


def _parse_seq(s):
    if isinstance(s, list):
        return s
    text = str(s).strip()
    if text.startswith("["):
        return ast.literal_eval(text)
    return [int(token) for token in text.split()] if text else []

class PopularityRecommender:
    def __init__(self):
        self.item_counts = {}
        self.sorted_items = []

    def fit(self, interactions_csv):
        df = pd.read_csv(interactions_csv)
        if "item_idx" in df.columns:
            item_ids = df["item_idx"].tolist()
        elif "item_sequence" in df.columns:
            item_ids = []
            for seq in df["item_sequence"]:
                item_ids.extend(_parse_seq(seq))
        else:
            raise ValueError("Expected either 'item_idx' or 'item_sequence' column")
        self.item_counts = Counter(item_ids)
        self.sorted_items = [item for item, _ in self.item_counts.most_common()]
        print(f"Fit complete: {len(self.item_counts):,} items")
        print(f"   Top 5: {self.sorted_items[:5]}")

    def score_candidates(self, candidates):
        return np.array([
            self.item_counts.get(int(c), 0)
            for c in candidates
        ], dtype=np.float32)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "Use pipeline.metrics.evaluate_popularity() for full-sort evaluation."
        )

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
