"""
popularity.py
=============
Popularity-based recommender — simplest baseline.

Idea: Count frequency of each item in the training data
-> Recommend top-K most popular items to everyone.

No training needed, no user personalization.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from evaluate import compute_metrics, print_results, compare_models


class PopularityRecommender:
    """
    Popularity-based recommender.

    Attributes:
        item_counts : dict {item_idx: count} — frequency count
        sorted_items: list — items sorted by descending popularity
    """

    def __init__(self):
        self.item_counts  = {}
        self.sorted_items = []

    # Train 
    def fit(self, interactions_csv):
        """
        Count frequency of each item from interactions.csv.

        Args:
            interactions_csv: path to data/processed/interactions.csv
        """
        df = pd.read_csv(interactions_csv)
        self.item_counts = Counter(df["item_idx"].tolist())

        # Sort by count in descending order
        self.sorted_items = [
            item for item, _ in self.item_counts.most_common()
        ]
        print(f"Fit complete: {len(self.item_counts):,} items")
        print(f"   Top 5 popular items: {self.sorted_items[:5]}")
        print(f"   Most popular count : {self.item_counts[self.sorted_items[0]]:,}")

    # Score 

    def score_candidates(self, candidates):
        """
        Returns popularity score for the list of candidates.

        Args:
            candidates: list or np.array of item_idx [n_candidates]
        Returns:
            np.array scores [n_candidates]
        """
        return np.array([
            self.item_counts.get(int(c), 0)
            for c in candidates
        ], dtype=np.float32)

    # Evaluate 
    def evaluate(self, eval_csv, k_list=(10, 20), num_neg=99):
        """
        Evaluate on val.csv hoặc test.csv.

        Protocol: 1 positive + 99 random negatives = 100 candidates
        Rank by popularity → compute HR@K and NDCG@K.

        Args:
            eval_csv: path to val.csv hoặc test.csv
            k_list  : list of K to compute metrics
            num_neg : number of negative samples (default 99)
        """
        df = pd.read_csv(eval_csv)
        all_scores = []

        # Load user history to avoid sampling duplicate negatives
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

            # Sample num_neg negatives
            neg_pool = [i for i in all_item_ids if i not in seen]
            neg_items = np.random.choice(
                neg_pool,
                size=min(num_neg, len(neg_pool)),
                replace=False
            ).tolist()

            # candidates[0] = target (positive)
            candidates = [tgt] + neg_items
            scores     = self.score_candidates(candidates)
            all_scores.append(scores)

        return compute_metrics(all_scores, k_list)

    # Recommend 
    def recommend(self, user_history=None, top_k=10):
        """
        Returns top-K most popular items.
        If user_history is provided, filter out seen items.

        Args:
            user_history: set of item_idx the user has seen (optional)
            top_k       : number of items to recommend
        Returns:
            list of item_idx
        """
        if user_history is None:
            return self.sorted_items[:top_k]

        recs = []
        for item in self.sorted_items:
            if item not in user_history:
                recs.append(item)
            if len(recs) == top_k:
                break
        return recs

    # Save / Load 
    def save(self, path="data/processed/popularity_model.json"):
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self.item_counts.items()}, f)
        print(f"Model saved → {path}")

    def load(self, path="data/processed/popularity_model.json"):
        with open(path) as f:
            raw = json.load(f)
        self.item_counts  = {int(k): v for k, v in raw.items()}
        self.sorted_items = [
            item for item, _ in Counter(self.item_counts).most_common()
        ]
        print(f"Model loaded ← {path}")


# Helper
def _parse_seq(s):
    import ast
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s))



# Main — run directly for train + evaluate
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--num_neg",    default=99,   type=int)
    parser.add_argument("--seed",       default=42,   type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # Fit 
    print("\n" + "="*45)
    print("  Popularity Recommender")
    print("="*45)

    model = PopularityRecommender()
    model.fit(data_dir / "interactions.csv")

    # Validate 
    print("\nEvaluating on Val set...")
    val_results = model.evaluate(
        data_dir / "val.csv",
        k_list=(10, 20),
        num_neg=args.num_neg
    )
    print_results("Popularity", val_results, phase="Val")

    # Test 
    print("\nEvaluating on Test set...")
    test_results = model.evaluate(
        data_dir / "test.csv",
        k_list=(10, 20),
        num_neg=args.num_neg
    )
    print_results("Popularity", test_results, phase="Test")

    # Save 
    model.save(data_dir / "popularity_model.json")

    print("\nDone!")
    print("\nTest results (used to compare with other models):")
    print(test_results)