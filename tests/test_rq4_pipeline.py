import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rq4_compare_rejects_user_missing_some_seeds():
    import pandas as pd
    from scripts import rq4_compare

    per_user = pd.DataFrame(
        [
            {"variant": "V0", "seed": 42, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.2},
            {"variant": "V0", "seed": 123, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.3},
            {"variant": "V1", "seed": 42, "user_idx": 1, "target_item": 10, "ndcg_at_10": 0.25},
            # V1 missing seed 123 for same user
        ]
    )
    with pytest.raises(RuntimeError, match="expected all users to have"):
        rq4_compare._join_by_user(per_user, "V1", "V0", "ndcg_at_10", expected_seed_count=2)
