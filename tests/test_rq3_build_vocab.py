import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq3_build_vocab import _get_train_item_idx


def test_get_train_item_idx_canonical_format(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "item_sequence,target_item\n"
        "1 2 3,4\n"
        "5 6 7,8\n"
        "1 9 10,11\n"
    )
    items = _get_train_item_idx(str(train_csv))
    # Targets are predicted, not fed to the model as input — only items in
    # `item_sequence` (the histories the model actually sees) belong in the
    # vocab-fit set.
    assert items == {1, 2, 3, 5, 6, 7, 9, 10}
