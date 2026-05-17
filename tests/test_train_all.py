import json
from pathlib import Path

import pandas as pd

from scripts.train_all import run_heuristic_model


def _write_processed_fixture(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)

    interactions = pd.DataFrame(
        [
            {"user_idx": 1, "item_idx": 1, "created_at": "2024-01-01 00:00:00", "confidence": 1.0},
            {"user_idx": 1, "item_idx": 2, "created_at": "2024-01-02 00:00:00", "confidence": 1.0},
            {"user_idx": 2, "item_idx": 2, "created_at": "2024-01-01 00:00:00", "confidence": 1.0},
            {"user_idx": 2, "item_idx": 3, "created_at": "2024-01-02 00:00:00", "confidence": 1.0},
        ]
    )
    interactions.to_csv(data_dir / "interactions.csv", index=False)

    val = pd.DataFrame(
        [
            {"user_idx": 1, "train_seq": "[1]", "target": 2},
            {"user_idx": 2, "train_seq": "[2]", "target": 3},
        ]
    )
    val.to_csv(data_dir / "val.csv", index=False)
    val.to_csv(data_dir / "test.csv", index=False)

    stats = {"n_users": 2, "n_items": 3}
    with open(data_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f)

    return stats


def test_run_heuristic_model_popularity_writes_experiment_artifact(tmp_path):
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "experiments"
    stats = _write_processed_fixture(data_dir)

    run_heuristic_model(
        "popularity",
        data_dir,
        stats,
        str(output_dir),
        model_kwargs={},
        train_kwargs={"batch_size": 2, "max_len": 5},
        seed=42,
    )

    assert (output_dir / "popularity" / "popularity_model.json").exists()


def test_run_heuristic_model_itemcf_writes_experiment_artifacts(tmp_path):
    data_dir = tmp_path / "processed"
    output_dir = tmp_path / "experiments"
    stats = _write_processed_fixture(data_dir)

    run_heuristic_model(
        "itemcf",
        data_dir,
        stats,
        str(output_dir),
        model_kwargs={"top_k_sim": 2},
        train_kwargs={"batch_size": 2, "max_len": 5},
        seed=42,
    )

    assert (output_dir / "itemcf" / "itemcf_sim.json").exists()
    assert (output_dir / "itemcf" / "itemcf_history.json").exists()
