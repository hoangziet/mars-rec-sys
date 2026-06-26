import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_rq2_report_writes_best_watch_artifact():
    """Test the write_outputs function directly without MLflow."""
    from scripts.rq2_report import write_outputs, VARIANT_ORDER

    selected = [
        {"variant": "baseline", "seed": 42, "val_ndcg_at_10": 0.3000, "test_NDCG_at_10": 0.2900},
        {"variant": "baseline", "seed": 123, "val_ndcg_at_10": 0.3010, "test_NDCG_at_10": 0.2910},
        {"variant": "wlwe", "seed": 42, "val_ndcg_at_10": 0.3200, "test_NDCG_at_10": 0.3150},
        {"variant": "wlwe", "seed": 123, "val_ndcg_at_10": 0.3190, "test_NDCG_at_10": 0.3140},
    ]
    output_dir = Path("/tmp/test_rq2_output")
    output_dir.mkdir(exist_ok=True)
    alpha_artifact = {"best_alpha": 1.0, "backbone": "bert4rec"}
    write_outputs(
        selected_runs=selected,
        alpha_artifact=alpha_artifact,
        output_dir=output_dir,
        benchmark_id="rq2-watch-test",
    )
    winner = json.loads((output_dir / "rq2_best_watch.json").read_text())
    assert winner["best_variant"] == "wlwe"
    assert winner["best_alpha"] == 1.0
    assert winner["backbone"] == "bert4rec"


def test_rq2_compare_holm_correction():
    """Test the compare function with Holm correction."""
    from training.stat_tests import apply_holm_correction

    rows = [
        {"baseline": "wl", "raw_p_value": 0.01},
        {"baseline": "we", "raw_p_value": 0.03},
        {"baseline": "wlwe", "raw_p_value": 0.20},
    ]
    result = apply_holm_correction(rows, p_key="raw_p_value")
    assert result[0]["significant_after_holm"]  # p=0.01 should survive Holm
    assert not result[2]["significant_after_holm"]  # p=0.20 should not
