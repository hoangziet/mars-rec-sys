"""Tests for the explicit baseline_variant handling in RQ4 compare."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_compare


# ---- _resolve_baseline ----

def test_resolve_baseline_picks_explicit_value_from_manifest():
    manifest = {"baseline_variant": "V0"}
    variants = ["V1", "V0", "V3"]
    assert rq4_compare._resolve_baseline(manifest, variants) == "V0"


def test_resolve_baseline_picks_V0_even_when_first_variant_is_V1():
    """Variants reordered → V0 is no longer variants[0]; baseline must still be V0."""
    manifest = {"baseline_variant": "V0"}
    variants = ["V1", "V0", "V3"]
    assert rq4_compare._resolve_baseline(manifest, variants) == "V0"


def test_resolve_baseline_picks_explicit_baseline_not_variants_zero():
    manifest = {"baseline_variant": "V2"}
    variants = ["V0", "V1", "V2", "V3"]
    assert rq4_compare._resolve_baseline(manifest, variants) == "V2"


def test_resolve_baseline_rejects_missing_field():
    with pytest.raises(RuntimeError, match="missing 'baseline_variant'"):
        rq4_compare._resolve_baseline({}, ["V0", "V1"])


def test_resolve_baseline_rejects_empty_field():
    with pytest.raises(RuntimeError, match="missing 'baseline_variant'"):
        rq4_compare._resolve_baseline({"baseline_variant": ""}, ["V0"])


def test_resolve_baseline_rejects_baseline_not_in_variants():
    manifest = {"baseline_variant": "V9"}
    with pytest.raises(RuntimeError, match="not in variants"):
        rq4_compare._resolve_baseline(manifest, ["V0", "V1"])


def test_resolve_baseline_rejects_duplicate_variant_ids():
    manifest = {"baseline_variant": "V0"}
    with pytest.raises(RuntimeError, match="duplicates"):
        rq4_compare._resolve_baseline(manifest, ["V0", "V1", "V0"])


def test_resolve_baseline_rejects_empty_variants():
    with pytest.raises(RuntimeError, match="variants list is empty"):
        rq4_compare._resolve_baseline({"baseline_variant": "V0"}, [])


# ---- main(): end-to-end baseline resolution against per-user CSVs ----

def _write_per_user_csv(per_user_dir, variant, seed, n_users=5):
    rows = []
    for user in range(n_users):
        rows.append({
            "variant": variant,
            "seed": seed,
            "user_idx": user,
            "target_item": 100 + user,
            "rank": 1,
            "hit_at_10": 1.0,
            "ndcg_at_10": 0.5,
            "hit_at_20": 1.0,
            "ndcg_at_20": 0.5,
        })
    pd.DataFrame(rows).to_csv(per_user_dir / f"{variant}_s{seed}.csv", index=False)


def test_main_uses_explicit_baseline_even_when_variants_reordered(tmp_path):
    per_user_dir = tmp_path / "per_user"
    per_user_dir.mkdir()
    # Reorder variants: V1 is first. Baseline must still be V0 (per manifest).
    variants = ["V1", "V0", "V3"]
    for variant in variants:
        for seed in (42, 43):
            _write_per_user_csv(per_user_dir, variant, seed)

    manifest = {
        "benchmark_id": "test",
        "variants": variants,
        "neural_seeds": [42, 43],
        "baseline_variant": "V0",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    output_dir = tmp_path / "out"
    saved = sys.argv
    sys.argv = [
        "rq4_compare.py",
        "--per-user-dir", str(per_user_dir),
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
    ]
    try:
        rq4_compare.main()
    finally:
        sys.argv = saved

    csv_path = output_dir / "rq4_comparison.csv"
    import csv
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Every primary row must use V0 as base, never V1 or V3.
    primary = [r for r in rows if r["comparison_type"] == "primary"]
    assert len(primary) == 2  # V1 vs V0 and V3 vs V0
    for r in primary:
        assert r["base_variant"] == "V0", f"unexpected base {r}"


def test_main_fails_when_manifest_has_no_baseline_field(tmp_path):
    per_user_dir = tmp_path / "per_user"
    per_user_dir.mkdir()
    for variant in ["V0", "V1"]:
        _write_per_user_csv(per_user_dir, variant, 42)

    manifest = {
        "benchmark_id": "test",
        "variants": ["V0", "V1"],
        "neural_seeds": [42],
        # baseline_variant intentionally missing
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    saved = sys.argv
    sys.argv = [
        "rq4_compare.py",
        "--per-user-dir", str(per_user_dir),
        "--manifest", str(manifest_path),
        "--output-dir", str(tmp_path / "out"),
    ]
    try:
        with pytest.raises(RuntimeError, match="missing 'baseline_variant'"):
            rq4_compare.main()
    finally:
        sys.argv = saved


def test_main_fails_when_baseline_not_in_variants(tmp_path):
    per_user_dir = tmp_path / "per_user"
    per_user_dir.mkdir()
    for variant in ["V0", "V1"]:
        _write_per_user_csv(per_user_dir, variant, 42)

    manifest = {
        "benchmark_id": "test",
        "variants": ["V0", "V1"],
        "neural_seeds": [42],
        "baseline_variant": "V9",  # not in variants
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    saved = sys.argv
    sys.argv = [
        "rq4_compare.py",
        "--per-user-dir", str(per_user_dir),
        "--manifest", str(manifest_path),
        "--output-dir", str(tmp_path / "out"),
    ]
    try:
        with pytest.raises(RuntimeError, match="not in variants"):
            rq4_compare.main()
    finally:
        sys.argv = saved