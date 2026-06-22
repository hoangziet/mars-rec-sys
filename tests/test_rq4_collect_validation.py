import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_collect


METADATA_VARIANTS = {
    "M0": {"use_structured": False, "use_text": False},
    "M1": {"use_structured": True, "use_text": False},
    "M2": {"use_structured": False, "use_text": True},
    "M3": {"use_structured": True, "use_text": True},
}


def _make_run(variant, seed, alpha, use_structured=False, use_text=False,
              drop_alpha=False, drop_flags=False, run_idx=0):
    run_id = f"run_{variant}_{seed}_{run_idx}"
    tags = {
        "reportable": "true",
        "ablation_variant": variant,
        "use_structured": str(use_structured).lower(),
        "use_text": str(use_text).lower(),
    }
    if not drop_alpha:
        tags["confidence_alpha"] = str(alpha)
    if drop_flags:
        tags.pop("use_structured", None)
        tags.pop("use_text", None)
    return {"variant": variant, "seed": seed, "run_id": run_id, "tags": tags}


def _to_tags_lookup(selected):
    return {r["run_id"]: r.pop("tags") for r in selected}


def _valid_grid(best_variant="M3"):
    """4 variants x 3 seeds, all valid given best_alpha=0.5 and best_variant.

    V0/V1: use_structured=False, use_text=False
    V2/V3: use_structured and use_text taken from the best_variant flags
    """
    flags = METADATA_VARIANTS[best_variant]
    use_meta_structured = flags["use_structured"]
    use_meta_text = flags["use_text"]
    runs = []
    for variant in ("V0", "V1", "V2", "V3"):
        alpha = 0.0 if variant in ("V0", "V2") else 0.5
        if variant in ("V2", "V3"):
            use_structured = use_meta_structured
            use_text = use_meta_text
        else:
            use_structured = False
            use_text = False
        for seed in (42, 123, 2024):
            runs.append(_make_run(variant, seed, alpha, use_structured, use_text))
    return runs


def test_validate_run_tags_accepts_valid_grid():
    selected = _valid_grid()
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert errors == []


def test_validate_run_tags_rejects_v0_with_nonzero_alpha():
    selected = _valid_grid()
    selected[0]["tags"]["confidence_alpha"] = "0.5"  # V0 seed=42
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V0" in e and "expected alpha=0" in e for e in errors)


def test_validate_run_tags_rejects_v2_with_nonzero_alpha():
    selected = _valid_grid()
    # Find V2 in grid
    v2_run = next(r for r in selected if r["variant"] == "V2")
    v2_run["tags"]["confidence_alpha"] = "0.5"
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V2" in e and "expected alpha=0" in e for e in errors)


def test_validate_run_tags_rejects_v1_with_wrong_alpha():
    selected = _valid_grid()
    v1_run = next(r for r in selected if r["variant"] == "V1")
    v1_run["tags"]["confidence_alpha"] = "0.0"  # should be 0.5
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V1" in e and "expected alpha=0.5" in e for e in errors)


def test_validate_run_tags_rejects_v3_with_wrong_alpha():
    selected = _valid_grid()
    v3_run = next(r for r in selected if r["variant"] == "V3")
    v3_run["tags"]["confidence_alpha"] = "0.3"  # should be 0.5
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V3" in e and "expected alpha=0.5" in e for e in errors)


def test_validate_run_tags_rejects_missing_alpha_tag():
    selected = _valid_grid()
    v0_run = next(r for r in selected if r["variant"] == "V0")
    del v0_run["tags"]["confidence_alpha"]
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V0" in e and "missing confidence_alpha" in e for e in errors)


def test_validate_run_tags_rejects_non_numeric_alpha_tag():
    selected = _valid_grid()
    v0_run = next(r for r in selected if r["variant"] == "V0")
    v0_run["tags"]["confidence_alpha"] = "not_a_number"
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V0" in e and "not numeric" in e for e in errors)


def test_validate_run_tags_rejects_v2_wrong_use_structured():
    selected = _valid_grid()
    v2_run = next(r for r in selected if r["variant"] == "V2")
    v2_run["tags"]["use_structured"] = "false"  # M3 expects true
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V2" in e and "use_structured=true" in e for e in errors)


def test_validate_run_tags_rejects_v3_wrong_use_text():
    selected = _valid_grid()
    v3_run = next(r for r in selected if r["variant"] == "V3")
    v3_run["tags"]["use_text"] = "false"  # M3 expects true
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V3" in e and "use_text=true" in e for e in errors)


def test_validate_run_tags_rejects_v0_with_non_false_use_text():
    selected = _valid_grid()
    v0_run = next(r for r in selected if r["variant"] == "V0")
    v0_run["tags"]["use_text"] = "true"  # V0 expects false
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V0" in e and "use_text=false" in e for e in errors)


def test_validate_run_tags_rejects_v1_with_non_false_use_structured():
    selected = _valid_grid()
    v1_run = next(r for r in selected if r["variant"] == "V1")
    v1_run["tags"]["use_structured"] = "true"  # V1 expects false
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V1" in e and "use_structured=false" in e for e in errors)


def test_validate_run_tags_v0_v1_missing_flags_is_ok():
    """V0/V1 may omit the use_structured/use_text tags entirely (legacy runs)."""
    selected = _valid_grid()
    for r in selected:
        if r["variant"] in ("V0", "V1"):
            r["tags"].pop("use_structured", None)
            r["tags"].pop("use_text", None)
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M3", metadata_variants=METADATA_VARIANTS,
    )
    assert errors == []


def test_validate_run_tags_uses_best_metadata_variant():
    """best_variant=M1 means use_text should be false for V2/V3."""
    # Build grid with M3 (V2/V3 have use_text=true); validator expects M1.
    selected = _valid_grid(best_variant="M3")
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, expected_alpha=0.5,
        best_metadata_variant="M1", metadata_variants=METADATA_VARIANTS,
    )
    assert any("V3" in e and "use_text=false" in e for e in errors)
