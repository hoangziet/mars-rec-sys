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


def _make_protocol(best_variant="M3", rq2_variant="wlwe", rq2_alpha=0.5):
    return {
        "best_metadata_variant": best_variant,
        "metadata_variants": METADATA_VARIANTS,
        "rq2_best_variant": rq2_variant,
        "rq2_best_alpha": rq2_alpha,
    }


def _make_run(variant, seed, watch_mode="none", watch_alpha=0.0,
              use_structured=False, use_text=False,
              drop_watch_mode=False, drop_watch_alpha=False, drop_flags=False, run_idx=0):
    run_id = f"run_{variant}_{seed}_{run_idx}"
    tags = {
        "reportable": "true",
        "ablation_variant": variant,
        "use_structured": str(use_structured).lower(),
        "use_text": str(use_text).lower(),
    }
    if not drop_watch_mode:
        tags["watch_mode"] = watch_mode
    if not drop_watch_alpha:
        tags["watch_alpha"] = str(watch_alpha)
    if drop_flags:
        tags.pop("use_structured", None)
        tags.pop("use_text", None)
    return {"variant": variant, "seed": seed, "run_id": run_id, "tags": tags}


def _to_tags_lookup(selected):
    return {r["run_id"]: r.pop("tags") for r in selected}


def _valid_grid(best_variant="M3"):
    """4 variants x 3 seeds, all valid given wlwe/0.5 and best_variant."""
    flags = METADATA_VARIANTS[best_variant]
    use_meta_structured = flags["use_structured"]
    use_meta_text = flags["use_text"]
    runs = []
    for variant in ("V0", "V1", "V2", "V3"):
        watch_mode = "both" if variant in ("V1", "V3") else "none"
        watch_alpha = 0.5 if variant in ("V1", "V3") else 0.0
        if variant in ("V2", "V3"):
            use_structured = use_meta_structured
            use_text = use_meta_text
        else:
            use_structured = False
            use_text = False
        for seed in (42, 123, 2024):
            runs.append(_make_run(variant, seed, watch_mode=watch_mode, watch_alpha=watch_alpha,
                                 use_structured=use_structured, use_text=use_text))
    return runs


def test_validate_run_tags_accepts_valid_grid():
    selected = _valid_grid()
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert errors == []


def test_validate_run_tags_rejects_v0_with_nonzero_watch_alpha():
    selected = _valid_grid()
    selected[0]["tags"]["watch_alpha"] = "0.5"  # V0 seed=42, should be 0
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V0" in e and "watch_alpha=0" in e for e in errors)


def test_validate_run_tags_rejects_v2_with_nonzero_watch_alpha():
    selected = _valid_grid()
    v2_run = next(r for r in selected if r["variant"] == "V2")
    v2_run["tags"]["watch_alpha"] = "0.5"
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V2" in e and "watch_alpha=0" in e for e in errors)


def test_validate_run_tags_rejects_v1_with_wrong_watch_alpha():
    selected = _valid_grid()
    v1_run = next(r for r in selected if r["variant"] == "V1")
    v1_run["tags"]["watch_alpha"] = "0.0"  # should be 0.5
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V1" in e and "watch_alpha=0.5" in e for e in errors)


def test_validate_run_tags_rejects_v3_with_wrong_watch_alpha():
    selected = _valid_grid()
    v3_run = next(r for r in selected if r["variant"] == "V3")
    v3_run["tags"]["watch_alpha"] = "0.3"  # should be 0.5
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V3" in e and "watch_alpha=0.5" in e for e in errors)


def test_validate_run_tags_rejects_v0_with_wrong_watch_mode():
    selected = _valid_grid()
    v0_run = next(r for r in selected if r["variant"] == "V0")
    v0_run["tags"]["watch_mode"] = "both"  # V0 should be none
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V0" in e and "watch_mode=none" in e for e in errors)


def test_validate_run_tags_rejects_v1_with_wrong_watch_mode():
    selected = _valid_grid()
    v1_run = next(r for r in selected if r["variant"] == "V1")
    v1_run["tags"]["watch_mode"] = "embedding"  # should be "both" for wlwe
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V1" in e and "watch_mode=both" in e for e in errors)


def test_validate_run_tags_rejects_v2_wrong_use_structured():
    selected = _valid_grid()
    v2_run = next(r for r in selected if r["variant"] == "V2")
    v2_run["tags"]["use_structured"] = "false"  # M3 expects true
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V2" in e and "use_structured=true" in e for e in errors)


def test_validate_run_tags_rejects_v3_wrong_use_text():
    selected = _valid_grid()
    v3_run = next(r for r in selected if r["variant"] == "V3")
    v3_run["tags"]["use_text"] = "false"  # M3 expects true
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V3" in e and "use_text=true" in e for e in errors)


def test_validate_run_tags_rejects_v0_with_non_false_use_text():
    selected = _valid_grid()
    v0_run = next(r for r in selected if r["variant"] == "V0")
    v0_run["tags"]["use_text"] = "true"  # V0 expects false
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
    )
    assert any("V0" in e and "use_text=false" in e for e in errors)


def test_validate_run_tags_rejects_v1_with_non_false_use_structured():
    selected = _valid_grid()
    v1_run = next(r for r in selected if r["variant"] == "V1")
    v1_run["tags"]["use_structured"] = "true"  # V1 expects false
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(),
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
        selected, tags_by_run, _make_protocol(),
    )
    assert errors == []


def test_validate_run_tags_uses_best_metadata_variant():
    """best_variant=M1 means use_text should be false for V2/V3."""
    selected = _valid_grid(best_variant="M3")
    tags_by_run = _to_tags_lookup(selected)
    errors = rq4_collect._validate_run_tags(
        selected, tags_by_run, _make_protocol(best_variant="M1"),
    )
    assert any("V3" in e and "use_text=false" in e for e in errors)


def test_validate_provenance_tags_rejects_preprocessing_version_mismatch():
    selected = [{"variant": "V0", "seed": 42, "run_id": "r1"}]
    tags_by_run = {
        "r1": {
            "per_user_complete": "true",
            "preprocessing_version": "v2",
            "data_source": "/tmp/data",
        }
    }
    protocol = {"preprocessing_version": "v1", "data_source": "/tmp/data"}
    errors = rq4_collect._validate_provenance_tags(selected, tags_by_run, protocol)
    assert any("preprocessing_version mismatch" in e for e in errors)


def test_validate_provenance_tags_rejects_data_source_mismatch():
    selected = [{"variant": "V0", "seed": 42, "run_id": "r1"}]
    tags_by_run = {
        "r1": {
            "per_user_complete": "true",
            "preprocessing_version": "v1",
            "data_source": "/tmp/other",
        }
    }
    protocol = {"preprocessing_version": "v1", "data_source": "/tmp/data"}
    errors = rq4_collect._validate_provenance_tags(selected, tags_by_run, protocol)
    assert any("data_source mismatch" in e for e in errors)
