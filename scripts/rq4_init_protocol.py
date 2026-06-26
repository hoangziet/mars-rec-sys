"""
scripts/rq4_init_protocol.py
=============================
Create the frozen protocol manifest for RQ4 ablation.

Must be run BEFORE training. The protocol manifest is immutable and
used by rq4_collect to validate exact run counts.

RQ4 is gSASRec-only: the backbone is frozen to "gsasrec" and validated
against the RQ2 and RQ3 winner artifacts. The RQ1 winner artifact is
no longer part of the RQ4 protocol.

Usage:
    uv run python scripts/rq4_init_protocol.py \\
        --benchmark-id rq4-ablation-v1 \\
        --rq2-winners experiments/rq2/.../rq2_best_alpha.json \\
        --rq3-winners experiments/rq3/.../rq3_best_variant.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999, 7, 21, 77, 314, 1337]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RQ4 frozen protocol manifest for gSASRec ablation.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--rq2-winners", required=True,
        help="Path to rq2_best_alpha.json")
    parser.add_argument("--rq3-winners", required=True,
        help="Path to rq3_best_variant.json")
    parser.add_argument("--variants", nargs="+", default=["V0", "V1", "V2", "V3"])
    parser.add_argument("--baseline-variant", default="V0",
        help="Explicit baseline variant ID for statistical comparison. "
             "Must exist in --variants.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    alpha_data = json.loads(Path(args.rq2_winners).read_text())
    variant_data = json.loads(Path(args.rq3_winners).read_text())
    if "best_alpha" not in alpha_data:
        raise ValueError(f"RQ2 winners JSON missing 'best_alpha': {args.rq2_winners}")
    if "best_variant" not in variant_data:
        raise ValueError(f"RQ3 winners JSON missing 'best_variant': {args.rq3_winners}")

    # RQ4 is gSASRec-only: both RQ2 and RQ3 winner artifacts must declare
    # backbone="gsasrec". This freezes the contract without falling back
    # to the RQ1 winner artifact.
    if alpha_data.get("backbone") != "gsasrec":
        raise RuntimeError(
            f"RQ2 winner artifact backbone must be 'gsasrec', got {alpha_data.get('backbone')!r}"
        )
    if variant_data.get("backbone") != "gsasrec":
        raise RuntimeError(
            f"RQ3 winner artifact backbone must be 'gsasrec', got {variant_data.get('backbone')!r}"
        )

    # Variants must be unique — duplicates would silently inflate run counts
    # and confuse the baseline lookup below.
    if len(set(args.variants)) != len(args.variants):
        seen: dict[str, int] = {}
        for v in args.variants:
            seen[v] = seen.get(v, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        raise ValueError(
            f"Variant IDs must be unique. Duplicates: {dupes}. "
            "Edit --variants before re-running."
        )

    # Explicit baseline requirement — never silently use variants[0].
    if not args.baseline_variant:
        raise ValueError(
            "--baseline-variant is required. Pass e.g. --baseline-variant V0."
        )
    if args.baseline_variant not in args.variants:
        raise ValueError(
            f"--baseline-variant={args.baseline_variant!r} not in --variants={args.variants}. "
            "Pick a baseline that exists in the variant list."
        )

    # Light provenance: RQ2 and RQ3 must share the same preprocessing
    # version and data source before we freeze the RQ4 protocol.
    rq2_pv = alpha_data.get("preprocessing_version")
    rq3_pv = variant_data.get("preprocessing_version")
    if not rq2_pv or not rq3_pv:
        raise RuntimeError("missing preprocessing_version in RQ2/RQ3 winner artifacts")
    if rq2_pv != rq3_pv:
        raise RuntimeError(
            f"RQ2 and RQ3 preprocessing_version differ: "
            f"RQ2={rq2_pv}, RQ3={rq3_pv}"
        )
    rq2_ds = alpha_data.get("data_source")
    rq3_ds = variant_data.get("data_source")
    if not rq2_ds or not rq3_ds:
        raise RuntimeError("missing data_source in RQ2/RQ3 winner artifacts")
    if rq2_ds != rq3_ds:
        raise RuntimeError(
            f"RQ2 and RQ3 data_source differ: "
            f"RQ2={rq2_ds}, RQ3={rq3_ds}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq4" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "rq4_protocol_manifest.json"
    if path.exists():
        raise RuntimeError(
            f"Protocol manifest already exists: {path}. "
            "Delete it first or use a new benchmark_id."
        )

    data_dir = Path(args.data_dir)

    manifest = {
        "benchmark_id": args.benchmark_id,
        "backbone": "gsasrec",
        "variants": args.variants,
        "baseline_variant": args.baseline_variant,
        "neural_seeds": args.seeds,
        "expected_runs": len(args.variants) * len(args.seeds),
        "best_alpha": float(alpha_data["best_alpha"]),
        "best_metadata_variant": str(variant_data["best_variant"]),
        "rq2_benchmark_id": alpha_data.get("benchmark_id"),
        "rq3_benchmark_id": variant_data.get("benchmark_id"),
        "preprocessing_version": rq2_pv,
        "data_source": rq2_ds,
        "metadata_variants": {
            "M0": {"use_structured": False, "use_text": False},
            "M1": {"use_structured": True,  "use_text": False},
            "M2": {"use_structured": False, "use_text": True},
            "M3": {"use_structured": True,  "use_text": True},
        },
    }

    path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Protocol manifest written: {path}")
    print(f"  Backbone: {manifest['backbone']}")
    print(f"  Baseline variant: {manifest['baseline_variant']}")
    print(f"  Best alpha: {manifest['best_alpha']}")
    print(f"  Best metadata variant: {manifest['best_metadata_variant']}")
    print(f"  Variants: {args.variants}")
    print(f"  Seeds: {len(args.seeds)}")
    print(f"  Expected runs: {manifest['expected_runs']}")


if __name__ == "__main__":
    main()
