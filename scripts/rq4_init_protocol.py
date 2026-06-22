"""
scripts/rq4_init_protocol.py
=============================
Create the frozen protocol manifest for RQ4 ablation.

Must be run BEFORE training. The protocol manifest is immutable and
used by rq4_collect to validate exact run counts.

Usage:
    uv run python scripts/rq4_init_protocol.py --benchmark-id rq4-ablation-v1 --best-alpha 0.5 --best-variant M3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999, 7, 21, 77, 314, 1337]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RQ4 frozen protocol manifest.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--best-alpha", type=float, required=True)
    parser.add_argument("--best-variant", required=True, choices=["M0", "M1", "M2", "M3"])
    parser.add_argument("--variants", nargs="+", default=["V0", "V1", "V2", "V3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rq2-benchmark-id", default=None)
    parser.add_argument("--rq3-benchmark-id", default=None)
    parser.add_argument("--preprocessing-version", default="mars-preprocess-v1")
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq4" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "rq4_protocol_manifest.json"
    if path.exists():
        raise RuntimeError(
            f"Protocol manifest already exists: {path}. "
            "Delete it first or use a new benchmark_id."
        )

    manifest = {
        "benchmark_id": args.benchmark_id,
        "variants": args.variants,
        "neural_seeds": args.seeds,
        "expected_runs": len(args.variants) * len(args.seeds),
        "best_alpha": args.best_alpha,
        "best_metadata_variant": args.best_variant,
        "rq2_benchmark_id": args.rq2_benchmark_id,
        "rq3_benchmark_id": args.rq3_benchmark_id,
        "git_commit": _get_git_commit(),
        "preprocessing_version": args.preprocessing_version,
    }

    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Protocol manifest written: {path}")
    print(f"  Variants: {args.variants}")
    print(f"  Seeds: {len(args.seeds)}")
    print(f"  Expected runs: {manifest['expected_runs']}")


if __name__ == "__main__":
    main()
