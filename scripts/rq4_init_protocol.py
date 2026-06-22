"""
scripts/rq4_init_protocol.py
=============================
Create the frozen protocol manifest for RQ4 ablation.

Must be run BEFORE training. The protocol manifest is immutable and
used by rq4_collect to validate exact run counts.

Usage:
    uv run python scripts/rq4_init_protocol.py \\
        --benchmark-id rq4-ablation-v1 \\
        --rq2-winners experiments/rq2/.../rq2_best_alpha.json \\
        --rq3-winners experiments/rq3/.../rq3_best_variant.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999, 7, 21, 77, 314, 1337]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RQ4 frozen protocol manifest.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--rq2-winners", required=True,
        help="Path to rq2_best_alpha.json")
    parser.add_argument("--rq3-winners", required=True,
        help="Path to rq3_best_variant.json")
    parser.add_argument("--variants", nargs="+", default=["V0", "V1", "V2", "V3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--preprocessing-version", default="mars-preprocess-v1")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_concat(glob_pattern: str) -> str:
    """Concatenate all files matching the glob, then SHA256."""
    h = hashlib.sha256()
    for path in sorted(glob.glob(glob_pattern)):
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def _get_git_long_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()

    alpha_data = json.loads(Path(args.rq2_winners).read_text())
    variant_data = json.loads(Path(args.rq3_winners).read_text())
    if "best_alpha" not in alpha_data:
        raise ValueError(f"RQ2 winners JSON missing 'best_alpha': {args.rq2_winners}")
    if "best_variant" not in variant_data:
        raise ValueError(f"RQ3 winners JSON missing 'best_variant': {args.rq3_winners}")

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
        "variants": args.variants,
        "neural_seeds": args.seeds,
        "expected_runs": len(args.variants) * len(args.seeds),
        "best_alpha": float(alpha_data["best_alpha"]),
        "best_metadata_variant": str(variant_data["best_variant"]),
        "rq2_benchmark_id": alpha_data.get("benchmark_id"),
        "rq3_benchmark_id": variant_data.get("benchmark_id"),
        "git_commit": _get_git_long_commit(),
        "preprocessing_version": args.preprocessing_version,
        "metadata_variants": {
            "M0": {"use_structured": False, "use_text": False},
            "M1": {"use_structured": True,  "use_text": False},
            "M2": {"use_structured": False, "use_text": True},
            "M3": {"use_structured": True,  "use_text": True},
        },
    }

    # Provenance SHA256s — these let rq4_ablation detect drift between
    # manifest creation time and training time.
    data_manifest_path = data_dir / "reports" / "preprocessing_report.json"
    if data_manifest_path.exists():
        manifest["data_manifest_sha256"] = _sha256_file(data_manifest_path)
    else:
        manifest["data_manifest_sha256"] = None

    if Path("configs/model").exists():
        manifest["config_sha256"] = _sha256_concat("configs/model/*.yaml")
    else:
        manifest["config_sha256"] = None

    text_emb_path = data_dir / "item_features" / "text_embeddings.pt"
    if text_emb_path.exists():
        manifest["text_artifact_sha256"] = _sha256_file(text_emb_path)
    else:
        manifest["text_artifact_sha256"] = None

    path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Self-hash: hash of canonical (sort_keys) JSON of the manifest, excluding
    # the field itself.  This is a content identifier, not a hash of the file
    # (the file is written without sort_keys for human readability).
    manifest_no_self_hash = {k: v for k, v in manifest.items() if k != "protocol_sha256"}
    manifest["protocol_sha256"] = hashlib.sha256(
        json.dumps(manifest_no_self_hash, sort_keys=True, indent=2).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Protocol manifest written: {path}")
    print(f"  Best alpha: {manifest['best_alpha']}")
    print(f"  Best metadata variant: {manifest['best_metadata_variant']}")
    print(f"  Git commit: {manifest['git_commit'][:12]}")
    print(f"  Data manifest SHA256: {(manifest.get('data_manifest_sha256') or 'N/A')[:12]}")
    print(f"  Config SHA256: {(manifest.get('config_sha256') or 'N/A')[:12]}")
    print(f"  Variants: {args.variants}")
    print(f"  Seeds: {len(args.seeds)}")
    print(f"  Expected runs: {manifest['expected_runs']}")


if __name__ == "__main__":
    main()
