"""Build a completed manifest for an already-finished study campaign by scanning
the experiment directory on disk."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.study_manifest import create_manifest, finalize_manifest, mark_completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--backbone", default="bert4rec")
    args = parser.parse_args()
    root = Path(args.output_dir)
    manifest_path = root / "benchmark_manifest.json"
    create_manifest(manifest_path, variants=args.variants, seeds=args.seeds, benchmark_id=args.benchmark_id, backbone=args.backbone)
    for variant in args.variants:
        for seed in args.seeds:
            metrics = root / variant / f"seed_{seed}" / args.backbone / "metrics.json"
            if metrics.exists():
                mark_completed(manifest_path, variant, seed)
    finalize_manifest(manifest_path)
    print(f"Created retroactive manifest: {manifest_path}")


if __name__ == "__main__":
    main()
