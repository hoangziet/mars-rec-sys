"""
scripts/rq4_collect.py
======================
RQ4: Collect ablation results from MLflow into local CSV/JSON.

Produces:
    rq4_runs.csv          — one row per (variant, seed)
    rq4_summary.json      — per-variant summary with validation_rank
    rq4_result_manifest.json — result metadata for rq4_compare

Validates against protocol manifest if --protocol is provided.

Usage:
    uv run python scripts/rq4_collect.py --benchmark-id rq4-ablation --protocol experiments/rq4/rq4-ablation/rq4_protocol_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq4_init_protocol import verify_protocol_hashes
from training.mlflow_utils import configure_mlflow

EXPERIMENT_NAME = "mars_final_ablation"
VARIANT_ORDER = {"V0": 0, "V1": 1, "V2": 2, "V3": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: collect ablation results from MLflow.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--protocol", default=None, help="Protocol manifest (validates exact runs, created before training)")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory (must match training data dir)")
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _get_run_tags(client, experiment_id):
    """Fetch tags from MLflow for all runs in the experiment.

    Returns a dict {run_id: tags_dict}.
    """
    run_tag_cache = {}
    for mlf_run in client.search_runs([experiment_id]):
        run_tag_cache[mlf_run.info.run_id] = mlf_run.data.tags
    return run_tag_cache


def _validate_run_tags(selected, tags_by_run, expected_alpha, best_metadata_variant, metadata_variants):
    """Validate that each run's tags match the expected config per variant.

    Returns a list of error strings (empty if all valid).  ``tags_by_run``
    is a {run_id: tags} dict produced by ``_get_run_tags``.
    """
    errors = []
    variant_flags = metadata_variants.get(best_metadata_variant, {})
    expected_structured = str(bool(variant_flags.get("use_structured", True))).lower()
    expected_text = str(bool(variant_flags.get("use_text", True))).lower()

    for r in selected:
        tags = tags_by_run.get(r["run_id"], {})
        variant = r["variant"]

        run_alpha_str = tags.get("confidence_alpha")
        if run_alpha_str is None:
            errors.append(
                f"{variant} seed={r['seed']}: missing confidence_alpha tag"
            )
        else:
            try:
                run_alpha = float(run_alpha_str)
            except (ValueError, TypeError):
                errors.append(
                    f"{variant} seed={r['seed']}: confidence_alpha tag is not numeric: {run_alpha_str!r}"
                )
            else:
                if variant in ("V0", "V2"):
                    if abs(run_alpha) > 1e-9:
                        errors.append(
                            f"{variant} seed={r['seed']}: expected alpha=0.0, got {run_alpha}"
                        )
                elif variant in ("V1", "V3"):
                    if abs(run_alpha - expected_alpha) > 1e-9:
                        errors.append(
                            f"{variant} seed={r['seed']}: expected alpha={expected_alpha}, got {run_alpha}"
                        )

        if variant in ("V2", "V3"):
            actual_structured = tags.get("use_structured", "missing")
            actual_text = tags.get("use_text", "missing")
            if actual_structured != expected_structured:
                errors.append(
                    f"{variant} seed={r['seed']}: expected use_structured={expected_structured}, "
                    f"got {actual_structured}"
                )
            if actual_text != expected_text:
                errors.append(
                    f"{variant} seed={r['seed']}: expected use_text={expected_text}, "
                    f"got {actual_text}"
                )
        else:
            actual_structured = tags.get("use_structured")
            actual_text = tags.get("use_text")
            if actual_structured is not None and actual_structured != "false":
                errors.append(
                    f"{variant} seed={r['seed']}: expected use_structured=false, got {actual_structured}"
                )
            if actual_text is not None and actual_text != "false":
                errors.append(
                    f"{variant} seed={r['seed']}: expected use_text=false, got {actual_text}"
                )

    return errors


def _validate_provenance_tags(selected: list[dict], tags_by_run: dict[str, dict],
                             protocol: dict) -> list[str]:
    """Each MLflow run must carry the exact provenance hashes from the protocol.

    A run with outdated or missing provenance tags was trained with a
    different data/config/artifact version and must not be included.
    """
    errors = []
    for r in selected:
        tags = tags_by_run.get(r["run_id"], {})
        variant = r["variant"]
        seed = r["seed"]
        prefix = f"{variant} seed={seed}"

        for key in ("protocol_sha256", "data_manifest_sha256", "config_sha256"):
            expected = protocol.get(key)
            if expected is None:
                continue
            actual = tags.get(key)
            if actual is None:
                errors.append(f"{prefix}: missing tag {key}")
            elif actual != expected:
                errors.append(
                    f"{prefix}: {key} mismatch — "
                    f"expected {expected[:12]}..., got {actual[:12]}..."
                )

        text_expected = protocol.get("text_artifact_sha256")
        if text_expected:
            text_actual = tags.get("text_artifact_sha256")
            if text_actual is None:
                errors.append(f"{prefix}: missing tag text_artifact_sha256")
            elif text_actual != text_expected:
                errors.append(
                    f"{prefix}: text_artifact_sha256 mismatch — "
                    f"expected {text_expected[:12]}..., got {text_actual[:12]}..."
                )

        git_expected = protocol.get("git_commit")
        if git_expected and git_expected != "unknown":
            git_actual = tags.get("git_commit")
            if git_actual is None:
                errors.append(f"{prefix}: missing tag git_commit")
            elif git_actual != git_expected:
                errors.append(
                    f"{prefix}: git_commit mismatch — "
                    f"expected {git_expected[:12]}, got {git_actual[:12]}"
                )

    return errors


def main() -> None:
    args = parse_args()
    configure_mlflow(mlflow_module=mlflow)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' does not exist")

    runs = client.search_runs([experiment.experiment_id])
    selected = []
    for run in runs:
        tags = run.data.tags
        if run.info.status != "FINISHED":
            continue
        if tags.get("reportable") != "true":
            continue
        if tags.get("benchmark_id") != args.benchmark_id:
            continue
        variant = tags.get("ablation_variant", "?")
        seed = int(run.data.params.get("seed", "0"))
        val_ndcg = run.data.metrics.get("best_val_ndcg_at_10")
        test_ndcg = run.data.metrics.get("test_NDCG_at_10")
        test_recall = run.data.metrics.get("test_Recall_at_10")
        test_ndcg20 = run.data.metrics.get("test_NDCG_at_20")
        test_recall20 = run.data.metrics.get("test_Recall_at_20")
        if val_ndcg is None or test_ndcg is None:
            continue
        selected.append({
            "variant": variant,
            "seed": seed,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "best_val_ndcg_at_10": val_ndcg,
            "test_NDCG_at_10": test_ndcg,
            "test_Recall_at_10": test_recall,
            "test_NDCG_at_20": test_ndcg20,
            "test_Recall_at_20": test_recall20,
        })

    if not selected:
        raise RuntimeError(f"No reportable runs found for benchmark {args.benchmark_id}")

    # Validate against protocol manifest if provided
    if args.protocol:
        protocol = json.loads(Path(args.protocol).read_text())

        # Reject collect if data/config/text artifacts drifted since the
        # protocol was frozen. Catches the case where someone re-runs collect
        # on a different machine with regenerated embeddings, etc.
        verify_protocol_hashes(
            manifest=protocol,
            data_dir=Path(args.data_dir),
            configs_glob="configs/model/*.yaml",
        )

        expected_variants = set(protocol["variants"])
        expected_seeds = {int(s) for s in protocol["neural_seeds"]}
        expected_runs = len(expected_variants) * len(expected_seeds)
        expected_alpha = float(protocol["best_alpha"])

        actual_variants = {r["variant"] for r in selected}
        actual_seeds = {r["seed"] for r in selected}
        actual_pairs = {(r["variant"], r["seed"]) for r in selected}

        errors = []
        if actual_variants != expected_variants:
            missing_v = sorted(expected_variants - actual_variants)
            extra_v = sorted(actual_variants - expected_variants)
            if missing_v:
                errors.append(f"Missing variants: {missing_v}")
            if extra_v:
                errors.append(f"Extra variants: {extra_v}")
        if actual_seeds != expected_seeds:
            missing_s = sorted(expected_seeds - actual_seeds)
            extra_s = sorted(actual_seeds - expected_seeds)
            if missing_s:
                errors.append(f"Missing seeds: {missing_s}")
            if extra_s:
                errors.append(f"Extra seeds: {extra_s}")
        if len(actual_pairs) != expected_runs:
            errors.append(f"Expected {expected_runs} runs, got {len(actual_pairs)}")
        pair_counts = {}
        for r in selected:
            key = (r["variant"], r["seed"])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        dupes = {k: v for k, v in pair_counts.items() if v > 1}
        if dupes:
            errors.append(f"Duplicate (variant, seed): {dupes}")

        # Validate tags against protocol
        tags_by_run = _get_run_tags(client, experiment.experiment_id)
        best_metadata_variant = protocol.get("best_metadata_variant", "M3")
        metadata_variants = protocol.get("metadata_variants", {})
        errors.extend(
            _validate_run_tags(
                selected,
                tags_by_run,
                expected_alpha,
                best_metadata_variant,
                metadata_variants,
            )
        )

        # Each MLflow run must carry provenance hashes matching the protocol
        errors.extend(_validate_provenance_tags(selected, tags_by_run, protocol))

        # Require all secondary metrics
        for r in selected:
            missing_metrics = []
            for m in ["test_Recall_at_10", "test_NDCG_at_20", "test_Recall_at_20"]:
                if r.get(m) is None:
                    missing_metrics.append(m)
            if missing_metrics:
                errors.append(f"{r['variant']} seed={r['seed']}: missing metrics {missing_metrics}")

        if errors:
            raise RuntimeError("Protocol validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / "rq4" / args.benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write rq4_runs.csv
    run_fields = ["variant", "seed", "run_id", "run_name", "best_val_ndcg_at_10",
                   "test_NDCG_at_10", "test_Recall_at_10", "test_NDCG_at_20", "test_Recall_at_20"]
    with open(output_dir / "rq4_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(selected)

    # Group by variant for summary
    by_variant: dict[str, list[dict]] = {}
    for r in selected:
        by_variant.setdefault(r["variant"], []).append(r)

    # Build summary with validation_rank
    summary_rows = []
    for variant in sorted(by_variant.keys(), key=lambda v: VARIANT_ORDER.get(v, 99)):
        runs_list = by_variant[variant]
        val_vals = [r["best_val_ndcg_at_10"] for r in runs_list]
        test_vals = [r["test_NDCG_at_10"] for r in runs_list]
        seeds = sorted({r["seed"] for r in runs_list})
        summary_rows.append({
            "model": variant,
            "variant": variant,
            "runs": len(runs_list),
            "seeds": seeds,
            "val_ndcg_at_10": {
                "mean": float(np.mean(val_vals)),
                "std": float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0,
            },
            "test_ndcg_at_10": {
                "mean": float(np.mean(test_vals)),
                "std": float(np.std(test_vals, ddof=1)) if len(test_vals) > 1 else 0.0,
            },
        })

    # Sort by val mean descending, assign validation_rank
    summary_rows.sort(key=lambda r: (-r["val_ndcg_at_10"]["mean"], VARIANT_ORDER.get(r["variant"], 99)))
    for rank, row in enumerate(summary_rows, start=1):
        row["validation_rank"] = rank

    with open(output_dir / "rq4_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Build manifest for rq4_compare
    all_seeds = sorted({r["seed"] for r in selected})
    all_variants = sorted(by_variant.keys(), key=lambda v: VARIANT_ORDER.get(v, 99))
    manifest = {
        "benchmark_id": args.benchmark_id,
        "variants": all_variants,
        "neural_seeds": all_seeds,
        "n_seeds": len(all_seeds),
        "n_variants": len(all_variants),
    }
    if args.protocol:
        for key in ("protocol_sha256", "data_manifest_sha256", "config_sha256",
                     "text_artifact_sha256", "git_commit"):
            val = protocol.get(key)
            if val is not None:
                manifest[key] = val
    with open(output_dir / "rq4_result_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Collected {len(selected)} runs across {len(by_variant)} variants")
    print(f"Variants: {all_variants}")
    print(f"Seeds: {all_seeds}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
