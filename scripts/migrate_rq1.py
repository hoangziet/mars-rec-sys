from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_utils import configure_mlflow

BACKFILL_VERSION = "mars-preprocess-v1"


def migrate_manifest(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return False

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("preprocessing_version") == BACKFILL_VERSION:
        print(f"Manifest already has preprocessing_version={BACKFILL_VERSION}, skipping")
        return True

    expected = manifest.get("dataset_version")
    if expected:
        print(f"Found dataset_version={expected} in manifest. Upgrading to preprocessing_version={BACKFILL_VERSION}")

    del manifest["dataset_version"]
    manifest["preprocessing_version"] = BACKFILL_VERSION
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated manifest: {manifest_path}")
    return True


def backfill_runs(benchmark_id: str, protocol_version: str) -> int:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("mars_benchmark")
    if experiment is None:
        raise RuntimeError("Experiment 'mars_benchmark' does not exist")

    runs = client.search_runs([experiment.experiment_id])
    count = 0

    for run in runs:
        tags = run.data.tags
        if tags.get("benchmark_id") != benchmark_id:
            continue
        if tags.get("protocol_version") != protocol_version:
            continue
        if tags.get("preprocessing_version") == BACKFILL_VERSION:
            continue

        client.set_tag(run.info.run_id, "preprocessing_version", BACKFILL_VERSION)
        count += 1

    return count


def main() -> None:
    configure_mlflow(mlflow_module=mlflow)

    benchmark_id = "rq1-v1"
    protocol_version = "rq1-v1"
    manifest_path = Path("experiments") / "benchmark" / benchmark_id / "benchmark_manifest.json"

    if migrate_manifest(manifest_path):
        count = backfill_runs(benchmark_id, protocol_version)
        print(f"Backfilled preprocessing_version={BACKFILL_VERSION} to {count} MLflow runs")
    else:
        print("Migration aborted — manifest not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
