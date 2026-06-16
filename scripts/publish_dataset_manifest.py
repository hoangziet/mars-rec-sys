from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from training.dataset_versioning import build_dataset_manifest, compare_against_latest_manifest, hash_directory
from training.mlflow_contract import SHARED_EXPERIMENTS
from training.mlflow_utils import configure_mlflow, get_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and publish a canonical dataset manifest.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-name", default="mars")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-record", default="data/processed/reports/dataset_freeze.json")
    return parser.parse_args()


def build_source_inventory(root: Path) -> list[str]:
    return sorted(path.relative_to(root.parent).as_posix() for path in root.rglob("*") if path.is_file())


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    output_record = Path(args.output_record)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory does not exist: {processed_dir}")

    configure_mlflow(mlflow_module=mlflow)

    raw_hash = hash_directory(raw_dir)
    processed_hash = hash_directory(processed_dir)
    preprocessing_hash = get_git_commit()

    payload = build_dataset_manifest(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        raw_data_hash=raw_hash,
        processed_data_hash=processed_hash,
        preprocessing_config_hash=preprocessing_hash,
        git_commit=get_git_commit(),
        source_files=build_source_inventory(raw_dir),
        processed_files=build_source_inventory(processed_dir),
        split_strategy="temporal_leave_one_out",
        preprocess_inputs={
            "config_source": "data.preprocess",
            "code_source": "data/preprocess.py",
        },
    )

    experiment_name = SHARED_EXPERIMENTS["datasets"]
    mlflow.set_experiment(experiment_name)

    latest = None
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs:
            latest_run = runs[0]
            artifacts = client.download_artifacts(latest_run.info.run_id, "manifest/dataset_manifest.json")
            latest = json.loads(Path(artifacts).read_text())

    comparison = compare_against_latest_manifest(current=payload, latest=latest)
    if comparison.name.startswith("REJECT"):
        raise RuntimeError(f"Dataset freeze rejected: {comparison.value}")

    run_name = f"dataset-{args.dataset_version}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "mars-rec-sys",
                "scope": "shared",
                "artifact_class": "dataset",
                "dataset_name": payload["dataset_name"],
                "dataset_version": payload["dataset_version"],
                "immutable": "true",
                "raw_data_hash": payload["raw_data_hash"],
                "processed_data_hash": payload["processed_data_hash"],
                "preprocessing_config_hash": payload["preprocessing_config_hash"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            manifest_path = tmp_root / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(payload, indent=2))
            mlflow.log_artifact(str(manifest_path), artifact_path="manifest")

        output_record.parent.mkdir(parents=True, exist_ok=True)
        output_record.write_text(
            json.dumps(
                {
                    "dataset_name": payload["dataset_name"],
                    "dataset_version": payload["dataset_version"],
                    "raw_data_hash": payload["raw_data_hash"],
                    "processed_data_hash": payload["processed_data_hash"],
                    "preprocessing_config_hash": payload["preprocessing_config_hash"],
                    "dataset_experiment": experiment_name,
                    "dataset_run_id": run.info.run_id,
                },
                indent=2,
            )
        )

        print(f"Published dataset manifest: {run.info.run_id}")
        print(f"Wrote local freeze record: {output_record}")


if __name__ == "__main__":
    main()
