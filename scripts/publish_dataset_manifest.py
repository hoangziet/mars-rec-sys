from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from training.mlflow_contract import SHARED_EXPERIMENTS
from training.mlflow_utils import configure_mlflow


def main() -> None:
    configure_mlflow(mlflow_module=mlflow)
    experiment_name = SHARED_EXPERIMENTS["datasets"]
    run_name = "dataset-mars-v1"

    payload = {
        "dataset_name": "mars",
        "dataset_version": "mars-v1",
        "raw_data_hash": "replace-me",
        "processed_data_hash": "replace-me",
        "preprocessing_config_hash": "replace-me",
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "mars-rec-sys",
                "scope": "shared",
                "artifact_class": "dataset",
                "dataset_name": payload["dataset_name"],
                "dataset_version": payload["dataset_version"],
                "immutable": "true",
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(payload, indent=2))
            mlflow.log_artifact(str(manifest_path), artifact_path="manifest")


if __name__ == "__main__":
    main()
