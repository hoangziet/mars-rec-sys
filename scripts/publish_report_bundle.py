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
    experiment_name = SHARED_EXPERIMENTS["reports"]
    run_name = "benchmark-summary-v1"

    summary = {
        "report_type": "benchmark",
        "report_version": "v1",
        "source_run_ids": [],
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "mars-rec-sys",
                "scope": "shared",
                "artifact_class": "report",
                "report_type": "benchmark",
                "report_version": "v1",
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            mlflow.log_artifact(str(summary_path), artifact_path="reports")


if __name__ == "__main__":
    main()
