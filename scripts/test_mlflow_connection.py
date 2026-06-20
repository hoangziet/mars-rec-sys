from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.artifacts import download_artifacts

from training.mlflow_contract import build_run_name
from training.mlflow_utils import configure_mlflow


def main() -> None:
    configure_mlflow(mlflow_module=mlflow)
    experiment_name = "phaseA_remote_test"
    run_name = build_run_name("connectivity-check", 0, variant="smoke")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("ping", "ok")
        mlflow.log_metric("score", 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "artifact-test.txt"
            artifact_path.write_text("mlflow artifact smoke test\n")
            mlflow.log_artifact(str(artifact_path))

        local_copy = download_artifacts(run_id=run.info.run_id, artifact_path="artifact-test.txt")
        content = Path(local_copy).read_text().strip()
        if content != "mlflow artifact smoke test":
            raise RuntimeError(f"Artifact readback mismatch: {content!r}")

        print("MLflow remote smoke test: PASS")
        print(f"Experiment: {experiment_name}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
