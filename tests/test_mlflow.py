import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.artifacts import download_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.mlflow_utils import configure_mlflow


def test_mlflow_round_trip_logs_params_metrics_and_artifact():
    configure_mlflow(mlflow_module=mlflow)
    experiment_name = "mars-recsys-test"

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="first-run") as run:
        mlflow.log_params(
            {
                "model": "dummy",
                "seed": 42,
                "learning_rate": 0.001,
            }
        )

        mlflow.log_metrics(
            {
                "ndcg_at_10": 0.182,
                "recall_at_10": 0.261,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = Path(tmp_dir) / "result.txt"
            artifact.write_text("MLflow artifact upload works.\n", encoding="utf-8")
            mlflow.log_artifact(str(artifact))

        local_copy = download_artifacts(run_id=run.info.run_id, artifact_path="result.txt")
        assert Path(local_copy).read_text(encoding="utf-8").strip() == "MLflow artifact upload works."
