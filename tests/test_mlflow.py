from pathlib import Path

import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("mars-recsys-test")

with mlflow.start_run(run_name="first-run"):
    mlflow.log_params(
        {
            "model": "dummy",
            "seed": 42,
            "learning_rate": 0.001,
        }
    )

    # MLflow metric names reject '@' on this server; sanitize like the trainer.
    mlflow.log_metrics(
        {
            "ndcg_at_10": 0.182,
            "recall_at_10": 0.261,
        }
    )

    artifact = Path("result.txt")
    artifact.write_text(
        "MLflow artifact upload works.",
        encoding="utf-8",
    )

    mlflow.log_artifact(str(artifact))

print("Logged successfully")