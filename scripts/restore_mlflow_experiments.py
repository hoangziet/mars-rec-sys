"""Restore all deleted MLflow experiments so they can be reused."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.tracking import MlflowClient
from training.mlflow_utils import configure_mlflow


def main() -> None:
    configure_mlflow(mlflow_module=mlflow)
    client = MlflowClient()

    all_exps = client.search_experiments()  # only active by default
    try:
        from mlflow.entities import ViewType

        deleted_exps = client.search_experiments(view_type=ViewType.DELETED_ONLY)
    except (ImportError, AttributeError, TypeError):
        try:
            deleted_exps = client.search_experiments(view_type="DELETED_ONLY")
        except Exception:
            deleted_exps = []

    if not deleted_exps:
        print(f"No deleted experiments. {len(all_exps)} active.")
        return

    print(f"Found {len(deleted_exps)} deleted experiments:")
    for e in deleted_exps:
        print(f"  {e.experiment_id}: {e.name}")

    restored = 0
    for e in deleted_exps:
        try:
            client.restore_experiment(e.experiment_id)
            print(f"  Restored: {e.name}")
            restored += 1
        except Exception as exc:
            print(f"  Failed: {e.name}: {exc}")

    print(f"Restored {restored}/{len(deleted_exps)} experiments.")


if __name__ == "__main__":
    main()
