from __future__ import annotations

import os
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class MlflowSettings:
    tracking_uri: str
    username: str
    password: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sanitize_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def load_mlflow_settings(*, load_dotenv_file: bool = True) -> MlflowSettings:
    if load_dotenv_file:
        load_dotenv(_repo_root() / ".env", override=False)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    missing = [
        name
        for name, value in {
            "MLFLOW_TRACKING_URI": tracking_uri,
            "MLFLOW_TRACKING_USERNAME": username,
            "MLFLOW_TRACKING_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required MLflow environment variables: "
            + ", ".join(missing)
            + ". Source .env before training."
        )

    return MlflowSettings(
        tracking_uri=tracking_uri,
        username=username,
        password=password,
    )


def assert_tracking_server_reachable(settings: MlflowSettings, *, client: MlflowClient) -> None:
    try:
        client.search_experiments()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach MLflow tracking server at "
            f"{settings.tracking_uri}. Check SSH tunnel, Nginx auth, and server availability."
        ) from exc


def ensure_experiment_active(*, mlflow_module: types.ModuleType, experiment_name: str):
    """Restore a soft-deleted experiment before calling ``set_experiment``.

    MLflow leaves deleted experiments in a soft-deleted state and
    ``get_experiment_by_name`` still returns them.  Reusing the same name
    later raises unless the experiment is restored first.
    """
    exp = mlflow_module.get_experiment_by_name(experiment_name)
    if exp is not None and getattr(exp, "lifecycle_stage", "active") != "deleted":
        return exp

    tracking = getattr(mlflow_module, "tracking", None)
    if tracking is None or not hasattr(tracking, "MlflowClient"):
        return None

    client = tracking.MlflowClient()

    if exp is not None:
        # Already resolved by name but soft-deleted — just restore it.
        client.restore_experiment(exp.experiment_id)
        return client.get_experiment(exp.experiment_id)

    # Not found by name at all; search across all view types in case a
    # stale instance exists only in the backend.
    entities = getattr(mlflow_module, "entities", None)
    view_type = getattr(getattr(entities, "ViewType", None), "ALL", None)
    search_kwargs = {"view_type": view_type} if view_type is not None else {}
    for candidate in client.search_experiments(**search_kwargs):
        if candidate.name != experiment_name:
            continue
        if getattr(candidate, "lifecycle_stage", None) == "deleted":
            client.restore_experiment(candidate.experiment_id)
            return client.get_experiment(candidate.experiment_id)
        return candidate
    return None


def collect_common_run_metadata(
    *,
    model_name: str,
    seed: int,
    phase: str,
    extra_params: dict,
) -> dict:
    payload = {
        "model": model_name,
        "seed": seed,
        "phase": phase,
    }
    payload.update(extra_params)
    return payload


def configure_mlflow(
    *,
    mlflow_module: types.ModuleType,
    client_factory: Callable[[], MlflowClient] | None = None,
    load_dotenv_file: bool = True,
) -> MlflowSettings:
    settings = load_mlflow_settings(load_dotenv_file=load_dotenv_file)
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.password
    mlflow_module.set_tracking_uri(settings.tracking_uri)

    if client_factory is None:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
    else:
        client = client_factory()

    assert_tracking_server_reachable(settings, client=client)
    return settings
