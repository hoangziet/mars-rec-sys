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


def collect_common_run_metadata(
    *,
    model_name: str,
    seed: int,
    phase: str,
    git_commit: str,
    dataset_version: str,
    extra_params: dict,
) -> dict:
    payload = {
        "model": model_name,
        "seed": seed,
        "phase": phase,
        "git_commit": git_commit,
        "dataset_version": dataset_version,
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


def build_run_name(model_name: str, seed: int, phase: str | None = None) -> str:
    if phase:
        return f"{phase}-{model_name}-seed-{seed}"
    return f"{model_name}-seed-{seed}"


def get_git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_dataset_version(stats_path: Path) -> str:
    if not stats_path.exists():
        return "unknown"
    return "mars-processed-v1"
