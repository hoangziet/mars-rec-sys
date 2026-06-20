import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import mlflow_utils


def test_load_mlflow_settings_reads_required_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")

    settings = mlflow_utils.load_mlflow_settings(load_dotenv_file=False)

    assert settings.tracking_uri == "http://127.0.0.1:8080"
    assert settings.username == "alice"
    assert settings.password == "secret"


def test_load_mlflow_settings_raises_when_required_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="Missing required MLflow environment variables"):
        mlflow_utils.load_mlflow_settings(load_dotenv_file=False)


def test_configure_mlflow_sets_tracking_uri_and_auth_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")

    calls = {}

    class DummyMlflow:
        @staticmethod
        def set_tracking_uri(uri):
            calls["uri"] = uri

    class DummyClient:
        def search_experiments(self):
            pass

    settings = mlflow_utils.configure_mlflow(
        mlflow_module=DummyMlflow,
        client_factory=lambda: DummyClient(),
        load_dotenv_file=False,
    )

    assert calls["uri"] == "http://127.0.0.1:8080"
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "alice"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "secret"
    assert settings.tracking_uri == "http://127.0.0.1:8080"


def test_sanitize_metric_name_replaces_at_symbol():
    assert mlflow_utils.sanitize_metric_name("Recall@10") == "Recall_at_10"
    assert mlflow_utils.sanitize_metric_name("NDCG@20") == "NDCG_at_20"
    assert mlflow_utils.sanitize_metric_name("train_loss") == "train_loss"
