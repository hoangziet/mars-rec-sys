import sys
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import mlflow_utils


def test_assert_tracking_server_reachable_wraps_mlflow_failure():
    class DummyClient:
        def search_experiments(self):
            raise MlflowException("boom")

    settings = mlflow_utils.MlflowSettings(
        tracking_uri="http://127.0.0.1:8080",
        username="alice",
        password="secret",
    )

    with pytest.raises(RuntimeError, match="Could not reach MLflow tracking server"):
        mlflow_utils.assert_tracking_server_reachable(settings, client=DummyClient())


def test_collect_common_run_metadata_includes_required_fields():
    data = mlflow_utils.collect_common_run_metadata(
        model_name="sasrec",
        seed=42,
        phase="benchmark",
        git_commit="abc123",
        extra_params={"lr": 1e-3, "batch_size": 256},
    )

    assert data["model"] == "sasrec"
    assert data["seed"] == 42
    assert data["phase"] == "benchmark"
    assert data["git_commit"] == "abc123"
    assert data["lr"] == 1e-3
    assert data["batch_size"] == 256


def test_trainer_configures_mlflow_before_run(monkeypatch, tmp_path):
    import training.trainer as trainer_module

    calls = []

    def fake_configure_mlflow(*, mlflow_module, client_factory=None, load_dotenv_file=True):
        calls.append("configured")

        class Settings:
            tracking_uri = "http://127.0.0.1:8080"

        return Settings()

    monkeypatch.setattr(trainer_module, "configure_mlflow", fake_configure_mlflow)

    class DummyMlflow:
        def set_experiment(self, name):
            calls.append(("experiment", name))

        def start_run(self, run_name=None):
            calls.append(("start_run", run_name))
            return object()

    monkeypatch.setitem(sys.modules, "mlflow", DummyMlflow())

    trainer = trainer_module.Trainer(
        "sasrec",
        "cpu",
        output_dir=tmp_path,
        use_mlflow=True,
        mlflow_config={"experiment_name": "mars_benchmark", "run_name": "benchmark-sasrec-seed-42"},
    )

    assert calls[0] == "configured"
    assert calls[1] == ("experiment", "mars_benchmark")
    assert calls[2] == ("start_run", "benchmark-sasrec-seed-42")


def test_train_script_uses_phase_to_select_experiment(monkeypatch, tmp_path):
    from training.mlflow_contract import get_experiment_name_for_phase

    assert get_experiment_name_for_phase("benchmark") == "mars_benchmark"
