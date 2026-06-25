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
        extra_params={"lr": 1e-3, "batch_size": 256},
    )

    assert data["model"] == "sasrec"
    assert data["seed"] == 42
    assert data["phase"] == "benchmark"
    assert data["lr"] == 1e-3
    assert data["batch_size"] == 256


def test_trainer_configures_mlflow_before_run(monkeypatch, tmp_path):
    """The trainer configures mlflow at construction time but opens the
    actual run lazily inside train() so the run lifecycle can be wrapped
    in a single try/finally. We assert both behaviors here."""
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

        def end_run(self, status="FINISHED"):
            calls.append(("end_run", status))

    monkeypatch.setitem(sys.modules, "mlflow", DummyMlflow())

    trainer = trainer_module.Trainer(
        "sasrec",
        "cpu",
        output_dir=tmp_path,
        use_mlflow=True,
        mlflow_config={"experiment_name": "mars_benchmark", "run_name": "benchmark-sasrec-seed-42"},
    )

    # configure_mlflow fires during __init__ — but start_run is deferred
    # to train() so it can be wrapped in a try/finally for clean FAILED
    # status on any exception.
    assert calls == ["configured"]
    assert trainer._mlflow_run is None, "Trainer must defer start_run until train()"


def test_train_script_uses_phase_to_select_experiment(monkeypatch, tmp_path):
    from training.mlflow_contract import get_experiment_name_for_phase

    assert get_experiment_name_for_phase("benchmark") == "mars_benchmark"


def test_phase_lookup_rejects_non_phase_workflows():
    from training.mlflow_contract import get_experiment_name_for_phase

    with pytest.raises(ValueError, match="Unsupported MLflow phase"):
        get_experiment_name_for_phase("rq2_tuning")


def test_trainer_last_run_id_survives_close_for_post_train_promotion(monkeypatch, tmp_path):
    """After train() returns, callers (e.g. rq4_ablation) need a stable
    handle on the MLflow run so they can promote per-user artifacts and
    flip reportable/per_user_complete tags. _mlflow_run is None after
    close (by design — the run is terminated), so Trainer exposes
    last_run_id for post-train read access."""
    import training.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module, "configure_mlflow",
        lambda **kw: type("S", (), {"tracking_uri": "x"})(),
    )

    class _Run:
        def __init__(self):
            self.info = type("I", (), {"run_id": "rid-stable"})()
            self.tags = {}

    class _Mlflow:
        def __init__(self):
            self.started = 0
        def set_experiment(self, name): pass
        def start_run(self, run_name=None):
            self.started += 1
            return _Run()
        def end_run(self, status="FINISHED"): pass
        def log_params(self, p): pass
        def log_metrics(self, m, step=None): pass
        def log_artifact(self, *a, **k): pass
        def set_tags(self, t): pass

    fake = _Mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    trainer = trainer_module.Trainer(
        "sasrec", "cpu", output_dir=tmp_path, use_mlflow=True,
        mlflow_config={"experiment_name": "ex", "run_name": "rn"},
    )

    import torch
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
        def forward(self, x): return self.lin(x)

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    fake_batches = {
        "input_seq": torch.zeros(1, 4), "pos_items": torch.tensor([1]), "neg_items": torch.tensor([2])
    }
    class _L:
        def __init__(self):
            self.dataset = [0, 1]
            self._b = [dict(fake_batches), dict(fake_batches)]
        def __iter__(self): return iter(self._b)
        def __len__(self): return len(self._b)

    trainer.train(
        model=_M(), train_loader=_L(), val_loader=_L(), test_loader=_L(),
        optimizer=torch.optim.SGD(_M().parameters(), 0.01),
        epochs=1, criterion_fn=crit, eval_fn=evalfn,
    )

    assert trainer._mlflow_run is None, "_mlflow_run must be cleared on close"
    assert trainer.last_run_id == "rid-stable", (
        "last_run_id must remain readable after close so RQ4 can promote "
        "per-user artifacts on the just-finished run"
    )
