"""Tests that the Trainer MLflow lifecycle handles every exception class.

The trainer must guarantee:
    - Success → run closed with status FINISHED.
    - Any exception → run closed with status FAILED.
    - ``end_run`` is called at most once.

To intercept the lazy ``import mlflow`` inside the Trainer, we install a
fake mlflow into ``sys.modules`` for the duration of the test.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import trainer as trainer_mod
from training.trainer import NoValidCheckpointError, Trainer


class _RecordingMlflow:
    """Fake mlflow module — installed in sys.modules."""

    def __init__(self):
        self.started = 0
        self.ended_with = []
        self.set_tags_calls = 0
        self.log_params_calls = 0
        self.log_metrics_calls = 0
        self.log_metrics_should_raise = False
        self.last_tags = {}

    def set_experiment(self, name): pass
    def set_tracking_uri(self, uri): pass
    def set_tag(self, *a, **kw): pass
    def get_experiment_by_name(self, name): return None
    def search_runs(self, *a, **kw): return []
    def search_experiments(self, *a, **kw): return []

    def start_run(self, run_name=None):
        self.started += 1
        return _RecordingRun(self, run_name)

    def end_run(self, status="FINISHED"):
        self.ended_with.append(status)

    def log_params(self, params):
        self.log_params_calls += 1

    def log_metrics(self, metrics, step=None):
        # log_metrics is also routed via the mlflow module in trainer.py.
        self.log_metrics_calls += 1
        if self.log_metrics_should_raise:
            self.log_metrics_should_raise = False
            raise RuntimeError("log_metrics exploded")

    def log_artifact(self, *args, **kwargs):
        pass

    def set_tags(self, tags):
        # Trainer routes set_tags through the mlflow module, not the run.
        self.last_tags.update(tags)
        self.set_tags_calls += 1


class _RecordingRun:
    def __init__(self, mlf, run_name):
        self.mlf = mlf
        self.run_name = run_name
        self.info = type("I", (), {"run_id": "rid"})()
        self.tags = {}

    def set_tags(self, tags):
        self.tags.update(tags)
        self.mlf.set_tags_calls += 1

    def log_params(self, params):
        self.mlf.log_params_calls += 1

    def log_metrics(self, metrics, step=None):
        self.mlf.log_metrics_calls += 1
        if self.mlf.log_metrics_should_raise:
            self.mlf.log_metrics_should_raise = False
            raise RuntimeError("log_metrics exploded")

    def log_artifact(self, *args, **kwargs): pass


class _NaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)
    def forward(self, x):
        return self.lin(x)


class _FakeLoader:
    def __init__(self, n=2):
        self.dataset = list(range(n))
        self._batches = [
            {"input_seq": torch.zeros(1, 4), "pos_items": torch.tensor([1]), "neg_items": torch.tensor([2])}
            for _ in range(n)
        ]
    def __iter__(self): return iter(self._batches)
    def __len__(self): return len(self._batches)


@pytest.fixture
def fake_mlflow(monkeypatch):
    """Install our fake mlflow into sys.modules AND bypass configure_mlflow
    (which would try to load .env and reach the real MLflow server)."""
    rec = _RecordingMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", rec)
    # The trainer imports configure_mlflow at module load time. Patch the
    # bound name inside trainer.py directly.
    monkeypatch.setattr(
        trainer_mod, "configure_mlflow",
        lambda **kw: type("S", (), {"tracking_uri": "x", "username": "u", "password": "p"})(),
    )
    return rec


def test_trainer_closes_run_with_FINISHED_on_success(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    trainer.train(
        model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
        test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
        epochs=1, criterion_fn=crit, eval_fn=evalfn,
        mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true", "per_user_complete": "true"}},
    )

    assert fake_mlflow.started == 1
    assert fake_mlflow.ended_with == ["FINISHED"], fake_mlflow.ended_with


def test_trainer_closes_run_with_FAILED_on_criterion_exception(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def exploding_crit(m, b, d):
        raise RuntimeError("training step exploded")

    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    with pytest.raises(RuntimeError, match="training step exploded"):
        trainer.train(
            model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
            test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
            epochs=1, criterion_fn=exploding_crit, eval_fn=evalfn,
            mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
        )

    assert fake_mlflow.started == 1
    assert fake_mlflow.ended_with == ["FAILED"], fake_mlflow.ended_with


def test_trainer_closes_run_with_FAILED_on_eval_exception(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def crit(m, b, d): return m(b["input_seq"]).sum()

    def exploding_eval(m, l, d):
        raise RuntimeError("eval step exploded")

    with pytest.raises(RuntimeError, match="eval step exploded"):
        trainer.train(
            model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
            test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
            epochs=1, criterion_fn=crit, eval_fn=exploding_eval,
            mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
        )

    assert fake_mlflow.ended_with == ["FAILED"], fake_mlflow.ended_with


def test_trainer_closes_run_with_FAILED_on_log_metrics_exception(tmp_path, fake_mlflow):
    fake_mlflow.log_metrics_should_raise = True
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    with pytest.raises(RuntimeError, match="log_metrics exploded"):
        trainer.train(
            model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
            test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
            epochs=1, criterion_fn=crit, eval_fn=evalfn,
            mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
        )

    assert fake_mlflow.ended_with == ["FAILED"], fake_mlflow.ended_with


def test_trainer_closes_run_with_FAILED_on_no_valid_checkpoint(tmp_path, fake_mlflow):
    trainer = Trainer("nan", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def nan_crit(m, b, d): return torch.tensor(float("nan"))
    def nan_eval(m, l, d): return {"NDCG@10": float("nan"), "Recall@10": float("nan")}

    with pytest.raises(NoValidCheckpointError):
        trainer.train(
            model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
            test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
            epochs=2, criterion_fn=nan_crit, eval_fn=nan_eval,
            mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
        )

    assert fake_mlflow.ended_with == ["FAILED"], fake_mlflow.ended_with


def test_trainer_calls_end_run_at_most_once_per_train(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    trainer.train(
        model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
        test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
        epochs=2, criterion_fn=crit, eval_fn=evalfn,
        mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
    )

    assert len(fake_mlflow.ended_with) == 1, fake_mlflow.ended_with


def test_trainer_does_not_log_tags_as_params(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    captured = {}
    real_log_params = fake_mlflow.log_params

    def capture(params):
        captured.update(params)
        return real_log_params(params)

    fake_mlflow.log_params = capture

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    trainer.train(
        model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
        test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
        epochs=1, criterion_fn=crit, eval_fn=evalfn,
        mlflow_params={
            "params": {"seed": 1, "lr": 0.001},
            "tags": {"reportable": "true", "per_user_complete": "true",
                     "backbone": "gsasrec", "variant": "V0"},
        },
    )

    for bad in ("reportable", "per_user_complete", "backbone", "variant", "tags"):
        assert bad not in captured, (
            f"tag-like key {bad!r} was flattened into MLflow params: {list(captured)}"
        )
    assert "seed" in captured
    assert "lr" in captured


def test_trainer_preserves_last_run_id_after_close(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    trainer.train(
        model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
        test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
        epochs=1, criterion_fn=crit, eval_fn=evalfn,
        mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true", "per_user_complete": "true"}},
    )

    assert trainer._mlflow_run is None
    assert trainer.last_run_id == "rid"


def test_trainer_preserves_last_run_id_on_no_valid_checkpoint(tmp_path, fake_mlflow):
    trainer = Trainer("nan", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    def nan_crit(m, b, d): return torch.tensor(float("nan"))
    def nan_eval(m, l, d): return {"NDCG@10": float("nan"), "Recall@10": float("nan")}

    with pytest.raises(NoValidCheckpointError):
        trainer.train(
            model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
            test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
            epochs=2, criterion_fn=nan_crit, eval_fn=nan_eval,
            mlflow_params={"params": {"seed": 1}, "tags": {"reportable": "true"}},
        )

    assert trainer._mlflow_run is None
    assert trainer.last_run_id == "rid"


def test_trainer_supports_legacy_mlflow_params_format(tmp_path, fake_mlflow):
    trainer = Trainer("ok", "cpu", run_dir=tmp_path, use_mlflow=True,
                      mlflow_config={"experiment_name": "ex", "run_name": "rn"})

    captured = {}
    real_log_params = fake_mlflow.log_params
    def capture(params):
        captured.update(params)
        return real_log_params(params)
    fake_mlflow.log_params = capture

    def crit(m, b, d): return m(b["input_seq"]).sum()
    def evalfn(m, l, d): return {"NDCG@10": 0.1, "Recall@10": 0.1}

    legacy_mlflow_params = {
        "model": "gsasrec",
        "seed": 42,
        "phase": "benchmark",
        "git_commit": "abc",
        "tags": {"backbone": "gsasrec", "reportable": "true"},
    }
    trainer.train(
        model=_NaNModel(), train_loader=_FakeLoader(), val_loader=_FakeLoader(),
        test_loader=_FakeLoader(), optimizer=torch.optim.SGD(_NaNModel().parameters(), 0.01),
        epochs=1, criterion_fn=crit, eval_fn=evalfn,
        mlflow_params=legacy_mlflow_params,
    )

    for k in ("model", "seed", "phase", "git_commit"):
        assert k in captured, f"missing legacy param {k}"
    assert "tags" not in captured