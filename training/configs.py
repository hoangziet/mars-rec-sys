"""
training/configs.py
===================
Configuration helpers.  The primary source of truth is the Hydra YAML config
hierarchy under configs/.  This module provides thin wrappers that load those
YAML files into plain dicts for consumers that do not use @hydra.main
(scripts/train_all.py, scripts/predict.py, tests).
"""

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

COMMON_NEURAL_EPOCHS = 50
COMMON_EARLY_STOP_PATIENCE = 10
COMMON_EARLY_STOP_MIN_DELTA = 1e-4


def _load_model_yaml(model_name: str) -> dict:
    """Load a model YAML config file and return its contents as a plain dict."""
    path = CONFIG_DIR / "model" / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for model '{model_name}' at {path}")
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def build_model_config(model_name: str) -> dict:
    """Return the {model_kwargs, train_kwargs} dict for a model."""
    raw = _load_model_yaml(model_name)
    return {
        "model_kwargs": dict(raw.get("model_kwargs", {})),
        "train_kwargs": dict(raw.get("train_kwargs", {})),
    }


# Lazily-built cache of all model configs.
_MODEL_CONFIGS_CACHE: dict | None = None


def _build_all_configs() -> dict:
    models = ["sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf", "itemcf", "popularity"]
    return {name: build_model_config(name) for name in models}


def get_model_configs() -> dict:
    global _MODEL_CONFIGS_CACHE
    if _MODEL_CONFIGS_CACHE is None:
        _MODEL_CONFIGS_CACHE = _build_all_configs()
    return _MODEL_CONFIGS_CACHE


# Module-level alias for backward compatibility.
# Consumers that access training.configs.MODEL_CONFIGS get the lazy cache.
def __getattr__(name: str):
    if name == "MODEL_CONFIGS":
        return get_model_configs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


DEFAULT_SEED = 42
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "experiments"
