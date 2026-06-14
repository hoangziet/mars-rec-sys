from __future__ import annotations

TRAINING_EXPERIMENTS = {
    "smoke": "mars_smoke",
    "benchmark": "mars_benchmark",
    "tuning": "mars_tuning",
    "ablation": "mars_ablation",
    "final": "mars_final",
}

SHARED_EXPERIMENTS = {
    "datasets": "mars_datasets",
    "reports": "mars_reports",
}

HEURISTIC_MODELS = {"popularity", "itemcf"}
SEQUENTIAL_BACKBONES = {"sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"}

ARTIFACT_PATHS = {
    "config_dir": "config",
    "dataset_dir": "dataset",
    "metrics_dir": "metrics",
    "plots_dir": "plots",
    "checkpoints_dir": "checkpoints",
    "reports_dir": "reports",
    "resolved_config": "config/resolved_config.yaml",
    "dataset_ref": "dataset/run_dataset_ref.json",
    "metrics_history": "metrics/history.json",
    "metrics_final": "metrics/final_metrics.json",
    "loss_plot": "plots/loss_plot.png",
    "metrics_plot": "plots/metrics_plot.png",
    "best_checkpoint": "checkpoints/best_model.pt",
    "run_summary": "reports/run_summary.json",
}


def get_experiment_name_for_phase(phase: str) -> str:
    try:
        return TRAINING_EXPERIMENTS[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLflow phase: {phase}") from exc


def infer_backbone(model_name: str) -> str:
    if model_name in HEURISTIC_MODELS:
        return "none"
    if model_name in SEQUENTIAL_BACKBONES:
        return model_name
    raise ValueError(f"Unsupported model for backbone inference: {model_name}")


def build_run_name(model_name: str, seed: int, *, variant: str, alpha: float | None = None) -> str:
    if alpha is None:
        return f"{model_name}-{variant}-s{seed}"
    return f"{model_name}-{variant}-a{alpha}-s{seed}"


def build_training_tags(
    *,
    model_name: str,
    phase: str,
    variant: str,
    git_commit: str,
    dataset_name: str,
    dataset_version: str,
    reportable: bool,
) -> dict[str, str]:
    return {
        "project": "mars-rec-sys",
        "phase": phase,
        "model": model_name,
        "variant": variant,
        "backbone": infer_backbone(model_name),
        "scope": "run",
        "artifact_class": "training",
        "git_commit": git_commit,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "reportable": str(reportable).lower(),
    }
