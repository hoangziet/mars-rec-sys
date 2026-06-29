from __future__ import annotations

SMOKE_EXPERIMENT_NAME = "mars_smoke"
BENCHMARK_EXPERIMENT_NAME = "mars_benchmark"
RQ2_EXPERIMENT_NAME = "mars_confidence_tuning"
RQ2_ALPHA_EXPERIMENT_NAME = "mars_watch_alpha_tuning"
RQ2_VARIANT_EXPERIMENT_NAME = "mars_watch_variant_comparison"
RQ3_EXPERIMENT_NAME = "mars_metadata_tuning"
RQ4_EXPERIMENT_NAME = "mars_final_ablation"

TRAINING_EXPERIMENTS = {
    "smoke": SMOKE_EXPERIMENT_NAME,
    "benchmark": BENCHMARK_EXPERIMENT_NAME,
    "rq2_tuning": RQ2_EXPERIMENT_NAME,
    "rq2_alpha_tuning": RQ2_ALPHA_EXPERIMENT_NAME,
    "rq2_variant_comparison": RQ2_VARIANT_EXPERIMENT_NAME,
    "rq3_tuning": RQ3_EXPERIMENT_NAME,
    "rq4_final": RQ4_EXPERIMENT_NAME,
}

SHARED_EXPERIMENTS = {
    "reports": "mars_reports",
}

HEURISTIC_MODELS = {"popularity", "itemcf"}
SEQUENTIAL_BACKBONES = {"sasrec", "gsasrec", "gru4rec", "bert4rec", "bprmf"}

ARTIFACT_PATHS = {
    "config_dir": "config",
    "metrics_dir": "metrics",
    "plots_dir": "plots",
    "checkpoints_dir": "checkpoints",
    "reports_dir": "reports",
    "resolved_config": "config/resolved_config.yaml",
    "metrics_history": "metrics/history.json",
    "metrics_final": "metrics/final_metrics.json",
    "loss_plot": "plots/loss_plot.png",
    "metrics_plot": "plots/metrics_plot.png",
    "best_checkpoint": "checkpoints/best_model.pt",
    "run_summary": "reports/run_summary.json",
}


def get_experiment_name_for_phase(phase: str) -> str:
    try:
        return {
            "smoke": SMOKE_EXPERIMENT_NAME,
            "benchmark": BENCHMARK_EXPERIMENT_NAME,
        }[phase]
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
        "reportable": str(reportable).lower(),
    }
