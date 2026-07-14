"""
scripts/aggregate_results.py
=============================
Aggregate RQ1, RQ2, RQ3 experiment results into a single JSON for visualisation.

Output: experiments/results.json

Usage:
    uv run python scripts/aggregate_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "experiments" / "results.json"

CAMPAIGNS = {
    "rq1": {
        "benchmark_id": "rq1-v1",
        "summary": "benchmark/rq1-v1/reports/rq1_summary.json",
        "runs": "benchmark/rq1-v1/reports/rq1_runs.csv",
        "stats": "benchmark/rq1-v1/stats/rq1_winner_vs_all.csv",
        "entity_key": "model",
    },
    "rq2": {
        "benchmark_id": "rq2-watch-variants",
        "summary": "rq2/rq2-watch-variants/reports/rq2_summary.json",
        "runs": "rq2/rq2-watch-variants/reports/rq2_runs.csv",
        "stats": "rq2/rq2-watch-variants/stats/rq2_statistical_comparison.csv",
        "entity_key": "variant",
    },
    "rq3": {
        "benchmark_id": "rq3-metadata-base-ce",
        "summary": "rq3/rq3-metadata-base-ce/reports/rq3_summary.json",
        "runs": "rq3/rq3-metadata-base-ce/reports/rq3_runs.csv",
        "stats": "rq3/rq3-metadata-base-ce/stats/rq3_statistical_comparison.csv",
        "entity_key": "variant",
    },
}


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_rq1(experiments_dir: Path) -> dict:
    cfg = CAMPAIGNS["rq1"]
    summary = _load_json(experiments_dir / cfg["summary"])
    runs = _load_csv(experiments_dir / cfg["runs"])
    stats = _load_csv(experiments_dir / cfg["stats"])

    runs_by_model: dict[str, list[dict]] = {}
    for r in runs:
        runs_by_model.setdefault(r["model"], []).append(r)

    models = []
    for entry in summary:
        model_name = entry["model"]
        per_seed = []
        for r in runs_by_model.get(model_name, []):
            seed_row = {"seed": int(r["seed"])}
            for k, v in r.items():
                if k in ("model", "seed", "run_id", "run_name"):
                    continue
                try:
                    seed_row[k] = float(v)
                except (ValueError, TypeError):
                    seed_row[k] = v
            per_seed.append(seed_row)

        models.append({
            "model": model_name,
            "rank": entry.get("validation_rank"),
            "runs": entry["runs"],
            "summary": {k: v for k, v in entry.items()
                        if k not in ("model", "runs", "validation_rank")},
            "per_seed": per_seed,
        })

    comparisons = []
    for row in stats:
        if not row.get("holm_adjusted_p_value", "").strip():
            continue
        comparisons.append({
            "winner": row.get("winner_model", row.get("winner", "")),
            "baseline": row.get("baseline_model", row.get("baseline", "")),
            "mean_diff": float(row["mean_difference"]),
            "holm_p": float(row["holm_adjusted_p_value"]),
            "significant": row["significant_after_holm"] == "True",
        })

    return {"benchmark_id": cfg["benchmark_id"], "models": models, "stat_comparisons": comparisons}


def _read_rq(experiments_dir: Path, rq_key: str) -> dict:
    cfg = CAMPAIGNS[rq_key]
    summary = _load_json(experiments_dir / cfg["summary"])
    runs = _load_csv(experiments_dir / cfg["runs"])
    stats = _load_csv(experiments_dir / cfg["stats"])

    runs_by_entity: dict[str, list[dict]] = {}
    for r in runs:
        runs_by_entity.setdefault(r[cfg["entity_key"]], []).append(r)

    entities = []
    for entry in summary:
        entity_name = entry[cfg["entity_key"]]
        per_seed = []
        for r in runs_by_entity.get(entity_name, []):
            seed_row = {"seed": int(r["seed"])}
            for k, v in r.items():
                if k in (cfg["entity_key"], "seed"):
                    continue
                try:
                    seed_row[k] = float(v)
                except (ValueError, TypeError):
                    seed_row[k] = v
            per_seed.append(seed_row)

        entities.append({
            cfg["entity_key"]: entity_name,
            "rank": entry["rank"],
            "n_seeds": entry["n_seeds"],
            "summary": {k: v for k, v in entry.items()
                        if k not in (cfg["entity_key"], "n_seeds", "rank")},
            "per_seed": per_seed,
        })

    comparisons = []
    for row in stats:
        if not row.get("holm_adjusted_p_value", "").strip():
            continue
        comparisons.append({
            "winner": row["winner"],
            "baseline": row["baseline"],
            "mean_diff": float(row["mean_difference"]),
            "holm_p": float(row["holm_adjusted_p_value"]),
            "significant": row["significant_after_holm"] == "True",
        })

    return {"benchmark_id": cfg["benchmark_id"], "variants": entities, "stat_comparisons": comparisons}


def main() -> None:
    experiments_dir = ROOT / "experiments"

    result = {
        "rq1": _read_rq1(experiments_dir),
        "rq2": _read_rq(experiments_dir, "rq2"),
        "rq3": _read_rq(experiments_dir, "rq3"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
