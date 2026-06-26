"""
scripts/rq4_subgroup.py
=======================
RQ4: Subgroup analysis on per-user results.

Reads per-user CSVs from rq4_ablation, derives subgroups from train data,
computes NDCG@10 per subgroup per variant, and reports improvement over V0.

Groups:
    - Users with watch signal vs without
    - Short history vs long history (median split)
    - Head items vs tail items (top 20% vs bottom 80% by train interaction count)
    - Items with complete metadata vs missing metadata

Usage:
    uv run python scripts/rq4_subgroup.py --per-user-dir ... --data-dir data/processed --manifest ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.loaders import parse_seq

PRIMARY_METRIC = "ndcg_at_10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4: subgroup analysis.")
    parser.add_argument("--per-user-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _derive_subgroups(data_dir: Path) -> dict:
    train_df = pd.read_csv(data_dir / "splits" / "train_sequences.csv")

    has_watch = {}
    if "watch_signal_sequence" in train_df.columns:
        for _, row in train_df.iterrows():
            uid = int(row["user_idx"])
            seq = parse_seq(row["watch_signal_sequence"])
            has_watch[uid] = any(v == 1 for v in seq) if seq else False
    else:
        for _, row in train_df.iterrows():
            uid = int(row["user_idx"])
            has_watch[uid] = False

    seq_lengths = dict(zip(train_df["user_idx"], train_df["sequence_length"]))
    median_len = int(train_df["sequence_length"].median())

    item_counts: dict[int, int] = {}
    for _, row in train_df.iterrows():
        seq = parse_seq(row.item_sequence)
        for item in seq:
            item_counts[item] = item_counts.get(item, 0) + 1
    sorted_items = sorted(item_counts.items(), key=lambda x: -x[1])
    n_head = max(1, int(len(sorted_items) * 0.2))
    head_items = {item for item, _ in sorted_items[:n_head]}

    meta_path = data_dir / "item_features" / "item_metadata.csv"
    complete_meta_items = set()
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        for _, row in meta_df.iterrows():
            missing = sum(1 for col in ["difficulty", "theme", "software", "job", "type"] if pd.isna(row.get(col)))
            if missing == 0:
                complete_meta_items.add(int(row["item_idx"]))

    return {
        "has_watch": has_watch,
        "seq_lengths": seq_lengths,
        "median_len": median_len,
        "head_items": head_items,
        "complete_meta_items": complete_meta_items,
    }


def _assign_subgroup(row, subgroups: dict) -> dict:
    uid = int(row["user_idx"])
    target = int(row["target_item"])

    watch = "has_watch" if subgroups["has_watch"].get(uid, False) else "no_watch"
    seq_len = subgroups["seq_lengths"].get(uid, 0)
    history = "short" if seq_len <= subgroups["median_len"] else "long"
    popularity = "head" if target in subgroups["head_items"] else "tail"
    meta = "complete_meta" if target in subgroups["complete_meta_items"] else "missing_meta"

    return {"watch": watch, "history": history, "popularity": popularity, "meta": meta}


def _compute_subgroup_metrics(per_user: pd.DataFrame, group_col: str) -> list[dict]:
    """Compute mean NDCG@10 per (subgroup, variant), averaging per user across seeds first."""
    results = []
    # Average per user across seeds first
    user_avg = per_user.groupby(["variant", "user_idx", "target_item", group_col])[PRIMARY_METRIC].mean().reset_index()

    for group_val in sorted(user_avg[group_col].unique()):
        subset = user_avg[user_avg[group_col] == group_val]
        for variant in sorted(subset["variant"].unique()):
            v_subset = subset[subset["variant"] == variant]
            values = v_subset[PRIMARY_METRIC].values
            results.append({
                "group": group_col,
                "subgroup": group_val,
                "variant": variant,
                "n_users": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            })
    return results


def _compute_improvements(subgroup_metrics: list[dict], baseline_variant: str) -> list[dict]:
    """Compute improvement over the explicit baseline for each subgroup."""
    by_key = {}
    for r in subgroup_metrics:
        by_key[(r["group"], r["subgroup"], r["variant"])] = r

    improvements = []
    for r in subgroup_metrics:
        if r["variant"] == baseline_variant:
            continue
        baseline = by_key.get((r["group"], r["subgroup"], baseline_variant))
        if baseline is None:
            continue
        diff = r["mean"] - baseline["mean"]
        rel = diff / baseline["mean"] if baseline["mean"] != 0 else None
        improvements.append({
            "group": r["group"],
            "subgroup": r["subgroup"],
            "variant": r["variant"],
            "baseline_variant": baseline_variant,
            "baseline_mean": baseline["mean"],
            "variant_mean": r["mean"],
            "absolute_improvement": diff,
            "relative_improvement": rel,
            "n_users": r["n_users"],
        })
    return improvements


def main() -> None:
    args = parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    variants = manifest["variants"]
    baseline_variant = manifest.get("baseline_variant", "V0")
    seeds = [int(s) for s in manifest["neural_seeds"]]
    data_dir = Path(args.data_dir)
    per_user_dir = Path(args.per_user_dir)

    # Load per-user data
    frames = []
    for variant in variants:
        for seed in seeds:
            path = per_user_dir / f"{variant}_s{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing per-user file: {path}")
            df = pd.read_csv(path)
            frames.append(df)
    per_user = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(per_user)} per-user rows across {len(variants)} variants x {len(seeds)} seeds")

    # Derive subgroups
    subgroups = _derive_subgroups(data_dir)

    # Assign subgroup labels
    labels = per_user.apply(lambda row: _assign_subgroup(row, subgroups), axis=1, result_type="expand")
    per_user = pd.concat([per_user, labels], axis=1)

    # Compute metrics per subgroup
    all_metrics = []
    for group_col in ["watch", "history", "popularity", "meta"]:
        all_metrics.extend(_compute_subgroup_metrics(per_user, group_col))

    improvements = _compute_improvements(all_metrics, baseline_variant=baseline_variant)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write subgroup metrics CSV
    metrics_fields = ["group", "subgroup", "variant", "n_users", "mean", "std"]
    pd.DataFrame(all_metrics).to_csv(output_dir / "rq4_subgroup_metrics.csv", index=False)

    # Write improvements CSV
    if improvements:
        pd.DataFrame(improvements).to_csv(output_dir / "rq4_subgroup_improvements.csv", index=False)

    # Write Markdown
    with open(output_dir / "rq4_subgroup_analysis.md", "w") as f:
        f.write("# RQ4 Subgroup Analysis\n\n")

        for group_col in ["watch", "history", "popularity", "meta"]:
            f.write(f"## {group_col}\n\n")
            group_metrics = [r for r in all_metrics if r["group"] == group_col]
            group_imps = [r for r in improvements if r["group"] == group_col]

            # Table: mean per variant
            subgroups_list = sorted({r["subgroup"] for r in group_metrics})
            f.write(f"| Subgroup | {' | '.join(variants)} |\n")
            f.write(f"| --- | {' | '.join(['---:'] * len(variants))} |\n")
            for sg in subgroups_list:
                vals = []
                for v in variants:
                    match = [r for r in group_metrics if r["subgroup"] == sg and r["variant"] == v]
                    vals.append(f"{match[0]['mean']:.4f}" if match else "-")
                f.write(f"| {sg} | {' | '.join(vals)} |\n")

            # Improvement table
            if group_imps:
                f.write(f"\n### Improvement over {baseline_variant}\n\n")
                f.write("| Subgroup | Variant | Δ | Relative | N |\n")
                f.write("| --- | --- | ---: | ---: | ---: |\n")
                for imp in group_imps:
                    rel = f"{imp['relative_improvement']*100:.2f}%" if imp["relative_improvement"] is not None else "-"
                    f.write(f"| {imp['subgroup']} | {imp['variant']} | {imp['absolute_improvement']:.6f} | {rel} | {imp['n_users']} |\n")
            f.write("\n")

    # Save thresholds
    thresholds = {
        "median_seq_length": subgroups["median_len"],
        "head_item_fraction": 0.2,
        "n_head_items": len(subgroups["head_items"]),
        "n_complete_meta_items": len(subgroups["complete_meta_items"]),
        "n_users_with_watch": sum(1 for v in subgroups["has_watch"].values() if v),
    }
    (output_dir / "rq4_subgroup_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"Subgroup analysis: {len(all_metrics)} metric rows, {len(improvements)} improvement rows")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
