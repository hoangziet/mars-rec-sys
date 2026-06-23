import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rq4_report


def _write_comparison_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "comparison", "comparison_type", "comp_variant", "base_variant",
        "n_users", "comp_mean", "base_mean", "mean_difference", "relative_improvement",
        "relative_improvement_pct", "abs_mean_difference",
        "wins", "ties", "losses",
        "bootstrap_ci_low", "bootstrap_ci_high",
        "permutation_p", "cohens_d",
        "seed_t_stat", "seed_t_p", "n_seeds",
        "holm_adjusted_p", "significant",
    ]
    with open(path, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_final_report_uses_significant_as_sole_conclusion(monkeypatch, tmp_path):
    comparison_dir = tmp_path / "cmp"
    comparison_dir.mkdir()
    rows = [
        {
            "comparison": "V1 vs V0",
            "comparison_type": "primary",
            "comp_variant": "V1",
            "base_variant": "V0",
            "n_users": 100,
            "comp_mean": "0.30",
            "base_mean": "0.29",
            "mean_difference": "0.01",
            "relative_improvement": "0.0345",
            "relative_improvement_pct": "3.45",
            "abs_mean_difference": "0.01",
            "wins": 70,
            "ties": 5,
            "losses": 25,
            "bootstrap_ci_low": "0.002",
            "bootstrap_ci_high": "0.018",
            "permutation_p": "0.012",
            "cohens_d": "0.30",
            "seed_t_stat": "2.5",
            "seed_t_p": "0.04",
            "n_seeds": "5",
            "holm_adjusted_p": "0.04",
            "significant": "True",
        }
    ]
    _write_comparison_csv(comparison_dir / "rq4_comparison.csv", rows)

    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "rq4_report.py",
        "--benchmark-id", "test",
        "--comparison-dir", str(comparison_dir),
        "--output-dir", str(out),
    ])
    rq4_report.main()

    text = (out / "rq4_final_report.md").read_text()
    assert "statistically significant" in text
    assert "V1 vs V0" in text
    # No more dual-conclusion or significance_label language
    assert "Label" not in text or "| Label |" not in text
    assert "inconclusive" not in text.lower()


def test_final_report_says_not_significant_correctly(monkeypatch, tmp_path):
    comparison_dir = tmp_path / "cmp"
    comparison_dir.mkdir()
    rows = [
        {
            "comparison": "V2 vs V0",
            "comparison_type": "primary",
            "comp_variant": "V2",
            "base_variant": "V0",
            "n_users": 100,
            "comp_mean": "0.31",
            "base_mean": "0.30",
            "mean_difference": "0.01",
            "relative_improvement": "0.0333",
            "relative_improvement_pct": "3.33",
            "abs_mean_difference": "0.01",
            "wins": 40,
            "ties": 30,
            "losses": 30,
            "bootstrap_ci_low": "-0.001",
            "bootstrap_ci_high": "0.021",
            "permutation_p": "0.08",
            "cohens_d": "0.10",
            "seed_t_stat": "1.1",
            "seed_t_p": "0.30",
            "n_seeds": "5",
            "holm_adjusted_p": "0.30",
            "significant": "False",
        }
    ]
    _write_comparison_csv(comparison_dir / "rq4_comparison.csv", rows)

    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "rq4_report.py",
        "--benchmark-id", "test",
        "--comparison-dir", str(comparison_dir),
        "--output-dir", str(out),
    ])
    rq4_report.main()

    text = (out / "rq4_final_report.md").read_text()
    assert "not significant" in text
