from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent.parent / "docs/research/report/final"
REPORT_MD = REPORT_DIR / "final_report.md"
REPORT_HTML = REPORT_DIR / "index.html"


def _read_report() -> str:
    assert REPORT_MD.is_file(), f"Missing final report markdown at {REPORT_MD}"
    return REPORT_MD.read_text(encoding="utf-8")


def test_final_report_deliverables_exist():
    assert REPORT_MD.is_file(), f"Missing final report markdown at {REPORT_MD}"
    assert REPORT_HTML.is_file(), f"Missing final report HTML at {REPORT_HTML}"


def test_report_defines_core_task_and_split_policy():
    content = _read_report()

    assert "next-distinct-course recommendation" in content
    assert "first encounter" in content
    assert "temporal leave-one-out" in content
    assert "train_sequences.csv" in content
    assert "val_sequences.csv" in content
    assert "test_sequences.csv" in content


def test_report_covers_rq1_to_rq3_only():
    content = _read_report()

    assert "**RQ1:**" in content
    assert "**RQ2:**" in content
    assert "**RQ3:**" in content
    assert "## 8. Kết luận" in content
    assert "Paired t-test trên test `NDCG@10` với Holm correction cho các neural baselines" in content


def test_report_documents_watch_alignment_and_reproducibility_limits():
    content = _read_report()

    assert "backward merge_asof" in content
    assert "0.13%" in content
    assert "MLflow" in content
    assert "artifact" in content
    assert "seed" in content
    assert "Giới hạn tái lập và tính hợp lệ" in content
