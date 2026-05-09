import tempfile
import unittest
from pathlib import Path

from scripts.data_quality_audit import run_audit


class DataQualityAuditTest(unittest.TestCase):
    def test_audit_reports_exact_metrics_for_multi_row_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "dataset_stats.json").write_text('{"n_users": 3, "n_items": 10}', encoding="utf-8")
            (base / "train.csv").write_text(
                "user_idx,item_sequence\n"
                "1,\"[1]\"\n"
                "2,\"[1,2,3]\"\n"
                "3,\"[2,3,4,5,6]\"\n",
                encoding="utf-8",
            )
            (base / "val.csv").write_text(
                "user_idx,train_seq,target\n"
                "1,\"[1]\",2\n"
                "2,\"[1,2,3]\",9\n",
                encoding="utf-8",
            )
            (base / "test.csv").write_text(
                "user_idx,train_seq,target\n"
                "1,\"[1]\",7\n"
                "2,\"[1,2,3]\",3\n",
                encoding="utf-8",
            )

            report = run_audit(base)

            self.assertEqual(report["n_users"], 3)
            self.assertEqual(report["n_items"], 10)
            self.assertEqual(report["train_seq_len_p50"], 3.0)
            self.assertEqual(report["train_seq_len_p90"], 4.6)
            self.assertEqual(report["cold_target_ratio"], 0.5)

    def test_audit_raises_clear_error_for_invalid_item_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "dataset_stats.json").write_text('{"n_users": 1, "n_items": 10}', encoding="utf-8")
            (base / "train.csv").write_text(
                "user_idx,item_sequence\n1,not-a-sequence\n",
                encoding="utf-8",
            )
            (base / "val.csv").write_text(
                "user_idx,train_seq,target\n1,\"[]\",1\n",
                encoding="utf-8",
            )
            (base / "test.csv").write_text(
                "user_idx,train_seq,target\n1,\"[]\",1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid item_sequence at row 1"):
                run_audit(base)


if __name__ == "__main__":
    unittest.main()
