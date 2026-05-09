import json
import tempfile
import unittest
from pathlib import Path

from scripts.calibration_suite import calibration_report_schema, main


class CalibrationSuiteTest(unittest.TestCase):
    def test_schema_contains_required_sections(self):
        schema = calibration_report_schema()
        self.assertIn("tiny_overfit", schema)
        self.assertIn("neg_mode_sensitivity", schema)
        self.assertIn("seed_stability", schema)

    def test_main_writes_report_with_data_quality_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "processed"
            out_path = Path(temp_dir) / "reports" / "calibration_report.json"
            data_dir.mkdir(parents=True)

            (data_dir / "dataset_stats.json").write_text(
                json.dumps({"n_users": 2, "n_items": 5}),
                encoding="utf-8",
            )
            (data_dir / "train.csv").write_text(
                "user_idx,item_sequence,seq_len,target,confidence\n"
                '1,"[1, 2, 3]",3,4,0.9\n'
                '2,"[2, 4]",2,5,0.7\n',
                encoding="utf-8",
            )
            (data_dir / "val.csv").write_text(
                "user_idx,train_seq,target\n"
                '1,"[1, 2, 3]",4\n'
                '2,"[2, 4]",5\n',
                encoding="utf-8",
            )
            (data_dir / "test.csv").write_text(
                "user_idx,train_seq,target\n"
                '1,"[1, 2, 3, 4]",5\n'
                '2,"[2, 4, 5]",1\n',
                encoding="utf-8",
            )

            exit_code = main(["--data_dir", str(data_dir), "--out", str(out_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_path.exists())

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("tiny_overfit", report)
            self.assertIn("neg_mode_sensitivity", report)
            self.assertIn("seed_stability", report)
            self.assertIn("data_quality", report)
            self.assertEqual(report["data_quality"]["n_users"], 2)
            self.assertEqual(report["data_quality"]["n_items"], 5)
            self.assertIn("cold_target_ratio", report["data_quality"])


if __name__ == "__main__":
    unittest.main()
