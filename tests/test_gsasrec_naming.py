import unittest

from run_all import build_run_record


class GSASRecNamingTest(unittest.TestCase):
    def test_gsasrec_run_record_contains_baseline_note(self):
        summary = {
            "test_results": {"HR@10": 0.1, "NDCG@10": 0.05, "HR@20": 0.2, "NDCG@20": 0.1},
            "best_val_ndcg": 0.05,
            "best_epoch": 1,
        }

        rec = build_run_record("gsasrec", 42, "random", 99, summary, "abc")

        self.assertIn("model_variant", rec)
        self.assertEqual(rec["model_variant"], "confidence_weighted_sasrec")


if __name__ == "__main__":
    unittest.main()
