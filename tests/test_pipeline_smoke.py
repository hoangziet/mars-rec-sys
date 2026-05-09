import json
import tempfile
import unittest
from pathlib import Path

from run_all import build_run_record


class PipelineSmokeTest(unittest.TestCase):
    def test_smoke_pipeline_outputs_run_records(self):
        summary = {
            "test_results": {"HR@10": 0.1, "NDCG@10": 0.05, "HR@20": 0.2, "NDCG@20": 0.1},
            "best_val_ndcg": 0.05,
            "best_epoch": 1,
        }
        record = build_run_record(
            model_name="popularity",
            seed=42,
            neg_mode="random",
            num_neg=99,
            summary=summary,
            commit_sha="smoke",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "comparison" / "run_records.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([record], f, indent=2)

            self.assertTrue(out_path.exists())
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["model"], "popularity")


class ReadmeProtocolDocTest(unittest.TestCase):
    def test_readme_mentions_neg_mode(self):
        txt = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("neg_mode", txt)

    def test_readme_mentions_implementation_fidelity_notes(self):
        txt = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("Implementation Fidelity Notes", txt)
        self.assertIn("confidence-weighted SASRec baseline", txt)


if __name__ == "__main__":
    unittest.main()
