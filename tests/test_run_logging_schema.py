import unittest

from run_all import build_run_record, aggregate_records


class RunLoggingSchemaTest(unittest.TestCase):
    def test_build_run_record_schema_and_field_mapping(self):
        summary = {
            "test_results": {"HR@10": 0.33, "NDCG@10": 0.21},
            "best_val_ndcg": 0.123,
            "best_epoch": 7,
        }

        record = build_run_record(
            model_name="sasrec",
            seed=42,
            neg_mode="random",
            num_neg=99,
            summary=summary,
            commit_sha="abc123",
        )

        required_keys = {"exp_id", "model", "seed", "eval_protocol", "metrics", "train_summary", "git"}
        self.assertTrue(required_keys.issubset(record.keys()))

        self.assertEqual(record["exp_id"], "sasrec_random_seed42")
        self.assertEqual(record["model"], "sasrec")
        self.assertEqual(record["seed"], 42)

        self.assertEqual(record["eval_protocol"]["neg_mode"], "random")
        self.assertEqual(record["eval_protocol"]["num_neg"], 99)
        self.assertEqual(record["eval_protocol"]["target_position"], 0)

        self.assertEqual(record["metrics"], summary["test_results"])
        self.assertEqual(record["train_summary"]["best_val_ndcg10"], 0.123)
        self.assertEqual(record["train_summary"]["best_epoch"], 7)



class AggregateMetricsTest(unittest.TestCase):
    def test_aggregate_records_groups_by_model_and_neg_mode(self):
        records = [
            {
                "model": "sasrec",
                "eval_protocol": {"neg_mode": "random"},
                "metrics": {"HR@10": 0.1, "NDCG@10": 0.05},
            },
            {
                "model": "sasrec",
                "eval_protocol": {"neg_mode": "random"},
                "metrics": {"HR@10": 0.3, "NDCG@10": 0.15},
            },
            {
                "model": "gru4rec",
                "eval_protocol": {"neg_mode": "popularity"},
                "metrics": {"HR@10": 0.2, "NDCG@10": 0.1},
            },
        ]

        aggregated = aggregate_records(records)

        self.assertIn("sasrec::random", aggregated)
        self.assertIn("gru4rec::popularity", aggregated)
        self.assertAlmostEqual(aggregated["sasrec::random"]["HR@10"]["mean"], 0.2)
        self.assertAlmostEqual(aggregated["sasrec::random"]["HR@10"]["std"], 0.1)
        self.assertAlmostEqual(aggregated["sasrec::random"]["NDCG@10"]["mean"], 0.1)
        self.assertAlmostEqual(aggregated["sasrec::random"]["NDCG@10"]["std"], 0.05)
        self.assertAlmostEqual(aggregated["gru4rec::popularity"]["HR@10"]["mean"], 0.2)
        self.assertAlmostEqual(aggregated["gru4rec::popularity"]["HR@10"]["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
