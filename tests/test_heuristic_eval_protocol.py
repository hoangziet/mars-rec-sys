import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataloader import get_eval_loader as real_get_eval_loader
from evaluate import evaluate_popularity
from run_all import build_item_popularity, run_heuristic_model


class HeuristicEvalProtocolTest(unittest.TestCase):
    def _write_processed_fixture(self, base_dir: Path):
        (base_dir / "dataset_stats.json").write_text(
            json.dumps({"n_users": 2, "n_items": 140}),
            encoding="utf-8",
        )
        (base_dir / "interactions.csv").write_text(
            "user_idx,item_idx\n"
            "1,1\n"
            "1,2\n"
            "1,3\n"
            "1,7\n"
            "1,8\n"
            "1,9\n"
            "1,10\n"
            "1,11\n"
            "1,12\n"
            "1,13\n"
            "1,14\n"
            "1,15\n"
            "1,16\n"
            "1,17\n"
            "1,18\n"
            "1,19\n"
            "1,20\n"
            "1,21\n"
            "1,22\n"
            "1,23\n"
            "1,24\n"
            "1,25\n"
            "1,26\n"
            "1,27\n"
            "1,28\n"
            "1,29\n"
            "1,30\n"
            "1,31\n"
            "1,32\n"
            "1,33\n"
            "1,34\n"
            "1,35\n"
            "1,36\n"
            "1,37\n"
            "1,38\n"
            "1,39\n"
            "1,40\n"
            "1,41\n"
            "1,42\n"
            "1,43\n"
            "1,44\n"
            "1,45\n"
            "1,46\n"
            "1,47\n"
            "1,48\n"
            "1,49\n"
            "1,50\n"
            "1,51\n"
            "1,52\n"
            "1,53\n"
            "1,54\n"
            "1,55\n"
            "1,56\n"
            "1,57\n"
            "1,58\n"
            "1,59\n"
            "1,60\n"
            "1,61\n"
            "1,62\n"
            "1,63\n"
            "1,64\n"
            "1,65\n"
            "1,66\n"
            "1,67\n"
            "1,68\n"
            "1,69\n"
            "1,70\n"
            "1,71\n"
            "1,72\n"
            "1,73\n"
            "1,74\n"
            "1,75\n"
            "1,76\n"
            "1,77\n"
            "1,78\n"
            "1,79\n"
            "1,80\n"
            "1,81\n"
            "1,82\n"
            "1,83\n"
            "1,84\n"
            "1,85\n"
            "1,86\n"
            "1,87\n"
            "1,88\n"
            "1,89\n"
            "1,90\n"
            "1,91\n"
            "1,92\n"
            "1,93\n"
            "1,94\n"
            "1,95\n"
            "1,96\n"
            "1,97\n"
            "1,98\n"
            "1,99\n"
            "1,100\n"
            "1,101\n"
            "2,4\n"
            "2,5\n"
            "2,6\n"
            "2,103\n"
            "2,104\n"
            "2,105\n"
            "2,106\n"
            "2,107\n"
            "2,108\n"
            "2,109\n"
            "2,110\n"
            "2,111\n"
            "2,112\n"
            "2,113\n"
            "2,114\n"
            "2,115\n"
            "2,116\n"
            "2,117\n"
            "2,118\n"
            "2,119\n"
            "2,120\n"
            "2,121\n"
            "2,122\n"
            "2,123\n"
            "2,124\n"
            "2,125\n"
            "2,126\n"
            "2,127\n"
            "2,128\n"
            "2,129\n"
            "2,130\n"
            "2,131\n"
            "2,132\n"
            "2,133\n"
            "2,134\n"
            "2,135\n"
            "2,136\n"
            "2,137\n"
            "2,138\n"
            "2,139\n"
            "2,140\n",
            encoding="utf-8",
        )
        (base_dir / "train.csv").write_text(
            "user_idx,item_sequence\n"
            '1,"[1, 2, 3]"\n'
            '2,"[4, 5, 6]"\n',
            encoding="utf-8",
        )
        eval_body = (
            "user_idx,train_seq,target\n"
            '1,"[1, 2, 3]",101\n'
            '2,"[4, 5, 6]",102\n'
        )
        (base_dir / "val.csv").write_text(eval_body, encoding="utf-8")
        (base_dir / "test.csv").write_text(eval_body, encoding="utf-8")

    def _assert_summary_contract(self, summary):
        self.assertIsInstance(summary["best_val_ndcg"], float)
        self.assertEqual(
            set(summary["test_results"].keys()),
            {"HR@10", "NDCG@10", "HR@20", "NDCG@20"},
        )
        for value in summary["test_results"].values():
            self.assertIsInstance(value, float)

    def test_evaluate_popularity_supports_ndarray_item_popularity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._write_processed_fixture(data_dir)
            stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
            item_popularity = build_item_popularity(data_dir, stats["n_items"])
            eval_loader = real_get_eval_loader(
                data_dir / "val.csv",
                stats,
                batch_size=2,
                max_len=5,
                num_neg=99,
                neg_mode="popularity",
                item_popularity=item_popularity,
            )

            ndarray_results = evaluate_popularity(item_popularity, eval_loader, k_list=(10, 20))
            mapping_results = evaluate_popularity(
                {idx: float(score) for idx, score in enumerate(item_popularity)},
                eval_loader,
                k_list=(10, 20),
            )

            self.assertEqual(
                set(ndarray_results.keys()),
                {"HR@10", "NDCG@10", "HR@20", "NDCG@20"},
            )
            for value in ndarray_results.values():
                self.assertIsInstance(value, float)
            self.assertEqual(ndarray_results, mapping_results)

    def test_popularity_uses_shared_eval_protocol_and_summary_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._write_processed_fixture(data_dir)
            stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
            item_popularity = build_item_popularity(data_dir, stats["n_items"])

            with patch("run_all.get_eval_loader", wraps=real_get_eval_loader) as get_eval_loader_mock:
                summary = run_heuristic_model(
                    model_name="popularity",
                    data_dir=data_dir,
                    stats=stats,
                    device="cpu",
                    output_dir=tmpdir,
                    model_kwargs={},
                    train_kwargs={"batch_size": 2, "max_len": 5, "num_neg": 99},
                    seed=7,
                    neg_mode="popularity",
                    item_popularity=item_popularity,
                )

            self._assert_summary_contract(summary)
            self.assertGreaterEqual(summary["best_val_ndcg"], 0.0)
            self.assertLessEqual(summary["best_val_ndcg"], 1.0)
            self.assertEqual(get_eval_loader_mock.call_count, 2)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["num_neg"], 99)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["max_len"], 5)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["neg_mode"], "popularity")
            self.assertIsNotNone(get_eval_loader_mock.call_args_list[0].kwargs["item_popularity"])
            self.assertTrue((data_dir / "popularity_model.json").exists())

    def test_itemcf_uses_shared_eval_protocol_and_summary_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self._write_processed_fixture(data_dir)
            stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))

            with patch("run_all.get_eval_loader", wraps=real_get_eval_loader) as get_eval_loader_mock:
                summary = run_heuristic_model(
                    model_name="itemcf",
                    data_dir=data_dir,
                    stats=stats,
                    device="cpu",
                    output_dir=tmpdir,
                    model_kwargs={"top_k_sim": 5},
                    train_kwargs={"batch_size": 2, "max_len": 5, "num_neg": 99},
                    seed=7,
                    neg_mode="random",
                )

            self._assert_summary_contract(summary)
            self.assertGreaterEqual(summary["best_val_ndcg"], 0.0)
            self.assertLessEqual(summary["best_val_ndcg"], 1.0)
            self.assertEqual(get_eval_loader_mock.call_count, 2)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["num_neg"], 99)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["max_len"], 5)
            self.assertEqual(get_eval_loader_mock.call_args_list[0].kwargs["neg_mode"], "random")
            self.assertTrue((data_dir / "itemcf_sim.json").exists())
            self.assertTrue((data_dir / "itemcf_history.json").exists())


if __name__ == "__main__":
    unittest.main()
