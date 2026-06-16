BENCHMARK_MODELS = sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf
RQ1_SEEDS = 42 123 2024 3407 9999

.PHONY: freeze-v1 rq1-smoke rq1-full rq1-report test

freeze-v1:
	uv run python scripts/publish_dataset_manifest.py --dataset-version mars-v1

rq1-smoke:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds 42 --benchmark-id rq1-smoke --protocol-version rq1-v1

rq1-full:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds $(RQ1_SEEDS) --benchmark-id rq1-v1 --protocol-version rq1-v1

rq1-report:
	uv run python scripts/report_rq1.py --benchmark-id rq1-v1 --output-dir report/rq1 --expected-neural-runs 5

test:
	uv run pytest tests/ -v
