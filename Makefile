BENCHMARK_MODELS = sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf
RQ1_SEEDS = 42 123 2024 3407 9999
BENCHMARK_ID ?= rq1-v1
EXPECTED_NEURAL_RUNS ?= 5
REPORT_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/reports
PAIRWISE_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/stats
PREPROCESSING_VERSION ?= mars-preprocess-v1

.PHONY: preprocess rq1-smoke rq1-full rq1-report rq1-compare test

preprocess:
	uv run python data/preprocess.py

rq1-smoke:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds 42 --benchmark-id rq1-smoke --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-full:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds $(RQ1_SEEDS) --benchmark-id rq1-v1 --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-report:
	uv run python scripts/report_rq1.py --benchmark-id $(BENCHMARK_ID) --output-dir $(REPORT_OUTPUT_DIR) --expected-neural-runs $(EXPECTED_NEURAL_RUNS)

rq1-compare:
	uv run python scripts/compare_rq1.py --runs-file $(REPORT_OUTPUT_DIR)/rq1_runs.csv --summary-file $(REPORT_OUTPUT_DIR)/rq1_summary.json --output-dir $(PAIRWISE_OUTPUT_DIR) --expected-pairs $(EXPECTED_NEURAL_RUNS)

test:
	uv run pytest tests/ -v
