BENCHMARK_MODELS = sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf
RQ1_SEEDS = 42 123 2024 3407 9999
BENCHMARK_ID ?= rq1-v1
SMOKE_BENCHMARK_ID ?= rq1-smoke
REPORT_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/reports
STATS_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/stats
PREPROCESSING_VERSION ?= mars-preprocess-v1

.PHONY: preprocess rq1-smoke rq1-full rq1-report rq1-compare test rq3-precompute rq3-tune rq3-report rq4-ablation rq4-compare rq4-report

preprocess:
	uv run python data/preprocess.py

rq1-smoke:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds 42 --benchmark-id $(SMOKE_BENCHMARK_ID) --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-full:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds $(RQ1_SEEDS) --benchmark-id $(BENCHMARK_ID) --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-report:
	uv run python scripts/report_rq1.py --benchmark-id $(BENCHMARK_ID) --output-dir $(REPORT_OUTPUT_DIR)

rq1-compare:
	uv run python scripts/rq1_compare.py \
		--runs-file $(REPORT_OUTPUT_DIR)/rq1_runs.csv \
		--summary-file $(REPORT_OUTPUT_DIR)/rq1_summary.json \
		--manifest experiments/benchmark/$(BENCHMARK_ID)/benchmark_manifest.json \
		--output-dir $(STATS_OUTPUT_DIR)

test:
	uv run pytest tests/ -v

# --- RQ2: Confidence Tuning ---
RQ2_ALPHAS ?= 0.0 0.25 0.5 1.0 2.0
RQ2_SEEDS ?= 42 123 2024
RQ2_BENCHMARK_ID ?= rq2-alpha-tune
RQ2_OUTPUT_DIR ?= experiments/rq2/$(RQ2_BENCHMARK_ID)

rq2-tune:
	uv run python scripts/rq2_tune_alpha.py --alphas $(RQ2_ALPHAS) --seeds $(RQ2_SEEDS) --benchmark-id $(RQ2_BENCHMARK_ID)

rq2-report:
	uv run python scripts/rq2_report.py --benchmark-id $(RQ2_BENCHMARK_ID) --output-dir $(RQ2_OUTPUT_DIR)

# --- RQ3: Metadata Tuning ---
RQ3_VARIANTS ?= M0 M1 M2 M3
RQ3_SEEDS ?= 42 123 2024
RQ3_BENCHMARK_ID ?= rq3-metadata-tune
RQ3_OUTPUT_DIR ?= experiments/rq3/$(RQ3_BENCHMARK_ID)

rq3-precompute:
	uv run python scripts/rq3_build_vocab.py
	uv run python scripts/rq3_precompute_embeddings.py

rq3-tune:
	uv run python scripts/rq3_tune_metadata.py --variants $(RQ3_VARIANTS) --seeds $(RQ3_SEEDS) --benchmark-id $(RQ3_BENCHMARK_ID)

rq3-report:
	uv run python scripts/rq3_report.py --benchmark-id $(RQ3_BENCHMARK_ID) --output-dir $(RQ3_OUTPUT_DIR)

# --- RQ4: Final Ablation ---
RQ4_SEEDS ?= 42 123 2024 3407 9999 7 21 77 314 1337
RQ4_BENCHMARK_ID ?= rq4-ablation
RQ4_OUTPUT_DIR ?= experiments/rq4/$(RQ4_BENCHMARK_ID)
RQ4_COMPARISON_DIR ?= experiments/rq4/$(RQ4_BENCHMARK_ID)

rq4-ablation:
	uv run python scripts/rq4_ablation.py --best-alpha $(RQ2_BEST_ALPHA) --best-variant $(RQ3_BEST_VARIANT) --seeds $(RQ4_SEEDS) --benchmark-id $(RQ4_BENCHMARK_ID)

rq4-compare:
	uv run python scripts/rq4_compare.py --runs-file $(RQ4_COMPARISON_DIR)/rq4_runs.csv --summary-file $(RQ4_COMPARISON_DIR)/rq4_summary.json --manifest experiments/benchmark/rq1-v1/benchmark_manifest.json --output-dir $(RQ4_COMPARISON_DIR)

rq4-report:
	uv run python scripts/rq4_report.py --benchmark-id $(RQ4_BENCHMARK_ID) --comparison-dir $(RQ4_COMPARISON_DIR) --output-dir $(RQ4_OUTPUT_DIR)
