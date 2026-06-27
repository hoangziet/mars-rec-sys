BENCHMARK_MODELS = sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf
RQ1_SEEDS = 42 123 2024 3407 9999
BENCHMARK_ID ?= rq1-v1
SMOKE_BENCHMARK_ID ?= rq1-smoke
DATA_DIR ?= data/processed
REPORT_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/reports
STATS_OUTPUT_DIR ?= experiments/benchmark/$(BENCHMARK_ID)/stats
PREPROCESSING_VERSION ?= mars-preprocess-v1
RQ4_BASELINE_VARIANT ?= V0

.PHONY: preprocess rq1-smoke rq1-full rq1-report rq1-compare test rq2-alpha rq2-alpha-report rq2-variants rq2-report rq2-compare rq2-all rq3-precompute rq3-tune rq3-report rq3-compare rq3-all rq4-init rq4-ablation rq4-collect rq4-compare rq4-subgroup rq4-report rq4-all

preprocess:
	uv run python data/preprocess.py

rq1-smoke:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds 42 --benchmark-id $(SMOKE_BENCHMARK_ID) --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-full:
	uv run python scripts/train_all.py $(BENCHMARK_MODELS) --seeds $(RQ1_SEEDS) --benchmark-id $(BENCHMARK_ID) --protocol-version rq1-v1 --preprocessing-version $(PREPROCESSING_VERSION)

rq1-report:
	uv run python scripts/rq1_report.py --benchmark-id $(BENCHMARK_ID) --output-dir $(REPORT_OUTPUT_DIR)

rq1-compare:
	uv run python scripts/rq1_compare.py \
		--runs-file $(REPORT_OUTPUT_DIR)/rq1_runs.csv \
		--summary-file $(REPORT_OUTPUT_DIR)/rq1_summary.json \
		--manifest experiments/benchmark/$(BENCHMARK_ID)/benchmark_manifest.json \
		--output-dir $(STATS_OUTPUT_DIR)

test:
	uv run pytest tests/ -v --ignore=tests/test_mlflow.py --ignore=tests/test_remote_mlflow_integration.py

# --- RQ2: Watch Integration ---
RQ2_ALPHAS ?= 0.25 0.5 1.0 2.0
RQ2_SEEDS ?= 42 123 2024
RQ2_VARIANT_SEEDS ?= 42 123 2024 3407 9999
RQ2_ALPHA_BENCHMARK_ID ?= rq2-watch-alpha
RQ2_VARIANT_BENCHMARK_ID ?= rq2-watch-variants
RQ2_ALPHA_REPORT_OUTPUT_DIR ?= experiments/rq2/$(RQ2_ALPHA_BENCHMARK_ID)/reports
RQ2_REPORT_OUTPUT_DIR ?= experiments/rq2/$(RQ2_VARIANT_BENCHMARK_ID)/reports
RQ2_STATS_OUTPUT_DIR ?= experiments/rq2/$(RQ2_VARIANT_BENCHMARK_ID)/stats

rq2-alpha:
	uv run python scripts/rq2_tune_alpha.py --alphas $(RQ2_ALPHAS) --seeds $(RQ2_SEEDS) --benchmark-id $(RQ2_ALPHA_BENCHMARK_ID) --data-dir $(DATA_DIR)

rq2-alpha-report:
	uv run python scripts/rq2_alpha_report.py --benchmark-id $(RQ2_ALPHA_BENCHMARK_ID) --output-dir $(RQ2_ALPHA_REPORT_OUTPUT_DIR)

rq2-variants:
	uv run python scripts/rq2_compare_variants.py --seeds $(RQ2_VARIANT_SEEDS) --benchmark-id $(RQ2_VARIANT_BENCHMARK_ID) --alpha-artifact $(RQ2_ALPHA_REPORT_OUTPUT_DIR)/rq2_best_alpha.json --data-dir $(DATA_DIR)

rq2-report:
	uv run python scripts/rq2_report.py --benchmark-id $(RQ2_VARIANT_BENCHMARK_ID) --alpha-artifact $(RQ2_ALPHA_REPORT_OUTPUT_DIR)/rq2_best_alpha.json --output-dir $(RQ2_REPORT_OUTPUT_DIR)

rq2-compare:
	uv run python scripts/rq2_compare.py --runs-file $(RQ2_REPORT_OUTPUT_DIR)/rq2_runs.csv --summary-file $(RQ2_REPORT_OUTPUT_DIR)/rq2_summary.json --output-dir $(RQ2_STATS_OUTPUT_DIR)

rq2-all: rq2-alpha rq2-alpha-report rq2-variants rq2-report rq2-compare

# --- RQ3: Metadata Tuning ---
RQ3_VARIANTS ?= M0 M1 M2 M3
RQ3_SEEDS ?= 42 123 2024 3407 9999
RQ3_BENCHMARK_ID ?= rq3-metadata-tune
RQ3_REPORT_OUTPUT_DIR ?= experiments/rq3/$(RQ3_BENCHMARK_ID)/reports
RQ3_STATS_OUTPUT_DIR ?= experiments/rq3/$(RQ3_BENCHMARK_ID)/stats

rq3-precompute:
	uv run python scripts/rq3_build_vocab.py --data-dir $(DATA_DIR)
	uv run python scripts/rq3_precompute_embeddings.py --data-dir $(DATA_DIR)

rq3-tune: rq3-precompute
	uv run python scripts/rq3_tune_metadata.py --rq2-winner $(RQ2_REPORT_OUTPUT_DIR)/rq2_best_watch.json --variants $(RQ3_VARIANTS) --seeds $(RQ3_SEEDS) --benchmark-id $(RQ3_BENCHMARK_ID) --data-dir $(DATA_DIR)

rq3-report:
	uv run python scripts/rq3_report.py --benchmark-id $(RQ3_BENCHMARK_ID) --rq2-winner $(RQ2_REPORT_OUTPUT_DIR)/rq2_best_watch.json --output-dir $(RQ3_REPORT_OUTPUT_DIR)

rq3-compare:
	uv run python scripts/rq3_compare.py --runs-file $(RQ3_REPORT_OUTPUT_DIR)/rq3_runs.csv --summary-file $(RQ3_REPORT_OUTPUT_DIR)/rq3_summary.json --output-dir $(RQ3_STATS_OUTPUT_DIR)

rq3-all: rq3-tune rq3-report rq3-compare

# --- RQ4: Final Ablation ---
RQ4_SEEDS ?= 42 123 2024 3407 9999 7 21 77 314 1337
RQ4_BENCHMARK_ID ?= rq4-ablation
RQ4_ROOT_DIR ?= experiments/rq4/$(RQ4_BENCHMARK_ID)
RQ4_REPORT_OUTPUT_DIR ?= $(RQ4_ROOT_DIR)/reports
RQ4_STATS_OUTPUT_DIR ?= $(RQ4_ROOT_DIR)/stats
RQ4_MANIFEST ?= $(RQ4_ROOT_DIR)/rq4_protocol_manifest.json
RQ2_WINNERS ?= $(RQ2_REPORT_OUTPUT_DIR)/rq2_best_watch.json
RQ3_WINNERS ?= $(RQ3_REPORT_OUTPUT_DIR)/rq3_best_variant.json

rq4-init:
	uv run python scripts/rq4_init_protocol.py \
		--benchmark-id $(RQ4_BENCHMARK_ID) \
		--rq2-winners $(RQ2_WINNERS) \
		--rq3-winners $(RQ3_WINNERS) \
		--baseline-variant $(RQ4_BASELINE_VARIANT) \
		--seeds $(RQ4_SEEDS) \
		--data-dir $(DATA_DIR) \
		--output-dir $(RQ4_ROOT_DIR)

rq4-ablation:
	uv run python scripts/rq4_ablation.py \
		--protocol $(RQ4_MANIFEST) \
		--data-dir $(DATA_DIR) \
		--output-dir experiments

rq4-collect:
	uv run python scripts/rq4_collect.py --benchmark-id $(RQ4_BENCHMARK_ID) --protocol $(RQ4_MANIFEST) --per-user-dir $(RQ4_ROOT_DIR)/per_user --data-dir $(DATA_DIR) --output-dir $(RQ4_STATS_OUTPUT_DIR)

rq4-compare:
	uv run python scripts/rq4_compare.py --per-user-dir $(RQ4_ROOT_DIR)/per_user --manifest $(RQ4_STATS_OUTPUT_DIR)/rq4_result_manifest.json --output-dir $(RQ4_STATS_OUTPUT_DIR)

rq4-subgroup:
	uv run python scripts/rq4_subgroup.py --per-user-dir $(RQ4_ROOT_DIR)/per_user --manifest $(RQ4_STATS_OUTPUT_DIR)/rq4_result_manifest.json --data-dir $(DATA_DIR) --output-dir $(RQ4_STATS_OUTPUT_DIR)

rq4-report:
	uv run python scripts/rq4_report.py --benchmark-id $(RQ4_BENCHMARK_ID) --comparison-dir $(RQ4_STATS_OUTPUT_DIR) --output-dir $(RQ4_REPORT_OUTPUT_DIR)

rq4-all: rq4-init rq4-ablation rq4-collect rq4-compare rq4-subgroup rq4-report
