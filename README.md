    

# mars-rec-sys

Sequential recommendation research codebase. Seven models from heuristics to Transformers, full-sort evaluation, MLflow tracking, lightweight RQ1 benchmark workflow.

## Repository Scope

Models:

- neural: `sasrec`, `gsasrec`, `gru4rec`, `bert4rec`, `bprmf`
- heuristic: `popularity`, `itemcf`

Metrics (full-sort against the entire item catalog):

- `NDCG@10`, `Recall@10`
- `NDCG@20`, `Recall@20`

## Quick Start

```bash
uv sync
make preprocess
```

## Workflow

Four-step pipeline: preprocess → benchmark runner → benchmark report → statistical comparison.

### 1. Preprocess

Run once per dataset change:

```bash
make preprocess
```

Produces `data/processed/` (train/val/test splits).

### 2. Benchmark runner

Runs all seven models and writes per-seed artifacts under `experiments/benchmark/<benchmark_id>/`.

```bash
make rq1-smoke
make rq1-full
```

- Neural models train on every seed in `--seeds`. With the default `RQ1_SEEDS = 42 123 2024 3407 9999`, you get five runs per neural model.
- Heuristic models (`popularity`, `itemcf`) run once per campaign because they are deterministic. The runner still produces a single output per heuristic model per `benchmark_id`.
- `benchmark_id` is the immutable campaign key. Interrupted campaigns with `status=running` can be safely resumed: `train_all.py` skips already-finished `(model, seed)` combinations and reruns only missing/failed ones. Once all expected runs finish, the manifest is auto-marked `completed`.
- `rq1_report.py` only accepts completed campaigns by default. A completed campaign with 5-seed results plus heuristic runs produces exactly the expected number of MLflow runs, preventing ambiguous seed-count or stale-run errors.

### 3. Benchmark report

Aggregates MLflow runs into `experiments/benchmark/<benchmark_id>/reports/`:

```bash
make rq1-report BENCHMARK_ID=rq1-smoke
make rq1-report BENCHMARK_ID=rq1-v1
```

Outputs:

- `rq1_runs.csv` — per-run metrics
- `rq1_summary.csv` — per-model mean ± std
- `rq1_summary.json` — same as CSV in JSON
- `rq1_table.md` — markdown table

### 4. Statistical comparison

The model ranked first by mean validation NDCG@10 is compared against every baseline.

Winner-versus-neural-baseline comparisons use two-sided paired t-tests across matched random seeds. Holm correction controls the family-wise error rate at α = 0.05.

ItemCF and Popularity are deterministic and are reported descriptively.

```bash
make rq1-compare BENCHMARK_ID=rq1-v1
```

Writes winner-versus-baseline stats to `experiments/benchmark/<benchmark_id>/stats/`:

- `rq1_winner_vs_all.csv` — one row per baseline with p-values and Holm-adjusted p-values
- `rq1_seed_pairs.csv` — per-seed paired values for every neural comparison
- `rq1_significance.md` — formatted summary table

## RQ2–RQ3 Workflow

RQ2–RQ3 are BERT4Rec-only follow-up studies.

The full flow:

```text
make rq2-alpha            -> RQ2 Stage A alpha grid on BERT4Rec-WL
   ↓
make rq2-alpha-report     -> reports/rq2_best_alpha.json
   ↓
make rq2-variants         -> RQ2 Stage B baseline/WL/WE/WLWE on BERT4Rec
   ↓
make rq2-report           -> reports/rq2_best_watch.json
   ↓
make rq2-compare          -> stats/rq2_statistical_comparison.csv
   ↓
make rq3-precompute       -> metadata/text artifacts
   ↓
make rq3-tune             -> RQ3 M0–M3 grid on BERT4Rec base (no watch)
   ↓
make rq3-report           -> reports/rq3_best_variant.json
   ↓
make rq3-compare          -> stats/rq3_statistical_comparison.csv
```

### Fairness scope

RQ1 uses a **shared benchmark contract with declared model-specific
exceptions**:

- same processed split, item catalog, full-sort ranking contract, and primary metrics
- same benchmark seed campaign for stochastic models
- same validation-selection rule (`mean validation NDCG@10` across the campaign)
- same early-stopping rule where a train loop exists
- same common optimization recipe for neural models: `batch_size=256`, `epochs=100`, `lr=1e-3`, `beta2=0.98`, `weight_decay=1e-4`, `gradient_clip=5.0`, no warmup scheduler

RQ1 is a next-distinct-course benchmark: preprocessing deduplicates `(user_id, item_id)` by first encounter, so validation and test targets must not already appear in the user history.

Model-specific objectives and data-pipeline differences are still allowed
when they are part of the implementation contract and documented up front.

Current declared exceptions:

- `GRU4Rec` uses a paper-near pairwise ranking objective (`bpr_max`) rather than the full-catalog CE fallback.
- `BERT4Rec` keeps a simplified masking pipeline, so it remains an adapted benchmark implementation rather than a full paper-faithful reproduction.

### Temporal watch signal

`engagement_score` is attached by a temporal backward join: the pipeline uses
the latest explicit watch event for the same `(user_id, item_id)` with
`explicit.created_at <= implicit.created_at`. Interactions without a temporal
match receive `engagement_score = 0.0` and `has_watch_signal = false`.

## Single-model runs

Use `scripts/train.py` for one-off neural runs, smoke checks, and tuning:

```bash
uv run python scripts/train.py model=sasrec
uv run python scripts/train.py model=gru4rec seed=123
```

`train.py` is a development tool, not part of the benchmark path.

## Model ranking

Models are ranked by validation `NDCG@10` (primary metric, checkpoint selection). Test metrics are for final evaluation only and are not used for ranking or early stopping.

## Data notes

**Engagement signal:** Engagement is attached by temporal backward join: for each implicit interaction, the pipeline uses the latest explicit watch event for the same `(user_id, item_id)` with `explicit.created_at <= implicit.created_at`. Interactions without a temporal match receive `engagement_score = 0.0` and `has_watch_signal = false`. Missing watch data defaults to 0.0. The confidence weighting formula is `confidence = 1 + alpha * clip(engagement, 0, 1)`.

**Metadata completeness:** Item metadata (`item_features/item_metadata.csv`) has varying missing rates: ~17% difficulty, ~15% theme, ~10% software, ~90% job, ~1% duration. Missing fields are encoded as MISSING tokens (index 1) in the vocabulary.

## Inference

Auto-discovers the best checkpoint from `experiments/benchmark/rq1-v1/`.

```bash
uv run python scripts/predict.py bert4rec --user_id 42 --top_k 10
uv run python scripts/predict.py sasrec --user_id 42 --show_titles --top_k 5
```

Or specify an explicit checkpoint:

```bash
uv run python scripts/predict.py bert4rec --checkpoint path/to/best_model.pt --user_id 42
```

## Makefile Shortcuts

```bash
make preprocess
make rq1-smoke
make rq1-report BENCHMARK_ID=rq1-smoke
make rq1-full
make rq1-report BENCHMARK_ID=rq1-v1
make rq1-compare BENCHMARK_ID=rq1-v1
make rq2-all
make rq3-all
make test
```

`rq1-report` and `rq1-compare` default to `BENCHMARK_ID=rq1-v1`.

## MLflow Conventions

### Training experiments

| Phase (script)               | Experiment                  |
| ---------------------------- | --------------------------- |
| Single-model smoke (`train.py phase=smoke`) | `mars_smoke`      |
| RQ1 smoke (`train_all.py`)   | `mars_benchmark`            |
| RQ1 (`benchmark`)            | `mars_benchmark`            |
| RQ2 alpha-tuning             | `mars_watch_alpha_tuning`   |
| RQ2 variant comparison       | `mars_watch_variant_comparison` |
| RQ3 (tuning)                 | `mars_metadata_tuning`      |

### Shared experiments

- `mars_reports` for shared report bundles

Run naming: `<benchmark-id>-<model>-<variant>-s<seed>` (e.g. `rq1-v1-sasrec-base-s42`).

## Local Environment

Create a local `.env`:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

Local code talks only to MLflow. It does not require direct PostgreSQL or MinIO credentials.

Verify MLflow connectivity:

```bash
uv run python scripts/test_mlflow_connection.py
```

## Testing

```bash
make test
```

## Project Structure

```text
data/
  preprocess.py              raw -> processed pipeline
  raw/                       local input data
  processed/                 local processed data
infra/                       VPS deployment reference and backup scripts
models/                      model implementations
pipeline/
  builder.py                 model/loss/eval factories
  loaders.py                 datasets and dataloaders
  metrics.py                 evaluation helpers
  optim.py                   optimizer/scheduler helpers
scripts/
  train.py                   single neural run (dev tool)
  train_all.py               benchmark runner for RQ1
  rq1_report.py              benchmark reporter (writes rq1_winner.json)
  rq1_compare.py             winner-versus-baseline statistical comparison
  rq2_tune_alpha.py          RQ2 Stage A alpha grid (BERT4Rec-WL)
  rq2_alpha_report.py        RQ2 alpha reporter (writes reports/rq2_best_alpha.json)
  rq2_compare_variants.py    RQ2 Stage B watch variant runner
  rq2_report.py              RQ2 reporter (writes reports/rq2_best_watch.json)
  rq2_compare.py             RQ2 statistical comparison
  rq3_precompute_embeddings.py
  rq3_tune_metadata.py       RQ3 metadata grid (BERT4Rec standalone, no watch)
  rq3_report.py              RQ3 reporter (writes rq3_best_variant.json)
  rq3_compare.py             RQ3 statistical comparison
  predict.py                 inference helper
  test_mlflow_connection.py  MLflow smoke test
  publish_report_bundle.py
training/
  configs.py
  trainer.py
  mlflow_contract.py
  mlflow_utils.py
  winner_artifact.py         RQ1 winner artifact contract
Makefile                     common local workflow shortcuts
tests/                       local pytest suite (gitignored)
docs/                        local specs/plans/research notes (gitignored)
```

## Infrastructure Notes

The project assumes a VPS-hosted MLflow stack with:

- PostgreSQL for tracking metadata
- MinIO for artifact storage
- MLflow server with `--serve-artifacts`
- Nginx on the VPS host as the authenticated entrypoint

Local access is typically done through SSH tunnels. Reference deployment files live under `infra/`.

## Implementation Notes

### SASRec

Pre-LN Transformer with causal self-attention and BCE loss.

### gSASRec

SASRec-style backbone with gBCE loss and multiple negatives.

### GRU4Rec

GRU encoder with a paper-near `bpr_max` ranking loss and sampled negatives.

### BERT4Rec

Bidirectional Transformer with masked item modeling; the current implementation uses a simplified masking pipeline rather than the canonical 80/10/10 corruption scheme.

### BPR-MF

Matrix factorization with Bayesian Personalized Ranking.
