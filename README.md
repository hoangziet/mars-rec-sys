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
- Each `benchmark_id` must be a unique campaign ID. Do not reuse the same ID by only deleting the local output directory — MLflow runs persist and the reporter will collect both old and new runs, causing duplicate seed or wrong run count errors. Always pick a new `benchmark_id` for a fresh campaign.

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

**Engagement signal:** `engagement_score` in `train_sequences.csv` is a per-user-item aggregate derived from `watch_percentage` in explicit ratings. It uses the global max watch percentage per (user, item) pair, not a temporally aligned per-event signal. Missing watch data defaults to 0.0. The confidence weighting formula is `confidence = 1 + alpha * clip(engagement, 0, 1)`.

**Metadata completeness:** Item metadata (`item_features/item_metadata.csv`) has varying missing rates: ~17% difficulty, ~15% theme, ~10% software, ~90% job, ~1% duration. Missing fields are encoded as MISSING tokens (index 1) in the vocabulary.

## Inference

```bash
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10 --show_titles
```

## Makefile Shortcuts

```bash
make preprocess
make rq1-smoke
make rq1-report BENCHMARK_ID=rq1-smoke
make rq1-full
make rq1-report BENCHMARK_ID=rq1-v1
make rq1-compare BENCHMARK_ID=rq1-v1
make test
```

`rq1-report` and `rq1-compare` default to `BENCHMARK_ID=rq1-v1`.

## MLflow Conventions

### Training experiments

| Phase | Experiment |
|-------|------------|
| `smoke` | `mars_smoke` |
| `benchmark` | `mars_benchmark` |
| `tuning` | `mars_tuning` |
| `ablation` | `mars_ablation` |
| `final` | `mars_final` |

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
  report_rq1.py              benchmark reporter
  rq1_compare.py             winner-versus-baseline statistical comparison
  predict.py                 inference helper
  test_mlflow_connection.py  MLflow smoke test
  publish_report_bundle.py
training/
  configs.py
  trainer.py
  mlflow_contract.py
  mlflow_utils.py
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

GRU encoder with cross-entropy loss over the full item catalog.

### BERT4Rec

Bidirectional Transformer with masked item modeling.

### BPR-MF

Matrix factorization with Bayesian Personalized Ranking.
