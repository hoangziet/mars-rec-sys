# mars-rec-sys

Sequential recommendation research codebase with full-sort evaluation, Hydra-based configuration, and MLflow tracking.

This repository is organized for experiment reproducibility:

- dataset freeze and versioning
- standardized MLflow metadata and artifact layout
- benchmark orchestration for RQ1
- local-first training with remote MLflow tracking

## What This Repo Contains

- 5 trainable neural recommenders:
  - `sasrec`
  - `gsasrec`
  - `gru4rec`
  - `bert4rec`
  - `bprmf`
- 2 heuristic baselines:
  - `popularity`
  - `itemcf`
- unified preprocessing pipeline
- full-sort evaluation with `Recall@K` and `NDCG@K`
- dataset freeze workflow with canonical MLflow dataset records
- benchmark runner and RQ1 reporter

## Quick Start

Install dependencies:

```bash
uv sync
```

Run preprocessing once:

```bash
uv run python data/preprocess.py
```

Verify MLflow connectivity:

```bash
uv run python scripts/test_mlflow_connection.py
```

Train one neural model:

```bash
uv run python scripts/train.py model=sasrec
```

## Required Local Environment

Create `.env` for MLflow client access:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

Local training code talks only to MLflow. It does not require direct PostgreSQL or MinIO credentials.

## Training Workflows

### 1. Single-model run

Use `scripts/train.py` for:

- smoke checks
- tuning/debug runs
- one-off neural experiments

Examples:

```bash
uv run python scripts/train.py model=sasrec
uv run python scripts/train.py model=gru4rec seed=123
uv run python scripts/train.py model=sasrec phase=smoke reportable=false model.train_kwargs.epochs=1 model.train_kwargs.batch_size=32
```

Notes:

- `train.py` supports neural models only
- `phase` defaults to `benchmark`
- `reportable=true` requires a canonical dataset freeze record

### 2. Benchmark orchestration for RQ1

Use `scripts/train_all.py` as the benchmark runner.

It runs:

- neural models across multiple seeds
- heuristic models once per benchmark campaign

Smoke benchmark:

```bash
uv run python scripts/train_all.py \
  sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf \
  --seeds 42 \
  --benchmark-id rq1-smoke \
  --protocol-version rq1-v1
```

Full benchmark:

```bash
uv run python scripts/train_all.py \
  sasrec gsasrec gru4rec bert4rec bprmf popularity itemcf \
  --seeds 42 123 2024 3407 9999 \
  --benchmark-id rq1-v1 \
  --protocol-version rq1-v1
```

### 3. RQ1 reporting

Aggregate MLflow runs into benchmark tables:

```bash
uv run python scripts/report_rq1.py \
  --benchmark-id rq1-v1 \
  --output-dir report/rq1 \
  --expected-neural-runs 5
```

Reporter outputs:

- `rq1_runs.csv`
- `rq1_summary.csv`
- `rq1_summary.json`
- `rq1_table.md`

## Dataset Freeze Workflow

Canonical dataset versions are published to the MLflow experiment `mars_datasets`.

Freeze the current dataset as `mars-v1`:

```bash
uv run python scripts/publish_dataset_manifest.py --dataset-version mars-v1
```

This command will:

- compute `raw_data_hash`
- compute `processed_data_hash`
- compute `preprocessing_config_hash`
- validate the proposed version against the latest canonical dataset run
- publish a canonical dataset manifest to MLflow
- write a local freeze record at:
  - `data/processed/reports/dataset_freeze.json`

Reportable benchmark/final runs reference that freeze record automatically.

## Makefile Shortcuts

Common workflows are available via `Makefile`:

```bash
make freeze-v1
make rq1-smoke
make rq1-full
make rq1-report
make test
```

## MLflow Conventions

### Experiment taxonomy

| Phase | Experiment |
|-------|------------|
| `smoke` | `mars_smoke` |
| `benchmark` | `mars_benchmark` |
| `tuning` | `mars_tuning` |
| `ablation` | `mars_ablation` |
| `final` | `mars_final` |

Shared experiments:

- `mars_datasets` for canonical dataset manifests
- `mars_reports` for shared report bundles

### Run naming

Examples:

- `sasrec-base-s42`
- `gsasrec-base-s123`
- `rq1-v1-sasrec-base-s42`
- `dataset-mars-v1`

### Reportable dataset metadata

Reportable runs log:

- `dataset_name`
- `dataset_version`
- `dataset_run_id`
- `raw_data_hash`
- `processed_data_hash`
- `preprocessing_config_hash`

## Evaluation

The repository uses full-sort ranking.

Primary metric:

- `NDCG@10`

Secondary metrics:

- `Recall@10`
- `NDCG@20`
- `Recall@20`

Checkpoint selection uses validation `NDCG@10`, then the selected checkpoint is evaluated on test.

## Project Structure

```text
configs/                     Hydra config root
data/
  preprocess.py              raw -> processed pipeline
  raw/                       local input data
  processed/                 local processed data and freeze record
infra/                       VPS deployment reference and backup scripts
models/                      model implementations
pipeline/
  builder.py                 model/loss/eval factories
  loaders.py                 datasets and dataloaders
  metrics.py                 evaluation helpers
  optim.py                   optimizer/scheduler helpers
scripts/
  train.py                   single neural run
  train_all.py               benchmark runner for RQ1
  report_rq1.py              benchmark aggregator/reporter
  predict.py                 inference helper
  test_mlflow_connection.py  MLflow smoke test
  publish_dataset_manifest.py
  publish_report_bundle.py
training/
  configs.py
  trainer.py
  mlflow_contract.py
  mlflow_utils.py
  dataset_versioning.py
tests/                       local pytest suite (gitignored)
docs/                        local specs/plans/research notes (gitignored)
```

## Testing

Run the local test suite:

```bash
uv run pytest tests/ -v
```

## Infrastructure Notes

The project assumes a VPS-hosted MLflow stack with:

- PostgreSQL for tracking metadata
- MinIO for artifacts
- MLflow server with `--serve-artifacts`
- Nginx on the VPS host as the authenticated entrypoint

Local access is typically done through SSH tunnels to the VPS.

Reference deployment files live under `infra/`.

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
