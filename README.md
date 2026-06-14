# mars-rec-sys

Sequential course recommendation with full-sort evaluation. 7 models from heuristics to Transformers, Hydra configs, MLflow tracking.

## Quick Start

```bash
uv sync                          # install deps
uv run python data/preprocess.py # run preprocessing once
uv run python scripts/train.py model=sasrec
```

## Models

| Model | Type | Reference |
|-------|------|-----------|
| **SASRec** | Sequential Transformer | Kang & McAuley, ICDM 2018 |
| **gSASRec** | Sequential Transformer | Petrov & Macdonald, RecSys 2023 |
| **GRU4Rec** | Sequential RNN | Hidasi et al., ICLR 2016 |
| **BERT4Rec** | Sequential Transformer | Sun et al., CIKM 2019 |
| **BPR-MF** | Matrix Factorization | Rendle et al., UAI 2009 |
| **Item-CF** | Heuristic | — |
| **Popularity** | Heuristic | — |

## Usage

### Single model

```bash
uv run python scripts/train.py model=sasrec
uv run python scripts/train.py model=gru4rec model.train_kwargs.epochs=100
uv run python scripts/train.py model=bprmf seed=123
```

Hydra overrides: `model=sasrec`, `model.train_kwargs.epochs=N`, `model.train_kwargs.lr=X`, `seed=N`.

Neural models: `sasrec | gsasrec | gru4rec | bert4rec | bprmf`
Heuristic models: `popularity | itemcf` (via `train_all.py`)

### All models

```bash
uv run python scripts/train_all.py
uv run python scripts/train_all.py sasrec gsasrec bert4rec
```

### Inference

```bash
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10 --show_titles
```

## Evaluation

All models use **full-sort ranking**: the target item is ranked against the entire item catalog, with items from the user's training history masked before ranking.

| Metric | Description |
|--------|-------------|
| Recall@K | Is the target in the top-K? |
| NDCG@K | Normalized Discounted Cumulative Gain — penalizes lower ranks |

Reported at K = 10 and K = 20. Primary metric: **NDCG@10** for checkpoint selection, early stopping, and model comparison.

Split: temporal leave-one-out — last interaction per user = test target, second-last = validation target.

## MLflow Tracking

Remote experiment tracking via MLflow.

```bash
# Required in .env
MLFLOW_TRACKING_URI=http://127.0.0.1:8080
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

**Experiment taxonomy:**

| Phase | Experiment |
|-------|-----------|
| smoke | `mars_smoke` |
| benchmark | `mars_benchmark` |
| tuning | `mars_tuning` |
| ablation | `mars_ablation` |
| final | `mars_final` |

Shared experiments: `mars_datasets` (canonical dataset manifests), `mars_reports` (aggregate result bundles).

```bash
uv run python scripts/test_mlflow_connection.py      # remote connectivity smoke test
uv run python scripts/publish_dataset_manifest.py     # publish dataset manifest
uv run python scripts/publish_report_bundle.py        # publish report bundle
```

See `docs/superpowers/specs/` for the full MLflow infrastructure specification.

## Project Structure

```
├── training/
│   ├── configs.py              # Hyperparameters
│   ├── trainer.py              # Unified training loop, checkpointing
│   ├── mlflow_contract.py      # Experiment taxonomy, run naming, tags
│   └── mlflow_utils.py         # MLflow configuration, validation helpers
│
├── pipeline/
│   ├── loaders.py              # Dataset classes, DataLoader factories
│   ├── builder.py              # Model / criterion / eval-fn factories
│   ├── metrics.py              # Full-sort eval, Recall/NDCG
│   └── optim.py                # Optimizer / scheduler helpers
│
├── models/
│   ├── sasrec.py, gsasrec.py, gru4rec.py
│   ├── bert4rec.py, bprmf.py
│   └── itemcf.py, popularity.py
│
├── scripts/
│   ├── train.py                # Hydra CLI: single-model training
│   ├── train_all.py            # All-models + comparison report
│   ├── predict.py              # Top-K inference from checkpoint
│   ├── test_mlflow_connection.py
│   ├── publish_dataset_manifest.py
│   └── publish_report_bundle.py
│
├── data/
│   ├── preprocess.py           # Raw → processed pipeline
│   ├── raw/                    # Input data
│   └── processed/              # Train/val/test splits, stats
│
├── configs/                    # Hydra YAML configs
├── infra/                      # VPS stack: compose, env, nginx, backup scripts
├── tests/                      # Pytest suite
└── docs/                       # Specs, plans, reports
```

## Infra

Single-VPS stack: PostgreSQL (tracking metadata), MinIO (artifact storage), MLflow server, Nginx (auth + reverse proxy).

```bash
docker compose -f infra/compose.yaml up -d
```

See `infra/` for compose, env templates, backup scripts, and restore checklist.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Implementation Notes

### SASRec
Pre-LN transformer, causal self-attention, `sqrt(d)` embedding scaling. 1-indexed positional embeddings (`padding_idx=0`). Padding positions zeroed after each sublayer. BCE loss.

### gSASRec
Same backbone as SASRec. gBCE loss with sampling-bias correction. `num_neg=32`, temperature `t=0.5`.

### GRU4Rec
GRU encoder, `bias=False`, embedding dropout, Xavier init. Cross-Entropy loss over full catalog — no negative sampling.

### BERT4Rec
Bidirectional Transformer, masked item modelling. Input LayerNorm on embedding sum. Weight-tied output projection with per-item bias. CE loss with `ignore_index=0` for padding.

### BPR-MF
User and item embeddings (1-indexed). Xavier uniform init. Bayesian Personalized Ranking with batch-level L2 regularization.
