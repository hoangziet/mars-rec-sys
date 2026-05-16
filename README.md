# mars-rec-sys

Sequential course recommendation system built on the MARS e-learning dataset.

Implements and compares 7 models — from simple heuristics to state-of-the-art Transformers — using a unified training and **full-sort evaluation** pipeline.

---

## Dataset

| Stat | Value |
|------|-------|
| Users | 15,989 |
| Items | 2,300 courses |
| Interactions | 264,880 |
| Sparsity | 99.3% |
| Split | Leave-one-out (val = second-last item, test = last item per user) |

Raw data: `data/raw/` — requires `implicit_ratings.csv` and `items.csv`.

---

## Models

| Model | Type | Loss | Key feature |
|-------|------|------|-------------|
| **SASRec** | Sequential / Transformer | BCE | Causal self-attention, Pre-LN, `sqrt(d)` embedding scaling |
| **gSASRec** | Sequential / Transformer | gBCE | Sampling-bias corrected BCE, 32 negatives |
| **GRU4Rec** | Sequential / RNN | Cross-Entropy | GRU with full-catalog CE loss, embedding dropout, `bias=False` |
| **BERT4Rec** | Sequential / Transformer | Cross-Entropy | Bidirectional encoder, masked item modelling, weight-tied head |
| **BPR-MF** | Matrix Factorization | BPR | Classic user-item embedding with L2 regularization |
| **Item-CF** | Heuristic | — | Item-item cosine similarity |
| **Popularity** | Heuristic | — | Global item frequency baseline |

---

## Evaluation Protocol

All models are evaluated using **full-sort ranking**: the target item is ranked against the **entire item catalog** (2,300 courses), with items from the user's training history masked to `-inf` before ranking.

This is stricter than the sampled-100-candidate protocol used in some original papers, and matches the RecBole standard for fair comparison.

| Metric | Description |
|--------|-------------|
| **HR@K** | Hit Rate — is the target item in the top-K recommendations? (0 or 1) |
| **NDCG@K** | Normalized Discounted Cumulative Gain — penalises lower ranks within top-K |

Reported at K = 10 and K = 20.

---

## Setup

**1. Install `uv`** (if not already installed):
```bash
pip install uv
```

**2. Install dependencies:**
```bash
uv sync
```

**3. Preprocess raw data:**
```bash
uv run python data/preprocess.py
```

Reads `data/raw/` and writes train / val / test splits plus `dataset_stats.json` to `data/processed/`.

---

## Training

### Single model

```bash
uv run python scripts/train.py <model_name> [options]
```

```bash
# Train with default config
uv run python scripts/train.py sasrec

# Override hyperparameters
uv run python scripts/train.py gsasrec --epochs 50 --lr 5e-4

# Custom seed
uv run python scripts/train.py bprmf --seed 123
```

| Flag | Description | Default |
|------|-------------|---------|
| `--data_dir` | Path to processed data | `data/processed` |
| `--output_dir` | Where to save artifacts | `experiments` |
| `--epochs` | Training epochs | per model config |
| `--lr` | Learning rate | per model config |
| `--batch_size` | Batch size | per model config |
| `--seed` | Random seed | `42` |

### All models + comparison report

```bash
# Train all 7 models sequentially
uv run python scripts/train_all.py

# Train a subset
uv run python scripts/train_all.py sasrec gsasrec bert4rec
```

### Inference

```bash
# Top-10 recommendations for user 42
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10

# Show course titles (requires items.csv)
uv run python scripts/predict.py sasrec --user_id 42 --top_k 10 --show_titles
```

---

## Output Structure

```
experiments/
├── sasrec/
│   ├── best_model.pt      # Best checkpoint (by val NDCG@10)
│   ├── metrics.json       # Per-epoch: train_loss, val_loss, HR@10, NDCG@10, ...
│   ├── loss_plot.png      # Train / val loss curves
│   └── metrics_plot.png   # HR@10, NDCG@10, HR@20, NDCG@20 curves
├── gru4rec/
│   └── ...
└── comparison/
    ├── comparison.json         # Test-set results per model
    ├── aggregate_metrics.json  # Aggregated metrics across models
    ├── run_records.json        # Full experiment metadata
    └── comparison.png          # Bar chart leaderboard
```

---

## Default Hyperparameters

| Model | emb / hidden | layers | heads | dropout | batch | epochs | lr |
|-------|-------------|--------|-------|---------|-------|--------|----|
| SASRec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| gSASRec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| GRU4Rec | emb=64, hid=128 | 1 | — | 0.2 | 512 | 20 | 1e-3 |
| BERT4Rec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| BPR-MF | 64 | — | — | — | 1024 | 20 | 1e-3 |

All configs live in `training/configs.py` — edit there to change defaults.

---

## Project Structure

```
mars-rec-sys/
├── training/
│   ├── configs.py          # Centralised hyperparameter configs for all models
│   └── trainer.py          # Unified training loop, checkpointing, experiment tracking
│
├── pipeline/
│   ├── loaders.py          # Dataset classes + DataLoader factories
│   │                       #   TrainSequenceDataset, MaskedSequenceDataset,
│   │                       #   BPRDataset, FullSortEvalDataset
│   ├── metrics.py          # Full-sort eval functions + HR/NDCG computation
│   └── builder.py          # Model / criterion / eval-fn factory functions
│
├── models/
│   ├── sasrec.py           # SASRec (Kang & McAuley, ICDM 2018)
│   ├── gsasrec.py          # gSASRec (Petrov & Macdonald, RecSys 2023)
│   ├── gru4rec.py          # GRU4Rec (Hidasi et al., ICLR 2016)
│   ├── bert4rec.py         # BERT4Rec (Sun et al., CIKM 2019)
│   ├── bprmf.py            # BPR-MF (Rendle et al., UAI 2009)
│   ├── itemcf.py           # Item-based Collaborative Filtering
│   └── popularity.py       # Popularity baseline
│
├── scripts/
│   ├── train.py            # Entry point: train a single model
│   ├── train_all.py        # Entry point: train all models + comparison report
│   └── predict.py          # Inference: top-K recommendations from checkpoint
│
├── data/
│   ├── preprocess.py       # Raw data → processed splits pipeline
│   ├── raw/                # Input: implicit_ratings.csv, items.csv, ...
│   └── processed/          # Output: train.csv, val.csv, test.csv, stats.json
│
├── tests/
│   ├── conftest.py         # sys.path setup for pytest
│   ├── test_metrics.py     # HR/NDCG correctness, _ranks_from_logits
│   ├── test_dataloader.py  # Dataset shapes, padding, history masking
│   └── test_models.py      # Forward pass, loss validity, no NaN, weight tying
│
└── docs/
    └── references/         # Reference implementations (SASRec, gSASRec, RecBole, ...)
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

49 tests covering metrics, data loading, and all 5 neural models.

---

## Implementation Notes

### SASRec
Pre-LN transformer with causal (unidirectional) self-attention. `sqrt(d)` scaling on item embeddings. 1-indexed positional embeddings (`padding_idx=0`). Padding positions are zeroed out after each sublayer to prevent NaN propagation. BCE loss with 1 random negative per positive.

### gSASRec
Architecturally identical to SASRec. Uses **gBCE loss** (generalised Binary Cross-Entropy) with logit transformation to correct for sampling bias in sparse data. `num_neg=32`, temperature `t=0.5` (paper defaults).

### GRU4Rec
GRU encoder with `bias=False`, embedding dropout, and Xavier initialisation (matching RecBole). Uses **Cross-Entropy loss over the full item catalog** — no negative sampling required. This is the correct setting for **user-based** sequential recommendation (as opposed to the original session-based BPR-max paper setting).

### BERT4Rec
Bidirectional Transformer encoder with masked item modelling (15% random mask). Input LayerNorm applied after embedding sum (BERT convention). MLM prediction head: `Linear → GELU → LayerNorm`. Weight-tied output projection with per-item bias. CE loss with `ignore_index=0` for padding.

### BPR-MF
User and item embeddings (1-indexed, `padding_idx=0`). Xavier uniform init. Bayesian Personalized Ranking loss with batch-level L2 regularisation applied to fetched embeddings only (not the full table).
