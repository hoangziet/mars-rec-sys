# mars-rec-sys
Course recommendation system on the MARS dataset

## 1. Python Environment Setup

This project uses `pyproject.toml` for dependency management and is pre-configured for `uv`.

**Step 1:** Install `uv` (if not already installed)
```bash
pip install uv
```

**Step 2:** Sync the environment and install dependencies (including PyTorch with CUDA support)
```bash
uv sync
```

## 2. Data Folders Setup

The project requires two main directories inside the root `data/` folder:

- **`data/raw/`**: The folder containing the raw original MARS data files (`explicit_ratings_en.csv`, `implicit_ratings_en.csv`, `items_en.csv`, `users_en.csv`).
- **`data/processed/`**: The folder containing the preprocessed data. You can run the following command to generate the processed files (such as `train.csv`, `val.csv`, `test.csv`, `dataset_stats.json`...):
  ```bash
  uv run python data/processed.py
  ```

**Sample directory structure:**
```text
data/
├── processed/       <-- Receives preprocessing results
├── raw/             <-- Download and place raw files here
└── processed.py     <-- Script to process data from raw -> processed
```

## 3. Train Models

### Train a single model

```bash
uv run python run_experiment.py <model_name> [options]
```

Available models: `sasrec`, `gsasrec`, `gru4rec`, `bert4rec`, `bprmf`, `itemcf`, `popularity`

```bash
# Train SASRec with default config
uv run python run_experiment.py sasrec

# Override hyperparameters
uv run python run_experiment.py gru4rec --epochs 50 --lr 5e-4 --batch_size 1024

# Train BPR-MF with custom seed
uv run python run_experiment.py bprmf --seed 123
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `--data_dir` | Path to processed data | `data/processed` |
| `--output_dir` | Path to save experiment artifacts | `experiments` |
| `--epochs` | Number of training epochs | (per model config) |
| `--lr` | Learning rate | (per model config) |
| `--batch_size` | Batch size | (per model config) |
| `--seed` | Random seed | `42` |

### Train all models + comparison

```bash
# Train all 7 models sequentially
uv run python run_all.py

# Train only selected models
uv run python run_all.py sasrec gru4rec bprmf
```

### Output structure

After training, artifacts are saved to `experiments/`:

```text
experiments/
├── sasrec/
│   ├── metrics.json       # Per-epoch: train_loss, val_loss, HR@10, NDCG@10, ...
│   ├── loss_plot.png      # Train/val loss curve
│   ├── metrics_plot.png   # HR@10, NDCG@10, HR@20, NDCG@20 curves
│   └── best_model.pt      # Best checkpoint (by val NDCG@10)
├── gru4rec/
│   └── ...
└── comparison/
    ├── comparison.json    # Aggregated test results for all models
    └── comparison.png     # Bar chart comparison
```

### Default hyperparameters

| Model | emb/hidden | layers | heads | dropout | batch_size | epochs | lr |
|-------|-----------|--------|-------|---------|------------|--------|-----|
| SASRec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| gSASRec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| GRU4Rec | emb=64, hid=128 | 1 | - | 0.2 | 512 | 20 | 1e-3 |
| BERT4Rec | 64 | 2 | 2 | 0.2 | 256 | 20 | 1e-3 |
| BPR-MF | 64 | - | - | - | 1024 | 20 | 1e-3 |
| Item-CF | top_k_sim=20 | - | - | - | - | - | - |
| Popularity | - | - | - | - | - | - | - |

All configs are centralized in `configs.py` — edit there to change defaults.

### Evaluation metrics

All models are evaluated with the same protocol: **1 positive + 99 random negatives = 100 candidates**, ranked by model score.

| Metric | Meaning |
|--------|---------|
| **HR@K** | Hit Rate — is the target item in top-K? (1 or 0) |
| **NDCG@K** | Normalized Discounted Cumulative Gain — penalizes lower ranks |

Reported at K=10 and K=20.
