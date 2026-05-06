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
