    

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

## RQ2–RQ4 Workflow

RQ2–RQ4 are gSASRec-only follow-up studies. RQ1 remains the benchmark
for model comparison, but it does not control the backbone used for
RQ2–RQ4. The backbone is frozen to `gsasrec` in each RQ2/RQ3/RQ4
script, and `rq4-init` re-validates that the RQ2 and RQ3 winner
artifacts also declare `gsasrec` as the backbone.

```text
RQ1 benchmark (any model)
   ↓
RQ2 alpha tuning on gSASRec
   ↓
RQ3 metadata tuning on gSASRec
   ↓
RQ4 final gSASRec ablation
```

The full flow:

```text
RQ1 benchmark
   ↓
make rq1-report           -> rq1_winner.json (for reporting/audit only;
                              not consumed by RQ2–RQ4)
   ↓
make rq2-tune             -> RQ2 alpha grid on gSASRec
   ↓
make rq2-report           -> rq2_best_alpha.json (backbone=gsasrec + provenance)
   ↓
make rq3-precompute       -> metadata/text artifacts
   ↓
make rq3-tune             -> RQ3 M0–M3 grid on gSASRec
   ↓
make rq3-report           -> rq3_best_variant.json (backbone=gsasrec + provenance)
   ↓
make rq4-init             -> frozen rq4_protocol_manifest.json
                            (backbone=gsasrec + baseline_variant + provenance)
   ↓
make rq4-ablation         -> V0–V3 × 10 seeds (gSASRec only)
   ↓
make rq4-collect          -> validates exact runs, on-disk per-user CSVs,
                              writes rq4_result_manifest.json
   ↓
make rq4-compare          -> user-level bootstrap/permutation + Holm correction
                              (uses explicit baseline_variant, not variants[0])
   ↓
make rq4-subgroup         -> subgroup metrics from per-user outputs
   ↓
make rq4-report           -> final markdown report
```

### RQ1 winner artifact (reporting/audit only)

`rq1_report` writes a small JSON file recording the RQ1 winner. It is
kept for reporting and audit, but it is no longer consumed by RQ2–RQ4:

```json
{
  "schema_version": 1,
  "benchmark_id": "rq1-2026-06-23",
  "winner_model": "gsasrec",
  "selection_metric": "best_val_ndcg_at_10",
  "selection_split": "val",
  "seed_set": [42, 123, 2024, 3407, 9999],
  "data_source": "/abs/path/data/processed",
  "preprocessing_version": "mars-preprocess-v1"
}
```

RQ2–RQ4 do not read this file. The backbone in RQ2/RQ3 is the
hardcoded constant `gsasrec` in each tune script; the backbone in RQ4
is whatever `rq4-init` recorded into the protocol manifest (also
`gsasrec`, re-validated by `rq4-ablation`).

### Fairness scope

RQ1 uses a **shared benchmark contract with declared model-specific
exceptions**:

- same processed split, item catalog, full-sort ranking contract, and primary metrics
- same benchmark seed campaign for stochastic models
- same validation-selection rule (`mean validation NDCG@10` across the campaign)
- same early-stopping rule where a train loop exists
- same common optimization recipe for neural models: `batch_size=256`, `epochs=50`, `lr=1e-3`, `beta2=0.98`, `weight_decay=1e-4`, `gradient_clip=5.0`, no warmup scheduler

Model-specific objectives and data-pipeline differences are still allowed
when they are part of the implementation contract and documented up front.

Current declared exceptions:

- `GRU4Rec` uses a paper-near pairwise ranking objective (`bpr_max`) rather than the full-catalog CE fallback.
- `BERT4Rec` keeps a simplified masking pipeline, so it remains an adapted benchmark implementation rather than a full paper-faithful reproduction.

### Provenance checks

RQ2/RQ3 winner artifacts carry `backbone`, `data_source`,
and `preprocessing_version` tags from the run that
produced them. `rq2-report` and `rq3-report` validate that **every
selected run agrees on all of these** and that the backbone is
`gsasrec` — any mismatch or missing tag fails the report rather than
silently writing a default. `rq4-init` re-checks that the RQ2 and RQ3
winner artifacts also declare `gsasrec` before freezing the protocol.

The RQ4 protocol uses **lightweight provenance** only:

- `preprocessing_version` and `data_source` from the validated RQ2/RQ3 winners
- `backbone = "gsasrec"`

There is **no SHA256 hashing** of the dataset manifest, configs, or text
embeddings, and **no git-commit runtime gate**. The research contract is
gSASRec-only; the protocol manifest is checked for backbone and
provenance consistency, not for byte-exact reproducibility of data or
code.

### RQ4 explicit baseline

The RQ4 protocol manifest declares an explicit `baseline_variant`
(default `V0`). `rq4-compare` reads this field and uses it as the
baseline — it never silently picks `variants[0]`. Reordering the
variant list or using a custom sweep will not silently change the
baseline.

### Atomic per-user artifact

`rq4-ablation` writes per-user CSVs through `scripts/rq4_per_user.py`:
write to temp file → validate → atomic rename → MLflow upload → only
then promote `per_user_complete=true` and `reportable=true`. `rq4-collect`
re-validates each on-disk CSV before including a run, so a stale or
partial file cannot be misread as valid.

### Temporal watch signal

`engagement_score` is attached by a temporal backward join: the pipeline uses
the latest explicit watch event for the same `(user_id, item_id)` with
`explicit.created_at <= implicit.created_at`. Interactions without a temporal
match receive `engagement_score = 0.0` and `has_watch_signal = false`.

### Required commands

```bash
make preprocess
make rq3-precompute
make rq2-tune
make rq2-report
make rq3-tune
make rq3-report
make rq4-init
make rq4-ablation
make rq4-collect
make rq4-compare
make rq4-subgroup
make rq4-report
```

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
make rq2-all
make rq3-all
make rq4-all
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
| RQ2 (tuning)                 | `mars_confidence_tuning`    |
| RQ3 (tuning)                 | `mars_metadata_tuning`      |
| RQ4 (final ablation)         | `mars_final_ablation`       |

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
  rq2_tune_alpha.py          RQ2 alpha grid (gSASRec only, backbone frozen in script)
  rq2_report.py              RQ2 reporter (writes rq2_best_alpha.json)
  rq3_precompute_embeddings.py
  rq3_tune_metadata.py       RQ3 metadata grid (gSASRec only, backbone frozen in script)
  rq3_report.py              RQ3 reporter (writes rq3_best_variant.json)
  rq4_init_protocol.py       freeze RQ4 protocol (--rq2-winners + --rq3-winners; backbone=gsasrec)
  rq4_ablation.py            RQ4 V0–V3 ablation runner
  rq4_per_user.py            atomic per-user CSV write helper
  rq4_collect.py             RQ4 collector (validates per-user files)
  rq4_compare.py             RQ4 statistical comparison
  rq4_subgroup.py            RQ4 subgroup metrics
  rq4_report.py              RQ4 final markdown report
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
tests/                       local pytest suite (tracked)
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
