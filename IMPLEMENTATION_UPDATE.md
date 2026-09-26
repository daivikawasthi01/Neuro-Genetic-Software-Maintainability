# Neuro-Genetic Software Maintainability Framework
## Implementation and Experimental Update

**Date:** 26 September 2026  
**Scope:** Tier 1 implementation from the revision and technical plans

## 1. Overview

The framework has been upgraded to improve experimental reliability,
reproducibility, statistical strength, and baseline fairness. The main
comparison was re-run on FastAPI using 20 trials instead of the previous
under-powered 5-trial protocol.

The implementation preserves the existing research pipeline:

```text
Repository mining → preprocessing → ANN tuning → GA feature selection
→ baseline comparison → statistical validation → reporting
```

## 2. New implementation work

### Reproducibility and provenance

- Added a shared reproducibility module for Python, NumPy, and PyTorch seeds.
- Added deterministic per-trial seeds using `base_seed + trial_index`.
- Added deterministic GA initialization and generation-level split seeds.
- Added repository commit hashes, configuration, timestamps, and dataset
  information to experiment outputs.
- Added `results/RUN_LOG.md` to record executed experiments and artifacts.
- Added `results/SEED_VERIFICATION.md` with repeatability checks.
- Added focused automated tests for seed behavior and metadata serialization.

### Experiment configuration

- Unified the main experiment default to 20 trials.
- Unified the five-fold stratified cross-validation protocol.
- Added the top-level `--seed` command-line option.
- Added dataset-specific result directories to prevent stale results from
  other repositories contaminating new experiments.
- Made GA checkpoints and interim results experiment-specific.

### Statistical reporting

The statistical module now explicitly reports:

- paired Wilcoxon signed-rank testing;
- two-sided alternative hypothesis;
- exact Wilcoxon statistic and p-value;
- paired Cohen’s d effect size;
- raw per-trial MSE values;
- means, standard deviations, and trial seeds.

### XGBoost baseline

- Added a 50-trial Optuna search for XGBoost hyperparameters.
- Preserved the original untuned XGBoost result for comparison.
- Added tuning for estimators, learning rate, tree depth, child weight,
  subsampling, column sampling, and regularization.

### Repository and environment reproducibility

- Added `scripts/fetch_repos.py`.
- Pinned Flask, Requests, and FastAPI to immutable commit hashes.
- Pinned Python dependencies to the verified working environment.
- Added reproducible commands to the project README.
- Added a repository audit documenting the current pipeline, feature schema,
  cache behavior, and experiment entry points.

### Backend startup improvement

- Updated `setup.sh` to install FastAPI backend dependencies.
- Updated `start.sh` to use the project virtual environment explicitly.
- Verified that the FastAPI backend starts and responds at
  `http://localhost:8000/api/status`.

## 3. Fresh FastAPI experiment

The FastAPI repository was freshly cloned and pinned to commit:

```text
192b12197eb04c2b4a691cce7d87261b21716714
```

Dataset details:

- Python files scanned: 1,138
- Valid dataset files: 1,136
- Cleaned dataset rows: 953
- Features after correlation filtering: 18
- Bug-prone files: 154, or 16.2%
- Correlation threshold: `|r| > 0.95`

The current implementation computes `weighted_methods_per_class` and
`docstring_presence` as actual features. This is documented as an audit
finding because the paper currently describes these features inconsistently.

## 4. Experimental results

All results below use 20 trials, seed `42`, and five-fold stratified
cross-validation.

| Method | Mean MSE | Std. Dev. |
|---|---:|---:|
| All-Features ANN | 0.187950 | 0.035194 |
| Random-Subset ANN | 0.264078 | 0.082200 |
| GA-Selected ANN | **0.123577** | 0.035504 |
| Untuned GA-XGBoost | 0.297551 | 0.053227 |
| Tuned GA-XGBoost | 0.284552 | 0.066058 |

### Statistical comparison: GA-ANN vs. All-Features ANN

- GA-ANN improvement: **34.25% lower MSE**
- Paired two-sided Wilcoxon statistic: `W = 0`
- p-value: `0.00000190735`
- Paired Cohen’s d: `1.9504`, classified as large

The GA-selected model therefore remained better than the All-Features model
under the equal-powered 20-trial protocol.

### GA feature-selection result

The GA selected 9 of 18 features:

- `avg_cyclomatic_complexity`
- `halstead_volume`
- `nesting_depth`
- `class_coupling`
- `docstring_presence`
- `commit_frequency`
- `code_age_days`
- `code_churn`
- `added_deleted_ratio`

The GA achieved a 50% reduction in the cleaned FastAPI feature space.

### Category ablation

The full 20-trial ablation produced these means:

| Feature categories | Mean MSE |
|---|---:|
| Structural only | 0.415484 |
| Textual only | 0.511490 |
| Evolutionary only | 0.349153 |
| Structural + Textual | 0.359854 |
| Structural + Evolutionary | **0.183963** |
| Textual + Evolutionary | 0.311774 |
| All categories | 0.187950 |

The results show that evolutionary features provide substantial predictive
value, especially when combined with structural features.

## 5. Main improvements over the previous implementation

| Area | Previous limitation | Improvement |
|---|---|---|
| Statistical power | Main comparison used only 5 trials | Main comparison now uses 20 trials |
| Randomness | Seed formulas differed across modules | One shared seed configuration |
| Statistical detail | Wilcoxon settings were implicit | Paired/two-sided settings and exact W are recorded |
| Reproducibility | No complete run manifest | Metadata, commit hashes, seeds, and run log added |
| Cache isolation | Checkpoints could be reused across runs | Dataset- and seed-specific checkpoints/results |
| XGBoost fairness | Only untuned XGBoost was reported | Tuned 50-trial XGBoost added and preserved alongside untuned results |
| Dataset revisions | Repository revisions were undocumented | Immutable repository commits recorded |
| Backend startup | `uvicorn` could be missing from the active shell | Backend dependencies and virtual-environment execution fixed |

## 6. Verification performed

- Python AST validation passed for modified modules.
- Reproducibility unit tests passed: 3 tests.
- Same-seed ANN runs produced identical metrics.
- Independent baseline and statistics runs produced bit-identical raw
  All-Features and GA-ANN trial arrays.
- All fresh JSON artifacts passed finite-value and metadata validation.
- README CLI commands and the new `--seed`/`--n-trials` options were checked.
- No manuscript `.tex` files were modified.

## 7. Artifacts

Fresh FastAPI artifacts are stored in:

```text
data/results/fastapi_dataset_clean/
├── ga_results.json
├── baseline_results.json
├── stats_results.json
└── ablation_results.json
```

Supporting documentation is stored in:

```text
results/REPO_AUDIT.md
results/REPOSITORIES.json
results/RUN_LOG.md
results/NUMBERS_CHANGED.md
results/SEED_VERIFICATION.md
results/XGBOOST_TUNING.md
```

## 8. Remaining work

Tier 2 has not been started. It remains available for future work and
includes:

- an NNCP-style numerical baseline;
- issue-tracker label validation;
- expansion to two additional Python repositories.

The current results should be used to update the manuscript only after the
research team reviews the changed numerical findings in
`results/NUMBERS_CHANGED.md`.
