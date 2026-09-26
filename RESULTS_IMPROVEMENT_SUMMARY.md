# Experimental Results and System Improvements

## 1. Main result

The system was upgraded from a five-trial experimental setup to a controlled
20-trial protocol with deterministic seeds, five-fold cross-validation, and
recorded experiment provenance.

For the regenerated FastAPI experiment:

| Method | Mean MSE | Standard Deviation |
|---|---:|---:|
| All-Features ANN | 0.187950 | 0.035194 |
| Random-Subset ANN | 0.264078 | 0.082200 |
| GA-Selected ANN | **0.123577** | 0.035504 |
| Untuned GA-XGBoost | 0.297551 | 0.053227 |
| Tuned GA-XGBoost | 0.284552 | 0.066058 |

The GA-selected ANN reduced FastAPI mean MSE by **34.25%** compared with the
All-Features ANN.

The paired two-sided Wilcoxon test reported:

- Statistic: `W = 0`
- p-value: `0.00000190735`
- Paired Cohen’s d: `1.9504`

This indicates a statistically significant improvement under the unified
20-trial protocol.

## 2. Results across repositories

| Repository | All-Features MSE | GA ANN MSE | Improvement |
|---|---:|---:|---:|
| FastAPI | 0.187950 | 0.123577 | **34.25% lower** |
| Flask | 0.391898 | 0.339299 | **13.42% lower** |
| Requests | 0.535466 | 0.397316 | **25.80% lower** |
| aiohttp | 1.560527 | 1.391042 | **10.86% lower** |
| Django* | 0.924691 | 0.936432 | 1.27% higher |

\* Django was evaluated using a deterministic 100-file bounded sample because
full-history traversal was not computationally bounded on the local machine.

The results show that GA improved the mean error on four of the five evaluated
datasets. The Django result is reported transparently as a case where GA did
not outperform the All-Features model.

## 3. Comparison with the NNCP-style baseline

The new NNCP-style baseline uses Category A structural features, Pearson
correlation filtering, the same ANN model, the same five-fold validation, and
the same 20-trial seed schedule.

| Repository | NNCP Mean MSE | GA ANN Mean MSE |
|---|---:|---:|
| Flask | 0.5071 | 0.3393 |
| Requests | 1.2693 | 0.3973 |
| FastAPI | 0.4968 | 0.1236 |

GA achieved lower error than the structural NNCP-style baseline on all three
original repositories.

## 4. What was improved

### Experimental reliability

- Increased the main comparison from 5 trials to 20 trials.
- Standardized all experiments on the same seed scheme:
  `trial_seed = base_seed + trial_index`.
- Added deterministic seeding for Python, NumPy, PyTorch, GA initialization,
  and cross-validation splits.
- Stored raw per-trial results instead of reporting only averages.

### Statistical quality

- Made Wilcoxon testing explicitly paired and two-sided.
- Added exact test statistics and p-values.
- Added paired Cohen’s effect size.
- Added mean, standard deviation, MAE, and R² reporting.

### Reproducibility

- Pinned Flask, Requests, FastAPI, Django, and aiohttp to exact commits.
- Added dataset SHA-256 hashes.
- Added source commit, configuration, seed, timestamp, and artifact paths to
  result files.
- Added experiment-specific GA checkpoints and result directories.
- Added reproducible repository acquisition and regeneration scripts.

### Fairness of model comparison

- Added an Optuna-tuned XGBoost baseline using a 50-trial budget.
- Preserved the untuned XGBoost result for transparency.
- Ensured ANN, GA, random-subset, and XGBoost comparisons use the same trial
  count and evaluation protocol.

### Broader validation

- Added an NNCP-style structural baseline.
- Evaluated two additional Python repositories.
- Validated the commit-message bug heuristic against GitHub pull-request
  labels.
- The heuristic achieved precision `0.0000`, recall `0.0000`, and Cohen’s
  kappa `-0.0396`, so it is not treated as reliable ground truth.

### Maintainability and deployment

- Added a Java metric collection entry point with explicit lexical metric
  definitions.
- Improved Docker dependency portability for Python 3.11 and CPU-only
  XGBoost.
- Added copy-pasteable README commands for dataset generation and experiment
  regeneration.
- Confirmed that manuscript files were not modified.

## 5. Overall conclusion

The implementation is now more reproducible, statistically stronger, and
fairer to competing models. The central result remains positive: GA-selected
features improved the ANN prediction error on FastAPI, Flask, Requests, and
aiohttp. At the same time, the Django result demonstrates that the method is
not universally superior, which provides a more balanced and scientifically
credible conclusion.

Detailed raw results are available in:

- `data/results/fastapi_dataset_clean/`
- `data/results/flask_t2_dataset_clean/`
- `data/results/requests_t2_dataset_clean/`
- `data/results/django_t2_dataset_clean/`
- `data/results/aiohttp_t2_dataset_clean/`
- `results/nncp_results.json`
