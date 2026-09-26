# Experiment Run Log

Entries are added by the experiment modules with task, script, configuration,
seed, artifact, and summary information. The smoke entries are retained and
the completed FastAPI Tier 1 rerun appears below them.
| Task 1.1/1.2/1.5 | `src.baseline.run_baselines` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 1, "xgb_tune_trials": 1}` | 2026-09-26T07:22:37.676413+00:00 | `data/results/smoke_baseline.json` | Unified baseline comparison with raw trial metrics and tuned XGBoost. |
| Task 1.3 | `src.tune.run_tuning` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 1}` | 2026-09-26T07:27:47.701712+00:00 | `data/results/fastapi_dataset_clean_best_hyperparams.json` | ANN hyperparameters tuned with deterministic Optuna seed. |
| Task 1.3 | `src.tune.run_tuning` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 50}` | 2026-09-26T07:31:00.817738+00:00 | `data/results/fastapi_dataset_clean_best_hyperparams.json` | ANN hyperparameters tuned with deterministic Optuna seed. |
| Task 1.1/1.2/1.5 | `src.baseline.run_baselines` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 20, "xgb_tune_trials": 50}` | 2026-09-26T07:40:01.317348+00:00 | `data/results/fastapi_dataset_clean/baseline_results.json` | Unified baseline comparison with raw trial metrics and tuned XGBoost. |
| Task 1.2 | `src.stats.run_significance_tests` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 20}` | 2026-09-26T07:41:37.421475+00:00 | `data/results/fastapi_dataset_clean/stats_results.json` | Paired two-sided Wilcoxon and paired Cohen effect size recorded. |
| Task 1.1/1.2 | `src.ablation.run_ablation` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "n_trials": 20}` | 2026-09-26T07:47:11.819882+00:00 | `data/results/fastapi_dataset_clean/ablation_results.json` | Ablation combinations evaluated with unified trial seeds. |
| Task 1.1/1.3 | `src.genetic_algorithm.FeatureSelectionGA.evolve` | `{"base_seed": 42, "dataset": "data/fastapi_dataset_clean.csv", "generations": 10, "population_size": 15}` | 2026-09-26T07:37:00+00:00 | `data/results/fastapi_dataset_clean/ga_results.json` | GA converged at generation 8 with 120 isolated evaluations. |
