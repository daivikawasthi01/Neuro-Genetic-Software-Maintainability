"""
baseline.py — Automated baseline comparison for research credibility.

Runs four parallel evaluations:
  ALL      — ANN trained on all features (mirrors base paper approach)
  RANDOM   — ANN trained on a random subset the same size as GA-selected
  GA       — ANN trained on GA-selected features
  XGB_GA   — XGBoost trained on GA-selected features (model comparison)

The XGBoost baseline answers a secondary research question:
  "Is the ANN the right model for this task, or would a tree-based model
   generalise better on this small tabular dataset?"
Including it strengthens the paper: either the ANN wins (justifying the
architecture) or XGBoost wins (which is a finding in itself).

Results saved to data/results/baseline_results.json.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from src.ann_model import train_and_evaluate_ann
from src.reproducibility import (
    DEFAULT_BASE_SEED,
    DEFAULT_N_FOLDS,
    append_run_log,
    make_metadata,
    seed_everything,
    trial_seed,
    write_json,
)


def _xgb_cv_mse(
    csv_file: str,
    feature_mask: list,
    n_folds: int        = 5,
    split_seed: int     = 42,
    log_transform: bool = True,
    params: dict | None = None,
) -> float:
    """
    5-fold stratified CV MSE for XGBoost on the given feature mask.
    Returns MSE on original bug-count scale (back-transformed if log_transform).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return float('nan')   # graceful skip if xgboost not installed

    df    = pd.read_csv(csv_file)
    y_all = df['target_bug_proneness'].values.astype(float)
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != 'target_bug_proneness']
    X_all = df[feature_cols].values.astype(float)

    mask_idx = [i for i, v in enumerate(feature_mask) if v == 1]
    X_all    = X_all[:, mask_idx]
    y_model  = np.log1p(y_all) if log_transform else y_all.copy()

    y_bins = np.clip(y_all.astype(int), 0, 2)
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    mses   = []

    for tr_idx, va_idx in skf.split(X_all, y_bins):
        scaler   = MinMaxScaler()
        X_tr     = scaler.fit_transform(X_all[tr_idx])
        X_va     = scaler.transform(X_all[va_idx])
        y_tr_m   = y_model[tr_idx]
        y_va_orig = y_all[va_idx]

        model_params = {
            'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'n_jobs': 1,  # Prevent OpenMP conflicts on macOS
            'random_state': split_seed, 'verbosity': 0,
            'early_stopping_rounds': 20,
            'eval_metric': 'rmse',
        }
        if params:
            model_params.update(params)
        model = XGBRegressor(**model_params)
        model.fit(X_tr, y_tr_m,
                  eval_set=[(X_va, y_va_orig if not log_transform else np.log1p(y_va_orig))],
                  verbose=False)

        preds = model.predict(X_va)
        if log_transform:
            preds = np.expm1(np.maximum(preds, 0))
        mses.append(float(np.mean((preds - y_va_orig) ** 2)))

    return float(np.mean(mses))


def _tune_xgb(csv_file: str, feature_mask: list, n_trials: int, seed: int) -> dict:
    """Tune XGBoost once using the same stratified CV objective as evaluation."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        return _xgb_cv_mse(csv_file, feature_mask, split_seed=seed,
                           params=params)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'best_cv_mse': float(study.best_value), 'n_trials': n_trials}


def _build_mask(n_total: int, selected_indices: list) -> list:
    mask = [0] * n_total
    for i in selected_indices:
        mask[i] = 1
    return mask


def run_baselines(
    csv_file: str,
    ga_chromosome: tuple,
    n_trials: int       = 20,
    base_seed: int      = DEFAULT_BASE_SEED,
    xgb_tune_trials: int = 50,
    tune_xgb: bool      = True,
    log_transform: bool = True,
    output_path: str    = "data/results/baseline_results.json",
) -> dict:
    """
    Runs n_trials of each baseline and returns MSE distributions.

    Args:
        csv_file:      Path to cleaned dataset CSV.
        ga_chromosome: Best chromosome tuple from GA.evolve().
        n_trials:      Number of independent ANN runs per method (different seeds).
        log_transform: Must match what was used during GA training.
        output_path:   Where to save the JSON results.

    Returns:
        {
          'all_features':    {'mses': [...], 'mean': x, 'std': x},
          'random_subset':   {'mses': [...], 'mean': x, 'std': x},
          'ga_selected':     {'mses': [...], 'mean': x, 'std': x},
          'n_trials':        n_trials,
          'ga_n_features':   k,
          'total_features':  n,
        }
    """
    df          = pd.read_csv(csv_file)
    num_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols   = [c for c in num_cols if c != 'target_bug_proneness']
    n_total     = len(feat_cols)
    n_ga        = int(sum(ga_chromosome))
    all_mask    = [1] * n_total
    ga_mask     = list(ga_chromosome)


    seed_everything(base_seed)
    xgb_params = None
    xgb_model_params = None
    if tune_xgb:
        print(f"\n[BASELINES] Tuning XGBoost ({xgb_tune_trials} Optuna trials)")
        xgb_params = _tune_xgb(csv_file, ga_mask, xgb_tune_trials, base_seed)
        xgb_model_params = {
            key: value for key, value in xgb_params.items()
            if key not in {'best_cv_mse', 'n_trials'}
        }

    results = {
        'all_features':  {'mses': []},
        'random_subset': {'mses': []},
        'ga_selected':   {'mses': []},
        'xgb_ga':        {'mses': []},   # XGBoost on GA features
        'xgb_ga_tuned':  {'mses': []},
    }

    print(f"\n[BASELINES] Running {n_trials} trials × 4 methods "
          f"(GA uses {n_ga}/{n_total} features)")

    for trial in range(n_trials):
        seed = trial_seed(base_seed, trial)
        seed_everything(seed)

        mse_all = train_and_evaluate_ann(
            csv_file, feature_mask=all_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform,
            return_metrics=True,
        )
        results['all_features']['mses'].append(mse_all['mse'])
        results['all_features'].setdefault('maes', []).append(mse_all['mae'])
        results['all_features'].setdefault('r2s', []).append(mse_all['r2'])

        rand_indices = random.sample(range(n_total), n_ga)
        rand_mask    = _build_mask(n_total, rand_indices)
        mse_rand     = train_and_evaluate_ann(
            csv_file, feature_mask=rand_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform,
            return_metrics=True,
        )
        results['random_subset']['mses'].append(mse_rand['mse'])
        results['random_subset'].setdefault('maes', []).append(mse_rand['mae'])
        results['random_subset'].setdefault('r2s', []).append(mse_rand['r2'])

        mse_ga = train_and_evaluate_ann(
            csv_file, feature_mask=ga_mask,
            use_kfold=True, split_seed=seed, log_transform=log_transform,
            return_metrics=True,
        )
        results['ga_selected']['mses'].append(mse_ga['mse'])
        results['ga_selected'].setdefault('maes', []).append(mse_ga['mae'])
        results['ga_selected'].setdefault('r2s', []).append(mse_ga['r2'])

        mse_xgb = _xgb_cv_mse(csv_file, ga_mask, split_seed=seed,
                               log_transform=log_transform)
        results['xgb_ga']['mses'].append(mse_xgb)
        if xgb_params:
            mse_xgb_tuned = _xgb_cv_mse(
                csv_file, ga_mask, split_seed=seed,
                log_transform=log_transform, params=xgb_model_params,
            )
            results['xgb_ga_tuned']['mses'].append(mse_xgb_tuned)

        print(f"  Trial {trial+1:02d}/{n_trials} — "
              f"All: {mse_all['mse']:.4f}  Rand: {mse_rand['mse']:.4f}  "
              f"GA-ANN: {mse_ga['mse']:.4f}  GA-XGB: {mse_xgb:.4f}")

    # Compute summary stats
    for key in results:
        mses                  = results[key]['mses']
        if not mses:
            results[key]['mean'] = None
            results[key]['std'] = None
            continue
        results[key]['mean']  = float(np.mean(mses))
        results[key]['std']   = float(np.std(mses))
        results[key]['min']   = float(np.min(mses))
        results[key]['max']   = float(np.max(mses))
        for metric in ('maes', 'r2s'):
            if metric in results[key]:
                results[key][f'{metric[:-1]}_mean'] = float(np.mean(results[key][metric]))
                results[key][f'{metric[:-1]}_std'] = float(np.std(results[key][metric]))

    results['n_trials']       = n_trials
    results['base_seed']       = base_seed
    results['trial_seeds']     = [trial_seed(base_seed, i) for i in range(n_trials)]
    results['n_folds']         = DEFAULT_N_FOLDS
    results['xgb_tuning']      = xgb_params
    results['metadata']        = make_metadata({
        'dataset': csv_file, 'n_trials': n_trials, 'base_seed': base_seed,
        'n_folds': DEFAULT_N_FOLDS, 'ga_chromosome': list(ga_chromosome),
        'xgb_tune_trials': xgb_tune_trials, 'tune_xgb': tune_xgb,
    })
    results['ga_n_features']  = n_ga
    results['total_features'] = n_total

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_json(output_path, results)
    append_run_log(
        'Task 1.1/1.2/1.5', 'src.baseline.run_baselines',
        {'dataset': csv_file, 'n_trials': n_trials, 'base_seed': base_seed,
         'xgb_tune_trials': xgb_tune_trials}, output_path,
        'Unified baseline comparison with raw trial metrics and tuned XGBoost.',
    )

    _print_summary(results)
    print(f"  Saved to: {output_path}")
    return results


def _print_summary(results: dict):
    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │  BASELINE COMPARISON SUMMARY                     │")
    print("  ├──────────────────────┬─────────────┬────────────┤")
    print("  │  Method              │  Mean MSE   │  Std Dev   │")
    print("  ├──────────────────────┼─────────────┼────────────┤")
    for label, key in [("All Features    ", "all_features"),
                       ("Random Subset   ", "random_subset"),
                       ("GA + ANN        ", "ga_selected"),
                       ("GA + XGBoost    ", "xgb_ga"),
                       ("GA + XGB tuned  ", "xgb_ga_tuned")]:
        data = results.get(key, {})
        m    = data.get('mean', float('nan'))
        s    = data.get('std',  float('nan'))
        print(f"  │  {label}  │  {m:.4f}     │  {s:.4f}    │")
    print("  └──────────────────────┴─────────────┴────────────┘")


if __name__ == "__main__":
    import json
    with open("data/results/ga_results.json") as f:
        ga_results = json.load(f)
    run_baselines(
        csv_file     = "data/flask_dataset.csv",
        ga_chromosome = tuple(ga_results['chromosome']),
        n_trials     = 10,
    )
