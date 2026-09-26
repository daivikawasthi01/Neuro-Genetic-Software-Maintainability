"""
ablation.py — Quantifies each feature category's contribution to prediction accuracy.

Answers the research question: "Does including evolutionary (Git) and textual
metrics actually help, or do structural metrics alone suffice?"

Tests 7 combinations:
  A only       — structural metrics (closest to base paper)
  B only       — textual metrics only
  C only       — evolutionary metrics only
  A + B        — structural + textual
  A + C        — structural + evolutionary
  B + C        — textual + evolutionary
  A + B + C    — all three categories (full model)

Results saved to data/results/ablation_results.json.
"""

import json
import os
import numpy as np
import pandas as pd

from src.ann_model import train_and_evaluate_ann
from src.constants import CATEGORY_A_STRUCTURAL, CATEGORY_B_TEXTUAL, CATEGORY_C_EVOLUTIONARY
from src.reproducibility import (
    DEFAULT_BASE_SEED,
    append_run_log,
    make_metadata,
    seed_everything,
    trial_seed,
    write_json,
)


ABLATION_COMBOS = {
    'A only (Structural)': ['A'],
    'B only (Textual)':    ['B'],
    'C only (Evolutionary)': ['C'],
    'A + B':               ['A', 'B'],
    'A + C':               ['A', 'C'],
    'B + C':               ['B', 'C'],
    'A + B + C (Full)':    ['A', 'B', 'C'],
}

CATEGORY_MAP = {
    'A': CATEGORY_A_STRUCTURAL,
    'B': CATEGORY_B_TEXTUAL,
    'C': CATEGORY_C_EVOLUTIONARY,
}


def _build_mask_for_combo(feature_names: list, categories: list) -> list:
    """Build binary mask selecting only the features in the given categories."""
    selected = set()
    for cat in categories:
        selected.update(CATEGORY_MAP[cat])
    return [1 if name in selected else 0 for name in feature_names]


def run_ablation(
    csv_file: str,
    n_trials: int       = 20,
    base_seed: int      = DEFAULT_BASE_SEED,
    log_transform: bool = True,
    output_path: str    = "data/results/ablation_results.json",
) -> dict:
    """
    Runs n_trials for each of the 7 feature combinations.

    Returns:
        {combo_name: {'mses': [...], 'mean': x, 'std': x, 'n_features': k}, ...}
    """
    df            = pd.read_csv(csv_file)
    num_cols      = df.select_dtypes(include='number').columns.tolist()
    feature_names = [c for c in num_cols if c != 'target_bug_proneness']

    print(f"\n[ABLATION] {len(ABLATION_COMBOS)} combinations x {n_trials} trials each")

    seed_everything(base_seed)
    results = {}

    for combo_name, categories in ABLATION_COMBOS.items():
        mask    = _build_mask_for_combo(feature_names, categories)
        n_feats = sum(mask)

        print(f"\n  {combo_name}  ({n_feats} features)")

        # FIX: guard against zero-feature mask. If constants.py names don't
        # match the CSV columns (e.g. after a schema change), all bits are 0.
        # Passing a zero-feature mask produces X[:, []]-shaped input, which
        # crashes nn.Linear(0, hidden1). Skip with a clear warning instead.
        if n_feats == 0:
            print(f"    [WARN] No matching features for '{combo_name}' — "
                  f"skipping. Check that constants.py names match CSV columns.")
            continue

        combo_mses = []
        for trial in range(n_trials):
            seed = trial_seed(base_seed, trial)
            seed_everything(seed)
            mse  = train_and_evaluate_ann(
                csv_file,
                feature_mask  = mask,
                use_kfold     = True,
                split_seed    = seed,
                log_transform = log_transform,
                return_metrics = True,
            )
            combo_mses.append(mse['mse'])
            print(f"    trial {trial+1:02d}/{n_trials}  MSE: {mse['mse']:.4f}")

        results[combo_name] = {
            'categories': categories,
            'n_features': n_feats,
            'mses':       combo_mses,
            'mean':       float(np.mean(combo_mses)),
            'std':        float(np.std(combo_mses)),
            'trial_seeds': [trial_seed(base_seed, i) for i in range(n_trials)],
        }

    payload = {
        'metadata': make_metadata({
            'dataset': csv_file, 'n_trials': n_trials, 'base_seed': base_seed,
            'protocol': '5-fold stratified CV',
        }),
        'results': results,
    }
    write_json(output_path, payload)
    append_run_log(
        'Task 1.1/1.2', 'src.ablation.run_ablation',
        {'dataset': csv_file, 'n_trials': n_trials, 'base_seed': base_seed},
        output_path, 'Ablation combinations evaluated with unified trial seeds.',
    )

    _print_summary(results)
    print(f"\n  Saved to: {output_path}")
    return results


def _print_summary(results: dict):
    print("\n  ┌──────────────────────────────────┬──────────┬──────────┬───────┐")
    print("  │  Combination                     │ Mean MSE │  Std Dev │ Feats │")
    print("  ├──────────────────────────────────┼──────────┼──────────┼───────┤")
    for name, data in sorted(results.items(), key=lambda x: x[1]['mean']):
        print(f"  │  {name:<32}  │  {data['mean']:.4f}  │  {data['std']:.4f}  │  {data['n_features']:2d}   │")
    print("  └──────────────────────────────────┴──────────┴──────────┴───────┘")


if __name__ == "__main__":
    run_ablation("data/flask_dataset.csv", n_trials=5)
