"""
held_out.py — Evaluate GA-selected features on a held-out (unseen) repository.

Tests true generalisation:
Evaluates whether the feature subset discovered by the GA on the training repo(s)
generalises to a completely held-out repository that the GA never saw during evolution.

Compares:
  - All-Features ANN on Held-Out Repo
  - GA-Selected Features ANN on Held-Out Repo
  - Random-Subset ANN on Held-Out Repo (matched cardinality)
"""

import argparse
import json
import os
import sys
import random
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ann_model import train_and_evaluate_ann


def evaluate_on_held_out(
    held_out_csv: str,
    ga_chromosome: tuple,
    n_trials: int       = 20,
    log_transform: bool = True,
    output_path: str    = "data/results/held_out_results.json",
) -> dict:
    """
    Evaluate GA feature selection on an unseen held-out repository.

    Args:
        held_out_csv: Path to preprocessed CSV of the held-out repository.
        ga_chromosome: Binary tuple/list indicating feature mask.
        n_trials: Number of repeated evaluation trials with varying seeds.
        log_transform: Whether to apply log1p transform on target variable.
        output_path: Output JSON path.

    Returns:
        Dictionary containing held-out evaluation metrics, Wilcoxon p-value,
        and Cohen's d effect size.
    """
    print(f"\n{'='*60}")
    print(f" HELD-OUT REPOSITORY EVALUATION (Zero-Shot Feature Transfer)")
    print(f"{'='*60}")
    print(f"  Held-out dataset : {held_out_csv}")
    print(f"  GA chromosome    : {sum(ga_chromosome)}/{len(ga_chromosome)} features")
    print(f"  Evaluation trials: {n_trials}")
    print(f"{'='*60}\n")

    df = pd.read_csv(held_out_csv)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != 'target_bug_proneness']
    n_total_feats = len(feat_cols)
    k_selected = int(sum(ga_chromosome))

    ga_mask = list(ga_chromosome)
    # Ensure length matches held-out features
    if len(ga_mask) != n_total_feats:
        if len(ga_mask) > n_total_feats:
            ga_mask = ga_mask[:n_total_feats]
        else:
            ga_mask = ga_mask + [0] * (n_total_feats - len(ga_mask))

    ga_mses     = []
    all_mses    = []
    random_mses = []

    for trial in range(1, n_trials + 1):
        seed = trial * 13

        # 1. GA-selected feature subset
        ga_mse = train_and_evaluate_ann(
            held_out_csv,
            feature_mask  = ga_mask,
            epochs        = 100,
            use_kfold     = True,
            log_transform = log_transform,
            split_seed    = seed,
        )
        ga_mse = min(ga_mse, 2.0)
        ga_mses.append(ga_mse)

        # 2. All-Features baseline
        all_mse = train_and_evaluate_ann(
            held_out_csv,
            feature_mask  = None,
            epochs        = 100,
            use_kfold     = True,
            log_transform = log_transform,
            split_seed    = seed,
        )
        all_mse = min(all_mse, 2.0)
        all_mses.append(all_mse)

        # 3. Random subset with same cardinality k
        rng = random.Random(seed)
        rand_indices = rng.sample(range(n_total_feats), min(k_selected, n_total_feats))
        rand_mask = [1 if i in rand_indices else 0 for i in range(n_total_feats)]
        rand_mse = train_and_evaluate_ann(
            held_out_csv,
            feature_mask  = rand_mask,
            epochs        = 100,
            use_kfold     = True,
            log_transform = log_transform,
            split_seed    = seed,
        )
        rand_mse = min(rand_mse, 2.0)
        random_mses.append(rand_mse)

        print(f"  Trial {trial:02d}/{n_trials} — "
              f"GA: {ga_mse:.4f}  |  All: {all_mse:.4f}  |  Random: {rand_mse:.4f}")

    ga_arr   = np.array(ga_mses)
    all_arr  = np.array(all_mses)
    rand_arr = np.array(random_mses)

    ga_mean   = float(ga_arr.mean())
    ga_std    = float(ga_arr.std())
    all_mean  = float(all_arr.mean())
    all_std   = float(all_arr.std())
    rand_mean = float(rand_arr.mean())
    rand_std  = float(rand_arr.std())

    improvement_vs_all = (all_mean - ga_mean) / (all_mean + 1e-9) * 100
    improvement_vs_rand = (rand_mean - ga_mean) / (rand_mean + 1e-9) * 100
    reduction_pct = (1.0 - k_selected / max(1, n_total_feats)) * 100

    differences = all_arr - ga_arr
    if np.all(differences == 0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(differences)

    diff_mean = float(differences.mean())
    diff_std  = float(differences.std())
    cohens_d  = diff_mean / (diff_std + 1e-9) if diff_std > 0 else 0.0

    if abs(cohens_d) < 0.2:
        effect_label = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_label = "small"
    elif abs(cohens_d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    results = {
        'held_out_dataset':         held_out_csv,
        'n_files':                  len(df),
        'n_features_total':         n_total_feats,
        'n_selected_features':      k_selected,
        'feature_reduction_pct':    reduction_pct,
        'ga_mean_mse':              ga_mean,
        'ga_std_mse':               ga_std,
        'all_features_mean_mse':    all_mean,
        'all_features_std_mse':     all_std,
        'random_mean_mse':          rand_mean,
        'random_std_mse':           rand_std,
        'improvement_vs_all_pct':   improvement_vs_all,
        'improvement_vs_rand_pct':  improvement_vs_rand,
        'wilcoxon_p_value':         float(p_val),
        'cohens_d':                 cohens_d,
        'effect_size':              effect_label,
        'significant':              bool(p_val < 0.05),
        'n_trials':                 n_trials,
        'ga_mses':                  [float(x) for x in ga_mses],
        'all_mses':                 [float(x) for x in all_mses],
        'random_mses':              [float(x) for x in random_mses],
    }

    width = 54
    print(f"\n  {'─' * width}")
    print(f"  HELD-OUT GENERALISATION RESULTS")
    print(f"  {'─' * width}")
    print(f"  GA-ANN MSE (Held-out)    : {ga_mean:.4f} ± {ga_std:.4f} ({k_selected}/{n_total_feats} feats)")
    print(f"  All-Features MSE         : {all_mean:.4f} ± {all_std:.4f} ({n_total_feats} feats)")
    print(f"  Random-Subset MSE        : {rand_mean:.4f} ± {rand_std:.4f} ({k_selected} feats)")
    print(f"  Improvement vs All-Feats : {improvement_vs_all:+.1f}%")
    print(f"  Improvement vs Random    : {improvement_vs_rand:+.1f}%")
    print(f"  Feature Reduction        : {reduction_pct:.1f}%")
    print(f"  Cohen's d (vs All-Feats) : {cohens_d:.3f} ({effect_label})")
    print(f"  Wilcoxon p-value         : {p_val:.4f}")
    print(f"  Significant (p < 0.05)   : {'YES ✓' if p_val < 0.05 else 'NO ✗'}")
    print(f"  {'─' * width}\n")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {output_path}")

    return results


def _load_chromosome(ga_results_path: str = "data/results/ga_results.json") -> tuple:
    if os.path.exists(ga_results_path):
        with open(ga_results_path) as f:
            data = json.load(f)
            return tuple(data.get('chromosome', []))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-Out Repository Evaluation")
    parser.add_argument("--held-out-csv", default="flask_dataset_processed.csv",
                        help="Path to preprocessed held-out CSV")
    parser.add_argument("--ga-results", default="data/results/ga_results.json",
                        help="Path to ga_results.json containing chromosome")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of repeated paired trials")
    parser.add_argument("--output", default="data/results/held_out_results.json",
                        help="Output path for held-out results JSON")
    args = parser.parse_args()

    chrom = _load_chromosome(args.ga_results)
    if chrom is None:
        df_tmp = pd.read_csv(args.held_out_csv)
        n_feats = len(df_tmp.select_dtypes(include=[np.number]).columns) - 1
        chrom = tuple([1 if i % 2 == 0 else 0 for i in range(n_feats)])

    evaluate_on_held_out(
        held_out_csv  = args.held_out_csv,
        ga_chromosome = chrom,
        n_trials      = args.n_trials,
        output_path   = args.output,
    )
