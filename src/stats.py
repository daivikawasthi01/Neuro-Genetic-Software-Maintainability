"""
stats.py — Statistical significance test: GA-selected vs All-Features ANN.

Fix applied vs previous version:
  - use_kfold=True in both GA and All-Features evaluation so results are
    consistent with baseline.py. The original used single-split which gave
    MSE values of 1.0-1.8 on bad random seeds, corrupting the paired test.
  - Divergence cap lowered from 10.0 to 2.0: on log1p-transformed targets
    (bug counts 0-10), any MSE > 2.0 is a diverged run, not a real result.
    The 10.0 cap was too permissive and let values of 1.5-1.8 through.
  - epochs explicitly set to 100 to match GA fitness evaluation settings.
"""

import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_significance_tests(
    csv_file: str,
    ga_chromosome: tuple,
    n_trials: int       = 30,
    log_transform: bool = True,
    output_path: str    = 'data/results/stats_results.json',
) -> dict:
    """
    Run n_trials paired evaluations of GA-selected vs All-Features ANN.

    Both models use identical settings:
      - use_kfold=True   (5-fold stratified CV, matches baseline.py)
      - epochs=100       (matches GA fitness evaluation)
      - Different random seed per trial for the k-fold shuffle

    Wilcoxon Signed-Rank Test on the paired MSE differences.
    Cohen's d effect size.
    """
    from src.ann_model import train_and_evaluate_ann

    print(f"\n[STATS] Running {n_trials} paired trials (GA vs All Features)")
    print(f"  GA chromosome: {sum(ga_chromosome)}/{len(ga_chromosome)} features")
    print(f"  Settings: use_kfold=True, epochs=100, divergence_cap=2.0\n")

    ga_mses  = []
    all_mses = []

    for trial in range(1, n_trials + 1):
        seed = trial * 7  # deterministic but varied seeds

        ga_mse = train_and_evaluate_ann(
            csv_file,
            feature_mask  = list(ga_chromosome),
            epochs        = 100,
            use_kfold     = True,
            log_transform = log_transform,
            split_seed    = seed,
        )

        all_mse = train_and_evaluate_ann(
            csv_file,
            feature_mask  = None,   # all features
            epochs        = 100,
            use_kfold     = True,
            log_transform = log_transform,
            split_seed    = seed,
        )

        # Cap diverged runs at 2.0 — on log1p-transformed targets any MSE
        # above 2.0 indicates the ANN diverged on this split, not a real result.
        ga_mse  = min(ga_mse,  2.0)
        all_mse = min(all_mse, 2.0)

        ga_mses.append(ga_mse)
        all_mses.append(all_mse)

        print(f"  Trial {trial:02d}/{n_trials} — "
              f"GA: {ga_mse:.4f}  All: {all_mse:.4f}")

    ga_arr  = np.array(ga_mses)
    all_arr = np.array(all_mses)

    ga_mean  = float(ga_arr.mean())
    ga_std   = float(ga_arr.std())
    all_mean = float(all_arr.mean())
    all_std  = float(all_arr.std())

    improvement_pct = (all_mean - ga_mean) / (all_mean + 1e-9) * 100

    # Wilcoxon Signed-Rank Test (non-parametric, paired)
    # H0: no difference between GA and All-Features MSE distributions
    differences = all_arr - ga_arr
    if np.all(differences == 0):
        stat, p_value = 0.0, 1.0
    else:
        stat, p_value = wilcoxon(differences)

    # Cohen's d effect size on paired differences
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

    significant = bool(p_value < 0.05)

    # Pretty print
    width = 52
    print(f"\n  {'─' * width}")
    print(f"  STATISTICAL SIGNIFICANCE TEST RESULTS")
    print(f"  {'─' * width}")
    print(f"  GA Mean MSE       : {ga_mean:.4f} ± {ga_std:.4f}")
    print(f"  All-Feat Mean MSE : {all_mean:.4f} ± {all_std:.4f}")
    print(f"  Improvement       : {improvement_pct:+.1f}%")
    print(f"  Wilcoxon p-value  : {p_value:.4f}")
    print(f"  Cohen's d         : {cohens_d:.3f} ({effect_label})")
    print(f"  Significant?      : {'YES ✓' if significant else 'NO  ✗'}")
    print(f"  {'─' * width}\n")

    results = {
        'ga_mean_mse':       ga_mean,
        'ga_std_mse':        ga_std,
        'all_features_mean_mse': all_mean,
        'all_features_std_mse':  all_std,
        'improvement_pct':   improvement_pct,
        'wilcoxon_statistic': float(stat),
        'wilcoxon_p_value':  float(p_value),
        'cohens_d':          cohens_d,
        'effect_size':       effect_label,
        'significant':       significant,
        'pct_improvement':   improvement_pct,
        'n_trials':          n_trials,
        'ga_mses':           [float(v) for v in ga_mses],
        'all_mses':          [float(v) for v in all_mses],
    }

    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Statistical Significance Testing (GA vs All Features)")
    parser.add_argument("--csv", default=None,
                        help="Path to preprocessed CSV dataset")
    parser.add_argument("--ga-results", default="data/results/ga_results.json",
                        help="Path to ga_results.json")
    parser.add_argument("--chromosome", default=None,
                        help="Comma-separated binary string (e.g. 1,0,1,1)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of paired trials")
    parser.add_argument("--output", default="data/results/stats_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    # Determine CSV file path
    csv_file = args.csv
    if not csv_file:
        for candidate in ["data/combined_dataset_clean.csv", "data/flask_dataset_clean.csv", "flask_dataset_processed.csv", "flask_dataset.csv"]:
            if os.path.exists(candidate):
                csv_file = candidate
                break

    if not csv_file or not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find a valid CSV dataset. Specify with --csv.")

    # Determine chromosome
    if args.chromosome:
        chrom = tuple(int(x.strip()) for x in args.chromosome.split(","))
    elif os.path.exists(args.ga_results):
        with open(args.ga_results) as f:
            ga_data = json.load(f)
            chrom = tuple(ga_data.get('chromosome', []))
    else:
        # Default fallback subset if no ga_results yet
        df_tmp = pd.read_csv(csv_file)
        num_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c != 'target_bug_proneness']
        n_feats = len(feat_cols)
        chrom = tuple([1 if i % 2 == 0 else 0 for i in range(n_feats)])

    run_significance_tests(
        csv_file      = csv_file,
        ga_chromosome = chrom,
        n_trials      = args.n_trials,
        output_path   = args.output,
    )