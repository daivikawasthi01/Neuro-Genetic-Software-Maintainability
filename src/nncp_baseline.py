"""NNCP-style Pearson-filtered structural baseline.

This baseline intentionally removes the GA step and uses only Category A
structural features. Highly correlated structural columns are filtered with a
Pearson absolute-correlation threshold, then the existing ANN and five-fold
CV evaluator are reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.ann_model import train_and_evaluate_ann
from src.constants import CATEGORY_A_STRUCTURAL
from src.reproducibility import (
    DEFAULT_BASE_SEED,
    DEFAULT_N_TRIALS,
    append_run_log,
    make_metadata,
    seed_everything,
    trial_seed,
    write_json,
)


def select_structural_features(csv_file: str, threshold: float = 0.95) -> list[str]:
    """Return Pearson-filtered Category A columns in original column order."""
    df = pd.read_csv(csv_file)
    available = [
        name for name in CATEGORY_A_STRUCTURAL
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name])
    ]
    if len(available) < 2:
        return available
    corr = df[available].corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [name for name in available if name not in drop]


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _paired_stats(nncp: list[float], ga: list[float]) -> dict | None:
    if len(nncp) != len(ga) or not nncp:
        return None
    differences = np.asarray(ga, dtype=float) - np.asarray(nncp, dtype=float)
    if np.all(differences == 0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(
            differences, alternative="two-sided", zero_method="wilcox", method="auto"
        )
    std = float(differences.std())
    return {
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
        "cohens_d_paired": float(differences.mean() / (std + 1e-12)),
        "alternative": "two-sided",
        "paired": True,
    }


def run_nncp_baseline(
    datasets: dict[str, str],
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    base_seed: int = DEFAULT_BASE_SEED,
    threshold: float = 0.95,
    output_path: str = "results/nncp_results.json",
) -> dict:
    results = {}
    for repo_name, csv_file in datasets.items():
        selected = select_structural_features(csv_file, threshold)
        df = pd.read_csv(csv_file)
        numeric = [
            name for name in df.select_dtypes(include="number").columns
            if name != "target_bug_proneness"
        ]
        mask = [1 if name in selected else 0 for name in numeric]
        metrics = {"mse": [], "mae": [], "r2": []}
        for index in range(n_trials):
            seed = trial_seed(base_seed, index)
            seed_everything(seed)
            output = train_and_evaluate_ann(
                csv_file,
                feature_mask=mask,
                use_kfold=True,
                split_seed=seed,
                log_transform=True,
                return_metrics=True,
            )
            for key in metrics:
                metrics[key].append(float(output[key]))

        summary = {
            "repo": repo_name,
            "dataset": csv_file,
            "pearson_threshold": threshold,
            "selected_features": selected,
            "n_features": len(selected),
            "trial_seeds": [trial_seed(base_seed, i) for i in range(n_trials)],
            "mse": _summary(metrics["mse"]),
            "mae": _summary(metrics["mae"]),
            "r2": _summary(metrics["r2"]),
        }
        baseline_path = Path("data/results") / Path(csv_file).stem / "baseline_results.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            ga_values = baseline.get("ga_selected", {}).get("mses", [])
            summary["ga_comparison"] = _paired_stats(metrics["mse"], ga_values)
            summary["ga_mses"] = ga_values
        else:
            summary["ga_comparison"] = None
            summary["ga_mses"] = []
        results[repo_name] = summary

    payload = {
        "metadata": make_metadata({
            "datasets": datasets,
            "n_trials": n_trials,
            "base_seed": base_seed,
            "pearson_threshold": threshold,
            "feature_scope": "Category A structural features only",
            "protocol": "5-fold stratified CV using the shared ANN evaluator",
        }),
        "results": results,
    }
    write_json(output_path, payload)
    append_run_log(
        "Task 2.1", "src.nncp_baseline.run_nncp_baseline",
        {"datasets": datasets, "n_trials": n_trials, "base_seed": base_seed,
         "pearson_threshold": threshold}, output_path,
        "NNCP-style Pearson-filtered structural baseline evaluated.",
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NNCP-style baseline")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output", default="results/nncp_results.json")
    parser.add_argument("datasets", nargs="+", help="repo_name=clean_dataset.csv")
    args = parser.parse_args()
    dataset_map = dict(item.split("=", 1) for item in args.datasets)
    run_nncp_baseline(
        dataset_map,
        n_trials=args.n_trials,
        base_seed=args.seed,
        threshold=args.threshold,
        output_path=args.output,
    )
