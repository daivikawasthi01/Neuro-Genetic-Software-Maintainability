# Numbers Changed

This file is intentionally maintained from executed experiment output. No
manuscript values are changed by the implementation.

## Tier 1 FastAPI rerun

Source artifacts: `data/results/fastapi_dataset_clean/`, base seed `42`,
20 trials, five-fold stratified CV, FastAPI commit
`192b12197eb04c2b4a691cce7d87261b21716714`.

| Metric | New executed value |
|---|---:|
| All-Features ANN mean MSE ± SD | 0.187950 ± 0.035194 |
| Random-Subset ANN mean MSE ± SD | 0.264078 ± 0.082200 |
| GA-ANN mean MSE ± SD | 0.123577 ± 0.035504 |
| Untuned GA-XGBoost mean MSE ± SD | 0.297551 ± 0.053227 |
| Tuned GA-XGBoost mean MSE ± SD | 0.284552 ± 0.066058 |
| GA-ANN improvement vs All-Features | +34.2504% |
| Paired two-sided Wilcoxon W | 0.0 |
| Paired two-sided Wilcoxon p | 0.00000190735 |
| Paired Cohen's d | 1.9504 (large) |

The equal-powered All-Features result is closer to the old 0.2289 Table 4
value than the old 0.1686 Table 6 value, but it is not the same because the
dataset revision and tuned protocol are now explicitly recorded. Under the
same 20 seeds and five-fold protocol, GA-ANN still beats All-Features.

The ablation's full A+B+C mean is also 0.187950, while the GA-selected mean is
0.123577. Therefore the old narrative claim that the GA result was worse than
the all-feature ablation is not reproduced under this unified run.

XGBoost tuning improved the GA-feature baseline from 0.297551 to 0.284552,
but it remained above both ANN conditions. The tuned search used 50 Optuna
trials; its best CV objective was 0.136983.

No stale GA cache was reused: the run used the dataset-specific checkpoint and
result directory, with 120 in-memory evaluations and generation split seeds
derived from base seed 42.
