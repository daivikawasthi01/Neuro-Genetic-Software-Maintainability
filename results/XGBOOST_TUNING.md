# XGBoost Tuning Record

The FastAPI Tier 1 baseline used 50 Optuna trials with the same five-fold
stratified CV protocol and GA-selected feature mask as the ANN comparison.

Best executed parameters:

```json
{
  "n_estimators": 342,
  "learning_rate": 0.19621641102949616,
  "max_depth": 2,
  "min_child_weight": 0.5816921826248967,
  "subsample": 0.7616186438288415,
  "colsample_bytree": 0.8868421704055205,
  "reg_alpha": 7.893764033350246e-08,
  "reg_lambda": 0.0022834751482411144,
  "best_cv_mse": 0.13698345445454818
}
```

On the 20-trial comparison, untuned GA-XGBoost mean MSE was `0.297551` and
tuned GA-XGBoost mean MSE was `0.284552`. Tuning improved the baseline but it
remained above All-Features ANN (`0.187950`) and GA-ANN (`0.123577`).
