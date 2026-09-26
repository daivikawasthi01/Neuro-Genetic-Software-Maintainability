# Seed Verification

Verified on 2026-09-26 using `data/fastapi_dataset_clean.csv`, seed `42`,
single split, and two training epochs:

```json
{
  "first": {"mse": 0.14167223401212445, "mae": 0.14444704505332126, "r2": 0.3558824483606695},
  "second": {"mse": 0.14167223401212445, "mae": 0.14444704505332126, "r2": 0.3558824483606695},
  "equal": true
}
```

The check covered Python/NumPy/PyTorch seeding and deterministic ANN data
loading for the same configuration. Full experiment artifacts record the
same base seed and their derived per-trial seeds.

The completed baseline and stats runs independently evaluated the same 20
FastAPI trial seeds. Their raw All-Features arrays and raw GA-ANN arrays were
bit-identical (`true` for both comparisons), providing a full-protocol
cross-stage repeatability check.
