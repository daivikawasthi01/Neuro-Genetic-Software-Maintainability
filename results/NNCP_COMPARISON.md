# NNCP-style structural baseline comparison

This Tier 2.1 baseline uses only Category A structural metrics and applies a
Pearson absolute-correlation filter at `|r| > 0.95`. It then uses the same
ANN architecture, five-fold protocol, 20 trials, and seed schedule as the
main comparison (`base_seed=42`, trial seed `42 + trial_index`). Raw trial
metrics and provenance are stored in [`nncp_results.json`](nncp_results.json).

| Repository | Structural features retained | NNCP mean MSE | GA ANN mean MSE | NNCP vs GA paired result |
|---|---:|---:|---:|---|
| Flask | 6 | 0.5071 | 0.3393 | Wilcoxon W=9.0, p=0.000063, paired d=-1.2268 |
| Requests | 6 | 1.2693 | 0.3973 | Wilcoxon W=0.0, p=0.000002, paired d=-3.7825 |
| FastAPI | 7 | 0.4968 | 0.1236 | Wilcoxon W=0.0, p=0.000002, paired d=-9.0192 |

The negative paired effect sizes are computed as `(GA MSE - NNCP MSE) / SD`
and therefore indicate lower error for GA on all three evaluated repositories.
The comparison is paired and two-sided; the complete raw MSE arrays remain in
the JSON artifact.
