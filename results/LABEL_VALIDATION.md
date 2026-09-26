# Bug-label heuristic validation

The Tier 2.2 validation samples 50 Flask commits deterministically (`seed=42`)
from the pinned local history and compares the commit-message keyword heuristic
with GitHub pull-request labels. A commit is treated as bug-related when its
associated pull request contains one of: `bug`, `bugfix`, `defect`,
`regression`, or `security`.

The exact commit hashes, messages, labels, and raw decisions are stored in
[`label_validation.json`](label_validation.json). The pinned Flask checkout
was deepened before sampling so the sample is not limited by a shallow clone.

## Result

- Sampled/evaluated commits: 50/50
- Precision: 0.0000
- Recall: 0.0000
- Cohen's kappa: -0.0396
- API lookup blockers: none

This result does not support the existing keyword heuristic as a reliable
ground-truth substitute. The comparison is deliberately conservative: an
unlabelled pull request is not counted as a bug fix, and the GitHub label
signal is only an external validation proxy rather than a manually reviewed
oracle. The raw sample should therefore be retained for a future manually
reviewed relabelling study.
