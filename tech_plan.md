# Agent Task Spec: Neuro-Genetic Maintainability Framework — Code Improvements

**Audience:** an AI coding agent (e.g. Claude Code) with read/write/execute access to this repository.
**Do not treat this file as prose to summarize — treat every numbered task as a unit of work with a concrete Definition of Done.**
**Work strictly in tier order. Do not start a Tier 2 task until every Tier 1 task's Definition of Done is satisfied and verified.**

This spec assumes the repository implements the pipeline described in "A Hybrid Neuro-Genetic Framework for Automated Software Maintainability Assessment" (AST/Git feature mining → GA feature selection → Optuna-tuned ANN evaluator → ablation/sensitivity/statistical analysis over Flask, Requests, FastAPI). Exact file names below are placeholders — **Step 0 requires you to map them to the real files before touching anything.**

---

## Ground rules (apply to every task below)

1. **Never fabricate or hand-edit a results number.** Every number that ends up in a table, plot, or report must come from an actual code run you executed in this session. If you cannot run something (e.g. no GPU, no network), say so explicitly instead of inventing a plausible-looking result.
2. **Don't silently change existing experiment configs that other tables depend on.** If Table 4's config and Table 6's config currently differ (they do — see Task 1.1), make the discrepancy an explicit, named config parameter rather than hardcoding a new number into one script.
3. **Preserve reproducibility as you go.** Every experiment you run or re-run must have its random seed(s), config, and git commit hash logged alongside its output (see Task 1.3).
4. **Version control discipline.** One commit per task below, with a commit message referencing the task number (e.g. `Task 1.2: increase RQ1 trial count to 20, unify with ablation protocol`). Do not squash tasks together.
5. **Log everything to a run manifest.** Maintain (create if absent) a `results/RUN_LOG.md` or `results/run_log.jsonl` that records: task ID, script invoked, config/seed, start/end time, output artifact path, and a one-line summary of what changed vs. the previous run.
6. **If a fix changes a headline number** (MSE, % reduction, p-value, effect size), **do not edit prose in the paper yourself.** Output a short `results/NUMBERS_CHANGED.md` diff (old value → new value → source script/run) so a human can update the manuscript. Your job is the code and the numbers; the manuscript text is out of scope for this agent.
7. **If Task 1.1's investigation shows the paper's core claim doesn't hold** (GA-selected subset does not actually beat the full feature set once trial counts are equalized), **do not paper over it.** Report it plainly in `results/NUMBERS_CHANGED.md` and flag it as a finding, not a bug to hide.

---

## Step 0 — Repository audit (required before Task 1.1)

Before any code changes, produce `results/REPO_AUDIT.md` answering:

1. Where does the GA loop live, and where is its trial/repeat-count parameterized (the thing that currently produces "5 trials" for Table 4 vs "20 trials" for Table 6)?
2. Where is the ANN defined and trained, and where are Optuna hyperparameters loaded/frozen from?
3. Where is the correlation filter (|r| > 0.95) applied, and does the code that builds the 19-feature list match Table 1 exactly (check for `weighted methods per class` or `docstring presence` — these appear in the paper's prose but are **not** supposed to be real features; confirm whether they exist in code as dead/unused columns, which would explain the paper-text inconsistency)?
5. Where is the Wilcoxon signed-rank test computed, and does the code specify `alternative` (one/two-sided) and whether the test is paired?
6. Where are random seeds currently set, if anywhere (numpy, torch, GA population init, train/test split)?
7. Where do Table 4, Table 5, Table 6, Table 7/8, and Figure 4/5/6/7 get generated from — are they separate scripts, or one pipeline with different config flags? List the exact entry points.
8. Is there an existing `requirements.txt` / `environment.yml` / lockfile? Is there a README with run instructions? Is there a results cache/memoization store (the paper mentions a memoization cache `M` in Algorithm 1) — where does it persist, and could stale cache entries be contaminating the Table 4 vs Table 6 discrepancy?

This audit is the map for every task below. Update file paths in the tasks as you discover the real ones.

---

## TIER 1 — Execute now, in order

### Task 1.1 — Diagnose and unify the Table 4 vs. Table 6 discrepancy
**Why:** Table 4 reports "All-Features ANN" MSE = 0.2289 (5 trials); Table 6's ablation reports the identical model ("A+B+C, All Features") at MSE = 0.1686 (20 trials) — a 26% gap for the same model on the same dataset. Since the GA-selected result (0.1890) sits *between* these two numbers, whether the paper's central claim ("GA beats all-features") holds at all depends entirely on which of these two baseline numbers is correct.

**Definition of Done:**
- A single, named experiment config controls trial count for *all* main-comparison conditions (All-Features, Random-Subset, GA-ANN, XGB-GA) in whatever script produces Table 4/5. No condition should use a different trial count than another in the same table.
- You have run this unified script at the *same* trial count used for the ablation (Table 6) on the *same* FastAPI dataset, and captured a fresh MSE for "All-Features ANN" under that protocol.
- `results/NUMBERS_CHANGED.md` states, in plain language, whether the new All-Features number is closer to 0.2289 or 0.1686, and whether GA-ANN still beats it once both are measured under equal statistical power. If it flips (GA no longer wins), say so explicitly and do not adjust anything to make it "look better."
- Check the memoization cache (Algorithm 1's `M`) for the ablation run — confirm it wasn't accidentally reused/stale across the Table 4 and Table 6 runs, which could itself explain divergent numbers. Note the finding either way.

**Suggested approach:** find where each table's script sets its trial-count constant, refactor to a shared CLI/config argument (e.g. `--n-trials`), and re-run both Table 4 and Table 6 generation with an identical value (start with 20).

---

### Task 1.2 — Raise the main RQ1 comparison to 20–30 trials with full statistical reporting
**Why:** professor's top-cited weakness — Wilcoxon p = 0.0625 on n = 5 is underpowered.

**Definition of Done:**
- Table 4/5 regenerated with n ≥ 20 trials (reuse the unified config from Task 1.1).
- The Wilcoxon test call explicitly specifies whether it's paired and one- or two-sided; both the exact statistic (W) and p-value are captured in the output artifact, not just p.
- Cohen's d is recomputed for the new trial count and stored alongside.
- Output artifact (CSV/JSON) contains, per condition: mean MSE, std MSE, all raw per-trial MSE values (not just the aggregate), so a human can re-derive any statistic later without re-running.
- `results/NUMBERS_CHANGED.md` updated with old vs. new p-value, W statistic, Cohen's d, and MSE means for every condition in Table 4/5.

---

### Task 1.3 — Deterministic seeding and run provenance
**Why:** professor's reproducibility weakness (#7); also a precondition for trusting Task 1.1/1.2's results.

**Definition of Done:**
- A single top-level seed parameter (e.g. `--seed`) deterministically seeds: numpy, Python's `random`, PyTorch (CPU and CUDA if applicable), the GA's population initialization, and the train/val/test split.
- Running the same script twice with the same seed produces bit-identical (or numerically identical to a reasonable tolerance) results; verify this with an actual repeated run and record the check in `results/REPO_AUDIT.md` or a new `results/SEED_VERIFICATION.md`.
- Every result artifact (CSV/JSON from Tasks 1.1, 1.2, and later tiers) includes the seed(s) used, either as a column/field or a companion metadata file.
- If trials are meant to vary stochastically (e.g. 20 independent trials should *not* all use the same seed), confirm the per-trial seeding scheme is documented (e.g. `seed = base_seed + trial_index`) and is itself deterministic given `base_seed`.

---

### Task 1.4 — Package the repository for reproducibility
**Why:** professor's reproducibility weakness (#7); this is the "public code/data repository" ask, scoped to what's realistic without a Docker image.

**Definition of Done:**
- `requirements.txt` (or `environment.yml`) exists, is complete, and is pinned to specific versions (not just package names) — generate it from the actual working environment (`pip freeze` or equivalent), then prune to only what's imported.
- `README.md` at repo root includes: one-paragraph project description, install instructions, and **exact commands to regenerate Table 4, Table 5, Table 6, Table 7/8, and Figures 4–7** from a clean checkout — every command should be copy-pasteable and actually tested by you in this session.
- A `data/` or `datasets/` section of the README documents exactly which commit/tag of Flask, Requests, and FastAPI were cloned (the paper says "cloned at a fixed tag" — find and record the actual tags/commit hashes used, or add code to pin and record them if this isn't currently done).
- Confirm there is no dependency on absolute local paths, hardcoded API keys, or machine-specific config; if any exist, parameterize them via environment variables or CLI args and document in the README.

---

### Task 1.5 — Give the XGBoost baseline a fair (tuned) comparison
**Why:** minor issue flagged — XGBoost-on-GA-features (MSE 0.3734) underperforms even the all-features ANN, which is suspicious for tabular data of this size and likely reflects an untuned model.

**Definition of Done:**
- An Optuna search (comparable trial budget to the ANN's 50-trial search — document the exact budget used) is added for XGBoost's key hyperparameters (e.g. `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`/`reg_lambda`), evaluated via the same 5-fold CV protocol used elsewhere in the pipeline.
- The tuned XGBoost is re-run under the same unified trial-count protocol as Task 1.1/1.2, on the same GA-selected feature subset.
- Result recorded in the same output artifact as Task 1.2 (so Table 4/5 can include a "XGB-GA (tuned)" row alongside the existing one) — do not overwrite the untuned result, keep both for comparison.
- `results/NUMBERS_CHANGED.md` states whether tuning closes the gap, and by how much.

**Tier 1 exit gate:** all five tasks' Definitions of Done are met, `results/NUMBERS_CHANGED.md` and `results/RUN_LOG.md` are up to date, and every regenerated table/artifact is reproducible from the README commands in Task 1.4. Do not proceed to Tier 2 until this is true.

---

## TIER 2 — Queue after Tier 1 exit gate passes

### Task 2.1 — Re-implement the NNCP baseline for a fair numeric comparison
**Why:** professor's weakness #5 — the paper's comparison to the base paper (Vescan & Barac-Antonescu) is methodological only, with no numbers.

**Definition of Done:**
- A new baseline script implements: Pearson-correlation-threshold feature selection (project-specific threshold, matching the base paper's Model 4 description) restricted to Category A (structural/CK-style) features only, feeding the *same* ANN architecture/hyperparameter search used elsewhere in this repo — no GA involved.
- Trained/evaluated with the framework's own bug-fix-count target variable (not lines-changed, since that's not available here) and the framework's own 5-fold CV protocol, on Flask/Requests/FastAPI, using the same unified trial count from Tier 1.
- Output artifact adds an "NNCP-style baseline" row to the Table 4/5-equivalent comparison, with the same statistical detail (mean/std MSE, MAE, R², Wilcoxon vs. GA-ANN if sample sizes allow).
- A short `results/NNCP_COMPARISON.md` explains any remaining methodological differences that couldn't be equalized (e.g. target variable definition), so the comparison's caveats are explicit rather than implied.

### Task 2.2 — Small-scale bug-fix-label validation against issue-tracker ground truth
**Why:** professor's weakness #4 — the bug-fix keyword-matched target is a known-noisy proxy, never validated.

**Definition of Done:**
- A script pulls a random (seeded) sample of ~50–100 commits from one repository (recommend Flask, smallest history) via the GitHub API (or local git log if API access isn't available — note which was used).
- For each sampled commit, the script/agent determines (a) whether the existing keyword-matching heuristic labeled it as bug-fix-related, and (b) an independent ground-truth signal — e.g. whether the commit message references a closed GitHub Issue/PR labeled `bug`, or a manual review flag if API-based linkage isn't feasible.
- Computes and reports precision/recall/agreement (e.g. Cohen's kappa) between the keyword heuristic and the ground-truth signal.
- Result and methodology written to `results/LABEL_VALIDATION.md`, including the exact sample (commit hashes) used so it can be audited or repeated.
- If GitHub API access is not available in this environment, produce the sampling script and a clear note of what's blocked, rather than fabricating agreement numbers.

### Task 2.3 — Extend the multi-repository evaluation
**Why:** professor's weakness #3 (partial progress — full 8–10 repo/multi-language scope is Tier 3, deferred).

**Definition of Done:**
- Pipeline successfully run end-to-end (feature mining → GA selection → evaluation) on 2 additional same-domain Python repositories not already covered — recommend `django` and `aiohttp`, or `httpx`, chosen for availability of accessible Git history and comparable size range to the existing three.
- Table 7/8-equivalent output extended to include the new repositories, generated by the same unified script/config as the existing three (no separate one-off script).
- `results/RUN_LOG.md` records wall-clock time for each new repo's GA run (the paper reports these per-repo, e.g. "~470s for FastAPI").
- No changes made to Flask/Requests/FastAPI's existing recorded results as a side effect of this task — re-verify they're unchanged after the pipeline extension.

**Tier 2 exit gate:** all three tasks' Definitions of Done are met and logged. Stop and report back — do not attempt Tier 3 without explicit human sign-off.

---

## TIER 3 — Backlog, do not implement without explicit instruction

Listed for awareness only. Do not start these even if Tier 1 and 2 finish early — they require decisions (which languages, which industrial codebase, licensing/access for a Docker registry) that need a human call:

- Full 8–10 repository, multi-language (Java) evaluation with a UIMS/QUES-style comparison.
- Full-dataset issue-tracker relabeling (beyond the Tier 2 sample validation).
- Docker image / containerized reproduction environment.

---

## Reporting format expected from the agent after each tier

At the end of Tier 1, and again at the end of Tier 2, produce a short summary containing:
1. Which tasks completed, and links/paths to their output artifacts.
2. Any Definition-of-Done item that could not be met, and why (missing data access, compute limits, etc.) — do not mark a task done if any criterion is unmet.
3. The contents of `results/NUMBERS_CHANGED.md` at that point, so a human can decide what needs to change in the manuscript.
4. Any finding that contradicts the paper's current claims (most importantly, the Task 1.1 outcome) stated plainly, even if inconvenient.