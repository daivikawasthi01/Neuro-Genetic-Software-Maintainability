# Revision Plan — "A Hybrid Neuro-Genetic Framework for Automated Software Maintainability Assessment"

**Authors:** Daivik Awasthi, Manik Gaur (NSUT) | **Prepared:** for fast, high-quality revision ahead of resubmission

This document has three parts:
1. A complete, section-by-section summary of the paper (so nothing gets lost while you revise).
2. A synthesis of your professor's review, organized by severity.
3. A **time-boxed improvement plan** — ordered so the highest-value, lowest-effort fixes come first. Several of the fixes below are things I found while cross-checking the paper's own numbers against each other, on top of what your professor flagged.

---

## PART 1 — Full Paper Summary

### Metadata
- **Title:** A Hybrid Neuro-Genetic Framework for Automated Software Maintainability Assessment Using Multidimensional Code Metrics
- **Domain:** Software maintainability prediction (defect-proneness / bug-fix-commit prediction at file level)
- **Core idea:** Combine (a) a 19-feature, 3-category metric taxonomy, (b) a Genetic Algorithm for automated feature selection, (c) an Optuna-tuned ANN as the fitness evaluator, and (d) a snapshot-based temporal isolation protocol to prevent data leakage.

### Abstract
Claims: 19-dimensional feature space (Structural / Textual / Evolutionary) from static analysis + Git history; GA-based binary feature selection with a dual-objective fitness (accuracy + parsimony); Optuna-tuned ANN evaluator; snapshot-based temporal isolation. Evaluated on Flask, Requests, FastAPI. Reports validation MSE of 0.139, 0.220, 0.254 with 44–71% feature reduction, and claims the GA consistently selects cross-dimensional (multi-category) subsets, validated by a sensitivity analysis.

### Section 1 — Introduction
- Motivates the problem: maintenance is 60–80% of lifecycle cost; manual review doesn't scale.
- Traces the research lineage: CK metrics (Li & Henry) → ML models (ANNs, hybrid FLANN+evolutionary, ensembles).
- Identifies **three literature gaps**: (1) feature spaces are structural-only, (2) feature selection is manual/Pearson-threshold-based, (3) temporal data leakage is rarely controlled for.
- Lists 5 contributions: the 3-category/19-feature taxonomy, GA-driven dual-objective feature selection, snapshot-based temporal isolation, Optuna-tuned ANN, and a multi-repository empirical evaluation with ablation/sensitivity/statistical testing.

### Section 2 — Related Work
- **2.1 Metrics-based prediction:** CK suite (Chidamber & Kemerer); Li & Henry's UIMS/QUES benchmark; Elmidaoui et al.'s systematic review (structural metrics dominate literature); Riaz et al.'s SLR (flags weak cross-project generalizability — which this paper claims to address).
- **2.2 ML approaches:** Kumar & Rath's hybrid FLANN + GA/PSO/CSA; Schnappinger et al. (decision tree, 80% precision, 3 Java projects); Alsolai & Roper (ensemble methods — proposal only, no empirical results); Gupta & Chug (Optimised ELM). The key related work is **Vescan & Barac-Antonescu (2025), "NNCP"** — Pearson-correlation-filtered CK metrics into a feed-forward ANN. This paper is explicitly positioned as extending NNCP on every axis (automated selection, richer features, temporal isolation, Bayesian tuning).
- **2.3 Evolutionary metrics / Git mining:** Nagappan & Ball (code churn → defect density); Rahman & Devanbu (process metrics > static metrics); Hassan (change entropy); Zimmermann et al. (network structure). Claims novelty: no prior framework combines all three metric dimensions inside one GA-driven selection loop.

### Section 3 — Proposed Framework
- **3.1 Architecture:** Four-stage pipeline — Data Collection (AST via `radon`/`ast`, Git mining via `GitPython`, snapshot isolation) → Preprocessing (correlation filter |r| > 0.95, log-normalize target, min-max scale to [0,1]) → Genetic Optimisation (GA + ANN evaluator) → Analysis.
- **3.2 Feature taxonomy (Table 1) — 19 features across 3 categories:**
  - **A — Structural (8):** avg cyclomatic complexity, halstead volume, number of methods per class, nesting depth, class coupling, maintainability index, LOC, num classes.
  - **B — Textual (5):** avg identifier length, comment ratio, blank line ratio, avg line length, code duplication %.
  - **C — Evolutionary (6):** commit frequency, author count, code churn, added/deleted ratio, days since last change, bug-fix ratio.
  - A correlation filter (|r| > 0.95) typically drops the space to 18 active dimensions.
- **3.3 Snapshot-based temporal isolation:** features computed from commits strictly before snapshot time `ts`; label = bug-fix-keyword commit count in `[ts, ts+Δ)`. This is the paper's strongest, most defensible methodological contribution.
- **3.4 ANN architecture:** 3-hidden-layer MLP (PyTorch), batch norm, dropout, Adam + weight decay, `ReduceLROnPlateau`. 50-trial Optuna Bayesian search (Table 2: LR, hidden sizes, dropout, weight decay, batch size, batchnorm on/off) on the **full feature set only**, then frozen for all GA/baseline/ablation runs. ReLU chosen over the base paper's softplus for faster convergence.
- **3.5 Genetic Algorithm:**
  - Binary chromosome (length ≤ 19; effectively ≤ 18 post-filtering).
  - Fitness: `F(c) = α·(1/(MSE+ε)) + β·(1 − k/n)` — jointly rewards accuracy and feature parsimony.
  - Algorithm 1: population init → memoized fitness evaluation → elitism (top-2) → tournament selection (k=3) → single-point crossover → adaptive mutation (μ₀=0.20 → μ_min=0.03, exponential decay) → early stopping on stagnation.

### Section 4 — Experimental Setup
- **4.1 Subjects (Table 3):** Flask (80 files, 18 active features, 3,600+ commits), Requests (35 files, 14 active features, 5,200+ commits), FastAPI (935 files, 18 active features, 2,100+ commits; GA run ≈ 470s).
- **4.2 Research Questions:** RQ1 (does GA-selection beat all-features?), RQ2 (per-category ablation), RQ3 (cross-repo consistency of GA selections), RQ4 (sensitivity to α, β, population size).
- **4.3 Statistical validation:** 5 independent trials, Wilcoxon signed-rank test (α=0.05), Cohen's d thresholds (per Cohen 1988). 5-fold CV used throughout — justified specifically for the small Requests repo (n=35 → a hold-out split would leave only ~7 test files).

### Section 5 — Results
- **5.1 RQ1 (Table 4, FastAPI, 5 trials):** All-Features ANN MSE=0.2289±0.0071 (18 feat); Random-Subset ANN MSE=0.2281±0.0392 (8 feat, essentially tied with all-features); **GA-ANN MSE=0.1890±0.0168 (8 feat, best)**; XGBoost-on-GA-features MSE=0.3734±0.0367 (worst of all four). Wilcoxon p=0.0625 (misses α=0.05), Cohen's d=1.615 (large). Table 5 repeats the comparison with MAE/R² — GA-ANN wins on all three metrics. Figure 4 visualizes this across all three repos.
- **5.2 GA convergence:** Figure 5 (FastAPI, 10 generations) — sharp MSE drops at generations 1–3 and 7. Figure 6 (Flask, 150 epochs) — train/val MSE curves converge without divergence, final val MSE 0.139.
- **5.3 RQ2 — Ablation (Table 6, FastAPI, stated as **20 trials**, contradicting Table 4's 5 trials): A-only (7 feat) MSE=0.2192 [baseline]; B-only (5 feat) MSE=0.3525 (+60.8%); C-only (6 feat) MSE=0.1919 (−12.5%); A+B (12 feat) MSE=0.2039 (−7.0%); A+C (13 feat) MSE=0.1767 (−19.4%); B+C (11 feat) MSE=0.1724 (−21.4%); **A+B+C / All Features (18 feat) MSE=0.1686 (−23.1%, the single best number in the whole table)**; GA-selected (8 feat) MSE=0.1890 (−13.8% vs A-only). Text concludes evolutionary features give the largest marginal gain and that GA-selection "outperforms even the all-features baseline."
- **5.4 RQ3 — Multi-repo generalization (Tables 7–8):** Flask (k=10, 44.4% reduction, MSE 0.1393, 4A/2B/4C); Requests (k=4, 71.4% reduction, MSE 0.2198, 2A/1B/1C); FastAPI (k=8, 55.6% reduction, MSE 0.2542, 3A/2B/3C). `avg cyclomatic complexity` selected in all three repos; `commit frequency`/`author count` in Flask+FastAPI; `days since last change` in Requests+FastAPI. Table 7 and Table 8 are internally consistent with each other (verified).
- **5.5 RQ4 — Sensitivity (Figure 7, Table 9):** MSE is flat (0.254–0.289) across the α×β×P grid; only α=1.0, β=2.0 is an outlier (steers to 6 features, MSE 0.289). Default (α=1.0, β=0.5, P=10) is near-optimal. Conclusion: framework is robust to hyperparameter choice (≤13.9% MSE variation across the whole grid).

### Section 6 — Discussion
- **6.1 Comparison with base paper (Table 10):** Purely methodological (feature-selection method, metric dimensions, optimisation goal, temporal integrity, target variable, ANN tuning, datasets) — no numerical head-to-head with NNCP. Claims evolutionary metrics give "the largest individual gain (−14.1% MSE)" and GA's cross-dimensional selection contributes "a further −4.5%" over the full feature set.
- **6.2 Generalizability:** Discusses why different repos need different feature-subset sizes (Requests: homogeneous → k=4; Flask: rich history → k=10, lowest MSE).
- **6.3 Practical implications:** CI/PR-prioritization use case; mentions an "interactive dashboard (FastAPI backend + React frontend)" as a contribution, with no evaluation of it.

### Section 7 — Threats to Validity
- **Construct validity:** target variable is a keyword-matched bug-fix-commit count — a known-noisy proxy; no validation against issue-tracker ground truth.
- **Internal validity:** ANN hyperparameters are Optuna-tuned once on the full feature set, then frozen for every GA/ablation run — not re-tuned per chromosome.
- **External validity:** only 3 Python web/HTTP repos; honestly notes that direct MSE comparison with NNCP (UIMS/QUES, Java, lines-changed target) is not meaningful due to language/granularity/target differences — this honesty is explicitly praised by your professor.
- **Conclusion validity:** cites the "20-trial repeated-measurement design" (conflicts with Table 4's 5 trials) and acknowledges compute cost as the limiting factor (FastAPI GA run ≈ 8 minutes... though 4.1 said ≈470s ≈ 7.8 min, consistent).

### Section 8 — Conclusion & Future Work
Summarizes the four design choices and the headline 17.4% MSE reduction / 44–71% feature reduction. Future work: other languages (Java, TypeScript) and industrial codebases; issue-tracker-based labeling; Pareto-front multi-objective GA; temporal non-stationarity of feature importance.

### References
21 references, spanning the CK-metrics lineage (Li & Henry, Chidamber & Kemerer), churn/process-metric literature (Nagappan & Ball, Rahman & Devanbu, Hassan, Zimmermann), the direct base paper (Vescan & Barac-Antonescu, 2025), and tooling citations (Optuna, PyTorch, Cohen's effect-size reference).

---

## PART 2 — Your Professor's Review, Organized

### What's already working (keep these, and say so explicitly in your response letter)
- Clear, coherent thesis integrating three metric dimensions + automated feature selection + leakage prevention.
- Snapshot-based temporal isolation protocol — flagged as a genuine methodological contribution many published papers skip.
- Dual-objective (accuracy + parsimony) fitness function with a thorough α/β sensitivity sweep.
- Ablation study (Table 6) showing evolutionary metrics carry independent signal.
- Honest limitations section, especially the admission that a numeric NNCP comparison isn't meaningful.
- Structure follows ESE/EMSE conventions.

### Major weaknesses (must fix for a high-impact resubmission)
| # | Issue | What's needed |
|---|---|---|
| 1 | Statistical power too low: Wilcoxon p=0.0625 on n=5 trials | 20–30 trials for the main RQ1 comparison; p should clear 0.05 (ideally 0.01) |
| 2 | Improvement is modest; random-subset (0.2281) ≈ all-features (0.2289) | Either strengthen the result or be more explicit that the gain comes from GA feature *quality*, not dimensionality reduction |
| 3 | Generalizability limited to 3 similar Python web/HTTP libraries | Target 8–10 repos, ≥1 non-Python language (Java, to also enable a UIMS/QUES comparison), ideally 1 industrial codebase |
| 4 | Target variable (keyword-matched bug-fix count) is a known-noisy proxy | Validate keyword labels against issue-tracker ground truth on at least a subset |
| 5 | Comparison to NNCP is methodological only, no numbers | Re-implement NNCP's pipeline on Flask/Requests/FastAPI for a fair numeric comparison |
| 6 | Numerical inconsistencies: −14.1% (text) vs −12.5% (Table 6); "20 trials" (§4.3) vs "5 trials" (Table 4); unreproducible −4.5% claim | Audit every number in the text against the tables and fix all of them |
| 7 | No reproducibility artifacts | Public code/data repo, Docker image, documented random seeds |

### Minor issues
- XGBoost-on-GA-features underperforms all-features ANN — surprising and under-discussed; likely an untuned-XGBoost artifact.
- The "interactive dashboard" is claimed as a contribution but never evaluated — drop it or evaluate it.
- Random-Subset ANN nearly matching All-Features ANN is a negative result for the core hypothesis and needs more discussion, not less.
- Optuna tuning is frozen from the full feature set; worth a sensitivity check on whether re-tuning on the GA-selected subset changes anything.
- Wilcoxon test details (paired? two-sided?) and exact statistics should be reported.

---

## PART 3 — Additional Issues Found While Cross-Checking the Numbers

These aren't in your professor's review but showed up when I checked every number against every other number. The first one is, I think, the single most important thing to fix before anything else — it goes to the core claim of the paper.

### 🔴 Critical: Table 6 actually contradicts the paper's central claim
Section 5.3 states: *"the GA-selected 8-feature cross-dimensional subset outperforms even the all-features (A+B+C) baseline."*

But Table 6's own numbers say otherwise:
- All-Features (A+B+C): MSE = **0.1686**
- GA-selected: MSE = **0.1890**

0.1890 is *worse* (higher MSE) than 0.1686 — a **+12.1% degradation**, not an improvement. The GA-selected subset only looks good against the *A-only baseline* (−13.8%), not against the full feature set. This directly undercuts RQ2's stated answer.

It gets more concerning when you line this up against Table 4: there, "All-Features ANN" is reported as MSE = 0.2289 (5 trials) — **35.7% higher** than the 0.1686 figure for the *same model on the same dataset* in Table 6 (20 trials). Since more trials should tighten your variance estimate, not shift the mean by this much, this strongly suggests the 5-trial "All-Features" baseline in Table 4 was an unlucky/noisy estimate — and once you use a fair, equally-powered trial count for *every* condition in Table 4 (which you need to do anyway for weakness #1), you may find GA-ANN no longer beats All-Features. **This needs to be resolved with real re-runs before anything else in the paper, because it's the thesis of RQ1 and RQ2.**

### 🟠 A feature is discussed that doesn't exist in the taxonomy
Section 3.2 says *"weighted methods per class is dropped in full-size repositories because it is near-perfectly collinear with avg cyclomatic complexity"* — but "weighted methods per class" (WMC) is not one of the 19 features in Table 1 or Table 8 (Category A lists "number of methods per class," a different, unweighted metric). This reads like a leftover sentence from CK-metrics literature that wasn't updated for your own feature set.

### 🟠 A second feature is discussed that doesn't exist in the taxonomy
Section 3.2 and Section 5.4 both refer to **"docstring presence"** as a Textual feature ("comment ratio and docstring presence directly measure documentation density"; "textual features, notably docstring presence and comment ratio, are selected in two of three repositories"). But Table 1's Category B list (5 features) has no "docstring presence" — and it's not a row in Table 8 either.

### 🟠 Section 5.4's own selection claim is also wrong about a feature that *does* exist
The same sentence claims comment ratio is "selected in two of three repositories." Table 8 shows comment ratio (B2) selected in **all three** repos (Flask, Requests, FastAPI all marked). So this sentence is inconsistent with your own data on two separate counts.

### 🟡 Minor wording
- 3.2 says "Halstead volume and effort quantify..." but only "halstead volume" is an actual feature (effort isn't separately listed) — tighten the wording.
- The paper calls the same feature "days since last change" (Table 1/8) and "code age days" (§5.4) — pick one name and use it everywhere.

---

## PART 4 — Prioritized Improvement Plan (built for limited time)

Ordered so you get the most credibility-per-hour. **Tier 0 needs no new experiments at all** — just careful editing — so do it first regardless of how much time you have.

### Tier 0 — Text-only fixes (a few hours total, no re-running anything)
1. **Fix every numerical inconsistency** your professor and I found: the −14.1%/−12.5% mismatch, the unreproducible −4.5%, the "20 trials" vs "5 trials" mismatch, and the "docstring presence"/"weighted methods per class" phantom features. Re-derive every percentage in the Discussion section directly from Table 6's numbers rather than writing them from memory.
2. **Rewrite the Section 5.3/6.1 narrative honestly** once the Tier-1 re-run below is done: state clearly whether GA-selection beats all-features or not, and if the margin is thin, say so — reviewers respect an honest "this gain is modest but consistent across category-based ablations" more than an overstated claim they can falsify from your own table.
3. **Add the missing statistical detail**: report the Wilcoxon test as paired/two-sided explicitly, and report the exact W statistic alongside p and Cohen's d.
4. **Fix the "code age days" vs "days since last change" naming** and the Halstead volume/effort wording.
5. **Decide on the dashboard**: either delete the "interactive dashboard" sentence in §6.3, or add 2–3 sentences describing it as an artifact (screenshot + one-paragraph description) without claiming it as an evaluated contribution.
6. **Add an explicit discussion paragraph** owning the Random-Subset ≈ All-Features result as a genuine negative-result finding (this is what a "threats-to-validity"-literate reviewer wants to see, and it costs you nothing but honesty).

### Tier 1 — Fast experimental fixes (re-uses your existing pipeline, ~1 day)
7. **Re-run the RQ1 main comparison (Table 4/5) at 20 trials**, matching the trial count you already use for the ablation (Table 6) — your infrastructure clearly already supports this since Table 6 used 20 trials. This single change: (a) directly answers your professor's #1 weakness (statistical power), and (b) resolves the Table 4 vs. Table 6 contradiction described above, since both will now be measured under the same protocol. Report the new Wilcoxon p and Cohen's d.
8. **Report and fix random seeds**, and add a short "Reproducibility" paragraph/appendix listing library versions, seeds, and hardware.
9. **Publish a code/data repository** (GitHub is enough — a Docker image is a nice-to-have, not essential if time is short) with a README, `requirements.txt`, and the exact commands to reproduce Tables 4–9.
10. **Give XGBoost a fair shot**: run the same Optuna search space (or a comparable one) on XGBoost's hyperparameters rather than using it untuned. This is cheap (XGBoost trains fast) and directly resolves the "surprisingly bad XGBoost" minor issue — either it improves and you get a stronger baseline, or it still underperforms and you can say so with confidence instead of speculation.

### Tier 2 — Medium-effort fixes (1–3 days, still realistic under time pressure)
11. **Add 2 more repositories** in the same domain you already have tooling for (e.g., Django, aiohttp, or httpx) rather than the full "8–10 repos" your professor asked for. This won't fully satisfy the generalizability ask, but going from 3→5 repos with the *same* pipeline is a low-risk way to show meaningful additional evidence without new infrastructure. Be upfront in the paper that full-scale multi-language validation remains future work (see Tier 3).
12. **Small-scale label validation**: pull ~50–100 commits from one repo (Flask is a good choice — smallest history) via the GitHub API, manually or semi-automatically check whether your bug-fix keyword match agrees with the repo's actual issue links/labels, and report a simple precision/recall or agreement number. This is a scoped, one-repo, one-day task that directly answers weakness #4 without needing to relabel the whole dataset.
13. **Re-implement a simple NNCP-style baseline** (Pearson-correlation-filtered CK metrics → the same ANN architecture, no GA) on your own three repos using your own bug-fix-count target. You already have all of Category A (your structural features are essentially the CK-style metrics NNCP uses), so this mostly reuses existing code with the GA step swapped for a correlation threshold. This gives you the numeric NNCP comparison your professor asked for, on equal footing with your own framework.

### Tier 3 — Larger asks: recommend deferring, and say so explicitly
- **Full 8–10 repository, multi-language (Java) evaluation.** This is the right ask for a journal-quality paper, but attempting it under real time pressure risks introducing new bugs or rushed results. Better to do 2–3 well-validated repos now (Tier 2) and state clearly in Future Work that a Java replication against UIMS/QUES is planned — this is a legitimate, professor-approved framing rather than a dodge.
- **Full issue-tracker relabeling of the whole dataset.** The scoped validation in Tier 2 (#12) is the right-sized version of this for your timeline.
- **Docker image.** Useful, but a clean GitHub repo with pinned dependencies and documented seeds gets you 90% of the reproducibility credit for a fraction of the effort.

---

## PART 5 — Suggested Execution Order

If you genuinely have only a few days:

| Day | Focus |
|---|---|
| Day 1 (few hrs) | Tier 0, items 1–6 — pure editing, immediately removes every "reviewers will catch this" red flag |
| Day 1–2 | Tier 1, item 7 — re-run at 20 trials; this is the load-bearing fix, do it before writing anything else about RQ1/RQ2 |
| Day 2 | Tier 1, items 8–10 — seeds, repo, XGBoost re-tune |
| Day 2–3 | Tier 2, item 13 — NNCP baseline (reuses existing structural features + ANN, no GA) |
| Day 3–4 | Tier 2, item 12 — small label-validation study on Flask |
| If time remains | Tier 2, item 11 — add 1–2 more repos |
| Final pass | Rewrite abstract/conclusion numbers to match whatever the Tier-1 re-run actually shows, even if the headline 17.4% changes |

---

### Note on the headline number
Everything downstream (abstract, conclusion, discussion) currently repeats the 17.4%/0.1890 figures from the under-powered 5-trial run. Once you redo Tier 1 item 7, **update every one of those numbers consistently** — an inconsistent headline number across abstract/results/conclusion is exactly the kind of thing that erodes reviewer trust fastest.

I'm happy to help directly with any of these next — for example, drafting the corrected Section 5.3/6.1 text once you have new numbers, writing the NNCP baseline re-implementation, or drafting the README for the code repository.