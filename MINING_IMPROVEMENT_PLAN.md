# Repository Mining Improvement Plan

## Objective

Make repository mining faster, reproducible, resumable, and easier to audit
without changing the existing feature names or the research target definition.

## Problems identified

1. Each file independently traversed Git history with
   `repo.iter_commits(paths=...)`.
2. Each file launched another `git log --numstat` subprocess for churn.
3. The snapshot boundary came from the current clock, so identical commands
   could produce different temporal windows on different days.
4. Interrupted mining lost completed work and offered no validated resume path.
5. There was no explicit command-line interface for reproducible mining
   configuration.

## Implementation plan

### Phase 1 — Shared repository history index

- Run one repository-wide `git log --numstat` command.
- Parse commit metadata, file paths, insertions, and deletions once.
- Reuse the indexed records for commit frequency, authors, bug-fix labels,
  target counts, churn, and code-age calculations.
- Retrieve historical file content from the indexed commit SHA.

### Phase 2 — Deterministic snapshot configuration

- Accept an explicit UTC `--as-of` timestamp.
- Derive the snapshot boundary from `as_of - timeframe_months`.
- Record the source commit and both timestamps in the cache configuration.
- Preserve the previous current-time behavior when `--as-of` is omitted, while
  printing the boundary so it can be pinned in a later run.

### Phase 3 — Safe resumable caching

- Cache each file's extracted feature dictionary.
- Validate cache reuse against schema version, source commit, timeframe,
  snapshot boundary, and selected file count.
- Save atomically through a temporary file and replace operation.
- Save periodically during mining so an interruption loses little work.

### Phase 4 — Operational interface and verification

- Add CLI flags for repository, output, time window, file cap, snapshot date,
  and cache path.
- Keep the existing Python API backward compatible.
- Verify first-run extraction, cache-hit reruns, syntax, and output integrity.

## Implemented result

The phases above are implemented in `src/data_collector.py`.

Example reproducible command:

```bash
./venv/bin/python -m src.data_collector \
  --repo test_repos/flask \
  --output data/flask_dataset_mined.csv \
  --timeframe-months 12 \
  --as-of 2026-09-27T00:00:00+00:00 \
  --cache data/flask_dataset_mined.csv.mining-cache.json
```

## Verification results

On the pinned Flask checkout (`d73fa1cdcbd8`), the new miner indexed 643
history paths once and extracted 83 files in approximately 1.73 seconds. A
repeat with the same configuration reused 83/83 cached files and completed in
approximately 1.11 seconds. The output was generated with the same source
commit, snapshot boundary, and cache metadata.

The benchmark demonstrates deterministic resumability and avoids repeated
per-file Git history work. It is not presented as a hardware-independent
speedup percentage; exact runtime depends on the machine, Git object cache,
and repository size.

## Compatibility and scope

- Existing structural, textual, evolutionary, and target feature names are
  preserved.
- The keyword bug-fix heuristic remains unchanged; its separate validation
  result is documented in `results/LABEL_VALIDATION.md`.
- Existing user changes to `setup.sh`, `start.sh`, frontend files, `plan.md`,
  and `tech_plan.md` were not included in this branch commit.
