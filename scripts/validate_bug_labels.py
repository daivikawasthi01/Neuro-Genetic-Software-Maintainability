#!/usr/bin/env python3
"""Validate the keyword bug-fix heuristic against GitHub PR labels.

The script samples local commits reproducibly, then uses the GitHub commit ->
pull-request endpoint as an independent signal. If the API is unavailable, it
writes the exact sample and blocker without inventing agreement statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import git
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

from src.reproducibility import (
    DEFAULT_BASE_SEED,
    append_run_log,
    make_metadata,
    write_json,
)


BUG_KEYWORDS = ("fix", "bug", "patch", "issue", "resolve", "error")
BUG_LABELS = {"bug", "bugfix", "defect", "regression", "security"}


def heuristic_label(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in BUG_KEYWORDS)


def github_pr_labels(owner: str, repo: str, sha: str, token: str | None) -> tuple[bool | None, str | None]:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/pulls"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "neuro-genetic-maintainability-label-validation",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            pulls = json.loads(response.read().decode("utf-8"))
        labels = {
            label.get("name", "").lower()
            for pull in pulls
            for label in pull.get("labels", [])
        }
        return bool(labels & BUG_LABELS), ",".join(sorted(labels))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return None, str(exc)


def run_validation(
    repo_path: str,
    owner: str,
    repo_name: str,
    *,
    sample_size: int = 50,
    seed: int = DEFAULT_BASE_SEED,
    output_path: str = "results/label_validation.json",
    token: str | None = None,
) -> dict:
    repository = git.Repo(repo_path)
    commits = list(repository.iter_commits())
    rng = random.Random(seed)
    if sample_size <= 0:
        selected = commits
    else:
        selected = rng.sample(commits, min(sample_size, len(commits)))
    records = []
    blocked = []
    for commit in selected:
        heuristic = heuristic_label(commit.message)
        ground_truth, detail = github_pr_labels(owner, repo_name, commit.hexsha, token)
        record = {
            "commit": commit.hexsha,
            "message": commit.message.splitlines()[0],
            "keyword_heuristic": heuristic,
            "github_bug_label": ground_truth,
            "github_labels": detail,
        }
        records.append(record)
        if ground_truth is None:
            blocked.append(record)

    payload = {
        "metadata": make_metadata({
            "repo_path": repo_path,
            "github_repository": f"{owner}/{repo_name}",
            "sample_size": sample_size,
            "sample_selection": "all local commits" if sample_size <= 0 else "seeded random sample",
            "seed": seed,
            "ground_truth": "GitHub pull-request labels containing a bug-like label",
        }),
        "sample": records,
        "blocked_commits": blocked,
    }
    valid = [row for row in records if row["github_bug_label"] is not None]
    if valid:
        y_true = np.asarray([row["github_bug_label"] for row in valid], dtype=bool)
        y_pred = np.asarray([row["keyword_heuristic"] for row in valid], dtype=bool)
        payload["metrics"] = {
            "n_evaluated": len(valid),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        }
        payload["status"] = "complete" if not blocked else "partial"
    else:
        payload["metrics"] = None
        payload["status"] = "blocked"
        payload["blocker"] = (
            "GitHub PR-label lookups were unavailable for the sampled commits; "
            "no agreement metrics were calculated."
        )

    write_json(output_path, payload)
    append_run_log(
        "Task 2.2", "scripts/validate_bug_labels.py",
        {"repo": repo_name, "sample_size": sample_size, "seed": seed},
        output_path, f"Bug-label validation status: {payload['status']}.",
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate keyword bug labels")
    parser.add_argument("--repo-path", default="test_repos/flask")
    parser.add_argument("--owner", default="pallets")
    parser.add_argument("--repo", default="flask")
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of commits to sample; use 0 for the full local history",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output", default="results/label_validation.json")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    run_validation(
        args.repo_path, args.owner, args.repo,
        sample_size=args.sample_size,
        seed=args.seed,
        output_path=args.output,
        token=args.token,
    )
