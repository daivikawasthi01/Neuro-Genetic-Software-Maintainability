#!/usr/bin/env python3
"""Collect a reproducible, parser-independent Java metrics dataset.

This is the Tier 3 language-extension entry point. It intentionally uses
conservative lexical metrics so a Java repository can be acquired and
audited without silently claiming that Python's Radon metrics are equivalent
to Java metrics. The output is suitable for a later cross-language adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import git
import pandas as pd

from src.reproducibility import DEFAULT_BASE_SEED, git_commit, write_json


BUG_WORDS = ("fix", "bug", "patch", "issue", "resolve", "error")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _metrics(path: Path, repo: git.Repo) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    code_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("//")]
    comments = sum(1 for line in lines if line.lstrip().startswith(("//", "/*", "*")))
    methods = len(re.findall(r"\b(?:public|private|protected|static|final|abstract|native|synchronized|\s)+[\w<>\[\], ?]+\s+\w+\s*\([^;{}]*\)\s*(?:throws [^{]+)?\{", text))
    classes = len(re.findall(r"\b(?:class|interface|enum|record)\s+\w+", text))
    imports = len(re.findall(r"^\s*import\s+", text, re.MULTILINE))
    control_flow = len(re.findall(r"\b(?:if|for|while|case|catch|switch|\?|&&|\|\|)\b", text))
    nesting = max(((len(line) - len(line.lstrip())) // 4 for line in lines), default=0)
    commits = list(repo.iter_commits(paths=str(path.relative_to(repo.working_tree_dir))))
    bug_commits = sum(
        any(word in commit.message.lower() for word in BUG_WORDS)
        for commit in commits
    )
    return {
        "file_name": str(path.relative_to(repo.working_tree_dir)).replace("\\", "/"),
        "loc": len(lines),
        "code_lines": len(code_lines),
        "comment_lines": comments,
        "method_count": methods,
        "class_count": classes,
        "import_count": imports,
        "cyclomatic_proxy": control_flow + 1,
        "nesting_depth": nesting,
        "commit_frequency": len(commits),
        "target_bug_proneness": bug_commits,
    }


def collect(repo_path: str, output: str, max_files: int | None, seed: int) -> dict:
    repo = git.Repo(repo_path)
    root = Path(repo.working_tree_dir)
    files = sorted(
        path for path in root.rglob("*.java")
        if ".git" not in path.parts and "target" not in path.parts
    )
    if max_files is not None:
        files = files[:max_files]
    rows = [_metrics(path, repo) for path in files]
    frame = pd.DataFrame(rows)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_path": str(root),
        "source_commit": git_commit(root),
        "seed": seed,
        "max_files": max_files,
        "n_files": len(rows),
        "dataset_sha256": _sha256(output_path),
        "metric_definition": "lexical Java metrics; cyclomatic_proxy is not Radon-equivalent",
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_json(metadata_path, metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path")
    parser.add_argument("output")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()
    print(json.dumps(collect(args.repo_path, args.output, args.max_files, args.seed), indent=2))
