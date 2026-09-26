"""Shared experiment configuration, seeding, and provenance utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASE_SEED = 42
DEFAULT_N_TRIALS = 20
DEFAULT_N_FOLDS = 5


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch when available."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def trial_seed(base_seed: int, trial_index: int) -> int:
    """Return the deterministic seed assigned to a zero-based trial."""
    return int(base_seed) + int(trial_index)


def git_commit(path: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_metadata(config: dict[str, Any], *, repo_root: str | Path = ".") -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "config": config,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def append_run_log(
    task_id: str,
    script: str,
    config: dict[str, Any],
    artifact: str,
    summary: str,
    *,
    path: str | Path = "results/RUN_LOG.md",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiment Run Log\n\n"
            "| Task | Script | Config / seed | Started (UTC) | Artifact | Summary |\n"
            "|---|---|---|---|---|---|\n"
        )
    started = datetime.now(timezone.utc).isoformat()
    compact_config = json.dumps(config, sort_keys=True).replace("|", "\\|")
    with path.open("a") as handle:
        handle.write(
            f"| {task_id} | `{script}` | `{compact_config}` | {started} | "
            f"`{artifact}` | {summary} |\n"
        )
