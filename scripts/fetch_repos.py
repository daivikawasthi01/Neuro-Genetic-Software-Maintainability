#!/usr/bin/env python3
"""Clone the evaluation repositories at recorded immutable revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORIES = {
    "flask": {
        "url": "https://github.com/pallets/flask.git",
        "commit": "d73fa1cdcbd8b1465c151db8924ba58b1dd14e35",
    },
    "requests": {
        "url": "https://github.com/psf/requests.git",
        "commit": "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
    },
    "fastapi": {
        "url": "https://github.com/fastapi/fastapi.git",
        "commit": "192b12197eb04c2b4a691cce7d87261b21716714",
    },
}


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def fetch(name: str, destination: Path) -> dict:
    spec = REPOSITORIES[name]
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", spec["url"], str(destination)], check=True)
    else:
        run(["git", "fetch", "--all", "--tags"], cwd=destination)
    run(["git", "checkout", "--detach", spec["commit"]], cwd=destination)
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual != spec["commit"]:
        raise RuntimeError(f"{name}: expected {spec['commit']}, got {actual}")
    return {**spec, "resolved_commit": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", choices=sorted(REPOSITORIES), default=list(REPOSITORIES))
    parser.add_argument("--root", default="test_repos")
    parser.add_argument("--manifest", default="results/REPOSITORIES.json")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {},
    }
    for name in args.names:
        print(f"[{name}] {REPOSITORIES[name]['commit']}")
        manifest["repositories"][name] = fetch(name, root / name)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
