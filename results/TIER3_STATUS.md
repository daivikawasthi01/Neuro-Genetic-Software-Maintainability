# Tier 3 implementation status

## Implemented

- `scripts/collect_java_metrics.py` provides a reproducible Java collection
  entry point with deterministic file ordering, optional file bounds, pinned
  source-commit metadata, dataset SHA-256, and explicitly named lexical metric
  definitions. It is an adapter scaffold, not a claim that lexical Java
  metrics are identical to Python/Radon metrics.
- `scripts/validate_bug_labels.py --sample-size 0` now supports full local
  commit-history validation when authenticated GitHub API capacity is
  available. Sampled and blocked rows remain persisted exactly.
- Docker Compose configuration was validated successfully. The Dockerfiles
  now install the CPU XGBoost wheel without pulling the optional NCCL package.
  The pinned API dependencies were made compatible with Python 3.11.

## Environment-specific verification

The default ARM64 Docker build cannot resolve the pinned XGBoost wheel. The
documented build target is therefore AMD64:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build backend research
```

On this run, the AMD64 build reached dependency installation but the host's
network became unavailable while downloading packages. Consequently, the
Compose file is syntax-verified, but a completed image build is not claimed.

## Still requires an external study run

The repository does not contain a selected Java upstream repository or a
manually reviewed full issue/PR ground-truth set. Those choices materially
change the research result, so Tier 3 adds the reproducible tooling and
records the blockers rather than inventing a Java comparison or full-history
accuracy number.
