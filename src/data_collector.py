import os
import ast
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from radon.complexity import cc_visit
from radon.metrics import h_visit
import git
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta


BUG_KEYWORDS = ('fix', 'bug', 'patch', 'issue', 'resolve', 'error')


@dataclass(frozen=True)
class HistoryRecord:
    sha: str
    committed_at: datetime
    author_email: str
    message: str
    insertions: int
    deletions: int


class RepositoryHistoryIndex:
    """One-pass Git history index shared by every file in a mining run.

    The previous implementation called ``repo.iter_commits(paths=...)`` and
    then launched ``git log --numstat`` once per file. This index performs one
    repository-wide walk and groups the resulting records by path, avoiding
    thousands of repeated object-database traversals and subprocesses.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.source_commit = self.repo.head.commit.hexsha
        self.by_path: dict[str, list[HistoryRecord]] = {}
        self._build()

    @staticmethod
    def _parse_int(value: str) -> int:
        try:
            return int(value) if value != '-' else 0
        except ValueError:
            return 0

    def _build(self) -> None:
        command = [
            'git', 'log', self.source_commit, '--no-renames', '--numstat',
            '--date=iso-strict',
            '--format=__COMMIT__%H%x09%aI%x09%ae%x09%s', '--', '.',
        ]
        result = subprocess.run(
            command, cwd=self.repo_path, capture_output=True, text=True,
            check=True,
        )
        current = None
        for line in result.stdout.splitlines():
            if line.startswith('__COMMIT__'):
                fields = line[len('__COMMIT__'):].split('\t', 3)
                if len(fields) == 4:
                    current = {
                        'sha': fields[0],
                        'committed_at': datetime.fromisoformat(fields[1]).astimezone(timezone.utc),
                        'author_email': fields[2],
                        'message': fields[3],
                    }
                continue
            if current is None:
                continue
            fields = line.split('\t', 2)
            if len(fields) != 3:
                continue
            insertions, deletions, path = fields
            record = HistoryRecord(
                sha=current['sha'],
                committed_at=current['committed_at'],
                author_email=current['author_email'],
                message=current['message'],
                insertions=self._parse_int(insertions),
                deletions=self._parse_int(deletions),
            )
            self.by_path.setdefault(path, []).append(record)

        for path in self.by_path:
            self.by_path[path].sort(key=lambda record: record.committed_at, reverse=True)

    def records_for(self, file_rel_path: str) -> list[HistoryRecord]:
        return self.by_path.get(file_rel_path, [])

    def historical_code(self, record: HistoryRecord, file_rel_path: str) -> str | None:
        try:
            blob = self.repo.commit(record.sha).tree[file_rel_path]
            return blob.data_stream.read().decode('utf-8', errors='replace')
        except (KeyError, AttributeError, git.exc.BadName):
            return None


class MiningCache:
    """Metadata-validated JSON cache for resumable feature extraction."""

    def __init__(self, path: str | None, metadata: dict):
        self.path = Path(path) if path else None
        self.metadata = metadata
        self.entries: dict[str, dict] = {}
        if self.path and self.path.exists():
            try:
                payload = json.loads(self.path.read_text())
                if payload.get('metadata') == metadata:
                    self.entries = payload.get('entries', {})
            except (OSError, json.JSONDecodeError):
                self.entries = {}

    def get(self, file_rel_path: str) -> dict | None:
        entry = self.entries.get(file_rel_path)
        return entry.get('features') if entry else None

    def put(self, file_rel_path: str, features: dict) -> None:
        self.entries[file_rel_path] = {'features': features}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps({
            'metadata': self.metadata,
            'entries': self.entries,
        }, indent=2, sort_keys=True) + '\n')
        temporary.replace(self.path)


def collect_structural_metrics(code):
    if not code:
        return {}

    metrics = {}

    try:
        blocks = cc_visit(code)
        metrics['avg_cyclomatic_complexity'] = (
            sum(b.complexity for b in blocks) / max(1, len(blocks))
        )
    except Exception:
        metrics['avg_cyclomatic_complexity'] = 0

    try:
        halstead = h_visit(code)
        metrics['halstead_volume'] = halstead.total.volume
        metrics['halstead_effort'] = halstead.total.effort
    except Exception:
        metrics['halstead_volume'] = 0
        metrics['halstead_effort'] = 0

    try:
        tree    = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        metrics['depth_of_inheritance_tree'] = max(
            [len(c.bases) for c in classes] + [0]
        )

        total_methods = sum(
            len([m for m in c.body if isinstance(m, ast.FunctionDef)])
            for c in classes
        )
        metrics['number_of_methods_per_class'] = total_methods / max(1, len(classes))
        metrics['weighted_methods_per_class']  = (
            metrics['number_of_methods_per_class']
            * metrics.get('avg_cyclomatic_complexity', 1)
        )

        _NESTING_TYPES = (ast.If, ast.For, ast.While, ast.With, ast.Try,
                          ast.ExceptHandler)

        def _max_nesting(node, depth=0):
            child_depths = [
                _max_nesting(child,
                             depth + (1 if isinstance(child, _NESTING_TYPES) else 0))
                for child in ast.iter_child_nodes(node)
            ]
            return max(child_depths) if child_depths else depth

        metrics['nesting_depth'] = _max_nesting(tree)

        defined_names = {c.name for c in classes} | {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        metrics['class_coupling'] = len(referenced_names - defined_names)

    except Exception:
        metrics['depth_of_inheritance_tree'] = 0
        metrics['number_of_methods_per_class'] = 0
        metrics['weighted_methods_per_class']  = 0
        metrics['nesting_depth']               = 0
        metrics['class_coupling']              = 0

    return metrics


def collect_textual_metrics(code):
    if not code:
        return {}

    lines       = code.split('\n')
    total_lines = len(lines)
    if total_lines == 0:
        return {}

    empty_lines   = sum(1 for l in lines if l.strip() == '')
    comment_lines = sum(1 for l in lines if l.strip().startswith('#'))

    metrics = {
        'comment_density':    comment_lines / total_lines,
        'whitespace_ratio':   empty_lines   / total_lines,
        'docstring_presence': 1 if ('"""' in code or "'''" in code) else 0,
    }

    try:
        tree        = ast.parse(code)
        identifiers = set(
            [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
            + [
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                     ast.AsyncFunctionDef))
            ]
        )
        metrics['avg_identifier_length'] = (
            sum(len(name) for name in identifiers) / max(1, len(identifiers))
        )
    except Exception:
        metrics['avg_identifier_length'] = 0

    non_empty = [l.strip() for l in lines if l.strip()]
    if non_empty:
        counts     = Counter(non_empty)
        duplicated = sum(v - 1 for v in counts.values() if v > 1)
        metrics['code_duplication_pct'] = duplicated / len(non_empty)
    else:
        metrics['code_duplication_pct'] = 0.0

    return metrics


def _get_churn_stats(repo_path: str, file_rel_path: str,
                     before_date) -> tuple:
    """
    Get total insertions and deletions for a file up to before_date using
    a single git log --numstat subprocess call.

    This replaces the old commit.stats.total approach which ran git diff
    once per commit (O(n_commits) diffs). git log --numstat retrieves all
    diff stats in one call regardless of commit count — typically 10-50ms
    per file, safe for a FastAPI backend.

    Returns (total_insertions, total_deletions).
    Binary files and renames produce '-' instead of numbers — these are
    treated as 0 and skipped safely.
    """
    before_str = before_date.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        result = subprocess.run(
            [
                "git", "log",
                "--numstat",        # tab-separated: insertions deletions filename
                "--format=",        # suppress commit headers — only stat lines
                f"--before={before_str}",
                "--follow",         # follow renames
                "--",
                file_rel_path,
            ],
            capture_output=True, text=True,
            cwd=repo_path, timeout=30
        )
    except subprocess.TimeoutExpired:
        return 0, 0
    except Exception:
        return 0, 0

    total_insertions = 0
    total_deletions  = 0

    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        try:
            # Binary files show '-' instead of a number — skip them
            ins  = int(parts[0]) if parts[0].strip() not in ('-', '') else 0
            dels = int(parts[1]) if parts[1].strip() not in ('-', '') else 0
            total_insertions += ins
            total_deletions  += dels
        except (ValueError, IndexError):
            continue

    return total_insertions, total_deletions



def collect_evolutionary_metrics_and_target(
        repo_path, file_rel_path, timeframe_months=12,
        history_index: RepositoryHistoryIndex | None = None,
        snapshot_date: datetime | None = None):
    """
    Extract Category C metrics and target ground truth (bug-proneness score).

    timeframe_months=12 (production default):
        12 months gives far more bug-fix commits in the target window and
        richer evolutionary metric history. The cloud deployment used 3 months
        to avoid timeouts on fresh shallow clones.

    No max_count cap on iter_commits:
        The max_count=100 limit was a cloud workaround. Locally we walk the
        full commit history for accurate commit_frequency, bug_fix_ratio, and
        author_count values.

    added_deleted_ratio and code_churn via git log --numstat:
        Single subprocess call per file (not one diff per commit). Returns
        total insertions and deletions across all past commits. ~10-50ms
        per file — safe for FastAPI backend.

    Fallback for empty past_commits:
        If no commits pre-date the snapshot, all available commits are used
        so the file is not silently dropped from the dataset.
    """
    try:
        index = history_index or RepositoryHistoryIndex(repo_path)
    except (git.exc.InvalidGitRepositoryError, subprocess.CalledProcessError):
        print(f"Not a valid git repository: {repo_path}")
        return {}, None

    commits_touching_file = index.records_for(file_rel_path)

    if not commits_touching_file:
        return {}, None

    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc) - relativedelta(months=timeframe_months)

    past_commits   = []
    future_commits = []

    for c in commits_touching_file:
        commit_date = c.committed_at
        if commit_date <= snapshot_date:
            past_commits.append(c)
        else:
            future_commits.append(c)

    # Fallback: if no commits pre-date the snapshot use all available commits
    if not past_commits and commits_touching_file:
        past_commits   = commits_touching_file
        future_commits = []

    if not past_commits:
        return {}, None

    past_authors   = set()
    past_bug_fixes = 0
    bug_keywords   = BUG_KEYWORDS

    for commit in past_commits:
        past_authors.add(commit.author_email)
        if any(kw in commit.message.lower() for kw in bug_keywords):
            past_bug_fixes += 1

    target_bug_fixes = sum(
        1 for c in future_commits
        if any(kw in c.message.lower() for kw in bug_keywords)
    )

    try:
        code_age_days = (
            snapshot_date
            - past_commits[-1].committed_at
        ).days
    except IndexError:
        code_age_days = 0

    # Churn is already available from the one-pass repository history index.
    insertions = sum(record.insertions for record in past_commits)
    deletions = sum(record.deletions for record in past_commits)
    code_churn          = insertions + deletions
    added_deleted_ratio = insertions / max(1, deletions)

    metrics = {
        'commit_frequency':     len(past_commits),
        'author_count':         len(past_authors),
        'bug_fix_ratio':        past_bug_fixes / max(1, len(past_commits)),
        'code_age_days':        max(0, code_age_days),
        'code_churn':           code_churn,
        'added_deleted_ratio':  added_deleted_ratio,
        'target_bug_proneness': target_bug_fixes,
    }

    # Retrieve file content at snapshot date
    historical_code  = None
    last_past_commit = past_commits[0]
    historical_code = index.historical_code(last_past_commit, file_rel_path)

    return metrics, historical_code


def extract_all_metrics_for_file(repo_path, file_rel_path,
                                 timeframe_months=12,
                                 history_index: RepositoryHistoryIndex | None = None,
                                 snapshot_date: datetime | None = None):
    evolutionary_metrics, historical_code = collect_evolutionary_metrics_and_target(
        repo_path, file_rel_path, timeframe_months=timeframe_months,
        history_index=history_index, snapshot_date=snapshot_date,
    )

    if historical_code is None:
        return {}

    structural = collect_structural_metrics(historical_code)
    textual    = collect_textual_metrics(historical_code)

    return {**structural, **textual, **evolutionary_metrics}


def build_dataset_from_repo(repo_path, output_csv_path,
                             timeframe_months=12, max_files=None,
                             as_of_date=None, cache_path=None):
    """
    Mine all Python files in repo_path and write a CSV to output_csv_path.

    timeframe_months (default 12):
        Production default. Controls snapshot lookback window.
        Cloud deployment used 3 months to avoid timeouts.
    max_files (default None):
        Optional deterministic cap applied after sorting relative paths. This
        is useful for bounded large-repository experiments and is recorded by
        the caller in the experiment configuration.
    as_of_date (default None):
        UTC boundary in ISO format. If omitted, the current UTC time is used
        and recorded so a later run can pin the same boundary.
    cache_path (default derived from output_csv_path):
        Metadata-validated resumable feature cache. A cache is ignored when
        its source commit or mining configuration differs.
    """
    print(f"\nScanning repository: {repo_path}")
    print(f"Snapshot window: {timeframe_months} months back")
    dataset = []

    history_index = RepositoryHistoryIndex(repo_path)
    if as_of_date:
        snapshot_boundary = datetime.fromisoformat(as_of_date)
        if snapshot_boundary.tzinfo is None:
            snapshot_boundary = snapshot_boundary.replace(tzinfo=timezone.utc)
        snapshot_boundary = snapshot_boundary.astimezone(timezone.utc)
    else:
        snapshot_boundary = datetime.now(timezone.utc)
    snapshot_date = snapshot_boundary - relativedelta(months=timeframe_months)

    py_files = []
    for root, _, files in os.walk(repo_path):
        if '.git' in root or '__pycache__' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    py_files.sort(key=lambda path: os.path.relpath(path, repo_path).replace('\\', '/'))
    if max_files is not None:
        py_files = py_files[:max_files]
    selected_file_count = len(py_files)
    cache = MiningCache(
        cache_path or f"{output_csv_path}.mining-cache.json",
        {
            'schema_version': 2,
            'source_commit': history_index.source_commit,
            'timeframe_months': timeframe_months,
            'snapshot_boundary': snapshot_boundary.isoformat(),
            'snapshot_date': snapshot_date.isoformat(),
            'selected_file_count': selected_file_count,
        },
    )
    cache_hits = 0
    print(f"Found {len(py_files)} Python files. Mining (this takes a while "
          f"on large repos)...")

    for i, abs_path in enumerate(py_files, 1):
        file_rel_path = os.path.relpath(abs_path, repo_path).replace('\\', '/')

        if i % 50 == 0 or i == len(py_files):
            print(f"  [{i:4d}/{len(py_files)}] {file_rel_path}")

        features = cache.get(file_rel_path)
        if features is not None:
            cache_hits += 1
        else:
            features = extract_all_metrics_for_file(
                repo_path, file_rel_path,
                timeframe_months=timeframe_months,
                history_index=history_index,
                snapshot_date=snapshot_date,
            )
            if features:
                cache.put(file_rel_path, features)

        if features:
            features['file_name'] = file_rel_path
            dataset.append(features)

        if i % 25 == 0:
            cache.save()

    df = pd.DataFrame(dataset)

    if df.empty or 'file_name' not in df.columns:
        raise ValueError(
            f"No valid Python files extracted from '{repo_path}'. "
            f"({len(py_files)} .py files found — all returned empty features.) "
            "Check that the repo has enough commit history."
        )

    cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
    df   = df[cols]

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    cache.save()

    bug_prone = (df['target_bug_proneness'] > 0).sum()
    print(f"\nDataset saved: {len(df)} files | "
          f"{bug_prone} bug-prone ({bug_prone/len(df)*100:.1f}%) | "
          f"→ {output_csv_path}")
    print(f"Mining history index: {len(history_index.by_path)} files | "
          f"cache hits: {cache_hits}/{len(py_files)} | "
          f"source commit: {history_index.source_commit[:12]}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Mine repository features')
    parser.add_argument('--repo', default='test_repos/flask')
    parser.add_argument('--output', default='data/flask_dataset.csv')
    parser.add_argument('--timeframe-months', type=int, default=12)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--as-of', default=None,
                        help='UTC ISO timestamp used as the snapshot boundary')
    parser.add_argument('--cache', default=None)
    args = parser.parse_args()
    build_dataset_from_repo(
        args.repo, args.output,
        timeframe_months=args.timeframe_months,
        max_files=args.max_files,
        as_of_date=args.as_of,
        cache_path=args.cache,
    )
