from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _to_jsonable(obj: Any) -> Any:
    """Convert common Python/numpy/path objects to JSON-serializable values."""
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # numpy scalar and similar objects usually provide `.item()`
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return obj


def _run_git(project_root: Path, args: Sequence[str]) -> str | None:
    try:
        out = subprocess.run(
            list(args),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return out.stdout.strip()


def collect_git_info(project_root: Path) -> dict[str, Any]:
    """Collect lightweight git metadata for experiment traceability."""
    branch = _run_git(project_root, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git(project_root, ["git", "rev-parse", "--short", "HEAD"])
    status = _run_git(project_root, ["git", "status", "--porcelain"])
    dirty = None if status is None else bool(status)
    return {
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
    }


def write_experiment_log(
    *,
    project_root: Path,
    task: str,
    mode: str,
    command: Sequence[str],
    status: str,
    duration_sec: float,
    config: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    error: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """Append one experiment record as JSONL and return the log file path."""
    project_root = Path(project_root).resolve()
    if log_dir is None:
        log_dir = project_root / "experiment_logs"
    else:
        log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{task}_{mode}.jsonl"
    now_local = datetime.now().astimezone()
    record = {
        "timestamp_local": now_local.isoformat(),
        "task": task,
        "mode": mode,
        "status": status,
        "duration_sec": round(float(duration_sec), 4),
        "command": list(command),
        "git": collect_git_info(project_root),
        "config": _to_jsonable(config or {}),
        "result": _to_jsonable(result or {}),
        "extra": _to_jsonable(extra or {}),
        "error": error,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return log_path
