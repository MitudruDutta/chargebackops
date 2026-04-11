"""Thread-safe storage for completed episode grading reports with file persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

try:
    from .models import GraderReport
except ImportError:  # pragma: no cover
    from models import GraderReport

_LOCK = Lock()
_REPORTS: dict[str, GraderReport] = {}
_LATEST_EPISODE_ID: str | None = None
_MAX_REPORTS = 100
_LOG_DIR = Path(os.getenv("EPISODE_LOG_DIR", "episode_logs"))


def _persist(report: GraderReport) -> None:
    """Append report to a JSON-lines log file (best-effort, non-blocking)."""

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / "episodes.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(report.model_dump(), default=str) + "\n")
    except OSError:
        pass


def record_report(report: GraderReport) -> None:
    """Store a finished grading report."""

    global _LATEST_EPISODE_ID
    with _LOCK:
        if len(_REPORTS) >= _MAX_REPORTS:
            oldest = next(iter(_REPORTS))
            del _REPORTS[oldest]
        _REPORTS[report.episode_id] = report
        _LATEST_EPISODE_ID = report.episode_id
    _persist(report)


def get_report(episode_id: str | None = None) -> GraderReport | None:
    """Return a report by id or the latest completed one."""

    with _LOCK:
        key = episode_id or _LATEST_EPISODE_ID
        if key is None:
            return None
        return _REPORTS.get(key)


def list_reports() -> list[GraderReport]:
    """Return all stored reports ordered by recency."""

    with _LOCK:
        return list(reversed(_REPORTS.values()))
