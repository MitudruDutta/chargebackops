"""Thread-safe storage for completed episode grading reports."""

from __future__ import annotations

from threading import Lock

try:
    from .models import GraderReport
except ImportError:  # pragma: no cover
    from models import GraderReport

_LOCK = Lock()
_REPORTS: dict[str, GraderReport] = {}
_LATEST_EPISODE_ID: str | None = None


def record_report(report: GraderReport) -> None:
    """Store a finished grading report."""

    global _LATEST_EPISODE_ID
    with _LOCK:
        _REPORTS[report.episode_id] = report
        _LATEST_EPISODE_ID = report.episode_id


def get_report(episode_id: str | None = None) -> GraderReport | None:
    """Return a report by id or the latest completed one."""

    with _LOCK:
        key = episode_id or _LATEST_EPISODE_ID
        if key is None:
            return None
        return _REPORTS.get(key)
