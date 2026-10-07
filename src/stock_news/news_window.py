from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def latest_completed_window(
    timezone_name: str = "Asia/Taipei",
    start_hour: int = 8,
    now: datetime | None = None,
    window_hours: int = 48,
) -> tuple[datetime, datetime]:
    """Return the latest completed [start, end) window as UTC datetimes."""
    if not 0 <= start_hour <= 23:
        raise ValueError("NEWS_WINDOW_START_HOUR must be between 0 and 23")
    if window_hours < 1:
        raise ValueError("NEWS_WINDOW_HOURS must be at least 1")
    zone = ZoneInfo(timezone_name)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    end_local = local_now.replace(
        hour=start_hour, minute=0, second=0, microsecond=0
    )
    if end_local > local_now:
        end_local -= timedelta(days=1)
    start_local = end_local - timedelta(hours=window_hours)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def next_run_time(
    timezone_name: str = "Asia/Taipei", start_hour: int = 8, now: datetime | None = None
) -> datetime:
    zone = ZoneInfo(timezone_name)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    next_local = local_now.replace(
        hour=start_hour, minute=0, second=0, microsecond=0
    )
    if next_local <= local_now:
        next_local += timedelta(days=1)
    return next_local.astimezone(timezone.utc)
