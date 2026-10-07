"""Exchange sessions shared by price capture, forecasting, and evaluation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any


def utc(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def calendar_name(stock: dict[str, Any]) -> str:
    if stock.get("calendar"):
        return str(stock["calendar"])
    market = str(stock.get("market", "")).upper()
    if market in {"TW", "TAIWAN"} or str(stock.get("ticker", "")).endswith(".TW"):
        return "XTAI"
    if market == "US":
        return "XNYS"
    raise ValueError(f"No exchange calendar configured for {stock.get('ticker')}")


@lru_cache(maxsize=32)
def _calendar(name: str, year: int) -> Any:
    import exchange_calendars as xcals

    return xcals.get_calendar(name, start=f"{year - 2}-01-01", end=f"{year + 2}-12-31")


def session_close(stock: dict[str, Any], session: str) -> datetime:
    calendar = _calendar(calendar_name(stock), int(session[:4]))
    return utc(calendar.session_close(session).to_pydatetime())


def session_pair(stock: dict[str, Any], issued_at: datetime) -> dict[str, Any]:
    """Find the completed reference session and first close after actual issuance."""
    issued = utc(issued_at)
    calendar = _calendar(calendar_name(stock), issued.year)
    local_date = issued.astimezone(calendar.tz).date()
    sessions = calendar.sessions_in_range(local_date - timedelta(days=45), local_date + timedelta(days=45))
    completed = []
    upcoming = []
    for session in sessions:
        close = utc(calendar.session_close(session).to_pydatetime())
        (completed if close <= issued else upcoming).append((session.strftime("%Y-%m-%d"), close))
    if not completed or not upcoming:
        raise ValueError("Cannot resolve reference and target sessions")
    return {
        "calendar": calendar_name(stock),
        "reference_session": completed[-1][0],
        "target_session": upcoming[0][0],
        "target_close_at": upcoming[0][1],
    }


def next_session(stock: dict[str, Any], session: str) -> str:
    calendar = _calendar(calendar_name(stock), int(session[:4]))
    return calendar.next_session(session).strftime("%Y-%m-%d")
