"""One canonical timestamp format for storage: UTC, fixed width, so string order == time order."""
from __future__ import annotations

from datetime import datetime, timezone

_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def to_iso(dt: datetime) -> str:
    return ensure_utc(dt).strftime(_FMT)


def from_iso(value: str) -> datetime:
    return datetime.strptime(value, _FMT).replace(tzinfo=timezone.utc)
