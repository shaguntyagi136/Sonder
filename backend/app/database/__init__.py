from __future__ import annotations

from app.core.config import Settings
from app.database.base import Repository


def create_repository(settings: Settings) -> Repository:
    """Factory: the only place that knows which concrete storage backend is in use."""
    from app.database.sqlite_repo import SQLiteRepository

    return SQLiteRepository(settings.sqlite_target)


__all__ = ["Repository", "create_repository"]
