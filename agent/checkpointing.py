"""
Checkpoint backend factory for LangGraph.

Prefers PostgreSQL when a connection string is provided, with automatic fallback
to in-memory checkpoints when Postgres is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver

DEFAULT_POSTGRES_URI = "postgresql://god@localhost:5432/ielts_memory"


@dataclass
class CheckpointerHandle:
    """Container for a checkpointer instance and optional cleanup callback."""

    checkpointer: Any
    backend: str
    close: Callable[[], None]


def resolve_postgres_uri(explicit_uri: str | None = None) -> str | None:
    """Resolve postgres URI from explicit value or supported environment variables."""
    if explicit_uri:
        return explicit_uri
    for key in ("LANGGRAPH_POSTGRES_URI", "POSTGRES_URI", "DATABASE_URL"):
        value = os.getenv(key)
        if value:
            return value
    return DEFAULT_POSTGRES_URI


def build_checkpointer(postgres_uri: str | None = None, setup: bool = True) -> CheckpointerHandle:
    """
    Build a checkpointer handle.

    If a Postgres URI is provided (or available via env), the function attempts
    to create a Postgres-backed checkpointer. On any failure, it falls back to
    MemorySaver so the application remains usable.
    """
    resolved_uri = resolve_postgres_uri(postgres_uri)
    if not resolved_uri:
        return CheckpointerHandle(checkpointer=MemorySaver(), backend="memory", close=lambda: None)

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except Exception:
        return CheckpointerHandle(checkpointer=MemorySaver(), backend="memory", close=lambda: None)

    try:
        maybe_cm = PostgresSaver.from_conn_string(resolved_uri)
        if hasattr(maybe_cm, "__enter__") and hasattr(maybe_cm, "__exit__"):
            context_manager = maybe_cm
            saver = context_manager.__enter__()

            def _close() -> None:
                context_manager.__exit__(None, None, None)
        else:
            saver = maybe_cm

            def _close() -> None:
                closer = getattr(saver, "close", None)
                if callable(closer):
                    closer()

        if setup and hasattr(saver, "setup"):
            saver.setup()

        return CheckpointerHandle(checkpointer=saver, backend="postgres", close=_close)
    except Exception:
        return CheckpointerHandle(checkpointer=MemorySaver(), backend="memory", close=lambda: None)
