"""Deterministic ID helpers for embedding artifacts."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


def deterministic_id(*parts: object) -> str:
    """Return a stable UUIDv5 string from ordered identity parts."""

    text = "::".join("" if part is None else str(part) for part in parts)
    return str(uuid5(NAMESPACE_URL, text))

