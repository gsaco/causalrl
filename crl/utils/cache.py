"""Lightweight in-memory cache for reusable nuisance fits."""

from __future__ import annotations

import weakref
from typing import Any, Callable

_CACHE: dict[int, tuple[weakref.ref[object], dict[Any, Any]]] = {}


def get_cache(dataset: object) -> dict[Any, Any]:
    """Return the cache dictionary for a dataset object."""

    key = id(dataset)
    entry = _CACHE.get(key)
    if entry is not None:
        ref_obj, cache = entry
        if ref_obj() is dataset:
            return cache
    cache: dict[Any, Any] = {}

    def _cleanup(_ref: weakref.ref[object]) -> None:
        _CACHE.pop(key, None)

    _CACHE[key] = (weakref.ref(dataset, _cleanup), cache)
    return cache


def get_or_set(dataset: object, key: Any, builder: Callable[[], Any]) -> Any:
    """Return cached value for key or build and store it."""

    cache = get_cache(dataset)
    if key in cache:
        return cache[key]
    value = builder()
    cache[key] = value
    return value


def clear_cache(dataset: object | None = None) -> None:
    """Clear cached items for a dataset or all datasets."""

    if dataset is None:
        _CACHE.clear()
        return
    _CACHE.pop(id(dataset), None)


__all__ = ["get_cache", "get_or_set", "clear_cache"]
