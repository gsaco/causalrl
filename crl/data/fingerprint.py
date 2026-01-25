"""Dataset fingerprint utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def fingerprint_dataset(dataset: Any, *, max_bytes: int = 1_000_000) -> str:
    """Return a stable fingerprint for a dataset.

    The fingerprint hashes shapes, dtypes, and a deterministic sample of values.
    """

    data = dataset.to_dict() if hasattr(dataset, "to_dict") else {}
    hasher = hashlib.sha256()
    hasher.update(type(dataset).__name__.encode("utf-8"))
    for key in sorted(data.keys()):
        hasher.update(str(key).encode("utf-8"))
        value = data[key]
        if isinstance(value, np.ndarray):
            hasher.update(_hash_array(value, max_bytes=max_bytes))
        elif value is None:
            hasher.update(b"None")
        elif isinstance(value, (int, float, str, bool)):
            hasher.update(str(value).encode("utf-8"))
        else:
            hasher.update(
                json.dumps(value, sort_keys=True, default=str).encode("utf-8")
            )
    return hasher.hexdigest()


def _hash_array(arr: np.ndarray, *, max_bytes: int) -> bytes:
    arr = np.asarray(arr)
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    if arr.size == 0:
        return hasher.digest()

    bytes_len = arr.nbytes
    if bytes_len <= max_bytes:
        hasher.update(arr.tobytes())
        return hasher.digest()

    stride = max(1, bytes_len // max_bytes)
    sample = arr.reshape(-1)[::stride]
    hasher.update(sample.tobytes())
    hasher.update(str(stride).encode("utf-8"))
    return hasher.digest()


__all__ = ["fingerprint_dataset"]
