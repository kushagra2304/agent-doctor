"""Small helpers shared across modules."""

from __future__ import annotations

import dataclasses
import math
import time
import uuid
import warnings
from datetime import date, datetime
from enum import Enum
from typing import Any

MAX_STR = 20_000
_warned: set[str] = set()


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def warn_once(key: str, message: str) -> None:
    """Warn about a recording problem once per process; never raise."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(f"agent-doctor: {message}", RuntimeWarning, stacklevel=3)


def to_jsonable(obj: Any, max_str: int = MAX_STR, _depth: int = 0) -> Any:
    """Convert any value into something json.dumps accepts, truncating long strings."""
    if _depth > 20:
        return repr(obj)[:200]
    d = _depth + 1
    if obj is None or isinstance(obj, bool | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, str):
        if len(obj) <= max_str:
            return obj
        return obj[:max_str] + f"...[truncated {len(obj) - max_str} chars]"
    if isinstance(obj, bytes | bytearray):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, Enum):
        return to_jsonable(obj.value, max_str, d)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_str, d) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v, max_str, d) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump(mode="json"), max_str, d)
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_jsonable(dataclasses.asdict(obj), max_str, d)
        except Exception:
            pass
    return repr(obj)[:max_str]


def pick(obj: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from dicts, objects and lists: pick(resp, "choices.0.message.content")."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, list | tuple):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
            continue
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                continue
            # HF output types are dict subclasses that also expose attributes
        nxt = getattr(cur, part, None)
        if nxt is None and isinstance(cur, dict):
            return default
        cur = nxt
    return default if cur is None else cur
