"""Small helpers shared across modules."""

from __future__ import annotations

import dataclasses
import math
import time
import uuid
import warnings
from collections.abc import Callable
from datetime import date, datetime
from enum import Enum
from typing import Any, Protocol

MAX_STR = 20_000
_warned: set[str] = set()


class Clock(Protocol):
    def time_ms(self) -> int: ...  # wall clock, Unix epoch milliseconds
    def perf(self) -> float: ...  # monotonic seconds, for durations
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def time_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def perf(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


_clock: Clock = SystemClock()
_id_factory: Callable[[str], str] | None = None


def get_clock() -> Clock:
    return _clock


def set_clock(clock: Clock) -> Clock:
    """Replace the clock used for all span timings; returns the previous one."""
    global _clock
    prev, _clock = _clock, clock
    return prev


def set_id_factory(factory: Callable[[str], str] | None) -> Callable[[str], str] | None:
    global _id_factory
    prev, _id_factory = _id_factory, factory
    return prev


def now_ms() -> int:
    return _clock.time_ms()


def new_id(prefix: str) -> str:
    if _id_factory is not None:
        return _id_factory(prefix)
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


def import_path(path: Any) -> Any:
    """Import a .py file. A file inside a package (folders with __init__.py) is imported by
    its dotted name, so its relative imports work; the folder above the package goes on
    sys.path, as pytest does."""
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    parts = [path.stem]
    root = path.parent
    while (root / "__init__.py").is_file():
        parts.insert(0, root.name)
        root = root.parent
    if len(parts) > 1:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return importlib.import_module(".".join(parts))
    name = f"_agent_doctor_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module
