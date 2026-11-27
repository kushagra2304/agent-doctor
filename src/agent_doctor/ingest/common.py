"""Shared helpers for importers: timestamps, message normalization, building valid runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agent_doctor.schema import Run, Span, SpanKind, SpanStatus, compute_args_hash

_ROLES = {"human": "user", "humanmessage": "user", "user": "user", "ai": "assistant",
          "aimessage": "assistant", "assistant": "assistant", "system": "system",
          "systemmessage": "system", "tool": "tool", "toolmessage": "tool", "function": "tool"}


def iso_ms(value: Any) -> int | None:
    """Epoch milliseconds from an ISO 8601 string, epoch seconds/ms/ns, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        v = float(value)
        if v > 1e17:
            return int(v / 1e6)  # nanoseconds
        if v > 1e14:
            return int(v / 1e3)  # microseconds
        if v > 1e11:
            return int(v)  # milliseconds
        return int(v * 1000)  # seconds
    text = str(value).strip()
    if text.isdigit():
        return iso_ms(int(text))
    text = text.replace("Z", "+00:00")
    if "." in text:  # Python < 3.11 accepts at most 6 fractional digits
        head, _, rest = text.partition(".")
        frac = rest[:len(rest) - len(rest.lstrip("0123456789"))]
        tz = rest[len(frac):]
        text = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def normalize_messages(value: Any) -> list[dict[str, Any]] | None:
    """Chat messages in the common shapes (OpenAI, OTel parts, LangChain) as role/content."""
    value = maybe_json(value)
    if isinstance(value, dict):
        for key in ("messages", "input", "prompt"):
            if key in value:
                return normalize_messages(value[key])
        return None
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        return None
    if len(value) == 1 and isinstance(value[0], list):  # LangChain batches: [[...]]
        value = value[0]
    out = []
    for m in value:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            continue
        if m.get("lc") and isinstance(m.get("kwargs"), dict):  # serialized LangChain message
            kind = str((m.get("id") or ["user"])[-1]).lower()
            out.append({"role": _ROLES.get(kind, "user"),
                        "content": _content_text(m["kwargs"].get("content"))})
            continue
        role = _ROLES.get(str(m.get("role") or m.get("type") or "user").lower(), "user")
        content = m.get("content") if "content" in m else m.get("parts")
        out.append({"role": role, "content": _content_text(content)})
    return out or None


def output_text(value: Any) -> str | None:
    """The text of a model output in the common shapes."""
    value = maybe_json(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msgs = normalize_messages(value) if isinstance(value, list) else None
    if msgs:
        return msgs[-1]["content"] or None
    if isinstance(value, dict):
        for path in (("content",), ("text",), ("output",), ("completion",),
                     ("choices", 0, "message", "content"), ("generations", 0, 0, "text"),
                     ("generations", 0, 0, "message", "kwargs", "content")):
            cur: Any = value
            for key in path:
                if isinstance(key, int):
                    cur = cur[key] if isinstance(cur, list) and len(cur) > key else None
                else:
                    cur = cur.get(key) if isinstance(cur, dict) else None
            if cur:
                return _content_text(cur)
    return json.dumps(value, default=str)


def tool_args(value: Any) -> dict[str, Any] | None:
    value = maybe_json(value)
    if value is None:
        return None
    return value if isinstance(value, dict) else {"input": value}


def build_run(run_id: str, agent_name: str, spans: list[dict[str, Any]], *, task_input: str | None = None,
              final_output: str | None = None, tags: dict[str, Any] | None = None) -> Run:
    """Make a valid Run from span dicts: fix orphan parents and times, link tool retries,
    and fill the task input and final output from the LLM calls when not given."""
    ids = {s["span_id"] for s in spans}
    failed: dict[tuple[str, str | None], str] = {}
    built = []
    for s in sorted(spans, key=lambda x: x["start_ms"]):
        s = {k: v for k, v in s.items() if v is not None}
        s["run_id"] = run_id
        if s.get("parent_id") not in ids or s.get("parent_id") == s["span_id"]:
            s.pop("parent_id", None)
        s["end_ms"] = max(s["end_ms"], s["start_ms"])
        s["input_tokens"] = max(int(s.get("input_tokens", 0)), int(s.get("cached_tokens", 0)))
        if s["kind"] == SpanKind.TOOL:
            key = (s["name"], compute_args_hash(s.get("tool_args")))
            if key in failed:
                s["retry_of"] = failed.pop(key)
            if s.get("status") == SpanStatus.ERROR:
                failed[key] = s["span_id"]
        built.append(Span(**s))
    llm = [s for s in built if s.kind is SpanKind.LLM]
    if task_input is None:
        task_input = next((m["content"] for s in llm for m in (s.input_messages or [])
                           if m.get("role") == "user"), "")
    if final_output is None and llm:
        final_output = llm[-1].output_text
    roots = [s for s in built if s.parent_id is None]
    status = SpanStatus.ERROR if any(r.is_error for r in roots) else SpanStatus.OK
    return Run(run_id=run_id, agent_name=agent_name or "imported-agent", task_input=task_input or "",
               final_output=final_output, status=status, spans=built, tags=tags or {})
