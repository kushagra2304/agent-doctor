"""Import traces from other tools, and redact personal data and secrets.

>>> from agent_doctor.ingest import load_runs
>>> runs = load_runs("export.json")                  # format detected automatically
>>> runs = load_runs("traces.jsonl", fmt="otel", redact=True)

Supported: Agent Doctor's own JSON, OpenTelemetry (OTLP JSON, GenAI conventions), Langfuse
and LangSmith exports. The importers are tolerant: fields they do not know are ignored,
and missing token counts become 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from agent_doctor.ingest import langfuse, langsmith, otel
from agent_doctor.ingest.redact import DEFAULT_PATTERNS, Redactor
from agent_doctor.schema import Run

Format = Literal["auto", "native", "otel", "langfuse", "langsmith"]
FORMATS = ("native", "otel", "langfuse", "langsmith")


def read_json(path: str | Path) -> Any:
    """A JSON file, or JSON lines (one value per line) returned as a list."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _is_native(data: Any) -> bool:
    items = data if isinstance(data, list) else [data]
    return bool(items) and all(isinstance(i, dict) and "spans" in i and "agent_name" in i for i in items)


def detect(data: Any) -> str:
    if _is_native(data):
        return "native"
    if otel.is_otel(data):
        return "otel"
    if langsmith.is_langsmith(data):
        return "langsmith"
    if langfuse.is_langfuse(data):
        return "langfuse"
    raise ValueError("unrecognized trace format; expected Agent Doctor JSON, OTLP JSON, a Langfuse "
                     "or a LangSmith export")


def parse_runs(data: Any, fmt: Format = "auto", redact: bool | Redactor = False) -> list[Run]:
    kind = detect(data) if fmt == "auto" else fmt
    if kind == "native":
        runs = [Run.model_validate(i) for i in (data if isinstance(data, list) else [data])]
    elif kind == "otel":
        runs = otel.parse(data)
    elif kind == "langfuse":
        runs = langfuse.parse(data)
    elif kind == "langsmith":
        runs = langsmith.parse(data)
    else:
        raise ValueError(f"unknown format {fmt!r}; use one of auto, {', '.join(FORMATS)}")
    if redact:
        redactor = redact if isinstance(redact, Redactor) else Redactor()
        runs = [redactor.run(r) for r in runs]
    return runs


def load_runs(path: str | Path, fmt: Format = "auto", redact: bool | Redactor = False) -> list[Run]:
    return parse_runs(read_json(path), fmt, redact)


__all__ = ["DEFAULT_PATTERNS", "FORMATS", "Redactor", "detect", "load_runs", "parse_runs", "read_json"]
