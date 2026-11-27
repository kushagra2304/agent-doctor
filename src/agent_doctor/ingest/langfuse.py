"""Langfuse exports: observations from the public API or a trace export.

Accepts ``{"data": [observation, ...]}`` (``GET /api/public/observations``), a list of
observations, or traces with an ``observations`` list (``GET /api/public/traces/{id}``).
Observations are grouped into one run per ``traceId``. ``GENERATION`` observations become
LLM calls, ``TOOL`` tool calls, ``RETRIEVER`` retrievals, everything else agent steps.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent_doctor.ingest.common import (
    build_run,
    iso_ms,
    maybe_json,
    normalize_messages,
    output_text,
    tool_args,
)
from agent_doctor.schema import Run, SpanKind, SpanStatus


def _items(data: Any) -> tuple[list[dict], dict[str, dict]]:
    """(observations, traces by id)."""
    if isinstance(data, dict):
        data = data.get("data", [data])
    observations, traces = [], {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("observations"), list):
            traces[item["id"]] = item
            observations += [o for o in item["observations"] if isinstance(o, dict)]
        elif "traceId" in item:
            observations.append(item)
    return observations, traces


def is_langfuse(data: Any) -> bool:
    observations, _ = _items(data)
    return bool(observations) and all("type" in o and "startTime" in o for o in observations[:5])


def _usage(o: dict[str, Any]) -> tuple[int, int, int]:
    details = o.get("usageDetails") or {}
    usage = o.get("usage") or {}
    inp = details.get("input", usage.get("input", usage.get("promptTokens", 0))) or 0
    out = details.get("output", usage.get("output", usage.get("completionTokens", 0))) or 0
    cached = sum(v for k, v in details.items() if "cache_read" in k and isinstance(v, int | float))
    return int(inp) + int(cached if "input" in details else 0), int(out), int(cached)


def parse(data: Any) -> list[Run]:
    observations, traces = _items(data)
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for o in observations:
        by_trace[o["traceId"]].append(o)
    runs = []
    for trace_id, obs in by_trace.items():
        spans = []
        for o in obs:
            start = iso_ms(o.get("startTime"))
            if start is None:
                continue
            kind = str(o.get("type", "SPAN")).upper()
            failed = str(o.get("level", "")).upper() == "ERROR"
            base = {"span_id": o["id"], "parent_id": o.get("parentObservationId"), "start_ms": start,
                    "end_ms": iso_ms(o.get("endTime")) or start,
                    "status": SpanStatus.ERROR if failed else SpanStatus.OK,
                    "error_message": o.get("statusMessage") if failed else None,
                    "name": str(o.get("name") or kind.lower())}
            if kind == "GENERATION":
                inp, out, cached = _usage(o)
                model = o.get("model") or base["name"]
                spans.append({**base, "kind": SpanKind.LLM, "name": str(model), "model": str(model),
                              "input_tokens": inp, "output_tokens": out, "cached_tokens": cached,
                              "input_messages": normalize_messages(o.get("input")),
                              "output_text": output_text(o.get("output"))})
            elif kind == "TOOL":
                spans.append({**base, "kind": SpanKind.TOOL, "tool_args": tool_args(o.get("input")),
                              "tool_output": maybe_json(o.get("output"))})
            else:
                step = SpanKind.RETRIEVAL if kind == "RETRIEVER" else SpanKind.AGENT_STEP
                spans.append({**base, "kind": step})
        trace = traces.get(trace_id, {})
        task = trace.get("input")
        runs.append(build_run(
            trace_id, str(trace.get("name") or "langfuse-agent"), spans,
            task_input=output_text(task) if task is not None else None,
            final_output=output_text(trace.get("output")) if trace.get("output") is not None else None,
            tags={"source": "langfuse",
                  **({"session_id": trace["sessionId"]} if trace.get("sessionId") else {})},
        ))
    return runs
