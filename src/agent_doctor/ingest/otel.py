"""OpenTelemetry traces (OTLP JSON) using the GenAI semantic conventions.

Accepts an OTLP export (``{"resourceSpans": [...]}``), a list of them, or JSON lines (as
written by the collector's file exporter). One trace becomes one run. LLM spans are the ones
with ``gen_ai.request.model`` or a chat-like ``gen_ai.operation.name``; tool spans have
``gen_ai.operation.name = execute_tool`` or ``gen_ai.tool.name``. Older attribute names
(``gen_ai.usage.prompt_tokens``, ``gen_ai.prompt``, ``gen_ai.system``) are understood too.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent_doctor.ingest.common import build_run, maybe_json, normalize_messages, output_text, tool_args
from agent_doctor.schema import Run, SpanKind, SpanStatus

LLM_OPERATIONS = {"chat", "text_completion", "generate_content", "completion"}


def _value(v: dict[str, Any]) -> Any:
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in v:
            return v[key]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return _attrs(v["kvlistValue"].get("values", []))
    return None


def _attrs(items: list[dict[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(items, dict):  # already a plain mapping
        return items
    return {a["key"]: _value(a.get("value", {})) for a in items or []}


def _first(attrs: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if attrs.get(k) not in (None, ""):
            return attrs[k]
    return None


def _batches(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "resourceSpans" in data:
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict) and "resourceSpans" in d]
    return []


def is_otel(data: Any) -> bool:
    return bool(_batches(data))


def _event_attr(span: dict[str, Any], event: str, key: str) -> Any:
    for e in span.get("events", []) or []:
        if e.get("name") == event:
            return _attrs(e.get("attributes")).get(key)
    return None


def parse(data: Any) -> list[Run]:
    traces: dict[str, list[tuple[dict, dict, dict]]] = defaultdict(list)
    for batch in _batches(data):
        for rs in batch.get("resourceSpans", []):
            resource = _attrs((rs.get("resource") or {}).get("attributes"))
            for ss in rs.get("scopeSpans", []) or rs.get("instrumentationLibrarySpans", []):
                for span in ss.get("spans", []):
                    traces[span["traceId"]].append((span, _attrs(span.get("attributes")), resource))

    runs = []
    for trace_id, items in traces.items():
        spans, agent = [], None
        for span, a, _resource in items:
            op = str(a.get("gen_ai.operation.name") or "").lower()
            status = span.get("status") or {}
            failed = status.get("code") in (2, "2", "STATUS_CODE_ERROR", "ERROR")
            base = {
                "span_id": span["spanId"], "parent_id": span.get("parentSpanId") or None,
                "start_ms": int(int(span["startTimeUnixNano"]) / 1e6),
                "end_ms": int(int(span.get("endTimeUnixNano") or span["startTimeUnixNano"]) / 1e6),
                "status": SpanStatus.ERROR if failed else SpanStatus.OK,
                "error_type": _first(a, "error.type", "exception.type") if failed else None,
                "error_message": ((status.get("message") or _first(a, "exception.message"))
                                  if failed else None),
            }
            model = _first(a, "gen_ai.response.model", "gen_ai.request.model")
            if op == "execute_tool" or "gen_ai.tool.name" in a:
                name = _first(a, "gen_ai.tool.name") or span.get("name", "tool")
                spans.append({**base, "kind": SpanKind.TOOL, "name": str(name),
                              "tool_args": tool_args(_first(a, "gen_ai.tool.call.arguments",
                                                            "input.value")),
                              "tool_output": maybe_json(_first(a, "gen_ai.tool.call.result",
                                                               "output.value"))})
            elif model or op in LLM_OPERATIONS:
                messages = (_first(a, "gen_ai.input.messages", "gen_ai.prompt")
                            or _event_attr(span, "gen_ai.content.prompt", "gen_ai.prompt"))
                output = (_first(a, "gen_ai.output.messages", "gen_ai.completion")
                          or _event_attr(span, "gen_ai.content.completion", "gen_ai.completion"))
                spans.append({**base, "kind": SpanKind.LLM, "name": str(model or span.get("name", "llm")),
                              "model": str(model) if model else None,
                              "provider": _first(a, "gen_ai.provider.name", "gen_ai.system"),
                              "input_tokens": int(_first(a, "gen_ai.usage.input_tokens",
                                                         "gen_ai.usage.prompt_tokens") or 0),
                              "output_tokens": int(_first(a, "gen_ai.usage.output_tokens",
                                                          "gen_ai.usage.completion_tokens") or 0),
                              "cached_tokens": int(_first(a, "gen_ai.usage.cache_read.input_tokens",
                                                          "gen_ai.usage.cache_read_input_tokens") or 0),
                              "input_messages": normalize_messages(messages),
                              "output_text": output_text(output)})
            else:
                agent = agent or _first(a, "gen_ai.agent.name")
                step = SpanKind.RETRIEVAL if op == "retrieval" else SpanKind.AGENT_STEP
                spans.append({**base, "kind": step, "name": str(span.get("name") or "step")})
            agent = agent or _first(a, "gen_ai.agent.name")
        service = items[0][2].get("service.name")
        runs.append(build_run(trace_id, str(agent or service or "otel-agent"), spans,
                              tags={"source": "otel"}))
    return runs
