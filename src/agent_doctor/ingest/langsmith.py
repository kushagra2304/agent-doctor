"""LangSmith runs, as exported with ``client.list_runs(...)`` and ``run.dict()`` / JSON.

Accepts a list of runs (or ``{"runs": [...]}``), flat or nested through ``child_runs``.
Runs are grouped into one Agent Doctor run per ``trace_id``. ``llm`` runs become LLM calls,
``tool`` runs tool calls, ``retriever`` runs retrievals, everything else (chains, prompts,
parsers) agent steps.
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


def _flatten(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for r in items:
        if isinstance(r, dict):
            out.append(r)
            out += _flatten(r.get("child_runs") or [])
    return out


def _runs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("runs", [data])
    return _flatten(data) if isinstance(data, list) else []


def is_langsmith(data: Any) -> bool:
    runs = _runs(data)
    return bool(runs) and all("run_type" in r for r in runs[:5])


def _model(r: dict[str, Any]) -> str | None:
    extra = r.get("extra") or {}
    meta = extra.get("metadata") or {}
    params = extra.get("invocation_params") or {}
    return meta.get("ls_model_name") or params.get("model") or params.get("model_name")


def _tokens(r: dict[str, Any]) -> tuple[int, int, int]:
    inp, out = r.get("prompt_tokens"), r.get("completion_tokens")
    cached = 0
    outputs = r.get("outputs") or {}
    usage = ((outputs.get("llm_output") or {}).get("token_usage")
             or (outputs.get("usage_metadata")) or {})
    details = (r.get("prompt_token_details") or usage.get("input_token_details") or {})
    cached = int(details.get("cache_read", 0) or 0)
    inp = inp if inp is not None else usage.get("prompt_tokens", usage.get("input_tokens", 0))
    out = out if out is not None else usage.get("completion_tokens", usage.get("output_tokens", 0))
    return int(inp or 0), int(out or 0), cached


def parse(data: Any) -> list[Run]:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for r in _runs(data):
        by_trace[str(r.get("trace_id") or r["id"])].append(r)
    runs = []
    for trace_id, items in by_trace.items():
        spans, root = [], None
        for r in items:
            start = iso_ms(r.get("start_time"))
            if start is None:
                continue
            kind = str(r.get("run_type", "chain")).lower()
            error = r.get("error")
            parent = str(r["parent_run_id"]) if r.get("parent_run_id") else None
            base = {"span_id": str(r["id"]), "parent_id": parent,
                    "start_ms": start, "end_ms": iso_ms(r.get("end_time")) or start,
                    "status": SpanStatus.ERROR if error else SpanStatus.OK,
                    "error_type": str(error).split(":", 1)[0][:100] if error else None,
                    "error_message": str(error) if error else None, "name": str(r.get("name") or kind)}
            if not r.get("parent_run_id"):
                root = r
            if kind == "llm":
                inp, out, cached = _tokens(r)
                model = _model(r) or base["name"]
                meta = (r.get("extra") or {}).get("metadata") or {}
                spans.append({**base, "kind": SpanKind.LLM, "name": model, "model": model,
                              "provider": meta.get("ls_provider"), "input_tokens": inp, "output_tokens": out,
                              "cached_tokens": cached, "input_messages": normalize_messages(r.get("inputs")),
                              "output_text": output_text(r.get("outputs"))})
            elif kind == "tool":
                outputs = r.get("outputs")
                if isinstance(outputs, dict) and set(outputs) == {"output"}:
                    outputs = outputs["output"]
                spans.append({**base, "kind": SpanKind.TOOL, "tool_args": tool_args(r.get("inputs")),
                              "tool_output": maybe_json(outputs)})
            else:
                step = SpanKind.RETRIEVAL if kind == "retriever" else SpanKind.AGENT_STEP
                spans.append({**base, "kind": step})
        root = root or {}
        task = root.get("inputs")
        runs.append(build_run(
            trace_id, str(root.get("name") or "langsmith-agent"), spans,
            task_input=output_text(task) if task else None,
            final_output=output_text(root.get("outputs")) if root.get("outputs") else None,
            tags={"source": "langsmith"},
        ))
    return runs
