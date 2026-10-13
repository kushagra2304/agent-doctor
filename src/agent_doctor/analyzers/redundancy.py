"""Repeated tool calls: the exact same call twice, or the same question reworded."""

from __future__ import annotations

from agent_doctor.analyzers.base import (
    Context,
    canonical,
    has_checkable_values,
    jaccard,
    leaves,
    prompt_text,
    strip_echo,
    value_appears,
    words,
)
from agent_doctor.schema import Finding, FindingType, Severity


def _arg_words(args: dict | None) -> set[str]:
    """Words in argument values only; argument names are the same on every call."""
    return words(" ".join(str(v) for _, v in leaves(args or {})))


def _fmt_args(args: dict | None) -> str:
    if not args:
        return "()"
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def exact_duplicates(ctx: Context) -> list[Finding]:
    groups: dict[tuple[str, str | None], list] = {}
    for s in ctx.tools:
        if s.is_error or s.span_id in ctx.loop_span_ids:
            continue
        groups.setdefault((s.name, s.args_hash), []).append(s)
    findings = []
    for (name, _), spans in groups.items():
        if len(spans) < 2:
            continue
        repeats = spans[1:]
        side = any(s.side_effects for s in spans)
        costs = [ctx.span_cost.get(s.span_id) for s in repeats]
        findings.append(ctx.finding(
            type=FindingType.REDUNDANT_TOOL_CALL,
            severity=Severity.HIGH if side else Severity.MEDIUM,
            title=f"{name} called {len(spans)}x with the same arguments",
            detail=(f"Steps {ctx.steps(spans)} all call {name}({_fmt_args(spans[0].tool_args)}). "
                    f"The first result could have been reused."
                    + (" This tool has side effects, so the action happened more than once."
                       if side else "")),
            suggestion=("Cache tool results within a run (key: tool name + arguments), or keep the "
                        "earlier result in the agent's state instead of fetching it again."),
            evidence=spans,
            wasted_ms=sum(s.duration_ms for s in repeats),
            wasted_cost_usd=sum(c for c in costs if c) or None,
            data={"tool": name, "calls": len(spans), "kind": "exact_duplicate_tool_call"},
        ))
    return findings


def near_duplicates(ctx: Context) -> list[Finding]:
    """Same tool, reworded arguments, same result (ignoring fields that echo the input)."""
    by_result: dict[tuple[str, str], list] = {}
    for s in ctx.tools:
        if s.is_error or s.tool_output is None or s.span_id in ctx.loop_span_ids:
            continue
        key = (s.name, canonical(strip_echo(s.tool_output, s.tool_args)))
        by_result.setdefault(key, []).append(s)
    findings = []
    for (name, _), spans in by_result.items():
        distinct = {s.args_hash: s for s in spans}
        if len(distinct) < 2:
            continue
        first, *rest = sorted(distinct.values(), key=lambda s: s.start_ms)
        base = _arg_words(first.tool_args)
        similar = [s for s in rest
                   if jaccard(base, _arg_words(s.tool_args)) >= ctx.t.near_duplicate_min_similarity]
        if not similar:
            continue
        group = [first, *similar]
        findings.append(ctx.finding(
            type=FindingType.REDUNDANT_TOOL_CALL,
            severity=Severity.LOW,
            title=f"{name} called {len(group)}x with reworded arguments and got the same result",
            detail=(f"Steps {ctx.steps(group)}: "
                    + " vs ".join(_fmt_args(s.tool_args) for s in group)
                    + ". Every call returned the same data."),
            suggestion=("Tell the model what it already searched for (or dedupe similar queries "
                        "before calling the tool) so it does not ask the same thing twice."),
            evidence=group,
            wasted_ms=sum(s.duration_ms for s in similar),
            data={"tool": name, "calls": len(group), "kind": "near_duplicate_tool_call"},
        ))
    return findings


def unused_outputs(ctx: Context) -> list[Finding]:
    """A tool result whose values never appear in any later LLM prompt."""
    findings = []
    for s in ctx.tools:
        if s.is_error or s.side_effects or s.tool_output is None:
            continue
        if not has_checkable_values(s.tool_output):
            continue
        later = [x for x in ctx.ok_llm if x.start_ms >= s.end_ms]
        if not later or any(x.input_messages is None for x in later):
            continue  # nothing to check against, or prompts were not captured
        text = "\n".join(prompt_text(x) for x in later)
        lower = text.lower()
        if any(value_appears(k, v, text, lower) for k, v in leaves(s.tool_output)):
            continue
        findings.append(ctx.finding(
            type=FindingType.UNUSED_TOOL_OUTPUT,
            severity=Severity.LOW,
            title=f"{s.name}'s result was never used",
            detail=(f"Step {ctx.steps([s])} ran {s.name} ({s.duration_ms} ms), but none of its "
                    f"values appear in the {len(later)} LLM call(s) after it."),
            suggestion=f"Remove the {s.name} call, or pass its result to the model if it was meant to help.",
            evidence=[s],
            wasted_ms=s.duration_ms,
            data={"tool": s.name, "kind": "unused_tool_output"},
        ))
    return findings
