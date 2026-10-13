"""Slow steps and sequential calls that could run in parallel."""

from __future__ import annotations

from agent_doctor.analyzers.base import Context, canonical, leaves
from agent_doctor.schema import Finding, FindingType, Severity, Span, SpanKind


def slow_steps(ctx: Context) -> list[Finding]:
    total = ctx.run.duration_ms
    if total <= 0:
        return []
    findings = []
    for s in ctx.tools:
        share = s.duration_ms / total
        if s.duration_ms < ctx.t.slow_step_min_ms or share < ctx.t.slow_step_share:
            continue
        findings.append(ctx.finding(
            type=FindingType.LATENCY,
            severity=Severity.HIGH if share >= 0.75 else Severity.MEDIUM,
            title=f"{s.name} took {s.duration_ms / 1000:.1f} s, {share:.0%} of the run",
            detail=f"Step {ctx.steps([s])} alone accounts for most of the run's {total / 1000:.1f} s.",
            suggestion=(f"Cache {s.name}'s results, add a timeout, or call it in the background "
                        "while the agent does other work."),
            evidence=[s],
            data={"tool": s.name, "share": round(share, 3), "kind": "slow_tool"},
        ))
    return findings


def _salient(output) -> set[str]:
    out = set()
    for _, v in leaves(output):
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float) and abs(v) >= 100:
            out.add(str(v))
        elif isinstance(v, str) and len(v.strip()) >= 3:
            out.add(v.strip().lower())
    return out


def _depends(later: Span, earlier: list[Span]) -> bool:
    args = canonical(later.tool_args).lower()
    return any(val in args for s in earlier for val in _salient(s.tool_output))


def parallelizable_calls(ctx: Context) -> list[Finding]:
    """Back-to-back tool calls (no LLM in between) whose inputs don't use each other's outputs."""
    groups: list[list[Span]] = []
    current: list[Span] = []
    seen_keys: set = set()
    for s in ctx.spans:
        if s.kind is SpanKind.AGENT_STEP and s.parent_id is None:
            continue
        usable = (s.kind is SpanKind.TOOL and not s.is_error and s.retry_of is None
                  and not s.side_effects and (s.name, s.args_hash) not in seen_keys)
        if usable and current and s.parent_id == current[-1].parent_id \
                and s.start_ms >= current[-1].end_ms and not _depends(s, current):
            current.append(s)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [s] if usable else []
        if s.kind is SpanKind.TOOL:
            seen_keys.add((s.name, s.args_hash))
    if len(current) >= 2:
        groups.append(current)

    findings = []
    for g in groups:
        sequential = sum(s.duration_ms for s in g)
        saving = sequential - max(s.duration_ms for s in g)
        if saving < ctx.t.parallel_min_saving_ms:
            continue
        names = [s.name for s in g]
        findings.append(ctx.finding(
            type=FindingType.PARALLELIZABLE_CALLS,
            severity=Severity.MEDIUM if saving >= 1000 else Severity.LOW,
            title=f"{len(g)} independent calls ran one after another ({', '.join(names)})",
            detail=(f"Steps {ctx.steps(g)} don't use each other's results, so they could run "
                    f"together: {sequential} ms in sequence vs about "
                    f"{sequential - saving} ms in parallel."),
            suggestion="Run independent tool calls concurrently (asyncio.gather or a thread pool).",
            evidence=g,
            wasted_ms=saving,
            data={"tools": names, "kind": "sequential_independent_calls"},
        ))
    return findings
