"""Retry storms, loops that make no progress, and failures the agent hid."""

from __future__ import annotations

import re

from agent_doctor.analyzers.base import Context
from agent_doctor.schema import Finding, FindingType, Severity, Span


def _gaps(chain: list[Span]) -> list[int]:
    return [b.start_ms - a.end_ms for a, b in zip(chain, chain[1:], strict=False)]


def _tool_chains(ctx: Context) -> list[list[Span]]:
    """Calls with the same tool and arguments that follow a failure of that call."""
    by_id = {s.span_id: s for s in ctx.tools}
    chains: list[list[Span]] = []
    open_chain: dict[tuple, list[Span]] = {}
    for s in ctx.tools:
        key = (s.name, s.args_hash)
        chain = open_chain.get(key)
        linked = s.retry_of is not None and s.retry_of in by_id
        if chain is not None and (linked or chain[-1].is_error):
            chain.append(s)
        else:
            chain = [s]
            chains.append(chain)
            open_chain[key] = chain
    return chains


def tool_retry_loops(ctx: Context) -> list[Finding]:
    findings = []
    for chain in _tool_chains(ctx):
        failures = [s for s in chain if s.is_error]
        if len(chain) < ctx.t.retry_min_attempts or len(failures) < 2:
            continue
        gaps = _gaps(chain)
        no_backoff = all(g < ctx.t.no_backoff_gap_ms for g in gaps)
        recovered = not chain[-1].is_error
        name = chain[0].name
        errors = sorted({s.error_type or "error" for s in failures})
        findings.append(ctx.finding(
            type=FindingType.RETRY_LOOP,
            severity=Severity.MEDIUM if recovered else Severity.HIGH,
            title=(f"{name} failed {len(failures)}x and was retried "
                   + ("with no backoff" if no_backoff else "repeatedly")
                   + ("" if recovered else ", never succeeding")),
            detail=(f"Steps {ctx.steps(chain)} call {name} with the same arguments "
                    f"({', '.join(errors)}). Gaps between attempts: "
                    + ", ".join(f"{g} ms" for g in gaps) + "."),
            suggestion=("Retry with exponential backoff and jitter (e.g. 0.5 s, 1 s, 2 s), cap the "
                        "attempts, and fall back gracefully when the service stays down."),
            evidence=chain,
            wasted_ms=sum(s.duration_ms for s in failures),
            data={"tool": name, "attempts": len(chain), "failures": len(failures),
                  "recovered": recovered, "no_backoff": no_backoff, "kind": "tool_retry_loop"},
        ))
    return findings


def llm_retry_loops(ctx: Context) -> list[Finding]:
    findings, chain = [], []
    for s in ctx.llm + [None]:  # sentinel flushes the last chain
        if s is not None and chain and chain[-1].is_error and s.model == chain[-1].model:
            chain.append(s)
            continue
        failures = [x for x in chain if x.is_error]
        if len(failures) >= 2:
            gaps = _gaps(chain)
            no_backoff = all(g < ctx.t.no_backoff_gap_ms for g in gaps)
            errors = sorted({x.error_type or "error" for x in failures})
            findings.append(ctx.finding(
                type=FindingType.RETRY_LOOP,
                severity=Severity.MEDIUM,
                title=(f"{len(failures)} failed LLM calls retried "
                       + ("with no backoff" if no_backoff else "repeatedly")),
                detail=(f"Steps {ctx.steps(chain)} call {chain[0].model}; "
                        f"{len(failures)} failed with {', '.join(errors)}."),
                suggestion=("Back off before retrying rate-limited calls (honour Retry-After), and "
                            "consider a fallback model or provider."),
                evidence=chain,
                wasted_ms=sum(x.duration_ms for x in failures),
                data={"model": chain[0].model, "failures": len(failures), "no_backoff": no_backoff,
                      "kind": "llm_retry_without_backoff"},
            ))
        chain = [s] if s is not None and s.is_error else []
    return findings


def no_progress_loops(ctx: Context) -> list[Finding]:
    if not ctx.loop_groups:
        return []
    tool_spans = sorted({s.span_id: s for g in ctx.loop_groups for s in g}.values(),
                        key=lambda s: s.start_ms)
    start, end = tool_spans[0].start_ms, tool_spans[-1].end_ms
    llm_between = [s for s in ctx.ok_llm if start <= s.start_ms and s.end_ms <= end]
    evidence = sorted(tool_spans + llm_between, key=lambda s: s.start_ms)
    repeats = sum(len(g) - 1 for g in ctx.loop_groups)
    names = sorted({g[0].name for g in ctx.loop_groups})
    tokens = sum(s.input_tokens + s.output_tokens for s in llm_between)
    cost = sum(ctx.span_cost.get(s.span_id) or 0 for s in llm_between)
    return [ctx.finding(
        type=FindingType.RETRY_LOOP,
        severity=Severity.HIGH,
        title=f"Loop with no progress: {', '.join(names)} returned the same result {repeats + len(names)}x",
        detail=(f"Between steps {ctx.steps([evidence[0]])} and {ctx.steps([evidence[-1]])} the agent "
                f"kept calling {', '.join(names)} and got nothing new, spending "
                f"{len(llm_between)} LLM call(s) deciding to continue."),
        suggestion=("Stop when a step returns nothing new: track what has been seen, cap iterations, "
                    "and check the prompt for instructions like 'always search again'."),
        evidence=evidence,
        wasted_ms=end - start,
        wasted_tokens=tokens or None,
        wasted_cost_usd=cost or None,
        data={"tools": names, "repeats": repeats, "kind": "no_progress_loop"},
    )]


_ADMITS_FAILURE = re.compile(
    r"unavailable|not available|unable to|couldn'?t|could not|can'?t (?:get|access|check|find|retrieve)|"
    r"cannot (?:get|access|check|find|retrieve)|temporarily|try again|having (?:trouble|issues)|"
    r"(?:is|was|are) down\b|not working|an? (?:error|issue|problem) (?:with|on|in)|"
    r"timed? ?out|time-out|(?:still|currently) pending|no (?:information|update)s? (?:yet|available)",
    re.I)


def _args_label(span: Span) -> str:
    args = span.tool_args or {}
    return ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])


def swallowed_failures(ctx: Context) -> list[Finding]:
    """A tool never succeeded, yet the run finished 'ok' with an answer that does not say so."""
    if ctx.run.status.value != "ok":
        return []
    findings = []
    for chain in _tool_chains(ctx):
        if not all(s.is_error for s in chain):
            continue
        last = chain[-1]
        name = last.name
        recovered = any(s.name == name and not s.is_error and s.start_ms > last.end_ms
                        for s in ctx.tools)
        answer = [s for s in ctx.ok_llm if s.start_ms >= last.end_ms]
        if recovered or not answer:
            continue
        final = ctx.run.final_output or answer[-1].output_text or ""
        if _ADMITS_FAILURE.search(final):
            continue  # the agent told the user the data was unavailable: honest, not swallowed
        # Name the arguments when the tool worked for others, so the title stays true.
        worked_elsewhere = any(s.name == name and not s.is_error for s in ctx.tools)
        label = f"{name}({_args_label(last)})" if worked_elsewhere and last.tool_args else name
        findings.append(ctx.finding(
            type=FindingType.ERROR,
            severity=Severity.HIGH,
            title=f"{label} never succeeded, but the agent answered anyway",
            detail=(f"All {len(chain)} call(s) to {label} failed (steps {ctx.steps(chain)}), and the "
                    f"run still ended 'ok' with an answer from step {ctx.steps([answer[-1]])} that "
                    f"does not mention the failure. The answer may be a guess."),
            suggestion=(f"When {name} fails, tell the model explicitly (or the user) instead of "
                        "letting it answer without the data; mark the run as degraded."),
            evidence=[*chain, answer[-1]],
            data={"tool": name, "failures": len(chain), "kind": "swallowed_tool_failure"},
        ))
    return findings

