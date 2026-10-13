"""Context bloat: prompts that keep growing because history and raw tool output pile up."""

from __future__ import annotations

from agent_doctor.analyzers.base import Context
from agent_doctor.schema import Finding, FindingType, Severity


def context_bloat(ctx: Context) -> list[Finding]:
    calls = ctx.ok_llm
    if len(calls) < ctx.t.bloat_min_calls:
        return []
    tokens = [s.input_tokens for s in calls]
    first, last = tokens[0], tokens[-1]
    if first <= 0 or last < ctx.t.bloat_growth_ratio * first:
        return []
    if last - first < ctx.t.bloat_min_growth_tokens:
        return []
    if any(b < 0.9 * a for a, b in zip(tokens, tokens[1:], strict=False)):
        return []  # not a steady pile-up

    extra = [t - first for t in tokens[1:]]
    cost = 0.0
    priced = True
    for s, n in zip(calls[1:], extra, strict=False):
        price = ctx.price(s)
        if price is None:
            priced = False
            continue
        cost += n * price.rates(s.input_tokens)[0] / 1_000_000
    ratio = last / first
    return [ctx.finding(
        type=FindingType.CONTEXT_BLOAT,
        severity=Severity.HIGH if ratio >= 3 else Severity.MEDIUM,
        title=f"Prompt grew {ratio:.1f}x over {len(calls)} LLM calls ({first:,} to {last:,} tokens)",
        detail=("Input tokens per call: " + " -> ".join(f"{t:,}" for t in tokens)
                + f". About {sum(extra):,} tokens were re-sent history or raw tool output."),
        suggestion=("Send the model only what the next step needs: summarize or trim tool results, "
                    "drop old turns, and keep large raw data out of the prompt."),
        evidence=calls,
        wasted_tokens=sum(extra),
        wasted_cost_usd=cost if priced and cost else None,
        data={"tokens": tokens, "ratio": round(ratio, 2), "kind": "context_bloat"},
    )]
