"""Cost waste: big models on trivial steps, and prompts that miss the provider's cache."""

from __future__ import annotations

from agent_doctor.analyzers.base import (
    Context,
    common_prefix_len,
    common_suffix_len,
    first_message,
    prompt_text,
)
from agent_doctor.schema import Finding, FindingType, Severity


def expensive_model_for_simple_step(ctx: Context) -> list[Finding]:
    findings = []
    for s in ctx.ok_llm:
        if s.output_tokens > ctx.t.tiny_output_tokens or s.attributes.get("tool_calls"):
            continue
        price = ctx.price(s)
        if price is None or price.rates(s.input_tokens)[0] < ctx.t.expensive_input_price:
            continue
        cost = ctx.span_cost.get(s.span_id)
        findings.append(ctx.finding(
            type=FindingType.COST,
            severity=Severity.MEDIUM,
            title=f"{s.model} used for a {s.output_tokens}-token answer",
            detail=(f"Step {ctx.steps([s])} sends {s.input_tokens:,} tokens to {s.model} "
                    f"(${price.rates(s.input_tokens)[0]}/1M input) to get a {s.output_tokens}-token "
                    "reply, which looks like a classification or yes/no decision."
                    + (f" It cost ${cost:.6f}." if cost else "")),
            suggestion=("Route simple decisions (classification, yes/no, picking a tool) to a small, "
                        "cheap model and keep the large model for the final answer."),
            evidence=[s],
            data={"model": s.model, "output_tokens": s.output_tokens, "cost_usd": cost,
                  "kind": "expensive_model_for_simple_step"},
        ))
    return findings


def prompt_cache_misses(ctx: Context) -> list[Finding]:
    """Consecutive calls share a long block of prompt, but nothing was served from cache."""
    calls = [s for s in ctx.ok_llm if s.input_messages]
    flagged: dict[str, object] = {}
    reasons: set[str] = set()
    stable_tokens: dict[str, int] = {}
    for a, b in zip(calls, calls[1:], strict=False):
        if b.cached_tokens > 0:
            continue
        full_a, full_b = prompt_text(a), prompt_text(b)
        prefix = common_prefix_len(full_a, full_b)
        sys_a, sys_b = first_message(a), first_message(b)
        head = common_prefix_len(sys_a, sys_b)
        tail = common_suffix_len(sys_a, sys_b)
        if prefix >= ctx.t.cache_min_shared_chars:
            reasons.add("shared_prefix_not_cached")
            stable = prefix
        elif head < ctx.t.cache_varying_head_chars and tail >= ctx.t.cache_min_shared_chars:
            reasons.add("dynamic_prompt_head")
            stable = tail
        else:
            continue
        flagged[a.span_id] = a
        flagged[b.span_id] = b
        stable_tokens[b.span_id] = stable // 4
    if not flagged:
        return []

    saving, priced = 0.0, True
    for sid, n in stable_tokens.items():
        span = ctx.run.get_span(sid)
        price = ctx.price(span)
        if price is None:
            priced = False
            continue
        inp, cached, _ = price.rates(span.input_tokens)
        saving += n * (inp - cached) / 1_000_000
    spans = sorted(flagged.values(), key=lambda s: s.start_ms)  # type: ignore[attr-defined]
    dynamic = "dynamic_prompt_head" in reasons
    return [ctx.finding(
        type=FindingType.COST,
        severity=Severity.MEDIUM,
        title=("Prompt caching is broken by text that changes at the start of the prompt"
               if dynamic else "A long shared prompt prefix is never served from cache"),
        detail=(f"Steps {ctx.steps(spans)} share about {max(stable_tokens.values()):,} tokens of "
                "identical prompt, but 0 tokens were cached."
                + (" The first message differs only in its opening characters (a timestamp or ID?),"
                   " which stops prefix caching." if dynamic else "")),
        suggestion=("Put stable content (system prompt, policies, tool definitions) first and move "
                    "anything that changes per call (time, user, IDs) to the end of the prompt."
                    if dynamic else
                    "Enable prompt caching with your provider and keep the shared prefix identical."),
        evidence=spans,
        wasted_tokens=sum(stable_tokens.values()),
        wasted_cost_usd=saving if priced and saving else None,
        data={"reasons": sorted(reasons), "kind": "prompt_cache_miss"},
    )]
