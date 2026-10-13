"""Shared pieces for the detectors: thresholds, the analysis context and text helpers.

Detectors are plain functions ``(ctx) -> list[Finding]``. They only read the run (what
a real agent records); they never call an LLM, so analysis is free and repeatable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from agent_doctor.pricing import CostReport, ModelPrice, PriceTable, estimate_cost
from agent_doctor.schema import Finding, Run, Span, SpanKind


@dataclass(frozen=True)
class Thresholds:
    """Every tunable number in one place. Defaults favour precision over recall."""

    retry_min_attempts: int = 3          # a failed call retried twice or more
    no_backoff_gap_ms: int = 250         # retries closer together than this have no backoff
    loop_min_repeats: int = 3            # same tool, same result, this many times
    near_duplicate_min_similarity: float = 0.3  # word overlap of reworded arguments
    slow_step_share: float = 0.5         # one step taking this share of the run...
    slow_step_min_ms: int = 1000         # ...and at least this long
    parallel_min_saving_ms: int = 300    # only suggest parallelism that saves this much
    bloat_min_calls: int = 3
    bloat_growth_ratio: float = 2.0      # last prompt this many times the first
    bloat_min_growth_tokens: int = 1000
    tiny_output_tokens: int = 5          # a "one-word" answer
    expensive_input_price: float = 1.0   # USD per 1M input tokens
    cache_min_shared_chars: int = 4000   # ~1,000 tokens: what providers typically cache
    cache_varying_head_chars: int = 200  # a dynamic header shorter than this breaks caching


DEFAULT_THRESHOLDS = Thresholds()

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "what", "which", "who", "how", "for", "of", "to",
    "in", "on", "and", "or", "my", "your", "me", "i", "do", "does", "can", "about", "with",
}
_COMMON_VALUES = {"true", "false", "none", "null", "ok", "yes", "no", "n/a"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def leaves(value: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    """Every scalar inside nested dicts and lists, with the key it sits under."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, str(k))
    elif isinstance(value, list | tuple):
        for v in value:
            yield from leaves(v, key)
    elif value is not None:
        yield key, value


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def strip_echo(output: Any, args: dict[str, Any] | None) -> Any:
    """Drop top-level output fields that just repeat an argument (e.g. "query")."""
    if not isinstance(output, dict) or not args:
        return output
    echoed = {canonical(v) for v in args.values()}
    return {k: v for k, v in output.items() if canonical(v) not in echoed}


def prompt_text(span: Span) -> str:
    parts: list[str] = []
    for m in span.input_messages or []:
        content = m.get("content") if isinstance(m, dict) else m
        if isinstance(content, list):
            parts += [str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content]
        elif content is not None:
            parts.append(str(content))
    return "\n".join(parts)


def first_message(span: Span) -> str:
    msgs = span.input_messages or []
    if not msgs:
        return ""
    content = msgs[0].get("content") if isinstance(msgs[0], dict) else msgs[0]
    return content if isinstance(content, str) else canonical(content)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def value_appears(key: str | None, value: Any, haystack: str, haystack_lower: str) -> bool:
    """Does this output value show up in later prompt text?"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        text = str(value)
        if abs(value) >= 100:
            return re.search(rf"(?<![\w.]){re.escape(text)}(?![\w])", haystack) is not None
        if not key:
            return False
        # small numbers only count next to their key: in_stock=3, "in_stock": 3
        return re.search(rf"{re.escape(key)}\W{{0,4}}{re.escape(text)}(?![\w.])", haystack) is not None
    text = str(value).strip()
    if len(text) < 4 or text.lower() in _COMMON_VALUES:
        return False
    return text.lower() in haystack_lower


def has_checkable_values(output: Any) -> bool:
    for key, v in leaves(output):
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float) and (abs(v) >= 100 or key):
            return True
        if isinstance(v, str) and len(v.strip()) >= 4 and v.strip().lower() not in _COMMON_VALUES:
            return True
    return False


class Context:
    """One run plus everything detectors share: prices, costs, step numbers, loop spans."""

    def __init__(self, run: Run, prices: PriceTable, thresholds: Thresholds = DEFAULT_THRESHOLDS):
        self.run = run
        self.prices = prices
        self.t = thresholds

    @cached_property
    def spans(self) -> list[Span]:
        return self.run.ordered_spans

    @cached_property
    def llm(self) -> list[Span]:
        return [s for s in self.spans if s.kind is SpanKind.LLM]

    @cached_property
    def ok_llm(self) -> list[Span]:
        return [s for s in self.llm if not s.is_error]

    @cached_property
    def tools(self) -> list[Span]:
        return [s for s in self.spans if s.kind is SpanKind.TOOL]

    @cached_property
    def cost(self) -> CostReport:
        return estimate_cost(self.run, self.prices)

    @cached_property
    def span_cost(self) -> dict[str, float | None]:
        return {s.span_id: s.cost_usd for s in self.cost.spans}

    @cached_property
    def step(self) -> dict[str, int]:
        return {s.span_id: i for i, s in enumerate(self.spans, start=1)}

    def price(self, span: Span) -> ModelPrice | None:
        return self.prices.find(span.model or span.name, span.provider)

    def steps(self, spans: list[Span]) -> str:
        return ", ".join(f"#{self.step[s.span_id]}" for s in spans)

    @cached_property
    def loop_groups(self) -> list[list[Span]]:
        """Groups of >= N successful calls to one tool that all returned the same result."""
        groups: dict[tuple[str, str], list[Span]] = {}
        for s in self.tools:
            if s.is_error or s.tool_output is None:
                continue
            key = (s.name, canonical(strip_echo(s.tool_output, s.tool_args)))
            groups.setdefault(key, []).append(s)
        return [g for g in groups.values() if len(g) >= self.t.loop_min_repeats]

    @cached_property
    def loop_span_ids(self) -> set[str]:
        return {s.span_id for g in self.loop_groups for s in g}

    def finding(self, **fields: Any) -> Finding:
        spans = fields.pop("evidence")
        fields["evidence_span_ids"] = [s.span_id for s in sorted(spans, key=lambda x: x.start_ms)]
        return Finding(run_id=self.run.run_id, **fields)


Detector = Callable[[Context], list[Finding]]
