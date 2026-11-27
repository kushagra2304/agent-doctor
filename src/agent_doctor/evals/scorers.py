"""Scorers: grade one attempt at one case.

A scorer is any callable ``(case, attempt) -> Score | None``; ``None`` means it does not
apply (for example ``exact_match`` on a case with no expected answer). Write your own with
the ``scorer`` decorator:

>>> @scorer(threshold=0.5)
... def polite(case, attempt):
...     return "please" in (attempt.output_text or "").lower()
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_doctor.analyzers import SEVERITY_RANK
from agent_doctor.evals.cases import EvalCase
from agent_doctor.schema import Finding, Run, Severity


class Score(BaseModel):
    name: str
    value: float
    passed: bool
    reason: str = ""


@dataclass
class Attempt:
    """What the agent did on one case, as scorers see it."""

    output: Any
    output_text: str | None
    run: Run | None
    error: str | None = None
    cost_usd: float = 0.0
    cost_complete: bool = True
    duration_ms: int = 0
    findings: list[Finding] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)


Scorer = Callable[[EvalCase, Attempt], "Score | None"]


def scorer_name(s: Scorer) -> str:
    return getattr(s, "name", None) or getattr(s, "__name__", type(s).__name__)


def scorer(fn: Callable | None = None, *, name: str | None = None, threshold: float = 1.0) -> Any:
    """Turn ``fn(case, attempt)`` returning bool, a number from 0 to 1, a Score or None into a
    scorer. Numbers pass at ``threshold`` or above."""

    def decorate(f: Callable) -> Scorer:
        label = name or f.__name__

        def run(case: EvalCase, attempt: Attempt) -> Score | None:
            out = f(case, attempt)
            if out is None or isinstance(out, Score):
                return out
            value = float(out)
            return Score(name=label, value=value, passed=value >= threshold)

        run.name = label  # type: ignore[attr-defined]
        run.__name__ = label
        return run

    return decorate(fn) if fn is not None else decorate


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def exact_match(normalize: bool = True) -> Scorer:
    """Output equals the expected answer (ignoring case, punctuation and spacing by default)."""

    @scorer(name="exact_match")
    def run(case: EvalCase, attempt: Attempt) -> Score | None:
        if case.expected is None:
            return None
        out = attempt.output_text or ""
        same = _norm(out) == _norm(case.expected) if normalize else out == case.expected
        return Score(name="exact_match", value=float(same), passed=same,
                     reason="" if same else f"expected {case.expected[:120]!r}")

    return run


def contains(*phrases: str, require_all: bool = True, case_sensitive: bool = False) -> Scorer:
    """Output contains the phrases (all of them, or any with ``require_all=False``)."""
    if not phrases:
        raise ValueError("contains() needs at least one phrase")

    @scorer(name="contains")
    def run(case: EvalCase, attempt: Attempt) -> Score:
        out = attempt.output_text or ""
        hay = out if case_sensitive else out.lower()
        found = [p for p in phrases if (p if case_sensitive else p.lower()) in hay]
        value = len(found) / len(phrases)
        ok = value == 1.0 if require_all else bool(found)
        missing = [p for p in phrases if p not in found]
        return Score(name="contains", value=value, passed=ok, reason=f"missing {missing}" if missing else "")

    return run


def matches(pattern: str, flags: int = re.IGNORECASE) -> Scorer:
    """Output matches a regular expression."""
    rx = re.compile(pattern, flags)

    @scorer(name="matches")
    def run(case: EvalCase, attempt: Attempt) -> Score:
        ok = bool(rx.search(attempt.output_text or ""))
        return Score(name="matches", value=float(ok), passed=ok,
                     reason="" if ok else f"no match for {pattern!r}")

    return run


def similarity(threshold: float = 0.8) -> Scorer:
    """Word-overlap F1 between output and expected answer, from 0 to 1."""

    @scorer(name="similarity")
    def run(case: EvalCase, attempt: Attempt) -> Score | None:
        if case.expected is None:
            return None
        out, exp = _norm(attempt.output_text or "").split(), _norm(case.expected).split()
        common = sum(min(out.count(w), exp.count(w)) for w in set(out))
        if not out or not exp or not common:
            f1 = float(out == exp)
        else:
            precision, recall = common / len(out), common / len(exp)
            f1 = 2 * precision * recall / (precision + recall)
        return Score(name="similarity", value=round(f1, 4), passed=f1 >= threshold)

    return run


def completed() -> Scorer:
    """The agent finished without raising an exception."""

    @scorer(name="completed")
    def run(case: EvalCase, attempt: Attempt) -> Score:
        ok = attempt.error is None
        return Score(name="completed", value=float(ok), passed=ok, reason=attempt.error or "")

    return run


def no_problems(min_severity: Severity | str = Severity.HIGH) -> Scorer:
    """Agent Doctor found no problems of this severity or worse in the run."""
    floor = SEVERITY_RANK[Severity(min_severity)]

    @scorer(name="no_problems")
    def run(case: EvalCase, attempt: Attempt) -> Score | None:
        if attempt.run is None:
            return None
        bad = [f for f in attempt.findings if SEVERITY_RANK[f.severity] <= floor]
        return Score(name="no_problems", value=float(not bad), passed=not bad,
                     reason="; ".join(f.title for f in bad[:3]))

    return run


def max_cost(usd: float) -> Scorer:
    """The run cost at most ``usd``. Not applied when some models are unpriced."""

    @scorer(name="max_cost")
    def run(case: EvalCase, attempt: Attempt) -> Score | None:
        if attempt.run is None or not attempt.cost_complete:
            return None
        ok = attempt.cost_usd <= usd
        return Score(name="max_cost", value=float(ok), passed=ok,
                     reason="" if ok else f"cost ${attempt.cost_usd:.6f} > ${usd}")

    return run


def max_latency(ms: int) -> Scorer:
    """The run took at most ``ms`` milliseconds."""

    @scorer(name="max_latency")
    def run(case: EvalCase, attempt: Attempt) -> Score | None:
        if attempt.run is None:
            return None
        ok = attempt.duration_ms <= ms
        return Score(name="max_latency", value=float(ok), passed=ok,
                     reason="" if ok else f"{attempt.duration_ms} ms > {ms} ms")

    return run
