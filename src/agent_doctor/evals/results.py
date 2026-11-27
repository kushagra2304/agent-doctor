"""Saved evaluation results, so past evals can be listed, shown and read over MCP."""

from __future__ import annotations

from pydantic import BaseModel

from agent_doctor._util import new_id
from agent_doctor.evals.runner import Comparison, EvalReport
from agent_doctor.store import TraceStore


class EvalRecord(BaseModel):
    eval_id: str
    suite: str
    created_ms: int
    headline: str
    reports: list[EvalReport]
    comparisons: list[Comparison]


def headline(reports: list[EvalReport], comparisons: list[Comparison]) -> str:
    """One line: each comparison's verdict, or the single variant's pass rate."""
    if comparisons:
        return "; ".join(f"{c.candidate} vs {c.baseline}: {c.verdict}"
                         + (f" ({len(c.regressed)} broken)" if c.regressed else "") for c in comparisons)
    return "; ".join(f"{r.name}: {r.pass_rate:.0%} pass" for r in reports)


def save_results(suite: str, reports: list[EvalReport], comparisons: list[Comparison],
                 store: TraceStore) -> EvalRecord:
    eval_id = new_id("eval")
    line = headline(reports, comparisons)
    store.save_eval(eval_id, suite, line, {
        "reports": [r.model_dump(mode="json") for r in reports],
        "comparisons": [c.model_dump(mode="json") for c in comparisons],
    })
    meta, _ = store.get_eval(eval_id)
    return EvalRecord(eval_id=eval_id, suite=suite, created_ms=meta.created_ms, headline=line,
                      reports=reports, comparisons=comparisons)


def load_results(eval_id: str, store: TraceStore) -> EvalRecord:
    """A saved evaluation by ID, unique prefix, or ``last``."""
    meta, data = store.get_eval(eval_id)
    return EvalRecord(eval_id=meta.eval_id, suite=meta.suite, created_ms=meta.created_ms,
                      headline=meta.headline, **data)
