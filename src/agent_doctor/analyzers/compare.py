"""Side-by-side comparison of two runs: what changed in cost, time, calls and problems."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from agent_doctor.analyzers.base import DEFAULT_THRESHOLDS, Thresholds
from agent_doctor.analyzers.report import AnalysisReport, analyze_run, problem_name
from agent_doctor.pricing import PriceTable, estimate_cost
from agent_doctor.schema import Run


class RunTotals(BaseModel):
    run_id: str
    agent_name: str
    status: str
    duration_ms: int
    llm_calls: int
    tool_calls: int
    errors: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    cost_complete: bool


class CountDiff(BaseModel):
    name: str
    a: int
    b: int


class ModelDiff(BaseModel):
    model: str
    a_calls: int
    b_calls: int
    a_cost_usd: float | None
    b_cost_usd: float | None


class RunComparison(BaseModel):
    a: RunTotals
    b: RunTotals
    delta: dict[str, float] = Field(description="b minus a, for every numeric total.")
    tool_calls: list[CountDiff] = Field(description="Tools called a different number of times.")
    models: list[ModelDiff] = Field(description="Models whose calls or cost differ.")
    problems_only_in_a: list[str]
    problems_only_in_b: list[str]
    problems_in_both: list[str]


def _totals(run: Run, report: AnalysisReport) -> RunTotals:
    return RunTotals(
        run_id=run.run_id, agent_name=run.agent_name, status=run.status.value,
        duration_ms=run.duration_ms, llm_calls=report.llm_calls, tool_calls=report.tool_calls,
        errors=run.error_count, input_tokens=run.total_input_tokens, output_tokens=run.total_output_tokens,
        cached_tokens=run.total_cached_tokens, cost_usd=report.total_cost_usd,
        cost_complete=report.fully_priced,
    )


def compare_runs(
    a: Run, b: Run, prices: PriceTable | None = None, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> RunComparison:
    prices = prices if prices is not None else PriceTable.load()
    rep_a, rep_b = analyze_run(a, prices, thresholds), analyze_run(b, prices, thresholds)
    ta, tb = _totals(a, rep_a), _totals(b, rep_b)
    numeric = [k for k, v in ta.model_dump().items()
               if isinstance(v, int | float) and not isinstance(v, bool)]
    delta = {k: getattr(tb, k) - getattr(ta, k) for k in numeric}

    tools_a = Counter(s.name for s in a.tool_spans)
    tools_b = Counter(s.name for s in b.tool_spans)
    tool_diffs = [CountDiff(name=t, a=tools_a[t], b=tools_b[t])
                  for t in sorted(tools_a.keys() | tools_b.keys()) if tools_a[t] != tools_b[t]]

    by_a = {m.model: m for m in estimate_cost(a, prices).by_model}
    by_b = {m.model: m for m in estimate_cost(b, prices).by_model}
    model_diffs = []
    for name in sorted(by_a.keys() | by_b.keys()):
        ma, mb = by_a.get(name), by_b.get(name)
        diff = ModelDiff(model=name, a_calls=ma.calls if ma else 0, b_calls=mb.calls if mb else 0,
                         a_cost_usd=ma.cost_usd if ma else 0.0, b_cost_usd=mb.cost_usd if mb else 0.0)
        if diff.a_calls != diff.b_calls or diff.a_cost_usd != diff.b_cost_usd:
            model_diffs.append(diff)

    pa = {problem_name(f) for f in rep_a.findings}
    pb = {problem_name(f) for f in rep_b.findings}
    return RunComparison(
        a=ta, b=tb, delta=delta, tool_calls=tool_diffs, models=model_diffs,
        problems_only_in_a=sorted(pa - pb), problems_only_in_b=sorted(pb - pa),
        problems_in_both=sorted(pa & pb),
    )
