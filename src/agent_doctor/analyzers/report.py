"""Run every detector over a run and rank what they find."""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, Field

from agent_doctor.analyzers import bloat, cost, latency, loops, redundancy
from agent_doctor.analyzers.base import DEFAULT_THRESHOLDS, Context, Detector, Thresholds
from agent_doctor.pricing import PriceTable
from agent_doctor.schema import Finding, Run, Severity

DETECTORS: list[Detector] = [
    loops.swallowed_failures,
    loops.tool_retry_loops,
    loops.llm_retry_loops,
    loops.no_progress_loops,
    redundancy.exact_duplicates,
    redundancy.near_duplicates,
    redundancy.unused_outputs,
    latency.slow_steps,
    latency.parallelizable_calls,
    bloat.context_bloat,
    cost.expensive_model_for_simple_step,
    cost.prompt_cache_misses,
]

_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}


class AnalysisReport(BaseModel):
    run_id: str
    agent_name: str
    status: str
    duration_ms: int
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    fully_priced: bool
    findings: list[Finding]
    wasted_cost_usd: float = Field(description="Sum of findings' estimates; they can overlap.")
    wasted_ms: int = Field(description="Sum of findings' estimates; they can overlap.")
    wasted_tokens: int
    step_numbers: dict[str, int] = Field(description="span_id -> step number, as `show` prints.")

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.severity.value] = out.get(f.severity.value, 0) + 1
        return out

    def summary(self, max_findings: int = 5) -> dict[str, Any]:
        """A compact view for MCP clients and dashboards."""
        return {
            "run_id": self.run_id,
            "agent": self.agent_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "cost_usd": round(self.total_cost_usd, 6),
            "cost_complete": self.fully_priced,
            "findings_total": len(self.findings),
            "findings_by_severity": self.counts,
            "estimated_waste": {"cost_usd": round(self.wasted_cost_usd, 6), "ms": self.wasted_ms,
                                "tokens": self.wasted_tokens},
            "top_findings": [
                {"severity": f.severity.value, "type": f.type.value, "title": f.title,
                 "suggestion": f.suggestion,
                 "steps": [self.step_numbers.get(s) for s in f.evidence_span_ids],
                 "wasted_cost_usd": f.wasted_cost_usd, "wasted_ms": f.wasted_ms}
                for f in self.findings[:max_findings]
            ],
        }


def analyze_run(
    run: Run,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    detectors: list[Detector] | None = None,
) -> AnalysisReport:
    ctx = Context(run, prices if prices is not None else PriceTable.load(), thresholds)
    findings: list[Finding] = []
    for detect in detectors or DETECTORS:
        try:
            findings += detect(ctx)
        except Exception as e:  # one broken detector must not hide the others' results
            warnings.warn(f"agent-doctor: detector {detect.__name__} failed: {e}",
                          RuntimeWarning, stacklevel=2)
    findings.sort(key=lambda f: (_RANK[f.severity], -(f.wasted_cost_usd or 0), -(f.wasted_ms or 0)))
    report = ctx.cost
    return AnalysisReport(
        run_id=run.run_id,
        agent_name=run.agent_name,
        status=run.status.value,
        duration_ms=run.duration_ms,
        llm_calls=len(ctx.llm),
        tool_calls=len(ctx.tools),
        input_tokens=run.total_input_tokens,
        output_tokens=run.total_output_tokens,
        total_cost_usd=report.total_cost_usd,
        fully_priced=report.fully_priced,
        findings=findings,
        wasted_cost_usd=sum(f.wasted_cost_usd or 0 for f in findings),
        wasted_ms=sum(f.wasted_ms or 0 for f in findings),
        wasted_tokens=sum(f.wasted_tokens or 0 for f in findings),
        step_numbers=ctx.step,
    )
