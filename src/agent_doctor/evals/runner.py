"""Run an agent over test cases, score it, and compare two versions case by case."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_doctor.analyzers import DEFAULT_THRESHOLDS, Thresholds, analyze_run, problem_name
from agent_doctor.evals import stats
from agent_doctor.evals.cases import EvalCase
from agent_doctor.evals.scorers import Attempt, Score, Scorer, scorer_name
from agent_doctor.pricing import PriceTable
from agent_doctor.replay import replay
from agent_doctor.replay.engine import ToolMode
from agent_doctor.schema import Run
from agent_doctor.store import RunNotFound, TraceStore, get_default_store


@dataclass
class Variant:
    """One version of the agent to evaluate.

    ``agent=None`` means "what was recorded": each case's source run is scored as it is,
    with nothing re-run. Otherwise ``agent(case.input, **case.kwargs)`` is called, replaying
    the case's source run when it has one (``replay=True``).
    """

    agent: Callable[..., Any] | None = None
    model_map: dict[str, str] | None = None
    replay: bool = True
    tools: ToolMode = "recorded"
    options: dict[str, Any] = field(default_factory=dict)  # more replay() keyword arguments


RECORDED = Variant()


class CaseResult(BaseModel):
    case_id: str
    output_text: str | None
    run_id: str | None
    error: str | None
    cost_usd: float
    cost_complete: bool
    duration_ms: int
    llm_calls: int
    tool_calls: int
    scores: list[Score]
    passed: bool = Field(description="Every scorer that applies passed (or none apply and it completed).")
    problems: list[str] = Field(description="Agent Doctor findings in the run, by problem name.")
    replay: dict[str, int] = Field(default_factory=dict, description="How tool calls were answered.")

    def score(self, name: str) -> Score | None:
        return next((s for s in self.scores if s.name == name), None)


class ScorerSummary(BaseModel):
    name: str
    cases: int = Field(description="Cases the scorer applied to.")
    pass_rate: float
    mean: float


class EvalReport(BaseModel):
    name: str
    results: list[CaseResult]
    scorers: list[ScorerSummary]
    pass_rate: float
    errors: int
    total_cost_usd: float
    mean_cost_usd: float
    cost_complete: bool
    mean_ms: float
    p50_ms: float
    p95_ms: float
    problems: dict[str, int] = Field(description="Problem name -> cases with it.")

    def result(self, case_id: str) -> CaseResult | None:
        return next((r for r in self.results if r.case_id == case_id), None)


def _attempt(run: Run | None, output: Any, error: str | None, prices: PriceTable,
             thresholds: Thresholds, tool_calls: dict[str, int] | None = None) -> Attempt:
    if run is None:
        return Attempt(output=output, output_text=None if output is None else str(output), run=None,
                       error=error or "no run was recorded", tool_calls=tool_calls or {})
    report = analyze_run(run, prices, thresholds)
    text = output if isinstance(output, str) else (run.final_output if output is None else str(output))
    return Attempt(output=output, output_text=text, run=run, error=error, cost_usd=report.total_cost_usd,
                   cost_complete=report.fully_priced, duration_ms=run.duration_ms,
                   findings=report.findings, tool_calls=tool_calls or {})


def _score(case: EvalCase, attempt: Attempt, scorers: list[Scorer]) -> CaseResult:
    scores = []
    for s in scorers:
        try:
            out = s(case, attempt)
        except Exception as e:  # a broken scorer counts as a failure, not a crash
            out = Score(name=scorer_name(s), value=0.0, passed=False, reason=f"scorer failed: {e}")
        if out is not None:
            scores.append(out)
    passed = all(s.passed for s in scores) if scores else attempt.error is None
    run = attempt.run
    return CaseResult(
        case_id=case.id, output_text=attempt.output_text, run_id=run.run_id if run else None,
        error=attempt.error, cost_usd=attempt.cost_usd, cost_complete=attempt.cost_complete,
        duration_ms=attempt.duration_ms, llm_calls=len(run.llm_spans) if run else 0,
        tool_calls=len(run.tool_spans) if run else 0, scores=scores, passed=passed,
        problems=sorted({problem_name(f) for f in attempt.findings}), replay=attempt.tool_calls,
    )


def summarize(name: str, results: list[CaseResult]) -> EvalReport:
    names = list(dict.fromkeys(s.name for r in results for s in r.scores))
    summaries = []
    for n in names:
        scores = [s for r in results if (s := r.score(n)) is not None]
        summaries.append(ScorerSummary(name=n, cases=len(scores),
                                       pass_rate=stats.mean([float(s.passed) for s in scores]),
                                       mean=stats.mean([s.value for s in scores])))
    ran = [r for r in results if r.run_id]
    ms = [float(r.duration_ms) for r in ran]
    problems = Counter(p for r in results for p in r.problems)
    return EvalReport(
        name=name, results=results, scorers=summaries,
        pass_rate=stats.mean([float(r.passed) for r in results]),
        errors=sum(1 for r in results if r.error),
        total_cost_usd=sum(r.cost_usd for r in results),
        mean_cost_usd=stats.mean([r.cost_usd for r in ran]),
        cost_complete=all(r.cost_complete for r in ran),
        mean_ms=stats.mean(ms), p50_ms=stats.percentile(ms, 50), p95_ms=stats.percentile(ms, 95),
        problems=dict(problems.most_common()),
    )


def evaluate(
    agent: Callable[..., Any] | Variant | None,
    cases: Iterable[EvalCase],
    scorers: list[Scorer],
    *,
    name: str = "candidate",
    store: TraceStore | Mapping[str, Run] | None = None,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    save: bool = False,
) -> EvalReport:
    """Run one variant over every case and score it.

    ``store`` is where source runs are looked up: a TraceStore (default: your trace
    database) or a mapping of run ID to Run. ``save=True`` saves replayed runs to the store.
    """
    variant = agent if isinstance(agent, Variant) else Variant(agent=agent)
    prices = prices if prices is not None else PriceTable.load()
    source = store if store is not None else get_default_store()

    def source_run(case: EvalCase) -> Run | None:
        if not case.source_run_id:
            return None
        try:
            if isinstance(source, TraceStore):
                return source.get_run(case.source_run_id)
            return source[case.source_run_id]
        except (RunNotFound, KeyError):
            raise ValueError(f"case {case.id}: source run {case.source_run_id!r} not found") from None

    results = []
    for case in cases:
        src = source_run(case)
        if variant.agent is None:
            if src is None:
                raise ValueError(f"case {case.id} has no source run, so it cannot be scored as recorded")
            attempt = _attempt(src, src.final_output, None, prices, thresholds)
        else:
            res = replay(
                src if variant.replay else None, variant.agent, args=(case.input,), kwargs=case.kwargs,
                tools=variant.tools if src is not None and variant.replay else "live",
                model_map=variant.model_map, label=name,
                save=save and isinstance(source, TraceStore),
                store=source if isinstance(source, TraceStore) else None, **variant.options,
            )
            attempt = _attempt(res.run, res.output, res.error, prices, thresholds, res.tool_calls)
        results.append(_score(case, attempt, scorers))
    return summarize(name, results)


# -- comparing two reports ----------------------------------------------------------
class ScorerComparison(BaseModel):
    name: str
    cases: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    baseline_mean: float
    candidate_mean: float
    mean_diff: float
    mean_diff_ci95: tuple[float, float]
    p_value: float = Field(description="Exact McNemar test on pass/fail; small means a real difference.")
    improved: list[str] = Field(default_factory=list, description="Cases this scorer now passes.")
    regressed: list[str] = Field(default_factory=list, description="Cases this scorer now fails.")


class Comparison(BaseModel):
    baseline: str
    candidate: str
    cases: int = Field(description="Cases present in both reports.")
    verdict: Literal["better", "worse", "mixed", "same"]
    headline: str
    significant: bool = Field(description="The pass/fail difference is unlikely to be chance (p < 0.05).")
    p_value: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    improved: list[str] = Field(description="Cases that failed before and pass now.")
    regressed: list[str] = Field(description="Cases that passed before and fail now.")
    scorers: list[ScorerComparison]
    cost_change_pct: float | None
    latency_change_pct: float | None
    latency_diff_ms_ci95: tuple[float, float]
    problems_fixed: dict[str, int] = Field(description="Problem -> how many fewer cases have it.")
    problems_new: dict[str, int] = Field(description="Problem -> how many more cases have it.")


def _pct(a: float, b: float) -> float | None:
    return None if a == 0 else round((b - a) / a * 100, 1)


def compare_reports(baseline: EvalReport, candidate: EvalReport, alpha: float = 0.05) -> Comparison:
    """Compare two reports on the cases they share, pairing results by case ID."""
    pairs = [(a, b) for a in baseline.results if (b := candidate.result(a.case_id)) is not None]
    if not pairs:
        raise ValueError("the two reports have no cases in common")
    improved = [a.case_id for a, b in pairs if not a.passed and b.passed]
    regressed = [a.case_id for a, b in pairs if a.passed and not b.passed]
    p = stats.mcnemar_p(len(regressed), len(improved))

    scorer_names = list(dict.fromkeys(s.name for a, b in pairs for s in [*a.scores, *b.scores]))
    scorer_rows = []
    for n in scorer_names:
        both = [(sa, sb) for a, b in pairs if (sa := a.score(n)) and (sb := b.score(n))]
        if not both:
            continue
        diffs = [sb.value - sa.value for sa, sb in both]
        ids = [a.case_id for a, b in pairs if a.score(n) and b.score(n)]
        s_improved = [i for i, (sa, sb) in zip(ids, both, strict=True) if not sa.passed and sb.passed]
        s_regressed = [i for i, (sa, sb) in zip(ids, both, strict=True) if sa.passed and not sb.passed]
        scorer_rows.append(ScorerComparison(
            name=n, cases=len(both),
            baseline_pass_rate=stats.mean([float(sa.passed) for sa, _ in both]),
            candidate_pass_rate=stats.mean([float(sb.passed) for _, sb in both]),
            baseline_mean=stats.mean([sa.value for sa, _ in both]),
            candidate_mean=stats.mean([sb.value for _, sb in both]),
            mean_diff=stats.mean(diffs), mean_diff_ci95=stats.bootstrap_ci(diffs),
            p_value=stats.mcnemar_p(len(s_regressed), len(s_improved)),
            improved=s_improved, regressed=s_regressed,
        ))

    ran = [(a, b) for a, b in pairs if a.run_id and b.run_id]
    cost_a, cost_b = sum(a.cost_usd for a, _ in ran), sum(b.cost_usd for _, b in ran)
    ms_a, ms_b = sum(a.duration_ms for a, _ in ran), sum(b.duration_ms for _, b in ran)
    cost_ok = all(a.cost_complete and b.cost_complete for a, b in ran)
    cost_change = _pct(cost_a, cost_b) if ran and cost_ok else None
    latency_change = _pct(ms_a, ms_b) if ran else None

    prob_a = Counter(p for a, _ in pairs for p in a.problems)
    prob_b = Counter(p for _, b in pairs for p in b.problems)
    fixed = {k: prob_a[k] - prob_b[k] for k in prob_a if prob_a[k] > prob_b[k]}
    new = {k: prob_b[k] - prob_a[k] for k in prob_b if prob_b[k] > prob_a[k]}

    if improved and regressed:
        verdict = "mixed"
    elif improved:
        verdict = "better"
    elif regressed:
        verdict = "worse"
    else:
        verdict = "same"
    significant = (improved or regressed) and p < alpha
    parts = {
        "better": f"{len(improved)} case(s) fixed, none broken",
        "worse": f"{len(regressed)} case(s) broken, none fixed",
        "mixed": f"{len(improved)} case(s) fixed, {len(regressed)} broken",
        "same": "same overall pass/fail on every case",
    }
    headline = parts[verdict]
    if verdict != "same":
        chance = "significant" if significant else "could be chance"
        headline += f" (p={p:.3f}, {chance} with {len(pairs)} cases)"
    # A case fails overall if any scorer fails it, so one scorer that fails everywhere hides
    # what the others did. Say what each scorer gained or lost.
    for row in scorer_rows:
        if (row.improved or row.regressed) and (row.improved, row.regressed) != (improved, regressed):
            moves = [f"{len(row.improved)} fixed" if row.improved else "",
                     f"{len(row.regressed)} broken" if row.regressed else ""]
            headline += (f"; {row.name}: {', '.join(m for m in moves if m)} "
                         f"({row.baseline_pass_rate:.0%} -> {row.candidate_pass_rate:.0%})")
    for label, change in (("cost", cost_change), ("time", latency_change)):
        if change is not None and abs(change) >= 1:
            headline += f"; {label} {'+' if change > 0 else ''}{change:.0f}%"
    if fixed:
        headline += f"; {sum(fixed.values())} fewer problem(s)"
    if new:
        headline += f"; {sum(new.values())} new problem(s)"

    return Comparison(
        baseline=baseline.name, candidate=candidate.name, cases=len(pairs), verdict=verdict,
        headline=headline, significant=bool(significant), p_value=p,
        baseline_pass_rate=stats.mean([float(a.passed) for a, _ in pairs]),
        candidate_pass_rate=stats.mean([float(b.passed) for _, b in pairs]),
        improved=improved, regressed=regressed, scorers=scorer_rows,
        cost_change_pct=cost_change, latency_change_pct=latency_change,
        latency_diff_ms_ci95=stats.bootstrap_ci([float(b.duration_ms - a.duration_ms) for a, b in ran]),
        problems_fixed=fixed, problems_new=new,
    )


def compare(
    cases: Iterable[EvalCase],
    scorers: list[Scorer],
    baseline: Callable[..., Any] | Variant | None,
    candidate: Callable[..., Any] | Variant | None,
    *,
    names: tuple[str, str] = ("baseline", "candidate"),
    **options: Any,
) -> tuple[EvalReport, EvalReport, Comparison]:
    """Evaluate two variants on the same cases and compare them. ``None`` = the recordings."""
    cases = list(cases)
    a = evaluate(baseline, cases, scorers, name=names[0], **options)
    b = evaluate(candidate, cases, scorers, name=names[1], **options)
    return a, b, compare_reports(a, b)
