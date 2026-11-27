"""Public API of the core engine.

Every interface (CLI, MCP server, Gradio demo) calls functions from here and holds no
logic of its own. Functions are added phase by phase:

- Phase 1: get_run, list_runs, estimate_cost
- Phase 3: analyze, analyze_runs
- Phase 4: scan, compare
- Phase 5: replay, evaluate, compare_variants
- Phase 6: recommend, recommend_for_runs, import_runs
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from agent_doctor.analyzers import (
    DEFAULT_THRESHOLDS,
    AnalysisReport,
    ScanReport,
    Thresholds,
    analyze_run,
    scan_reports,
)
from agent_doctor.analyzers.compare import RunComparison, compare_runs
from agent_doctor.evals import Comparison, EvalCase, EvalReport, Variant
from agent_doctor.evals import compare as _compare_variants
from agent_doctor.evals import evaluate as _evaluate
from agent_doctor.evals.scorers import Scorer
from agent_doctor.ingest import Format, Redactor, load_runs
from agent_doctor.pricing import CostReport, PriceTable
from agent_doctor.pricing import estimate_cost as _estimate_cost
from agent_doctor.recommender import FixPlan, KnowledgeBase, recommend_many
from agent_doctor.recommender import recommend as _recommend
from agent_doctor.recommender.recommend import LLM
from agent_doctor.replay import ReplayResult
from agent_doctor.replay import replay as _replay
from agent_doctor.schema import Run
from agent_doctor.store import StoredRun, TraceStore, get_default_store


def _store(store: TraceStore | None) -> TraceStore:
    return store if store is not None else get_default_store()


def get_run(run_id: str, store: TraceStore | None = None) -> Run:
    """Load a run by its ID or a unique prefix of it."""
    s = _store(store)
    return s.get_run(s.resolve_run_id(run_id))


def list_runs(
    agent: str | None = None, limit: int = 20, store: TraceStore | None = None
) -> list[StoredRun]:
    return _store(store).list_runs(agent=agent, limit=limit)


def estimate_cost(
    run: Run | str, store: TraceStore | None = None, prices: PriceTable | None = None
) -> CostReport:
    """Cost of a run, given as a Run or as a run ID."""
    if isinstance(run, str):
        run = get_run(run, store)
    return _estimate_cost(run, prices)


def analyze(
    run: Run | str,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> AnalysisReport:
    """Find wasted cost, time and reliability problems in one run."""
    if isinstance(run, str):
        run = get_run(run, store)
    return analyze_run(run, prices, thresholds)


def analyze_runs(
    agent: str | None = None,
    limit: int = 50,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[AnalysisReport]:
    """Analyze the most recent runs (newest first)."""
    s = _store(store)
    prices = prices if prices is not None else PriceTable.load()
    return [analyze_run(s.get_run(r.run_id), prices, thresholds)
            for r in s.list_runs(agent=agent, limit=limit)]


def scan(
    agent: str | None = None,
    limit: int = 50,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ScanReport:
    """Which problems are most common and most costly across the most recent runs."""
    return scan_reports(analyze_runs(agent, limit, store, prices, thresholds))


def compare(
    run_a: Run | str,
    run_b: Run | str,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> RunComparison:
    """What changed between two runs: cost, time, calls per tool and model, and problems."""
    if isinstance(run_a, str):
        run_a = get_run(run_a, store)
    if isinstance(run_b, str):
        run_b = get_run(run_b, store)
    return compare_runs(run_a, run_b, prices, thresholds)


def replay(
    run: Run | str, agent: Callable[..., Any], store: TraceStore | None = None, **options: Any
) -> ReplayResult:
    """Run ``agent`` again on a saved run's input, with tool calls answered from the recording.
    ``options`` are passed to ``agent_doctor.replay.replay``."""
    if isinstance(run, str):
        run = get_run(run, store)
    return _replay(run, agent, store=store, **options)


def evaluate(
    agent: Callable[..., Any] | Variant | None,
    cases: Iterable[EvalCase],
    scorers: list[Scorer],
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    **options: Any,
) -> EvalReport:
    """Score one version of the agent on test cases (``agent=None``: the recorded runs)."""
    return _evaluate(agent, cases, scorers, store=_store(store), prices=prices, **options)


def compare_variants(
    cases: Iterable[EvalCase],
    scorers: list[Scorer],
    baseline: Callable[..., Any] | Variant | None,
    candidate: Callable[..., Any] | Variant | None,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    **options: Any,
) -> tuple[EvalReport, EvalReport, Comparison]:
    """Evaluate two versions on the same cases and compare them case by case."""
    return _compare_variants(cases, scorers, baseline, candidate, store=_store(store), prices=prices,
                             **options)


def recommend(
    run: Run | str,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    kb: KnowledgeBase | None = None,
    llm: LLM | None = None,
) -> FixPlan:
    """A ranked fix plan for one run, from the playbooks; ``llm`` also writes it up as prose."""
    if isinstance(run, str):
        run = get_run(run, store)
    return _recommend(analyze_run(run, prices), run, kb=kb, llm=llm)


def recommend_for_runs(
    agent: str | None = None,
    limit: int = 50,
    store: TraceStore | None = None,
    prices: PriceTable | None = None,
    kb: KnowledgeBase | None = None,
    llm: LLM | None = None,
) -> FixPlan:
    """The fixes that remove the most waste across the most recent runs."""
    s = _store(store)
    prices = prices if prices is not None else PriceTable.load()
    runs = [s.get_run(r.run_id) for r in s.list_runs(agent=agent, limit=limit)]
    return recommend_many([(r, analyze_run(r, prices)) for r in runs], kb=kb, llm=llm)


def import_runs(
    path: str, fmt: Format = "auto", redact: bool | Redactor = False, store: TraceStore | None = None
) -> list[str]:
    """Load runs from a file (Agent Doctor JSON, OTLP JSON, Langfuse or LangSmith) and save them."""
    s = _store(store)
    runs = load_runs(path, fmt, redact)
    for run in runs:
        s.save_run(run)
    return [r.run_id for r in runs]
