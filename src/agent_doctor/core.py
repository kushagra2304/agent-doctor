"""Public API of the core engine.

Every interface (CLI, MCP server, Gradio demo) calls functions from here and holds no
logic of its own. Functions are added phase by phase:

- Phase 1: get_run, list_runs, estimate_cost
- Phase 3: analyze, analyze_runs
- Phase 5: run_eval_suite, compare_prompts, replay
"""

from __future__ import annotations

from agent_doctor.analyzers import DEFAULT_THRESHOLDS, AnalysisReport, Thresholds, analyze_run
from agent_doctor.pricing import CostReport, PriceTable
from agent_doctor.pricing import estimate_cost as _estimate_cost
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
