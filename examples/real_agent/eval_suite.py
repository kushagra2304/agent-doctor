"""Evaluation suite for the real agent: does the v2 prompt fix what v1 got wrong?

    python -m examples.real_agent.record            # once, to record the v1 runs
    agent-doctor --db examples/real_agent/traces.db eval examples/real_agent/eval_suite.py \\
        --prices examples/real_agent/local_pricing.yaml

Needs a local Ollama with qwen2.5:3b and qwen2.5:1.5b pulled. Variants:

- ``v1-recorded``: the recorded runs, scored as they are (the baseline)
- ``v1-rerun``: v1 run again, to show how much a model at temperature 0 still varies
- ``v2``: the v2 prompt
- ``v2-1.5b``: the v2 prompt on the smaller model, via model_map, no code change

The tools are small in-memory tables, so every variant runs them live; send_reply only
returns {"delivered": true}, so it is safe to run too.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("AD_BASE_URL", "http://localhost:11434/v1")  # the recordings are local

from agent_doctor.evals import EvalCase, Suite, Variant, scorers  # noqa: E402
from agent_doctor.pricing import PriceTable  # noqa: E402
from agent_doctor.store import TraceStore  # noqa: E402

from .agent import MODEL, handle, handle_v2  # noqa: E402
from .questions import QUESTIONS, missing_facts  # noqa: E402

HERE = Path(__file__).resolve().parent
STORE = TraceStore(os.environ.get("AGENT_DOCTOR_DB", HERE / "traces.db"))
FACTS = dict(QUESTIONS)
LIVE = {"tools": "live", "options": {"allow_side_effects": True}}


def load_cases() -> list[EvalCase]:
    """The latest recorded v1 run of each question (re-runs by the eval are not saved)."""
    latest: dict[str, str] = {}
    for row in sorted(STORE.list_runs(limit=500), key=lambda r: r.start_ms or 0):
        if row.task_input in FACTS and any(s.prompt_version == "v1" for s in STORE.get_run(row.run_id).spans):
            latest[row.task_input] = row.run_id
    if not latest:
        raise ValueError(f"no recorded v1 runs in {STORE.path}; run python -m examples.real_agent.record")
    return [EvalCase(id=f"q{i}", input=q, source_run_id=latest[q], tags={"facts": FACTS[q]})
            for i, (q, _) in enumerate(QUESTIONS, start=1) if q in latest]


def _tools(attempt) -> list[str]:
    run = attempt.run
    return [s.name for s in run.ordered_spans if s.kind.value == "tool"] if run else []


@scorers.scorer
def correct(case, attempt):
    """Every fact the question needs is in the answer."""
    miss = missing_facts(attempt.output_text or "", case.tags["facts"])
    return scorers.Score(name="correct", value=float(not miss), passed=not miss,
                         reason=f"missing {', '.join(miss)}" if miss else "")


@scorers.scorer
def looked_up(case, attempt):
    """Used at least one lookup tool rather than answering from memory."""
    return any(t != "send_reply" for t in _tools(attempt))


@scorers.scorer
def replied_once(case, attempt):
    """Called send_reply exactly once, as the prompt asks."""
    return _tools(attempt).count("send_reply") == 1


suite = Suite(
    name="real-agent",
    cases=load_cases,
    scorers=[correct, looked_up, replied_once, scorers.completed(), scorers.no_problems()],
    variants={
        "v1-recorded": None,
        "v1-rerun": Variant(handle, **LIVE),
        "v2": Variant(handle_v2, **LIVE),
        "v2-1.5b": Variant(handle_v2, model_map={MODEL: "qwen2.5:1.5b"}, **LIVE),
    },
    store=STORE,
    prices=PriceTable.load(HERE / "local_pricing.yaml"),
)
