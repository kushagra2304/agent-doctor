"""Evaluation suite for the bad agent: are the fixes better than what was recorded?

    agent-doctor eval examples/bad_agent/eval_suite.py

Three variants, on all 32 fixture runs:

- ``recorded``: the runs as they were recorded, planted problems and all (the baseline)
- ``fixed``: the agent with every agent-side fault turned off, replayed against the
  recorded tool results
- ``fixed-small``: the same, with ``model_map`` sending every large-model call to the
  small model, without changing the agent's code

Faults that live outside the agent stay as recorded: a tracking API that is down is still
down in the replay, and the provider still rate-limits. Time is simulated, so the suite
runs in a second and gives the same numbers on every machine.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from agent_doctor.evals import Suite, Variant, cases_from_runs, scorers
from agent_doctor.pricing import PriceTable
from agent_doctor.schema import Run
from agent_doctor.testing import deterministic_ids, simulated_time

from .agent import LARGE, SMALL, Faults, SupportAgent
from .fake_llm import FakeProvider
from .generate import DEFAULT_OUT, START_MS
from .scenarios import SCENARIOS

PRICES = Path(__file__).resolve().parents[1] / "demo_pricing.yaml"
BY_NAME = {s.name: s for s in SCENARIOS}


def load_fixtures(folder: Path = DEFAULT_OUT) -> tuple[dict[str, Run], dict]:
    runs = [Run.model_validate(r) for r in json.loads((folder / "runs.json").read_text(encoding="utf-8"))]
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))["runs"]
    return {r.run_id: r for r in runs}, labels


RUNS, LABELS = load_fixtures()


def _kwargs(run: Run) -> dict:
    sc = BY_NAME[LABELS[run.run_id]["scenario"]]
    return {"scenario": sc.name, "customer_id": sc.customer_id, "order_id": sc.order_id, "sku": sc.sku}


CASES = cases_from_runs(RUNS.values(), expected=lambda r: LABELS[r.run_id]["expected_output"],
                        kwargs=_kwargs)


def fixed_agent():
    """The support agent with its own faults fixed. One fake provider per variant, so the
    prompt cache warms up across cases as it would in production."""
    provider = FakeProvider()

    def handle(question: str, *, scenario: str, customer_id: str, order_id: str | None,
               sku: str | None) -> str:
        env = BY_NAME[scenario].faults
        faults = Faults(llm_rate_limit=env.llm_rate_limit)  # the provider's behaviour, not the agent's
        return SupportAgent(provider, faults).run(question, customer_id=customer_id, order_id=order_id,
                                                  sku=sku).answer

    return handle


@contextmanager
def repeatable():
    with simulated_time(START_MS + 86_400_000), deterministic_ids("eval"):
        yield


suite = Suite(
    name="support-bot",
    cases=CASES,
    scorers=[scorers.exact_match(), scorers.completed(), scorers.no_problems("high")],
    variants={
        "recorded": None,
        "fixed": fixed_agent(),
        "fixed-small": Variant(fixed_agent(), model_map={LARGE: SMALL}),
    },
    store=RUNS,
    prices=PriceTable.load(PRICES),
    context=repeatable,
)
