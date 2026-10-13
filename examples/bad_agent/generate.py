"""Generate the bad-agent fixtures: tests/fixtures/bad_agent/runs.json and labels.json.

From the repository root:

    python -m examples.bad_agent.generate            # write the fixture files
    python -m examples.bad_agent.generate --check    # fail if the files are out of date
    python -m examples.bad_agent.generate --save     # also save the runs to your trace DB

Time is simulated and IDs are seeded, so the output is identical on every machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_doctor.schema import Run
from agent_doctor.store import get_default_store
from agent_doctor.testing import deterministic_ids, simulated_time

from .agent import SupportAgent, expected_answer
from .fake_llm import FakeProvider
from .scenarios import SCENARIOS

SEED = 7
IST = timezone(timedelta(hours=5, minutes=30))
START_MS = int(datetime(2026, 10, 1, 9, 0, tzinfo=IST).timestamp() * 1000)
RUN_SPACING_MS = 7 * 60_000
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bad_agent"


def generate(seed: int = SEED) -> tuple[list[Run], dict]:
    provider = FakeProvider()
    runs: list[Run] = []
    labels: dict = {
        "generator": "python -m examples.bad_agent.generate",
        "seed": seed,
        "note": "Answer key for analyzer tests. 'planted' lists every problem put into a run "
                "and the spans that prove it. Analyzers must never read this file.",
        "runs": {},
    }
    with simulated_time(START_MS) as clock, deterministic_ids(seed):
        for i, sc in enumerate(SCENARIOS):
            clock.set_time_ms(START_MS + i * RUN_SPACING_MS)
            result = SupportAgent(provider, sc.faults).run(
                sc.question, customer_id=sc.customer_id, order_id=sc.order_id, sku=sc.sku,
                tags={"scenario": sc.name},
            )
            expected = expected_answer(sc.question, sc.order_id, sc.sku)
            runs.append(result.run)
            labels["runs"][result.run.run_id] = {
                "scenario": sc.name,
                "description": sc.description,
                "faults": sc.faults.active(),
                "clean": not sc.faults.active(),
                "expected_output": expected,
                "answer_correct": result.answer == expected,
                "planted": [p.as_dict() for p in result.planted],
            }
    return runs, labels


def render(runs: list[Run], labels: dict) -> dict[str, str]:
    return {
        "runs.json": json.dumps([r.model_dump(mode="json") for r in runs], indent=1) + "\n",
        "labels.json": json.dumps(labels, indent=2) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the files differ.")
    parser.add_argument("--save", action="store_true", help="Also save runs to the trace DB.")
    args = parser.parse_args(argv)

    runs, labels = generate()
    files = render(runs, labels)
    if args.check:
        stale = [n for n, text in files.items()
                 if not (args.out / n).is_file()
                 or (args.out / n).read_text(encoding="utf-8").replace("\r\n", "\n") != text]
        if stale:
            print(f"out of date: {', '.join(stale)}. Run: python -m examples.bad_agent.generate")
            return 1
        print("fixtures are up to date")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8", newline="\n")
    planted = sum(len(v["planted"]) for v in labels["runs"].values())
    print(f"wrote {len(runs)} runs and {planted} planted problems to {args.out}")
    if args.save:
        store = get_default_store()
        for run in runs:
            store.save_run(run)
        print(f"saved {len(runs)} runs to {store.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
