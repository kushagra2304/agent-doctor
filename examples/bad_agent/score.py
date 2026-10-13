"""Grade the analyzers against the bad-agent answer key.

    python -m examples.bad_agent.score            # summary table
    python -m examples.bad_agent.score -v         # every miss and unexpected finding

A planted problem counts as found when a finding of the same type points at at least
half of the planted evidence spans. Findings on clean runs are false alarms; findings
on faulty runs that match nothing planted are reported as unexpected.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent_doctor.analyzers import analyze_run
from agent_doctor.pricing import PriceTable
from agent_doctor.schema import Finding, Run

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bad_agent"
PRICES = Path(__file__).resolve().parents[1] / "demo_pricing.yaml"
MIN_OVERLAP = 0.5


@dataclass
class Score:
    planted: int = 0
    found: int = 0
    false_alarms: list[tuple[str, Finding]] = field(default_factory=list)
    unexpected: list[tuple[str, Finding]] = field(default_factory=list)
    missed: list[tuple[str, dict]] = field(default_factory=list)
    by_kind: dict[str, list[int]] = field(default_factory=dict)  # kind -> [planted, found]

    @property
    def recall(self) -> float:
        return self.found / self.planted if self.planted else 1.0

    @property
    def precision(self) -> float:
        wrong = len(self.false_alarms) + len(self.unexpected)
        return self.found / (self.found + wrong) if (self.found + wrong) else 1.0


def matches(finding: Finding, planted: dict) -> bool:
    if finding.type.value != planted["type"]:
        return False
    want = set(planted["evidence_span_ids"])
    return len(want & set(finding.evidence_span_ids)) / len(want) >= MIN_OVERLAP


def load() -> tuple[dict[str, Run], dict]:
    runs = json.loads((FIXTURES / "runs.json").read_text(encoding="utf-8"))
    labels = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))["runs"]
    return {r["run_id"]: Run.model_validate(r) for r in runs}, labels


def score(runs: dict[str, Run] | None = None, labels: dict | None = None) -> Score:
    if runs is None or labels is None:
        runs, labels = load()
    prices = PriceTable.load(PRICES)
    result = Score()
    for rid, run in runs.items():
        lab = labels[rid]
        findings = analyze_run(run, prices).findings  # analyzers never see the labels
        used: set[int] = set()
        for p in lab["planted"]:
            result.planted += 1
            row = result.by_kind.setdefault(p["kind"], [0, 0])
            row[0] += 1
            hit = next((i for i, f in enumerate(findings) if matches(f, p)), None)
            if hit is None:
                result.missed.append((lab["scenario"], p))
            else:
                result.found += 1
                row[1] += 1
                used.add(hit)
        for i, f in enumerate(findings):
            if i in used or any(matches(f, p) for p in lab["planted"]):
                continue
            (result.false_alarms if lab["clean"] else result.unexpected).append((lab["scenario"], f))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade analyzers against the answer key.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    s = score()
    print(f"{'planted problem':32} found")
    for kind, (planted, found) in sorted(s.by_kind.items()):
        print(f"  {kind:30} {found}/{planted}" + ("" if found == planted else "  <-- missed"))
    print(f"\nrecall {s.recall:.0%} ({s.found}/{s.planted})   precision {s.precision:.0%}   "
          f"false alarms on clean runs: {len(s.false_alarms)}   unexpected: {len(s.unexpected)}")
    if args.verbose or s.missed or s.false_alarms or s.unexpected:
        for scen, p in s.missed:
            print(f"MISSED     {scen:24} {p['kind']}")
        for scen, f in s.false_alarms:
            print(f"FALSE      {scen:24} {f.type.value}: {f.title}")
        for scen, f in s.unexpected:
            print(f"UNEXPECTED {scen:24} {f.type.value}: {f.title}")
    return 0 if (not s.missed and not s.false_alarms) else 1


if __name__ == "__main__":
    sys.exit(main())
