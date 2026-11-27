"""Copy the demo data into space/data/ and precompute the evaluation results.

Run from the repository root before uploading the Space:

    python space/sync.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "space" / "data"


def main() -> int:
    sys.path.insert(0, str(REPO))
    from examples.bad_agent.eval_suite import suite

    DATA.mkdir(exist_ok=True)
    for src in [REPO / "tests" / "fixtures" / "bad_agent" / "runs.json",
                REPO / "tests" / "fixtures" / "bad_agent" / "labels.json",
                REPO / "examples" / "demo_pricing.yaml"]:
        shutil.copy(src, DATA / src.name)
    reports, comparisons = suite.run()
    payload = {
        "reports": [r.model_dump(mode="json", exclude={"results"}) for r in reports],
        "comparisons": [c.model_dump(mode="json") for c in comparisons],
    }
    (DATA / "eval.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote demo data to {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
