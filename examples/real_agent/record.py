"""Record the real agent answering every question, into its own trace database.

From the repository root (needs a Hugging Face login: hf auth login):

    python -m examples.real_agent.record
    agent-doctor --db examples/real_agent/traces.db scan
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent / "traces.db"
os.environ.setdefault("AGENT_DOCTOR_DB", str(DB))

from examples.real_agent.agent import MODEL, handle  # noqa: E402
from examples.real_agent.questions import QUESTIONS  # noqa: E402


def main() -> int:
    print(f"model {MODEL}; saving runs to {os.environ['AGENT_DOCTOR_DB']}")
    for i, (question, _) in enumerate(QUESTIONS, start=1):
        try:
            answer = handle(question)
        except Exception as e:  # keep going: a failed run is recorded too, and worth analyzing
            answer = f"(failed: {type(e).__name__}: {e})"
        print(f"{i:2}. {question}\n    -> {answer[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
