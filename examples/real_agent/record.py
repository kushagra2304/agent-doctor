"""Record the real agent answering every question, into its own trace database.

From the repository root, with a free local model (Ollama, then: ollama pull qwen2.5:3b):

    AD_BASE_URL=http://localhost:11434/v1 python -m examples.real_agent.record
    agent-doctor --db examples/real_agent/traces.db scan --prices examples/real_agent/local_pricing.yaml

Or leave AD_BASE_URL unset to use Hugging Face Inference Providers (hf auth login).
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
