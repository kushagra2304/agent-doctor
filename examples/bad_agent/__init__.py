"""A support agent with planted problems, used to build Agent Doctor's test fixtures.

Regenerate the fixtures from the repository root with:

    python -m examples.bad_agent.generate
"""

from .agent import AgentResult, Faults, SupportAgent, expected_answer
from .fake_llm import FakeInferenceClient, FakeProvider

__all__ = ["AgentResult", "FakeInferenceClient", "FakeProvider", "Faults", "SupportAgent",
           "expected_answer"]
