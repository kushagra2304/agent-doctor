"""Replay engine: re-run saved runs with a new prompt or model, tools answered from the recording.

>>> from agent_doctor.replay import replay
>>> result = replay(run, handle, model_map={"gpt-4o": "gpt-4o-mini"})
>>> result.output_text, result.run.duration_ms
"""

from agent_doctor.replay.engine import (
    ReplayedToolError,
    ReplayMiss,
    ReplayResult,
    ReplaySession,
    ToolPlan,
    load_agent,
    replay,
)

__all__ = [
    "ReplayMiss",
    "ReplayResult",
    "ReplaySession",
    "ReplayedToolError",
    "ToolPlan",
    "load_agent",
    "replay",
]
