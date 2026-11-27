"""Agent Doctor: find out why your AI agent is expensive, slow or unreliable."""

__version__ = "0.5.0"

from agent_doctor.schema import (  # noqa: E402
    Finding,
    FindingType,
    Run,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    compute_args_hash,
)
from agent_doctor.sdk import (  # noqa: E402
    RunRecorder,
    current_run,
    instrument,
    record_run,
    span,
    trace_run,
    trace_tool,
)

__all__ = [
    "Finding",
    "FindingType",
    "Run",
    "RunRecorder",
    "Severity",
    "Span",
    "SpanKind",
    "SpanStatus",
    "__version__",
    "compute_args_hash",
    "current_run",
    "instrument",
    "record_run",
    "span",
    "trace_run",
    "trace_tool",
]
