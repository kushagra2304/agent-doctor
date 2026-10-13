"""The normalized trace schema.

Every trace, whatever its source (our SDK, OpenTelemetry, Langfuse, LangSmith),
is converted into these models. Everything else in Agent Doctor reads them:
the store, the analyzers, the replay engine and the MCP tools.

- ``Span``: one step of a run (one LLM call, tool call, retrieval or agent step).
- ``Run``: one complete execution of an agent for one input; holds its spans.
- ``Finding``: one problem an analyzer found in a run, with its evidence.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"


class SpanKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    AGENT_STEP = "agent_step"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class FindingType(str, Enum):
    COST = "cost"
    REDUNDANT_TOOL_CALL = "redundant_tool_call"
    UNUSED_TOOL_OUTPUT = "unused_tool_output"
    RETRY_LOOP = "retry_loop"
    LATENCY = "latency"
    PARALLELIZABLE_CALLS = "parallelizable_calls"
    CONTEXT_BLOAT = "context_bloat"
    ERROR = "error"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def compute_args_hash(args: dict[str, Any] | None) -> str | None:
    """Stable short hash of tool arguments, used to spot repeated tool calls.

    Key order does not matter: ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    hash the same. Values that are not JSON types are hashed by their ``str()``.
    """
    if args is None:
        return None
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)


class Span(_Model):
    """One step inside a run."""

    span_id: str = Field(min_length=1)
    parent_id: str | None = None
    run_id: str = Field(min_length=1)
    kind: SpanKind
    name: str = Field(min_length=1, description="Model name for LLM spans, tool name otherwise.")
    start_ms: int = Field(ge=0, description="Start time, Unix epoch milliseconds.")
    end_ms: int = Field(ge=0, description="End time, Unix epoch milliseconds.")
    status: SpanStatus = SpanStatus.OK
    error_type: str | None = None
    error_message: str | None = None

    # LLM fields
    model: str | None = None
    provider: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0, description="Input tokens served from cache.")
    time_to_first_token_ms: int | None = Field(default=None, ge=0)
    prompt_version: str | None = Field(default=None, description="Hash or label of the prompt.")
    input_messages: list[dict[str, Any]] | None = None
    output_text: str | None = None

    # Tool fields
    tool_args: dict[str, Any] | None = None
    args_hash: str | None = None
    tool_output: Any | None = Field(default=None, description="Stored so runs can be replayed.")
    side_effects: bool = Field(default=False, description="True = never call live during replay.")
    retry_of: str | None = Field(default=None, description="span_id this span retries.")

    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Span:
        if self.end_ms < self.start_ms:
            raise ValueError(f"span {self.span_id}: end_ms is before start_ms")
        if self.cached_tokens > self.input_tokens:
            raise ValueError(f"span {self.span_id}: cached_tokens exceeds input_tokens")
        ttft = self.time_to_first_token_ms
        if ttft is not None and ttft > self.duration_ms:
            raise ValueError(f"span {self.span_id}: time_to_first_token_ms exceeds the duration")
        if self.kind is SpanKind.LLM and not self.model:
            # The span name is the model name for LLM spans, so default to it.
            object.__setattr__(self, "model", self.name)
        if self.status is SpanStatus.ERROR and not self.error_type:
            object.__setattr__(self, "error_type", "unknown")
        if self.tool_args is not None and self.args_hash is None:
            object.__setattr__(self, "args_hash", compute_args_hash(self.tool_args))
        if self.retry_of == self.span_id:
            raise ValueError(f"span {self.span_id}: a span cannot retry itself")
        if self.parent_id == self.span_id:
            raise ValueError(f"span {self.span_id}: a span cannot be its own parent")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_error(self) -> bool:
        return self.status is SpanStatus.ERROR


class Run(_Model):
    """One complete execution of an agent for one input."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_version: str | None = None
    task_input: str
    final_output: str | None = None
    status: SpanStatus = SpanStatus.OK
    spans: list[Span] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Run:
        ids: set[str] = set()
        for span in self.spans:
            if span.run_id != self.run_id:
                raise ValueError(
                    f"span {span.span_id} has run_id {span.run_id!r}, expected {self.run_id!r}"
                )
            if span.span_id in ids:
                raise ValueError(f"duplicate span_id {span.span_id!r}")
            ids.add(span.span_id)
        for span in self.spans:
            if span.parent_id is not None and span.parent_id not in ids:
                raise ValueError(f"span {span.span_id}: unknown parent_id {span.parent_id!r}")
            if span.retry_of is not None and span.retry_of not in ids:
                raise ValueError(f"span {span.span_id}: unknown retry_of {span.retry_of!r}")
        return self

    # Views
    @property
    def ordered_spans(self) -> list[Span]:
        """Spans sorted by start time (ties keep their original order)."""
        return sorted(self.spans, key=lambda s: s.start_ms)

    @property
    def llm_spans(self) -> list[Span]:
        return [s for s in self.ordered_spans if s.kind is SpanKind.LLM]

    @property
    def tool_spans(self) -> list[Span]:
        return [s for s in self.ordered_spans if s.kind is SpanKind.TOOL]

    def get_span(self, span_id: str) -> Span:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        raise KeyError(span_id)

    def children(self, span_id: str) -> list[Span]:
        return [s for s in self.ordered_spans if s.parent_id == span_id]

    # Totals
    @property
    def start_ms(self) -> int | None:
        return min((s.start_ms for s in self.spans), default=None)

    @property
    def end_ms(self) -> int | None:
        return max((s.end_ms for s in self.spans), default=None)

    @property
    def duration_ms(self) -> int:
        if not self.spans:
            return 0
        return self.end_ms - self.start_ms  # type: ignore[operator]

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.llm_spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.llm_spans)

    @property
    def total_cached_tokens(self) -> int:
        return sum(s.cached_tokens for s in self.llm_spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.spans if s.is_error)

    @property
    def models_used(self) -> list[str]:
        return sorted({s.model for s in self.llm_spans if s.model})

    # I/O
    @classmethod
    def from_json_file(cls, path: str) -> Run:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "spans": len(self.spans),
            "llm_calls": len(self.llm_spans),
            "tool_calls": len(self.tool_spans),
            "errors": self.error_count,
            "duration_ms": self.duration_ms,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "models": self.models_used,
        }


class Finding(_Model):
    """One problem an analyzer found, with the evidence behind it."""

    run_id: str = Field(min_length=1)
    type: FindingType
    severity: Severity
    title: str = Field(min_length=1, description="One line, e.g. 'search_orders called 3x'.")
    detail: str = ""
    suggestion: str = Field(default="", description="What to change to fix it.")
    evidence_span_ids: list[str] = Field(min_length=1)
    wasted_cost_usd: float | None = Field(default=None, ge=0)
    wasted_ms: int | None = Field(default=None, ge=0)
    wasted_tokens: int | None = Field(default=None, ge=0)
    data: dict[str, Any] = Field(default_factory=dict)

    def check_evidence(self, run: Run) -> None:
        """Raise if any evidence span is missing from the run."""
        if run.run_id != self.run_id:
            raise ValueError(f"finding is for run {self.run_id!r}, got {run.run_id!r}")
        known = {s.span_id for s in run.spans}
        missing = [sid for sid in self.evidence_span_ids if sid not in known]
        if missing:
            raise ValueError(f"evidence spans not in run: {missing}")
