"""Collect spans for the run that is currently executing.

The active run and the current parent span live in context variables, so nested
calls (a tool that calls an LLM) are linked correctly, including under asyncio.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar, Token
from typing import Any

from agent_doctor import config
from agent_doctor._util import new_id, now_ms, to_jsonable, warn_once
from agent_doctor.schema import Run, Span, SpanKind, SpanStatus
from agent_doctor.store import TraceStore, get_default_store

_active_run: ContextVar[RunRecorder | None] = ContextVar("agent_doctor_run", default=None)
_active_parent: ContextVar[str | None] = ContextVar("agent_doctor_parent", default=None)


def current_run() -> RunRecorder | None:
    """The run being recorded right now, or None."""
    return _active_run.get()


def current_parent_id() -> str | None:
    return _active_parent.get()


class Timer:
    def __init__(self) -> None:
        self.start_ms = now_ms()
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def end_ms(self) -> int:
        return self.start_ms + self.elapsed_ms()


def error_fields(exc: BaseException | None) -> dict[str, Any]:
    if exc is None:
        return {}
    return {
        "status": SpanStatus.ERROR,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000] or None,
    }


class RunRecorder:
    """Records one run. Use as a context manager, or through ``@trace_run``.

    >>> with RunRecorder("support-bot", task_input="Where is my order?") as run:
    ...     run.final_output = answer(...)
    """

    def __init__(
        self,
        agent: str,
        *,
        task_input: str = "",
        agent_version: str | None = None,
        prompt_version: str | None = None,
        tags: dict[str, Any] | None = None,
        store: TraceStore | None = None,
        save: bool = True,
        run_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.task_input = task_input
        self.agent_version = agent_version
        self.prompt_version = prompt_version
        self.tags = dict(tags or {})
        self.store = store
        self.save = save
        self.run_id = run_id or new_id("run")
        self.root_span_id = new_id("sp")
        self.final_output: Any = None
        self.run: Run | None = None
        self._spans: list[Span] = []
        self._lock = threading.Lock()
        self._timer: Timer | None = None
        self._tokens: list[Token] = []

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> RunRecorder:
        self._timer = Timer()
        return self

    def finish(self, exc: BaseException | None = None) -> Run | None:
        """Build the Run, save it, and return it. Never raises."""
        if self._timer is None:
            self.start()
        assert self._timer is not None
        try:
            end_ms = max(self._timer.end_ms(), *(s.end_ms for s in self._spans), self._timer.start_ms)
            root = Span(
                span_id=self.root_span_id, run_id=self.run_id, kind=SpanKind.AGENT_STEP,
                name=self.agent, start_ms=self._timer.start_ms, end_ms=end_ms,
                **error_fields(exc),
            )
            output = self.final_output
            if output is not None and not isinstance(output, str):
                import json

                output = json.dumps(to_jsonable(output))
            run = Run(
                run_id=self.run_id, agent_name=self.agent, agent_version=self.agent_version,
                task_input=str(self.task_input)[:10_000], final_output=output,
                status=SpanStatus.ERROR if exc is not None else SpanStatus.OK,
                spans=[root, *self.spans], tags=to_jsonable(self.tags),
            )
        except Exception as e:  # a bug here must not break the user's agent
            warn_once(f"build:{type(e).__name__}", f"could not build run {self.run_id}: {e}")
            return None
        self.run = run
        if self.save and config.recording_enabled():
            try:
                (self.store or get_default_store()).save_run(run)
            except Exception as e:
                warn_once(f"save:{type(e).__name__}", f"could not save run {self.run_id}: {e}")
        return run

    def __enter__(self) -> RunRecorder:
        self.start()
        self._tokens = [_active_run.set(self), _active_parent.set(self.root_span_id)]
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        tok_run, tok_parent = self._tokens
        try:
            _active_parent.reset(tok_parent)
            _active_run.reset(tok_run)
        except ValueError:  # exited in a different context; fall back to clearing
            _active_parent.set(None)
            _active_run.set(None)
        self.finish(exc)
        return False

    async def __aenter__(self) -> RunRecorder:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    # -- spans ------------------------------------------------------------------
    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def add_span(self, **fields: Any) -> Span | None:
        """Create a span in this run. Returns None (and warns once) if it is invalid."""
        fields.setdefault("run_id", self.run_id)
        if fields.get("kind") in (SpanKind.LLM, "llm") and self.prompt_version:
            fields.setdefault("prompt_version", self.prompt_version)
        try:
            span = Span(**fields)
        except Exception as e:
            warn_once(f"span:{type(e).__name__}", f"dropped an invalid span: {e}")
            return None
        with self._lock:
            self._spans.append(span)
        return span

    def find_failed_tool(self, name: str, args_hash: str | None) -> str | None:
        """Latest failed call of this tool with the same arguments (for retry_of)."""
        with self._lock:
            for span in reversed(self._spans):
                if (
                    span.kind is SpanKind.TOOL
                    and span.name == name
                    and span.args_hash == args_hash
                    and span.status is SpanStatus.ERROR
                ):
                    return span.span_id
        return None


class SpanHandle:
    """Returned by ``span()``; set ``output`` or ``attributes`` on it inside the block."""

    def __init__(self) -> None:
        self.output: Any = None
        self.attributes: dict[str, Any] = {}


class span:  # noqa: N801  (used like a function: ``with span("plan"):``)
    """Record a custom step, e.g. ``with span("retrieve", kind="retrieval") as s:``.

    Spans created inside the block become its children. Outside a run it does nothing.
    """

    def __init__(self, name: str, kind: SpanKind | str = SpanKind.AGENT_STEP, **attributes: Any):
        self.name = name
        self.kind = SpanKind(kind)
        self.handle = SpanHandle()
        self.handle.attributes.update(attributes)
        self._rec: RunRecorder | None = None

    def __enter__(self) -> SpanHandle:
        rec = current_run()
        if rec is None or not config.recording_enabled():
            return self.handle
        self._rec = rec
        self._id = new_id("sp")
        self._parent = current_parent_id()
        self._token = _active_parent.set(self._id)
        self._timer = Timer()
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._rec is None:
            return False
        try:
            _active_parent.reset(self._token)
        except ValueError:
            _active_parent.set(self._parent)
        out = self.handle.output
        self._rec.add_span(
            span_id=self._id, parent_id=self._parent, kind=self.kind, name=self.name,
            start_ms=self._timer.start_ms, end_ms=self._timer.end_ms(),
            output_text=None if out is None else (out if isinstance(out, str) else repr(out))[:20_000],
            attributes=to_jsonable(self.handle.attributes),
            **error_fields(exc),
        )
        return False

    async def __aenter__(self) -> SpanHandle:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)
