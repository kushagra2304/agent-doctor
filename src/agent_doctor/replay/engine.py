"""Replay a saved run: call the agent again on the same input, with tool calls answered
from the recording, so a new prompt or model can be tested against real past traffic.

LLM calls go live (they are what is being tested). Tool calls are matched to recorded
calls by tool name and arguments, in order; a matched call returns the recorded output, or
raises the recorded error, without running the tool. Tools marked ``side_effects=True``
are never run unless ``allow_side_effects=True``.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib
import inspect
import sys
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_doctor import config
from agent_doctor._util import SystemClock, get_clock, import_path
from agent_doctor.schema import Run, Span, SpanStatus
from agent_doctor.sdk.recorder import RunRecorder, _active_replay
from agent_doctor.store import TraceStore, get_default_store

ToolMode = Literal["recorded", "live"]
MissMode = Literal["live", "error"]

# Values of the span attribute "replay": how each tool call was answered.
RECORDED, REPEATED, STUBBED = "recorded", "recorded-repeat", "stubbed"
LIVE, UNRECORDED = "live", "live-unrecorded"


class ReplayMiss(Exception):
    """A tool call that matches no recorded call, with ``on_miss="error"``."""


class ReplayedToolError(Exception):
    """A recorded tool error whose exception class could not be found to raise again."""

    def __init__(self, error_type: str | None, message: str):
        super().__init__(f"{error_type or 'Error'}: {message}" if message else error_type or "Error")
        self.error_type = error_type


def _find_exception(name: str | None, fn: Callable) -> type[Exception] | None:
    """The exception class called ``name``, looked up where the tool is defined, then in builtins."""
    if not name:
        return None
    module_globals = getattr(inspect.unwrap(fn), "__globals__", {})
    for namespace in (module_globals, vars(builtins)):
        obj = namespace.get(name)
        if isinstance(obj, type) and issubclass(obj, Exception):
            return obj
    return None


@dataclass
class ToolPlan:
    """How to answer one tool call during a replay."""

    action: Literal["live", "recorded", "error", "stub"]
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None
    raise_exc: BaseException | None = None
    delay_s: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def wait(self) -> None:
        if self.delay_s:
            get_clock().sleep(self.delay_s)

    async def wait_async(self) -> None:
        if self.delay_s:
            clock = get_clock()
            if isinstance(clock, SystemClock):
                await asyncio.sleep(self.delay_s)
            else:
                clock.sleep(self.delay_s)

    def exception(self, fn: Callable) -> BaseException:
        if self.raise_exc is not None:
            return self.raise_exc
        message = self.error_message or ""
        cls = _find_exception(self.error_type, fn)
        if cls is not None:
            try:
                return cls(message)
            except Exception:  # needs other constructor arguments
                pass
        return ReplayedToolError(self.error_type, message)


def _replayable(span: Span) -> bool:
    """A recorded tool call whose result is known: an error, or an output that was captured."""
    return span.is_error or span.attributes.get("output_captured", True)


class ReplaySession:
    """Answers tool calls from a recorded run and collects the runs recorded meanwhile.

    The SDK finds it through a context variable and calls ``plan_tool`` for every tool
    call and ``on_run_finished`` for every finished run.
    """

    def __init__(
        self,
        source: Run | None = None,
        *,
        tools: ToolMode = "recorded",
        on_miss: MissMode = "live",
        model_map: dict[str, str] | None = None,
        allow_side_effects: bool = False,
        side_effect_result: Any = None,
        tool_latency: Literal["recorded", "none"] = "recorded",
    ) -> None:
        self.source = source
        self.tools = tools
        self.on_miss = on_miss
        self.model_map = dict(model_map or {})
        self.allow_side_effects = allow_side_effects
        self.side_effect_result = side_effect_result
        self.tool_latency = tool_latency
        self.runs: list[Run] = []
        self.counts: Counter[str] = Counter()
        self._queues: dict[tuple[str, str | None], deque[Span]] = {}
        self._last: dict[tuple[str, str | None], Span] = {}
        for span in source.tool_spans if source is not None else []:
            if _replayable(span):
                self._queues.setdefault((span.name, span.args_hash), deque()).append(span)

    def _next(self, name: str, args_hash: str | None) -> tuple[Span | None, bool]:
        """The next recorded call with these arguments; repeats the last one once used up."""
        key = (name, args_hash)
        queue = self._queues.get(key)
        if queue:
            span = queue.popleft()
            self._last[key] = span
            return span, False
        return self._last.get(key), True

    def _from_recording(self, span: Span, repeated: bool) -> ToolPlan:
        attrs = {"replay": REPEATED if repeated else RECORDED, "replay_of_span": span.span_id}
        delay = span.duration_ms / 1000 if self.tool_latency == "recorded" else 0.0
        if span.status is SpanStatus.ERROR:
            return ToolPlan("error", error_type=span.error_type, error_message=span.error_message,
                            delay_s=delay, attributes=attrs)
        return ToolPlan("recorded", output=span.tool_output, delay_s=delay, attributes=attrs)

    def plan_tool(self, name: str, args_hash: str | None, side_effects: bool) -> ToolPlan:
        if side_effects and self.allow_side_effects:
            plan = ToolPlan("live", attributes={"replay": LIVE})
        elif side_effects:
            span, repeated = self._next(name, args_hash)
            plan = (self._from_recording(span, repeated) if span is not None
                    else ToolPlan("stub", output=self.side_effect_result))
            plan.attributes["replay"] = STUBBED  # never run, whatever the recording says
        elif self.tools == "live":
            plan = ToolPlan("live", attributes={"replay": LIVE})
        else:
            span, repeated = self._next(name, args_hash)
            if span is not None:
                plan = self._from_recording(span, repeated)
            elif self.on_miss == "error":
                plan = ToolPlan("error", raise_exc=ReplayMiss(
                    f"{name} was called with arguments that were never recorded"),
                    attributes={"replay": UNRECORDED})
            else:
                plan = ToolPlan("live", attributes={"replay": UNRECORDED})
        self.counts[plan.attributes["replay"]] += 1
        return plan

    def on_run_finished(self, run: Run) -> None:
        self.runs.append(run)


@dataclass
class ReplayResult:
    original: Run | None
    run: Run | None = field(repr=False)
    output: Any
    error: str | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    """How tool calls were answered: recorded, recorded-repeat, live, live-unrecorded, stubbed."""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def output_text(self) -> str | None:
        if self.output is None:
            return self.run.final_output if self.run is not None else None
        return self.output if isinstance(self.output, str) else str(self.output)


def replay(
    run: Run | None,
    agent: Callable[..., Any],
    args: tuple | list | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolMode = "recorded",
    on_miss: MissMode = "live",
    model_map: dict[str, str] | None = None,
    allow_side_effects: bool = False,
    side_effect_result: Any = None,
    tool_latency: Literal["recorded", "none"] = "recorded",
    label: str | None = None,
    save: bool = False,
    store: TraceStore | None = None,
) -> ReplayResult:
    """Run ``agent`` again on a saved run's input and return the new run.

    ``agent`` is called as ``agent(*args, **kwargs)``; with no ``args`` it gets the run's
    task input. It may be decorated with ``@trace_run`` or open its own ``RunRecorder``;
    either way the new run is captured. Exceptions from the agent are caught and reported
    in ``ReplayResult.error``. ``run=None`` runs the agent live with side effects still
    blocked, which is how evaluation runs cases that have no recording.

    - ``tools``: "recorded" answers tool calls from the recording; "live" runs them all
      (except side-effect tools).
    - ``on_miss``: for a tool call that was never recorded, "live" runs it, "error" raises
      ``ReplayMiss`` inside the agent.
    - ``model_map``: swap models without changing code, e.g. ``{"gpt-4o": "gpt-4o-mini"}``.
    - ``tool_latency``: "recorded" waits as long as the recorded call took, so durations
      compare fairly with the original; "none" answers at once.
    """
    if not config.recording_enabled():
        raise RuntimeError("replay needs recording; unset AGENT_DOCTOR_DISABLED")
    if args is None:
        if run is None:
            raise ValueError("pass args when there is no run to take the input from")
        args = (run.task_input,)
    session = ReplaySession(
        run, tools=tools, on_miss=on_miss, model_map=model_map, allow_side_effects=allow_side_effects,
        side_effect_result=side_effect_result, tool_latency=tool_latency,
    )
    task_input = run.task_input if run is not None else str(args[0]) if args else ""
    outer = RunRecorder(run.agent_name if run is not None else getattr(agent, "__name__", "agent"),
                        task_input=task_input, save=False)
    output: Any = None
    error: str | None = None
    token = _active_replay.set(session)
    try:
        with outer:
            output = agent(*args, **(kwargs or {}))
            if inspect.isawaitable(output):
                output = asyncio.run(_await(output))
            outer.final_output = output
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        _active_replay.reset(token)

    # An agent that opens its own RunRecorder records a separate run; prefer it.
    inner = [r for r in session.runs if r is not outer.run]
    new_run = inner[-1] if inner else outer.run
    if new_run is not None:
        tags = {**new_run.tags, "replay_of": run.run_id if run is not None else None}
        if label:
            tags["replay_label"] = label
        if model_map:
            tags["model_map"] = dict(model_map)
        new_run.tags = {k: v for k, v in tags.items() if v is not None}
        if save:
            (store or get_default_store()).save_run(new_run)
    return ReplayResult(original=run, run=new_run, output=output, error=error,
                        tool_calls=dict(session.counts))


async def _await(awaitable: Any) -> Any:
    return await awaitable


def load_agent(spec: str) -> Callable[..., Any]:
    """Import an agent from ``"package.module:function"`` or ``"path/to/file.py:function"``.

    The attribute may be dotted (``"app:bot.handle"``).
    """
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(f"expected 'module:function' or 'file.py:function', got {spec!r}")
    if target.endswith(".py"):
        module = import_path(target)
    else:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(target)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj
