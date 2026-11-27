"""``@trace_run`` and ``@trace_tool``."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any, TypeVar

from agent_doctor import config
from agent_doctor._util import new_id, to_jsonable, warn_once
from agent_doctor.schema import SpanKind, compute_args_hash
from agent_doctor.sdk.recorder import (
    RunRecorder,
    Timer,
    _active_parent,
    current_parent_id,
    current_replay,
    current_run,
    error_fields,
    span,
)
from agent_doctor.store import TraceStore

F = TypeVar("F", bound=Callable[..., Any])
_SKIP = {"self", "cls"}


def _signature(fn: Callable) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def bound_args(sig: inspect.Signature | None, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Call arguments by parameter name, defaults filled in, self/cls dropped."""
    if sig is not None:
        try:
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            return {k: to_jsonable(v) for k, v in ba.arguments.items() if k not in _SKIP}
        except TypeError:
            pass
    return {"args": to_jsonable(list(args)), "kwargs": to_jsonable(kwargs)}


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable(value))


def _task_input(sig, args, kwargs, input_arg: str | None) -> str:
    named = bound_args(sig, args, kwargs)
    if input_arg and input_arg in named:
        value = named[input_arg]
        return value if isinstance(value, str) else json.dumps(value)
    for value in named.values():
        if isinstance(value, str):
            return value
    return json.dumps(named) if named else ""


def trace_run(
    fn: F | None = None,
    *,
    agent: str | None = None,
    agent_version: str | None = None,
    prompt_version: str | None = None,
    tags: dict[str, Any] | None = None,
    input_arg: str | None = None,
    store: TraceStore | None = None,
    save: bool = True,
) -> Any:
    """Record each call of the decorated function as one agent run.

    Usable bare (``@trace_run``) or with options (``@trace_run(agent="support-bot")``).
    The first string argument (or ``input_arg``) becomes the run's task input and the
    return value becomes its final output. If a run is already active, the call is
    recorded as a step inside it instead of starting a new run.
    """

    def decorate(f: F) -> F:
        name = agent or f.__name__
        sig = _signature(f)

        def make_recorder(args, kwargs) -> RunRecorder:
            return RunRecorder(
                name, task_input=_task_input(sig, args, kwargs, input_arg),
                agent_version=agent_version, prompt_version=prompt_version, tags=tags,
                store=store, save=save,
            )

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                if not config.recording_enabled():
                    return await f(*args, **kwargs)
                if current_run() is not None:
                    async with span(name):
                        return await f(*args, **kwargs)
                async with make_recorder(args, kwargs) as rec:
                    result = await f(*args, **kwargs)
                    rec.final_output = _as_text(result)
                    return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not config.recording_enabled():
                return f(*args, **kwargs)
            if current_run() is not None:
                with span(name):
                    return f(*args, **kwargs)
            with make_recorder(args, kwargs) as rec:
                result = f(*args, **kwargs)
                rec.final_output = _as_text(result)
                return result

        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate


class _ToolCall:
    def __init__(self, rec: RunRecorder, tool_name: str, args: dict, side_effects: bool):
        self.rec = rec
        self.tool_name = tool_name
        self.args = args
        self.side_effects = side_effects
        self.args_hash = compute_args_hash(args)
        self.retry_of = rec.find_failed_tool(tool_name, self.args_hash)
        self.span_id = new_id("sp")
        self.parent = current_parent_id()
        self.token = _active_parent.set(self.span_id)
        self.timer = Timer()

    def end(self, result: Any = None, exc: BaseException | None = None, capture: bool = True,
            attributes: dict[str, Any] | None = None):
        try:
            _active_parent.reset(self.token)
        except ValueError:
            _active_parent.set(self.parent)
        attrs = dict(attributes or {})
        if not capture:
            attrs["output_captured"] = False  # so replay knows None is not the real output
        self.rec.add_span(
            span_id=self.span_id, parent_id=self.parent, kind=SpanKind.TOOL, name=self.tool_name,
            start_ms=self.timer.start_ms, end_ms=self.timer.end_ms(), tool_args=self.args,
            args_hash=self.args_hash, retry_of=self.retry_of, side_effects=self.side_effects,
            tool_output=to_jsonable(result) if (capture and exc is None) else None,
            attributes=attrs, **error_fields(exc),
        )


def _replay_plan(call: _ToolCall | None, sig, args, kwargs, tool_name: str, side_effects: bool):
    """Ask the active replay session how to handle this call; None when not replaying."""
    session = current_replay()
    if session is None:
        return None
    args_hash = call.args_hash if call is not None else compute_args_hash(bound_args(sig, args, kwargs))
    return session.plan_tool(tool_name, args_hash, side_effects)


def _served(call: _ToolCall | None, plan: Any, f: Callable) -> Any:
    """Answer a call from the replay plan (recorded output, recorded error or stub)."""
    if plan.action == "error":
        exc = plan.exception(f)
        if call is not None:
            call.end(exc=exc, attributes=plan.attributes)
        raise exc
    if call is not None:
        call.end(plan.output, attributes=plan.attributes)
    return plan.output


def trace_tool(
    fn: F | None = None,
    *,
    name: str | None = None,
    side_effects: bool = False,
    capture_output: bool = True,
) -> Any:
    """Record each call of the decorated function as a tool span.

    ``side_effects=True`` marks tools that change the world (send email, write to a
    database, take payment); replay and evaluation never call them live. A call with the same
    arguments as an earlier failed call is linked to it as a retry. Outside a run the
    function runs normally and nothing is recorded.
    """

    def decorate(f: F) -> F:
        tool_name = name or f.__name__
        sig = _signature(f)

        def begin(args, kwargs) -> _ToolCall | None:
            rec = current_run()
            if rec is None or not config.recording_enabled():
                return None
            try:
                return _ToolCall(rec, tool_name, bound_args(sig, args, kwargs), side_effects)
            except Exception as e:
                warn_once(f"tool:{type(e).__name__}", f"could not record tool {tool_name}: {e}")
                return None

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                call = begin(args, kwargs)
                plan = _replay_plan(call, sig, args, kwargs, tool_name, side_effects)
                if plan is not None and plan.action != "live":
                    await plan.wait_async()
                    return _served(call, plan, f)
                if call is None:
                    return await f(*args, **kwargs)
                attrs = plan.attributes if plan is not None else None
                try:
                    result = await f(*args, **kwargs)
                except BaseException as e:
                    call.end(exc=e, attributes=attrs)
                    raise
                call.end(result, capture=capture_output, attributes=attrs)
                return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            call = begin(args, kwargs)
            plan = _replay_plan(call, sig, args, kwargs, tool_name, side_effects)
            if plan is not None and plan.action != "live":
                plan.wait()
                return _served(call, plan, f)
            if call is None:
                return f(*args, **kwargs)
            attrs = plan.attributes if plan is not None else None
            try:
                result = f(*args, **kwargs)
            except BaseException as e:
                call.end(exc=e, attributes=attrs)
                raise
            call.end(result, capture=capture_output, attributes=attrs)
            return result

        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate
