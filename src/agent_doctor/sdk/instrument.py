"""``instrument(client)``: record every LLM call a client makes.

Supported clients:

- Hugging Face ``InferenceClient`` / ``AsyncInferenceClient`` (``chat_completion`` and
  ``chat.completions.create``, which forwards to it)
- OpenAI-compatible clients (``client.chat.completions.create``), sync or async
- Anthropic clients (``client.messages.create``), sync or async

Streaming responses are wrapped so tokens and time-to-first-token are captured when
the stream ends. Calls made outside a run are each saved as a one-call run.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from agent_doctor import config
from agent_doctor._util import new_id, pick, to_jsonable, warn_once
from agent_doctor.schema import SpanKind
from agent_doctor.sdk.recorder import (
    RunRecorder,
    Timer,
    current_parent_id,
    current_replay,
    current_run,
    error_fields,
)

HF, OPENAI, ANTHROPIC = "huggingface", "openai", "anthropic"
_FLAG = "_agent_doctor_instrumented"


def _detect(client: Any) -> str | None:
    if callable(getattr(client, "chat_completion", None)):
        return HF
    completions = pick(client, "chat.completions")
    if completions is not None and callable(getattr(completions, "create", None)):
        return OPENAI
    messages = getattr(client, "messages", None)
    if messages is not None and callable(getattr(messages, "create", None)):
        return ANTHROPIC
    return None


def instrument(
    client: Any,
    *,
    agent: str | None = None,
    provider: str | None = None,
    capture_messages: bool = True,
) -> Any:
    """Wrap ``client`` in place so its LLM calls are recorded; returns the same client.

    ``agent`` names one-call runs made outside ``@trace_run``. ``provider`` is stored on
    each span (used for pricing); by default it is taken from the client.
    ``capture_messages=False`` stores token counts but not prompt or response text.
    """
    if getattr(client, _FLAG, False):
        return client
    kind = _detect(client)
    if kind is None:
        raise TypeError(
            "instrument() supports Hugging Face InferenceClient, OpenAI-compatible and "
            f"Anthropic clients; got {type(client).__name__}"
        )
    if kind == HF:
        target, attr = client, "chat_completion"
        hf_provider = getattr(client, "provider", None)
        prov = provider or (hf_provider if hf_provider not in (None, "auto") else HF)
    elif kind == OPENAI:
        target, attr = pick(client, "chat.completions"), "create"
        prov = provider or OPENAI
    else:
        target, attr = client.messages, "create"
        prov = provider or ANTHROPIC

    settings = dict(
        kind=kind, provider=prov, agent=agent, capture=capture_messages,
        default_model=getattr(client, "model", None),
    )
    setattr(target, attr, _wrap(getattr(target, attr), settings))
    try:
        setattr(client, _FLAG, True)
    except Exception:
        pass
    return client


def _swap_model(settings: dict, kwargs: dict) -> tuple[dict, str | None]:
    """Apply the active replay's model_map. Returns the new kwargs and the replaced model."""
    session = current_replay()
    model_map = getattr(session, "model_map", None)
    if not model_map:
        return kwargs, None
    model = kwargs.get("model") or settings["default_model"]
    new = model_map.get(model)
    if not new or new == model:
        return kwargs, None
    return {**kwargs, "model": new}, model


def _begin(settings: dict, args: tuple, kwargs: dict) -> tuple[_LLMCall | None, dict]:
    kwargs, replaced = _swap_model(settings, kwargs)
    call = _LLMCall.begin(settings, args, kwargs)
    if call is not None and replaced:
        call.extra["replaced_model"] = replaced
    return call, kwargs


def _wrap(original: Any, settings: dict) -> Any:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def awrapper(*args, **kwargs):
            call, kwargs = _begin(settings, args, kwargs)
            if call is None:
                return await original(*args, **kwargs)
            try:
                response = await original(*args, **kwargs)
            except BaseException as e:
                call.fail(e)
                raise
            if call.stream:
                return _AsyncStream(response, call)
            call.finish(response)
            return response

        return awrapper

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        call, kwargs = _begin(settings, args, kwargs)
        if call is None:
            return original(*args, **kwargs)
        try:
            response = original(*args, **kwargs)
        except BaseException as e:
            call.fail(e)
            raise
        if call.stream:
            return _SyncStream(response, call)
        call.finish(response)
        return response

    return wrapper


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if pick(m, "role") == "user":
            content = pick(m, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(pick(p, "text", "")) for p in content).strip()
    return ""


class _LLMCall:
    """State of one LLM call while it runs."""

    def __init__(self, settings: dict, args: tuple, kwargs: dict):
        self.kind = settings["kind"]
        self.provider = settings["provider"]
        self.capture = settings["capture"]
        messages = kwargs.get("messages")
        if messages is None and args and isinstance(args[0], list):
            messages = args[0]  # HF chat_completion(messages, ...)
        self.messages = messages
        self.model = kwargs.get("model") or settings["default_model"]
        self.stream = bool(kwargs.get("stream"))
        rec = current_run()
        self.implicit = rec is None
        if rec is None:
            rec = RunRecorder(
                settings["agent"] or "unnamed-agent", task_input=_last_user_text(messages)
            ).start()
        self.rec = rec
        self.parent = rec.root_span_id if self.implicit else current_parent_id()
        self.span_id = new_id("sp")
        self.timer = Timer()
        # filled from the response
        self.input_tokens = self.output_tokens = self.cached_tokens = 0
        self.usage_seen = False
        self.text_parts: list[str] = []
        self.tool_calls: list[str] = []
        self.finish_reason: str | None = None
        self.ttft_ms: int | None = None
        self.extra: dict[str, Any] = {}
        self.done = False

    @classmethod
    def begin(cls, settings: dict, args: tuple, kwargs: dict) -> _LLMCall | None:
        if not config.recording_enabled():
            return None
        try:
            return cls(settings, args, kwargs)
        except Exception as e:
            warn_once(f"llm-begin:{type(e).__name__}", f"could not record an LLM call: {e}")
            return None

    # -- parsing ----------------------------------------------------------------
    def _usage_openai(self, usage: Any) -> None:
        if usage is None:
            return
        self.usage_seen = True
        self.input_tokens = int(pick(usage, "prompt_tokens", 0) or 0)
        self.output_tokens = int(pick(usage, "completion_tokens", 0) or 0)
        self.cached_tokens = int(pick(usage, "prompt_tokens_details.cached_tokens", 0) or 0)

    def _usage_anthropic(self, usage: Any) -> None:
        if usage is None:
            return
        self.usage_seen = True
        base = int(pick(usage, "input_tokens", 0) or 0)
        read = int(pick(usage, "cache_read_input_tokens", 0) or 0)
        write = int(pick(usage, "cache_creation_input_tokens", 0) or 0)
        # Anthropic's input_tokens excludes cache reads and writes; ours includes them.
        self.input_tokens = base + read + write
        self.cached_tokens = read
        if write:
            self.extra["cache_creation_tokens"] = write
        out = pick(usage, "output_tokens")
        if out is not None:
            self.output_tokens = int(out)

    def parse_response(self, resp: Any) -> None:
        self.model = pick(resp, "model") or self.model
        if self.kind == ANTHROPIC:
            self._usage_anthropic(pick(resp, "usage"))
            for block in pick(resp, "content", []) or []:
                btype = pick(block, "type")
                if btype == "text":
                    self.text_parts.append(str(pick(block, "text", "")))
                elif btype == "tool_use":
                    self.tool_calls.append(str(pick(block, "name", "?")))
            self.finish_reason = pick(resp, "stop_reason")
            return
        self._usage_openai(pick(resp, "usage"))
        message = pick(resp, "choices.0.message")
        content = pick(message, "content")
        if content:
            self.text_parts.append(str(content))
        for tc in pick(message, "tool_calls", []) or []:
            self.tool_calls.append(str(pick(tc, "function.name", "?")))
        self.finish_reason = pick(resp, "choices.0.finish_reason")

    def on_chunk(self, chunk: Any) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = self.timer.elapsed_ms()
        if self.kind == ANTHROPIC:
            ctype = pick(chunk, "type")
            if ctype == "message_start":
                self.model = pick(chunk, "message.model") or self.model
                self._usage_anthropic(pick(chunk, "message.usage"))
            elif ctype == "content_block_delta":
                text = pick(chunk, "delta.text")
                if text:
                    self.text_parts.append(str(text))
            elif ctype == "content_block_start" and pick(chunk, "content_block.type") == "tool_use":
                self.tool_calls.append(str(pick(chunk, "content_block.name", "?")))
            elif ctype == "message_delta":
                out = pick(chunk, "usage.output_tokens")
                if out is not None:
                    self.output_tokens = int(out)
                self.finish_reason = pick(chunk, "delta.stop_reason") or self.finish_reason
            return
        self.model = pick(chunk, "model") or self.model
        text = pick(chunk, "choices.0.delta.content")
        if text:
            self.text_parts.append(str(text))
        for tc in pick(chunk, "choices.0.delta.tool_calls", []) or []:
            fname = pick(tc, "function.name")
            if fname:
                self.tool_calls.append(str(fname))
        self.finish_reason = pick(chunk, "choices.0.finish_reason") or self.finish_reason
        usage = pick(chunk, "usage")
        if usage is not None:
            self._usage_openai(usage)

    # -- recording --------------------------------------------------------------
    def _record(self, exc: BaseException | None = None) -> None:
        if self.done:
            return
        self.done = True
        try:
            end_ms = self.timer.end_ms()
            ttft = self.ttft_ms
            if ttft is not None:
                ttft = min(ttft, end_ms - self.timer.start_ms)
            attrs: dict[str, Any] = dict(self.extra)
            if self.tool_calls:
                attrs["tool_calls"] = self.tool_calls
            if self.finish_reason:
                attrs["finish_reason"] = str(self.finish_reason)
            if self.stream:
                attrs["stream"] = True
            if not self.usage_seen and exc is None:
                attrs["usage_missing"] = True
            text = "".join(self.text_parts) or None
            model = str(self.model) if self.model else "unknown-model"
            self.rec.add_span(
                span_id=self.span_id, parent_id=self.parent, kind=SpanKind.LLM, name=model,
                model=model, provider=self.provider, start_ms=self.timer.start_ms, end_ms=end_ms,
                input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                cached_tokens=min(self.cached_tokens, self.input_tokens),
                time_to_first_token_ms=ttft,
                input_messages=to_jsonable(self.messages) if self.capture and self.messages else None,
                output_text=to_jsonable(text) if self.capture else None,
                attributes=attrs, **error_fields(exc),
            )
            if self.implicit:
                self.rec.final_output = text if self.capture else None
                self.rec.finish(exc)
        except Exception as e:
            warn_once(f"llm-record:{type(e).__name__}", f"could not record an LLM call: {e}")

    def finish(self, response: Any) -> None:
        try:
            self.parse_response(response)
        except Exception as e:
            warn_once(f"llm-parse:{type(e).__name__}", f"could not read an LLM response: {e}")
        self._record()

    def fail(self, exc: BaseException) -> None:
        self._record(exc)

    def chunk(self, chunk: Any) -> None:
        try:
            self.on_chunk(chunk)
        except Exception as e:
            warn_once(f"llm-chunk:{type(e).__name__}", f"could not read a stream chunk: {e}")


class _SyncStream:
    """Passes a stream through unchanged, recording the call when it ends or closes."""

    def __init__(self, inner: Any, call: _LLMCall):
        self._inner = inner
        self._it = iter(inner)
        self._call = call

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._it)
        except StopIteration:
            self._call._record()
            raise
        except BaseException as e:
            self._call._record(e)
            raise
        self._call.chunk(chunk)
        return chunk

    def close(self) -> None:
        self._call._record()
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncStream:
    def __init__(self, inner: Any, call: _LLMCall):
        self._inner = inner
        self._it = inner.__aiter__()
        self._call = call

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._call._record()
            raise
        except BaseException as e:
            self._call._record(e)
            raise
        self._call.chunk(chunk)
        return chunk

    async def aclose(self) -> None:
        self._call._record()
        for name in ("aclose", "close"):
            fn = getattr(self._inner, name, None)
            if callable(fn):
                result = fn()
                if inspect.isawaitable(result):
                    await result
                return

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
