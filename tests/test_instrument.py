import asyncio

import pytest

from agent_doctor import instrument, trace_run
from agent_doctor.schema import SpanKind, SpanStatus

from .fakes import (
    FakeAnthropicClient,
    FakeAsyncHFClient,
    FakeHFClient,
    FakeOpenAIClient,
    anthropic_response,
    chat_response,
    stream_chunks,
)


def only_run(store):
    rows = store.list_runs()
    assert len(rows) == 1
    return store.get_run(rows[0].run_id)


def test_hf_call_recorded(store):
    client = instrument(FakeHFClient(
        [chat_response("hello", 120, 7, model="Qwen/Qwen2.5-7B-Instruct",
                       tool_calls=["search"])], model="Qwen/Qwen2.5-7B-Instruct", provider="together"))

    @trace_run(store=store)
    def agent(msg: str):
        return client.chat_completion([{"role": "user", "content": msg}], max_tokens=50)

    resp = agent("hi there")
    assert resp["choices"][0]["message"]["content"] == "hello"  # response unchanged
    llm = only_run(store).llm_spans[0]
    assert llm.model == "Qwen/Qwen2.5-7B-Instruct" and llm.provider == "together"
    assert (llm.input_tokens, llm.output_tokens) == (120, 7)
    assert llm.output_text == "hello"
    assert llm.input_messages == [{"role": "user", "content": "hi there"}]
    assert llm.attributes["tool_calls"] == ["search"]
    assert llm.attributes["finish_reason"] == "stop"


def test_hf_openai_style_proxy_is_recorded(store):
    client = instrument(FakeHFClient([chat_response("x", 5, 1)]))

    @trace_run(store=store)
    def agent(msg: str):
        client.chat.completions.create(messages=[{"role": "user", "content": msg}])

    agent("hi")
    assert len(only_run(store).llm_spans) == 1


def test_default_provider_for_hf(store):
    client = instrument(FakeHFClient([chat_response()], provider="auto"))

    @trace_run(store=store)
    def agent(msg: str):
        client.chat_completion([{"role": "user", "content": msg}])

    agent("x")
    assert only_run(store).llm_spans[0].provider == "huggingface"


def test_stream_passes_chunks_and_records_usage(store):
    chunks = stream_chunks(["Hel", "lo"], prompt=30, completion=2)
    client = instrument(FakeHFClient([chunks]))

    @trace_run(store=store)
    def agent(msg: str):
        return [c for c in client.chat_completion([{"role": "user", "content": msg}], stream=True)]

    got = agent("x")
    assert got == chunks
    llm = only_run(store).llm_spans[0]
    assert llm.output_text == "Hello"
    assert (llm.input_tokens, llm.output_tokens) == (30, 2)
    assert llm.time_to_first_token_ms is not None
    assert llm.attributes["stream"] is True and "usage_missing" not in llm.attributes


def test_stream_without_usage_is_flagged(store):
    client = instrument(FakeHFClient([stream_chunks(["a"], with_usage=False)]))

    @trace_run(store=store)
    def agent(msg: str):
        list(client.chat_completion([{"role": "user", "content": msg}], stream=True))

    agent("x")
    assert only_run(store).llm_spans[0].attributes["usage_missing"] is True


def test_stream_closed_early_still_recorded(store):
    client = instrument(FakeHFClient([stream_chunks(["a", "b", "c"])]))

    @trace_run(store=store)
    def agent(msg: str):
        with client.chat_completion([{"role": "user", "content": msg}], stream=True) as s:
            next(s)

    agent("x")
    assert only_run(store).llm_spans[0].output_text == "a"


def test_call_outside_run_saves_one_call_run(isolated_home):
    from agent_doctor.store import get_default_store

    client = instrument(FakeHFClient([chat_response("pong", 9, 1)]), agent="pinger")
    client.chat_completion([{"role": "user", "content": "ping"}])
    store = get_default_store()
    run = store.get_run(store.list_runs()[0].run_id)
    assert run.agent_name == "pinger"
    assert run.task_input == "ping" and run.final_output == "pong"
    assert [s.kind for s in run.ordered_spans] == [SpanKind.AGENT_STEP, SpanKind.LLM]


def test_client_error_recorded_and_raised(store):
    client = instrument(FakeHFClient([ConnectionError("rate limited")]))

    @trace_run(store=store)
    def agent(msg: str):
        try:
            client.chat_completion([{"role": "user", "content": msg}])
        except ConnectionError:
            return "fallback"

    assert agent("x") == "fallback"
    llm = only_run(store).llm_spans[0]
    assert llm.status is SpanStatus.ERROR and llm.error_type == "ConnectionError"


def test_openai_client_with_cached_tokens(store):
    client = instrument(FakeOpenAIClient([chat_response("ok", 2000, 10, model="gpt-5.6-luna",
                                                        cached=1500)]))

    @trace_run(store=store)
    def agent(msg: str):
        client.chat.completions.create(model="gpt-5.6-luna",
                                       messages=[{"role": "user", "content": msg}])

    agent("x")
    llm = only_run(store).llm_spans[0]
    assert llm.provider == "openai" and llm.cached_tokens == 1500 and llm.input_tokens == 2000


def test_anthropic_client_counts_cache_reads(store):
    client = instrument(FakeAnthropicClient([anthropic_response(
        "hi", input_tokens=100, output_tokens=20, cache_read=900, cache_write=50, tool="lookup")]))

    @trace_run(store=store)
    def agent(msg: str):
        client.messages.create(model="claude-haiku-5-5", max_tokens=100,
                               messages=[{"role": "user", "content": msg}])

    agent("x")
    llm = only_run(store).llm_spans[0]
    assert llm.provider == "anthropic"
    assert llm.input_tokens == 1050 and llm.cached_tokens == 900 and llm.output_tokens == 20
    assert llm.attributes["cache_creation_tokens"] == 50
    assert llm.attributes["tool_calls"] == ["lookup"]


def test_async_client_and_stream(store):
    client = instrument(FakeAsyncHFClient([chat_response("one", 10, 1),
                                           stream_chunks(["t", "wo"], prompt=12, completion=2)]))

    @trace_run(store=store)
    async def agent(msg: str):
        await client.chat_completion([{"role": "user", "content": msg}])
        stream = await client.chat_completion([{"role": "user", "content": msg}], stream=True)
        return [c async for c in stream]

    asyncio.run(agent("x"))
    llms = only_run(store).llm_spans
    assert [s.output_text for s in llms] == ["one", "two"]
    assert llms[1].input_tokens == 12


def test_instrument_twice_records_once(store):
    client = instrument(instrument(FakeHFClient([chat_response()])))

    @trace_run(store=store)
    def agent(msg: str):
        client.chat_completion([{"role": "user", "content": msg}])

    agent("x")
    assert len(only_run(store).llm_spans) == 1


def test_capture_messages_off(store):
    client = instrument(FakeHFClient([chat_response("secret")]), capture_messages=False)

    @trace_run(store=store)
    def agent(msg: str):
        client.chat_completion([{"role": "user", "content": msg}])

    agent("x")
    llm = only_run(store).llm_spans[0]
    assert llm.output_text is None and llm.input_messages is None and llm.input_tokens == 10


def test_unsupported_client():
    with pytest.raises(TypeError, match="instrument"):
        instrument(object())


def test_weird_response_never_breaks(store):
    client = instrument(FakeHFClient([object()]))

    @trace_run(store=store)
    def agent(msg: str):
        return client.chat_completion([{"role": "user", "content": msg}])

    agent("x")  # no exception
    assert only_run(store).llm_spans[0].attributes.get("usage_missing") is True
