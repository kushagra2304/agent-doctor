import asyncio

import pytest

from agent_doctor import current_run, instrument, record_run, span, trace_run, trace_tool
from agent_doctor.schema import SpanKind, SpanStatus
from agent_doctor.store import TraceStore

from .fakes import FakeHFClient, chat_response


def only_run(store: TraceStore):
    rows = store.list_runs()
    assert len(rows) == 1, rows
    return store.get_run(rows[0].run_id)


def test_trace_run_records_tools_and_llm(store):
    client = instrument(FakeHFClient([chat_response("plan", 50, 5), chat_response("done", 80, 9)]))

    @trace_tool
    def search(query: str, limit: int = 3):
        return {"hits": [query] * limit}

    @trace_run(agent="bot", agent_version="1.0", prompt_version="v2", tags={"env": "test"},
               store=store)
    def agent(message: str) -> str:
        client.chat_completion([{"role": "user", "content": message}])
        search("cats")
        return client.chat_completion([{"role": "user", "content": "go"}])["choices"][0][
            "message"]["content"]

    assert agent("find cats") == "done"
    run = only_run(store)
    assert run.agent_name == "bot" and run.agent_version == "1.0"
    assert run.task_input == "find cats" and run.final_output == "done"
    assert run.status is SpanStatus.OK and run.tags == {"env": "test"}
    kinds = [s.kind for s in run.ordered_spans]
    assert kinds == [SpanKind.AGENT_STEP, SpanKind.LLM, SpanKind.TOOL, SpanKind.LLM]
    root = run.ordered_spans[0]
    assert all(s.parent_id == root.span_id for s in run.ordered_spans[1:])
    tool = run.tool_spans[0]
    assert tool.tool_args == {"query": "cats", "limit": 3}  # default filled in
    assert tool.tool_output == {"hits": ["cats"] * 3}
    assert all(s.prompt_version == "v2" for s in run.llm_spans)
    assert run.total_input_tokens == 130


def test_bare_decorator_and_input_arg(store):
    @trace_run(store=store, input_arg="question")
    def ask(user_id: str, question: str):
        return {"answer": 42}

    ask("u1", "meaning of life?")
    run = only_run(store)
    assert run.agent_name == "ask"
    assert run.task_input == "meaning of life?"
    assert run.final_output == '{"answer": 42}'


def test_bare_trace_run_without_parentheses(isolated_home):
    @trace_run
    def plain(x: str):
        return x.upper()

    assert plain("hi") == "HI"
    from agent_doctor.store import get_default_store

    assert get_default_store().list_runs()[0].agent_name == "plain"


def test_run_error_is_recorded_and_raised(store):
    @trace_run(store=store)
    def broken(msg: str):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        broken("x")
    run = only_run(store)
    assert run.status is SpanStatus.ERROR
    assert run.ordered_spans[0].error_type == "ValueError"


def test_tool_error_and_retry_link(store):
    calls = {"n": 0}

    @trace_tool
    def flaky(order_id: str):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow carrier")
        return "ok"

    @trace_run(store=store)
    def agent(msg: str):
        for _ in range(5):
            try:
                return flaky(order_id="1234")
            except TimeoutError:
                continue

    assert agent("x") == "ok"
    tools = only_run(store).tool_spans
    assert [t.status for t in tools] == [SpanStatus.ERROR, SpanStatus.ERROR, SpanStatus.OK]
    assert tools[0].error_type == "TimeoutError" and tools[0].error_message == "slow carrier"
    assert tools[0].retry_of is None
    assert tools[1].retry_of == tools[0].span_id
    assert tools[2].retry_of == tools[1].span_id
    assert tools[2].tool_output == "ok"


def test_side_effects_name_and_no_output(store):
    @trace_tool(name="email.send", side_effects=True, capture_output=False)
    def send(to: str):
        return "secret receipt"

    @trace_run(store=store)
    def agent(msg: str):
        send("a@b.c")

    agent("x")
    tool = only_run(store).tool_spans[0]
    assert tool.name == "email.send" and tool.side_effects is True
    assert tool.tool_output is None


def test_method_tool_drops_self(store):
    class Tools:
        @trace_tool
        def lookup(self, key: str):
            return key

    @trace_run(store=store)
    def agent(msg: str):
        Tools().lookup("k")

    agent("x")
    assert only_run(store).tool_spans[0].tool_args == {"key": "k"}


def test_tool_outside_run_is_not_recorded(store):
    @trace_tool
    def t(x):
        return x * 2

    assert t(2) == 4
    assert current_run() is None
    assert store.count_runs() == 0


def test_nested_trace_run_becomes_step(store):
    @trace_run(agent="sub-agent", store=store)
    def sub(task: str):
        return "sub done"

    @trace_run(agent="main", store=store)
    def main(task: str):
        return sub("inner")

    main("outer")
    run = only_run(store)
    assert run.agent_name == "main"
    names = [s.name for s in run.ordered_spans]
    assert names == ["main", "sub-agent"]


def test_span_context_nests_children(store):
    @trace_tool
    def fetch(doc: str):
        return doc

    with record_run("rag-bot", task_input="q", store=store) as rec:
        with span("retrieve", kind="retrieval", index="docs") as s:
            fetch("a")
            s.output = ["a"]
        rec.final_output = "answer"

    run = only_run(store)
    retrieve = next(x for x in run.spans if x.name == "retrieve")
    assert retrieve.kind is SpanKind.RETRIEVAL
    assert retrieve.attributes == {"index": "docs"}
    assert run.tool_spans[0].parent_id == retrieve.span_id
    assert run.final_output == "answer"


def test_span_outside_run_is_noop():
    with span("nothing") as s:
        s.output = 1


def test_async_runs_are_isolated(store):
    @trace_tool
    async def lookup(key: str):
        await asyncio.sleep(0.01)
        return key

    @trace_run(agent="async-bot", store=store)
    async def agent(msg: str):
        await lookup(msg)
        await lookup(msg + "!")
        return msg

    async def main():
        return await asyncio.gather(agent("a"), agent("b"))

    assert asyncio.run(main()) == ["a", "b"]
    runs = [store.get_run(r.run_id) for r in store.list_runs()]
    assert len(runs) == 2
    for run in runs:
        args = [t.tool_args["key"] for t in run.tool_spans]
        assert args == [run.task_input, run.task_input + "!"]


def test_disabled_records_nothing(store, monkeypatch):
    monkeypatch.setenv("AGENT_DOCTOR_DISABLED", "1")

    @trace_run(store=store)
    def agent(msg: str):
        return msg

    assert agent("x") == "x"
    assert store.count_runs() == 0


def test_save_failure_never_breaks_agent():
    class BrokenStore:
        def save_run(self, run):
            raise OSError("disk full")

    @trace_run(store=BrokenStore())
    def agent(msg: str):
        return "still works"

    with pytest.warns(RuntimeWarning, match="could not save run"):
        assert agent("x") == "still works"
