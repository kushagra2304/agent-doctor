"""Importers (OpenTelemetry, Langfuse, LangSmith) and redaction."""

import json

import pytest
from typer.testing import CliRunner

from agent_doctor.analyzers import analyze_run
from agent_doctor.cli import app
from agent_doctor.ingest import Redactor, detect, load_runs, parse_runs
from agent_doctor.ingest.common import iso_ms, normalize_messages
from agent_doctor.schema import SpanKind

NS = 1_760_000_000_000_000_000


def kv(d):
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            out.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, int):
            out.append({"key": k, "value": {"intValue": str(v)}})
        else:
            out.append({"key": k, "value": {"stringValue": v}})
    return out


def otel_span(sid, parent, start_ms, end_ms, attrs, name="span", error=False):
    s = {"traceId": "t1", "spanId": sid, "name": name, "startTimeUnixNano": str(NS + start_ms * 1_000_000),
         "endTimeUnixNano": str(NS + end_ms * 1_000_000), "attributes": kv(attrs)}
    if parent:
        s["parentSpanId"] = parent
    if error:
        s["status"] = {"code": 2, "message": "timed out"}
    return s


TRACK = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "track",
         "gen_ai.tool.call.arguments": '{"order_id": "1234"}'}
OTEL = {"resourceSpans": [{"resource": {"attributes": kv({"service.name": "shop-bot"})}, "scopeSpans": [{
    "spans": [
        otel_span("root", None, 0, 2000, {"gen_ai.operation.name": "invoke_agent",
                                           "gen_ai.agent.name": "support-bot"}, name="invoke_agent"),
        otel_span("l1", "root", 10, 400, {
            "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o",
            "gen_ai.provider.name": "openai",
            "gen_ai.usage.input_tokens": 1200, "gen_ai.usage.output_tokens": 3,
            "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [
                {"type": "text", "content": "Where is order 1234?"}]}]),
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [
                {"type": "text", "content": "order_status"}]}]),
        }),
        otel_span("t1", "root", 410, 900, {**TRACK}, error=True),
        otel_span("t2", "root", 901, 1300, {**TRACK, "gen_ai.tool.call.result": '{"eta": "Friday"}'}),
        otel_span("l2", "root", 1310, 1990, {"gen_ai.request.model": "gpt-4o",
                                              "gen_ai.usage.prompt_tokens": 1500,
                                              "gen_ai.usage.completion_tokens": 20,
                                              "gen_ai.completion": "It arrives Friday."}),
        otel_span("orphan", "missing-parent", 5, 6, {}, name="http"),
    ]}]}]}


def test_otel_import():
    assert detect(OTEL) == "otel"
    (run,) = parse_runs(OTEL)
    assert run.run_id == "t1" and run.agent_name == "support-bot"
    assert run.task_input == "Where is order 1234?" and run.final_output == "It arrives Friday."
    llm = run.llm_spans
    assert [s.model for s in llm] == ["gpt-4o", "gpt-4o"] and llm[0].provider == "openai"
    assert (llm[0].input_tokens, llm[1].input_tokens) == (1200, 1500)
    t_fail, t_ok = run.tool_spans
    assert t_fail.is_error and t_fail.error_message == "timed out" and t_ok.retry_of == t_fail.span_id
    assert t_ok.tool_output == {"eta": "Friday"} and t_ok.tool_args == {"order_id": "1234"}
    assert run.get_span("orphan").parent_id is None
    assert analyze_run(run).run_id == "t1"  # analyzers run on imported traces


def test_otel_json_lines(tmp_path):
    path = tmp_path / "traces.jsonl"
    second = json.dumps(OTEL).replace('"traceId": "t1"', '"traceId": "t2"')
    path.write_text(json.dumps(OTEL) + "\n" + second + "\n")
    assert sorted(r.run_id for r in load_runs(path)) == ["t1", "t2"]


LANGFUSE = {"data": [
    {"id": "g1", "traceId": "lf1", "type": "GENERATION", "name": "router", "model": "claude-haiku-5-5",
     "startTime": "2026-10-01T10:00:00.000Z", "endTime": "2026-10-01T10:00:00.400Z",
     "input": [{"role": "system", "content": "Route."}, {"role": "user", "content": "refund?"}],
     "output": {"role": "assistant", "content": "refund"},
     "usageDetails": {"input": 800, "output": 2, "cache_read_input_tokens": 600}},
    {"id": "tool1", "traceId": "lf1", "type": "TOOL", "name": "search_kb", "parentObservationId": "g1",
     "startTime": "2026-10-01T10:00:00.500Z", "endTime": "2026-10-01T10:00:00.700Z",
     "input": {"query": "refund policy"}, "output": {"days": 30}},
    {"id": "s1", "traceId": "lf1", "type": "SPAN", "name": "lookup", "level": "ERROR",
     "statusMessage": "db down", "startTime": "2026-10-01T10:00:00.710123456Z"},
]}


def test_langfuse_import():
    assert detect(LANGFUSE) == "langfuse"
    (run,) = parse_runs(LANGFUSE)
    llm = run.llm_spans[0]
    assert llm.model == "claude-haiku-5-5" and llm.cached_tokens == 600 and llm.input_tokens == 1400
    assert llm.output_text == "refund" and run.task_input == "refund?"
    tool = run.tool_spans[0]
    assert tool.tool_args == {"query": "refund policy"} and tool.tool_output == {"days": 30}
    step = run.get_span("s1")
    assert step.is_error and step.error_message == "db down" and run.status.value == "error"


LANGSMITH = [{
    "id": "root", "trace_id": "ls1", "parent_run_id": None, "name": "SupportAgent", "run_type": "chain",
    "start_time": "2026-10-01T10:00:00", "end_time": "2026-10-01T10:00:02",
    "inputs": {"question": "Is SKU-1 in stock?"}, "outputs": {"output": "Yes, 3 left."},
    "child_runs": [
        {"id": "llm1", "trace_id": "ls1", "parent_run_id": "root", "name": "ChatOpenAI", "run_type": "llm",
         "start_time": "2026-10-01T10:00:00.100", "end_time": "2026-10-01T10:00:00.900",
         "extra": {"metadata": {"ls_model_name": "gpt-4o-mini", "ls_provider": "openai"}},
         "inputs": {"messages": [[{"lc": 1, "type": "constructor", "id": ["langchain", "HumanMessage"],
                                   "kwargs": {"content": "Is SKU-1 in stock?"}}]]},
         "outputs": {"generations": [[{"text": "check_stock"}]]},
         "prompt_tokens": 300, "completion_tokens": 4},
        {"id": "tool1", "trace_id": "ls1", "parent_run_id": "root", "name": "check_stock", "run_type": "tool",
         "start_time": "2026-10-01T10:00:01", "end_time": "2026-10-01T10:00:01.500",
         "inputs": {"sku": "SKU-1"}, "outputs": {"output": {"in_stock": 3}}},
        {"id": "tool2", "trace_id": "ls1", "parent_run_id": "root", "name": "notify", "run_type": "tool",
         "start_time": "2026-10-01T10:00:01.600", "end_time": "2026-10-01T10:00:01.700",
         "inputs": {"input": "x"}, "error": "TimeoutError: slow"},
    ]}]


def test_langsmith_import():
    assert detect(LANGSMITH) == "langsmith"
    (run,) = parse_runs({"runs": LANGSMITH})
    assert run.agent_name == "SupportAgent" and run.final_output == "Yes, 3 left."
    llm = run.llm_spans[0]
    assert (llm.model, llm.provider, llm.input_tokens, llm.output_tokens) == ("gpt-4o-mini", "openai", 300, 4)
    assert llm.input_messages == [{"role": "user", "content": "Is SKU-1 in stock?"}]
    assert llm.output_text == "check_stock"
    stock, notify = run.tool_spans
    assert stock.tool_output == {"in_stock": 3} and notify.error_type == "TimeoutError"


def test_native_and_unknown_formats(sample_dict):
    assert detect(sample_dict) == "native" and parse_runs([sample_dict])[0].run_id == "run_482"
    with pytest.raises(ValueError, match="unrecognized"):
        detect({"hello": 1})
    with pytest.raises(ValueError, match="unknown format"):
        parse_runs(sample_dict, fmt="nope")  # type: ignore[arg-type]


def test_helpers():
    assert iso_ms("2026-10-01T00:00:00Z") == iso_ms(1790812800) == iso_ms(1790812800000) == 1790812800000
    assert iso_ms("2026-10-01T00:00:00.123456789+00:00") == 1790812800123
    assert normalize_messages("hi") == [{"role": "user", "content": "hi"}]
    assert normalize_messages({"messages": [{"type": "ai", "content": [{"text": "ok"}]}]}) == [
        {"role": "assistant", "content": "ok"}]


# -- redaction ----------------------------------------------------------------------------
def test_redactor_text():
    r = Redactor()
    text = ("Mail asha.rao@example.com or call +91 98765 43210. Card 4111 1111 1111 1111, "
            "order 1234-5678-9012-3456. Key sk-proj-abcdefghijklmnopqrstuvwx, "
            "token Bearer abcdefghijklmnop1234, "
            "AWS AKIAABCDEFGHIJKLMNOP, host 10.0.0.12, ticket 5678")
    out = r.text(text)
    for secret in ["asha.rao", "98765", "4111 1111", "sk-proj", "abcdefghijklmnop1234", "AKIA", "10.0.0.12"]:
        assert secret not in out, secret
    assert "[EMAIL]" in out and "[CARD]" in out and "[API_KEY]" in out and "[PHONE]" in out
    assert "1234-5678-9012-3456" in out  # fails the Luhn check, so it is not a card
    assert "order" in out and "ticket 5678" in out  # short numbers survive
    assert Redactor(extra=[("ORDER", r"ORD-\d+")]).text("see ORD-77") == "see [ORDER]"


def test_redact_run_keeps_args_hash(sample_run):
    run = sample_run.model_copy(deep=True)
    tool = run.tool_spans[0]
    tool.tool_args = {"email": "a@b.co", "password": "hunter2"}
    tool.args_hash = "keep-me"
    run.task_input = "I am a@b.co"
    clean = Redactor().run(run)
    t = clean.get_span(tool.span_id)
    assert t.tool_args == {"email": "[EMAIL]", "password": "[REDACTED]"} and t.args_hash == "keep-me"
    assert clean.task_input == "I am [EMAIL]" and run.task_input == "I am a@b.co"


def test_recording_with_redaction(store, monkeypatch):
    from agent_doctor import trace_run, trace_tool

    monkeypatch.setenv("AGENT_DOCTOR_REDACT", "1")

    @trace_tool
    def lookup(email: str) -> dict:
        return {"owner": email}

    @trace_run(agent="bot", store=store)
    def handle(q: str) -> str:
        return str(lookup("asha@example.com"))

    handle("hello asha@example.com")
    saved = store.get_run(store.resolve_run_id("last"))
    assert "asha@example.com" not in saved.model_dump_json()
    assert saved.tool_spans[0].tool_args == {"email": "[EMAIL]"}


def test_cli_import_formats(tmp_path):
    path = tmp_path / "lf.json"
    path.write_text(json.dumps(LANGFUSE))
    db = str(tmp_path / "t.db")
    runner = CliRunner()
    r = runner.invoke(app, ["--db", db, "import", str(path), "--redact"])
    assert r.exit_code == 0 and "imported lf1" in r.stdout
    r = runner.invoke(app, ["--db", db, "import", str(path), "--format", "otel"])
    assert r.exit_code == 0 and "0 run(s)" in r.stdout  # nothing in it looks like OTLP
    bad = tmp_path / "bad.json"
    bad.write_text('{"nope": 1}')
    assert runner.invoke(app, ["--db", db, "import", str(bad)]).exit_code == 1
    assert any(s.kind is SpanKind.TOOL for s in load_runs(path)[0].spans)
