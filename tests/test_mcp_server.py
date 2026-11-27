"""MCP server tests, through a real MCP client connected in-process."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from agent_doctor import core  # noqa: E402
from agent_doctor.mcp_server import create_server  # noqa: E402
from agent_doctor.store import TraceStore  # noqa: E402

BAD_AGENT_RUNS = Path(__file__).parent / "fixtures" / "bad_agent" / "runs.json"
TOOLS = {"list_runs", "analyze_run", "explain_cost", "show_run", "get_step", "scan_runs", "compare_runs",
         "recommend_fixes", "search_playbooks", "list_evals", "get_eval"}


@pytest.fixture
def prices_file(tmp_path) -> Path:
    p = tmp_path / "prices.yaml"
    p.write_text('models:\n  - {match: "qwen/*", input: 1.0, output: 2.0}\n')
    return p


@pytest.fixture
def loaded(store, sample_path):
    store.import_json_file(sample_path)
    store.import_json_file(BAD_AGENT_RUNS)
    return store


def call(server, tool, **args):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(go())


def ok(server, tool, **args) -> dict:
    result = call(server, tool, **args)
    assert not result.is_error, result.content
    return result.structured_content


def error_text(server, tool, **args) -> str:
    result = call(server, tool, **args)
    assert result.is_error
    return result.content[0].text


def test_tools_are_listed_and_read_only(store):
    async def go():
        async with Client(create_server(store)) as client:
            return (await client.list_tools()).tools
    tools = asyncio.run(go())
    assert {t.name for t in tools} == TOOLS
    assert all(t.annotations.read_only_hint and t.description for t in tools)
    run_id = next(t for t in tools if t.name == "analyze_run").input_schema["properties"]["run_id"]
    assert "last" in run_id["description"]


def test_list_runs(loaded):
    out = ok(create_server(loaded), "list_runs", limit=5)
    assert out["total_runs"] == 33 and len(out["runs"]) == 5
    assert "support-bot" in out["agents"]
    assert {"run_id", "agent", "duration_ms", "cost_usd"} <= out["runs"][0].keys()


def test_analyze_run_cites_step_numbers(loaded):
    out = ok(create_server(loaded), "analyze_run", run_id="run_482", max_findings=3)
    assert out["run_id"] == "run_482" and out["findings_total"] >= 1
    assert len(out["top_findings"]) <= 3
    top = out["top_findings"][0]
    assert top["suggestion"] and top["detail"] and top["problem"]
    assert all(isinstance(s, int) for s in top["steps"])


def test_explain_cost_tells_how_to_price_unknown_models(loaded, prices_file):
    out = ok(create_server(loaded), "explain_cost", run_id="run_482")
    assert out["cost_complete"] is False
    assert out["unpriced_models"] == ["Qwen/Qwen2.5-72B-Instruct"]
    assert "pricing.yaml" in out["how_to_price"]

    out = ok(create_server(loaded, prices_file), "explain_cost", run_id="run_482")
    assert out["cost_complete"] is True and out["total_cost_usd"] > 0
    assert "unpriced_models" not in out
    shares = [s["share"] for s in out["costliest_steps"]]
    assert shares == sorted(shares, reverse=True) and abs(sum(shares) - 1) < 0.01


def test_show_run_outline(loaded):
    out = ok(create_server(loaded), "show_run", run_id="run_48")
    steps = out["steps"]
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    assert any(s.get("retry_of") == 6 for s in steps)
    assert any(s.get("side_effects") for s in steps)
    assert all("input_messages" not in s for s in steps)  # full content is get_step's job


def test_get_step_returns_content_and_citing_findings(loaded):
    server = create_server(loaded)
    report = core.analyze("run_482", loaded)
    cited = report.step_numbers[report.findings[0].evidence_span_ids[0]]
    out = ok(server, "get_step", run_id="run_482", step=cited)
    assert out["step"] == cited
    assert any(f["title"] == report.findings[0].title for f in out["cited_by_findings"])

    out = ok(server, "get_step", run_id="run_482", step=2, max_chars=100)
    assert all(len(m.get("content", "")) < 140 for m in out.get("input_messages", []))

    assert "there is no step 999" in error_text(server, "get_step", run_id="run_482", step=999)


def test_bad_run_ids_explain_what_to_do(loaded):
    server = create_server(loaded)
    assert "list_runs" in error_text(server, "analyze_run", run_id="nope")
    assert "longer prefix" in error_text(server, "show_run", run_id="run_")


def test_scan_runs(loaded):
    out = ok(create_server(loaded), "scan_runs", limit=100)
    assert out["runs_analyzed"] == 33 and out["runs_with_problems"] > 0
    first = out["problems"][0]
    assert first["worst_severity"] == "high" and 0 < first["share"] <= 1
    assert len(first["example_run_ids"]) <= 3
    assert out["worst_runs"][0]["top_finding"]


def test_compare_runs(loaded):
    server = create_server(loaded)
    out = ok(server, "compare_runs", run_a="run_482", run_b="run_482")
    assert all(v == 0 for v in out["delta"].values())
    assert out["tool_calls"] == [] and out["problems_only_in_a"] == out["problems_only_in_b"] == []

    other = json.loads(BAD_AGENT_RUNS.read_text(encoding="utf-8"))[0]["run_id"]
    out = ok(server, "compare_runs", run_a="run_482", run_b=other)
    assert out["b"]["run_id"] == other
    assert out["delta"]["duration_ms"] == out["b"]["duration_ms"] - out["a"]["duration_ms"]


def test_bad_prices_file_is_a_tool_error(loaded, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [{match: 1}]\n")
    assert error_text(create_server(loaded, bad), "explain_cost", run_id="last")


def test_cli_serves_over_stdio(sample_path, tmp_path, prices_file):
    """The real `agent-doctor mcp` process: nothing but protocol messages may reach stdout."""
    db = tmp_path / "stdio.db"
    with TraceStore(db) as s:
        s.import_json_file(sample_path)
    exe = shutil.which("agent-doctor", path=os.path.dirname(sys.executable))
    if exe is None:
        pytest.skip("agent-doctor script is not installed")
    params = StdioServerParameters(command=exe, args=["--db", str(db), "mcp", "--prices", str(prices_file)],
                                   env={**os.environ})

    async def go():
        async with Client(params) as client:
            return await client.call_tool("explain_cost", {"run_id": "last"})
    result = asyncio.run(go())
    assert not result.is_error, result.content
    assert result.structured_content["run_id"] == "run_482"
    assert result.structured_content["cost_complete"] is True


def _saved_eval(store):
    from agent_doctor.evals import CaseResult, Score, compare_reports
    from agent_doctor.evals.runner import summarize

    def result(case_id, passed, answer):
        return CaseResult(case_id=case_id, output_text=answer, run_id=f"r-{case_id}", error=None,
                          cost_usd=0.01, cost_complete=True, duration_ms=100, llm_calls=1, tool_calls=0,
                          scores=[Score(name="exact_match", value=float(passed), passed=passed,
                                        reason="" if passed else "expected 'Friday'")],
                          passed=passed, problems=[])

    base = summarize("v1", [result("a", True, "Friday"), result("b", True, "Monday")])
    cand = summarize("v2", [result("a", False, "Saturday"), result("b", True, "Monday")])
    return core.save_eval("support-bot", [base, cand], [compare_reports(base, cand)], store)


def test_saved_evals_over_mcp(store):
    server = create_server(store)
    assert "list_evals" in error_text(server, "get_eval")  # nothing saved yet
    rec = _saved_eval(store)
    listed = ok(server, "list_evals")["evals"]
    assert listed[0]["eval_id"] == rec.eval_id and "v2 vs v1: worse (1 broken)" == listed[0]["result"]

    out = ok(server, "get_eval")
    assert out["eval_id"] == rec.eval_id and [v["name"] for v in out["versions"]] == ["v1", "v2"]
    assert out["versions"][1]["scorers"] == {"exact_match": 0.5}
    comparison = out["comparisons"][0]
    assert comparison["verdict"] == "worse" and comparison["regressed"] == ["a"]
    assert comparison["cases"] == 2 and "changed_cases" not in comparison

    out = ok(server, "get_eval", eval_id=rec.eval_id[:12], include_cases=True)
    (broke,) = out["comparisons"][0]["changed_cases"]
    assert broke["change"] == "broke" and broke["baseline"]["answer"] == "Friday"
    assert broke["candidate"]["answer"] == "Saturday"
    assert broke["candidate"]["failed"] == ["exact_match: expected 'Friday'"]
