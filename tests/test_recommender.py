"""Recommender: playbook loading, retrieval, ranking, provider filtering, CLI and LLM prompt."""

import json

import pytest
from typer.testing import CliRunner

from agent_doctor.analyzers import analyze_run
from agent_doctor.cli import app
from agent_doctor.pricing import PriceTable
from agent_doctor.recommender import (
    KnowledgeBase,
    build_prompt,
    default_knowledge_base,
    llm_from_spec,
    recommend,
    recommend_many,
    run_providers,
)
from agent_doctor.recommender.knowledge import parse_playbook
from agent_doctor.schema import Run
from tests.test_analyzers import PRICES, B

KINDS = {"exact_duplicate_tool_call", "near_duplicate_tool_call", "unused_tool_output", "tool_retry_loop",
         "llm_retry_without_backoff", "no_progress_loop", "swallowed_tool_failure", "slow_tool",
         "sequential_independent_calls", "context_bloat", "expensive_model_for_simple_step",
         "prompt_cache_miss"}


def test_every_detector_kind_has_a_playbook():
    kb = default_knowledge_base()
    covered = {k for b in kb.playbooks for k in b.problems}
    assert KINDS <= covered
    for b in kb.playbooks:
        assert b.section("Why") and b.section("Fix") and b.section("Verify"), b.id
        assert set(b.problems) <= KINDS, f"{b.id} names an unknown problem"


def test_search_finds_relevant_playbooks():
    kb = default_knowledge_base()
    assert kb.search("cache_control claude prompt caching")[0][0].id == "anthropic-prompt-caching"
    assert kb.search("exponential backoff jitter retry")[0][0].id == "retry-with-backoff"
    assert kb.search("asyncio gather concurrent")[0][0].id == "parallel-tool-calls"
    assert kb.search("zzzz qqqq") == []


def test_parse_playbook_and_extra_dirs(tmp_path):
    with pytest.raises(ValueError, match="front matter"):
        parse_playbook("no front matter", "x.md")
    (tmp_path / "mine.md").write_text(
        "---\nid: retry-with-backoff\ntitle: Our retry helper\nproblems: [tool_retry_loop]\n---\n"
        "## Why\nHouse rules.\n## Fix\nUse `ourlib.retry`.\n## Verify\nRun the suite.\n")
    kb = KnowledgeBase.load(tmp_path)
    assert kb.get("retry-with-backoff").title == "Our retry helper"  # yours wins on the same id
    assert len(kb.playbooks) == len(default_knowledge_base().playbooks)


def _bad_agent_runs():
    from examples.bad_agent.eval_suite import LABELS, RUNS
    return RUNS, LABELS


DEMO_PRICES = PriceTable.load("examples/demo_pricing.yaml")


def test_plan_for_one_run_cites_steps_and_ranks_by_severity():
    runs, labels = _bad_agent_runs()
    run = next(r for r in runs.values() if labels[r.run_id]["scenario"] == "everything-wrong-1")
    report = analyze_run(run, DEMO_PRICES)
    plan = recommend(report, run)
    assert plan.scope == run.run_id and plan.recommendations
    ids = [r.playbook_id for r in plan.recommendations]
    assert {"cache-tool-results", "trim-context", "route-to-small-model", "stable-prompt-prefix"} <= set(ids)
    assert "anthropic-prompt-caching" not in ids and "openai-prompt-caching" not in ids  # demo provider
    severities = [r.severity for r in plan.recommendations]
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    assert [order[s.value] for s in severities] == sorted(order[s.value] for s in severities)
    dup = plan.recommendations[ids.index("cache-tool-results")]
    assert dup.steps and all(isinstance(s, int) for s in dup.steps) and "def run_cache" in dup.fix


def test_provider_specific_playbooks_need_that_provider():
    b = B()
    for _ in range(3):
        b.llm(model="big", inp=2000, out=200, prompt="same long prompt")
    run = b.run()
    for span in run.spans:
        if span.kind.value == "llm":
            span.provider = "anthropic"
    assert run_providers(run) == {"anthropic"}
    report = analyze_run(run, PRICES)
    ids = [r.playbook_id for r in recommend(report, run).recommendations]
    if any(f.data.get("kind") == "prompt_cache_miss" for f in report.findings):
        assert ids[0] == "anthropic-prompt-caching" or "anthropic-prompt-caching" in ids
    assert "openai-prompt-caching" not in ids
    assert run_providers(Run(run_id="r", agent_name="a", task_input="", spans=[
        dict(span_id="s", run_id="r", kind="llm", name="gpt-4o-mini", start_ms=0, end_ms=1)])) == {"openai"}


def test_plan_across_runs_and_clean_runs():
    runs, labels = _bad_agent_runs()
    pairs = [(r, analyze_run(r, DEMO_PRICES)) for r in runs.values()]
    plan = recommend_many(pairs)
    assert plan.scope == "32 runs" and not plan.unmatched
    top = plan.recommendations[0]
    assert top.severity.value == "high" and len(top.runs) >= 2
    clean = next(r for r in runs.values() if labels[r.run_id]["clean"])
    assert recommend(analyze_run(clean, DEMO_PRICES), clean).recommendations == []


def test_llm_gets_evidence_and_playbooks():
    runs, labels = _bad_agent_runs()
    run = next(r for r in runs.values() if labels[r.run_id]["scenario"] == "context-bloat-1")
    report = analyze_run(run, DEMO_PRICES)
    seen = []
    plan = recommend(report, run, llm=lambda prompt: seen.append(prompt) or "1. Trim context [trim-context]")
    assert plan.narrative == "1. Trim context [trim-context]"
    prompt = seen[0]
    assert "[trim-context]" in prompt and run.run_id in prompt and "cite playbooks by ID" in prompt
    assert build_prompt(plan, [report]) == prompt


def test_llm_spec_validation():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        llm_from_spec("nope:model")
    with pytest.raises(ValueError, match="name a model"):
        llm_from_spec("openai")


runner = CliRunner()


def test_cli_recommend_and_playbooks(tmp_path):
    db = str(tmp_path / "t.db")
    assert runner.invoke(app, ["--db", db, "import", "tests/fixtures/bad_agent/runs.json"]).exit_code == 0
    r = runner.invoke(app, ["--db", db, "recommend", "--prices", "examples/demo_pricing.yaml"])
    assert r.exit_code == 0, r.output
    assert "Fix plan for 32 runs" in r.stdout and "Fixes:" in r.stdout and "Verify:" in r.stdout
    r = runner.invoke(app, ["--db", db, "recommend", "last", "--json", "--full"])
    plan = json.loads(r.stdout)
    assert plan["scope"].startswith("run_")
    r = runner.invoke(app, ["playbooks", "prompt caching"])
    assert r.exit_code == 0 and "stable-prompt-prefix" in r.stdout
    r = runner.invoke(app, ["playbooks", "--show", "circuit-breaker"])
    assert r.exit_code == 0 and "class CircuitBreaker" in r.stdout
    assert runner.invoke(app, ["playbooks", "--show", "nope"]).exit_code == 1
    assert runner.invoke(app, ["--db", db, "recommend", "--llm", "nope:x"]).exit_code == 2
