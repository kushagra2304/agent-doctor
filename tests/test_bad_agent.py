"""Checks on the bad-agent fixtures: the answer key must be complete and truthful.

If you change the bad agent, regenerate with ``python -m examples.bad_agent.generate``.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from agent_doctor.schema import FindingType, Run, SpanKind
from examples.bad_agent.generate import generate, render

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bad_agent"


@pytest.fixture(scope="module")
def runs() -> dict[str, Run]:
    data = json.loads((FIXTURE_DIR / "runs.json").read_text(encoding="utf-8"))
    return {r["run_id"]: Run.model_validate(r) for r in data}


@pytest.fixture(scope="module")
def labels() -> dict:
    return json.loads((FIXTURE_DIR / "labels.json").read_text(encoding="utf-8"))["runs"]


def planted(labels, kind):
    return [(rid, p) for rid, lab in labels.items() for p in lab["planted"] if p["kind"] == kind]


def test_fixtures_are_up_to_date():
    runs, labels = generate()
    files = render(runs, labels)
    for name, text in files.items():
        on_disk = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
        assert on_disk == json.loads(text), f"{name} is stale: python -m examples.bad_agent.generate"


def test_generation_is_repeatable():
    a, b = generate(), generate()
    assert render(*a) == render(*b)


def test_every_run_has_a_label(runs, labels):
    assert set(runs) == set(labels)
    assert len(runs) >= 30


def test_evidence_spans_exist(runs, labels):
    for rid, lab in labels.items():
        ids = {s.span_id for s in runs[rid].spans}
        for p in lab["planted"]:
            assert p["evidence_span_ids"], p
            assert set(p["evidence_span_ids"]) <= ids, (lab["scenario"], p["kind"])
            FindingType(p["type"])


def test_every_finding_type_is_covered_twice(labels):
    counts = Counter(p["type"] for lab in labels.values() for p in lab["planted"])
    for t in FindingType:
        assert counts[t.value] >= 2, f"no fixture plants {t.value}"


def test_every_kind_is_planted_at_least_twice(labels):
    kinds = Counter(p["kind"] for lab in labels.values() for p in lab["planted"])
    assert all(n >= 2 for n in kinds.values()), kinds
    assert len(kinds) >= 12


def test_clean_runs_have_no_problems(runs, labels):
    clean = [rid for rid, lab in labels.items() if lab["clean"]]
    assert len(clean) >= 6
    for rid in clean:
        run = runs[rid]
        assert labels[rid]["planted"] == [] and labels[rid]["answer_correct"]
        assert run.error_count == 0
        keys = [(s.name, s.args_hash) for s in run.tool_spans]
        assert len(keys) == len(set(keys)), "clean run repeats a tool call"
        ok = [s for s in run.llm_spans if not s.is_error]
        assert max(s.input_tokens for s in ok) < 1.2 * min(s.input_tokens for s in ok)
        assert all(s.duration_ms < 2000 for s in run.tool_spans)


def test_exact_duplicates_share_args(runs, labels):
    for rid, p in planted(labels, "exact_duplicate_tool_call"):
        a, b = (runs[rid].get_span(i) for i in p["evidence_span_ids"])
        assert a.name == b.name and a.args_hash == b.args_hash


def test_near_duplicates_differ_in_args_only(runs, labels):
    for rid, p in planted(labels, "near_duplicate_tool_call"):
        a, b = (runs[rid].get_span(i) for i in p["evidence_span_ids"])
        assert a.name == b.name and a.args_hash != b.args_hash
        assert a.tool_output["results"] == b.tool_output["results"]


def test_tool_retry_loops_are_linked(runs, labels):
    for rid, p in planted(labels, "tool_retry_loop"):
        spans = [runs[rid].get_span(i) for i in p["evidence_span_ids"]]
        assert len(spans) >= 3
        assert all(s.is_error for s in spans[:-1])
        for prev, cur in zip(spans, spans[1:], strict=False):
            assert cur.retry_of == prev.span_id
            assert cur.start_ms - prev.end_ms < 100  # no backoff


def test_swallowed_failures_end_in_wrong_answer(runs, labels):
    hits = planted(labels, "swallowed_tool_failure")
    assert hits
    for rid, p in hits:
        *tools, answer = (runs[rid].get_span(i) for i in p["evidence_span_ids"])
        assert all(t.is_error for t in tools)
        assert answer.kind is SpanKind.LLM and not answer.is_error
        assert runs[rid].status.value == "ok"
        assert labels[rid]["answer_correct"] is False


def test_unused_output_never_reaches_an_llm(runs, labels):
    for rid, p in planted(labels, "unused_tool_output"):
        run = runs[rid]
        tool = run.get_span(p["evidence_span_ids"][0])
        later = [s for s in run.llm_spans if s.start_ms >= tool.end_ms]
        prompt = json.dumps([s.input_messages for s in later])
        assert later and tool.tool_output["name"] not in prompt


def test_sequential_calls_are_independent_and_back_to_back(runs, labels):
    for rid, p in planted(labels, "sequential_independent_calls"):
        spans = [runs[rid].get_span(i) for i in p["evidence_span_ids"]]
        for prev, cur in zip(spans, spans[1:], strict=False):
            assert cur.start_ms >= prev.end_ms
            assert all(str(v) not in json.dumps(cur.tool_args) for v in prev.tool_output.values()
                       if isinstance(v, str) and len(v) > 3)


def test_no_progress_loop_repeats_the_same_results(runs, labels):
    for rid, p in planted(labels, "no_progress_loop"):
        tools = [runs[rid].get_span(i) for i in p["evidence_span_ids"]]
        searches = [s for s in tools if s.name == "search_kb"]
        assert len(searches) >= 3
        assert len({json.dumps(s.tool_output["results"]) for s in searches}) == 1


def test_slow_tool_dominates_the_run(runs, labels):
    for rid, p in planted(labels, "slow_tool"):
        span = runs[rid].get_span(p["evidence_span_ids"][0])
        assert span.duration_ms / runs[rid].duration_ms > 0.5


def test_context_bloat_grows(runs, labels):
    for rid, p in planted(labels, "context_bloat"):
        tokens = [runs[rid].get_span(i).input_tokens for i in p["evidence_span_ids"]]
        assert tokens == sorted(tokens) and tokens[-1] > 2.5 * tokens[0]


def test_big_model_routing_uses_large_model(runs, labels):
    for rid, p in planted(labels, "expensive_model_for_simple_step"):
        span = runs[rid].get_span(p["evidence_span_ids"][0])
        assert span.model == "demo-large" and span.output_tokens <= 5
    clean_routes = [runs[r].llm_spans[0] for r, lab in labels.items() if lab["clean"]]
    assert all(s.model == "demo-small" for s in clean_routes)


def test_prompt_cache_misses(runs, labels):
    for rid, p in planted(labels, "prompt_cache_miss"):
        assert all(runs[rid].get_span(i).cached_tokens == 0 for i in p["evidence_span_ids"])
    clean = [r for r, lab in labels.items() if lab["clean"]]
    assert all(runs[r].total_cached_tokens > 0 for r in clean)


def test_llm_retries_have_no_backoff(runs, labels):
    for rid, p in planted(labels, "llm_retry_without_backoff"):
        spans = [runs[rid].get_span(i) for i in p["evidence_span_ids"]]
        assert all(s.is_error and s.kind is SpanKind.LLM for s in spans[:-1])
        assert not spans[-1].is_error
        assert all(b.start_ms - a.end_ms < 100 for a, b in zip(spans, spans[1:], strict=False))


def test_side_effect_tool_is_marked(runs):
    for run in runs.values():
        replies = [s for s in run.tool_spans if s.name == "send_reply"]
        assert len(replies) == 1 and replies[0].side_effects
