import copy

import pytest
from pydantic import ValidationError

from agent_doctor.schema import (
    Finding,
    FindingType,
    Run,
    Severity,
    Span,
    SpanKind,
    SpanStatus,
    compute_args_hash,
)


def make_span(**overrides) -> Span:
    data = dict(
        span_id="a", run_id="r1", kind="tool", name="search", start_ms=100, end_ms=200
    )
    data.update(overrides)
    return Span(**data)


# --- sample trace -----------------------------------------------------------


def test_sample_trace_loads(sample_run: Run):
    assert sample_run.run_id == "run_482"
    assert len(sample_run.spans) == 9
    assert len(sample_run.llm_spans) == 3
    assert len(sample_run.tool_spans) == 5


def test_sample_totals(sample_run: Run):
    assert sample_run.total_input_tokens == 4000 + 8200 + 22000
    assert sample_run.total_output_tokens == 60 + 55 + 140
    assert sample_run.total_cached_tokens == 4000
    assert sample_run.duration_ms == 9100
    assert sample_run.error_count == 1
    assert sample_run.models_used == ["Qwen/Qwen2.5-72B-Instruct"]


def test_sample_duplicate_tool_calls_share_hash(sample_run: Run):
    s3, s5 = sample_run.get_span("s3"), sample_run.get_span("s5")
    assert s3.args_hash is not None
    assert s3.args_hash == s5.args_hash


def test_retry_with_reordered_args_has_same_hash(sample_run: Run):
    s6, s7 = sample_run.get_span("s6"), sample_run.get_span("s7")
    assert s7.retry_of == "s6"
    assert s6.args_hash == s7.args_hash


def test_round_trip(sample_run: Run, tmp_path):
    path = tmp_path / "out.json"
    sample_run.to_json_file(str(path))
    again = Run.from_json_file(str(path))
    assert again == sample_run


def test_children_and_order(sample_run: Run):
    kids = sample_run.children("s1")
    assert [s.span_id for s in kids] == ["s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    starts = [s.start_ms for s in sample_run.ordered_spans]
    assert starts == sorted(starts)


def test_summary_keys(sample_run: Run):
    summary = sample_run.summary()
    assert summary["llm_calls"] == 3
    assert summary["errors"] == 1


# --- args hash --------------------------------------------------------------


def test_args_hash_ignores_key_order():
    assert compute_args_hash({"a": 1, "b": 2}) == compute_args_hash({"b": 2, "a": 1})


def test_args_hash_differs_for_different_values():
    assert compute_args_hash({"q": "x"}) != compute_args_hash({"q": "y"})


def test_args_hash_none():
    assert compute_args_hash(None) is None


def test_explicit_args_hash_is_kept():
    span = make_span(tool_args={"q": 1}, args_hash="custom")
    assert span.args_hash == "custom"


# --- span validation --------------------------------------------------------


def test_end_before_start_rejected():
    with pytest.raises(ValidationError, match="end_ms is before start_ms"):
        make_span(start_ms=200, end_ms=100)


def test_negative_tokens_rejected():
    with pytest.raises(ValidationError):
        make_span(kind="llm", input_tokens=-1)


def test_cached_tokens_cannot_exceed_input():
    with pytest.raises(ValidationError, match="cached_tokens"):
        make_span(kind="llm", input_tokens=10, cached_tokens=20)


def test_ttft_cannot_exceed_duration():
    with pytest.raises(ValidationError, match="time_to_first_token_ms"):
        make_span(kind="llm", time_to_first_token_ms=500)


def test_llm_model_defaults_to_name():
    span = make_span(kind="llm", name="gpt-x")
    assert span.model == "gpt-x"
    assert span.kind is SpanKind.LLM


def test_error_span_gets_error_type():
    span = make_span(status="error")
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "unknown"
    assert span.is_error


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        make_span(tokens=5)


def test_self_parent_rejected():
    with pytest.raises(ValidationError, match="own parent"):
        make_span(parent_id="a")


def test_duration_and_total_tokens():
    span = make_span(kind="llm", input_tokens=7, output_tokens=3)
    assert span.duration_ms == 100
    assert span.total_tokens == 10


# --- run validation ---------------------------------------------------------


def test_span_with_wrong_run_id_rejected(sample_dict: dict):
    bad = copy.deepcopy(sample_dict)
    bad["spans"][1]["run_id"] = "other"
    with pytest.raises(ValidationError, match="expected 'run_482'"):
        Run.model_validate(bad)


def test_duplicate_span_ids_rejected(sample_dict: dict):
    bad = copy.deepcopy(sample_dict)
    bad["spans"][2]["span_id"] = "s2"
    with pytest.raises(ValidationError, match="duplicate span_id"):
        Run.model_validate(bad)


def test_unknown_parent_rejected(sample_dict: dict):
    bad = copy.deepcopy(sample_dict)
    bad["spans"][1]["parent_id"] = "nope"
    with pytest.raises(ValidationError, match="unknown parent_id"):
        Run.model_validate(bad)


def test_unknown_retry_of_rejected(sample_dict: dict):
    bad = copy.deepcopy(sample_dict)
    bad["spans"][6]["retry_of"] = "nope"
    with pytest.raises(ValidationError, match="unknown retry_of"):
        Run.model_validate(bad)


def test_empty_run():
    run = Run(run_id="r", agent_name="a", task_input="hi")
    assert run.duration_ms == 0
    assert run.start_ms is None
    assert run.summary()["spans"] == 0


# --- findings ---------------------------------------------------------------


def test_finding_valid(sample_run: Run):
    finding = Finding(
        run_id="run_482",
        type=FindingType.REDUNDANT_TOOL_CALL,
        severity=Severity.MEDIUM,
        title="search_orders called twice with the same arguments",
        evidence_span_ids=["s3", "s5"],
        wasted_ms=400,
    )
    finding.check_evidence(sample_run)


def test_finding_needs_evidence():
    with pytest.raises(ValidationError):
        Finding(
            run_id="r", type="cost", severity="low", title="x", evidence_span_ids=[]
        )


def test_finding_missing_evidence_detected(sample_run: Run):
    finding = Finding(
        run_id="run_482", type="latency", severity="high", title="slow", evidence_span_ids=["zz"]
    )
    with pytest.raises(ValueError, match="evidence spans not in run"):
        finding.check_evidence(sample_run)


def test_finding_negative_waste_rejected():
    with pytest.raises(ValidationError):
        Finding(
            run_id="r", type="cost", severity="low", title="x",
            evidence_span_ids=["a"], wasted_cost_usd=-1.0,
        )
