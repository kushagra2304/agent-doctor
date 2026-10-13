"""Detector tests on small hand-built runs, including cases the bad agent never produces."""

import pytest

from agent_doctor.analyzers import Thresholds, analyze_run
from agent_doctor.pricing import ModelPrice, PriceTable
from agent_doctor.schema import FindingType, Run, Severity

PRICES = PriceTable([
    ModelPrice(match="big", input=3.0, cached_input=0.3, output=15.0),
    ModelPrice(match="small", input=0.1, output=0.4),
])
SYSTEM = "You are a helpful agent. " + "Policy line. " * 400  # ~5,200 chars


class B:
    """Tiny builder: spans laid out back to back unless a gap is given."""

    def __init__(self):
        self.spans = [dict(span_id="root", run_id="r", kind="agent_step", name="agent",
                           start_ms=0, end_ms=0)]
        self.t = 0

    def _add(self, gap, dur, **kw):
        self.t += gap
        sid = f"s{len(self.spans)}"
        self.spans.append(dict(span_id=sid, run_id="r", parent_id="root", start_ms=self.t,
                               end_ms=self.t + dur, **kw))
        self.t += dur
        return sid

    def tool(self, name, args, out=None, dur=100, gap=10, error=False, **kw):
        extra = dict(status="error", error_type="Timeout") if error else dict(tool_output=out)
        return self._add(gap, dur, kind="tool", name=name, tool_args=args, **extra, **kw)

    def llm(self, model="small", inp=1000, out=20, cached=0, prompt="hi", system=SYSTEM,
            dur=300, gap=10, error=False, **kw):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        extra = dict(status="error", error_type="RateLimit") if error else {}
        return self._add(gap, dur, kind="llm", name=model, model=model, input_tokens=inp,
                         output_tokens=out, cached_tokens=cached, input_messages=msgs, **extra, **kw)

    def run(self, status="ok") -> Run:
        self.spans[0]["end_ms"] = self.t + 10
        return Run(run_id="r", agent_name="a", task_input="q", status=status, spans=self.spans)


def kinds(run, **kw):
    return sorted(f.data["kind"] for f in analyze_run(run, PRICES, **kw).findings)


def test_clean_run_has_no_findings():
    b = B()
    b.llm(prompt="route")
    b.tool("lookup", {"id": "42"}, {"name": "Widget Pro", "price": 1299})
    b.llm(model="big", prompt="Facts: Widget Pro costs 1299", out=40, cached=900)
    assert kinds(b.run()) == []


def test_exact_duplicate_and_side_effect_severity():
    b = B()
    b.tool("send_email", {"to": "a@b.c"}, {"sent": True}, side_effects=True)
    b.tool("send_email", {"to": "a@b.c"}, {"sent": True}, side_effects=True)
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.data["kind"] == "exact_duplicate_tool_call" and f.severity is Severity.HIGH
    assert f.wasted_ms == 100


def test_retry_with_backoff_is_not_called_no_backoff():
    b = B()
    first = b.tool("api", {"q": 1}, error=True)
    second = b.tool("api", {"q": 1}, error=True, gap=1000, retry_of=first)
    b.tool("api", {"q": 1}, {"v": "result text"}, gap=2000, retry_of=second)
    b.llm(prompt="result text")
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.data["kind"] == "tool_retry_loop" and f.data["no_backoff"] is False
    assert "repeatedly" in f.title and f.severity is Severity.MEDIUM


def test_single_retry_is_normal():
    b = B()
    first = b.tool("api", {"q": 1}, error=True)
    b.tool("api", {"q": 1}, {"v": "result text"}, retry_of=first)
    b.llm(prompt="result text")
    assert kinds(b.run()) == []


def test_failure_then_success_with_other_args_is_not_swallowed():
    b = B()
    b.tool("search", {"q": "a"}, error=True)
    b.tool("search", {"q": "b"}, {"hit": "useful doc"})
    b.llm(prompt="useful doc")
    assert "swallowed_tool_failure" not in kinds(b.run())


def test_swallowed_failure_needs_ok_status():
    b = B()
    b.tool("api", {"q": 1}, error=True)
    b.llm()
    assert "swallowed_tool_failure" in kinds(b.run())
    assert "swallowed_tool_failure" not in kinds(b.run(status="error"))


def test_dependent_calls_are_not_parallelizable():
    b = B()
    b.tool("get_user", {"id": "u1"}, {"email": "asha@example.com"}, dur=800)
    b.tool("get_orders", {"email": "asha@example.com"}, {"n": 3}, dur=800)
    b.llm(prompt="asha@example.com has n=3 orders")
    assert "sequential_independent_calls" not in kinds(b.run())


def test_independent_calls_are_parallelizable():
    b = B()
    b.tool("weather", {"city": "Pune"}, {"sky": "sunny"}, dur=800)
    b.tool("news", {"topic": "cricket"}, {"headline": "India win"}, dur=600)
    b.llm(prompt="sunny; India win")
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.type is FindingType.PARALLELIZABLE_CALLS and f.wasted_ms == 600


def test_parallel_threshold_is_configurable():
    b = B()
    b.tool("weather", {"city": "Pune"}, {"sky": "sunny"}, dur=200)
    b.tool("news", {"topic": "cricket"}, {"headline": "India win"}, dur=200)
    b.llm(prompt="sunny; India win")
    run = b.run()
    assert kinds(run) == []
    assert kinds(run, thresholds=Thresholds(parallel_min_saving_ms=100)) == [
        "sequential_independent_calls"]


def test_cheap_model_short_answer_is_fine_but_big_model_is_flagged():
    b = B()
    b.llm(model="small", out=2)
    b.llm(model="big", out=2, cached=1000)
    found = analyze_run(b.run(), PRICES).findings
    assert [f.data["kind"] for f in found] == ["expensive_model_for_simple_step"]
    assert found[0].evidence_span_ids == ["s2"]


def test_unpriced_model_is_never_called_expensive():
    b = B()
    b.llm(model="mystery-model", out=1)
    assert kinds(b.run()) == []


def test_unused_output_needs_captured_prompts():
    b = B()
    b.tool("profile", {"id": "u1"}, {"name": "Asha Rao", "tier": "gold"})
    b.llm(prompt="no profile data here")
    run = b.run()
    assert kinds(run) == ["unused_tool_output"]
    data = run.model_dump()
    for s in data["spans"]:
        if s["kind"] == "llm":
            s["input_messages"] = None
    assert kinds(Run.model_validate(data)) == []  # can't tell, so stay quiet


def test_small_numbers_must_sit_next_to_their_key():
    b = B()
    b.tool("stock", {"sku": "X"}, {"in_stock": 3})
    b.llm(prompt="Rule 3.1 applies")  # a stray 3 is not the stock count
    assert kinds(b.run()) == ["unused_tool_output"]
    b = B()
    b.tool("stock", {"sku": "X"}, {"in_stock": 3})
    b.llm(prompt="in_stock=3")
    assert kinds(b.run()) == []


def test_steady_bloat_flagged_spike_not():
    b = B()
    for n in (1000, 1800, 2600, 3400):
        b.llm(inp=n, cached=900, prompt=f"step {n}")
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.type is FindingType.CONTEXT_BLOAT and f.wasted_tokens == 800 + 1600 + 2400
    b = B()
    for n in (1000, 3500, 1000):  # one big call, not a pile-up
        b.llm(inp=n, cached=900)
    assert kinds(b.run()) == []


def test_cache_miss_from_dynamic_header():
    b = B()
    b.llm(model="big", system="Time: 10:00:01\n" + SYSTEM, out=30)
    b.llm(model="big", system="Time: 10:00:05\n" + SYSTEM, out=30)
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.data["kind"] == "prompt_cache_miss" and "dynamic_prompt_head" in f.data["reasons"]
    assert f.wasted_cost_usd == pytest.approx(len(SYSTEM) // 4 * 2.7 / 1e6, rel=0.01)


def test_short_prompts_are_not_cache_candidates():
    b = B()
    b.llm(system="short system prompt")
    b.llm(system="short system prompt")
    assert kinds(b.run()) == []


def test_no_progress_loop_and_no_double_reporting():
    b = B()
    for q in ("a b", "a b c", "a b d"):
        b.tool("search", {"q": q}, {"q": q, "hits": ["same doc"]})
        b.tool("read", {"id": "D1"}, {"text": "same doc"})
        b.llm(prompt="same doc", cached=900)
    found = kinds(b.run())
    assert found == ["no_progress_loop"]  # not also duplicates or near-duplicates


def test_slow_step():
    b = B()
    b.tool("db", {"q": 1}, {"rows": "many rows"}, dur=3000)
    b.llm(prompt="many rows")
    (f,) = analyze_run(b.run(), PRICES).findings
    assert f.type is FindingType.LATENCY and f.severity is Severity.HIGH


def test_broken_detector_does_not_hide_others():
    def broken(ctx):
        raise ZeroDivisionError

    from agent_doctor.analyzers import DETECTORS

    b = B()
    b.tool("db", {"q": 1}, {"rows": "many rows"}, dur=3000)
    b.llm(prompt="many rows")
    with pytest.warns(RuntimeWarning, match="broken"):
        report = analyze_run(b.run(), PRICES, detectors=[broken, *DETECTORS])
    assert len(report.findings) == 1


def test_report_summary_and_ranking(sample_run):
    report = analyze_run(sample_run, PRICES)
    summary = report.summary(max_findings=3)
    assert summary["run_id"] == "run_482" and summary["findings_total"] == len(report.findings)
    ranks = [f.severity for f in report.findings]
    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    assert [order[r] for r in ranks] == sorted(order[r] for r in ranks)
    assert all(isinstance(s, int) for f in summary["top_findings"] for s in f["steps"])


def test_bad_agent_scorecard_is_perfect():
    from examples.bad_agent.score import score

    s = score()
    assert s.missed == [] and s.false_alarms == [] and s.unexpected == []
    assert s.found == s.planted >= 36


def test_near_duplicate_ignores_argument_names():
    b = B()
    b.tool("lookup", {"order_id": "1234"}, {"status": "shipped"})
    b.tool("lookup", {"order_id": "9999"}, {"status": "shipped"})
    b.llm(prompt="status shipped", cached=900)
    assert kinds(b.run()) == []
    b = B()
    b.tool("search", {"query": "refund policy"}, {"hits": ["KB-12"]})
    b.tool("search", {"query": "what is the refund policy"}, {"hits": ["KB-12"]})
    b.llm(prompt="KB-12", cached=900)
    assert kinds(b.run()) == ["near_duplicate_tool_call"]
