"""Evaluation suite: scorers, statistics, cases, comparisons, and the bad agent's suite."""

import json
import textwrap

import pytest
from typer.testing import CliRunner

from agent_doctor import trace_run, trace_tool
from agent_doctor.cli import app
from agent_doctor.evals import (
    Attempt,
    CaseResult,
    EvalCase,
    Score,
    Variant,
    cases_from_runs,
    compare,
    compare_reports,
    evaluate,
    load_cases,
    save_cases,
    scorer,
    scorers,
    stats,
)
from agent_doctor.evals.runner import summarize
from agent_doctor.pricing import ModelPrice, PriceTable
from agent_doctor.schema import Finding, FindingType, Severity
from agent_doctor.testing import simulated_time, sleep

PRICES = PriceTable([ModelPrice(match="*", input=1.0, output=2.0)])


def attempt(text="Order 1234 arrives Friday.", error=None, **kw):
    return Attempt(output=text, output_text=text, run=kw.pop("run", None), error=error, **kw)


CASE = EvalCase(id="c1", input="Where is 1234?", expected="order 1234 arrives friday")


# -- scorers ------------------------------------------------------------------------
def test_exact_match_and_similarity():
    assert scorers.exact_match()(CASE, attempt()).passed  # case and punctuation ignored
    assert not scorers.exact_match(normalize=False)(CASE, attempt()).passed
    assert scorers.exact_match()(EvalCase(id="x", input="q"), attempt()) is None  # no expected answer
    sim = scorers.similarity(threshold=0.5)(CASE, attempt("Order 1234 arrives Monday."))
    assert 0.7 < sim.value < 0.8 and sim.passed


def test_contains_and_matches():
    s = scorers.contains("1234", "friday")(CASE, attempt())
    assert s.passed and s.value == 1
    s = scorers.contains("1234", "monday")(CASE, attempt())
    assert not s.passed and s.value == 0.5 and "monday" in s.reason
    assert scorers.contains("1234", "monday", require_all=False)(CASE, attempt()).passed
    assert scorers.matches(r"\bfri(day)?\b")(CASE, attempt()).passed
    with pytest.raises(ValueError):
        scorers.contains()


def test_run_based_scorers(sample_run):
    finding = Finding(run_id="r", type=FindingType.RETRY_LOOP, severity=Severity.MEDIUM, title="loop",
                      evidence_span_ids=["s"])
    a = attempt(run=sample_run, cost_usd=0.02, duration_ms=1500, findings=[finding])
    assert scorers.no_problems("high")(CASE, a).passed
    assert not scorers.no_problems("medium")(CASE, a).passed
    assert scorers.max_cost(0.05)(CASE, a).passed and not scorers.max_cost(0.01)(CASE, a).passed
    assert scorers.max_latency(2000)(CASE, a).passed and not scorers.max_latency(1000)(CASE, a).passed
    assert scorers.max_cost(0.01)(CASE, attempt(run=sample_run, cost_complete=False)) is None
    assert scorers.no_problems()(CASE, attempt()) is None  # nothing ran
    assert not scorers.completed()(CASE, attempt(error="ValueError: x")).passed


def test_custom_scorer_decorator():
    @scorer(threshold=0.5)
    def length_ok(case, a):
        return min(1.0, len(a.output_text) / 100)

    s = length_ok(CASE, attempt())
    assert s.name == "length_ok" and not s.passed and 0 < s.value < 0.5

    @scorer(name="polite")
    def polite(case, a):
        return "please" in a.output_text

    assert polite(CASE, attempt("yes please")) == Score(name="polite", value=1.0, passed=True)


# -- statistics -----------------------------------------------------------------------
def test_stats():
    assert stats.mcnemar_p(0, 0) == 1.0
    assert stats.mcnemar_p(0, 5) == pytest.approx(0.0625)
    assert stats.mcnemar_p(0, 6) == pytest.approx(0.03125)
    assert stats.mcnemar_p(3, 3) == 1.0
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5 and stats.percentile([], 50) == 0
    diffs = [1.0, 2.0, 3.0, 4.0]
    lo, hi = stats.bootstrap_ci(diffs)
    assert lo < 2.5 < hi and (lo, hi) == stats.bootstrap_ci(diffs)  # repeatable


# -- cases ----------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_cases_round_trip(tmp_path, suffix):
    cases = [CASE, EvalCase(id="c2", input="refund?", kwargs={"customer": "C-1"}, source_run_id="r2")]
    path = tmp_path / f"cases{suffix}"
    save_cases(cases, path)
    assert load_cases(path) == cases


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]))
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_cases_from_runs(sample_run):
    cases = cases_from_runs([sample_run], expected={"run_482": "yes"},
                            kwargs=lambda r: {"agent": r.agent_name})
    assert cases[0].id == cases[0].source_run_id == "run_482"
    assert cases[0].expected == "yes" and cases[0].kwargs == {"agent": "support-bot"}


# -- evaluate and compare ---------------------------------------------------------------
class Shop:
    def __init__(self):
        self.live = 0

    @trace_tool
    def stock(self, sku: str) -> dict:
        self.live += 1
        sleep(0.5)
        return {"sku": sku, "count": {"A": 3, "B": 0}[sku]}


def bot(shop, wording="{n} left"):
    @trace_run(agent="shop-bot")
    def handle(question: str, sku: str) -> str:
        n = shop.stock(sku)["count"]
        return wording.format(n=n) if n else "sold out"

    return handle


@pytest.fixture
def recorded_cases(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    handle = bot(Shop())
    with simulated_time():
        for sku in ("A", "B"):
            handle(f"stock of {sku}?", sku=sku)
    runs = [store.get_run(r.run_id) for r in store.list_runs()]
    expected = {"stock of A?": "3 left", "stock of B?": "sold out"}
    return cases_from_runs(runs, expected=lambda r: expected[r.task_input],
                           kwargs=lambda r: {"sku": r.task_input[-2]})


def test_recorded_baseline_and_replayed_candidate(store, recorded_cases):
    shop = Shop()
    with simulated_time():
        base, cand, cmp = compare(recorded_cases, [scorers.exact_match()], None, bot(shop, "only {n}"),
                                  store=store, prices=PRICES)
    assert shop.live == 0  # tools came from the recordings
    assert base.pass_rate == 1.0 and cand.pass_rate == 0.5
    assert cmp.verdict == "worse" and len(cmp.regressed) == 1 and not cmp.significant
    assert cmp.headline.startswith("1 case(s) broken, none fixed (p=1.000, could be chance")
    assert cmp.latency_change_pct == 0  # recorded tool latency reproduced
    regressed = cand.result(cmp.regressed[0])
    assert regressed.output_text == "only 3" and regressed.replay == {"recorded": 1}


def test_live_variant_and_missing_sources(store, recorded_cases):
    shop = Shop()
    with simulated_time():
        report = evaluate(Variant(bot(shop), replay=False), recorded_cases, [scorers.exact_match()],
                          store=store, prices=PRICES)
    assert shop.live == 2 and report.pass_rate == 1.0
    with pytest.raises(ValueError, match="not found"):
        evaluate(None, recorded_cases, [], store={})
    with pytest.raises(ValueError, match="no source run"):
        evaluate(None, [EvalCase(id="x", input="q")], [], store=store)


def test_crashing_agent_and_broken_scorer(store, recorded_cases):
    @trace_run(agent="shop-bot")
    def crash(question, sku):
        raise RuntimeError("boom")

    def broken(case, a):
        raise KeyError("oops")

    report = evaluate(crash, recorded_cases, [scorers.completed(), broken], store=store, prices=PRICES)
    assert report.errors == 2 and report.pass_rate == 0
    reasons = {s.name: s.reason for s in report.results[0].scores}
    assert reasons["completed"] == "RuntimeError: boom" and "scorer failed" in reasons["broken"]


def test_comparison_verdicts_and_problem_counts():
    def result(case_id, passed, problems=(), cost=1.0):
        from agent_doctor.evals import CaseResult
        return CaseResult(case_id=case_id, output_text="", run_id="r", error=None, cost_usd=cost,
                          cost_complete=True, duration_ms=100, llm_calls=1, tool_calls=0,
                          scores=[Score(name="s", value=float(passed), passed=passed)], passed=passed,
                          problems=list(problems))

    base = summarize("a", [result(f"c{i}", False, ["context bloat"]) for i in range(8)])
    cand = summarize("b", [result(f"c{i}", True, cost=0.5) for i in range(8)])
    cmp = compare_reports(base, cand)
    assert cmp.verdict == "better" and cmp.significant and cmp.p_value < 0.01
    assert cmp.cost_change_pct == -50 and cmp.problems_fixed == {"context bloat": 8}
    assert "cost -50%" in cmp.headline and "8 fewer problem(s)" in cmp.headline
    mixed = summarize("c", [result("c0", True), *[result(f"c{i}", False) for i in range(1, 8)]])
    before = summarize("d", [result("c0", False), result("c1", True)])
    assert compare_reports(before, mixed).verdict == "mixed"
    assert compare_reports(base, base).verdict == "same"
    with pytest.raises(ValueError, match="no cases in common"):
        compare_reports(base, summarize("e", [result("zz", True)]))
    assert "; s:" not in cmp.headline  # the only scorer moved with the overall result: no repeat


def test_regression_in_one_scorer_is_not_hidden_by_another():
    """Found on a real agent: every case failed 'replied_once', so overall pass/fail never moved
    and the comparison said 'same' while correctness fell from 70% to 50%."""
    def result(case_id, correct):
        scores = [Score(name="correct", value=float(correct), passed=correct),
                  Score(name="replied_once", value=0.0, passed=False)]
        return CaseResult(case_id=case_id, output_text="", run_id="r", error=None, cost_usd=0,
                          cost_complete=True, duration_ms=100, llm_calls=1, tool_calls=0,
                          scores=scores, passed=False, problems=[])

    base = summarize("v1", [result(f"q{i}", i < 7) for i in range(10)])
    cand = summarize("v2", [result(f"q{i}", i < 4 or i == 9) for i in range(10)])
    cmp = compare_reports(base, cand)
    assert cmp.verdict == "same" and cmp.regressed == []
    (correct,) = [s for s in cmp.scorers if s.name == "correct"]
    assert correct.regressed == ["q4", "q5", "q6"] and correct.improved == ["q9"]
    assert "correct: 1 fixed, 3 broken (70% -> 50%)" in cmp.headline
    assert "replied_once" not in cmp.headline


# -- the bad agent's suite ---------------------------------------------------------------
def test_bad_agent_suite_shows_fixes_without_regressions():
    from examples.bad_agent.eval_suite import suite

    (recorded, fixed, small), (vs_fixed, vs_small) = suite.run()
    assert len(recorded.results) == 32
    assert recorded.problems and sum(fixed.problems.values()) < sum(recorded.problems.values())
    assert vs_fixed.regressed == [] and vs_small.regressed == []
    assert vs_fixed.cost_change_pct < -50 and vs_small.cost_change_pct < vs_fixed.cost_change_pct
    # the tracking API is still down in the replay, so those answers stay wrong
    exact = {s.name: s for s in fixed.scorers}["exact_match"]
    assert exact.pass_rate == {s.name: s for s in recorded.scorers}["exact_match"].pass_rate < 1
    assert "swallowed tool failure" in fixed.problems and "context bloat" not in fixed.problems


# -- CLI --------------------------------------------------------------------------------
runner = CliRunner()


def test_cli_eval_passes_and_fails_on_regression(tmp_path):
    r = runner.invoke(app, ["eval", "examples/bad_agent/eval_suite.py", "-v", "recorded", "-v", "fixed"])
    assert r.exit_code == 0, r.output
    assert "fixed vs recorded: better" in r.stdout and "fixed-small" not in r.stdout

    suite_file = tmp_path / "suite.py"
    suite_file.write_text(textwrap.dedent("""
        from agent_doctor.evals import EvalCase, Suite, scorers
        suite = Suite(name="tiny", cases=[EvalCase(id="a", input="hi", expected="hello")],
                      scorers=[scorers.exact_match()],
                      variants={"good": lambda q: "hello", "bad": lambda q: "bye"})
    """))
    r = runner.invoke(app, ["eval", str(suite_file)])
    assert r.exit_code == 1 and "broke a" in r.stdout
    r = runner.invoke(app, ["eval", str(suite_file), "--json"])
    assert r.exit_code == 1 and json.loads(r.stdout)["comparisons"][0]["regressed"] == ["a"]
    assert runner.invoke(app, ["eval", str(suite_file), "-v", "nope"]).exit_code == 2


def test_cli_replay(tmp_path, sample_path):
    db = str(tmp_path / "t.db")
    assert runner.invoke(app, ["--db", db, "import", str(sample_path)]).exit_code == 0
    agent_file = tmp_path / "bot.py"
    agent_file.write_text(textwrap.dedent("""
        from agent_doctor import trace_run
        @trace_run(agent="support-bot")
        def handle(question):
            return "fixed answer"
    """))
    r = runner.invoke(app, ["--db", db, "replay", "run_482", "--agent", f"{agent_file}:handle", "--save"])
    assert r.exit_code == 0, r.output
    assert "after:  fixed answer" in r.stdout and "Before and after" in r.stdout and "Saved as" in r.stdout
    r = runner.invoke(app, ["--db", db, "replay", "run_482", "--agent", f"{agent_file}:handle", "--json"])
    out = json.loads(r.stdout)
    assert out["output"] == "fixed answer" and out["comparison"]["a"]["run_id"] == "run_482"
    assert runner.invoke(app, ["--db", db, "replay", "last", "--agent", "nope:x"]).exit_code == 2
    assert runner.invoke(app, ["--db", db, "replay", "last", "--agent", f"{agent_file}:handle",
                               "-m", "bad"]).exit_code == 2


def test_saved_eval_results_store_and_cli(tmp_path):
    from agent_doctor import core
    from agent_doctor.store import EvalNotFound, TraceStore

    store = TraceStore(tmp_path / "e.db")
    with pytest.raises(EvalNotFound):
        core.get_eval("last", store)
    a = summarize("a", [CaseResult(case_id="c", output_text="x", run_id="r", error=None, cost_usd=0,
                                   cost_complete=True, duration_ms=1, llm_calls=0, tool_calls=0,
                                   scores=[], passed=True, problems=[])])
    first = core.save_eval("suite-one", [a], [], store)
    second = core.save_eval("suite-two", [a, a], [compare_reports(a, a)], store)
    assert first.headline == "a: 100% pass" and second.headline == "a vs a: same"
    assert [e.eval_id for e in core.list_evals(store=store)] == [second.eval_id, first.eval_id]
    assert [e.suite for e in core.list_evals("suite-one", store=store)] == ["suite-one"]
    loaded = core.get_eval(first.eval_id[:12], store)  # a unique prefix works too
    assert loaded.reports == [a] and loaded.suite == "suite-one"
    assert core.get_eval("last", store).eval_id == second.eval_id

    db = str(tmp_path / "e.db")
    r = runner.invoke(app, ["--db", db, "evals"])
    assert r.exit_code == 0 and second.eval_id in r.stdout and "suite-two" in r.stdout
    r = runner.invoke(app, ["--db", db, "evals", second.eval_id])
    assert r.exit_code == 0 and "a vs a: same" in r.stdout
    assert runner.invoke(app, ["--db", db, "evals", "nope"]).exit_code == 1


def test_cli_eval_records_results(tmp_path):
    suite_file = tmp_path / "suite.py"
    suite_file.write_text(textwrap.dedent("""
        from agent_doctor.evals import EvalCase, Suite, scorers
        suite = Suite(name="tiny", cases=[EvalCase(id="a", input="hi", expected="hello")],
                      scorers=[scorers.exact_match()], variants={"only": lambda q: "hello"})
    """))
    db = str(tmp_path / "t.db")
    r = runner.invoke(app, ["--db", db, "eval", str(suite_file), "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["eval_id"].startswith("eval_")
    r = runner.invoke(app, ["--db", db, "eval", str(suite_file), "--no-record"])
    assert r.exit_code == 0 and "Saved as" not in r.stdout
    r = runner.invoke(app, ["--db", db, "evals", "--json"])
    assert len(json.loads(r.stdout)) == 1  # --no-record saved nothing
