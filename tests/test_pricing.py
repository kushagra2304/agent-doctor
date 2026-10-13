import pytest
from pydantic import ValidationError

from agent_doctor.pricing import ModelPrice, PriceTable, estimate_cost
from agent_doctor.schema import Run


def table(*entries) -> PriceTable:
    return PriceTable([ModelPrice(**e) for e in entries])


def test_builtin_loads():
    t = PriceTable.builtin()
    assert len(t) > 10
    assert all(e.origin == "built-in" for e in t.entries)


def test_builtin_matches_dated_and_prefixed_ids():
    t = PriceTable.builtin()
    assert t.find("claude-sonnet-5-5-20261001").match == "claude-sonnet-5-5*"
    assert t.find("anthropic/claude-haiku-4-5").match == "claude-haiku-4-5*"
    assert t.find("claude-opus-4-6").input == 5.0
    assert t.find("claude-opus-4-1-20250805").input == 15.0
    assert t.find("Qwen/Qwen2.5-72B-Instruct") is None


def test_exact_beats_glob_and_longest_glob_wins():
    t = table({"match": "m*", "input": 1, "output": 1},
              {"match": "model-x*", "input": 2, "output": 2},
              {"match": "model-x", "input": 3, "output": 3})
    assert t.find("model-x").input == 3
    assert t.find("model-x-2024").input == 2
    assert t.find("mini").input == 1


def test_name_normalization():
    t = table({"match": "GPT 5.6 Luna", "input": 1, "output": 1})
    assert t.find("gpt-5.6-luna") is not None
    assert t.find("GPT_5.6_LUNA") is not None


def test_provider_filter():
    t = table({"match": "llama*", "provider": "groq", "input": 1, "output": 1},
              {"match": "llama*", "provider": "together", "input": 2, "output": 2})
    assert t.find("llama-3", "together").input == 2
    assert t.find("llama-3", "groq").input == 1
    assert t.find("llama-3", None).input == 1  # no provider on the span: first match


def test_earlier_entries_win_ties():
    t = table({"match": "x*", "input": 9, "output": 9}, {"match": "x*", "input": 1, "output": 1})
    assert t.find("x1").input == 9


def test_cost_math_with_cache():
    p = ModelPrice(match="m", input=2.0, cached_input=0.5, output=10.0)
    # 1,000 uncached * 2 + 1,000 cached * 0.5 + 500 out * 10, per 1M
    assert p.cost_usd(2000, 500, cached_tokens=1000) == pytest.approx(0.0075)


def test_cached_defaults_to_input_price():
    p = ModelPrice(match="m", input=2.0, output=10.0)
    assert p.cost_usd(1000, 0, cached_tokens=1000) == pytest.approx(0.002)


def test_long_context_tier():
    p = ModelPrice(match="m", input=0.1, output=0.5,
                   long_context={"above_input_tokens": 100_000, "input": 0.5, "output": 2.5})
    assert p.rates(100_000)[0] == 0.1
    assert p.rates(100_001)[0] == 0.5
    assert p.cost_usd(200_000, 1000) == pytest.approx(0.1 + 0.0025)


def test_user_file_overrides_builtin(tmp_path, monkeypatch):
    f = tmp_path / "p.yaml"
    f.write_text('models:\n  - {match: "claude-sonnet-5-5*", input: 99, output: 99}\n')
    monkeypatch.setenv("AGENT_DOCTOR_PRICING", str(f))
    t = PriceTable.load()
    entry = t.find("claude-sonnet-5-5")
    assert entry.input == 99 and entry.origin == str(f)


def test_extra_file_beats_user_file(tmp_path, monkeypatch):
    user = tmp_path / "user.yaml"
    user.write_text('models:\n  - {match: "m", input: 1, output: 1}\n')
    extra = tmp_path / "extra.yaml"
    extra.write_text('models:\n  - {match: "m", input: 7, output: 7}\n')
    monkeypatch.setenv("AGENT_DOCTOR_PRICING", str(user))
    assert PriceTable.load(extra).find("m").input == 7


def test_bad_unit_rejected():
    with pytest.raises(ValueError, match="per_million_tokens"):
        PriceTable.parse({"unit": "per_token", "models": []}, "x")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        PriceTable.parse({"models": [{"match": "m", "input": 1, "output": 1, "oops": 1}]}, "x")


def test_estimate_cost_sample(sample_run):
    t = table({"match": "qwen/qwen2.5-72b-instruct", "input": 1.0, "cached_input": 0.25,
               "output": 2.0})
    r = estimate_cost(sample_run, t)
    expected = (4000 * 1 + 60 * 2 + 8200 * 1 + 55 * 2 + 18000 * 1 + 4000 * 0.25 + 140 * 2) / 1e6
    assert r.total_cost_usd == pytest.approx(expected)
    assert r.fully_priced and r.unpriced_models == []
    assert r.most_expensive_span_id == "s8"
    assert r.cache_savings_usd == pytest.approx(4000 * 0.75 / 1e6)
    assert r.by_model[0].calls == 3
    assert r.projected_cost_usd(1000) == pytest.approx(expected * 1000)


def test_estimate_cost_unpriced(sample_run):
    r = estimate_cost(sample_run, PriceTable([]))
    assert r.total_cost_usd == 0
    assert not r.fully_priced
    assert r.unpriced_models == ["Qwen/Qwen2.5-72B-Instruct"]
    assert all(s.cost_usd is None for s in r.spans if s.kind == "llm")
    assert r.by_model[0].cost_usd is None


def test_tool_cost_attribute(sample_dict):
    sample_dict["spans"][2]["attributes"] = {"cost_usd": 0.005}
    r = estimate_cost(Run.model_validate(sample_dict), PriceTable([]))
    assert r.tool_cost_usd == pytest.approx(0.005)
    assert r.total_cost_usd == pytest.approx(0.005)
