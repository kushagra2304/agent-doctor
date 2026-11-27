"""Replay: recorded tool results served back, side effects blocked, models swapped."""

import asyncio

import pytest

from agent_doctor import instrument, trace_run, trace_tool
from agent_doctor.replay import ReplayedToolError, load_agent, replay
from agent_doctor.sdk import RunRecorder
from agent_doctor.testing import deterministic_ids, simulated_time, sleep
from tests.fakes import FakeHFClient, chat_response


class Outage(Exception):
    pass


class World:
    """Tools whose real calls are counted, so tests can tell served from live."""

    def __init__(self, fail_times=0):
        self.live = []
        self.fail_times = fail_times
        self.weather = "sunny"

    @trace_tool
    def lookup(self, city: str) -> dict:
        self.live.append(("lookup", city))
        sleep(0.4)
        if len([c for c in self.live if c[0] == "lookup"]) <= self.fail_times:
            raise Outage(f"weather API down for {city}")
        return {"city": city, "weather": self.weather}

    @trace_tool(side_effects=True)
    def notify(self, text: str) -> dict:
        self.live.append(("notify", text))
        return {"sent": True}

    @trace_tool(capture_output=False)
    def secret(self, key: str) -> str:
        self.live.append(("secret", key))
        return "s3cret"


def make_agent(world, retries=0, cities=("Pune",)):
    @trace_run(agent="weather-bot")
    def agent(question: str) -> str:
        found = []
        for city in cities:
            for attempt in range(retries + 1):
                try:
                    found.append(world.lookup(city)["weather"])
                    break
                except Outage:
                    if attempt == retries:
                        found.append("unknown")
        world.notify(f"answered {question}")
        return f"{question}: {', '.join(found)}"

    return agent


def record(agent, question, store):
    with simulated_time(), deterministic_ids("rec"):
        agent(question)
    return store.get_run(store.resolve_run_id("last"))


@pytest.fixture
def recorded(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.store.sqlite.get_default_store", lambda: store)
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    world = World()
    return world, record(make_agent(world), "weather?", store)


def test_tools_are_served_from_the_recording(recorded):
    old_world, run = recorded
    world = World()
    world.weather = "raining"  # the live tool would now say something else
    with simulated_time(), deterministic_ids("replay"):
        result = replay(run, make_agent(world))
    assert result.ok and result.output == "weather?: sunny"
    assert world.live == []  # nothing ran for real: lookup served, notify blocked
    assert result.tool_calls == {"recorded": 1, "stubbed": 1}
    lookup = next(s for s in result.run.tool_spans if s.name == "lookup")
    assert lookup.attributes["replay"] == "recorded" and lookup.duration_ms == 400  # recorded latency
    assert result.run.tags["replay_of"] == run.run_id and result.run.run_id != run.run_id


def test_no_latency_answers_at_once(recorded):
    _, run = recorded
    with simulated_time():
        result = replay(run, make_agent(World()), tool_latency="none")
    assert all(s.duration_ms == 0 for s in result.run.tool_spans)


def test_side_effects_run_only_when_allowed(recorded):
    _, run = recorded
    world = World()
    with simulated_time():
        replay(run, make_agent(world), allow_side_effects=True)
    assert world.live == [("notify", "answered weather?")]


def test_unrecorded_calls_run_live_or_fail(recorded):
    _, run = recorded
    world = World()
    with simulated_time():
        result = replay(run, make_agent(world, cities=("Pune", "Goa")))
    assert world.live == [("lookup", "Goa")]
    assert result.tool_calls["live-unrecorded"] == 1

    with simulated_time():
        result = replay(run, make_agent(World(), cities=("Pune", "Goa")), on_miss="error")
    assert result.error and "ReplayMiss" in result.error and "never recorded" in result.error


def test_live_tools_mode_still_blocks_side_effects(recorded):
    _, run = recorded
    world = World()
    with simulated_time():
        result = replay(run, make_agent(world), tools="live")
    assert world.live == [("lookup", "Pune")] and result.tool_calls == {"live": 1, "stubbed": 1}


def test_recorded_errors_are_raised_again_with_their_class(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    run = record(make_agent(World(fail_times=2), retries=3), "weather?", store)
    assert [s.status.value for s in run.tool_spans if s.name == "lookup"] == ["error", "error", "ok"]
    world = World()
    with simulated_time():
        result = replay(run, make_agent(world, retries=3))
    # Outage was raised and caught by the agent twice, then the success was served
    assert result.output == "weather?: sunny" and world.live == []
    errors = [s for s in result.run.tool_spans if s.is_error]
    assert [s.error_type for s in errors] == ["Outage", "Outage"]

    # More calls than were recorded: the last recorded result is served again
    with simulated_time():
        result = replay(run, make_agent(World(), retries=3, cities=("Pune", "Pune")))
    assert result.output == "weather?: sunny, sunny"
    assert result.tool_calls["recorded-repeat"] == 1


def test_unknown_error_class_becomes_replayed_tool_error(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    run = record(make_agent(World(fail_times=9)), "weather?", store)
    span = next(s for s in run.tool_spans if s.name == "lookup")
    span.error_type = "GoneFromTheCodebase"
    seen = []

    @trace_run(agent="weather-bot")
    def agent(question):
        try:
            World().lookup("Pune")
        except Exception as e:
            seen.append(e)
        return "x"

    with simulated_time():
        replay(run, agent)
    assert isinstance(seen[0], ReplayedToolError) and "weather API down" in str(seen[0])


def test_uncaptured_outputs_are_not_replayed(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    world = World()

    @trace_run(agent="secret-bot")
    def agent(q):
        return world.secret("k")

    run = record(agent, "q", store)
    world.live.clear()
    with simulated_time():
        result = replay(run, agent)
    assert result.output == "s3cret" and world.live == [("secret", "k")]


def test_model_map_swaps_models_without_code_changes(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    client = instrument(FakeHFClient([chat_response("hi", model="big-model"),
                                      chat_response("hi", model="small-model")], model="big-model"))

    @trace_run(agent="chat-bot")
    def agent(q):
        return client.chat_completion(messages=[{"role": "user", "content": q}])["choices"][0]["message"][
            "content"]

    run = record(agent, "hello", store)
    with simulated_time():
        result = replay(run, agent, model_map={"big-model": "small-model"})
    assert client.calls[-1]["model"] == "small-model"
    llm = result.run.llm_spans[0]
    assert llm.model == "small-model" and llm.attributes["replaced_model"] == "big-model"
    assert result.run.tags["model_map"] == {"big-model": "small-model"}
    assert run.llm_spans[0].model == "big-model"


def test_agent_with_its_own_recorder_and_saving(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    world = World()

    def agent(question):
        with RunRecorder("manual-bot", task_input=question, save=False) as rec:
            rec.final_output = world.lookup("Pune")["weather"]
        return rec.final_output

    with simulated_time():
        with RunRecorder("manual-bot", task_input="q", store=store) as rec:
            world.lookup("Pune")
    world.live.clear()
    with simulated_time():
        result = replay(rec.run, agent, label="v2", save=True, store=store)
    assert result.run.agent_name == "manual-bot" and len(result.run.tool_spans) == 1
    assert world.live == []
    saved = store.get_run(result.run.run_id)
    assert saved.tags == {"replay_of": rec.run.run_id, "replay_label": "v2"}


def test_agent_exceptions_are_reported(recorded):
    _, run = recorded

    @trace_run(agent="weather-bot")
    def broken(q):
        raise ValueError("bad prompt")

    result = replay(run, broken)
    assert not result.ok and result.error == "ValueError: bad prompt"
    assert result.run.status.value == "error"


def test_async_agent_and_tools(store, monkeypatch):
    monkeypatch.setattr("agent_doctor.sdk.recorder.get_default_store", lambda: store)
    calls = []

    @trace_tool
    async def fetch(x: int) -> int:
        calls.append(x)
        return x * 2

    @trace_run(agent="async-bot")
    async def agent(q):
        return sum([await fetch(1), await fetch(2)])

    with simulated_time():
        asyncio.run(agent("q"))
    run = store.get_run(store.resolve_run_id("last"))
    calls.clear()
    with simulated_time():
        result = replay(run, agent)
    assert result.output == 6 and calls == []


def test_replay_needs_recording(recorded, monkeypatch):
    monkeypatch.setenv("AGENT_DOCTOR_DISABLED", "1")
    with pytest.raises(RuntimeError, match="AGENT_DOCTOR_DISABLED"):
        replay(recorded[1], make_agent(World()))


def test_load_agent(tmp_path):
    f = tmp_path / "my_agent.py"
    f.write_text("class Bot:\n    def handle(self, q):\n        return q.upper()\nbot = Bot()\n")
    assert load_agent(f"{f}:bot.handle")("hi") == "HI"
    assert load_agent("tests.test_replay:make_agent") is make_agent
    with pytest.raises(ValueError, match="module:function"):
        load_agent("no_colon")
