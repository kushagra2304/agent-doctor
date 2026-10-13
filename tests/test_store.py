import json

import pytest

from agent_doctor.schema import Run
from agent_doctor.store import AmbiguousRunId, RunNotFound, TraceStore, get_default_store


def renamed(run: Run, new_id: str, shift_ms: int = 0) -> Run:
    data = run.model_dump(mode="json")
    data["run_id"] = new_id
    for s in data["spans"]:
        s["run_id"] = new_id
        s["start_ms"] += shift_ms
        s["end_ms"] += shift_ms
    return Run.model_validate(data)


def test_save_and_get_round_trip(store, sample_run):
    store.save_run(sample_run)
    assert store.get_run("run_482") == sample_run


def test_get_missing_raises(store):
    with pytest.raises(RunNotFound):
        store.get_run("nope")


def test_save_twice_replaces(store, sample_run):
    store.save_run(sample_run)
    changed = sample_run.model_copy(update={"final_output": "changed"})
    store.save_run(changed)
    assert store.count_runs() == 1
    assert store.get_run("run_482").final_output == "changed"
    with store._lock:
        n = store._conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    assert n == len(sample_run.spans)


def test_list_newest_first_and_filter(store, sample_run):
    store.save_run(sample_run)
    store.save_run(renamed(sample_run, "run_newer", shift_ms=60_000))
    other = renamed(sample_run, "run_other").model_copy(update={"agent_name": "other-bot"})
    store.save_run(other)
    ids = [r.run_id for r in store.list_runs(agent="support-bot")]
    assert ids == ["run_newer", "run_482"]
    assert store.count_runs() == 3
    assert store.agents() == ["other-bot", "support-bot"]
    row = store.list_runs(limit=1)[0]
    assert row.run_id == "run_newer" and row.llm_calls == 3 and row.error_count == 1


def test_resolve_prefix(store, sample_run):
    store.save_run(sample_run)
    store.save_run(renamed(sample_run, "run_483"))
    store.save_run(renamed(sample_run, "abc_1"))
    assert store.resolve_run_id("run_482") == "run_482"
    assert store.resolve_run_id("abc") == "abc_1"
    with pytest.raises(AmbiguousRunId):
        store.resolve_run_id("run_48")
    with pytest.raises(RunNotFound):
        store.resolve_run_id("zzz")
    # "_" must be literal, not a LIKE wildcard
    with pytest.raises(RunNotFound):
        store.resolve_run_id("abc_2")


def test_delete_cascades(store, sample_run):
    store.save_run(sample_run)
    assert store.delete_run("run_482") is True
    assert store.delete_run("run_482") is False
    with store._lock:
        assert store._conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0


def test_import_file_with_list(store, sample_run, tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps([sample_run.model_dump(mode="json"),
                                renamed(sample_run, "run_b").model_dump(mode="json")]))
    assert store.import_json_file(path) == ["run_482", "run_b"]
    assert store.count_runs() == 2


def test_file_store_persists(tmp_path, sample_run):
    db = tmp_path / "sub" / "traces.db"
    with TraceStore(db) as s:
        s.save_run(sample_run)
    with TraceStore(db) as s:
        assert s.get_run("run_482").run_id == "run_482"


def test_default_store_uses_home(isolated_home):
    store = get_default_store()
    assert store.path == str(isolated_home / "traces.db")
    assert get_default_store() is store


def test_resolve_last(store, sample_run):
    with pytest.raises(RunNotFound):
        store.resolve_run_id("last")
    store.save_run(sample_run)
    store.save_run(renamed(sample_run, "run_newest", shift_ms=60_000))
    assert store.resolve_run_id("last") == "run_newest"
    assert store.resolve_run_id("LATEST") == "run_newest"
