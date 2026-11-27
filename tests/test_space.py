"""The Hugging Face Space's view functions (the Gradio layer itself is not imported)."""

import importlib.util
import json
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "space" / "app.py"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("space_app", SPACE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_views(app):
    runs, labels, prices = app.load_demo()
    assert len(runs) == 32
    run = next(r for r in runs if labels[r.run_id]["scenario"] == "everything-wrong-1")
    assert app.run_label(run, labels).startswith("everything-wrong-1")
    summary, findings, plan, steps = app.analyze_one(run, prices, labels)
    assert run.run_id in summary and "problem(s)" in summary and "Planted" in summary
    assert findings and all(len(row) == len(app.FINDING_HEADERS) for row in findings)
    assert "Fix plan" in plan and "Verify" in plan
    assert [row[0] for row in steps] == list(range(1, len(run.spans) + 1))
    head, rows, all_plan = app.scan_view(runs, prices)
    assert "32" in head and rows and rows[0][1] == "high" and "Fix plan for 32 runs" in all_plan


def test_upload(app, tmp_path):
    assert app.analyze_upload(None, "auto", True)[0] == []
    bad = tmp_path / "bad.json"
    bad.write_text('{"x": 1}')
    runs, msg = app.analyze_upload(str(bad), "auto", True)
    assert runs == [] and msg.startswith("Could not read")
    good = tmp_path / "run.json"
    data = json.loads((Path(__file__).parent / "fixtures" / "sample_run.json").read_text())
    data["task_input"] = "my email is asha@example.com"
    good.write_text(json.dumps(data))
    runs, msg = app.analyze_upload(str(good), "auto", True)
    assert len(runs) == 1 and "redacted" in msg and runs[0].task_input == "my email is [EMAIL]"


def test_eval_view_without_data(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "EVAL_FILE", tmp_path / "missing.json")
    text, rows = app.eval_view()
    assert "sync.py" in text and rows == []
