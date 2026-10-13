import json

import pytest

from agent_doctor.cli import main


def test_validate_ok(sample_path, capsys):
    assert main(["validate", str(sample_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "run_482"
    assert out["tool_calls"] == 5


def test_validate_invalid(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_id": "r"}))
    assert main(["validate", str(bad)]) == 1
    assert "invalid trace" in capsys.readouterr().err


def test_validate_missing_file(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.json")]) == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "agent-doctor" in capsys.readouterr().out
