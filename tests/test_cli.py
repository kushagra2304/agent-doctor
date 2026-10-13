import json

from typer.testing import CliRunner

from agent_doctor.cli import app

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, list(args))


def test_validate_ok(sample_path):
    r = invoke("validate", str(sample_path))
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["run_id"] == "run_482"


def test_validate_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_id": "r"}))
    r = invoke("validate", str(bad))
    assert r.exit_code == 1
    assert "invalid trace" in r.output


def test_validate_missing(tmp_path):
    assert invoke("validate", str(tmp_path / "missing.json")).exit_code == 2


def test_version():
    r = invoke("--version")
    assert r.exit_code == 0 and "agent-doctor" in r.stdout


def test_import_runs_show_cost(sample_path, tmp_path):
    db = str(tmp_path / "t.db")
    r = invoke("--db", db, "import", str(sample_path))
    assert r.exit_code == 0, r.output
    assert "imported run_482" in r.stdout

    r = invoke("--db", db, "runs")
    assert r.exit_code == 0 and "run_482" in r.stdout and "support-bot" in r.stdout

    r = invoke("--db", db, "show", "run_4")
    assert r.exit_code == 0, r.output
    assert "retry of #6" in r.stdout and "side effects" in r.stdout

    r = invoke("--db", db, "cost", "run_482")
    assert r.exit_code == 0 and "No price for: Qwen/Qwen2.5-72B-Instruct" in r.stdout

    prices = tmp_path / "p.yaml"
    prices.write_text('models:\n  - {match: "qwen/*", input: 1.0, output: 2.0}\n')
    r = invoke("--db", db, "cost", "run_482", "--prices", str(prices), "--json")
    assert r.exit_code == 0, r.output
    report = json.loads(r.stdout)
    assert report["fully_priced"] is True and report["total_cost_usd"] > 0


def test_unknown_run(tmp_path):
    r = invoke("--db", str(tmp_path / "t.db"), "show", "nope")
    assert r.exit_code == 1 and "no run" in r.output


def test_empty_runs(tmp_path):
    r = invoke("--db", str(tmp_path / "t.db"), "runs")
    assert r.exit_code == 0 and "No runs" in r.stdout


def test_pricing_lookup():
    r = invoke("pricing", "-m", "claude-sonnet-5-5")
    assert r.exit_code == 0 and "claude-sonnet-5-5*" in r.stdout
    assert invoke("pricing", "-m", "made-up-model").exit_code == 1
    assert invoke("pricing").exit_code == 0
