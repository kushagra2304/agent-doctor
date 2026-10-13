from pathlib import Path

import pytest

from agent_doctor import _util
from agent_doctor.schema import Run
from agent_doctor.store import TraceStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.agent-doctor during tests."""
    home = tmp_path / "ad-home"
    monkeypatch.setenv("AGENT_DOCTOR_HOME", str(home))
    monkeypatch.delenv("AGENT_DOCTOR_DB", raising=False)
    monkeypatch.delenv("AGENT_DOCTOR_PRICING", raising=False)
    monkeypatch.delenv("AGENT_DOCTOR_DISABLED", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    _util._warned.clear()
    return home


@pytest.fixture
def store() -> TraceStore:
    s = TraceStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def sample_path() -> Path:
    return FIXTURES / "sample_run.json"


@pytest.fixture
def sample_run(sample_path: Path) -> Run:
    return Run.from_json_file(str(sample_path))


@pytest.fixture
def sample_dict(sample_run: Run) -> dict:
    return sample_run.model_dump(mode="json")
