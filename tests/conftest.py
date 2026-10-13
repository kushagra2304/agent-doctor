from pathlib import Path

import pytest

from agent_doctor.schema import Run

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_path() -> Path:
    return FIXTURES / "sample_run.json"


@pytest.fixture
def sample_run(sample_path: Path) -> Run:
    return Run.from_json_file(str(sample_path))


@pytest.fixture
def sample_dict(sample_run: Run) -> dict:
    return sample_run.model_dump(mode="json")
