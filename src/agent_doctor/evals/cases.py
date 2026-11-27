"""Test cases: an input, optionally the expected answer, and the recorded run to replay."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_doctor.schema import Run


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str
    expected: str | None = Field(default=None, description="The correct answer, if known.")
    kwargs: dict[str, Any] = Field(default_factory=dict,
                                   description="Extra keyword arguments for the agent call.")
    source_run_id: str | None = Field(default=None, description="Recorded run to replay tools from.")
    tags: dict[str, Any] = Field(default_factory=dict)


def cases_from_runs(
    runs: Iterable[Run],
    expected: Mapping[str, str] | Callable[[Run], str | None] | None = None,
    kwargs: Callable[[Run], dict[str, Any]] | None = None,
) -> list[EvalCase]:
    """One case per recorded run, replaying that run's tool calls.

    ``expected`` maps run IDs to correct answers, or computes one from the run.
    ``kwargs`` computes extra arguments for the agent call from the run.
    """
    cases = []
    for run in runs:
        exp = expected(run) if callable(expected) else (expected or {}).get(run.run_id)
        cases.append(EvalCase(id=run.run_id, input=run.task_input, expected=exp,
                              kwargs=kwargs(run) if kwargs else {}, source_run_id=run.run_id,
                              tags=dict(run.tags)))
    return cases


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read cases from JSON or YAML: a list of cases, or ``{"cases": [...]}``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    items = data.get("cases", []) if isinstance(data, dict) else data
    cases = [EvalCase.model_validate(item) for item in items]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: case IDs must be unique")
    return cases


def save_cases(cases: Iterable[EvalCase], path: str | Path) -> None:
    data = {"cases": [c.model_dump(exclude_defaults=True) for c in cases]}
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
