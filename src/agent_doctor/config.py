"""Settings, read from environment variables at call time (so tests can change them).

| Variable | Default | Meaning |
| --- | --- | --- |
| AGENT_DOCTOR_HOME | ~/.agent-doctor | Folder for local data |
| AGENT_DOCTOR_DB | $AGENT_DOCTOR_HOME/traces.db | SQLite trace database |
| AGENT_DOCTOR_PRICING | $AGENT_DOCTOR_HOME/pricing.yaml | Your own prices (override built-ins) |
| AGENT_DOCTOR_DISABLED | unset | Set to 1 to turn recording off |
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("AGENT_DOCTOR_HOME") or Path.home() / ".agent-doctor")


def db_path() -> Path:
    env = os.environ.get("AGENT_DOCTOR_DB")
    return Path(env) if env else data_dir() / "traces.db"


def user_pricing_path() -> Path:
    env = os.environ.get("AGENT_DOCTOR_PRICING")
    return Path(env) if env else data_dir() / "pricing.yaml"


def recording_enabled() -> bool:
    return os.environ.get("AGENT_DOCTOR_DISABLED", "").strip().lower() not in {"1", "true", "yes"}
