"""SQLite trace store.

Each run is stored twice: as full JSON (``runs.run_json``, the source of truth) and
as one row per span (``spans``) so later phases can query across runs quickly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from agent_doctor import config
from agent_doctor._util import now_ms
from agent_doctor.schema import Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    agent_name    TEXT NOT NULL,
    agent_version TEXT,
    status        TEXT NOT NULL,
    start_ms      INTEGER,
    duration_ms   INTEGER NOT NULL,
    span_count    INTEGER NOT NULL,
    llm_calls     INTEGER NOT NULL,
    tool_calls    INTEGER NOT NULL,
    error_count   INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    task_input    TEXT,
    saved_ms      INTEGER NOT NULL,
    run_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_name, start_ms);
CREATE INDEX IF NOT EXISTS idx_runs_start ON runs(start_ms);

CREATE TABLE IF NOT EXISTS spans (
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    span_id       TEXT NOT NULL,
    parent_id     TEXT,
    kind          TEXT NOT NULL,
    name          TEXT NOT NULL,
    model         TEXT,
    status        TEXT NOT NULL,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    args_hash     TEXT,
    PRIMARY KEY (run_id, span_id)
);
CREATE INDEX IF NOT EXISTS idx_spans_tool ON spans(name, args_hash);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class StoredRun:
    """One row of ``list_runs``: the run's totals without its spans."""

    run_id: str
    agent_name: str
    agent_version: str | None
    status: str
    start_ms: int | None
    duration_ms: int
    span_count: int
    llm_calls: int
    tool_calls: int
    error_count: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    task_input: str | None


class RunNotFound(KeyError):
    pass


class AmbiguousRunId(KeyError):
    pass


class TraceStore:
    """Save and load runs. ``path=":memory:"`` keeps everything in memory (handy in tests)."""

    def __init__(self, path: str | Path | None = None):
        self.path = str(path) if path is not None else str(config.db_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    # -- writing ----------------------------------------------------------------
    def save_run(self, run: Run) -> None:
        """Insert a run, or replace it if a run with the same ID exists."""
        s = run.summary()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM spans WHERE run_id = ?", (run.run_id,))
            self._conn.execute(
                """INSERT OR REPLACE INTO runs VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id, run.agent_name, run.agent_version, run.status.value,
                    run.start_ms, run.duration_ms, s["spans"], s["llm_calls"], s["tool_calls"],
                    s["errors"], s["input_tokens"], s["output_tokens"], s["cached_tokens"],
                    run.task_input[:500], now_ms(), run.model_dump_json(),
                ),
            )
            self._conn.executemany(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run.run_id, sp.span_id, sp.parent_id, sp.kind.value, sp.name, sp.model,
                        sp.status.value, sp.start_ms, sp.end_ms, sp.input_tokens,
                        sp.output_tokens, sp.cached_tokens, sp.args_hash,
                    )
                    for sp in run.spans
                ],
            )

    def delete_run(self, run_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return cur.rowcount > 0

    # -- reading ----------------------------------------------------------------
    def get_run(self, run_id: str) -> Run:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return Run.model_validate_json(row["run_json"])

    def resolve_run_id(self, run_id_or_prefix: str) -> str:
        """Accept a full run ID, a unique prefix of one (like git commit hashes), or
        ``last`` / ``latest`` for the most recent run."""
        if run_id_or_prefix.lower() in {"last", "latest"}:
            recent = self.list_runs(limit=1)
            if not recent:
                raise RunNotFound(run_id_or_prefix)
            return recent[0].run_id
        with self._lock:
            exact = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id_or_prefix,)
            ).fetchone()
            if exact:
                return exact["run_id"]
            pattern = run_id_or_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id LIKE ? ESCAPE '\\' LIMIT 6",
                (pattern + "%",),
            ).fetchall()
        if not rows:
            raise RunNotFound(run_id_or_prefix)
        if len(rows) > 1:
            ids = ", ".join(r["run_id"] for r in rows[:5])
            raise AmbiguousRunId(f"{run_id_or_prefix!r} matches several runs: {ids}")
        return rows[0]["run_id"]

    def list_runs(
        self, agent: str | None = None, limit: int = 20, offset: int = 0
    ) -> list[StoredRun]:
        """Most recent runs first."""
        sql = "SELECT * FROM runs"
        params: list = []
        if agent:
            sql += " WHERE agent_name = ?"
            params.append(agent)
        sql += " ORDER BY start_ms DESC, saved_ms DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            StoredRun(**{k: r[k] for k in StoredRun.__dataclass_fields__}) for r in rows
        ]

    def count_runs(self, agent: str | None = None) -> int:
        with self._lock:
            if agent:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM runs WHERE agent_name = ?", (agent,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    def agents(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT agent_name FROM runs ORDER BY agent_name"
            ).fetchall()
        return [r["agent_name"] for r in rows]

    # -- import helpers ---------------------------------------------------------
    def import_json_file(self, path: str | Path) -> list[str]:
        """Load a file holding one run or a JSON list of runs. Returns the saved IDs."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        runs = [Run.model_validate(item) for item in items]
        for run in runs:
            self.save_run(run)
        return [r.run_id for r in runs]

    # -- lifecycle --------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_default_stores: dict[str, TraceStore] = {}
_default_lock = threading.Lock()


def get_default_store() -> TraceStore:
    """The store at the configured path, opened once per process and path."""
    path = str(config.db_path())
    with _default_lock:
        store = _default_stores.get(path)
        if store is None:
            store = TraceStore(path)
            _default_stores[path] = store
        return store
