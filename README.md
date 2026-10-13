# Agent Doctor

Find out why your AI agent is expensive, slow or unreliable, and how to fix it.

Agent Doctor records every step of an agent run (LLM calls, tool calls, retries,
errors), finds wasted cost and time, and recommends concrete fixes. It exposes
all of this as MCP tools, so you can ask Claude or Cursor *"why was run_482 so
expensive?"* and get a real answer. A replay engine lets you test a new prompt
or model against saved runs before you deploy.

> Status: **Phase 0** (schema, packaging, tests). Not usable yet. See the roadmap below.

## Quick start (development)

Windows (PowerShell). Needs Python 3.10 or newer; check with `python --version`.
These commands use the venv's own Python, so no activation step is needed:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\agent-doctor validate tests/fixtures/sample_run.json
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
agent-doctor validate tests/fixtures/sample_run.json
```

## The trace schema

Every trace is normalized into three models in `src/agent_doctor/schema.py`:

| Model | What it is |
| --- | --- |
| `Span` | One step: an LLM call, tool call, retrieval or agent step, with timing, tokens, inputs and outputs |
| `Run` | One full agent execution for one input; holds its spans and validates that they fit together |
| `Finding` | One problem an analyzer found, with evidence span IDs and wasted cost or time |

Tool arguments are hashed (`args_hash`, key order ignored) so repeated tool calls
are easy to spot. Tools with `side_effects=True` are never called live during replay.

## Roadmap

- [x] Phase 0: setup, schema, tests, CI
- [ ] Phase 1: SDK (`instrument()`, `@trace_run`, `@trace_tool`), SQLite store, pricing, cost CLI
- [ ] Phase 2: demo agent with planted problems
- [ ] Phase 3: analyzers (redundancy, loops, latency, context bloat)
- [ ] Phase 4: MCP server
- [ ] Phase 5: evaluation suite and replay
- [ ] Phase 6: recommender (RAG), Hugging Face Space demo, launch

## License

MIT
