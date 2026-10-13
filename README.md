# Agent Doctor

Find out why your AI agent is expensive, slow or unreliable, and how to fix it.

Agent Doctor records every step of an agent run (LLM calls, tool calls, retries,
errors), finds wasted cost and time, and recommends concrete fixes. It exposes
all of this as MCP tools, so you can ask Claude or Cursor *"why was run_482 so
expensive?"* and get a real answer. A replay engine lets you test a new prompt
or model against saved runs before you deploy.

> Status: **Phase 1**. Recording, storage and cost work. Analysis and MCP tools come next (see roadmap).

## Quick start (development)

Windows (PowerShell). Needs Python 3.10 or newer; check with `python --version`.
These commands use the venv's own Python, so no activation step is needed:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,hf]"
.venv\Scripts\python -m pytest
.venv\Scripts\agent-doctor validate tests/fixtures/sample_run.json
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"
pytest
agent-doctor validate tests/fixtures/sample_run.json
```

## Record your agent

```python
from huggingface_hub import InferenceClient
from agent_doctor import instrument, trace_run, trace_tool

client = instrument(InferenceClient(model="Qwen/Qwen2.5-7B-Instruct"))  # every LLM call recorded

@trace_tool                      # each call becomes a tool span (args, output, errors, retries)
def search_orders(order_id: str): ...

@trace_tool(side_effects=True)   # never called live during replay
def send_reply(text: str): ...

@trace_run(agent="support-bot", prompt_version="v3")   # each call becomes one run
def handle(message: str) -> str: ...
```

`instrument()` also works with OpenAI-compatible and Anthropic clients, sync or async,
streaming or not. Runs are saved to `~/.agent-doctor/traces.db`. Recording never breaks
your agent: if anything goes wrong it warns once and carries on. Set
`AGENT_DOCTOR_DISABLED=1` to turn it off.

Try it without an API key:

```bash
python examples/offline_demo.py
agent-doctor runs --prices examples/demo_pricing.yaml
agent-doctor show last              # every step, with retries and errors
agent-doctor cost last --prices examples/demo_pricing.yaml
```

Or with a real open model (needs `HF_TOKEN`): `python examples/hf_quickstart.py`.

## Commands

| Command | What it does |
| --- | --- |
| `agent-doctor runs` | List recent runs with duration, calls, errors, tokens and cost |
| `agent-doctor show RUN` | Every step of a run as a tree, with retries, errors and cached tokens |
| `agent-doctor cost RUN` | Cost per LLM call and per model, cache savings, cost of 1,000 runs |
| `agent-doctor import FILE` | Load runs from JSON |
| `agent-doctor validate FILE` | Check a trace file against the schema |
| `agent-doctor pricing [-m MODEL]` | List prices, or show which price a model gets |

`RUN` is a run ID from `agent-doctor runs`, a unique prefix of it, or `last` for the most recent run. Add `--json` to `cost` for machine-readable output.

## Pricing

Built-in prices (USD per 1M tokens) cover current Claude and OpenAI models, each with
its source and the date it was checked. Open models routed through Hugging Face
Inference Providers are billed at each provider's rate, which Hugging Face does not
publish as a table, so add the rate you pay to `~/.agent-doctor/pricing.yaml`:

```yaml
models:
  # example numbers: put in the rate your provider charges
  - {match: "Qwen/Qwen2.5-7B-Instruct", provider: together, input: 0.30, output: 0.30}
```

Your entries win over the built-ins; `*` is a wildcard. Models with no price are
reported as "unpriced" instead of being counted as free.

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
- [x] Phase 1: SDK (`instrument()`, `@trace_run`, `@trace_tool`), SQLite store, pricing, cost CLI
- [ ] Phase 2: demo agent with planted problems
- [ ] Phase 3: analyzers (redundancy, loops, latency, context bloat)
- [ ] Phase 4: MCP server
- [ ] Phase 5: evaluation suite and replay
- [ ] Phase 6: recommender (RAG), Hugging Face Space demo, launch

## License

MIT
