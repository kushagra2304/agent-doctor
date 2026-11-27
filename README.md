# Agent Doctor

Find out why your AI agent is expensive, slow or unreliable, and how to fix it.

Agent Doctor records every step of an agent run (LLM calls, tool calls, retries,
errors), finds wasted cost and time, and recommends concrete fixes. It exposes
all of this as MCP tools, so you can ask Claude or Cursor *"why was run_482 so
expensive?"* and get a real answer. A replay engine lets you test a new prompt
or model against saved runs before you deploy.

> Status: **0.5.0, ready for a first release**. Recording, cost, problem detection, fix plans, the MCP server, replay, evaluation and importers from other tracing tools all work.

## Install

```bash
pip install agentdoctor               # the package is "agentdoctor"; the command is agent-doctor
pip install "agentdoctor[mcp]"        # plus the MCP server for Claude, Cursor and other clients
pip install "agentdoctor[mcp,hf]"     # plus Hugging Face InferenceClient support
```

In Python, import it as `agent_doctor`. Try it in the browser first: the
[Hugging Face Space](space/README.md) runs the demo with no install.

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

## Find the problems

```bash
agent-doctor analyze last          # one run: problems, evidence steps, estimated waste, fixes
agent-doctor scan -n 100           # many runs: which problems are most common and costly
```

Twelve deterministic detectors (plain code, no LLM calls, so analysis is free and instant):

| Detector | Finds |
| --- | --- |
| Exact duplicate tool call | The same tool called again with the same arguments |
| Near-duplicate tool call | The same question reworded, returning the same result |
| Unused tool output | A tool result that never reaches any later LLM call |
| Tool retry loop | A failing call retried 2+ times, flagged when there is no backoff |
| LLM retry without backoff | Rate-limited or failed LLM calls retried immediately |
| No-progress loop | The agent keeps calling tools that return nothing new |
| Swallowed failure | A tool never worked, yet the run ended "ok" with an answer |
| Slow step | One tool taking most of the run |
| Sequential independent calls | Back-to-back calls that could run in parallel |
| Context bloat | Prompts growing steadily as history and raw tool output pile up |
| Expensive model for a simple step | A large model producing a one-word decision |
| Prompt cache miss | A long shared prompt never served from cache, e.g. a timestamp at the top |

Each finding names the steps that prove it (as numbered by `agent-doctor show`) and
estimates the wasted cost, time or tokens. All thresholds live in
`agent_doctor.analyzers.Thresholds`.

**Graded against an answer key.** On the [bad agent](examples/bad_agent/README.md)'s 32
runs the detectors find all 36 planted problems with no false alarms on the clean runs:
`python -m examples.bad_agent.score`.

## Get a fix plan

```bash
agent-doctor recommend last                     # one run
agent-doctor recommend -n 200                   # the fixes that matter most across recent runs
agent-doctor recommend last --llm anthropic     # also have Claude write the plan up
```

Each finding is matched to fix playbooks: why the problem matters, how to fix it with code,
and how to verify the fix with `replay` or `eval`. Fixes are ranked by severity and by the
waste they remove. Provider-specific advice (prompt caching for Claude or OpenAI) only shows
up when the run actually used that provider.

Retrieval is deterministic and free: no LLM is needed. With `--llm anthropic` (default model
`claude-opus-5-5`), `--llm openai:MODEL` or `--llm hf:MODEL`, an LLM also writes a short plan
from the run's evidence and the retrieved playbooks, citing them by ID. Over MCP, Claude or
Cursor gets the playbooks from `recommend_fixes` and writes the plan itself.

Browse the playbooks with `agent-doctor playbooks [QUERY]` and `agent-doctor playbooks --show ID`.
Add your team's own as markdown files with front matter and pass `--playbooks DIR`; see
[the built-in ones](src/agent_doctor/recommender/playbooks/) for the format.

## Bring traces from other tools

```bash
agent-doctor import otel-export.json            # OpenTelemetry (OTLP JSON, GenAI conventions)
agent-doctor import langfuse-observations.json  # Langfuse API or trace export
agent-doctor import langsmith-runs.json         # LangSmith runs (flat or with child_runs)
agent-doctor import traces.jsonl --redact       # JSON lines too; scrub personal data first
```

The format is detected automatically (`--format` to force one). One trace becomes one run;
LLM calls, tool calls, errors and token counts are mapped onto Agent Doctor's schema, and
retries of failed tool calls are linked, so every detector works on imported traces.

**Redaction.** `--redact` on import, or `AGENT_DOCTOR_REDACT=1` while recording, replaces
emails, phone numbers, card numbers (Luhn-checked), API keys, bearer tokens, JWTs, AWS keys and
IP addresses with placeholders, and masks values under keys like `password` or `api_key`.
Duplicate-call detection keeps working, because argument hashes are taken before redaction.
Add your own patterns with `agent_doctor.ingest.Redactor(extra=[("ORDER", r"ORD-\d+")])`.

## Ask Claude or Cursor about your runs (MCP)

```bash
pip install "agentdoctor[mcp]"
claude mcp add agent-doctor -- agent-doctor mcp      # Claude Code
```

For Claude Desktop, Cursor or any other MCP client, add this to its MCP config
(`claude_desktop_config.json`, `.cursor/mcp.json`). If `agent-doctor` is installed in a
virtualenv, give the full path to it as `command`:

```json
{
  "mcpServers": {
    "agent-doctor": {"command": "agent-doctor", "args": ["mcp"]}
  }
}
```

Add `"--prices", "path/to/pricing.yaml"` after `"mcp"` for extra prices, or `"--db", "path/to/traces.db"`
before it for another database. Then ask things like *"why was the last run so expensive?"*
or *"what problems does support-bot have most often?"*.

The server has nine read-only tools. They never change your data or call your agent:

| Tool | Answers |
| --- | --- |
| `list_runs` | Which runs are there, and what did each cost? |
| `analyze_run` | What went wrong in this run, which steps prove it, and how do I fix it? |
| `explain_cost` | Where did the money go: per model, per step, cache savings, cost of 1,000 runs |
| `show_run` | What happened, step by step? |
| `get_step` | The full prompt, output, tool arguments and result of one step, and the findings that cite it |
| `scan_runs` | Which problems are most common and most costly across many runs? |
| `compare_runs` | Why did this run cost or take more than that one? |
| `recommend_fixes` | How do I fix it? The playbooks that remove the most waste, with code and how to verify |
| `search_playbooks` | Find a fix playbook by topic, e.g. "prompt caching with Claude" |

## Test a change before you deploy

### Replay a run

Run your agent again on a saved run's input, with a new prompt, model or code:

```bash
agent-doctor replay last --agent my_app.bot:handle                    # handle(task_input)
agent-doctor replay last --agent my_app.bot:handle -m gpt-4o=gpt-4o-mini
```

You get the old and new answer side by side, the change in time, calls, tokens and cost,
and which problems were fixed, which are new, and which are still there.

- **Tool calls are answered from the recording.** A call matching a recorded one (same tool,
  same arguments) returns the recorded output, or raises the recorded error again with its
  original exception class, without running the tool. Each one also takes as long as the
  recorded call did, so durations compare fairly (`--no-latency` to skip that).
- **Side-effect tools never run.** `@trace_tool(side_effects=True)` tools are stubbed, so a
  replay never sends an email or charges a card.
- **New calls run live**, unless you pass `--strict`. `--live-tools` runs every tool live
  (side-effect tools still never run).
- **LLM calls are live.** They are what you are testing. `-m OLD=NEW` swaps models without
  touching your code.

From Python: `agent_doctor.replay.replay(run, handle, model_map={...})`.

### Evaluate and compare versions

An evaluation suite is a Python file with test cases, scorers and the versions to compare.
The first version is the baseline. `None` means "the runs as they were recorded", so you can
compare a new prompt with what production actually did:

```python
# evals/support_bot.py
from agent_doctor.evals import Suite, Variant, cases_from_runs, load_cases, scorers
from my_app.bot import handle, handle_v2

suite = Suite(
    name="support-bot",
    cases=load_cases("evals/support_bot.yaml"),     # or cases_from_runs(saved runs)
    scorers=[scorers.exact_match(), scorers.completed(), scorers.no_problems("high")],
    variants={
        "recorded": None,
        "v2-prompt": handle_v2,
        "mini": Variant(handle, model_map={"gpt-4o": "gpt-4o-mini"}),
    },
)
```

```bash
agent-doctor eval evals/support_bot.py
```

Each case is replayed against its recorded run when it has one (`source_run_id`), and run
live otherwise. Side-effect tools are blocked either way. Every new run also goes through the
twelve detectors, so the report shows pass rate, cost, latency and problems for each
version. The comparison pairs results case by case: which cases broke, which got fixed, and
an exact McNemar test that says whether the difference could be chance. **The command exits
with code 1 if any case that passed before fails now**, so it can gate a deploy in CI.

Built-in scorers: `exact_match`, `contains`, `matches` (regex), `similarity` (word overlap),
`completed`, `no_problems`, `max_cost`, `max_latency`. Write your own with `@scorer`:

```python
from agent_doctor.evals import scorer

@scorer
def mentions_order_id(case, attempt):
    return case.kwargs["order_id"] in (attempt.output_text or "")
```

Try it on the bad agent: `agent-doctor eval examples/bad_agent/eval_suite.py`. Fixing the
agent's own faults cuts cost by 61% and removes 24 of the 36 problems with no answer getting
worse. Sending large-model calls to the small model (`model_map`) cuts cost by 94%. The
replays keep the recorded environment, so the tracking API that was down is still down, and
those two answers stay wrong.

## Testing your own agent

`agent_doctor.testing` gives you simulated time (`simulated_time()`, `sleep()`) and
repeatable IDs (`deterministic_ids()`), so recorded runs are identical on every machine.
The [bad agent](examples/bad_agent/README.md) uses them to build the fixtures.

## Commands

| Command | What it does |
| --- | --- |
| `agent-doctor analyze RUN` | Problems in one run, with evidence, estimated waste and fixes |
| `agent-doctor scan` | Problems across many runs, most serious and costly first |
| `agent-doctor replay RUN --agent M:F` | Run your agent again on a saved run, tools answered from the recording |
| `agent-doctor eval SUITE.py` | Evaluate and compare agent versions; exits 1 if any case regressed |
| `agent-doctor mcp` | Run the MCP server over stdio (needs the `mcp` extra) |
| `agent-doctor runs` | List recent runs with duration, calls, errors, tokens and cost |
| `agent-doctor show RUN` | Every step of a run as a tree, with retries, errors and cached tokens |
| `agent-doctor cost RUN` | Cost per LLM call and per model, cache savings, cost of 1,000 runs |
| `agent-doctor recommend [RUN]` | Ranked fix plan from the playbooks; `--llm` to have an LLM write it up |
| `agent-doctor playbooks [QUERY]` | List, search (`QUERY`) or read (`--show ID`) the fix playbooks |
| `agent-doctor import FILE` | Load runs: Agent Doctor JSON, OTLP JSON, Langfuse or LangSmith; `--redact` |
| `agent-doctor validate FILE` | Check a trace file against the schema |
| `agent-doctor pricing [-m MODEL]` | List prices, or show which price a model gets |

`RUN` is a run ID from `agent-doctor runs`, a unique prefix of it, or `last` for the most recent run. Add `--json` to `cost` or `analyze` for machine-readable output.

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
- [x] Phase 2: bad agent with 12 kinds of planted problems; 32 fixture runs with an answer key ([details](examples/bad_agent/README.md))
- [x] Phase 3: 12 deterministic detectors, `analyze` and `scan`; 36/36 planted problems found, 0 false alarms
- [x] Phase 4: MCP server with 7 read-only tools (`agent-doctor mcp`)
- [x] Phase 5: replay (recorded tools, blocked side effects, model swaps) and evaluation suite (scorers, paired statistics, CI exit code)
- [x] Phase 6: recommender (15 playbooks, deterministic retrieval, optional LLM write-up), OpenTelemetry / Langfuse / LangSmith importers, redaction, Hugging Face Space demo, release workflow
- [ ] Launch: publish to PyPI and the Space

## License

MIT
