# Changelog

## 0.5.0

First public release, on PyPI as `agent-doctor-cli` (PyPI treats `agentdoctor` as the same name
as the existing `agent-doctor`).
The command is still `agent-doctor` and the import is still `agent_doctor`.

- **Recommender**: `agent-doctor recommend [RUN]` turns findings into a ranked fix plan from a
  library of 15 playbooks (why, fix with code, how to verify). Retrieval is deterministic
  (problem match + BM25 + provider filter); `--llm anthropic|openai:MODEL|hf:MODEL` also has an
  LLM write the plan from the evidence and the retrieved playbooks. `agent-doctor playbooks`
  lists and searches them; add your own with `--playbooks DIR`.
- **Importers**: `agent-doctor import` reads OpenTelemetry (OTLP JSON, GenAI conventions),
  Langfuse and LangSmith exports, as well as Agent Doctor JSON.
- **Redaction**: `--redact` on import, or `AGENT_DOCTOR_REDACT=1` while recording, scrubs
  emails, phone numbers, card numbers, API keys, tokens and IP addresses.
- **Saved evaluations**: `agent-doctor eval` saves its results; `agent-doctor evals [ID]`
  lists or shows them.
- **MCP**: new `recommend_fixes`, `search_playbooks`, `list_evals` and `get_eval` tools.
- **Hugging Face Space** demo in `space/`.

Fixes found by running it on a real tool-calling agent (`examples/real_agent/`):

- Eval comparisons now say what each scorer fixed and broke. Before, one scorer that failed
  every case kept the overall verdict at "same" and hid a drop in the others.
  `ScorerComparison` has new `improved` and `regressed` case lists.
- Replayed and evaluated runs keep the agent's `@trace_run` labels (`prompt_version`,
  `agent_version`, tags). They used to be dropped.
- "Swallowed tool failure" no longer fires when the answer tells the user the data is
  unavailable or timed out. When the tool worked for other inputs, the title names the
  failing arguments.
- New example: `examples/real_agent/`, with a recording script, an eval suite comparing two
  prompts and two models, and a write-up of what it found.

## 0.4.0

- Replay: `agent-doctor replay RUN --agent module:function` re-runs your agent on a saved run,
  with tool calls answered from the recording, side-effect tools blocked, and `-m OLD=NEW`
  model swaps.
- Evaluation: `agent-doctor eval SUITE.py` scores agent versions with built-in or custom
  scorers, compares them case by case with an exact McNemar test, and exits 1 on regressions.

## 0.3.0

- MCP server (`agent-doctor mcp`) with seven read-only tools for Claude, Cursor and other
  MCP clients.

## 0.2.0

- Twelve deterministic detectors, `agent-doctor analyze` and `agent-doctor scan`.
- The bad agent: 32 fixture runs with 36 planted problems and an answer key.

## 0.1.0

- Recording SDK (`instrument()`, `@trace_run`, `@trace_tool`), SQLite store, pricing and
  `agent-doctor cost`.
