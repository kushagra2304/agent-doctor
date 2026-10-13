# The bad agent

A customer-support agent with switchable, deliberately planted problems. Its runs are
the test fixtures for Agent Doctor's analyzers: each run comes with an answer key
saying exactly which problems were planted and which spans prove them.

Everything is fake and free: a fake LLM provider with two models (`demo-large`,
`demo-small`), realistic token counts, latency, prompt caching and rate limits, and
in-memory tools. Time is simulated and IDs are seeded, so the fixtures are identical on
every machine.

## Regenerate the fixtures

From the repository root:

```bash
python -m examples.bad_agent.generate          # writes tests/fixtures/bad_agent/
python -m examples.bad_agent.generate --check  # exits 1 if the files are stale
python -m examples.bad_agent.generate --save   # also saves the runs to your trace DB
```

After `--save`, browse them like real runs:

```bash
agent-doctor runs -n 40 --prices examples/demo_pricing.yaml
agent-doctor show <run id>
```

## What is planted

| Fault flag | Kind in `labels.json` | Finding type | What it looks like |
| --- | --- | --- | --- |
| `duplicate_lookup` | `exact_duplicate_tool_call` | redundant_tool_call | `search_orders` twice, same arguments |
| `near_duplicate_search` | `near_duplicate_tool_call` | redundant_tool_call | the same search, reworded, same results |
| `unused_tool` | `unused_tool_output` | unused_tool_output | customer profile fetched, never used |
| `tracking_failures=3` | `tool_retry_loop` | retry_loop | tracking fails 3 times, retried instantly |
| `tracking_down` | `tool_retry_loop` + `swallowed_tool_failure` | retry_loop, error | tracking never works; the agent guesses a date |
| `search_loop` | `no_progress_loop` | retry_loop | re-searches the same article 3 times |
| `llm_rate_limit=3` | `llm_retry_without_backoff` | retry_loop | three 429s retried with no backoff |
| `sequential_independent` | `sequential_independent_calls` | parallelizable_calls | three independent lookups in a row |
| `slow_inventory` | `slow_tool` | latency | one tool takes 4.5 s of a 5 s run |
| `context_bloat` | `context_bloat` | context_bloat | input grows ~3x as raw history piles up |
| `big_model_routing` | `expensive_model_for_simple_step` | cost | large model for a one-word decision |
| `no_prompt_cache` | `prompt_cache_miss` | cost | a timestamp at the top of the prompt breaks caching |

There are 32 runs: 6 clean runs (2 per question type), 2 runs per fault, and 2 runs
with five faults at once. `labels.json` also records each run's correct answer and
whether the agent got it right, for the evaluation suite in Phase 5.

**Analyzers must never read `labels.json` or run tags.** They only see what a real
agent would record.
