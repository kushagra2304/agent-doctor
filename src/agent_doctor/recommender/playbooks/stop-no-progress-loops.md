---
id: stop-no-progress-loops
title: Stop loops that make no progress
problems: [no_progress_loop]
effort: low
---
## Why

The agent keeps calling tools that return what it already has. Each round costs a model
call and a tool call and adds nothing. The cause is often in the prompt ("always search
again to be thorough") or in a loop with no stopping rule.

## Fix

1. **Check the prompt** for instructions that push the model to keep going, and remove them.
2. **Cap iterations** and stop when a round brings nothing new:

```python
seen = set()
for step in range(MAX_STEPS):          # e.g. 5
    result = run_tool(next_action)
    fingerprint = json.dumps(result, sort_keys=True)
    if fingerprint in seen:
        break                          # nothing new: answer with what we have
    seen.add(fingerprint)
```

3. **Tell the model** when a tool returns something it has seen before, so it can decide to
answer instead of searching again.

## Verify

Replay the looping runs with the new prompt (`agent-doctor replay RUN --agent ...`): LLM and
tool calls should drop, and `agent-doctor eval` should show the answers did not get worse.
