---
id: stable-prompt-prefix
title: Keep the start of the prompt identical between calls
problems: [prompt_cache_miss]
effort: low
---
## Why

Providers cache prompts by prefix: the cached part has to match the earlier call exactly,
from the first character. A timestamp, request ID or user name at the top of the system
prompt changes the very beginning, so nothing after it can be reused. You pay full price
for the same long instructions on every call.

## Fix

Order the prompt from most stable to least stable:

1. System prompt, policies, examples, tool definitions (identical for everyone)
2. Per-user or per-session context
3. Per-call content: the current time, the question, fresh tool results

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},                       # never changes
    {"role": "user", "content": f"Current time: {now}\nQuestion: {q}"},  # changes per call
]
```

Avoid anything random in the stable part: no shuffled examples, no dict ordering that can
change, no "Request ID" lines.

## Verify

`agent-doctor show RUN` lists cached tokens per call. After the change, every call after the
first should show a large cached count, and `agent-doctor cost RUN` reports the savings.
