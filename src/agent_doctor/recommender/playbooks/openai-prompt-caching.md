---
id: openai-prompt-caching
title: Get prompt caching to work with OpenAI models
problems: [prompt_cache_miss]
providers: [openai]
effort: low
---
## Why

OpenAI applies prompt caching automatically to long prompts (1,024 tokens or more), matching
on the prefix. There is nothing to turn on, so a cache miss almost always means the start of
the prompt changes between calls, or the prompt is too short to qualify.

## Fix

Put static content first and dynamic content last. Tool definitions and images count as
part of the prompt too, so keep their order fixed:

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},                 # static, first
        {"role": "user", "content": f"{context}\n\nQuestion: {q}"},   # dynamic, last
    ],
    tools=TOOLS,                                                       # same list, same order
)
```

The response's `usage.prompt_tokens_details.cached_tokens` shows how many input tokens were
served from cache; Agent Doctor records it for every call.

## Verify

`agent-doctor show RUN`: calls after the first should list cached tokens. `agent-doctor cost
RUN` reports how much caching saved.
