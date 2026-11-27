---
id: anthropic-prompt-caching
title: Turn on prompt caching for Claude
problems: [prompt_cache_miss, context_bloat]
providers: [anthropic]
effort: low
---
## Why

Claude caches a prompt prefix only when you ask for it. Without a cache breakpoint, the
same long system prompt and tool definitions are billed at the full input price on every
call. Cached reads are billed at a small fraction of the input price.

## Fix

Mark the end of the stable part with `cache_control`. Everything up to and including that
block is cached:

```python
response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,               # long and identical across calls
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": question}],
)
```

The cached prefix must be identical byte for byte, so keep timestamps and IDs out of it
(see the stable prompt prefix playbook). Very short prompts are not cached: there is a
minimum cacheable length that depends on the model. `response.usage` reports
`cache_creation_input_tokens` and `cache_read_input_tokens`; Agent Doctor records the read
tokens as cached tokens.

## Verify

`agent-doctor cost RUN` should show cached input tokens on every call after the first, and a
"Prompt caching saved" line.
