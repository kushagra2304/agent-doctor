---
id: llm-rate-limit-backoff
title: Back off on rate-limited LLM calls
problems: [llm_retry_without_backoff]
effort: low
---
## Why

A 429 (rate limit) or overloaded error means "slow down". Retrying at once is almost
guaranteed to fail again, and it counts against the same limit.

## Fix

Let the provider SDK retry for you. The official OpenAI and Anthropic Python clients retry
429 and 5xx errors with exponential backoff and honour the `Retry-After` header; raise the
number of attempts instead of writing your own loop:

```python
from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(max_retries=5)
anthropic_client = Anthropic(max_retries=5)
```

If you call the provider through another library, wrap the call with backoff (see the
`tenacity` example in the tool retry playbook) and read `Retry-After` when it is present.

Rate limits that hit often are a capacity problem: spread traffic over time, cut tokens per
call (see the context trimming playbook), or add a fallback model on a different provider.

## Verify

`agent-doctor analyze` should no longer report "LLM retry without backoff"; the failed
attempts should be spaced out in `agent-doctor show RUN`.
