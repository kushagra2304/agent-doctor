---
id: parallel-tool-calls
title: Run independent tool calls at the same time
problems: [sequential_independent_calls, slow_tool]
effort: low
---
## Why

Several tool calls run one after another even though none of them needs another's result.
The run takes the sum of their durations instead of the longest one.

## Fix

With async tools, use `asyncio.gather`:

```python
customer, promos, policy = await asyncio.gather(
    get_customer(customer_id),
    get_promotions(locale),
    search_kb("refund policy"),
)
```

With sync tools, use a thread pool:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as pool:
    f1 = pool.submit(get_customer, customer_id)
    f2 = pool.submit(get_promotions, locale)
    f3 = pool.submit(search_kb, "refund policy")
    customer, promos, policy = f1.result(), f2.result(), f3.result()
```

Agent Doctor records tool calls from worker threads only if the run context is copied into
them: submit `contextvars.copy_context().run` with the function, or use async tools. Many
models can also request several tool calls in one turn; execute those concurrently too.

## Verify

`agent-doctor show RUN`: the calls should now overlap in time, and the run should be shorter
by roughly the duration of all but the slowest call.
