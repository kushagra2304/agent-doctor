---
id: cache-tool-results
title: Cache tool results within a run
problems: [exact_duplicate_tool_call, near_duplicate_tool_call]
effort: low
---
## Why

The agent asks a tool the same question twice in one run. The second call costs time (and
money, if the tool is a paid API) and returns nothing new. It usually happens because the
earlier result fell out of the prompt, or because two code paths fetch the same thing.

## Fix

Memoize tool calls for the length of one run, keyed by tool name and arguments:

```python
import functools, json

def run_cache(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = json.dumps([args, kwargs], sort_keys=True, default=str)
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    wrapper.cache_clear = cache.clear  # call at the start of every run
    return wrapper

@run_cache      # outside @trace_tool, so a cache hit is not recorded as another call
@trace_tool
def search_orders(order_id: str): ...
```

Clear the cache at the start of each run so data never goes stale across users. Do not cache
tools with side effects. If two code paths need the same data, fetch it once and keep it in
the agent's state instead.

## Verify

Replay a run that had the duplicate: the second call should disappear from
`agent-doctor replay RUN --agent ...`, and the answer should not change.
