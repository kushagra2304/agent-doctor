---
id: speed-up-slow-tools
title: Speed up or hide a slow tool
problems: [slow_tool]
effort: medium
---
## Why

One tool takes most of the run. Users wait for it on every request, and no prompt or model
change can fix that.

## Fix

Pick what fits the data:

- **Cache across runs** when the data changes slowly (inventory, policies, catalog). A short
  time-to-live keeps it fresh:

```python
import time

_cache: dict[str, tuple[float, dict]] = {}

def cached_inventory(sku: str, ttl_s: float = 60) -> dict:
    hit = _cache.get(sku)
    if hit and time.monotonic() - hit[0] < ttl_s:
        return hit[1]
    value = check_inventory(sku)
    _cache[sku] = (time.monotonic(), value)
    return value
```

- **Add a timeout** so a slow call fails fast instead of holding the whole run.
- **Start it early**: kick the call off in the background as soon as you know you need it,
  and do the model call in the meantime (see the parallel calls playbook).
- **Fix the tool itself**: an index, a smaller query, or a cheaper endpoint.

## Verify

`agent-doctor scan`: the slow-tool share of run time should drop. Replays keep the recorded
tool latency, so measure this one on new live runs.
