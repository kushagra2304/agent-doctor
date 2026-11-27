---
id: circuit-breaker
title: Stop calling a service that is down
problems: [tool_retry_loop, swallowed_tool_failure]
effort: medium
---
## Why

When a dependency is down, every run retries it, waits, and fails anyway. Across many runs
this wastes far more time than any single retry policy can save, and it keeps hammering a
service that is trying to recover.

## Fix

Use a circuit breaker: after N failures in a row, skip the call for a cool-down period and
fail fast with a clear error. The agent can then say the data is unavailable right away.

```python
import time

class CircuitBreaker:
    def __init__(self, failures=3, cooldown_s=30):
        self.failures, self.cooldown_s = failures, cooldown_s
        self.count, self.opened_at = 0, None

    def call(self, fn, *args, **kwargs):
        if self.opened_at and time.monotonic() - self.opened_at < self.cooldown_s:
            raise RuntimeError(f"{fn.__name__} is unavailable (circuit open)")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.count += 1
            if self.count >= self.failures:
                self.opened_at = time.monotonic()
            raise
        self.count, self.opened_at = 0, None
        return result

tracking_breaker = CircuitBreaker()
tracking = tracking_breaker.call(get_tracking, order_id, carrier)
```

Keep one breaker per dependency, shared across runs in the same process. Combine it with
backoff for short blips, and with an honest fallback message for the user.

## Verify

`agent-doctor scan` across recent runs: time lost to this tool should drop while it is down.
