---
id: retry-with-backoff
title: Retry failing tools with exponential backoff and a cap
problems: [tool_retry_loop]
effort: low
---
## Why

A failing call retried immediately usually fails again for the same reason: the service is
overloaded, timing out or briefly down. Instant retries add load at the worst moment and
burn the run's time budget.

## Fix

Wait longer before each attempt, add jitter so many clients do not retry in lockstep, and
cap the attempts. With the `tenacity` library:

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

@trace_tool
@retry(wait=wait_random_exponential(multiplier=0.5, max=8),
       stop=stop_after_attempt(4),
       retry=retry_if_exception_type((TimeoutError, ConnectionError)),
       reraise=True)
def get_tracking(order_id: str, carrier: str): ...
```

Or by hand:

```python
import random, time

for attempt in range(4):
    try:
        return get_tracking(order_id, carrier)
    except (TimeoutError, ConnectionError):
        if attempt == 3:
            raise
        time.sleep(min(8, 0.5 * 2 ** attempt) * random.uniform(0.5, 1.5))
```

Only retry errors that can succeed on a second try (timeouts, 429, 5xx), never validation
errors. Set a timeout on the call itself so one hang cannot eat the whole run.

## Verify

`agent-doctor analyze` should report the retry loop without the "no backoff" note. If the
service is often down, see the circuit breaker playbook.
