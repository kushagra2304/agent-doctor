---
id: surface-tool-failures
title: Tell the model (and the user) when a tool failed
problems: [swallowed_tool_failure]
effort: low
---
## Why

A tool never returned data, yet the run ended "ok" with a confident answer. The model filled
the gap with a guess ("should arrive in 3-5 days"). This is the most dangerous kind of
failure: nothing looks wrong, but the user got made-up information.

## Fix

Put the failure in front of the model explicitly, and give it an honest way out:

```python
try:
    facts.append(f"eta={get_tracking(order_id, carrier)['eta']}")
except ServiceDown:
    facts.append("tracking: UNAVAILABLE (the carrier API is down). Do not guess a date; "
                 "say tracking is temporarily unavailable.")
    degraded = True
```

Also mark the run itself as degraded so it shows up in monitoring, for example by adding a
tag or raising after the reply is sent. Check your evaluation suite for this case: add a
scorer that fails answers containing a date when tracking was unavailable.

## Verify

`agent-doctor eval` with a `contains` or custom scorer on the affected cases, then
`agent-doctor analyze RUN`: the swallowed failure finding should be gone.
