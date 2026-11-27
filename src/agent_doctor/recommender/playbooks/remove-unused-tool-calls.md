---
id: remove-unused-tool-calls
title: Remove tool calls whose results are never used
problems: [unused_tool_output]
effort: low
---
## Why

A tool is called, but nothing from its result reaches any later model call or the final
answer. It is pure waste: time, rate-limit budget and sometimes money. Typical causes are a
"fetch just in case" step, a leftover from an older prompt, or a result that is fetched and
then dropped by a formatting step.

## Fix

Decide which of the two it is:

1. **Not needed:** delete the call, or make it conditional on the question type.
2. **Needed but lost:** pass the fields the model needs into the prompt. Keep it short:

```python
profile = get_customer(customer_id)
facts.append(f"customer: {profile['name']}, tier {profile['tier']}")
```

If the data is only sometimes useful, let the model ask for it as a tool call instead of
fetching it up front.

## Verify

Run your evaluation suite with and without the call (`agent-doctor eval`): pass rates should
not drop, and time per run should.
