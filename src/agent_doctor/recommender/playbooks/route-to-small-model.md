---
id: route-to-small-model
title: Use a small model for simple steps
problems: [expensive_model_for_simple_step]
effort: low
---
## Why

A large model is used to produce a one-word decision: a routing label, yes/no, or which tool
to call. Small models handle these steps well at a fraction of the price and latency. Keep
the large model for the step that writes the answer.

## Fix

Pick the model per step instead of per agent:

```python
ROUTER_MODEL = "your-small-model"   # classification, yes/no, tool choice
ANSWER_MODEL = "your-large-model"   # the reply the user reads

intent = llm(ROUTER_MODEL, "Classify: order_status, refund or stock", question)
answer = llm(ANSWER_MODEL, task_prompt, facts)
```

Give the small model a closed set of answers and validate its output, falling back to the
large model if it returns something unexpected.

## Verify

Test the swap on real traffic before changing code: replay with a model map and compare.

```bash
agent-doctor replay RUN --agent my_app:handle -m your-large-model=your-small-model
agent-doctor eval evals/suite.py   # with a Variant(handle, model_map={...})
```

Ship it only if the evaluation shows no broken cases.
