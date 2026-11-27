---
id: trim-context
title: Send the model only what the next step needs
problems: [context_bloat, prompt_cache_miss]
effort: medium
---
## Why

Every call re-sends the whole conversation, including raw tool output (JSON with dozens of
fields the model never uses). Input tokens grow with every step, so cost and latency grow
too, and long, noisy prompts can make answers worse.

## Fix

1. **Compact tool results** before they enter the prompt. Keep the fields that matter:

```python
def compact_order(order: dict) -> str:
    return f"order {order['order_id']}: status={order['status']}, eta={order['eta']}"

facts.append(compact_order(search_orders(order_id)))   # not json.dumps(order)
```

2. **Keep a running state** (a short list of facts) instead of the full history, or keep only
the last few turns plus a summary of older ones.
3. **Store big data outside the prompt** and give the model an ID or a short summary; let it
call a tool if it needs details.

## Verify

`agent-doctor replay RUN --agent ...` shows the input-token change directly. Run
`agent-doctor eval` to confirm the answers stay correct with less context.
