---
id: dedupe-model-queries
title: Tell the model what it already looked up
problems: [near_duplicate_tool_call, no_progress_loop]
effort: low
---
## Why

The model rewords a query it already ran ("refund policy", then "what is the refund policy
for orders") and gets the same result back. A cache keyed on exact arguments cannot catch
this, because the arguments differ. The model simply does not know, or does not remember,
that it already has the answer.

## Fix

Keep a short list of the queries already made and their result IDs, and put it in the
prompt next to the task:

```python
seen = {}  # query -> short summary of the result

def search(query: str):
    result = search_kb(query)
    seen[query] = ", ".join(r["article_id"] for r in result["results"])
    return result

prompt += "\nAlready searched: " + "; ".join(f"{q!r} -> {ids}" for q, ids in seen.items())
prompt += "\nDo not search again for something already covered above."
```

For a stronger guard, normalize queries before calling the tool (lowercase, drop stop
words) and return the earlier result when the normalized query matches.

## Verify

`agent-doctor replay RUN --agent ...`: the near-duplicate finding should be gone and the
tool-call count should drop by one per duplicate.
