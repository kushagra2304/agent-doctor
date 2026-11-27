---
title: Agent Doctor
emoji: 🩺
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Find out why your AI agent is expensive, slow or unreliable
---

# Agent Doctor demo

Find out why an AI agent is expensive, slow or unreliable, and how to fix it.

- **Bad agent demo**: a support agent with 36 planted problems across 32 runs. Pick a run to
  see what went wrong, which steps prove it, and the fix plan.
- **Analyze your trace**: upload an Agent Doctor, OpenTelemetry (OTLP JSON), Langfuse or
  LangSmith export. It is analyzed in memory, redacted by default, and never stored.
- **Across all runs**: the most common and most costly problems, and the fixes that matter most.
- **Did the fix work?**: the fixed agent replayed against the recorded tool results.

Source and docs: https://github.com/kushagra2304/agent-doctor

## Updating this Space

From the repository root:

```bash
python space/sync.py      # copies the demo data and precomputes the evaluation
```

Then upload the `space/` folder (including `space/data/`) to the Space repository.
