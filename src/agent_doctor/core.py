"""Public API of the core engine.

Every interface (SDK, CLI, MCP server, Gradio demo) calls functions from here and
holds no logic of its own. Functions are added phase by phase:

- Phase 1: estimate_cost(run_id)
- Phase 3: analyze(run_id)
- Phase 5: run_eval_suite(...), compare_prompts(...), replay(...)
"""
