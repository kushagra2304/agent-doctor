"""MCP server: lets Claude, Cursor and other MCP clients ask about recorded runs.

Start it with ``agent-doctor mcp`` (stdio). Every tool is read-only and calls the core
engine; this module only shapes the results into compact JSON for a model to read.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field, ValidationError

from agent_doctor import __version__, core
from agent_doctor._util import to_jsonable
from agent_doctor.pricing import PriceTable
from agent_doctor.schema import Run, Span, SpanKind
from agent_doctor.store import AmbiguousRunId, EvalNotFound, RunNotFound, TraceStore, get_default_store

INSTRUCTIONS = """\
Agent Doctor answers questions about recorded AI agent runs: why a run was expensive,
slow or unreliable, and how to fix it.

- A run ID can be a full ID, a unique prefix of one, or "last" for the most recent run.
- Steps are numbered 1, 2, 3... in start order; findings cite these step numbers as evidence.
- Waste figures are estimates and can overlap between findings, so do not add them up as exact.
- A cost marked incomplete means some models have no price; say so instead of calling them free.

Typical flow: list_runs -> analyze_run (problems and fixes) -> explain_cost (where the money
went) -> get_step (look at the evidence). Use scan_runs for patterns across many runs and
compare_runs to explain why one run differs from another. For how to fix something, use
recommend_fixes (or search_playbooks) and base your advice on the playbooks it returns.
To answer "did the new prompt or model break anything?", use list_evals and get_eval: they
show saved evaluation results, including which test cases broke or got fixed.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)
SHORT = 200  # characters kept from each string in step outlines

RunId = Annotated[str, Field(description='Run ID, a unique prefix of it, or "last".')]
Agent = Annotated[str | None, Field(description="Only this agent's runs.")]


def _iso(epoch_ms: int | None) -> str | None:
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop empty fields so outlines stay short. Zeros are kept: a $0 cost is information."""
    return {k: v for k, v in d.items() if not (v is None or v is False or v in ("", [], {}))}


def _steps(run: Run) -> dict[str, int]:
    return {s.span_id: i for i, s in enumerate(run.ordered_spans, start=1)}


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn expected failures into messages the model can act on."""
    try:
        yield
    except AmbiguousRunId as e:
        raise ToolError(f"{e.args[0]}. Use a longer prefix.") from None
    except RunNotFound as e:
        raise ToolError(f"No run {e.args[0]!r}. Call list_runs to see the available run IDs.") from None
    except EvalNotFound as e:
        raise ToolError(f"No saved evaluation {e.args[0]!r}. Call list_evals to see them; evaluations "
                        "are saved by `agent-doctor eval`.") from None
    except (OSError, ValueError, ValidationError) as e:
        raise ToolError(str(e)) from None


def create_server(store: TraceStore | None = None, prices_file: Path | None = None) -> MCPServer:
    """Build the server. Prices are reloaded on every call, so edits to pricing.yaml apply at once."""

    def db() -> TraceStore:
        return store if store is not None else get_default_store()

    def prices() -> PriceTable:
        return PriceTable.load(prices_file)

    def load(run_id: str) -> Run:
        return core.get_run(run_id, db())

    server = MCPServer("agent-doctor", version=__version__, instructions=INSTRUCTIONS,
                       log_level="WARNING")

    @server.tool(annotations=READ_ONLY)
    def list_runs(
        agent: Agent = None,
        limit: Annotated[int, Field(ge=1, le=200, description="How many runs, newest first.")] = 20,
    ) -> dict[str, Any]:
        """List recent agent runs with their duration, calls, errors, tokens and cost."""
        with _tool_errors():
            s, table = db(), prices()
            rows = []
            for r in core.list_runs(agent, limit, s):
                cost = core.estimate_cost(s.get_run(r.run_id), prices=table)
                rows.append(_compact({
                    "run_id": r.run_id, "agent": r.agent_name, "started": _iso(r.start_ms),
                    "status": r.status, "duration_ms": r.duration_ms, "llm_calls": r.llm_calls,
                    "tool_calls": r.tool_calls, "errors": r.error_count,
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "cost_usd": round(cost.total_cost_usd, 6),
                    "cost_incomplete": not cost.fully_priced,
                    "task": to_jsonable(r.task_input, 120),
                }))
            return {"total_runs": s.count_runs(agent), "agents": s.agents(), "runs": rows}

    @server.tool(annotations=READ_ONLY)
    def analyze_run(
        run_id: RunId,
        max_findings: Annotated[int, Field(ge=1, le=50, description="Most serious first.")] = 10,
    ) -> dict[str, Any]:
        """Find wasted cost, time and reliability problems in one run.

        Returns each problem with its severity, the steps that prove it, the estimated waste
        and a concrete fix.
        """
        with _tool_errors():
            return core.analyze(load(run_id), prices=prices()).summary(max_findings)

    @server.tool(annotations=READ_ONLY)
    def explain_cost(
        run_id: RunId,
        top_steps: Annotated[int, Field(ge=1, le=100, description="How many costliest steps.")] = 10,
    ) -> dict[str, Any]:
        """Break down where a run's money went: per model, per step, cache savings, and the
        projected cost of 1,000 runs like it."""
        with _tool_errors():
            run = load(run_id)
            report = core.estimate_cost(run, prices=prices())
            steps = _steps(run)
            total = report.total_cost_usd
            costly = sorted((s for s in report.spans if s.cost_usd), key=lambda s: -(s.cost_usd or 0))
            out: dict[str, Any] = {
                "run_id": run.run_id, "agent": run.agent_name,
                "total_cost_usd": round(total, 6), "cost_complete": report.fully_priced,
                "llm_cost_usd": round(report.llm_cost_usd, 6),
                "tool_cost_usd": round(report.tool_cost_usd, 6),
                "cache_savings_usd": round(report.cache_savings_usd, 6),
                "tokens": {"input": report.input_tokens, "cached": report.cached_tokens,
                           "output": report.output_tokens},
                "cost_of_1000_runs_usd": round(report.projected_cost_usd(1000), 4),
                "by_model": [m.model_dump() for m in report.by_model],
                "costliest_steps": [_compact({
                    "step": steps.get(s.span_id), "kind": s.kind, "model": s.model, "name": s.name,
                    "input_tokens": s.input_tokens, "cached_tokens": s.cached_tokens,
                    "output_tokens": s.output_tokens, "cost_usd": s.cost_usd,
                    "share": round((s.cost_usd or 0) / total, 3) if total else None,
                }) for s in costly[:top_steps]],
            }
            if report.unpriced_models:
                out["unpriced_models"] = report.unpriced_models
                out["how_to_price"] = ("Add the models to ~/.agent-doctor/pricing.yaml, e.g. "
                                       f'models: [{{match: "{report.unpriced_models[0]}", '
                                       "input: 0.30, output: 0.30}] (USD per 1M tokens).")
            return out

    @server.tool(annotations=READ_ONLY)
    def show_run(run_id: RunId) -> dict[str, Any]:
        """Outline every step of a run in order: LLM and tool calls, timing, tokens, retries,
        errors and tool arguments. Use get_step for a step's full prompt and output."""
        with _tool_errors():
            run = load(run_id)
            steps = _steps(run)
            outline = []
            for span in run.ordered_spans:
                llm = span.kind is SpanKind.LLM
                outline.append(_compact({
                    "step": steps[span.span_id], "parent": steps.get(span.parent_id or ""),
                    "kind": span.kind.value, "name": span.name, "duration_ms": span.duration_ms,
                    "status": span.status.value,
                    "error": to_jsonable(f"{span.error_type}: {span.error_message or ''}", SHORT)
                    if span.is_error else None,
                    "input_tokens": span.input_tokens if llm else None,
                    "cached_tokens": span.cached_tokens or None,
                    "output_tokens": span.output_tokens if llm else None,
                    "retry_of": steps.get(span.retry_of or ""),
                    "side_effects": span.side_effects,
                    "args": to_jsonable(span.tool_args, SHORT),
                    "calls_tools": span.attributes.get("tool_calls"),
                }))
            return _compact({
                "run_id": run.run_id, "agent": run.agent_name, "agent_version": run.agent_version,
                "status": run.status.value, "started": _iso(run.start_ms),
                "duration_ms": run.duration_ms, "task": to_jsonable(run.task_input, 500),
                "final_output": to_jsonable(run.final_output, 500), "tags": run.tags,
                "steps": outline,
            })

    @server.tool(annotations=READ_ONLY)
    def get_step(
        run_id: RunId,
        step: Annotated[int, Field(ge=1, description="Step number, as show_run and findings cite it.")],
        max_chars: Annotated[int, Field(ge=100, le=50_000,
                                        description="Characters kept from each long string.")] = 4000,
    ) -> dict[str, Any]:
        """Everything recorded for one step: prompt messages, model output, tool arguments and
        output, errors, tokens and cost, plus the findings that cite it as evidence."""
        with _tool_errors():
            run = load(run_id)
            ordered = run.ordered_spans
            if step > len(ordered):
                raise ToolError(f"Run {run.run_id} has {len(ordered)} steps; there is no step {step}.")
            span: Span = ordered[step - 1]
            steps = _steps(run)
            table = prices()
            cost = next(s for s in core.estimate_cost(run, prices=table).spans
                        if s.span_id == span.span_id)
            report = core.analyze(run, prices=table)
            data = span.model_dump(mode="json", exclude={"run_id", "parent_id", "retry_of"})
            data.update(step=step, duration_ms=span.duration_ms,
                        parent_step=steps.get(span.parent_id or ""),
                        retry_of_step=steps.get(span.retry_of or ""), cost_usd=cost.cost_usd)
            data["cited_by_findings"] = [
                {"severity": f.severity.value, "title": f.title, "suggestion": f.suggestion}
                for f in report.findings if span.span_id in f.evidence_span_ids
            ]
            return _compact(to_jsonable(data, max_chars))

    @server.tool(annotations=READ_ONLY)
    def scan_runs(
        agent: Agent = None,
        limit: Annotated[int, Field(ge=1, le=1000, description="How many recent runs to analyze.")] = 50,
    ) -> dict[str, Any]:
        """Analyze many recent runs: which problems are most common and most costly, and which
        runs to look at first."""
        with _tool_errors():
            return core.scan(agent, limit, db(), prices()).model_dump(mode="json")

    @server.tool(annotations=READ_ONLY)
    def compare_runs(run_a: RunId, run_b: RunId) -> dict[str, Any]:
        """Compare two runs: differences in cost, time, tokens, calls per tool and per model,
        and which problems appear in only one of them. Deltas are run_b minus run_a."""
        with _tool_errors():
            return core.compare(load(run_a), load(run_b), prices=prices()).model_dump(mode="json")

    @server.tool(annotations=READ_ONLY)
    def recommend_fixes(
        run_id: Annotated[str | None, Field(description='One run (ID, prefix or "last"); omit to cover '
                                            "recent runs.")] = None,
        agent: Agent = None,
        limit: Annotated[int, Field(ge=1, le=500, description="Without run_id: recent runs to cover.")] = 50,
        max_fixes: Annotated[int, Field(ge=1, le=20, description="Most important first.")] = 5,
    ) -> dict[str, Any]:
        """Recommend fixes from Agent Doctor's playbooks, ranked by how much waste they remove.

        Each fix comes with why it matters, how to apply it (with code), and how to verify it
        with replay or eval. Use these playbooks as the source for your advice and cite their IDs.
        """
        with _tool_errors():
            if run_id:
                plan = core.recommend(load(run_id), prices=prices())
            else:
                plan = core.recommend_for_runs(agent, limit, db(), prices())
            data = plan.model_dump(mode="json", exclude={"narrative"})
            data["recommendations"] = data["recommendations"][:max_fixes]
            for r in data["recommendations"]:
                r.pop("score", None)
            return data

    @server.tool(annotations=READ_ONLY)
    def search_playbooks(
        query: Annotated[str, Field(description="What to fix, e.g. 'prompt caching with Claude'.")],
        limit: Annotated[int, Field(ge=1, le=20)] = 3,
    ) -> dict[str, Any]:
        """Search the fix playbooks by text and return the best matches in full."""
        from agent_doctor.recommender import default_knowledge_base

        hits = default_knowledge_base().search(query, limit)
        return {"query": query, "playbooks": [
            {"id": b.id, "title": b.title, "fixes": b.problems, "providers": b.providers or ["any"],
             "effort": b.effort, "body": b.body} for b, _ in hits]}

    @server.tool(annotations=READ_ONLY)
    def list_evals(
        suite: Annotated[str | None, Field(description="Only this suite's evaluations.")] = None,
        limit: Annotated[int, Field(ge=1, le=200, description="How many, newest first.")] = 20,
    ) -> dict[str, Any]:
        """List saved evaluation results (from `agent-doctor eval`), newest first, each with a
        one-line verdict per compared version."""
        with _tool_errors():
            rows = core.list_evals(suite, limit, db())
            return {"evals": [{"eval_id": r.eval_id, "suite": r.suite, "created": _iso(r.created_ms),
                               "result": r.headline} for r in rows]}

    @server.tool(annotations=READ_ONLY)
    def get_eval(
        eval_id: Annotated[str, Field(description='Saved evaluation ID, a prefix, or "last".')] = "last",
        include_cases: Annotated[bool, Field(description="Also return the broken and fixed test cases, "
                                             "with both versions' answers.")] = False,
    ) -> dict[str, Any]:
        """One saved evaluation: pass rates, cost, latency and problems per version, and for each
        comparison the verdict, significance, and which test cases broke or got fixed."""
        with _tool_errors():
            rec = core.get_eval(eval_id, db())
            by_name = {r.name: r for r in rec.reports}
            out: dict[str, Any] = {
                "eval_id": rec.eval_id, "suite": rec.suite, "created": _iso(rec.created_ms),
                "versions": [{
                    "name": r.name, "cases": len(r.results), "pass_rate": round(r.pass_rate, 4),
                    "scorers": {s.name: round(s.pass_rate, 4) for s in r.scorers}, "errors": r.errors,
                    "total_cost_usd": round(r.total_cost_usd, 6), "cost_complete": r.cost_complete,
                    "mean_ms": round(r.mean_ms), "p95_ms": round(r.p95_ms), "problems": r.problems,
                } for r in rec.reports],
                "comparisons": [c.model_dump(mode="json") for c in rec.comparisons],
            }
            if include_cases:
                def case(report_name: str, case_id: str) -> dict[str, Any]:
                    res = by_name[report_name].result(case_id)
                    if res is None:
                        return {}
                    return _compact({"answer": to_jsonable(res.output_text, 300), "error": res.error,
                                     "failed": [f"{s.name}: {s.reason}" if s.reason else s.name
                                                for s in res.scores if not s.passed],
                                     "run_id": res.run_id, "problems": res.problems})

                for c in out["comparisons"]:
                    changes = (("broke", c["regressed"]), ("fixed", c["improved"]))
                    c["changed_cases"] = [{"case_id": cid, "change": change,
                                           "baseline": case(c["baseline"], cid),
                                           "candidate": case(c["candidate"], cid)}
                                          for change, ids in changes for cid in ids[:20]]
            return out

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
