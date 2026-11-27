"""Command-line interface: ``agent-doctor <command>``."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent_doctor import __version__, core
from agent_doctor.analyzers import AnalysisReport, analyze_run, problem_name
from agent_doctor.pricing import CostReport, PriceTable, estimate_cost
from agent_doctor.schema import Run, Severity, SpanKind, SpanStatus
from agent_doctor.store import AmbiguousRunId, RunNotFound, TraceStore, get_default_store

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Find out why your AI agent is expensive, slow or unreliable.",
)
console = Console()
err = Console(stderr=True)
_state: dict[str, Path | None] = {"db": None}


# -- formatting -------------------------------------------------------------------
def fmt_usd(value: float | None) -> str:
    if value is None:
        return "unpriced"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.6f}".rstrip("0")
    return f"${value:,.4f}"


def fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "-"
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def fmt_time(epoch_ms: int | None) -> str:
    if epoch_ms is None:
        return "-"
    return datetime.fromtimestamp(epoch_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def fmt_int(n: int) -> str:
    return f"{n:,}"


# -- helpers ----------------------------------------------------------------------
def _store() -> TraceStore:
    db = _state["db"]
    return TraceStore(db) if db is not None else get_default_store()


def _load_run(run_id: str) -> Run:
    store = _store()
    try:
        return store.get_run(store.resolve_run_id(run_id))
    except AmbiguousRunId as e:
        err.print(f"[red]error:[/red] {e.args[0]}")
    except RunNotFound:
        err.print(f"[red]error:[/red] no run {run_id!r} in {store.path}")
        err.print("List runs with: agent-doctor runs")
    raise typer.Exit(1)


def _prices(prices_file: Path | None) -> PriceTable:
    try:
        return PriceTable.load(prices_file)
    except (OSError, ValueError, ValidationError) as e:
        err.print(f"[red]error:[/red] could not load prices: {e}")
        raise typer.Exit(1) from None


def _step_numbers(run: Run) -> dict[str, int]:
    return {s.span_id: i for i, s in enumerate(run.ordered_spans, start=1)}


def _depths(run: Run) -> dict[str, int]:
    parents = {s.span_id: s.parent_id for s in run.spans}
    depths: dict[str, int] = {}
    for sid in parents:
        d, cur, seen = 0, parents[sid], set()
        while cur is not None and cur in parents and cur not in seen:
            seen.add(cur)
            d += 1
            cur = parents[cur]
        depths[sid] = d
    return depths


PricesOpt = typer.Option(None, "--prices", help="Extra pricing YAML file (checked first).")


# -- commands ---------------------------------------------------------------------
def _version(value: bool) -> None:
    if value:
        console.print(f"agent-doctor {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    db: Path | None = typer.Option(
        None, "--db", help="Trace database (default ~/.agent-doctor/traces.db or $AGENT_DOCTOR_DB)."
    ),
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    _state["db"] = db


@app.command()
def validate(path: Path = typer.Argument(..., help="Path to a run JSON file.")) -> None:
    """Check that a trace file matches the schema."""
    if not path.is_file():
        err.print(f"[red]error:[/red] file not found: {path}")
        raise typer.Exit(2)
    try:
        run = Run.from_json_file(str(path))
    except ValidationError as e:
        err.print(f"[red]invalid trace:[/red] {path}\n{e}")
        raise typer.Exit(1) from None
    console.print_json(json.dumps(run.summary()))


@app.command("import")
def import_(
    paths: list[Path] = typer.Argument(..., help="Trace files: JSON or JSON lines."),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto, native, otel, langfuse or langsmith."),
    redact: bool = typer.Option(False, "--redact", help="Scrub emails, phone numbers, cards, keys "
                                "and tokens before saving."),
) -> None:
    """Load runs into the trace database: Agent Doctor JSON, OpenTelemetry (OTLP JSON),
    Langfuse or LangSmith exports."""
    store = _store()
    total = 0
    for path in paths:
        try:
            ids = core.import_runs(str(path), fmt, redact, store)  # type: ignore[arg-type]
        except (OSError, ValueError, ValidationError) as e:
            err.print(f"[red]error:[/red] {path}: {e}")
            raise typer.Exit(1) from None
        total += len(ids)
        console.print(f"{path}: imported {', '.join(ids)}")
    console.print(f"[green]{total} run(s) saved to {store.path}[/green]")


@app.command()
def runs(
    agent: str | None = typer.Option(None, "--agent", "-a", help="Only this agent's runs."),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="How many runs to show."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """List recent runs, newest first."""
    store = _store()
    rows = store.list_runs(agent=agent, limit=limit)
    if not rows:
        console.print(f"No runs in {store.path} yet.")
        console.print("Record one with @trace_run, or load a file: agent-doctor import FILE")
        return
    prices = _prices(prices_file)
    table = Table(title=f"Runs ({store.count_runs(agent)} total)", title_justify="left")
    for col, justify in [
        ("Run ID", "left"), ("Agent", "left"), ("Started", "left"), ("Duration", "right"),
        ("LLM", "right"), ("Tools", "right"), ("Errors", "right"), ("Tokens in/out", "right"),
        ("Cost", "right"),
    ]:
        table.add_column(col, justify=justify, no_wrap=col == "Run ID")  # type: ignore[arg-type]
    for r in rows:
        report = estimate_cost(store.get_run(r.run_id), prices)
        cost = fmt_usd(report.total_cost_usd) if report.fully_priced else (
            f"{fmt_usd(report.total_cost_usd)}+" if report.total_cost_usd else "unpriced"
        )
        errors = f"[red]{r.error_count}[/red]" if r.error_count else "0"
        table.add_row(
            r.run_id, r.agent_name, fmt_time(r.start_ms), fmt_ms(r.duration_ms),
            str(r.llm_calls), str(r.tool_calls), errors,
            f"{fmt_int(r.input_tokens)} / {fmt_int(r.output_tokens)}", cost,
        )
    console.print(table)


@app.command()
def show(run_id: str = typer.Argument(..., help="Run ID, a unique prefix of it, or 'last'.")) -> None:
    """Show every step of a run."""
    run = _load_run(run_id)
    steps = _step_numbers(run)
    depths = _depths(run)
    status = "[red]error[/red]" if run.status is SpanStatus.ERROR else "[green]ok[/green]"
    console.print(f"[bold]{run.run_id}[/bold]  agent [bold]{run.agent_name}[/bold]  {status}  "
                  f"{fmt_ms(run.duration_ms)}  started {fmt_time(run.start_ms)}")
    console.print(f"[dim]input:[/dim]  {run.task_input[:300]}")
    if run.final_output:
        console.print(f"[dim]output:[/dim] {run.final_output[:300]}")

    table = Table(show_lines=False)
    for col, justify in [("#", "right"), ("Step", "left"), ("Kind", "left"), ("Duration", "right"),
                         ("Tokens in/out", "right"), ("Status", "left"), ("Notes", "left")]:
        table.add_column(col, justify=justify, no_wrap=col == "Step")  # type: ignore[arg-type]
    for span in run.ordered_spans:
        notes = []
        if span.retry_of:
            notes.append(f"retry of #{steps.get(span.retry_of, '?')}")
        if span.side_effects:
            notes.append("side effects")
        if span.cached_tokens:
            notes.append(f"{fmt_int(span.cached_tokens)} cached")
        if span.time_to_first_token_ms is not None:
            notes.append(f"first token {fmt_ms(span.time_to_first_token_ms)}")
        if span.attributes.get("tool_calls"):
            notes.append("calls " + ", ".join(span.attributes["tool_calls"]))
        if span.error_message:
            notes.append(span.error_message[:80])
        tokens = (f"{fmt_int(span.input_tokens)} / {fmt_int(span.output_tokens)}"
                  if span.kind is SpanKind.LLM else "")
        st = f"[red]{span.error_type or 'error'}[/red]" if span.is_error else "ok"
        table.add_row(
            str(steps[span.span_id]), "  " * depths[span.span_id] + span.name, span.kind.value,
            fmt_ms(span.duration_ms), tokens, st, "; ".join(notes),
        )
    console.print(table)


def _print_cost(report: CostReport, run: Run) -> None:
    steps = _step_numbers(run)
    total = report.total_cost_usd
    table = Table(title="Cost per LLM call", title_justify="left")
    for col, justify in [("#", "right"), ("Model", "left"), ("Input", "right"),
                         ("Cached", "right"), ("Output", "right"), ("Cost", "right"),
                         ("Share", "right")]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for s in report.spans:
        if s.kind != SpanKind.LLM.value and not s.cost_usd:
            continue
        share = f"{s.cost_usd / total:.0%}" if (s.cost_usd and total) else "-"
        table.add_row(
            str(steps.get(s.span_id, "?")), s.model or s.name, fmt_int(s.input_tokens),
            fmt_int(s.cached_tokens), fmt_int(s.output_tokens), fmt_usd(s.cost_usd), share,
        )
    console.print(table)

    if len(report.by_model) > 1:
        bm = Table(title="By model", title_justify="left")
        for col in ["Model", "Calls", "Input", "Output", "Cost"]:
            bm.add_column(col, justify="left" if col == "Model" else "right")
        for m in report.by_model:
            bm.add_row(m.model, str(m.calls), fmt_int(m.input_tokens), fmt_int(m.output_tokens),
                       fmt_usd(m.cost_usd))
        console.print(bm)

    if report.fully_priced:
        console.print(f"[bold]Total: {fmt_usd(total)}[/bold]")
    elif total:
        console.print(f"[bold]Total: {fmt_usd(total)}[/bold] [yellow](priced calls only)[/yellow]")
    else:
        console.print("[bold]Total: unknown[/bold] [yellow](no price for these models)[/yellow]")
    if report.tool_cost_usd:
        console.print(f"  LLM {fmt_usd(report.llm_cost_usd)} + tools {fmt_usd(report.tool_cost_usd)}")
    if report.cache_savings_usd:
        console.print(f"Prompt caching saved {fmt_usd(report.cache_savings_usd)} on this run.")
    if report.most_expensive_span_id and total:
        top = next(s for s in report.spans if s.span_id == report.most_expensive_span_id)
        console.print(f"Most expensive step: #{steps.get(top.span_id)} ({top.model or top.name}), "
                      f"{(top.cost_usd or 0) / total:.0%} of the cost.")
    if total:
        console.print(f"1,000 runs like this: ~{fmt_usd(report.projected_cost_usd(1000))}")
    if report.unpriced_models:
        console.print(f"[yellow]No price for: {', '.join(report.unpriced_models)}[/yellow]")
        example = report.unpriced_models[0]
        console.print("Add one to ~/.agent-doctor/pricing.yaml (USD per 1M tokens):\n"
                      f"[dim]models:\n  - {{match: \"{example}\", input: 0.0, output: 0.0}}[/dim]")


@app.command()
def cost(
    run_id: str = typer.Argument(..., help="Run ID, a unique prefix of it, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Show what a run cost, per step and per model."""
    run = _load_run(run_id)
    report = estimate_cost(run, _prices(prices_file))
    if as_json:
        console.print_json(report.model_dump_json())
        return
    console.print(f"[bold]{run.run_id}[/bold]  agent [bold]{run.agent_name}[/bold]  "
                  f"{fmt_int(report.input_tokens)} tokens in, {fmt_int(report.output_tokens)} out")
    _print_cost(report, run)


@app.command()
def pricing(
    model: str | None = typer.Option(None, "--model", "-m", help="Show which price this model gets."),
    provider: str | None = typer.Option(None, "--provider", help="Provider, used with --model."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """List known model prices (USD per 1M tokens)."""
    table_obj = _prices(prices_file)
    if model:
        entry = table_obj.find(model, provider)
        if entry is None:
            console.print(f"[yellow]No price matches {model!r}.[/yellow]")
            raise typer.Exit(1)
        inp, cached, out = entry.rates(0)
        console.print(f"{model} -> {entry.match!r} from {entry.origin}: "
                      f"input ${inp}, cached ${cached}, output ${out} per 1M tokens")
        return
    table = Table(title=f"Model prices, USD per 1M tokens ({len(table_obj)} entries)",
                  title_justify="left")
    for col in ["Pattern", "Provider", "Input", "Cached", "Output", "Long context", "From"]:
        table.add_column(col, justify="right" if col in {"Input", "Cached", "Output"} else "left")
    for e in table_obj.entries:
        long_ctx = (f"> {fmt_int(e.long_context.above_input_tokens)} in: "
                    f"${e.long_context.input} / ${e.long_context.output}") if e.long_context else ""
        table.add_row(e.match, e.provider or "", f"${e.input}",
                      f"${e.cached_input if e.cached_input is not None else e.input}",
                      f"${e.output}", long_ctx, e.origin if e.origin == "built-in" else Path(e.origin).name)
    console.print(table)


_SEV_STYLE = {Severity.HIGH: "bold red", Severity.MEDIUM: "yellow", Severity.LOW: "cyan",
              Severity.INFO: "dim"}


def _waste(cost: float | None, ms: int | None, tokens: int | None = None) -> str:
    parts = []
    if cost:
        parts.append(f"~{fmt_usd(cost)}")
    if ms:
        parts.append(fmt_ms(ms))
    if tokens:
        parts.append(f"{fmt_int(tokens)} tok")
    return ", ".join(parts) or "-"


def _print_analysis(report: AnalysisReport) -> None:
    status = "[red]error[/red]" if report.status == "error" else "[green]ok[/green]"
    cost = fmt_usd(report.total_cost_usd) if report.fully_priced else (
        "cost unknown (unpriced models)" if not report.total_cost_usd
        else f"{fmt_usd(report.total_cost_usd)}+")
    console.print(f"[bold]{report.run_id}[/bold]  agent [bold]{report.agent_name}[/bold]  {status}  "
                  f"{fmt_ms(report.duration_ms)}  {report.llm_calls} LLM / {report.tool_calls} tool calls"
                  f"  {cost}")
    if not report.findings:
        console.print("[green]No problems found.[/green]")
        return
    counts = report.counts
    sev = ", ".join(f"{counts[k]} {k}" for k in ("high", "medium", "low", "info") if k in counts)
    console.print(f"{len(report.findings)} finding(s): {sev}.  Estimated waste: "
                  f"{_waste(report.wasted_cost_usd, report.wasted_ms, report.wasted_tokens)}"
                  " [dim](estimates can overlap)[/dim]")
    table = Table(show_lines=False)
    for col, justify in [("", "right"), ("Severity", "left"), ("Problem", "left"),
                         ("Steps", "left"), ("Waste", "right")]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for i, f in enumerate(report.findings, start=1):
        steps = ", ".join(f"#{report.step_numbers.get(s, '?')}" for s in f.evidence_span_ids)
        table.add_row(str(i), f"[{_SEV_STYLE[f.severity]}]{f.severity.value}[/]", f.title, steps,
                      _waste(f.wasted_cost_usd, f.wasted_ms, f.wasted_tokens))
    console.print(table)
    console.print("[bold]How to fix[/bold]")
    for i, f in enumerate(report.findings, start=1):
        console.print(f"{i}. [bold]{problem_name(f).capitalize()}[/bold]: {f.suggestion}")
        console.print(f"   [dim]{f.detail}[/dim]")


@app.command()
def analyze(
    run_id: str = typer.Argument(..., help="Run ID, a unique prefix of it, or 'last'."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Find wasted cost, time and reliability problems in a run, with fixes."""
    run = _load_run(run_id)
    report = analyze_run(run, _prices(prices_file))
    if as_json:
        console.print_json(report.model_dump_json())
        return
    _print_analysis(report)


@app.command()
def scan(
    agent: str | None = typer.Option(None, "--agent", "-a", help="Only this agent's runs."),
    limit: int = typer.Option(50, "--limit", "-n", min=1, help="How many recent runs to analyze."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Analyze many runs and show which problems are most common and most costly."""
    store = _store()
    if not store.count_runs(agent):
        console.print(f"No runs in {store.path} yet.")
        return
    report = core.scan(agent, limit, store, _prices(prices_file))
    console.print(f"Analyzed {report.runs_analyzed} run(s): {report.runs_with_problems} with problems, "
                  f"{report.runs_analyzed - report.runs_with_problems} clean. "
                  f"Total cost {fmt_usd(report.total_cost_usd)}.")
    if not report.problems:
        console.print("[green]No problems found.[/green]")
        return
    table = Table(title="Problems across runs", title_justify="left")
    for col, justify in [("Problem", "left"), ("Worst", "left"), ("Runs", "right"),
                         ("Share", "right"), ("Est. waste", "right"), ("Time lost", "right")]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for p in report.problems:
        table.add_row(p.problem, f"[{_SEV_STYLE[p.worst_severity]}]{p.worst_severity.value}[/]",
                      str(p.runs), f"{p.share:.0%}",
                      f"~{fmt_usd(p.wasted_cost_usd)}" if p.wasted_cost_usd else "-",
                      fmt_ms(p.wasted_ms) if p.wasted_ms else "-")
    console.print(table)
    console.print("[bold]Runs to look at first[/bold]")
    for r in report.worst_runs:
        console.print(f"  {r.run_id}  {r.findings} finding(s), est. waste "
                      f"{_waste(r.wasted_cost_usd, r.wasted_ms)}: {r.top_finding}")
    console.print("[dim]Details: agent-doctor analyze RUN_ID[/dim]")


def _model_map(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        old, sep, new = pair.partition("=")
        if not sep or not old or not new:
            err.print(f"[red]error:[/red] --model expects OLD=NEW, got {pair!r}")
            raise typer.Exit(2)
        out[old] = new
    return out


def _delta(a: float, b: float, fmt: Callable[[Any], str]) -> str:
    if a == b:
        return "[dim]same[/dim]"
    pct = f" ({(b - a) / a:+.0%})" if a else ""
    sign, color = ("+", "red") if b > a else ("-", "green")
    return f"[{color}]{sign}{fmt(abs(b - a))}{pct}[/{color}]"


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run ID, a unique prefix of it, or 'last'."),
    agent: str = typer.Option(..., "--agent", help="module:function or file.py:function, "
                              "called with the run's input."),
    model: list[str] = typer.Option([], "--model", "-m", help="Swap a model: OLD=NEW (repeatable)."),
    live_tools: bool = typer.Option(False, "--live-tools", help="Run tools live instead of "
                                    "replaying recorded results (side-effect tools never run)."),
    strict: bool = typer.Option(False, "--strict", help="Fail tool calls that were never recorded "
                                "instead of running them live."),
    no_latency: bool = typer.Option(False, "--no-latency", help="Answer recorded tool calls at once "
                                    "instead of waiting as long as they took."),
    save: bool = typer.Option(False, "--save", help="Save the new run to the trace database."),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Run your agent again on a saved run's input and compare the new run with the old one."""
    from agent_doctor.replay import load_agent

    run = _load_run(run_id)
    try:
        fn = load_agent(agent)
    except Exception as e:
        err.print(f"[red]error:[/red] could not load agent {agent!r}: {e}")
        raise typer.Exit(2) from None
    prices = _prices(prices_file)
    models = _model_map(model)
    result = core.replay(run, fn, _store(), model_map=models or None,
                         tools="live" if live_tools else "recorded", on_miss="error" if strict else "live",
                         tool_latency="none" if no_latency else "recorded", save=save)
    diff = core.compare(run, result.run, prices=prices) if result.run is not None else None
    if as_json:
        console.print_json(json.dumps({
            "original_run_id": run.run_id, "replay_run_id": result.run.run_id if result.run else None,
            "output": result.output_text, "error": result.error, "tool_calls": result.tool_calls,
            "comparison": diff.model_dump(mode="json") if diff else None,
        }))
        return
    console.print(f"Replayed [bold]{run.run_id}[/bold] with {agent}"
                  + (f", models {models}" if models else ""))
    if result.tool_calls:
        console.print("Tool calls: " + ", ".join(f"{n} {k}" for k, n in sorted(result.tool_calls.items())))
    if result.error:
        console.print(f"[red]The agent raised {result.error}[/red]")
    console.print(f"[dim]before:[/dim] {(run.final_output or '')[:300]}")
    same = (result.output_text or "") == (run.final_output or "")
    console.print(f"[dim]after:[/dim]  {(result.output_text or '')[:300]}"
                  + ("  [dim](same)[/dim]" if same else ""))
    if diff is None:
        raise typer.Exit(1)
    a, b = diff.a, diff.b
    table = Table(title="Before and after", title_justify="left")
    for col in ["", "Before", "After", "Change"]:
        table.add_column(col, justify="right" if col else "left")
    rows = [("Duration", a.duration_ms, b.duration_ms, fmt_ms), ("LLM calls", a.llm_calls, b.llm_calls, str),
            ("Tool calls", a.tool_calls, b.tool_calls, str), ("Errors", a.errors, b.errors, str),
            ("Input tokens", a.input_tokens, b.input_tokens, fmt_int),
            ("Output tokens", a.output_tokens, b.output_tokens, fmt_int)]
    for label, x, y, f in rows:
        table.add_row(label, f(x), f(y), _delta(x, y, f))
    cost_known = a.cost_complete and b.cost_complete
    table.add_row("Cost", fmt_usd(a.cost_usd) if a.cost_complete else "unpriced",
                  fmt_usd(b.cost_usd) if b.cost_complete else "unpriced",
                  _delta(a.cost_usd, b.cost_usd, fmt_usd) if cost_known else "-")
    console.print(table)
    if diff.problems_only_in_a:
        console.print("[green]Fixed:[/green] " + ", ".join(diff.problems_only_in_a))
    if diff.problems_only_in_b:
        console.print("[red]New problems:[/red] " + ", ".join(diff.problems_only_in_b))
    if diff.problems_in_both:
        console.print("[yellow]Still there:[/yellow] " + ", ".join(diff.problems_in_both))
    if save and result.run is not None:
        console.print(f"Saved as {result.run.run_id}. Details: agent-doctor analyze {result.run.run_id}")


def _knowledge(extra: list[Path]):
    from agent_doctor.recommender import KnowledgeBase, default_knowledge_base

    try:
        return KnowledgeBase.load(*extra) if extra else default_knowledge_base()
    except (OSError, ValueError, ValidationError) as e:
        err.print(f"[red]error:[/red] could not load playbooks: {e}")
        raise typer.Exit(2) from None


PlaybooksOpt = typer.Option([], "--playbooks", help="Folder of your own playbooks (repeatable).")


@app.command()
def recommend(
    run_id: str | None = typer.Argument(None, help="Run ID, prefix or 'last'. Omit to cover recent runs."),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Without RUN: only this agent's runs."),
    limit: int = typer.Option(50, "--limit", "-n", min=1, help="Without RUN: how many recent runs."),
    llm: str | None = typer.Option(None, "--llm", help="Also have an LLM write the plan: anthropic[:MODEL], "
                                   "openai:MODEL or hf:MODEL."),
    full: bool = typer.Option(False, "--full", help="Show each playbook's whole fix, with code."),
    playbooks: list[Path] = PlaybooksOpt,
    as_json: bool = typer.Option(False, "--json", help="Print the plan as JSON."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Recommend fixes, ranked by how much waste they remove, from Agent Doctor's playbooks."""
    from agent_doctor.recommender import llm_from_spec

    kb = _knowledge(playbooks)
    writer = None
    if llm:
        try:
            writer = llm_from_spec(llm)
        except (ImportError, ValueError) as e:
            err.print(f"[red]error:[/red] {escape(str(e))}")
            raise typer.Exit(2) from None
    prices = _prices(prices_file)
    try:
        if run_id:
            plan = core.recommend(_load_run(run_id), prices=prices, kb=kb, llm=writer)
        else:
            plan = core.recommend_for_runs(agent, limit, _store(), prices, kb=kb, llm=writer)
    except Exception as e:  # the optional LLM call can fail in many provider-specific ways
        if writer is None:
            raise
        err.print(f"[red]error:[/red] the LLM call failed: {type(e).__name__}: {e}")
        raise typer.Exit(1) from None
    if as_json:
        console.print_json(plan.model_dump_json())
        return
    if not plan.recommendations:
        console.print("[green]Nothing to fix: no problems found.[/green]" if not plan.unmatched
                      else "No playbook covers these findings yet.")
    else:
        console.print(f"[bold]Fix plan for {plan.scope}[/bold]"
                      + (f"  [dim](providers: {', '.join(plan.providers)})[/dim]" if plan.providers else ""))
    for i, r in enumerate(plan.recommendations, start=1):
        where = (f"steps {', '.join(f'#{s}' for s in r.steps)}" if r.steps
                 else f"{len(r.runs)} run(s)")
        console.print(f"\n{i}. [{_SEV_STYLE[r.severity]}]{r.title}[/]  [dim]{r.playbook_id}, effort "
                      f"{r.effort}, est. waste {_waste(r.wasted_cost_usd, r.wasted_ms)}[/dim]")
        console.print(f"   Fixes: {', '.join(r.problems)} ({where})")
        fix = (r.fix if full else r.fix.split("\n\n")[0]).replace("**", "")
        console.print("   " + fix.replace("\n", "\n   "), markup=False, highlight=False)
        if r.verify:
            verify = r.verify if full else " ".join(r.verify.split("\n\n")[0].split())
            console.print("   Verify: " + verify.replace("\n", "\n   "), style="dim", markup=False,
                          highlight=False)
    if plan.unmatched:
        console.print(f"\n[yellow]No playbook yet for:[/yellow] {'; '.join(plan.unmatched)}")
    if plan.narrative:
        console.print(f"\n[bold]Plan written by {llm}[/bold]")
        console.print(plan.narrative, markup=False, highlight=False)
    elif plan.recommendations and not full:
        console.print("\n[dim]--full shows the code for each fix; --llm anthropic has an LLM write "
                      "the plan.[/dim]")


@app.command("playbooks")
def playbooks_(
    query: str | None = typer.Argument(None, help="Search text; omit to list every playbook."),
    show: str | None = typer.Option(None, "--show", help="Print one playbook in full, by ID."),
    playbooks: list[Path] = PlaybooksOpt,
) -> None:
    """List, search or read the fix playbooks the recommender uses."""
    kb = _knowledge(playbooks)
    if show:
        book = kb.get(show)
        if book is None:
            err.print(f"[red]error:[/red] no playbook {show!r}. List them with: agent-doctor playbooks")
            raise typer.Exit(1)
        console.print(f"[bold]{book.title}[/bold]  [dim]{book.id}, fixes {', '.join(book.problems)}[/dim]")
        console.print(book.body, markup=False, highlight=False)
        return
    rows = kb.search(query, limit=10) if query else [(b, None) for b in kb.playbooks]
    table = Table(title=f"Playbooks matching {query!r}" if query else f"Playbooks ({len(kb.playbooks)})",
                  title_justify="left")
    for col in ["ID", "Title", "Fixes", "Providers", "Effort"]:
        table.add_column(col)
    for book, _score in rows:
        table.add_row(book.id, book.title, ", ".join(book.problems), ", ".join(book.providers) or "any",
                      book.effort)
    console.print(table)
    console.print("[dim]Read one: agent-doctor playbooks --show ID[/dim]")


def _rate(x: float) -> str:
    return f"{x:.0%}"


@app.command("eval")
def eval_(
    suite_file: Path = typer.Argument(..., help="Python file that defines a Suite."),
    variant: list[str] = typer.Option([], "--variant", "-v", help="Only these variants; the first is "
                                      "the baseline (repeatable)."),
    save: bool = typer.Option(False, "--save", help="Save replayed runs to the trace database."),
    as_json: bool = typer.Option(False, "--json", help="Print the reports as JSON."),
    prices_file: Path | None = PricesOpt,
) -> None:
    """Run an evaluation suite and compare each variant with the first one.

    Exits with code 1 if any case that passed with the baseline fails with another variant.
    """
    from agent_doctor.evals import load_suite

    try:
        suite = load_suite(suite_file)
    except Exception as e:
        err.print(f"[red]error:[/red] could not load {suite_file}: {e}")
        raise typer.Exit(2) from None
    try:
        reports, comparisons = suite.run(variant or None, save=save,
                                         prices=_prices(prices_file) if prices_file else None)
    except ValueError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2) from None
    regressed = any(c.regressed for c in comparisons)
    if as_json:
        console.print_json(json.dumps({"suite": suite.name,
                                       "reports": [r.model_dump(mode="json") for r in reports],
                                       "comparisons": [c.model_dump(mode="json") for c in comparisons]}))
        raise typer.Exit(1 if regressed else 0)

    table = Table(title=f"{suite.name}: {len(reports[0].results)} case(s)", title_justify="left")
    scorer_names = list(dict.fromkeys(s.name for r in reports for s in r.scorers))
    for col in ["Variant", "Pass", *scorer_names, "Errors", "Cost", "Mean time", "p95", "Problems"]:
        table.add_column(col, justify="left" if col == "Variant" else "right")
    for r in reports:
        by_name = {s.name: s for s in r.scorers}
        cost = fmt_usd(r.total_cost_usd) if r.cost_complete else f"{fmt_usd(r.total_cost_usd)}+"
        table.add_row(r.name, _rate(r.pass_rate),
                      *[_rate(by_name[s].pass_rate) if s in by_name else "-" for s in scorer_names],
                      str(r.errors), cost, fmt_ms(int(r.mean_ms)), fmt_ms(int(r.p95_ms)),
                      str(sum(r.problems.values())))
    console.print(table)
    for c, rep in zip(comparisons, reports[1:], strict=True):
        color = {"better": "green", "worse": "red", "mixed": "yellow", "same": "dim"}[c.verdict]
        console.print(f"[bold]{c.candidate}[/bold] vs {c.baseline}: [{color}]{c.verdict}[/{color}]. "
                      f"{c.headline}")
        for case_id in c.regressed[:10]:
            res = rep.result(case_id)
            failed = [s.name + (f" ({s.reason})" if s.reason else "") for s in res.scores if not s.passed]
            console.print(f"  [red]broke[/red] {case_id}: {'; '.join(failed)[:200]}")
        for case_id in c.improved[:10]:
            console.print(f"  [green]fixed[/green] {case_id}")
        if c.problems_new:
            console.print("  [red]new problems:[/red] "
                          + ", ".join(f"{k} (+{v})" for k, v in c.problems_new.items()))
    raise typer.Exit(1 if regressed else 0)


@app.command()
def mcp(prices_file: Path | None = PricesOpt) -> None:
    """Run the MCP server over stdio, for Claude, Cursor and other MCP clients."""
    try:
        from agent_doctor.mcp_server import create_server
    except ImportError:
        err.print("[red]error:[/red] the MCP server needs the 'mcp' extra: "
                  'pip install "agentdoctor\\[mcp]"')
        raise typer.Exit(1) from None
    if prices_file is not None:
        _prices(prices_file)  # fail now with a clear message, not on the first tool call
    create_server(_store(), prices_file).run("stdio")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
