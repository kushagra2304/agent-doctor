"""Command-line interface: ``agent-doctor <command>``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agent_doctor import __version__
from agent_doctor.pricing import CostReport, PriceTable, estimate_cost
from agent_doctor.schema import Run, SpanKind, SpanStatus
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
def import_(paths: list[Path] = typer.Argument(..., help="Run JSON files (one run or a list).")) -> None:
    """Load runs from JSON files into the trace database."""
    store = _store()
    total = 0
    for path in paths:
        try:
            ids = store.import_json_file(path)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
