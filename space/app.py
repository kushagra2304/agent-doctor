"""Agent Doctor demo for Hugging Face Spaces.

Four tabs: browse the bad agent's 32 recorded runs, upload your own trace (analyzed in
memory and never stored), see the problems across all runs, and see whether the fixes
worked. Run locally with ``python space/app.py`` after ``python space/sync.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_doctor.analyzers import AnalysisReport, analyze_run, problem_name, scan_reports
from agent_doctor.ingest import Redactor, parse_runs, read_json
from agent_doctor.pricing import PriceTable
from agent_doctor.recommender import FixPlan, recommend, recommend_many
from agent_doctor.schema import Run

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _find(name: str, fallback: Path) -> Path:
    local = HERE / "data" / name
    return local if local.is_file() else fallback


RUNS_FILE = _find("runs.json", REPO / "tests" / "fixtures" / "bad_agent" / "runs.json")
LABELS_FILE = _find("labels.json", REPO / "tests" / "fixtures" / "bad_agent" / "labels.json")
PRICES_FILE = _find("demo_pricing.yaml", REPO / "examples" / "demo_pricing.yaml")
EVAL_FILE = HERE / "data" / "eval.json"


def load_demo() -> tuple[list[Run], dict[str, dict], PriceTable]:
    runs = parse_runs(read_json(RUNS_FILE), fmt="native")
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))["runs"] if LABELS_FILE.is_file() else {}
    return runs, labels, PriceTable.load(PRICES_FILE)


def _usd(x: float | None) -> str:
    if x is None:
        return "unpriced"
    return f"${x:.6f}" if x < 0.01 else f"${x:,.4f}"


def run_label(run: Run, labels: dict[str, dict]) -> str:
    info = labels.get(run.run_id, {})
    name = info.get("scenario") or run.tags.get("scenario") or run.agent_name
    return f"{name}  ({run.run_id})"


# -- views (plain data, so they can be tested without Gradio) ---------------------------
def summary_md(run: Run, report: AnalysisReport, labels: dict[str, dict] | None = None) -> str:
    info = (labels or {}).get(run.run_id, {})
    cost = _usd(report.total_cost_usd) + ("" if report.fully_priced else " (some models unpriced)")
    lines = [f"### {run.agent_name} · `{run.run_id}`",
             f"**Status** {run.status.value} · **Time** {run.duration_ms:,} ms · **LLM calls** "
             f"{report.llm_calls} · **Tool calls** {report.tool_calls} · **Cost** {cost}",
             f"**Input:** {run.task_input[:300]}", f"**Answer:** {(run.final_output or '-')[:300]}"]
    if info.get("description"):
        lines.append(f"*Planted:* {info['description']}")
    if report.findings:
        lines.append(f"**{len(report.findings)} problem(s)**, estimated waste {_usd(report.wasted_cost_usd)} "
                     f"and {report.wasted_ms:,} ms (estimates can overlap).")
    else:
        lines.append("**No problems found.**")
    return "\n\n".join(lines)


FINDING_HEADERS = ["Severity", "Problem", "What happened", "Steps", "Est. waste"]


def findings_rows(report: AnalysisReport) -> list[list[Any]]:
    rows = []
    for f in report.findings:
        steps = ", ".join(f"#{report.step_numbers.get(s, '?')}" for s in f.evidence_span_ids)
        waste = ", ".join(x for x in [_usd(f.wasted_cost_usd) if f.wasted_cost_usd else "",
                                      f"{f.wasted_ms:,} ms" if f.wasted_ms else ""] if x) or "-"
        rows.append([f.severity.value, problem_name(f), f.title, steps, waste])
    return rows


STEP_HEADERS = ["#", "Step", "Kind", "Duration (ms)", "Tokens in/out", "Status", "Notes"]


def step_rows(run: Run) -> list[list[Any]]:
    steps = {s.span_id: i for i, s in enumerate(run.ordered_spans, start=1)}
    rows = []
    for s in run.ordered_spans:
        notes = []
        if s.retry_of:
            notes.append(f"retry of #{steps.get(s.retry_of, '?')}")
        if s.cached_tokens:
            notes.append(f"{s.cached_tokens:,} cached")
        if s.side_effects:
            notes.append("side effects")
        if s.error_message:
            notes.append(s.error_message[:80])
        tokens = f"{s.input_tokens:,} / {s.output_tokens:,}" if s.kind.value == "llm" else ""
        rows.append([steps[s.span_id], s.name, s.kind.value, s.duration_ms, tokens, s.status.value,
                     "; ".join(notes)])
    return rows


def plan_md(plan: FixPlan, limit: int = 5) -> str:
    if not plan.recommendations:
        return "Nothing to fix." if not plan.unmatched else "No playbook covers these findings yet."
    parts = [f"### Fix plan for {plan.scope}"]
    for i, r in enumerate(plan.recommendations[:limit], start=1):
        where = f"steps {', '.join(f'#{s}' for s in r.steps)}" if r.steps else f"{len(r.runs)} run(s)"
        parts.append(f"**{i}. {r.title}** · `{r.playbook_id}` · {r.severity.value} · effort {r.effort}\n\n"
                     f"Fixes {', '.join(r.problems)} ({where}).\n\n{r.fix}\n\n*Verify:* {r.verify}")
    return "\n\n---\n\n".join(parts)


def analyze_one(run: Run, prices: PriceTable, labels: dict[str, dict] | None = None):
    report = analyze_run(run, prices)
    return (summary_md(run, report, labels), findings_rows(report), plan_md(recommend(report, run)),
            step_rows(run))


def analyze_upload(path: str | None, fmt: str, redact: bool) -> tuple[list[Run], str]:
    """Parse an uploaded trace file. Returns (runs, message)."""
    if not path:
        return [], "Upload a trace file (JSON or JSON lines)."
    p = Path(path)
    if p.stat().st_size > MAX_UPLOAD_BYTES:
        return [], f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MB."
    try:
        runs = parse_runs(read_json(p), fmt=fmt, redact=Redactor() if redact else False)  # type: ignore[arg-type]
    except Exception as e:  # show the problem to the visitor instead of a stack trace
        return [], f"Could not read this file: {type(e).__name__}: {e}"
    if not runs:
        return [], "No runs found in this file."
    note = " Personal data and secrets were redacted." if redact else ""
    return runs, f"Found {len(runs)} run(s).{note}"


SCAN_HEADERS = ["Problem", "Worst", "Runs", "Share", "Est. waste", "Time lost (ms)"]


def scan_view(runs: list[Run], prices: PriceTable) -> tuple[str, list[list[Any]], str]:
    reports = [analyze_run(r, prices) for r in runs]
    scan = scan_reports(reports)
    head = (f"Analyzed **{scan.runs_analyzed}** runs: {scan.runs_with_problems} with problems, "
            f"{scan.runs_analyzed - scan.runs_with_problems} clean. Total cost {_usd(scan.total_cost_usd)}.")
    rows = [[p.problem, p.worst_severity.value, p.runs, f"{p.share:.0%}",
             _usd(p.wasted_cost_usd) if p.wasted_cost_usd else "-", p.wasted_ms] for p in scan.problems]
    plan = recommend_many(list(zip(runs, reports, strict=True)))
    return head, rows, plan_md(plan, limit=6)


EVAL_HEADERS = ["Version", "Correct answers", "No serious problems", "Cost", "Mean time (ms)", "Problems"]


def eval_view() -> tuple[str, list[list[Any]]]:
    if not EVAL_FILE.is_file():
        return "Run `python space/sync.py` to include the evaluation results.", []
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    rows = []
    for r in data["reports"]:
        by = {s["name"]: s for s in r["scorers"]}
        rows.append([r["name"], f"{by.get('exact_match', {}).get('pass_rate', 0):.0%}",
                     f"{by.get('no_problems', {}).get('pass_rate', 0):.0%}", _usd(r["total_cost_usd"]),
                     round(r["mean_ms"]), sum(r["problems"].values())])
    lines = ["Each version answers the same 32 recorded questions. Tool calls are replayed from the "
             "recordings, so every version sees the same data, and a tracking API that was down is "
             "still down."]
    lines += [f"- **{c['candidate']}** vs {c['baseline']}: {c['verdict']}. {c['headline']}"
              for c in data["comparisons"]]
    return "\n".join(lines), rows


# -- the Gradio app ---------------------------------------------------------------------
INTRO = """# Agent Doctor
Find out why your AI agent is expensive, slow or unreliable, and how to fix it.
Twelve deterministic detectors (no LLM calls) find the problems; fixes come from a library
of playbooks. `pip install agent-doctor` to use it on your own agent, from the CLI or from
Claude and Cursor over MCP."""


def build_demo():
    import gradio as gr

    runs, labels, prices = load_demo()
    by_label = {run_label(r, labels): r for r in runs}
    choices = list(by_label)

    def show_demo(label: str):
        return analyze_one(by_label[label], prices, labels)

    def upload(path, fmt, redact):
        found, msg = analyze_upload(path, fmt, redact)
        names = [f"{r.agent_name} ({r.run_id})" for r in found]
        state = {n: r.model_dump(mode="json") for n, r in zip(names, found, strict=True)}
        first = names[0] if names else None
        return msg, gr.update(choices=names, value=first), state

    def show_upload(name, state):
        if not name or not state or name not in state:
            return "", [], "", []
        return analyze_one(Run.model_validate(state[name]), prices)

    with gr.Blocks(title="Agent Doctor") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Bad agent demo"):
            gr.Markdown("A support agent with planted problems. Pick a run.")
            pick = gr.Dropdown(choices, value=choices[0], label="Run")
            summary = gr.Markdown()
            findings = gr.Dataframe(headers=FINDING_HEADERS, label="Problems", wrap=True)
            plan = gr.Markdown()
            steps = gr.Dataframe(headers=STEP_HEADERS, label="Every step", wrap=True)
            pick.change(show_demo, pick, [summary, findings, plan, steps])
            demo.load(show_demo, pick, [summary, findings, plan, steps])
        with gr.Tab("Analyze your trace"):
            gr.Markdown("Agent Doctor JSON, OpenTelemetry (OTLP JSON), Langfuse or LangSmith exports. "
                        "Files are analyzed in memory and not stored.")
            with gr.Row():
                file = gr.File(label="Trace file", file_types=[".json", ".jsonl"], type="filepath")
                fmt = gr.Dropdown(["auto", "native", "otel", "langfuse", "langsmith"], value="auto",
                                  label="Format")
                redact = gr.Checkbox(value=True, label="Redact personal data and secrets")
            msg = gr.Markdown()
            uploaded = gr.State({})
            which = gr.Dropdown([], label="Run")
            u_summary = gr.Markdown()
            u_findings = gr.Dataframe(headers=FINDING_HEADERS, label="Problems", wrap=True)
            u_plan = gr.Markdown()
            u_steps = gr.Dataframe(headers=STEP_HEADERS, label="Every step", wrap=True)
            gr.Button("Analyze").click(upload, [file, fmt, redact], [msg, which, uploaded])
            which.change(show_upload, [which, uploaded], [u_summary, u_findings, u_plan, u_steps])
        with gr.Tab("Across all runs"):
            head, rows, all_plan = scan_view(runs, prices)
            gr.Markdown(head)
            gr.Dataframe(value=rows, headers=SCAN_HEADERS, label="Problems across runs", wrap=True)
            gr.Markdown(all_plan)
        with gr.Tab("Did the fix work?"):
            text, eval_rows = eval_view()
            gr.Markdown(text)
            gr.Dataframe(value=eval_rows, headers=EVAL_HEADERS, label="Evaluation", wrap=True)
    return demo


if __name__ == "__main__":
    build_demo().launch()
