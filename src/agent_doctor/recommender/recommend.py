"""Turn findings into a ranked fix plan, grounded in the playbooks (retrieval-augmented).

Retrieval is deterministic: a playbook matches a finding when it lists the finding's kind,
ranked by BM25 similarity to the finding's text and boosted when it targets a provider the
run actually used. An LLM is optional; when given, it writes a short plan from the run's
evidence and the retrieved playbooks, citing them by ID.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from agent_doctor.analyzers import SEVERITY_RANK, AnalysisReport, problem_name
from agent_doctor.recommender.knowledge import KnowledgeBase, Playbook, default_knowledge_base
from agent_doctor.schema import Finding, Run, Severity

LLM = Callable[[str], str]

_PROVIDER_HINTS = {"claude": "anthropic", "gpt-": "openai", "o1": "openai", "o3": "openai", "o4": "openai"}


def run_providers(run: Run) -> set[str]:
    """Providers a run used, from span providers and model names."""
    found = set()
    for s in run.llm_spans:
        if s.provider:
            found.add(s.provider.lower())
        model = (s.model or "").lower().rsplit("/", 1)[-1]
        found.update(p for hint, p in _PROVIDER_HINTS.items() if model.startswith(hint))
    return found


class Recommendation(BaseModel):
    playbook_id: str
    title: str
    effort: str
    severity: Severity = Field(description="Worst severity among the findings it addresses.")
    problems: list[str]
    findings: list[str] = Field(description="Titles of the findings it addresses.")
    steps: list[int] = Field(description="Evidence steps, for a single-run plan.")
    runs: list[str] = Field(description="Runs with these findings.")
    wasted_cost_usd: float = Field(description="Sum over its findings; overlaps other recommendations.")
    wasted_ms: int
    why: str
    fix: str
    verify: str
    source: str
    score: float


class FixPlan(BaseModel):
    scope: str
    providers: list[str]
    recommendations: list[Recommendation]
    unmatched: list[str] = Field(description="Findings no playbook covers.")
    narrative: str | None = Field(default=None, description="The LLM-written plan, if an LLM was used.")


@dataclass
class _Item:
    finding: Finding
    run_id: str
    steps: list[int]
    providers: set[str]


def _query(f: Finding) -> str:
    return f"{problem_name(f)} {f.title} {f.detail} {f.suggestion}"


def _candidates(kb: KnowledgeBase, item: _Item, per_finding: int) -> list[tuple[Playbook, float]]:
    kind = str(item.finding.data.get("kind", item.finding.type.value))
    bm25 = kb.bm25(_query(item.finding))
    top = max(bm25) or 1.0
    scored = []
    for book, s in zip(kb.playbooks, bm25, strict=True):
        if book.providers and not (set(book.providers) & item.providers):
            continue  # advice for a provider this run does not use
        if kind not in book.problems:
            continue
        score = 2.0 + s / top + (0.5 if book.providers else 0.0)
        scored.append((book, round(score, 4)))
    scored.sort(key=lambda x: -x[1])
    return scored[:per_finding]


def _plan(items: list[_Item], scope: str, kb: KnowledgeBase, per_finding: int) -> FixPlan:
    recs: dict[str, Recommendation] = {}
    unmatched = []
    for item in items:
        f = item.finding
        matches = _candidates(kb, item, per_finding)
        if not matches:
            unmatched.append(f.title)
            continue
        for book, score in matches:
            rec = recs.get(book.id)
            if rec is None:
                rec = recs[book.id] = Recommendation(
                    playbook_id=book.id, title=book.title, effort=book.effort, severity=f.severity,
                    problems=[], findings=[], steps=[], runs=[], wasted_cost_usd=0.0, wasted_ms=0,
                    why=book.section("Why"), fix=book.section("Fix"), verify=book.section("Verify"),
                    source=book.source, score=score)
            if problem_name(f) not in rec.problems:
                rec.problems.append(problem_name(f))
            if f.title not in rec.findings:
                rec.findings.append(f.title)
            rec.steps = sorted(set(rec.steps) | set(item.steps))
            if item.run_id not in rec.runs:
                rec.runs.append(item.run_id)
            rec.wasted_cost_usd += f.wasted_cost_usd or 0
            rec.wasted_ms += f.wasted_ms or 0
            rec.score = max(rec.score, score)
            if SEVERITY_RANK[f.severity] < SEVERITY_RANK[rec.severity]:
                rec.severity = f.severity
    ranked = sorted(recs.values(), key=lambda r: (SEVERITY_RANK[r.severity], -r.wasted_cost_usd,
                                                  -r.wasted_ms, -len(r.runs), -r.score))
    providers = sorted(set().union(*(i.providers for i in items))) if items else []
    return FixPlan(scope=scope, providers=providers, recommendations=ranked, unmatched=unmatched)


def recommend(
    report: AnalysisReport,
    run: Run,
    kb: KnowledgeBase | None = None,
    llm: LLM | None = None,
    per_finding: int = 2,
) -> FixPlan:
    """A fix plan for one analyzed run."""
    providers = run_providers(run)
    items = [_Item(f, run.run_id, [report.step_numbers[s] for s in f.evidence_span_ids
                                   if s in report.step_numbers], providers) for f in report.findings]
    plan = _plan(items, run.run_id, kb or default_knowledge_base(), per_finding)
    if llm is not None and plan.recommendations:
        plan.narrative = llm(build_prompt(plan, [report]))
    return plan


def recommend_many(
    analyzed: Iterable[tuple[Run, AnalysisReport]],
    kb: KnowledgeBase | None = None,
    llm: LLM | None = None,
    per_finding: int = 1,
) -> FixPlan:
    """One fix plan across many runs: the playbooks that remove the most waste, most often."""
    pairs = list(analyzed)
    items = [_Item(f, run.run_id, [], run_providers(run)) for run, rep in pairs for f in rep.findings]
    plan = _plan(items, f"{len(pairs)} runs", kb or default_knowledge_base(), per_finding)
    if llm is not None and plan.recommendations:
        plan.narrative = llm(build_prompt(plan, [rep for _, rep in pairs[:5]]))
    return plan


def build_prompt(plan: FixPlan, reports: list[AnalysisReport], max_recs: int = 6) -> str:
    """The prompt for the optional plan-writing LLM: evidence first, then the playbooks."""
    lines = [
        "You are reviewing an AI agent's recorded runs. Agent Doctor's detectors found the problems "
        "below, and these playbooks were retrieved as fixes. Write a short, prioritized fix plan "
        "for the developer: what to change first and why, concretely for this agent.",
        "Rules: use only the evidence and playbooks given; cite playbooks by ID in [brackets]; do "
        "not invent numbers; say which change to verify with `agent-doctor replay` or "
        "`agent-doctor eval`. Plain prose with a numbered list, at most 250 words.",
        "",
        f"## Scope: {plan.scope} (providers: {', '.join(plan.providers) or 'unknown'})",
        "",
        "## Findings",
    ]
    for rep in reports:
        lines.append(f"Run {rep.run_id} ({rep.agent_name}): {rep.llm_calls} LLM calls, "
                     f"{rep.tool_calls} tool calls, {rep.duration_ms} ms, cost ${rep.total_cost_usd:.6f}")
        for f in rep.findings[:8]:
            steps = ", ".join(f"#{rep.step_numbers.get(s, '?')}" for s in f.evidence_span_ids)
            lines.append(f"- [{f.severity.value}] {f.title} (steps {steps}): {f.detail}")
    lines += ["", "## Retrieved playbooks"]
    for r in plan.recommendations[:max_recs]:
        lines += [f"### [{r.playbook_id}] {r.title} (effort: {r.effort}; fixes: {', '.join(r.problems)})",
                  r.why, r.fix[:1500], ""]
    return "\n".join(lines)
