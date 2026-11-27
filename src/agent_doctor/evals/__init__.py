"""Evaluation suite: test cases, scorers, and paired statistics to compare agent versions.

>>> from agent_doctor.evals import cases_from_runs, compare, scorers
>>> cases = cases_from_runs(runs, expected=answers)
>>> recorded, new, cmp = compare(cases, [scorers.exact_match()], baseline=None, candidate=handle_v2)
>>> cmp.headline
'2 case(s) fixed, none broken (p=0.500, could be chance with 12 cases); cost -38%'
"""

from agent_doctor.evals import scorers
from agent_doctor.evals.cases import EvalCase, cases_from_runs, load_cases, save_cases
from agent_doctor.evals.results import EvalRecord, load_results, save_results
from agent_doctor.evals.runner import (
    RECORDED,
    CaseResult,
    Comparison,
    EvalReport,
    ScorerComparison,
    ScorerSummary,
    Variant,
    compare,
    compare_reports,
    evaluate,
)
from agent_doctor.evals.scorers import Attempt, Score, scorer
from agent_doctor.evals.suite import Suite, load_suite

__all__ = [
    "RECORDED",
    "Attempt",
    "CaseResult",
    "Comparison",
    "EvalCase",
    "EvalRecord",
    "EvalReport",
    "Score",
    "ScorerComparison",
    "ScorerSummary",
    "Suite",
    "Variant",
    "cases_from_runs",
    "compare",
    "compare_reports",
    "evaluate",
    "load_cases",
    "load_results",
    "load_suite",
    "save_cases",
    "save_results",
    "scorer",
    "scorers",
]
