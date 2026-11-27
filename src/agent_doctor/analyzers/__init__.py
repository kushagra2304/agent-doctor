"""Analyzers: deterministic detectors for wasted cost, time and reliability problems.

>>> from agent_doctor.analyzers import analyze_run
>>> report = analyze_run(run)
>>> for f in report.findings: print(f.severity, f.title)
"""

from agent_doctor.analyzers.base import DEFAULT_THRESHOLDS, Context, Thresholds
from agent_doctor.analyzers.report import (
    DETECTORS,
    SEVERITY_RANK,
    AnalysisReport,
    ProblemStats,
    RunDigest,
    ScanReport,
    analyze_run,
    problem_name,
    scan_reports,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DETECTORS",
    "SEVERITY_RANK",
    "AnalysisReport",
    "Context",
    "ProblemStats",
    "RunDigest",
    "ScanReport",
    "Thresholds",
    "analyze_run",
    "problem_name",
    "scan_reports",
]
