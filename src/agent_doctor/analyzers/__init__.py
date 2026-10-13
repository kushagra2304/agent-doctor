"""Analyzers: deterministic detectors for wasted cost, time and reliability problems.

>>> from agent_doctor.analyzers import analyze_run
>>> report = analyze_run(run)
>>> for f in report.findings: print(f.severity, f.title)
"""

from agent_doctor.analyzers.base import DEFAULT_THRESHOLDS, Context, Thresholds
from agent_doctor.analyzers.report import DETECTORS, AnalysisReport, analyze_run

__all__ = ["DEFAULT_THRESHOLDS", "DETECTORS", "AnalysisReport", "Context", "Thresholds", "analyze_run"]
