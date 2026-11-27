"""A Suite bundles cases, scorers and variants so ``agent-doctor eval FILE.py`` can run it.

```python
# evals/support_bot.py
from agent_doctor.evals import Suite, Variant, load_cases, scorers
from my_app import handle, handle_v2

suite = Suite(
    name="support-bot",
    cases=load_cases("evals/support_bot.yaml"),
    scorers=[scorers.exact_match(), scorers.completed()],
    variants={"recorded": None, "v2-prompt": handle_v2, "mini": Variant(handle, model_map={...})},
)
```

The first variant is the baseline; every other variant is compared with it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_doctor._util import import_path
from agent_doctor.evals.cases import EvalCase
from agent_doctor.evals.runner import Comparison, EvalReport, Variant, compare_reports, evaluate
from agent_doctor.evals.scorers import Scorer
from agent_doctor.pricing import PriceTable
from agent_doctor.schema import Run
from agent_doctor.store import TraceStore


@dataclass
class Suite:
    name: str
    cases: list[EvalCase] | Callable[[], list[EvalCase]]
    scorers: list[Scorer]
    variants: dict[str, Callable[..., Any] | Variant | None]
    store: TraceStore | Mapping[str, Run] | None = None
    prices: PriceTable | None = None
    context: Callable[[], AbstractContextManager] | None = None
    """Entered around each variant's run, e.g. simulated time for a repeatable demo."""
    options: dict[str, Any] = field(default_factory=dict)  # more evaluate() keyword arguments

    def run(self, variants: list[str] | None = None, save: bool = False,
            prices: PriceTable | None = None) -> tuple[list[EvalReport], list[Comparison]]:
        names = variants or list(self.variants)
        unknown = [n for n in names if n not in self.variants]
        if unknown:
            raise ValueError(f"unknown variant(s) {unknown}; this suite has {list(self.variants)}")
        cases = self.cases() if callable(self.cases) else self.cases
        reports = []
        for n in names:
            with self.context() if self.context else contextlib.nullcontext():
                reports.append(evaluate(self.variants[n], cases, self.scorers, name=n, store=self.store,
                                        prices=prices or self.prices, save=save, **self.options))
        return reports, [compare_reports(reports[0], r) for r in reports[1:]]


def load_suite(path: str | Path) -> Suite:
    """Import a Python file and return its Suite (the variable ``suite``, or the only Suite)."""
    module = import_path(path)
    if isinstance(getattr(module, "suite", None), Suite):
        return module.suite
    found = [v for v in vars(module).values() if isinstance(v, Suite)]
    if len(found) != 1:
        raise ValueError(f"{Path(path).name} must define one Suite (name it `suite`); found {len(found)}")
    return found[0]
