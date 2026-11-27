"""Small, dependency-free statistics for paired comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` from 0 to 100."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test: chance of a split this lopsided if A and B were equal.

    ``only_a`` is how many cases passed with A only, ``only_b`` with B only.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: Sequence[float], level: float = 0.95, samples: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Confidence interval for the mean of paired differences, by resampling (repeatable)."""
    if not diffs:
        return (0.0, 0.0)
    if len(diffs) == 1:
        return (diffs[0], diffs[0])
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(samples))
    alpha = (1 - level) / 2
    return (percentile(means, alpha * 100), percentile(means, (1 - alpha) * 100))
