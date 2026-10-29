"""
evaluation/stats.py - Statistical Significance & Confidence Intervals
=====================================================================

Provides pure-Python (no SciPy dependency) statistical utilities:
  - Wilson score interval for binomial proportions (e.g. Hit@k, Answer Correctness)
  - Exact two-sided McNemar paired test for comparing two retrievers/chunking strategies
"""

from __future__ import annotations

import math
from typing import Tuple


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the two-sided Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Total number of trials.
    confidence : float
        Confidence level, default 0.95 (uses z = 1.95996).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) bounded in [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0

    # Common z values or standard normal inverse approx
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    else:
        # Standard approximation for normal quantile
        # rational approximation (Abramowitz and Stegun 26.2.23)
        alpha = 1.0 - confidence
        p = 1.0 - alpha / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def mcnemar_exact(b: int, c: int) -> float:
    """Compute the exact two-sided p-value for McNemar's paired test.

    b : number of items where method 1 succeeded and method 2 failed
    c : number of items where method 1 failed and method 2 succeeded

    Under H0, the discordant pairs follow Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    # Sum binomial probabilities from 0 to k
    # P(X <= k) under Binomial(n, 0.5) = sum_{i=0..k} comb(n, i) * 0.5^n
    cum_prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2.0 * cum_prob)
    return p_value

