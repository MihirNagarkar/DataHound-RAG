"""Unit tests for statistical significance and confidence interval calculations."""
import pytest
from evaluation.stats import wilson_ci, mcnemar_exact


def test_wilson_ci_bounds():
    lower, upper = wilson_ci(8, 8, confidence=0.95)
    assert abs(lower - 0.676) <= 0.002, f"Expected lower ~0.676, got {lower:.4f}"
    assert upper == 1.0

    lower_0, upper_0 = wilson_ci(0, 8, confidence=0.95)
    assert lower_0 == 0.0
    assert abs(upper_0 - (1.0 - 0.676)) <= 0.002

    l_half, u_half = wilson_ci(15, 30, confidence=0.95)
    assert 0.0 < l_half < 0.5 < u_half < 1.0


def test_mcnemar_exact():
    # Prompt specification: mcnemar_exact(b=4, c=0) == 0.125
    p_val = mcnemar_exact(4, 0)
    assert abs(p_val - 0.125) < 1e-6, f"Expected 0.125, got {p_val}"

    # Symmetric case
    assert mcnemar_exact(5, 5) == 1.0

    # No discordance
    assert mcnemar_exact(0, 0) == 1.0

    # Extreme discordance (e.g. b=15, c=0 -> 2 * 0.5^15 ~ 6e-5)
    assert mcnemar_exact(15, 0) < 0.001

