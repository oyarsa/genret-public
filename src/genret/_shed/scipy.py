"""Typed wrapper for scipy.stats.spearmanr."""

# pyright: basic

from typing import Any

import numpy.typing as npt
from scipy.stats import spearmanr  # type: ignore[import-untyped]


def spearman_correlation(
    a: npt.NDArray[Any],
    b: npt.NDArray[Any],
) -> float:
    """Compute Spearman rank correlation between two arrays.

    Args:
        a: First array.
        b: Second array (same length as *a*).

    Returns:
        Spearman correlation coefficient (rho).
    """
    result = spearmanr(a, b, axis=None)
    return float(result.correlation)  # type: ignore[union-attr]
