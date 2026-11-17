"""Typed wrapper for ``pytest.approx``."""

import pytest


def approx(
    expected: float,
    rel: float | None = None,
    abs: float | None = None,
    nan_ok: bool = False,
) -> float:
    """Typesafe wrapper around ``pytest.approx``.

    ``pytest.approx`` returns an ``ApproxBase`` that compares equal to
    floats but isn't typed as ``float``. This wrapper hides the cast.

    Args:
        expected: Expected value.
        rel: Relative tolerance.
        abs: Absolute tolerance.
        nan_ok: Whether NaN == NaN.

    Returns:
        A value that compares approximately equal to *expected*.
    """
    return pytest.approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)  # type: ignore
