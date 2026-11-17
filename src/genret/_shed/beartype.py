"""Typed wrapper for beartype runtime type checking."""

# pyright: basic

from typing import Any, TypeIs

from beartype.door import is_bearable  # type: ignore[import-untyped]


def is_type[T](obj: object, t: Any) -> TypeIs[T]:  # pyright: ignore[reportInvalidTypeVarUse]
    """Check if *obj* conforms to type *t* at runtime.

    Args:
        obj: Object to check.
        t: Type to check against (may be a generic alias).

    Returns:
        ``True`` if *obj* is bearable as *t*.
    """
    return is_bearable(obj, t)
