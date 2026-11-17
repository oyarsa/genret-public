"""Typed wrappers for numpy operations with imprecise stubs."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def savez(path: Path, data: dict[str, npt.NDArray[Any]]) -> None:
    """Save a dict of arrays as a compressed ``.npz`` file.

    ``np.savez`` expects ``**NDArray`` kwargs, but unpacking a typed dict
    requires a ``cast(dict[str, Any], ...)`` to satisfy checkers. This
    wrapper hides that.

    Args:
        path: Output ``.npz`` path.
        data: Mapping from string keys to arrays.
    """
    np.savez(path, **data)  # type: ignore


def savez_compressed(path: Path, data: dict[str, npt.NDArray[Any]]) -> None:
    """Save a dict of arrays as a compressed ``.npz`` file.

    Same stub problem as :func:`savez`, made worse in numpy 2.5, whose
    signature gained an ``allow_pickle`` keyword that collides with
    unpacked array kwargs.

    Args:
        path: Output ``.npz`` path.
        data: Mapping from string keys to arrays.
    """
    np.savez_compressed(path, **data)  # type: ignore
