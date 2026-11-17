"""Numpy array type aliases shared across the processing package.

These aliases encode intended dimensionality in their names for documentation
purposes. The type checker treats them as equivalent — shape is not enforced.
"""

import numpy as np
import numpy.typing as npt

type Float16Array2D = npt.NDArray[np.float16]
"""2-D array of float16 values, e.g. an `(N, N)` or `(N, K)` score matrix."""

type Float32Array2D = npt.NDArray[np.float32]
"""2-D array of float32 values, e.g. an `(N, D)` embedding matrix."""

type Float32Array3D = npt.NDArray[np.float32]
"""3-D array of float32 values, e.g. a padded `(N, max_k, D)` embedding tensor."""

type Int32Array2D = npt.NDArray[np.int32]
"""2-D array of int32 values, e.g. an `(N, K)` candidate index matrix."""
