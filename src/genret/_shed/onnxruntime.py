"""Typed wrapper for onnxruntime inference sessions."""

# pyright: basic

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime


def create_cpu_session(model_path: Path) -> onnxruntime.InferenceSession:
    """Create a CPU inference session for an ONNX model."""
    return onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )


def run_session(
    session: onnxruntime.InferenceSession,
    output_name: str,
    inputs: dict[str, npt.NDArray[np.int64]],
) -> npt.NDArray[np.float32]:
    """Run a session and return the named output as a float32 array."""
    (out,) = session.run([output_name], inputs)
    return np.asarray(out, dtype=np.float32)
