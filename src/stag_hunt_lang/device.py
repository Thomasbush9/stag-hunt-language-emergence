"""Torch device selection shared by local and HPC training entry points."""

from typing import Literal

DeviceName = Literal["auto", "cpu", "mps", "cuda"]


def resolve_torch_device(requested: DeviceName = "auto") -> str:
    """Return an available torch device without importing torch at package import.

    CUDA is preferred over MPS in auto mode so the same experiment CLI naturally
    uses the HPC accelerators. Local Apple Silicon runs fall back to MPS.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyTorch is not installed; run `uv sync --extra train`") from exc

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return requested

