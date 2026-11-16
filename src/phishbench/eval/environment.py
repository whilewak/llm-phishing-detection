"""Record the hardware, software and memory a run used."""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
from importlib import metadata
from pathlib import Path

_TRACKED_PACKAGES = ("laya", "torch", "transformers", "mlx", "mlx-lm", "anthropic", "openai", "pydantic")


def peak_rss_mb() -> float:
    """Peak resident memory of this process. Does not include GPU memory."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def gpu_memory_mb() -> dict[str, float] | None:
    """GPU memory held by PyTorch, if torch is already loaded and a GPU is in use.

    On Apple Silicon this is MPS driver memory (where model weights live), which
    RSS does not count.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if torch.backends.mps.is_available():
        return {
            "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / 2**20,
            "mps_current_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
        }
    if torch.cuda.is_available():
        return {
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_current_allocated_mb": torch.cuda.memory_allocated() / 2**20,
        }
    return None


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def describe_environment() -> dict:
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "git_commit": _git_commit(),
    }
