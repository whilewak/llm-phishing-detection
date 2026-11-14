"""Detectors. Each module is imported on demand so heavy dependencies
(torch, laya, MLX, API clients) load only for the model being run."""

from __future__ import annotations

import importlib
from typing import Any


def load_detector(spec: str, params: dict[str, Any] | None = None):
    """Build a detector from "package.module:ClassName" and keyword params."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"detector spec must look like 'module:Class', got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**(params or {}))
