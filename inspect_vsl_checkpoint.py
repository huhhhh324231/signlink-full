# -*- coding: utf-8 -*-
"""Inspect the original VSL checkpoint metadata if it can be unpickled."""

from __future__ import annotations

import sys
import traceback

import numpy as np

# Some checkpoints saved with NumPy 2.x refer to numpy._core while this venv
# currently has NumPy 1.x. These aliases let pickle resolve those module names.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

import torch


def describe(value, indent: str = "") -> None:
    if hasattr(value, "shape"):
        print(f"{indent}shape={tuple(value.shape)} dtype={getattr(value, 'dtype', None)}")
    elif isinstance(value, dict):
        print(f"{indent}dict len={len(value)} keys={list(value.keys())[:30]}")
        for key, item in list(value.items())[:12]:
            print(f"{indent}  {key!r}: {type(item).__name__} {repr(item)[:160]}")
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{type(value).__name__} len={len(value)} first={repr(value[:30])[:500]}")
    else:
        print(f"{indent}{type(value).__name__}: {repr(value)[:500]}")


for path in ["best_model.pth", "best_vsl_transformer.pth"]:
    print("\n" + "=" * 80)
    print("loading", path, flush=True)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        print("loaded:", type(obj).__name__)
        describe(obj)
        if isinstance(obj, dict):
            for key, value in obj.items():
                print("\nKEY:", key)
                describe(value, indent="  ")
    except Exception:
        traceback.print_exc()
