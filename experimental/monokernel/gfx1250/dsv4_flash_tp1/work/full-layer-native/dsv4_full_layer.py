"""Experimental gfx1250 Flash layer-0 monokernel integration (default off)."""
import os
import sys
from pathlib import Path

def prepare():
    root = Path(os.environ.get("ATOM_DSV4_FULL_LAYER_ROOT", "/work/full-layer-native"))
    if not (root / "dsv4_full_layer_runtime.py").is_file():
        raise RuntimeError(f"Missing full-layer experimental artifacts: {root}")
    sys.path.insert(0, str(root))
    from dsv4_full_layer_runtime import ServingLayer
    return ServingLayer()
