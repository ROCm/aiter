"""Native K128 warp-pipeline GEMM for gfx1250; preparation is outside launch timing."""

import importlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
CONFIGS = json.loads((ROOT / "configs.json").read_text())


@lru_cache(None)
def _adapter(split, padded, scale_layout="mcontiguous"):
    variant = "row_padded" if split and padded else "row" if split else "base"
    stem = "mcontiguous_" if scale_layout == "mcontiguous" else "b64_"
    kernel = importlib.import_module("." + stem + variant, __name__)
    name = __name__ + "._adapter_" + scale_layout + "_" + variant
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / ("adapter.py" if scale_layout == "mcontiguous" else "b64_adapter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod._example = lambda: kernel
    return mod


def prepare_gemm_warp_pipeline(
    a, w, a_scales, w_scales, *, scale_layout="mcontiguous", **overrides
):
    if scale_layout not in ("mcontiguous", "b64"):
        raise ValueError("scale_layout must be mcontiguous or b64")
    key = f"{a.shape[0]},{w.shape[0]},{a.shape[1]}"
    cfg = dict(CONFIGS.get(key, {}))
    cfg.update(overrides)
    for ignored in ["scale_mode", "num_warps"]:
        cfg.pop(ignored, None)
    split = cfg.pop("register_split_output", False)
    padded = cfg.pop("output_padding", False)
    if not cfg.get("native_scale128", True):
        raise ValueError("Public runtime requires native K128 scales")
    cfg["native_scale128"] = True
    if split:
        cfg["output_buffers"] = 2
    return _adapter(split, padded, scale_layout).prepare_gemm_warp_pipeline(
        a, w, a_scales, w_scales, **cfg
    )


def gemm_warp_pipeline_cdna5(a, w, a_scales, w_scales, **kwargs):
    return prepare_gemm_warp_pipeline(a, w, a_scales, w_scales, **kwargs)()
