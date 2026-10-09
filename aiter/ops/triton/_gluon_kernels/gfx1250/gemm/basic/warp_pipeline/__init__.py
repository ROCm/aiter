"""Native K128 M-contiguous GEMM with padded, two-buffer output stores."""

import json
from pathlib import Path

from .adapter import prepare_gemm_warp_pipeline as _prepare

CONFIGS = json.loads((Path(__file__).parent / "configs.json").read_text())


def prepare_gemm_warp_pipeline(a, w, a_scales, w_scales, **overrides):
    key = f"{a.shape[0]},{w.shape[0]},{a.shape[1]}"
    cfg = dict(CONFIGS.get(key, {}))
    cfg.update(overrides)
    for key, required in {
        "scale_mode": "mcontiguous",
        "scale_layout": "mcontiguous",
        "output_buffers": 2,
        "native_scale128": True,
        "register_split_output": True,
        "output_padding": True,
        "num_warps": 8,
    }.items():
        if key in cfg and cfg.pop(key) != required:
            raise ValueError(f"{key} must be {required!r}")
    return _prepare(a, w, a_scales, w_scales, **cfg)


def gemm_warp_pipeline_cdna5(a, w, a_scales, w_scales, **kwargs):
    return prepare_gemm_warp_pipeline(a, w, a_scales, w_scales, **kwargs)()
