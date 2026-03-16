# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
import os


def get_gfx():
    """Detect GPU arch: honour GPU_ARCHS env, fall back to chip_info, default gfx942."""
    env = os.environ.get("GPU_ARCHS", "")
    if env and env != "native":
        return env.split(";")[-1].strip()
    try:
        import sys
        this_dir = os.path.dirname(os.path.abspath(__file__))
        jit_utils = os.path.abspath(os.path.join(this_dir, "..", "aiter", "jit", "utils"))
        if jit_utils not in sys.path:
            sys.path.insert(0, jit_utils)
        from chip_info import get_gfx as _get_gfx
        return _get_gfx()
    except Exception:
        return "gfx942"

_DTYPE_SHORT = {
    "fp8": "F8",
    "int8": "I8",
    "bf16": "B16",
    "fp16": "F16",
}


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    q_dtype_a: str          # "fp8" | "int8"
    q_dtype_w: str          # "fp8" | "int8"
    dtype: str              # output dtype: "bf16" | "fp16"
    lds_stage: int          # 1 or 2
    use_cshuffle_epilog: int  # 0 or 1
    use_async_copy: int       # 0 or 1
    waves_per_eu: int         # 0=no hint, 1-4=occupancy limit
    sScheduler: str           # "Default"

    @property
    def name(self) -> str:
        qa = _DTYPE_SHORT.get(self.q_dtype_a, self.q_dtype_a.upper())
        qw = _DTYPE_SHORT.get(self.q_dtype_w, self.q_dtype_w.upper())
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        return "_".join([
            "flydsl",
            "bpreshuflle",
            "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
            qa,
            qw,
            dt,
            "x".join(map(str, [
                self.lds_stage,
                self.use_cshuffle_epilog,
                self.use_async_copy,
                self.waves_per_eu,
            ])),
            self.sScheduler.lower(),
        ])


def _ki(tile_m, tile_n, tile_k, lds_stage,
        cshuffle=0, async_copy=0, waves_per_eu=0,
        scheduler="Default",
        q_dtype_a="fp8", q_dtype_w="fp8", dtype="bf16"):
    return kernelInstance(
        tile_m, tile_n, tile_k,
        q_dtype_a, q_dtype_w, dtype,
        lds_stage, cshuffle, async_copy,
        waves_per_eu, scheduler,
    )


# fmt: off
# ---------------------------------------------------------------------------
# Base tile configurations: (tile_m, tile_n, tile_k)
# ---------------------------------------------------------------------------

# lds_stage=2 tiles shared by gfx942 and gfx950
_base_tiles_lds2_common = [
    # small M (decode / token-gen)
    (16,  64,  256), (16,  64,  512),
    (16,  128, 256), (16,  128, 512), (16,  256, 256), (16,  256, 512),
    (16,  512, 256), (16,  192, 256),
    # M=32
    (32,  64,  128), (32,  64,  256), (32,  64,  512), (32,  128, 128),
    (32,  128, 256), (32,  192, 128), (32,  192, 256), (32,  256, 128),
    (32,  256, 256),
    # M=48
    (48,  64,  256), (48,  128, 256), (48,  192, 256), (48,  256, 256),
    # M=64
    (64,  64,  128), (64,  64,  256), (64,  128, 128), (64,  128, 256),
    (64,  192, 128), (64,  192, 256), (64,  256, 64),  (64,  256, 128),
    (64,  256, 256),
    # M=80
    (80,  64,  256), (80,  128, 256), (80,  192, 256), (80,  256, 256),
    # M=96
    (96,  64,  128), (96,  64,  256), (96,  128, 128), (96,  128, 256),
    (96,  192, 128), (96,  192, 256), (96,  256, 128), (96,  256, 256),
    # M=112
    (112, 64,  256), (112, 128, 256), (112, 192, 256), (112, 256, 256),
    # M=128
    (128, 64,  128), (128, 64,  256), (128, 128, 64),  (128, 128, 128),
    (128, 128, 256), (128, 192, 128), (128, 192, 256), (128, 256, 128),
    # M=160/192/224/256
    (160, 192, 128),
    (192, 64,  128), (192, 128, 128),
    (224, 64,  128), (224, 128, 128), (224, 192, 128),
    (256, 64,  128), (256, 128, 128), (256, 192, 128),
]

# gfx950 has one extra lds_stage=2 tile
_base_tiles_lds2_950_extra = [
    (256, 256, 128),
]

# lds_stage=1 tiles (same for both archs)
_base_tiles_lds1 = [
    (16,  64,  256), (16,  64,  512),
    (16,  128, 256), (16,  128, 512), (16,  256, 256), (16,  256, 512),
    (16,  512, 256),
    (32,  64,  128), (32,  64,  256), (32,  64,  512), (32,  128, 128),
    (32,  128, 256),
    (64,  64,  128), (64,  64,  256), (64,  128, 128), (64,  128, 256),
    (64,  256, 128),
    (128, 64,  128), (128, 128, 128), (128, 128, 256), (128, 256, 128),
]

# ---------------------------------------------------------------------------
# Combo sweep: lds_stage x cshuffle x async_copy x waves_per_eu
# ---------------------------------------------------------------------------
_LDS_STAGES      = (1, 2)
_CSHUFFLE_VALS   = (0, 1)
_ASYNC_COPY_VALS = (0, 1)
_WAVES_PER_EU    = (0, 1, 2, 3, 4)


def _build_kernels_list(tiles_lds2, tiles_lds1):
    tiles_by_lds = {2: tiles_lds2, 1: tiles_lds1}
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for lds in _LDS_STAGES:
                    for tm, tn, tk in tiles_by_lds[lds]:
                        kl[idx] = _ki(tm, tn, tk, lds, csh, acp, wpe)
                        idx += 1
    return kl


kernels_list_942 = _build_kernels_list(
    _base_tiles_lds2_common, _base_tiles_lds1)
kernels_list_950 = _build_kernels_list(
    _base_tiles_lds2_common + _base_tiles_lds2_950_extra, _base_tiles_lds1)
# fmt: on

default_kernels_dict_942 = {
    (-1): _ki(128,  128,    128,    2, 0, 0, 2, "Default"),
    (-2): _ki(16,   64,     512,    2, 0, 0, 2, "Default"),
    (-3): _ki(32,   64,     512,    2, 0, 0, 2, "Default"),
    (-4): _ki(64,   256,    64,     2, 0, 0, 2, "Default"),
    (-5): _ki(128,  128,    64,     2, 0, 0, 2, "Default"),
    (-6): _ki(128,  64,     128,    2, 0, 0, 2, "Default"),
    (-7): _ki(64,   256,    128,    2, 0, 0, 2, "Default"),
}

default_kernels_dict_950 = {
    (-1): _ki(128,  256,    256,    2, 0, 0, 2, "Default"),
    (-2): _ki(16,   64,     512,    2, 0, 0, 2, "Default"),
    (-3): _ki(32,   64,     512,    2, 0, 0, 2, "Default"),
    (-4): _ki(128,  128,    128,    2, 0, 0, 2, "Default"),
}

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
