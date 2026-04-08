# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL blockscale bpreshuffle GEMM kernel instance definitions for tuning.

Mirrors the pattern in ``flydsl_gemm_a8w8_bpreshuffle_common.py`` but for the
blockscale variant (per-block scales, ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128).
"""

from dataclasses import dataclass
import os


def get_gfx():
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


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    out_dtype: str              # "bf16" | "fp16"
    scale_block_k: int          # 128
    use_cshuffle_epilog: int    # 0 or 1
    use_async_copy: int         # 0 or 1
    waves_per_eu: int           # 0=no hint, 1-4
    sScheduler: str             # "Default"

    @property
    def name(self) -> str:
        dt = {"bf16": "B16", "fp16": "F16"}.get(self.out_dtype, self.out_dtype.upper())
        return "_".join([
            "flydsl",
            "blockscale_bpreshuffle",
            "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
            f"sbk{self.scale_block_k}",
            dt,
            "x".join(map(str, [
                self.use_cshuffle_epilog,
                self.use_async_copy,
                self.waves_per_eu,
            ])),
            self.sScheduler.lower(),
        ])


def _ki(tile_m, tile_n, tile_k,
        cshuffle=0, async_copy=0, waves_per_eu=0,
        scheduler="Default",
        out_dtype="bf16", scale_block_k=128):
    return kernelInstance(
        tile_m, tile_n, tile_k,
        out_dtype, scale_block_k,
        cshuffle, async_copy, waves_per_eu, scheduler,
    )


# fmt: off
_base_tiles_common = [
    # small M (decode / token-gen)
    (16,  64,  256),
    (16,  128, 256),
    # M=32
    (32,  64,  128), (32,  64,  256),
    (32,  128, 128), (32,  128, 256),
    # M=64
    (64,  64,  128), (64,  64,  256),
    (64,  128, 128), (64,  128, 256),
    (64,  256, 128),
    # M=128
    (128, 64,  128), (128, 128, 128),
    (128, 128, 256), (128, 256, 128),
    # M=256
    (256, 128, 128), (256, 256, 128),
]

_CSHUFFLE_VALS   = (0,)
_ASYNC_COPY_VALS = (0,)
_WAVES_PER_EU    = (0,)
# fmt: on


def _build_kernels_list(tiles):
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for tm, tn, tk in tiles:
                    if tk % 128 != 0:
                        continue
                    kl[idx] = _ki(tm, tn, tk, csh, acp, wpe)
                    idx += 1
    return kl


kernels_list_942 = _build_kernels_list(_base_tiles_common)
kernels_list_950 = _build_kernels_list(_base_tiles_common)

default_kernels_dict_942 = {
    (-1): _ki(64, 128, 128),
}
default_kernels_dict_950 = {
    (-1): _ki(64, 128, 128),
}

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
