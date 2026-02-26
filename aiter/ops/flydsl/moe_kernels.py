# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management utilities.

Provides:
- flydsl_kernel_name: construct kernel name from compile parameters
- parse_flydsl_kernel_name: parse kernel name back to parameters
- get_flydsl_stage1_kernels: enumerate valid stage1 configurations
- get_flydsl_stage2_kernels: enumerate valid stage2 configurations
- compile_flydsl_moe_stage1: compile stage1 kernel (or return cached)
- compile_flydsl_moe_stage2: compile stage2 kernel (or return cached)
"""

import re
from typing import Dict, Optional

_KERNEL_NAME_RE = re.compile(
    r"flydsl_moe(\d+)_a(\w+)_w(\w+)_(\w+)_t(\d+)x(\d+)x(\d+)(?:_(\w+))?"
)


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
) -> str:
    """Construct a FlyDSL MOE kernel name encoding compile parameters.

    Format: flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}[_{mode}]

    Args:
        stage: 1 or 2
        a_dtype: activation dtype ("fp8", "fp4", "fp16", etc.)
        b_dtype: weight dtype ("fp4", "fp8", "fp16", etc.)
        out_dtype: output dtype ("bf16", "f16", "fp8", etc.)
        tile_m: M tile size (block_m)
        tile_n: N tile size
        tile_k: K tile size
        mode: optional mode suffix for stage2 ("reduce", "atomic")
    """
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    return name


def parse_flydsl_kernel_name(name: str) -> Optional[Dict]:
    """Parse a FlyDSL kernel name into its component parameters.

    Returns None if the name is not a valid FlyDSL kernel name.
    """
    if not name or not name.startswith("flydsl_moe"):
        return None
    m = _KERNEL_NAME_RE.match(name)
    if not m:
        return None
    return {
        "stage": int(m.group(1)),
        "a_dtype": m.group(2),
        "b_dtype": m.group(3),
        "out_dtype": m.group(4),
        "tile_m": int(m.group(5)),
        "tile_n": int(m.group(6)),
        "tile_k": int(m.group(7)),
        "mode": m.group(8) or "",
    }


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return a dict of kernelName -> params for all supported FlyDSL stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [16, 32, 64]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                name = flydsl_kernel_name(1, a_dtype, b_dtype, out_dtype, tm, tn, tk)
                kernels[name] = {
                    "stage": 1,
                    "a_dtype": a_dtype,
                    "b_dtype": b_dtype,
                    "out_dtype": out_dtype,
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "MPerBlock": tm,
                }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return a dict of kernelName -> params for all supported FlyDSL stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [32, 64]
    modes = ["atomic", "reduce"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    kernels[name] = {
                        "stage": 2,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "mode": mode,
                        "MPerBlock": tm,
                    }
    return kernels


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
):
    """Compile FlyDSL stage1 kernel (or return cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            use_cshuffle_epilog=(out_dtype == "fp8"),
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
):
    """Compile FlyDSL stage2 kernel (or return cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )
