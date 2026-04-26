# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Derived from ROCm/FlyDSL kernels/moe_blockscale_2stage.py (Apache-2.0).
# Upstream ref: ROCm/FlyDSL commit 8bee73f (verified to contain the source).
"""FlyDSL Blockscale MOE kernel management (per_1x128 FP8/FP8, g1u1).

Mirrors aiter/ops/flydsl/moe_kernels.py for the BLOCKSCALE variant used by
DSV4 and other models with ``weight_block_size: [128, 128]``
(``QuantType.per_1x128``).

Kernel name pattern:
    flydsl_moe{stage}_afp8_wfp8_{out}_blockscale_t{M}x{N}x{K}[_{mode}]

Source of truth: ROCm/FlyDSL ``kernels/moe_blockscale_2stage.py`` —
``compile_moe_blockscale_gemm1`` and ``compile_moe_blockscale_gemm2``.

Sprint 4 (issue sunway513/atom#37 W4.5 accuracy half) ports the kernel
implementation here so aiter's ``fused_moe`` dispatcher can route DSV4's
per_1x128 quant scheme through FlyDSL instead of CK MoE (which has an
ABI mismatch with this checkpoint, producing gibberish outputs).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

_KERNEL_PARAMS: Dict[str, Dict] = {}

# Kernel-name suffix parser: optional waves_per_eu (_w{N}), b_nt (_bnt{N}),
# xcd_swizzle (_xcd{N}). All optional — defaults handled in lookup.
_SUFFIX_RE = re.compile(
    r"(?:_w(?P<wpe>\d+))?(?:_bnt(?P<bnt>\d+))?(?:_xcd(?P<xcd>\d+))?$"
)


def flydsl_blockscale_kernel_name(
    stage: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    waves_per_eu: Optional[int] = None,
    b_nt: Optional[int] = None,
    xcd_swizzle: Optional[int] = None,
    mode: str = "",
) -> str:
    """Construct kernel name.

    Pattern: ``flydsl_moe{stage}_afp8_wfp8_bf16_blockscale_t{M}x{N}x{K}[_{mode}][_w{WPE}][_bnt{B}][_xcd{X}]``.
    """
    name = f"flydsl_moe{stage}_afp8_wfp8_bf16_blockscale_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if waves_per_eu is not None and waves_per_eu != 1:
        name += f"_w{waves_per_eu}"
    if b_nt is not None and b_nt != 0:
        name += f"_bnt{b_nt}"
    if xcd_swizzle is not None and xcd_swizzle > 0:
        name += f"_xcd{xcd_swizzle}"
    return name


def get_flydsl_blockscale_stage1_kernels() -> Dict[str, Dict]:
    """Generate FlyDSL blockscale stage1 kernel registry (FP8/FP8, bf16 out)."""
    kernels: Dict[str, Dict] = {}
    # Tile space: tile_m for token block size; tile_n for inter_dim block;
    # tile_k for K-stride. ScaleBlockK is fixed to 128 (per_1x128 quant).
    for tm in (32, 64, 128):
        for tn in (128, 256):
            # tile_k must be a multiple of ScaleBlockK=128 to align with the
            # block-scale stride. Allow tile_k=128 (single block) and
            # tile_k=256 (two blocks).
            for tk in (128, 256):
                for wpe in (1, 2, 3, 4):
                    for bnt in (0, 2):
                        name = flydsl_blockscale_kernel_name(
                            1, tm, tn, tk, waves_per_eu=wpe, b_nt=bnt
                        )
                        kernels[name] = {
                            "stage": 1,
                            "tile_m": tm,
                            "tile_n": tn,
                            "tile_k": tk,
                            "waves_per_eu": wpe,
                            "b_nt": bnt,
                            "scale_block_k": 128,
                            "a_dtype": "fp8",
                            "b_dtype": "fp8",
                            "out_dtype": "bf16",
                        }
    _KERNEL_PARAMS.update(kernels)
    return kernels


def get_flydsl_blockscale_stage2_kernels() -> Dict[str, Dict]:
    """Generate FlyDSL blockscale stage2 kernel registry (FP8/FP8, bf16 out).

    Stage2 uses ``_atomic`` mode by default (atomic add into output buffer).
    Optional ``_persist`` mode for persistent-CTA execution.
    """
    kernels: Dict[str, Dict] = {}
    for tm in (32, 64, 128):
        for tn in (128, 256):
            for tk in (128, 256):
                for mode in ("atomic", "atomic_persist"):
                    for wpe in (1, 2, 3, 4):
                        for bnt in (0, 2):
                            name = flydsl_blockscale_kernel_name(
                                2, tm, tn, tk,
                                waves_per_eu=wpe, b_nt=bnt, mode=mode,
                            )
                            kernels[name] = {
                                "stage": 2,
                                "tile_m": tm,
                                "tile_n": tn,
                                "tile_k": tk,
                                "waves_per_eu": wpe,
                                "b_nt": bnt,
                                "scale_block_k": 128,
                                "a_dtype": "fp8",
                                "b_dtype": "fp8",
                                "out_dtype": "bf16",
                                "mode": mode,
                            }
    _KERNEL_PARAMS.update(kernels)
    return kernels


def get_flydsl_blockscale_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Auto-populates registry on first call."""
    if not _KERNEL_PARAMS:
        get_flydsl_blockscale_stage1_kernels()
        get_flydsl_blockscale_stage2_kernels()
    return _KERNEL_PARAMS.get(name)


# ---------------------------------------------------------------------------
# Compile + run entry points (Task 2 will fill these in by porting from
# ROCm/FlyDSL kernels/moe_blockscale_2stage.py)
# ---------------------------------------------------------------------------

def compile_moe_blockscale_gemm1(*args, **kwargs):  # pragma: no cover
    """Stage1 GEMM compile entry — ported in Task 2."""
    raise NotImplementedError(
        "compile_moe_blockscale_gemm1 will be ported from FlyDSL upstream "
        "in Task 2 of the W4.5 accuracy plan."
    )


def compile_moe_blockscale_gemm2(*args, **kwargs):  # pragma: no cover
    """Stage2 GEMM compile entry — ported in Task 2."""
    raise NotImplementedError(
        "compile_moe_blockscale_gemm2 will be ported from FlyDSL upstream "
        "in Task 2 of the W4.5 accuracy plan."
    )


def flydsl_moe_blockscale_stage1(*args, **kwargs):  # pragma: no cover
    """Stage1 Python wrapper — implemented in Task 2."""
    raise NotImplementedError(
        "flydsl_moe_blockscale_stage1 will be ported in Task 2."
    )


def flydsl_moe_blockscale_stage2(*args, **kwargs):  # pragma: no cover
    """Stage2 Python wrapper — implemented in Task 2."""
    raise NotImplementedError(
        "flydsl_moe_blockscale_stage2 will be ported in Task 2."
    )
