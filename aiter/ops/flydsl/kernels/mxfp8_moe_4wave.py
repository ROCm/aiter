# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Four-wave A8W4 MoE tiles with sorted FP8 activation output."""

from .gemm_mxfp8_4wave import compile_mxfp8_gemm_4w
from .mxfp8_moe_8wave import _store_factory


def compile_mxfp8_moe_gemm_4w(
    *,
    K,
    stage,
    xcd_swizzle=1,
    logical_k=None,
    gather_a=False,
    tile_m=128,
    tile_n=256,
    expert_block_m=128,
    b_k=None,
    dynamic_rows=True,
    swiglu_limit=None,
    activation_type="silu",
    b_dtype="fp4",
    fuse_quant=False,
):
    """Four-wave G1 and compatible G2; G2 keeps the existing BF16 reduction ABI."""
    assert stage in (1, 2) and activation_type == "silu" and b_dtype == "fp4"
    assert dynamic_rows
    assert fuse_quant == (stage == 1)
    store = _store_factory(
        activation=stage == 1,
        transpose=stage == 2,
        activation_type=activation_type,
        swiglu_limit=0.0 if swiglu_limit is None else swiglu_limit,
        fuse_quant=fuse_quant,
    )
    return compile_mxfp8_gemm_4w(
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        expert_block_m=expert_block_m,
        b_k=b_k,
        xcd_swizzle=xcd_swizzle,
        logical_k=logical_k,
        gather_a=gather_a,
        store_factory=store,
    )
