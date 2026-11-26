# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which FlyDSL stage-1 kernels can rotate their SwiGLU output in-kernel.

``inter_hadamard`` rotates each block of that many activation columns before
the fused fp4/fp8 output quantization. A block has to sit inside the columns
one lane run holds, so narrow tiles are rejected before any tracing.
"""

import pytest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported
from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    _parse_mxfp4_g2_kname,
)

_MXMOE_ARGS = {
    "D_HIDDEN": 6144,
    "D_INTER": 768,
    "BM": 64,
    "use_nt": True,
    "inline_quant": False,
}


def test_mxmoe_names_carry_the_block_size():
    name = "flydsl_mxmoe_g1_a4w4_64x256x256_nt_swiglu_xcd2"
    assert _parse_mxfp4_g1_kname(name)["inter_hadamard"] == 0
    assert _parse_mxfp4_g1_kname(name + "_had128")["inter_hadamard"] == 128
    with pytest.raises(ValueError, match="GEMM1 option"):
        _parse_mxfp4_g2_kname("flydsl_mxmoe_g2_a4w4_64x256x256_atomic_had128")


def test_mxmoe_rotation_needs_bn256():
    _assert_supported(**_MXMOE_ARGS, BN=256, inter_hadamard=128)
    with pytest.raises(NotImplementedError, match="BN256"):
        _assert_supported(**_MXMOE_ARGS, BN=128, inter_hadamard=128)
    with pytest.raises(NotImplementedError, match="power of two"):
        _assert_supported(**_MXMOE_ARGS, BN=256, inter_hadamard=96)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tile_n": 64, "out_dtype": "fp4"}, "does not fit"),
        (
            {"tile_n": 128, "out_dtype": "fp4", "gate_mode": "interleave"},
            "does not fit",
        ),
        ({"tile_n": 128, "out_dtype": "bf16"}, "fused fp4/fp8 output"),
        ({"tile_n": 128, "out_dtype": "fp4", "k_batch": 2}, "fused fp4/fp8 output"),
    ],
)
def test_moe1_rejects_tiles_that_cannot_hold_a_block(kwargs, match):
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import (
        GateMode,
        compile_mixed_moe_gemm1,
    )

    kwargs = dict(kwargs)
    kwargs["gate_mode"] = GateMode(kwargs.get("gate_mode", "separated"))
    with pytest.raises(ValueError, match=match):
        compile_mixed_moe_gemm1(
            model_dim=6144,
            inter_dim=768,
            experts=128,
            topk=4,
            tile_m=32,
            tile_k=256,
            doweight_stage1=False,
            a_dtype="fp4",
            b_dtype="fp4",
            act="swiglu",
            inter_hadamard=128,
            **kwargs,
        )
