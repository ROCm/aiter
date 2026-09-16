# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 MoE numerics across inter_dim 256-alignment.

CK-Tile's 2-stage MXFP4 stage-2 (``moe_cktile2stages_gemm2``) reduces over
``inter_dim`` and indexes its e8m0 weight scales in groups of 8 blocks
(8 * 32 = 256 elements). When ``inter_dim`` is not a multiple of 256 the host
pads the scale group dimension (``shuffle_scale`` rounds it up to a multiple of
8) and the kernel reads the wrong groups, silently returning a badly wrong
result -- no error, no NaN, only wrong numbers. The dispatch-only coverage is
in ``op_tests/test_moe_mxfp4_inter_dim_dispatch.py`` so standard Aiter CI
collects it. This file keeps the GPU numerics cases for FlyDSL toolchains that
can compile the a16w4 port.

Weights are prepared the way the production vLLM AITER_MXFP4_MXFP4 path prepares
them (``shuffle_weight(..., (16, 16))`` + ``e8m0_shuffle``), not with the
a16w4-specific shuffles, because that is the layout the broken shapes actually
arrive in.

Run:
    pytest op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA (gfx942/gfx950) required for the MXFP4 2-stage MoE kernels",
)

NON_256_ALIGNED = [128, 384, 640]

MODEL_DIM = 6144
E = 32
TOPK = 4
TOKEN = 32


def _rel_l2(actual, ref):
    return ((actual.float() - ref.float()).norm() / ref.float().norm()).item()


@_SKIP
@pytest.mark.parametrize("inter_dim", NON_256_ALIGNED)
def test_mxfp4_swiglu_non_256_aligned_numerics(inter_dim):
    """The re-routed shapes must actually compute the right answer."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dtype = dtypes.bf16
    dev = "cuda"

    inp = torch.randn((TOKEN, MODEL_DIM), dtype=dtype, device=dev) / 10
    w1 = torch.randn((E, inter_dim * 2, MODEL_DIM), dtype=dtype, device=dev) / 10
    w2 = torch.randn((E, MODEL_DIM, inter_dim), dtype=dtype, device=dev) / 10
    score = torch.randn((TOKEN, E), dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(inp, score, TOPK, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(E, MODEL_DIM, inter_dim // 2)
    w1_s = w1_s.view(E, inter_dim * 2, MODEL_DIM // 32)
    w2_s = w2_s.view(E, MODEL_DIM, inter_dim // 32)

    ref1 = torch_moe_stage1(
        inp,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_s,
        doweight=False,
    )
    ref = torch_moe_stage2(
        ref1.view(TOKEN, TOPK, inter_dim),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_s,
        a2_scale=None,
        doweight=True,
    )

    # Production AITER_MXFP4_MXFP4 weight preparation.
    f4 = dtypes.fp4x2
    w1_sh = shuffle_weight(w1_q.view(f4), (16, 16))
    w2_sh = shuffle_weight(w2_q.view(f4), (16, 16))
    s0, s1, _ = w1_s.shape
    w1_ss = e8m0_shuffle(w1_s.view(s0 * s1, -1)).view(s0, s1, -1)
    s0, s1, _ = w2_s.shape
    w2_ss = e8m0_shuffle(w2_s.view(s0 * s1, -1)).view(s0, s1, -1)

    out = fused_moe(
        inp,
        w1_sh,
        w2_sh,
        topk_weights,
        topk_ids,
        w1_scale=w1_ss,
        w2_scale=w2_ss,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        doweight_stage1=False,
    )

    assert not out.isnan().any().item(), f"inter_dim={inter_dim}: output has NaN"
    err = _rel_l2(out, ref)
    # mxfp4 quantisation alone lands near 5e-3; the CK-Tile scale mis-indexing
    # this test guards against produces 0.7 or more.
    assert err < 5e-2, f"inter_dim={inter_dim}: rel L2 {err:.6f} too large"
