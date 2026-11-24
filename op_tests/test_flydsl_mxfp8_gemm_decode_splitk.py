# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-3 decode MXFP8 GEMM (MiniMax-M3.1 TP4 QKV, N 2560 x K 6144, 65-128 rows) against the public GEMM path.

Reference: the same bf16 activation through aiter's MXFP8 quantizer + ``gemm_a8w8_blockscale`` (row-major codes, other K
order, bf16 output). The three fp32 K-third slabs summed agree with it to the bf16 output rounding; the activation codes the
kernel reads are the (16, 16)-layout codes of ``mxfp8_quantize_shuffled`` padded to whole m-blocks (padding rows are
zero and never stored). A ragged row count, rows of a non-zero m-block (odd row-tile count: re-paired A scales) and a CUDA
graph replay with changed inputs are covered.
"""

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight

torch.set_default_device("cuda")
N, K = 2560, 6144
needs_gfx950 = pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 only")


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.05).to(dtypes.fp8)
    sc = torch.randint(120, 128, (N, K // 32), generator=g, dtype=torch.uint8)
    return (
        shuffle_weight(w, layout=(16, 16)),
        sc.view(dtypes.fp8_e8m0),
        shuffle_scale_a16w4(sc.view(dtypes.fp8_e8m0), 1, False),
    )


@needs_gfx950
@pytest.mark.parametrize("m", [65, 72, 80, 81, 88, 96, 97, 112, 128])
def test_splitk_gemm_matches_public_gemm(m):
    from aiter.ops.flydsl.kernels.mxfp8_gemm_decode_splitk import (
        code_rows,
        mxfp8_gemm_decode_splitk,
    )
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled

    wsh, w_scale, chain_scale = _weights()
    g = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn(m, K, generator=g) * 2).to(torch.bfloat16)
    codes, sc = mxfp8_quantize_shuffled(x, code_rows(m))
    slab = mxfp8_gemm_decode_splitk(codes, sc, wsh, chain_scale, m)
    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = gemm_a8w8_blockscale(xq, wsh, xs, w_scale, dtypes.bf16, isBpreshuffled=True)
    tot = slab[0] + slab[1] + slab[2]
    rel = ((tot - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 4e-3, f"m={m}: relative error {rel}"
    assert torch.isfinite(slab).all()
    # a K third is a third of the contraction: no slab equals the whole sum
    assert (slab[0] - tot).abs().max() > 0


@needs_gfx950
def test_splitk_gemm_graph_replay_matches_eager():
    from aiter.ops.flydsl.kernels.mxfp8_gemm_decode_splitk import (
        code_rows,
        mxfp8_gemm_decode_splitk,
    )
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled

    m = 100
    wsh, _, chain_scale = _weights()
    xs_ = [
        (
            torch.randn(m, K, generator=torch.Generator(device="cuda").manual_seed(i))
            * 2
        ).to(torch.bfloat16)
        for i in range(2)
    ]
    xin = xs_[0].clone()
    out = torch.empty(3, m, N, dtype=torch.float32)

    def call():
        codes, sc = mxfp8_quantize_shuffled(xin, code_rows(m))
        mxfp8_gemm_decode_splitk(codes, sc, wsh, chain_scale, m, out=out)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for xi in xs_ * 2:
        xin.copy_(xi)
        call()
        torch.cuda.synchronize()
        eager = out.clone()
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
