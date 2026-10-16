# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_dtypes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx1250",
    reason="batched_gemm_a16w8 is a gfx1250 gluon kernel",
)

_, e4m3_type = get_fp8_dtypes()


def generate_inputs(B, M, N, K, strided_x=False, transpose_bm=False, output=True):
    """X (B, M, K) bf16, WQ (B, N, K) fp8, per-tensor w_scale, optional Y.

    strided_x mimics the MLA q_nope operand: a K-slice of an (M, B, K + 64)
    buffer viewed as (B, M, K), so neither the batch nor the M stride is dense.
    """
    torch.manual_seed(0)
    if strided_x:
        buf = torch.randn((M, B, K + 64), dtype=torch.bfloat16, device="cuda")
        x = buf[..., :K].transpose(0, 1)
    else:
        x = torch.randn((B, M, K), dtype=torch.bfloat16, device="cuda")
    weight = (torch.randn((B, N, K), device="cuda") * 0.1).to(e4m3_type)
    w_scale = torch.tensor(0.02, dtype=torch.float32, device="cuda")
    y = None
    if output:
        shape = (M, B, N) if transpose_bm else (B, M, N)
        y = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    return x, weight, w_scale, y


def ref_heads(B):
    """Batch entries checked against the reference: first, last and a spread."""
    return sorted({0, 1, B // 3, B // 2, B - 2, B - 1} & set(range(B)))


def run_torch(x, weight, w_scale, heads):
    """fp32 CPU reference for the selected batch entries, as (len(heads), M, N).

    Computed on the CPU on purpose: on gfx1250 torch's own GPU bmm/matmul
    intermittently returns corrupted rows or faults for the larger shapes
    here (B=128, M=3000) after torch.cuda.empty_cache(), which would make
    the reference, not the kernel under test, the thing being checked.
    """
    xs = x[heads].float().cpu()
    ws = (weight[heads].float() * w_scale.float()).cpu()
    return torch.bmm(xs, ws.transpose(1, 2))


def check(out, x, weight, w_scale, transpose_bm):
    heads = ref_heads(x.shape[0])
    got = (out.transpose(0, 1) if transpose_bm else out)[heads].float().cpu()
    ref = run_torch(x, weight, w_scale, heads)
    # bf16 output rounding: 2^-8 relative, plus a small floor near zero.
    triton.testing.assert_close(ref, got, atol=1e-3, rtol=1e-2)


def run_gluon(x, weight, w_scale, y, transpose_bm, config=None):
    from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import batched_gemm_a16w8

    return batched_gemm_a16w8(
        x, weight, w_scale, YQ=y, transpose_bm=transpose_bm, config=config
    )


# DSR1 MLA absorbed BMMs: W_UK (N=512, K=128) and W_UV (N=128, K=512), 128 heads.
MLA_SHAPES = [
    (128, m, n, k)
    for m in [1, 7, 64, 256, 1536, 3000]
    for (n, k) in [(512, 128), (128, 512)]
]
GENERIC_SHAPES = [
    (16, 1, 128, 64),
    (16, 100, 384, 320),
    (4, 257, 72, 1024),
    (8, 1024, 1024, 1024),
]


@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("transpose_bm", [True, False])
@pytest.mark.parametrize("strided_x", [True, False])
@pytest.mark.parametrize("b, m, n, k", MLA_SHAPES + GENERIC_SHAPES)
def test_batched_gemm_a16w8(b, m, n, k, strided_x, transpose_bm, output):
    torch.cuda.empty_cache()
    x, weight, w_scale, y = generate_inputs(b, m, n, k, strided_x, transpose_bm, output)
    out = run_gluon(x, weight, w_scale, y, transpose_bm)
    if output:
        assert out.data_ptr() == y.data_ptr()
    check(out, x, weight, w_scale, transpose_bm)


def _config(store_mode, bm=128, bn=128, bk=64, num_warps=4, num_buffers=2, wg_per_cu=1):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "num_warps": num_warps,
        "NUM_BUFFERS": num_buffers,
        "WG_PER_CU": wg_per_cu,
        "STORE_MODE": store_mode,
        "C_PAD": 16,
    }


@pytest.mark.parametrize(
    "config",
    [
        _config(1),
        _config(2),
        _config(2, bm=256, bk=128),
        _config(1, bm=64, bn=256, bk=128, num_buffers=3),
        _config(2, bm=64, bn=256, bk=128, num_buffers=3),
        _config(1, bm=64, bn=64, bk=256, wg_per_cu=2),
        _config(2, bm=128, bn=256, bk=64, num_warps=8),
    ],
)
@pytest.mark.parametrize("transpose_bm", [True, False])
@pytest.mark.parametrize(
    "b, m, n, k", [(128, 1536, 512, 128), (128, 1536, 128, 512), (16, 100, 384, 320)]
)
def test_batched_gemm_a16w8_configs(b, m, n, k, transpose_bm, config):
    """Both epilogues (coalesced buffer_store and padded TDM store) over tile shapes."""
    torch.cuda.empty_cache()
    x, weight, w_scale, y = generate_inputs(b, m, n, k, True, transpose_bm)
    out = run_gluon(x, weight, w_scale, y, transpose_bm, config=dict(config))
    check(out, x, weight, w_scale, transpose_bm)
