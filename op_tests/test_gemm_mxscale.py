# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for ``aiter.gemm_a8w8_mxscale`` (OCP MXFP8 dense GEMM) on
gfx1250.

The op (E4M3 x E4M3) consumes 1x32 E8M0 block scales and is routed through the
FlyDSL gfx1250 backend. Skipped on non-gfx1250 hardware.
"""

import pytest
import torch

import aiter
from aiter.utility import dtypes, fp4_utils
from aiter.ops.quant import per_1x32_f8_scale_f8_quant

from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.mxscale_layout import SCALE_BLOCK

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="MXScale GEMM ops require a gfx1250 device",
)


def _dequant_fp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    qf = q.view(torch.float8_e4m3fn).to(torch.float32)
    sf = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).to(torch.float32)
    return qf * sf.repeat_interleave(SCALE_BLOCK, dim=1)


def _metrics(out: torch.Tensor, ref: torch.Tensor):
    out_f, ref_f = out.float(), ref.float()
    rel = (out_f - ref_f).abs().sum() / ref_f.abs().sum().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0)
    return rel.item(), cos.item()


@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 256),
        (512, 1024, 512),
        (1, 4096, 4096),
        (333, 512, 1024),  # unaligned M
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_mxscale(M, N, K, dtype):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0

    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )

    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(dtype)
    out = aiter.gemm_a8w8_mxscale(aq, bq, a_s, b_s, dtype=dtype)

    assert out.shape == (M, N)
    assert out.dtype == dtype
    rel, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (M={M},N={N},K={K})"
    assert rel < 0.05, f"rel L1={rel} too high (M={M},N={N},K={K})"


def test_gemm_a8w8_mxscale_out_tensor():
    """The out= path must write into the caller-provided tensor."""
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ret = aiter.gemm_a8w8_mxscale(aq, bq, a_s, b_s, out=out, dtype=torch.bfloat16)
    assert ret.data_ptr() == out.data_ptr()
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low"


def test_gemm_a8w8_mxscale_split_k():
    """split_k > 1 uses the buffer-store atomic-accumulation path."""
    from aiter.ops.flydsl.mxscale_gemm import (
        flydsl_mxscale_gemm,
        flydsl_mxscale_kernel_name,
    )

    torch.manual_seed(0)
    M, N, K = 256, 256, 1024  # K big enough for split_k=2 + num_buffers=2
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(torch.bfloat16)
    name = flydsl_mxscale_kernel_name(
        data_format="fp8",
        out_dtype="bf16",
        tile_m=128,
        tile_n=128,
        tile_k=128,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=2,
    )
    out = flydsl_mxscale_gemm(
        aq, bq, a_s, b_s, data_format="fp8", out_dtype="bf16", kernel_name=name
    )
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"split_k cosine={cos} too low"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
