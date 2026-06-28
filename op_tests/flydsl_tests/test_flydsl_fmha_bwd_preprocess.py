# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test for FlyDSL FMHA backward preprocess kernel.

Verifies that delta = rowsum(O * dO) matches a reference torch computation.
"""

import torch
import pytest
from aiter.ops.flydsl.kernels.fmha_bwd_preprocess import build_fmha_bwd_preprocess_module


def ref_delta(o, do):
    """Reference: delta[b, h, m] = sum_d(O[b,h,m,d] * dO[b,h,m,d])."""
    return (o.float() * do.float()).sum(dim=-1)  # [B, H, Sq]


@pytest.mark.parametrize("B,H,Sq,D,dtype", [
    (1, 5, 128,  128, torch.bfloat16),
    (2, 8, 256,  128, torch.bfloat16),
    (1, 1, 64,   128, torch.float16),
    (1, 5, 1024, 128, torch.bfloat16),
])
def test_fmha_bwd_preprocess(B, H, Sq, D, dtype):
    torch.manual_seed(42)
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"

    # BSHD layout → permute to BHSD for the kernel
    o_bshd  = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    do_bshd = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")

    # Permute to BHSD: [B, H, Sq, D]
    o  = o_bshd.permute(0, 2, 1, 3).contiguous()
    do = do_bshd.permute(0, 2, 1, 3).contiguous()

    delta = torch.zeros(B, H, Sq, dtype=torch.float32, device="cuda")

    # Strides in elements (BHSD layout, contiguous)
    # o: [B, H, Sq, D]  → strides = [H*Sq*D, Sq*D, D, 1]
    stride_ob, stride_oh, stride_om, stride_ok = o.stride()
    stride_dob, stride_doh, stride_dom, stride_dok = do.stride()
    # delta: [B, H, Sq]
    stride_deltab, stride_deltah, stride_deltam = delta.stride()

    launcher = build_fmha_bwd_preprocess_module(head_dim=D, dtype=dtype_str)
    launcher(
        o, do, delta,
        stride_ob,  stride_oh,  stride_om,  stride_ok,
        stride_dob, stride_doh, stride_dom, stride_dok,
        stride_deltab, stride_deltah, stride_deltam,
        Sq, B, H,
    )
    torch.cuda.synchronize()

    ref = ref_delta(o, do)  # [B, H, Sq]

    max_err = (delta - ref).abs().max().item()
    print(f"B={B} H={H} Sq={Sq} D={D} {dtype_str}: max_err={max_err:.2e}")
    assert max_err < 1e-3, f"max error {max_err} too large"


if __name__ == "__main__":
    test_fmha_bwd_preprocess(1, 5, 128,  128, torch.bfloat16)
    test_fmha_bwd_preprocess(2, 8, 256,  128, torch.bfloat16)
    test_fmha_bwd_preprocess(1, 5, 1024, 128, torch.bfloat16)
    print("All tests passed.")
