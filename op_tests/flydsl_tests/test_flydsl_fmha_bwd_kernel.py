# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test for FlyDSL FMHA backward main kernel (non-causal, dK/dV/dQ).

Verifies dQ, dK, dV against torch autograd reference.
"""

import torch
import math
from aiter.ops.flydsl.kernels.fmha_bwd_preprocess import (
    build_fmha_bwd_preprocess_module,
)
from aiter.ops.flydsl.kernels.fmha_bwd_kernel import build_fmha_bwd_kernel_module


def ref_forward(q, k, v, sm_scale, causal=False):
    """Reference forward pass, returns O and softmax_lse."""
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    scores = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * sm_scale
    if causal:
        mask = torch.triu(torch.ones(Sq, Sk, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Sq]
    attn = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhmn,bhnd->bhmd", attn, v.float()).to(q.dtype)
    return o, lse.float()


def ref_backward(q, k, v, o, do, lse, sm_scale):
    """Reference backward using torch autograd."""
    B, H, Sq, D = q.shape
    q2 = q.float().detach().requires_grad_(True)
    k2 = k.float().detach().requires_grad_(True)
    v2 = v.float().detach().requires_grad_(True)
    scores = torch.einsum("bhmd,bhnd->bhmn", q2, k2) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhmn,bhnd->bhmd", attn, v2)
    out.backward(do.float())
    return q2.grad, k2.grad, v2.grad


def run_test(B, H, Sq, D, dtype=torch.bfloat16):
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    # BHSD layout
    q = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")
    do = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")

    # Reference forward
    o, lse = ref_forward(q, k, v, sm_scale)  # lse: [B, H, Sq]

    # Reference backward
    dq_ref, dk_ref, dv_ref = ref_backward(q, k, v, o, do, lse, sm_scale)

    # --- FlyDSL backward ---

    # Pass 1: preprocess → delta
    delta = torch.zeros(B, H, Sq, dtype=torch.float32, device="cuda")
    preprocess = build_fmha_bwd_preprocess_module(head_dim=D, dtype=dtype_str)
    sb_o, sh_o, sm_o, sd_o = o.stride()
    sb_do, sh_do, sm_do, sd_do = do.stride()
    sb_d, sh_d, sm_d = delta.stride()
    preprocess(
        o,
        do,
        delta,
        sb_o,
        sh_o,
        sm_o,
        sd_o,
        sb_do,
        sh_do,
        sm_do,
        sd_do,
        sb_d,
        sh_d,
        sm_d,
        Sq,
        B,
        H,
    )
    torch.cuda.synchronize()

    # Pass 2: main backward → dQ, dK, dV
    dq = torch.zeros(B, H, Sq, D, dtype=torch.float32, device="cuda")
    dk = torch.zeros(B, H, Sq, D, dtype=torch.float32, device="cuda")
    dv = torch.zeros(B, H, Sq, D, dtype=torch.float32, device="cuda")

    bwd = build_fmha_bwd_kernel_module(head_dim=D, block_m=16, dtype=dtype_str)

    sq_b, sq_h, sq_m, sq_d = q.stride()
    sk_b, sk_h, sk_n, sk_d = k.stride()
    sv_b, sv_h, sv_n, sv_d = v.stride()
    sdo_b, sdo_h, sdo_m, sdo_d = do.stride()
    sdq_b, sdq_h, sdq_m, sdq_d = dq.stride()
    sdk_b, sdk_h, sdk_n, sdk_d = dk.stride()
    sdv_b, sdv_h, sdv_n, sdv_d = dv.stride()
    sl_b, sl_h, sl_m = lse.stride()
    sdelta_b, sdelta_h, sdelta_m = delta.stride()

    bwd(
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        sm_scale,
        sq_b,
        sq_h,
        sq_m,
        sk_b,
        sk_h,
        sk_n,
        sv_b,
        sv_h,
        sv_n,
        sdo_b,
        sdo_h,
        sdo_m,
        sdq_b,
        sdq_h,
        sdq_m,
        sdk_b,
        sdk_h,
        sdk_n,
        sdv_b,
        sdv_h,
        sdv_n,
        sl_b,
        sl_h,
        sl_m,
        sdelta_b,
        sdelta_h,
        sdelta_m,
        Sq,
        Sq,
        H,
        B,
    )
    torch.cuda.synchronize()

    # Compare
    atol = 1e-2
    dq_err = (dq.float() - dq_ref).abs().max().item()
    dk_err = (dk.float() - dk_ref).abs().max().item()
    dv_err = (dv.float() - dv_ref).abs().max().item()
    print(
        f"B={B} H={H} Sq={Sq} D={D} {dtype_str}:"
        f"  dQ_err={dq_err:.2e}  dK_err={dk_err:.2e}  dV_err={dv_err:.2e}"
    )
    assert dq_err < atol, f"dQ max error {dq_err} too large"
    assert dk_err < atol, f"dK max error {dk_err} too large"
    assert dv_err < atol, f"dV max error {dv_err} too large"
    print("  PASS")


if __name__ == "__main__":
    run_test(1, 5, 64, 128, torch.bfloat16)
    run_test(1, 5, 128, 128, torch.bfloat16)
    print("All tests passed.")
