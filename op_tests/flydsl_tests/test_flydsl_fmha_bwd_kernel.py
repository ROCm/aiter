# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the FlyDSL FMHA backward (non-causal, fused dQ/dK/dV).

Two paths, both checked against a torch-autograd reference:
  run_test         — the fused kernel directly (BHSD, fp32 outputs); small + larger shapes.
  run_wrapper_test — the full production wrapper flydsl_flash_attn_backward
                     (BSHD, 3-pass preprocess -> fused kernel -> fp32->dtype cast).

Constraints: bf16 only (kernel hardcodes bf16 MFMA); Sk must be a multiple of
BLOCK_N (64) — max_size buffer resources aren't bounds-checked, so a partial last
K-tile reads garbage.
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

    # Pass 2: fused dQ/dK/dV kernel (outer-K, inner-Q) — writes dQ (atomic), dK, dV
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


def run_wrapper_test(B, H, Sq, D, dtype=torch.bfloat16):
    """Exercise the full production path: flydsl_flash_attn_backward (BSHD layout,
    3-pass preprocess->fused kernel->cast). Output is cast to `dtype`, so tolerance
    is looser than the direct fp32-kernel test above.
    """
    from aiter.ops.flydsl.fmha_bwd_kernels import flydsl_flash_attn_backward

    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    # BSHD layout (what the wrapper expects)
    q = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    do = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")

    # Reference in BHSD (transpose), then torch autograd
    qb, kb, vb, dob = (t.transpose(1, 2) for t in (q, k, v, do))
    o_b, lse = ref_forward(qb, kb, vb, sm_scale)  # o_b: BHSD, lse: [B,H,Sq]
    dq_ref, dk_ref, dv_ref = ref_backward(qb, kb, vb, o_b, dob, lse, sm_scale)
    out = o_b.transpose(1, 2).contiguous()  # BSHD

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    handled = flydsl_flash_attn_backward(
        q, k, v, out, do, lse, dq, dk, dv, sm_scale, causal=False
    )
    assert handled, f"wrapper returned False (unsupported shape B={B} H={H} Sq={Sq})"
    torch.cuda.synchronize()

    # FlyDSL grads are BSHD → transpose to BHSD to compare with the reference
    atol = 2e-2  # output is cast to dtype (bf16/fp16)
    dq_err = (dq.transpose(1, 2).float() - dq_ref).abs().max().item()
    dk_err = (dk.transpose(1, 2).float() - dk_ref).abs().max().item()
    dv_err = (dv.transpose(1, 2).float() - dv_ref).abs().max().item()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(
        f"[wrapper] B={B} H={H} Sq={Sq} D={D} {dtype_str}:"
        f"  dQ_err={dq_err:.2e}  dK_err={dk_err:.2e}  dV_err={dv_err:.2e}"
    )
    assert dq_err < atol, f"dQ max error {dq_err} too large"
    assert dk_err < atol, f"dK max error {dk_err} too large"
    assert dv_err < atol, f"dV max error {dv_err} too large"
    print("  PASS")


if __name__ == "__main__":
    print("== direct fused kernel (BHSD, fp32 out) ==")
    run_test(1, 5, 64, 128, torch.bfloat16)
    run_test(1, 5, 128, 128, torch.bfloat16)
    run_test(1, 5, 512, 128, torch.bfloat16)  # larger shape
    # NOTE: bf16 only — the kernel hardcodes bf16 MFMA + bf16 bit-extraction, so
    # fp16 produces NaN (the wrapper rejects fp16 → Triton fallback).
    # NOTE: Sk must be a multiple of BLOCK_N (64). The kernel uses max_size buffer
    # resources (no bounds-checking), so a partial last K-tile reads garbage and
    # corrupts output — e.g. Sq=Sk=80 gives ~2.0 error. The host wrapper enforces
    # Sq % 64 == 0 for this reason; the 75600 profiler measures timing on garbage.

    print("== host wrapper (BSHD, full 3-pass + cast) ==")
    run_wrapper_test(1, 5, 256, 128, torch.bfloat16)
    run_wrapper_test(1, 5, 512, 128, torch.bfloat16)

    print("All tests passed.")
