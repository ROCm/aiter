#!/usr/bin/env python3
"""Attention backward fp8 test — @flyc.kernel API.

Kernel implementation lives in `kernels/attn_bwd_mxfp8_gfx950.py`.
"""

import logging
import torch
import pytest

from aiter.ops.triton.quant.mxfp8_quant import downcast_to_mxfp8, upcast_from_mxfp8
from aiter.ops.flydsl.kernels.attn_bwd_mxfp8_gfx950 import compile_attn_bwd_mxfp8_gfx950
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())

def check_result(test_out, ref_out, atol=0.01, rtol=0.01, pass_pct=95.0):
    """Compare outputs and print result. Returns (passed, max_delta, pct_close)."""
    close_mask = torch.isclose(test_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct
    if passed:
        return True
    
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")


def mx_quant(x, dim=-1):
    x_fp8, x_scale = downcast_to_mxfp8(x, torch.float8_e4m3fn, dim)
    x_fp32 = upcast_from_mxfp8(x_fp8, x_scale, torch.float32, dim)
    return x_fp32.contiguous(), x_fp8.contiguous(), x_scale.contiguous()


def run_torch(q_fp32_head, q_fp32_m, k_fp32_head, k_fp32_n, v, do_fp32_head, do_fp32_m, m, D, sm_scale, causal, dtype=torch.float32):
    seqlen = q_fp32_head.shape[1]
    device = q_fp32_head.device
    v_f32 = v.to(torch.float32)
    qk = torch.matmul(q_fp32_head, k_fp32_head.transpose(-2, -1)) * sm_scale
    p = torch.exp(qk - m[:, :, None])
    if causal:
        mask = torch.tril(torch.ones((seqlen, seqlen), device=device)) #.T
        p[:, mask == 0] = 0.0
    
    ppT, _, _ = mx_quant(p, -2)
    ppT = ppT.transpose(-2, -1)
    dv =  torch.matmul(ppT, do_fp32_m)
    dp = torch.matmul(do_fp32_head, v_f32.transpose(-2, -1))
    ds = p * (dp - D[:, :, None])
    dsT, _, _ = mx_quant(ds, -1)
    dsT = dsT.transpose(-2, -1)
    ds, _, _ = mx_quant(ds, -2)
    dk = torch.matmul(dsT, q_fp32_m) * sm_scale
    dq = torch.matmul(ds, k_fp32_n) * sm_scale

    return dq, dk, dv


@pytest.mark.parametrize("batch", [2, 8, 45, 256])
@pytest.mark.parametrize("seqlen", [128, 1024, 1152, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("tile_m", [64, 128])
@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_attn_bwd_flyc(
    batch, seqlen, head_dim,
    tile_m, tile_n, 
    causal,
    waves_per_eu: int = 0,
):
    tile_head = head_dim
    if tile_m == 128 and tile_head == 128:
        pytest.skip("Too large block size")

    torch.manual_seed(0)

    sm_scale = 0.5
    _wpe = int(waves_per_eu)
    launch_fn = compile_attn_bwd_mxfp8_gfx950(
        seqlen=seqlen, head_dim=head_dim,
        tile_m=tile_m, tile_n=tile_n, tile_head=tile_head,
        sm_scale=sm_scale,
        causal=causal,
        waves_per_eu=_wpe,
    )

    device = torch.device("cuda")
    q_fp32 = torch.randn(batch, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    k_fp32 = torch.randn(batch, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    v_fp32 = torch.randn(batch, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    o_fp32 = torch.randn(batch, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5
    do_fp32 = torch.randn(batch, seqlen, head_dim, device=device, dtype=torch.float32) * 0.5

    qk = q_fp32 @ k_fp32.transpose(-2, -1)
    qk = qk * sm_scale
    m = qk.max(dim=-1)[0]
    p = (qk - m[:, :, None]).exp()
    l = p.sum(dim=-1)
    p = p / l[:, :, None]
    o_fp32 = torch.matmul(p, v_fp32)
    m = m + torch.log(l)
    D = (o_fp32 * do_fp32).sum(dim=-1)

    q_fp32_head, q_quant_head, q_scale_head = mx_quant(q_fp32, -1)
    q_fp32_m, q_quant_m, q_scale_m = mx_quant(q_fp32, -2)
    k_fp32_head, k_quant_head, k_scale_head = mx_quant(k_fp32, -1)
    k_fp32_n, k_quant_n, k_scale_n = mx_quant(k_fp32, -2)
    v_fp32, v_quant, v_scale = mx_quant(v_fp32)
    do_fp32_head, do_quant_head, do_scale_head = mx_quant(do_fp32, -1)
    do_fp32_m, do_quant_m, do_scale_m = mx_quant(do_fp32, -2)
    
    dq_ref, dk_ref, dv_ref = run_torch(q_fp32_head, q_fp32_m, k_fp32_head, k_fp32_n, v_fp32, do_fp32_head, do_fp32_m, m, D, sm_scale, causal, dtype=torch.float32)
    dq_fly = torch.zeros((batch, seqlen, head_dim), dtype=torch.float32, device=device)
    dk_fly = torch.zeros((batch, seqlen, head_dim), dtype=torch.bfloat16, device=device)
    dv_fly = torch.zeros((batch, seqlen, head_dim), dtype=torch.bfloat16, device=device)

    def launch_kernel(dq, dk, dv, q_quant_head, q_scale_head, q_quant_m, q_scale_m, k_quant_head, k_scale_head, k_quant_n, k_scale_n, v, v_scale, do_quant_head, do_scale_head, do_quant_m, do_scale_m, m, D, batch):
        launch_fn(
            dq.contiguous().view(-1),
            dk.contiguous().view(-1),
            dv.contiguous().view(-1),
            q_quant_head.contiguous().view(-1),
            q_scale_head.contiguous().view(-1),
            q_quant_m.contiguous().view(-1),
            q_scale_m.contiguous().view(-1),
            k_quant_head.contiguous().view(-1),
            k_scale_head.contiguous().view(-1),
            k_quant_n.contiguous().view(-1),
            k_scale_n.contiguous().view(-1),
            v.contiguous().view(-1),
            v_scale.contiguous().view(-1),
            do_quant_head.contiguous().view(-1),
            do_scale_head.contiguous().view(-1),
            do_quant_m.contiguous().view(-1),
            do_scale_m.contiguous().view(-1),
            m.contiguous().view(-1),
            D.contiguous().view(-1),
            batch,
            torch.cuda.current_stream(),
        )

    launch_kernel(
        dq_fly,
        dk_fly,
        dv_fly,
        q_quant_head,
        q_scale_head,
        q_quant_m,
        q_scale_m,
        k_quant_head,
        k_scale_head,
        k_quant_n,
        k_scale_n,
        v_quant,
        v_scale,
        do_quant_head,
        do_scale_head,
        do_quant_m,
        do_scale_m,
        m,
        D,
        batch
    )

    dq_fly_fp32 = dq_fly.to(torch.float32)
    dk_fly_fp32 = dk_fly.to(torch.float32)
    dv_fly_fp32 = dv_fly.to(torch.float32)

    assert check_result(dq_fly_fp32, dq_ref, rtol=0.01, atol=0.01, pass_pct=99.0)
    assert check_result(dk_fly_fp32, dk_ref, rtol=0.01, atol=0.01, pass_pct=99.0)
    assert check_result(dv_fly_fp32, dv_ref, rtol=0.01, atol=0.01, pass_pct=99.0)
