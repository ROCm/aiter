# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the decode small-M MX-FP8 GEMM ops (gfx950).

Each op (``mxfp8_gemv`` dense, ``smallm_mxfp8_moe_grouped_gemm`` MoE via the
``grouped_gemm_mxfp8`` wrapper) is compared against a PyTorch reference that
dequantizes the SAME e4m3/e8m0 inputs the kernel reads (so the only difference
is fp8 matrix-core accumulation vs bf16 matmul -> ~28 dB / cosine ~0.999).

gfx950-only: the kernels use mfma_scale_f32_16x16x128_f8f6f4 and a host
gfx950 guard; on other archs they raise, so the module is skipped there.
"""
import pytest
import torch

from aiter.ops.smallm_gemm_mxfp8 import grouped_gemm_mxfp8, mxfp8_gemv

DEVICE = "cuda"
FP8_MAX = 448.0  # e4m3 max magnitude


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return ""


requires_gfx950 = pytest.mark.skipif(
    "gfx95" not in _gcn_arch(),
    reason="decode small-M MX-FP8 GEMMs are a CDNA4 (gfx95x) feature.",
)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


# ── self-contained MX-FP8 (e4m3 data + e8m0 1x32 scales) quant / dequant ──────
def quant_mxfp8(x: torch.Tensor):
    """x [..., K] -> (q e4m3 [..., K], e8m0 uint8 [..., K//32])."""
    K = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], K // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp)
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    e8m0 = (exp + 127).to(torch.uint8)
    return (
        q.reshape(*x.shape[:-1], K).contiguous(),
        e8m0.squeeze(-1).reshape(*x.shape[:-1], K // 32).contiguous(),
    )


def dequant_mxfp8(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    K = q.shape[-1]
    qb = q.float().reshape(*q.shape[:-1], K // 32, 32)
    exp = e8m0.reshape(*e8m0.shape, 1).float() - 127.0
    return (qb * torch.exp2(exp)).reshape(*q.shape[:-1], K)


def moe_align(topk_ids: torch.Tensor, block_m: int, E: int):
    """Reference moe_align_block_size: per-expert token slots padded to block_m,
    pad slots marked with the sentinel ``M`` (== num_valid_tokens)."""
    flat = topk_ids.reshape(-1).to(torch.int32)
    M = flat.numel()
    sorted_ids, expert_ids = [], []
    for e in range(E):
        idx = (flat == e).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        npad = ((n + block_m - 1) // block_m) * block_m
        blk = torch.full((npad,), M, dtype=torch.int32, device=topk_ids.device)
        blk[:n] = idx.to(torch.int32)
        sorted_ids.append(blk)
        expert_ids.append(
            torch.full((npad // block_m,), e, dtype=torch.int32, device=topk_ids.device)
        )
    sorted_ids = torch.cat(sorted_ids)
    expert_ids = torch.cat(expert_ids)
    num_post = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_post


# ── dense GEMV (+ MFMA crossover) ─────────────────────────────────────────────
# (K, N) are M3's qkv / o_proj / gate_up / down shapes (all in the allowlist).
@requires_gfx950
@pytest.mark.parametrize("K,N", [(6144, 2304), (2048, 6144), (6144, 1536), (1536, 6144)])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 32, 64])
@torch.inference_mode()
def test_mxfp8_gemv(K, N, M):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)

    got = mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16)
    if got is None:
        # autotune (or the allowlist) routes this shape to the Triton fallback.
        pytest.skip(f"({M},{K},{N}) dispatches to Triton (use_hip=0)")
    # Reference consumes the SAME quantized bits the kernel reads.
    ref = torch.nn.functional.linear(dequant_mxfp8(xq, xs), dequant_mxfp8(wq, ws))
    assert got.shape == (M, N)
    assert _relerr(got, ref) < 5e-2


@requires_gfx950
@torch.inference_mode()
def test_mxfp8_gemv_unsupported_returns_none():
    # A (K, N) outside the measured allowlist must fall back (return None).
    M, K, N = 4, 4096, 4096
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    xq, xs = quant_mxfp8(x)
    wq, ws = quant_mxfp8(w)
    assert mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16) is None


# ── MoE grouped GEMM ──────────────────────────────────────────────────────────
def _ref_grouped(a_deq, w_deq, sorted_ids, expert_ids, num_valid, a_div, block_m, wt):
    """Per-slot reference: out[slot] = a_deq[slot // a_div] @ w_deq[expert].T."""
    Np = sorted_ids.shape[0]
    N = w_deq.shape[1]
    out = torch.zeros(num_valid, N, device=a_deq.device, dtype=torch.float32)
    for blk in range(Np // block_m):
        e = int(expert_ids[blk])
        for s in range(blk * block_m, (blk + 1) * block_m):
            tok = int(sorted_ids[s])
            if tok >= num_valid:
                continue
            row = a_deq[tok // a_div]
            o = (row.float() @ w_deq[e].float().T)
            if wt is not None:
                o = o * float(wt[tok])
            out[tok] = o
    return out


@requires_gfx950
@pytest.mark.parametrize(
    "E,N,K,a_div,has_w,T,top_k",
    [
        (128, 1536, 6144, 4, False, 4, 4),  # gemm1 (gate_up): M_routed=16
        (128, 6144, 768, 1, True, 8, 1),    # gemm2 weighted: M_routed=8
        (128, 6144, 768, 1, False, 8, 1),   # gemm2 no-combine
    ],
)
@torch.inference_mode()
def test_grouped_gemm_mxfp8(E, N, K, a_div, has_w, T, top_k):
    torch.manual_seed(0)
    block_m = 64
    M = T * top_k  # num_valid_tokens
    M_act = T if a_div == top_k else M  # gemm1 reads per-token; gemm2 per-slot

    a = torch.randn(M_act, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
    aq, asc = quant_mxfp8(a)
    wq, wsc = quant_mxfp8(w)

    topk_ids = torch.randint(0, E, (T, top_k), device=DEVICE, dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align(topk_ids, block_m, E)
    wt = None
    if has_w:
        wt = torch.rand(M, device=DEVICE, dtype=torch.float32)

    got = grouped_gemm_mxfp8(
        aq, asc, wq, wsc, sorted_ids, expert_ids, num_post,
        M, top_k, block_m, torch.bfloat16, a_div,
        mul_weight_by=wt, topk_ids=topk_ids,
    )
    assert got is not None, "shape should be in the HIP MoE envelope"
    ref = _ref_grouped(
        dequant_mxfp8(aq, asc), dequant_mxfp8(wq, wsc),
        sorted_ids, expert_ids, M, a_div, block_m, wt,
    )
    assert got.shape == (M, N)
    assert _relerr(got, ref) < 5e-2
