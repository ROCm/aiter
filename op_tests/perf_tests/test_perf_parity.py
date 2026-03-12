# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance parity tests for AITER kernels vs PyTorch reference.

Each test measures wall-clock latency of an AITER kernel and its torch
reference implementation, then asserts the AITER kernel is at least as
fast (within a tolerance).  This catches performance regressions that
correctness-only tests miss.

Run:
    pytest op_tests/perf_tests/test_perf_parity.py -v --tb=short

Environment:
    AITER_PERF_MIN_SPEEDUP  – minimum required speedup (default 0.95)
    AITER_PERF_WARMUP       – warmup iterations for do_bench (default 25)
    AITER_PERF_REP          – measurement iterations (default 100)
"""

import os
import json
import datetime
import pytest
import torch
import torch.nn.functional as F
import triton

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_SPEEDUP = float(os.getenv("AITER_PERF_MIN_SPEEDUP", "0.95"))
WARMUP = int(os.getenv("AITER_PERF_WARMUP", "25"))
REP = int(os.getenv("AITER_PERF_REP", "100"))

DEVICE = "cuda"
RESULTS: list[dict] = []


def _bench(fn, **kwargs):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP, **kwargs)


def _record(name, shape_info, torch_ms, aiter_ms):
    speedup = torch_ms / aiter_ms if aiter_ms > 0 else float("inf")
    RESULTS.append(
        {
            "kernel": name,
            "shape": shape_info,
            "torch_ms": round(torch_ms, 4),
            "aiter_ms": round(aiter_ms, 4),
            "speedup": round(speedup, 3),
            "pass": speedup >= MIN_SPEEDUP,
        }
    )
    return speedup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_no_aiter():
    try:
        import aiter  # noqa: F401
    except ImportError:
        pytest.skip("aiter not installed")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ===================================================================
# 1. GEMM A8W8 (FP8 quantized GEMM)
# ===================================================================
GEMM_A8W8_SHAPES = [
    # (M, N, K) – representative decode / prefill shapes
    (1, 7168, 2048),     # DeepSeek decode qkv_proj
    (1, 2048, 7168),     # DeepSeek decode o_proj
    (1, 18432, 7168),    # DeepSeek decode gate_up_proj
    (1, 7168, 9216),     # DeepSeek decode down_proj
    (32, 7168, 2048),    # small batch
    (128, 7168, 2048),   # medium batch
    (256, 14336, 4096),  # Llama-style FFN
]


@pytest.fixture(autouse=True)
def _setup():
    _skip_if_no_gpu()
    _skip_if_no_aiter()


def _torch_gemm_a8w8(x_f32, weight_f32, x_scale, w_scale, bias, out_dtype):
    out = F.linear(x_f32, weight_f32)
    scale = torch.matmul(x_scale, w_scale)
    out = (out * scale).to(out_dtype)
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("M,N,K", GEMM_A8W8_SHAPES, ids=lambda *a: f"{a}")
def test_gemm_a8w8_perf(M, N, K):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    fp8_dtype = torch.float8_e4m3fnuz
    out_dtype = torch.bfloat16

    x = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    weight = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
    x_scale = torch.ones(M, 1, device=DEVICE, dtype=torch.float32)
    w_scale = torch.ones(1, N, device=DEVICE, dtype=torch.float32)
    bias = torch.zeros(N, device=DEVICE, dtype=out_dtype)
    y = torch.empty(M, N, device=DEVICE, dtype=out_dtype)

    x_fp8 = x.to(fp8_dtype)
    w_fp8 = weight.to(fp8_dtype)

    torch_ms = _bench(
        lambda: _torch_gemm_a8w8(x, weight, x_scale, w_scale, bias, out_dtype)
    )
    aiter_ms = _bench(
        lambda: gemm_a8w8(x_fp8, w_fp8, x_scale, w_scale, bias, out_dtype, y)
    )
    speedup = _record(
        "gemm_a8w8", f"M={M} N={N} K={K}", torch_ms, aiter_ms
    )
    assert speedup >= MIN_SPEEDUP, (
        f"gemm_a8w8 M={M} N={N} K={K}: speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# 2. RMSNorm
# ===================================================================
RMSNORM_SHAPES = [
    (1, 4096),
    (1, 7168),
    (32, 4096),
    (128, 4096),
    (128, 7168),
    (256, 7168),
    (1024, 4096),
    (4096, 7168),
]


def _torch_rmsnorm(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


@pytest.mark.parametrize("M,N", RMSNORM_SHAPES, ids=lambda *a: f"{a}")
def test_rmsnorm_perf(M, N):
    from aiter import rms_norm

    eps = 1e-6
    x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(N, device=DEVICE, dtype=torch.bfloat16)

    torch_ms = _bench(lambda: _torch_rmsnorm(x, weight, eps))
    aiter_ms = _bench(lambda: rms_norm(x, weight, eps))
    speedup = _record("rmsnorm", f"M={M} N={N}", torch_ms, aiter_ms)
    assert speedup >= MIN_SPEEDUP, (
        f"rmsnorm M={M} N={N}: speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# 3. RMSNorm + Residual Add (fused)
# ===================================================================
RMSNORM_ADD_SHAPES = [
    (1, 7168),
    (32, 7168),
    (128, 7168),
    (256, 7168),
]


def _torch_rmsnorm_add(x, residual, weight, eps):
    residual_out = x + residual
    variance = residual_out.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = residual_out * torch.rsqrt(variance + eps)
    return (normed * weight).to(x.dtype), residual_out


@pytest.mark.parametrize("M,N", RMSNORM_ADD_SHAPES, ids=lambda *a: f"{a}")
def test_rmsnorm_add_perf(M, N):
    from aiter import rmsnorm2d_fwd_with_add

    eps = 1e-6
    x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(N, device=DEVICE, dtype=torch.bfloat16)

    torch_ms = _bench(lambda: _torch_rmsnorm_add(x, residual, weight, eps))

    def aiter_fn():
        out = torch.empty_like(x)
        res_out = torch.empty_like(residual)
        rmsnorm2d_fwd_with_add(out, x, residual, res_out, weight, eps)
        return out, res_out

    aiter_ms = _bench(aiter_fn)
    speedup = _record("rmsnorm_add", f"M={M} N={N}", torch_ms, aiter_ms)
    assert speedup >= MIN_SPEEDUP, (
        f"rmsnorm_add M={M} N={N}: speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# 4. TopK Softmax (MoE routing)
# ===================================================================
TOPK_SHAPES = [
    (1, 256, 8),     # decode, 256 experts, top-8
    (32, 256, 8),    # small batch
    (128, 256, 8),   # medium batch
    (1, 64, 6),      # GPT-OSS-120B style
    (32, 64, 6),
]


def _torch_topk_softmax(gating_output, top_k, renormalize):
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


@pytest.mark.parametrize(
    "tokens,experts,top_k", TOPK_SHAPES, ids=lambda *a: f"{a}"
)
def test_topk_softmax_perf(tokens, experts, top_k):
    from aiter import topk_softmax

    gating = torch.randn(tokens, experts, device=DEVICE, dtype=torch.float32)
    topk_w = torch.empty(tokens, top_k, device=DEVICE, dtype=torch.float32)
    topk_i = torch.empty(tokens, top_k, device=DEVICE, dtype=torch.int32)
    token_expert_i = torch.empty(
        tokens, top_k, device=DEVICE, dtype=torch.int32
    )

    torch_ms = _bench(lambda: _torch_topk_softmax(gating, top_k, True))

    def aiter_fn():
        topk_softmax(topk_w, topk_i, token_expert_i, gating, True)
        return topk_w, topk_i

    aiter_ms = _bench(aiter_fn)
    speedup = _record(
        "topk_softmax",
        f"tokens={tokens} experts={experts} top_k={top_k}",
        torch_ms,
        aiter_ms,
    )
    assert speedup >= MIN_SPEEDUP, (
        f"topk_softmax tokens={tokens} experts={experts} top_k={top_k}: "
        f"speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# 5. Fused MoE (end-to-end MoE forward)
# ===================================================================
MOE_SHAPES = [
    # (tokens, hidden, intermediate, experts, top_k)
    (1, 7168, 2048, 256, 8),      # DeepSeek V3 decode
    (32, 7168, 2048, 256, 8),     # small batch
    (1, 4096, 14336, 8, 2),       # Mixtral-style
]


def _torch_fused_moe(hidden, w1, w2, topk_weights, topk_ids, num_experts):
    """Simple torch reference for MoE: scatter-gather approach."""
    tokens, D = hidden.shape
    _, top_k = topk_ids.shape
    N = w2.shape[1]  # intermediate_size for w2

    out = torch.zeros(tokens, D, device=hidden.device, dtype=hidden.dtype)
    for e in range(num_experts):
        mask = (topk_ids == e).any(dim=-1)
        if not mask.any():
            continue
        h_e = hidden[mask].to(torch.float32)
        # gate_up = h @ w1[e].T, then SiLU gating
        gate_up = h_e @ w1[e].T.float()
        gate, up = gate_up.chunk(2, dim=-1)
        h_e = F.silu(gate) * up
        h_e = h_e @ w2[e].T.float()
        # weight by topk_weights
        for k in range(top_k):
            slot_mask = mask & (topk_ids[:, k] == e)
            if slot_mask.any():
                w = topk_weights[slot_mask, k].unsqueeze(-1).float()
                out[slot_mask] += (h_e[: slot_mask.sum()] * w).to(out.dtype)
    return out


@pytest.mark.parametrize(
    "tokens,hidden,inter,experts,top_k",
    MOE_SHAPES,
    ids=lambda *a: f"{a}",
)
def test_fused_moe_perf(tokens, hidden, inter, experts, top_k):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    dtype = torch.bfloat16
    h = torch.randn(tokens, hidden, device=DEVICE, dtype=dtype)
    w1 = torch.randn(experts, 2 * inter, hidden, device=DEVICE, dtype=dtype)
    w2 = torch.randn(experts, hidden, inter, device=DEVICE, dtype=dtype)

    # generate routing
    gating = torch.randn(tokens, experts, device=DEVICE, dtype=torch.float32)
    scores = torch.softmax(gating, dim=-1)
    topk_w, topk_i = torch.topk(scores, top_k, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(dtype)
    topk_i = topk_i.to(torch.int32)

    torch_ms = _bench(
        lambda: _torch_fused_moe(h, w1, w2, topk_w, topk_i, experts)
    )

    aiter_ms = _bench(
        lambda: fused_moe(
            h, w1, w2, topk_w, topk_i,
            None,
            ActivationType.Swiglu,
            QuantType.No,
            False,
        )
    )
    speedup = _record(
        "fused_moe",
        f"tok={tokens} h={hidden} inter={inter} E={experts} k={top_k}",
        torch_ms,
        aiter_ms,
    )
    assert speedup >= MIN_SPEEDUP, (
        f"fused_moe tok={tokens}: speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# 6. MLA Decode (Multi-head Latent Attention decode)
# ===================================================================
MLA_DECODE_SHAPES = [
    # (batch, nhead, kv_len, kv_lora_rank, qk_rope_head_dim)
    (1, 16, 4096, 512, 64),     # TP=2, 4K context
    (1, 16, 16384, 512, 64),    # TP=2, 16K context
    (8, 16, 4096, 512, 64),     # batch=8
    (1, 128, 4096, 512, 64),    # TP=1
]


def _torch_mla_decode_ref(q, kv_flat, sm_scale, kv_lora_rank):
    """Simple torch MLA decode reference: q @ K^T * scale, softmax, @ V."""
    B, nhead, D = q.shape
    S = kv_flat.shape[0]
    V_dim = kv_lora_rank

    q_f = q.float()
    k_f = kv_flat[:, :D].float()
    v_f = kv_flat[:, :V_dim].float()

    # [B, nhead, D] x [S, D]^T -> [B, nhead, S]
    scores = torch.einsum("bhd,sd->bhs", q_f, k_f) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    # [B, nhead, S] x [S, V] -> [B, nhead, V]
    out = torch.einsum("bhs,sv->bhv", attn, v_f)
    return out.to(q.dtype)


@pytest.mark.parametrize(
    "batch,nhead,kv_len,lora_rank,rope_dim",
    MLA_DECODE_SHAPES,
    ids=lambda *a: f"{a}",
)
def test_mla_decode_perf(batch, nhead, kv_len, lora_rank, rope_dim):
    from aiter.mla import mla_decode_fwd

    dtype = torch.bfloat16
    D = lora_rank + rope_dim  # total head dim

    q = torch.randn(batch, nhead, D, device=DEVICE, dtype=dtype)
    kv_buffer = torch.randn(kv_len, 1, 1, D, device=DEVICE, dtype=dtype)
    o = torch.zeros(batch, nhead, lora_rank, device=DEVICE, dtype=dtype)

    # Build paged KV metadata (contiguous, 1 token per page)
    qo_indptr = torch.arange(
        0, batch + 1, device=DEVICE, dtype=torch.int32
    )
    kv_indptr = torch.arange(
        0, (batch + 1) * kv_len, kv_len, device=DEVICE, dtype=torch.int32
    )
    kv_indices = torch.arange(
        0, batch * kv_len, device=DEVICE, dtype=torch.int32
    )
    kv_last_page_lens = torch.ones(batch, device=DEVICE, dtype=torch.int32)

    sm_scale = 1.0 / (D ** 0.5)
    kv_flat = kv_buffer.view(-1, D)

    torch_ms = _bench(
        lambda: _torch_mla_decode_ref(q, kv_flat[:kv_len], sm_scale, lora_rank)
    )

    def aiter_fn():
        o.zero_()
        mla_decode_fwd(
            q, kv_buffer, o, qo_indptr,
            kv_indptr, kv_indices, kv_last_page_lens,
            max_seqlen_q=1, sm_scale=sm_scale,
        )
        return o

    aiter_ms = _bench(aiter_fn)
    speedup = _record(
        "mla_decode",
        f"B={batch} nhead={nhead} kv={kv_len} D={D}",
        torch_ms,
        aiter_ms,
    )
    assert speedup >= MIN_SPEEDUP, (
        f"mla_decode B={batch} nhead={nhead} kv={kv_len}: "
        f"speedup {speedup:.3f}x < {MIN_SPEEDUP}x"
    )


# ===================================================================
# Report generation (runs after all tests)
# ===================================================================
@pytest.fixture(scope="session", autouse=True)
def _write_report(request):
    yield
    if not RESULTS:
        return
    # Print summary table
    print("\n" + "=" * 80)
    print("AITER Performance Parity Report")
    print(f"MIN_SPEEDUP={MIN_SPEEDUP}  WARMUP={WARMUP}  REP={REP}")
    print("=" * 80)
    print(
        f"{'Kernel':<20} {'Shape':<45} {'Torch(ms)':>10} "
        f"{'AITER(ms)':>10} {'Speedup':>8} {'Status':>8}"
    )
    print("-" * 80)
    for r in RESULTS:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['kernel']:<20} {r['shape']:<45} {r['torch_ms']:>10.4f} "
            f"{r['aiter_ms']:>10.4f} {r['speedup']:>7.3f}x "
            f"{'✓' if r['pass'] else '✗':>6}"
        )
    print("=" * 80)

    passed = sum(1 for r in RESULTS if r["pass"])
    total = len(RESULTS)
    print(f"Results: {passed}/{total} passed\n")

    # Write JSON report
    report_path = os.getenv(
        "AITER_PERF_REPORT_PATH", "perf_parity_report.json"
    )
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "min_speedup": MIN_SPEEDUP,
            "warmup": WARMUP,
            "rep": REP,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "results": RESULTS,
        "summary": {"passed": passed, "total": total},
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
