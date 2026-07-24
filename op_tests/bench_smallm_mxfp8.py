# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Microbenchmark: decode small-M MX-FP8 HIP GEMV vs the Triton dot_scaled path
it replaces (the vLLM/sglang fallback). MI355X / gfx950.

  python op_tests/bench_smallm_mxfp8.py
"""

import torch
import triton
import triton.language as tl

from aiter.ops.smallm_gemm_mxfp8 import mxfp8_gemv

DEVICE = "cuda"
FP8_MAX = 448.0


def quant_mxfp8(x):
    K = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], K // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    q = (xb / torch.exp2(exp)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    e8m0 = (exp + 127).to(torch.uint8)
    return (
        q.reshape(*x.shape[:-1], K).contiguous(),
        e8m0.squeeze(-1).reshape(*x.shape[:-1], K // 32).contiguous(),
    )


@triton.jit
def _triton_mxfp8_linear(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    sxm,
    sxk,
    sxsm,
    sxsk,
    swn,
    swk,
    swsn,
    swsk,
    som,
    son,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    xs_ptrs = xs_ptr + offs_m[:, None] * sxsm + offs_sk[None, :] * sxsk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
    ws_ptrs = ws_ptr + offs_n[:, None] * swsn + offs_sk[None, :] * swsk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk
        xs_ptrs += (BLOCK_K // 32) * sxsk
        ws_ptrs += (BLOCK_K // 32) * swsk
    o_ptrs = out_ptr + offs_m[:, None] * som + offs_n[None, :] * son
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def triton_dense(xq, xs, wq, ws):
    """Stock Triton dot_scaled dense GEMM (BLOCK 64/128/128, w8) -- the fallback."""
    M, K = xq.shape
    N = wq.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=xq.device)
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _triton_mxfp8_linear[grid](
        xq,
        xs,
        wq,
        ws,
        out,
        M,
        N,
        K,
        xq.stride(0),
        xq.stride(1),
        xs.stride(0),
        xs.stride(1),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=8,
    )
    return out


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us/call


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    shapes = [
        (6144, 2304, "qkv"),
        (2048, 6144, "o_proj"),
        (6144, 1536, "gate_up"),
        (1536, 6144, "mlp_down"),
    ]
    Ms = [1, 2, 4, 8, 16, 32, 64]
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"# MX-FP8 dense GEMV: HIP vs stock Triton dot_scaled  ({arch})")
    print(
        f"{'shape':10} {'K':>5} {'N':>5} {'M':>4} {'HIP us':>8} {'Triton us':>10} {'speedup':>8}"
    )
    for K, N, name in shapes:
        w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
        wq, ws = quant_mxfp8(w)
        for M in Ms:
            x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
            xq, xs = quant_mxfp8(x)
            if mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16) is None:
                continue
            t_hip = bench(lambda: mxfp8_gemv(xq, xs, wq, ws, torch.bfloat16))
            t_tri = bench(lambda: triton_dense(xq, xs, wq, ws))
            print(
                f"{name:10} {K:>5} {N:>5} {M:>4} {t_hip:>8.2f} {t_tri:>10.2f} "
                f"{t_tri / t_hip:>7.2f}x"
            )


def _skip_in_ci() -> bool:
    # CI runs every op_tests/*.py via `python3 <file>` (60-min timeout). This is
    # a microbenchmark, not a correctness test, and triggers JIT compilation, so
    # make it a no-op in CI (AITER_TEST set) and on non-gfx95x devices.
    import os

    if os.environ.get("AITER_TEST"):
        return True
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950"
    except Exception:
        return True


if __name__ == "__main__":
    if _skip_in_ci():
        print("skip bench_smallm_mxfp8: dev microbenchmark (gfx95x only, not under CI)")
    else:
        main()
