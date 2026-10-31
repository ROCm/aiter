# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused-vs-unfused op-level latency for the Kimi-K3 KDA whole-layer ops.

Answers "why a separate op": times the single fused decode/prefill kernel
against the native aiter path it replaces -- the same component sequence the
correctness tests validate against (``test_kimi_kda_fused_layer.py``), i.e.

  decode : qkvg matmul + beta/forget matmul + forget-B matmul
           + native ``fused_kda_decode`` (conv + recurrence + gated-RMSNorm)
           + O matmul
  prefill: native causal conv1d+SiLU + ``chunk_kimi_delta_attn`` (chunked
           gated-delta mixer) + gated-RMSNorm

The projection matmuls run at bf16 (production precision), not the fp32 the
correctness reference upcasts to -- so this measures the real unfused path, not
a reference-precision one. gfx950 + Triton >= 3.8.

    python op_tests/op_benchmarks/triton/bench_kimi_kda_fused_vs_unfused.py
"""

import torch

from aiter import logger
from aiter.ops.triton.gated_delta_net.causal_conv1d_prefill import (
    causal_conv1d_split_qkv_triton_fn,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode
from aiter.ops.triton.kimi_delta_attn import (
    chunk_kimi_delta_attn,
    kimi_kda_fused_layer_decode,
    kimi_kda_fused_layer_prefill,
)

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
DEV = "cuda"
LOWER_BOUND, NORM_EPS = -5.0, 1e-5


def _rnd(shape, dtype):
    if dtype == I32:
        return torch.zeros(shape, dtype=I32, device=DEV)
    return (torch.randn(shape, device=DEV, dtype=torch.float32) * 0.05).to(dtype)


def _time(fn, *, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------
def _decode_inputs(m):
    slots = max(m, 64)
    return {
        "x": _rnd((m, 7168), BF16),
        "qkvg_weight": _rnd((6144, 7168), BF16),
        "beta_forget_weight": _rnd((144, 7168), BF16),
        "output_weight": _rnd((7168, 1536), BF16),
        "forget_weight": _rnd((1536, 128), BF16),
        "conv_weight": _rnd((4608, 4), F32),
        "a_log": _rnd((12,), F32),
        "dt_bias": _rnd((1536,), F32),
        "norm_weight": _rnd((128,), BF16),
        "conv_state": _rnd((slots, 3, 4608), BF16),
        "state": _rnd((slots, 12, 128, 128), F32),
        "state_indices": torch.arange(m, dtype=I32, device=DEV),
    }


def _fused_decode(ins):
    out = torch.empty((ins["x"].shape[0], 7168), dtype=BF16, device=DEV)
    return lambda: kimi_kda_fused_layer_decode(
        **ins, lower_bound=LOWER_BOUND, norm_eps=NORM_EPS, output_tensor=out
    )


def _unfused_decode(ins):
    """Native component sequence the fused decode replaces, at bf16 precision.

    Mirrors ``_ref_decode`` in test_kimi_kda_fused_layer.py (validated
    equivalent) but keeps the projection matmuls in bf16 -- the native path's
    real precision -- rather than the fp32 the correctness reference uses.
    The conv/recurrence caches are V-first [N,3*lp,STATE_LEN]; clone once so the
    native op's in-place updates don't reshape the fused cache under us.
    """
    m = ins["x"].shape[0]
    x, qkvg_w = ins["x"], ins["qkvg_weight"]
    bf_w, out_w, fgt_w = (
        ins["beta_forget_weight"],
        ins["output_weight"],
        ins["forget_weight"],
    )
    conv_ref = ins["conv_state"].clone().transpose(1, 2).contiguous()
    state_ref = ins["state"].clone()
    sidx, a_log, dt_bias, nw, cw = (
        ins["state_indices"],
        ins["a_log"],
        ins["dt_bias"],
        ins["norm_weight"],
        ins["conv_weight"],
    )
    cu = torch.arange(0, m + 1, dtype=torch.int64, device=DEV)

    def run():
        qkvg = x @ qkvg_w.T  # (m, 6144)
        beta_forget = x @ bf_w.T  # (m, 144)
        mixed_qkv = qkvg[:, :4608].contiguous()
        out_gate = qkvg[:, 4608:6144].contiguous()
        forget_a = beta_forget[:, :128]
        beta = beta_forget[:, 128:140].reshape(1, m, 12).contiguous()
        gate = (forget_a @ fgt_w.T).reshape(1, m, 12, 128).contiguous()
        core = fused_kda_decode(
            mixed_qkv,
            conv_ref,
            cw,
            gate,
            beta,
            out_gate,
            a_log,
            dt_bias,
            state_ref,
            sidx,
            cu,
            nw,
            NORM_EPS,
            head_dim=128,
            num_local_heads=12,
            lower_bound=LOWER_BOUND,
        )
        return core @ out_w.T

    return run


# --------------------------------------------------------------------------
# prefill
# --------------------------------------------------------------------------
def _prefill_inputs(m, n):
    slots = max(n, 4)
    base = m // n
    lens = [base] * n
    lens[-1] += m - base * n
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=I32, device=DEV
    )
    return {
        "qkv": _rnd((m, 4608), BF16),
        "gate": _rnd((m, 1536), BF16),
        "forget_a": _rnd((m, 128), BF16),
        "beta": _rnd((m, 12), BF16),
        "forget_weight": _rnd((1536, 128), BF16),
        "conv_weight": _rnd((4608, 4), F32),
        "a_log": _rnd((12,), F32),
        "dt_bias": _rnd((1536,), F32),
        "norm_weight": _rnd((128,), BF16),
        "conv_state": _rnd((slots, 3, 4608), BF16),
        "state": _rnd((slots, 12, 128, 128), F32),
        "state_indices": torch.arange(n, dtype=I32, device=DEV),
        "cu_seqlens": cu,
        "has_initial_state": torch.zeros(n, dtype=torch.bool, device=DEV),
    }


def _fused_prefill(ins):
    return lambda: kimi_kda_fused_layer_prefill(
        **ins, lower_bound=LOWER_BOUND, norm_eps=NORM_EPS
    )


def _unfused_prefill(ins):
    """Native component sequence the fused prefill replaces (bf16).

    Mirrors ``_ref_prefill`` in test_kimi_kda_fused_layer.py (validated
    equivalent): native causal conv1d+SiLU, forget-B decay matmul, the chunked
    gated-delta mixer, and a gated RMSNorm -- all at bf16.
    """
    m = ins["qkv"].shape[0]
    n = ins["state_indices"].numel()
    cu = ins["cu_seqlens"]
    lens = (cu[1:] - cu[:-1]).tolist()
    sidx, has_init = ins["state_indices"], ins["has_initial_state"]
    cw, fgt_w, a_log, dt_bias, nw = (
        ins["conv_weight"].to(BF16),
        ins["forget_weight"],
        ins["a_log"],
        ins["dt_bias"],
        ins["norm_weight"],
    )
    x = ins["qkv"].T.contiguous()
    conv_ref = ins["conv_state"].clone().transpose(1, 2).contiguous()
    forget_a, gate_in, beta_in = ins["forget_a"], ins["gate"], ins["beta"]
    h0 = torch.zeros(n, 12, 128, 128, device=DEV, dtype=F32)
    cul = cu.long()

    def run():
        q, k, v = causal_conv1d_split_qkv_triton_fn(
            x,
            cw,
            None,
            conv_ref,
            cu,
            lens,
            k_dim=1536,
            v_dim=1536,
            cache_indices=sidx,
            has_initial_state=has_init,
            activation="silu",
        )
        q = q.reshape(1, m, 12, 128)
        k = k.reshape(1, m, 12, 128)
        v = v.reshape(1, m, 12, 128)
        g = (forget_a @ fgt_w.T).reshape(1, m, 12, 128)
        beta = beta_in.reshape(1, m, 12).contiguous()
        core, _ = chunk_kimi_delta_attn(
            q,
            k,
            v,
            g,
            beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=128**-0.5,
            initial_state=h0,
            output_final_state=True,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=LOWER_BOUND,
            cu_seqlens=cul,
        )
        cf = core.reshape(m, 12, 128).float()
        inv = torch.rsqrt((cf * cf).mean(-1, keepdim=True) + NORM_EPS)
        gate = torch.sigmoid(gate_in.reshape(m, 12, 128).float())
        return (cf * inv * nw.float() * gate).reshape(m, 1536).to(BF16)

    return run


def main():
    logger.info(
        "%-9s%-14s%12s%12s%10s", "op", "shape", "fused_us", "unfused_us", "speedup"
    )
    for m in (1, 8, 32, 64, 96, 128, 192, 256):
        ins = _decode_inputs(m)
        f = _time(_fused_decode(ins))
        u = _time(_unfused_decode(ins))
        logger.info("%-9s%-14s%12.1f%12.1f%9.2fx", "decode", f"m={m}", f, u, u / f)
    for m, n in ((1024, 1), (2048, 4), (4096, 8), (8192, 32)):
        ins = _prefill_inputs(m, n)
        f = _time(_fused_prefill(ins))
        u = _time(_unfused_prefill(ins))
        logger.info(
            "%-9s%-14s%12.1f%12.1f%9.2fx", "prefill", f"m={m} n={n}", f, u, u / f
        )


if __name__ == "__main__":
    main()
