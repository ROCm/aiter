# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention Prefill (chunk_gated_delta_h) regressions.

Usage:
    pytest -sv aiter/ops/flydsl/test_flydsl_linear_attention_prefill.py
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL Linear Attention Prefill tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        flydsl_chunk_gated_delta_rule_fwd_h,
    )
    from aiter.ops.flydsl.kernels import (
        chunk_gated_delta_h as flydsl_chunk_gated_delta_h_mod,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL Linear Attention Prefill kernels: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")


# -- Global test configuration ------------------------------------------

K = 128
V = 128
Hk = 16
Hv = 64
TP_LIST = [8]
BT = 64
MAX_NUM_BATCHED_TOKENS = 32768
FULL_PROMPT_LENS = [8192]
NUM_WARMUP = 5
NUM_ITERS = 100


# -- Helper functions ---------------------------------------------------


def _build_context_lens(full_prompt_len, max_tokens=MAX_NUM_BATCHED_TOKENS):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


def _build_cu_seqlens(context_lens, device="cuda"):
    scheduled_q_lens = context_lens
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(scheduled_q_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return scheduled_q_lens, cu_seqlens


def _build_chunk_offsets(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(lens, chunk_size),
        ]
    ).cumsum(-1)


def _make_inputs(
    context_lens, tp=1, dtype=torch.bfloat16, device="cuda", with_initial_state=True
):
    Hg = Hk // tp
    H = Hv // tp
    scheduled_q_lens, cu_seqlens = _build_cu_seqlens(context_lens, device=device)
    T_total = int(cu_seqlens[-1].item())
    N = len(scheduled_q_lens)
    B = 1

    k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, K, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(T_total, H, dtype=torch.float32, device=device).abs() * -0.5
    g = g.cumsum(dim=0)

    w_c = w_orig.permute(0, 2, 1, 3).contiguous()
    u_c = u_orig.permute(0, 2, 1, 3).contiguous()

    initial_state = None
    if with_initial_state:
        initial_state = (
            torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01
        )

    return k, w_orig, u_orig, w_c, u_c, g, initial_state, cu_seqlens, scheduled_q_lens


# -- Pure-PyTorch reference ----------------------------------------------


def ref_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
):
    """Reference in FP32 for correctness checking."""
    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(triton.cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = (
        torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [
                (s, cu_seqlens[s].item(), cu_seqlens[s + 1].item()) for s in range(N)
            ]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = triton.cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(
                    V_dim, K_dim, dtype=torch.float32, device=k.device
                )
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start : bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start : bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start : bos + t_end, i_h] = b_v

                    last_idx = bos + t_end - 1
                    g_last = g[last_idx, i_h].float()
                    g_chunk = g[bos + t_start : bos + t_end, i_h].float()

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    gate = torch.where(
                        mask[:actual_bt].bool(),
                        torch.exp(g_last - g_chunk),
                        torch.zeros_like(g_chunk),
                    )
                    b_v_gated = b_v * gate.unsqueeze(-1)

                    h_state = h_state * torch.exp(g_last)
                    k_chunk = k[b_idx, bos + t_start : bos + t_end, i_hg].float()
                    b_v_gated_cast = b_v_gated.to(k.dtype).float()
                    h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


def _normalize_opt_v_new(vn_opt):
    """Convert opt v_new layout [B, H, T, V] back to [B, T, H, V]."""
    return vn_opt.permute(0, 2, 1, 3).contiguous()


# -- Triton opt_vk reference ---------------------------------------------

_FLA_CHUNK_SIZE_OPT_VK = 64


def _check_platform():
    try:
        backend = triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        backend = "cpu"
    return {"cuda": "nvidia", "hip": "amd", "xpu": "intel"}.get(backend, backend)


_use_cuda_graph = _check_platform() == "nvidia"


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
        for BV in [16, 32, 64]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    use_cuda_graph=_use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_opt_vk(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    T_flat,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    if IS_VARLEN:
        w += ((i_h * T_flat + bos) * K).to(tl.int64)
    else:
        w += (((i_n * H + i_h) * T_flat) * K).to(tl.int64)
    if IS_VARLEN:
        v += ((i_h * T_flat + bos) * V).to(tl.int64)
    else:
        v += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
    if SAVE_NEW_VALUE:
        if IS_VARLEN:
            v_new += ((i_h * T_flat + bos) * V).to(tl.int64)
        else:
            v_new += (((i_n * H + i_h) * T_flat) * V).to(tl.int64)
    stride_v = V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, 64),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 128),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 192),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(
                v_new, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= tl.exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= tl.exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= tl.exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= tl.exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_opt_vk(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = _FLA_CHUNK_SIZE_OPT_VK,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        NT = sum(triton.cdiv(int(seq_len), BT) for seq_len in lens.tolist())
        chunk_offsets = _build_chunk_offsets(cu_seqlens, BT)

    assert K <= 256, "Current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_opt_vk[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        T_flat=T_flat,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state


def _bench_fn(fn, *args, **kwargs):
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(NUM_ITERS):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / NUM_ITERS * 1000


def _prepare_flydsl_kernel_launch(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    wu_contiguous=True,
    bv=0,
):
    mod = flydsl_chunk_gated_delta_h_mod
    batch_size, total_tokens, hg, head_k_dim = k.shape
    bt = chunk_size

    if wu_contiguous:
        num_heads = w.shape[1]
        head_v_dim = u.shape[-1]
        t_flat = w.shape[2]
    else:
        num_heads = u.shape[-2]
        head_v_dim = u.shape[-1]
        t_flat = w.shape[1]

    if cu_seqlens is None:
        num_seqs, num_chunks, chunk_offsets = (
            batch_size,
            triton.cdiv(total_tokens, bt),
            None,
        )
    else:
        num_seqs = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_chunks = sum(triton.cdiv(int(seq_len), bt) for seq_len in lens.tolist())
        chunk_offsets = _build_chunk_offsets(cu_seqlens, bt).to(torch.int32)

    use_g = g is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    if bv <= 0:
        shape_key = (
            head_k_dim,
            head_v_dim,
            bt,
            num_heads,
            hg,
            t_flat,
            num_seqs,
            use_g,
            use_h0,
            output_final_state,
            save_new_value,
            is_varlen,
            wu_contiguous,
        )
        if shape_key not in mod._autotune_cache:
            mod.chunk_gated_delta_rule_fwd_h_flydsl(
                k,
                w,
                u,
                g=g,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                save_new_value=save_new_value,
                cu_seqlens=cu_seqlens,
                wu_contiguous=wu_contiguous,
            )
            torch.cuda.synchronize()
        bv = mod._autotune_cache[shape_key]

    launch_fn = mod._get_or_compile(
        head_k_dim,
        head_v_dim,
        bt,
        bv,
        num_heads,
        hg,
        use_g,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        wu_contiguous,
    )

    h = k.new_empty(batch_size, num_chunks, num_heads, head_v_dim, head_k_dim)
    final_state = (
        k.new_empty(num_seqs, num_heads, head_v_dim, head_k_dim, dtype=torch.float32)
        if output_final_state
        else None
    )
    v_new_buf = k.new_empty(batch_size, num_heads, t_flat, head_v_dim, dtype=u.dtype)

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    cu_arg = (
        cu_seqlens.to(torch.int32) if cu_seqlens is not None else dummy.to(torch.int32)
    )
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)
    stream = torch.cuda.current_stream()

    def _launch():
        mod._launch_kernel(
            launch_fn,
            bv,
            head_v_dim,
            num_seqs,
            num_heads,
            k,
            u,
            w,
            vn_arg,
            g_arg,
            h,
            h0_arg,
            ht_arg,
            cu_arg,
            co_arg,
            total_tokens,
            t_flat,
            stream,
        )

    return _launch


def _prepare_triton_opt_vk_kernel_launch(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=_FLA_CHUNK_SIZE_OPT_VK,
    save_new_value=True,
    cu_seqlens=None,
):
    batch_size, total_tokens, hg, head_k_dim = k.shape
    num_heads = w.shape[1]
    head_v_dim = u.shape[-1]
    t_flat = w.shape[2]
    bt = chunk_size

    if cu_seqlens is None:
        num_seqs, num_chunks, chunk_offsets = (
            batch_size,
            triton.cdiv(total_tokens, bt),
            None,
        )
    else:
        num_seqs = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_chunks = sum(triton.cdiv(int(seq_len), bt) for seq_len in lens.tolist())
        chunk_offsets = _build_chunk_offsets(cu_seqlens, bt).to(torch.int32)

    h = k.new_empty(batch_size, num_chunks, num_heads, head_v_dim, head_k_dim)
    final_state = (
        k.new_empty(num_seqs, num_heads, head_v_dim, head_k_dim, dtype=torch.float32)
        if output_final_state
        else None
    )
    v_new = (
        k.new_empty(batch_size, num_heads, t_flat, head_v_dim, dtype=u.dtype)
        if save_new_value
        else None
    )

    def grid(meta):
        return (triton.cdiv(head_v_dim, meta["BV"]), num_seqs * num_heads)

    def _launch():
        chunk_gated_delta_rule_fwd_kernel_h_opt_vk[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=total_tokens,
            T_flat=t_flat,
            H=num_heads,
            Hg=hg,
            K=head_k_dim,
            V=head_v_dim,
            BT=bt,
        )

    return _launch


# -- Correctness tests ---------------------------------------------------

PERF_SHAPES = [
    pytest.param(tp, fpl, id=f"TP{tp}_full{fpl}")
    for tp in TP_LIST
    for fpl in FULL_PROMPT_LENS
]


class TestCorrectness:
    """Correctness against PyTorch reference."""

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_correctness_flydsl(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)

        h_fly, vn_fly, fs_fly = flydsl_chunk_gated_delta_rule_fwd_h(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k,
            w_orig,
            u_orig,
            g=g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )

        torch.testing.assert_close(h_fly.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_fly).float(), vn_ref.float(), atol=1e-1, rtol=1e-1
        )
        torch.testing.assert_close(fs_fly.float(), fs_ref.float(), atol=1e-1, rtol=1e-1)


class TestPerformance:
    """Kernel-only performance comparison: FlyDSL vs Triton opt_vk."""

    @pytest.mark.parametrize("tp, full_prompt_len", PERF_SHAPES)
    def test_perf_comparison(self, tp, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, _, _, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, tp=tp)
        total_tokens = int(cu[-1].item())
        hg = Hk // tp
        h = Hv // tp

        flydsl_launch = _prepare_flydsl_kernel_launch(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        triton_vk_launch = _prepare_triton_opt_vk_kernel_launch(
            k,
            w_c,
            u_c,
            g=g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )
        us_fly = _bench_fn(flydsl_launch)
        us_triton_vk = _bench_fn(triton_vk_launch)
        speedup = us_triton_vk / us_fly if us_fly > 0 else float("inf")

        print(
            f"\n[K5 FlyDSL kernel-only TP={tp} Hg={hg} H={h} T={total_tokens}] {us_fly:.2f} us"
        )
        print(
            f"[K5 Triton opt_vk kernel-only TP={tp} T={total_tokens}] {us_triton_vk:.2f} us"
        )
        print(f"[Speedup FlyDSL/Triton opt_vk kernel-only] {speedup:.3f}x")
