# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused paged flash-attention prefill (extend) kernel for page_size=1 KV pools.

A single Triton kernel serving the same seam as the CK-tile
`mha_batch_prefill_func`: it walks each request's FULL kv range directly out of the
paged pool via `kv_page_indices`, with bottom-right aligned causal masking, and
splits the KV loop into a fully-unmasked BULK range (no `tl.where` at all) and a
masked diagonal TAIL range. There is no prefix/extend split, no materialized
K_Extend/V_Extend gather, and no host sync.

It uses no LDS: K and V tiles stay in registers, which is the main structural
difference from the CK-tile pipeline (`qr_ks_vs_async`, which stages K/V through
LDS). On gfx950 at head_dim 256 that is worth ~1.4x on the kernel and +5.71%
end-to-end on an SGLang Qwen3.8-MoE TP8 server; see the PR body for the full
measurement.

Scope is deliberately narrow and enforced by `is_supported()`: causal only,
page_size 1, ONE kv head in the cache view, and a power-of-two head dim from
`SUPPORTED_HEAD_DIMS`. Anything else must fall back to the CK-tile path -- see the
note on `is_supported()` for why each condition is load-bearing.
"""

import os

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- config
# One static config. Deliberately not @triton.autotune: autotuning at runtime
# re-benchmarks inside the serving path, which is exactly what the
# do_not_specialize note below exists to avoid.
_DEFAULT_CFG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "kpack": 2,
    "matrix_instr_nonkdim": 16,
    # How the softmax scale reaches the QK product. 2 folds only the power-of-two part
    # of sm_scale*log2e into q, which is exact in bf16, and applies the small remainder
    # to the product. Mode 1 folds the whole thing for at most ~2% of kernel time (1.8%
    # and 0.0% in two processes), but its rounding error is data-dependent: on a
    # distribution with large late kv values it measured 8.0x the CK-tile path's error
    # against an fp32 reference (7.8e-3 vs 9.8e-4), where mode 2 matches CK-tile to
    # 1.00x. Not worth ~2% on a path that silently replaces a more accurate kernel.
    "PRESCALE_Q": 2,
    "DOT_ACC": 0,
    "SWAP_GRID": 1,  # head-major grid: 8 GQA heads of one (req, m-tile) dispatch together
    "CONTIG": 0,  # contiguous-page fast path measured SLOWER (11.6 vs 10.5 ms) -> off
    "NUM_XCDS": 8,
}


def _cfg():
    env = os.environ.get("AITER_FPP_CFG", "")
    if not env:
        return dict(_DEFAULT_CFG)
    import json

    c = dict(_DEFAULT_CFG)
    c.update(json.loads(env))
    return c


# NUM_M / BS are RUNTIME scalars, never tl.constexpr and never specialized: they are derived from
# (max_seqlen_q, batch_size), which the live chunked-prefill scheduler varies call-to-call.  Baking
# them into the JIT key forces a fresh compile per ragged shape (elevated live TTFT).  They are only
# read by the SWAP_GRID==2 flat-grid path; the default SWAP_GRID==1 grid never touches them.
@triton.jit(do_not_specialize=["NUM_M", "BS"])
def _fused_paged_prefill_kernel(
    Q,
    K,
    V,
    Out,
    CU_Q,
    KV_INDPTR,
    KV_PAGE,
    qk_scale,
    q_pre_scale,
    stride_qt,
    stride_qh,
    stride_ot,
    stride_oh,
    stride_kp,
    stride_vp,
    NUM_M,
    BS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CONTIG: tl.constexpr,
    SWAP_GRID: tl.constexpr,
    H_Q: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    PRESCALE_Q: tl.constexpr,
    DOT_ACC: tl.constexpr,
):
    if SWAP_GRID == 2:
        pid = tl.program_id(0)
        n_tot = NUM_M * H_Q * BS
        chunk = (n_tot + NUM_XCDS - 1) // NUM_XCDS
        rid = (pid % NUM_XCDS) * chunk + pid // NUM_XCDS
        if rid >= n_tot:
            return
        cur_head = rid % H_Q
        pid_m = (rid // H_Q) % NUM_M
        cur_b = rid // (H_Q * NUM_M)
    elif SWAP_GRID == 1:
        cur_head = tl.program_id(0)
        pid_m = tl.program_id(1)
        cur_b = tl.program_id(2)
    else:
        pid_m = tl.program_id(0)
        cur_head = tl.program_id(1)
        cur_b = tl.program_id(2)

    q_start = tl.load(CU_Q + cur_b).to(tl.int32)
    q_end = tl.load(CU_Q + cur_b + 1).to(tl.int32)
    q_len = q_end - q_start

    m_start = pid_m * BLOCK_M
    if m_start >= q_len:
        return

    kv_start = tl.load(KV_INDPTR + cur_b).to(tl.int32)
    kv_end = tl.load(KV_INDPTR + cur_b + 1).to(tl.int32)
    kv_len = kv_end - kv_start
    delta = kv_len - q_len

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < q_len
    q_ptrs = (
        Q
        + (q_start + offs_m)[:, None] * stride_qt
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    if PRESCALE_Q > 0:
        q = (q * q_pre_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # fully unmasked range: every row m in the tile can see kv j < m_start + delta + 1
    n_bulk = ((m_start + delta + 1) // BLOCK_N) * BLOCK_N
    # last kv position any row in this tile can see (exclusive)
    n_end = tl.minimum(m_start + BLOCK_M + delta, kv_len)

    page_base = KV_PAGE + kv_start
    pg0 = tl.load(KV_PAGE + kv_start).to(tl.int32)

    # ---------------- BULK: no mask at all ----------------
    for start_n in range(0, n_bulk, BLOCK_N):
        if CONTIG:
            pg = pg0 + start_n + offs_n
        else:
            pg = tl.load(page_base + start_n + offs_n).to(tl.int32)
        k = tl.load(K + pg[None, :] * stride_kp + offs_d[:, None])
        qk = tl.dot(q, k)
        if PRESCALE_Q != 1:
            qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + pg[:, None] * stride_vp + offs_d[None, :])
        if DOT_ACC:
            acc = tl.dot(p.to(v.dtype), v, acc)
        else:
            acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # ---------------- TAIL: causal diagonal ----------------
    for start_n in range(n_bulk, n_end, BLOCK_N):
        n_idx = start_n + offs_n
        mask_n = n_idx < kv_len
        if CONTIG:
            pg = pg0 + tl.where(mask_n, n_idx, 0)
        else:
            pg = tl.load(page_base + n_idx, mask=mask_n, other=0).to(tl.int32)
        k = tl.load(K + pg[None, :] * stride_kp + offs_d[:, None])
        qk = tl.dot(q, k)
        if PRESCALE_Q != 1:
            qk = qk * qk_scale
        causal = n_idx[None, :] <= (offs_m[:, None] + delta)
        qk = tl.where(causal & mask_n[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + pg[:, None] * stride_vp + offs_d[None, :])
        if DOT_ACC:
            acc = tl.dot(p.to(v.dtype), v, acc)
        else:
            acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out
        + (q_start + offs_m)[:, None] * stride_ot
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


SUPPORTED_HEAD_DIMS = (64, 256)
"""Head dims this kernel is both CORRECT and FASTER on, measured on gfx950.

Restricted rather than "any power of two" on purpose:

* 192 (and any non-power-of-two) fails to COMPILE -- `tl.arange(0, HEAD_DIM)`
  requires a power of two. Loud, but still must not be dispatched.
* 128 is correct but SLOWER than CK-tile: measured 0.930x (CK reaches 863 TFLOP/s
  at that head dim, versus 663 at 256). CK's tiling is tuned for 128, which is the
  head dim most models use; there is nothing to win there.
* 64 (1.445x) and 256 (1.277x) are where CK leaves throughput on the table.
"""


def is_supported(
    q,
    k_cache,
    v_cache=None,
    *,
    causal=True,
    logits_soft_cap=0.0,
    alibi_slopes=None,
    return_lse=False,
    return_attn_probs=False,
    window_size=(-1, -1),
    sink_ptr=None,
    sink_size=0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    kv_block_descale=None,
    dropout_p=0.0,
    kv_last_page_lens=None,
    block_table=None,
    seqlen_k=None,
):
    """Return (supported, reason). Callers MUST gate on this before dispatching.

    This covers every argument of `aiter.ops.mha.mha_batch_prefill_func` that changes
    what the op computes, as of that function's current signature. Arguments the
    CK-tile path itself ignores (`deterministic`) are not listed. Note that the caller
    names these explicitly, so an argument added to that function later will not reach
    this predicate: it has to be added here in the same change, or calls using it will
    be dispatched with it silently ignored.

    Most conditions describe a feature this kernel does not implement, and getting
    one wrong would be loud. The four that would be SILENT are:

    * ``k_cache``/``v_cache`` must present exactly ONE kv head. K/V are addressed as
      ``base + page * stride(0) + offs_d`` with no kv-head term, so with more than one
      kv head every query head reads kv head 0 and the kernel returns a
      plausible-looking wrong answer (measured rms error ~1.0 at num_kv_heads=2).
    * ``kv_block_descale`` is a per-page K/V scale the kernel never applies.
    * the head dim of V must equal that of Q, since one ``HEAD_DIM`` serves both.
    * the last dim of q/k/v must be unit-stride, because ``offs_d`` is added to the
      base pointer directly rather than multiplied by a stride.

    One kv head is the normal case for a GQA model whose kv heads are replicated
    across tensor-parallel ranks (4 kv heads at TP8 gives 1 per rank), which is the
    configuration this was developed and measured against.
    """
    # --- features not implemented (all would be loud, but must not dispatch) ---
    if not causal:
        return False, "non-causal not implemented"
    if logits_soft_cap:
        return False, "logits_soft_cap not implemented"
    if alibi_slopes is not None:
        return False, "alibi_slopes not implemented"
    if return_lse or return_attn_probs:
        return False, "return_lse / return_attn_probs not implemented"
    if tuple(window_size) != (-1, -1):
        return False, "sliding window not implemented"
    if sink_ptr is not None or sink_size:
        return False, "attention sinks not implemented"
    if q_descale is not None or k_descale is not None or v_descale is not None:
        return False, "fp8 descales not implemented"
    if dropout_p:
        return False, "dropout not implemented"
    # block_table / kv_last_page_lens / seqlen_k select CK-only paged layouts
    if block_table is not None:
        return False, "block_table layout not implemented"
    if kv_last_page_lens is not None:
        return False, "kv_last_page_lens not implemented"
    if seqlen_k is not None:
        return False, "seqlen_k not implemented"

    # --- conditions that would be SILENT if they were not checked here ---
    if kv_block_descale is not None:
        return False, "kv_block_descale (per-page K/V scale) not applied by this kernel"
    if q.dim() != 3:
        return False, f"expected q of rank 3 (tokens, heads, dim), got {q.dim()}"
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        if t is None:
            continue
        # rank 3 == the page_size=1 linear layout; shape[1] is the kv-head count
        if t.dim() != 3 or t.shape[1] != 1:
            return False, (
                f"requires page_size=1 with a single kv head, got "
                f"{name} {tuple(t.shape)}"
            )
    if v_cache is not None and v_cache.shape[-1] != q.shape[-1]:
        return False, (
            f"head dim of V ({v_cache.shape[-1]}) must equal that of Q "
            f"({q.shape[-1]})"
        )
    for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache)):
        if t is not None and t.stride(-1) != 1:
            return False, f"{name} must be unit-stride in its last dim"
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False, f"head_dim {q.shape[-1]} not in {SUPPORTED_HEAD_DIMS}"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {q.dtype} not supported"
    if k_cache.dtype != q.dtype or (v_cache is not None and v_cache.dtype != q.dtype):
        return False, "q/k/v dtypes must match"
    return True, ""


_LOG2E = 1.4426950408889634


def mha_batch_prefill_func(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    max_seqlen_k,
    causal=True,
    logits_soft_cap=0.0,
    alibi_slopes=None,
    return_lse=False,
    return_attn_probs=False,
    window_size=(-1, -1),
    sink_ptr=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out=None,
    softmax_scale=None,
):
    ok, why = is_supported(
        q,
        k_cache,
        v_cache,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
        window_size=window_size,
        sink_ptr=sink_ptr,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    if not ok:
        raise NotImplementedError(
            f"fused_paged_prefill does not support this call: {why}"
        )

    _, H, D = q.shape
    sm_scale = softmax_scale if softmax_scale is not None else D**-0.5
    cfg = _cfg()
    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]

    _mode = int(cfg.get("PRESCALE_Q", 0))
    _full = sm_scale * _LOG2E
    if _mode == 1:
        _q_pre, _qk_scale = _full, 1.0
    elif _mode == 2:
        # split into an EXACT power-of-two prescale (bit-exact in bf16) + a residual on qk
        import math as _m

        _e = _m.floor(_m.log2(_full))
        _q_pre = 2.0**_e
        _qk_scale = _full / _q_pre
    else:
        _q_pre, _qk_scale = 1.0, _full

    if out is None:
        o = torch.empty_like(q)
    else:
        if out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
            raise ValueError("out must match q in shape/dtype and be unit-stride")
        o = out
    bs = cu_seqlens_q.shape[0] - 1
    num_m = triton.cdiv(int(max_seqlen_q), BLOCK_M)

    _swap = int(cfg.get("SWAP_GRID", 0))
    if _swap == 2:
        _grid = (num_m * H * bs, 1, 1)
    elif _swap == 1:
        _grid = (H, num_m, bs)
    else:
        _grid = (num_m, H, bs)
    _fused_paged_prefill_kernel[_grid](
        q,
        k_cache,
        v_cache,
        o,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        _qk_scale,
        _q_pre,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_cache.stride(0),
        v_cache.stride(0),
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CONTIG=bool(cfg.get("CONTIG", 0)),
        SWAP_GRID=_swap,
        NUM_M=num_m,
        H_Q=H,
        BS=bs,
        NUM_XCDS=int(cfg.get("NUM_XCDS", 8)),
        PRESCALE_Q=int(_mode),
        DOT_ACC=bool(cfg.get("DOT_ACC", 1)),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        kpack=cfg["kpack"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )
    return o
