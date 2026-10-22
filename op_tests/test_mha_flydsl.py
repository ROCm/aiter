# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit test + perf sweep for the FlyDSL FMHA forward-prefill kernels on gfx1250.

Two layouts, one table each:
  - ``thd``  packed ``[total_tokens, H, D]``, variable-length via ``cu_seqlens``
  - ``bshd`` batched ``[B, S, H, D]``, uniform ``seq_len``

Covers causal, non-causal, sq != sk, seqlen_k == 0, mixed zero/nonzero batches,
GQA/MQA, attention sink, sliding window, and return_lse.

Both suites drive the public ``aiter.ops.mha`` entry points, so a case is only
swept when the routing in ``aiter/ops/mha.py`` + ``aiter/ops/flydsl/fmha_kernels.py``
lands on the FlyDSL m32x8 kernel; anything else would be testing CK/ASM/Triton.

Usage:
    python op_tests/test_mha_flydsl.py
"""

import argparse
import itertools
import math
import os
import random

# This test exists to exercise the FlyDSL m32x8 kernel, so take every shape it can
# serve away from ASM/CK. Read live by both dispatchers, so setting it here is enough.
os.environ["AITER_ENABLE_EXPERIMENTAL"] = "1"

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    supports_gqa_ratio,
)
from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import dtypes

SUPPORTED_GFX = ["gfx1250"]

# (d_qk, d_v) pairs the FlyDSL kernels can serve at all. Anything else is
# rejected up front rather than silently falling through to CK/Triton.
SUPPORTED_D_QK_V = [
    (256, 128),
    (192, 128),
    (128, 128),
    (64, 64),
    (256, 64),
    (192, 64),
    (128, 64),
    (64, 128),
]

# Kernels that can be timed per case. "flydsl" is the path under test; "triton"
# is an optional cross-check for plain (no sink / no window) thd configs.
KERNELS = ["flydsl", "triton"]

# Pin the arg-rotation count. Left to itself, run_perftest derives it from
# `free_memory` at call time, so two rows timed in one process rotate a
# different number of times and land in different L2 states -- their `us`
# columns then are not comparable with each other.
_ROTATE = 4

# Timing loop size. This file is a correctness harness: 101 timed iterations per
# row used to dominate a sweep's wall clock for a `us` column nobody reads here.
# Default is now a token loop; `--bench` restores a converged one. Either way,
# measure perf with op_tests/op_benchmarks/flydsl/bench_mha_fwd.py, not with this script.
_ITERS = 3
_WARMUP = 1


# ============================================================================
# Masks / reference
# ============================================================================


def _expand_kv(t, gqa, dim=-2):
    """Broadcast nheads_k KV heads up to nheads_q for the reference."""
    return t if gqa == 1 else t.repeat_interleave(gqa, dim=dim)


def _local_mask(sq, sk, window_left, window_right, device):
    """Sliding-window mask, bottom-right aligned (offset ``sk - sq``). Returns a
    ``[sq, sk]`` bool tensor, True where masked out (outside the band).
    ``window_left``/``window_right == -1`` means infinite on that side.

    Query row ``s`` attends kv col ``j`` in the inclusive band
    ``s + (sk - sq) - left <= j <= s + (sk - sq) + right``, so
    ``_local_mask(sq, sk, -1, 0)`` is exactly the bottom-right causal mask.
    """
    row_idx = torch.arange(sq, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(sk, device=device, dtype=torch.long)
    offset = sk - sq
    masks = []
    if window_right >= 0:
        masks.append(col_idx > row_idx + offset + window_right)
    if window_left >= 0:
        masks.append(col_idx < row_idx + offset - window_left)
    if not masks:
        return torch.zeros(sq, sk, device=device, dtype=torch.bool)
    m = masks[0]
    for extra in masks[1:]:
        m = torch.logical_or(m, extra)
    return m


def _effective_window(causal, window_size):
    """Effective window: ``causal`` pins the right bound to 0 and leaves the left
    bound as given, matching ``flash_attn_*_m32x8``."""
    return window_size[0], (0 if causal else window_size[1])


def _make_sink(nheads_q, device):
    """Per-head fp32 sink logits in the scaled-score domain (one extra virtual
    zero-value KV column in the softmax denominator). Spread so some heads' sink
    dominates the running max and some are negligible."""
    return torch.linspace(-2.0, 5.0, nheads_q, dtype=torch.float32, device=device)


def _apply_sink(qk, sink):
    """Append the per-head virtual sink column to scaled scores ``qk``
    (``[H_q, sq, sk]``) so softmax/logsumexp pick it up."""
    if sink is None:
        return qk
    sink_col = sink.to(qk.dtype).view(-1, 1, 1).expand(qk.shape[0], qk.shape[1], 1)
    return torch.cat([qk, sink_col], dim=-1)


def run_torch_varlen(q, k, v, cu_q, cu_k, scale, causal, return_lse, sink, window_size):
    """PyTorch reference for varlen THD layout, per-batch.

    Causal is routed through the local (sliding-window) mask: ``causal`` forces
    the effective right window to 0 while leaving the left window as given, so
    ``_local_mask(sq, sk, -1, 0)`` reproduces the pure-causal mask exactly.
    A batch with ``kv_len == 0`` yields O = 0 and LSE = -inf (or ``sink[h]``).
    """
    B = len(cu_q) - 1
    gqa = q.shape[1] // k.shape[1]
    wl, wr = _effective_window(causal, window_size)
    apply_mask = causal or tuple(window_size) != (-1, -1)
    outs, lses = [], []
    for b in range(B):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        qb = q[cu_q[b] : cu_q[b + 1]].float()
        kb = _expand_kv(k[cu_k[b] : cu_k[b + 1]].float(), gqa)
        vb = _expand_kv(v[cu_k[b] : cu_k[b + 1]].float(), gqa)
        qk = torch.bmm(qb.permute(1, 0, 2), kb.permute(1, 2, 0)) * scale
        if apply_mask:
            qk = qk.masked_fill(
                _local_mask(sq, sk, wl, wr, qk.device).unsqueeze(0), float("-inf")
            )
        qk_aug = _apply_sink(qk, sink)
        if return_lse:
            lses.append(torch.logsumexp(qk_aug, dim=-1))
        p = torch.softmax(qk_aug, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # all-masked rows: softmax(-inf) = NaN -> 0
        if sink is not None:
            p = p[..., :-1]  # drop virtual sink column (zero value -> 0 contribution)
        outs.append(torch.bmm(p, vb.permute(1, 0, 2)).permute(1, 0, 2))
    out = torch.cat(outs, dim=0)
    return (out, lses) if return_lse else out


def run_torch_batch(q, k, v, scale, causal, return_lse, sink, window_size):
    """PyTorch reference for batched BSHD layout, per-batch.

    Loops over the batch so peak scratch stays at one ``[H_q, sq, sk]`` score
    matrix instead of ``[B, H_q, sq, sk]``. Causal is routed through the local
    (sliding-window) mask (see ``run_torch_varlen``).
    """
    B, sq, _, _ = q.shape
    sk = k.shape[1]
    gqa = q.shape[2] // k.shape[2]
    wl, wr = _effective_window(causal, window_size)
    apply_mask = causal or tuple(window_size) != (-1, -1)
    outs, lses = [], []
    for b in range(B):
        qb = q[b].float().permute(1, 0, 2)  # [H_q, sq, d_qk]
        kb = _expand_kv(k[b].float(), gqa).permute(1, 0, 2)
        vb = _expand_kv(v[b].float(), gqa).permute(1, 0, 2)
        qk = torch.bmm(qb, kb.transpose(1, 2)) * scale
        if apply_mask:
            qk = qk.masked_fill(
                _local_mask(sq, sk, wl, wr, qk.device).unsqueeze(0), float("-inf")
            )
        qk_aug = _apply_sink(qk, sink)
        if return_lse:
            lses.append(torch.logsumexp(qk_aug, dim=-1))  # [H_q, sq]
        p = torch.softmax(qk_aug, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        if sink is not None:
            p = p[..., :-1]
        outs.append(torch.bmm(p, vb).permute(1, 0, 2))  # [sq, H_q, d_v]
    out = torch.stack(outs, dim=0)
    return (out, torch.stack(lses, dim=0)) if return_lse else out  # lse [B, H_q, sq]


# ============================================================================
# Roofline
# ============================================================================


def _flops(shapes, H_q, d_qk, d_v, causal):
    """FLOPs for forward: sum per-batch QK^T + PV, causal halves each batch."""
    total = 0
    for sq, sk in shapes:
        f = H_q * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
        total += f // 2 if causal else f
    return total


def _nbytes(tot_q, tot_k, H_q, H_kv, d_qk, d_v, elem_size, return_lse):
    """Q + K + V reads and O (+ LSE) writes. Ignores re-reads of K/V across
    M-blocks, so this is the compulsory-traffic lower bound."""
    n = (
        tot_q * H_q * d_qk  # Q
        + tot_k * H_kv * d_qk  # K
        + tot_k * H_kv * d_v  # V
        + tot_q * H_q * d_v  # O
    ) * elem_size
    if return_lse:
        n += tot_q * H_q * 4
    return n


# ============================================================================
# Dispatch filter -- only sweep pairs the FlyDSL kernel can serve
# ============================================================================


def _flydsl_serves(d_qk, d_v):
    """True when the FlyDSL m32x8 kernel serves this (d_qk, d_v).

    AITER_ENABLE_EXPERIMENTAL is forced on at import, so both routers hand us every
    pair we support regardless of what ASM or CK would otherwise claim; the pair is
    the only thing left to filter on.
    """
    return (d_qk, d_v) in SUPPORTED_D_QK_V


# ============================================================================
# Timed entry points (module level so run_perftest can rotate their tensor args)
# ============================================================================


def _call_varlen(q, k, v, cu_q, cu_k, max_sq, max_sk, scale, causal, window, lse, sink):
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_sq,
        max_sk,
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        return_lse=lse,
        sink_ptr=sink,
    )


def _call_varlen_triton(
    q, k, v, cu_q, cu_k, max_sq, max_sk, scale, causal, window, lse, sink
):
    """Cross-check candidate. Shares ``_call_varlen``'s signature so the same
    tensors get rotated; ``window``/``lse``/``sink`` are ignored."""
    from aiter.ops.triton.attention.mha import flash_attn_varlen_func as _tri

    return _tri(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_sq,
        max_seqlen_k=max_sk,
        softmax_scale=scale,
        causal=causal,
    )


def _call_batch(q, k, v, scale, causal, window, lse, sink):
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        return_lse=lse,
        sink_ptr=sink,
    )


# ============================================================================
# Test functions -- one @benchmark per arg signature, one table each
# ============================================================================


@benchmark()
def test_mha_flydsl_varlen(
    seqs_q,
    seqs_k,
    H_q,
    H_kv,
    d_qk,
    d_v,
    dtype,
    causal,
    return_lse,
    sink,
    window,
    kernels,
):
    """One varlen (thd) case. ``seqs_q``/``seqs_k`` are per-batch lengths."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    assert (d_qk, d_v) in SUPPORTED_D_QK_V, f"unsupported d_qk/d_v {(d_qk, d_v)}"
    assert H_q % H_kv == 0, f"nheads_q={H_q} must be a multiple of nheads_kv={H_kv}"

    B = len(seqs_q)
    cu_q = [0]
    cu_k = [0]
    for s in seqs_q:
        cu_q.append(cu_q[-1] + s)
    for s in seqs_k:
        cu_k.append(cu_k[-1] + s)
    total_q, total_k = cu_q[-1], cu_k[-1]
    max_sq, max_sk = max(seqs_q), max(seqs_k)

    q = torch.randn(total_q, H_q, d_qk, dtype=dtype, device=device)
    k = torch.randn(total_k, H_kv, d_qk, dtype=dtype, device=device)
    v = torch.randn(total_k, H_kv, d_v, dtype=dtype, device=device)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=device)

    scale = 1.0 / math.sqrt(d_qk)
    sink_t = _make_sink(H_q, device) if sink else None

    ref_result = run_torch_varlen(
        q, k, v, cu_q, cu_k, scale, causal, return_lse, sink_t, window
    )
    ref, ref_lses = ref_result if return_lse else (ref_result, None)

    candidates = {"flydsl": _call_varlen}
    # Triton has no sink / sliding-window support here, so cross-check it only on
    # plain configs; elsewhere its column stays nan.
    if "triton" in kernels and not sink and tuple(window) == (-1, -1):
        candidates["triton"] = _call_varlen_triton

    flops = _flops(list(zip(seqs_q, seqs_k)), H_q, d_qk, d_v, causal)
    nbytes = _nbytes(
        total_q, total_k, H_q, H_kv, d_qk, d_v, q.element_size(), return_lse
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        data, us = run_perftest(
            fn,
            q,
            k,
            v,
            cu_q_t,
            cu_k_t,
            max_sq,
            max_sk,
            scale,
            causal,
            window,
            return_lse and name == "flydsl",
            sink_t,
            num_rotate_args=_ROTATE,
            num_iters=_ITERS,
            num_warmup=_WARMUP,
        )
        # Only the flydsl call was asked for an LSE.
        if isinstance(data, (tuple, list)):
            out = data[0]
            lse = data[1] if (return_lse and name == "flydsl") else None
        else:
            out, lse = data, None

        assert tuple(out.shape) == (
            total_q,
            H_q,
            d_v,
        ), f"{name}: bad out shape {tuple(out.shape)}"
        err = checkAllclose(
            ref,
            out.float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} thd out: ",
        )

        if lse is not None:
            # Kernel LSE is fp32 [nheads_q, total_q] (the aiter varlen convention,
            # asserted here because test_mha_varlen.py, test_flydsl_fmha.py and the
            # bwd test all slice it as lse[:, cu[b]:cu[b+1]]); the reference is
            # per batch [nheads_q, sq]. Empty-KV batches carry -inf (or sink[h])
            # and are compared the same way -- isclose(-inf, -inf) holds.
            assert lse.dtype == torch.float32, f"lse dtype {lse.dtype}"
            assert tuple(lse.shape) == (
                H_q,
                total_q,
            ), f"bad thd lse shape {tuple(lse.shape)}"
            for b in range(B):
                got = lse[:, cu_q[b] : cu_q[b + 1]]
                if seqs_k[b] == 0:
                    exp_lse = (
                        sink_t.view(-1, 1).expand(H_q, seqs_q[b])
                        if sink_t is not None
                        else torch.full((H_q, seqs_q[b]), float("-inf"), device=device)
                    )
                else:
                    exp_lse = ref_lses[b]
                err = max(
                    err,
                    checkAllclose(
                        exp_lse,
                        got,
                        rtol=1e-2,
                        atol=1e-2,
                        msg=f"{name} thd lse batch {b}: ",
                    ),
                )

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mha_flydsl_batch(
    B, sq, sk, H_q, H_kv, d_qk, d_v, dtype, causal, return_lse, sink, window
):
    """One batched (bshd) case: ``[B, S, H, D]``, uniform seq_len, no cu_seqlens."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    assert (d_qk, d_v) in SUPPORTED_D_QK_V, f"unsupported d_qk/d_v {(d_qk, d_v)}"
    assert H_q % H_kv == 0, f"nheads_q={H_q} must be a multiple of nheads_kv={H_kv}"

    q = torch.randn(B, sq, H_q, d_qk, dtype=dtype, device=device)
    k = torch.randn(B, sk, H_kv, d_qk, dtype=dtype, device=device)
    v = torch.randn(B, sk, H_kv, d_v, dtype=dtype, device=device)

    scale = 1.0 / math.sqrt(d_qk)
    sink_t = _make_sink(H_q, device) if sink else None

    ref_result = run_torch_batch(q, k, v, scale, causal, return_lse, sink_t, window)
    ref, ref_lse = ref_result if return_lse else (ref_result, None)

    flops = _flops([(sq, sk)] * B, H_q, d_qk, d_v, causal)
    nbytes = _nbytes(B * sq, B * sk, H_q, H_kv, d_qk, d_v, q.element_size(), return_lse)

    ret = {"gfx": get_gfx()}
    data, us = run_perftest(
        _call_batch,
        q,
        k,
        v,
        scale,
        causal,
        window,
        return_lse,
        sink_t,
        num_rotate_args=_ROTATE,
        num_iters=_ITERS,
        num_warmup=_WARMUP,
    )
    out, lse = data if return_lse else (data, None)

    assert tuple(out.shape) == (
        B,
        sq,
        H_q,
        d_v,
    ), f"bad out shape {tuple(out.shape)}"
    err = checkAllclose(ref, out.float(), rtol=1e-2, atol=1e-2, msg="flydsl bshd out: ")
    if lse is not None:
        # Kernel LSE is fp32 [B, nheads_q, sq], the layout test_mha.py asserts.
        assert lse.dtype == torch.float32, f"lse dtype {lse.dtype}"
        assert tuple(lse.shape) == (
            B,
            H_q,
            sq,
        ), f"bad bshd lse shape {tuple(lse.shape)}"
        err = max(
            err,
            checkAllclose(ref_lse, lse, rtol=1e-2, atol=1e-2, msg="flydsl bshd lse: "),
        )

    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = err
    return ret


# ============================================================================
# Shapes
# ============================================================================

# thd: (seqs_q, seqs_k, nheads_q, nheads_kv) -- per-batch lengths.
THD_SHAPES = [
    # --- basic sq == sk ---
    ([128], [128], 1, 1),
    ([184], [184], 128, 128),
    ([341], [341], 128, 128),
    ([5], [5], 128, 128),
    # --- multi-batch ---
    ([481, 100, 401], [481, 100, 401], 128, 128),
    # --- sq != sk ---
    ([128], [512], 1, 1),
    ([128], [256], 1, 1),
    ([128, 128], [512, 512], 1, 1),
    ([128], [512], 2, 2),
    ([128, 128], [256, 256], 2, 2),
    # --- sq << sk (decode-like) ---
    ([72], [600], 1, 1),
    ([72], [600], 2, 2),
    ([1], [512], 1, 1),
    ([1], [512], 2, 2),
    ([16], [1024], 2, 2),
    ([72, 72], [600, 600], 2, 2),
    ([1, 128], [512, 1024], 2, 2),
    ([72, 1], [600, 256], 4, 4),
    # --- noncausal various sq/sk ---
    ([128, 256], [128, 256], 1, 1),
    ([128, 256], [256, 384], 2, 2),
    ([300], [300], 2, 2),
    ([128, 128], [256, 256], 4, 4),
    # --- cu_q != cu_k (chunked prefill) ---
    ([693, 692, 461], [693, 692, 701], 128, 128),
    # --- seqlen_k == 0 (output must be all zeros) ---
    ([128], [0], 1, 1),
    ([256], [0], 2, 2),
    ([128, 128], [0, 0], 1, 1),
    ([300], [0], 4, 4),
    # --- mixed seqlen_k == 0 (some batches zero) ---
    ([128, 128], [0, 128], 1, 1),
    ([128, 128, 128], [0, 0, 128], 1, 1),
    # --- GQA (nheads_q > nheads_kv) ---
    ([128], [128], 2, 1),
    ([256], [256], 8, 1),  # MQA
    ([341], [341], 8, 2),  # non-tile-multiple seq
    ([512], [512], 16, 4),
    ([128], [128], 32, 1),  # gqa 32: packed rows are a head SLICE, not whole seqs
    ([128], [128], 48, 1),  # gqa 48: only num_q_tiles=1 packs it
    ([128], [512], 6, 3),  # non-power-of-two nheads
    ([128, 256], [256, 384], 8, 2),  # multi-batch, sq != sk
    ([72], [600], 32, 4),  # decode-like
    ([481, 100, 401], [481, 100, 401], 32, 8),
    ([128, 128], [0, 128], 8, 2),  # mixed seqlen_k == 0
    ([1024], [1024], 32, 4),
    # --- larger shapes ---
    ([512], [512], 128, 128),
    ([1024], [1024], 128, 128),
    ([256, 256, 256, 256], [256, 256, 256, 256], 128, 128),
    ([128], [2048], 128, 128),
    ([1], [512], 128, 128),
]

# bshd: (batch, seqlen_q, seqlen_k, nheads_q, nheads_kv).
BSHD_SHAPES = [
    # --- basic sq == sk ---
    (1, 128, 128, 1, 1),
    (1, 128, 128, 8, 8),
    (2, 256, 256, 8, 8),
    (4, 512, 512, 8, 8),
    # --- non-tile-multiple / tiny seq ---
    (1, 5, 5, 8, 8),
    (1, 341, 341, 4, 4),
    (2, 184, 184, 4, 4),
    # --- sq != sk ---
    (1, 128, 512, 1, 1),
    (2, 128, 512, 4, 4),
    (2, 300, 1024, 2, 2),
    # --- sq << sk (decode-like) ---
    (1, 1, 512, 8, 8),
    (2, 16, 1024, 4, 4),
    # --- GQA (nheads_q > nheads_kv) ---
    (1, 128, 128, 2, 1),
    (2, 256, 256, 8, 1),  # MQA
    (1, 341, 341, 8, 2),  # non-tile-multiple seq
    (2, 512, 512, 16, 4),
    (1, 128, 128, 32, 1),  # gqa 32: packed rows are a head SLICE, not whole seqs
    (1, 128, 128, 48, 1),  # gqa 48: only num_q_tiles=1 packs it
    (1, 128, 512, 6, 3),  # non-power-of-two nheads, sq != sk
    (1, 16, 1024, 32, 4),  # decode-like
    (2, 1024, 1024, 32, 8),
    # --- larger shapes ---
    (1, 2048, 2048, 8, 8),
    (1, 4096, 4096, 4, 4),
]


def _rand_seqs(B, max_sq, max_sk, rng):
    """Random per-batch lengths with ``1 <= sq_i <= sk_i <= max_sk``."""
    if min(max_sq, max_sk) < 1:
        raise SystemExit(
            f"--rand-seqlens draws in [1, len] and has no form at zero "
            f"(got -sq {max_sq} -sk {max_sk})"
        )
    seqs_k = [rng.randint(1, max_sk) for _ in range(B)]
    seqs_q = [rng.randint(1, min(max_sq, sk)) for sk in seqs_k]
    return seqs_q, seqs_k


def _gqa_packs(H_q, H_kv):
    """True when the router will serve this head pair (same rule, same source)."""
    return H_q % H_kv == 0 and supports_gqa_ratio(H_q // H_kv)


# Per-axis values present in the built-in lists; an axis the user does not name
# inherits these, so a partial -b/-nh/-nhkv/-sq/-sk run is no longer silently ignored.
_AXIS_DEFAULTS = {
    "batch_size": sorted(
        {len(sq) for sq, _, _, _ in THD_SHAPES} | {B for B, *_ in BSHD_SHAPES}
    ),
    "nheads": sorted(
        {h for *_, h, _ in THD_SHAPES} | {h for _, _, _, h, _ in BSHD_SHAPES}
    ),
    "nheads_kv": sorted({k for *_, k in THD_SHAPES} | {k for *_, k in BSHD_SHAPES}),
    "seqlen_q": sorted(
        {x for sq, _, _, _ in THD_SHAPES for x in sq}
        | {sq for _, sq, _, _, _ in BSHD_SHAPES}
    ),
    "seqlen_k": sorted(
        {x for _, sk, _, _ in THD_SHAPES for x in sk}
        | {sk for _, _, sk, _, _ in BSHD_SHAPES}
    ),
}


def _shape_lists(args):
    """``(thd, bshd)`` shape lists for this run.

    With no ``-b/-nh/-nhkv/-sq/-sk`` at all, the curated built-in lists run as-is.
    Name any of them and the sweep becomes the cross product of those axes, each
    unnamed axis taking the values it has in the built-in lists (``_AXIS_DEFAULTS``).
    ``--max-cases`` then bounds what that product costs. With ``--rand-seqlens N``
    each combination contributes N random varlen draws instead of one uniform shape.
    """
    named = [args.batch_size, args.nheads, args.nheads_kv, args.seqlen_q, args.seqlen_k]
    if not any(named):
        if args.rand_seqlens:
            aiter.logger.warning(
                "--rand-seqlens needs one of -b/-nh/-nhkv/-sq/-sk; using built-in shapes"
            )
        return THD_SHAPES, BSHD_SHAPES

    def axis(name, given):
        vals = given or _AXIS_DEFAULTS[name]
        if args.rand_seqlens and not given and name.startswith("seqlen"):
            # The curated empty-KV cases contribute 0, which has no random form; they
            # stay covered by the built-in lists.
            vals = [v for v in vals if v > 0]
        return vals

    # -nhkv unset falls back to -nh (plain MHA) as the flag documents; only with neither
    # given does it take the built-in values.
    kv_heads = args.nheads_kv or args.nheads or _AXIS_DEFAULTS["nheads_kv"]
    rng = random.Random(args.seed)
    thd, bshd, skipped = [], [], 0
    for B, H_q, H_kv, sq, sk in itertools.product(
        axis("batch_size", args.batch_size),
        axis("nheads", args.nheads),
        kv_heads,
        axis("seqlen_q", args.seqlen_q),
        axis("seqlen_k", args.seqlen_k),
    ):
        if not _gqa_packs(H_q, H_kv):
            skipped += 1
            continue
        bshd.append((B, sq, sk, H_q, H_kv))
        for _ in range(args.rand_seqlens or 1):
            if args.rand_seqlens:
                seqs_q, seqs_k = _rand_seqs(B, sq, sk, rng)
            else:
                seqs_q, seqs_k = [sq] * B, [sk] * B
            thd.append((seqs_q, seqs_k, H_q, H_kv))
    if skipped:
        aiter.logger.warning(
            "skipped %d head combos the router declines (nheads %% nheads_kv, or a "
            "GQA ratio whose head group would split across waves)",
            skipped,
        )
    if not thd and not bshd:
        raise SystemExit(
            "no requested shape survives the head-packing invariant -- nothing to run"
        )
    return thd, bshd


def _str2d_qk_v(v):
    """Parse a ``d_qk,d_v`` pair and check it against ``SUPPORTED_D_QK_V``."""
    pair = dtypes.str2tuple(v)
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or tuple(pair) not in [tuple(p) for p in SUPPORTED_D_QK_V]
    ):
        raise argparse.ArgumentTypeError(
            f"d_qk,d_v must be one of {SUPPORTED_D_QK_V}, got {v!r}"
        )
    return tuple(pair)


def _str2window(v):
    """Parse a ``left,right`` sliding-window bound. ``i``/``inf`` (or ``-1``) mean
    infinite on that side; ``i`` is preferred because argparse reads a leading
    ``-1`` as an option string."""
    try:
        parts = [p.strip().lower() for p in v.strip("()").split(",")]
        if len(parts) != 2:
            raise ValueError("expected two comma-separated bounds")
        return tuple(-1 if p in ("i", "inf", "-1") else int(p) for p in parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid window: {v}") from e


def summarize(title, rows):
    if not rows:
        aiter.logger.warning("%s: no case routes to FlyDSL for this config", title)
        return
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def check_routing_declines():
    """Options m32x8 cannot honour must never reach it -- they belong to CK.

    Both escapes found so far (padded cu_seqlens, logits_soft_cap) were silent: the
    kernel ran and returned plausible-looking numbers computed under the wrong contract.
    A numeric sweep cannot catch that, so assert on the dispatch itself.
    """
    import aiter.ops.flydsl.fmha_kernels as fk

    dev, L, D, H, H_kv = "cuda", 128, 64, 8, 2
    cu = torch.tensor([0, L], dtype=dtypes.i32, device=dev)
    q = torch.randn(L, H, D, dtype=dtypes.bf16, device=dev)
    k = torch.randn(L, H_kv, D, dtype=dtypes.bf16, device=dev)
    v = torch.randn(L, H_kv, D, dtype=dtypes.bf16, device=dev)

    served = {}
    real = fk.flydsl_flash_attn_varlen_func

    def spy(*a, **kw):
        out = real(*a, **kw)
        served["hit"] = out is not None
        return out

    def ran_on_flydsl(**kw):
        served.clear()
        fk.flydsl_flash_attn_varlen_func = spy
        try:
            flash_attn_varlen_func(q, k, v, cu, cu, L, L, causal=True, **kw)
        finally:
            fk.flydsl_flash_attn_varlen_func = real
        return served.get("hit", False)

    assert ran_on_flydsl(), "baseline d64/64 should be served by m32x8"
    for label, kw in [
        ("logits_soft_cap", {"logits_soft_cap": 20.0}),
        # the screen keys off presence, not value -- these never reach the kernel
        ("cu_seqlens_q_padded", {"cu_seqlens_q_padded": cu}),
        ("cu_seqlens_k_padded", {"cu_seqlens_k_padded": cu}),
    ]:
        assert not ran_on_flydsl(**kw), f"{label} must not route to m32x8"
    aiter.logger.info("routing: m32x8 correctly declines %d unsupported options", 3)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL FMHA forward-prefill unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL FMHA forward-prefill unit test & perf sweep (gfx1250).",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
        help="Element dtype (q/k/v share one).\n        e.g.: -d bf16",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        choices=["thd", "bshd"],
        default=["thd", "bshd"],
        help="Layouts to sweep.\n        e.g.: -l thd",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        type=str,
        nargs="*",
        choices=KERNELS,
        default=["flydsl"],
        help="Kernels to time per thd case.\n        e.g.: -k flydsl triton",
    )
    parser.add_argument(
        "-s",
        "--d_qk_v",
        type=_str2d_qk_v,
        nargs="*",
        default=[],
        help="(d_qk, d_v) for both suites. Empty (the default) runs every supported\n"
        f"        pair.\n        Supported: {SUPPORTED_D_QK_V}.\n        e.g.: -s 192,128",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="Causal masking.\n        e.g.: -c 1",
    )
    parser.add_argument(
        "-e",
        "--return_lse",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="Return log-sum-exp.\n        e.g.: -e 0",
    )
    parser.add_argument(
        "-n",
        "--sink",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="Attention sink.\n        e.g.: -n 1",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="*",
        default=None,
        help="Batch size. With -nh/-sq/-sk, replaces the built-in shape lists.\n"
        "        e.g.: -b 1 2 4",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        nargs="*",
        default=None,
        help="Number of query heads.\n        e.g.: -nh 8 32",
    )
    parser.add_argument(
        "-nhkv",
        "--nheads_kv",
        type=int,
        nargs="*",
        default=None,
        help="Number of key/value heads (GQA); must divide --nheads.\n"
        "        Defaults to --nheads (plain MHA).\n        e.g.: -nhkv 1 2",
    )
    parser.add_argument(
        "-sq",
        "--seqlen_q",
        type=int,
        nargs="*",
        default=None,
        help="Query sequence length.\n        e.g.: -sq 128 512",
    )
    parser.add_argument(
        "-sk",
        "--seqlen_k",
        type=int,
        nargs="*",
        default=None,
        help="Key sequence length.\n        e.g.: -sk 512 2048",
    )
    parser.add_argument(
        "--rand-seqlens",
        type=int,
        default=0,
        metavar="N",
        help="Draw N random per-batch thd seqlen sets per shape combo instead of\n"
        "        a uniform one (sq_i <= sk_i). Needs -b/-nh/-sq/-sk. 0 = off.\n"
        "        e.g.: --rand-seqlens 4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for --rand-seqlens and --max-cases sampling.\n        e.g.: --seed 7",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=256,
        help="Cap each layout's sweep at N cases, sampled at random from the full\n"
        "        cross product (0 = no cap).\n        e.g.: --max-cases 0",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=_str2window,
        nargs="*",
        default=[(-1, -1)],
        help="Sliding-window (left,right) bounds; 'i' = infinite on that side.\n"
        "        ('i' rather than -1: argparse reads a leading -1 as a flag.)\n"
        "        e.g.: -w i,i 64,64 128,0 256,i i,256",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Restore a converged timing loop (101 iters / 2 warmup) so the us and\n"
        "        TFLOPS columns mean something. Off by default -- a correctness sweep\n"
        "        pays for the compile and the allclose only.\n"
        "        e.g.: --bench",
    )
    args = parser.parse_args()

    if args.bench:
        global _ITERS, _WARMUP
        _ITERS, _WARMUP = 101, 2

    causals = [bool(c) for c in args.causal]
    lses = [bool(e) for e in args.return_lse]
    sinks = [bool(n) for n in args.sink]
    windows = [tuple(w) for w in args.window]
    check_routing_declines()
    thd_shapes, bshd_shapes = _shape_lists(args)
    # No -s: run every pair the kernel serves. With -b/-nh/-sq/-sk also unset that is
    # the full default sweep CI runs; any explicit flag narrows only what it names.
    d_qk_v = args.d_qk_v or SUPPORTED_D_QK_V
    case_rng = random.Random(args.seed)

    def _cases(hdims, shapes, label):
        """The sweep for one layout, randomly thinned to --max-cases.

        The axes multiply fast (hdim pairs x shapes x causal x lse x sink x window), so
        sample the product rather than trim an axis: every combination stays reachable,
        and a different --seed walks a different slice of it. The sample is drawn over
        product INDICES -- materializing millions of tuples to keep a few hundred costs
        hundreds of MB on a partial-axis run. ``_flydsl_serves`` depends only on the hdim
        axis, so filtering it up front leaves the index space exact."""
        axes = [
            [h for h in hdims if _flydsl_serves(*h)],
            shapes,
            args.dtype,
            causals,
            lses,
            sinks,
            windows,
        ]
        total = math.prod(len(a) for a in axes)
        if not args.max_cases or total <= args.max_cases:
            return list(itertools.product(*axes))
        aiter.logger.info(
            "%s: sampling %d of %d cases (--seed %d; --max-cases 0 runs all)",
            label,
            args.max_cases,
            total,
            args.seed,
        )
        picked = []
        for idx in case_rng.sample(range(total), args.max_cases):
            combo = []
            for a in reversed(axes):
                idx, j = divmod(idx, len(a))
                combo.append(a[j])
            picked.append(tuple(reversed(combo)))
        return picked

    if "thd" in args.layout:
        rows = []
        for (d_qk, d_v), shape, dtype, causal, lse, sink, window in _cases(
            d_qk_v, thd_shapes, "thd"
        ):
            seqs_q, seqs_k, H_q, H_kv = shape
            rows.append(
                test_mha_flydsl_varlen(
                    seqs_q,
                    seqs_k,
                    H_q,
                    H_kv,
                    d_qk,
                    d_v,
                    dtype,
                    causal,
                    lse,
                    sink,
                    window,
                    tuple(args.kernels),
                )
            )
        summarize("FlyDSL FMHA prefill thd summary", rows)

    if "bshd" in args.layout:
        rows = []
        for (d_qk, d_v), shape, dtype, causal, lse, sink, window in _cases(
            d_qk_v, bshd_shapes, "bshd"
        ):
            B, sq, sk, H_q, H_kv = shape
            rows.append(
                test_mha_flydsl_batch(
                    B,
                    sq,
                    sk,
                    H_q,
                    H_kv,
                    d_qk,
                    d_v,
                    dtype,
                    causal,
                    lse,
                    sink,
                    window,
                )
            )
        summarize("FlyDSL FMHA prefill bshd summary", rows)


if __name__ == "__main__":
    main()
