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
import random

import pandas as pd
import torch

import aiter
from aiter.jit.core import is_experimental_enabled
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import dtypes

SUPPORTED_GFX = ["gfx1250"]

# (d_qk, d_v) pairs the FlyDSL kernels can serve at all. Anything else is
# rejected up front rather than silently falling through to CK/Triton.
SUPPORTED_D_QK_V = [(256, 128), (192, 128), (128, 128)]

# Kernels that can be timed per case. "flydsl" is the path under test; "triton"
# is an optional cross-check for plain (no sink / no window) thd configs.
KERNELS = ["flydsl", "triton"]

# Pin the arg-rotation count. Left to itself, run_perftest derives it from
# `free_memory` at call time, so two rows timed in one process rotate a
# different number of times and land in different L2 states -- their `us`
# columns then are not comparable with each other.
_ROTATE = 4


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
# Routing mirrors -- only sweep configs that really reach the FlyDSL kernel
# ============================================================================


def _flydsl_serves_thd(d_qk, d_v, causal, sink, window):
    """True when ``flash_attn_varlen_func`` lands on the FlyDSL m32x8 kernel.

    Mirrors ``aiter/ops/mha.py`` (the PR3039 gfx1250 ASM gate) and
    ``flydsl_flash_attn_varlen_func`` in ``aiter/ops/flydsl/fmha_kernels.py``.
    """
    if d_v != 128:
        return False
    exp = is_experimental_enabled()
    if d_qk == 256:
        return exp  # else CK
    if d_qk not in (128, 192):
        return False
    # The gfx1250 ASM kernel claims d128 only for plain causal attention with
    # no sink and no finite window; non-causal / sink / windowed d128 is ours.
    asm_claims_d128 = (
        d_qk == 128 and not exp and causal and not sink and tuple(window) == (-1, -1)
    )
    return not asm_claims_d128


def _flydsl_serves_bshd(d_qk, d_v):
    """True when ``flash_attn_func`` lands on the FlyDSL m32x8 BSHD kernel.
    No ASM/sibling competes for BSHD, so routing is head-dim only."""
    if d_v != 128:
        return False
    return d_qk in (128, 192) or (d_qk == 256 and is_experimental_enabled())


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
            # Kernel LSE is [total_q, nheads_q]; the reference is per batch
            # [nheads_q, sq]. Empty-KV batches carry -inf (or sink[h]) and are
            # compared the same way -- isclose(-inf, -inf) holds.
            for b in range(B):
                if seqs_k[b] == 0:
                    exp_lse = (
                        sink_t.view(1, -1).expand(seqs_q[b], H_q)
                        if sink_t is not None
                        else torch.full((seqs_q[b], H_q), float("-inf"), device=device)
                    )
                    got = lse[cu_q[b] : cu_q[b + 1]]
                else:
                    exp_lse = ref_lses[b]
                    got = lse[cu_q[b] : cu_q[b + 1]].permute(1, 0)
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
        # Kernel LSE is [B, nheads_q, sq]; the reference matches that layout.
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
    (1, 128, 512, 6, 3),  # non-power-of-two nheads, sq != sk
    (1, 16, 1024, 32, 4),  # decode-like
    (2, 1024, 1024, 32, 8),
    # --- larger shapes ---
    (1, 2048, 2048, 8, 8),
    (1, 4096, 4096, 4, 4),
]


def _rand_seqs(B, max_sq, max_sk, rng):
    """Random per-batch lengths with ``1 <= sq_i <= sk_i <= max_sk``."""
    seqs_k = [rng.randint(1, max_sk) for _ in range(B)]
    seqs_q = [rng.randint(1, min(max_sq, sk)) for sk in seqs_k]
    return seqs_q, seqs_k


def _shape_lists(args):
    """``(thd, bshd)`` shape lists for this run.

    The ``-b/-nh/-nhkv/-sq/-sk`` axes replace the built-in lists when all of
    ``-b/-nh/-sq/-sk`` are given, so a single shape can be driven from the CLI;
    their cross product is the sweep. With ``--rand-seqlens N`` each combination
    contributes N random varlen draws instead of one uniform ``[SQ]*B`` shape.
    """
    if not (args.batch_size and args.nheads and args.seqlen_q and args.seqlen_k):
        if args.rand_seqlens:
            aiter.logger.warning(
                "--rand-seqlens needs -b/-nh/-sq/-sk; using built-in shapes"
            )
        return THD_SHAPES, BSHD_SHAPES

    nheads_kv = args.nheads_kv or args.nheads
    rng = random.Random(args.seed)
    thd, bshd, skipped = [], [], 0
    for B, H_q, H_kv, sq, sk in itertools.product(
        args.batch_size, args.nheads, nheads_kv, args.seqlen_q, args.seqlen_k
    ):
        if H_q % H_kv:
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
        aiter.logger.warning("skipped %d combos where nheads %% nheads_kv", skipped)
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
        default=[(192, 128)],
        help="(d_qk, d_v) for the thd suite.\n"
        f"        Supported: {SUPPORTED_D_QK_V}.\n        e.g.: -s 192,128",
    )
    parser.add_argument(
        "-bs",
        "--batch_d_qk_v",
        type=_str2d_qk_v,
        nargs="*",
        default=[(128, 128)],
        help="(d_qk, d_v) for the bshd suite.\n"
        f"        Supported: {SUPPORTED_D_QK_V}.\n        e.g.: -bs 128,128",
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
        help="Seed for --rand-seqlens.\n        e.g.: --seed 7",
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
    args = parser.parse_args()

    causals = [bool(c) for c in args.causal]
    lses = [bool(e) for e in args.return_lse]
    sinks = [bool(n) for n in args.sink]
    windows = [tuple(w) for w in args.window]
    thd_shapes, bshd_shapes = _shape_lists(args)

    if "thd" in args.layout:
        rows = []
        for (d_qk, d_v), shape, dtype, causal, lse, sink, window in itertools.product(
            args.d_qk_v, thd_shapes, args.dtype, causals, lses, sinks, windows
        ):
            if not _flydsl_serves_thd(d_qk, d_v, causal, sink, window):
                continue  # CK / ASM / sibling would run -- not what this test gates
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
        for (d_qk, d_v), shape, dtype, causal, lse, sink, window in itertools.product(
            args.batch_d_qk_v, bshd_shapes, args.dtype, causals, lses, sinks, windows
        ):
            if not _flydsl_serves_bshd(d_qk, d_v):
                continue
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
