# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-KV for the gfx1250 FP8 varlen ASM forward (``fmha_fwd_fp8_varlen_asm``, D192x128).

The kernel launches ``ceil(max_seqlen_q / 256) x heads x batch`` workgroups and each one walks the whole key range.
A chunked-prefix prefill call -- 1024 new tokens, 12 heads, one sequence, a 16384-key cached chunk -- is 48
workgroups on a 256-CU GPU, so it takes about as long as a call with four times the queries.

For a non-causal call whose grid would leave CUs idle, each sequence's keys are split into ``S`` contiguous parts,
presented to the same kernel as ``S`` sequences of one varlen batch, sequence-major ``(b, 0), (b, 1), ...`` so the
split ``cu_seqlens`` stay monotone and K/V are read in place.  One small kernel writes the split ``cu_seqlens`` and
repeats each query row ``S`` times; another combines the ``S`` partial outputs by their LSE.  The result has the
layout and meaning of the unsplit call: ``out`` [total_q, H, Dv] bf16 and natural-log ``lse`` [total_q, H, 1] fp32.
Everything stays on the device (no host sync).  Causal calls are never split: a part's bottom-right mask would be
wrong.

``AITER_FMHA_FP8_SPLIT_KV``: unset or ``1`` = automatic, ``0`` = off, ``N > 1`` = always ``N`` parts.
"""

import functools
import os

import torch
import triton
import triton.language as tl

# Query rows per workgroup of the fmha_fp8_pertensor_hd192x128_256x64 kernels.
_TILE_Q = 256
# Fewer keys per part than this and the per-part fixed cost outweighs the extra parallelism.
_MIN_KEYS_PER_PART = 512
# Below this many keys the whole call is short enough that the two extra launches cost more than they save.
_MIN_KEYS_TO_SPLIT = 8192


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def split_factor(
    batch: int,
    heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    device: torch.device,
) -> int:
    """Key parts per sequence: doubled while the grid stays within the CU count and every part keeps at least
    ``_MIN_KEYS_PER_PART`` keys.  1 means "do not split"."""
    mode = os.environ.get("AITER_FMHA_FP8_SPLIT_KV", "1")
    if causal or mode == "0":
        return 1
    if mode not in ("", "1"):
        return max(1, int(mode))
    if int(max_seqlen_k) < _MIN_KEYS_TO_SPLIT:
        return 1
    tiles = -(-int(max_seqlen_q) // _TILE_Q) * heads * batch
    index = device.index if device.index is not None else torch.cuda.current_device()
    target = _num_cus(index)
    parts = 1
    while (
        tiles * parts * 2 <= target
        and int(max_seqlen_k) // (parts * 2) >= _MIN_KEYS_PER_PART
    ):
        parts *= 2
    return parts


@triton.jit
def _seq_of(cu_q, t, B, BB: tl.constexpr):
    """(start, length) of the varlen sequence holding query row ``t``."""
    ob = tl.arange(0, BB)
    nxt = tl.load(cu_q + 1 + ob, mask=ob < B, other=2147483647)
    b = tl.sum((nxt <= t).to(tl.int32))
    start = tl.load(cu_q + b)
    return start, tl.load(cu_q + b + 1) - start


@triton.jit
def _split_prep_kernel(
    q,
    q2,
    cu_q,
    cu_k,
    cu_q2,
    cu_k2,
    B,
    S: tl.constexpr,
    BB: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program ``t`` copies query row ``t`` (``ROW`` int32 words) to its ``S`` copies; program 0 also writes the
    split ``cu_seqlens``: part ``j`` of sequence ``b`` holds keys ``[ck_b + j*lk_b//S, ck_b + (j+1)*lk_b//S)``.
    """
    t = tl.program_id(0)
    start, lq = _seq_of(cu_q, t, B, BB)
    o = tl.arange(0, BLOCK)
    vals = tl.load(q + t.to(tl.int64) * ROW + o, mask=o < ROW)
    for j in tl.static_range(S):
        dst = (S * start + j * lq + (t - start)).to(tl.int64)
        tl.store(q2 + dst * ROW + o, vals, mask=o < ROW)
    if t == 0:
        ob = tl.arange(0, BB)[:, None]
        oj = tl.arange(0, S)[None, :]
        mb = ob < B
        cq = tl.load(cu_q + ob, mask=mb, other=0)
        lqb = tl.load(cu_q + ob + 1, mask=mb, other=0) - cq
        ck = tl.load(cu_k + ob, mask=mb, other=0)
        lkb = tl.load(cu_k + ob + 1, mask=mb, other=0) - ck
        idx = ob * S + oj
        tl.store(cu_q2 + idx, S * cq + oj * lqb, mask=mb & (oj < S))
        tl.store(cu_k2 + idx, ck + (oj * lkb) // S, mask=mb & (oj < S))
        tl.store(cu_q2 + B * S, S * tl.load(cu_q + B))
        tl.store(cu_k2 + B * S, tl.load(cu_k + B))


@triton.jit
def _split_merge_kernel(
    o2,
    l2,
    cu_q,
    out,
    lse,
    B,
    s_o2_t,
    s_o2_h,
    s_out_t,
    s_out_h,
    S: tl.constexpr,
    BB: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    D: tl.constexpr,
):
    """Combines query row ``t``'s ``S`` partial outputs by their LSE.  A row whose parts are all empty (-inf LSE)
    gets a zero output and -inf LSE, as the kernel gives an empty sequence."""
    t = tl.program_id(0)
    start, lq = _seq_of(cu_q, t, B, BB)
    oh = tl.arange(0, BH)
    od = tl.arange(0, D)
    mh = oh < H
    m = tl.full([BH], float("-inf"), tl.float32)
    for j in tl.static_range(S):
        r = (S * start + j * lq + (t - start)).to(tl.int64)
        m = tl.maximum(m, tl.load(l2 + r * H + oh, mask=mh, other=float("-inf")))
    ms = tl.where(m == float("-inf"), 0.0, m)
    acc = tl.zeros([BH, D], tl.float32)
    den = tl.zeros([BH], tl.float32)
    for j in tl.static_range(S):
        r = (S * start + j * lq + (t - start)).to(tl.int64)
        w = tl.exp(tl.load(l2 + r * H + oh, mask=mh, other=float("-inf")) - ms)
        oj = tl.load(
            o2 + r * s_o2_t + oh[:, None] * s_o2_h + od[None, :],
            mask=mh[:, None],
            other=0.0,
        )
        acc += w[:, None] * oj.to(tl.float32)
        den += w
    o = tl.where(den[:, None] > 0, acc / den[:, None], 0.0)
    tl.store(
        out + t.to(tl.int64) * s_out_t + oh[:, None] * s_out_h + od[None, :],
        o.to(out.dtype.element_ty),
        mask=mh[:, None],
    )
    tl.store(
        lse + t.to(tl.int64) * H + oh,
        tl.where(den > 0, ms + tl.log(den), float("-inf")),
        mask=mh,
    )


try:  # binds a launch's arguments once per shape: an eager prefill call would otherwise wait on the host
    from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch

    _split_prep_kernel = fast_launch(_split_prep_kernel)
    _split_merge_kernel = fast_launch(_split_merge_kernel)
except ImportError:  # pragma: no cover
    pass


def can_split(q: torch.Tensor) -> bool:
    """The query copy moves rows as int32 words of a contiguous q."""
    return q.is_contiguous() and (q.shape[1] * q.shape[2] * q.element_size()) % 4 == 0


def fmha_fwd_fp8_varlen_asm_split(
    raw_kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    parts: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``raw_kernel`` is ``aiter.ops.mha._fmha_fwd_fp8_varlen_asm``.  Same arguments and result as the non-causal
    unsplit call, which always returns the LSE."""
    total_q, H, Dq = q.shape
    Dv = v.size(2)
    S = parts
    B = cu_seqlens_q.numel() - 1
    BB = triton.next_power_of_2(max(B, 1))
    row = H * Dq * q.element_size() // 4
    cu_q = cu_seqlens_q.to(torch.int32)
    cu_k = cu_seqlens_k.to(torch.int32)
    # One workspace for the repeated queries, the split cu_seqlens and the partial results.
    n_q2 = S * total_q * H * Dq * q.element_size()
    n_o2 = S * total_q * H * Dv * 2
    n_l2 = S * total_q * H * 4
    n_cu = (B * S + 1) * 4
    off_o2 = -(-n_q2 // 256) * 256
    off_l2 = off_o2 + -(-n_o2 // 256) * 256
    off_cu = off_l2 + -(-n_l2 // 256) * 256
    ws = torch.empty(off_cu + 2 * n_cu, dtype=torch.uint8, device=q.device)
    q2 = ws[:n_q2].view(q.dtype).view(S * total_q, H, Dq)
    o2 = ws[off_o2 : off_o2 + n_o2].view(torch.bfloat16).view(S * total_q, H, Dv)
    l2 = ws[off_l2 : off_l2 + n_l2].view(torch.float32).view(S * total_q, H, 1)
    cu_q2 = ws[off_cu : off_cu + n_cu].view(torch.int32)
    cu_k2 = ws[off_cu + n_cu : off_cu + 2 * n_cu].view(torch.int32)
    _split_prep_kernel[(total_q,)](
        q=q.view(torch.int32),
        q2=q2.view(torch.int32),
        cu_q=cu_q,
        cu_k=cu_k,
        cu_q2=cu_q2,
        cu_k2=cu_k2,
        B=B,
        S=S,
        BB=BB,
        ROW=row,
        BLOCK=triton.next_power_of_2(row),
        num_warps=4,
    )
    raw_kernel(
        q2,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        o2,
        l2,
        cu_q2,
        cu_k2,
        int(max_seqlen_q),
        int(-(-int(max_seqlen_k) // S)),
        float(softmax_scale),
        False,
        True,
    )
    if out is None:
        out = torch.empty((total_q, H, Dv), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_q, H, 1), dtype=torch.float32, device=q.device)
    _split_merge_kernel[(total_q,)](
        o2=o2,
        l2=l2,
        cu_q=cu_q,
        out=out,
        lse=lse,
        B=B,
        s_o2_t=o2.stride(0),
        s_o2_h=o2.stride(1),
        s_out_t=out.stride(0),
        s_out_h=out.stride(1),
        S=S,
        BB=BB,
        H=H,
        BH=triton.next_power_of_2(H),
        D=Dv,
        num_warps=4,
    )
    return out, lse
