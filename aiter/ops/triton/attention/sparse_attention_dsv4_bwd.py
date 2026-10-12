# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

Counterpart to the DSv4 sparse prefill forward. The op is the official V4 form: shared-KV GQA
where ``K == V == kv`` is a single dense 512-wide tensor, RoPE already applied in place
caller-side, scale ``1/sqrt(512)``, ``attn_sink`` folded into the softmax denominator only, and
``topk_indices == -1`` masked out.

    P     = exp(Q@kv^T * scale - lse)
    dP    = dO@kv^T
    delta = rowsum(O * dO)
    dS    = P * (dP - delta) * scale
    dQ    = dS @ kv
    dKV   = scatter_add over top-k of  sum_h ( dS*Q + P*dO )

Pipeline (five kernels + one torch reduction), per rank chunk:

    delta            triton    rowsum(O*dO)
    dQ               gluon     also emits this chunk's dS / P
    dKV-interm       gluon     interm[t, slot, d] = sum_h (dS*Q + P*dO)
    CSR build        torch     inverted top-k index (sort + searchsorted)
    dKV gather       triton    reduce interm over the top-k mapping, atomic-free
    d_sink           torch     26 us, not worth a kernel

``lse`` and ``o`` come from the forward. The merged Gluon prefill kernel produces both::

    from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    o, lse = mla_gluon(..., has_pe=False, attn_sink=sink, return_lse=True)

Its ``lse`` is sink-inclusive (the sink is folded into ``e_max``/``e_sum`` before
``lse = e_max + log(e_sum)``), which is the convention this backward expects.

Measured on gfx950 (MI355X) at ``T=4096 H=128 topk=512`` with a realistic SWA(128)+pool top-k:
delta 0.178 / dQ 1.391 / interm 1.152 / CSR build 0.130 / gather 0.503 / d_sink 0.026 ms,
3.380 ms total = 407 TFLOPS.
"""

from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_attention_dsv4_bwd import (
    _dkv_interm_v4_kernel,
    _dq_v4_kernel,
)
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4_bwd import (
    _bwd_dkv_gather_acc_v4,
    _delta_v4_kernel,
)
from aiter.ops.triton.utils._triton import arch_info

_BLOCK_H_DQ = 64
_TILE_K_DQ = 32
_BD_DKV = 256
_TILE_K_DKV = 128


def sparse_mla_bwd_dq(
    q,
    kv,
    do,
    topk,
    lse,
    delta,
    dq,
    chunk_dS,
    chunk_P,
    scale,
    r_start,
    R_CHUNK,
    BLOCK_H=64,
    TILE_K=32,
    is_first_chunk=True,
):
    """Launch the dQ kernel for one rank chunk. Writes ``dq`` (RMW when not the first chunk)
    plus this chunk's ``chunk_dS`` / ``chunk_P``."""
    T, H, D = q.shape
    _dq_v4_kernel[(T, triton.cdiv(H, BLOCK_H))](
        q,
        kv,
        do,
        topk,
        lse,
        delta,
        dq,
        chunk_dS,
        chunk_P,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        do.stride(0),
        do.stride(1),
        dq.stride(0),
        dq.stride(1),
        topk.stride(0),
        chunk_dS.stride(0),
        chunk_dS.stride(1),
        scale,
        H,
        r_start,
        R_CHUNK=R_CHUNK,
        BLOCK_H=BLOCK_H,
        TILE_K=TILE_K,
        D=D,
        IS_FIRST_CHUNK=is_first_chunk,
        num_warps=4,
        waves_per_eu=1,
    )


def sparse_mla_bwd_dkv_interm_v4(
    q,
    do,
    chunk_dS,
    chunk_P,
    R_CHUNK,
    BD=256,
    TILE_K=128,
    MFMA_K=32,
    DUAL_STAGE=1,
    H_POW2=None,
    num_warps=4,
    interm=None,
):
    """V4 dKV-interm, Q/dO read once. Returns interm [T, R_CHUNK, D] bf16.

    ``BD`` splits D across ``grid.y``; dS/P are re-read once per D block, so a larger BD moves
    less of them. ``MFMA_K=32`` is the CDNA4 16x16x32 depth.
    """
    T, H, D = q.shape
    assert R_CHUNK % TILE_K == 0
    assert D % BD == 0
    h_pow2 = H_POW2 or triton.next_power_of_2(H)
    if interm is None:
        interm = torch.empty(T, R_CHUNK, D, dtype=torch.bfloat16, device=q.device)
    _dkv_interm_v4_kernel[(T, D // BD)](
        q,
        do,
        chunk_dS,
        chunk_P,
        interm,
        q.stride(0),
        q.stride(1),
        do.stride(0),
        do.stride(1),
        chunk_dS.stride(0),
        chunk_dS.stride(1),
        interm.stride(0),
        interm.stride(1),
        H,
        R_CHUNK=R_CHUNK,
        TILE_K=TILE_K,
        NH=h_pow2,
        BD=BD,
        D=D,
        MFMA_K=MFMA_K,
        DUAL_STAGE=DUAL_STAGE,
        num_warps=num_warps,
    )
    return interm


def delta_v4(o, do, out=None, BLOCK_R=8, num_warps=8):
    # BLOCK_R=8 keeps each lane loading >= 8 bf16, i.e. a dwordx4; narrower blocks drop to a
    # dword and the kernel loses most of its bandwidth.
    """o[T,H,D] bf16, do[T,H,D] bf16 -> delta[T,H] fp32 = sum_d o*do.

    ``do`` must already be the D-wide (lora) slice, contiguous — same contract as the dQ kernel.
    """
    assert o.shape == do.shape and o.is_contiguous() and do.is_contiguous()
    T, H, D = o.shape
    n_rows = T * H
    if out is None:
        out = torch.empty(T, H, dtype=torch.float32, device=o.device)
    _delta_v4_kernel[(triton.cdiv(n_rows, BLOCK_R),)](
        o,
        do,
        out,
        n_rows,
        D=D,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )
    return out


def build_inverted_topk(topk_indices_slice, num_kv):
    """CSR inverted index over ``num_kv`` KV rows.

    One stable sort yields both the permutation (``inv_data``) and the sorted keys;
    ``inv_ptr[k] = searchsorted(sorted, k, 'left')`` = the number of entries with value < k.
    Invalid (-1) entries sort to the front, so ``inv_ptr[0]`` starts past them and they are
    never visited.

    The sort key is narrowed to int16 when ``num_kv`` fits, which is what keeps the radix sort
    to two byte-passes.

    Returns ``inv_ptr[num_kv+1]`` int32, ``inv_data[T*R]`` int32.
    """
    # row_ids is the searchsorted query: [0 .. num_kv], one per KV row plus the end sentinel.
    # Its dtype must match `keys` -- searchsorted is built per branch for that reason, not by
    # accident.
    flat_kv = topk_indices_slice.reshape(-1)  # [T*R] int32; -1 = invalid
    # row_ids reaches num_kv, so int16 holds it as long as num_kv is within the type;
    # the -1 sentinel is fine either way.
    if num_kv <= torch.iinfo(torch.int16).max:
        keys = flat_kv.to(torch.int16)
        row_ids = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int16)
    else:
        keys = flat_kv.to(torch.int32)
        row_ids = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int32)
    sorted_vals, inv_data = torch.sort(keys, stable=True)
    inv_ptr = torch.searchsorted(sorted_vals, row_ids).to(torch.int32)
    return inv_ptr, inv_data.to(torch.int32)


def dkv_gather_acc(
    interm, inv_ptr, inv_data, dkv_acc, BLOCK_E=64, num_warps=8, accumulate=True
):
    """interm[T,R,D] bf16 -> dkv_acc[num_kv,D] fp32 via the entry-blocked CSR gather.

    Grid is ``num_kv`` (from ``dkv_acc``), not ``T``, so a compressed-pool KV works.
    """
    _, _, D = interm.shape
    num_kv = dkv_acc.shape[0]
    _bwd_dkv_gather_acc_v4[(num_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv_acc,
        interm.stride(1),
        dkv_acc.stride(0),
        D=D,
        BLOCK_E=BLOCK_E,
        ACCUMULATE=accumulate,
        num_warps=num_warps,
    )


@dataclass
class _BwdPlan:
    """Validated inputs, tile choices and workspace for one backward call.

    Holds the values that flow between phases (``delta``, the CSR pair, the accumulator) so
    ``_bwd_phases`` can hand out independent thunks that still compose into the real pipeline.
    """

    q: torch.Tensor
    kv: torch.Tensor
    do: torch.Tensor
    o: torch.Tensor
    lse: torch.Tensor
    topk_indices: torch.Tensor
    attn_sink: object
    scale: float
    T: int
    H: int
    D: int
    TOPK: int
    num_kv: int
    R_CHUNK: int
    tk_dq: int
    tk_dkv: int
    delta: torch.Tensor
    dq: torch.Tensor
    dkv_acc: torch.Tensor
    chunk_dS: torch.Tensor
    chunk_P: torch.Tensor
    interm: torch.Tensor
    inv_ptr: object = None
    inv_data: object = None
    d_sink: object = None

    def build_csr(self, r):
        self.inv_ptr, self.inv_data = build_inverted_topk(
            self.topk_indices[:, r : r + self.R_CHUNK], self.num_kv
        )

    def compute_d_sink(self):
        # d_sink[h] = -sum_t exp(sink[h] - lse[t,h]) * delta[t,h]
        self.d_sink = -(
            torch.exp(self.attn_sink[None, :].float() - self.lse) * self.delta
        ).sum(dim=0)

    def result(self):
        return self.dq, self.dkv_acc.to(self.kv.dtype), self.d_sink


def _plan_bwd(
    q, kv, do, o, lse, topk_indices, attn_sink=None, scale=None, R_CHUNK=None
):
    """Validate the inputs, pick the tile widths and allocate the workspace for one call.

    Split out from ``sparse_mla_bwd_dsv4`` so the op benchmark can build the same plan and then
    time ``_bwd_phases`` against it. Argument meanings are documented on the public entry.
    """
    assert (
        arch_info.get_arch() == "gfx950"
    ), f"sparse_mla_bwd_dsv4 requires gfx950 (CDNA4), got {arch_info.get_arch()}"

    for name, t in (
        ("q", q),
        ("kv", kv),
        ("do", do),
        ("o", o),
        ("lse", lse),
        ("topk_indices", topk_indices),
    ):
        if not t.is_cuda:
            raise RuntimeError(
                f"sparse_mla_bwd_dsv4 requires CUDA/HIP tensors, {name} is on {t.device}"
            )

    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"sparse_mla_bwd_dsv4 expects bf16 q, got {q.dtype}")
    for name, t in (("kv", kv), ("do", do), ("o", o)):
        if t.dtype != q.dtype:
            raise RuntimeError(f"{name} dtype mismatch: {name}={t.dtype}, q={q.dtype}")

    T, H, D = q.shape
    assert topk_indices.ndim == 2 and topk_indices.shape[0] == T, (
        f"topk_indices must be [T, TOPK] with T={T}, got {tuple(topk_indices.shape)} -- the dQ "
        "grid is sized from q, so a shorter index tensor is read past its end"
    )
    TOPK = topk_indices.shape[1]
    num_kv = kv.shape[0]
    assert D == 512, f"DSv4 sparse-MLA backward is fixed to head_dim 512, got {D}"
    assert kv.shape[-1] == D, f"kv must be [num_kv, {D}], got {tuple(kv.shape)}"
    assert (
        do.shape == q.shape
    ), f"do must match q {tuple(q.shape)}, got {tuple(do.shape)}"
    assert o.shape == q.shape, f"o must match q {tuple(q.shape)}, got {tuple(o.shape)}"
    assert lse.shape == (T, H), f"lse must be [{T}, {H}], got {tuple(lse.shape)}"
    assert num_kv >= T, f"num_kv ({num_kv}) must be >= T ({T})"
    if attn_sink is not None:
        # Optional, so it misses the device sweep above; d_sink does GPU math with it.
        if not attn_sink.is_cuda:
            raise RuntimeError(
                f"sparse_mla_bwd_dsv4 requires CUDA/HIP tensors, attn_sink is on "
                f"{attn_sink.device}"
            )
        if attn_sink.dtype != torch.float32:
            raise RuntimeError(
                f"sparse_mla_bwd_dsv4 expects fp32 attn_sink, got {attn_sink.dtype}"
            )
        assert attn_sink.shape == (
            H,
        ), f"attn_sink must be [{H}], got {tuple(attn_sink.shape)}"
    assert (
        topk_indices.dtype == torch.int32
    ), f"topk_indices must be int32, got {topk_indices.dtype}"
    for name, t in (
        ("q", q),
        ("kv", kv),
        ("do", do),
        ("o", o),
        ("topk_indices", topk_indices),
    ):
        assert t.is_contiguous(), f"{name} must be contiguous"

    if scale is None:
        scale = 1.0 / (D**0.5)
    if R_CHUNK is None:
        R_CHUNK = TOPK
    lse = lse.float().contiguous()

    # Both mfma tiles must divide the chunk width; step down rather than making it the
    # caller's problem, so a small R_CHUNK still works.
    tk_dkv = next(
        (t for t in (_TILE_K_DKV, 64, 32) if t <= R_CHUNK and R_CHUNK % t == 0), None
    )
    tk_dq = next(
        (t for t in (_TILE_K_DQ, 32) if t <= R_CHUNK and R_CHUNK % t == 0), None
    )
    assert (
        tk_dkv is not None and tk_dq is not None
    ), f"R_CHUNK={R_CHUNK} must be a multiple of 32 (it is the mfma tile width)"

    # A tail chunk narrower than R_CHUNK is wrong in three places at once. The kernels take
    # R_CHUNK as a constexpr and mask the top-k load against it rather than against TOPK, so
    # they read past the end of each row -- into the next token's indices for tokens 0..T-2,
    # and past the tensor entirely for the last one (measured: that faults the GPU). The CSR
    # build meanwhile gets a torch-clamped, narrower slice, and the gather then indexes interm,
    # which is still R_CHUNK wide, with entries encoded against that narrower width. Require
    # the divisor rather than paying a second compile for a narrower tail.
    assert TOPK % R_CHUNK == 0, (
        f"R_CHUNK={R_CHUNK} must divide TOPK={TOPK} -- a partial tail chunk reads past the end "
        "of each top-k row and desynchronizes the CSR index space from interm"
    )

    return _BwdPlan(
        q=q,
        kv=kv,
        do=do,
        lse=lse,
        topk_indices=topk_indices,
        attn_sink=attn_sink,
        scale=scale,
        T=T,
        H=H,
        D=D,
        TOPK=TOPK,
        num_kv=num_kv,
        R_CHUNK=R_CHUNK,
        tk_dq=tk_dq,
        tk_dkv=tk_dkv,
        delta=torch.empty(T, H, dtype=torch.float32, device=q.device),
        dq=torch.empty_like(q),
        dkv_acc=torch.zeros(num_kv, D, dtype=torch.float32, device=q.device),
        chunk_dS=torch.empty(T, H, R_CHUNK, dtype=torch.bfloat16, device=q.device),
        chunk_P=torch.empty(T, H, R_CHUNK, dtype=torch.bfloat16, device=q.device),
        interm=torch.empty(T, R_CHUNK, D, dtype=torch.bfloat16, device=q.device),
        o=o,
    )


def _bwd_phases(plan):
    """``(name, thunk)`` for every phase of one backward call, in execution order.

    ``sparse_mla_bwd_dsv4`` runs these; the op benchmark times them one at a time. Sharing the
    sequence is the point -- a per-phase measurement is only meaningful if it is measuring the
    phases the op actually runs, with the same tile configuration and the same workspace.
    """
    yield "delta", lambda: delta_v4(plan.o, plan.do, out=plan.delta)

    for r in range(0, plan.TOPK, plan.R_CHUNK):
        yield (
            "dq",
            lambda r=r: sparse_mla_bwd_dq(
                plan.q,
                plan.kv,
                plan.do,
                plan.topk_indices,
                plan.lse,
                plan.delta,
                plan.dq,
                plan.chunk_dS,
                plan.chunk_P,
                plan.scale,
                r,
                plan.R_CHUNK,
                BLOCK_H=_BLOCK_H_DQ,
                TILE_K=plan.tk_dq,
                is_first_chunk=(r == 0),
            ),
        )
        yield (
            "interm",
            lambda: sparse_mla_bwd_dkv_interm_v4(
                plan.q,
                plan.do,
                plan.chunk_dS,
                plan.chunk_P,
                plan.R_CHUNK,
                BD=_BD_DKV,
                TILE_K=plan.tk_dkv,
                interm=plan.interm,
            ),
        )
        yield "csr_build", lambda r=r: plan.build_csr(r)
        # dkv_acc was just zeroed, so the first chunk can write instead of read-modify-write.
        # Unchunked -- the default -- that is the only chunk, and it saves reading back the
        # whole [num_kv, 512] fp32 accumulator.
        yield (
            "gather",
            lambda r=r: dkv_gather_acc(
                plan.interm,
                plan.inv_ptr,
                plan.inv_data,
                plan.dkv_acc,
                accumulate=(r > 0),
            ),
        )

    if plan.attn_sink is not None:
        yield "d_sink", plan.compute_d_sink


def sparse_mla_bwd_dsv4(
    q,
    kv,
    do,
    o,
    lse,
    topk_indices,
    attn_sink=None,
    scale=None,
    R_CHUNK=None,
):
    """Backward for the DSv4 sparse-MLA prefill attention. gfx950 (CDNA4) only.

    Args:
        q:            [T, H, 512]      bf16
        kv:           [num_kv, 512]    bf16, K == V. ``num_kv >= T``; rows ``T..num_kv-1`` are
                                       the compressed pool, which ``topk_indices`` may reference.
        do:           [T, H, 512]      bf16, gradient of the attention output
        o:            [T, H, 512]      bf16, the forward output
        lse:          [T, H]           fp32, sink-inclusive log-sum-exp from the forward
        topk_indices: [T, TOPK]        int32, -1 marks an invalid slot
        attn_sink:    [H]              fp32 per-head sink bias, or None
        scale:        softmax scale, defaults to ``1/sqrt(512)``
        R_CHUNK:      split the rank dimension into chunks of this width. ``None`` (default)
                      runs unchunked, which is what you want. Chunking exists only to bound the
                      ``interm`` intermediate, which is ``T*TOPK*512`` bf16 (2.0 GiB at
                      T=4096, TOPK=512); it costs a dQ read-modify-write between chunks and one
                      CSR build per chunk. Must be a multiple of 32 (the mfma tile width) and
                      must divide ``TOPK`` -- a partial tail chunk is rejected rather than
                      handled, since the chunk width is a kernel constexpr.

    Returns:
        dq [T, H, 512] bf16, dkv [num_kv, 512] bf16, d_sink [H] fp32 (None if no ``attn_sink``)
    """
    plan = _plan_bwd(q, kv, do, o, lse, topk_indices, attn_sink, scale, R_CHUNK)
    for _, run in _bwd_phases(plan):
        run()
    return plan.result()


__all__ = ["sparse_mla_bwd_dsv4"]
