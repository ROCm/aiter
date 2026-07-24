# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 (CDNA3 / MI300X) Gluon fp8_mqa_logits scoring kernel.

CDNA3 sibling of the gfx950 kernel (``_gluon_kernels/gfx950/attention/
fp8_mqa_logits.py``). Two differences from the gfx950 source drive the port:

  * MFMA: gfx950's ``cdna4.mfma_scaled`` (microscaled, [32, 32, 64]) has no CDNA3
    equivalent, so this uses the native CDNA3 ``cdna3.mfma``
    (v_mfma_f32_32x32x16_fp8_fp8, version=3, [32, 32, 16]) on FNUZ e4m3 operands
    -- 2x the K-rate of the ``input_precision="ieee"`` f16-upcast generic kernel.
  * KV load: the gluon ``cdna3`` dialect exposes no async-copy / load-to-shared
    op (only ``cdna4.async_copy`` provides ``buffer_load_to_shared`` etc.), so
    rather than gfx950's shared-memory double-buffered pipeline each KV tile is
    ``cdna3.buffer_load``-ed straight into registers (Q + weights hoisted out of
    the loop, LICM). The gfx942 HW itself does have direct global->LDS
    (``global_load_lds``); it is the gluon lowering that is absent here, not the
    instruction.

``transposed=False`` keeps the per-row reduction over query heads mostly in
registers, so the plain ``gl.sum(axis=0)`` here is both correct and fast, and the
relu is a non-propagating ``gl.maximum`` (no NaN can reach it -- see relu_f32). On
MI300X at M=4096, N=32768, H=32 it runs at ~592 TFLOPs -- ~2.2x the generic Triton
fallback (266 TFLOPs) that gfx942 used before. ``transposed=True``
sends the head reduction cross-lane and is slower than even the generic kernel, so
``transposed=False`` is the right layout. gfx950's folded-FMA head reduction is
wired in (shared, arch-agnostic; see NUM_CHAINS) and enabled when the Triton build
has the constexpr-tuple ``permute`` (USE_FOLDED_REDUCTION probes for PR #9751);
otherwise NUM_CHAINS is 0 (the naive ``gl.sum``, bit-identical). The fold was
validated bit-exact vs the oracle across H {32,64} / D {64,128} with the actual
#9751 change (``_unwrap_if_constexpr``) applied to Triton 3.6.0.

The kernel is launched on a 2D grid ``(seq_len, n_chunks)`` for tile-level load
balancing: each work item scores up to CHUNK KV tiles of one query row, and
``logits[row, n]`` is independent across ``n`` (the reduction is over heads
*within* a tile), so a row's tile range splits freely across work items with no
cross-workgroup reduction. This keeps every workgroup at ~equal work, removing
the long-pole tail that a 1-row-per-workgroup grid leaves on triangular (causal
prefill) windows -- measured +30% at M=2048 N=16384 causal on MI300X.

Validated on MI300X (gfx942, Triton 3.6.0) against a torch.einsum oracle: exact
on in-window logits across partial-tile edge shapes, multi-tile (N up to 16384),
dense/causal/sliding windows, non-power-of-2 M, and H in {32, 64}.
"""
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Same reduction technique, arch agnostic (the MFMA layout is passed in), as gfx1250
# reuses it. NUM_CHAINS=0 falls back to the naive `s*w; gl.sum` path bit-for-bit.
from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
    _weighted_sum_fma_fold,
)


@gluon.jit
def relu_f32(x):
    # relu = max(x, 0); no NaN reaches here (fnuz e4m3 has no inf, masked tail uses
    # other=0.0), so non-propagating gl.maximum is correct. A single-instr inline-asm
    # v_max would drop the sNaN-canonicalize (~+4%) but gl.inline_asm_elementwise is
    # miscompiled here (bit-exact in isolation, but the LLVM AMDGPU regalloc corrupts
    # its operands under this kernel's register pressure), so keep gl.maximum.
    return gl.maximum(x, 0.0)


# --- CDNA3 register load / store helpers (honor the dispatch's buffer flags) ---
# cdna3 buffer ops raise on an explicit ``mask=None`` (they unconditionally try
# to broadcast offsets against the mask), so _masked_load branches on ``mask is
# None`` and omits the kwarg when unmasked. Use if/else (not early return): gluon
# keeps the fall-through in the mask-is-None specialization, so an early return
# would still compile the masked ``gl.load(..., other=...)`` and trip "other
# without mask". Mirrors the gfx950 helpers.
@gluon.jit
def _masked_load(ptr, offsets, USE_BUFFER_LOAD: gl.constexpr, mask=None):
    if mask is None:
        if USE_BUFFER_LOAD:
            out = gl.amd.cdna3.buffer_load(ptr, offsets)
        else:
            out = gl.load(ptr + offsets)
    else:
        if USE_BUFFER_LOAD:
            out = gl.amd.cdna3.buffer_load(ptr, offsets, mask=mask, other=0.0)
        else:
            out = gl.load(ptr + offsets, mask=mask, other=0.0)
    return out


@gluon.jit
def _load_kv_tile(
    KV_ptr, offs_d, offs_n, kv_pos,
    stride_kv_d: gl.constexpr, stride_kv_s,
    USE_BUFFER_LOAD: gl.constexpr, mask=None,
):
    offsets = offs_d[:, None] * stride_kv_d + (kv_pos + offs_n)[None, :] * stride_kv_s
    return _masked_load(KV_ptr, offsets, USE_BUFFER_LOAD, mask)


@gluon.jit
def _load_kv_scales(
    kv_scales_ptr, kv_pos, offs, USE_BUFFER_LOAD: gl.constexpr, mask=None,
):
    return _masked_load(kv_scales_ptr, kv_pos + offs, USE_BUFFER_LOAD, mask)


@gluon.jit
def _store_logits(
    logits_row, kv_pos, offs, scores, stride_logits_k,
    USE_BUFFER_STORE: gl.constexpr, mask=None,
):
    offsets = (kv_pos + offs) * stride_logits_k
    if mask is None:
        if USE_BUFFER_STORE:
            gl.amd.cdna3.buffer_store(scores, logits_row, offsets)
        else:
            gl.store(logits_row + offsets, scores)
    else:
        if USE_BUFFER_STORE:
            gl.amd.cdna3.buffer_store(scores, logits_row, offsets, mask=mask)
        else:
            gl.store(logits_row + offsets, scores, mask=mask)


@gluon.jit
def _score_tile(
    mfma_q, mfma_k, w_block, kv_scales,
    NUM_HEADS: gl.constexpr, BLOCK_KV: gl.constexpr, mfma_layout: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
):
    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    acc = gl.zeros([NUM_HEADS, BLOCK_KV], dtype=gl.float32, layout=mfma_layout)
    scores = gl.amd.cdna3.mfma(mfma_q, mfma_k, acc)
    scores = relu_f32(scores)
    # weighted head reduction -> [BLOCK_KV]. NUM_CHAINS=0 is the naive `s*w; gl.sum`
    # (in-register under transposed=False); NUM_CHAINS>0 folds it into parallel FMA
    # chains for shorter dependency depth (needs a Triton with constexpr-tuple permute).
    scores = _weighted_sum_fma_fold(
        scores, w_block, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS
    )
    scores = scores * kv_scales  # per-kv-token scale (kv_scales >= 0)
    return scores


@gluon.jit
def _gluon_fp8_mqa_logits_kernel(
    Q_ptr,  # fp8 (fnuz e4m3) [seq_len, NUM_HEADS, HEAD_SIZE]
    KV_ptr,  # fp8 (fnuz e4m3) [seq_len_kv, HEAD_SIZE]
    kv_scales_ptr,  # fp32   [seq_len_kv]
    weights_ptr,  # fp32   [seq_len, NUM_HEADS]
    cu_start_ptr,  # int32  [seq_len]
    cu_end_ptr,  # int32  [seq_len]
    logits_ptr,  # fp32   [seq_len, seq_len_kv]
    seq_len: gl.int32,
    seq_len_kv: gl.int32,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    stride_q_s: gl.int32,
    stride_q_h: gl.constexpr,
    stride_q_d: gl.constexpr,
    stride_kv_s: gl.int32,
    stride_kv_d: gl.constexpr,
    stride_w_s: gl.int32,
    stride_w_h: gl.constexpr,
    stride_logits_s: gl.int32,
    stride_logits_k: gl.int32,
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,  # unused: this kernel register-loads KV, no LDS pipeline
    NUM_CHAINS: gl.constexpr,  # folded-FMA head reduction; 0 = naive gl.sum
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    CHUNK: gl.constexpr,  # tiles per work item; 2D grid (seq_len, n_chunks)
):
    # Tile-balanced work item: program_id = (row, chunk). Each work item covers up
    # to CHUNK KV tiles of one row starting at chunk*CHUNK*BLOCK_KV. logits[row, n]
    # is independent across n (the reduction is over heads within a tile), so a
    # row's tile range can be split freely across work items. Out-of-window items
    # do nothing, so every active item does ~equal work -- this fixes the
    # triangular (causal prefill) load imbalance that a 1-row-per-workgroup grid
    # leaves as a long-pole tail.
    row_id = gl.program_id(axis=0)
    chunk_id = gl.program_id(axis=1)

    if not USE_BUFFER_LOAD:
        stride_kv_s = stride_kv_s.to(gl.int64)
    # The per-row logits base offset (row_id * stride_logits_s) can approach
    # INT32_MAX for large seq_len * seq_len_kv, so compute it in int64 to never
    # wrap the base pointer. The buffer-store *within-row* offset stays 32-bit
    # (it uses stride_logits_k, i.e. one element), so the 2 GiB buffer-op cap on
    # ``use_buffer_store`` still holds.
    stride_logits_s = stride_logits_s.to(gl.int64)

    # CDNA3 native fp8 MFMA: v_mfma_f32_32x32x16_fp8_fp8. transposed=False keeps
    # the head reduction mostly in-register (see the module docstring for perf).
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[32, 32, 16],
        transposed=False,
        warps_per_cta=[NUM_WARPS, 1],
    )
    K_WIDTH: gl.constexpr = 16
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=K_WIDTH
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=K_WIDTH
    )

    # Q [NUM_HEADS, HEAD_SIZE] contiguous along HEAD_SIZE
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    # K [HEAD_SIZE, BLOCK_KV] contiguous along HEAD_SIZE (dim 0). threads_per_warp
    # [8, 8] keeps the buffer_load coalesced (16 contiguous fp8 / thread, same load
    # count as [4, 16]) but sits much closer to dot_b, so the convert_layout to the
    # MFMA operand is far cheaper: v_cndmask 176 -> 48, ds_bpermute 84 -> 20 across
    # the kernel -- ~+3% causal / ~+4% dense on this VALU-issue-bound kernel.
    blocked_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    start_ind = gl.maximum(gl.load(cu_start_ptr + row_id), 0)
    end_ind = gl.minimum(gl.load(cu_end_ptr + row_id), seq_len_kv)
    p_start = start_ind + chunk_id * (CHUNK * BLOCK_KV)

    if p_start < end_ind:
        chunk_end = gl.minimum(end_ind, p_start + CHUNK * BLOCK_KV)

        # --- Load Q + weights once (LICM within this work item) ---
        offs_h_q = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, blocked_q))
        offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, blocked_q))
        q_offsets = (
            row_id * stride_q_s
            + offs_h_q[:, None] * stride_q_h
            + offs_d_q[None, :] * stride_q_d
        )
        if USE_BUFFER_LOAD:
            q = gl.amd.cdna3.buffer_load(Q_ptr, q_offsets, cache=".cg")
        else:
            q = gl.load(Q_ptr + q_offsets, cache_modifier=".cg")
        mfma_q = gl.convert_layout(q, dot_a_layout)

        offs_h_w = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
        w_offsets = row_id * stride_w_s + offs_h_w[:, None] * stride_w_h
        if USE_BUFFER_LOAD:
            w_block = gl.amd.cdna3.buffer_load(weights_ptr, w_offsets, cache=".cg")
        else:
            w_block = gl.load(weights_ptr + w_offsets, cache_modifier=".cg")

        kv_arange = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
        offs_d_kv = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, blocked_kv))
        offs_n_kv = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, blocked_kv))

        logits_row = logits_ptr + row_id * stride_logits_s
        num_full_tiles = (chunk_end - p_start) // BLOCK_KV

        kv_pos = p_start
        # --- full tiles (no mask) ---
        for _ in tl.range(0, num_full_tiles):
            k_tile = _load_kv_tile(
                KV_ptr, offs_d_kv, offs_n_kv, kv_pos, stride_kv_d, stride_kv_s,
                USE_BUFFER_LOAD,
            )
            mfma_k = gl.convert_layout(k_tile, dot_b_layout)
            kv_scales = _load_kv_scales(kv_scales_ptr, kv_pos, kv_arange, USE_BUFFER_LOAD)
            scores = _score_tile(
                mfma_q, mfma_k, w_block, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout,
                NUM_CHAINS,
            )
            _store_logits(
                logits_row, kv_pos, kv_arange, scores, stride_logits_k, USE_BUFFER_STORE
            )
            kv_pos += BLOCK_KV

        # --- masked tail (only when the chunk end isn't tile-aligned, i.e. the
        # last chunk of a row whose window isn't a BLOCK_KV multiple) ---
        if kv_pos < chunk_end:
            tail_mask_1d = (kv_pos + kv_arange) < chunk_end
            tail_mask_2d = (kv_pos + offs_n_kv)[None, :] < chunk_end
            k_tile = _load_kv_tile(
                KV_ptr, offs_d_kv, offs_n_kv, kv_pos, stride_kv_d, stride_kv_s,
                USE_BUFFER_LOAD, mask=tail_mask_2d,
            )
            mfma_k = gl.convert_layout(k_tile, dot_b_layout)
            kv_scales = _load_kv_scales(
                kv_scales_ptr, kv_pos, kv_arange, USE_BUFFER_LOAD, mask=tail_mask_1d
            )
            scores = _score_tile(
                mfma_q, mfma_k, w_block, kv_scales, NUM_HEADS, BLOCK_KV, mfma_layout,
                NUM_CHAINS,
            )
            _store_logits(
                logits_row, kv_pos, kv_arange, scores, stride_logits_k, USE_BUFFER_STORE,
                mask=tail_mask_1d,
            )
