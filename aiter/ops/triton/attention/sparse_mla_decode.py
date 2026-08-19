# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse MLA decode (gfx950 gluon): separated-rope geometry — GLM-5 / DSA-style
``kv_lora_rank`` latent ⊕ appended decoupled rope, token-granular top-k gather.

This is the separated-rope sibling of ``pa_decode_sparse``: both drive the same
gluon kernel (``_gluon_kernels/gfx950/attention/pa_decode_sparse.py``), but the
two model families have different enough calling structures that they get
different APIs — DSv4 (rope *inside* the 512-wide row, K == V, optional SWA +
top-k two-loop, attention sink) stays on ``pa_decode_sparse``; MLA-lineage
models (rope *appended*: QK over ``kv_lora_rank + qk_rope_head_dim``, V = the
latent only, single top-k segment, no sink) use this module.

The calling convention follows the asm path this replaces
(``aiter.mla.mla_decode_fwd`` as invoked by vLLM's ``ROCMAiterMLASparseImpl``),
with two deliberate deviations, both favorable: no head padding (the asm kernel
repeat_interleaves H<16 to 16 and computes the duplicate rows; here H=8 runs
natively at BLOCK_M=8), and q stays bf16 (no q-quant kernel; <1% accuracy
effect).

Cache formats, inferred from the buffer:

    [nb, block, R] / [slots, 1, 1, R] / [slots, R] bf16      full-row bf16
    same shapes in fp8/uint8  (+ scalar ``kv_scale``)        flat per-tensor fp8
        — vLLM's production layout: ``concat_and_cache_mla(...,"fp8",
        scale=layer._k_scale)`` quantizes the whole row (rope included) with
        one scale. The kernel never dequantizes in the tile loop: the K-side
        scale folds into qk_scale and the V-side scale into p (exact; see the
        kernel docstring).
    [blocks, block_size, 656] uint8                          vLLM ``fp8_ds_mla``
        — 512 fp8 | 4×f32 per-128 scales | 64 bf16 rope.

Indices are GLOBAL slot ids (``triton_convert_req_index_to_global_index``
output). NOTE: that converter maps -1 / out-of-range entries to slot 0 despite
its docstring, so the production stream carries no -1 sentinels and the default
``has_invalid=False`` matches it (and skips the sentinel clamp+mask). Pass
``has_invalid=True`` for index streams that really carry -1 entries — they are
then masked out, which is *more* correct than the converter's slot-0
substitution.
"""

import math
import os as _os

import torch

from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse as _pa_decode_sparse_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse_reduce as _pa_decode_sparse_reduce_gfx950,
)
from aiter.ops.triton.attention.pa_decode_sparse import (
    _as_int32_contiguous_1d,
    _decode_num_splits_occ,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

MAX_BYTES = 2**31 - 1


def _check_out(out, q, kv_lora_rank):
    """Caller-supplied output buffer, or a fresh one. The attention output is
    [C, H, kv_lora_rank]; a wider buffer (e.g. a q-shaped scratch) is accepted
    and written in its leading kv_lora_rank columns."""
    C, H = q.shape[0], q.shape[1]
    if out is None:
        return torch.empty(
            (C, H, kv_lora_rank), dtype=torch.bfloat16, device=q.device
        )
    assert out.shape[0] == C and out.shape[1] == H, (
        f"out shape {tuple(out.shape)} != [{C}, {H}, >= {kv_lora_rank}]"
    )
    assert out.shape[2] >= kv_lora_rank and out.stride(2) == 1
    assert out.dtype == torch.bfloat16 and out.device == q.device
    return out


def _infer_cache_format(kv, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale):
    """-> (fmt, cache, alt_ptr_tensor, scl_ptr_tensor, block_size, fp8_fnuz).

    Pointer roles follow the kernel's Seg contract: alt = the bf16 view
    ("dsmla" rope tail) or the cache itself ("bf16"); scl = the f32 scalar
    k_scale ("tensor") or the f32 view of the cache ("dsmla")."""
    if kv.ndim == 4:  # [slots, 1, 1, R] -- the asm mla_decode_fwd view
        assert kv.shape[1] == 1 and kv.shape[2] == 1
        kv = kv.reshape(kv.shape[0], kv.shape[3])
    elif kv.ndim == 3 and kv.shape[2] == d_qk:
        # the vLLM paged cache [nb, block_size, R]: indices are global slot
        # ids and rows are contiguous, so the flat view is stride-identical.
        assert kv.stride(2) == 1 and kv.stride(1) == kv.shape[2] * kv.stride(2)
        kv = kv.reshape(-1, d_qk)

    fp8_dtypes = (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if kv.ndim == 2 and kv.shape[1] == d_qk:
        if kv.dtype == torch.bfloat16:
            return "bf16", kv, kv, None, 1, False
        assert kv.dtype in fp8_dtypes, f"unsupported flat cache dtype {kv.dtype}"
        assert kv_scale is not None, (
            "flat fp8 cache needs the per-tensor kv_scale (layer._k_scale)"
        )
        assert kv_scale.dtype == torch.float32
        u8 = kv.view(torch.uint8)
        fnuz = kv.dtype == torch.float8_e4m3fnuz
        return "tensor", u8, u8, kv_scale.reshape(1), 1, fnuz
    if (
        kv.ndim == 3
        and kv.element_size() == 1
        and kv.shape[2] == kv_lora_rank + 4 * (kv_lora_rank // 128) + 2 * qk_rope_head_dim
    ):
        # vLLM fp8_ds_mla: 512 fp8 | 4 f32 per-128 scales | 64 bf16 rope = 656 B
        u8 = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
        assert u8.stride(2) == 1 and u8.stride(1) == u8.shape[2], (
            "fp8_ds_mla rows must be contiguous 656-byte records"
        )
        return (
            "dsmla",
            u8,
            u8.view(torch.bfloat16),
            u8.view(torch.float32),
            u8.shape[1],
            False,
        )
    raise ValueError(f"unrecognized sparse-MLA kv cache: {tuple(kv.shape)} {kv.dtype}")


def sparse_mla_decode_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse (top-k gathered) MLA decode, separated-rope geometry.

    Args:
        q: ``[C, H, kv_lora_rank + qk_rope_head_dim]`` bf16 decode queries.
            H = per-rank query heads; 8 (TP8) and 16 (TP4) both run natively.
        kv_buffer: the KV pool in one of the formats in the module docstring.
        kv_indptr: ``[C + 1]`` int32 prefix sum of per-query index counts
            (vLLM builds ``cumsum(min(seq_len, topk))``; uniform ``C * topk``
            stepping also works).
        kv_indices: flat int32 GLOBAL slot ids into the pool.
        softmax_scale: the layer's softmax scale.
        kv_scale: ``[1]`` f32 per-tensor cache scale — required for the flat
            fp8 format (vLLM's ``layer._k_scale``), unused otherwise.
        kv_splits: split-K override; default picks by the occupancy policy.
        skip_reduce: with split-K active, return ``(part_acc, part_m,
            part_l)`` instead of launching the combine.
        has_invalid: index stream carries -1 sentinels (masked out). Default
            False — the production converter emits none.
        out: optional ``[C, H, >= kv_lora_rank]`` bf16 destination (written in
            ``[..., :kv_lora_rank]``).

    Returns:
        ``[C, H, kv_lora_rank]`` bf16 attention output (the latent V; callers
        run ``v_up_proj`` after, exactly as with the asm path).
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert arch_info.get_arch() == "gfx950", "sparse_mla_decode_fwd is gfx950-only"
    assert q.dtype == torch.bfloat16, "q stays bf16 (no q-quant; see module doc)"
    num_queries, num_heads, d_qk = q.shape
    assert d_qk == kv_lora_rank + qk_rope_head_dim, (
        f"q last dim {d_qk} != {kv_lora_rank} + {qk_rope_head_dim}"
    )
    _LOGGER.info(
        f"SPARSE_MLA_DECODE C={num_queries} H={num_heads} d_qk={d_qk} "
        f"nnz={kv_indices.shape[0]}"
    )

    fmt, cache, alt, scl, block_size, fp8_fnuz = _infer_cache_format(
        kv_buffer, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale
    )
    kv_indices = _as_int32_contiguous_1d(kv_indices)
    kv_indptr = _as_int32_contiguous_1d(kv_indptr)
    assert kv_indptr.numel() == num_queries + 1
    # HAS_SINK=False: MLA-lineage models carry no attention sink. The kernel
    # still wants a live pointer for the elided-at-compile-time argument slot.
    attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    # Tuned launch config (gfx950 / MI355), shared with the DSv4 wrapper; same
    # env knobs so interleaved A/B sweeps drive both paths together.
    BLOCK_M, BLOCK_K, MFMA_K, waves_per_eu = 16, 64, 16, 0
    _bk = _os.environ.get("AITER_PA_DECODE_BLOCK_K")
    if _bk:
        BLOCK_K = int(_bk)
    _bm = _os.environ.get("AITER_PA_DECODE_BLOCK_M")
    if _bm:
        BLOCK_M = int(_bm)
    elif num_heads < 16:
        # H=8 (TP8): BLOCK_M=8 computes 8 real rows where a masked BLOCK_M=16
        # pads to 16 -- against the asm kernel's repeat_interleave to >=16
        # heads this is the free half of the measured win.
        BLOCK_M = max(8, 1 << (num_heads - 1).bit_length())
    _mk = _os.environ.get("AITER_PA_DECODE_MFMA_K")
    if _mk:
        MFMA_K = int(_mk)
    num_warps = max(1, BLOCK_K // 16)
    gather_tw1 = int(_os.environ.get("AITER_PA_DECODE_TW1", "32"))
    lds_pad = int(_os.environ.get("AITER_PA_DECODE_LDS_PAD", "8"))

    num_rows = cache.shape[0] * block_size if cache.ndim >= 2 else cache.shape[0]
    avg_topk = kv_indices.numel() / max(1, num_queries)

    cs0_align = 1
    for _a in (16, 8, 4, 2):
        if int(cache.stride(0)) % _a == 0:
            cs0_align = _a
            break
    _ca = _os.environ.get("AITER_PA_DECODE_CS0_ALIGN")
    if _ca:
        cs0_align = int(_ca)

    # buffer_load carries a 32-bit offset (2 GB cap); a production pool crosses
    # it at ~3.7M cached tokens, so the 64-bit gl.load path is first-class here
    # (measured: same speed as the buffer path, +2-3 VGPRs, no spills).
    use_buffer_load = max_addressable_bytes(cache) < MAX_BYTES
    idx_use_buffer_load = (
        _os.environ.get("AITER_PA_DECODE_IDX_BUF", "1") == "1"
        and max_addressable_bytes(kv_indices) < MAX_BYTES
    )
    _fg = _os.environ.get("AITER_PA_DECODE_FORCE_GLOBAL_LOAD", "")
    if _fg in ("1", "main"):
        use_buffer_load = False

    HEAD_ALIGNED = num_heads % BLOCK_M == 0
    heads_blocks = (num_heads + BLOCK_M - 1) // BLOCK_M
    out = _check_out(out, q, kv_lora_rank)

    _sp = _os.environ.get("AITER_PA_DECODE_SPLITS")

    def _pick_splits(hb):
        if _sp:
            return max(1, int(_sp))
        if kv_splits is not None:
            return max(1, int(kv_splits))
        return _decode_num_splits_occ(num_queries, hb, avg_topk, 0.0, BLOCK_K)

    num_splits = _pick_splits(heads_blocks)

    # Short top-k BLOCK_M halving gate (real parallelism at small launches).
    _tiles = max(1, math.ceil(avg_topk / BLOCK_K)) if avg_topk > 0 else 1
    if (
        _bm is None
        and BLOCK_M == 16
        and num_heads % 8 == 0
        and _tiles <= 2
        and num_queries * heads_blocks * num_splits <= get_num_sms()
    ):
        _hb8 = (num_heads + 7) // 8
        _ns8 = _pick_splits(_hb8)
        if _ns8 >= num_splits and num_queries * _hb8 * _ns8 <= get_num_sms():
            BLOCK_M = 8
            HEAD_ALIGNED = num_heads % BLOCK_M == 0
            heads_blocks = _hb8
            num_splits = _ns8

    if num_splits > 1:
        part_m = torch.empty(
            (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)
        _pab = (
            _os.environ.get("AITER_PA_DECODE_PART_BF16", "1") == "1" and not skip_reduce
        )
        part_acc = torch.empty(
            (num_queries, num_splits, num_heads, kv_lora_rank),
            dtype=torch.bfloat16 if _pab else torch.float32,
            device=q.device,
        )
        pm_stride0, pm_stride_s = part_m.stride(0), part_m.stride(1)
        pa_stride0, pa_stride_s, pa_stride_h = (
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
        )
    else:
        part_m = part_l = part_acc = out  # unused placeholders (never dereferenced)
        pm_stride0 = pm_stride_s = pa_stride0 = pa_stride_s = pa_stride_h = 0

    # Dequant chunking: split the axis that still has per-lane register repeats.
    col_reps = kv_lora_rank // (gather_tw1 * 16)
    chunk_axis = 1 if col_reps >= 4 else 0
    _nc = _os.environ.get("AITER_PA_DECODE_NOPE_CHUNK")
    if chunk_axis == 0:
        nope_chunk = int(_nc) if _nc else max(1, BLOCK_K // 4)
    else:
        nope_chunk = int(_nc) if _nc else min(128, kv_lora_rank)

    if chunk_axis == 0 or nope_chunk < kv_lora_rank:
        waves_per_eu = 2
    one_wg_per_cu = (
        use_buffer_load and num_queries * heads_blocks * num_splits <= get_num_sms()
    )
    if one_wg_per_cu:
        waves_per_eu = 1
    _wp = _os.environ.get("AITER_PA_DECODE_WPEU")
    if _wp:
        waves_per_eu = int(_wp)

    adaptive_splits = (
        num_splits > 1 and _os.environ.get("AITER_PA_DECODE_ADAPTIVE", "1") == "1"
    )

    _go = _os.environ.get("AITER_PA_DECODE_GRID_ORDER", "qsh")
    _ax = {"q": num_queries, "s": num_splits, "h": heads_blocks}
    assert sorted(_go) == ["h", "q", "s"], f"bad AITER_PA_DECODE_GRID_ORDER {_go!r}"
    # The separated-rope fp8 formats never trace the legacy peeled masked tail
    # (kernel static_assert); bf16 keeps the knob.
    uni_tile = (
        True
        if fmt in ("tensor", "dsmla")
        else _os.environ.get("AITER_PA_DECODE_UNI_TILE", "1") == "1"
    )

    grid = tuple(_ax[c] for c in _go)
    _pa_decode_sparse_gfx950[grid](
        q,
        cache,
        alt,
        kv_indices,
        kv_indptr,
        cache,       # extra_* segment: unread placeholders (HAS_EXTRA=False)
        alt,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        part_m,
        part_l,
        part_acc,
        scl,         # f32 side-channel: k_scale ("tensor") / f32 view ("dsmla")
        scl,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        cache.stride(0),
        cache.stride(0),
        num_rows,
        num_rows,
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        num_heads,
        HAS_EXTRA=False,
        HAS_SINK=False,
        MAIN_FMT=fmt,
        EXTRA_FMT=fmt,
        MAIN_BLOCK_SIZE=block_size,
        EXTRA_BLOCK_SIZE=block_size,
        CS0_ALIGN=cs0_align,
        NOPE_DIM=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        HEAD_SIZE=kv_lora_rank,
        ROPE_SEPARATE=True,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=HEAD_ALIGNED,
        MFMA_K=MFMA_K,
        GATHER_TW1=gather_tw1,
        LDS_PAD=lds_pad,
        NOPE_CHUNK=nope_chunk,
        CHUNK_AXIS=chunk_axis,
        PART_STORE_CACHE=_os.environ.get("AITER_PA_DECODE_PART_ST", ""),
        UNI_TILE=uni_tile,
        GRID_ORDER=_go,
        MAIN_SPLITS=num_splits,
        ADAPTIVE_SPLITS=adaptive_splits,
        ASM_DEQ=False,
        MAIN_USE_BUFFER_LOAD=use_buffer_load,
        EXTRA_USE_BUFFER_LOAD=use_buffer_load,
        IDX_BUFFER_LOAD=idx_use_buffer_load,
        HAS_INVALID=has_invalid,
        FP8_FNUZ=fp8_fnuz,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if num_splits == 1:
        return out
    if skip_reduce:
        return part_acc, part_m, part_l

    red_block_m = int(_os.environ.get("AITER_PA_DECODE_REDUCE_BLOCK_M", "1"))
    red_warps = int(_os.environ.get("AITER_PA_DECODE_REDUCE_WARPS", "1"))
    rgrid = (num_queries, (num_heads + red_block_m - 1) // red_block_m)
    _pa_decode_sparse_reduce_gfx950[rgrid](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        num_heads,
        HAS_SINK=False,
        HEAD_SIZE=kv_lora_rank,
        BLOCK_M=red_block_m,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=num_heads % red_block_m == 0,
        ADAPTIVE_SPLITS=adaptive_splits,
        PART_LOAD_CACHE=_os.environ.get("AITER_PA_DECODE_PART_LD", ""),
        num_warps=red_warps,
    )
    return out
