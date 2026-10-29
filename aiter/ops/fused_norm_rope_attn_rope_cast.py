# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused optional Q RMSNorm, RoPE, sparse attention, inverse RoPE and MXFP8 on gfx950.

Legacy cache records contain 512 FP8 data bytes followed by 16 E8M0 scale bytes.
"""

import math

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

# The ctypes ABI follows these signatures positionally. Keep argument order in
# sync with csrc/kernels/mla/fused_norm_rope_attn_rope_cast/fused_norm_rope_attn_rope_cast.cu.
MD_NAME = "module_fused_norm_rope_attn_rope_cast"

__all__ = [
    "fused_norm_rope_attn_rope_cast",
    "fused_norm_rope_attn_rope_cast_bf16",
    "fused_norm_rope_attn_rope_cast_legacy",
]


@compile_ops(MD_NAME, ffi_type="ctypes")
def _fused_norm_rope_attn_rope_cast_legacy_fwd(
    q: torch.Tensor,
    pool: torch.Tensor,
    prefix: torch.Tensor,
    prefix_indptr: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    extend_kv: torch.Tensor | None,
    extend: torch.Tensor | None,
    extend_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    debug_out: torch.Tensor | None,
    softmax_scale: float,
    q_is_rotated: bool,
    fp8_qk: bool = False,
    q_scale: torch.Tensor | None = None,
    q_norm: bool = False,
    norm_eps: float = 1e-6,
) -> None:
    """Write caller-allocated outputs; see :func:`fused_norm_rope_attn_rope_cast_legacy`."""


def fused_norm_rope_attn_rope_cast_legacy(
    q: torch.Tensor,
    pool: torch.Tensor,
    prefix: torch.Tensor,
    prefix_indptr: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    extend_kv: torch.Tensor | None = None,
    extend: torch.Tensor | None = None,
    extend_indptr: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    q_is_rotated: bool = True,
    out: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    fp8_qk: bool = False,
    q_scale: torch.Tensor | None = None,
    q_norm: bool = False,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(out, out_scale, max_logits, lse)`` in one kernel launch.

    ``q`` is contiguous BF16 ``[T,H,512]``, H in {8,16,32,64,128}, in
    ordinary head-major layout. Set ``q_is_rotated=False`` to apply forward
    RoPE to its final 64 channels. KV rows must already have RoPE applied.

    Set ``q_norm=True, q_is_rotated=False`` to apply unweighted RMSNorm
    over each Q head's 512 channels before RoPE. The reduction uses FP32:
    ``q / sqrt(mean(q**2) + norm_eps)``, rounded to BF16 before RoPE.
    ``norm_eps`` defaults to 1e-6 and must be positive, finite and normal
    in FP32. FP32 reduction order can change BF16 rounding at boundaries;
    norm need not be bitwise identical to a separate RMSNorm implementation.
    Norm is disabled by default and cannot be used with already rotated
    or prequantized Q. KV is not normalized by this entry.

    Set ``fp8_qk=True`` for scaled FP8 QK with fused group32 Q quantization
    after optional RoPE. This opt-in mode supports H64/H128 with FP8 KV
    and no BF16 extend. PV remains BF16 and accumulation remains FP32.
    Q and KV use one E8M0 scale per group of 32 values.
    Alternatively supply FP8 E4M3FN ``q[T,H,512]`` and contiguous uint8
    ``q_scale[T,H,16]`` with ``fp8_qk=True, q_is_rotated=True`` to consume
    already rotated and quantized Q. Q then requires 32-byte alignment;
    scales use finite E8M0 codes. This mode skips all Q preparation.

    BF16 mode: ``pool`` is contiguous BF16 ``[S,512]`` and ``prefix``
    contains int32 row indices. Values
    are read directly from the supplied BF16 cache, with no intermediate
    allocation or dequantization. See :func:`fused_norm_rope_attn_rope_cast_bf16`.

    Tagged FP8 mode: ``pool`` is contiguous uint8 ``[bytes]`` or ``[bytes,1]``. Each packed
    row has 512 E4M3FN bytes followed by 16 finite E8M0 group32 scales.
    ``prefix`` is int64 packed addresses ``(byte_offset << 1) | 1`` with
    int32 CSR ``prefix_indptr[T+1]``. Addresses are relative to pool.data_ptr;
    the pool base and live row offsets must be 4-byte aligned. Invalid
    addresses/tags are masked. Only live CSR ranges are read.

    Optional ``extend_kv[S,512]`` is BF16 with int32 ``extend`` row indices
    and ``extend_indptr[T+1]`` CSR. Invalid row indices are masked. All three
    tensors must be supplied together. CSR pointers must be monotonic and
    in [0, capacity], with at most 1,073,741,567 indices per source; positions (int32/int64 [T]) must be in [0,P).
    ``cos``/``sin`` are BF16 [P,32], using adjacent-pair RoPE.

    Outputs are E4M3FN ``[T,H/8,4096]``, E8M0 ``[T,H/8,128]``, and FP32
    max/LSE ``[T,H]``. Attention is rounded to BF16 before inverse RoPE and
    MXFP8 quantization. ``attn_sink[H]`` (FP32) enters only the softmax
    denominator; max/LSE exclude it. Empty rows return zero output, -inf
    max, and +inf LSE.

    Q and selected KV values must be finite; sink may also be -inf. All
    tensors must be contiguous on q.device. Q, extend_kv and out
    need 16-byte alignment. Output buffers must not overlap inputs or one
    another. Warm up before graph capture and reuse preallocated outputs.
    This entry does not allocate a rearrangement or intermediate KV tensor.
    """
    return _fused_norm_rope_attn_rope_cast_legacy_with_debug(
        q=q,
        pool=pool,
        prefix=prefix,
        prefix_indptr=prefix_indptr,
        cos=cos,
        sin=sin,
        positions=positions,
        extend_kv=extend_kv,
        extend=extend,
        extend_indptr=extend_indptr,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        q_is_rotated=q_is_rotated,
        out=out,
        out_scale=out_scale,
        max_logits=max_logits,
        lse=lse,
        fp8_qk=fp8_qk,
        q_scale=q_scale,
        q_norm=q_norm,
        norm_eps=norm_eps,
    )


def _fused_norm_rope_attn_rope_cast_legacy_with_debug(
    q: torch.Tensor,
    pool: torch.Tensor,
    prefix: torch.Tensor,
    prefix_indptr: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    extend_kv: torch.Tensor | None = None,
    extend: torch.Tensor | None = None,
    extend_indptr: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    q_is_rotated: bool = True,
    out: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    debug_out: torch.Tensor | None = None,
    fp8_qk: bool = False,
    q_scale: torch.Tensor | None = None,
    q_norm: bool = False,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal test entry with an optional pre-quantization FP32 output."""
    if not isinstance(q, torch.Tensor) or q.device.type != "cuda" or q.ndim != 3:
        raise ValueError("q must be a GPU tensor [T,H,512]")
    t, h, d = q.shape
    if h not in (8, 16, 32, 64, 128) or d != 512:
        raise ValueError("Supported H=8/16/32/64/128 and D=512")
    if get_gfx_runtime() != "gfx950":
        raise ValueError("fused_norm_rope_attn_rope_cast_legacy requires gfx950")
    norm_eps = float(norm_eps)
    if (
        not math.isfinite(norm_eps)
        or not torch.finfo(torch.float32).tiny
        <= norm_eps
        <= torch.finfo(torch.float32).max
    ):
        raise ValueError("norm_eps must be positive, finite and normal in float32")
    if q_norm and (q_is_rotated or q_scale is not None):
        raise ValueError("q_norm requires raw BF16 Q and q_is_rotated=False")
    scale = 512**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or not 0 < scale <= torch.finfo(torch.float32).max:
        raise ValueError("softmax_scale must be positive and finite in float32")

    def buffer(x, dtype, shape):
        if x is None:
            return torch.empty(shape, dtype=dtype, device=q.device)
        if x.dtype != dtype:
            raise ValueError(f"output must have dtype {dtype}")
        return x

    out = buffer(out, torch.float8_e4m3fn, (t, h // 8, 4096))
    out_scale = buffer(out_scale, torch.float8_e8m0fnu, (t, h // 8, 128))
    max_logits = buffer(max_logits, torch.float32, (t, h))
    lse = buffer(lse, torch.float32, (t, h))
    _fused_norm_rope_attn_rope_cast_legacy_fwd(
        q,
        pool,
        prefix,
        prefix_indptr,
        cos,
        sin,
        positions,
        out,
        out_scale,
        max_logits,
        lse,
        extend_kv,
        extend,
        extend_indptr,
        attn_sink,
        debug_out,
        scale,
        q_is_rotated,
        fp8_qk,
        q_scale,
        q_norm,
        norm_eps,
    )
    return out, out_scale, max_logits, lse


def fused_norm_rope_attn_rope_cast_bf16(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused attention reading a contiguous BF16 KV workspace directly.

    ``kv`` is BF16 ``[S,512]`` or ``[S,1,512]``, already rotated. ``indices``
    is a flat int32 row-index buffer with int32 CSR ``indptr[T+1]``. Negative
    and out-of-range indices are masked. The singleton-head view is zero-copy.
    This follows the BF16 workspace input strategy used by mega prefill;
    Q, RoPE, outputs, optional buffers and numerical semantics are those of
    :func:`fused_norm_rope_attn_rope_cast_legacy`, not the full FlashMLA tensor ABI. In particular,
    Q remains head-major and RoPE tables are BF16. No gather/dequant or
    index conversion is performed in this function.
    """
    kv = _bf16_kv_rows(kv)
    return fused_norm_rope_attn_rope_cast_legacy(
        q, kv, indices, indptr, cos, sin, positions, **kwargs
    )


@compile_ops(MD_NAME, ffi_type="ctypes")
def _fused_norm_rope_attn_rope_cast_fwd(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
    kv_indices_extend: torch.Tensor | None,
    kv_indptr_extend: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    max_logits: torch.Tensor | None,
    lse: torch.Tensor | None,
    debug_out: torch.Tensor | None,
    q_is_rotated: bool,
    q_norm: bool,
    norm_eps: float,
    num_splits: int = 1,
    workspace: torch.Tensor | None = None,
) -> None:
    """Write outputs for the two-source FP8 interface, optionally with split-KV."""


def fused_norm_rope_attn_rope_cast(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    kv_indices_extend: torch.Tensor | None = None,
    kv_indptr_extend: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float = 512**-0.5,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    backend: str | None = None,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    out_scale: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    q_is_rotated: bool = True,
    q_norm: bool = False,
    norm_eps: float = 1e-6,
    is_decode: bool = False,
    num_splits: int = 0,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fuse Q preparation, two-source FP8 attention and output RoPE/quantization.

    The Q/KV argument names and CSR semantics follow ``mla_sparse_prefill_v41``.
    Both KV sources contain already rotated FP8 E4M3FN ``[S,512]`` values and
    E8M0 ``[S,16]`` group32 scales. Scales may have E8M0 or uint8 dtype. Each
    source has its own int32 indices and int32 ``[T+1]`` CSR pointers; indices
    address rows of that source. Invalid indices are masked. CSR pointers must
    be monotonic and within their index-buffer capacity. Padding outside the
    live CSR ranges is not read. Each index buffer is limited to 1,073,741,567
    elements. Both sources contribute to one softmax.

    Values and scales need unit channel stride and a positive, 4-byte-aligned
    row stride covering their width. Row strides may differ between all four
    views. In particular, ``cache[:, :512]`` and ``cache[:, 512:]`` from a
    contiguous ``[S,528]`` byte cache are consumed directly, without copies,
    index concatenation or a dequantized KV workspace. Empty sources are valid;
    omit all four extend arguments together to use the prefix alone.

    ``q`` is contiguous ``[T,H,512]``, H16/H32/H64/H128, in head-major layout. Prepared
    FP8 Q requires contiguous E8M0/uint8 ``q_scale[T,H,16]`` and
    ``q_is_rotated=True``. Alternatively pass BF16 Q and ``q_scale=None``;
    group32 Q quantization is fused. Set ``q_is_rotated=False`` for forward
    RoPE, and optionally ``q_norm=True`` for unweighted per-head RMSNorm.
    Norm uses FP32 reduction with ``norm_eps``, rounds to BF16 before RoPE,
    and requires raw BF16 Q. It never normalizes KV.

    ``cos``/``sin`` are contiguous BF16 ``[P,32]`` tables for adjacent-pair
    RoPE of the final 64 channels. ``positions`` is contiguous int32/int64
    ``[T]``, with values in [0,P). ``attn_sink[H]`` is FP32 or None; it enters
    only the softmax denominator. Selected values/scales must be finite, and
    sink may also be -inf. ``backend`` accepts None or "opus" on gfx950.

    Returns ``(out, out_scale, max_logits, lse)``. Unlike the attention-only
    entry, out is E4M3FN ``[T,H/8,4096]`` after BF16 attention rounding,
    inverse RoPE and group32 quantization; scales are row-layout E8M0
    ``[T,H/8,128]``. Provide ``out``/``out_scale`` to reuse buffers. Optional
    ``max_logits``/``lse`` are caller-allocated FP32 ``[T,H]``; omitted buffers
    remain None and are not allocated or written. Their values exclude sink;
    empty rows produce zero out, -inf max and +inf LSE.

    ``num_splits=0`` automatically chooses a split count for both prefill and
    decode from host-visible batch and index capacities, without reading
    device CSR pointers. Set ``num_splits=1`` to force the single-kernel path.
    ``is_decode=True`` enables explicit counts 2..32 and caller-supplied
    workspace for tuning. Counts above one launch FP8 partial attention
    followed by FP32 reduction and the fused
    output transform. Q preparation remains fused into each partial kernel.
    Split attention changes FP32 summation order. Sink is applied once during
    reduction; max/LSE still exclude it.

    Optional ``workspace`` is disjoint, contiguous FP32 ``[T,num_splits,H,516]``
    scratch with opaque contents. It holds partial accumulators, not KV.
    When omitted it is allocated for split attention. For allocation-free
    calls, provide it with ``is_decode=True`` and an explicit split count.
    No workspace is used with one split.

    All tensors must be on q.device. Output buffers must be contiguous and
    disjoint from inputs and each other. Warm up before graph capture.
    The previous pool/tagged-address interface is available explicitly as
    :func:`fused_norm_rope_attn_rope_cast_legacy`.
    """
    return _fused_norm_rope_attn_rope_cast_with_debug(
        q=q,
        q_scale=q_scale,
        unified_kv=unified_kv,
        unified_kv_scale=unified_kv_scale,
        kv_indices_prefix=kv_indices_prefix,
        kv_indptr_prefix=kv_indptr_prefix,
        kv=kv,
        kv_scale=kv_scale,
        kv_indices_extend=kv_indices_extend,
        kv_indptr_extend=kv_indptr_extend,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        out=out,
        lse=lse,
        backend=backend,
        cos=cos,
        sin=sin,
        positions=positions,
        out_scale=out_scale,
        max_logits=max_logits,
        q_is_rotated=q_is_rotated,
        q_norm=q_norm,
        norm_eps=norm_eps,
        is_decode=is_decode,
        num_splits=num_splits,
        workspace=workspace,
    )


def _fused_norm_rope_attn_rope_cast_with_debug(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    kv_indices_extend: torch.Tensor | None = None,
    kv_indptr_extend: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float = 512**-0.5,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    backend: str | None = None,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    out_scale: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    debug_out: torch.Tensor | None = None,
    q_is_rotated: bool = True,
    q_norm: bool = False,
    norm_eps: float = 1e-6,
    is_decode: bool = False,
    num_splits: int = 0,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Internal test entry with an optional pre-quantization FP32 output."""
    if backend not in (None, "opus"):
        raise ValueError("backend must be None or 'opus'")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.device.type != "cuda":
        raise ValueError("q must be a GPU tensor [T,H,512]")
    t, h, d = q.shape
    if h not in (16, 32, 64, 128) or d != 512:
        raise ValueError("two-source FP8 attention requires H16/H32/H64/H128 and D512")
    if type(is_decode) is not bool:
        raise ValueError("is_decode must be bool")
    if type(num_splits) is not int or not 0 <= num_splits <= 32:
        raise ValueError("num_splits must be an integer in [0,32]")
    if not is_decode and (num_splits not in (0, 1) or workspace is not None):
        raise ValueError("split workspace and num_splits > 1 require is_decode=True")
    if num_splits == 0:
        num_splits = 1
        if t:
            capacity = kv_indices_prefix.numel()
            if kv_indices_extend is not None:
                capacity += kv_indices_extend.numel()
            grid = t * ((h + 63) // 64)
            min_rows_per_split = 128 if h == 16 else 256
            work_splits = max(1, capacity // (t * min_rows_per_split))
            if grid <= 32:
                work_splits = max(work_splits, min(4, capacity // (t * 128)))
            if grid <= 4:
                work_splits = max(work_splits, capacity // (t * 128))
            limit = min(32, max(1, 256 // grid), work_splits)
            if h == 16:
                tile_splits = (kv_indices_prefix.numel() + t * 64 - 1) // (t * 64)
                if kv_indices_extend is not None:
                    tile_splits += (kv_indices_extend.numel() + t * 64 - 1) // (t * 64)
                if is_decode and 1 < tile_splits <= min(16, 256 // grid):
                    limit = tile_splits
                # Long lists amortize a second resident CTA per CU. Avoid extra
                # partials when fewer splits give the same longest tile loop.
                if grid >= 64 and capacity >= t * 2048:
                    limit = min(32, max(1, 512 // grid), work_splits)
                    tiles_per_split = (tile_splits + limit - 1) // limit
                    limit = (tile_splits + tiles_per_split - 1) // tiles_per_split
            # Avoid partial-output traffic on H64 grids that already have
            # enough independent requests for these short lists.
            if h == 64 and t >= 128 and capacity < t * 1024:
                limit = 1
            # The 32-head single path fills a 256-CTA grid for short lists
            # without partial-output traffic or a separate reduction launch.
            if (
                t * (h // 64) == 128
                and capacity < t * 768
                and kv_indices_prefix.numel() > 0
                and kv_indices_extend is not None
                and kv_indices_extend.numel() > 0
            ):
                limit = 1
            if h == 32 and grid <= 32:
                limit = min(limit, max(1, 128 // grid))
            if h in (16, 32):
                num_splits = limit
            else:
                while num_splits * 2 <= limit:
                    num_splits *= 2
    if num_splits > 1 and workspace is None:
        workspace = torch.empty(
            (t, num_splits, h, 516), device=q.device, dtype=torch.float32
        )
    if num_splits == 1 and workspace is not None:
        raise ValueError("workspace requires num_splits > 1")
    if out is None:
        out = torch.empty((t, h // 8, 4096), device=q.device, dtype=torch.float8_e4m3fn)
    if out_scale is None:
        out_scale = torch.empty(
            (t, h // 8, 128), device=q.device, dtype=torch.float8_e8m0fnu
        )
    _fused_norm_rope_attn_rope_cast_fwd(
        q,
        q_scale,
        unified_kv,
        unified_kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_scale,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        float(softmax_scale),
        cos,
        sin,
        positions,
        out,
        out_scale,
        max_logits,
        lse,
        debug_out,
        q_is_rotated,
        q_norm,
        float(norm_eps),
        num_splits,
        workspace,
    )
    return out, out_scale, max_logits, lse


def _bf16_kv_rows(kv):
    if kv.dtype != torch.bfloat16 or not kv.is_contiguous():
        raise ValueError("kv must be contiguous BF16 [S,512] or [S,1,512]")
    if kv.ndim == 3 and kv.shape[1:] == (1, 512):
        kv = kv.view(kv.shape[0], 512)
    if kv.ndim != 2 or kv.shape[1] != 512:
        raise ValueError("kv must be BF16 [S,512] or [S,1,512]")
    return kv
