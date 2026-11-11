# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-base a packed V4 NoPE cache onto a per-D-block scale.

The served cache carries one E8M0 per 32 contiguous D elements *per row*
(MXFP8 1x32). That works for QK^T, whose reduction axis is D, but not for P*V,
whose reduction axis is KV: there the scale varies along the thing being summed,
so it cannot be hoisted or expressed through the MFMA's scale operand.

A 32(KV) x 32(D) tile scale does not fix it either, because prefill *gathers*
its KV rows by top-k index -- two rows in one tile generally come from different
chunks of the cache. The scale has to be independent of the row entirely.

So this pass re-bases every row onto a single set of ``448/32 = 14`` E8M0
exponents, shared across all rows, chosen from the global per-D-block max. The
scale then depends only on D, which for P*V is the *output* axis, so it factors
out of the reduction and is applied once to the accumulator. QK^T is unaffected
beyond reading uniform values.

Storage is unchanged: the 14 bytes still live at offset 448 of each row, now
holding the same values in every row, so the consuming kernel's scale load does
not move.

:func:`to_dma_kv_layout` does the unit-scale re-base and the RoPE fold in one
out-of-place pass; it is what ``pa_sparse_prefill_fp8_opus`` runs. All of this is
a placeholder for the KV cache writer emitting the layout directly.
"""

import torch
import triton
import triton.language as tl

_D_NOPE = 448
_D_PADDED = 512
_BLK = 32
_NBLK = _D_NOPE // _BLK  # 14
_FP8_MAX = 448.0


@triton.jit
def _blockwise_amax_kernel(
    data_ptr,
    byte_ptr,
    amax_ptr,
    n_rows,
    stride_row,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
):
    """Per-D-block max of |dequantized value| over every row."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_ok = rows < n_rows

    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        off = rows[:, None] * stride_row + cols[None, :]
        q = tl.load(data_ptr + off, mask=row_ok[:, None], other=0.0).to(tl.float32)
        # The E8M0 bytes share storage with the fp8 payload, so they have to be
        # read through a uint8 view -- an fp8 load would decode them as floats.
        e = tl.load(byte_ptr + rows * stride_row + D_NOPE + b, mask=row_ok, other=0)
        scale = tl.exp2(e.to(tl.float32) - 127.0)
        tl.atomic_max(amax_ptr + b, tl.max(tl.abs(q * scale[:, None])))


@triton.jit
def _rebase_kernel(
    data_ptr,
    byte_ptr,
    exp_ptr,
    n_rows,
    stride_row,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
):
    """Requantize each row onto the shared per-D-block exponents."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_ok = rows < n_rows

    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        off = rows[:, None] * stride_row + cols[None, :]
        q = tl.load(data_ptr + off, mask=row_ok[:, None], other=0.0).to(tl.float32)
        e_old = tl.load(byte_ptr + rows * stride_row + D_NOPE + b, mask=row_ok, other=0)
        e_new = tl.load(exp_ptr + b)
        # Both scales are powers of two, so re-basing is an exponent difference.
        ratio = tl.exp2(e_old.to(tl.float32) - e_new.to(tl.float32))
        tl.store(
            data_ptr + off, (q * ratio[:, None]).to(tl.float8e4nv), mask=row_ok[:, None]
        )
        # The trailer is not rewritten here: warps of this program read a row's
        # old exponent on their own schedule, so writing it in the same pass
        # races. The caller writes the new exponents afterwards.


def rebase_kv_nope_to_dblock_scale(
    *caches: torch.Tensor, unit_scale: bool = True
) -> torch.Tensor:
    """Rewrite each cache in place so its E8M0 scales depend only on the D block.

    All caches passed together are re-based onto one shared exponent set. That
    is required, not cosmetic: prefill accumulates the prefix and extend regions
    into the same output, so a single scale has to cover both.

    Args:
      caches: one or more ``[rows, 512]`` fp8 tensors (448 NoPE + 14 E8M0 at
        offset 448 + pad).
      unit_scale: when True (default) every block is re-based to 2^0, so the
        stored values *are* the real ones and the consumer needs no epilogue
        scale at all. Set False to keep the max-targeting exponent, which buys
        underflow headroom but then requires the reader to apply 2^(e-127) per
        D block to its accumulator.

    Returns:
      ``[14]`` uint8 tensor of the shared exponents, also written into every
      row's trailer so existing readers need no change.
    """
    assert caches, "need at least one cache"
    for c in caches:
        assert c.dtype == torch.float8_e4m3fn, c.dtype
        assert c.shape[-1] == _D_PADDED and c.is_contiguous()
    dev = caches[0].device

    if unit_scale:
        exps = torch.full((_NBLK,), 127, dtype=torch.uint8, device=dev)
    else:
        amax = torch.zeros(_NBLK, dtype=torch.float32, device=dev)
        for c in caches:
            _blockwise_amax_kernel[(triton.cdiv(c.shape[0], 64),)](
                c,
                c.view(torch.uint8),
                amax,
                c.shape[0],
                _D_PADDED,
                BLOCK_ROWS=64,
                NBLK=_NBLK,
                BLK=_BLK,
                D_NOPE=_D_NOPE,
            )
        # Same exponent rule the reference quantizer uses: land the block max in
        # (224, 448] so it fills e4m3's range without reaching NaN.
        e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / _FP8_MAX))
        e = torch.where(amax == 0, torch.zeros_like(e), e)
        exps = (e + 127).clamp(0, 255).to(torch.uint8)

    for c in caches:
        _rebase_kernel[(triton.cdiv(c.shape[0], 64),)](
            c,
            c.view(torch.uint8),
            exps,
            c.shape[0],
            _D_PADDED,
            BLOCK_ROWS=64,
            NBLK=_NBLK,
            BLK=_BLK,
            D_NOPE=_D_NOPE,
        )
        c.view(torch.uint8)[:, _D_NOPE : _D_NOPE + _NBLK] = exps
    return exps


@triton.jit
def _fold_rope_kernel(
    data_ptr,
    rope_ptr,
    n_rows,
    stride_row,
    stride_rope,
    BLOCK_ROWS: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """Quantize the bf16 RoPE tail to fp8 and park it in slots [448, 512)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_ok = rows < n_rows
    cols = tl.arange(0, D_ROPE)
    r = tl.load(
        rope_ptr + rows[:, None] * stride_rope + cols[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        data_ptr + rows[:, None] * stride_row + D_NOPE + cols[None, :],
        r.to(tl.float8e4nv),
        mask=row_ok[:, None],
    )


def fold_rope_into_nope(kv_nope: torch.Tensor, kv_rope: torch.Tensor) -> None:
    """Pack the fp8 RoPE tail into the NoPE row's trailing slots, in place.

    After :func:`rebase_kv_nope_to_dblock_scale` with ``unit_scale=True`` the
    E8M0 bytes at [448, 462) are all 127 and carry no information, so the tail
    is free. Folding RoPE there makes the row a complete 512-wide fp8 V vector,
    which is what lets a single ``global_load_lds`` stage the whole of V --
    rows are then 512 B, so one 1024 B DMA chunk covers exactly two of them.

    The consumer must therefore treat the KV scale as a constant 2^0 rather
    than reading the (now overwritten) trailer, and consume V's RoPE as fp8
    (the OPUS kernel takes QK^T's RoPE in bf16 from the RoPE source).
    """
    assert kv_nope.shape[0] == kv_rope.shape[0]
    assert kv_nope.shape[1] == _D_PADDED and kv_rope.shape[1] == 64
    rows = kv_nope.shape[0]
    _fold_rope_kernel[(triton.cdiv(rows, 64),)](
        kv_nope,
        kv_rope,
        rows,
        _D_PADDED,
        kv_rope.stride(0),
        BLOCK_ROWS=64,
        D_NOPE=_D_NOPE,
        D_ROPE=64,
    )


@triton.jit
def _to_dma_layout_kernel(
    src_ptr,
    src_byte_ptr,
    rope_ptr,
    dst_ptr,
    n_rows,
    stride_src,
    stride_rope,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_PADDED: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """Re-base each row to unit scale and fold its fp8 RoPE, into a new tensor."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_ok = rows < n_rows
    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        q = tl.load(
            src_ptr + rows[:, None] * stride_src + cols[None, :],
            mask=row_ok[:, None],
            other=0.0,
        ).to(tl.float32)
        e = tl.load(
            src_byte_ptr + rows * stride_src + D_NOPE + b, mask=row_ok, other=127
        )
        v = q * tl.exp2(e.to(tl.float32) - 127.0)[:, None]
        tl.store(
            dst_ptr + rows[:, None] * D_PADDED + cols[None, :],
            tl.clamp(v, -448.0, 448.0).to(tl.float8e4nv),
            mask=row_ok[:, None],
        )
    rc = tl.arange(0, D_ROPE)
    r = tl.load(
        rope_ptr + rows[:, None] * stride_rope + rc[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        dst_ptr + rows[:, None] * D_PADDED + D_NOPE + rc[None, :],
        tl.clamp(r, -448.0, 448.0).to(tl.float8e4nv),
        mask=row_ok[:, None],
    )


def to_dma_kv_layout(kv_nope: torch.Tensor, kv_rope: torch.Tensor) -> torch.Tensor:
    """Out-of-place, single-pass equivalent of :func:`rebase_kv_nope_to_dblock_scale`
    (unit scale) followed by :func:`fold_rope_into_nope`.

    Args:
      kv_nope: ``[rows, 512]`` fp8 cache as served: 448 NoPE values whose E8M0
        scales (one per 32 D elements, or a 64-element scale written twice)
        sit at bytes [448, 462).
      kv_rope: ``[rows, 64]`` bf16 RoPE.

    Returns:
      A new ``[rows, 512]`` fp8 tensor in the layout the DMA kernel reads: the
      NoPE values at scale 2^0 in [0, 448) and fp8 RoPE in [448, 512). Values
      beyond e4m3's +-448 saturate. The inputs are not modified.
    """
    assert kv_nope.dtype == torch.float8_e4m3fn, kv_nope.dtype
    assert (
        kv_nope.dim() == 2 and kv_nope.shape[1] == _D_PADDED and kv_nope.stride(1) == 1
    )
    assert kv_rope.dtype == torch.bfloat16, kv_rope.dtype
    assert kv_rope.shape == (kv_nope.shape[0], 64) and kv_rope.stride(1) == 1
    rows = kv_nope.shape[0]
    out = torch.empty((rows, _D_PADDED), dtype=kv_nope.dtype, device=kv_nope.device)
    if rows:
        _to_dma_layout_kernel[(triton.cdiv(rows, 64),)](
            kv_nope,
            kv_nope.view(torch.uint8),
            kv_rope,
            out,
            rows,
            kv_nope.stride(0),
            kv_rope.stride(0),
            BLOCK_ROWS=64,
            NBLK=_NBLK,
            BLK=_BLK,
            D_NOPE=_D_NOPE,
            D_PADDED=_D_PADDED,
            D_ROPE=64,
        )
    return out
