# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention APIs.

``flydsl_flash_attn_batch_fwd`` / ``flydsl_flash_attn_varlen_fwd`` dispatch by
arch and dtype: gfx950 fp8 to ``kernels/fmha_gfx950``, gfx1250 bf16/f16 to the
m32x8 prefill kernel, anything else ``None`` so the caller falls through to
CK/Triton.

``flydsl_flash_attn_func`` serves gfx11 (RDNA3 / RDNA3.5) with
``flash_attn_func_gfx11`` -- any sequence lengths, self- and cross-attention,
GQA / MQA, ``head_dim % 8 == 0``, bf16/f16, no padding -- and gfx1201 (RDNA4)
by wrapping the `flash_attn_func_gfx1201` kernel with:
  - Build cache keyed by shape, dtype, masking, and tuning parameters.
  - Automatic Q and KV sequence padding to their respective tile sizes.
  - BSHD ([B, S, H, D]) input/output convention to match upstream
    flash-attention layout.
  - Exact masking of padded non-causal K/V columns.
  - Non-causal cross-attention with independent Q and KV sequence lengths.
"""

from __future__ import annotations

import math
import os
import warnings
from contextlib import nullcontext
from functools import lru_cache

import torch
import torch.nn.functional as F

from .fmha_bwd_gfx942 import flash_attn_varlen_bwd_d192_gfx942
from .kernels.flash_attn_func_gfx11 import build_flash_attn_func_module_gfx11
from .kernels.flash_attn_varlen_func_gfx11 import (
    build_flash_attn_varlen_func_module_gfx11,
)
from .kernels.fmha_gfx1201.flash_attn_func import (
    build_flash_attn_func_module,
    get_flash_attn_lds_bytes,
)
from .kernels.fmha_gfx1201.flash_attn_func_fp8 import (
    build_flash_attn_func_module as build_flash_attn_fp8_func_module,
)
from .kernels.fmha_gfx1201.flash_attn_func_fp8 import get_flash_attn_fp8_lds_bytes
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a8w8_m32x8 import (
    TORCH_O_DTYPE as _A8W8_GFX1250_OUT_DTYPE,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a8w8_m32x8 import (
    flash_attn_a8w8_m32x8_supported as flash_attn_a8w8_gfx1250_supported,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a8w8_m32x8 import (
    flash_attn_batch_a8w8_m32x8 as flash_attn_batch_a8w8_gfx1250,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a8w8_m32x8 import (
    flash_attn_paged_a8w8_m32x8 as flash_attn_paged_a8w8_gfx1250,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a8w8_m32x8 import (
    flash_attn_varlen_a8w8_m32x8 as flash_attn_varlen_a8w8_gfx1250,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    flash_attn_a16w16_m32x8_supported as flash_attn_a16w16_gfx1250_supported,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    flash_attn_batch_m32x8 as flash_attn_batch_a16w16_gfx1250,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    flash_attn_paged_m32x8 as flash_attn_paged_a16w16_gfx1250,
)
from .kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8 import (
    flash_attn_varlen_m32x8 as flash_attn_varlen_a16w16_gfx1250,
)

__all__ = [
    "flydsl_flash_attn_batch_fwd",
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_varlen_bwd",
    "flydsl_flash_attn_varlen_fwd",
]

_FP8_DTYPES = (torch.float8_e4m3fn,)
_GFX1201_LDS_CAPACITY_BYTES = 65536
_GFX1201_KERNEL_INT32_MAX = (1 << 31) - 1
_GFX1201_BUFFER_MAX_BYTES = 1 << 32


def _pick_gfx1201_tiles(seq_len: int, head_dim: int, causal: bool) -> tuple[int, int]:
    """Select the best known gfx1201 tile for the production shape envelope."""
    if causal:
        return 128, 32
    if head_dim <= 64:
        return 128, 64
    if seq_len < 2048:
        return 128, 32
    return 256, 64


def _gfx1201_fmha_lds_bytes(head_dim: int, block_n: int, *, fp8: bool) -> int:
    """Return the exact static LDS allocation for a selected gfx1201 kernel."""
    if fp8:
        return get_flash_attn_fp8_lds_bytes(head_dim, block_n)
    return get_flash_attn_lds_bytes(head_dim, block_n)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16, got {dtype!r}")


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the half-open byte range touched by a non-empty tensor view."""
    storage_ptr = tensor.untyped_storage().data_ptr()
    first = tensor.data_ptr() - storage_ptr
    last = first
    element_size = tensor.element_size()
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (size - 1) * stride * element_size
        first += min(0, delta)
        last += max(0, delta)
    return first, last + element_size


def _storage_overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.device != rhs.device:
        return False
    if lhs.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr():
        return False
    lhs_first, lhs_last = _storage_byte_range(lhs)
    rhs_first, rhs_last = _storage_byte_range(rhs)
    return lhs_first < rhs_last and rhs_first < lhs_last


def _has_unsupported_internal_overlap(tensor: torch.Tensor) -> bool:
    checker = getattr(torch, "_debug_has_internal_overlap", None)
    if checker is None:
        # Older PyTorch versions do not expose the overlap checker. Accept only
        # contiguous storage there rather than risk concurrent writes aliasing.
        return not tensor.is_contiguous()
    # 0 means no overlap. Treat both definite overlap and "too hard" as unsafe.
    return int(checker(tensor)) != 0


def _validate_gfx1201_launch_limits(
    *,
    batch: int,
    seq_len: int,
    seq_len_kv_real: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    fp8: bool,
) -> None:
    """Validate integer arguments and signed-Int32 V# buffer indices."""
    for name, value in (
        ("padded query sequence length", seq_len),
        ("real KV sequence length", seq_len_kv_real),
        ("padded KV sequence length", seq_len_kv),
    ):
        if value > _GFX1201_KERNEL_INT32_MAX:
            raise ValueError(
                f"{name}={value} exceeds the gfx1201 kernel Int32 limit "
                f"({_GFX1201_KERNEL_INT32_MAX})"
            )
    if not fp8:
        v_batch_bytes = seq_len_kv * num_heads * head_dim * 2
        if v_batch_bytes >= _GFX1201_BUFFER_MAX_BYTES:
            raise ValueError(
                f"one BF16/F16 V batch requires {v_batch_bytes} bytes, but the "
                "gfx1201 buffer descriptor requires a byte count below 2^32"
            )
    else:
        # The FP8 kernel's typed V# helpers narrow flattened Q/K/V/O element
        # offsets to Int32. Bound the padded tensors before materializing them
        # so a large batch cannot wrap those buffer indices.
        for name, elements in (
            ("padded Q/O", batch * seq_len * num_heads * head_dim),
            ("padded K/V", batch * seq_len_kv * num_heads * head_dim),
        ):
            if elements > _GFX1201_KERNEL_INT32_MAX:
                raise ValueError(
                    f"{name} requires {elements} elements, exceeding the gfx1201 "
                    f"FP8 buffer-index Int32 limit ({_GFX1201_KERNEL_INT32_MAX})"
                )


@lru_cache(maxsize=64)
def _get_kernel(
    device_index: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
    tail_mask: bool,
    cross_attn: bool,
    lds_vec_width: int,
):
    # device_index intentionally participates in the cache key; the builder
    # observes the active device selected by the caller's device context.
    # lds_vec_width participates because it changes the cooperative load layout.
    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        sm_scale=softmax_scale,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
        lds_vec_width=lds_vec_width,
    )


@lru_cache(maxsize=64)
def _get_fp8_gfx1201_kernel(
    device_index: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    softmax_scale: float | None,
    tail_mask: bool,
    cross_attn: bool,
):
    # device_index intentionally participates in the cache key.
    return build_flash_attn_fp8_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="bf16",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        sm_scale=softmax_scale,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
    )


@lru_cache(maxsize=64)
def _get_kernel_gfx11(
    device_index: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    softmax_scale: float | None,
    kv_tail: bool,
    num_waves: int,
    block_n: int,
    head_group: int | None,
    token_strides: tuple[int, int, int],
    return_lse: bool,
):
    # device_index intentionally participates in the cache key.
    return build_flash_attn_func_module_gfx11(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=softmax_scale,
        kv_tail=kv_tail,
        num_waves=num_waves,
        block_n=block_n,
        vt_rows=4 if block_n >= 64 else 2,
        head_group=head_group,
        q_token_stride=token_strides[0],
        k_token_stride=token_strides[1],
        v_token_stride=token_strides[2],
        num_kv_heads=num_kv_heads,
        return_lse=return_lse,
    )


@lru_cache(maxsize=64)
def _get_varlen_kernel_gfx11(
    device_index: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    softmax_scale: float | None,
    num_waves: int,
    block_n: int,
    vt_rows: int,
    head_group: int | None,
    token_strides: tuple[int, int, int],
    return_lse: bool,
    rope: str | None = None,
):
    # device_index intentionally participates in the cache key.
    return build_flash_attn_varlen_func_module_gfx11(
        rope=rope,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=softmax_scale,
        num_waves=num_waves,
        block_n=block_n,
        vt_rows=vt_rows,
        head_group=head_group,
        q_token_stride=token_strides[0],
        k_token_stride=token_strides[1],
        v_token_stride=token_strides[2],
        num_kv_heads=num_kv_heads,
        return_lse=return_lse,
    )


def _gfx11_direct(t: torch.Tensor) -> bool:
    """True if the gfx11 kernel reads ``t`` in place: head_dim and heads packed
    (strides 1 and head_dim), token / batch strides multiples of 8 elements and a
    16-byte aligned start. Covers plain BSHD and views of a packed tensor such as
    ``qkv[:, :, i]`` of ``[B, S, 3, H, D]``. Strides of size-1 dims are ignored
    (PyTorch leaves them arbitrary, e.g. a ``[B, H, 1, D]`` transpose)."""
    batch, seq, num_heads, head_dim = t.shape
    s_b, s_t, s_h, s_d = t.stride()
    return (
        s_d == 1
        and (num_heads == 1 or s_h == head_dim)
        and (seq == 1 or (s_t >= num_heads * head_dim and s_t % 8 == 0))
        and (batch == 1 or (s_b % 8 == 0 and 0 <= s_b < 2**31))
        and t.data_ptr() % 16 == 0
    )


def _gfx11_strides(t: torch.Tensor) -> tuple[int, int]:
    """(token stride, batch stride) the gfx11 kernel uses for ``t``; a size-1
    dim gets the plain BSHD value (token) or 0 (batch), whatever its stride."""
    batch, seq, num_heads, head_dim = t.shape
    return (
        t.stride(1) if seq > 1 else num_heads * head_dim,
        t.stride(0) if batch > 1 else 0,
    )


def _gfx11_config(
    arch: str, seq_q: int, seq_kv: int, num_heads: int, head_dim: int
) -> tuple[int, int, int | None]:
    """(num_waves, block_n, head_group) for the gfx11 kernel.

    On gfx1151, long head_dim-128 attention runs faster with 512-row query
    blocks and at most 8 heads walked together, which keeps fewer K/V pages
    live at once: 1.02x-1.10x with seq_q and seq_kv of 4096 or more. Short
    sequences, short K/V (cross-attention), head_dim below 128 and head_dim
    160 / 256 measured flat or slower with it, so they keep (16, 32, all heads).
    """
    if arch == "gfx1151" and head_dim == 128 and min(seq_q, seq_kv) >= 4096:
        return 32, 32, max(g for g in range(1, 9) if num_heads % g == 0)
    return 16, 32, None


def _gfx11_varlen_config(
    arch: str,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    head_dim: int,
    batch: int,
    causal: bool,
) -> tuple[int, int, int | None, int]:
    """(num_waves, block_n, head_group, vt_rows) for the gfx11 kernel on a
    varlen call; ``vt_rows`` is the number of keys per LDS write in the V
    transpose.

    Causal (decoder prefill), measured on gfx1151 with 14 / 2 heads at head_dim
    64, 32 / 8 and 16 / 8 heads at head_dim 128 and 8 / 2 heads at head_dim 256,
    on single and batched prefills of 256-8192 tokens, query chunks over a 2-4x
    longer prefix and speculative-decode batches (4-8 queries over 2048-4096
    keys):

    - 128-row query blocks everywhere: under the mask the blocks of one sequence
      do very different amounts of work, and smaller blocks balance them. The
      batched rule's 256 rows lost 5-45% on prefills and about 2x on speculative
      decode, where a 4-row sequence pads to a whole block.
    - head_dim up to 64: 64 keys, 2 keys per transpose write.
    - head_dim up to 128: 32 keys. A chunk of queries over a prefix at least
      twice as long walks a long KV range per block and takes 64 keys with 8-key
      transpose writes, in 256-row blocks from 512 queries over 4096+ keys,
      where 128 rows lost 4-8% (single and batched chunks); a 6144+ token
      prefill takes the same 256-row tile (2%).
    - head_dim above 128: 32 keys (64 do not fit the LDS at head_dim 256) with
      8-key transpose writes, 17-33% faster than 2-key writes.

    Non-causal, head_dim up to 64, measured on gfx1151 with 16-head and
    12-head head_dim-64 ViTs on the 863 ``cu_seqlens`` patterns of a recorded
    vLLM video workload (1-16 images of 144-8160 tokens), weighted by call
    count. The first rule took the query tile of 128 / 192 / 256 rows that pads
    the longest sequence least (ties to the larger) and 64 keys with 4-key
    transpose writes for at most four sequences of 512+ tokens, else 32 keys.
    Every (rows) x (32 / 64 keys) x (2 / 4 transpose keys) combination was then
    measured on the patterns under 1280 tokens, and a band changes only where
    one choice beats the first rule on both head counts without losing on any
    pattern:

    - 128-row blocks with 2-key writes: 32 keys at 193-511 tokens (9% over
      the first rule), 64 keys at 640-767 tokens (5-6%) and from 1024 tokens
      (7% up to 1280, 3-3.5% beyond).
    - the first rule stays up to 192 tokens, at 512-639 and 768-1023 tokens,
      and at 256-383 tokens with more than four sequences: there its
      padding-least tile wins, by up to 23% on 3 x 160 tokens, and the other
      choices lose on some patterns of one head count.

    Whole workload: 3.6% (16 heads) and 3.1% (12 heads) over the first rule,
    no pattern slower by more than 2.2%.

    Non-causal head_dim 65-72 (the 16 x 72 ViTs of Qwen3-VL 8B+ and Qwen3.5
    9B+), same patterns and grid, against the batched rule's 256-row / 32-key
    tile: the larger K / V tiles favour 192 rows where head_dim 64 took 128.
    Up to 192 tokens and at 384-511 the padding-least tile with 32 keys (50%
    and 2%); 193-383 tokens 128 rows / 32 keys (32%); 512-767 tokens with at
    most four sequences the padding-least of 128 / 192 rows with 64 keys, with
    more sequences 32 keys below 640 and the batched tile above (10%);
    768-1023 with at most four sequences the padding-least of 192 / 256 rows
    with 64 keys, else the batched tile (3%); 1024-1279 the padding-least tile
    with 64 keys for at most four sequences, else 32 (5%); 192 rows / 64 keys
    at 1280-1535, and up to 2047 for at most four sequences (6%); from there
    the padding-least of 192 / 256 rows with 64 keys (3%). All with 2-key
    writes. Whole workload 3.7%, no pattern slower beyond the 2.3% noise
    between two runs of the same tile.

    Other non-causal head dims keep the batched rule with 2-key writes.
    """
    if causal:
        if head_dim <= 64:
            return 8, 64, None, 2
        if head_dim <= 128:
            chunk = max_seqlen_k >= 2 * max_seqlen_q
            if max_seqlen_k >= 4096 and (
                max_seqlen_q >= 6144 or (chunk and max_seqlen_q >= 512)
            ):
                return 16, 64, None, 8
            if chunk:
                return 8, 64, None, 8
            return 8, 32, None, 2
        return 8, 32, None, 8
    if head_dim <= 64:
        mx = max_seqlen_q
        if (
            mx <= 192
            or 512 <= mx < 640
            or 768 <= mx < 1024
            or (256 <= mx < 384 and batch > 4)
        ):
            # The first rule: the query tile that pads the longest sequence
            # least, 64 keys for at most four sequences of 512+ tokens.
            num_waves = min(
                (8, 12, 16), key=lambda n: (-(-mx // (16 * n)) * 16 * n, -n)
            )
            block_n = 64 if batch <= 4 and mx >= 512 else 32
            return num_waves, block_n, None, 4 if block_n == 64 else 2
        if mx < 512:
            return 8, 32, None, 2
        return 8, 64, None, 2
    if head_dim <= 72:
        mx = max_seqlen_q
        few = batch <= 4

        def pad_least(cands: tuple[int, ...]) -> int:
            return min(cands, key=lambda n: (-(-mx // (16 * n)) * 16 * n, -n))

        if mx <= 192 or 384 <= mx < 512:
            return pad_least((8, 12, 16)), 32, None, 2
        if mx < 384:
            return 8, 32, None, 2
        if mx < 768:
            if few:
                return pad_least((8, 12)), 64, None, 2
            if mx < 640:
                return pad_least((8, 12, 16)), 32, None, 2
            return 16, 32, None, 2
        if mx < 1024:
            return (pad_least((12, 16)), 64, None, 2) if few else (16, 32, None, 2)
        if mx < 1280:
            return pad_least((8, 12, 16)), 64 if few else 32, None, 2
        if mx < 1536 or (mx < 2048 and few):
            return 12, 64, None, 2
        return pad_least((12, 16)), 64, None, 2
    num_waves, block_n, head_group = _gfx11_config(
        arch, max_seqlen_q, max_seqlen_k, num_heads, head_dim
    )
    return num_waves, block_n, head_group, 4 if block_n >= 64 else 2


def _gfx11_varlen_rope_config(
    max_seqlen: int, num_heads: int, batch: int
) -> tuple[int, int]:
    """(num_waves, block_n) for the gfx11 varlen kernel with RoPE in the kernel.

    Every query block rotates the K tiles it walks, so fewer, larger query
    blocks pay less for it than in the plain varlen kernel. Measured on gfx1151
    with 16-head and 12-head head_dim-64 ViTs on the 863 cu_seqlens patterns of
    a recorded vLLM video workload (1-16 images of 144-8160 tokens), weighted by
    call count:

    - query tile: 128 rows (8 waves) for sequences under 512 tokens (192 and
      256 rows lost 1-7% there). Longer ones take 512 rows (32 waves) when that
      pads the longest sequence at most 3% more than 192 rows (12 waves) and
      still gives at least 64 workgroups; otherwise 256 rows (16 waves) from
      3000 tokens, which gained 2% over 192 rows, and 192 rows below.
    - KV tile: 64 keys (4-12% faster than 32) unless there are more than four
      sequences of under 512 tokens, where 32 keys keep more blocks resident.

    The caller builds this kernel with ``vt_rows=2``: with the rotation in the
    loop, 2 keys per V transpose write beat 4 by 0.3-0.6% at 64 keys on every
    tile, and by 2-3% on the short sequences.
    """
    block_n = 32 if max_seqlen < 512 and batch > 4 else 64
    if max_seqlen < 512:
        return 8, block_n

    def padded(rows: int) -> int:
        return -(-max_seqlen // rows) * rows

    big_blocks = -(-max_seqlen // 512) * num_heads * batch
    if padded(512) <= 1.03 * padded(192) and big_blocks >= 64:
        return 32, block_n
    return (16 if max_seqlen >= 3000 else 12), block_n


def _gfx11_thd_direct(t: torch.Tensor) -> bool:
    """THD counterpart of ``_gfx11_direct``: ``[total, heads, head_dim]`` is read
    in place when head_dim and heads are packed, the token stride is a multiple
    of 8 elements and at least one row, and the start is 16-byte aligned. Covers
    plain THD and views of a packed ``[total, 3, H, D]`` tensor."""
    total, num_heads, head_dim = t.shape
    s_t, s_h, s_d = t.stride()
    return (
        s_d == 1
        and (num_heads == 1 or s_h == head_dim)
        and (total == 1 or (s_t >= num_heads * head_dim and s_t % 8 == 0))
        and t.data_ptr() % 16 == 0
    )


def _flydsl_flash_attn_func_gfx11(
    q, k, v, causal, softmax_scale, stream, out, return_lse=False, arch=""
):
    """gfx11 launch; the caller has validated devices, shapes, dtypes and ``out``.
    Returns ``out``, or ``(out, lse)`` with ``return_lse`` (fp32 ``[B, H, Sq]``)."""
    batch, seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[1]

    num_waves, block_n, head_group = _gfx11_config(
        arch, seq_q, seq_kv, num_heads, head_dim
    )
    with torch.cuda.device(q.device.index):
        launch_stream = (
            stream if stream is not None else torch.cuda.current_stream(q.device)
        )
        with torch.cuda.stream(launch_stream):
            for tensor in (q, k, v):
                tensor.record_stream(launch_stream)
            if out is not None:
                out.record_stream(launch_stream)

            # Plain BSHD and packed views (e.g. qkv[:, :, i] of [B, S, 3, H, D])
            # are read in place; anything else is copied. .contiguous() returns
            # an already-contiguous tensor as is, so a misaligned one is cloned.
            def _readable(t):
                if _gfx11_direct(t):
                    return t
                c = t.contiguous()
                return (
                    c
                    if _gfx11_direct(c)
                    else t.clone(memory_format=torch.contiguous_format)
                )

            q_c, k_c, v_c = (_readable(t) for t in (q, k, v))
            strides = [_gfx11_strides(t) for t in (q_c, k_c, v_c)]
            for t, seq, (tok, _) in zip(
                (q_c, k_c, v_c), (seq_q, seq_kv, seq_kv), strides
            ):
                # Per-batch buffer descriptors address with 32-bit byte offsets.
                rows = (seq - 1) * tok + t.shape[2] * head_dim
                if rows * t.element_size() >= 2**31:
                    raise ValueError(
                        "one batch slice of q/k/v must be smaller than 2 GiB"
                    )
            # The kernel writes O as plain BSHD: allocate it that way (empty_like
            # would keep the strides of a dense permuted q, e.g. a seq-first
            # transpose) and write into out only if it is plain BSHD and aligned.
            write_in_place = (
                out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0
            )
            o = (
                out
                if write_in_place
                else torch.empty(q.shape, dtype=q.dtype, device=q.device)
            )
            lse = (
                torch.empty(
                    batch, num_heads, seq_q, dtype=torch.float32, device=q.device
                )
                if return_lse
                else None
            )
            exe = _get_kernel_gfx11(
                device_index=q.device.index,
                num_heads=num_heads,
                num_kv_heads=k.shape[2],
                head_dim=head_dim,
                causal=causal,
                dtype_str=_torch_dtype_to_str(q.dtype),
                softmax_scale=softmax_scale,
                kv_tail=seq_kv % block_n != 0,
                num_waves=num_waves,
                block_n=block_n,
                head_group=head_group,
                token_strides=tuple(tok for tok, _ in strides),
                return_lse=return_lse,
            )
            exe(
                q_c,
                k_c,
                v_c,
                o,
                batch,
                seq_q,
                seq_kv,
                stream=launch_stream,
                q_batch_stride=strides[0][1],
                k_batch_stride=strides[1][1],
                v_batch_stride=strides[2][1],
                lse=lse,
            )
            if out is not None and not write_in_place:
                out.copy_(o)
                o = out
    return (o, lse) if return_lse else o


def _gfx11_varlen_supported(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    softmax_scale,
    causal,
    dropout_p,
    window_size,
    bias,
    alibi_slopes,
    sink,
    block_table,
    return_attn_probs,
    descales,
    out,
) -> bool:
    """Gate for the gfx11 varlen kernel: packed THD bf16 / f16, head_dim % 8 == 0,
    GQA by head ratio, int32 ``cu_seqlens`` of equal length on the device, and
    none of the options the kernel has no code for. Causal takes separate tables:
    the mask is bottom-right aligned, as in flash-attn."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3 or k.shape != v.shape:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16) or not (
        q.dtype == k.dtype == v.dtype
    ):
        return False
    if q.shape[2] != k.shape[2] or q.shape[2] % 8 != 0 or q.shape[2] == 0:
        return False
    if k.shape[1] < 1 or q.shape[1] % k.shape[1] != 0:
        return False
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        return False
    for cu in (cu_seqlens_q, cu_seqlens_k):
        if (
            not torch.is_tensor(cu)
            or cu.dtype != torch.int32
            or cu.dim() != 1
            or cu.numel() < 2
            or cu.device != q.device
        ):
            return False
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        return False
    if softmax_scale is not None and (
        torch.is_tensor(softmax_scale) or not (0 < float(softmax_scale) < float("inf"))
    ):
        return False
    if dropout_p != 0.0 or any(w >= 0 for w in window_size[:2]):
        return False
    if len(window_size) > 2 and window_size[2] != 0:
        return False
    if (
        bias is not None
        or alibi_slopes is not None
        or sink is not None
        or block_table is not None
        or return_attn_probs
        or any(d is not None for d in descales)
    ):
        return False
    return out is None or (
        out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
    )


def _flydsl_flash_attn_varlen_gfx11(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    out,
    return_lse,
    arch,
):
    """gfx11 varlen launch; ``_gfx11_varlen_supported`` has accepted the call.
    Returns ``out`` (``[total_q, H, D]``), or ``(out, lse)`` with ``return_lse``
    (lse fp32 ``[H, total_q]``, flash-attn's varlen convention)."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.numel() - 1
    num_waves, block_n, head_group, vt_rows = _gfx11_varlen_config(
        arch, max_seqlen_q, max_seqlen_k, num_heads, head_dim, batch, causal
    )
    # Switching the current device costs several microseconds; a 208-token ViT
    # image takes 13. Only do it when q lives on another device.
    device_ctx = (
        nullcontext()
        if q.device.index == torch.cuda.current_device()
        else torch.cuda.device(q.device.index)
    )
    with device_ctx:
        stream = torch.cuda.current_stream(q.device)

        def _readable(t):
            if _gfx11_thd_direct(t):
                return t
            c = t.contiguous()
            return (
                c
                if _gfx11_thd_direct(c)
                else t.clone(memory_format=torch.contiguous_format)
            )

        q_c, k_c, v_c = (_readable(t) for t in (q, k, v))
        cu_q, cu_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
        token_strides = tuple(
            t.stride(0) if t.shape[0] > 1 else t.shape[1] * head_dim
            for t in (q_c, k_c, v_c)
        )
        for t, seq, tok in zip(
            (q_c, k_c, v_c), (max_seqlen_q, max_seqlen_k, max_seqlen_k), token_strides
        ):
            # Per-sequence buffer descriptors address with 32-bit byte offsets.
            rows = (seq - 1) * tok + t.shape[1] * head_dim
            if rows * t.element_size() >= 2**31:
                raise ValueError("one sequence of q/k/v must be smaller than 2 GiB")
        write_in_place = (
            out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0
        )
        o = (
            out
            if write_in_place
            else torch.empty(q.shape, dtype=q.dtype, device=q.device)
        )
        lse = (
            torch.empty(num_heads, total_q, dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        exe = _get_varlen_kernel_gfx11(
            device_index=q.device.index,
            num_heads=num_heads,
            num_kv_heads=k.shape[1],
            head_dim=head_dim,
            causal=causal,
            dtype_str=_torch_dtype_to_str(q.dtype),
            softmax_scale=softmax_scale,
            num_waves=num_waves,
            block_n=block_n,
            vt_rows=vt_rows,
            head_group=head_group,
            token_strides=token_strides,
            return_lse=return_lse,
        )
        exe(q_c, k_c, v_c, o, cu_q, cu_k, batch, max_seqlen_q, stream=stream, lse=lse)
        if out is not None and not write_in_place:
            out.copy_(o)
            o = out
    return (o, lse) if return_lse else o


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    softmax_scale: float | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL Flash Attention on gfx11 (RDNA3 / RDNA3.5) or gfx1201 (RDNA4).

    Args:
        q: tensor with shape ``[batch, seqlen_q, num_heads, head_dim]`` (BSHD).
        k, v: tensors with shape ``[batch, seqlen_kv, num_heads, head_dim]``.
            All three inputs must share dtype, batch, num_heads, and head_dim;
            on gfx11 k/v may have fewer heads than q (GQA / MQA, num_heads a
            multiple of the K/V heads).
        causal: apply causal masking when ``True``. Causal cross-attention is
            not supported.
        waves_per_eu: kernel occupancy hint passed to the gfx1201 builder.
        daz: enable denormals-are-zero on the gfx1201 kernel.
        stream: optional launch stream. The caller owns normal PyTorch stream
            dependencies between producers and consumers.
        softmax_scale: optional positive finite QK scale. Defaults to
            ``1 / sqrt(head_dim)``.
        q_descale, k_descale, v_descale: one-element float32 device tensors
            required for FP8 input (gfx1201 only; gfx11 takes BF16/F16).
        out: optional output tensor. It must not alias an input. FP8 input
            requires BF16 output.
        return_lse: also return the fp32 softmax log-sum-exp (natural log),
            ``[batch, num_heads, seqlen_q]``, as in flash-attn. gfx11 only.

    Returns:
        Output with Q's shape. Its dtype matches Q for BF16/F16 input and is
        BF16 for FP8 input.
        With ``return_lse``, a tuple ``(out, lse)``.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible, dimensions are
            empty, the output aliases an input, or the kernel's ``head_dim``
            constraints are not met.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            "q/k/v must reside on the same device, got "
            f"q={q.device} k={k.device} v={v.device}"
        )
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:  # noqa: BLE001
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    # gfx110x / gfx115x (RDNA3 / RDNA3.5). Not gfx117x: its WMMA operands follow
    # the gfx12 layout although the target is gfx11.
    is_gfx11 = arch_base.startswith(("gfx110", "gfx115"))
    if not (is_gfx11 or arch_base.startswith("gfx1201")):
        raise ValueError(
            "flydsl_flash_attn_func requires gfx110x / gfx115x or gfx1201, "
            f"got {arch!r}"
        )
    if return_lse and not is_gfx11:
        raise ValueError("return_lse is only supported on gfx110x / gfx115x")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"expected 4D BSHD tensors, got {q.dim()}/{k.dim()}/{v.dim()} dimensions"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k/v shapes must match, got {tuple(k.shape)}/{tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    # gfx11 also takes fewer K/V heads than query heads (GQA / MQA).
    heads_ok = q.shape[2] == k.shape[2] or (
        is_gfx11 and k.shape[2] > 0 and q.shape[2] % k.shape[2] == 0
    )
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3] or not heads_ok:
        raise ValueError(
            "q/k must share batch, num_heads (gfx11: num_heads a multiple of "
            "the K/V heads), and head dimension, got "
            f"{tuple(q.shape)}/{tuple(k.shape)}"
        )
    if stream is not None:
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                "stream must be a torch.cuda.Stream or None, got "
                f"{type(stream).__name__}"
            )
        if stream.device != q.device:
            raise ValueError(f"stream must belong to {q.device}, got {stream.device}")

    batch, seq_len_real, num_heads, head_dim = q.shape
    seq_len_kv_real = k.shape[1]
    if batch == 0 or seq_len_real == 0 or seq_len_kv_real == 0 or num_heads == 0:
        raise ValueError(
            "batch, query sequence length, KV sequence length, and num_heads "
            "must all be non-zero"
        )
    is_cross = seq_len_real != seq_len_kv_real
    if causal and is_cross:
        raise ValueError("causal cross-attention is not supported")
    if is_gfx11:
        if head_dim % 8 != 0:
            raise ValueError(f"gfx11 kernel requires head_dim % 8 == 0, got {head_dim}")
    elif head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    if softmax_scale is not None:
        if torch.is_tensor(softmax_scale):
            if softmax_scale.numel() != 1:
                raise ValueError("softmax_scale must be a scalar")
            if softmax_scale.device.type != "cpu":
                raise ValueError("tensor softmax_scale must be on CPU")
            softmax_scale = softmax_scale.item()
        softmax_scale = float(softmax_scale)
        if not math.isfinite(softmax_scale) or softmax_scale <= 0:
            raise ValueError("softmax_scale must be finite and positive")

    is_fp8 = q.dtype in _FP8_DTYPES
    if is_fp8 and is_gfx11:
        raise ValueError("the gfx11 kernel supports bf16/f16 only, got FP8 input")
    if is_fp8:
        for name, scale in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            if (
                not torch.is_tensor(scale)
                or scale.dtype != torch.float32
                or scale.numel() != 1
                or scale.device != q.device
            ):
                raise ValueError(
                    f"{name} must be a one-element float32 tensor on {q.device}"
                )
        dtype_str = "bf16"
    else:
        if any(scale is not None for scale in (q_descale, k_descale, v_descale)):
            raise ValueError("q/k/v descales are only valid for FP8 input")
        dtype_str = _torch_dtype_to_str(q.dtype)

    output_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if out is not None:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}")
        if out.dtype != output_dtype:
            raise ValueError(f"out must have dtype {output_dtype}")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}")
        if _has_unsupported_internal_overlap(out):
            raise ValueError(
                "out must not have internal overlap or unsupported striding"
            )
        alias_inputs = [q, k, v]
        if is_fp8:
            alias_inputs.extend((q_descale, k_descale, v_descale))
        if any(_storage_overlaps(out, tensor) for tensor in alias_inputs):
            raise ValueError("out must not overlap q, k, v, or FP8 descale storage")
    in_place = _gfx11_direct if is_gfx11 else (lambda t: t.is_contiguous())
    materialized_inputs = not (in_place(q) and in_place(k) and in_place(v))
    temporary_output = out is not None and not out.is_contiguous()
    if materialized_inputs or temporary_output:
        parts = []
        if materialized_inputs:
            parts.append("non-contiguous Q/K/V")
        if temporary_output:
            parts.append("a non-contiguous out")
        warnings.warn(
            "flydsl_flash_attn_func materializes "
            + " and ".join(parts)
            + "; provide contiguous tensors to avoid the copy",
            stacklevel=2,
        )

    if is_gfx11:
        return _flydsl_flash_attn_func_gfx11(
            q, k, v, causal, softmax_scale, stream, out, return_lse, arch_base
        )

    block_m, block_n = _pick_gfx1201_tiles(seq_len_real, head_dim, causal)
    lds_bytes = _gfx1201_fmha_lds_bytes(head_dim, block_n, fp8=is_fp8)
    if lds_bytes > _GFX1201_LDS_CAPACITY_BYTES and block_n != 32:
        # Keep the selected query tile, but fall back to the narrower KV tile
        # before rejecting a shape that the kernel can safely represent.
        block_n = 32
        lds_bytes = _gfx1201_fmha_lds_bytes(head_dim, block_n, fp8=is_fp8)
    if lds_bytes > _GFX1201_LDS_CAPACITY_BYTES:
        kernel_kind = "FP8" if is_fp8 else dtype_str.upper()
        raise ValueError(
            f"gfx1201 {kernel_kind} attention with head_dim={head_dim} and "
            f"BLOCK_N={block_n} requires {lds_bytes} bytes of LDS, exceeding "
            f"the {_GFX1201_LDS_CAPACITY_BYTES}-byte hardware limit"
        )

    # Pad sequence lengths to their respective tile sizes. Non-causal padded
    # K/V columns are masked in the kernel; padded query rows are sliced off.
    seq_len_pad = ((seq_len_real + block_m - 1) // block_m) * block_m
    seq_len_kv_pad = (
        seq_len_pad
        if not is_cross
        else ((seq_len_kv_real + block_n - 1) // block_n) * block_n
    )
    tail_mask = not causal and seq_len_kv_real % block_n != 0
    _validate_gfx1201_launch_limits(
        batch=batch,
        seq_len=seq_len_pad,
        seq_len_kv_real=seq_len_kv_real,
        seq_len_kv=seq_len_kv_pad,
        num_heads=num_heads,
        head_dim=head_dim,
        fp8=is_fp8,
    )

    def _pad_seq(tensor: torch.Tensor, pad: int) -> torch.Tensor:
        tensor = tensor.contiguous()
        if pad == 0:
            return tensor
        if tensor.dtype in _FP8_DTYPES:
            # Some backends do not implement constant padding for float8.
            # Padding the byte representation is equivalent because FP8 is
            # one byte per element and the all-zero bit pattern represents 0.
            return F.pad(tensor.view(torch.uint8), (0, 0, 0, 0, 0, pad)).view(
                tensor.dtype
            )
        return F.pad(tensor, (0, 0, 0, 0, 0, pad))

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        launch_stream = (
            stream if stream is not None else torch.cuda.current_stream(q.device)
        )

        with torch.cuda.stream(launch_stream):
            # This is normal allocator lifetime tracking, not a readiness
            # protocol: callers establish producer ordering themselves.
            for tensor in (q, k, v):
                tensor.record_stream(launch_stream)
            if out is not None:
                out.record_stream(launch_stream)
            if is_fp8:
                for tensor in (q_descale, k_descale, v_descale):
                    tensor.record_stream(launch_stream)
            q_p = _pad_seq(q, seq_len_pad - seq_len_real)
            k_p = _pad_seq(k, seq_len_kv_pad - seq_len_kv_real)
            v_p = _pad_seq(v, seq_len_kv_pad - seq_len_kv_real)
            write_in_place = (
                out is not None and seq_len_pad == seq_len_real and out.is_contiguous()
            )
            o_p = out if write_in_place else torch.empty_like(q_p, dtype=output_dtype)

            if is_fp8:
                exe = _get_fp8_gfx1201_kernel(
                    device_index=q.device.index,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=softmax_scale,
                    tail_mask=tail_mask,
                    cross_attn=is_cross,
                )
                exe(
                    q_p.reshape(-1),
                    k_p.reshape(-1),
                    v_p.reshape(-1),
                    o_p.reshape(-1),
                    batch,
                    seq_len_pad,
                    seq_len_kv_real,
                    seq_len_kv_pad,
                    q_descale,
                    k_descale,
                    v_descale,
                    stream=launch_stream,
                )
            else:
                exe = _get_kernel(
                    device_index=q.device.index,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    causal=causal,
                    dtype_str=dtype_str,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    block_m=block_m,
                    block_n=block_n,
                    softmax_scale=softmax_scale,
                    tail_mask=tail_mask,
                    cross_attn=is_cross,
                    lds_vec_width=(
                        16
                        if os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1")
                        == "1"
                        else 8
                    ),
                )
                exe(
                    # Preserve the original rank: FlyDSL's tensor ABI stores
                    # each dimension as signed Int32, so flattening a legal
                    # 2^31-element tensor would make its sole dimension overflow.
                    q_p,
                    k_p,
                    v_p,
                    o_p,
                    batch,
                    seq_len_pad,
                    seq_len_kv_real,
                    seq_len_kv_pad,
                    stream=launch_stream,
                )

            result = o_p[:, :seq_len_real, :, :] if seq_len_pad != seq_len_real else o_p
            if out is not None and not write_in_place:
                out.copy_(result)
                result = out
            elif out is None:
                result = result.contiguous()

    return result


def _check_out(out, q, v_hdim, out_dtype=None):
    """Validate a caller-provided ``out`` tensor. ``out_dtype`` defaults to q's -- the
    8-bit gfx1250 kernel writes bf16 output, so it passes its own."""
    assert out is not None, "out must be provided"
    assert torch.is_tensor(out), f"out must be a tensor, got {type(out).__name__}"
    want_dtype = q.dtype if out_dtype is None else out_dtype
    assert out.dtype == want_dtype, f"out dtype {out.dtype} != expected {want_dtype}"
    assert out.device == q.device, f"out device {out.device} != q device {q.device}"
    assert out.stride(-1) == 1, "out must have a contiguous last dimension"
    want = (*tuple(q.shape[:-1]), v_hdim)
    assert (
        tuple(out.shape) == want
    ), f"out shape {tuple(out.shape)} != expected {want} (v_hdim={v_hdim})"


def _is_per_tensor_descale(d, q):
    """A per-tensor fp32 descale, as gfx950 defines it."""
    return (
        torch.is_tensor(d)
        and d.dtype == torch.float32
        and d.numel() == 1
        and d.device == q.device
    )


def _has_implicit_bshd_batch_stride(t, seq_len):
    """The BSHD kernels take no batch stride: they address batch b at
    ``b * seq_len * stride_seq``. A view whose own batch stride differs from that
    product (e.g. [S,B,H,D] storage permuted to [B,S,H,D]) reads the wrong rows, or
    off the end. Irrelevant at batch 1, which never forms a nonzero batch offset."""
    return t.shape[0] == 1 or t.stride(0) == seq_len * t.stride(1)


_A8W8_GFX1250_IN_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _fp8_gfx950_supported(
    q,
    k,
    v,
    *,
    softmax_scale,
    dropout_p,
    window_size,
    bias,
    alibi_slopes,
    sink,
    return_attn_probs,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    out,
) -> bool:
    """Gate for the gfx950 fp8 kernel.

    It needs e4m3fn Q/K/V with per-tensor descales and a positive, finite
    softmax scale, and writes bf16. Reject anything else so it falls through rather than
    silently dropping the feature.
    """
    if q.dtype is not torch.float8_e4m3fn or q_descale is None:
        return False

    from .kernels.flash_attn_func_fp8_gfx950 import (
        _is_valid_softmax_scale,
        flydsl_flash_attn_fp8_supported,
    )

    if q.dim() not in (3, 4) or k.dim() != q.dim() or v.dim() != q.dim():
        return False
    if not (k.dtype == v.dtype == torch.float8_e4m3fn):
        return False
    if out is not None and (out.dtype != torch.bfloat16 or not out.is_contiguous()):
        return False
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        return False
    if any(
        s is None
        or not torch.is_tensor(s)
        or s.dtype != torch.float32
        or s.numel() != 1
        or s.device != q.device
        for s in (q_descale, k_descale, v_descale)
    ):
        return False
    qk_hdim = q.shape[-1]
    if not _is_valid_softmax_scale(softmax_scale):
        return False
    nq, nkv = q.shape[-2], k.shape[-2]
    if not flydsl_flash_attn_fp8_supported(
        q.device, nq, nkv, qk_hdim, v.shape[-1], dtype=q.dtype
    ):
        return False
    return (
        k.shape[-1] == qk_hdim
        and nkv > 0
        and nq % nkv == 0
        and dropout_p == 0.0
        and all(w < 0 for w in window_size[:2])
        and (len(window_size) < 3 or window_size[2] == 0)
        and bias is None
        and alibi_slopes is None
        and sink is None
        and block_table is None
        and not return_attn_probs
    )


def flydsl_flash_attn_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    sink=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """FlyDSL MHA forward, varlen THD layout.

    Returns the result if FlyDSL can handle this configuration,
    otherwise returns None so the caller falls through to Triton/CK.
    """
    from ...jit.utils.chip_info import get_gfx_runtime

    gfxip = get_gfx_runtime()
    if gfxip == "gfx950":
        if (
            q.dtype is torch.float8_e4m3fn
            and q.dim() == 3
            and _fp8_gfx950_supported(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                sink=sink,
                return_attn_probs=return_attn_probs,
                block_table=block_table,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out,
            )
        ):
            from .kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func

            return flydsl_flash_attn_fp8_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_k,
                cross_seqlen=max_seqlen_q != max_seqlen_k,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out,
                return_lse=return_lse,
            )
    elif gfxip == "gfx1250":
        # 8-bit Q/K/V go to the a8w8 sibling, which writes bf16 output; everything
        # else to a16w16, whose output follows the input.
        _is_b8 = q.dtype in _A8W8_GFX1250_IN_DTYPES
        # a8w8 applies per-tensor fp32 descales (gfx950's contract); a16w16 has none,
        # and a partial set is ambiguous, so both fall through.
        _descales = (q_descale, k_descale, v_descale)
        if any(d is not None for d in _descales) and not (
            _is_b8 and all(_is_per_tensor_descale(d, q) for d in _descales)
        ):
            return None
        if any(t.stride(-1) != 1 for t in (q, k, v)):
            return None
        if (
            sink is not None
            and torch.is_tensor(sink)
            and sink.dim() > 0
            and sink.stride(-1) != 1
        ):
            return None
        if softmax_scale is not None and not (0 < softmax_scale < float("inf")):
            return None
        _out_dtype = _A8W8_GFX1250_OUT_DTYPE if _is_b8 else q.dtype
        _supported = (
            flash_attn_a8w8_gfx1250_supported
            if _is_b8
            else flash_attn_a16w16_gfx1250_supported
        )
        if not _supported(
            "varlen",
            q,
            k,
            v,
            dropout_p=dropout_p,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            sink=sink,
            block_table=block_table,
            return_attn_probs=return_attn_probs,
        ):
            return None
        if out is None:
            out = q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=_out_dtype)
        else:
            _check_out(out, q, v.shape[-1], out_dtype=_out_dtype)
        return (
            flash_attn_varlen_a8w8_gfx1250
            if _is_b8
            else flash_attn_varlen_a16w16_gfx1250
        )(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            out=out,
            return_lse=return_lse,
            sink=sink,
            **(
                {
                    "q_descale": q_descale,
                    "k_descale": k_descale,
                    "v_descale": v_descale,
                }
                if _is_b8
                else {}
            ),
        )

    elif gfxip.startswith(("gfx110", "gfx115")):
        # gfx11 (RDNA3 / RDNA3.5), not gfx117x: its WMMA follows the gfx12 layout.
        try:
            _max_q, _max_k = int(max_seqlen_q), int(max_seqlen_k)
        except (TypeError, ValueError):
            return None
        if _max_q < 1 or _max_k < 0:
            return None
        if not _gfx11_varlen_supported(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            dropout_p=dropout_p,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            sink=sink,
            block_table=block_table,
            return_attn_probs=return_attn_probs,
            descales=(q_descale, k_descale, v_descale),
            out=out,
        ):
            return None
        return _flydsl_flash_attn_varlen_gfx11(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            _max_q,
            _max_k,
            None if softmax_scale is None else float(softmax_scale),
            bool(causal),
            out,
            return_lse,
            gfxip,
        )

    return None


def flydsl_flash_attn_varlen_qkv_rope_fwd(
    qkv: torch.Tensor,
    num_heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Vision-encoder attention from the packed qkv projection, RoPE included.

    For ViT blocks such as the Qwen2.5-VL / Qwen3-VL vision attention: ``qkv`` is
    the qkv linear output, one row ``[q (H*D) | k (H*D) | v (H*D)]`` per packed
    token, as ``[total, 3*H*D]``, ``[total, 1, 3*H*D]`` or ``[total, 3, H, D]``
    (inner dims contiguous, any row stride). Q and K are rotated with NeoX RoPE
    over the whole head (``x[d] * cos[d] - x[d + D/2] * sin[d]``,
    ``x[d + D/2] * cos[d] + x[d] * sin[d]``) using ``cos`` / ``sin`` of shape
    ``[total, D/2]`` (bf16 or fp32, one row per token), then attend within each
    ``cu_seqlens`` sequence (self-attention, non-causal). Equivalent to the
    separate rotary + ``flydsl_flash_attn_varlen_fwd``, without writing the
    rotated Q / K to memory.

    Returns ``[total, H, D]`` (``out`` if given), or None when the call is not
    supported here, so the caller falls back to the unfused path. gfx110x /
    gfx115x, bf16 / f16 qkv, ``head_dim % 16 == 0``.
    """
    from ...jit.utils.chip_info import get_gfx_runtime

    gfxip = get_gfx_runtime()
    if not gfxip.startswith(("gfx110", "gfx115")):
        return None
    if not (torch.is_tensor(qkv) and qkv.is_cuda) or num_heads < 1:
        return None
    if qkv.dtype not in (torch.bfloat16, torch.float16):
        return None
    total = qkv.shape[0]
    if qkv.dim() == 4:
        if qkv.shape[1] != 3 or qkv.shape[2] != num_heads:
            return None
        head_dim = qkv.shape[3]
        inner_ok = (
            qkv.stride(3) == 1
            and qkv.stride(2) == head_dim
            and qkv.stride(1) == num_heads * head_dim
        )
    elif qkv.dim() in (2, 3):
        if qkv.dim() == 3 and qkv.shape[1] != 1:
            return None
        width = qkv.shape[-1]
        if width % (3 * num_heads):
            return None
        head_dim = width // (3 * num_heads)
        inner_ok = qkv.stride(-1) == 1
    else:
        return None
    if not inner_ok or head_dim % 16 != 0 or total < 1:
        return None
    tok = qkv.stride(0) if total > 1 else 3 * num_heads * head_dim
    if tok % 8 != 0 or qkv.data_ptr() % 16 != 0:
        return None
    for t in (cos, sin):
        if (
            not torch.is_tensor(t)
            or t.dim() != 2
            or t.shape != (total, head_dim // 2)
            or t.dtype not in (torch.bfloat16, torch.float32)
            or t.device != qkv.device
            or t.stride(1) != 1
            or (t.stride(0) * t.element_size()) % 16 != 0
            or t.data_ptr() % 16 != 0
        ):
            return None
    if cos.dtype != sin.dtype or cos.stride(0) != sin.stride(0):
        return None
    if (
        not torch.is_tensor(cu_seqlens)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.dim() != 1
        or cu_seqlens.numel() < 2
        or cu_seqlens.device != qkv.device
    ):
        return None
    try:
        max_seqlen = int(max_seqlen)
    except (TypeError, ValueError):
        return None
    if max_seqlen < 1:
        return None
    if softmax_scale is not None:
        if torch.is_tensor(softmax_scale) or not (
            0 < float(softmax_scale) < float("inf")
        ):
            return None
        softmax_scale = float(softmax_scale)
    if out is not None and (
        out.shape != (total, num_heads, head_dim)
        or out.dtype != qkv.dtype
        or out.device != qkv.device
    ):
        return None
    # Per-sequence buffer descriptors address with 32-bit byte offsets.
    if ((max_seqlen - 1) * tok + 3 * num_heads * head_dim) * 2 >= 2**31:
        return None

    batch = cu_seqlens.numel() - 1
    num_waves, block_n = _gfx11_varlen_rope_config(max_seqlen, num_heads, batch)
    head_group = None
    row = qkv.view(total, 3 * num_heads * head_dim) if qkv.dim() != 4 else None
    if row is None:
        q, k, v = qkv.unbind(1)
    else:
        part = num_heads * head_dim
        q, k, v = (
            row[:, i * part : (i + 1) * part].view(total, num_heads, head_dim)
            for i in range(3)
        )
    device_ctx = (
        nullcontext()
        if qkv.device.index == torch.cuda.current_device()
        else torch.cuda.device(qkv.device.index)
    )
    with device_ctx:
        write_in_place = (
            out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0
        )
        o = (
            out
            if write_in_place
            else torch.empty(
                total, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device
            )
        )
        exe = _get_varlen_kernel_gfx11(
            device_index=qkv.device.index,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            causal=False,
            dtype_str=_torch_dtype_to_str(qkv.dtype),
            softmax_scale=softmax_scale,
            num_waves=num_waves,
            block_n=block_n,
            # With RoPE in the loop, 2 keys per V transpose write measured
            # faster than 4 at 64 keys (see _gfx11_varlen_rope_config).
            vt_rows=2,
            head_group=head_group,
            token_strides=(tok, tok, tok),
            return_lse=False,
            rope="bf16" if cos.dtype == torch.bfloat16 else "f32",
        )
        cu = cu_seqlens.contiguous()
        exe(
            q,
            k,
            v,
            o,
            cu,
            cu,
            batch,
            max_seqlen,
            stream=torch.cuda.current_stream(qkv.device),
            cos=cos,
            sin=sin,
        )
        if out is not None and not write_in_place:
            out.copy_(o)
            o = out
    return o


def flydsl_flash_attn_paged_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    *,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size=(-1, -1),
    return_lse: bool = False,
    dropout_p: float = 0.0,
    logits_soft_cap: float = 0.0,
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    seqlen_k=None,
    out=None,
    sink=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """Paged-KV prefill, for a page-indexed K/V cache.

    ``k``/``v`` are the page pool: ``[num_blocks, page_size, nheads_k, hdim]``, or
    ``[num_blocks, nheads_k, hdim]`` at page size 1. ``qo_indptr`` carries this batch's Q
    token ranges; ``kv_indptr`` is a cumulative PAGE count (not tokens) into
    ``kv_page_indices``, and ``kv_last_page_lens`` gives each request's fill of its last
    page.

    Returns the result when FlyDSL can serve this configuration, otherwise None so the
    caller falls through to CK. Never raises on an unsupported shape.
    """
    from ...jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        return None
    # Options the paged kernel does not implement. Silently honouring any of these would
    # return wrong numbers, so each one falls through.
    if logits_soft_cap != 0.0 or block_table is not None or seqlen_k is not None:
        return None
    # Q is ragged thd-style [total_q, nheads_q, hdim]; the kernel indexes it as such.
    if q.dim() != 3:
        return None
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        return None
    _metadata = (qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens)
    if any(
        t is None
        or t.dtype != torch.int32
        or t.dim() != 1
        or t.stride(0) != 1
        or t.device != q.device
        for t in _metadata
    ):
        return None
    # Batch comes from qo_indptr; the KV side must cover it. Lower bounds only, so an
    # oversized preallocated buffer still dispatches. kv_page_indices is left to the
    # kernel: checking it needs kv_indptr[-1], a device read that would sync here.
    _batch = qo_indptr.numel() - 1
    if (
        _batch < 1
        or kv_indptr.numel() < _batch + 1
        or kv_last_page_lens.numel() < _batch
    ):
        return None
    if (
        sink is not None
        and torch.is_tensor(sink)
        and sink.dim() > 0
        and sink.stride(-1) != 1
    ):
        return None
    if softmax_scale is not None and not (0 < softmax_scale < float("inf")):
        return None
    # 8-bit Q/K/V go to the a8w8 sibling, which writes bf16 output; everything else to
    # a16w16, whose output follows the input.
    _is_b8 = q.dtype in _A8W8_GFX1250_IN_DTYPES
    # a8w8 applies per-tensor fp32 descales (gfx950's contract); a16w16 has none,
    # and a partial set is ambiguous, so both fall through.
    _descales = (q_descale, k_descale, v_descale)
    if any(d is not None for d in _descales) and not (
        _is_b8 and all(_is_per_tensor_descale(d, q) for d in _descales)
    ):
        return None
    _supported = (
        flash_attn_a8w8_gfx1250_supported
        if _is_b8
        else flash_attn_a16w16_gfx1250_supported
    )
    if not _supported(
        "paged",
        q,
        k,
        v,
        dropout_p=dropout_p,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        sink=sink,
        return_attn_probs=return_attn_probs,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
    ):
        return None

    if max_seqlen_q is None:
        max_seqlen_q = int((qo_indptr[1:] - qo_indptr[:-1]).max())
    if max_seqlen_k is None:
        page_size = 1 if k.dim() == 3 else k.shape[1]
        pages = kv_indptr[1 : _batch + 1] - kv_indptr[:_batch]
        max_seqlen_k = int(
            ((pages - 1).clamp(min=0) * page_size + kv_last_page_lens[:_batch]).max()
        )

    _out_dtype = _A8W8_GFX1250_OUT_DTYPE if _is_b8 else q.dtype
    if out is None:
        out = q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=_out_dtype)
    else:
        _check_out(out, q, v.shape[-1], out_dtype=_out_dtype)
    _entry = (
        flash_attn_paged_a8w8_gfx1250 if _is_b8 else flash_attn_paged_a16w16_gfx1250
    )
    # a16w16 has no descale parameters at all; the gate above already guaranteed none
    # were given in that case, so only the 8-bit entry is handed them.
    _descale_kwargs = (
        {"q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale}
        if _is_b8
        else {}
    )
    return _entry(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        out=out,
        return_lse=return_lse,
        sink=sink,
        **_descale_kwargs,
    )


def flydsl_flash_attn_batch_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    sink=None,
    out=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """FlyDSL MHA forward, batched BSHD ``[B, S, H, D]`` layout.

    Routes gfx950 fp8 to the dual-wave kernel and gfx1250 bf16/f16 to the
    dedicated BSHD m32x8 kernel (uniform ``seq_len``, no ``cu_seqlens`` —
    CUDA-graph safe). Returns the result if FlyDSL can handle this
    configuration, otherwise returns ``None`` so the caller falls through
    to Triton/CK.
    """
    from ...jit.utils.chip_info import get_gfx_runtime

    gfxip = get_gfx_runtime()
    if gfxip == "gfx950":
        if (
            q.dtype is torch.float8_e4m3fn
            and q.dim() == 4
            and _fp8_gfx950_supported(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                sink=sink,
                return_attn_probs=return_attn_probs,
                block_table=None,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out,
            )
        ):
            from .kernels.flash_attn_func_fp8_gfx950 import flydsl_flash_attn_fp8_func

            return flydsl_flash_attn_fp8_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out,
                return_lse=return_lse,
            )
    # deterministic is a backward-only flag (this forward is atomic-free), and
    # flash_attn_func defaults it True -- gating on it would reject every request.
    elif gfxip == "gfx1250" and q.dim() == 4:
        _is_b8 = q.dtype in _A8W8_GFX1250_IN_DTYPES
        # a8w8 applies per-tensor fp32 descales (gfx950's contract); a16w16 has none,
        # and a partial set is ambiguous, so both fall through.
        _descales = (q_descale, k_descale, v_descale)
        if any(d is not None for d in _descales) and not (
            _is_b8 and all(_is_per_tensor_descale(d, q) for d in _descales)
        ):
            return None
        if any(t.stride(-1) != 1 for t in (q, k, v)):
            return None
        if (
            sink is not None
            and torch.is_tensor(sink)
            and sink.dim() > 0
            and sink.stride(-1) != 1
        ):
            return None
        if softmax_scale is not None and not (0 < softmax_scale < float("inf")):
            return None
        if not all(_has_implicit_bshd_batch_stride(t, t.shape[1]) for t in (q, k, v)):
            return None
        if (
            out is not None
            and out.dim() == 4
            and not _has_implicit_bshd_batch_stride(out, out.shape[1])
        ):
            return None
        _out_dtype = _A8W8_GFX1250_OUT_DTYPE if _is_b8 else q.dtype
        _supported = (
            flash_attn_a8w8_gfx1250_supported
            if _is_b8
            else flash_attn_a16w16_gfx1250_supported
        )
        if not _supported(
            "batch",
            q,
            k,
            v,
            dropout_p=dropout_p,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            sink=sink,
            block_table=None,
            return_attn_probs=return_attn_probs,
        ):
            return None
        if out is None:
            out = q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=_out_dtype)
        else:
            _check_out(out, q, v.shape[-1], out_dtype=_out_dtype)
        return (
            flash_attn_batch_a8w8_gfx1250 if _is_b8 else flash_attn_batch_a16w16_gfx1250
        )(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            out=out,
            return_lse=return_lse,
            sink=sink,
            **(
                {
                    "q_descale": q_descale,
                    "k_descale": k_descale,
                    "v_descale": v_descale,
                }
                if _is_b8
                else {}
            ),
        )

    return None


def flydsl_flash_attn_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
):
    """FlyDSL MHA backward, varlen THD layout.

    Returns ``(dq, dk, dv, softmax_d)`` to match ``mha_varlen_bwd`` and
    ``fmha_v3_varlen_bwd``.  The gradients are the same tensors that were passed
    in -- the kernel fills them in place -- and ``softmax_d`` is the ``[H, T]``
    fp32 ``rowsum(dO*O)`` those two also return.

    PRECONDITION: the caller has established this configuration is supported --
    causal varlen THD self-attention, d_qk=192 / d_v=128, bf16, no GQA,
    contiguous, ``[H, T]`` fp32 LSE, no dropout / sliding window / alibi / sink /
    padded cu_seqlens, on gfx942.  The authoritative gate is
    ``can_impl_fmha_bwd_flydsl`` inside ``_flash_attn_varlen_backward`` in
    ``aiter/ops/mha.py``; the screened feature arguments are absent from this
    signature precisely because that gate has already established they are unset,
    leaving no configuration for this function to branch on.
    """
    dq, dk, dv, softmax_d = flash_attn_varlen_bwd_d192_gfx942(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        dq=dq,
        dk=dk,
        dv=dv,
    )
    return dq, dk, dv, softmax_d
