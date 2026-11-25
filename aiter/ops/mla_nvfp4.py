# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 R1 NVFP4 MLA: fixed gains, page64, one query, 128 heads.

All decode inputs are already quantized. The producers below write the native
NoPE/AUX layout directly; callers own the BF16 active tail and CSR publication.
"""

import math
import struct

import torch

from ..jit.core import compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm

G_Q = 6.72
G_KV = 89.6


# v7 loads the N128 CO name; do not reuse v6 wrappers with the old CO path.
@compile_ops("module_mla_nvfp4_v7", ffi_type="ctypes")
def mla_nvfp4_asm_fwd(
    q_nope: torch.Tensor,
    q_aux: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    params: torch.Tensor,
) -> None: ...


@compile_ops("module_mla_nvfp4_v7", ffi_type="ctypes")
def _mla_nvfp4_quantize_q(
    q: torch.Tensor, q_nope: torch.Tensor, q_aux: torch.Tensor
) -> None: ...


@compile_ops("module_mla_nvfp4_v7", ffi_type="ctypes")
def _mla_nvfp4_update_cache(
    tiles: torch.Tensor,
    page_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
) -> None: ...


def prepare_mla_nvfp4_params(
    *, device, physical_pages: int, sm_scale: float, rebase_threshold: float = 1.0
) -> torch.Tensor:
    """Create the immutable int32[4] ABI bit buffer before graph capture.

    Gains are G_Q=6.72 and G_KV=89.6. This is NOT four numeric
    integers: the first three words contain IEEE FP32 bits. Recreate this
    buffer if cache capacity or sm_scale changes. Do not modify it while in use.
    """
    if (
        isinstance(physical_pages, bool)
        or not isinstance(physical_pages, int)
        or not 1 <= physical_pages <= 0x7FFFFFFF
    ):
        raise ValueError("physical_pages must be an integer in [1, INT32_MAX]")
    if not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite")
    if rebase_threshold != 1.0:
        raise ValueError("rebase_threshold must be 1 for the O34a N128 decoder")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("params must be prepared on a GPU device")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare NVFP4 params before graph capture")
    try:
        packed = struct.pack(
            "<fffI",
            float(sm_scale) * math.log2(math.e) / (G_Q * G_KV),
            1.0 / G_KV,
            float(rebase_threshold),
            physical_pages,
        )
    except OverflowError as exc:
        raise ValueError("sm_scale overflows FP32 score scaling") from exc
    return torch.tensor(struct.unpack("<4i", packed), dtype=torch.int32, device=device)


def mla_decode_nvfp4_fwd(
    q_nope: torch.Tensor,
    q_aux: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    out: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    *,
    params: torch.Tensor,
    lse: torch.Tensor | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Write BF16 out[B,128,512] using quantized probability mass normalization.

    Q is uint8 [B,128,256] + [B,128,128]. KV is uint8 [P,16384] +
    [P,8192] in the native shuffled layout, with 256-byte aligned KV bases.
    CSR is contiguous nonnegative GPU int32: indptr[B+1] counts pages,
    indices contains physical page IDs, last[B] is 1..64 (0 for empty rows).
    Params must come from prepare_mla_nvfp4_params with the same capacity.

    Caller guarantees valid CSR, finite live data/scales and per-row length
    <= INT32_MAX. Empty rows produce O=0. No split-K, MTP or CP.
    This CO does not produce LSE; return_lse=True is unsupported. The optional
    lse buffer is accepted for ABI compatibility and remains untouched. No LSE
    scratch is allocated. Warm up before capture. All native launches
    use the query device's current stream; callers manage cross-stream lifetime.
    """
    if return_lse:
        raise ValueError("O34a NVFP4 MLA does not produce LSE; use return_lse=False")
    require_gfx1250_asm("mla_decode_nvfp4_fwd")
    mla_nvfp4_asm_fwd(
        q_nope,
        q_aux,
        kv_nope,
        kv_aux,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        out,
        lse,
        params,
    )
    return out, None


def mla_nvfp4_quantize_q(
    q: torch.Tensor,
    q_nope: torch.Tensor | None = None,
    q_aux: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize absorbed, RoPE-rotated BF16 Q[B,128,576] on the current stream.

    QN uses per-row block16 scales; QR and its residual QE each use block16.
    Supply both output buffers for an allocation-free producer. Valid inputs
    must be finite. Uses positive E4M3 scale floor 2**-9 and finite saturation.
    """
    if q_nope is None:
        q_nope = torch.empty((*q.shape[:2], 256), dtype=torch.uint8, device=q.device)
    if q_aux is None:
        q_aux = torch.empty((*q.shape[:2], 128), dtype=torch.uint8, device=q.device)
    _mla_nvfp4_quantize_q(q, q_nope, q_aux)
    return q_nope, q_aux


def mla_nvfp4_update_cache(
    tiles: torch.Tensor,
    page_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
) -> None:
    """Requantize BF16 tiles[T,16,576] directly into the page64 cache.

    page_ids/tile_ids/valid_tokens are GPU int32[T]. Each update names a unique
    (physical page, tile 0..3) and 0..16 valid rows. Unused BF16 rows may be NaN.
    C uses one scale per 16-token x 16-channel block; KR/KE use per-token block16.
    The caller retains original BF16 values for the whole active tile: a new
    outlier requires rewriting earlier tokens in that tile, not dequantizing
    old FP4. No other stream/request may concurrently write the same tile.
    Publish page lengths/CSR after this operation on the same stream or event.
    This function does not own a tail ring, page allocator or prefix copy-on-write.
    AUX always uses the N16 token-brick layout consumed by the N128 decoder.
    """
    _mla_nvfp4_update_cache(tiles, page_ids, tile_ids, valid_tokens, kv_nope, kv_aux)


# Its own HIP module (csrc/kernels/mla/nvfp4/decode_prep.cu): it needs no ASM
# code object, and editing it leaves the decoder's module alone.
@compile_ops(
    "module_mla_nvfp4_decode_prep", fc_name="mla_nvfp4_decode_prep", develop=True
)
def _mla_nvfp4_decode_prep(
    q_rope: torch.Tensor,
    q_aux: torch.Tensor,
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    cos_cache: torch.Tensor | None = None,
    sin_cache: torch.Tensor | None = None,
) -> None: ...


def mla_nvfp4_decode_prep(
    q_rope: torch.Tensor,
    q_aux: torch.Tensor,
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    cos_cache: torch.Tensor | None = None,
    sin_cache: torch.Tensor | None = None,
) -> None:
    """One decode step's Q RoPE quantization and KV cache append, one launch.

    For each of the B tokens (one per request), with BF16 q_rope[B,128,64],
    k_nope[B,512] and k_rope[B,64]:

    * q_aux[B,128,128] bytes 0..63 and 96..127 as mla_nvfp4_quantize_q writes
      them; bytes 64..95 (the NoPE scales) are left as written.
    * The token's row ``positions % 16`` of its request's BF16 tail tile
      ``ring[req_ids, :, :]`` ([R,16,576]: 512 NoPE then 64 rotated RoPE
      channels) is set, and the tile's rows 0..row are requantized into page
      ``slots // 64``, tile ``positions % 64 // 16`` of kv_nope / kv_aux, as
      mla_nvfp4_update_cache writes it (N16 AUX). A tile's first token also
      writes the zero codes and 0x7F scales of its later rows.

    With BF16 cos_cache/sin_cache [max_position,32] (GPT-J pairs, as aiter's
    rotary embedding caches them), q_rope and k_rope are rotated here at
    ``positions`` with the rounding of aiter's Triton RoPE
    (``RotaryEmbedding.forward_triton``, AITER_ROPE_TRITON_BACKEND=1); without
    them they are already rotated.
    A padding row (slot -1) gets zero Q codes and 0x7F scales and writes no
    KV; nor does a token whose page or request lies outside the cache or ring.

    q_rope rows must be 128 head strides apart (contiguous, or the RoPE slice
    of a [B,128,D] Q); req_ids is int32, positions and slots int64, all [B]. The
    caller owns the tail tiles and their recycling (a request's first token of
    a tile overwrites row 0) and must not repeat a request within a batch.
    The written tensors (q_aux, ring, kv_nope, kv_aux) must not overlap each
    other or the inputs; ring rows that overlap one another are rejected.
    gfx1250 only. Graph-safe; warm up before capture.
    """
    if (cos_cache is None) != (sin_cache is None):
        raise ValueError("cos_cache and sin_cache come together")
    if cos_cache is not None:
        # The rotary embedding's caches may carry singleton dims.
        cos_cache = cos_cache.reshape(cos_cache.shape[0], -1)
        sin_cache = sin_cache.reshape(sin_cache.shape[0], -1)
    # The launch takes the current device's stream.
    with torch.cuda.device(q_rope.device):
        _mla_nvfp4_decode_prep(
            q_rope,
            q_aux,
            ring,
            req_ids,
            k_nope,
            k_rope,
            positions,
            slots,
            kv_nope,
            kv_aux,
            cos_cache,
            sin_cache,
        )


__all__ = [
    "mla_decode_nvfp4_fwd",
    "mla_nvfp4_decode_prep",
    "mla_nvfp4_quantize_q",
    "mla_nvfp4_update_cache",
    "prepare_mla_nvfp4_params",
]
