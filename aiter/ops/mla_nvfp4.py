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
G_P = 2688.0


@compile_ops("module_mla_nvfp4", ffi_type="ctypes")
def mla_nvfp4_asm_fwd(
    q_nope: torch.Tensor,
    q_aux: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    params: torch.Tensor,
) -> None: ...


@compile_ops("module_mla_nvfp4", ffi_type="ctypes")
def _mla_nvfp4_quantize_q(
    q: torch.Tensor, q_nope: torch.Tensor, q_aux: torch.Tensor
) -> None: ...


@compile_ops("module_mla_nvfp4", ffi_type="ctypes")
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

    Gains are G_Q=6.72, G_KV=89.6, G_P=2688. This is NOT four numeric
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
    if rebase_threshold not in (0.0, 1.0, 6.0):
        raise ValueError("rebase_threshold must be 0, 1 or 6; use 1 by default")
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
            1.0 / G_P,
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
    """Write BF16 out[B,128,512], optionally return natural-log FP32 LSE.

    Q is uint8 [B,128,256] + [B,128,128]. KV is uint8 [P,16384] +
    [P,8192] in the native shuffled layout, with 256-byte aligned KV bases.
    CSR is contiguous nonnegative GPU int32: indptr[B+1] counts pages,
    indices contains physical page IDs, last[B] is 1..64 (0 for empty rows).
    Params must come from prepare_mla_nvfp4_params with the same capacity.

    Caller guarantees valid CSR, finite live data/scales and per-row length
    <= INT32_MAX. Empty rows produce O=0,LSE=-inf. No split-K, MTP or CP.
    The CO always writes LSE: pass preallocated lse even when return_lse=False
    to avoid allocating scratch. Warm up before capture. All native launches
    use the query device's current stream; callers manage cross-stream lifetime.
    """
    require_gfx1250_asm("mla_decode_nvfp4_fwd")
    if lse is None:
        lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=q_nope.device)
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
    return out, lse if return_lse else None


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
    """
    _mla_nvfp4_update_cache(tiles, page_ids, tile_ids, valid_tokens, kv_nope, kv_aux)


__all__ = [
    "mla_decode_nvfp4_fwd",
    "mla_nvfp4_quantize_q",
    "mla_nvfp4_update_cache",
    "prepare_mla_nvfp4_params",
]
