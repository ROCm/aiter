# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused DSV4.1 sparse attention, inverse RoPE and group32 MXFP8 on gfx950.

See ``docs/fused_dsv41_mxfp8.md`` for the packed-cache contract and examples.
"""

import math

import torch

from ..jit.core import compile_ops

__all__ = ["fused_dsv41_mxfp8", "fused_dsv41_mxfp8_fwd"]


@compile_ops("module_fused_dsv41_mxfp8", develop=True)
def fused_dsv41_mxfp8_fwd(
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
) -> None:
    """Write caller-allocated outputs; see :func:`fused_dsv41_mxfp8`."""


def fused_dsv41_mxfp8(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(out, out_scale, max_logits, lse)`` in one kernel launch.

    ``q`` is contiguous BF16 ``[T,H,512]``, H in {8,16,32,64,128}, in
    ordinary head-major layout. Set ``q_is_rotated=False`` to apply forward
    RoPE to its final 64 channels. KV rows must already have RoPE applied.

    ``pool`` is contiguous uint8 ``[bytes]`` or ``[bytes,1]``. Each packed
    row has 512 E4M3FN bytes followed by 16 finite E8M0 group32 scales.
    ``prefix`` is int64 packed addresses ``(byte_offset << 1) | 1`` with
    int32 CSR ``prefix_indptr[T+1]``. Addresses are relative to pool.data_ptr;
    the pool base and live row offsets must be 4-byte aligned. Invalid
    addresses/tags are masked. Only live CSR ranges are read.

    Optional ``extend_kv[S,512]`` is BF16 with int32 ``extend`` row indices
    and ``extend_indptr[T+1]`` CSR. Invalid row indices are masked. All three
    tensors must be supplied together. CSR pointers must be monotonic and
    in [0, capacity]; positions (int32/int64 [T]) must be in [0,P).
    ``cos``/``sin`` are BF16 [P,32], using adjacent-pair RoPE.

    Outputs are E4M3FN ``[T,H/8,4096]``, E8M0 ``[T,H/8,128]``, and FP32
    max/LSE ``[T,H]``. Attention is rounded to BF16 before inverse RoPE and
    MXFP8 quantization. ``attn_sink[H]`` (FP32) enters only the softmax
    denominator; max/LSE exclude it. Empty rows return zero output, -inf
    max, and +inf LSE. ``debug_out`` optionally receives the FP32 result
    after inverse RoPE and before quantization, with shape [T,H,512].

    Q and selected KV values must be finite; sink may also be -inf. All
    tensors must be contiguous on q.device. Q, extend_kv, out and debug_out
    need 16-byte alignment. Output buffers must not overlap inputs or one
    another. Warm up before graph capture and reuse preallocated outputs.
    This entry does not allocate a rearrangement or intermediate KV tensor.
    """
    if not isinstance(q, torch.Tensor) or q.device.type != "cuda" or q.ndim != 3:
        raise ValueError("q must be a GPU tensor [T,H,512]")
    t, h, d = q.shape
    if h not in (8, 16, 32, 64, 128) or d != 512:
        raise ValueError("Supported H=8/16/32/64/128 and D=512")
    if not torch.cuda.get_device_properties(q.device).gcnArchName.startswith("gfx950"):
        raise ValueError("fused_dsv41_mxfp8 requires gfx950")
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
    with torch.cuda.device(q.device):
        fused_dsv41_mxfp8_fwd(
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
        )
    return out, out_scale, max_logits, lse
