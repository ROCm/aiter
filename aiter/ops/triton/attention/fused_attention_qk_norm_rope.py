# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Qwen3-Next attention prologue (gfx950): Q/K norm + partial RoPE.

A single launch replaces the Qwen3-Next full-attention prologue: the per-head
``[Q||gate]`` split, Q/K GemmaRMSNorm (``x*(1+w)``), and partial NeoX RoPE
(rotary_dim 64, reuse_freqs). The op reads the packed ``(rows, 2560)``
projection (4 Q heads + 1 KV head, head_dim 256) and returns
``(query, gate, key, value)``. The KV-cache write is left to the attention
backend (RadixAttention), so this op does not touch the cache.

Dispatch routes by batch size to three schedules selected from an on-device
bake-off across every decode/prefill tier (see the benchmark):

* ``decode`` (Triton m64 tile) for ``rows <= 12288`` -- ~31-33us flat, best
  from one token through the small prefill chunks.
* ``m128_m256`` (Gluon) for ``12288 < rows <= 24576`` -- wins the 16384 chunk.
* ``prefill`` (Gluon) for ``rows > 24576`` -- a large fixed cost that only the
  32768-token batches amortize.

Support is reported by :func:`fused_attention_qk_norm_rope_supported` rather
than asserted, so callers fall back to the native (unfused) Q/K-norm + RoPE
path instead of crashing. The ``decode`` schedule needs only gfx950 (CDNA4
tiles); the mid/large Gluon schedules also need the Gluon CDNA dialect
(Triton 3.7+), so rows above 12288 fall back to native without it.
"""

import functools
from typing import NamedTuple

import torch

_PACKED = 2560
_HEAD_DIM = 256
_QUERY_HEADS = 4
_ROTARY_DIM = 64

# (rows) -> schedule key. Thresholds come from an on-device median-of-60
# crossover sweep on gfx950: the Triton m64 tile wins flat from 1 through
# 12288 (every decode tier and the small prefill chunks), m128_m256 wins from
# 14336 through 24576 (the 16384 chunk: 35us vs m64 43us, prefill 45us), and
# prefill's large fixed cost only wins at 32768 (50us vs m128 71us, m64 93us).
_DECODE_MAX = 12288
_M128_MAX = 24576


@functools.lru_cache(maxsize=1)
def _arch_supported() -> tuple[bool, str]:
    """Probe the GPU arch lazily and fail closed if it cannot be read."""
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001 - any probe failure => unsupported
        return False, f"could not probe GPU arch ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
    """Cached probe: can this Triton compile the mid/large Gluon schedules?

    The tiles use the Gluon CDNA dialect (blocked layouts, gathers, buffer
    ops); probe those imports directly rather than a version number. They
    compile from Triton 3.7 on, so any build that exposes the dialect works.
    """
    try:
        import triton.experimental.gluon  # noqa: F401
        from triton.experimental.gluon.language.amd.cdna3 import (  # noqa: F401
            buffer_load,
        )
    except ImportError as exc:
        return False, f"Gluon CDNA dialect unavailable ({exc})"
    return True, ""


def _select_schedule(rows: int) -> str | None:
    """Return the schedule key for a covered batch ``rows``, else ``None``."""
    if rows < 1:
        return None
    if rows <= _DECODE_MAX:
        return "decode"
    if rows <= _M128_MAX:
        return "m128_m256"
    return "prefill"


def fused_attention_qk_norm_rope_supported(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether this prologue call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``; the reason is meant to be
    logged once by the caller on its native fallback path.
    """
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason

    if (
        not isinstance(projected_qkv_gate, torch.Tensor)
        or not projected_qkv_gate.is_cuda
        or projected_qkv_gate.ndim != 2
    ):
        return False, "projected_qkv_gate must be a 2-D CUDA matrix"
    rows, packed = projected_qkv_gate.shape
    if packed != _PACKED:
        return (
            False,
            f"packed width must be {_PACKED} (4Q+1KV, head_dim 256), got {packed}",
        )
    schedule = _select_schedule(rows)
    if schedule is None:
        return False, f"rows={rows} outside covered range"
    # Only the mid/large schedules are Gluon; decode (Triton) runs without 3.8,
    # so bigger batches fall back to native when the Gluon dialect is absent.
    if schedule != "decode":
        gluon_ok, gluon_reason = _gluon_supported()
        if not gluon_ok:
            return False, gluon_reason
    device = projected_qkv_gate.device
    if projected_qkv_gate.dtype != torch.bfloat16:
        return False, "projected_qkv_gate must be bf16"
    # The kernels index the packed matrix at a fixed row stride, so it must be
    # row-major contiguous.
    if not projected_qkv_gate.is_contiguous():
        return False, "projected_qkv_gate must be contiguous"

    for name, t, shape in (
        ("q_norm_weight", q_norm_weight, (_HEAD_DIM,)),
        ("k_norm_weight", k_norm_weight, (_HEAD_DIM,)),
    ):
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if tuple(t.shape) != shape:
            return False, f"{name} must be {shape}, got {tuple(t.shape)}"
        if not t.is_contiguous():
            return False, f"{name} must be contiguous"

    # The kernels read the table at a fixed stride of 64 (32 cos then 32 sin
    # per position), so it must be contiguous with exactly rotary_dim columns.
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.device != device
        or cos_sin_cache.shape[-1] != _ROTARY_DIM
        or not cos_sin_cache.is_contiguous()
    ):
        return (
            False,
            (
                f"cos_sin_cache must be a contiguous table on {device} with "
                f"exactly {_ROTARY_DIM} columns"
            ),
        )
    if (
        not isinstance(positions, torch.Tensor)
        or positions.device != device
        or positions.ndim != 1
        or positions.shape[0] < rows
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        return (
            False,
            (
                f"positions must be a 1-D int32/int64 CUDA tensor on {device} "
                f"covering {rows} rows"
            ),
        )
    # The m128_m256 Gluon tile reads positions at unit stride; the decode and
    # prefill tiles pass the stride, so they accept a strided view.
    if schedule == "m128_m256" and positions.stride(0) != 1:
        return False, "positions must be contiguous for the m128_m256 schedule"
    return True, ""


def fused_attention_qk_norm_rope(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    rotary_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Qwen3-Next attention prologue in a single launch.

    Args:
        projected_qkv_gate: ``(rows, 2560)`` bf16 packed projection. Per token,
            five 512-wide slabs: Q heads 0..3 each ``[Q(256)||gate(256)]`` then
            the KV slab ``[K(256)||V(256)]``.
        q_norm_weight, k_norm_weight: ``(256,)`` GemmaRMSNorm gains.
        positions: ``(rows,)`` int RoPE positions.
        cos_sin_cache: ``(max_pos, 64)`` table (32 cos then 32 sin per position).
        eps: RMSNorm epsilon. rotary_dim: must be 64.

    Returns ``(query, gate, key, value)``: ``query`` ``(rows,4,256)`` and
    ``key`` ``(rows,1,256)`` are normed+roped bf16; ``gate`` ``(rows,4*256)``
    and ``value`` ``(rows,1,256)`` are raw bf16 views of the projection. The
    caller writes K/V to the attention cache.

    Narrow by design; guard with
    :func:`fused_attention_qk_norm_rope_supported` and fall back to the native
    path when it returns ``False``.
    """
    ok, reason = fused_attention_qk_norm_rope_supported(
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
    )
    if not ok:
        raise ValueError(
            f"fused_attention_qk_norm_rope does not support this call: {reason}"
        )
    if rotary_dim != _ROTARY_DIM:
        raise ValueError(f"rotary_dim must be {_ROTARY_DIM}, got {rotary_dim}")

    schedule = _select_schedule(projected_qkv_gate.shape[0])
    launcher = {
        "decode": _launch_decode,
        "m128_m256": _launch_m128_m256,
        "prefill": _launch_prefill,
    }[schedule]
    return launcher(
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        eps=eps,
    )


# --- host launchers (tile modules stay torch-free; imported lazily here) ---
def _launch_decode(
    projected_qkv_gate,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    *,
    eps,
):
    from aiter.ops.triton._triton_kernels.attention.fused_attention_qk_norm_rope.decode import (
        _decoupled_heads,
    )

    rows, packed_width = projected_qkv_gate.shape
    heads = (packed_width - 512) // 512
    output_stride = (heads + 1) * 256
    qk = torch.empty(
        (rows, heads + 1, 256),
        dtype=projected_qkv_gate.dtype,
        device=projected_qkv_gate.device,
    )
    gate = torch.empty(
        (rows, heads * 256),
        dtype=projected_qkv_gate.dtype,
        device=projected_qkv_gate.device,
    )
    query = qk.as_strided((rows, heads, 256), (output_stride, 256, 1))
    key = qk.as_strided((rows, 1, 256), (output_stride, 256, 1), heads * 256)
    value = projected_qkv_gate[:, heads * 512 + 256 :].view(rows, 1, 256)
    _decoupled_heads[(rows, heads + 6)](
        positions,
        projected_qkv_gate,
        cos_sin_cache,
        eps,
        q_norm_weight,
        k_norm_weight,
        qk,
        gate,
        HEADS=heads,
        POSITION_STRIDE=positions.stride(0),
        num_warps=1,
    )
    return query, gate, key, value


def _launch_m128_m256(
    projected_qkv_gate,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    *,
    eps,
):
    from aiter.ops.triton._gluon_kernels.gfx950.attention.fused_attention_qk_norm_rope.m128_m256 import (
        _attention_prologue,
    )

    rows, packed_width = projected_qkv_gate.shape
    query_heads = packed_width // 512 - 1
    device = projected_qkv_gate.device
    query = torch.empty((rows, query_heads, 256), dtype=torch.bfloat16, device=device)
    gate = torch.empty((rows, query_heads * 256), dtype=torch.bfloat16, device=device)
    key = torch.empty((rows, 1, 256), dtype=torch.bfloat16, device=device)
    value = projected_qkv_gate[:, -256:].view(rows, 1, 256)
    kv_input = projected_qkv_gate[:, -512:]
    # M128 favors one head per wave; M256 reuses rotary data across four Q heads.
    heads_per_wave = 4 if rows >= 256 else 1
    query_groups = query_heads // heads_per_wave
    _attention_prologue[(rows, query_groups + 1)](
        projected_qkv_gate,
        kv_input,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        query,
        gate,
        key,
        eps,
        QUERY_HEADS=query_heads,
        HEADS_PER_WAVE=heads_per_wave,
        num_warps=1,
    )
    return query, gate, key, value


class _PrefillConfig(NamedTuple):
    q_block: int
    k_block: int
    warps: int
    q_vector: int
    k_vector: int
    q_lanes: int
    k_lanes: int
    group: int
    partitions: int
    key_first: bool
    io_policy: int
    store_cache: str
    q_prefetch: int
    k_prefetch: int
    q_aux_stage: int
    k_aux_stage: int
    output_layout: int
    output_gap: int
    waves_per_eu: int


def _small_span(tensor: torch.Tensor) -> bool:
    """Prove signed 32-bit byte addressing, including holes in strided views."""
    return (
        all(s >= 0 for s in tensor.stride())
        and (sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1)
        * tensor.element_size()
        < 2**31
    )


_PREFILL_FIELDS = _PrefillConfig._fields


@functools.lru_cache(maxsize=1)
def _prefill_table() -> dict:
    """Load and cache the tuned gfx950 prefill launch table."""
    from aiter.ops.triton.utils.config_utils import (
        load_config_json,
        resolve_config_dir,
    )

    cfg_dir = resolve_config_dir(
        "attention", "FUSED-ATTENTION-QK-NORM-ROPE", backend="gluon"
    )
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


def _prefill_config(rows) -> _PrefillConfig:
    """Shape-only launch policy; Q blocks count heads, K blocks count tokens.

    The tuned per-bucket values and the I/O policy bit field come from the
    measured gfx950 JSON; see the kernel module for the bit meanings.
    """
    from aiter.ops.triton.utils.config_utils import select_leq_config

    entry = select_leq_config(_prefill_table(), rows, prefix="M_LEQ_")
    return _PrefillConfig(*(entry[field] for field in _PREFILL_FIELDS))


def _allocate_prefill_outputs(packed, output_layout, gap):
    """Allocate planar or token-major Q/gate/K views of one backing buffer."""
    rows, width = packed.shape
    heads = (width - 512) // 512
    if output_layout == 0:
        q_size = rows * heads * 256
        backing = packed.new_empty((q_size * 2 + rows * 256 + 2 * gap,))
        query = backing[:q_size].view(rows, heads, 256)
        gate = backing[q_size + gap : 2 * q_size + gap].view(rows, heads * 256)
        key = backing[2 * q_size + 2 * gap :].view(rows, 1, 256)
    else:
        backing = packed.new_empty((rows, (2 * heads + 1) * 256 + gap))
        query = backing[:, : heads * 256].view(rows, heads, 256)
        gate = backing[:, heads * 256 : 2 * heads * 256]
        key = backing[:, 2 * heads * 256 : (2 * heads + 1) * 256].view(rows, 1, 256)
    value = packed[:, heads * 512 + 256 :].view(rows, 1, 256)
    return query, gate, key, value


def _launch_prefill(
    projected_qkv_gate,
    q_norm_weight,
    k_norm_weight,
    positions,
    cos_sin_cache,
    *,
    eps,
):
    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.attention.fused_attention_qk_norm_rope.prefill import (
        _fused_asymmetric,
    )

    rows, width = projected_qkv_gate.shape
    heads = (width - 512) // 512
    config = _prefill_config(rows)
    query, gate, key, value = _allocate_prefill_outputs(
        projected_qkv_gate, config.output_layout, config.output_gap
    )
    (
        qb,
        kb,
        warps,
        qv,
        kv,
        ql,
        kl,
        group,
        partitions,
        key_first,
        io_policy,
        cache,
        qp,
        kp,
        qe,
        ke,
        _output_layout,
        _gap,
        waves_per_eu,
    ) = config
    packed_small = _small_span(projected_qkv_gate)
    output_small = all(_small_span(t) for t in (query, gate, key))
    positions_small = _small_span(positions)
    table_small = _small_span(cos_sin_cache)
    narrow = packed_small and output_small
    # Keep each proven-safe buffer path; use int64 arithmetic only where the
    # reachable byte span exceeds signed int32.
    if not packed_small:
        io_policy &= ~0x21
    if not output_small:
        io_policy &= ~0x108
    if not positions_small:
        io_policy = (io_policy & ~0x42) | (1 << 19)
    if not table_small:
        io_policy = (io_policy & ~0x84) | (1 << 21)
    programs = triton.cdiv(rows, group) * (group * heads // qb + group // kb)
    _fused_asymmetric[(programs,)](
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        query,
        gate,
        key,
        eps,
        rows,
        heads,
        _ROTARY_DIM,
        positions.stride(0),
        query.stride(0),
        key.stride(0),
        qb,
        kb,
        qv,
        kv,
        ql,
        kl,
        warps,
        group,
        partitions,
        key_first,
        io_policy,
        narrow,
        cache,
        qp,
        kp,
        qe,
        ke,
        num_warps=warps,
        waves_per_eu=waves_per_eu,
    )
    return query, gate, key, value
