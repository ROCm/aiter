# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-kernel Gated DeltaNet chunked prefill on gfx950 (FlyDSL).

One launch replaces the l2norm, prepare, hidden-state and output stages of the
chunked prefill: every workgroup keeps its slice of the recurrent state on chip
across the sequence's 64-token chunks, so the per-chunk states ``h`` never
reach HBM. The pool state is updated in place. The kernel can also write the
fp32 state at one requested chunk boundary per sequence, which serving stacks
use to snapshot the recurrent state for prefix caching.

Callers gate on :func:`gdn_fused_prefill_supported` (passing ``max_seqlen`` so
it can also reject batches the multi-kernel path runs faster) and keep their
multi-kernel path wherever it returns False.
"""

from __future__ import annotations

import functools

import torch

from .kernels.gdr_prefill import compile_gdn_fused_prefill
from .kernels.tensor_shim import _run_compiled

_KD = 128
_VD = 128
# Buffer loads/stores take a 32-bit byte offset from each workgroup's base, so
# every operand's span must stay below 2^32 bytes. That also keeps the 32-bit
# element offsets and token strides the launch ABI passes below 2^31.
_MAX_BYTES = 2**32


@functools.cache
def _is_gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx950"
    except Exception:  # noqa: BLE001
        return False


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _pick_bv(N: int, H: int, device: torch.device) -> int:
    """V slice per workgroup.

    LDS allows one workgroup per CU, and each walks its sequence's chunks
    serially. When the BV=16 grid still fits in one pass over the CUs, the
    narrower slice shortens every chunk step; otherwise BV=32 recomputes the
    V-independent work half as often.
    """
    return 16 if (_VD // 16) * N * H <= _num_cus(device) else 32


def _span(t: torch.Tensor, head_dim: int) -> int:
    """Elements from ``t``'s first element through its last [T, heads, dim] row.

    ``head_dim=1`` covers the [1, T, H] g / beta tensors (unit head stride).
    """
    T, heads = t.shape[1], t.shape[2]
    return (T - 1) * t.stride(1) + (heads - 1) * t.stride(2) + head_dim


def _flat(t: torch.Tensor, head_dim: int) -> torch.Tensor:
    """1-D view over ``t``'s storage span; the kernel applies the strides."""
    return t.as_strided((_span(t, head_dim),), (1,), t.storage_offset())


def _token_major(t: torch.Tensor, head_dim: int) -> bool:
    """[1, T, heads, head_dim] with contiguous heads; any token stride."""
    return (
        t.dim() == 4
        and t.shape[0] == 1
        and t.shape[3] == head_dim
        and t.stride(3) == 1
        and t.stride(2) == head_dim
        and t.stride(1) >= t.shape[2] * head_dim
    )


def gdn_fused_prefill_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    max_seqlen: int | None = None,
) -> bool:
    """Whether :func:`gdn_fused_prefill_fwd` serves these operands.

    Supported: gfx950; bf16 q/k [1, T, Hg, 128] and v [1, T, H, 128] with
    contiguous heads (token stride free, e.g. views of a packed projection);
    fp32 g/beta [1, T, H] with unit head stride; a contiguous bf16 state pool
    [slots, H, 128, 128]; H % Hg == 0; q/k/v/g/beta/o spans under 4 GiB
    (32-bit buffer byte offsets).

    With ``max_seqlen`` (the longest sequence in the batch) it also returns
    False where the multi-kernel path is expected to be faster, i.e. unless
    ``max_seqlen <= 1024 or H * T >= 24 * max_seqlen``.
    """
    if not (_token_major(q, _KD) and _token_major(k, _KD) and _token_major(v, _VD)):
        return False
    T, Hg = q.shape[1], q.shape[2]
    H = v.shape[2]
    # The fused kernel walks each sequence's chunks serially at ~2.5 us per
    # chunk whatever H is; the multi-kernel path spreads all chunks over the
    # GPU and serializes only its state scan. Thresholds measured on MI355X
    # against chunk_gated_delta_rule_opt_vk (op_tests/test_gdn_fused_prefill.py).
    if max_seqlen is not None and not (max_seqlen <= 1024 or H * T >= 24 * max_seqlen):
        return False
    tensors = (q, k, v, g, beta, ssm_states)
    return (
        T > 0
        and k.shape == q.shape
        and v.shape[1] == T
        and H % Hg == 0
        and q.dtype is torch.bfloat16
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and g.dtype is torch.float32
        and beta.dtype is torch.float32
        and ssm_states.dtype is torch.bfloat16
        and g.shape == (1, T, H)
        and beta.shape == (1, T, H)
        and g.stride(2) == 1
        and beta.stride(2) == 1
        and ssm_states.dim() == 4
        and tuple(ssm_states.shape[1:]) == (H, _VD, _KD)
        and ssm_states.is_contiguous()
        and all(t.is_cuda and t.device == q.device for t in tensors)
        and all(
            _span(t, d) * t.element_size() < _MAX_BYTES
            for t, d in ((q, _KD), (k, _KD), (v, _VD), (g, 1), (beta, 1))
        )
        and T * H * _VD * 2 < _MAX_BYTES  # o
        and _is_gfx950()
    )


def gdn_fused_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    track_state: torch.Tensor | None = None,
    track_chunk_idx: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    stream=None,
) -> torch.Tensor:
    """Varlen GDN chunked prefill (BT=64) with in-kernel q/k l2norm.

    Args:
        q, k: [1, T, Hg, 128] bf16, un-normalized.
        v: [1, T, H, 128] bf16.
        g: [1, T, H] fp32 per-token log decay (natural log).
        beta: [1, T, H] fp32, post-sigmoid.
        ssm_states: [slots, H, V, K] bf16 state pool, read and updated in place.
            Each sequence starts from its slot's current contents unless
            ``has_initial_state`` says otherwise, so without that flag the
            slot of a new sequence must be zeroed beforehand.
        cache_indices: [N] pool slot per sequence. Non-negative entries must
            be unique within the batch: sequences sharing a slot race on its
            read-modify-write. Negative entries (padding) are skipped: no pool
            slot or track row is touched and that sequence's rows of ``o``
            are left uninitialized.
        cu_seqlens: [N + 1] token offsets of the packed sequences.
        track_state: optional fp32 [>= N, H, V, K]. Row n receives the state
            before chunk ``track_chunk_idx[n]`` of sequence n. Rows with a
            negative or out-of-range chunk index are left untouched.
        track_chunk_idx: [N] int chunk index per sequence, required with
            ``track_state``.
        has_initial_state: optional [N] bool (or int) per sequence. False
            rows start from a zero state and ignore their slot's contents; the
            final state is still written to the slot. None loads every slot.
        scale: attention scale, defaults to 128 ** -0.5.
        stream: launch stream; defaults to the current stream.

    Returns:
        o: [1, T, H, 128] bf16, freshly allocated. Only the tokens of
        non-padded sequences are written; rows outside
        ``[cu_seqlens[0], cu_seqlens[-1])`` and the rows of padded sequences
        are left uninitialized.
    """
    if not gdn_fused_prefill_supported(q, k, v, g, beta, ssm_states):
        raise ValueError(
            "gdn_fused_prefill_fwd: unsupported operands "
            f"q={tuple(q.shape)} {q.dtype}, v={tuple(v.shape)} {v.dtype}, "
            f"g={g.dtype}, beta={beta.dtype}, ssm_states={tuple(ssm_states.shape)} "
            f"{ssm_states.dtype}; gate on gdn_fused_prefill_supported."
        )
    T, Hg = q.shape[1], q.shape[2]
    H = v.shape[2]
    N = cu_seqlens.numel() - 1
    if cache_indices.numel() < N:
        raise ValueError(
            f"cache_indices has {cache_indices.numel()} entries for {N} sequences"
        )
    for name, tensor in (
        ("cu_seqlens", cu_seqlens),
        ("cache_indices", cache_indices),
        ("track_state", track_state),
        ("track_chunk_idx", track_chunk_idx if track_state is not None else None),
        ("has_initial_state", has_initial_state),
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
    cu = cu_seqlens.to(torch.int32).contiguous()
    idx = cache_indices[:N].to(torch.int32).contiguous()

    has_track = track_state is not None
    if has_track:
        if track_chunk_idx is None:
            raise ValueError("track_state needs track_chunk_idx")
        if (
            track_state.dtype is not torch.float32
            or not track_state.is_contiguous()
            or track_state.shape[0] < N
            or tuple(track_state.shape[1:]) != (H, _VD, _KD)
        ):
            raise ValueError(
                "track_state must be contiguous fp32 [>= N, H, 128, 128], got "
                f"{tuple(track_state.shape)} {track_state.dtype}"
            )
        trs = track_state.view(-1)
        if track_chunk_idx.dim() != 1 or track_chunk_idx.numel() < N:
            raise ValueError(f"track_chunk_idx must be 1-D with at least {N} entries")
        tri = track_chunk_idx[:N].to(torch.int32).contiguous()
    else:
        # Not dereferenced when HAS_TRACK is off.
        trs = ssm_states.view(-1)
        tri = idx

    has_init_flag = has_initial_state is not None
    if has_init_flag:
        if has_initial_state.dim() != 1 or has_initial_state.numel() < N:
            raise ValueError(f"has_initial_state must be 1-D with at least {N} entries")
        init = has_initial_state[:N].to(torch.int32).contiguous()
    else:
        # Not dereferenced when HAS_INIT_FLAG is off.
        init = idx

    o = q.new_empty(1, T, H, _VD)
    if N == 0:
        return o
    if stream is None:
        stream = torch.cuda.current_stream()

    exe = compile_gdn_fused_prefill(
        H=H,
        Hg=Hg,
        BV=_pick_bv(N, H, q.device),
        HAS_TRACK=has_track,
        HAS_INIT_FLAG=has_init_flag,
        scale=_KD**-0.5 if scale is None else float(scale),
    )
    _run_compiled(
        exe,
        _flat(q, _KD),
        _flat(k, _KD),
        _flat(v, _VD),
        _flat(g, 1),
        _flat(beta, 1),
        cu,
        idx,
        ssm_states.view(-1),
        o.view(-1),
        trs,
        tri,
        init,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        g.stride(1),
        beta.stride(1),
        T,
        N * H,
        stream,
    )
    return o


__all__ = ["gdn_fused_prefill_fwd", "gdn_fused_prefill_supported"]
