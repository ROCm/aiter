# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated DeltaNet chunked prefill on gfx1151 (FlyDSL).

Short prompts use FP32 recurrence or one persistent kernel; longer Qwen
batches use parallel U/W preparation followed by a persistent scan. Matrix
paths walk 16-token chunks with FP32 state on chip, avoiding per-chunk state
traffic to HBM. Tiny prompts and weak-decay sequences use FP32 recurrence.

The BF16 pool is updated in place; optional FP32 snapshots retain the
64-token serving contract. Callers gate on gdn_fused_prefill_supported and
keep their multi-kernel path for unsupported operands. max_seqlen is accepted
for compatibility without a performance gate on gfx1151.
"""

from __future__ import annotations

import torch

from .gdn_fused_prefill import (
    _KD,
    _MAX_BYTES,
    _VD,
    _is_gfx1151,
    _span,
    _token_major,
)
from .gdn_fused_prefill import (
    _flat as _flat_nonempty,
)
from .kernels.tensor_shim import _run_compiled


def _flat(t: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Reuse the token-major storage view, also accepting an empty RDNA shard."""
    if t.shape[1] == 0:
        return t.as_strided((0,), (1,), t.storage_offset())
    return _flat_nonempty(t, head_dim)


def gdn_fused_prefill_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    max_seqlen: int | None = None,
) -> bool:
    """Whether gdn_fused_prefill_fwd serves these operands.

    Supported: gfx1151; bf16 q/k [1, T, Hg, 128] and v [1, T, H, 128] with
    contiguous heads (token stride free, e.g. views of a packed projection);
    fp32 g/beta [1, T, H] with unit head stride; a contiguous bf16 state pool
    [slots, H, 128, 128]; H % Hg == 0; q/k/v/g/beta/o spans under 4 GiB
    (32-bit buffer byte offsets).

    Empty token shards are supported. The state pool must span less than 4 GiB.
    max_seqlen is accepted for API compatibility without a performance gate.
    """
    if not (_token_major(q, _KD) and _token_major(k, _KD) and _token_major(v, _VD)):
        return False
    T, Hg = q.shape[1], q.shape[2]
    H = v.shape[2]
    if Hg == 0 or H == 0:
        return False
    tensors = (q, k, v, g, beta, ssm_states)
    return (
        k.shape == q.shape
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
        and ssm_states.numel() * 2 < _MAX_BYTES
        and all(t.is_cuda and t.device == q.device for t in tensors)
        and all(
            (0 if T == 0 else _span(t, d)) * t.element_size() < _MAX_BYTES
            for t, d in ((q, _KD), (k, _KD), (v, _VD), (g, 1), (beta, 1))
        )
        and T * H * _VD * 2 < _MAX_BYTES  # o
        and _is_gfx1151()
    )


def _use_parallel_prefill(T: int, N: int, H: int, Hg: int) -> bool:
    """Amortize the extra launch and U/W traffic on longer Qwen batches.

    The token threshold uses the batch average, avoiding host reads of varlen
    metadata during graph capture. The bound covers each BF16 U/W buffer.
    """
    return (
        Hg == 16
        and H in (32, 48)
        and 1 <= N <= 128
        and T >= N * 512
        and (T // 16 + N) * H * 16 * _KD * 2 < _MAX_BYTES
    )


def _use_recurrent_prefill(T: int, N: int, H: int, Hg: int) -> bool:
    """Select FP32 recurrence for short prompts."""
    if T <= N:
        return True
    if Hg != 16 or H not in (32, 48):
        return False
    return (
        (N <= 4 and T <= 2 * N)
        or (N <= (4 if H == 32 else 2) and T <= 4 * N)
        or (N == 1 and H == 48 and T <= 8)
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
    """Varlen GDN prefill with fused q/k l2norm and BT=64 snapshots.

    Parallel preparation allocates U/W and coefficient workspace; the single
    kernel paths need no scratch. Recurrent and tracked states remain FP32.

    Args:
        q, k: [1, T, Hg, 128] bf16, un-normalized.
        v: [1, T, H, 128] bf16.
        g: [1, T, H] fp32 per-token log decay (natural log).
        beta: [1, T, H] fp32, post-sigmoid.
        ssm_states: [slots, H, V, K] bf16 state pool, read and updated in place.
            Each sequence starts from its slot's current contents unless
            has_initial_state says otherwise, so without that flag the
            slot of a new sequence must be zeroed beforehand.
        cache_indices: [N] pool slot per sequence. Non-negative entries must
            be unique within the batch: sequences sharing a slot race on its
            read-modify-write. Negative entries (padding) are skipped: no pool
            slot or track row is touched and that sequence's rows of o
            are left uninitialized.
            On gfx1151, entries beyond the pool size are also skipped.
        cu_seqlens: [N + 1] token offsets of the packed sequences.
        track_state: optional fp32 [>= N, H, V, K]. Row n receives the state
            before chunk track_chunk_idx[n] of sequence n. Rows with a
            negative or out-of-range chunk index are left untouched. The
            buffer must be contiguous and span less than 4 GiB.
        track_chunk_idx: [N] int chunk index per sequence, required with
            track_state.
        has_initial_state: optional [N] bool (or int) per sequence. False
            rows start from a zero state and ignore their slot's contents; the
            final state is still written to the slot. None loads every slot.
        scale: attention scale, defaults to 128 ** -0.5.
        stream: optional torch.cuda.Stream; defaults to the current stream.

    Returns:
        o: [1, T, H, 128] bf16, freshly allocated. Only the tokens of
        non-padded sequences are written; rows outside
        [cu_seqlens[0], cu_seqlens[-1]) and the rows of padded sequences
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
    if cu_seqlens.dim() != 1 or N < 0:
        raise ValueError("cu_seqlens must be 1-D with at least one entry")
    if cache_indices.dim() != 1:
        raise ValueError("cache_indices must be 1-D")
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
    # Metadata copies and workspace must use the same stream as the kernels.
    with torch.cuda.stream(stream):
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
                or track_state.numel() * 4 >= _MAX_BYTES
            ):
                raise ValueError(
                    "track_state must be contiguous fp32 [>= N, H, 128, 128], got "
                    f"{tuple(track_state.shape)} {track_state.dtype}"
                )
            trs = track_state.view(-1)
            if track_chunk_idx.dim() != 1 or track_chunk_idx.numel() < N:
                raise ValueError(
                    f"track_chunk_idx must be 1-D with at least {N} entries"
                )
            tri = track_chunk_idx[:N].to(torch.int32).contiguous()
        else:
            # Not dereferenced when HAS_TRACK is off.
            trs = ssm_states.view(-1)
            tri = idx

        has_init_flag = has_initial_state is not None
        if has_init_flag:
            if has_initial_state.dim() != 1 or has_initial_state.numel() < N:
                raise ValueError(
                    f"has_initial_state must be 1-D with at least {N} entries"
                )
            init = has_initial_state[:N].to(torch.int32).contiguous()
        else:
            # Not dereferenced when HAS_INIT_FLAG is off.
            init = idx

        o = q.new_empty(1, T, H, _VD)
        if N == 0:
            return o
        if stream is None:
            stream = torch.cuda.current_stream()

        if _use_parallel_prefill(T, N, H, Hg):
            from .kernels.gdr_prefill.gdn_chunk_prefill_rdna import (
                BT,
                compile_prepare_gdn_prefill_rdna,
                compile_scan_gdn_prefill_rdna,
            )

            # Sum ceil(length/BT) <= floor(T/BT)+N for packed, disjoint sequences.
            # The spare slots keep allocation and the grid metadata-only in graphs.
            nc = T // BT + N
            elements = nc * H * BT * _KD
            u, w = q.new_empty(elements), q.new_empty(elements)
            p = g.new_empty(nc * H * BT * BT)
            factors = g.new_empty(nc * H * (2 * BT + 1))
            common = {
                "device_index": q.device.index,
                "H": H,
                "Hg": Hg,
                "N": N,
                "SLOTS": ssm_states.shape[0],
            }
            prepare = compile_prepare_gdn_prefill_rdna(
                scale=_KD**-0.5 if scale is None else float(scale), **common
            )
            _run_compiled(
                prepare,
                _flat(q, _KD),
                _flat(k, _KD),
                _flat(v, _VD),
                _flat(g, 1),
                _flat(beta, 1),
                cu,
                idx,
                u,
                w,
                p,
                factors,
                q.stride(1),
                k.stride(1),
                v.stride(1),
                g.stride(1),
                beta.stride(1),
                T,
                nc * H,
                stream,
            )
            scan = compile_scan_gdn_prefill_rdna(
                HAS_TRACK=has_track,
                HAS_INIT_FLAG=has_init_flag,
                scale=_KD**-0.5 if scale is None else float(scale),
                **common,
            )
            _run_compiled(
                scan,
                _flat(q, _KD),
                cu,
                idx,
                ssm_states.view(-1),
                o.view(-1),
                trs,
                tri,
                init,
                u,
                w,
                _flat(k, _KD),
                p,
                factors,
                _flat(v, _VD),
                _flat(g, 1),
                _flat(beta, 1),
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

        from .kernels.gdr_prefill.gdn_fused_prefill_rdna import (
            compile_gdn_fused_prefill_rdna,
            compile_recurrent_gdn_prefill_rdna,
        )

        compiler = compile_gdn_fused_prefill_rdna
        compile_kwargs = {
            "device_index": q.device.index,
            "SLOTS": ssm_states.shape[0],
        }
        # Overlap output and state updates on long Qwen batches. Dispatch depends
        # only on tensor shapes, so it does not read GPU metadata during capture.
        if _use_recurrent_prefill(T, N, H, Hg):
            compiler = compile_recurrent_gdn_prefill_rdna
        else:
            compile_kwargs["OVERLAP_OUTPUT"] = (
                Hg == 16 and H in (32, 48) and N > 1 and T >= N * 512
            )
        exe = compiler(
            H=H,
            Hg=Hg,
            BV=64,
            HAS_TRACK=has_track,
            HAS_INIT_FLAG=has_init_flag,
            scale=_KD**-0.5 if scale is None else float(scale),
            **compile_kwargs,
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
