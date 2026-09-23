# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Context-parallel GDN K5 for long prefills (forward only)."""

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.gdn_segment_scan import (
    _gdn_segment_kernel,
    _gdn_segment_scan_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_K = 128
_V = 128
_BT = 64

# Tuned on gfx950 across the 8k/16k/32k budgets this path is gated to; see the
# sweep in op_tests/op_benchmarks/triton/bench_gdn_segment_scan.py.
_TARGET_SEGMENTS = 16
_SEGMENT_BV = 64
_SEGMENT_WARPS = 4
_SCAN_BV = 16
_SCAN_WARPS = 4


@functools.lru_cache(maxsize=64)
def _build_segments(
    seq_lens: tuple[int, ...],
    chunks_per_segment: int,
    device: torch.device,
):
    chunk_base: list[int] = []
    nchunks: list[int] = []
    tok_base: list[int] = []
    tok_end: list[int] = []
    seq_id: list[int] = []
    is_last: list[int] = []
    seq_seg_offsets = [0]
    global_chunk = 0
    global_token = 0
    for i, length in enumerate(seq_lens):
        n_chunks = triton.cdiv(length, _BT)
        n_segments = max(1, triton.cdiv(n_chunks, chunks_per_segment))
        for s in range(n_segments):
            c0 = s * chunks_per_segment
            count = min(chunks_per_segment, n_chunks - c0)
            chunk_base.append(global_chunk + c0)
            nchunks.append(count)
            tok_base.append(global_token + c0 * _BT)
            tok_end.append(
                min(global_token + (c0 + count) * _BT, global_token + length)
            )
            seq_id.append(i)
            is_last.append(1 if s == n_segments - 1 else 0)
        seq_seg_offsets.append(len(chunk_base))
        global_chunk += n_chunks
        global_token += length
    # Pad the rows to a multiple of four int32 so every descriptor pointer is
    # 16-byte aligned. Triton specialises pointer arguments on that alignment,
    # so an unpadded stride silently compiles a second variant of the kernel
    # whenever the segment count is not a multiple of four -- chosen by batch
    # shape, at the cost of a recompile. The grid never reaches the padding.
    pad = [0] * (triton.cdiv(len(chunk_base), 4) * 4 - len(chunk_base))
    desc = torch.tensor(
        [
            row + pad
            for row in (chunk_base, nchunks, tok_base, tok_end, seq_id, is_last)
        ],
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.tensor(seq_seg_offsets, dtype=torch.int32, device=device)
    return (
        desc,
        offsets,
        len(chunk_base),
        max(
            seq_seg_offsets[i + 1] - seq_seg_offsets[i]
            for i in range(len(seq_seg_offsets) - 1)
        ),
    )


def gdn_segment_scan_fwd(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    seq_lens: tuple[int, ...],
    state_indices: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    snapshot_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.float32,
    chunks_per_segment: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    r"""
    Segmented affine-scan GDN K5 hidden-state pass (forward only).

    Produces the same outputs as the serial K5 chunk pass, but recovers
    occupancy on long, narrow prefills: each sequence is cut into segments, the
    affine map ``h' = h @ A + b`` of each segment is extracted in one launch, a
    short scan composes those maps into per-segment incoming states, and every
    segment is then replayed in parallel from its true state.

    Args:
        k (torch.Tensor): keys of shape `[1, T_flat, HG, 128]`.
        w (torch.Tensor): WY-transform `w` of shape `[1, H, T_flat, 128]`.
        u (torch.Tensor): WY-transform `u` of shape `[1, H, T_flat, 128]`.
        g (torch.Tensor): head-major, log2(e)-scaled cumulative gate of shape
            `[1, H, T_flat]`.
        initial_state (torch.Tensor | None): entry state, either dense
            `[N, H, 128, 128]` or a pool addressed by `state_indices`. `None`
            starts every sequence from zero.
        output_final_state (bool): write each sequence's final state.
        seq_lens (tuple[int, ...]): host-side length of each packed sequence;
            must sum to `T_flat`.
        state_indices (torch.Tensor | None): int32 pool slot per sequence.
            Required when `initial_state` is a pool rather than dense.
        inplace_final_state (bool): write the final state into
            `initial_state` instead of a fresh tensor.
        snapshot_dtype (torch.dtype): dtype of the per-chunk `h` snapshots.
        state_dtype (torch.dtype): dtype of the final state.
        chunks_per_segment (int | None): segment size in 64-token chunks.
            `None` targets 16 segments per batch.

    Returns:
        `(h_snapshots, v_new, final_state)` with the same layout and contract
        as the serial K5 pass. `final_state` is `None` unless
        `output_final_state`.

    Special considerations:
        Only the narrow production GDN shape is supported: `B=1`,
        `K=V=128`, head-major log2-scaled `g`. Callers that do not meet those
        constraints must use the serial K5 path.
    """
    B, _T, HG, K = k.shape
    H, T_flat, V = w.shape[1], w.shape[2], u.shape[-1]
    if (B, K, V) != (1, _K, _V):
        raise ValueError(
            f"GDN segment scan requires B=1,K=V=128; got B={B},K={K},V={V}."
        )
    if g.shape != (B, H, T_flat):
        raise ValueError(f"GDN segment scan needs head-major g; got {tuple(g.shape)}.")
    seq_lens = tuple(int(length) for length in seq_lens)
    if sum(seq_lens) != T_flat:
        raise ValueError(f"seq_lens sum to {sum(seq_lens)}, expected {T_flat}.")

    chunks_per_seq = tuple(triton.cdiv(length, _BT) for length in seq_lens)
    max_chunks = max(chunks_per_seq)
    total_chunks = sum(chunks_per_seq)
    if chunks_per_segment is None:
        # Across 8k/16k/32k token budgets and N=1..3, the optimum keeps roughly
        # 16 segments in flight. This scales from 8 chunks/segment at 8k tokens
        # through 32 chunks/segment at 32k tokens.
        chunks_per_segment = triton.cdiv(total_chunks, _TARGET_SEGMENTS)
    chunks_per_segment = min(max_chunks, max(1, chunks_per_segment))

    desc, seq_seg_offsets, num_segments, max_segments = _build_segments(
        seq_lens, chunks_per_segment, k.device
    )
    _LOGGER.info(
        "gdn_segment_scan_fwd: seq_lens=%s total_chunks=%s chunks_per_segment=%s "
        "num_segments=%s max_segments=%s",
        seq_lens,
        total_chunks,
        chunks_per_segment,
        num_segments,
        max_segments,
    )
    seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq, seg_is_last = desc
    # The regular K5 output contract.
    snapshots = torch.empty(
        B, total_chunks, H, V, K, dtype=snapshot_dtype, device=k.device
    )
    v_new = torch.empty(B, H, T_flat, V, dtype=u.dtype, device=k.device)
    if output_final_state:
        final_state = (
            initial_state
            if inplace_final_state
            else torch.empty(len(seq_lens), H, V, K, dtype=state_dtype, device=k.device)
        )
    else:
        final_state = None

    # Segmentation has no benefit when the sequence is already short.
    if max_segments == 1:
        # One segment per sequence, so segment id equals sequence id and the
        # incoming state is just the entry state. The replay kernel addresses
        # h_in by segment, so an indexed pool has to be gathered first --
        # feeding it the pool directly would read slot i for sequence i.
        if initial_state is not None and state_indices is not None:
            h_in = initial_state.index_select(0, state_indices)
        else:
            h_in = initial_state
    else:
        b_seg = torch.empty(num_segments, H, V, K, dtype=torch.float32, device=k.device)
        a_seg = torch.empty(
            num_segments, H, V, K, dtype=torch.bfloat16, device=k.device
        )
        # One launch extracts both summaries: start at h=0 with u for b_seg, and
        # at h=I with u=0 for a_seg.
        _gdn_segment_kernel[(triton.cdiv(V, _SEGMENT_BV), num_segments * H)](
            k=k,
            u=u,
            w=w,
            g=g,
            h_in=None,
            h_out=b_seg,
            a_out=a_seg,
            h_snapshots=None,
            v_new=None,
            final_state=None,
            state_indices=state_indices,
            seg_chunk_base=seg_chunk_base,
            seg_nchunks=seg_nchunks,
            seg_tok_base=seg_tok_base,
            seg_tok_end=seg_tok_end,
            seg_seq=seg_seq,
            seg_is_last=seg_is_last,
            T_FLAT=T_flat,
            H=H,
            HG=HG,
            K=K,
            V=V,
            BV=_SEGMENT_BV,
            DUAL_SUMMARY=True,
            HAS_H_IN=False,
            WRITE_OUTPUTS=False,
            STORE_H_OUT=True,
            STORE_FINAL=False,
            USE_STATE_INDICES=state_indices is not None,
            STATE_BF16=state_dtype is torch.bfloat16,
            num_warps=_SEGMENT_WARPS,
            num_stages=1,
        )
        h_in = torch.empty(num_segments, H, V, K, dtype=torch.float32, device=k.device)
        _gdn_segment_scan_kernel[(triton.cdiv(V, _SCAN_BV), len(seq_lens) * H)](
            a_seg=a_seg,
            b_seg=b_seg,
            h_in=h_in,
            h0=initial_state,
            state_indices=state_indices,
            seq_seg_offsets=seq_seg_offsets,
            H=H,
            K=K,
            V=V,
            BV=_SCAN_BV,
            HAS_H0=initial_state is not None,
            USE_STATE_INDICES=state_indices is not None,
            num_warps=_SCAN_WARPS,
            num_stages=1,
        )

    _gdn_segment_kernel[(triton.cdiv(V, _SEGMENT_BV), num_segments * H)](
        k=k,
        u=u,
        w=w,
        g=g,
        h_in=h_in,
        h_out=None,
        a_out=None,
        h_snapshots=snapshots,
        v_new=v_new,
        final_state=final_state,
        state_indices=state_indices,
        seg_chunk_base=seg_chunk_base,
        seg_nchunks=seg_nchunks,
        seg_tok_base=seg_tok_base,
        seg_tok_end=seg_tok_end,
        seg_seq=seg_seq,
        seg_is_last=seg_is_last,
        T_FLAT=T_flat,
        H=H,
        HG=HG,
        K=K,
        V=V,
        BV=_SEGMENT_BV,
        DUAL_SUMMARY=False,
        HAS_H_IN=h_in is not None,
        WRITE_OUTPUTS=True,
        STORE_H_OUT=False,
        STORE_FINAL=output_final_state,
        USE_STATE_INDICES=state_indices is not None,
        STATE_BF16=state_dtype is torch.bfloat16,
        num_warps=_SEGMENT_WARPS,
        num_stages=1,
    )
    return snapshots, v_new, final_state


__all__ = ["gdn_segment_scan_fwd"]
