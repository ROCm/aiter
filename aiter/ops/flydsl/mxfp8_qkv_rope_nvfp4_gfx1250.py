# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MiniMax-M3 decode QKV chain: the native-MXFP8 QKV GEMM, the q/k Gemma
RMS norm + partial RoPE, bf16 q / index-q and the NVFP4 K / V / index-K row
inserts in one launch (kernels/mxfp8_qkv_rope_nvfp4_gfx1250.py).

The bytes are those of ``gemm_a8w8_blockscale(isBpreshuffled=True)`` followed by
``fused_qknorm_idxrqknorm`` (NVFP4 row-layout K/V, NVFP4 index rows) on the same
pre-norm rows: the tail reproduces fused_qknorm_idxrqknorm bit for bit, the GEMM
rows differ from the tuned GEMM's only by its FP32 K order (~1e-4 of the bf16
values flip one ulp).
"""

from __future__ import annotations

import functools
import struct

import torch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

D = 128
NQ, NKV, ROT = 16, 1, 64
K_DIM = 6144
# (max rows, config) per number of index-q heads; config = (tile_m, ns, ks, kw,
# wm, wn, depth, astage, wcpol, owner_keeps). Graph-replay sweeps on MI455X,
# MiniMax-M3.1 TP4 shapes, cold weights (bench_qkv_chain.py / validate_k3.py). A
# cluster is one head's ns x ks workgroups; head-aligned clusters with ns = 1 take a
# whole head (128 columns) a workgroup and keep the cluster count <= the concurrent
# cluster slots (16 shader engines x 16 // (ns * ks)), so every cluster is resident.
_CHAIN_ROWS = {
    # niq: [(max rows, cfg)]. Cold-weight graph replay on MI455X (24 weight
    # copies): non-temporal weight loads (wcpol 1) everywhere; one m-tile up to 96
    # rows (65-96: 6 K-waves, owner-kept rows), 64-row tiles past that; from 49
    # rows the activation slab goes to LDS by TDM (astage 2). Up to 16 rows a head
    # is 4 column-split x 4 K-split workgroups (clusters of 16, 32 columns a
    # workgroup): -0.6..-0.9 us a call against ns 2 at 1-16 rows (same K order, same
    # bytes).
    1: [
        (16, (16, 4, 4, 6, 1, 2, 2, 1, 1, 0)),
        (32, (32, 2, 4, 6, 1, 2, 2, 1, 1, 0)),
        (48, (48, 2, 4, 6, 1, 2, 2, 1, 1, 0)),
        (64, (64, 2, 4, 6, 1, 2, 2, 2, 1, 0)),
        (80, (80, 2, 4, 6, 1, 2, 2, 2, 1, 1)),
        (96, (96, 2, 4, 6, 1, 2, 2, 2, 1, 1)),
        (192, (64, 2, 4, 4, 1, 2, 2, 2, 1, 1)),
    ],
    # indexer CP (4 index-q heads, 23 heads): past 128 rows the 69 clusters of the
    # 64-row tiles outrun the shader engines and the unfused chain is as fast.
    4: [
        (16, (16, 4, 4, 6, 1, 2, 2, 1, 1, 0)),
        (32, (32, 2, 4, 6, 1, 2, 2, 1, 1, 0)),
        (48, (48, 2, 4, 6, 1, 2, 2, 1, 1, 0)),
        (64, (64, 2, 4, 6, 1, 2, 2, 2, 1, 0)),
        (80, (80, 2, 4, 6, 1, 2, 2, 2, 1, 1)),
        (96, (96, 2, 4, 6, 1, 2, 2, 2, 1, 1)),
        (128, (64, 2, 4, 4, 1, 2, 2, 2, 1, 1)),
    ],
}
MAX_TOKENS = max(r for rows in _CHAIN_ROWS.values() for r, _ in rows)


def mxfp8_qkv_chain_config(m: int, num_index_heads: int = 1):
    """The chain config for ``m`` tokens (10 fields), or None where the unfused GEMM +
    fused_qknorm_idxrqknorm keeps the call."""
    for max_rows, cfg in _CHAIN_ROWS.get(num_index_heads, ()):
        if 0 < m <= max_rows:
            return cfg
    return None


def mxfp8_qkv_chain_supported(
    m: int, num_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int,
    hidden_size: int, num_index_heads: int = 1,
) -> bool:  # fmt: skip
    """Whether the fused chain takes this call (gfx1250, the M3 TP4 head shape)."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return (
        get_gfx_runtime() == "gfx1250"
        and mxfp8_qkv_chain_config(m, num_index_heads) is not None
        and (num_heads, num_kv_heads, head_dim, rotary_dim, hidden_size)
        == (NQ, NKV, D, ROT, K_DIM)
    )


@functools.cache
def _launcher():
    from .kernels.mxfp8_qkv_rope_nvfp4_gfx1250 import (
        launch_mxfp8_qkv_norm_rope_nvfp4_insert,
    )

    return launch_mxfp8_qkv_norm_rope_nvfp4_insert


@capture_scoped_cache
def _slab_cached(device: torch.device, stream_id: int, numel: int) -> torch.Tensor:
    """FP32 split partial planes, one per (device, stream); a capture gets its own."""
    del stream_id
    with persistent_alloc(device):
        return torch.empty(numel, dtype=torch.float32, device=device)


def _slab(device, numel):
    bucket = 1 << max(20, (numel - 1).bit_length())  # token counts share a bucket
    return _slab_cached(device, torch.cuda.current_stream(device).cuda_stream, bucket)


def mxfp8_qkv_norm_rope_nvfp4_insert(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    index_cache: torch.Tensor,
    index_scale: torch.Tensor,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    eps: float,
    *,
    num_index_heads: int = 1,
    qkv_out: torch.Tensor | None = None,
    config: tuple | None = None,
) -> None:
    """One launch of the QKV chain.

    Args:
        x, x_scale: e4m3 [M, 6144] (unit column stride, 16-byte row stride) and its
            row-major e8m0 [M, 192] (the AR + RMSNorm + MXFP8 producer's output).
        w, w_scale: the (16, 16)-preshuffled e4m3 [N, 6144] QKV weight, N =
            (16 + 2 + num_index_heads + 1) * 128 ([q | k | v | index-q | index-k]),
            and its 32x4-shuffled e8m0 [N, 192] (shuffle_mxfp8_bpreshuffle_scale).
        cos_sin_cache: bf16 [max_pos, 64] ([cos | sin], rotary 64); positions,
            slot_mapping, index_slot_mapping: int64 [M] (slot -1 skips the token).
        k_cache, v_cache: u8 [pages, 1, 128, 64] NVFP4 payloads; k_scale, v_scale:
            their E4M3 planes [pages, 1, 128, 8] (u8 / e4m3 views, shared strides).
        index_cache, index_scale: u8 [pages, 128, 64] and [pages, 128, 8].
        q_out: bf16 [M, 2048]; index_q_out: bf16 [M, num_index_heads * 128].
        qkv_out: optional bf16 [M, N], the reduced pre-norm rows (for checks).
    """
    from .kernels.tensor_shim import _run_compiled, ptr_arg
    import flydsl.expr as fx

    m = x.shape[0]
    if m == 0:
        return
    niq = num_index_heads
    n = (NQ + 2 * NKV + niq + 1) * D
    cfg = config or mxfp8_qkv_chain_config(m, num_index_heads)
    if cfg is None:
        raise ValueError(f"mxfp8_qkv_norm_rope_nvfp4_insert: no config for M={m}")
    if (
        tuple(w.shape) != (n, K_DIM)
        or x.shape[1] != K_DIM
        or x.stride(1) != 1
        or x.stride(0) % 16
        or tuple(x_scale.shape) != (m, K_DIM // 32)
        or tuple(w_scale.shape) != (n, K_DIM // 32)
        or not (
            x_scale.is_contiguous() and w_scale.is_contiguous() and w.is_contiguous()
        )
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: operand shapes / strides")
    if (
        k_cache.shape[1:] != (NKV, 128, D // 2)
        or k_scale.shape[1:] != (NKV, 128, D // 16)
        or k_cache.stride() != v_cache.stride()
        or k_scale.stride() != v_scale.stride()
        or k_cache.stride()[2:] != (D // 2, 1)
        or k_scale.stride()[2:] != (D // 16, 1)
        or index_cache.shape[1:] != (128, D // 2)
        or index_scale.shape[1:] != (128, D // 16)
        or index_cache.stride()[1:] != (D // 2, 1)
        or index_scale.stride()[1:] != (D // 16, 1)
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: NVFP4 row-layout caches")
    if (
        q_out.dtype != torch.bfloat16
        or not q_out.is_contiguous()
        or tuple(q_out.shape) != (m, NQ * D)
        or index_q_out.dtype != torch.bfloat16
        or not index_q_out.is_contiguous()
        or tuple(index_q_out.shape) != (m, niq * D)
        or cos_sin_cache.shape[1] != ROT
        or not cos_sin_cache.is_contiguous()
        or eps < 2.0**-126  # v_rsq_f32 without ocml's denormal scaling
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: outputs / rope / eps")
    cfg = tuple(cfg) + (1, 0, 0)[len(cfg) - 7 :] if len(cfg) < 10 else tuple(cfg)
    tile_m, ns, ks, kw, wm, wn, depth, astage, wcpol, owner_keeps = cfg
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(x.device, ks * mpad * n)
    write_qkv = qkv_out is not None
    eps_bits = struct.unpack("<i", struct.pack("<f", float(eps)))[0]
    stream = torch.cuda.current_stream(x.device)
    spec = (
        tile_m,
        ns,
        ks,
        kw,
        wm,
        wn,
        depth,
        niq,
        write_qkv,
        astage,
        wcpol,
        owner_keeps,
    )
    _run_compiled(
        _launcher(),
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(slab),
        ptr_arg(q_out), ptr_arg(index_q_out), ptr_arg(qkv_out if write_qkv else q_out),
        ptr_arg(q_norm_weight), ptr_arg(k_norm_weight), ptr_arg(index_q_norm_weight),
        ptr_arg(index_k_norm_weight), ptr_arg(cos_sin_cache), ptr_arg(positions),
        ptr_arg(slot_mapping), ptr_arg(index_slot_mapping), ptr_arg(k_cache),
        ptr_arg(v_cache), ptr_arg(k_scale), ptr_arg(v_scale), ptr_arg(index_cache),
        ptr_arg(index_scale),
        m, x.stride(0), mpad, cos_sin_cache.shape[0],
        k_cache.stride(0), k_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
        index_cache.stride(0), index_scale.stride(0), eps_bits,
        fx.Stream(stream),
        *spec,
        specialization_key=("mxfp8_qkv_chain", x.device.index) + spec,
    )  # fmt: skip
