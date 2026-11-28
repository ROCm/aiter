# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant KDA chunk forward (prefill), Gluon kernels (gfx950).

A token's output bits do not depend on the batch, the packing of sequences, or where a
sequence is cut at a chunk boundary: chunks are 64 rows aligned to each sequence start,
tiles and launch configs are fixed, every sum has a fixed order and the state is FP32.
"""

import functools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.batch_invariant.kda_bi import (
    _kda_bi_chunk_table_kernel,
    _kda_bi_intra_kernel,
    _kda_bi_out_kernel,
    _kda_bi_seq_chunks_kernel,
    _kda_bi_seq_order_kernel,
    _kda_bi_state_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = ["BI_CONTRACT_VERSION", "kda_bi_fwd"]

# Bump when a change can change the output bits; callers compare it at startup.
BI_CONTRACT_VERSION = 2
CHUNK = 64
HEAD_DIM = 128
_LAUNCH = ("num_warps",)


@functools.cache
def _config(name: str) -> dict:
    """Pinned launch values of one kernel (constexpr names upper case)."""
    cfg = load_config_json(
        f"{resolve_config_dir('attention', 'KDA-BI', backend='gluon')}/DEFAULT.json",
        required=False,
    )
    if not cfg or name not in cfg:
        raise NotImplementedError(f"kda_bi: no pinned config '{name}' for this arch")
    return {k if k in _LAUNCH else k.upper(): v for k, v in cfg[name].items()}


def _state_bv(n_seq_heads: int) -> int:
    """V block of the state pass: the widest one that still gives min_programs programs
    (packed batches), else the narrowest (long sequences). The output bits do not depend
    on it."""
    grid = _config("state_grid")
    choices = sorted(grid["BV_CHOICES"])
    fits = [
        bv for bv in choices if n_seq_heads * (HEAD_DIM // bv) >= grid["MIN_PROGRAMS"]
    ]
    return fits[-1] if fits else choices[0]


def _fuse_out(n_seq_heads: int) -> bool:
    """Compute the output inside the state pass (no bf16 state copies, one launch less)
    when the state grid is wide enough; the bits are the same either way."""
    return n_seq_heads >= _config("state_grid")["FUSE_OUT_MIN_SEQ_HEADS"]


def kda_bi_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    output_boundary_state: bool = False,
):
    """KDA chunk forward with in-kernel q/k L2 norm and the lower-bound gate.

    q, k, v, g: bf16 [1, T, H, 128] (g is the raw gate input); beta: [1, T, H], activated;
    A_log: [H]; dt_bias: [H * 128]; cu_seqlens: [N + 1]; initial_state: fp32 [N, H, 128, 128].
    With cu_seqlens_cpu (same values on the host) the call does not sync and can be captured.

    Returns (o bf16 [1, T, H, 128], final_state fp32 [N, H, 128, 128] or None), plus the state
    after each sequence's last complete chunk when output_boundary_state."""
    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(
            "kda_bi: q must be [1, T, H, 128] (pack sequences with cu_seqlens)"
        )
    _, T, H, Dk = q.shape
    if Dk != HEAD_DIM:
        raise ValueError(f"kda_bi: head dim must be {HEAD_DIM}, got {Dk}")
    for name, t in (("k", k), ("v", v), ("g", g)):
        if t.shape != q.shape:
            raise ValueError(
                f"kda_bi: {name} shape {tuple(t.shape)} != q {tuple(q.shape)}"
            )
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g)):
        if t.dtype != torch.bfloat16:
            raise ValueError(f"kda_bi: {name} must be bf16, got {t.dtype}")
    if beta.shape != (1, T, H) or beta.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("kda_bi: beta must be bf16 or fp32 [1, T, H]")
    if A_log.numel() != H or dt_bias.numel() != H * HEAD_DIM:
        raise ValueError("kda_bi: A_log must have H and dt_bias H * 128 elements")
    if not -5.0 <= float(lower_bound) < 0.0:
        raise ValueError(f"kda_bi: lower_bound must be in [-5, 0), got {lower_bound}")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("kda_bi: cu_seqlens must be [N + 1]")
    N = cu_seqlens.numel() - 1
    if initial_state is not None and (
        initial_state.dtype != torch.float32
        or tuple(initial_state.shape) != (N, H, HEAD_DIM, HEAD_DIM)
    ):
        raise ValueError(f"kda_bi: initial_state must be fp32 [{N}, {H}, 128, 128]")
    if T >= 2**31:
        raise ValueError("kda_bi: T must be below 2**31")
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)

    cu_host = cu_seqlens.cpu() if cu_seqlens_cpu is None else cu_seqlens_cpu
    lens = (cu_host[1:] - cu_host[:-1]).to(torch.int64)
    n_chunks = int(((lens + CHUNK - 1) // CHUNK).sum())
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    a_log = A_log.to(torch.float32).contiguous()
    dtb = dt_bias.to(torch.float32).contiguous()
    dev = q.device
    o = torch.empty_like(v)
    final = (
        torch.empty(N, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev)
        if output_final_state
        else None
    )
    bound = (
        torch.empty(N, H, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev)
        if output_boundary_state
        else None
    )
    h0 = None if initial_state is None else initial_state.contiguous()

    if n_chunks > 0:
        # chunk table on the device (no host sync): chunks per sequence, then each chunk's rows
        cu = cu_seqlens.to(dev)
        seq_chunks = torch.empty(N + 1, dtype=torch.int32, device=dev)
        chunks = torch.empty(n_chunks, 2, dtype=torch.int32, device=dev)
        c_seq, c_tab = _config("seq_chunks"), _config("chunk_table")
        _kda_bi_seq_chunks_kernel[(1,)](cu, seq_chunks, N, **c_seq)
        _kda_bi_chunk_table_kernel[(triton.cdiv(n_chunks, c_tab["BLOCK"]),)](
            cu, seq_chunks, chunks, N, n_chunks, max(N.bit_length(), 1), **c_tab
        )
        qg, kg, w, u = (torch.empty_like(q) for _ in range(4))
        aqk = torch.empty(1, T, H, CHUNK, dtype=torch.bfloat16, device=dev)
        glast = torch.empty(n_chunks, H, HEAD_DIM, dtype=torch.float32, device=dev)
        fuse = _fuse_out(N * H)
        # per-chunk bf16 states and v_new only feed the separate out kernel
        hs = (
            torch.empty(
                n_chunks, H, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=dev
            )
            if not fuse
            else aqk
        )
        vn = torch.empty_like(v) if not fuse else v
        ci, cs, co = _config("intra"), _config("state"), _config("out")
        _kda_bi_intra_kernel[(n_chunks, H)](
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dtb,
            chunks,
            qg,
            kg,
            w,
            u,
            aqk,
            glast,
            float(lower_bound),
            scale,
            H,
            **ci,
            enable_fp_fusion=False,
        )
        dummy = glast
        cs = {**cs, "BV": _state_bv(N * H)}
        # longest sequences first (device-side, no sync)
        seq_order = seq_chunks
        if N > 1:
            seq_order = torch.empty(N, dtype=torch.int32, device=dev)
            c_ord = _config("seq_order")
            _kda_bi_seq_order_kernel[(triton.cdiv(N, c_ord["BLOCK"]),)](
                seq_chunks, seq_order, N, **c_ord
            )
        _kda_bi_state_kernel[(N * H * (HEAD_DIM // cs["BV"]),)](
            w,
            u,
            kg,
            glast,
            chunks,
            seq_chunks,
            seq_order,
            dummy if h0 is None else h0,
            dummy if final is None else final,
            dummy if bound is None else bound,
            hs,
            vn,
            qg,
            aqk,
            o,
            scale,
            H,
            n_chunks,
            HAS_H0=h0 is not None,
            STORE_HT=final is not None,
            STORE_HB=bound is not None,
            FUSE_OUT=fuse,
            ORDERED=N > 1,
            **cs,
            enable_fp_fusion=False,
        )
        if not fuse:
            _kda_bi_out_kernel[(n_chunks, H, HEAD_DIM // co["BV"])](
                qg,
                aqk,
                vn,
                hs,
                o,
                chunks,
                scale,
                H,
                **co,
                enable_fp_fusion=False,
            )
    else:
        for t in (final, bound):
            if t is not None:
                t.copy_(h0) if h0 is not None else t.zero_()
    if output_boundary_state:
        return o, final, bound
    return o, final
