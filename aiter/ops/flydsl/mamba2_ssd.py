# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mamba-2 SSD chunked forward (FlyDSL).

Same contract as TileGym's ``mamba2_chunk_forward``::

    x    [B, T, H, P] bf16      a  [H] f32        d [H] f32
    b, c [B, T, G, N] bf16      dt [B, T, H] bf16 (post-softplus)
    init_state [B, H, P, N] f32
    -> out [B, T, H, P] bf16, final_state [B, H, P, N] f32
"""

import functools

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .kernels.mamba2_ssd_kernels import (
    build_chunk_cumsum,
    build_compute_out,
    build_fwd_pass_state,
)

# The kernels use the gfx11 (RDNA3 / RDNA3.5) v16-operand WMMA.
SUPPORTED_GFX = ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"]

_CUMSUM_BLOCK_H = 32


def default_config(head_size, state_size):
    """(rm, rn) for fwd_pass_state and rp for compute_out."""
    rm = 2 if head_size % 32 == 0 else 1
    rn = 2 if state_size % 32 == 0 else 1
    rp = 4 if head_size % 64 == 0 else (2 if head_size % 32 == 0 else 1)
    return {"rm": rm, "rn": rn, "rp": rp}


@functools.cache
def _cumsum(H, L):
    return build_chunk_cumsum(H, L, _CUMSUM_BLOCK_H)


@functools.cache
def _state(H, G, P, N, L, rm, rn):
    return build_fwd_pass_state(H, G, P, N, L, rm, rn)


@functools.cache
def _out(H, G, P, N, L, rp):
    return build_compute_out(H, G, P, N, L, rp)


def mamba2_chunk_forward_flydsl(x, a, b, c, d, dt, init_state, chunk_size=64, config=None):
    if get_gfx() not in SUPPORTED_GFX:
        raise RuntimeError(f"mamba2_chunk_forward_flydsl needs gfx11 WMMA; got {get_gfx()}")
    B, T, H, P = x.shape
    G, N = b.shape[2], b.shape[3]
    L = chunk_size
    if T % L:
        raise ValueError(f"num_tokens={T} must be a multiple of chunk_size={L}")
    if L % 16 or P % 16 or N % 16 or H % G:
        raise ValueError("chunk_size, head_size and state_size must be multiples of 16")
    for t, dtype in ((x, torch.bfloat16), (b, torch.bfloat16), (c, torch.bfloat16), (dt, torch.bfloat16)):
        if t.dtype != dtype or not t.is_contiguous():
            raise ValueError("x, b, c, dt must be contiguous bf16")
    cfg = {**default_config(P, N), **(config or {})}
    nc = T // L

    a = a.float().contiguous()
    d = d.float().contiguous()
    init_state = init_state.float().contiguous()
    decay = torch.empty(B, H, T, device=x.device, dtype=torch.float32)
    dt_bht = torch.empty(B, H, T, device=x.device, dtype=torch.float32)
    hiddens = torch.empty(B, H, nc, P, N, device=x.device, dtype=torch.bfloat16)
    final_state = torch.empty(B, H, P, N, device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)

    with torch.cuda.device(x.device):
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        _run_compiled(_cumsum(H, L), a, dt, decay, dt_bht, T, nc, B, stream)
        _run_compiled(
            _state(H, G, P, N, L, cfg["rm"], cfg["rn"]),
            x, b, decay, dt_bht, init_state, hiddens, final_state, T, B, stream,
        )
        _run_compiled(
            _out(H, G, P, N, L, cfg["rp"]),
            x, b, c, d, decay, dt_bht, hiddens, out, T, nc, B, stream,
        )
    return out, final_state
