# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mamba-2 SSD chunked forward (FlyDSL).

Same contract as TileGym's ``mamba2_chunk_forward``::

    x    [B, T, H, P] bf16      a  [H] f32        d [H] f32
    b, c [B, T, G, N] bf16      dt [B, T, H] bf16 (post-softplus)
    init_state [B, H, P, N] f32
    -> out [B, T, H, P] bf16, final_state [B, H, P, N] f32

Tile configs are autotuned once per shape (TileGym's tune-once / cache /
launch pattern); set ``AITER_MAMBA2_AUTOTUNE=0`` to use the defaults.
"""

import functools
import itertools
import os

import flydsl.expr as fx
import torch
from flydsl.autotune import do_bench

from aiter import logger
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
_TILE_REPEATS = (1, 2, 4)  # 16-wide WMMA tiles per wave along a dimension
_AUTOTUNE = os.environ.get("AITER_MAMBA2_AUTOTUNE", "1") == "1"
_tuned = {}


def _state_configs(P, N):
    for rm, rn in itertools.product(_TILE_REPEATS, _TILE_REPEATS):
        if P % (16 * rm) == 0 and N % (16 * rn) == 0:
            yield {"rm": rm, "rn": rn}


def _out_configs(P):
    for rp, vec_loads in itertools.product(_TILE_REPEATS, (False, True)):
        if P % (16 * rp) == 0:
            yield {"rp": rp, "vec_loads": vec_loads}


def default_config(head_size, state_size):
    """Untuned fallback: widest tiles that divide the head and state sizes."""
    state = list(_state_configs(head_size, state_size))[-1]
    rp = list(_out_configs(head_size))[-1]["rp"]
    return {**state, "rp": rp, "vec_loads": False}


@functools.cache
def _cumsum(H, L):
    return build_chunk_cumsum(H, L, _CUMSUM_BLOCK_H)


@functools.cache
def _state(H, G, P, N, L, rm, rn):
    return build_fwd_pass_state(H, G, P, N, L, rm, rn)


@functools.cache
def _out(H, G, P, N, L, rp, vec_loads):
    return build_compute_out(H, G, P, N, L, rp, vec_loads)


def _pick(name, key, configs, launch):
    """Benchmark every config once for this shape and cache the fastest."""
    if (name, key) not in _tuned:
        timed = []
        for cfg in configs:
            launch(cfg)  # compile outside the timed region
            timed.append((do_bench(lambda: launch(cfg)), cfg))
        ms, best = min(timed, key=lambda t: t[0])
        logger.info(
            f"mamba2 {name} {key}: {best} ({ms * 1e3:.1f} us of {len(timed)} configs)"
        )
        _tuned[name, key] = best
    return _tuned[name, key]


def mamba2_chunk_forward_flydsl(
    x, a, b, c, d, dt, init_state, chunk_size=64, config=None
):
    if get_gfx() not in SUPPORTED_GFX:
        raise RuntimeError(
            f"mamba2_chunk_forward_flydsl needs gfx11 WMMA; got {get_gfx()}"
        )
    B, T, H, P = x.shape
    G, N = b.shape[2], b.shape[3]
    L = chunk_size
    if T % L:
        raise ValueError(f"num_tokens={T} must be a multiple of chunk_size={L}")
    if L % 16 or P % 16 or N % 16 or H % G:
        raise ValueError("chunk_size, head_size and state_size must be multiples of 16")
    for t, dtype in (
        (x, torch.bfloat16),
        (b, torch.bfloat16),
        (c, torch.bfloat16),
        (dt, torch.bfloat16),
    ):
        if t.dtype != dtype or not t.is_contiguous():
            raise ValueError("x, b, c, dt must be contiguous bf16")
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

        def launch_state(cfg):
            _run_compiled(
                _state(H, G, P, N, L, cfg["rm"], cfg["rn"]),
                x,
                b,
                decay,
                dt_bht,
                init_state,
                hiddens,
                final_state,
                T,
                B,
                stream,
            )

        def launch_out(cfg):
            _run_compiled(
                _out(H, G, P, N, L, cfg["rp"], cfg["vec_loads"]),
                x,
                b,
                c,
                d,
                decay,
                dt_bht,
                hiddens,
                out,
                T,
                nc,
                B,
                stream,
            )

        _run_compiled(_cumsum(H, L), a, dt, decay, dt_bht, T, nc, B, stream)
        key = (B, T, G, H, N, P, L)
        if config is not None:
            cfg = {**default_config(P, N), **config}
            state_cfg = out_cfg = cfg
        elif _AUTOTUNE:
            state_cfg = _pick(
                "fwd_pass_state", key, list(_state_configs(P, N)), launch_state
            )
            out_cfg = None  # tuned below, once the hiddens it reads are final
        else:
            state_cfg = out_cfg = default_config(P, N)

        launch_state(state_cfg)  # the tuning runs above clobbered hiddens / final_state
        if out_cfg is None:
            out_cfg = _pick("compute_out", key, list(_out_configs(P)), launch_out)
        launch_out(out_cfg)
    return out, final_state
