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
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
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


def _state_configs(P, N, L):
    for rm, rn, ks in itertools.product(_TILE_REPEATS, _TILE_REPEATS, _TILE_REPEATS):
        if P % (16 * rm) == 0 and N % (16 * rn) == 0 and L % (16 * ks) == 0:
            yield {"rm": rm, "rn": rn, "ks": ks}


def _out_configs(P):
    # TileGym _compute_out_configs: plain grid, or persistent at 1-2 blocks per CU.
    for rp, (persistent, occupancy) in itertools.product(
        _TILE_REPEATS, ((False, 0), (True, 1), (True, 2))
    ):
        if P % (16 * rp) == 0:
            yield {"rp": rp, "persistent": persistent, "occupancy": occupancy}


def default_config(head_size, state_size, chunk_size):
    """Untuned fallback: widest tiles that divide the head and state sizes."""
    state = list(_state_configs(head_size, state_size, chunk_size))[-1]
    return {**state, **list(_out_configs(head_size))[-1]}


@functools.cache
def _cumsum(H, L):
    return build_chunk_cumsum(H, L, _CUMSUM_BLOCK_H)


@functools.cache
def _state(H, G, P, N, L, rm, rn, ks):
    return build_fwd_pass_state(H, G, P, N, L, rm, rn, ks)


@functools.cache
def _out(H, G, P, N, L, rp, persistent):
    return build_compute_out(H, G, P, N, L, rp, persistent)


def _tuning_key(*args):
    return tuple(
        (
            (tuple(arg.shape), tuple(arg.stride()), arg.dtype, str(arg.device))
            if torch.is_tensor(arg)
            else arg
        )
        for arg in args
    )


def _tuned_config(name, key, configs, launch, config):
    """TileGym's tune-once / cache / launch: an explicit ``config`` wins, else
    benchmark every candidate once per shape and keep the fastest."""
    if config is not None:
        return config
    if not _AUTOTUNE:
        return configs[-1]
    if (name, key) not in _tuned:
        timed = []
        for cfg in configs:
            launch(cfg)  # compile outside the timed region
            timed.append((do_bench(lambda: launch(cfg)), cfg))
        ms, best = min(timed, key=lambda t: t[0])
        logger.info(
            f"mamba2 {name}: {best} ({ms * 1e3:.1f} us, best of {len(timed)} configs)"
        )
        _tuned[name, key] = best
    return _tuned[name, key]


def _validate_chunk_size(num_tokens, chunk_size):
    if chunk_size <= 0 or num_tokens % chunk_size != 0:
        raise ValueError("num_tokens must be divisible by a positive chunk_size")


def _chunk_cumsum(a, dt, chunk_size, stream):
    """dt:[b,t,h] -> (decay_bht [b,h,t], dt_bht [b,h,t]), decay in log2 units."""
    batch_size, num_tokens, num_heads = dt.shape
    decay_bht = dt.new_empty(batch_size, num_heads, num_tokens, dtype=torch.float32)
    dt_bht = torch.empty_like(decay_bht)
    _run_compiled(
        _cumsum(num_heads, chunk_size),
        a,
        dt,
        decay_bht,
        dt_bht,
        num_tokens,
        num_tokens // chunk_size,
        batch_size,
        stream,
    )
    return decay_bht, dt_bht


def _fwd_pass_state(x, b, dt_bht, decay_bht, init_state, chunk_size, stream, config):
    """Return chunk-entry states (narrowed to x.dtype) and the fp32 final state."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[2], b.shape[3]
    num_chunks = num_tokens // chunk_size
    hiddens = x.new_empty(batch_size, num_heads, num_chunks, head_size, state_size)
    final_state = torch.empty_like(init_state)

    def launch(cfg):
        _run_compiled(
            _state(
                num_heads,
                num_groups,
                head_size,
                state_size,
                chunk_size,
                cfg["rm"],
                cfg["rn"],
                cfg["ks"],
            ),
            x,
            b,
            decay_bht,
            dt_bht,
            init_state,
            hiddens,
            final_state,
            num_tokens,
            batch_size,
            stream,
        )

    configs = list(_state_configs(head_size, state_size, chunk_size))
    key = _tuning_key(x, b, init_state, chunk_size)
    launch(_tuned_config("fwd_pass_state", key, configs, launch, config))
    return hiddens, final_state


def _compute_out(x, b, c, d, dt_bht, decay_bht, hiddens, chunk_size, stream, config):
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[2], b.shape[3]
    out = torch.empty_like(x)
    num_chunks = num_tokens // chunk_size

    def launch(cfg):
        if cfg["persistent"]:
            num_programs = cfg["occupancy"] * get_cu_num()
        else:
            num_programs = batch_size * num_chunks * num_heads * head_size
            num_programs //= 16 * cfg["rp"]
        _run_compiled(
            _out(
                num_heads,
                num_groups,
                head_size,
                state_size,
                chunk_size,
                cfg["rp"],
                cfg["persistent"],
            ),
            x,
            b,
            c,
            d,
            decay_bht,
            dt_bht,
            hiddens,
            out,
            num_tokens,
            batch_size,
            num_programs,
            stream,
        )

    configs = list(_out_configs(head_size))
    key = _tuning_key(x, b, chunk_size)
    launch(_tuned_config("compute_out", key, configs, launch, config))
    return out


def mamba2_chunk_forward_flydsl(
    x, a, b, c, d, dt, init_state, chunk_size=64, config=None
):
    """Return the output and fp32 final state; chunk-entry states are kept in
    c.dtype, as in TileGym's mamba2_chunk_forward."""
    if get_gfx() not in SUPPORTED_GFX:
        raise RuntimeError(
            f"mamba2_chunk_forward_flydsl needs gfx11 WMMA; got {get_gfx()}"
        )
    _validate_chunk_size(x.shape[1], chunk_size)
    head_size, state_size = x.shape[3], b.shape[3]
    if chunk_size % 16 or head_size % 16 or state_size % 16:
        raise ValueError("chunk_size, head_size and state_size must be multiples of 16")
    if x.shape[2] % b.shape[2]:
        raise ValueError("num_heads must be a multiple of num_groups")
    for t in (x, b, c, dt):
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError("x, b, c, dt must be contiguous bf16")
    if config is not None:
        config = {**default_config(head_size, state_size, chunk_size), **config}

    a = a.float().contiguous()
    d = d.float().contiguous()
    init_state = init_state.float().contiguous()
    with torch.cuda.device(x.device):
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        decay_bht, dt_bht = _chunk_cumsum(a, dt, chunk_size, stream)
        hiddens, final_state = _fwd_pass_state(
            x, b, dt_bht, decay_bht, init_state, chunk_size, stream, config
        )
        out = _compute_out(
            x, b, c, d, dt_bht, decay_bht, hiddens, chunk_size, stream, config
        )
    return out, final_state
