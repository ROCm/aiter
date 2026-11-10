# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mamba-2 SSD chunked forward and backward (FlyDSL).

Contract::

    x    [B, T, H, P] bf16      a  [H] f32        d [H] f32
    b, c [B, T, G, N] bf16      dt [B, T, H] bf16 (post-softplus)
    init_state [B, H, P, N] f32
    -> out [B, T, H, P] bf16, final_state [B, H, P, N] f32

Tile configs are autotuned once per shape (tune once, cache, launch); set ``AITER_MAMBA2_AUTOTUNE=0`` to use the defaults.
"""

import functools
import itertools
import math
import os

import flydsl.expr as fx
import torch
from flydsl.autotune import do_bench

from aiter import logger
from aiter.jit.utils.chip_info import (
    get_cu_num,
    get_gfx_runtime,
    get_lds_capacity_bytes,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .kernels.mamba2_ssd_kernels import (
    MAX_THREADS,
    TILE,
    WAVE,
    build_chunk_cumsum,
    build_compute_out,
    build_fwd_pass_state,
    compute_out_lds_bytes,
    fwd_pass_state_lds_bytes,
)

# RDNA3 / RDNA3.5 and RDNA4: wave32 bf16 16x16x16 WMMA (see ``wmma_layout``).
SUPPORTED_GFX = [
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1200",
    "gfx1201",
]

_CUMSUM_BLOCK_H = 32
_TILE_REPEATS = (1, 2, 4)  # 16-wide WMMA tiles per wave along a dimension
_AUTOTUNE = os.environ.get("AITER_MAMBA2_AUTOTUNE", "1") == "1"
_tuned = {}


def _state_configs(P, N, L):
    lds = get_lds_capacity_bytes(get_gfx_runtime())
    for rm, rn, ks in itertools.product(_TILE_REPEATS, _TILE_REPEATS, _TILE_REPEATS):
        if P % (16 * rm) or N % (16 * rn) or L % (16 * ks):
            continue
        if fwd_pass_state_lds_bytes(rm, rn, ks) > lds:
            continue
        yield {"rm": rm, "rn": rn, "ks": ks}


def _out_configs(P, N, L):
    # Plain grid, or persistent at 1-2 blocks per CU;
    # the state-tile K loop runs 4 tiles per trip or fully unrolled.
    lds = get_lds_capacity_bytes(get_gfx_runtime())
    nk = N // 16
    kus = sorted({nk} | ({4} if nk % 4 == 0 else set()))
    for rp, ku, (persistent, occupancy) in itertools.product(
        _TILE_REPEATS, kus, ((False, 0), (True, 1), (True, 2))
    ):
        if P % (16 * rp) or compute_out_lds_bytes(L, rp) > lds:
            continue
        yield {"rp": rp, "ku": ku, "persistent": persistent, "occupancy": occupancy}


def default_config(head_size, state_size, chunk_size):
    """Untuned fallback: widest tiles that divide the sizes and fit in LDS."""
    state = list(_state_configs(head_size, state_size, chunk_size))[-1]
    return {**state, **list(_out_configs(head_size, state_size, chunk_size))[-1]}


# Compiled handles are per device (``_run_compiled`` caches one per launcher),
# so every factory takes the device index as its first cache key.
@functools.cache
def _cumsum(device, H, L):
    return build_chunk_cumsum(H, L, _CUMSUM_BLOCK_H)


@functools.cache
def _state(device, H, G, P, N, L, rm, rn, ks, backward=False, state_f32=False):
    return build_fwd_pass_state(
        get_gfx_runtime(), H, G, P, N, L, rm, rn, ks, backward, state_f32
    )


@functools.cache
def _out(device, H, G, P, N, L, rp, ku, persistent):
    return build_compute_out(get_gfx_runtime(), H, G, P, N, L, rp, ku, persistent)


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
    """Tune once / cache / launch: an explicit ``config`` wins, else
    benchmark every candidate once per shape and keep the fastest."""
    if config is not None:
        return config
    if not _AUTOTUNE:
        return configs[-1]
    if (name, key) not in _tuned:
        timed = []
        for cfg in configs:
            launch(cfg)  # compile outside the timed region
            timed.append((do_bench(lambda cfg=cfg: launch(cfg)), cfg))
        ms, best = min(timed, key=lambda t: t[0])
        logger.info(
            f"mamba2 {name}: {best} ({ms * 1e3:.1f} us, best of {len(timed)} configs)"
        )
        _tuned[name, key] = best
    return _tuned[name, key]


_MAX_BUFFER_BYTES = 2**31
# compute_out runs one wave per 16 tokens of a chunk.
MAX_CHUNK_SIZE = MAX_THREADS // WAVE * TILE


def _check_inputs(x, a, b, c, d, dt, init_state, chunk_size, state_bytes=2):
    """Reject inputs the kernels would index out of bounds; see the contract."""
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        raise RuntimeError(f"mamba2 FlyDSL kernels support {SUPPORTED_GFX}; got {gfx}")
    if x.dim() != 4 or b.dim() != 4:
        raise ValueError(f"x and b must be 4-D, got {x.dim()}-D and {b.dim()}-D")
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[2], b.shape[3]
    expected = {
        "b": (batch_size, num_tokens, num_groups, state_size),
        "c": (batch_size, num_tokens, num_groups, state_size),
        "dt": (batch_size, num_tokens, num_heads),
        "a": (num_heads,),
        "d": (num_heads,),
        "init_state": (batch_size, num_heads, head_size, state_size),
    }
    given = {"b": b, "c": c, "dt": dt, "a": a, "d": d, "init_state": init_state}
    for name, shape in expected.items():
        if tuple(given[name].shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(given[name].shape)}"
            )
    if not 0 < chunk_size <= MAX_CHUNK_SIZE or num_tokens % chunk_size:
        raise ValueError(
            f"chunk_size must be in (0, {MAX_CHUNK_SIZE}] and divide num_tokens"
        )
    if chunk_size % 16 or head_size % 16 or state_size % 16:
        raise ValueError("chunk_size, head_size and state_size must be multiples of 16")
    if num_heads % num_groups:
        raise ValueError("num_heads must be a multiple of num_groups")
    for t in (x, b, c, dt):
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError("x, b, c, dt must be contiguous bf16")
    for t in (a, b, c, d, dt, init_state):
        if t.device != x.device:
            raise ValueError("all inputs must be on x's device")
    # Buffer descriptors use 32-bit byte offsets over the whole buffer.
    hidden_bytes = (
        batch_size
        * num_heads
        * (num_tokens // chunk_size)
        * head_size
        * state_size
        * state_bytes
    )
    largest = max(
        x.nbytes, b.nbytes, batch_size * num_heads * num_tokens * 4, hidden_bytes
    )
    if largest >= _MAX_BUFFER_BYTES:
        raise ValueError(
            f"a {largest}-byte buffer exceeds the {_MAX_BUFFER_BYTES}-byte limit of "
            "32-bit buffer offsets; split the batch or the heads"
        )


def _chunk_cumsum(a, dt, chunk_size, stream):
    """dt:[b,t,h] -> (decay_bht [b,h,t], dt_bht [b,h,t]), decay in log2 units."""
    batch_size, num_tokens, num_heads = dt.shape
    decay_bht = dt.new_empty(batch_size, num_heads, num_tokens, dtype=torch.float32)
    dt_bht = torch.empty_like(decay_bht)
    _run_compiled(
        _cumsum(dt.device.index, num_heads, chunk_size),
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


def _fwd_pass_state(
    x,
    b,
    dt_bht,
    decay_bht,
    init_state,
    chunk_size,
    stream,
    config,
    backward=False,
    state_dtype=None,
):
    """Return chunk-entry states (x.dtype unless ``state_dtype``) and the fp32
    final state. With ``backward``, x / b / init_state are dout / c / dfinal_state
    and this returns the per-chunk state gradients and the initial-state gradient.
    """
    state_dtype = x.dtype if state_dtype is None else state_dtype
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[2], b.shape[3]
    num_chunks = num_tokens // chunk_size
    hiddens = x.new_empty(
        batch_size, num_heads, num_chunks, head_size, state_size, dtype=state_dtype
    )
    final_state = torch.empty_like(init_state)

    def launch(cfg):
        _run_compiled(
            _state(
                x.device.index,
                num_heads,
                num_groups,
                head_size,
                state_size,
                chunk_size,
                cfg["rm"],
                cfg["rn"],
                cfg["ks"],
                backward,
                state_dtype == torch.float32,
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
    key = _tuning_key(x, b, init_state, chunk_size, backward, state_dtype)
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
                x.device.index,
                num_heads,
                num_groups,
                head_size,
                state_size,
                chunk_size,
                cfg["rp"],
                cfg["ku"],
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

    configs = list(_out_configs(head_size, state_size, chunk_size))
    key = _tuning_key(x, b, chunk_size)
    launch(_tuned_config("compute_out", key, configs, launch, config))
    return out


def mamba2_chunk_forward_flydsl(
    x, a, b, c, d, dt, init_state, chunk_size=64, config=None
):
    """Return the output and fp32 final state; chunk-entry states are kept in
    c.dtype."""
    _check_inputs(x, a, b, c, d, dt, init_state, chunk_size)
    if x.shape[1] == 0:  # nothing to scan: the state passes through
        return torch.empty_like(x), init_state.float().clone()
    head_size, state_size = x.shape[3], b.shape[3]
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


def _chunked(t, chunk_size):
    """[B, T, H, ...] -> [B, H, num_chunks, chunk_size, ...]."""
    B, T, H = t.shape[:3]
    return t.reshape(B, T // chunk_size, chunk_size, H, *t.shape[3:]).movedim(3, 1)


def _unchunked(t):
    """[B, H, num_chunks, chunk_size, ...] -> [B, T, H, ...]."""
    B, H, nc, L = t.shape[:4]
    return t.movedim(1, 3).reshape(B, nc * L, H, *t.shape[4:])


def _chunk_grads(x, a, b, c, d, dt_bht, decay_bht, hiddens, dout, dhiddens, L):
    """Per-chunk gradients from the chunk states and state gradients (fp32).

    Numerics: no small quantity is formed as a difference of large, separately
    rounded ones (the decay gradient's reverse cumsum is a direct suffix scan;
    ddt is accumulated without dt and never divided by it), and the decay
    gradient of the shift-invariant terms -- those depending only on d_t - d_s
    within a chunk -- is re-centred to sum to exactly zero, as the exact
    gradient does.
    """
    B, T, H, _ = x.shape
    G, N = b.shape[2], b.shape[3]
    HPG = H // G
    X, DO = _chunked(x.float(), L), _chunked(dout.float(), L)
    Bm = _chunked(b.float().repeat_interleave(HPG, 2), L)
    Cm = _chunked(c.float().repeat_interleave(HPG, 2), L)
    DT = dt_bht.view(B, H, T // L, L)
    dec = decay_bht.view(B, H, T // L, L) * math.log(2.0)  # log2 -> natural units
    gl = dec[..., -1]
    causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
    seg = torch.exp(  # mask before exp: the upper triangle would overflow
        torch.where(causal, dec[..., :, None] - dec[..., None, :], -torch.inf)
    )
    w_b = DT * torch.exp(gl[..., None] - dec)  # weight of B_s in the state update

    cb_raw = Cm @ Bm.transpose(-1, -2)
    cb = cb_raw * seg
    hm = DO @ X.transpose(-1, -2)  # dout_t . x_s, without dt
    dcb = hm * DT[..., None, :] * seg
    m = dcb * cb_raw

    dx = DO * d[None, :, None, None, None] + DT[..., None] * (cb.transpose(-1, -2) @ DO)
    dx = dx + (Bm * w_b[..., None]) @ dhiddens.transpose(-1, -2)
    u = X @ dhiddens
    q = torch.exp(gl[..., None] - dec) * (u * Bm).sum(-1)
    ddt = (cb * hm).sum(-2) + q
    dD = (DO * X).sum((2, 3, 4))  # [B, H]

    v = DO @ hiddens
    e_d = torch.exp(dec)[..., None]
    dc = v * e_d + dcb @ Bm
    db = dcb.transpose(-1, -2) @ Cm + u * w_b[..., None]

    dd = (v * Cm * e_d).sum(-1)
    dd_shift = m.sum(-1) - m.sum(-2) - DT * q
    dd_shift[..., -1] += (DT * q).sum(-1)
    dd = dd + dd_shift - dd_shift.mean(-1, keepdim=True)
    dd[..., -1] += (dhiddens * hiddens).sum((-1, -2)) * torch.exp(gl)
    dd_rev = dd.flip(-1).cumsum(-1).flip(-1)  # suffix scan
    ddt = ddt + dd_rev * a[None, :, None, None]
    da = (DT * dd_rev).sum((0, 2, 3))

    db = _unchunked(db).reshape(B, T, G, HPG, N).sum(3)
    dc = _unchunked(dc).reshape(B, T, G, HPG, N).sum(3)
    return _unchunked(dx), da, db, dc, dD.sum(0), _unchunked(ddt)


def mamba2_chunk_backward_flydsl(
    x, a, b, c, d, dt, init_state, dout, dfinal_state, chunk_size=64, config=None
):
    """Return dx, da, db, dc, dd, ddt and the initial-state gradient.

    The chunk states (fp32) and the state gradients (the forward recurrence run
    in reverse) come from the FlyDSL pass_state kernel.
    """
    _check_inputs(x, a, b, c, d, dt, init_state, chunk_size, state_bytes=4)
    if (
        dout.shape != x.shape
        or dout.dtype != torch.bfloat16
        or not dout.is_contiguous()
        or dout.device != x.device
    ):
        raise ValueError("dout must be contiguous bf16 shaped like x, on x's device")
    if dfinal_state.shape != init_state.shape or dfinal_state.device != x.device:
        raise ValueError("dfinal_state must be shaped like init_state, on x's device")
    if x.shape[1] == 0:  # no tokens: only the state gradient passes through
        z = torch.zeros_like
        return (
            z(x),
            z(a, dtype=torch.float32),
            z(b),
            z(c),
            z(d, dtype=torch.float32),
            z(dt),
            dfinal_state.float().clone(),
        )
    head_size, state_size = x.shape[3], b.shape[3]
    if config is not None:
        config = {**default_config(head_size, state_size, chunk_size), **config}

    a = a.float().contiguous()
    d = d.float().contiguous()
    init_state = init_state.float().contiguous()
    dfinal_state = dfinal_state.float().contiguous()
    with torch.cuda.device(x.device):
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        decay_bht, dt_bht = _chunk_cumsum(a, dt, chunk_size, stream)
        hiddens, _ = _fwd_pass_state(
            x,
            b,
            dt_bht,
            decay_bht,
            init_state,
            chunk_size,
            stream,
            config,
            state_dtype=torch.float32,
        )
        dhiddens, dinit_state = _fwd_pass_state(
            dout,
            c,
            dt_bht,
            decay_bht,
            dfinal_state,
            chunk_size,
            stream,
            config,
            backward=True,
            state_dtype=torch.float32,
        )
        dx, da, db, dc, dd, ddt = _chunk_grads(
            x, a, b, c, d, dt_bht, decay_bht, hiddens, dout, dhiddens, chunk_size
        )
    return (
        dx.to(x.dtype),
        da,
        db.to(b.dtype),
        dc.to(c.dtype),
        dd,
        ddt.to(dt.dtype),
        dinit_state,
    )
