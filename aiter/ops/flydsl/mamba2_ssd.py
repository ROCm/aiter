# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mamba-2 SSD chunked scan (FlyDSL), forward and backward.

``mamba_chunk_scan_combined`` takes the arguments of ``mamba_ssm``'s function of
the same name, including variable-length batches (``cu_seqlens``), and is
differentiable.

Internally every batch is one flat run of whole chunks: each sequence starts a
new chunk, and a ragged sequence's last chunk is padded with ``dt = 0`` tokens,
which change neither the state nor any gradient. Tile configs are autotuned
once per shape bucket; set ``AITER_MAMBA2_AUTOTUNE=0`` to use the defaults.
"""

import functools
import itertools
import math
import os
from dataclasses import dataclass

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
from aiter.ops.triton._triton_kernels.gated_delta_net.gated_delta_rule_utils import (
    tensor_cache,
)
from aiter.ops.triton._triton_kernels.gated_delta_net.utils.index import (
    prepare_chunk_offsets,
    prepare_lens,
)

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


def _tuning_key(x, b, num_seqs, *flags):
    """Shape bucket: varlen batches vary in length, so token and sequence counts
    are rounded up to powers of two."""

    def bucket(n):
        return 1 << max(n - 1, 0).bit_length()

    return (
        x.shape[1:],
        b.shape[1:],
        bucket(x.shape[0]),
        bucket(num_seqs),
        x.device,
        *flags,
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


@dataclass(frozen=True)
class _Layout:
    """Where each token sits in the padded flat chunk layout."""

    real_tokens: int  # cu_seqlens[-1]
    num_tokens: int  # padded: num_chunks * chunk_size
    seq_chunks: torch.Tensor  # int32 [S + 1], first chunk of each sequence
    dst: torch.Tensor | None  # padded position of each token; None if no padding


@tensor_cache  # one layout per cu_seqlens tensor, shared by every layer
def _layout(cu_seqlens, chunk_size):
    lens = prepare_lens(cu_seqlens)
    offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    first, last, shortest, ragged, num_chunks = torch.stack(  # one host sync
        [
            cu_seqlens[0],
            cu_seqlens[-1],
            lens.min(),
            (lens % chunk_size).any(),
            offsets[-1],
        ]
    ).tolist()
    if first != 0 or shortest < 0:
        raise ValueError("cu_seqlens must start at 0 and never decrease")
    dst = None
    if ragged:  # repeat_interleave, as empty sequences have no position 0
        seq = torch.repeat_interleave(torch.arange(len(lens), device=lens.device), lens)
        pos = torch.arange(last, device=lens.device) - cu_seqlens[seq]
        dst = offsets[seq] * chunk_size + pos
    return _Layout(last, num_chunks * chunk_size, offsets.int(), dst)


@functools.lru_cache(maxsize=64)
def _uniform_layout(batch, seqlen, chunk_size, device):
    """Layout of ``batch`` equal-length sequences, without a host sync."""
    per_seq = -(-seqlen // chunk_size)
    offsets = torch.arange(batch + 1, device=device, dtype=torch.int32) * per_seq
    dst = None
    if seqlen % chunk_size:
        rows = torch.arange(batch, device=device)[:, None] * (per_seq * chunk_size)
        dst = (rows + torch.arange(seqlen, device=device)).flatten()
    return _Layout(batch * seqlen, batch * per_seq * chunk_size, offsets, dst)


def _pad(t, layout):
    """[T, ...] -> [num_tokens, ...], zero padding."""
    if layout.dst is None:
        return t
    return t.new_zeros(layout.num_tokens, *t.shape[1:]).index_copy(0, layout.dst, t)


def _unpad(t, layout):
    return t if layout.dst is None else t.index_select(0, layout.dst)


def _check_buffers(
    num_tokens, num_heads, head_size, num_groups, state_size, num_chunks
):
    # Buffer descriptors use 32-bit byte offsets over the whole buffer.
    largest = max(
        num_tokens * num_heads * head_size * 2,
        num_tokens * num_groups * state_size * 2,
        num_heads * num_tokens * 4,
        num_heads * num_chunks * head_size * state_size * 4,  # fp32 in the backward
    )
    if largest >= _MAX_BUFFER_BYTES:
        raise ValueError(
            f"a {largest}-byte buffer exceeds the {_MAX_BUFFER_BYTES}-byte limit of "
            "32-bit buffer offsets; split the batch"
        )


def _chunk_cumsum(a, dt, chunk_size, stream):
    """dt [T, H] f32 -> (decay [H, T] in log2 units, dt [H, T])."""
    num_tokens, num_heads = dt.shape
    decay_ht = dt.new_empty(num_heads, num_tokens, dtype=torch.float32)
    dt_ht = torch.empty_like(decay_ht)
    _run_compiled(
        _cumsum(dt.device.index, num_heads, chunk_size),
        a,
        dt,
        decay_ht,
        dt_ht,
        num_tokens,
        num_tokens // chunk_size,
        1,
        stream,
    )
    return decay_ht, dt_ht


def _pass_state(
    x,
    b,
    dt_ht,
    decay_ht,
    init_state,
    layout,
    chunk_size,
    stream,
    config,
    backward=False,
    state_dtype=torch.bfloat16,
):
    """Walk each sequence's chunks; return the state entering every chunk and each
    sequence's final state. With ``backward`` (x, b, init_state = dout, c,
    dfinal_state) the walk runs in reverse and returns the state gradients."""
    num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[1], b.shape[2]
    num_chunks = num_tokens // chunk_size
    hiddens = x.new_empty(
        num_heads, num_chunks, head_size, state_size, dtype=state_dtype
    )
    final_state = torch.empty_like(init_state)
    num_seqs = init_state.shape[0]

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
            decay_ht,
            dt_ht,
            init_state,
            hiddens,
            final_state,
            layout.seq_chunks,
            num_tokens,
            num_seqs,
            stream,
        )

    configs = list(_state_configs(head_size, state_size, chunk_size))
    key = _tuning_key(x, b, num_seqs, chunk_size, backward, state_dtype)
    launch(_tuned_config("pass_state", key, configs, launch, config))
    return hiddens, final_state


def _compute_out(x, b, c, d, dt_ht, decay_ht, hiddens, chunk_size, stream, config):
    num_tokens, num_heads, head_size = x.shape
    num_groups, state_size = b.shape[1], b.shape[2]
    out = torch.empty_like(x)
    num_chunks = num_tokens // chunk_size

    def launch(cfg):
        if cfg["persistent"]:
            num_programs = cfg["occupancy"] * get_cu_num()
        else:
            num_programs = num_chunks * num_heads * head_size // (16 * cfg["rp"])
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
            decay_ht,
            dt_ht,
            hiddens,
            out,
            num_tokens,
            1,
            num_programs,
            stream,
        )

    configs = list(_out_configs(head_size, state_size, chunk_size))
    key = _tuning_key(x, b, 1, chunk_size)
    launch(_tuned_config("compute_out", key, configs, launch, config))
    return out


def _chunked(t, chunk_size):
    """[T, H, ...] -> [1, H, num_chunks, chunk_size, ...]."""
    T, H = t.shape[:2]
    return t.reshape(T // chunk_size, chunk_size, H, *t.shape[2:]).movedim(2, 0)[None]


def _unchunked(t):
    """[1, H, num_chunks, chunk_size, ...] -> [T, H, ...]."""
    H, nc, L = t.shape[1:4]
    return t[0].movedim(0, 2).reshape(nc * L, H, *t.shape[4:])


def _chunk_grads(x, a, b, c, d, dt_bht, decay_bht, hiddens, dout, dhiddens, L):
    """Per-chunk gradients from the chunk states and state gradients (fp32).

    Numerics: no small quantity is formed as a difference of large, separately
    rounded ones (the decay gradient's reverse cumsum is a direct suffix scan;
    ddt is accumulated without dt and never divided by it), and the decay
    gradient of the shift-invariant terms -- those depending only on d_t - d_s
    within a chunk -- is re-centred to sum to exactly zero, as the exact
    gradient does.
    """
    T, H, _ = x.shape
    B, G, N = 1, b.shape[1], b.shape[2]
    HPG = H // G
    X, DO = _chunked(x.float(), L), _chunked(dout.float(), L)
    Bm = _chunked(b.float().repeat_interleave(HPG, 1), L)
    Cm = _chunked(c.float().repeat_interleave(HPG, 1), L)
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

    db = _unchunked(db).reshape(T, G, HPG, N).sum(2)
    dc = _unchunked(dc).reshape(T, G, HPG, N).sum(2)
    return _unchunked(dx), da, db, dc, dD.sum(0), _unchunked(ddt)


def _forward(x, dt, a, b, c, d, init_state, layout, chunk_size, config):
    with torch.cuda.device(x.device):
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        decay_ht, dt_ht = _chunk_cumsum(a, dt, chunk_size, stream)
        hiddens, final_state = _pass_state(
            x, b, dt_ht, decay_ht, init_state, layout, chunk_size, stream, config
        )
        out = _compute_out(
            x, b, c, d, dt_ht, decay_ht, hiddens, chunk_size, stream, config
        )
    return out, final_state


def _backward(x, dt, a, b, c, d, init_state, dout, dfinal, layout, chunk_size, config):
    """Chunk states (fp32) and state gradients come from the pass_state kernel run
    both ways; the per-chunk gradients are computed from them in torch."""
    with torch.cuda.device(x.device):
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        decay_ht, dt_ht = _chunk_cumsum(a, dt, chunk_size, stream)
        fp32 = {"state_dtype": torch.float32}
        hiddens, _ = _pass_state(
            x,
            b,
            dt_ht,
            decay_ht,
            init_state,
            layout,
            chunk_size,
            stream,
            config,
            **fp32,
        )
        dhiddens, dinit = _pass_state(
            dout,
            c,
            dt_ht,
            decay_ht,
            dfinal,
            layout,
            chunk_size,
            stream,
            config,
            backward=True,
            **fp32,
        )
        # pass_state keeps states as [H, C, P, N]; the torch stage wants [1, H, C, ...].
        dx, da, db, dc, dd, ddt = _chunk_grads(
            x,
            a,
            b,
            c,
            d,
            dt_ht[None],
            decay_ht[None],
            hiddens[None],
            dout,
            dhiddens[None],
            chunk_size,
        )
    return dx.to(x.dtype), ddt, da, db.to(b.dtype), dc.to(c.dtype), dd, dinit


class _ChunkScan(torch.autograd.Function):
    """Scan over the padded flat layout: x [T, H, P], dt [T, H] f32, b / c [T, G, N],
    a / d [H] f32, init_state [S, H, P, N] f32 -> out, final_state [S, H, P, N]."""

    @staticmethod
    def forward(ctx, x, dt, a, b, c, d, init_state, layout, chunk_size, config):
        ctx.save_for_backward(x, dt, a, b, c, d, init_state)
        ctx.layout, ctx.chunk_size, ctx.config = layout, chunk_size, config
        return _forward(x, dt, a, b, c, d, init_state, layout, chunk_size, config)

    @staticmethod
    def backward(ctx, dout, dfinal):
        x, dt, a, b, c, d, init_state = ctx.saved_tensors
        dout = torch.zeros_like(x) if dout is None else dout.to(x.dtype).contiguous()
        dfinal = torch.zeros_like(init_state) if dfinal is None else dfinal.float()
        grads = _backward(
            x,
            dt,
            a,
            b,
            c,
            d,
            init_state,
            dout,
            dfinal.contiguous(),
            ctx.layout,
            ctx.chunk_size,
            ctx.config,
        )
        return (*grads, None, None, None)


def _cu_seqlens(x, cu_seqlens, seq_idx):
    """Packed sequences' cu_seqlens, or None for a batch of equal-length rows."""
    batch = x.shape[0]
    if cu_seqlens is None and seq_idx is None:
        return None
    if batch != 1:
        raise ValueError("packed sequences (cu_seqlens / seq_idx) need batch size 1")
    if cu_seqlens is None:
        counts = torch.bincount(seq_idx[0].long())
        cu_seqlens = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2 or cu_seqlens.device != x.device:
        raise ValueError("cu_seqlens must be 1-D with >= 2 entries, on x's device")
    return cu_seqlens


def _check_inputs(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, num_seqs):
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        raise RuntimeError(f"mamba2 FlyDSL kernels support {SUPPORTED_GFX}; got {gfx}")
    if x.dim() != 4 or B.dim() != 4:
        raise ValueError(f"x and B must be 4-D, got {x.dim()}-D and {B.dim()}-D")
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2], B.shape[3]
    expected = {
        "dt": (dt, (batch, seqlen, nheads)),
        "A": (A, (nheads,)),
        "B": (B, (batch, seqlen, ngroups, dstate)),
        "C": (C, (batch, seqlen, ngroups, dstate)),
        "z": (z, x.shape),
        "dt_bias": (dt_bias, (nheads,)),
        "initial_states": (initial_states, (num_seqs, nheads, headdim, dstate)),
    }
    if D is not None:
        expected["D"] = (D, (nheads,) if D.dim() == 1 else (nheads, headdim))
    for name, (t, shape) in expected.items():
        if t is None:
            continue
        if tuple(t.shape) != tuple(shape):
            raise ValueError(
                f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}"
            )
        if t.device != x.device:
            raise ValueError(f"{name} must be on x's device")
    if not 0 < chunk_size <= MAX_CHUNK_SIZE or chunk_size % 16:
        raise ValueError(
            f"chunk_size must be a multiple of 16 in [16, {MAX_CHUNK_SIZE}]"
        )
    if headdim % 16 or dstate % 16:
        raise ValueError("headdim and dstate must be multiples of 16")
    if nheads % ngroups:
        raise ValueError("nheads must be a multiple of ngroups")
    for name, t in (("x", x), ("B", B), ("C", C)):
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bf16, got {t.dtype}")


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False,
    config=None,
):
    """Mamba-2 SSD scan with ``mamba_ssm.ops.triton.ssd_combined``'s interface.

    x [b, l, h, p], dt [b, l, h], A [h], B / C [b, l, g, n] (x, B, C in bf16),
    D [h] or [h, p], z [b, l, h, p], dt_bias [h], initial_states [s, h, p, n]
    with s = b, or the number of sequences when packed. Packed sequences (b = 1)
    are given by ``cu_seqlens`` [s + 1] or ``seq_idx`` [1, l]; each starts from its
    own initial state. Returns ``out`` [b, l, h, p], plus the final state of each
    batch row (``return_final_states``) and of each packed sequence
    (``return_varlen_states``), in fp32.
    """
    batch, seqlen, nheads, headdim = x.shape
    cu = _cu_seqlens(x, cu_seqlens, seq_idx)
    num_seqs = batch if cu is None else cu.numel() - 1
    _check_inputs(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, num_seqs)
    if return_varlen_states and cu_seqlens is None and seq_idx is None:
        raise ValueError("return_varlen_states needs cu_seqlens or seq_idx")
    dstate = B.shape[3]

    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)
    if tuple(dt_limit) != (0.0, float("inf")):
        dt = dt.clamp(*dt_limit)
    if initial_states is None:
        initial_states = x.new_zeros(
            num_seqs, nheads, headdim, dstate, dtype=torch.float32
        )
    d_head = (
        D.float() if D is not None and D.dim() == 1 else A.new_zeros(nheads).float()
    )

    if cu is None:
        layout = _uniform_layout(batch, seqlen, chunk_size, x.device)
    else:
        layout = _layout(cu, chunk_size)
    if layout.real_tokens != batch * seqlen:
        raise ValueError(f"cu_seqlens must end at seqlen={seqlen}")
    num_tokens = batch * seqlen
    _check_buffers(
        layout.num_tokens,
        nheads,
        headdim,
        B.shape[2],
        dstate,
        layout.num_tokens // chunk_size,
    )
    if config is not None:
        config = {**default_config(headdim, dstate, chunk_size), **config}

    def flat(t):
        return _pad(t.reshape(num_tokens, *t.shape[2:]).contiguous(), layout)

    if num_tokens == 0:
        out, states = x.new_empty(x.shape), initial_states.float()
    else:
        out, states = _ChunkScan.apply(
            flat(x),
            flat(dt),
            A.float().contiguous(),
            flat(B),
            flat(C),
            d_head,
            initial_states.float().contiguous(),
            layout,
            chunk_size,
            config,
        )
        out = _unpad(out, layout).reshape(x.shape)
    if D is not None and D.dim() == 2:
        out = out + (x * D.to(x.dtype)).to(out.dtype)
    if z is not None:
        out = out * torch.nn.functional.silu(z.to(out.dtype))
    results = [out]
    if return_final_states:
        results.append(
            states if cu_seqlens is None and seq_idx is None else states[-1:]
        )
    if return_varlen_states:
        results.append(states)
    return results[0] if len(results) == 1 else tuple(results)
