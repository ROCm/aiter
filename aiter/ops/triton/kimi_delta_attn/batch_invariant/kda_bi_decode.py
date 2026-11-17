# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant KDA decode: one new token per request, bitwise equal to row p of
kda_bi_fwd (Gluon kernels, gfx950)."""

import functools
from dataclasses import dataclass

import torch

from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.batch_invariant.kda_bi_decode import (
    _kda_bi_cache_derive_kernel,
    _kda_bi_cache_fill_kernel,
    _kda_bi_cache_vnew_kernel,
    _kda_bi_decode_intra_kernel,
    _kda_bi_decode_out_kernel,
    _kda_bi_decode_scatter_kernel,
    _kda_bi_decode_step_kernel,
)
from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi import (
    _config as _kda_bi_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = [
    "KdaBiDecodeCache",
    "kda_bi_cache_fill",
    "kda_bi_cache_reset",
    "kda_bi_decode",
    "kda_bi_decode_cache",
]

BT = 64  # chunk length of the prefill contract
HEAD_DIM = 128


@functools.cache
def _config(name: str) -> dict:
    """Pinned launch values of one kernel."""
    if get_arch() != "gfx950":
        raise NotImplementedError("kda_bi_decode: Gluon kernels for gfx950 only")
    path = resolve_config_dir("attention", "KDA-BI-DECODE", backend="gluon")
    cfg = load_config_json(f"{path}/DEFAULT.json")[name]
    return dict(cfg)


@dataclass
class KdaBiDecodeCache:
    """Open-chunk rows of each slot ([n_slots, H, 64, ...]), all as kda_bi computes them:
    q, k (bf16, L2-normed), v (bf16), g (fp32 gate before the scan), beta (fp32); qg, w, u
    (bf16), the fp32 solve rows xw, xu, aqk (bf16 [.., 64, 64]) and the gate cumsum gc
    (fp32, log2) and v_new (bf16) of the rows already seen; hb: bf16(h_b), transposed
    ([n_slots, H, V, K]); ai: rows of the open block's (I + L)^-1 before the + I (fp32
    [n_slots, H, 64, 16]); pos: [n_slots, H] int32, the next row.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    qg: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    xw: torch.Tensor
    xu: torch.Tensor
    aqk: torch.Tensor
    gc: torch.Tensor
    vn: torch.Tensor
    hb: torch.Tensor
    ai: torch.Tensor
    pos: torch.Tensor

    @property
    def n_slots(self) -> int:
        return self.q.shape[0]

    @property
    def num_heads(self) -> int:
        return self.q.shape[1]


def kda_bi_decode_cache(n_slots: int, num_heads: int, device) -> KdaBiDecodeCache:
    """Allocate an empty decode cache (every slot at row 0): 284 KiB per slot and head."""
    z = functools.partial(torch.zeros, device=device)
    shape = (n_slots, num_heads, BT, HEAD_DIM)
    bf = functools.partial(z, shape, dtype=torch.bfloat16)
    return KdaBiDecodeCache(
        q=bf(), k=bf(), v=bf(), g=z(shape, dtype=torch.float32),
        beta=z((n_slots, num_heads, BT), dtype=torch.float32),
        qg=bf(), w=bf(), u=bf(),
        xw=z(shape, dtype=torch.float32), xu=z(shape, dtype=torch.float32),
        aqk=z((n_slots, num_heads, BT, BT), dtype=torch.bfloat16),
        gc=z(shape, dtype=torch.float32),
        vn=bf(),
        hb=z((n_slots, num_heads, HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16),
        ai=z((n_slots, num_heads, BT, 16), dtype=torch.float32),
        pos=z((n_slots, num_heads), dtype=torch.int32),
    )  # fmt: skip


def kda_bi_cache_reset(cache: KdaBiDecodeCache, slots: torch.Tensor) -> None:
    """Start a new request in each slot: its open chunk is empty. Stale rows are masked."""
    _check_index("slots", slots, None)
    cache.pos.index_fill_(0, slots.long(), 0)


def _check_index(name, t, n):
    if t.dtype != torch.int32 or t.dim() != 1:
        raise ValueError(f"kda_bi_decode: {name} must be a 1-D int32 tensor")
    if n is not None and t.numel() != n:
        raise ValueError(f"kda_bi_decode: {name} must have {n} entries")


def _check_rows(name, x, R, H, dtypes):
    """x: [R, H, D] (or [R, H] for beta) with a unit last stride."""
    if x.dtype not in dtypes:
        raise ValueError(f"kda_bi_decode: {name} must be {dtypes}, got {x.dtype}")
    if x.shape[:2] != (R, H):
        raise ValueError(f"kda_bi_decode: {name} must be [{R}, {H}, ...]")
    if x.dim() == 3 and (x.shape[2] != HEAD_DIM or x.stride(2) != 1):
        raise ValueError(
            f"kda_bi_decode: {name} must have head dim {HEAD_DIM} with unit stride"
        )


def _check_state(state, H):
    if state.dtype != torch.float32:
        raise ValueError("kda_bi_decode: state must be float32")
    if state.dim() != 4 or state.shape[1:] != (H, HEAD_DIM, HEAD_DIM):
        raise ValueError(f"kda_bi_decode: state must be [n, {H}, 128, 128]")
    if state.stride(3) != 1 or state.stride(2) != HEAD_DIM:
        raise ValueError("kda_bi_decode: each [K, V] state block must be contiguous")


def _check_gate(A_log, dt_bias, H):
    if A_log.numel() != H or dt_bias.numel() != H * HEAD_DIM:
        raise ValueError(
            "kda_bi_decode: A_log must have H and dt_bias H * 128 elements"
        )


def _check_lower_bound(lower_bound):
    if not -5.0 <= lower_bound < 0.0:
        raise ValueError("kda_bi_decode: lower_bound must be in [-5, 0)")


def kda_bi_cache_fill(
    cache: KdaBiDecodeCache,
    slots: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    boundary_state: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    state_slots: torch.Tensor | None = None,
    scale: float | None = None,
) -> None:
    """After kda_bi_fwd on N prompts: put each prompt's open-chunk rows (its last T mod 64
    tokens) in its cache slot. state[state_slots] must hold each prompt's boundary state
    (the state after its last complete chunk), or pass it as boundary_state ([N, H, K, V])
    to have it written there. The state must not change while a chunk is open.

    q, k, v, g: [1, T, H, 128] bf16 and beta [1, T, H], A_log, dt_bias, lower_bound and
    scale as given to kda_bi_fwd. cu_seqlens: [N + 1] int32 or int64; slots, state_slots:
    [N] int32."""
    N = cu_seqlens.numel() - 1
    H = cache.num_heads
    T = q.shape[1]
    _check_index("slots", slots, N)
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g)):
        _check_rows(name, x[0], T, H, (torch.bfloat16,))
    _check_rows("beta", beta[0], T, H, (torch.bfloat16, torch.float32))
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("kda_bi_decode: cu_seqlens must be int32 or int64")
    _check_lower_bound(lower_bound)
    _check_gate(A_log, dt_bias, H)
    scale = HEAD_DIM**-0.5 if scale is None else scale
    cfg = _config("fill")
    _kda_bi_cache_fill_kernel[(N, H)](
        v, beta, v.stride(1), v.stride(2), beta.stride(1), beta.stride(2),
        cu_seqlens, slots, cache.pos, cache.v, cache.beta, H,
        D=HEAD_DIM, BT=BT, num_warps=cfg["num_warps"],
    )  # fmt: skip
    c = cache
    _kda_bi_cache_derive_kernel[(N, H)](
        q, k, g, q.stride(1), k.stride(1), g.stride(1), q.stride(2), k.stride(2),
        g.stride(2), c.q, c.k, c.g, c.v, c.beta,
        A_log.to(torch.float32).contiguous(), dt_bias.to(torch.float32).contiguous(),
        cu_seqlens, slots,
        c.qg, c.w, c.u, c.xw, c.xu, c.aqk, c.gc, c.ai,
        float(lower_bound), float(scale), H,
        D=HEAD_DIM, BT=BT, **_kda_bi_config("intra"), enable_fp_fusion=False,
    )  # fmt: skip
    if state is None:
        raise ValueError("kda_bi_decode: cache_fill needs state (v_new uses h_b)")
    _check_state(state, H)
    idx = slots if state_slots is None else state_slots
    _check_index("state_slots", idx, N)
    if boundary_state is not None:
        state.index_copy_(0, idx.long(), boundary_state.to(state.dtype))
    co = _kda_bi_config("out")
    _kda_bi_cache_vnew_kernel[(N, H, HEAD_DIM // co["BV"])](
        c.w, c.u, c.vn, c.hb, cu_seqlens, slots, idx, state, state.stride(0),
        state.stride(1), H,
        D=HEAD_DIM, BT=BT, **co, enable_fp_fusion=False,
    )  # fmt: skip


def kda_bi_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state: torch.Tensor,
    cache: KdaBiDecodeCache,
    slots: torch.Tensor,
    state_slots: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Append one token to each request and return its output, bitwise equal to the
    same row of kda_bi_fwd on the whole sequence.

    q, k, v, g: [R, H, 128] bf16 raw (pre-L2-norm q, k; pre-gate g), last dim contiguous.
    beta: [R, H] bf16 or fp32 (activated). state: [n, H, 128, 128] fp32, the boundary state
    of each request (any slot and head strides, contiguous [K, V] blocks); it changes only
    when a token closes a 64-token chunk. slots: [R] int32 cache slots (distinct);
    state_slots: [R] int32 rows of state (default: slots). Returns o [R, H, 128] bf16.
    No host sync: a call is CUDA-graph capturable. All offsets are 64-bit."""
    R, H = q.shape[0], q.shape[1]
    if H != cache.num_heads:
        raise ValueError("kda_bi_decode: head count differs from the cache")
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g)):
        _check_rows(name, x, R, H, (torch.bfloat16,))
    _check_rows("beta", beta, R, H, (torch.bfloat16, torch.float32))
    _check_state(state, H)
    _check_lower_bound(lower_bound)
    _check_gate(A_log, dt_bias, H)
    _check_index("slots", slots, R)
    state_slots = slots if state_slots is None else state_slots
    _check_index("state_slots", state_slots, R)
    if scale is None:
        scale = HEAD_DIM**-0.5
    if R == 0:
        return torch.empty((0, H, HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    c = cache
    o = torch.empty((R, H, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    kg = torch.empty((R, H, BT, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    glast = torch.empty((R, H, HEAD_DIM), dtype=torch.float32, device=q.device)
    a_log = A_log.to(torch.float32).contiguous()
    dtb = dt_bias.to(torch.float32).contiguous()
    strides = (
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        g.stride(0), g.stride(1), beta.stride(0), beta.stride(1),
    )  # fmt: skip
    # num_warps fixes the kda_bi helpers' layouts, so it comes from kda_bi; BV and the
    # launch split do not change the bits.
    nw = _kda_bi_config("intra")["num_warps"]
    if _kda_bi_config("out")["num_warps"] != nw:
        raise NotImplementedError(
            "kda_bi_decode: kda_bi intra and out num_warps differ"
        )
    bv = _config("out")["bv"]
    st = (state, state.stride(0), state.stride(1))
    if R <= _config("step")["max_requests_one_launch"]:
        _kda_bi_decode_step_kernel[(R, H)](
            q, k, v, g, beta, a_log, dtb, *strides,
            slots, state_slots, c.pos, c.q, c.k, c.v, c.g, c.beta,
            c.qg, c.w, c.u, c.xw, c.xu, c.aqk, c.gc, c.ai, c.vn, c.hb, kg, glast, *st, o,
            float(lower_bound), float(scale), H,
            D=HEAD_DIM, BT=BT, BV=_config("step")["bv"], num_warps=nw,
            enable_fp_fusion=False,
        )  # fmt: skip
        return o
    pcur = torch.empty((R, H), dtype=torch.int32, device=q.device)
    _kda_bi_decode_scatter_kernel[(R, H)](
        q, k, v, g, beta, a_log, dtb, *strides,
        slots, c.pos, pcur, c.q, c.k, c.v, c.g, c.beta, float(lower_bound), H,
        D=HEAD_DIM, BT=BT, num_warps=nw, enable_fp_fusion=False,
    )  # fmt: skip
    _kda_bi_decode_intra_kernel[(R, H)](
        c.q, c.k, c.v, c.g, c.beta, slots, c.pos, pcur,
        c.qg, c.w, c.u, c.xw, c.xu, c.aqk, c.gc, c.ai, kg, glast, float(scale), H,
        D=HEAD_DIM, BT=BT, num_warps=nw, enable_fp_fusion=False,
    )  # fmt: skip
    _kda_bi_decode_out_kernel[(R, H, HEAD_DIM // bv)](
        c.qg, c.w, c.u, c.aqk, c.vn, c.hb, kg, glast, slots, state_slots, pcur, *st,
        o, float(scale), H,
        D=HEAD_DIM, BT=BT, BV=bv, num_warps=nw, enable_fp_fusion=False,
    )  # fmt: skip
    return o
