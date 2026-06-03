# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL grouped jagged x dense BMM with a fused bias add.

Computes, for a jagged batch of ``B`` variable-length sequences packed into a
single ``(L, D)`` tensor with ``B + 1`` row offsets::

    out[s:e] = jagged[s:e] @ dense[b] + bias[b]        # broadcast bias (B, K)
    out[s:e] = jagged[s:e] @ dense[b] + bias[s:e]      # elementwise bias (L, K)

where ``[s, e)`` is the row span of batch ``b``. This is the FlyDSL kernel
behind ``jagged_dense_bmm_broadcast_add`` (HSTU / generative-recommenders).

A single HGEMM-style kernel (``kernels/jagged_dense_bmm.py``) backs every shape.
Per-shape tile configs come from ``jagged_dense_bmm_dispatch.json`` (built by an
offline autotune); shapes not in the table fall back to an empirically derived,
D-bucketed heuristic. Callers normally pass only the tensors and let dispatch
pick the tiling, but every tile knob can be overridden by keyword for tuning.

Usage::

    from aiter.ops.flydsl.jagged_dense_bmm_dispatch import (
        flydsl_jagged_dense_bmm_broadcast_add,
    )
    out = flydsl_jagged_dense_bmm_broadcast_add(
        max_seq_len=max_seq_len,
        seq_offsets=seq_offsets,   # int32/int64, shape (B + 1,)
        jagged=jagged,             # bf16, shape (L, D)
        dense=dense,               # bf16, shape (B, D, K)
        bias=bias,                 # bf16, shape (B, K)
    )
"""

from __future__ import annotations

import functools
import json
import os
import weakref
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

import flydsl.expr as fx

from .gemm_kernels import _ptr_view_safe, _normalize_launch_stream
from .kernels.jagged_dense_bmm import compile_jagged_dense_bmm_kernel
from .kernels.tensor_shim import _run_compiled
from .utils import is_flydsl_available

__all__ = [
    "flydsl_jagged_dense_bmm_broadcast_add",
    "flydsl_jagged_dense_bmm_add",
]

# Heuristic fallback order for shapes absent from the dispatch table. Empirically
# tuned across the gfx942 catalog: 128x128x32 with B-operand staged through LDS
# dominates; small (D, K) <= 256 prefer 64x64x32.
_HEURISTIC_CONFIGS: Tuple[dict, ...] = (
    dict(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        stages=2,
        block_m_warps=2,
        block_n_warps=2,
        block_k_warps=1,
        b_to_lds=True,
        waves_per_eu=0,
    ),
    dict(tile_m=64, tile_n=64, tile_k=32, b_to_lds=True),
    dict(tile_m=128, tile_n=64, tile_k=32, b_to_lds=True),
    dict(tile_m=128, tile_n=128, tile_k=64, b_to_lds=False),
)

_DISPATCH_CACHE: dict[tuple, dict] = {}
_DISPATCH_TABLE: Optional[dict] = None
# data_ptr -> (weakref(dense), {subkey: panel}). Keyed on data_ptr with a
# weakref-identity guard: a hit is valid only if the *same live tensor object*
# still occupies that address. This avoids both (a) id()/ptr reuse across freed
# tensors returning a STALE transposed panel and (b) torch.Tensor.__eq__
# (element-wise) breaking dict/WeakKey lookups.
_DENSE_B_CACHE: dict[int, tuple] = {}


def _dispatch_json_paths() -> Tuple[Path, ...]:
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON")
    if env:
        return (Path(env),)
    return (Path(__file__).resolve().parent / "jagged_dense_bmm_dispatch.json",)


def _load_dispatch_table() -> dict:
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is not None:
        return _DISPATCH_TABLE
    for path in _dispatch_json_paths():
        if path.is_file():
            data = json.loads(path.read_text())
            _DISPATCH_TABLE = {
                "winners": dict(data.get("winners") or {}),
                "fallback": dict(data.get("fallback") or {}),
            }
            return _DISPATCH_TABLE
    _DISPATCH_TABLE = {"winners": {}, "fallback": {}}
    return _DISPATCH_TABLE


def _shape_id(*, batch_size: int, d_in: int, k_out: int, max_seq_len: int) -> str:
    return f"B{batch_size}D{d_in}K{k_out}N{max_seq_len}"


def _pick_tile_n(k_out: int) -> int:
    if k_out % 256 == 0:
        return 256
    if k_out % 128 == 0:
        return 128
    if k_out % 64 == 0:
        return 64
    raise ValueError(f"output dim K={k_out} must be divisible by 64/128/256")


def _pick_tile_k(d_in: int) -> int:
    if d_in % 64 == 0 and d_in // 64 >= 2:
        return 64
    if d_in % 32 == 0:
        return 32
    raise ValueError(f"reduction dim D={d_in} must be divisible by 32 or 64")


def _config_valid(cfg: dict, *, d_in: int, k_out: int) -> bool:
    return k_out % cfg["tile_n"] == 0 and d_in % cfg["tile_k"] == 0


def _normalize_cfg(cfg: dict) -> dict:
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": int(cfg["tile_n"]),
        "tile_k": int(cfg["tile_k"]),
        "stages": int(cfg.get("stages", 2)),
        "block_m_warps": int(cfg.get("block_m_warps", 2)),
        "block_n_warps": int(cfg.get("block_n_warps", 2)),
        "block_k_warps": int(cfg.get("block_k_warps", 1)),
        "waves_per_eu": int(cfg.get("waves_per_eu", 0)),
        "b_to_lds": bool(cfg.get("b_to_lds", False)),
        # XCD/L2-aware grid remap (HipKittens Algorithm 1): 0=off, >0=group chunk
        # per XCD. The bijection requires batch % (8*chunk)==0, enforced at launch.
        "xcd_remap": int(cfg.get("xcd_remap", 0)),
    }


def _lds_fits(*, tile_m: int, tile_n: int, tile_k: int, stages: int, b_to_lds: bool) -> bool:
    as_b = stages * tile_m * tile_k * 2
    bs_b = stages * tile_n * tile_k * 2 if b_to_lds else 0
    c_b = tile_m * tile_n * 2
    return max(as_b + bs_b, c_b) <= 65536


def _d_bucket(d_in: int) -> str:
    if d_in <= 256:
        return "d_le_256"
    if d_in <= 512:
        return "d_le_512"
    if d_in <= 1024:
        return "d_le_1024"
    return "d_big"


# Strong default derived empirically from the gfx942 catalog tune: 128x128x32
# with the B operand staged through LDS dominates; small (D, K) <= 256 prefer
# 64x64x32.
_STRONG_DEFAULT = dict(
    tile_m=128, tile_n=128, tile_k=32, stages=2,
    block_m_warps=2, block_n_warps=2, block_k_warps=1, b_to_lds=True, waves_per_eu=0,
)


def _strong_fallback(*, d_in: int, k_out: int, fallback_rules: dict) -> dict:
    bucket = _d_bucket(d_in)
    base = dict(_STRONG_DEFAULT)
    by_bucket = (fallback_rules or {}).get("by_d_bucket") or {}
    if bucket in by_bucket and isinstance(by_bucket[bucket], dict):
        base.update(by_bucket[bucket].get("config") or {})
    elif (fallback_rules or {}).get("global"):
        base.update(fallback_rules["global"])

    stages = int(base.get("stages", 2))
    bmw = int(base.get("block_m_warps", 2))
    bnw = int(base.get("block_n_warps", 2))

    # tile_n: largest candidate dividing K and compatible with bnw warps.
    tile_n = 0
    for cand in (int(base.get("tile_n", 128)), 128, 64):
        if k_out % cand == 0 and cand % (bnw * 16) == 0:
            tile_n = cand
            break
    if tile_n == 0:
        bnw = 1
        tile_n = _pick_tile_n(k_out)

    # tile_k: prefer 32 (proven best); require D%tk==0 and D//tk>=stages.
    tile_k = 0
    for cand in (int(base.get("tile_k", 32)), 32, 64):
        if d_in % cand == 0 and d_in // cand >= stages:
            tile_k = cand
            break
    if tile_k == 0:
        tile_k = _pick_tile_k(d_in)

    # tile_m: keep multiple of bmw*16; small (D, K) <= 256 -> 64.
    tile_m = int(base.get("tile_m", 128))
    if d_in <= 256 and k_out <= 256:
        tile_m = min(tile_m, 64)
    if tile_m % (bmw * 16) != 0:
        tile_m = bmw * 16

    # Ensure LDS fit; degrade b_to_lds, then tile_m.
    b_to_lds = bool(base.get("b_to_lds", True))
    if not _lds_fits(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages, b_to_lds=b_to_lds):
        b_to_lds = False
        while tile_m > bmw * 16 and not _lds_fits(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages, b_to_lds=b_to_lds
        ):
            tile_m //= 2

    cfg = dict(
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages,
        block_m_warps=bmw, block_n_warps=bnw, block_k_warps=1,
        b_to_lds=b_to_lds, waves_per_eu=int(base.get("waves_per_eu", 0)),
    )
    return _normalize_cfg(cfg)


def _heuristic_dispatch(*, d_in: int, k_out: int) -> dict:
    rules = _load_dispatch_table().get("fallback") or {}
    cfg = _strong_fallback(d_in=d_in, k_out=k_out, fallback_rules=rules)
    if _config_valid(cfg, d_in=d_in, k_out=k_out):
        return cfg
    # Last resort: walk the static heuristic list, then a divisibility-safe pick.
    for cfg in _HEURISTIC_CONFIGS:
        if _config_valid(cfg, d_in=d_in, k_out=k_out):
            return _normalize_cfg(cfg)
    tile_n = _pick_tile_n(k_out)
    tile_m = 64 if d_in * tile_n * 2 > 32768 else 128
    return _normalize_cfg(
        dict(tile_m=tile_m, tile_n=tile_n, tile_k=_pick_tile_k(d_in))
    )


def _resolve_dispatch(
    *,
    batch_size: int,
    d_in: int,
    k_out: int,
    max_seq_len: int,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    stages: Optional[int] = None,
    block_m_warps: Optional[int] = None,
    block_n_warps: Optional[int] = None,
    block_k_warps: Optional[int] = None,
    b_to_lds: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
) -> dict:
    """Return the full tile config for a shape (explicit override > table > heuristic)."""
    explicit = (
        tile_m,
        tile_n,
        tile_k,
        stages,
        block_m_warps,
        block_n_warps,
        block_k_warps,
        b_to_lds,
        waves_per_eu,
    )
    if all(v is not None for v in explicit):
        return _normalize_cfg(
            dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                stages=stages,
                block_m_warps=block_m_warps,
                block_n_warps=block_n_warps,
                block_k_warps=block_k_warps,
                b_to_lds=b_to_lds,
                waves_per_eu=waves_per_eu,
            )
        )

    key = (d_in, k_out, max_seq_len, batch_size)
    cached = _DISPATCH_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    table = _load_dispatch_table()
    sid = _shape_id(batch_size=batch_size, d_in=d_in, k_out=k_out, max_seq_len=max_seq_len)
    entry = table["winners"].get(sid)
    if entry is not None:
        cfg = _normalize_cfg(entry)
        if _config_valid(cfg, d_in=d_in, k_out=k_out):
            _DISPATCH_CACHE[key] = dict(cfg)
            return dict(cfg)

    cfg = _heuristic_dispatch(d_in=d_in, k_out=k_out)
    _DISPATCH_CACHE[key] = dict(cfg)
    return dict(cfg)


def _dense_b_panel(
    dense: Tensor,
    *,
    batch_size: int,
    n_dim: int,
    k_dim: int,
    transpose_dense: bool,
) -> Tensor:
    """Layout dense as (B*N, K) row-major; transpose of genrec ``(B, D, K)``.

    Cached per live ``dense`` object so static weights transpose once (steady
    state) while distinct tensors never collide (correctness across sweeps).
    """
    if not transpose_dense:
        return dense.contiguous().reshape(batch_size * n_dim, k_dim)
    ptr = dense.data_ptr()
    subkey = (batch_size, n_dim, k_dim, int(getattr(dense, "_version", 0)))
    entry = _DENSE_B_CACHE.get(ptr)
    sub = None
    if entry is not None:
        wr, cached_sub = entry
        if wr() is dense:  # same live tensor still at this address
            hit = cached_sub.get(subkey)
            if hit is not None:
                return hit
            sub = cached_sub
        # else: address reused by a different tensor -> stale, refresh below
    if sub is None:
        sub = {}
        try:
            _DENSE_B_CACHE[ptr] = (weakref.ref(dense), sub)
        except TypeError:
            sub = None  # non-weakrefable; skip caching (always correct)
    dense_b = dense.transpose(1, 2).contiguous().reshape(batch_size * n_dim, k_dim)
    if sub is not None:
        sub[subkey] = dense_b
    return dense_b


@functools.lru_cache(maxsize=256)
def _get_launcher(
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    b_to_lds: bool,
    waves_per_eu: int,
    has_bias: bool,
    elementwise: bool,
    xcd_remap: int = 0,
    m_tiles: int = 0,
):
    return compile_jagged_dense_bmm_kernel(
        "bf16",
        n,
        k,
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        STAGES=stages,
        BLOCK_M_WARPS=block_m_warps,
        BLOCK_N_WARPS=block_n_warps,
        BLOCK_K_WARPS=block_k_warps,
        B_TO_LDS=b_to_lds,
        WAVES_PER_EU=waves_per_eu,
        HAS_BIAS=has_bias,
        ELEMENTWISE=elementwise,
        XCD_REMAP=xcd_remap,
        M_TILES=m_tiles,
    )


def _launch_jagged_dense_bmm(
    *,
    max_seq_len: int,
    seq_offsets: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Optional[Tensor],
    out: Tensor,
    has_bias: bool,
    elementwise: bool,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    stages: Optional[int] = None,
    block_m_warps: Optional[int] = None,
    block_n_warps: Optional[int] = None,
    block_k_warps: Optional[int] = None,
    b_to_lds: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
    out_n: Optional[int] = None,
    red_k: Optional[int] = None,
    transpose_dense: bool = True,
) -> None:
    B = dense.shape[0]
    d_in = dense.shape[1]
    k_out = dense.shape[2]
    n_dim = out_n if out_n is not None else k_out
    k_dim = red_k if red_k is not None else d_in

    disp = _resolve_dispatch(
        batch_size=B,
        d_in=k_dim,
        k_out=n_dim,
        max_seq_len=max_seq_len,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        b_to_lds=b_to_lds,
        waves_per_eu=waves_per_eu,
    )

    dense_b = _dense_b_panel(
        dense,
        batch_size=B,
        n_dim=n_dim,
        k_dim=k_dim,
        transpose_dense=transpose_dense,
    )
    if not seq_offsets.is_contiguous():
        seq_offsets = seq_offsets.contiguous()
    if seq_offsets.dtype != torch.int32:
        seq_offsets = seq_offsets.to(torch.int32)

    launch_stream = _normalize_launch_stream(jagged.device, stream)
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=jagged.device)
    if has_bias:
        bias_arg = _ptr_view_safe(bias if elementwise else bias.reshape(B * k_out))
    else:
        bias_arg = _ptr_view_safe(dummy_i32)

    m_tiles = (max_seq_len + disp["tile_m"] - 1) // disp["tile_m"]
    # XCD/L2-aware grid remap: only enable when the bijection holds
    # (batch % (8*chunk)==0); otherwise fall back to identity (off).
    xcd_remap = int(disp.get("xcd_remap", 0))
    if xcd_remap > 0 and (B % (8 * xcd_remap) != 0):
        xcd_remap = 0
    m_tiles_hint = m_tiles if xcd_remap > 0 else 0
    launcher = _get_launcher(
        n_dim,
        k_dim,
        disp["tile_m"],
        disp["tile_n"],
        disp["tile_k"],
        disp["stages"],
        disp["block_m_warps"],
        disp["block_n_warps"],
        disp["block_k_warps"],
        disp["b_to_lds"],
        disp["waves_per_eu"],
        has_bias,
        elementwise,
        xcd_remap,
        m_tiles_hint,
    )
    _run_compiled(
        launcher,
        _ptr_view_safe(out),
        _ptr_view_safe(jagged),
        _ptr_view_safe(dense_b),
        bias_arg,
        _ptr_view_safe(seq_offsets),
        max_seq_len,
        B,
        m_tiles,
        _ptr_view_safe(dummy_i32),
        _ptr_view_safe(dummy_i32),
        fx.Stream(launch_stream),
    )


def _bwd_dense_bias_ref(
    seq_offsets: Tensor,
    jagged: Tensor,
    d_out: Tensor,
    *,
    elementwise: bool,
) -> Tuple[Tensor, Tensor]:
    """Reference d_dense / d_bias matching Triton ``_jagged_jagged_bmm_reduce_sum``."""
    B = int(seq_offsets.numel()) - 1
    d_in = jagged.shape[1]
    k_out = d_out.shape[1]
    d_dense = torch.zeros((B, d_in, k_out), dtype=d_out.dtype, device=d_out.device)
    if elementwise:
        d_bias = d_out
    else:
        d_bias = torch.zeros((B, k_out), dtype=d_out.dtype, device=d_out.device)
        for b in range(B):
            s, e = int(seq_offsets[b].item()), int(seq_offsets[b + 1].item())
            if e > s:
                d_bias[b] = d_out[s:e].sum(dim=0)
    for b in range(B):
        s, e = int(seq_offsets[b].item()), int(seq_offsets[b + 1].item())
        if e > s:
            d_dense[b] = jagged[s:e].transpose(0, 1) @ d_out[s:e]
    return d_dense, d_bias


class _FlydslJaggedDenseBmmAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, max_seq_len, seq_offsets, jagged, dense, bias, elementwise):
        if jagged.dtype != torch.bfloat16:
            raise ValueError("only bf16 is supported")
        if not jagged.is_cuda:
            raise ValueError("tensors must be on CUDA/ROCm device")
        L, d_in = jagged.shape
        B, d_chk, k_out = dense.shape
        if d_chk != d_in:
            raise ValueError("jagged.shape[1] must match dense.shape[1] (D)")
        if seq_offsets.shape[0] != B + 1:
            raise ValueError("seq_offsets must have length B+1")
        if elementwise and bias.shape != (L, k_out):
            raise ValueError("elementwise bias must be (L, K)")
        if not elementwise and bias.shape != (B, k_out):
            raise ValueError("broadcast bias must be (B, K)")
        if not jagged.is_contiguous():
            jagged = jagged.contiguous()
        if not bias.is_contiguous():
            bias = bias.contiguous()
        out = torch.empty((L, k_out), dtype=jagged.dtype, device=jagged.device)
        _launch_jagged_dense_bmm(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
            out=out,
            has_bias=True,
            elementwise=elementwise,
        )
        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.max_seq_len = max_seq_len
        ctx.B, ctx.K, ctx.D = B, k_out, d_in
        ctx.elementwise = elementwise
        return out

    @staticmethod
    def backward(ctx, d_out):
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        _launch_jagged_dense_bmm(
            max_seq_len=ctx.max_seq_len,
            seq_offsets=seq_offsets,
            jagged=d_out,
            dense=dense,
            bias=None,
            out=d_jagged,
            has_bias=False,
            elementwise=False,
            out_n=ctx.D,
            red_k=ctx.K,
            transpose_dense=False,
        )
        d_dense, d_bias = _bwd_dense_bias_ref(
            seq_offsets, jagged, d_out, elementwise=ctx.elementwise
        )
        return None, None, d_jagged, d_dense, d_bias, None


def flydsl_jagged_dense_bmm_add(
    *,
    max_seq_len: int,
    seq_offsets: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Tensor,
    elementwise: bool = False,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    stages: Optional[int] = None,
    block_m_warps: Optional[int] = None,
    block_n_warps: Optional[int] = None,
    block_k_warps: Optional[int] = None,
    b_to_lds: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tensor:
    """Grouped jagged x dense BMM plus a fused bias add (broadcast or elementwise).

    Tile knobs default to ``None`` (per-shape dispatch); pass them to force a
    specific tiling for tuning. ``elementwise=True`` adds a per-row ``(L, K)``
    bias instead of a per-batch ``(B, K)`` broadcast bias.
    """
    if not is_flydsl_available():
        raise RuntimeError("flydsl is not installed")
    if out is None:
        if jagged.requires_grad or dense.requires_grad or bias.requires_grad:
            return _FlydslJaggedDenseBmmAddFunction.apply(
                max_seq_len, seq_offsets, jagged, dense, bias, elementwise
            )
        out = torch.empty(
            (jagged.shape[0], dense.shape[2]), dtype=jagged.dtype, device=jagged.device
        )
    elif out.shape != (jagged.shape[0], dense.shape[2]):
        raise ValueError("out shape mismatch")
    _launch_jagged_dense_bmm(
        max_seq_len=max_seq_len,
        seq_offsets=seq_offsets,
        jagged=jagged,
        dense=dense,
        bias=bias,
        out=out,
        has_bias=True,
        elementwise=elementwise,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        b_to_lds=b_to_lds,
        waves_per_eu=waves_per_eu,
        stream=stream,
    )
    return out


def flydsl_jagged_dense_bmm_broadcast_add(
    *,
    max_seq_len: int,
    seq_offsets: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Tensor,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    stages: Optional[int] = None,
    block_m_warps: Optional[int] = None,
    block_n_warps: Optional[int] = None,
    block_k_warps: Optional[int] = None,
    b_to_lds: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tensor:
    """Grouped jagged x dense BMM plus a per-batch broadcast bias ``(B, K)``."""
    return flydsl_jagged_dense_bmm_add(
        max_seq_len=max_seq_len,
        seq_offsets=seq_offsets,
        jagged=jagged,
        dense=dense,
        bias=bias,
        elementwise=False,
        out=out,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        b_to_lds=b_to_lds,
        waves_per_eu=waves_per_eu,
        stream=stream,
    )
