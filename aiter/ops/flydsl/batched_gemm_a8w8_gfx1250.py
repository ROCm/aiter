# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 strided-batched mxfp8 GEMMs: mxfp8_128 and mxfp8_32.

mxfp8_128 (``run_bmm_a8w8_mxfp8_128_gfx1250``):

===========  ==========================  ===========================
operand      shape                       notes
===========  ==========================  ===========================
``XQ``       ``[M, B, K]`` fp8           M-outer, K-contiguous
``WQ``       ``[B, N, K]`` fp8           16x16 preshuffled (ops.shuffle.shuffle_weight)
``x_scale``  ``[M, B, K//128]`` e8m0     row-major, same as the 2-D op's contract
``w_scale``  ``[B, N//128, K//128]`` e8m0
``Out``      ``[M, B, N]`` bf16/fp16
===========  ==========================  ===========================

mxfp8_32 (``run_bmm_a8w8_mxfp8_32_gfx1250``): same XQ / WQ / Out, a row-major
``[M, B, K//32]`` e8m0 x_scale (``inverse_rope_group_quant(..., scale_layout=
"row")``), and either w_scale layout:

* 1x32: ``[B, N, K//32]`` e8m0 holding ``shuffle_scale_n32k4(raw, experts_cnt=B)``
  bytes (the n32k4 layout every gfx1250 mxfp8_32 GEMM reads), or
* 32x32: the raw row-major ``[B, N//32, K//32]`` e8m0 block scale, read as is.
  A 32x32 block is a 1x32 scale repeated over its 32 rows, so the kernel loads
  one word per 32 output columns -- no expansion, at load or per call.
"""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_lds_capacity_bytes

from .mxfp8_bpreshuffle_gemm_gfx1250 import BASE_NAME_SUFFIX_RE

BLOCK_K = 128
BLOCK_N = 128
BMM_WMMA_NAME_PREFIX = "flydsl_bmm_mxfp8_128_wmma"
MX32_BLOCK_K = 32
BMM_MX32_WMMA_NAME_PREFIX = "flydsl_bmm_mxfp8_32_wmma"

_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_PRELOAD_MIN_TILE_M = 128
_MAX_HEURISTIC_TILE_M = 128
_LDS_BYTES = get_lds_capacity_bytes("gfx1250")
_SUFFIX_RE = BASE_NAME_SUFFIX_RE
_BMM_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(BMM_WMMA_NAME_PREFIX)}_{_SUFFIX_RE}(?P<preload>_pre)?$"
)
# mxfp8_32 has no whole-K scale preload: its scale panels are 4x (A) and 32x
# (n32k4 B) the mxfp8_128 ones.
_BMM_MX32_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(BMM_MX32_WMMA_NAME_PREFIX)}_{_SUFFIX_RE}$"
)

_launch_gemm_a8w8 = None
_run_compiled = None
_ptr_arg = None
_check_e8m0 = None
_fx = None


def _lazy_import():
    global _launch_gemm_a8w8, _run_compiled, _ptr_arg, _check_e8m0, _fx
    if _launch_gemm_a8w8 is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_gfx1250 import launch_gemm_a8w8
    from .kernels.tensor_shim import _run_compiled as run_compiled
    from .kernels.tensor_shim import check_e8m0, ptr_arg

    _launch_gemm_a8w8 = launch_gemm_a8w8
    _run_compiled = run_compiled
    _ptr_arg = ptr_arg
    _check_e8m0 = check_e8m0
    _fx = fx_mod


def parse_bmm_kernel_name(name: str) -> dict[str, int] | None:
    """Parse a tuned batched kernelName, or ``None`` if it is not one of ours."""
    match = _BMM_KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    groups = match.groupdict()
    preload = groups.pop("preload")
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["preload"] = int(preload is not None)
    return cfg


@functools.lru_cache(maxsize=1024)
def _parse_bmm_mx32_kernel_name_cached(name: str) -> dict[str, int] | None:
    match = _BMM_MX32_KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    cfg = {key: int(value) for key, value in match.groupdict().items()}
    cfg["preload"] = 0
    return cfg


def parse_bmm_mx32_kernel_name(name: str) -> dict[str, int] | None:
    """Parse a batched mxfp8_32 kernelName, or ``None`` if it is not one.

    Every launch parses its kernel name, so the regex result is cached per name;
    each caller gets its own copy of the dict."""
    cfg = _parse_bmm_mx32_kernel_name_cached(name)
    return None if cfg is None else dict(cfg)


def bmm_kernel_name(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    preload: bool = False,
) -> str:
    """Canonical tuned-CSV name for a batched config (split-K/cluster are fixed)."""
    return (
        f"{BMM_WMMA_NAME_PREFIX}_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk1_cm1_cn1"
        f"{'_pre' if preload else ''}"
    )


def _align(value: int, to: int) -> int:
    return -(-value // to) * to


def preload_lds_bytes(c: dict[str, int], k: int) -> int:
    """LDS a config needs with the whole-K scale panels staged."""
    stage_a = _align(c["tile_m"] * (c["tile_k"] + 16), 16)
    stage_b = _align(c["tile_n"] * c["tile_k"], 16)
    pitch = _align(stage_a + stage_b, 1024)
    ks = k // BLOCK_K
    panel = _align(c["tile_m"] * ks, 16) + _align(
        max(1, c["tile_n"] // BLOCK_N) * ks, 16
    )
    c_pad = 8 if c["tile_n"] >= 128 else 0
    c_store = _align(c["tile_m"] * (c["tile_n"] + c_pad) * 2, 128)
    return max(pitch * c["num_buffers"] + panel, c_store)


def bmm_mx32_kernel_name(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
) -> str:
    """Canonical name for a batched mxfp8_32 config."""
    return (
        f"{BMM_MX32_WMMA_NAME_PREFIX}_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk1_cm1_cn1"
    )


def mx32_lds_bytes(c: dict[str, int], w_scale_32x32: bool = False) -> int:
    """LDS the mxfp8_32 kernel allocates for a config (its own arithmetic)."""
    tm, tn, tk = c["tile_m"], c["tile_n"], c["tile_k"]
    stage_a = _align(tm * (tk + 16 * -(-tk // 1024)), 16)
    stage_b = _align(tn * tk, 16)
    stage_sa = _align(tm * (tk // MX32_BLOCK_K), 16)
    stage_sb = _align((tn // 32) * (tk // 32 if w_scale_32x32 else tk), 16)
    pitch = _align(stage_a + stage_b + stage_sa + stage_sb, 1024)
    c_pad = 8 if tn >= 128 else 0
    c_store = _align(tm * (tn + c_pad) * 2, 128)
    return max(pitch * c["num_buffers"], c_store)


def bmm_candidates(b: int, m: int, n: int, k: int) -> list[dict[str, int]]:
    """Every batched config that can legally run this shape, tuner-ordered."""
    from .gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        is_compute_kernel,
        kernel_fits_shape,
        kernels_list,
    )

    out = []
    for ki in kernels_list.values():
        if is_compute_kernel(ki) or ki.split_k != 1:
            continue
        if ki.cluster_m != 1 or ki.cluster_n != 1:
            continue
        if not kernel_fits_shape(ki, m, n, k):
            continue
        out.append(
            {
                "tile_m": ki.tile_m,
                "tile_n": ki.tile_n,
                "tile_k": ki.tile_k,
                "m_warp": ki.m_warp,
                "n_warp": ki.n_warp,
                "num_buffers": ki.num_buffers,
            }
        )
    return out


@functools.lru_cache(maxsize=1024)
def pick_bmm_kernel_name(b: int, m: int, n: int, k: int) -> str:
    """Heuristic config for a shape with no tuned row."""
    cands = bmm_candidates(b, m, n, k)
    if not cands:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] no batched mxfp8_128 kernel fits "
            f"B={b}, M={m}, N={n}, K={k}"
        )
    num_cu = get_cu_num() or 256
    want_tm = min(_MAX_HEURISTIC_TILE_M, max(16, 1 << (m - 1).bit_length()))

    def rank(c):
        wgs = -(-m // c["tile_m"]) * (n // c["tile_n"]) * b
        tail = -(-wgs // num_cu) * num_cu - wgs
        return (
            tail,
            abs(c["tile_m"] - want_tm),
            -c["tile_n"],
            -c["num_buffers"],
            -c["tile_k"],
        )

    c = min(cands, key=rank)
    preload = (
        c["tile_m"] >= _PRELOAD_MIN_TILE_M and preload_lds_bytes(c, k) <= _LDS_BYTES
    )
    return bmm_kernel_name(
        c["tile_m"],
        c["tile_n"],
        c["tile_k"],
        c["m_warp"],
        c["n_warp"],
        c["num_buffers"],
        preload=preload,
    )


@functools.lru_cache(maxsize=1024)
def _pick_bmm_mx32_kernel_name_exact(
    b: int, m: int, n: int, k: int, w_scale_32x32: bool = False
) -> str:
    """Heuristic batched mxfp8_32 config for a shape with no tuned row."""
    # A K that 128 does not divide (DeepSeek-V4.1's shared-expert down
    # projection has K = 576 at TP4) runs as a ragged last k-tile, so configs
    # are chosen for K rounded up to a whole block and must keep enough tiles.
    k_round = -(-k // BLOCK_K) * BLOCK_K
    cands = [
        c
        for c in bmm_candidates(b, m, n, k_round)
        if c["tile_n"] % 32 == 0
        and mx32_lds_bytes(c, w_scale_32x32) <= _LDS_BYTES
        and (k == k_round or -(-k // c["tile_k"]) >= c["num_buffers"])
    ]
    if not cands:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] no batched mxfp8_32 kernel fits "
            f"B={b}, M={m}, N={n}, K={k}"
        )
    num_cu = get_cu_num() or 256

    def rank(c):
        # Waves times a workgroup's cost per K step: its MACs at ~128 per
        # operand byte, its operand bytes (tm + tn), and a fixed overhead.
        # Small M spreads the weight read over many workgroups; large M takes
        # the biggest tiles that fill whole waves. Ties: tile_k 512, three
        # buffers, fewer M warps. Fitted on MI455X over every candidate at
        # DeepSeek-V4.1's linear and wo_a shapes (M 1..8192, B 1/2/8).
        tm, tn = c["tile_m"], c["tile_n"]
        wgs = -(-m // tm) * (n // tn) * b
        cost = -(-wgs // num_cu) * (tm * tn / 128 + tm + tn + 256)
        return (
            cost,
            abs(c["tile_k"].bit_length() - 10),
            abs(c["num_buffers"] - 3),
            c["m_warp"],
        )

    c = min(cands, key=rank)
    return bmm_mx32_kernel_name(
        c["tile_m"],
        c["tile_n"],
        c["tile_k"],
        c["m_warp"],
        c["n_warp"],
        c["num_buffers"],
    )


@functools.cache
def _mx32_m_class_params() -> tuple[tuple[tuple[int, int | None], ...], bool]:
    """How M enters ``_pick_bmm_mx32_kernel_name_exact``.

    The pick reads M only through ``bmm_candidates`` -> ``kernel_fits_shape``
    on the non-compute, split-K 1, 1x1-cluster instances (M parity when an
    instance preshuffles A; ``ceil(M / tile_m)`` against ``_MAX_M_BLOCKS`` when
    ``tile_m < _LARGE_M_TILE_MIN``) and through its rank (``ceil(M / tile_m)``).
    So ``ceil(M / tile_m)`` per candidate tile_m -- saturated just past the
    block limit where that limit already rejects the tile -- plus the parity
    when it matters, determines the pick. Returns
    ``((tile_m, saturation or None), ...), parity_matters``.
    """
    from .gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        _LARGE_M_TILE_MIN,
        _MAX_M_BLOCKS,
        is_compute_kernel,
        kernels_list,
    )

    tiles = set()
    parity = False
    for ki in kernels_list.values():
        if is_compute_kernel(ki) or ki.split_k != 1:
            continue
        if ki.cluster_m != 1 or ki.cluster_n != 1:
            continue
        tiles.add(ki.tile_m)
        parity = parity or bool(ki.a_preshuffle)
    params = tuple(
        (tm, _MAX_M_BLOCKS + 1 if tm < _LARGE_M_TILE_MIN else None)
        for tm in sorted(tiles)
    )
    return params, parity


# (b, n, k, w_scale_32x32, M class) -> kernel name. Prefill hands the GEMM a new
# M nearly every step; keying the heuristic on the M class instead of the exact
# M turns those into hits (a few hundred classes per weight shape vs. every M).
_MX32_PICK_BY_M_CLASS: dict = {}


def pick_bmm_mx32_kernel_name(
    b: int, m: int, n: int, k: int, w_scale_32x32: bool = False
) -> str:
    """Heuristic batched mxfp8_32 config for a shape with no tuned row.

    Same answer as ``_pick_bmm_mx32_kernel_name_exact(b, m, n, k, w_scale_32x32)``,
    memoised per M class (see ``_mx32_m_class_params``)."""
    params, parity = _mx32_m_class_params()
    m_class = tuple(
        (-(-m // tm) if sat is None else min(-(-m // tm), sat)) for tm, sat in params
    )
    key = (b, n, k, bool(w_scale_32x32), m_class, (m & 1) if parity else 0, m > 0)
    name = _MX32_PICK_BY_M_CLASS.get(key)
    if name is None:
        name = _pick_bmm_mx32_kernel_name_exact(b, m, n, k, w_scale_32x32)
        _MX32_PICK_BY_M_CLASS[key] = name
    return name


def _pick_bmm_mx32_cache_clear() -> None:
    _MX32_PICK_BY_M_CLASS.clear()
    _pick_bmm_mx32_kernel_name_exact.cache_clear()
    _mx32_m_class_params.cache_clear()


pick_bmm_mx32_kernel_name.cache_clear = _pick_bmm_mx32_cache_clear


def _check_bmm_operands(
    tag: str, XQ: Tensor, WQ: Tensor, Out: Tensor
) -> tuple[int, int, int, int, str]:
    """Shared operand checks of the batched runners; returns (m, b, n, k, out)."""
    if get_gfx() != "gfx1250":
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] batched {tag} requires gfx1250, got {get_gfx()}"
        )
    if XQ.dim() != 3 or WQ.dim() != 3 or Out.dim() != 3:
        raise RuntimeError(
            "[FlyDSL gfx1250 bmm] XQ/WQ/Out must be 3-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}, {tuple(Out.shape)}"
        )
    m, b, k = XQ.shape
    wb, n, wk = WQ.shape
    if (wb, wk) != (b, k):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] WQ {tuple(WQ.shape)} does not match "
            f"XQ {tuple(XQ.shape)}; expected [{b}, N, {k}]"
        )
    if tuple(Out.shape) != (m, b, n):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] Out must be [{m}, {b}, {n}], "
            f"got {tuple(Out.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 bmm] XQ/WQ must be 1-byte fp8 storage")
    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] unsupported out dtype {Out.dtype}; expected "
            "bf16/fp16"
        )
    if XQ.stride() != (b * k, k, 1):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] XQ must be contiguous [M,B,K], got strides "
            f"{tuple(XQ.stride())}"
        )
    if Out.stride() != (b * n, n, 1):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] Out must be contiguous [M,B,N], got strides "
            f"{tuple(Out.stride())}"
        )
    if not WQ.is_contiguous():
        raise RuntimeError("[FlyDSL gfx1250 bmm] WQ must be contiguous")
    return m, b, n, k, out_dtype


def run_bmm_a8w8_mxfp8_32_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
    w_scale_32x32: bool = False,
) -> Tensor:
    """Run the batched mxfp8_32 GEMM; writes into ``Out`` and returns it.

    ``w_scale_32x32`` selects the raw ``[B, N//32, K//32]`` block scale; else
    w_scale is the n32k4-shuffled 1x32 ``[B, N, K//32]`` (see module doc).
    """
    m, b, n, k, out_dtype = _check_bmm_operands("mxfp8_32", XQ, WQ, Out)
    _lazy_import()
    if n % 32 or k % MX32_BLOCK_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] mxfp8_32 needs N%32==0 and K%{MX32_BLOCK_K}==0, "
            f"got N={n}, K={k}"
        )
    if k % BLOCK_K and not w_scale_32x32:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] K={k} is not a multiple of {BLOCK_K}; a ragged "
            "last k-tile needs the row-major 32x32 w_scale (an n32k4 scale does "
            "not keep its leading K contiguous)"
        )
    k_blocks = k // MX32_BLOCK_K
    _check_e8m0(x_scale, "x_scale", "FlyDSL gfx1250 bmm", (m, b, k_blocks))
    w_shape = (b, n // 32, k_blocks) if w_scale_32x32 else (b, n, k_blocks)
    _check_e8m0(w_scale, "w_scale", "FlyDSL gfx1250 bmm", w_shape)

    name = kernel_name or pick_bmm_mx32_kernel_name(b, m, n, k, w_scale_32x32)
    cfg = parse_bmm_mx32_kernel_name(name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx1250 bmm] unrecognised kernelName: {name!r}")
    if cfg["split_k"] != 1 or cfg["cluster_m"] != 1 or cfg["cluster_n"] != 1:
        raise ValueError(
            f"[FlyDSL gfx1250 bmm] split-K / clusters are not supported on the "
            f"batched path, got {name!r}"
        )
    tile_n, tile_k = cfg["tile_n"], cfg["tile_k"]
    k_tail = k % tile_k != 0
    if n % tile_n or tile_n % 32 or (k_tail and not w_scale_32x32):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs N%{tile_n}==0, tile_n%32==0 and "
            f"K%{tile_k}==0 (or a 32x32 w_scale for a ragged K), got N={n}, K={k}"
        )
    k_tiles = -(-k // tile_k)
    if k_tiles < cfg["num_buffers"]:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs >= {cfg['num_buffers']} K-tiles, "
            f"got {k_tiles}"
        )
    if m == 0:
        return Out

    out_is_f16 = 1 if out_dtype == "f16" else 0
    bmm_spec = (
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        out_is_f16,
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        MX32_BLOCK_K,
        1,  # split_k
        True,  # batched
        0,  # preload_ks
        False,  # a_preshuffle
        False,  # a_scale_preshuffled
        bool(w_scale_32x32),
        k_tail,
    )
    # Cached CompiledFunction per constexpr set; see _run_compiled.
    _run_compiled(
        _launch_gemm_a8w8,
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(x_scale),
        _ptr_arg(w_scale),
        m,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
        n,
        k,
        b * k_blocks,  # stride_ascale_k: x_scale row stride, [M, B, K//32]
        b * k,  # lda: XQ is [M, B, K]
        b * n,  # ldc: Out is [M, B, N]
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        out_is_f16,
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        MX32_BLOCK_K,
        1,  # split_k
        True,  # batched
        0,  # preload_ks
        b,
        False,  # a_preshuffle
        False,  # a_scale_preshuffled
        bool(w_scale_32x32),
        k_tail,
        specialization_key=bmm_spec,
    )
    return Out


def run_bmm_a8w8_mxfp8_128_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
) -> Tensor:
    """Run the batched mxfp8_128 GEMM; writes into ``Out`` and returns it."""
    m, b, n, k, out_dtype = _check_bmm_operands("mxfp8_128", XQ, WQ, Out)
    _lazy_import()
    if n % BLOCK_N or k % BLOCK_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] N/K must be multiples of {BLOCK_N}/{BLOCK_K}, "
            f"got N={n}, K={k}"
        )

    k_blocks = k // BLOCK_K
    _check_e8m0(x_scale, "x_scale", "FlyDSL gfx1250 bmm", (m, b, k_blocks))
    _check_e8m0(w_scale, "w_scale", "FlyDSL gfx1250 bmm", (b, n // BLOCK_N, k_blocks))

    name = kernel_name or pick_bmm_kernel_name(b, m, n, k)
    cfg = parse_bmm_kernel_name(name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx1250 bmm] unrecognised kernelName: {name!r}")
    if cfg["split_k"] != 1 or cfg["cluster_m"] != 1 or cfg["cluster_n"] != 1:
        raise ValueError(
            f"[FlyDSL gfx1250 bmm] split-K / clusters are not supported on the "
            f"batched path, got {name!r}"
        )
    tile_n, tile_k = cfg["tile_n"], cfg["tile_k"]
    preload_ks = k_blocks if cfg["preload"] else 0
    if preload_ks:
        need = preload_lds_bytes(cfg, k)
        if need > _LDS_BYTES:
            raise RuntimeError(
                f"[FlyDSL gfx1250 bmm] {name!r} needs {need} B of LDS with the "
                f"K={k} scale panels, over the {_LDS_BYTES} B budget"
            )
    if n % tile_n or k % tile_k:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs N%{tile_n}==0 and K%{tile_k}==0, "
            f"got N={n}, K={k}"
        )
    if k // tile_k < cfg["num_buffers"]:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs >= {cfg['num_buffers']} K-tiles, "
            f"got {k // tile_k}"
        )

    bmm_spec = (
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        1 if out_dtype == "f16" else 0,
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        BLOCK_K,
        1,  # split_k
        True,  # batched
        preload_ks,
        False,  # a_preshuffle
    )
    # Cached CompiledFunction per constexpr set; see _run_compiled.
    _run_compiled(
        _launch_gemm_a8w8,
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(x_scale),
        _ptr_arg(w_scale),
        m,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
        n,
        k,
        b * k_blocks,  # stride_ascale_k: x_scale row stride, [M, B, K//128]
        b * k,  # lda: XQ is [M, B, K]
        b * n,  # ldc: Out is [M, B, N]
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        1 if out_dtype == "f16" else 0,
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        BLOCK_K,
        1,  # split_k
        True,  # batched
        preload_ks,
        b,
        specialization_key=bmm_spec,
    )
    return Out


__all__ = [
    "run_bmm_a8w8_mxfp8_32_gfx1250",
    "run_bmm_a8w8_mxfp8_128_gfx1250",
]
