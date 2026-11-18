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
import math
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import (
    get_cu_num,
    get_gfx,
    get_gfx_runtime,
    get_lds_capacity_bytes,
)

from .mxfp8_bpreshuffle_gemm_gfx1250 import (
    BASE_NAME_SUFFIX_RE,
    PIPELINED_BY_SUFFIX,
    PIPELINED_DEFAULT,
    default_pipelined_k_loop,
    is_compute_wmma_kernel_name,
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
)

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
# (n32k4 B) the mxfp8_128 ones. A trailing ``_pl`` / ``_nopl`` selects the
# software-pipelined / single-fence K loop (see check_pipelined_k_loop);
# without either, default_pipelined_k_loop picks it.
_BMM_MX32_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(BMM_MX32_WMMA_NAME_PREFIX)}_{_SUFFIX_RE}(?P<pipelined>_pl|_nopl)?$"
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
    groups = match.groupdict()
    pipelined = PIPELINED_BY_SUFFIX[groups.pop("pipelined")]
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["preload"] = 0
    if pipelined == PIPELINED_DEFAULT:
        # The batched path has no M-dependent config fix-up, so the K loop is
        # final here.
        pipelined = default_pipelined_k_loop(
            mx32=True,
            preload=False,
            use_cluster=cfg["cluster_m"] > 1 or cfg["cluster_n"] > 1,
            split_k=cfg["split_k"],
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n"],
            tile_k=cfg["tile_k"],
            m_warp=cfg["m_warp"],
            n_warp=cfg["n_warp"],
        )
    cfg["pipelined"] = pipelined
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
    tag: str, XQ: Tensor, WQ: Tensor, Out: Tensor, fused_fp8_out: bool = False
) -> tuple[int, int, int, int, str]:
    """Shared operand checks of the batched runners; returns (m, b, n, k, out).

    ``fused_fp8_out``: the GEMM quantizes its C tile to e4m3 in the epilogue, so
    ``Out`` is a 1-byte fp8 buffer. The returned out name stays ``bf16``: the
    quant reads values rounded to bf16, either from a bf16 C tile staged in LDS
    or, on the accumulator branch, from the accumulators rounded to bf16 in
    registers (no bf16 staging there).
    """
    if get_gfx() not in ("gfx1250", "gfx1250-strict"):
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
    if fused_fp8_out:
        if Out.element_size() != 1:
            raise RuntimeError(
                "[FlyDSL gfx1250 bmm] fused fp8 Out must be 1-byte e4m3 storage, "
                f"got {Out.dtype}"
            )
        out_dtype = "bf16"  # dtype the epilogue rounds C to before the e4m3 quant
    else:
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
    y_scale_out: Tensor | None = None,
) -> Tensor:
    """Run the batched mxfp8_32 GEMM; writes into ``Out`` and returns it.

    ``w_scale_32x32`` selects the raw ``[B, N//32, K//32]`` block scale; else
    w_scale is the n32k4-shuffled 1x32 ``[B, N, K//32]`` (see module doc).

    ``y_scale_out``: when given, the GEMM quantizes its bf16 C tile to e4m3 in
    the epilogue instead of writing bf16 -- ``Out`` is then the e4m3 ``[M,B,N]``
    buffer and ``y_scale_out`` the row-major ``[M, (B*N)//32]`` e8m0 scale, the
    ``(x, x_scale)`` pair a downstream native-quant GEMM reads. Bit-matches a
    standalone ATOM ``quantize_fp8`` of the bf16 output.
    """
    fused = y_scale_out is not None
    m, b, n, k, out_dtype = _check_bmm_operands(
        "mxfp8_32", XQ, WQ, Out, fused_fp8_out=fused
    )
    if fused:
        gn32 = b * n // 32
        if y_scale_out.element_size() != 1 or y_scale_out.numel() != m * gn32:
            raise RuntimeError(
                f"[FlyDSL gfx1250 bmm] y_scale_out must be {m * gn32} e8m0 bytes "
                f"([M, (B*N)//32]), got {tuple(y_scale_out.shape)} {y_scale_out.dtype}"
            )
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
    if is_compute_wmma_kernel_name(name):
        if fused:
            raise ValueError(
                f"[FlyDSL gfx1250 bmm] compute kernel {name!r} has no fused quant "
                "epilogue; drop y_scale_out or use a non-compute kernel"
            )
        if m:
            from aiter.utility import dtypes

            run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
                XQ[:, 0],
                WQ[0],
                x_scale[:, 0].view(dtypes.fp8_e8m0),
                w_scale[0].view(dtypes.fp8_e8m0),
                Out[:, 0],
                name,
                w_scale_32x32=w_scale_32x32,
                batch=b,
            )
        return Out
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
        fused,  # fuse_quant_out
        cfg["pipelined"],
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
        _ptr_arg(y_scale_out) if fused else _ptr_arg(Out),  # arg_scale_out
        fused,  # fuse_quant_out
        cfg["pipelined"],
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
        False,  # a_preshuffle
        False,  # a_scale_preshuffled
        False,  # b_scale_32x32
        False,  # k_tail
        _ptr_arg(Out),  # arg_scale_out
        specialization_key=bmm_spec,
    )
    return Out


# The fused gate_up tiles were measured against GEMM + silu_and_mul +
# quantize_fp8 only on a gfx1250 part with this many CUs and at these
# (2*I, K) gate_up shapes (the DSV4.1 shared expert). gate_up_silu_supported
# is False anywhere else, so an unmeasured shape keeps the caller's unfused
# path.
_GATE_UP_SILU_CU_NUM = 256
_GATE_UP_SILU_SHAPES = ((4608, 5120),)
# (largest M, kernelName) at those shapes. Each name pins its K loop with
# ``_pl``, the loop the tile was timed with, instead of inheriting the default
# pick. Above the last bound the 2-D compute GEMM followed by silu_and_mul and
# quantize_fp8 was faster, so gate_up_silu_kernel_for leaves those rows to the
# caller's unfused path; the launch itself runs any M (see
# run_gate_up_silu_mxfp8_32_gfx1250).
_GATE_UP_SILU_TILES = (
    (384, "flydsl_bmm_mxfp8_32_wmma_t64x128x256_mw2_nw2_nb4_sk1_cm1_cn1_pl"),
    (512, "flydsl_bmm_mxfp8_32_wmma_t64x256x256_mw1_nw4_nb3_sk1_cm1_cn1_pl"),
    (640, "flydsl_bmm_mxfp8_32_wmma_t192x128x256_mw2_nw2_nb3_sk1_cm1_cn1_pl"),
    (768, "flydsl_bmm_mxfp8_32_wmma_t96x256x256_mw2_nw4_nb3_sk1_cm1_cn1_pl"),
    (960, "flydsl_bmm_mxfp8_32_wmma_t192x128x256_mw2_nw2_nb3_sk1_cm1_cn1_pl"),
    (1792, "flydsl_bmm_mxfp8_32_wmma_t128x256x256_mw2_nw4_nb3_sk1_cm1_cn1_pl"),
    (2048, "flydsl_bmm_mxfp8_32_wmma_t192x128x256_mw2_nw2_nb3_sk1_cm1_cn1_pl"),
    (4096, "flydsl_bmm_mxfp8_32_wmma_t128x256x256_mw2_nw4_nb3_sk1_cm1_cn1_pl"),
)
# Largest row count the fused path was measured to win at.
GATE_UP_SILU_MAX_ROWS = _GATE_UP_SILU_TILES[-1][0]
_GFX1250_ARCHS = ("gfx1250", "gfx1250-strict")
# 1-byte storage the launch reads / writes: e4m3 codes and e8m0 scales, or
# their raw bytes.
_FP8_STORAGE = (torch.float8_e4m3fn, torch.uint8)
_E8M0_STORAGE = (torch.float8_e8m0fnu, torch.uint8)


def gate_up_silu_kernel_ok(name: str | None) -> bool:
    """Whether a batched mxfp8_32 kernelName can run the fused SiLU epilogue:
    split-K/cluster 1 and 64-column multiples per tile and per wave."""
    cfg = parse_bmm_mx32_kernel_name(name) if name else None
    if cfg is None:
        return False
    return (
        cfg["split_k"] == 1
        and cfg["cluster_m"] == 1
        and cfg["cluster_n"] == 1
        and cfg["tile_n"] % 64 == 0
        and (cfg["tile_n"] // cfg["n_warp"]) % 64 == 0
    )


def _gate_up_silu_tiles_shape(cfg: dict[str, int], n2: int, k: int) -> bool:
    """Whether a fused-epilogue config tiles a [n2, k] gate_up weight."""
    return (
        n2 % 128 == 0
        and (n2 // 2) % (cfg["tile_n"] // 2) == 0
        and k % cfg["tile_k"] == 0
        and k // cfg["tile_k"] >= cfg["num_buffers"]
    )


def _check_gate_up_silu_tiles() -> None:
    # Every table name is checked once here, at import, so the per-call pick
    # is a plain table walk.
    for _, name in _GATE_UP_SILU_TILES:
        cfg = parse_bmm_mx32_kernel_name(name)
        if not gate_up_silu_kernel_ok(name) or cfg["pipelined"] != 1:
            raise AssertionError(f"bad fused gate_up tile {name!r}")
        for n2, k in _GATE_UP_SILU_SHAPES:
            if not _gate_up_silu_tiles_shape(cfg, n2, k):
                raise AssertionError(f"{name!r} does not tile [{n2}, {k}]")


_check_gate_up_silu_tiles()


@torch.compiler.assume_constant_result
def _gate_up_silu_device_measured() -> bool:
    """Whether the running GPU is the gfx1250 part the fused tiles were
    measured on; False when that cannot be determined (no GPU, an unknown arch,
    mixed or unreadable CU counts)."""
    try:
        return (
            get_gfx_runtime() in _GFX1250_ARCHS and get_cu_num() == _GATE_UP_SILU_CU_NUM
        )
    except (RuntimeError, KeyError, AssertionError, ValueError):
        return False


def gate_up_silu_supported(n2: int, k: int) -> bool:
    """Whether the fused gate_up path applies to a [n2, k] gate_up weight on
    this device: gfx1250 with the CU count and at a (n2, k) the tiles were
    measured on. Independent of the row count, so a model can decide once,
    when it is built, whether to route its gate_up through the fused op."""
    if not _gate_up_silu_device_measured():
        return False
    return any(n2 == s_n2 and k == s_k for s_n2, s_k in _GATE_UP_SILU_SHAPES)


def _gate_up_silu_tile(m: int) -> str:
    """The table's tile for ``m`` rows; past the table, its largest bucket's."""
    for bound, name in _GATE_UP_SILU_TILES:
        if m <= bound:
            return name
    return _GATE_UP_SILU_TILES[-1][1]


def gate_up_silu_kernel_for(m: int, n2: int, k: int) -> str | None:
    """kernelName for ``run_gate_up_silu_mxfp8_32_gfx1250`` at ``[m, k] x
    [n2, k]``, or None where the fused path does not apply: not
    ``gate_up_silu_supported(n2, k)``, or an M outside 1..GATE_UP_SILU_MAX_ROWS,
    which the unfused path serves better.

    A host-side helper for code that sees the real row count (eager code, or
    the body of a custom op). Do not branch on it in a forward that
    torch.compile traces: a compiled graph can be replayed for other row
    counts without re-checking guards, which freezes the choice made for the
    first one. The registered op ``gate_up_silu_mxfp8_32`` picks the tile from
    the runtime row count itself."""
    if not 1 <= m <= GATE_UP_SILU_MAX_ROWS or not gate_up_silu_supported(n2, k):
        return None
    return _gate_up_silu_tile(m)


def gate_up_silu_limit(swiglu_limit: float) -> float:
    """``swiglu_limit`` as the fused kernel takes it, with the meaning aiter
    ``silu_and_mul`` gives it: a positive limit clamps (gate from above at L, up
    to [-L, L]), 0 clamps nothing, and a negative or NaN limit is an error.
    +inf also clamps nothing, so it runs the same kernel as 0."""
    limit = float(swiglu_limit)
    if not limit >= 0.0:
        raise ValueError(
            "[FlyDSL gfx1250 gate_up silu] swiglu_limit must be >= 0 "
            f"(0 = no clamp), got {swiglu_limit!r}"
        )
    return 0.0 if math.isinf(limit) else limit


def run_gate_up_silu_mxfp8_32_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    y_scale_out: Tensor,
    swiglu_limit: float,
    kernel_name: str | None = None,
) -> Tensor:
    """gate_up GEMM with the SwiGLU activation and its MXFP8 quant fused.

    ``XQ`` [M, K] e4m3 (row-contiguous) with row-major ``x_scale`` [M, K//32]
    e8m0; ``WQ`` the contiguous [2*I, K] gate_up weight ([gate; up] rows)
    16x16-preshuffled, ``w_scale`` its raw [2*I//32, K//32] 32x32 e8m0 block
    scale. Writes ``Out`` [M, I] e4m3 and ``y_scale_out`` [M, I//32] e8m0
    (either may also be uint8 storage): ``quantize_fp8`` of aiter
    ``silu_and_mul(gemm_out, swiglu_limit)`` on the bf16 GEMM output, with the
    same scales and the same e4m3 codes up to one step. The epilogue's exp/rcp
    can round a bf16 activation to the neighbour of silu_and_mul's, which can
    move an e4m3 rounding midpoint; with a limit bf16 cannot represent, the
    gate is also clamped in f32 where silu_and_mul rounds the clamped gate to
    bf16. ``swiglu_limit`` follows silu_and_mul (see ``gate_up_silu_limit``).

    ``kernel_name`` None picks the tile from ``XQ``'s row count; past
    GATE_UP_SILU_MAX_ROWS that is the largest bucket's tile, which computes the
    same values but was not measured to beat the unfused chain there.

    This is the raw FlyDSL launch; a torch.compile region calls the registered
    op ``aiter.ops.batched_gemm_op_a8w8.gate_up_silu_mxfp8_32`` instead.
    """
    tag = "[FlyDSL gfx1250 gate_up silu]"
    limit = gate_up_silu_limit(swiglu_limit)
    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            f"{tag} XQ/WQ must be 2-D, got {tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    m, k = XQ.shape
    n2, wk = WQ.shape
    inter = n2 // 2
    if wk != k or n2 % 128 or k % BLOCK_K:
        raise RuntimeError(
            f"{tag} bad shapes XQ {tuple(XQ.shape)} WQ {tuple(WQ.shape)}"
        )
    if XQ.dtype not in _FP8_STORAGE or WQ.dtype not in _FP8_STORAGE:
        raise RuntimeError(
            f"{tag} XQ/WQ must be e4m3 (or uint8 storage), got {XQ.dtype}, "
            f"{WQ.dtype}"
        )
    # The launch passes raw pointers with lda = K.
    if XQ.stride() != (k, 1):
        raise RuntimeError(
            f"{tag} XQ must be row-contiguous [M, K], got strides {tuple(XQ.stride())}"
        )
    if not WQ.is_contiguous():
        raise RuntimeError(f"{tag} WQ must be contiguous")
    if (
        tuple(Out.shape) != (m, inter)
        or Out.dtype not in _FP8_STORAGE
        or not Out.is_contiguous()
        or tuple(y_scale_out.shape) != (m, inter // 32)
        or y_scale_out.dtype not in _E8M0_STORAGE
        or not y_scale_out.is_contiguous()
    ):
        raise RuntimeError(
            f"{tag} Out must be contiguous e4m3 [{m}, {inter}] and y_scale_out "
            f"e8m0 [{m}, {inter // 32}], got {Out.dtype} {tuple(Out.shape)}, "
            f"{y_scale_out.dtype} {tuple(y_scale_out.shape)}"
        )
    name = _gate_up_silu_tile(m) if kernel_name is None else kernel_name
    if not gate_up_silu_kernel_ok(name):
        raise ValueError(f"{tag} unsupported kernel {name!r}")
    cfg = parse_bmm_mx32_kernel_name(name)
    tile_n, tile_k = cfg["tile_n"], cfg["tile_k"]
    if not _gate_up_silu_tiles_shape(cfg, n2, k):
        raise RuntimeError(f"{tag} {name!r} does not tile I={inter}, K={k}")
    gfx = get_gfx_runtime()
    if gfx not in _GFX1250_ARCHS:
        raise RuntimeError(f"{tag} requires gfx1250, got {gfx}")
    _lazy_import()
    k_blocks = k // MX32_BLOCK_K
    _check_e8m0(x_scale, "x_scale", "FlyDSL gfx1250 gate_up silu", (m, k_blocks))
    _check_e8m0(w_scale, "w_scale", "FlyDSL gfx1250 gate_up silu", (n2 // 32, k_blocks))
    if m == 0:
        return Out
    spec = (
        "gate_up_silu",
        XQ.device.index,
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        cfg["num_buffers"],
        cfg["pipelined"],
        limit,
    )
    _run_compiled(
        _launch_gemm_a8w8,
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(x_scale),
        _ptr_arg(w_scale),
        m,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
        n2,
        k,
        k_blocks,  # stride_ascale_k: x_scale row stride, [M, K//32]
        k,  # lda
        inter,  # ldc: Out is [M, I]
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        0,  # out_is_f16: bf16 C rounding before the activation
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        MX32_BLOCK_K,
        1,  # split_k
        True,  # batched (B = 1)
        0,  # preload_ks
        1,  # batch
        False,  # a_preshuffle
        False,  # a_scale_preshuffled
        True,  # b_scale_32x32
        False,  # k_tail
        _ptr_arg(y_scale_out),
        True,  # fuse_quant_out
        cfg["pipelined"],
        True,  # fuse_act_mul
        limit,  # act_limit: 0 = no clamp
        specialization_key=spec,
    )
    return Out


__all__ = [
    "GATE_UP_SILU_MAX_ROWS",
    "gate_up_silu_kernel_for",
    "gate_up_silu_kernel_ok",
    "gate_up_silu_limit",
    "gate_up_silu_supported",
    "run_bmm_a8w8_mxfp8_32_gfx1250",
    "run_bmm_a8w8_mxfp8_128_gfx1250",
    "run_gate_up_silu_mxfp8_32_gfx1250",
]
