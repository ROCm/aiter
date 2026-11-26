# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 strided-batched mxfp8 GEMM with 32- or 128-wide e8m0 blocks.

===========  =============================  ================================
operand      shape                          notes
===========  =============================  ================================
``XQ``       ``[M, B, K]`` fp8              M-outer, K-contiguous
``WQ``       ``[B, N, K]`` fp8              16x16 preshuffled (ops.shuffle.shuffle_weight)
``x_scale``  ``[M, B, K//XK]`` e8m0         row-major, XK = 32 or 128
``w_scale``  ``[B, N//WN, K//WK]`` e8m0     WN x WK = 32x32, 128x128 or 1x32
``Out``      ``[M, B, N]`` bf16
===========  =============================  ================================

The block edges come from the scale shapes. Tuned configs are rows of the
mxscale bpreshuffle CSV keyed by w_scale_block and arrive here as a kernelName
(see bmm_kernel_name).
"""

from __future__ import annotations

import functools
import inspect
import itertools
import re
from types import MappingProxyType

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .aot_launch import aot_factory
from .kernels.bmm_a8w8_mxscale_gfx950 import (
    SCALE_BLOCKS,
    W_SCALE_ROWS,
    bmm_config_name,
    check_bmm_config,
    launch_bmm_a8w8_mxscale,
    xcd_swizzle_runs,
)
from .kernels.kernels_common import ceildiv
from .kernels.tensor_shim import (
    _run_compiled,
    check_e8m0,
    current_raw_stream,
    ptr_arg,
)

BMM_MFMA_NAME_PREFIX = "flydsl_bmm_mxfp8_mfma"

_BMM_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(BMM_MFMA_NAME_PREFIX)}"
    r"_t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)"
    r"_w(?P<m_warp>\d+)x(?P<n_warp>\d+)_nb(?P<num_buffers>\d+)_sk(?P<splits>\d+)"
    r"(?:_bd(?P<b_ahead>\d+))?(?P<b_nt>_nt)?"
    r"(?P<xcd_order>_xcd(?P<xcd_swizzle>[1-9]\d*)?)?"
    r"(?P<per_stage>_sps)?$"
)


@functools.cache
def _launch_config(name: str) -> MappingProxyType | None:
    """Launch arguments of a kernelName, parsed once per name (read-only), or
    None if the name breaks a rule of its own (xcd_swizzle_runs). Whether they
    run at a shape is check_bmm_config's call."""
    match = _BMM_KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    groups = match.groupdict()
    b_ahead = groups.pop("b_ahead")
    b_nt = groups.pop("b_nt")
    xcd_order = groups.pop("xcd_order")
    xcd_swizzle = groups.pop("xcd_swizzle")
    per_stage = groups.pop("per_stage")
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["b_direct"] = b_ahead is not None
    cfg["b_ahead"] = int(b_ahead or 1)
    cfg["b_nt"] = b_nt is not None
    cfg["xcd_order"] = xcd_order is not None
    cfg["xcd_swizzle"] = int(xcd_swizzle or 0)
    cfg["scale_preload"] = per_stage is None
    if not xcd_swizzle_runs(cfg["xcd_swizzle"], cfg["xcd_order"], cfg["splits"]):
        return None
    return MappingProxyType(cfg)


def parse_bmm_kernel_name(name: str) -> dict | None:
    """Parse a tuned kernelName into launch arguments, or ``None`` if it is not
    one of ours."""
    cfg = _launch_config(name)
    return None if cfg is None else dict(cfg)


def bmm_kernel_name(**cfg) -> str:
    """Canonical tuned-CSV name of a config's launch arguments (see
    bmm_config_name)."""
    return BMM_MFMA_NAME_PREFIX + bmm_config_name(**cfg)


# Shapes without a tuned row, by the largest M a tier serves: small M streams
# the weight through narrow tiles and deep LDS rings (split K is sized in
# pick_bmm_kernel_name); large M reads it straight into registers.
_FALLBACK_FIELDS = ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "num_buffers")
_FALLBACK = [
    (16, (16, 32, 256, 1, 1, 4), False),
    (32, (32, 64, 512, 1, 4, 3), False),
    (64, (64, 32, 512, 4, 1, 3), False),
    (256, (64, 64, 256, 2, 2, 2), False),
    (1024, (128, 128, 128, 1, 4, 2), False),
    (2048, (128, 128, 128, 2, 2, 2), False),
    (float("inf"), (256, 256, 128, 1, 4, 2), True),
]
_SPLITS = (1, 2, 4, 8, 16)
# M-major group sizes of the xcd_order run that a tuner times (xcd_swizzle).
_XCD_SWIZZLES = (2, 3, 4, 6, 8)


def bmm_config_runs(b: int, n: int, k: int, scale_args: dict, cfg: dict) -> bool:
    """Whether cfg runs the shape; ``scale_args`` are check_bmm_config's."""
    try:
        check_bmm_config(n, k, b, **cfg, **scale_args)
    except ValueError:
        return False
    return True


def _variants(tier: dict, b_direct: bool):
    """The tier's config, then ever looser ones for shapes it cannot take:
    shorter K tiles, narrower N tiles, fewer stages, B through LDS, per-stage
    scales."""
    for preload, tile_k, tile_n, stages, direct in itertools.product(
        (True, False),
        (t for t in (512, 256, 128) if t <= tier["tile_k"]),
        (t for t in (256, 128, 64, 32) if t <= tier["tile_n"]),
        range(tier["num_buffers"], 1, -1),
        dict.fromkeys((b_direct, False)),
    ):
        yield {
            **tier,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "n_warp": min(tier["n_warp"], tile_n // 16),
            "num_buffers": stages,
            "b_direct": direct,
            "scale_preload": preload,
        }


def _max_split(cfg: dict, tiles: int, runs) -> int:
    """The largest power-of-two K split that keeps about 256 workgroups in
    flight and that cfg runs at, or 0. More splits only shrink a preloaded
    scale panel, so a config too big for LDS whole may fit split."""
    splits = (
        [1] if cfg["b_direct"] else [s for s in _SPLITS if s == 1 or tiles * s <= 256]
    )
    return max((s for s in splits if runs({**cfg, "splits": s})), default=0)


@functools.lru_cache(maxsize=1024)
def pick_bmm_kernel_name(
    b: int,
    m: int,
    n: int,
    k: int,
    x_scale_k: int = 32,
    w_scale_n: int = 32,
    w_scale_k: int = 32,
    x_scale_transposed: bool = False,
) -> str:
    """Heuristic config for a shape with no tuned row: the M tier's tile, or
    the first looser variant the shape can run, split in K until about 256
    workgroups are in flight. Raises ValueError only for shapes no config of
    the kernel can run."""
    tile, b_direct = next((t, d) for max_m, t, d in _FALLBACK if m <= max_m)
    scale_args = {
        "x_scale_k": x_scale_k,
        "w_scale_n": w_scale_n,
        "w_scale_k": w_scale_k,
        "x_scale_transposed": x_scale_transposed,
    }
    runs = functools.partial(bmm_config_runs, b, n, k, scale_args)
    for cfg in _variants(dict(zip(_FALLBACK_FIELDS, tile)), b_direct):
        tiles = ceildiv(m, cfg["tile_m"]) * (n // cfg["tile_n"]) * b
        splits = _max_split(cfg, tiles, runs)
        if splits:
            break
    else:
        raise ValueError(
            f"[FlyDSL gfx950 bmm] no kernel config runs B={b} N={n} K={k} "
            f"with 1x{x_scale_k} / {w_scale_n}x{w_scale_k} scale blocks"
        )
    # The XCD tile order wants the N x batch tiles in 8s, as split K does. Its
    # M-major swizzle is measured on 1x32 w_scales only.
    xcd = {"xcd_order": True}
    if w_scale_n == 1 and splits == 1:
        xcd["xcd_swizzle"] = 4
    xcd = xcd if cfg["b_direct"] and runs({**cfg, **xcd}) else {}
    return bmm_kernel_name(**cfg, splits=splits, **xcd)


def bmm_candidate_names(
    b: int, m: int, n: int, k: int, seeds=(), **scale_args
) -> list[str]:
    """kernelNames a tuner times for a shape: ``seeds`` (tuned names of nearby
    M), the heuristic pick, and their K splits from half to double that run
    it. ``scale_args`` are check_bmm_config's. Split K keeps its own tile
    order, so a split variant drops the seed's xcd_order and xcd_swizzle (a
    split ``_xcd`` name runs the same kernel as the plain one)."""
    runs = functools.partial(bmm_config_runs, b, n, k, scale_args)
    names = set(seeds)
    try:
        names.add(pick_bmm_kernel_name(b, m, n, k, **scale_args))
    except ValueError:
        pass
    out = set()
    for cfg in filter(None, map(parse_bmm_kernel_name, names)):
        base = cfg["splits"]
        for splits in {base, *(s for s in _SPLITS if base / 2 <= s <= base * 2)}:
            variant = {**cfg, "splits": splits}
            if splits > 1:
                variant.update(xcd_order=False, xcd_swizzle=0)
            if runs(variant):
                out.add(bmm_kernel_name(**variant))
        for swizzle in (0, *_XCD_SWIZZLES):
            variant = {**cfg, "splits": 1, "xcd_order": True, "xcd_swizzle": swizzle}
            if runs(variant):
                out.add(bmm_kernel_name(**variant))
    return sorted(out)


def _x_scale_column_major(m: int, x_scale_transposed: bool) -> bool:
    """Whether the kernel reads x_scale column-major. A single row reads the
    same either way and takes the row-major kernel: a column-major dword there
    would span four columns and run past the tensor's end."""
    return x_scale_transposed and m > 1


# One arrival counter per output tile of a split-K launch; the last split to
# arrive resets it, so a buffer per (device, stream) is reused forever.
_SPLIT_COUNTERS = 1 << 16


@capture_scoped_cache
def _split_counters(device: int, stream: int) -> Tensor:
    with persistent_alloc(torch.device("cuda", device)):
        return torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)


def _scale_block(extent: int, blocks: int, what: str, edges=SCALE_BLOCKS) -> int:
    """The block edge that splits extent into blocks: one of edges."""
    if blocks and extent % blocks == 0 and extent // blocks in edges:
        return extent // blocks
    raise RuntimeError(
        f"[FlyDSL gfx950 bmm] {what} of {extent} in {blocks} scale blocks is not "
        f"a {edges} block"
    )


def _require_config(name: str) -> MappingProxyType:
    cfg = _launch_config(name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx950 bmm] unrecognised kernelName: {name!r}")
    return cfg


@functools.cache
def _warn_unrecognised(name: str) -> None:
    logger.warning(
        f"[FlyDSL gfx950 bmm] unrecognised kernelName {name!r}; "
        "using the heuristic's pick"
    )


# The launcher's runtime parameters: seven pointers, M and the stream. Every
# later one is a Constexpr, baked into the kernel it compiles.
_RUNTIME_PARAMS = 9
_SCALE_PARAMS = ("x_scale_k", "w_scale_n", "w_scale_k", "x_scale_transposed")


@functools.lru_cache(maxsize=1024)
def _constexprs(
    name: str, n: int, k: int, b: int, *scale_args, fuse_quant_out: bool = False
) -> tuple:
    """A kernel's Constexpr arguments in parameter order, the key of its
    compiled launcher. ``scale_args`` are the _SCALE_PARAMS values."""
    bound = inspect.signature(launch_bmm_a8w8_mxscale.func).bind(
        *(None,) * _RUNTIME_PARAMS,
        n,
        k,
        b,
        **_require_config(name),
        **dict(zip(_SCALE_PARAMS, scale_args)),
        fuse_quant_out=fuse_quant_out,
    )
    bound.apply_defaults()
    return bound.args[_RUNTIME_PARAMS:]


@aot_factory(
    "bmm_a8w8_mxscale_gfx950",
    sources=[
        "kernels/bmm_a8w8_mxscale_gfx950.py",
        "kernels/mfma_preshuffle_pipeline.py",
        "../../utility/mx_types.py",
        "kernels/quant_utils.py",
    ],
)
@functools.cache
def _bound_launcher(constexprs: tuple):
    """launch_bmm_a8w8_mxscale with its Constexprs bound, so that every
    parameter left is a runtime argument the AOT cache can export."""

    @flyc.jit
    def launch_bmm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_partials: fx.Pointer,
        arg_counters: fx.Pointer,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        launch_bmm_a8w8_mxscale(
            arg_c,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_partials,
            arg_counters,
            i32_m,
            stream,
            *constexprs,
        )

    # The call inlines while tracing; keep the kernel's compile hints.
    launch_bmm.compile_hints.update(launch_bmm_a8w8_mxscale.compile_hints)
    return launch_bmm


def _launch(tensors, m, stream, constexprs: tuple) -> None:
    """The one call into the kernel, shared by the runtime and AOT paths so
    both build the same cache entry. ``tensors`` are Out, XQ, WQ, x_scale,
    w_scale, split-K partials and counters; only their addresses reach the
    kernel. Under compile-only (AOT) it builds the entry without launching."""
    _run_compiled(
        _bound_launcher.lazy(constexprs),
        *map(ptr_arg, tensors),
        m,
        stream,
        specialization_key=tensors[0].device,
    )


def compile_bmm_a8w8_mxfp8_gfx950(
    kernel_name: str,
    b: int,
    n: int,
    k: int,
    x_scale_k: int,
    w_scale_n: int,
    w_scale_k: int,
    x_scale_transposed: bool = False,
) -> None:
    """Compile ``kernel_name`` for ``[B, N, K]`` with 1 x x_scale_k x_scale
    blocks (column-major if x_scale_transposed) and w_scale_n x w_scale_k
    w_scale blocks, as run_bmm_a8w8_mxfp8_gfx950 would launch it. M is a
    runtime argument, so this serves every M. Call it under
    ``aiter.aot.flydsl.common.compile_only_env``."""
    constexprs = _constexprs(
        kernel_name, n, k, b, x_scale_k, w_scale_n, w_scale_k, x_scale_transposed
    )
    placeholder = torch.empty(0, dtype=torch.uint8)
    _launch((placeholder,) * 7, 1, fx.Stream(0), constexprs)


def run_bmm_a8w8_mxfp8_gfx950(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
    x_scale_transposed: bool = False,
    y_scale_out: Tensor | None = None,
) -> Tensor:
    """``Out[m, b] = dequant(XQ[m, b]) @ dequant(WQ[b]).T``; writes into ``Out``
    and returns it. The e8m0 block edges (1x32 or 1x128 for ``x_scale``, 32x32
    or 128x128 for ``w_scale``) are read off the scale shapes. ``kernel_name``
    is a tuned row's kernelName, else (or if it does not parse, with a
    warning) :func:`pick_bmm_kernel_name` chooses. A ``w_scale`` block may
    also be a single row, 1x32 only.
    x_scale_transposed (B = 1, 1x128 blocks) reads ``x_scale``'s bytes
    column-major, [K / 128, M], the blockscale convention, under its
    [M, 1, K / 128] shape; a single row takes the row-major kernel.
    ``y_scale_out`` fuses the output's group-32 MXFP8 quant into the epilogue:
    ``Out`` is then e4m3 and ``y_scale_out`` receives the row-major
    [M, B * N // 32] e8m0 scale bytes, as quantize_fp8 of the kernel's bf16
    output would. The fused kernel runs one split, so a split-K kernelName
    runs unsplit; its bf16 values may then differ from the split-K kernel's
    by an fp32 rounding (another reduction order)."""
    if get_gfx() != "gfx950":
        raise RuntimeError(
            f"[FlyDSL gfx950 bmm] batched mxfp8 requires gfx950, got {get_gfx()}"
        )
    if XQ.dim() != 3 or WQ.dim() != 3 or Out.dim() != 3:
        raise RuntimeError(
            "[FlyDSL gfx950 bmm] XQ/WQ/Out must be 3-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}, {tuple(Out.shape)}"
        )
    m, b, k = XQ.shape
    n = WQ.shape[1]
    if tuple(WQ.shape) != (b, n, k) or tuple(Out.shape) != (m, b, n):
        raise RuntimeError(
            f"[FlyDSL gfx950 bmm] expected WQ [{b}, N, {k}] and Out [{m}, {b}, N], "
            f"got {tuple(WQ.shape)} and {tuple(Out.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx950 bmm] XQ/WQ must be 1-byte fp8 storage")
    fuse = y_scale_out is not None
    if fuse:
        if Out.element_size() != 1:
            raise RuntimeError(
                f"[FlyDSL gfx950 bmm] a fused quant Out must be 1-byte e4m3, "
                f"got {Out.dtype}"
            )
        if y_scale_out.device.type != "cuda" or y_scale_out.device != XQ.device:
            raise RuntimeError(
                f"[FlyDSL gfx950 bmm] y_scale_out must be on the launch GPU "
                f"{XQ.device}, got {y_scale_out.device}"
            )
        if (
            n % 32
            or y_scale_out.element_size() != 1
            or y_scale_out.numel() != m * b * n // 32
            or not y_scale_out.is_contiguous()
        ):
            raise RuntimeError(
                f"[FlyDSL gfx950 bmm] fused quant needs N % 32 == 0 and a "
                f"contiguous 1-byte y_scale_out of {m} x {b * n // 32}, got N={n}, "
                f"{tuple(y_scale_out.shape)} {y_scale_out.dtype}"
            )
    elif Out.dtype != torch.bfloat16:
        raise RuntimeError(f"[FlyDSL gfx950 bmm] Out must be bf16, got {Out.dtype}")
    if not (XQ.is_contiguous() and WQ.is_contiguous() and Out.is_contiguous()):
        raise RuntimeError("[FlyDSL gfx950 bmm] XQ/WQ/Out must be contiguous")
    if x_scale.dim() != 3 or tuple(x_scale.shape[:2]) != (m, b):
        raise RuntimeError(
            f"[FlyDSL gfx950 bmm] x_scale must be [{m}, {b}, K / block], "
            f"got {tuple(x_scale.shape)}"
        )
    if w_scale.dim() != 3 or w_scale.shape[0] != b:
        raise RuntimeError(
            f"[FlyDSL gfx950 bmm] w_scale must be [{b}, N / block, K / block], "
            f"got {tuple(w_scale.shape)}"
        )
    check_e8m0(x_scale, "x_scale", "FlyDSL gfx950 bmm")
    check_e8m0(w_scale, "w_scale", "FlyDSL gfx950 bmm")
    scale_args = (
        _scale_block(k, x_scale.shape[2], "x_scale K"),
        _scale_block(n, w_scale.shape[1], "w_scale N", W_SCALE_ROWS),
        _scale_block(k, w_scale.shape[2], "w_scale K"),
        _x_scale_column_major(m, x_scale_transposed),
    )
    if m == 0:
        return Out

    if kernel_name and _launch_config(kernel_name) is None:
        _warn_unrecognised(kernel_name)
        kernel_name = None
    if fuse and not kernel_name and m > _FUSED_LARGE_M:
        kernel_name = _FUSED_LARGE_M_KERNEL  # no flydsl row: see _FUSED_LARGE_M
    name = kernel_name or pick_bmm_kernel_name(b, m, n, k, *scale_args)
    if fuse:
        name = _fused_quant_name(name, b, m, n, k, scale_args)
    cfg = _require_config(name)
    stream = current_raw_stream(XQ.device)
    partials = counters = Out
    if fuse:
        partials = y_scale_out  # the kernel writes the scale bytes there
    elif cfg["splits"] > 1:
        tiles = ceildiv(m, cfg["tile_m"]) * (n // cfg["tile_n"]) * b
        if tiles > _SPLIT_COUNTERS:
            raise RuntimeError(
                f"[FlyDSL gfx950 bmm] {name}: {tiles} split-K tiles exceed the "
                f"{_SPLIT_COUNTERS} arrival counters"
            )
        partials = torch.empty(
            tiles * cfg["splits"] * cfg["tile_m"] * cfg["tile_n"],
            dtype=torch.float32,
            device=XQ.device,
        )
        counters = _split_counters(XQ.device.index, stream)
    tensors = (Out, XQ, WQ, x_scale, w_scale, partials, counters)
    _launch(
        tensors,
        m,
        fx.Stream(stream),
        _constexprs(name, n, k, b, *scale_args, fuse_quant_out=fuse),
    )
    return Out


_FUSED_LARGE_M = 1024
_FUSED_LARGE_M_KERNEL = BMM_MFMA_NAME_PREFIX + "_t128x256x128_w1x4_nb3_sk1_bd2_xcd"


def _fused_quant_falls_back(name: str) -> bool:
    cfg = _launch_config(name)
    return (
        cfg is not None and cfg["splits"] >= 4 and cfg["tile_m"] * cfg["tile_n"] >= 2048
    )


@functools.lru_cache(maxsize=1024)
def _fused_quant_name(name: str, b: int, m: int, n: int, k: int, scale_args) -> str:
    """The kernel a fused-quant launch runs for ``name``: the same tile at one
    split (the quant epilogue has no split-K reduction), with xcd_swizzle
    dropped if one split cannot take it; the heuristic's unsplit pick when the
    tile cannot, or its tile_n leaves a 32-column group straddling two tiles."""
    scale = dict(zip(_SCALE_PARAMS, scale_args))
    for cfg in (
        _require_config(name),
        _require_config(pick_bmm_kernel_name(b, m, n, k, *scale_args)),
    ):
        cfg = {**cfg, "splits": 1}
        if not xcd_swizzle_runs(cfg["xcd_swizzle"], cfg["xcd_order"], 1):
            cfg["xcd_swizzle"] = 0
        if cfg["tile_n"] % 32 == 0 and bmm_config_runs(b, n, k, scale, cfg):
            return bmm_kernel_name(**cfg)
    raise NotImplementedError(
        f"[FlyDSL gfx950 bmm] no unsplit kernel with a 32-column tile_n runs the "
        f"fused quant for B={b}, M={m}, N={n}, K={k}"
    )


__all__ = [
    "bmm_candidate_names",
    "bmm_kernel_name",
    "compile_bmm_a8w8_mxfp8_gfx950",
    "parse_bmm_kernel_name",
    "pick_bmm_kernel_name",
    "run_bmm_a8w8_mxfp8_gfx950",
]
