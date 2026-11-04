# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL FP8 MQA logits (DeepSeek lightning indexer) API.

Wraps the gfx942/gfx950 kernel builders in
``aiter.ops.flydsl.kernels.mqa_logits.fp8_mqa_logits`` with:
  - The kernel-variant registry and shape-adaptive auto-selection
    (``_auto_variant`` / ``KERNEL_VARIANTS`` / ``DEFAULT_VARIANT``).
  - A build cache keyed by shape/variant/dtype-conversion flags
    (``compile_fp8_mqa_logits``).
  - Output-column alignment, and the occupancy-aware KV-column split
    (``grid.y``) heuristic that fills the device for small-M shapes
    (``_auto_num_splits``). Query rows are not padded.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import torch

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

from .kernels.mqa_logits.fp8_mqa_logits import (
    _MFMA16,
    _MFMA16_K128,
    _MFMA32_K64,
    _build_kernel_mfma_lds_pipe,
    _build_kernel_mfma_r_w,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "DEFAULT_VARIANT",
    "KERNEL_VARIANTS",
    "compile_fp8_mqa_logits",
    "flydsl_fp8_mqa_logits",
]

# Default KV tile width (columns processed per inner-loop iteration).
_BLOCK_KV = 128

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}

# Resolved once at import so the variant registry below can be built statically.
# The guard keeps the module importable on a host with no GPU (CI collecting
# tests, doc builds), where ``get_gfx()`` raises.
#
# The sentinel is deliberately not a real arch.
# Here the arch selects the kernel registry, so naming a real arch
# would register variants that cannot run and defer the failure to a compile or
# launch error. Returning ``"unknown"`` instead leaves ``_VARIANT_BUILDERS`` empty and lets
# ``_auto_variant`` raise NotImplementedError naming the arch. Hence, the import succeeds,
# and the first actual use fails with a clear message. ``_split_policy`` already
# treats an unrecognised arch as gfx942, so it needs no separate handling.
try:
    _ARCH = get_gfx()
except Exception:  # noqa: BLE001
    _ARCH = "unknown"


@dataclass(frozen=True)
class _SplitPolicy:
    """Per-arch tuning for the ``grid.y`` KV-column split (``_auto_num_splits``).

    Fields
    ------
    min_seq_len_kv : int
        Never split below this ``seq_len_kv``. 0 means "no gate" -- let
        ``min_tiles_per_split`` do the limiting, which it does automatically
        once the window holds fewer than that many tiles.
    min_tiles_per_split : int
        Smallest chunk, in BKV tiles, a split may own. Below it the per-block
        fixed cost (Q/weight preload, plus the LDS builder's pipeline prologue)
        stops being amortized. Note this is denominated in *tiles*, so its
        column-equivalent scales with the variant's ``block_kv``.
    cu_oversub : int
        Target total blocks as a multiple of the device CU count. Used only by
        the fixed-oversubscription path (``block_overhead_tiles`` is None, or
        ``blocks_per_cu`` is unknown: a launcher's first call during graph
        capture, see ``_max_blocks_per_cu``).
    fallback_cu : int
        Nominal CU count to assume when the device query fails.
    block_overhead_tiles : int | None
        Per-block fixed cost, in BKV tiles, of the occupancy-aware cost model
        (``_splits_by_wave_cost``). None keeps the ``cu_oversub`` target.
    """

    min_seq_len_kv: int
    min_tiles_per_split: int
    cu_oversub: int
    fallback_cu: int
    block_overhead_tiles: int | None = None


_SPLIT_POLICIES = {
    # Tuned on MI300X (304 CU) against the direct-load builder at BKV=128,
    # where min_tiles_per_split=8 is 1024 KV columns.
    "gfx942": _SplitPolicy(
        min_seq_len_kv=4096,
        min_tiles_per_split=8,
        cu_oversub=4,
        fallback_cu=304,
        block_overhead_tiles=8,  # fitted on MI325X
    ),
    "gfx950": _SplitPolicy(
        min_seq_len_kv=0,
        min_tiles_per_split=2,
        cu_oversub=4,
        fallback_cu=256,
        block_overhead_tiles=8,
    ),
}


def _split_policy() -> _SplitPolicy:
    """Split-policy constants for the current arch (gfx942's, conservatively,
    for anything unrecognized)."""
    return _SPLIT_POLICIES.get(_ARCH, _SPLIT_POLICIES["gfx942"])


@lru_cache(maxsize=8)
def _device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); the arch's nominal
    count if the query fails."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return _split_policy().fallback_cu


# Upper bound on the grid.y splits the cost model searches.
_MAX_SPLITS = 64


@lru_cache(maxsize=256)
def _splits_by_wave_cost(grid_x, effective_cus, window_tiles, max_splits, overhead):
    """``num_splits`` in [1, min(max_splits, _MAX_SPLITS)] minimizing the
    wave-quantized cost
    ``ceil(grid_x * s / effective_cus) * (ceil(window_tiles / s) + overhead)``."""
    return min(
        range(1, min(max_splits, _MAX_SPLITS) + 1),
        key=lambda s: -(-(grid_x * s) // effective_cus)
        * (-(-window_tiles // s) + overhead),
    )


def _max_blocks_per_cu(launcher, args):
    """Max resident blocks per CU of *launcher*'s kernel, cached on the launcher.

    None on a first call during graph capture: FlyDSL's occupancy query loads
    a scratch copy of the kernel module and unloads it, and HIP rejects
    ``hipModuleUnload`` during stream capture.
    """
    if not hasattr(launcher, "_max_blocks_per_cu"):
        if torch.cuda.is_current_stream_capturing():
            return None
        # Meta tensors make flyc.compile compile only, without launching.
        args = [a.to("meta") if isinstance(a, torch.Tensor) else a for a in args]
        launcher._max_blocks_per_cu = flyc.compile(launcher, *args).max_blocks_per_cu()
    return launcher._max_blocks_per_cu


def _auto_num_splits(
    seq_len: int,
    seq_len_kv: int,
    rows_per_block: int,
    block_kv: int,
    device_index: int,
    blocks_per_cu: int | None = None,
) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    For small-M / large-N shapes the ``ceil(seq_len/RPB)`` row grid leaves the
    device block-starved; splitting each row's window across ``grid.y`` recovers
    occupancy at no correctness cost (logits[m,n] are independent across n).

    Two paths, chosen per-arch by ``_SplitPolicy.block_overhead_tiles``:

    - Occupancy-aware (gfx942, gfx950): how full the device is depends on the
      kernel instance, not just the shape -- resident blocks per CU depend on
      the compiled kernel's register and LDS use. With ``blocks_per_cu`` known,
      this path forms ``cu_count * blocks_per_cu`` effective CUs and minimizes
      ``_splits_by_wave_cost`` over them.
    - Fixed oversubscription (``block_overhead_tiles`` None, or
      ``blocks_per_cu`` unknown): target
      ``cu_oversub * cu_count`` blocks and return 1 once the row grid alone
      reaches that.
    """
    pol = _split_policy()
    grid_x = (seq_len + rows_per_block - 1) // rows_per_block
    if grid_x == 0 or seq_len_kv < pol.min_seq_len_kv:
        return 1
    cu_count = _device_cu_count(device_index)
    max_splits = max(1, (seq_len_kv // block_kv) // pol.min_tiles_per_split)
    if pol.block_overhead_tiles is not None and blocks_per_cu is not None:
        # The window bounds live on the device; the full KV extent stands in.
        return _splits_by_wave_cost(
            grid_x,
            cu_count * blocks_per_cu,
            seq_len_kv // block_kv,
            max_splits,
            pol.block_overhead_tiles,
        )
    target_blocks = pol.cu_oversub * cu_count
    if grid_x >= target_blocks:
        return 1
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


# Kernel-variant registry (arch-dependent).
#
# gfx942 keeps its original ``"mfma_r<RPB>_w<WPB>"`` tags unchanged: RPB query
# rows per block, WPB waves per block, block_kv fixed at _BLOCK_KV.
#
# gfx950 variants carry the MFMA shape and block_kv in the tag, because there
# the atom and tile width both vary, and every other knob of the two builders
# has a segment too, so a gfx950 tag *is* a config:
#     "mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>"         direct load
#     "[_lds<NB>[_pd<PD>][_swp][_rs][_ls][_ts][_dg]]"
#                                                LDS pipeline (PD default 2)
#     "[_h<fma|fma2>]"                           head_acc (default add)
#     "[_wpe<N>]"                                waves_per_eu hint (default 2)
#     "[_ns<N>]"                                 fixed grid.y KV split
#                                                (default _auto_num_splits)
# The ``_lds`` suffix selects the LDS-pipelined builder, in which all WPB waves
# share one staged KV tile and partition rows, so a block owns RPB*WPB rows.
# gfx950 has no hand-picked registry: any tag in this grammar builds on demand
# (``_variant_builder``), and the tuned dispatch table ``_GFX950_ROUTES`` is a
# list of tags. Only gfx942's fixed ``mfma_r<RPB>_w<WPB>`` family is registered
# up front.
#
# Each entry hardcodes its own block_kv, overriding whatever the caller passed
# to ``compile_fp8_mqa_logits``.


def _mk_builder(
    rpb,
    wpb,
    *,
    mfma=_MFMA16,
    bkv=None,
    lds=None,
    prefetch_depth=2,
    sw_pipe=False,
    rs_head=False,
    lds_scales=False,
    head_acc="add",
    tile_skip=False,
    dma_guard=False,
):
    """Registry entry factory.

    ``lds`` is None for the direct-load builder, else the LDS slot count.
    ``prefetch_depth`` controls the software-pipeline depth for LDS variants:
    tiles 0..PD-1 are prefetched into flight before the steady-state loop, and
    each iteration issues one new DMA while waiting for the oldest in-flight tile.
    Defaults to 2. Variants with PD != 2 append ``_pd{PD}``
    to their tag so the cache and registry can hold both simultaneously.
    Constraint: ``(PD-1) * NUM_ASYNC_LOADS ≤ 63`` (gfx9 vmcnt encoding limit).

    Also records ``mfma.MFMA_M`` / ``mfma.MFMA_K`` in ``_VARIANT_MFMA_M`` /
    ``_VARIANT_MFMA_K`` so ``_resolve_variant`` can reject a ``num_heads`` or
    ``head_size`` the chosen variant's atom cannot tile (the kernel builders
    only assert this deep inside, e.g. H=16 against an MFMA_M=32 atom) with a
    clear, host-side error instead.
    """
    extra = {} if bkv is None else {"block_kv": bkv}
    if head_acc != "add":
        extra["head_acc"] = head_acc
    if lds is None:
        builder = lambda **kw: _build_kernel_mfma_r_w(
            **{**kw, **extra}, rows_per_block=rpb, waves_per_block=wpb, mfma=mfma
        )
    else:
        builder = lambda **kw: _build_kernel_mfma_lds_pipe(
            **{**kw, **extra},
            rows_per_block=rpb,
            waves_per_block=wpb,
            mfma=mfma,
            num_buffers=lds,
            prefetch_depth=prefetch_depth,
            sw_pipe=sw_pipe,
            rs_head=rs_head,
            lds_scales=lds_scales,
            tile_skip=tile_skip,
            dma_guard=dma_guard,
        )
    return builder, mfma


_VARIANT_BUILDERS = {}
_VARIANT_MFMA_M = {}
_VARIANT_MFMA_K = {}


def _register_variants(entries):
    for tag, (builder, mfma) in entries.items():
        _VARIANT_BUILDERS[tag] = builder
        _VARIANT_MFMA_M[tag] = mfma.MFMA_M
        _VARIANT_MFMA_K[tag] = mfma.MFMA_K


def _variant_lds_bytes(tag, head_size):
    """LDS bytes ``tag``'s kernel allocates at ``head_size`` (0: direct load)."""
    m = _TAG_RE.match(tag)
    if m is None or not m.group("lds"):
        return 0
    bkv = int(m.group("bkv")) if m.group("bkv") else _BLOCK_KV
    # lds_scales appends the tile's f32 kv_scales, padded to whole 64-lane DMAs.
    scale_bytes = -(-bkv // 64) * 64 * 4 if m.group("ls") else 0
    return int(m.group("lds")) * (bkv * head_size + scale_bytes)


def _variant_supports(tag, num_heads, head_size):
    """Whether ``tag``'s MFMA atom tiles ``num_heads`` x ``head_size`` and its
    LDS slots fit."""
    if _variant_builder(tag) is None:
        return False
    return (
        num_heads % _VARIANT_MFMA_M[tag] == 0
        and head_size % _VARIANT_MFMA_K[tag] == 0
        and _variant_lds_bytes(tag, head_size) <= get_lds_capacity_bytes(_ARCH)
    )


if _ARCH == "gfx942":
    _register_variants(
        {f"mfma_r{r}_w{w}": _mk_builder(r, w) for r in (1, 2, 4) for w in (1, 2, 4)}
    )

DEFAULT_VARIANT = (
    "mfma_r2_w4"
    if _ARCH == "gfx942"
    else ("mfma32x32x64_bkv64_r1_w1_lds2_swp_rs_hfma" if _ARCH == "gfx950" else None)
)

# Parses both tag schemes (see the grammar above); ``shape`` is None for the
# gfx942 tags, ``bkv`` None means _BLOCK_KV.
_TAG_RE = re.compile(
    r"^mfma(?P<shape>\d+x\d+x\d+)?(?:_bkv(?P<bkv>\d+))?"
    r"_r(?P<rpb>\d+)_w(?P<wpb>\d+)"
    r"(?:_lds(?P<lds>\d+)(?:_pd(?P<pd>\d+))?(?P<swp>_swp)?(?P<rs>_rs)?(?P<ls>_ls)?"
    r"(?P<ts>_ts)?(?P<dg>_dg)?)?"
    r"(?:_h(?P<ha>fma2?))?(?:_wpe(?P<wpe>\d+))?(?:_ns(?P<ns>\d+))?$"
)

_GFX950_ATOMS = {"32x32x64": _MFMA32_K64, "16x16x128": _MFMA16_K128}


def _parse_variant(tag):
    """(block_kv, rows_per_block_effective) for the row grid and KV splits.

    For ``_lds`` variants the WPB waves partition rows within one shared KV
    tile, so a block owns RPB*WPB rows. A short tail is masked in the kernel.
    """
    m = _TAG_RE.match(tag)
    if m is None:
        return _BLOCK_KV, 1
    bkv = int(m.group("bkv")) if m.group("bkv") else _BLOCK_KV
    rpb, wpb = int(m.group("rpb")), int(m.group("wpb"))
    return bkv, (rpb * wpb if m.group("lds") else rpb)


def _variant_launch_knobs(tag):
    """(compile-hint overrides, fixed num_splits or None) of a tag."""
    m = _TAG_RE.match(tag)
    if m is None:
        return {}, None
    hints = {"waves_per_eu": int(m.group("wpe"))} if m.group("wpe") else {}
    return hints, (int(m.group("ns")) if m.group("ns") else None)


def _variant_builder(tag):
    """The builder of ``tag``: a registered one, or on gfx950 any tag in the
    grammar above, built and registered on first use. None if neither."""
    if tag in _VARIANT_BUILDERS:
        return _VARIANT_BUILDERS[tag]
    m = _TAG_RE.match(tag) if _ARCH == "gfx950" else None
    if m is None or m.group("shape") not in _GFX950_ATOMS or not m.group("bkv"):
        return None
    lds = m.group("lds")
    _register_variants(
        {
            tag: _mk_builder(
                int(m.group("rpb")),
                int(m.group("wpb")),
                mfma=_GFX950_ATOMS[m.group("shape")],
                bkv=int(m.group("bkv")),
                lds=int(lds) if lds else None,
                prefetch_depth=int(m.group("pd") or 2),
                sw_pipe=bool(m.group("swp")),
                rs_head=bool(m.group("rs")),
                lds_scales=bool(m.group("ls")),
                head_acc=m.group("ha") or "add",
                tile_skip=bool(m.group("ts")),
                dma_guard=bool(m.group("dg")),
            )
        }
    )
    return _VARIANT_BUILDERS[tag]


@dataclass(frozen=True)
class _Route:
    """One tuned dispatch entry: ``tag`` serves ``seq_len <= max_seq_len`` and
    ``seq_len_kv <= max_seq_len_kv`` (``None``: unbounded) not taken by an
    earlier entry."""

    max_seq_len: int | None
    tag: str
    max_seq_len_kv: int | None = None


# Tuned gfx950 routes, (num_heads, head_size) class -> routes by seq_len,
# shortest first, then by seq_len_kv within one seq_len bound, the last of each
# unbounded. A route is tuned at the class's own (num_heads, head_size): other
# sizes that map to the class run untuned and may spill.
_GFX950_LARGE_HEAD_FALLBACK = "mfma32x32x64_bkv64_r1_w4_lds2_rs_hfma_wpe4"
_GFX950_ROUTES = {
    (32, 64): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds4_pd1_swp_rs_hfma_wpe3", 4096),
        _Route(256, "mfma32x32x64_bkv128_r1_w4_lds3_pd1_swp_rs_hfma"),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_lds3_pd1_swp_rs_hfma_wpe1", 4096),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_lds3_swp_rs_hfma"),
        _Route(None, "mfma32x32x64_bkv64_r1_w4_lds4_rs_hfma_wpe1"),
    ),
    (32, 128): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds4_pd1_rs_hfma_wpe3", 4096),
        _Route(256, "mfma32x32x64_bkv64_r1_w4_lds2_rs_hfma_wpe4"),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_lds3_pd1_swp_rs_hfma_wpe1", 4096),
        _Route(2048, "mfma32x32x64_bkv64_r1_w4_lds2_rs_hfma_wpe4"),
        _Route(None, "mfma16x16x128_bkv64_r1_w4_lds2_pd1_swp_rs_ls_hfma2_wpe3", 4096),
        _Route(None, "mfma32x32x64_bkv128_r1_w4_lds3_swp_rs_hfma"),
    ),
    (64, 64): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds3_pd1_rs_hfma", 4096),
        _Route(256, "mfma32x32x64_bkv128_r1_w1_lds2_pd1_swp_rs_hfma_wpe1"),
        _Route(2048, "mfma32x32x64_bkv64_r1_w1_lds3_swp_rs_hfma_wpe4", 4096),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_lds4_swp_rs_hfma_wpe1"),
        _Route(None, "mfma32x32x64_bkv128_r1_w4_lds4_swp_rs_ts_dg_hfma_wpe1", 4096),
        _Route(None, "mfma32x32x64_bkv128_r1_w4_lds4_swp_rs_hfma_wpe1"),
    ),
    (64, 128): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds4_pd1_swp_rs_hfma_wpe3", 4096),
        _Route(256, "mfma32x32x64_bkv128_r1_w4_lds3_pd1_swp_rs_hfma_wpe1"),
        _Route(2048, "mfma32x32x64_bkv64_r1_w1_lds2_pd1_swp_rs_hfma_wpe1", 4096),
        _Route(2048, "mfma32x32x64_bkv64_r2_w1_lds2_swp_rs_hfma_wpe1"),
        _Route(None, "mfma32x32x64_bkv128_r1_w4_lds3_swp_rs_hfma", 4096),
        _Route(None, "mfma32x32x64_bkv64_r1_w4_lds4_rs_ts_hfma_wpe1"),
    ),
    (128, 64): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds2_pd1_swp_rs_hfma", 4096),
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds4_swp_rs_hfma_wpe4"),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_hfma_wpe1", 4096),
        _Route(2048, "mfma32x32x64_bkv128_r1_w4_lds4_swp_rs_hfma_wpe1"),
        _Route(None, "mfma32x32x64_bkv64_r1_w1_lds2_swp_rs_hfma", 4096),
        _Route(None, "mfma32x32x64_bkv64_r1_w4_lds4_rs_hfma_wpe1"),
    ),
    (128, 128): (
        _Route(256, "mfma32x32x64_bkv64_r1_w1_lds4_pd1_swp_rs_hfma", 4096),
        _Route(256, "mfma32x32x64_bkv128_r1_w1_lds2_pd1_swp_rs_hfma_wpe1"),
        _Route(2048, "mfma32x32x64_bkv64_r1_w1_lds2_swp_rs_hfma_wpe4", 4096),
        _Route(2048, "mfma32x32x64_bkv64_r1_w4_lds4_rs_hfma_wpe1"),
        _Route(None, "mfma32x32x64_bkv64_r1_w1_lds2_swp_rs_hfma_wpe4", 4096),
        _Route(None, "mfma32x32x64_bkv64_r1_w4_lds4_rs_hfma_wpe1"),
    ),
}
for _routes in _GFX950_ROUTES.values():
    _bounds = [r.max_seq_len for r in _routes]
    _uniq = list(dict.fromkeys(_bounds))
    assert _uniq[-1] is None and _uniq[:-1] == sorted(
        set(_uniq[:-1])
    ), f"routes must have increasing bounds and end unbounded: {_bounds}"
    for _b in _uniq:
        _kv = [r.max_seq_len_kv for r in _routes if r.max_seq_len == _b]
        assert _kv[-1] is None and _kv[:-1] == sorted(
            set(_kv[:-1])
        ), f"seq_len_kv bounds must increase and end unbounded: {_kv}"


def _auto_variant(seq_len, seq_len_kv, num_heads, head_size=128):
    """Pick a variant from the problem shape.

    gfx942: RPB from ``seq_len * seq_len_kv`` thresholds; in the middle band an
        odd ``seq_len`` stays at RPB 1. WPB=2 packs more column
        tiles per wave when M and N are both large, else WPB=4 for more
        wavefronts on small-M / short-window shapes.

    gfx950: the tuned ``_GFX950_ROUTES`` table. ``num_heads`` maps to the
        nearest tuned class (<= 32, 64, >= 128) and ``head_size`` to 64 or
        128 (>= 128); within the class, the first ``_Route`` whose
        ``max_seq_len`` admits ``seq_len`` and whose ``max_seq_len_kv`` admits
        ``seq_len_kv`` wins. Routed tags use the
        32x32x64 atom (K=64), except in the 128 head_size class, where a
        16x16x128 atom (K=128) is also allowed: head sizes are powers of two,
        so every head_size >= 128 is a multiple of 128. Routes are tuned at
        head_size 64 and 128; a route whose LDS slots do not fit the device
        at a larger head_size (``_variant_lds_bytes``) is skipped, and
        ``_GFX950_LARGE_HEAD_FALLBACK`` is used if none fits.
    """
    if _ARCH == "gfx942":
        rpb2_min_elems = 2**19
        rpb4_min_elems = 2**21
        elems = seq_len * seq_len_kv
        if elems < rpb2_min_elems:
            rpb = 1
        elif elems < rpb4_min_elems:
            rpb = 2 if seq_len % 2 == 0 else 1
        else:
            rpb = 4
        wpb = 2 if (seq_len >= 2048 and seq_len_kv >= 8192) else 4
        return f"mfma_r{rpb}_w{wpb}"
    if _ARCH == "gfx950":
        h = 32 if num_heads <= 32 else (128 if num_heads >= 128 else 64)
        routes = _GFX950_ROUTES[(h, 128 if head_size >= 128 else 64)]
        # Routes are tuned at D=64/128; a larger head size may overflow a
        # route's LDS slots (a build-time error), so the first admitted route
        # whose slots fit is selected, else the fixed fallback.
        return next(
            (
                r.tag
                for r in routes
                if (r.max_seq_len is None or seq_len <= r.max_seq_len)
                and (r.max_seq_len_kv is None or seq_len_kv <= r.max_seq_len_kv)
                and _variant_lds_bytes(r.tag, head_size)
                <= get_lds_capacity_bytes(_ARCH)
            ),
            _GFX950_LARGE_HEAD_FALLBACK,
        )
    raise NotImplementedError(
        f"fp8_mqa_logits has no FlyDSL variants for arch {_ARCH!r}; "
        "supported: gfx942, gfx950"
    )


def _auto_variants():
    """Every tag ``_auto_variant`` can return (registered, so their atoms are
    known to ``_variant_supports``)."""
    if _ARCH == "gfx942":
        return [
            f"mfma_r{rpb}_w{wpb}" for rpb, wpb in itertools.product((1, 2, 4), (2, 4))
        ]
    elif _ARCH == "gfx950":
        tags = {r.tag for routes in _GFX950_ROUTES.values() for r in routes}
        for t in tags:
            _variant_builder(t)
        return sorted(tags)
    else:
        raise NotImplementedError(
            f"fp8_mqa_logits has no FlyDSL variants for arch {_ARCH!r}; "
            "supported: gfx942, gfx950"
        )


# The variants worth naming: gfx942's registry, gfx950's routed tags (any other
# tag in the gfx950 grammar also builds, on demand).
KERNEL_VARIANTS = tuple(_auto_variants() if _ARCH == "gfx950" else _VARIANT_BUILDERS)


def _resolve_variant(variant, seq_len, seq_len_kv, num_heads, head_size=128):
    """Effective variant: explicit ``variant=`` > env var > shape-adaptive."""
    tag = (
        variant
        or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        or _auto_variant(seq_len, seq_len_kv, num_heads, head_size)
    )
    if _variant_builder(tag) is None:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {tag!r} for arch {_ARCH}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    # The public API only checks num_heads is a power of two; a variant's atom
    # additionally requires num_heads % MFMA_M == 0 (e.g. gfx950's H<=32 tags
    # all use an MFMA_M=32 atom, so H=16 has no compatible variant at all).
    # Catch that here with a clear message instead of the builder's internal
    # `assert H % MFMA_M == 0` surfacing deep inside kernel construction.
    mfma_m = _VARIANT_MFMA_M[tag]
    if num_heads % mfma_m != 0:
        raise NotImplementedError(
            f"fp8_mqa_logits: num_heads={num_heads} is not a multiple of "
            f"MFMA_M={mfma_m} required by variant {tag!r} on arch {_ARCH}; "
            f"no compatible variant is registered for this num_heads on {_ARCH}"
        )
    mfma_k = _VARIANT_MFMA_K[tag]
    if head_size % mfma_k != 0:
        raise NotImplementedError(
            f"fp8_mqa_logits: head_size={head_size} is not a multiple of "
            f"MFMA_K={mfma_k} required by variant {tag!r} on arch {_ARCH}"
        )
    return tag


def _policy_num_splits(launcher, args, variant, seq_len, seq_len_kv, device_index):
    """The split policy's ``num_splits`` for ``launcher``, built from
    ``variant``; ``args`` are its launch arguments (for occupancy)."""
    bkv, rows_per_block = _parse_variant(variant)
    pol = _split_policy()
    blocks_per_cu = None
    if pol.block_overhead_tiles is not None:
        blocks_per_cu = _max_blocks_per_cu(launcher, args)
    return _auto_num_splits(
        seq_len,
        seq_len_kv,
        rows_per_block,
        bkv,
        device_index,
        blocks_per_cu,
    )


@lru_cache(maxsize=32)
def compile_fp8_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int = _BLOCK_KV,
    paged: bool = False,
    # None only on an unsupported/undetected arch, where DEFAULT_VARIANT is None
    # and no variant exists; the membership check below rejects it.
    variant: str | None = DEFAULT_VARIANT,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    ``num_heads``/``head_size`` are compile-time constants (powers of two, D in
    {64, 128}); ``variant`` is an ``mfma_r<RPB>_w<WPB>`` tag (see
    ``KERNEL_VARIANTS``); ``convert_q_fn``/``convert_kv_fn`` mark an FP8 FN
    operand whose -0 (0x80) byte the kernel patches to FNUZ +0.
    ``clean_logits`` selects whether the kernel also writes -inf to the
    out-of-window positions; like the convert flags it is a compile-time
    specialization, so the False kernel carries none of that code. ``paged`` is
    reserved for a future variant and must be False.
    """
    if paged:
        raise NotImplementedError(
            "Paged FlyDSL fp8_mqa_logits is Phase 2 and not implemented yet."
        )
    builder = None if variant is None else _variant_builder(variant)
    if builder is None:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    launcher = builder(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=block_kv,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
        clean_logits=clean_logits,
    )
    hints, _ = _variant_launch_knobs(variant)
    launcher.compile_hints = {**_DEFAULT_COMPILE_HINTS, **hints}
    return launcher


def flydsl_fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    stream=None,
    variant=None,
):
    """FlyDSL gfx942/gfx950 FP8 MQA logits -- drop-in replacement for the Triton ``fp8_mqa_logits``.

    Q:            [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:           [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:    [seq_len_kv], dtype float32
    weights:      [seq_len, NUM_HEADS], dtype float32
    cu_starts:    [seq_len], dtype int32, per-row window start (inclusive)
    cu_ends:      [seq_len], dtype int32, per-row window end (exclusive)
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i])
                  in row i are written as -inf -- by the kernel itself, as part
                  of the same launch; the output is never pre-filled. If False,
                  the kernel skips those positions and the caller owns whatever
                  is left there.
    stream:       optional HIP stream; defaults to the current stream.
    variant:      optional kernel-variant tag (see ``KERNEL_VARIANTS``). If None,
                  taken from ``FLYDSL_FP8_MQA_LOGITS_VARIANT`` or, failing that,
                  chosen adaptively from the problem shape (``_auto_variant``).

    Returns
    -------
    logits: [seq_len, seq_len_kv], dtype float32.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    # The gfx950 launcher assumes flat, contiguous row-major Q/KV: it derives
    # per-row/per-column byte offsets from the logical shape alone, with no
    # stride operand forwarded to the kernel. For the contiguous case, this is a no-op.
    Q = Q.contiguous()
    KV = KV.contiguous()

    # FlyDSL's DLPack tensor adaptor rejects 0-dim tensors, but the per-token
    # ``kv_scales`` collapses to a scalar when seq_len_kv == 1 (and ``weights``
    # could too). Reshape the 1-D / 2-D inputs back to their logical rank so the
    # kernel always sees indexable tensors (matches the Triton pointer path).
    kv_scales = kv_scales.reshape(seq_len_kv)
    weights = weights.reshape(seq_len, num_heads)
    cu_starts = cu_starts.reshape(seq_len)
    cu_ends = cu_ends.reshape(seq_len)

    # The gfx942 fp8 MFMA reads operands as e4m3 FNUZ (bias 8). For an e4m3 FN
    # operand (OCP, bias 7) the same byte encodes exactly 2x the FNUZ value (the
    # only data byte that differs is FN -0 = 0x80, which is FNUZ NaN), so we pass
    # the raw bytes through, let the kernel patch 0x80 -> +0, and undo the 2x per
    # FN operand by scaling kv_scales -- ReLU is positive-homogeneous, so
    # logits = sum_h ReLU(QK*scale)*w is preserved.
    _fnuz = torch.float8_e4m3fnuz
    _fn = torch.float8_e4m3fn
    _arch = get_gfx()
    if _arch == "gfx950":
        # gfx950's scaled MFMA atoms (_MFMA16_K128 / _MFMA32_K64) require
        # native FN operands and reject FNUZ outright.
        assert (
            Q.dtype == _fn and KV.dtype == _fn
        ), f"gfx950 fp8_mqa_logits requires native e4m3fn Q/KV; got {Q.dtype}, {KV.dtype}"
    else:
        assert Q.dtype in (_fnuz, _fn) and KV.dtype in (
            _fnuz,
            _fn,
        ), f"Q/KV must be e4m3 fp8 (fnuz or fn); got {Q.dtype}, {KV.dtype}"
    # Only gfx942 needs that conversion; other fp8 archs read operands in their
    # native dtype, so the FN->FNUZ recast there would corrupt them.
    convert_q_fn = _arch == "gfx942" and Q.dtype != _fnuz
    convert_kv_fn = _arch == "gfx942" and KV.dtype != _fnuz
    scale_mul = (2.0 if convert_q_fn else 1.0) * (2.0 if convert_kv_fn else 1.0)
    if scale_mul != 1.0:
        kv_scales = kv_scales.to(torch.float32) * scale_mul

    variant = _resolve_variant(variant, seq_len, seq_len_kv, num_heads, head_size)

    _BKV, _ROWS_PER_BLOCK = _parse_variant(variant)

    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=_BKV,
        paged=False,
        variant=variant,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
        clean_logits=bool(clean_logits),
    )

    # No torch.full even when clean_logits: the kernel writes -inf itself.
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    logits = torch.empty(
        (seq_len, seq_len_kv_aligned),
        dtype=torch.float32,
        device=Q.device,
    )[:, :seq_len_kv]

    if stream is None:
        stream = torch.cuda.current_stream()

    tensors = (Q, KV, kv_scales, weights, cu_starts, cu_ends, logits)
    scalars = (int(seq_len), int(seq_len_kv), int(logits.stride(0)))

    with torch.cuda.device(Q.device.index):
        # A tag's _ns<N> fixes the split; otherwise the arch's split policy.
        _, num_splits = _variant_launch_knobs(variant)
        if num_splits is None:
            num_splits = _policy_num_splits(
                launcher,
                (*tensors, *scalars, 1, stream),
                variant,
                seq_len,
                seq_len_kv,
                Q.device.index,
            )
        _run_compiled(launcher, *tensors, *scalars, int(num_splits), stream)

    return logits[:seq_len, :]
