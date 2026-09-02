# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL FP8 MQA logits (DeepSeek lightning indexer) API.

Wraps the gfx942/gfx950 kernel builders in
``aiter.ops.flydsl.kernels.mqa_logits.fp8_mqa_logits`` with:
  - The kernel-variant registry and shape-adaptive auto-selection
    (``_auto_variant`` / ``KERNEL_VARIANTS`` / ``DEFAULT_VARIANT``).
  - A build cache keyed by shape/variant/dtype-conversion flags
    (``compile_fp8_mqa_logits``).
  - Host-side seq_len padding, output-column alignment, and the KV-column
    split (``grid.y``) heuristic that fills the device for small-M shapes.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mqa_logits.fp8_mqa_logits import (
    _MFMA16,
    _MFMA16_K128,
    _MFMA32_K64,
    _build_kernel_mfma_lds_pipe,
    _build_kernel_mfma_r_w,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "KERNEL_VARIANTS",
    "DEFAULT_VARIANT",
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
        Target total blocks as a multiple of the device CU count.
    fallback_cu : int
        Nominal CU count to assume when the device query fails.
    """

    min_seq_len_kv: int
    min_tiles_per_split: int
    cu_oversub: int
    fallback_cu: int


_SPLIT_POLICIES = {
    # Tuned on MI300X (304 CU) against the direct-load builder at BKV=128,
    # where min_tiles_per_split=8 is 1024 KV columns.
    "gfx942": _SplitPolicy(
        min_seq_len_kv=4096, min_tiles_per_split=8, cu_oversub=4, fallback_cu=304
    ),
    # Tuned on MI355X (256 CU) against the LDS-pipelined builder.
    "gfx950": _SplitPolicy(
        min_seq_len_kv=0, min_tiles_per_split=2, cu_oversub=4, fallback_cu=256
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


def _auto_num_splits(
    seq_len_padded: int,
    seq_len_kv: int,
    rows_per_block: int,
    block_kv: int,
    device_index: int,
) -> int:
    """KV-column splits (grid.y) to fill the device when the row grid is small.

    For small-M / large-N shapes the ``ceil(seq_len/RPB)`` row grid leaves the
    device block-starved; splitting each row's window across ``grid.y`` recovers
    occupancy at no correctness cost (logits[m,n] are independent across n).
    Returns 1 once the row grid alone oversubscribes the device. The three
    tuning constants are per-arch -- see ``_SPLIT_POLICIES``.
    """
    pol = _split_policy()
    grid_x = seq_len_padded // rows_per_block
    if grid_x == 0 or seq_len_kv < pol.min_seq_len_kv:
        return 1
    target_blocks = pol.cu_oversub * _device_cu_count(device_index)
    if grid_x >= target_blocks:
        return 1
    max_splits = max(1, (seq_len_kv // block_kv) // pol.min_tiles_per_split)
    return max(1, min(math.ceil(target_blocks / grid_x), max_splits))


# Kernel-variant registry (arch-dependent).
#
# gfx942 keeps its original ``"mfma_r<RPB>_w<WPB>"`` tags unchanged: RPB query
# rows per block, WPB waves per block, block_kv fixed at _BLOCK_KV.
#
# gfx950 variants carry the MFMA shape and block_kv in the tag, because there
# the atom and tile width both vary:
#     "mfma<MxNxK>_bkv<B>_r<RPB>_w<WPB>[_lds<NUM_BUFFERS>]"
# The ``_lds`` suffix selects the LDS-pipelined builder, in which all WPB waves
# share one staged KV tile and partition rows, so a block owns RPB*WPB rows.
#
# Each entry hardcodes its own block_kv, overriding whatever the caller passed
# to ``compile_fp8_mqa_logits``.


def _mk_builder(
    rpb, wpb, *, mfma=_MFMA16, bkv=None, lds=None, swizzle=True, prefetch_depth=2
):
    """Registry entry factory.

    ``lds`` is None for the direct-load builder, else the LDS slot count.
    ``prefetch_depth`` controls the software-pipeline depth for LDS variants:
    tiles 0..PD-1 are prefetched into flight before the steady-state loop, and
    each iteration issues one new DMA while waiting for the oldest in-flight tile.
    Defaults to 2. Variants with PD != 2 append ``_pd{PD}``
    to their tag so the cache and registry can hold both simultaneously.
    Constraint: ``(PD-1) * NUM_ASYNC_LOADS ≤ 63`` (gfx9 vmcnt encoding limit).

    Also records ``mfma.MFMA_M`` in ``_VARIANT_MFMA_M`` so ``_resolve_variant``
    can reject a ``num_heads`` the chosen variant's atom cannot tile (the
    kernel builders only assert this deep inside, e.g. H=16 against an
    MFMA_M=32 atom) with a clear, host-side error instead.
    """
    extra = {} if bkv is None else {"block_kv": bkv}
    if lds is None:
        builder = lambda **kw: _build_kernel_mfma_r_w(  # noqa: E731
            **{**kw, **extra}, rows_per_block=rpb, waves_per_block=wpb, mfma=mfma
        )
    else:
        builder = lambda **kw: _build_kernel_mfma_lds_pipe(  # noqa: E731
            **{**kw, **extra},
            rows_per_block=rpb,
            waves_per_block=wpb,
            mfma=mfma,
            swizzle=swizzle,
            num_buffers=lds,
            prefetch_depth=prefetch_depth,
        )
    return builder, mfma.MFMA_M


_VARIANT_BUILDERS = {}
_VARIANT_MFMA_M = {}


def _register_variants(entries):
    for tag, (builder, mfma_m) in entries.items():
        _VARIANT_BUILDERS[tag] = builder
        _VARIANT_MFMA_M[tag] = mfma_m


if _ARCH == "gfx942":
    _register_variants(
        {f"mfma_r{r}_w{w}": _mk_builder(r, w) for r in (1, 2, 4) for w in (1, 2, 4)}
    )

if _ARCH == "gfx950":
    # CDNA4 scaled MFMA atoms (K=128/64): gfx950-only, since those instructions
    # require native FN operands and do not exist on gfx942.
    _K64 = _MFMA32_K64
    _K128 = _MFMA16_K128
    _register_variants(
        {
            # -- direct load: every wave fetches its own KV tile, no LDS --
            "mfma16x16x128_bkv128_r1_w1": _mk_builder(1, 1, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r2_w1": _mk_builder(2, 1, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r1_w2": _mk_builder(1, 2, mfma=_K128, bkv=128),
            "mfma16x16x128_bkv128_r2_w2": _mk_builder(2, 2, mfma=_K128, bkv=128),
            "mfma32x32x64_bkv128_r1_w1": _mk_builder(1, 1, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r2_w1": _mk_builder(2, 1, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r1_w2": _mk_builder(1, 2, mfma=_K64, bkv=128),
            "mfma32x32x64_bkv128_r2_w2": _mk_builder(2, 2, mfma=_K64, bkv=128),
            # -- LDS double-buffered: WPB waves share one staged KV tile --
            "mfma32x32x64_bkv64_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv64_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv64_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K64, bkv=64, lds=2
            ),
            "mfma32x32x64_bkv128_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv128_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv128_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K64, bkv=128, lds=2
            ),
            "mfma32x32x64_bkv256_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K64, bkv=256, lds=2
            ),
            "mfma32x32x64_bkv256_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K64, bkv=256, lds=2
            ),
            "mfma16x16x128_bkv64_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=64, lds=2
            ),
            "mfma16x16x128_bkv128_r1_w2_lds2": _mk_builder(
                1, 2, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv128_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv128_r2_w4_lds2": _mk_builder(
                2, 4, mfma=_K128, bkv=128, lds=2
            ),
            "mfma16x16x128_bkv256_r2_w2_lds2": _mk_builder(
                2, 2, mfma=_K128, bkv=256, lds=2
            ),
            # -- LDS triple-buffered: same in-flight depth as _lds2 but the
            #    reader/writer barrier is elided (num_buffers > prefetch_depth) --
            "mfma32x32x64_bkv64_r1_w2_lds3": _mk_builder(
                1, 2, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv64_r2_w2_lds3": _mk_builder(
                2, 2, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv64_r2_w4_lds3": _mk_builder(
                2, 4, mfma=_K64, bkv=64, lds=3
            ),
            "mfma32x32x64_bkv128_r1_w2_lds3": _mk_builder(
                1, 2, mfma=_K64, bkv=128, lds=3
            ),
            "mfma32x32x64_bkv128_r2_w4_lds3": _mk_builder(
                2, 4, mfma=_K64, bkv=128, lds=3
            ),
            # -- K128/bkv64 triple-buffered (complement the _lds2 entry above).
            #
            # At H=32, mfma16x16x128 gives M_TILES=2 and N_TILES=4 per BKV-64
            # tile (8 MFMAs/wave), vs mfma32x32x64's M_TILES=1 N_TILES=2
            # (4 MFMAs/wave).  Despite the 2x MFMA advantage, K128 measured
            # slower for H=32 square shapes: MFMA_N=16 (vs 32) doubles the
            # scatter-write count per BKV tile and adds an extra shuffle in the
            # head-reduce butterfly, erasing the MFMA gain.  K64 + WPB=4 is
            # the preferred auto-selection for H<=32; these variants are kept
            # as exploration coverage and may perform better for higher H. --
            "mfma16x16x128_bkv64_r1_w2_lds3": _mk_builder(
                1, 2, mfma=_K128, bkv=64, lds=3
            ),
            "mfma16x16x128_bkv64_r2_w2_lds3": _mk_builder(
                2, 2, mfma=_K128, bkv=64, lds=3
            ),
            "mfma16x16x128_bkv128_r4_w4_lds3": _mk_builder(
                4, 4, mfma=_K128, bkv=128, lds=3
            ),
            "mfma16x16x128_bkv128_r2_w2_lds3": _mk_builder(
                2, 2, mfma=_K128, bkv=128, lds=3
            ),
        }
    )

KERNEL_VARIANTS = tuple(_VARIANT_BUILDERS.keys())
# None on an unsupported/undetected arch: there is no variant to name when
# _VARIANT_BUILDERS is empty, and compile_fp8_mqa_logits' membership check then
# rejects it with the available-variants list rather than a confusing KeyError.
DEFAULT_VARIANT = (
    "mfma_r2_w4"
    if _ARCH == "gfx942"
    else ("mfma32x32x64_bkv64_r1_w2_lds3" if _ARCH == "gfx950" else None)
)

# Parses both tag schemes; group 1 is the shape (None for the gfx942 tags),
# then block_kv (None -> _BLOCK_KV), RPB, WPB, and the LDS buffer count.
_TAG_RE = re.compile(
    r"^mfma(?P<shape>\d+x\d+x\d+)?(?:_bkv(?P<bkv>\d+))?"
    r"_r(?P<rpb>\d+)_w(?P<wpb>\d+)(?:_lds(?P<lds>\d+))?$"
)


def _parse_variant(tag):
    """(block_kv, rows_per_block_effective) for host-side padding and splitting.

    For ``_lds`` variants the WPB waves partition rows within one shared KV
    tile, so a block owns RPB*WPB rows and seq_len must be padded to that.
    """
    m = _TAG_RE.match(tag)
    if m is None:
        return _BLOCK_KV, 1
    bkv = int(m.group("bkv")) if m.group("bkv") else _BLOCK_KV
    rpb, wpb = int(m.group("rpb")), int(m.group("wpb"))
    return bkv, (rpb * wpb if m.group("lds") else rpb)


def _auto_variant(seq_len, seq_len_kv, num_heads):
    """Pick a variant from the problem shape.

    gfx942 (unchanged): RPB=2 always; WPB=2 packs more column tiles per wave
    when M and N are both large, else WPB=4 for more wavefronts on small-M /
    short-window shapes.

    gfx950 H>=128: mfma32x32x64 at r=1 always -- ample compute, more blocks.

    gfx950 H<=32: mfma32x32x64 with WPB=4 for small/square shapes,
        WPB=2 r=2 for streaming / large-square.
        K64 gives M_TILES=1 at H=32 -- half the compute of H=64 -- so the
        smaller tile grid benefits from extra wavefronts per block (WPB=4)
        rather than more blocks (WPB=2), which keeps the SIMD units busier
        when the row grid alone under-saturates the device.  For large or
        streaming shapes the row grid is already sufficient to fill the device,
        so WPB=2 with r=2 (more row reuse per KV load) is preferred.

    gfx950 H in (32, 128): mfma32x32x64 with r=2 for streaming / large-square
        shapes (KV pressure high), r=1 otherwise.
    """
    if _ARCH == "gfx942":
        wpb = 2 if (seq_len >= 2048 and seq_len_kv >= 8192) else 4
        return f"mfma_r2_w{wpb}"
    if _ARCH == "gfx950":
        if num_heads >= 128:
            return "mfma32x32x64_bkv64_r1_w2_lds3"
        streaming = seq_len_kv > 2 * seq_len
        large_square = seq_len >= 8192 and seq_len_kv >= seq_len
        if num_heads <= 32:
            if streaming or large_square:
                return "mfma32x32x64_bkv64_r2_w2_lds3"
            return "mfma32x32x64_bkv64_r2_w4_lds3"
        r = 2 if streaming or large_square else 1
        return f"mfma32x32x64_bkv64_r{r}_w2_lds3"
    raise NotImplementedError(
        f"fp8_mqa_logits has no FlyDSL variants for arch {_ARCH!r}; "
        "supported: gfx942, gfx950"
    )


def _resolve_variant(variant, seq_len, seq_len_kv, num_heads):
    """Effective variant: explicit ``variant=`` > env var > shape-adaptive."""
    tag = (
        variant
        or os.environ.get("FLYDSL_FP8_MQA_LOGITS_VARIANT")
        or _auto_variant(seq_len, seq_len_kv, num_heads)
    )
    if tag not in _VARIANT_BUILDERS:
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
    return tag


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
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(
            f"unknown fp8_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    launcher = _VARIANT_BUILDERS[variant](
        num_heads=num_heads,
        head_size=head_size,
        block_kv=block_kv,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
        clean_logits=clean_logits,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
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
    # stride operand forwarded to the kernel (unlike the Triton launcher,
    # which does pass strides). A strided view (e.g. ``base[:, ::2]``) would
    # silently be read at the wrong offsets, so make both contiguous here --
    # a no-op copy in the common already-contiguous case.
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

    variant = _resolve_variant(variant, seq_len, seq_len_kv, num_heads)

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

    # The kernels require seq_len padded to a multiple of the rows a block owns,
    # so every block owns exactly that many. Padded rows get empty windows
    # (start == end == 0) so the kernel writes nothing for them; the output is
    # sliced back to the original seq_len after the launch.
    seq_len_padded = (
        (seq_len + _ROWS_PER_BLOCK - 1) // _ROWS_PER_BLOCK
    ) * _ROWS_PER_BLOCK
    if seq_len_padded != seq_len:
        pad = seq_len_padded - seq_len
        Q = torch.cat([Q, Q.new_zeros((pad, num_heads, head_size))], dim=0)
        weights = torch.cat([weights, weights.new_zeros((pad, num_heads))], dim=0)
        cu_starts = torch.cat([cu_starts, cu_starts.new_zeros(pad)], dim=0)
        cu_ends = torch.cat([cu_ends, cu_ends.new_zeros(pad)], dim=0)

    # No torch.full even when clean_logits: the kernel now writes -inf itself,
    # at exactly the out-of-window positions it would otherwise skip.
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    logits = torch.empty(
        (seq_len_padded, seq_len_kv_aligned),
        dtype=torch.float32,
        device=Q.device,
    )[:, :seq_len_kv]

    num_splits = _auto_num_splits(
        seq_len_padded, seq_len_kv, _ROWS_PER_BLOCK, _BKV, Q.device.index
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.device(Q.device.index):
        _run_compiled(
            launcher,
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            int(seq_len_padded),
            int(seq_len_kv),
            int(logits.stride(0)),
            int(num_splits),
            stream,
        )

    return logits[:seq_len, :]
