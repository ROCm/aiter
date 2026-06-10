# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Linear Attention Prefill K5 host wrapper (gated delta rule).

This module hosts ``chunk_gated_delta_rule_fwd_h_flydsl`` -- the host
wrapper around the K5 hidden-state recurrence FlyDSL kernel
(``compile_chunk_gated_delta_h``). It performs PyTorch tensor
preparation, chooses ``BV`` with a rule-based grid/CU heuristic, manages
the compiled kernel cache, and handles the launch stream. The kernel-
compile module ``kernels.chunk_gated_delta_h`` is kept ``torch``-free,
mirroring the layering used by ``kernels.gdr_decode``.

For an end-to-end GDN forward that uses this K5 wrapper, call
``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` with
``use_chunk_flydsl=True``.
"""

from __future__ import annotations

import csv
import functools
import math
import warnings
from pathlib import Path

import torch
import triton
from packaging.version import Version

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from ..triton._triton_kernels.gated_delta_rule.utils import (
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)


# The K5 kernel passes every tensor slot as ``fx.Pointer`` (raw data pointer).
# The kernel body wraps each one as ``GTensor(..., shape=(-1,))`` and never
# reads the FlyDSL memref shape/stride, so the pointer ABI produces identical
# device code while skipping the per-launch DLPack export + layout-buffer
# packing that the default layout-dynamic ``fx.Tensor`` memref incurs under
# flydsl >=0.2.0. ``fx.Pointer`` host wrapping (``flyc.from_c_void_p`` +
# ``PointerAdaptor`` fast dispatch) only exists from 0.2.0, so this op alone
# requires 0.2.0 (the rest of aiter.ops.flydsl only needs >=0.1.8).
_K5_MIN_FLYDSL_VERSION = Version("0.2.0")
_installed_flydsl_version = Version(
    getattr(flydsl, "__version__", "0").split("+")[0]
)
if _installed_flydsl_version < _K5_MIN_FLYDSL_VERSION:
    raise ImportError(
        "FlyDSL K5 linear-attention prefill requires `flydsl` "
        f">=`{_K5_MIN_FLYDSL_VERSION}` (for the fx.Pointer argument ABI), "
        f"but got `{getattr(flydsl, '__version__', 'unknown')}`."
    )


def _as_ptr(t: torch.Tensor):
    """Wrap a torch tensor as a flydsl ``Pointer`` argument (raw data ptr).

    Uses ``fx.Uint8`` element type: the K5 kernel re-types every slot inside
    the body via ``GTensor(ptr, dtype=...)``, so the host-side element type is
    irrelevant to codegen and only needs to be a valid 1-byte unit.
    """
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP K5 convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
]


# -- K5 host wrapper (FlyDSL kernel + rule-based BV selection) ------------

_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16

# The trace-calibrated BV carve-outs in ``_target_bv_for_shape`` were swept
# exclusively on gfx950 (V=128, BT=64, 256 CUs). They assume gfx950's CU
# count and wave-occupancy behavior, so they are gated to gfx950 -- on any
# other arch the carve-outs return ``None`` and the generic CU-fill default
# applies instead (matches the pre-calibration behavior, no regression).
# ``get_rocm_arch()`` may return a feature-suffixed string like
# ``gfx950:sramecc+:xnack-``; normalize before matching.
_IS_GFX950 = get_rocm_arch().split(":")[0].startswith("gfx950")

# gfx950 has 256 CUs. Used as the fallback CU count when the live device
# query is unavailable (e.g. CPU-only meta runs).
_GFX950_CU_COUNT = 256

# ---------------------------------------------------------------------------
# Tuned-csv BV lookup (csv-best preferred, rule-based fallback)
# ---------------------------------------------------------------------------
# Offline-tuned table mapping a shape family to its measured-best BV. This is
# the SAME csv the AOT path (``aiter/aot/flydsl/chunk_gdn_h.py``) uses as its
# pre-compile seed list. At runtime we consult it FIRST for an exact-match
# best BV; on a miss (the table is sparse -- a few dozen rows) we fall back to
# the rule-based ``_target_bv_for_shape`` / CU-fill heuristic, so coverage is
# never worse than the pure-rule path.
#
# Built once per process (mirrors ``GDR_GLOBAL_CONFIG_MAP`` in
# ``linear_attention_kernels``): we keep, per key, the BV of the row with the
# smallest measured ``duration``.
_TUNED_BV_CSV = Path(__file__).resolve().parent / "chunk_gdn_h_tuned.csv"
# key = (arch, dtype, K, V, BT, H, Hg, T_flat, N, is_varlen) -> (BV, duration)
_TUNED_BV_MAP: dict[tuple, tuple[int, float]] | None = None


def _parse_csv_bool(s: str) -> bool:
    return str(s).strip() in ("True", "true", "1", "yes")


def _load_tuned_bv_map() -> dict[tuple, tuple[int, float]]:
    """Parse ``chunk_gdn_h_tuned.csv`` into a best-BV lookup, once per process.

    Keeps the BV from the minimum-``duration`` row for each shape-family key.
    Malformed rows are skipped so a partially hand-edited csv can never break
    the runtime path (we just fall back to the rule for those shapes).
    """
    global _TUNED_BV_MAP
    if _TUNED_BV_MAP is not None:
        return _TUNED_BV_MAP

    out: dict[tuple, tuple[int, float]] = {}
    try:
        with open(_TUNED_BV_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    key = (
                        row["arch"],
                        row["dtype"],
                        int(row["K"]),
                        int(row["V"]),
                        int(row["BT"]),
                        int(row["H"]),
                        int(row["Hg"]),
                        int(row["T_flat"]),
                        int(row["N"]),
                        _parse_csv_bool(row["is_varlen"]),
                    )
                    bv = int(row["BV"])
                    dur = float(row["duration"])
                except (KeyError, TypeError, ValueError):
                    continue
                prev = out.get(key)
                if prev is None:
                    out[key] = (bv, dur)
                    continue
                # Ambiguity guard: the lookup key intentionally omits the
                # use_g/use_gk/use_h0/store_fs/save_vn/wu_contig switches (they
                # are not tuned as independent BV dimensions today). If a future
                # csv edit introduces two rows that share this key but disagree
                # on BV, the min-duration row silently wins -- which could be a
                # row tuned under a different switch combo than the caller's.
                # Surface it so the csv author can decide whether the switch
                # belongs in the key.
                if bv != prev[0]:
                    warnings.warn(
                        "FlyDSL K5 tuned csv: conflicting BV for shape key "
                        f"{key}: BV={prev[0]} (dur={prev[1]:.1f}) vs BV={bv} "
                        f"(dur={dur:.1f}); keeping the lower-duration row. If "
                        "these rows differ only by a use_*/store_fs/save_vn/"
                        "wu_contig switch, consider adding that switch to the "
                        "lookup key.",
                        stacklevel=2,
                    )
                if dur < prev[1]:
                    out[key] = (bv, dur)
    except OSError:
        # No csv on disk (e.g. trimmed deployment): rule-only path.
        pass

    _TUNED_BV_MAP = out
    return out


def _lookup_csv_bv(
    *,
    dtype_str: str | None,
    K: int | None,
    BT: int | None,
    H: int,
    Hg: int,
    V: int,
    T_flat: int,
    N: int,
    is_varlen: bool,
) -> int | None:
    """Exact-match best BV from the tuned csv, or ``None`` on miss.

    Returns ``None`` whenever any key field needed to form the csv key is
    unavailable (``dtype_str`` / ``K`` / ``BT`` are optional on the
    ``_heuristic_bv`` signature for backward compatibility), so callers that
    don't pass them simply skip straight to the rule-based path.
    """
    if dtype_str is None or K is None or BT is None:
        return None
    table = _load_tuned_bv_map()
    if not table:
        return None
    hit = table.get(
        (
            get_rocm_arch(),
            dtype_str,
            K,
            V,
            BT,
            H,
            Hg,
            T_flat,
            N,
            is_varlen,
        )
    )
    return hit[0] if hit is not None else None


@functools.lru_cache(maxsize=None)
def _cu_count(device_index: int) -> int:
    """Number of compute units (CTA "wave" width) for the target device.

    The grid-fill heuristic targets "one wave of CTAs over the device's
    CUs"; the wave width is the CU count, which is arch/SKU-specific (256
    on gfx950, but differs on gfx942 and others). We read it from the live
    device properties (``multi_processor_count``) instead of hardcoding
    256, falling back to the gfx950 value only when the query fails.
    """
    try:
        idx = device_index if device_index >= 0 else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        cu = int(getattr(props, "multi_processor_count", 0) or 0)
        if cu > 0:
            return cu
    except Exception:
        pass
    return _GFX950_CU_COUNT


# ---------------------------------------------------------------------------
# Host-side overhead caches
# ---------------------------------------------------------------------------
# The flyc launcher requires every tensor argument to be an ``fx.Tensor``
# (``None`` is not accepted), and the K5 kernel reads the offset arrays as
# int32 (``GTensor(..., dtype=T.i32, ...)``). The Triton-side cached prologue
# helpers (``prepare_chunk_offsets`` / ``prepare_rebased_cu_seqlens``) return
# int64, so the launch path was previously doing ``.to(torch.int32)`` on
# every forward, even though the underlying int64 tensor is identity-stable
# across forwards thanks to ``@tensor_cache``. We sidestep that by:
#
#   1. ``_as_int32``: attaches the int32 view directly onto the int64 tensor
#      as a private attribute (``Tensor`` objects accept arbitrary
#      attributes). The first forward casts once; every subsequent forward
#      is a pure ``getattr`` -- no ATen op dispatch, no allocator hit, no
#      D2D copy. Lifetime is bound to the int64 tensor itself, so when the
#      upstream ``@tensor_cache`` evicts an entry the int32 copy is freed
#      automatically (no ``id``-recycling hazard, unlike a global dict).
#
#   2. Null-arg launch placeholder: on the ``cu_seqlens is None`` (batched)
#      path the kernel reads neither ``cu_seqlens`` nor ``chunk_offsets``
#      (and the ``use_*`` guards skip the optional gate/state loads), but
#      ``@flyc.jit`` still requires a real tensor in every slot. A local
#      ``torch.empty(0)`` (one per forward, MoE-style) satisfies that
#      without any global cache.
_INT32_ATTR = "_flydsl_int32_view"
_PROLOGUE_ATTR = "_flydsl_prologue_cache"


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    """Return an int32 narrowing of ``t``, cached on the tensor itself.

    ``t`` is expected to come from one of the ``@tensor_cache``-decorated
    prologue helpers (so its identity is stable across forwards). The
    cached int32 result lives as an attribute on ``t`` itself, which keeps
    cache invalidation trivially correct: when the upstream cache evicts
    ``t``, the int32 copy is collected with it.
    """
    if t.dtype == torch.int32:
        return t
    cached = getattr(t, _INT32_ATTR, None)
    if cached is None:
        cached = t.to(torch.int32)
        try:
            object.__setattr__(t, _INT32_ATTR, cached)
        except (AttributeError, TypeError):
            # Some tensor subclasses or autograd-tracked tensors disallow
            # ad-hoc attributes; fall back to the uncached cast (still
            # correct, just no longer hot-path-optimised for this caller).
            pass
    return cached


def _resolve_prologue(
    cu_seqlens: torch.Tensor,
    BT: int,
    num_decodes: int,
    num_decode_tokens: int,
):
    """Resolve the per-shape varlen prologue in one cached lookup.

    Each of ``prepare_chunk_offsets`` / ``prepare_num_chunks`` /
    ``prepare_rebased_cu_seqlens`` is already ``@tensor_cache``-decorated, so
    every call is "just" a tuple compare + dict lookup. That is still
    ~0.55us each via the upstream Python wrapper (≈1.7us total across the
    three calls), so we collapse them into a single tuple attached
    directly to ``cu_seqlens`` (keyed by ``(BT, num_decodes,
    num_decode_tokens)``). After the first forward on a given
    ``cu_seqlens`` tensor, this is one ``getattr`` + one dict get on a
    tiny dict, ~0.15us.

    The exact minimum segment length (``min_seqlen``) is cached here too.
    Reading it requires a ``min().item()`` host<->device sync (~5us), but the
    sync is paid once per ``cu_seqlens`` tensor, not per forward. That lets
    the BV heuristic keep the min-based balanced-split carve-outs without a
    separate launch-plan cache.

    Returns ``(NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)``.
    """
    cache_key = (BT, num_decodes, num_decode_tokens)
    cache = getattr(cu_seqlens, _PROLOGUE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            object.__setattr__(cu_seqlens, _PROLOGUE_ATTR, cache)
        except (AttributeError, TypeError):
            # Subclass disallows ad-hoc attrs; fall through to recomputing
            # (still correct, just slower).
            cache = None
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    chunk_offsets = prepare_chunk_offsets(
        cu_seqlens, BT, num_decodes, num_decode_tokens
    )
    NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
    kernel_cu_seqlens = prepare_rebased_cu_seqlens(
        cu_seqlens, num_decodes, num_decode_tokens
    )
    N = len(kernel_cu_seqlens) - 1
    if N >= 1:
        seg_lens = kernel_cu_seqlens[1:] - kernel_cu_seqlens[:-1]
        min_seqlen = int(seg_lens.min().item())
    else:
        min_seqlen = None
    result = (NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)
    if cache is not None:
        cache[cache_key] = result
    return result


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]


def _grid_ctas(*, H: int, V: int, N: int, BV: int) -> int:
    return max(1, N) * H * ((V + BV - 1) // BV)


def _select_bv_for_grid(*, H: int, V: int, N: int, target_ctas: int) -> int:
    """Choose the largest legal BV whose grid still covers target_ctas."""
    legal = sorted(_legal_bv_candidates(V), reverse=True)
    if not legal:
        return _DEFAULT_BV
    for bv in legal:
        if _grid_ctas(H=H, V=V, N=N, BV=bv) >= target_ctas:
            return bv
    # If even BV=16 cannot reach the target, use it to maximize grid size.
    return legal[-1]


def _target_bv_for_shape(
    *,
    H: int,
    Hg: int,
    T_flat: int,
    N: int,
    is_varlen: bool,
    min_seqlen: int | None = None,
) -> int | None:
    """Return the calibrated BV regime before legality/grid adjustment.

    Calibration scope (gfx950, V=128, BT=64, is_varlen=True):
      * H==32, Hg==16  : N in {2, 3} swept on T_flat in [2000, 25000].
        Outside that (T_flat, N) cube the rule deliberately returns
        ``None`` so the grid-fill default applies -- matches the
        pre-20260604 behavior. The N=2 / N=3 carve-outs are the only
        new behavior; everything else is preserved exactly.
      * H==16          : 32k-context many-seq carve-out (unchanged).

    Args:
        min_seqlen: smallest segment length in the (varlen) batch, i.e.
            ``min(cu_seqlens[1:] - cu_seqlens[:-1])``. Used by the
            "balanced-split" carve-outs to distinguish "segments ~= T/N each"
            from "one dominant segment + small tails" -- a distinction the
            average length cannot make (a balanced and a skewed split can
            share the same average), so the EXACT min is required. The
            caller computes it once and caches it on the ``cu_seqlens``
            tensor (see ``_resolve_prologue``) so the per-forward path pays
            no ``.item()`` host<->device sync. When None, those sub-rules
            fall through to the T_flat-only logic.

    If you extend this rule, please keep:
      (a) every return statement guarded by an explicit T_flat range
          you actually measured;
      (b) the "no data -> return None" fallthrough at the end of each
          branch so untested combos can't silently regress.

    All carve-outs below were swept on gfx950 only, so they are skipped on
    other arches (``_IS_GFX950`` guard) -- non-gfx950 falls through to the
    generic CU-fill default, preserving the pre-calibration behavior.
    """
    if not _IS_GFX950:
        return None
    if is_varlen and H == 32 and Hg == 16:
        if N == 2:
            # Calibrated range: T_flat in [2000, 25000]. Two flips
            # measured on H=32/Hg=16/V=128/gfx950 (notes:
            # _bv_sweep_n2_20260604):
            #   T_flat <  8000 : BV=32 (grid 256, exactly one wave)
            #   8000 <= T_flat < 12000 : BV=16
            #   12000 <= T_flat < 13000 : BV=32 (narrow tail-fit window)
            #   13000 <= T_flat <= 25000 : BV=16
            # T_flat outside [2000, 25000] is NOT covered; fall through.
            #
            # Balanced-split carve-out (bench20260604_051030, n=2 T~16k
            # cluster, 134 cases): when both segments are roughly
            # balanced (min_seqlen >= 6300), BV=32 wins by 17-76us per
            # case across T_flat in [12000, 20000]. The earlier
            # T_flat-only rule misses this because it assumed n=2 with
            # T>=13000 was always "long single-segment dominant"; large
            # min_seqlen indicates the opposite (two comparable runs).
            # The (T_flat, min_seqlen) window was sweep-validated to
            # avoid any regression vs the T_flat-only rule on 44
            # measured (T_flat, head) points (notes: _bv_sweep_n2_balanced).
            if (
                12000 <= T_flat <= 20000
                and min_seqlen is not None
                and min_seqlen >= 6300
            ):
                return 32
            if 2000 <= T_flat <= 25000:
                if T_flat < 8000:
                    return 32
                if 12000 <= T_flat < 13000:
                    return 32
                return 16
            # else: untested range, fall through to default
        elif N == 3:
            # Calibrated range: T_flat in [8000, 30000]. Across this
            # whole range BV=32 (grid=N*H*ceil(V/32) = 384, ~1.5 waves
            # on 256 CUs) measured 22-95us faster than the prior BV=64
            # / grid-fill choice, including the bench20260603 cluster
            # T~=16384 cu=[0,head,head+10000,T] (~85us per case, 200+
            # cases). T_flat outside this range is NOT covered; fall
            # through.
            #
            # Balanced-split carve-out (notes: _bv_sweep_n3_balanced,
            # 20 measured (T_flat, min_seg, max_seg) points): when the
            # smallest segment is >= 3000 the three segments are large
            # enough that BV=64 (grid 192, exactly 0.75 wave on 256 CUs)
            # wins by 11-74us across T_flat in [10000, 25000]. The
            # earlier rule missed this because the original calibration
            # only swept skewed splits (head << T) where one tiny
            # segment makes BV=64 padding-bound. Validated decision
            # boundary: min_seg <= 2384 -> BV=32 still wins, min_seg
            # >= 3000 -> BV=64 wins; no regression observed on the
            # skewed-split cluster (which has min_seg << 3000).
            if T_flat >= 10000 and min_seqlen is not None and min_seqlen >= 3000:
                return 64
            if 8000 <= T_flat <= 30000:
                return 32
            # else: untested range, fall through to default
        # N==1 and N>=4 are NOT touched by this branch -- the original
        # behavior (return None -> grid-fill default) is preserved.
    if is_varlen and H == 16 and T_flat >= 32768 and N >= 7:
        return 64
    return None


def _heuristic_bv(
    *,
    H: int,
    Hg: int,
    V: int,
    T_flat: int,
    N: int,
    is_varlen: bool,
    min_seqlen: int | None = None,
    device_index: int | None = None,
    dtype_str: str | None = None,
    K: int | None = None,
    BT: int | None = None,
) -> int:
    """Pick a sensible BV for the requested shape.

    Selection order:
      1. Exact-match best BV from the offline-tuned csv
         (``chunk_gdn_h_tuned.csv``), when the full key is available and the
         row's BV is legal for this ``V``. This gives the measured optimum
         for shapes that were actually swept.
      2. Otherwise the rule-based heuristic below (CTA/CU grid-fill plus the
         trace-calibrated carve-outs), which generalizes to shapes the sparse
         csv does not cover.

    The csv table is parsed once per process, then this is just an exact-key
    dict lookup plus scalar arithmetic. It runs once per forward (MoE/GEMM-
    style host recompute), while expensive inputs such as the exact
    ``min_seqlen`` are cached separately by ``_resolve_prologue``.

    Rules calibrated against a 27-point sweep matrix on gfx950 (20 in-csv
    shapes + 7 csv-uncovered probes). The 27 points span H in
    {8,16,24,32,48,64,128} and T_local in [256, 128000]; see
    flydsl_bv_sweep.log + flydsl_heuristic_verify.log.

      * First pick a target CTA count, then choose the largest legal BV whose
        grid ``N * H * ceil(V / BV)`` still reaches that target. Larger BV
        reduces per-CTA overhead; smaller BV exposes more CTAs for CU
        utilization.

      * ``is_varlen=False`` -- target one wave of CTAs over the device's CUs
        (live ``multi_processor_count``; 256 on gfx950).

      * ``is_varlen=True`` -- the target grid depends on (H, T_local) jointly:
          H <= 8:
            short chunks target the BV=64 grid; medium chunks target BV=32;
            long chunks target BV=16.
          H in (8, 16]:
            long chunks target BV=32; shorter chunks target BV=64.
          H == 32, Hg == 16:
            target grid follows the bench333/407 production trace: single
            sequence needs BV=16 grid; N=2/3 use total-T windows; N>=4 has
            enough grid at BV=64.
          H > 16:
            target the BV=64 grid unless a more specific regime above applies.

    Coverage: the rule matches the AOT seed CSV plus the measured bench333 /
    bench407 probes used during calibration. Shapes far outside the sampled
    (H, T_local) grid may still be suboptimal; extend the calibration sweep
    when production reports new shape families.

    Args:
        H: number of v-heads (per TP rank).
        V: head_v_dim.
        T_flat: flat token count fed to the kernel (sum of context lens
            in varlen, ``B*T`` otherwise).
        N: number of sequences in the batch (varlen) or batch size.
        is_varlen: whether the kernel runs in variable-length mode.
        Hg: number of k-heads (per TP rank). Currently only used to scope
            trace-calibrated rules to the K5 H=32/Hg=16 family.

    Returns:
        A BV from ``_BV_CANDIDATES`` that satisfies ``BV <= V`` and
        ``V % BV == 0``. If the rule's first choice is illegal for this
        V (rare: V<16 or V not divisible by 16), falls back to the
        largest legal candidate, then finally to ``_DEFAULT_BV``.
    """
    # 1. csv-best exact match (preferred). Only honoured when the row's BV is
    #    legal for this V; an out-of-range / non-divisor BV in a hand-edited
    #    csv silently falls through to the rule rather than launching a bad
    #    grid.
    csv_bv = _lookup_csv_bv(
        dtype_str=dtype_str,
        K=K,
        BT=BT,
        H=H,
        Hg=Hg,
        V=V,
        T_flat=T_flat,
        N=N,
        is_varlen=is_varlen,
    )
    if csv_bv is not None and csv_bv in _legal_bv_candidates(V):
        return csv_bv

    # 2. rule-based fallback (generalizes beyond the sparse csv).
    target_bv = _target_bv_for_shape(
        H=H,
        Hg=Hg,
        T_flat=T_flat,
        N=N,
        is_varlen=is_varlen,
        min_seqlen=min_seqlen,
    )
    if target_bv is not None:
        target_ctas = _grid_ctas(H=H, V=V, N=N, BV=target_bv)
    else:
        # Generic default: target one wave of CTAs over the device's CUs.
        # Use the live CU count (256 on gfx950, differs on other arches)
        # rather than a hardcoded gfx950 value.
        idx = device_index if device_index is not None else -1
        target_ctas = _cu_count(idx)
    return _select_bv_for_grid(H=H, V=V, N=N, target_ctas=target_ctas)


@functools.lru_cache(maxsize=None)
def _get_or_compile(
    K,
    V,
    BT,
    BV,
    H,
    Hg,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
    state_bf16=False,
    g_log2_scaled=False,
):
    """Compile (and cache) the K5 kernel for one compile-time config.

    Cached via ``lru_cache`` keyed on the full compile-time constant set,
    mirroring the gemm/moe/gdr_decode flydsl ops. ``maxsize=None`` because
    the number of distinct configs is naturally bounded by the compile-time
    constant combinations.
    """
    return compile_chunk_gated_delta_h(
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        H=H,
        Hg=Hg,
        USE_G=use_g,
        USE_GK=use_gk,
        USE_INITIAL_STATE=use_h0,
        STORE_FINAL_STATE=store_fs,
        SAVE_NEW_VALUE=save_vn,
        IS_VARLEN=is_varlen,
        WU_CONTIGUOUS=wu_contig,
        STATE_DTYPE_BF16=state_bf16,
        G_IS_LOG2_SCALED=g_log2_scaled,
    )


def _resolve_state_dtype(initial_state, state_dtype):
    """Mirror the legacy state-dtype resolution. Cheap; runs every call."""
    if initial_state is not None:
        resolved = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}; pass them "
                f"consistently or omit state_dtype."
            )
    elif state_dtype is not None:
        resolved = state_dtype
    else:
        resolved = torch.float32
    if resolved not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved}."
        )
    return resolved


def chunk_gated_delta_rule_fwd_h_flydsl(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, H, T_total] f32 cumulative gate, head-major contiguous
            (matches Triton VK / HIP K5), or None. Must be a
            ``contiguous()`` tensor with stride-1 along the T dimension.
            Caller passes ``g`` in natural-log space; when
            ``use_exp2=True`` the K1+K2 producer is expected to have
            already pre-scaled ``g`` by ``log2(e)`` (i.e. ``g`` is in
            log2 space) -- this matches the Triton VK convention and is
            NOT re-scaled by this wrapper.
        gk: [T_total, H, K] f32 per-K cumulative gate (natural-log
            space), or None. Pre-scaled to log2 space inside the wrapper
            when ``use_exp2=True``, mirroring
            ``chunk_gated_delta_rule_fwd_h_opt_vk``.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.
        state_dtype: optional initial/final state dtype (float32 or bfloat16).
        use_exp2: whether ``g`` is in log2 space. Standalone K5 callers pass
            natural-log ``g`` by default; end-to-end prefill passes the Triton
            K1 ``use_exp2`` setting through explicitly.
        num_decodes: number of leading decode-only sequences to skip in
            ``cu_seqlens``. When nonzero, ``cu_seqlens`` is the ORIGINAL,
            cache-stable metadata tensor (decode prefix included) and the
            data tensors (``k/w/u/g/...``) are expected to be pre-sliced to
            the prefill region; the offsets are rebased internally via the
            cached ``prepare_rebased_cu_seqlens``.
        num_decode_tokens: number of leading decode tokens stripped from the
            data tensors; subtracted from the rebased offsets so they index
            from token 0 of the prefill region.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection prefers an exact match from ``chunk_gdn_h_tuned.csv``
    (the same file AOT uses as its pre-compilation seed list), then falls
    back to the rule-based heuristic for shapes the sparse csv does not cover.
    """
    # All shape/flag-derived launch products (BV, launch_fn, grid dims,
    # output-buffer shapes, ...) are recomputed inline per forward, MoE/GEMM-
    # style: no per-shape launch-plan cache. This is affordable because each
    # individually-expensive input is itself cached -- the compiled kernel via
    # ``_get_or_compile`` (lru_cache), the varlen prologue + exact
    # ``min_seqlen`` via the per-cu_seqlens cache in ``_resolve_prologue``, and
    # the int32 offset views via ``_as_int32``. Everything else here is cheap
    # host-side arithmetic.

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    g_log2_scaled = bool(use_exp2)

    # State-dtype validation: cheap, and raises on bad / conflicting dtypes.
    # ``state_bf16`` is the only derived bit the compile key needs.
    resolved_state_dtype = _resolve_state_dtype(initial_state, state_dtype)
    state_bf16 = resolved_state_dtype is torch.bfloat16

    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    assert K <= 256

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
        kernel_cu_seqlens = None
        is_varlen = False
        min_seqlen = None
    else:
        # The exact ``min_seqlen`` comes from the per-``cu_seqlens`` cache, so
        # its ``.item()`` host<->device sync is paid once per cu_seqlens tensor
        # (not per forward) -- letting the heuristic use the exact min without
        # a per-call stall and without a launch-plan cache.
        NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen = _resolve_prologue(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        is_varlen = True

    BV = _heuristic_bv(
        H=H,
        Hg=Hg,
        V=V,
        T_flat=T_flat,
        N=N,
        is_varlen=is_varlen,
        min_seqlen=min_seqlen,
        device_index=k.device.index if k.device.type == "cuda" else -1,
        dtype_str=str(k.dtype),
        K=K,
        BT=BT,
    )

    launch_fn = _get_or_compile(
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        True,
        state_bf16=state_bf16,
        g_log2_scaled=g_log2_scaled,
    )

    # Null-arg placeholder for the @flyc.jit slots the kernel ignores on this
    # path (one local scalar tensor allocated per forward, MoE-style -- no
    # global cache). Sized 1 (not 0) so its ``data_ptr()`` is always a valid
    # non-null device address for the launcher's unused arg slots.
    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    int32_dummy = dummy.to(torch.int32) if not is_varlen else None
    cu_arg = (
        _as_int32(kernel_cu_seqlens) if kernel_cu_seqlens is not None else int32_dummy
    )
    co_arg = _as_int32(chunk_offsets) if chunk_offsets is not None else int32_dummy
    stream = torch.cuda.current_stream(k.device)

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H

    h_shape = (B, NT, H, V, K)
    vn_shape = (B, H, T_flat, V)
    vn_dtype = u.dtype
    fs_shape = (N, H, V, K) if output_final_state else None
    fs_dtype = resolved_state_dtype if output_final_state else None
    save_vn = save_new_value

    # G contiguity guard. We only check ``is_contiguous`` (not the full
    # shape): a mismatched ``g.shape[-1]`` vs ``w.shape[2]`` (T_flat) can
    # only happen if the caller breaks the documented [B, H, T_flat]
    # contract -- in which case strides diverge and ``is_contiguous()``
    # catches the common modes (transposed view, slice) for ~50ns.
    if g is not None and not g.is_contiguous():
        raise AssertionError(
            "FlyDSL K5: ``g`` must be contiguous (head-major [B, H, T_flat] "
            f"or [H, T_flat]); got strides={g.stride()}, shape={tuple(g.shape)}."
        )

    # gk pre-scaling: still per-call work (allocates a new tensor). Cannot
    # be cached without aliasing across forwards; an upstream producer
    # change to emit log2-space gk directly would eliminate this entirely.
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2

    h = k.new_empty(h_shape)
    v_new_buf = k.new_empty(vn_shape, dtype=vn_dtype)
    final_state = (
        k.new_empty(fs_shape, dtype=fs_dtype) if fs_shape is not None else None
    )

    # The 11 tensor slots, wrapped as raw fx.Pointer args. Keep the torch
    # tensors referenced as locals (``k``/``u``/``h``/... above) so the storage
    # stays alive across the (synchronous) launch -- ``from_c_void_p`` only
    # captures the data pointer, not a reference to the tensor.
    tensor_args = tuple(
        _as_ptr(t)
        for t in (
            k,
            u,
            w,
            v_new_buf,
            g if g is not None else dummy,
            gk if gk is not None else dummy,
            h,
            initial_state if initial_state is not None else dummy,
            final_state if final_state is not None else dummy,
            cu_arg,
            co_arg,
        )
    )

    launch_fn(
        *tensor_args,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )

    return h, (v_new_buf if save_vn else None), final_state
