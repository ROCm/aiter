# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Host entry for the FlyDSL implicit-GEMM convolution (no DSL).

x: (N, C, D, H, W) bf16 NCDHW by default, weight: (K, C/groups, T, R, S) bf16.
Returns (N, K, Do, Ho, Wo) bf16. Layouts NCDHW/NDHWC are independent.
"""

import functools
import math
import os
import weakref

import torch

from .kernels.conv.conv3d_gfx950_utils import (
    BF16_BYTES,
    DEFAULT_TILE,
    LDG_VEC,
    MAX_DYN_DIVIDEND,
    SPLITK_AUTO_MAX_STAGING_BYTES,
    SPLITK_MAX_STAGING_BYTES,
    TILE_K,
    _as_stream,
    out_extent,
)
from .kernels.conv.conv3d_im2col import PADDING_MODES
from .kernels.conv.conv3d_implicit_gfx950 import (
    compile_conv3d_implicit,
    make_conv3d_implicit_param,
)
from .kernels.conv.conv3d_transpose import (
    TR_MAX_BIG_S,
    TR_VEC,
    compile_transpose_ncdhw_ndhwc,
)

# ---------------------------------------------------------------------------
# Which cards this op runs on
# ---------------------------------------------------------------------------

# TILE_K=32 is mfma_f32_16x16x32_bf16 (CDNA4). Allow-list so an unsupported
# card fails at the entry, including the 1x1 matmul path.
SUPPORTED_GFX = ("gfx950",)


def _check_supported_arch():
    """Refuse an arch whose MFMA this kernel is written against, before compiling."""
    from aiter.jit.utils.chip_info import get_gfx

    # get_gfx matches the tuned table / tuner stamp; strip feature suffixes.
    gfx = get_gfx().split(":", 1)[0]
    if gfx not in SUPPORTED_GFX:
        raise RuntimeError(
            f"flydsl_conv_implicit requires one of {list(SUPPORTED_GFX)}, got {gfx}. "
            f"Its MMA is mfma_f32_16x16x32_bf16 (TILE_K={TILE_K}), which is CDNA4 "
            f"only; use torch.nn.functional.conv{{1,2,3}}d on this device."
        )


# Tuned launch configs from csrc/flydsl_conv3d/conv3d_tune.py, which times
# NDHWC in and out. Layout is not in the key: an NCDHW call gets the same row.

# Lookup key = untuned CSV header. Shared with the tuner and AOT.
TUNED_KEY_COLUMNS = (
    "N",
    "C",
    "D",
    "H",
    "W",
    "K",
    "kT",
    "kH",
    "kW",
    "stride_d",
    "stride_h",
    "stride_w",
    "pad_d",
    "pad_h",
    "pad_w",
    "dil_d",
    "dil_h",
    "dil_w",
    "groups",
    "bias",
)
TUNED_RESULT_COLUMNS = ("tile_m", "tile_n", "wave_m", "wave_n", "wgm")
TUNED_DEVICE_COLUMNS = ("gfx", "cu_num")

# Result column, not part of the key. Empty cell means FlyDSL; other backends skipped.
TUNED_LIBTYPE_COLUMN = "libtype"
LIBTYPE_FLYDSL = "flydsl"

_MATMUL_FAST_PATH_INT_COLS = (
    "groups",
    "kT",
    "kH",
    "kW",
    "stride_d",
    "stride_h",
    "stride_w",
    "pad_d",
    "pad_h",
    "pad_w",
)


def _parse_tuned_bool(value) -> bool:
    """Parse a tuned-CSV bias cell. Do not use ``bool("False")``."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):  # an empty cell
        return False
    if isinstance(value, (int, float)):
        return bool(int(value))
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "nan", "<na>", "none"}:
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    raise ValueError(f"Expected True/False, got {value!r}")


def _is_matmul_shape(*, groups, kt, kh, kw, st, sh, sw, pt, ph, pw) -> bool:
    """1x1x1, unit stride, no pad: answered with ``torch.matmul``, no kernel."""
    return (
        groups == 1
        and kt == kh == kw == 1
        and st == sh == sw == 1
        and pt == ph == pw == 0
    )


def _is_matmul_fast_path(shape) -> bool:
    """``_is_matmul_shape`` for a tuned/untuned CSV row, keyed by column name."""
    vals = {c: int(shape[c]) for c in _MATMUL_FAST_PATH_INT_COLS}
    return _is_matmul_shape(
        groups=vals["groups"],
        kt=vals["kT"],
        kh=vals["kH"],
        kw=vals["kW"],
        st=vals["stride_d"],
        sh=vals["stride_h"],
        sw=vals["stride_w"],
        pt=vals["pad_d"],
        ph=vals["pad_h"],
        pw=vals["pad_w"],
    )


def _implicit_param_from_problem(
    n,
    c,
    d,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pt,
    ph,
    pw,
    dt,
    dh,
    dw,
    groups,
    has_bias,
    splitk,
    tile,
    wgm,
    out_ndhwc,
    pad_mode="zeros",
    dyn_hw=False,
):
    """Compile param for a caller-facing problem (unpadded ``C``)."""
    return make_conv3d_implicit_param(
        n,
        groups * _pad_channels(c // groups),
        d,
        h,
        w,
        k,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        pt,
        ph,
        pw,
        dt,
        dh,
        dw,
        pad_mode,
        has_bias,
        splitk,
        tile,
        wgm,
        groups,
        out_ndhwc,
        dyn_hw,
    )


# Runtime D/H/W so one artifact serves every resolution. Set 0 to compile per
# size (tiny fixed-shape workloads). BIG_IN/BIG_OUT/M-chunking stay static.
AITER_CONV3D_DYN_HW = int(os.environ.get("AITER_CONV3D_DYN_HW", "1"))


def _dyn_hw_ok(n, c_padded, d, h, w, k, npq, tile):
    """Whether variable-resolution can express this problem. Else per-shape compile."""
    if not AITER_CONV3D_DYN_HW:
        return False
    # The magic-number reciprocals are derived for dividends under 2**31, and
    # every division the gather does is of a GEMM row or a remainder of one.
    if npq >= MAX_DYN_DIVIDEND:
        return False
    # BIG_IN / BIG_OUT swap in a different addressing scheme whose reach is
    # checked against the extents themselves, so leave those static.
    if n * c_padded * d * h * w > 0x7FFFFFFF:
        return False
    if k * npq * BF16_BYTES > 0x7FFFFFFF:
        return False
    # M chunking would need the chunk count as one more runtime value; it only
    # engages past ~16M tiles, which nothing here reaches.
    tile_m, _, wave_m, wave_n = tile
    block_threads = wave_m * wave_n * 64
    grid_m = (npq + tile_m - 1) // tile_m
    return grid_m <= 0xFFFFFFFF // block_threads


# (device, shape) pairs already reported by _log_tuned_lookup, so the report
# costs one line per shape rather than one per conv call.
_TUNED_LOOKUP_LOGGED = set()


@functools.lru_cache(maxsize=1)
def _load_tuned_table():
    """``{(gfx, cu_num, *shape): (tile, wgm, splitk)}``. Empty dict on any failure."""
    try:
        import pandas as pd

        from aiter.jit.core import AITER_CONFIGS

        path = AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE
        if not path or not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        missing = [
            c
            for c in (
                *TUNED_DEVICE_COLUMNS,
                *TUNED_KEY_COLUMNS,
                *TUNED_RESULT_COLUMNS,
            )
            if c not in df.columns
        ]
        if missing:
            from aiter import logger

            logger.warning(
                f"conv3d_implicit: tuned config {path} is missing column(s) "
                f"{missing}; every conv falls back to the heuristic tile."
            )
            return {}

        has_libtype = TUNED_LIBTYPE_COLUMN in df.columns
        skipped_libtypes = {}

        table = {}
        for row in df.itertuples(index=False):
            if has_libtype:
                # An empty cell reads back as NaN, and str(NaN) is the truthy
                # "nan" -- which would look like a backend named nan and drop a
                # row the caller meant as FlyDSL.
                raw_libtype = getattr(row, TUNED_LIBTYPE_COLUMN, None)
                libtype = "" if pd.isna(raw_libtype) else str(raw_libtype).strip()
                if libtype and libtype != LIBTYPE_FLYDSL:
                    skipped_libtypes[libtype] = skipped_libtypes.get(libtype, 0) + 1
                    continue
            key = (str(row.gfx).strip(), int(row.cu_num)) + tuple(
                (
                    _parse_tuned_bool(getattr(row, c))
                    if c == "bias"
                    else int(getattr(row, c))
                )
                for c in TUNED_KEY_COLUMNS
            )
            raw_sk = getattr(row, "splitK", None)
            if raw_sk is None:
                raw_sk = getattr(row, "splitk", None)
            try:
                splitk = (
                    int(raw_sk) if raw_sk is not None and str(raw_sk) != "" else None
                )
            except (TypeError, ValueError):
                splitk = None
            if splitk is not None:
                splitk = splitk or 1
            table[key] = (
                (
                    int(row.tile_m),
                    int(row.tile_n),
                    int(row.wave_m),
                    int(row.wave_n),
                ),
                int(row.wgm),
                splitk,
            )
        if skipped_libtypes:
            from aiter import logger

            logger.info(
                f"conv3d_implicit: tuned config {path} holds rows for backends this "
                "dispatch does not serve, skipped: "
                + ", ".join(
                    f"{n} x {TUNED_LIBTYPE_COLUMN}={lt}"
                    for lt, n in sorted(skipped_libtypes.items())
                )
            )
        return table
    except Exception as exc:  # noqa: BLE001  a bad config table must never break a conv
        from aiter import logger

        logger.warning(
            f"conv3d_implicit: could not read the tuned config table "
            f"({type(exc).__name__}: {exc}); every conv falls back to the "
            f"heuristic tile."
        )
        return {}


def _log_tuned_lookup(table, dev, key, hit, borrowed=False):
    """Log once per (device, shape): hits need AITER_LOG_TUNED_CONFIG; misses always."""
    if not table or (dev, key) in _TUNED_LOOKUP_LOGGED:
        return
    _TUNED_LOOKUP_LOGGED.add((dev, key))
    if hit is not None:
        from aiter.jit.core import AITER_LOG_TUNED_CONFIG

        if not AITER_LOG_TUNED_CONFIG:
            return

    from aiter import logger

    shape = ",".join(f"{c}={v}" for c, v in zip(TUNED_KEY_COLUMNS, key))
    if hit is not None:
        splitk = hit[2]
        sk_s = f", splitK={splitk}" if splitk is not None else ""
        if borrowed:
            logger.info(
                f"conv3d_implicit: {shape} has no tuned row of its own on "
                f"gfx={dev[0]}, cu_num={dev[1]}; borrowing this layer's nearest "
                f"tuned resolution, tile={hit[0]}, wgm={hit[1]}. Tune this shape "
                f"for the exact config."
            )
            return
        logger.info(
            f"conv3d_implicit: {shape} is tuned on gfx={dev[0]}, "
            f"cu_num={dev[1]}; running tile={hit[0]}, wgm={hit[1]}{sk_s}."
        )
        return

    elsewhere = sorted({k[:2] for k in table if k[2:] == key})
    if elsewhere:
        logger.warning(
            f"conv3d_implicit: {shape} is tuned for {elsewhere} but not for this "
            f"device (gfx={dev[0]}, cu_num={dev[1]}); using the heuristic tile. "
            f"Re-run csrc/flydsl_conv3d/conv3d_tune.py on this device."
        )
    else:
        logger.info(
            f"conv3d_implicit: {shape} has no tuned row for gfx={dev[0]}, "
            f"cu_num={dev[1]}; using the heuristic tile."
        )


# Resolution fields of a lookup key vs layer identity.
_RESOLUTION_KEY_IDX = frozenset(
    TUNED_KEY_COLUMNS.index(c) for c in ("N", "D", "H", "W")
)

# Borrow only within this npq ratio of the nearest tuned row of the same layer.
_BORROW_NPQ_RATIO = 4.0


def _layer_key(key):
    """The part of a lookup key that identifies the layer, not the resolution."""
    return tuple(v for i, v in enumerate(key) if i not in _RESOLUTION_KEY_IDX)


def _key_npq(key):
    """The implicit GEMM's row count for one lookup key."""
    f = dict(zip(TUNED_KEY_COLUMNS, key))
    return (
        f["N"]
        * out_extent(f["D"], f["pad_d"], f["dil_d"], f["kT"], f["stride_d"])
        * out_extent(f["H"], f["pad_h"], f["dil_h"], f["kH"], f["stride_h"])
        * out_extent(f["W"], f["pad_w"], f["dil_w"], f["kW"], f["stride_w"])
    )


@functools.lru_cache(maxsize=1)
def _tuned_rows_by_layer():
    """The tuned table regrouped as ``{(gfx, cu_num, *layer): [(npq, hit), ...]}``."""
    by_layer = {}
    for key, hit in _load_tuned_table().items():
        dev, shape = key[:2], key[2:]
        by_layer.setdefault(dev + _layer_key(shape), []).append((_key_npq(shape), hit))
    return by_layer


def _borrow_tuned_tile(dev, key):
    """Nearest same-layer tuned tile by npq, or None. Does not borrow splitK.

    Two bars, both of which drop back to ``_pick_tile``: the row has to be
    within ``_BORROW_NPQ_RATIO`` of the npq being served, and its tile has to
    still fill the device there (``_tile_fills_device``).
    """
    rows = _tuned_rows_by_layer().get(dev + _layer_key(key))
    if not rows:
        return None
    want = _key_npq(key)
    if want <= 0:
        return None
    npq, hit = min(rows, key=lambda row: abs(row[0] - want))
    if not npq or not 1 / _BORROW_NPQ_RATIO <= want / npq <= _BORROW_NPQ_RATIO:
        return None
    f = dict(zip(TUNED_KEY_COLUMNS, key))
    if not _tile_fills_device(want, f["K"], f["groups"], hit[0], dev[1]):
        return None
    return (hit[0], hit[1], None)


def _lookup_tuned_tile(key, device):
    """Tuned (tile, wgm, splitk) for this device, or a borrowed same-layer row."""
    del device
    if key is None:
        return None
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        dev = (get_gfx(), get_cu_num())
    except Exception:  # noqa: BLE001  same: degrade to the heuristic
        return None
    table = _load_tuned_table()
    hit = table.get((*dev, *key))
    borrowed = hit is None
    if borrowed:
        hit = _borrow_tuned_tile(dev, key)
    _log_tuned_lookup(table, dev, key, hit, borrowed=borrowed)
    return hit


# Fallback tile / WGM when the table has no row.
TILE_LADDER = ((128, 128, 2, 4), (64, 64, 2, 2), (32, 32, 1, 2))

TILE_MIN_WAVES_PER_CU = 6

TILE_MIN_N_FILL = 0.75

# 256-wide N only when K/groups fills one tile (>= TILE_MIN_N_FILL).
TILE_WIDE_N = (256, 256, 2, 4)
TILE_WIDE_N_MIN_KG = int(TILE_WIDE_N[1] * TILE_MIN_N_FILL)

# L2 swizzle: needs >1 N-tile and enough blocks in flight.
WGM_L2_SWIZZLE = 8
WGM_MIN_BLOCKS_PER_CU = 4


def _num_cu(device):
    """CU count for tile/split-K heuristics: ``chip_info.get_cu_num()``, then torch, else 256."""
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        cu = int(get_cu_num())
    except Exception:  # noqa: BLE001 -- no rocminfo (CPU-only host); try torch
        cu = 0
    if cu > 0:
        return cu
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # noqa: BLE001 -- probe failure falls back to gfx950's count
        return 256


def _blocks(npq, kg, groups, tile):
    tile_m, tile_n = tile[0], tile[1]
    return ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)


def _tile_fills_device(npq, k, groups, tile, num_cu):
    """Whether ``tile`` still has the waves to fill ``num_cu`` CUs at this npq.

    The bar ``_pick_tile`` holds its own ladder to, factored out so a borrowed
    tile is held to it as well: a tuned tile transfers to another resolution of
    its layer only while M stays large enough that the tile shape is not what
    decides occupancy. Below the bar it measured up to 2.0x slower than the
    heuristic's own pick, which is why ``_borrow_tuned_tile`` drops it there.
    """
    kg = k // groups
    return _blocks(npq, kg, groups, tile) * tile[2] * tile[3] >= (
        TILE_MIN_WAVES_PER_CU * num_cu
    )


def _pick_tile(npq, k, groups, device):
    kg = k // groups
    num_cu = _num_cu(device)

    # Single-n-tile wide case first; see TILE_WIDE_N. The wave check keeps it off
    # problems too small to fill the device, where the halved M grid would hurt.
    if TILE_WIDE_N_MIN_KG <= kg <= TILE_WIDE_N[1] and _tile_fills_device(
        npq, k, groups, TILE_WIDE_N, num_cu
    ):
        return TILE_WIDE_N

    # A tile wider than kg is still worth its masked columns: it keeps more waves per
    # block and halves the A traffic per output element. Below TILE_MIN_N_FILL the
    # wasted columns take over; the wave-count check below demotes it again when the
    # problem is too small to fill the device.
    legal = [t for t in TILE_LADDER if kg >= t[1] * TILE_MIN_N_FILL] or [
        TILE_LADDER[-1]
    ]
    for tile in legal:
        if _tile_fills_device(npq, k, groups, tile, num_cu):
            return tile
    return legal[-1]


def _pick_wgm(npq, k, groups, tile, device):
    """L2 swizzle grouping for the chosen tile; see WGM_L2_SWIZZLE."""
    kg = k // groups
    if (kg + tile[1] - 1) // tile[1] < 2:
        return 1
    if _blocks(npq, kg, groups, tile) < WGM_MIN_BLOCKS_PER_CU * _num_cu(device):
        return 1
    return WGM_L2_SWIZZLE


def _dispatch(exe, *args, stream=None):
    """Run a builder's launcher, pre-compiling on first use.

    ``extra_args`` are the trailing scalars a variable-resolution conv3d takes
    on top of its tensors; empty for every other kernel. ``exe.compile``
    appends them itself, so they are only spelled out on the steady-state call.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, *getattr(exe, "extra_args", ()), _as_stream(stream))


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    big = n * c * s > 0x7FFFFFFF
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % TR_VEC == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    if big and s > TR_MAX_BIG_S:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    _dispatch(
        exe, out, x, stream=torch.cuda.current_stream() if stream is None else stream
    )
    return out


_WEIGHT_CACHE = {}


def _pad_channels(c):
    return (c + LDG_VEC - 1) // LDG_VEC * LDG_VEC


LAYOUTS = {
    3: ("NCDHW", "NDHWC"),
    2: ("NCHW", "NHWC"),
    1: ("NCW", "NWC"),
}


def _check_layouts(rank, input_layout, output_layout):
    names = LAYOUTS[rank]
    for what, v in (("input_layout", input_layout), ("output_layout", output_layout)):
        assert v in names, f"{what} must be one of {names}, got {v!r}"


def _shape_ncdhw(x, ndhwc):
    """Unpack a 5-D input in either layout to (n, c, d, h, w)."""
    if ndhwc:
        n, d, h, w, c = x.shape
    else:
        n, c, d, h, w = x.shape
    return n, c, d, h, w


def _pad_spatial(x, ndhwc, pads, mode="constant"):
    """Pad (D, H, W) with torch's (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi) ordering."""
    if mode == "constant":
        return torch.nn.functional.pad(x, ((0, 0) + pads) if ndhwc else pads), ndhwc
    if ndhwc:
        x = x.permute(0, 4, 1, 2, 3)
    return torch.nn.functional.pad(x, pads, mode=mode), False


def _big_in(n, c, groups, d, h, w, pt, ph, pw):
    """Whether the kernel's 64-bit BIG_IN address path would engage for this input."""
    cp = _pad_channels(c // groups) * groups
    return n * cp * (d + 2 * pt) * (h + 2 * ph) * (w + 2 * pw) > 0x7FFFFFFF


def _evict_weight(key, _ref):
    """weakref callback: drop the entry the dead weight was pinning."""
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _WEIGHT_CACHE[key]


def _prep_weight(w, k, kt, kh, kw, c):
    """Pack (K, C, T, R, S) -> (K, T*R*S*Cpad), memoized on the source weight."""
    anchor = w._base if w._base is not None else w
    key = w.data_ptr()
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is anchor and ent[2] == stamp:
        return ent[1]
    cp = _pad_channels(c)
    wsrc = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cp - c)) if cp != c else w
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cp)
    _WEIGHT_CACHE[key] = (
        weakref.ref(anchor, functools.partial(_evict_weight, key)),
        wk,
        stamp,
    )
    return wk


def _resolve_splitk(
    splitk, npq, crs, k, device, tile=DEFAULT_TILE, groups=1, num_cu=None
):
    """K splits to launch. ``num_cu`` is for AOT (row's CU count, no GPU)."""
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if npq * k * 4 > SPLITK_MAX_STAGING_BYTES:
        return 1
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        kg = k // groups
        base = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        if (
            npq < 4096
            or k_tiles < 16
            or kg % tile_n != 0
            or npq % tile_m != 0
            or crs % TILE_K != 0
            # Not SPLITK_MAX_STAGING_BYTES: the top half of that window is only
            # addressable through the unsigned reinterpretation the constant
            # documents, and a split nobody asked for does not go there.
            or npq * k * 4 > SPLITK_AUTO_MAX_STAGING_BYTES
        ):
            sk = 1
        else:
            num_cu = _num_cu(device) if num_cu is None else int(num_cu)
            if base >= (3 * num_cu) // 4:
                sk = 1
            else:
                sk = min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def _as_tuple(v, rank, name):
    if isinstance(v, int):
        return (v,) * rank
    t = tuple(v)
    if len(t) == 1:
        return t * rank
    assert (
        len(t) == rank
    ), f"{name} must be an int or a sequence of 1 or {rank} ints, got {tuple(v)}"
    return t


def _resolve_padding(padding, kernel, stride, dilation):
    """Normalize torch's ``padding`` argument to a (low, high) pair of per-axis triples."""
    if not isinstance(padding, str):
        p = _as_tuple(padding, 3, "padding")
        assert min(p) >= 0, f"negative padding is not supported, got (pt, ph, pw) = {p}"
        return p, p
    if padding == "valid":
        return (0, 0, 0), (0, 0, 0)
    if padding != "same":
        raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
    assert all(
        s == 1 for s in stride
    ), f"padding='same' is not supported for strided convolutions, got stride {tuple(stride)}"
    total = [dl * (kn - 1) for kn, dl in zip(kernel, dilation)]
    return tuple(t // 2 for t in total), tuple(t - t // 2 for t in total)


def _conv3d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    splitk=None,
    stream=None,
    tile=None,
    wgm=None,
    input_layout="NCDHW",
    output_layout="NCDHW",
):
    _check_layouts(3, input_layout, output_layout)

    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape

    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        assert (
            t is None or t.is_cuda
        ), f"flydsl_conv_implicit needs GPU tensors; {name} is on {t.device}"
    assert (
        x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), f"flydsl_conv_implicit is a bf16-only kernel; got x={x.dtype}, weight={weight.dtype}"
    assert bias is None or (bias.dim() == 1 and bias.numel() == k), (
        f"bias must be a 1-D tensor of {k} elements, one per output channel; "
        f"got shape {tuple(bias.shape)}"
    )
    groups = int(groups)
    assert groups >= 1, f"groups must be >= 1, got {groups}"
    assert c % groups == 0, f"in-channels {c} not divisible by groups {groups}"
    assert k % groups == 0, f"out-channels {k} not divisible by groups {groups}"
    assert wc == c // groups, f"weight in-channels {wc} != C/groups = {c // groups}"
    st, sh, sw = _as_tuple(stride, 3, "stride")

    assert (
        min(st, sh, sw) >= 1
    ), f"non-positive stride is not supported, got (st, sh, sw) = {(st, sh, sw)}"
    dt, dh, dw = _as_tuple(dilation, 3, "dilation")
    assert min(dt, dh, dw) >= 1, f"dilation must be >= 1, got {(dt, dh, dw)}"
    pad_lo, pad_hi = _resolve_padding(padding, (kt, kh, kw), (st, sh, sw), (dt, dh, dw))
    pt, ph, pw = pad_lo
    assert (
        padding_mode in PADDING_MODES
    ), f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"

    if padding_mode in ("reflect", "circular"):
        for ax, (p, ext) in enumerate(zip(map(max, pad_lo, pad_hi), (d, h, w))):
            if padding_mode == "reflect":
                assert (
                    p < ext
                ), f"reflect padding {p} must be < input extent {ext} on spatial axis {ax}"
            else:
                assert (
                    p <= ext
                ), f"circular padding {p} must be <= input extent {ext} on spatial axis {ax}"

    # Tuned-table key: caller shape, before pad/channel rewrite. Asymmetric pad → None.
    tuned_key = (
        (
            n,
            c,
            d,
            h,
            w,
            k,
            kt,
            kh,
            kw,
            st,
            sh,
            sw,
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            groups,
            bias is not None,
        )
        if padding_mode == "zeros" and pad_lo == pad_hi
        else None
    )

    if pad_lo != pad_hi:
        if padding_mode == "zeros":
            x, in_ndhwc = _pad_spatial(
                x, in_ndhwc, (0, pad_hi[2] - pw, 0, pad_hi[1] - ph, 0, pad_hi[0] - pt)
            )
        else:
            x, in_ndhwc = _pad_spatial(
                x,
                in_ndhwc,
                (pw, pad_hi[2], ph, pad_hi[1], pt, pad_hi[0]),
                mode=padding_mode,
            )
            pt = ph = pw = 0
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)

    inline_pad = padding_mode != "zeros" and bool(pt or ph or pw)
    if inline_pad and _big_in(n, c, groups, d, h, w, pt, ph, pw):
        x, in_ndhwc = _pad_spatial(
            x, in_ndhwc, (pw, pw, ph, ph, pt, pt), mode=padding_mode
        )
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
        pt = ph = pw = 0
        inline_pad = False
    pad_mode = padding_mode if inline_pad else "zeros"

    # Same predicate the tuner/AOT skip. Post-normalisation pads, so "same"→0 lands here.
    if _is_matmul_shape(
        groups=groups, kt=kt, kh=kh, kw=kw, st=st, sh=sh, sw=sw, pt=pt, ph=ph, pw=pw
    ):
        wm = weight.reshape(k, c)
        if in_ndhwc:
            y = torch.matmul(x.reshape(n * d * h * w, c), wm.t()).reshape(n, d, h, w, k)
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y if out_ndhwc else y.permute(0, 4, 1, 2, 3).contiguous()
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y.permute(0, 2, 3, 4, 1).contiguous() if out_ndhwc else y

    do = out_extent(d, pt, dt, kt, st)
    ho = out_extent(h, ph, dh, kh, sh)
    wo = out_extent(w, pw, dw, kw, sw)
    assert (
        min(do, ho, wo) >= 1
    ), f"dilated filter is larger than the padded input: output ({do}, {ho}, {wo})"
    npq = n * do * ho * wo

    if n == 0:
        empty = (0, do, ho, wo, k) if out_ndhwc else (0, k, do, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    cg = c // groups
    cgp = _pad_channels(cg)
    c_in = c
    if cgp != cg:
        if in_ndhwc:
            x = torch.nn.functional.pad(
                x.reshape(n, d, h, w, groups, cg), (0, cgp - cg)
            )
            x = x.reshape(n, d, h, w, groups * cgp)
        else:
            x = torch.nn.functional.pad(
                x.reshape(n, groups, cg, d, h, w), (0, 0, 0, 0, 0, 0, 0, cgp - cg)
            )
            x = x.reshape(n, groups * cgp, d, h, w)
    c = groups * cgp
    crs = cgp * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = (
        bias.to(torch.float32).contiguous()
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )

    x_ndhwc = x.contiguous() if in_ndhwc else _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, wc)

    def _run(the_tile, the_wgm=1, tuned_splitk=None):
        # Caller splitk= wins; else freeze the CSV value so AOT and runtime share a key.
        sk_arg = splitk if splitk is not None else tuned_splitk
        sk = _resolve_splitk(sk_arg, npq, crs, k, x.device, the_tile, groups)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
            y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        dyn_hw = _dyn_hw_ok(
            n, groups * _pad_channels(c_in // groups), d, h, w, k, npq, the_tile
        )
        exe = compile_conv3d_implicit(
            _implicit_param_from_problem(
                n,
                c_in,
                d,
                h,
                w,
                k,
                kt,
                kh,
                kw,
                st,
                sh,
                sw,
                pt,
                ph,
                pw,
                dt,
                dh,
                dw,
                groups,
                has_bias,
                sk,
                the_tile,
                the_wgm,
                out_ndhwc,
                pad_mode,
                dyn_hw,
            )
        )
        _dispatch(exe, y, x_ndhwc, w_packed, bias_arg, stream=launch_stream)
        return y, sk

    forced_wgm = None if wgm is None else max(1, int(wgm))
    tuned_splitk = None
    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_wgm = 1 if forced_wgm is None else forced_wgm
    else:
        hit = _lookup_tuned_tile(tuned_key, x.device)
        if hit is not None:
            chosen_tile, chosen_wgm, tuned_splitk = hit
            if forced_wgm is not None:
                chosen_wgm = forced_wgm
        else:
            chosen_tile = _pick_tile(npq, k, groups, x.device)
            chosen_wgm = (
                _pick_wgm(npq, k, groups, chosen_tile, x.device)
                if forced_wgm is None
                else forced_wgm
            )

    y, sk = _run(chosen_tile, chosen_wgm, tuned_splitk=tuned_splitk)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        if out_ndhwc:
            return y.view(n, do, ho, wo, k).to(torch.bfloat16)
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y


def _conv2d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    input_layout="NCHW",
    output_layout="NCHW",
    **kwargs,
):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    _check_layouts(2, input_layout, output_layout)
    sh, sw = _as_tuple(stride, 2, "stride")
    dh, dw = _as_tuple(dilation, 2, "dilation")

    if isinstance(padding, str):
        p3 = padding
    else:
        ph, pw = _as_tuple(padding, 2, "padding")
        p3 = (0, ph, pw)
    k, wc, r, s = weight.shape

    if input_layout == "NHWC":
        n, h, w, c = x.shape
        x5, in5 = x.reshape(n, 1, h, w, c), "NDHWC"
    else:
        n, c, h, w = x.shape
        x5, in5 = x.reshape(n, c, 1, h, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NHWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, sh, sw),
        padding=p3,
        dilation=(1, dh, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NHWC":
        return y5.reshape(y5.shape[0], y5.shape[2], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    input_layout="NCW",
    output_layout="NCW",
    **kwargs,
):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    _check_layouts(1, input_layout, output_layout)
    (sw,) = _as_tuple(stride, 1, "stride")
    (dw,) = _as_tuple(dilation, 1, "dilation")
    if isinstance(padding, str):
        p3 = padding
    else:
        p3 = (0, 0, _as_tuple(padding, 1, "padding")[0])
    k, wc, s = weight.shape
    if input_layout == "NWC":
        n, w, c = x.shape
        x5, in5 = x.reshape(n, 1, 1, w, c), "NDHWC"
    else:
        n, c, w = x.shape
        x5, in5 = x.reshape(n, c, 1, 1, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, 1, sw),
        padding=p3,
        dilation=(1, 1, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NWC":
        return y5.reshape(y5.shape[0], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def flydsl_conv_implicit(
    x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs
):
    """Implicit-GEMM conv; rank from ``weight.dim() - 2`` (1/2/3).

    Layouts are independent: NCDHW/NDHWC (or NCHW/NHWC, NCW/NWC). Weight stays
    KC*. NDHWC input skips the pre-transpose; NDHWC output skips the split-K
    transpose but loses the vectorised store.

    ``padding``: int, per-axis tuple, ``"valid"``, or ``"same"`` (stride 1).
    ``groups``: C and K must divide; depthwise (C/groups==1) is the slow case.
    Raises ``RuntimeError`` outside ``SUPPORTED_GFX``.
    """
    _check_supported_arch()
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError(
            f"flydsl_conv_implicit supports 1D/2D/3D; got filter rank {weight.dim()}"
        )
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    impl = {3: _conv3d_impl, 2: _conv2d_impl, 1: _conv1d_impl}[spatial_rank]
    y = impl(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        **kwargs,
    )
    return y.squeeze(0) if unbatched else y
