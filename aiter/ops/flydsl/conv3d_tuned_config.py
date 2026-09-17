# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tuned launch configs for the FlyDSL implicit-GEMM conv3d.

The runtime counterpart to ``csrc/flydsl_conv3d/conv3d_tune.py``: it reads the
rows that tuner writes and hands ``_conv3d_impl`` the tile and WGM to launch
with, or nothing when this shape was never tuned on this device.

Split out of ``kernels/conv3d_implicit.py`` so the kernel module holds the DSL
and its host dispatch, the way ``tuned_gemm.py`` keeps the GEMM lookups out of
``kernels/gemm_a16w16_gfx950.py``. Nothing here touches FlyDSL, and nothing here
imports the kernel module, so the tuner and the AOT pass can read the column
order without pulling in the compiler.
"""

import functools
import os

import torch

# Column order of the conv3d_bf16_untuned family header, and therefore of the
# lookup key. Single source for the three readers of that CSV: this lookup,
# csrc/flydsl_conv3d/conv3d_tune.py, and aiter/aot/flydsl/conv.py.
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

# (device, shape) pairs already reported by _log_tuned_lookup, so the report
# costs one line per shape rather than one per conv call.
_TUNED_LOOKUP_LOGGED = set()


@functools.lru_cache(maxsize=1)
def _load_tuned_table():
    """Parse the tuned config CSV into ``{(gfx, cu_num, *shape): (tile, wgm)}``.

    The device columns stay in the key instead of filtering the frame, as in
    ``gemm_op_a8w8.get_CKGEMM_config`` and ``tuned_gemm.get_GEMM_A16W16_config``:
    a miss can then tell "tuned, but on another device" from "never tuned", and
    _log_tuned_lookup says which.

    Returns an empty dict on any failure. A missing or malformed table must
    degrade to the heuristic, never break a conv.
    """
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

        table = {}
        for row in df.itertuples(index=False):
            key = (str(row.gfx).strip(), int(row.cu_num)) + tuple(
                bool(getattr(row, c)) if c == "bias" else int(getattr(row, c))
                for c in TUNED_KEY_COLUMNS
            )
            table[key] = (
                (
                    int(row.tile_m),
                    int(row.tile_n),
                    int(row.wave_m),
                    int(row.wave_n),
                ),
                int(row.wgm),
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


def _log_tuned_lookup(table, dev, key, hit):
    """Report once per (device, shape) where this conv's launch config came from.

    A hit is reported only under AITER_LOG_TUNED_CONFIG, as in the GEMM
    lookups; a miss is reported unconditionally, since dropping a tuned config
    without a word is the outcome worth seeing. An empty table is silent either
    way: shipping no tuned rows for this op is the normal state, and the
    heuristic is the intended answer there.
    """
    if not table or (dev, key) in _TUNED_LOOKUP_LOGGED:
        return
    from aiter import logger

    shape = ",".join(f"{c}={v}" for c, v in zip(TUNED_KEY_COLUMNS, key))
    if hit is not None:
        from aiter.jit.core import AITER_LOG_TUNED_CONFIG

        if not AITER_LOG_TUNED_CONFIG:
            return
        _TUNED_LOOKUP_LOGGED.add((dev, key))
        logger.info(
            f"conv3d_implicit: {shape} is tuned on gfx={dev[0]}, "
            f"cu_num={dev[1]}; running tile={hit[0]}, wgm={hit[1]}."
        )
        return

    _TUNED_LOOKUP_LOGGED.add((dev, key))
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


def _lookup_tuned_tile(key, device):
    """Offline-tuned launch config for this exact problem, or None."""
    if key is None:
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        # Not chip_info.get_gfx_runtime(), which the GEMM lookups use: that reads
        # rocminfo's first GPU, while this has to answer for the device the conv
        # launches on. The two agree except on a mixed-arch host.
        dev = (props.gcnArchName.split(":")[0], props.multi_processor_count)
    except Exception:  # noqa: BLE001  same: degrade to the heuristic
        return None
    table = _load_tuned_table()
    hit = table.get((*dev, *key))
    _log_tuned_lookup(table, dev, key, hit)
    return hit


# ---------------------------------------------------------------------------
# Fallback heuristic: the tile and WGM to run when the table has no row.
#
# These live here rather than in conv3d_policy because they are the runtime
# decision, as tuned_gemm's default_config is, while the policy module is the
# tuner's candidate enumeration -- and because the kernel module calls them, so
# putting them in the policy would make the kernel import a module that imports
# the kernel back. _resolve_splitk stays in the kernel: its window is asserted
# inside the kernel body, and it keys on TILE_K and DEFAULT_TILE.
# ---------------------------------------------------------------------------

TILE_LADDER = ((128, 128, 2, 4), (64, 64, 2, 2), (32, 32, 1, 2))

TILE_MIN_WAVES_PER_CU = 6

TILE_MIN_N_FILL = 0.75

# A 256-wide N tile only pays when it spans K/groups in ONE tile: it halves the M tiles
# (and with them the A traffic per output element) for at most 1/(1-TILE_MIN_N_FILL) of
# masked columns. Past 256 the second tile is mostly mask -- measured on gfx950, K/groups
# = 384 is 1.5x slower on 256x256 than on three clean 128-wide tiles, while K/groups = 192
# is 1.08-1.17x faster. Hence a closed range, not a "wider is better" ladder step.
TILE_WIDE_N = (256, 256, 2, 4)
TILE_WIDE_N_MIN_KG = int(TILE_WIDE_N[1] * TILE_MIN_N_FILL)

# Grouped-M L2 swizzle. It only has something to reuse when the N grid has more than one
# tile (with a single n-tile the regrouping is a no-op that still costs index math), and
# it needs enough blocks in flight for the grouped weight tile to stay hot. Below this it
# measured neutral-to-negative on every VAE shape.
WGM_L2_SWIZZLE = 8
WGM_MIN_BLOCKS_PER_CU = 4


def _num_cu(device):
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # noqa: BLE001 -- probe failure falls back to gfx950's count
        return 256


def _blocks(npq, kg, groups, tile):
    tile_m, tile_n = tile[0], tile[1]
    return ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)


def _pick_tile(npq, k, groups, device):
    kg = k // groups
    target = TILE_MIN_WAVES_PER_CU * _num_cu(device)

    # Single-n-tile wide case first; see TILE_WIDE_N. The wave check keeps it off
    # problems too small to fill the device, where the halved M grid would hurt.
    if (
        TILE_WIDE_N_MIN_KG <= kg <= TILE_WIDE_N[1]
        and _blocks(npq, kg, groups, TILE_WIDE_N) * TILE_WIDE_N[2] * TILE_WIDE_N[3]
        >= target
    ):
        return TILE_WIDE_N

    # A tile wider than kg is still worth its masked columns: it keeps more waves per
    # block and halves the A traffic per output element. Below TILE_MIN_N_FILL the
    # wasted columns take over; the wave-count check below demotes it again when the
    # problem is too small to fill the device.
    legal = [t for t in TILE_LADDER if kg >= t[1] * TILE_MIN_N_FILL] or [
        TILE_LADDER[-1]
    ]
    for tile_m, tile_n, wave_m, wave_n in legal:
        if _blocks(npq, kg, groups, (tile_m, tile_n)) * wave_m * wave_n >= target:
            return (tile_m, tile_n, wave_m, wave_n)
    return legal[-1]


def _pick_wgm(npq, k, groups, tile, device):
    """L2 swizzle grouping for the chosen tile; see WGM_L2_SWIZZLE."""
    kg = k // groups
    if (kg + tile[1] - 1) // tile[1] < 2:
        return 1
    if _blocks(npq, kg, groups, tile) < WGM_MIN_BLOCKS_PER_CU * _num_cu(device):
        return 1
    return WGM_L2_SWIZZLE
