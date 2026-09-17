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
