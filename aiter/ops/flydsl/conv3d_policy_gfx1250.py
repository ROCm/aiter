# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware launch-config enumeration for gfx1250 FlyDSL conv3d.

Closed-form legality via ``validate_launch_config_gfx1250``. Sweeps
``block_rows`` (R such that R|tile_m and cols%16==0). ``wgm`` is unused on
gfx1250 and always reported as 1 for CSV schema parity.
"""

import itertools

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.conv.conv3d_gfx1250_utils import (
    BF16_BYTES,
    DEFAULT_BLOCK_ROWS,
    LDS_PAD,
    TILE_K,
    WMMA_M,
    WMMA_N,
    make_tile_config_gfx1250,
    validate_launch_config_gfx1250,
)

__all__ = [
    "TILE_K",
    "get_flydsl_conv3d_configs_gfx1250",
    "is_legal_tile_gfx1250",
    "tile_kernel_name_gfx1250",
]

TILE_M_VALUES = (32, 64, 96, 128, 192, 256)
TILE_N_VALUES = (32, 48, 64, 96, 128, 192, 256)
WAVE_M_VALUES = (1, 2, 4)
WAVE_N_VALUES = (1, 2, 4)
# br=1 historically NaN'd on some tiles; keep the heuristic ladder's set.
BLOCK_ROWS_VALUES = (2, 4, 8, 16)

BASELINE_TILES = (
    (128, 128, 2, 4, 8),
    (128, 64, 2, 2, 8),
    (64, 64, 2, 2, 4),
    (64, 32, 2, 1, 4),
    (32, 32, 1, 1, 2),
)

MAX_N_ACC = 32
MIN_WAVES = 2
MAX_WAVES = 16


def is_legal_tile_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows):
    return (
        validate_launch_config_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows)
        is None
    )


def tile_kernel_name_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows):
    return (
        f"conv3d_implicit_gfx1250_t{tile_m}x{tile_n}_w{wave_m}x{wave_n}_r{block_rows}"
    )


def lds_bytes_tap(tile_m, tile_n, num_buffers=2):
    lds_k = TILE_K + LDS_PAD
    a = num_buffers * (tile_m * lds_k + LDS_PAD)
    b = num_buffers * (tile_n * lds_k + LDS_PAD)
    return (a + b) * BF16_BYTES


def max_lds_bytes():
    return get_lds_capacity_bytes() // 2


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _n_fill(tile_n, kg):
    return kg / (_ceil_div(kg, tile_n) * tile_n)


def _block_rows_for(tile_m):
    out = []
    for br in BLOCK_ROWS_VALUES:
        if tile_m % br == 0 and (tile_m // br) % WMMA_M == 0:
            out.append(br)
    if not out and tile_m % DEFAULT_BLOCK_ROWS == 0:
        out.append(DEFAULT_BLOCK_ROWS)
    return out


def _sweep(npq, kg, groups, num_cu, min_n_fill, min_waves, check_waste, check_grid):
    scored = []
    lds_limit = max_lds_bytes()
    for tile_m, tile_n, wave_m, wave_n in itertools.product(
        TILE_M_VALUES, TILE_N_VALUES, WAVE_M_VALUES, WAVE_N_VALUES
    ):
        waves = wave_m * wave_n
        if waves < min_waves or waves > MAX_WAVES:
            continue
        n_acc = (tile_m // wave_m // WMMA_M) * (tile_n // wave_n // WMMA_N)
        if n_acc > MAX_N_ACC:
            continue
        if lds_bytes_tap(tile_m, tile_n) > lds_limit:
            continue
        fill = _n_fill(tile_n, kg)
        if fill < min_n_fill and tile_n != kg:
            continue
        if (
            check_waste
            and tile_n > kg
            and _ceil_div(kg, tile_n) * tile_n - kg >= tile_n // 2
        ):
            continue
        blocks = _ceil_div(npq, tile_m) * groups * _ceil_div(kg, tile_n)
        if check_grid and blocks * waves < num_cu:
            continue
        for block_rows in _block_rows_for(tile_m):
            if not is_legal_tile_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows):
                continue
            # Try building cfg (asserts divisibility already covered).
            try:
                make_tile_config_gfx1250(
                    (tile_m, tile_n, wave_m, wave_n), block_rows
                )
            except AssertionError:
                continue
            scored.append(
                (
                    (-round(fill, 4), -(tile_m * tile_n), blocks, block_rows),
                    (tile_m, tile_n, wave_m, wave_n, 1, block_rows),
                )
            )
    return scored


_RELAXATIONS = (
    (0.5, MIN_WAVES, True, True),
    (0.5, MIN_WAVES, False, True),
    (0.34, 2, False, True),
    (0.0, 1, False, False),
)


def get_flydsl_conv3d_configs_gfx1250(
    npq,
    kg,
    groups,
    num_cu,
    max_configs=96,
):
    """Candidate ``(tile_m, tile_n, wave_m, wave_n, wgm=1, block_rows)`` tuples."""
    scored = []
    for min_n_fill, min_waves, check_waste, check_grid in _RELAXATIONS:
        scored = _sweep(
            npq, kg, groups, num_cu, min_n_fill, min_waves, check_waste, check_grid
        )
        if scored:
            break

    scored.sort(key=lambda item: item[0])
    configs = [config for _, config in scored[:max_configs]]

    seen = set(configs)
    for tile_m, tile_n, wave_m, wave_n, block_rows in BASELINE_TILES:
        if not is_legal_tile_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows):
            continue
        candidate = (tile_m, tile_n, wave_m, wave_n, 1, block_rows)
        if candidate not in seen:
            seen.add(candidate)
            configs.append(candidate)
    return configs
