# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware launch-config enumeration for FlyDSL conv3d.

Closed-form legality via ``validate_launch_config``. Includes 48/96/192 N tiles
so VAE Cout 96/192/384 can fill TILE_N, unlike the runtime power-of-two ladder.
"""

import itertools

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.conv.conv3d_gfx950_utils import (
    BF16_BYTES,
    MFMA_M,
    MFMA_N,
    TILE_K,
    TILES_PER_BARRIER,
    validate_launch_config,
)

__all__ = [
    "TILE_K",
    "get_flydsl_conv3d_configs",
    "is_legal_tile",
    "tile_kernel_name",
]

# Taken from the kernel rather than restated here, as gemm_a16w16_policy takes its
# from gemm_a16w16_gfx950: each of these is fixed next to the assert or the MFMA
# shape that decides it, and a second copy would drift without any symptom other
# than a candidate sweep quietly disagreeing with what can compile.
PIPE_STAGES = 2 * TILES_PER_BARRIER

TILE_M_VALUES = (64, 96, 128, 192, 256, 384)
TILE_N_VALUES = (32, 48, 64, 96, 128, 192, 256)
WAVE_M_VALUES = (1, 2, 3, 4)
WAVE_N_VALUES = (1, 2, 3, 4, 6)
WGM_VALUES = (1, 4, 8)

# Kernel table + heuristic ladder, unioned so the tuned pick cannot lose to default.
# Keep in step with conv_kernels.TILE_LADDER by hand (order is the tuner's measure order).
BASELINE_TILES = (
    (128, 128, 2, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 2, 4),
    (256, 256, 4, 4),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
    (64, 64, 2, 2),
    (32, 32, 1, 2),
)

# acc VGPRs = 4 * MI_M * MI_N per lane. Past this the kernel spills and the
# config is slower than anything it could win on tile shape.
MAX_N_ACC = 32

# Small workgroups were assumed to have too little in flight to overlap global
# latency, and this floor sat at 4. Measuring both VAE tables against the
# unpruned legal space -- every is_legal_tile config, 75 of the 76 shapes timed
# on gfx950 -- showed the assumption costs more than it saves. The floor is what
# hid the fastest config for 26 of the 29 shapes the pruned sweep lost, and the
# winners it hid are 2- and 3-wave tiles: worst case 6.85% (C=96 482x834 K=96,
# where (192,96,3,1) wins), 1.28% across the two VAEs weighted by kernel time.
#
# At 2 that weighted loss drops to 0.18% and no shape is off by more than 2.60%,
# for 17% more candidates (100.0 -> 116.9 per shape over the 76). Nearly all of
# that widening lands on kg=96, which is where the losses were: kg=3 and kg=32
# are unchanged, since _RELAXATIONS already had to drop the floor for them.
# Going below 2 is pointless -- it adds 6 candidates in total and moves no
# winner, because a legal tile needs two waves before TILE_N can reach 96.
MIN_WAVES = 2
MAX_WAVES = 16


def is_legal_tile(tile_m, tile_n, wave_m, wave_n):
    """Would ``compile_conv3d_implicit`` accept this launch config?

    Asks the kernel rather than restating its asserts, as
    ``gemm_a16w16_policy`` asks ``make_gemm_a16w16_param_and_validate``: a
    second copy of the arithmetic here would prune compilable configs the day
    it drifted, and a sweep that never measures a config leaves no trace.
    """
    return validate_launch_config(tile_m, tile_n, wave_m, wave_n) is None


def tile_kernel_name(tile_m, tile_n, wave_m, wave_n, wgm):
    return f"conv3d_implicit_t{tile_m}x{tile_n}_w{wave_m}x{wave_n}_g{wgm}"


def lds_bytes(tile_m, tile_n):
    return PIPE_STAGES * (tile_m + tile_n) * TILE_K * BF16_BYTES


def max_lds_bytes():
    """Half the per-workgroup LDS the chip table reports for this arch.

    A candidate above half cannot keep two workgroups resident, which costs more
    latency hiding than a wider tile buys. Read at call time, not import: the
    figure is per-arch, and hardcoding CDNA3/4's 160 KiB here would silently
    mis-prune anywhere else.
    """
    return get_lds_capacity_bytes() // 2


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _n_fill(tile_n, kg):
    """Fraction of the N tile columns that carry real output channels.

    Masked columns are zero-filled on the B load but still go through MFMA, so
    this is a direct multiplier on achievable throughput, not just on traffic.
    """
    return kg / (_ceil_div(kg, tile_n) * tile_n)


def _sweep(npq, kg, groups, num_cu, min_n_fill, min_waves, check_waste, check_grid):
    """One filtering pass. Returns ``[(sort_key, config), ...]``, unsorted."""
    scored = []
    lds_limit = max_lds_bytes()
    for tile_m, tile_n, wave_m, wave_n in itertools.product(
        TILE_M_VALUES, TILE_N_VALUES, WAVE_M_VALUES, WAVE_N_VALUES
    ):
        if not is_legal_tile(tile_m, tile_n, wave_m, wave_n):
            continue

        waves = wave_m * wave_n
        if waves < min_waves or waves > MAX_WAVES:
            continue

        n_acc = (tile_m // wave_m // MFMA_M) * (tile_n // wave_n // MFMA_N)
        if n_acc > MAX_N_ACC:
            continue

        if lds_bytes(tile_m, tile_n) > lds_limit:
            continue

        fill = _n_fill(tile_n, kg)
        if fill < min_n_fill and tile_n != kg:
            continue

        # An N tile wider than the whole of kg past the first tile is all mask.
        if (
            check_waste
            and tile_n > kg
            and _ceil_div(kg, tile_n) * tile_n - kg >= tile_n // 2
        ):
            continue

        blocks = _ceil_div(npq, tile_m) * groups * _ceil_div(kg, tile_n)
        # Below one wave per CU the launch cannot fill the device no matter how
        # good the tile is. Split-K is resolved separately and may rescue some
        # of these, so the floor is deliberately loose.
        if check_grid and blocks * waves < num_cu:
            continue

        for wgm in WGM_VALUES:
            # The grouped-M swizzle regroups the N grid; with a single N tile it
            # is a no-op that still costs index math.
            if wgm > 1 and _ceil_div(kg, tile_n) < 2:
                continue
            # Rank: exact N fit first, then larger tiles (more reuse per byte),
            # then fewer blocks. Only decides which survive ``max_configs``.
            scored.append(
                (
                    (-round(fill, 4), -(tile_m * tile_n), blocks, wgm),
                    (tile_m, tile_n, wave_m, wave_n, wgm),
                )
            )
    return scored


# Progressive relaxation until a candidate survives (narrow kg cannot meet
# wave floor and mask-waste at once).
_RELAXATIONS = (
    # (min_n_fill, min_waves, check_waste, check_grid)
    (0.5, MIN_WAVES, True, True),
    (0.5, MIN_WAVES, False, True),
    (0.34, 2, False, True),
    (0.0, 1, False, False),
)


def get_flydsl_conv3d_configs(
    npq,
    kg,
    groups,
    num_cu,
    max_configs=96,
):
    """Candidate ``(tile_m, tile_n, wave_m, wave_n, wgm)`` tuples; never empty."""
    scored = []
    for min_n_fill, min_waves, check_waste, check_grid in _RELAXATIONS:
        scored = _sweep(
            npq, kg, groups, num_cu, min_n_fill, min_waves, check_waste, check_grid
        )
        if scored:
            break

    scored.sort(key=lambda item: item[0])
    configs = [config for _, config in scored[:max_configs]]

    # Union BASELINE_TILES at every WGM_VALUES (heuristic picks tile and wgm independently).
    seen = set(configs)
    for tile in BASELINE_TILES:
        if not is_legal_tile(*tile):
            continue
        for wgm in WGM_VALUES:
            candidate = (*tile, wgm)
            if candidate not in seen:
                seen.add(candidate)
                configs.append(candidate)
    return configs
