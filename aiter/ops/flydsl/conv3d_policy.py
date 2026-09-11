# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware policy selection for the FlyDSL implicit-GEMM conv3d.

Counterpart to ``gemm_a16w16_policy.py``. Enumerates ``(TILE_M, TILE_N, WAVE_M,
WAVE_N, WGM)`` launch configurations, keeps the legal ones, and prunes to a set
small enough to tune.

Legality is decided by arithmetic, not by compiling. ``compile_conv3d_implicit``
asserts its constraints at trace time, so a candidate sweep that discovered them
by try/except would pay a full compile per rejected config. :func:`is_legal_tile`
is the closed form of those asserts.

The reason this exists rather than the fixed eight-entry table it replaces: the
kernel's own ``TILE_LADDER`` only offers N tiles of 128/64/32 plus a 256 wide
case, all powers of two. A VAE whose ``Cout`` ladder is 96/192/384 lands two of
its three rungs on 75% N occupancy, and the masked columns still issue MFMA. The
enumeration below includes 48/96/192 so an exact fit can be expressed at all.
"""

import itertools

__all__ = [
    "TILE_K",
    "get_flydsl_conv3d_configs",
    "is_legal_tile",
    "tile_kernel_name",
]

# Mirrors the module-level constants in kernels/conv3d_implicit.py. TILE_K is
# asserted equal to 32 by the kernel; the rest follow from the MFMA shape.
TILE_K = 32
LDG_VEC = 8
MFMA_M = 16
MFMA_N = 16
WARP_SIZE = 64
BF16_BYTES = 2
# PIPE_STAGES = 2 * TILES_PER_BARRIER, both fixed in the kernel.
PIPE_STAGES = 4

TILE_M_VALUES = (64, 96, 128, 192, 256, 384)
TILE_N_VALUES = (32, 48, 64, 96, 128, 192, 256)
WAVE_M_VALUES = (1, 2, 3, 4)
WAVE_N_VALUES = (1, 2, 3, 4, 6)
WGM_VALUES = (1, 4, 8)

# Kernel default, used as the guaranteed-present fallback.
DEFAULT_TILE = (128, 128, 2, 4)

# The kernel's own candidate table and heuristic ladder. These are unioned into
# every sweep so that the tuned pick can never come out worse than the shipped
# default -- whatever ``_pick_tile`` would have chosen is always measured too.
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

# Per-CU LDS on CDNA3/4. A candidate above half of this cannot keep two
# workgroups resident, which costs more latency hiding than a wider tile buys.
LDS_CAPACITY_BYTES = 160 * 1024
MAX_LDS_BYTES = LDS_CAPACITY_BYTES // 2

# acc VGPRs = 4 * MI_M * MI_N per lane. Past this the kernel spills and the
# config is slower than anything it could win on tile shape.
MAX_N_ACC = 32

# Two-wave workgroups exist in the legal space but have too little to overlap
# global latency with; they are only reachable through ``allow_narrow``.
MIN_WAVES = 4
MAX_WAVES = 16


def is_legal_tile(tile_m, tile_n, wave_m, wave_n):
    """Closed form of ``compile_conv3d_implicit``'s launch-config asserts.

    With ``TILE_K == 32`` and ``BLOCK_VECS == LDG_VEC * WAVE_M * WAVE_N *
    WARP_SIZE``, the two ``BLOCK_VECS`` divisibility asserts reduce to
    ``TILE_{M,N} % (16 * WAVE_M * WAVE_N) == 0``, which implies the separate
    ``TILE_M % (WAVE_M * 16)`` / ``TILE_N % (WAVE_N * 16)`` asserts and
    guarantees ``LDG_{A,B}_COUNT >= 1``.
    """
    waves = wave_m * wave_n
    if waves * WARP_SIZE > 1024:  # BLOCK_THREADS <= 1024
        return False
    step = MFMA_M * waves
    return tile_m % step == 0 and tile_n % step == 0


def tile_kernel_name(tile_m, tile_n, wave_m, wave_n, wgm):
    return f"conv3d_implicit_t{tile_m}x{tile_n}_w{wave_m}x{wave_n}_g{wgm}"


def lds_bytes(tile_m, tile_n):
    return PIPE_STAGES * (tile_m + tile_n) * TILE_K * BF16_BYTES


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

        if lds_bytes(tile_m, tile_n) > MAX_LDS_BYTES:
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


# Progressive relaxation. A narrow ``kg`` (``conv_out`` has 32) cannot satisfy
# the wave floor and the mask-waste rule at once: an exact 32-wide N tile forces
# WAVE_M*WAVE_N <= 2, while reaching four waves needs a 64-wide tile that is
# half mask. Rather than pick one rule to weaken globally, drop them in order
# until something survives.
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
    """Return the candidate launch configs worth tuning for one problem.

    Args:
        npq: GEMM M extent, ``n * do * ho * wo``.
        kg: GEMM N extent, ``k // groups``.
        groups: Convolution groups; one tile never spans two groups.
        num_cu: Compute units on the target device.
        max_configs: Cap on the enumerated part, to bound tuning time. The
            baseline tiles are unioned in afterwards and are not subject to it.

    Returns:
        List of ``(tile_m, tile_n, wave_m, wave_n, wgm)`` tuples, never empty.
    """
    scored = []
    for min_n_fill, min_waves, check_waste, check_grid in _RELAXATIONS:
        scored = _sweep(
            npq, kg, groups, num_cu, min_n_fill, min_waves, check_waste, check_grid
        )
        if scored:
            break

    scored.sort(key=lambda item: item[0])
    configs = [config for _, config in scored[:max_configs]]

    # Union in whatever the shipped heuristic could pick, so the sweep always
    # measures the incumbent and "tuned is never worse than default" is
    # checkable from the same run.
    seen = set(configs)
    for tile in BASELINE_TILES:
        if not is_legal_tile(*tile):
            continue
        candidate = (*tile, 1)
        if candidate not in seen:
            seen.add(candidate)
            configs.append(candidate)
    return configs
