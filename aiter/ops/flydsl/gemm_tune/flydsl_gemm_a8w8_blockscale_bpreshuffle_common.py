# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate catalog and tuning filters for gfx942 FP8 blockscale preshuffle GEMM.

kernelName encodes the tile and compile options, including split-K; the CSV
splitK column stays zero. Output dtype comes from the caller.
This module remains importable without FlyDSL.
"""

import re
from dataclasses import dataclass
from itertools import product

# Shared with the kernel; importing these constants does not require FlyDSL.
MFMA_MN = 16  # v_mfma_f32_16x16x32_fp8_fp8 output tile edge
WAVE = 64
SCALE_BLOCK = 128

# The kernel forms A and B byte offsets and output element offsets in signed 32 bits,
# and its buffer descriptors count bytes in 32 bits.
OFFSET_LIMIT = 2**31

# Fixed per-stream buffers retain their addresses for captured graphs.
SPLIT_K_WORKSPACE_ELEMS = 4 * 1024 * 1024  # 16 MiB of fp32 partials
SPLIT_K_SEMAPHORE_SLOTS = 4096  # 16 KiB of Int32 arrival counters

_PREFIX = "flydsl_blockscale_bpreshuffle"

_NAME_RE = re.compile(
    rf"^{_PREFIX}_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)"
    r"_w(?P<num_waves>\d+)"
    r"_ac(?P<use_async_copy>[01])"
    r"_cs(?P<use_cshuffle_epilog>[01])"
    # Missing _sk preserves older unsplit names.
    r"(?:_sk(?P<split_k>\d+))?$"
)


@dataclass(frozen=True)
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    num_waves: int = 4
    use_async_copy: bool = False
    use_cshuffle_epilog: bool = False
    split_k: int = 1

    @property
    def name(self) -> str:
        split_suffix = f"_sk{self.split_k}" if self.split_k > 1 else ""
        return (
            f"{_PREFIX}_{self.tile_m}x{self.tile_n}x{self.tile_k}"
            f"_w{self.num_waves}_ac{int(self.use_async_copy)}"
            f"_cs{int(self.use_cshuffle_epilog)}{split_suffix}"
        )


def parse_kernel_name(kernel_name: str) -> kernelInstance | None:
    """Parse a catalog name, returning None when its format is unrecognized."""
    match = _NAME_RE.match(kernel_name or "")
    if match is None:
        return None
    return kernelInstance(
        tile_m=int(match.group("tile_m")),
        tile_n=int(match.group("tile_n")),
        tile_k=int(match.group("tile_k")),
        num_waves=int(match.group("num_waves")),
        use_async_copy=match.group("use_async_copy") == "1",
        use_cshuffle_epilog=match.group("use_cshuffle_epilog") == "1",
        split_k=int(match.group("split_k") or 1),
    )


def tile_validation_error(
    *,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_waves: int,
    use_cshuffle_epilog: bool,
    split_k: int,
) -> str | None:
    """Return the first tile/shape constraint violation, or None."""
    if tile_k % SCALE_BLOCK != 0:
        return f"tile_k ({tile_k}) must be divisible by the scale block ({SCALE_BLOCK})"
    if k % tile_k != 0:
        return f"K ({k}) must be divisible by tile_k ({tile_k})"
    if split_k < 1:
        return f"split_k must be at least 1, got {split_k}"
    # Every split runs the same compile-time tile count.
    if (k // tile_k) % split_k != 0:
        return f"split_k ({split_k}) must divide the K tile count ({k // tile_k})"
    # The CShuffle epilogue stages a 2-byte image that the fp32 partials do not fit.
    if split_k > 1 and use_cshuffle_epilog:
        return "split_k > 1 supports the direct epilogue only"
    # The N grid uses floor division, so partial tiles would leave columns unwritten.
    if n % tile_n != 0:
        return f"N ({n}) must be divisible by tile_n ({tile_n})"
    if tile_n % (num_waves * MFMA_MN) != 0:
        return (
            f"tile_n ({tile_n}) must split into whole {MFMA_MN}-wide MFMA tiles across "
            f"{num_waves} waves"
        )
    # The M loops step in whole MFMA tiles; a remainder would never be computed.
    if tile_m % MFMA_MN != 0:
        return f"tile_m ({tile_m}) must be a multiple of {MFMA_MN}"
    # XOR swizzling requires a power-of-two count of 16-byte chunks per row.
    lds_chunks_per_row = tile_k // 16
    if lds_chunks_per_row & (lds_chunks_per_row - 1):
        return f"tile_k ({tile_k}) must be 16 times a power of two for the LDS swizzle"
    total_threads = num_waves * WAVE
    bytes_a_per_tile = tile_m * tile_k
    if bytes_a_per_tile % total_threads != 0:
        return (
            f"tile_m*tile_k must be divisible by {total_threads}: "
            f"tile_m={tile_m}, tile_k={tile_k}"
        )
    # The A loader supports only 8-byte or 16-byte chunks per thread.
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    if bytes_per_thread_a % 8 != 0:
        return f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by 8"
    return None


def max_lds_bytes_for_tune() -> int:
    """Addressable LDS limit for the current target."""
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes

    return get_lds_capacity_bytes()


def estimated_lds_bytes(config: kernelInstance) -> int:
    """Return planned LDS bytes, or zero if the kernel cannot be imported."""
    try:
        from aiter.ops.flydsl.kernels.gemm_blockscale_preshuffle import plan_lds
    except Exception:  # noqa: BLE001  # pragma: no cover - needs a FlyDSL install
        return 0
    return plan_lds(
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        use_cshuffle_epilog=config.use_cshuffle_epilog,
        use_async_copy=config.use_async_copy,
        split_k=config.split_k,
    )


def tile_is_valid(config: kernelInstance, n: int, k: int) -> bool:
    """Check tile constraints and LDS capacity when the kernel is importable."""
    validation_error = tile_validation_error(
        n=n,
        k=k,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        num_waves=config.num_waves,
        use_cshuffle_epilog=config.use_cshuffle_epilog,
        split_k=config.split_k,
    )
    if validation_error is not None:
        return False
    lds_bytes = estimated_lds_bytes(config)
    return not lds_bytes or lds_bytes <= max_lds_bytes_for_tune()


def offsets_fit(m: int, n: int, k: int, tile_m: int) -> bool:
    """Whether M * K, N * K and M * N, with M rounded up to tile_m, stay below 2**31."""
    padded_m = -(-m // tile_m) * tile_m
    return max(padded_m * k, n * k, padded_m * n) < OFFSET_LIMIT


def split_k_buffers_fit(m: int, n: int, tile_m: int, tile_n: int, split_k: int) -> bool:
    """Whether a split kernel's partials and counters fit the wrapper's buffers."""
    if split_k <= 1:
        return True
    output_tiles = -(-m // tile_m) * -(-n // tile_n)
    return (
        split_k * m * n <= SPLIT_K_WORKSPACE_ELEMS
        and output_tiles <= SPLIT_K_SEMAPHORE_SLOTS
    )


# Performance heuristics apply only to the reduced tuning search.
_DECODE_M_MAX = 256
_LARGE_M_TILE_ALLOWLIST = {  # M >= 8192
    (32, 128, 128),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
    (128, 128, 128),
    (128, 256, 128),
}
_MID_M_TILE_DENYLIST = {  # 1024 <= M < 8192
    (64, 64, 128),
    (64, 64, 256),
    (256, 128, 128),
}


def passes_tuning_heuristics(
    config: kernelInstance, m: int, n: int, cu_num: int
) -> bool:
    """Apply the reduced-search heuristics to an already-valid configuration."""
    tile_shape = (config.tile_m, config.tile_n, config.tile_k)
    unsplit_workgroups = -(-m // config.tile_m) * (n // config.tile_n)
    total_workgroups = unsplit_workgroups * config.split_k
    min_workgroups = max(4, min(16, n // 64))
    if total_workgroups < min_workgroups:
        return False
    if m <= _DECODE_M_MAX:
        if config.tile_m >= 128:
            return False
    elif config.use_async_copy:
        return False
    if m >= 8192 and tile_shape not in _LARGE_M_TILE_ALLOWLIST:
        return False
    if 1024 <= m < 8192 and tile_shape in _MID_M_TILE_DENYLIST:
        return False
    if config.split_k > 1:
        return (
            config.tile_m <= 64
            and unsplit_workgroups < cu_num
            and total_workgroups <= 4 * cu_num
        )
    return True


# Candidate order determines kernel IDs recorded in tuning results.
TILE_CANDIDATES = (
    (16, 64, 256),
    (16, 128, 256),
    (32, 64, 128),
    (32, 64, 256),
    (32, 128, 128),
    (32, 128, 256),
    (64, 64, 128),
    (64, 64, 256),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
    (128, 128, 128),
    (128, 256, 128),
    (256, 128, 128),
)

WAVE_CANDIDATES = (4, 8)
# (use_async_copy, use_cshuffle_epilog); (False, False) must stay first.
FLAG_CANDIDATES = ((False, False), (True, False), (False, True), (True, True))

SPLIT_K_CANDIDATES = (2, 4, 5, 8, 10)


def _build_kernels_list() -> dict[int, kernelInstance]:
    # Preserve flag/wave/tile order; filtering here would renumber existing IDs.
    configs = []
    for flags, num_waves, tile in product(
        FLAG_CANDIDATES, WAVE_CANDIDATES, TILE_CANDIDATES
    ):
        use_async_copy, use_cshuffle_epilog = flags
        tile_m, tile_n, tile_k = tile
        configs.append(
            kernelInstance(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                num_waves=num_waves,
                use_async_copy=use_async_copy,
                use_cshuffle_epilog=use_cshuffle_epilog,
            )
        )

    # Split-K follows all unsplit IDs and fixes async copy and CShuffle to False.
    for split_k, num_waves, tile in product(
        SPLIT_K_CANDIDATES, WAVE_CANDIDATES, TILE_CANDIDATES
    ):
        tile_m, tile_n, tile_k = tile
        configs.append(
            kernelInstance(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                num_waves=num_waves,
                split_k=split_k,
            )
        )
    return dict(enumerate(configs))


kernels_list = _build_kernels_list()


def candidate_kernel_ids(
    m: int, n: int, k: int, cu_num: int, full_space: bool = False
) -> list[int]:
    """Return IDs passing resource checks; full_space skips performance heuristics."""
    kernel_ids = []
    for kernel_id, config in kernels_list.items():
        if not offsets_fit(m, n, k, config.tile_m):
            continue
        if not tile_is_valid(config, n, k):
            continue
        if not split_k_buffers_fit(m, n, config.tile_m, config.tile_n, config.split_k):
            continue
        if full_space or passes_tuning_heuristics(config, m, n, cu_num):
            kernel_ids.append(kernel_id)
    return kernel_ids
