# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
import functools
import json
import os.path

# Triton
import triton
import triton.language as tl

# AITER
from ..utils.core import AITER_TRITON_CONFIGS_PATH
from ..utils._triton import arch_info
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd


# Kernel config.
# ------------------------------------------------------------------------------


@functools.lru_cache()
def get_config(
    gmm_type: str, M: int, K: int, N: int, G: int, accumulate: bool = False
) -> dict[str, int]:
    assert gmm_type in {
        "gmm",
        "ptgmm",
        "nptgmm",
    }, f"'{gmm_type}' is an invalid GMM variant."
    if not hasattr(get_config, "_config_dict"):
        config_filename = os.path.join(
            AITER_TRITON_CONFIGS_PATH, f"{arch_info.get_arch()}-GMM.json"
        )
        assert os.path.exists(config_filename) and os.path.isfile(
            config_filename
        ), f"'{config_filename}' isn't an existent file."
        with open(config_filename, "r") as config_file:
            get_config._config_dict = json.load(config_file)
            assert all(
                gmm_type in get_config._config_dict
                for gmm_type in {"gmm", "ptgmm", "nptgmm"}
            ), "Not all GMM variants are present in the configuration file."
    # TODO: Fine tune GMM kernels and use (M, K, N, G) shape to query the best
    #       config in the dictionary.
    assert (
        "default" in get_config._config_dict[gmm_type]
    ), "Default configuration is absent."
    key = "accumulate" if accumulate else "default"
    return get_config._config_dict[gmm_type][key]


# Common code shared by GMM and TGMM kernels.
# ------------------------------------------------------------------------------


# XCD remapping followed by 1D PID to 2D grid mapping.
@triton.jit
def _remap_xcd_tile_grid(
    tile_in_mm,
    num_row_tiles,
    num_col_tiles,
    GROUP_SIZE: tl.constexpr = 1,
    NUM_XCDS: tl.constexpr = 8,
):
    return pid_grid(
        remap_xcd(tile_in_mm, num_row_tiles * num_col_tiles, NUM_XCDS=NUM_XCDS),
        num_row_tiles,
        num_col_tiles,
        GROUP_SIZE_M=GROUP_SIZE,
    )


# GMM kernel.
# ------------------------------------------------------------------------------


@triton.heuristics(
    {
        "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda META: META["K"] % META["BLOCK_SIZE_K"]
        == 0,
    }
)
@triton.jit
def gmm_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    bias_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_RHS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input row of lhs and output row of out. Each group reads some rows of
    # lhs and writes some rows to out.
    last_m = 0

    # Loop through all (m, K, N) MM problems:
    #   (m, K) x (K, N) = (m, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        num_m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        # num_m_tiles can be zero if group is empty
        tl.device_assert(num_m_tiles >= 0, "num_m_tiles < 0")

        num_tiles = num_m_tiles * num_n_tiles
        # num_tiles can be zero if group is empty
        tl.device_assert(num_tiles >= 0, "num_tiles < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            tile_m, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_m_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
            )

            # Do regular MM:

            tl.device_assert(tile_m * BLOCK_SIZE_M >= 0, "tile_m * BLOCK_SIZE_M < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_m = (
                tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            ) % m
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

            lhs_ptrs = lhs_ptr + (last_m + offs_lhs_m[:, None]) * K + offs_k[None, :]

            if TRANS_RHS:
                rhs_ptrs = (
                    rhs_ptr
                    + g.to(tl.int64) * K * N
                    + offs_k[:, None]
                    + offs_rhs_n[None, :] * K
                )
            else:
                rhs_ptrs = (
                    rhs_ptr
                    + g.to(tl.int64) * K * N
                    + offs_k[:, None] * N
                    + offs_rhs_n[None, :]
                )

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if K_DIVISIBLE_BY_BLOCK_SIZE_K:
                    lhs = tl.load(lhs_ptrs)
                    rhs = tl.load(rhs_ptrs)
                else:
                    k_mask_limit = K - k * BLOCK_SIZE_K
                    lhs = tl.load(
                        lhs_ptrs, mask=offs_k[None, :] < k_mask_limit, other=0
                    )
                    rhs = tl.load(
                        rhs_ptrs, mask=offs_k[:, None] < k_mask_limit, other=0
                    )

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                lhs_ptrs += BLOCK_SIZE_K

                if TRANS_RHS:
                    rhs_ptrs += BLOCK_SIZE_K
                else:
                    rhs_ptrs += BLOCK_SIZE_K * N

            # Add bias if enabled
            if USE_BIAS:
                offs_bias_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                    0, BLOCK_SIZE_N
                )
                bias_ptrs = bias_ptr + g.to(tl.int64) * N + offs_bias_n
                bias = tl.load(bias_ptrs, mask=offs_bias_n < N, other=0.0)
                # Convert bias to float32 to match accumulator precision
                bias = bias.to(tl.float32)
                # Broadcast bias across M dimension and add in float32
                acc += bias[None, :]

            # Convert to output dtype after all computations
            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            out_ptrs = (
                out_ptr + (last_m + offs_out_m[:, None]) * N + offs_out_n[None, :]
            )

            tl.store(
                out_ptrs,
                acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            # Go to the next tile by advancing number of programs.
            tile += GRID_DIM
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.

        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")

        last_m += m
        # last_m can be zero if group 0 is skipped
        tl.device_assert(last_m >= 0, "last_m < 0 (at update)")
        tl.device_assert(last_m <= M, "last_m > M (at update)")


# Persistent TGMM kernel.
# ------------------------------------------------------------------------------


@triton.jit
def tgmm_persistent_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    bias_grad_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_LHS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
    COMPUTE_BIAS_GRAD: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.device_assert(num_k_tiles > 0, "num_k_tiles <= 0")

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    num_tiles = num_k_tiles * num_n_tiles
    tl.device_assert(num_tiles > 0, "num_tiles <= 0")

    # Current tile. Each program computes multiple tiles of each group.
    tile = tl.program_id(0)
    tl.device_assert(tile >= 0, "tile < 0 (at initialization)")

    # Tile limit of last MM problem (inclusive).
    last_mm_tile = 0

    # Last input column of lhs and input row of rhs. Each group reads some
    # columns of lhs and some rows of rhs.
    last_m = 0

    # Loop through all (K, m, N) MM problems:
    #   (K, m) x (m, N) = (K, N)
    #   sum(m) = M
    for g in range(G):
        # Get m dimension of current MM problem.
        m = tl.load(group_sizes_ptr + g)
        # m can be zero if group is empty
        tl.device_assert(m >= 0, "m < 0")

        # Loop through tiles of current MM problem.
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            # Figure out tile coordinates in current MM problem.
            tile_in_mm = tile - last_mm_tile
            tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

            tile_k, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_k_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
            )

            # Do regular MM:

            tl.device_assert(tile_k * BLOCK_SIZE_K >= 0, "tile_k * BLOCK_SIZE_K < 0")
            tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

            offs_lhs_k = (
                tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            ) % K
            offs_rhs_n = (
                tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            ) % N
            offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

            if TRANS_LHS:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] + (last_m + offs_m[None, :]) * K
                )
            else:
                lhs_ptrs = (
                    lhs_ptr + offs_lhs_k[:, None] * M + (last_m + offs_m[None, :])
                )

            rhs_ptrs = rhs_ptr + (last_m + offs_m[:, None]) * N + offs_rhs_n[None, :]

            loop_m = tl.cdiv(m, BLOCK_SIZE_M)
            m_divisible_by_block_m = m % BLOCK_SIZE_M == 0
            if not m_divisible_by_block_m:
                loop_m -= 1

            acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

            # Initialize bias accumulator
            bias_acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

            for _ in range(0, loop_m):
                lhs = tl.load(lhs_ptrs)
                rhs = tl.load(rhs_ptrs)

                acc += tl.dot(lhs, rhs, input_precision="ieee")

                # Accumulate for bias gradient: sum lhs across M dimension
                if COMPUTE_BIAS_GRAD and tile_n == 0:
                    bias_acc += tl.sum(
                        lhs, axis=1
                    )  # Sum across M dimension [K, M] -> [K]

                if TRANS_LHS:
                    lhs_ptrs += BLOCK_SIZE_M * K
                else:
                    lhs_ptrs += BLOCK_SIZE_M

                rhs_ptrs += BLOCK_SIZE_M * N

            if not m_divisible_by_block_m:
                offs_lhs_k = (
                    tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                ) % K
                offs_rhs_n = (
                    tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                ) % N
                offs_m = loop_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                lhs = tl.load(lhs_ptrs, mask=offs_m[None, :] < m, other=0)
                rhs = tl.load(rhs_ptrs, mask=offs_m[:, None] < m, other=0)
                acc += tl.dot(lhs, rhs, input_precision="ieee")

                # Accumulate last chunk for bias gradient
                if COMPUTE_BIAS_GRAD and tile_n == 0:
                    bias_acc += tl.sum(lhs, axis=1)

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_k = tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            out_ptrs = (
                out_ptr
                + g.to(tl.int64) * K * N
                + offs_out_k[:, None] * N
                + offs_out_n[None, :]
            )

            mask = (offs_out_k[:, None] < K) & (offs_out_n[None, :] < N)
            if ACCUMULATE:
                # Load existing values and add to them (like beta=1 in BLAS)
                old_vals = tl.load(out_ptrs, mask=mask, other=0.0)
                tl.store(out_ptrs, acc + old_vals, mask=mask)
            else:
                # Overwrite output (like beta=0 in BLAS)
                tl.store(out_ptrs, acc, mask=mask)

            # Store bias gradient (only for first N tile, sum across all M)
            if COMPUTE_BIAS_GRAD and tile_n == 0:
                # Keep as float32 for atomic_add (bf16 not supported for atomics)
                bias_grad_ptrs = bias_grad_ptr + g.to(tl.int64) * K + offs_out_k
                # Use atomic add since multiple K-tiles may write to same expert's bias
                tl.atomic_add(
                    bias_grad_ptrs, bias_acc, mask=offs_out_k < K, sem="relaxed"
                )

            # Go to the next tile by advancing number of programs.
            tile += GRID_DIM
            tl.device_assert(tile > 0, "tile <= 0 (at update)")

        # Get ready to go to the next MM problem.

        last_mm_tile += num_tiles
        # last_mm_tile can be zero if group 0 is skipped
        tl.device_assert(last_mm_tile >= 0, "last_mm_tile < 0 (at update)")

        last_m += m
        # last_m can be zero if group 0 is skipped
        tl.device_assert(last_m >= 0, "last_m < 0 (at update)")
        tl.device_assert(last_m <= M, "last_m > M (at update)")


# Regular non-persistent TGMM kernel.
# ------------------------------------------------------------------------------


@triton.heuristics({"BLOCK_SIZE_G": lambda META: triton.next_power_of_2(META["G"])})
@triton.jit
def tgmm_non_persistent_kernel(
    # Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    bias_grad_ptr,
    # Tensor shapes:
    M: int,
    K: int,
    N: int,
    G: int,
    # Meta-parameters:
    TRANS_LHS: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    COMPUTE_BIAS_GRAD: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    # Get group ID from grid.
    g = tl.program_id(0)
    tl.device_assert(g >= 0, "g < 0")
    tl.device_assert(g < G, "g >= G")

    # Get m dimension of current MM group.
    m = tl.load(group_sizes_ptr + g)
    # m can be zero if group is empty.
    tl.device_assert(m >= 0, "m < 0")

    # Skip empty groups.
    if m == 0:
        return

    # Compute sum(group_sizes) until current group g.
    # It's the starting column of lhs and starting row of rhs.
    offs_g = tl.arange(0, BLOCK_SIZE_G)
    group_sizes = tl.load(group_sizes_ptr + offs_g, mask=offs_g < g, other=0)
    start_m = tl.sum(group_sizes)

    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tl.device_assert(num_k_tiles > 0, "num_k_tiles <= 0")

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")

    # Get MM tile from grid.
    tile_in_mm = tl.program_id(1)
    tl.device_assert(tile_in_mm >= 0, "tile_in_mm < 0")

    tile_k, tile_n = _remap_xcd_tile_grid(
        tile_in_mm, num_k_tiles, num_n_tiles, GROUP_SIZE=GROUP_SIZE
    )

    tl.device_assert(tile_k * BLOCK_SIZE_K >= 0, "tile_k * BLOCK_SIZE_K < 0")
    tl.device_assert(tile_n * BLOCK_SIZE_N >= 0, "tile_n * BLOCK_SIZE_N < 0")

    offs_lhs_k = (tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_rhs_n = (tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    if TRANS_LHS:
        lhs_ptrs = lhs_ptr + offs_lhs_k[:, None] + (start_m + offs_m[None, :]) * K
    else:
        lhs_ptrs = lhs_ptr + offs_lhs_k[:, None] * M + (start_m + offs_m[None, :])

    rhs_ptrs = rhs_ptr + (start_m + offs_m[:, None]) * N + offs_rhs_n[None, :]

    loop_m = tl.cdiv(m, BLOCK_SIZE_M)
    m_divisible_by_block_m = m % BLOCK_SIZE_M == 0
    if not m_divisible_by_block_m:
        loop_m -= 1

    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    # Initialize bias accumulator
    bias_acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

    for _ in range(0, loop_m):
        lhs = tl.load(lhs_ptrs)
        rhs = tl.load(rhs_ptrs)

        acc += tl.dot(lhs, rhs, input_precision="ieee")

        # Accumulate for bias gradient: sum lhs across M dimension
        if COMPUTE_BIAS_GRAD and tile_n == 0:
            bias_acc += tl.sum(lhs, axis=1)  # [K, M] -> [K]

        if TRANS_LHS:
            lhs_ptrs += BLOCK_SIZE_M * K
        else:
            lhs_ptrs += BLOCK_SIZE_M

        rhs_ptrs += BLOCK_SIZE_M * N

    if not m_divisible_by_block_m:
        offs_lhs_k = (
            tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        ) % K
        offs_rhs_n = (
            tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        ) % N
        offs_m = loop_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        lhs = tl.load(lhs_ptrs, mask=offs_m[None, :] < m, other=0)
        rhs = tl.load(rhs_ptrs, mask=offs_m[:, None] < m, other=0)
        acc += tl.dot(lhs, rhs, input_precision="ieee")
        # Accumulate last chunk for bias gradient
        if COMPUTE_BIAS_GRAD and tile_n == 0:
            bias_acc += tl.sum(lhs, axis=1)

    acc = acc.to(out_ptr.type.element_ty)

    offs_out_k = tile_k.to(tl.int64) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_out_n = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_ptrs = (
        out_ptr + g.to(tl.int64) * K * N + offs_out_k[:, None] * N + offs_out_n[None, :]
    )

    mask = (offs_out_k[:, None] < K) & (offs_out_n[None, :] < N)
    if ACCUMULATE:
        # Load existing values and add to them (like beta=1 in BLAS)
        old_vals = tl.load(out_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, acc + old_vals, mask=mask)
    else:
        # Overwrite output (like beta=0 in BLAS)
        tl.store(out_ptrs, acc, mask=mask)

    # Store bias gradient (only for first N tile, sum across all M)
    if COMPUTE_BIAS_GRAD and tile_n == 0:
        # Keep as float32 for atomic_add (bf16/fp16 not supported for atomics)
        bias_grad_ptrs = bias_grad_ptr + g.to(tl.int64) * K + offs_out_k
        # Use atomic add since multiple K-tiles may write to same expert's bias
        tl.atomic_add(bias_grad_ptrs, bias_acc, mask=offs_out_k < K, sem="relaxed")
