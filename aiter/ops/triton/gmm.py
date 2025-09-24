# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# Triton
import triton

# AITER: general utility functions
from aiter.ops.triton.utils.device_info import get_num_sms

# AITER: GMM utility functions
from aiter.ops.triton.utils.gmm_common import (
    DTYPE,
    is_power_of_2,
    check_input_device_dtype,
    get_gmm_shape,
    get_gmm_output,
    get_gmm_transposition,
    get_tgmm_shape,
    get_tgmm_output,
    get_tgmm_transposition,
)

# AITER: GMM Triton kernels
from aiter.ops.triton._triton_kernels.gmm import (
    gmm_kernel,
    tgmm_persistent_kernel,
    tgmm_non_persistent_kernel,
)


# GMM PyTorch wrapper.
# ------------------------------------------------------------------------------


def _gmm_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: Tensor,
    grid_dim: int,
) -> tuple[int]:
    assert N > 0, f"N must be positive, it's {N}."
    assert is_power_of_2(
        block_size_m
    ), f"M-dimension tile size must be a power of 2 (it's {block_size_m})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert torch.all(group_sizes >= 0).item(), "All group_sizes must be non-negative."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_m_tiles = (group_sizes + block_size_m - 1) // block_size_m
    assert torch.all(num_m_tiles >= 0).item(), "All num_m_tiles must be non-negative."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = torch.sum(num_m_tiles * num_n_tiles).item()
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = int(min(grid_dim, num_tiles))
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_gmm_shape(lhs, rhs, group_sizes)

    out = get_gmm_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    trans_rhs, _ = get_gmm_transposition(lhs, rhs, out)

    # TODO: Read best config from JSON file.
    config = {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "GRID_DIM": get_num_sms(),
        "num_warps": 8,
        "num_stages": 1,
    }

    grid = _gmm_grid(
        N,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        group_sizes,
        config["GRID_DIM"],
    )

    # fmt: off
    gmm_kernel[grid](
        # Tensor pointers:
        lhs, rhs, group_sizes, out,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_RHS=trans_rhs,
        **config,
    )
    # fmt: on

    return out


# Persistent TGMM PyTorch wrapper.
# ------------------------------------------------------------------------------


def _ptgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
    grid_dim: int,
) -> tuple[int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    assert grid_dim > 0, f"Grid dimension must be positive (it's {grid_dim})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles = G * num_k_tiles * num_n_tiles
    assert num_tiles > 0, f"num_tiles must be positive, it's {num_tiles}."
    num_programs = min(grid_dim, num_tiles)
    assert num_programs > 0, f"num_programs must be positive, it's {num_programs}."
    return (num_programs,)


def ptgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_tgmm_shape(lhs, rhs, group_sizes)

    out = get_tgmm_output(
        K,
        N,
        G,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    trans_lhs, _ = get_tgmm_transposition(lhs, rhs, out)

    # TODO: Read best config from JSON file.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_K": 256,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "GRID_DIM": get_num_sms(),
        "num_warps": 8,
        "num_stages": 1,
    }

    grid = _ptgmm_grid(
        K,
        N,
        G,
        config["BLOCK_SIZE_K"],
        config["BLOCK_SIZE_N"],
        config["GRID_DIM"],
    )

    # fmt: off
    tgmm_persistent_kernel[grid](
        # Tensor pointers:
        lhs, rhs, group_sizes, out,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_LHS=trans_lhs,
        **config,
    )
    # fmt: on

    return out


# Regular non-persistent TGMM PyTorch wrapper.
# ------------------------------------------------------------------------------


def _nptgmm_grid(
    K: int,
    N: int,
    G: int,
    block_size_k: int,
    block_size_n: int,
) -> tuple[int, int]:
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}."
    assert G > 0, f"G must be positive, it's {G}."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size must be a power of 2 (it's {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size must be a power of 2 (it's {block_size_n})."
    num_k_tiles = triton.cdiv(K, block_size_k)
    assert num_k_tiles > 0, f"num_k_tiles must be positive, it's {num_k_tiles}."
    num_n_tiles = triton.cdiv(N, block_size_n)
    assert num_n_tiles > 0, f"num_n_tiles must be positive, it's {num_n_tiles}."
    num_tiles_per_mm = num_k_tiles * num_n_tiles
    assert (
        num_tiles_per_mm > 0
    ), f"num_tiles_per_mm must be positive, it's {num_tiles_per_mm}."
    return (G, num_tiles_per_mm)


def nptgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, K, N, G = get_tgmm_shape(lhs, rhs, group_sizes)

    out = get_tgmm_output(
        K,
        N,
        G,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    trans_lhs, _ = get_tgmm_transposition(lhs, rhs, out)

    # TODO: Read best config from JSON file.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_K": 256,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "num_warps": 8,
        "num_stages": 1,
    }

    grid = _nptgmm_grid(
        K,
        N,
        G,
        config["BLOCK_SIZE_K"],
        config["BLOCK_SIZE_N"],
    )

    # fmt: off
    tgmm_non_persistent_kernel[grid](
        # Tensor pointers:
        lhs, rhs, group_sizes, out,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_LHS=trans_lhs,
        **config,
    )
    # fmt: on

    return out
