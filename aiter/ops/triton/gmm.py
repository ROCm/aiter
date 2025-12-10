# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# Triton
import triton

# AITER: GMM utility functions
from aiter.ops.triton.utils.gmm_common import (
    DTYPE,
    is_power_of_2,
    check_input_device_dtype,
    check_bias_shape_stride,
    get_gmm_shape,
    get_gmm_output,
    get_gmm_transposition,
    get_tgmm_shape,
    get_tgmm_output,
    get_tgmm_bias_grad,
    get_tgmm_transposition,
)

# AITER: GMM Triton kernels
from aiter.ops.triton._triton_kernels.gmm import (
    gmm_kernel,
    tgmm_persistent_kernel,
    tgmm_non_persistent_kernel,
    get_config,
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
    config: dict[str, int] | None = None,
    bias: Tensor | None = None,
) -> Tensor:
    """
    Perform Group Matrix Multiplication (GMM): out = lhs @ rhs + bias

    lhs rows are divided into G groups. Each group of lhs rows is matrix multiplied with a plane of
    rhs 3D tensor and then stored in a slice of out. In PyTorch parlance, it can be implemented as
    follows for a given group g:
        out[group_start:group_end, :] = lhs[group_start:group_end, :] @ rhs[g] + bias[g]

    The size of each group, and their respective start and end positions are specified by
    group_sizes tensor. For instance, suppose that group_sizes = [3, 2, 4, 1]. In this particular
    case we have 4 groups. The 1st group starts at 0 and ends at 2, the second group starts at 3 and
    ends at 4, the third group starts at 5 and ends at 8, and the fourth and final group consists of
    just the 10th (last) row of lhs.

    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side 2D input tensor. Shape: (M, K).
        lhs data type must be torch.float16 or torch.bfloat16, and must match rhs data type.
        lhs must be on the same device of rhs and group_sizes.
    rhs : torch.Tensor
        Right-hand side 3D input tensor. Shape: (G, K, N).
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 and all its elements must be non-negative.
        group_sizes must be on the same device of lhs and rhs.
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (M, N), its data type must match preferred_element_type
        and it must be on the same device of other input tensors.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.
    bias : torch.Tensor or None, optional
        Optional bias tensor. Shape: (G, N).
        If provided, bias data type must match lhs and rhs data type, and bias must be on the same
        device as other input tensors. Each group g adds bias[g] to the output.

    Returns
    -------
    torch.Tensor
        The computed output 2D tensor. Shape: (M, N).
        Output tensor data type is given by preferred_element_type.
        If existing_out is provided then existing_out is also returned.

    Implementation Notes
    --------------------
    - GMM is implemented with a persistent Triton kernel.
    - lhs must be row-major (lhs.stride() == (K, 1)).
    - rhs can be row-major (rhs.stride() == (K * N, N, 1)) or column-major (rhs.stride() ==
      (K * N, 1, K)). If rhs is row-major then kernel parameter TRANS_RHS == False, this is useful
      for implementing forward pass. If rhs is column-major then kernel parameter TRANS_RHS == True,
      this is useful for computing the lhs derivative in the backward pass, while fusing the
      transposition.
    - out must be row-major (out.stride() == (N, 1)).
    - bias must be row-major (bias.stride() == (N, 1)) if provided.
    """
    use_bias = bias is not None
    check_input_device_dtype(lhs, rhs, group_sizes, bias)

    M, K, N, G = get_gmm_shape(lhs, rhs, group_sizes)

    if use_bias:
        check_bias_shape_stride(bias, G, N)

    out = get_gmm_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    trans_rhs, _ = get_gmm_transposition(lhs, rhs, out)

    if config is None:
        config = get_config("gmm", M, K, N, G)

    assert all(
        key in config
        and isinstance(config[key], int)
        and (
            is_power_of_2(config[key])
            if key.startswith("BLOCK_SIZE_")
            else config[key] > 0
        )
        for key in {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_K",
            "BLOCK_SIZE_N",
            "GROUP_SIZE",
            "GRID_DIM",
        }
    ), "Invalid GMM kernel config."

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
        lhs, rhs, group_sizes, out, bias,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_RHS=trans_rhs,
        USE_BIAS=use_bias,
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
    config: dict[str, int] | None = None,
    bias_grad: Tensor | None = None,
    accumulate: bool = False,
) -> Tensor:
    """
    Perform a Group Matrix Multiplication (GMM) variant: out = lhs @ rhs

    lhs columns and rhs rows are divided into G groups. Each group of lhs is matrix multiplied with
    the respective group of rhs and then stored in a plane of the output 3D tensor. In PyTorch
    parlance, it can be implemented as follows for a given group g:
        out[g] = lhs[:, group_start:group_end] @ rhs[group_start:group_end, :]

    The 't' in the operator name derives from MaxText implementation
    (https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/kernels/megablox/gmm.py),
    which served as the initial inspiration for this one. TGMM differs from GMM in terms of tensor
    shapes. GMM does (M, K) @ (G, K, N) = (M, N) while TGMM does (K, M) @ (M, N) = (G, K, N).

    The 'p' in the operator name means that it is implemented with a persistent kernel. There is
    also the non-persistent variation, which is implemented with a regular kernel. Please take a
    look at nptgmm operator. Both ptgmm and nptgmm implement the same computation, choosing one or
    the other is a matter of performance for the target workload.

    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side 2D input tensor. Shape: (K, M).
        lhs data type must be torch.float16 or torch.bfloat16, and must match rhs data type.
        lhs must be on the same device of rhs and group_sizes.
    rhs : torch.Tensor
        Right-hand side 2D input tensor. Shape: (M, N).
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 and all its elements must be non-negative.
        group_sizes must be on the same device of lhs and rhs.
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (G, K, N), its data type must match
        preferred_element_type and it must be on the same device of other input tensors.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.
    bias_grad : torch.Tensor or None, optional
        Optional bias gradient output tensor. Shape: (G, K).
        If provided, the kernel will compute the bias gradient and write it to this tensor.
        bias_grad must be torch.float32 (kernel uses atomic_add which requires float32),
    accumulate : bool, optional
        Whether to accumulate into existing output tensor values. Default is False.
        If False, output will be overwritten with fresh computation.
        If True, results will be added to existing output tensor values.

    Returns
    -------
    torch.Tensor
        The computed output 3D tensor. Shape: (G, K, N).
        Output tensor data type is given by preferred_element_type.
        If existing_out is provided then existing_out is also returned.

    Implementation Notes
    --------------------
    - PTGMM is implemented with a persistent Triton kernel.
    - lhs can be row-major (lhs.stride() == (M, 1)) or column-major (lhs.stride() == (1, K)). If lhs
      is row-major then kernel parameter TRANS_LHS == False. If lhs is column-major then kernel
      parameter TRANS_LHS == True, this is useful for computing the rhs derivative in the backward
      pass, while fusing the transposition.
    - rhs must be row-major (rhs.stride() == (N, 1)).
    - out must be row-major (out.stride() == (K * N, N, 1)).
    """
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

    if config is None:
        config = get_config("ptgmm", M, K, N, G, accumulate)

    assert all(
        key in config
        and isinstance(config[key], int)
        and (
            is_power_of_2(config[key])
            if key.startswith("BLOCK_SIZE_")
            else config[key] > 0
        )
        for key in {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_K",
            "BLOCK_SIZE_N",
            "GROUP_SIZE",
            "GRID_DIM",
        }
    ), "Invalid PTGMM kernel config."

    # Bias gradient handling.
    # -----------------------
    # Get or validate bias gradient tensor.
    compute_bias_grad = bias_grad is not None
    bias_grad_ptr = get_tgmm_bias_grad(
        K,
        G,
        device=lhs.device,
        existing_bias_grad=bias_grad,
    )

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
        lhs, rhs, group_sizes, out, bias_grad_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_LHS=trans_lhs,
        COMPUTE_BIAS_GRAD=compute_bias_grad,
        ACCUMULATE=accumulate,
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
    config: dict[str, int] | None = None,
    bias_grad: Tensor | None = None,
    accumulate: bool = False,
) -> Tensor:
    """
    Perform a Group Matrix Multiplication (GMM) variant: out = lhs @ rhs

    lhs columns and rhs rows are divided into G groups. Each group of lhs is matrix multiplied with
    the respective group of rhs and then stored in a plane of the output 3D tensor. In PyTorch
    parlance, it can be implemented as follows for a given group g:
        out[g] = lhs[:, group_start:group_end] @ rhs[group_start:group_end, :]

    The 't' in the operator name derives from MaxText implementation
    (https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/kernels/megablox/gmm.py),
    which served as the initial inspiration for this one. TGMM differs from GMM in terms of tensor
    shapes. GMM does (M, K) @ (G, K, N) = (M, N) while TGMM does (K, M) @ (M, N) = (G, K, N).

    The 'np' in the operator name means that it is implemented with a non-persistent, i.e. regular
    kernel. There is also the persistent variation, which is implemented with a persistent kernel.
    Please take a look at ptgmm operator. Both nptgmm and ptgmm implement the same computation,
    choosing one or the other is a matter of performance for the target workload.

    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side 2D input tensor. Shape: (K, M).
        lhs data type must be torch.float16 or torch.bfloat16, and must match rhs data type.
        lhs must be on the same device of rhs and group_sizes.
    rhs : torch.Tensor
        Right-hand side 2D input tensor. Shape: (M, N).
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 and all its elements must be non-negative.
        group_sizes must be on the same device of lhs and rhs.
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (G, K, N), its data type must match
        preferred_element_type and it must be on the same device of other input tensors.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.
    bias_grad : torch.Tensor or None, optional
        Optional bias gradient output tensor. Shape: (G, K).
        If provided, the kernel will compute the bias gradient and write it to this tensor.
        bias_grad must be torch.float32 (kernel uses atomic_add which requires float32),
    accumulate : bool, optional
        Whether to accumulate into existing output tensor values. Default is False.
        If False, output will be overwritten with fresh computation.
        If True, results will be added to existing output tensor values.

    Returns
    -------
    torch.Tensor
        The computed output 3D tensor. Shape: (G, K, N).
        Output tensor data type is given by preferred_element_type.
        If existing_out is provided then existing_out is also returned.

    Implementation Notes
    --------------------
    - NPTGMM is implemented with a non-persistent regular Triton kernel.
    - lhs can be row-major (lhs.stride() == (M, 1)) or column-major (lhs.stride() == (1, K)). If lhs
      is row-major then kernel parameter TRANS_LHS == False. If lhs is column-major then kernel
      parameter TRANS_LHS == True, this is useful for computing the rhs derivative in the backward
      pass, while fusing the transposition.
    - rhs must be row-major (rhs.stride() == (N, 1)).
    - out must be row-major (out.stride() == (K * N, N, 1)).
    """
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

    # Bias gradient handling.
    # -----------------------
    # Get or validate bias gradient tensor.
    compute_bias_grad = bias_grad is not None
    bias_grad_ptr = get_tgmm_bias_grad(
        K,
        G,
        device=lhs.device,
        existing_bias_grad=bias_grad,
    )

    if config is None:
        config = get_config("nptgmm", M, K, N, G, accumulate)

    assert all(
        key in config
        and isinstance(config[key], int)
        and (
            is_power_of_2(config[key])
            if key.startswith("BLOCK_SIZE_")
            else config[key] > 0
        )
        for key in {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_K",
            "BLOCK_SIZE_N",
            "GROUP_SIZE",
        }
    ), "Invalid NPTGMM kernel config."

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
        lhs, rhs, group_sizes, out, bias_grad_ptr,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_LHS=trans_lhs,
        COMPUTE_BIAS_GRAD=compute_bias_grad,
        ACCUMULATE=accumulate,
        **config,
    )
    # fmt: on

    return out
