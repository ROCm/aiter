# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
from typing import Literal
import warnings

# PyTorch
import torch
from torch import Tensor

# Triton
from triton import cdiv

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

_NUM_XCDS: int = 8
_TILE_COUNTER_STRIDE: int = 64

# Per-(device, stream) cache for the work stealing tile counter.
# Layout: [xcd_{0}_slot, ..., xcd{_NUM_XCDS-1}_slot, global_slot]
# It's one slot per XCD plus a global slot at the last position.
# Each slot _TILE_COUNTER_STRIDE int32 elements wide (256 B = 1 MI355X L2 line)
# to avoid false-sharing across XCDs.
# A single scratch buffer is reused across launches to avoid an allocator
# round-trip on every call.
_GMM_TILE_COUNTER_CACHE: dict[tuple[torch.device, int], Tensor] = {}


# Get work stealing tile counter.
def _get_gmm_tile_counter(device: torch.device) -> Tensor:
    stream = torch.cuda.current_stream(device=device).cuda_stream
    tile_counter = _GMM_TILE_COUNTER_CACHE.get((device, stream))
    if tile_counter is None:
        tile_counter = torch.zeros(
            (_NUM_XCDS + 1) * _TILE_COUNTER_STRIDE, dtype=torch.int32, device=device
        )
        _GMM_TILE_COUNTER_CACHE[(device, stream)] = tile_counter
    else:
        tile_counter.zero_()
    return tile_counter


# Compute total number of GMM tiles. Transfer `group_sizes` tensor to host.
def _gmm_total_tiles(
    group_sizes: torch.Tensor | list[int],
    N: int,
    block_size_m: int,
    block_size_n: int,
) -> int:
    group_sizes_list: list[int] = (
        group_sizes.cpu().tolist()
        if isinstance(group_sizes, torch.Tensor)
        else group_sizes
    )
    all_m_tiles = (cdiv(group_size, block_size_m) for group_size in group_sizes_list)
    n_tiles = cdiv(N, block_size_n)
    num_tiles = sum(m_tiles * n_tiles for m_tiles in all_m_tiles)
    return num_tiles


# Compute number of GMM tiles per XCD. It's used with work stealing kernel.
def _gmm_tiles_per_xcd(
    group_sizes: torch.Tensor,
    M: int,
    N: int,
    block_size_m: int,
    block_size_n: int,
    mode: Literal["global", "per_xcd"] = "global",
    num_tiles: int | None = None,
) -> int:
    assert mode in (
        "global",
        "per_xcd",
    ), f"GMM tiles per XCD work stealing mode must be in {{'global', 'per_xcd'}} (got '{mode}')."
    assert (num_tiles is None) or (num_tiles > 0), "GMM must contain at least one tile."

    if mode == "global":
        return 0

    assert mode == "per_xcd"
    if num_tiles is None:
        G = group_sizes.shape[0]
        num_tiles_estimate = G * cdiv(M // G, block_size_m) * cdiv(N, block_size_n)
        # If we have too few tiles then moving `group_sizes` to host and calculating the exact
        # number isn't worthwhile.
        num_tiles = (
            num_tiles_estimate
            if num_tiles_estimate < 1.5e5
            else _gmm_total_tiles(group_sizes, N, block_size_m, block_size_n)
        )
    tiles_per_xcd = cdiv(num_tiles, _NUM_XCDS)
    return tiles_per_xcd


def _gmm_grid(
    N: int,
    block_size_m: int,
    block_size_n: int,
    group_sizes: Tensor,
    grid_dim: int,
    # Minimum number of programs to launch. Per-XCD work stealing requires at
    # least one program per XCD (i.e. `min_programs == _NUM_XCDS`). Otherwise,
    # the per-XCD regions of the absent XCDs are never claimed and their output
    # tiles are left untouched.
    min_programs: int = 0,
    # Expensive assertions launch GPU kernels on `group_sizes` and dominate the
    # host-side launch cost. Only enable them in development.
    enable_expensive_assertions: bool = False,
) -> tuple[int, int | None]:
    assert (
        min_programs >= 0
    ), f"Minimum number of programs must be non-negative (got {min_programs})."

    num_tiles: int | None
    num_programs: int

    if enable_expensive_assertions:
        group_sizes_list: list[int] = group_sizes.cpu().tolist()
        assert all(group_size >= 0 for group_size in group_sizes_list), (
            "All GMM group sizes must be non-negative, but at least one element "
            "of group_sizes is negative."
        )
        num_tiles = _gmm_total_tiles(group_sizes_list, N, block_size_m, block_size_n)
        assert num_tiles > 0, (
            "GMM has no tiles to launch: group_sizes must contain at least one "
            f"non-empty group (computed {num_tiles} total tiles)."
        )
        num_programs = min(grid_dim, num_tiles)
    else:
        # Cheap-path default. The kernel handle the case where grid_dim exceeds the
        # total tile count: extra programs just exit without doing any work.
        num_tiles = None
        num_programs = grid_dim

    # Floor must be applied after the `min(grid_dim, num_tiles)` clamp above, so
    # the clamp can't drop the program count below the per-XCD minimum. Extra
    # programs (when this exceeds the tile count) just exit without doing work.
    num_programs = max(num_programs, min_programs)

    return num_programs, num_tiles


def gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    bias: Tensor | None = None,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    work_stealing: bool = False,
    work_stealing_mode: Literal["global", "per_xcd"] | None = None,
    config: dict[str, int] | None = None,
    grid_dim: int | None = None,
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
        Right-hand side 3D input tensor. Shape is (G, K, N) when rhs is non-transposed. When rhs is
        transposed, two physically equivalent metadata layouts are supported: shape (G, K, N) with a
        column-major-like stride, and shape (G, N, K) with a row-major stride. See Implementation
        Notes below for the supported (shape, stride) combinations.
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 or torch.int64, and all its elements must be
        non-negative.
        group_sizes must be on the same device of lhs and rhs.
    bias : torch.Tensor or None, optional
        Optional bias tensor. Shape: (G, N).
        If provided, bias data type must match lhs and rhs data type, and bias must be on the same
        device as other input tensors. Each group g adds bias[g] to the output.
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (M, N), its data type must match preferred_element_type
        and it must be on the same device of other input tensors.
    work_stealing : bool, defaults to False
        Enable work stealing, i.e. dynamic load-balancing where CUs with no assigned tiles "steal"
        the next available tile to be computed.
    work_stealing_mode : Literal["global", "per_xcd"] or None, optional
        Work stealing behavior with respect to atomic synchronization. In "global" mode, all work
        groups fetch the next tile from a shared atomic counter. In "per_xcd" mode, we have XCDs + 1
        atomic counters: the work groups start fetching from a per-XCD shared counter and then
        proceed to the global counter to drain the remaining tiles.
        This argument only makes sense when `work_stealing` is True. If `work_stealing` is True and
        `work_stealing_mode` is None, fall-back to "global" behavior.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.
    grid_dim : positive int or None, optional
        Optional override for GRID_DIM config. It's useful to override it while doing performance
        experiments or launching the GMM kernel in parallel with a comms kernel (reserve some CUs
        for comms).

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
    - rhs supports three storage layouts. The two transposed layouts are physically
      equivalent (same memory ordering, K varies fastest, then N, then G); only the
      tensor metadata (shape and stride) differs. Both transposed layouts select
      kernel parameter TRANS_RHS == True and produce identical byte offsets in the
      kernel's pointer arithmetic, so they execute the same code:
        * Non-transposed: shape (G, K, N), stride (K*N, N, 1). Kernel parameter
          TRANS_RHS == False. Useful for the forward pass.
        * Transposed (layout 1): shape (G, K, N), stride (K*N, 1, K). Kernel parameter
          TRANS_RHS == True. The (K, N) sub-matrix per group is column-major.
        * Transposed (layout 2): shape (G, N, K), stride (K*N, K, 1). Kernel parameter
          TRANS_RHS == True. The (N, K) sub-matrix per group is row-major.
      Both transposed layouts are useful for computing the lhs derivative in the
      backward pass while fusing the transposition. The choice between layout 1 and
      layout 2 is purely a metadata preference of the calling code.
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
    ), (
        "Invalid GMM kernel config: each of BLOCK_SIZE_M, BLOCK_SIZE_K, "
        "BLOCK_SIZE_N, GROUP_SIZE and GRID_DIM must be present, with BLOCK_SIZE_* "
        f"being powers of 2 and the rest positive integers. Got: {config}."
    )

    # Override grid dimension, if optional argument is provided.
    assert (grid_dim is None) or (
        grid_dim > 0
    ), f"grid_dim must be None or a positive integer (got {grid_dim})."
    if grid_dim is not None and grid_dim != config["GRID_DIM"]:
        warnings.warn(
            f"Overriding GMM grid dim with {grid_dim} (it was {config['GRID_DIM']})."
        )
        # Copy before mutating: when `config` comes from `get_config` it's the
        # dict cached by `@functools.lru_cache`, so an in-place write would leak
        # the override into subsequent calls.
        config = dict(config)
        config["GRID_DIM"] = grid_dim

    resolved_ws_mode: str = (
        "global" if work_stealing_mode is None else work_stealing_mode
    )

    num_programs, num_tiles = _gmm_grid(
        N,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        group_sizes,
        config["GRID_DIM"],
        # per_xcd partitions tiles by XCD residue class, so every XCD needs at
        # least one claiming program; global drains a single counter and needs no floor.
        min_programs=(
            _NUM_XCDS if work_stealing and resolved_ws_mode == "per_xcd" else 0
        ),
    )

    tile_counter: Tensor | None
    tiles_per_xcd: int | None
    if work_stealing:
        tile_counter = _get_gmm_tile_counter(lhs.device)
        tiles_per_xcd = _gmm_tiles_per_xcd(
            group_sizes,
            M,
            N,
            config["BLOCK_SIZE_M"],
            config["BLOCK_SIZE_N"],
            mode=resolved_ws_mode,
            num_tiles=num_tiles,
        )
    else:
        tile_counter, tiles_per_xcd = None, None

    # fmt: off
    gmm_kernel[(num_programs,)](
        # Tensor pointers:
        lhs, rhs, group_sizes, out, bias,
        # Work stealing parameters:
        tile_counter, tiles_per_xcd,
        # Tensor shapes:
        M, K, N, G,
        # Meta-parameters:
        TRANS_RHS=trans_rhs,
        USE_BIAS=use_bias,
        # Work stealing meta-parameters:
        WORK_STEALING=work_stealing,
        NUM_XCDS=_NUM_XCDS,
        TILE_COUNTER_STRIDE=_TILE_COUNTER_STRIDE,
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
    assert K > 0, f"Number of output rows K must be positive (got K = {K})."
    assert N > 0, f"Number of output columns N must be positive (got N = {N})."
    assert G > 0, f"Number of groups G must be positive (got G = {G})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size BLOCK_SIZE_K must be a power of 2 (got {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size BLOCK_SIZE_N must be a power of 2 (got {block_size_n})."
    assert (
        grid_dim > 0
    ), f"Grid dimension (number of programs) must be positive (got {grid_dim})."
    num_k_tiles = cdiv(K, block_size_k)
    num_n_tiles = cdiv(N, block_size_n)
    num_tiles = G * num_k_tiles * num_n_tiles
    num_programs = min(grid_dim, num_tiles)
    return (num_programs,)


def ptgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    bias_grad: Tensor | None = None,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    accumulate: bool = False,
    config: dict[str, int] | None = None,
    grid_dim: int | None = None,
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
        Left-hand side 2D input tensor. Shape is (K, M) when lhs is non-transposed. When lhs is
        transposed, two physically equivalent metadata layouts are supported: shape (K, M) with a
        column-major stride, and shape (M, K) with a row-major stride. See Implementation Notes
        below for the supported (shape, stride) combinations.
        lhs data type must be torch.float16 or torch.bfloat16, and must match rhs data type.
        lhs must be on the same device of rhs and group_sizes.
    rhs : torch.Tensor
        Right-hand side 2D input tensor. Shape: (M, N).
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 or torch.int64, and all its elements must be
        non-negative.
        group_sizes must be on the same device of lhs and rhs.
    bias_grad : torch.Tensor or None, optional
        Optional bias gradient output tensor. Shape: (G, K).
        If provided, the kernel will compute the bias gradient and write it to this tensor.
        bias_grad must be torch.float32 (kernel uses atomic_add which requires float32),
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (G, K, N), its data type must match
        preferred_element_type and it must be on the same device of other input tensors.
    accumulate : bool, optional
        Whether to accumulate into existing output tensor values. Default is False.
        If False, output will be overwritten with fresh computation.
        If True, results will be added to existing output tensor values.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.
    grid_dim : positive int or None, optional
        Optional override for GRID_DIM config. It's useful to override it while doing performance
        experiments or launching the persistent TGMM kernel in parallel with a comms kernel (reserve
        some CUs for comms).

    Returns
    -------
    torch.Tensor
        The computed output 3D tensor. Shape: (G, K, N).
        Output tensor data type is given by preferred_element_type.
        If existing_out is provided then existing_out is also returned.

    Implementation Notes
    --------------------
    - PTGMM is implemented with a persistent Triton kernel.
    - lhs supports three storage layouts. The two transposed layouts are physically
      equivalent (same memory ordering, K varies fastest, then M); only the tensor
      metadata (shape and stride) differs. Both transposed layouts select kernel
      parameter TRANS_LHS == True and produce identical byte offsets in the kernel's
      pointer arithmetic, so they execute the same code:
        * Non-transposed: shape (K, M), stride (M, 1). Kernel parameter
          TRANS_LHS == False.
        * Transposed (layout 1): shape (K, M), stride (1, K). Kernel parameter
          TRANS_LHS == True. lhs is column-major.
        * Transposed (layout 2): shape (M, K), stride (K, 1). Kernel parameter
          TRANS_LHS == True. lhs is row-major over the swapped shape.
      Both transposed layouts are useful for computing the rhs derivative in the
      backward pass while fusing the transposition. The choice between layout 1 and
      layout 2 is purely a metadata preference of the calling code.
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
    ), (
        "Invalid PTGMM kernel config: each of BLOCK_SIZE_M, BLOCK_SIZE_K, "
        "BLOCK_SIZE_N, GROUP_SIZE and GRID_DIM must be present, with BLOCK_SIZE_* "
        f"being powers of 2 and the rest positive integers. Got: {config}."
    )

    # Override grid dimension, if optional argument is provided.
    assert (grid_dim is None) or (
        grid_dim > 0
    ), f"grid_dim must be None or a positive integer (got {grid_dim})."
    if grid_dim is not None and grid_dim != config["GRID_DIM"]:
        warnings.warn(
            f"Overriding PTGMM grid dim with {grid_dim} (it was {config['GRID_DIM']})."
        )
        # Copy before mutating: when `config` comes from `get_config` it's the
        # dict cached by `@functools.lru_cache`, so an in-place write would leak
        # the override into subsequent calls.
        config = dict(config)
        config["GRID_DIM"] = grid_dim

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
    assert K > 0, f"Number of output rows K must be positive (got K = {K})."
    assert N > 0, f"Number of output columns N must be positive (got N = {N})."
    assert G > 0, f"Number of groups G must be positive (got G = {G})."
    assert is_power_of_2(
        block_size_k
    ), f"K-dimension tile size BLOCK_SIZE_K must be a power of 2 (got {block_size_k})."
    assert is_power_of_2(
        block_size_n
    ), f"N-dimension tile size BLOCK_SIZE_N must be a power of 2 (got {block_size_n})."
    num_k_tiles = cdiv(K, block_size_k)
    num_n_tiles = cdiv(N, block_size_n)
    num_tiles_per_mm = num_k_tiles * num_n_tiles
    return (G, num_tiles_per_mm)


def nptgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    bias_grad: Tensor | None = None,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    accumulate: bool = False,
    config: dict[str, int] | None = None,
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
        Left-hand side 2D input tensor. Shape is (K, M) when lhs is non-transposed. When lhs is
        transposed, two physically equivalent metadata layouts are supported: shape (K, M) with a
        column-major stride, and shape (M, K) with a row-major stride. See Implementation Notes
        below for the supported (shape, stride) combinations.
        lhs data type must be torch.float16 or torch.bfloat16, and must match rhs data type.
        lhs must be on the same device of rhs and group_sizes.
    rhs : torch.Tensor
        Right-hand side 2D input tensor. Shape: (M, N).
        rhs data type must be torch.float16 or torch.bfloat16, and must match lhs data type.
        rhs must be on the same device of lhs and group_sizes.
    group_sizes : torch.Tensor
        1D input tensor describing group sizes. Shape: (G,).
        group_sizes data type must be torch.int32 or torch.int64, and all its elements must be
        non-negative.
        group_sizes must be on the same device of lhs and rhs.
    bias_grad : torch.Tensor or None, optional
        Optional bias gradient output tensor. Shape: (G, K).
        If provided, the kernel will compute the bias gradient and write it to this tensor.
        bias_grad must be torch.float32 (kernel uses atomic_add which requires float32),
    preferred_element_type : torch.dtype, optional
        Desired data type for output tensor. Default is torch.bfloat16.
        Supported output types are torch.float16 and torch.bfloat16.
    existing_out : torch.Tensor or None, optional
        Preallocated output tensor. Default is None.
        If provided, results are written into this tensor. Otherwise, a new output tensor is
        allocated.
        If provided then it must have shape (G, K, N), its data type must match
        preferred_element_type and it must be on the same device of other input tensors.
    accumulate : bool, optional
        Whether to accumulate into existing output tensor values. Default is False.
        If False, output will be overwritten with fresh computation.
        If True, results will be added to existing output tensor values.
    config : dict[str, int] or None, optional
        Optional dictionary with kernel metaparameters. If absent, config will be queried from
        internal tuning database.

    Returns
    -------
    torch.Tensor
        The computed output 3D tensor. Shape: (G, K, N).
        Output tensor data type is given by preferred_element_type.
        If existing_out is provided then existing_out is also returned.

    Implementation Notes
    --------------------
    - NPTGMM is implemented with a non-persistent regular Triton kernel.
    - lhs supports three storage layouts. The two transposed layouts are physically
      equivalent (same memory ordering, K varies fastest, then M); only the tensor
      metadata (shape and stride) differs. Both transposed layouts select kernel
      parameter TRANS_LHS == True and produce identical byte offsets in the kernel's
      pointer arithmetic, so they execute the same code:
        * Non-transposed: shape (K, M), stride (M, 1). Kernel parameter
          TRANS_LHS == False.
        * Transposed (layout 1): shape (K, M), stride (1, K). Kernel parameter
          TRANS_LHS == True. lhs is column-major.
        * Transposed (layout 2): shape (M, K), stride (K, 1). Kernel parameter
          TRANS_LHS == True. lhs is row-major over the swapped shape.
      Both transposed layouts are useful for computing the rhs derivative in the
      backward pass while fusing the transposition. The choice between layout 1 and
      layout 2 is purely a metadata preference of the calling code.
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
    ), (
        "Invalid NPTGMM kernel config: each of BLOCK_SIZE_M, BLOCK_SIZE_K, "
        "BLOCK_SIZE_N and GROUP_SIZE must be present, with BLOCK_SIZE_* being "
        f"powers of 2 and GROUP_SIZE a positive integer. Got: {config}."
    )

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
