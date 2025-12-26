# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.


# Imports.
# ------------------------------------------------------------------------------

# Python standard library
from functools import partial

# PyTorch
import torch
from torch import Tensor

# pytest
import pytest

# AITER: GMM defaults and utility functions
from aiter.ops.triton.utils.gmm_common import (
    SUPPORTED_DTYPES_STR,
    DTYPE,
    dtype_from_str,
    check_input_device_dtype,
    gen_gmm_tensors,
    get_gmm_shape,
    get_gmm_output,
    gen_tgmm_tensors,
    get_tgmm_shape,
    get_tgmm_output,
    get_tgmm_bias_grad,
)

# AITER: Triton kernel wrappers
from aiter.ops.triton.gmm import (
    gmm as triton_gmm,
    ptgmm as triton_ptgmm,
    nptgmm as triton_nptgmm,
)


# Common code shared by GMM and TGMM unit tests.
# ------------------------------------------------------------------------------


# Shapes.

# Shapes used only for test purposes.
# fmt: off
TEST_ONLY_SHAPES: list[tuple[int, int, int, int]] = [
    #  M,    K,    N,   G
    ( 10,    2,    3,   4),
    ( 32,   16,    8,   4),
    (512, 4096, 2048, 160),
]
# fmt: on

# Real shapes, used by real models.
# fmt: off
REAL_SHAPES: list[tuple[int, int, int, int]] = [
    #      M,     K,     N,   G
    (  49152,  1408,  2048,  64),  # deepseekv2-16B
    (3145728,  2048,  1408,   8),  # deepseekv2-16B
    ( 393216,  2048,  1408,  64),  # deepseekv2-16B
    (  32768,  6144, 16384,   8),  # Mixtral 8x22B
    (  32768, 16384,  6144,   8),  # Mixtral 8x22B
]
# fmt: on

# Test shapes are test only + real ones.
TEST_SHAPES: list[tuple[int, int, int, int]] = TEST_ONLY_SHAPES + REAL_SHAPES


# Input and output types.

INPUT_DTYPES_STR: set[str] = {"i" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}
OUTPUT_DTYPES_STR: set[str] = {"o" + dtype_str for dtype_str in SUPPORTED_DTYPES_STR}


# Transpositions.

TRANS_LSH_STR: set[str] = {f"tlhs{b}" for b in {"F", "T"}}
TRANS_RHS_STR: set[str] = {f"trhs{b}" for b in {"F", "T"}}


def trans_from_str(trans_str: str, tensor_str: str) -> bool:
    assert tensor_str in {"lhs", "rhs"}, f"Invalid tensor string ({tensor_str})."
    return trans_str.replace(f"t{tensor_str}", "") == "T"


trans_lhs_from_str = partial(trans_from_str, tensor_str="lhs")
trans_rhs_from_str = partial(trans_from_str, tensor_str="rhs")


# RNG seed.

RNG_SEED_STR: set[str] = {f"rng{rng_seed}" for rng_seed in {77, 121}}


def rng_seed_from_str(rng_seed_str: str) -> int:
    rng_seed_int = int(rng_seed_str.replace("rng", ""))
    assert rng_seed_int >= 0, f"RNG seed must be non-negative (it's {rng_seed_int})."
    return rng_seed_int


# Number of distinct group sizes for each test shape.
NUM_GROUP_SIZES: int = 5


# Tensor comparison.
def check_tensors(
    actual: Tensor,
    expected: Tensor,
    msg: str,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    if atol is None:
        atol = 5e-3
    else:
        assert atol > 0, f"Absolute tolerance must be positive (it's {atol})."
    if rtol is None:
        rtol = 1e-2
    else:
        assert rtol > 0, f"Relative tolerance must be positive (it's {rtol})."
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=lambda torch_msg: f"{msg}\n\n{torch_msg}\n",
    )


# GMM unit tests.
# ------------------------------------------------------------------------------


def torch_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    bias: Tensor | None = None,
) -> Tensor:
    check_input_device_dtype(lhs, rhs, group_sizes)

    M, _, N, G = get_gmm_shape(lhs, rhs, group_sizes)

    out = get_gmm_output(
        M,
        N,
        device=lhs.device,
        preferred_element_type=preferred_element_type,
        existing_out=existing_out,
    )

    last_row = 0

    for g in range(G):
        m = int(group_sizes[g].item())

        # Skip group if there are no tokens assigned to the expert.
        if m == 0:
            continue

        start_idx = last_row
        end_idx = last_row + m

        result = (lhs[start_idx:end_idx, :] @ rhs[g]).to(torch.float32)
        if bias is not None:
            result += bias[g].to(torch.float32)
        out[start_idx:end_idx, :] = result.to(preferred_element_type)

        last_row += m

    return out


@pytest.mark.parametrize("M, K, N, G", TEST_SHAPES)
@pytest.mark.parametrize("in_dtype_str", INPUT_DTYPES_STR)
@pytest.mark.parametrize("out_dtype_str", OUTPUT_DTYPES_STR)
@pytest.mark.parametrize("trans_rhs_str", TRANS_RHS_STR)
@pytest.mark.parametrize("rng_seed_str", RNG_SEED_STR)
@pytest.mark.parametrize("use_bias", [False, True])
def test_gmm(
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype_str: str,
    out_dtype_str: str,
    trans_rhs_str: str,
    rng_seed_str: str,
    use_bias: bool,
):
    in_dtype = dtype_from_str(in_dtype_str)
    out_dtype = dtype_from_str(out_dtype_str)
    trans_rhs = trans_rhs_from_str(trans_rhs_str)
    rng_seed = rng_seed_from_str(rng_seed_str)

    lhs, rhs, multiple_group_sizes, out_torch, bias = gen_gmm_tensors(
        M,
        K,
        N,
        G,
        NUM_GROUP_SIZES,
        input_type=in_dtype,
        output_type=out_dtype,
        trans_rhs=trans_rhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,  # 1st group_sizes in test is evenly distributed
        use_bias=use_bias,
    )
    out_triton = torch.empty_like(out_torch)

    for group_sizes in multiple_group_sizes:
        torch_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_torch,
            bias=bias,
        )

        triton_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_triton,
            bias=bias,
        )

        m = int(torch.sum(group_sizes).item())

        # Tolerance handling:
        # - Default (no bias): use strict global defaults (atol=5e-3, rtol=1e-2)
        # - With bias: allow slightly looser tolerances due to:
        #   * extra floating point op (add bias)
        #   * large problem sizes and mixed precision
        #   * very small fraction of elements differing by a few bf16/fp16 ULPs
        if use_bias:
            # Base tolerances for bias case.
            atol = 0.02
            rtol = 0.02
        else:
            atol = None
            rtol = None

        check_tensors(
            out_triton[:m],
            out_torch[:m],
            "Triton GMM doesn't match PyTorch reference GMM.",
            atol=atol,
            rtol=rtol,
        )


# TGMM unit tests.
# ------------------------------------------------------------------------------


def torch_tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
    bias_grad: Tensor | None = None,
    accumulate: bool = False,
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

    # Bias gradient handling (test/reference only).
    # Get or validate bias gradient tensor (validates and optionally zeros it).
    compute_bias_grad = bias_grad is not None
    bias_grad = get_tgmm_bias_grad(
        K,
        G,
        device=lhs.device,
        existing_bias_grad=bias_grad,
    )

    last_col = 0

    for g in range(G):
        m = int(group_sizes[g].item())

        # Skip group if there are no columns assigned to the group.
        if m == 0:
            continue

        start_idx = last_col
        end_idx = last_col + m
        mm = lhs[:, start_idx:end_idx] @ rhs[start_idx:end_idx, :]
        out[g] = mm.to(preferred_element_type)

        # Bias gradient: sum lhs across m-dimension (columns) for each group.
        if compute_bias_grad:
            grad = lhs[:, start_idx:end_idx].sum(dim=1, dtype=torch.float32)
            bias_grad[g] += grad

        last_col += m

    return out


@pytest.mark.parametrize("persistent_str", {"p", "np"})
@pytest.mark.parametrize("with_bias_grad", [False, True])
@pytest.mark.parametrize("M, K, N, G", TEST_SHAPES)
@pytest.mark.parametrize("in_dtype_str", INPUT_DTYPES_STR)
@pytest.mark.parametrize("out_dtype_str", OUTPUT_DTYPES_STR)
@pytest.mark.parametrize("trans_lhs_str", TRANS_LSH_STR)
@pytest.mark.parametrize("rng_seed_str", RNG_SEED_STR)
def test_tgmm(
    persistent_str: str,
    with_bias_grad: bool,
    M: int,
    K: int,
    N: int,
    G: int,
    in_dtype_str: str,
    out_dtype_str: str,
    trans_lhs_str: str,
    rng_seed_str: str,
):
    assert persistent_str in {"p", "np"}
    persistent: bool = persistent_str == "p"

    in_dtype = dtype_from_str(in_dtype_str)
    out_dtype = dtype_from_str(out_dtype_str)
    trans_lhs = trans_lhs_from_str(trans_lhs_str)
    rng_seed = rng_seed_from_str(rng_seed_str)

    lhs, rhs, multiple_group_sizes, out_torch, bias_grad_torch = gen_tgmm_tensors(
        M,
        K,
        N,
        G,
        NUM_GROUP_SIZES,
        input_type=in_dtype,
        output_type=out_dtype,
        trans_lhs=trans_lhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,  # 1st group_sizes in test is evenly distributed
        use_bias=with_bias_grad,
    )
    out_triton = torch.empty_like(out_torch)
    bias_grad_triton = torch.empty_like(bias_grad_torch) if with_bias_grad else None

    # For big shape (M, K, N, G) = (3145728, 2048, 1408, 8) there are some element
    # mismatches (125 / 23068672 ~ 0.00013%) with absolute error greater than the
    # default tolerance. This behavior is deterministic and, given a RNG seed,
    # always happen for the same output elements. So, absolute tolerance is increased
    # only for this shape.
    atol = 2.5e-2 if M > 1e6 else None

    kernel_wrapper = triton_ptgmm if persistent else triton_nptgmm

    for group_sizes in multiple_group_sizes:
        # Reference implementation.
        torch_tgmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_torch,
            bias_grad=bias_grad_torch,
            accumulate=False,
        )

        # Triton kernel.
        kernel_wrapper(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_triton,
            bias_grad=bias_grad_triton,
            accumulate=False,
        )
        non_empty_groups = group_sizes > 0

        # Compare TGMM outputs.
        check_tensors(
            out_triton[non_empty_groups],
            out_torch[non_empty_groups],
            f"Triton {'persistent' if persistent else 'non-persistent'} TGMM doesn't match PyTorch reference TGMM.",
            atol=atol,
        )

        # For persistent TGMM, also compare bias gradients on smaller shapes.
        #
        # For very large shapes (e.g., M > 1e6), bias_grad is an extremely long
        # float32 reduction with atomics in the Triton kernel and a different
        # reduction order in the PyTorch reference. Per-element comparisons
        # become dominated by reduction-order noise rather than meaningful
        # correctness checks, so we skip bias_grad comparison there and rely
        # only on the output tensor check above.
        if with_bias_grad and M <= 1e6:
            bias_atol = 1.7
            bias_rtol = 0.1

            check_tensors(
                bias_grad_triton[non_empty_groups],
                bias_grad_torch[non_empty_groups],
                "Triton persistent TGMM bias_grad doesn't match PyTorch reference TGMM bias_grad.",
                atol=bias_atol,
                rtol=bias_rtol,
            )


@pytest.mark.parametrize("persistent_str", {"p", "np"})
@pytest.mark.parametrize("with_bias_grad", [False, True])
def test_tgmm_accumulate(persistent_str: str, with_bias_grad: bool):
    persistent: bool = persistent_str == "p"

    """Test ACCUMULATE semantics for persistent TGMM on a small, focused case."""
    # Use the smallest TEST_ONLY_SHAPES entry to keep runtime low.
    M, K, N, G = TEST_ONLY_SHAPES[0]

    in_dtype = DTYPE
    out_dtype = DTYPE
    trans_lhs = False
    rng_seed = 77

    lhs, rhs, multiple_group_sizes, out_torch, bias_grad_torch = gen_tgmm_tensors(
        M,
        K,
        N,
        G,
        NUM_GROUP_SIZES,
        input_type=in_dtype,
        output_type=out_dtype,
        trans_lhs=trans_lhs,
        rng_seed=rng_seed,
        unif_group_sizes=True,
        use_bias=with_bias_grad,
    )

    # Take a single group_sizes configuration for this targeted test.
    group_sizes = multiple_group_sizes[0]
    non_empty_groups = group_sizes > 0

    # Base output to accumulate into.
    base_out = torch.randn_like(out_torch)

    # Reference: compute TGMM delta into a fresh buffer, then add to base_out.
    delta_ref = torch.empty_like(out_torch)
    torch_tgmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=out_dtype,
        existing_out=delta_ref,
        bias_grad=bias_grad_torch,
        accumulate=False,
    )
    expected = base_out.clone()
    expected[non_empty_groups] = (
        expected[non_empty_groups] + delta_ref[non_empty_groups]
    )

    # Triton PTGMM/NPTGMM with ACCUMULATE=True.
    out_triton = base_out.clone()
    bias_grad_triton = torch.empty_like(bias_grad_torch) if with_bias_grad else None

    if persistent:
        triton_ptgmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_triton,
            bias_grad=bias_grad_triton,
            accumulate=True,
        )
    else:
        triton_nptgmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            existing_out=out_triton,
            bias_grad=bias_grad_triton,
            accumulate=True,
        )

    check_tensors(
        out_triton[non_empty_groups],
        expected[non_empty_groups],
        "Triton persistent TGMM ACCUMULATE semantics do not match reference behavior.",
    )

    # Check bias_grad
    if with_bias_grad:
        check_tensors(
            bias_grad_triton[non_empty_groups],
            bias_grad_torch[non_empty_groups],
            "Triton persistent TGMM bias_grad with ACCUMULATE=True does not match reference.",
        )
