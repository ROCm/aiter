# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# Imports.
# ------------------------------------------------------------------------------

# PyTorch
import torch
from torch import Tensor

# AITER: logging
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER: AiterTritonLogger = AiterTritonLogger()


# Supported data types.
# ------------------------------------------------------------------------------

# Supported data types, as strings.
SUPPORTED_DTYPES_STR: set[str] = {"fp16", "bf16"}


# Convert string data type to PyTorch data type.
def dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    dtype_str = dtype_str[1:] if dtype_str[0] in {"i", "o"} else dtype_str
    assert (
        dtype_str in SUPPORTED_DTYPES_STR
    ), "String data type isn't in set of supported string data types."
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


# Supported data types, as PyTorch types.
SUPPORTED_DTYPES: set[torch.dtype] = {
    dtype_from_str(dtype_str) for dtype_str in SUPPORTED_DTYPES_STR
}


# Convert PyTorch data type to string data type.
def str_from_dtype(dtype: torch.dtype) -> str:
    assert (
        dtype in SUPPORTED_DTYPES
    ), "PyTorch data type isn't in set of supported PyTorch data types."
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]


# Default data type, as string.
DTYPE_STR: str = "bf16"
assert (
    DTYPE_STR in SUPPORTED_DTYPES_STR
), "Default string data type isn't in set of supported string data types."


# Default data type, as PyTorch type.
DTYPE: torch.dtype = dtype_from_str(DTYPE_STR)


# Supported integer data types for group sizes tensor.
# ------------------------------------------------------------------------------

# Supported group sizes data types, as strings.
SUPPORTED_GROUP_SIZES_DTYPES_STR: set[str] = {"int32", "int64"}


# Convert string data type to PyTorch data type.
def group_sizes_dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.strip().lower()
    assert (
        dtype_str in SUPPORTED_GROUP_SIZES_DTYPES_STR
    ), "String data type isn't in set of supported string data types."
    return {"int32": torch.int32, "int64": torch.int64}[dtype_str]


# Supported data types, as PyTorch types.
SUPPORTED_GROUP_SIZES_DTYPES: set[torch.dtype] = {
    group_sizes_dtype_from_str(dtype_str)
    for dtype_str in SUPPORTED_GROUP_SIZES_DTYPES_STR
}


# Convert PyTorch data type to string data type.
def str_from_group_sizes_dtype(dtype: torch.dtype) -> str:
    assert (
        dtype in SUPPORTED_GROUP_SIZES_DTYPES
    ), "PyTorch data type isn't in set of supported PyTorch data types."
    return {torch.int32: "int32", torch.int64: "int64"}[dtype]


# Default data type, as string.
GROUP_SIZES_DTYPE_STR: str = "int32"
assert (
    GROUP_SIZES_DTYPE_STR in SUPPORTED_GROUP_SIZES_DTYPES_STR
), "Default string data type isn't in set of supported string data types."


# Default data type, as PyTorch type.
GROUP_SIZES_DTYPE: torch.dtype = group_sizes_dtype_from_str(GROUP_SIZES_DTYPE_STR)


def check_group_sizes_dtype(dtype: torch.dtype) -> None:
    assert dtype in SUPPORTED_GROUP_SIZES_DTYPES, (
        f"group_sizes data type must be one of {SUPPORTED_GROUP_SIZES_DTYPES}, "
        f"got {dtype}."
    )


# Other defaults.
# ------------------------------------------------------------------------------

# Default device.
DEVICE: torch.device | str = "cuda"

# Default RNG seed for input generation.
RNG_SEED: int = 0

# Default number of group sizes.
NUM_GROUP_SIZES: int = 1

# Default transposition (NN).
TRANS_LHS: bool = False
TRANS_RHS: bool = False


# Parameter checking functions.
# ------------------------------------------------------------------------------


def is_power_of_2(x: int) -> bool:
    return (x > 0) and (x & (x - 1) == 0)


def check_input_device_dtype(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor, bias: Tensor | None = None
) -> None:
    assert (
        lhs.device == rhs.device == group_sizes.device
    ), f"All input tensors must be in the same device (lhs = {lhs.device}, rhs = {rhs.device}, group_sizes = {group_sizes.device})."
    assert (
        lhs.dtype == rhs.dtype
    ), f"lhs and rhs types must match (lhs = {lhs.dtype}, rhs = {rhs.dtype})."
    check_group_sizes_dtype(group_sizes.dtype)

    if bias is not None:
        assert (
            bias.device == lhs.device
        ), f"bias must be on the same device as lhs (bias = {bias.device}, lhs = {lhs.device})."
        assert (
            bias.dtype == lhs.dtype
        ), f"bias dtype must match lhs dtype (bias = {bias.dtype}, lhs = {lhs.dtype})."


def check_bias_shape_stride(bias: Tensor, G: int, N: int) -> None:
    assert bias.shape == (
        G,
        N,
    ), f"bias must have shape (G, N) = ({G}, {N}), got {bias.shape}."
    assert bias.stride() == (N, 1), "bias must be row-major (bias.stride() == (N, 1))."


# Generation of group sizes.
# ------------------------------------------------------------------------------


# Probabilities for generating random group sizes.
UNUSED_TOKENS_PROB: float = 0.0
UNUSED_EXPERTS_PROB: float = 0.1


def gen_uniform_group_sizes(
    M: int,
    G: int,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
) -> Tensor:
    assert M >= 0, f"Number of tokens M must be non-negative (it's {M})."
    assert G > 0, f"Number of experts G must be positive (it's {G})."
    check_group_sizes_dtype(group_sizes_dtype)

    base = M // G
    remainder = M % G
    group_sizes = torch.full((G,), base, dtype=group_sizes_dtype, device=device)
    if remainder > 0:
        group_sizes[:remainder] += 1

    assert (
        len(group_sizes) == G
    ), f"Group sizes don't have {G} elements (it's {len(group_sizes)})."
    assert torch.all(group_sizes >= 0).item(), "All group sizes must be non-negative."
    assert (
        torch.sum(group_sizes).item() == M
    ), f"Group sizes don't add up to total tokens {M}."
    assert (
        group_sizes.dtype == group_sizes_dtype
    ), f"Group sizes must be {group_sizes_dtype} (it's {group_sizes.dtype})."

    return group_sizes


def gen_group_sizes(
    M: int,
    G: int,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    rng_seed: int | None = RNG_SEED,
    unused_tokens_prob: float = UNUSED_TOKENS_PROB,
    unused_experts_prob: float = UNUSED_EXPERTS_PROB,
) -> Tensor:
    assert M >= 0, f"Number of tokens M must be non-negative (it's {M})."
    assert G > 0, f"Number of experts G must be positive (it's {G})."
    assert (
        0 <= unused_tokens_prob <= 1
    ), f"Probability of unused tokens must be in [0, 1] interval (it's {unused_tokens_prob})."
    assert (
        0 <= unused_experts_prob <= 1
    ), f"Probability of unused experts must be in [0, 1] interval (it's {unused_experts_prob})."
    check_group_sizes_dtype(group_sizes_dtype)

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    if unused_tokens_prob > 0:
        # Optionally drop tokens to simulate routing sparsity, some tokens may not be routed.
        num_unused_tokens = M
        while num_unused_tokens == M:
            num_unused_tokens = int(
                torch.binomial(
                    torch.tensor(float(M), device=device),
                    torch.tensor(unused_tokens_prob, device=device),
                ).item()
            )
    else:
        num_unused_tokens = 0
    num_used_tokens = M - num_unused_tokens
    assert (
        num_unused_tokens >= 0
    ), f"Number of unused tokens must be non-negative (it's {num_unused_tokens})."
    assert (
        num_used_tokens > 0
    ), f"Number of used tokens must be positive (it's {num_used_tokens})."
    assert (
        num_used_tokens + num_unused_tokens == M
    ), f"Unused + used tokens don't add up total tokens ({num_used_tokens} + {num_unused_tokens} != {M})."

    if num_unused_tokens > 0:
        _LOGGER.debug(
            f"Group sizes generation: dropped {num_unused_tokens} token{'s' if num_unused_tokens > 1 else ''}.",
        )

    if unused_experts_prob > 0:
        # Some experts may have zero tokens assigned to them.
        num_used_experts = 0
        while num_used_experts == 0:
            used_experts = torch.nonzero(
                torch.rand((G,), device=device) >= unused_experts_prob
            ).squeeze()
            num_used_experts = used_experts.numel()
    else:
        used_experts = torch.arange(0, G, device=device)
        num_used_experts = G
    num_unused_experts = G - num_used_experts
    assert (
        num_unused_experts >= 0
    ), f"Number of unused experts must be non-negative (it's {num_unused_experts})."
    assert (
        num_used_experts >= 1
    ), f"At least one expert must be used (it's {num_used_experts})."
    assert (
        num_unused_experts + num_used_experts == G
    ), f"Unused + used experts don't add up total experts ({num_unused_experts} + {num_used_experts} != {G})."

    if num_unused_experts > 0:
        _LOGGER.debug(
            f"Group sizes generation: dropped {num_unused_experts} expert{'s' if num_unused_experts > 1 else ''}.",
        )

    group_sizes = torch.bincount(
        used_experts[
            torch.randint(low=0, high=num_used_experts, size=(num_used_tokens,))
        ],
        minlength=G,
    ).to(group_sizes_dtype)

    assert (
        len(group_sizes) == G
    ), f"Group sizes don't have {G} elements (it's {len(group_sizes)})."
    assert torch.all(group_sizes >= 0).item(), "All group sizes must be non-negative."
    assert (
        torch.sum(group_sizes).item() == num_used_tokens
    ), f"Group sizes don't add up to used tokens {num_used_tokens}."
    assert (
        group_sizes.dtype == group_sizes_dtype
    ), f"Group sizes must be {group_sizes_dtype} (it's {group_sizes.dtype})."

    return group_sizes


def gen_multiple_group_sizes(
    num_group_sizes: int,
    M: int,
    G: int,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    rng_seed: int | None = RNG_SEED,
    unused_tokens_prob: float = UNUSED_TOKENS_PROB,
    unused_experts_prob: float = UNUSED_EXPERTS_PROB,
    group_sizes_0: Tensor | None = None,
) -> list[Tensor]:
    assert (
        num_group_sizes > 0
    ), f"Number of group sizes to be generated must be positive, it's {num_group_sizes}."
    check_group_sizes_dtype(group_sizes_dtype)
    if group_sizes_0 is not None:
        assert group_sizes_0.dtype == group_sizes_dtype, (
            f"group_sizes_0 dtype ({group_sizes_0.dtype}) must match requested "
            f"group_sizes_dtype ({group_sizes_dtype})."
        )
    multiple_group_sizes = [
        gen_group_sizes(
            M,
            G,
            group_sizes_dtype=group_sizes_dtype,
            device=device,
            rng_seed=rng_seed if g == 0 else None,
            unused_tokens_prob=unused_tokens_prob,
            unused_experts_prob=unused_experts_prob,
        )
        for g in range(
            num_group_sizes if group_sizes_0 is None else num_group_sizes - 1
        )
    ]
    if group_sizes_0 is not None:
        multiple_group_sizes.insert(0, group_sizes_0)
    assert (
        len(multiple_group_sizes) == num_group_sizes
    ), f"Expecting {num_group_sizes} distinct group sizes (it's {len(multiple_group_sizes)})."
    return multiple_group_sizes


# GMM helpers: tensor generation.
# ------------------------------------------------------------------------------


def gen_gmm_input(
    M: int,
    K: int,
    N: int,
    G: int,
    preferred_element_type: torch.dtype = DTYPE,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    trans_rhs: bool = TRANS_RHS,
    alt_trans: bool = False,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."
    check_group_sizes_dtype(group_sizes_dtype)

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    lhs = torch.randn((M, K), dtype=torch.float32, device=device)
    lhs = lhs.to(preferred_element_type)

    if trans_rhs:
        # Two physically equivalent transposed layouts are supported. They share the
        # same memory ordering (K varies fastest, then N, then G); only the tensor
        # metadata (shape/stride) differs.
        if alt_trans:
            # Transposed layout 2: shape (G, N, K), stride (K*N, K, 1). The (N, K)
            # sub-matrix per group is row-major.
            rhs = torch.randn((G, N, K), dtype=torch.float32, device=device)
        else:
            # Transposed layout 1: shape (G, K, N), stride (K*N, 1, K). The (K, N)
            # sub-matrix per group is column-major.
            rhs = torch.randn((G, N, K), dtype=torch.float32, device=device).permute(
                0, 2, 1
            )
    else:
        # alt_trans is ignored when trans_rhs is False; only the non-transposed
        # row-major layout is supported in that case.
        rhs = torch.randn((G, K, N), dtype=torch.float32, device=device)
    rhs = rhs.to(preferred_element_type)

    group_sizes = (
        gen_uniform_group_sizes(
            M,
            G,
            group_sizes_dtype=group_sizes_dtype,
            device=device,
        )
        if unif_group_sizes
        else gen_group_sizes(
            M,
            G,
            group_sizes_dtype=group_sizes_dtype,
            device=device,
            rng_seed=None,
        )
    )

    return lhs, rhs, group_sizes


def gen_gmm_output(
    M: int,
    N: int,
    preferred_element_type: torch.dtype = DTYPE,
    device: torch.device | str = DEVICE,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    out = torch.empty((M, N), dtype=preferred_element_type, device=device)

    return out


def gen_gmm_tensors(
    M: int,
    K: int,
    N: int,
    G: int,
    num_group_sizes: int,
    input_type: torch.dtype = DTYPE,
    output_type: torch.dtype = DTYPE,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    trans_lhs: bool = False,
    trans_rhs: bool = TRANS_RHS,
    alt_trans: bool = False,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
    use_bias: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor, Tensor | None]:
    lhs, rhs, group_sizes_0 = gen_gmm_input(
        M,
        K,
        N,
        G,
        preferred_element_type=input_type,
        group_sizes_dtype=group_sizes_dtype,
        device=device,
        trans_rhs=trans_rhs,
        alt_trans=alt_trans,
        rng_seed=rng_seed,
        unif_group_sizes=unif_group_sizes,
    )
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes,
        M,
        G,
        group_sizes_dtype=group_sizes_dtype,
        device=device,
        rng_seed=None,
        group_sizes_0=group_sizes_0,
    )
    out = gen_gmm_output(M, N, preferred_element_type=output_type, device=device)
    bias = None
    if use_bias:
        torch.manual_seed(rng_seed + 1000)  # Different seed for bias
        bias = torch.randn(G, N, dtype=input_type, device=device)

    return lhs, rhs, multiple_group_sizes, out, bias


# GMM helpers: get information from tensors.
# ------------------------------------------------------------------------------


def get_gmm_shape(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 3, f"rhs must have 3 dimensions (it's {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (it's {group_sizes.dim()})."

    M, lhs_k = lhs.shape
    # rhs supports three layouts (see gmm() docstring):
    #   * Non-transposed:        shape (G, K, N), stride (K*N, N, 1).
    #   * Transposed (layout 1): shape (G, K, N), stride (K*N, 1, K).
    #   * Transposed (layout 2): shape (G, N, K), stride (K*N, K, 1).
    # Non-transposed and transposed layout 1 share shape (G, K, N), so K is taken
    # from lhs to disambiguate which dimension of rhs is N.
    rhs_g, rhs_d1, rhs_d2 = rhs.shape
    K = lhs_k
    if rhs_d1 == K:
        # Either non-transposed or transposed layout 1: shape (G, K, N).
        N = rhs_d2
    elif rhs_d2 == K:
        # Transposed layout 2: shape (G, N, K).
        N = rhs_d1
    else:
        raise AssertionError(
            f"rhs shape {tuple(rhs.shape)} doesn't match K = {K} from lhs"
            f" (expected (G, K, N) or (G, N, K))."
        )
    group_sizes_g = group_sizes.shape[0]

    assert (
        rhs_g == group_sizes_g
    ), f"G dimension of rhs and group_sizes don't match (rhs = {rhs_g}, group_sizes = {group_sizes_g})."
    G = rhs_g

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G


def get_gmm_output(
    M: int,
    N: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert M > 0, f"Number of out rows M must be positive (M = {M})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."

    if existing_out is not None:
        assert (
            existing_out.device == device
        ), f"Existing output device and provided device don't match (existing = {existing_out.device}, provided = {device})."
        assert (
            existing_out.dtype == preferred_element_type
        ), f"Existing output type and preferred output type don't match (existing = {existing_out.dtype}, preferred = {preferred_element_type})."
        assert existing_out.shape == (
            M,
            N,
        ), f"Existing output shape and GMM shape don't match (existing = {tuple(existing_out.shape)}, provided = {(M, N)})."
        return existing_out

    return gen_gmm_output(
        M,
        N,
        device=device,
        preferred_element_type=preferred_element_type,
    )


def get_gmm_transposition(lhs: Tensor, rhs: Tensor, out: Tensor) -> tuple[bool, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 3, f"rhs must have 3 dimensions (it's {rhs.dim()})."
    assert out.dim() == 2, f"out must have 2 dimensions (it's {out.dim()})."

    lhs_m, lhs_k = lhs.shape
    out_m, out_n = out.shape

    assert (
        lhs_m == out_m
    ), f"M dimension of lhs and out don't match (lhs = {lhs_m}, out = {out_m})."
    M = lhs_m
    K = lhs_k
    N = out_n

    # Three rhs layouts are accepted (see gmm() docstring):
    #   * Non-transposed:        shape (G, K, N), stride (K*N, N, 1) -> TRANS_RHS=False.
    #   * Transposed (layout 1): shape (G, K, N), stride (K*N, 1, K) -> TRANS_RHS=True.
    #   * Transposed (layout 2): shape (G, N, K), stride (K*N, K, 1) -> TRANS_RHS=True.
    # Both transposed layouts produce identical byte offsets in the kernel's
    # TRANS_RHS branch and therefore execute the same code; the difference is
    # purely metadata.
    G, rhs_d1, rhs_d2 = rhs.shape
    is_kn_shape = (rhs_d1 == K) and (rhs_d2 == N)  # (G, K, N)
    is_nk_shape = (rhs_d1 == N) and (rhs_d2 == K)  # (G, N, K)
    assert is_kn_shape or is_nk_shape, (
        f"rhs shape {tuple(rhs.shape)} must be (G, K, N) = ({G}, {K}, {N}) or "
        f"(G, N, K) = ({G}, {N}, {K})."
    )

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    is_lhs_row_major = lhs.stride() == (K, 1)
    assert is_lhs_row_major, "lhs must be row-major."

    rhs_stride = rhs.stride()
    is_rhs_not_transposed = is_kn_shape and rhs_stride == (K * N, N, 1)
    is_rhs_transposed_layout_1 = is_kn_shape and rhs_stride == (K * N, 1, K)
    is_rhs_transposed_layout_2 = is_nk_shape and rhs_stride == (K * N, K, 1)
    num_matches = (
        int(is_rhs_not_transposed)
        + int(is_rhs_transposed_layout_1)
        + int(is_rhs_transposed_layout_2)
    )
    # When K == N, shape (G, K, N) and (G, N, K) are indistinguishable, and so are
    # the strides for non-transposed and transposed layout 2: (K*N, N, 1) and
    # (K*N, K, 1) collapse to the same tuple. The two interpretations correspond
    # to different mathematical operations (see TRANS_RHS branches in the kernel),
    # so we cannot disambiguate from shape+stride alone in that case. Transposed
    # layout 1 stays unambiguous because its stride pattern (K*N, 1, K) differs.
    assert num_matches == 1, (
        "rhs must match exactly one supported layout: "
        "non-transposed (shape (G, K, N), stride (K*N, N, 1)), "
        "transposed layout 1 (shape (G, K, N), stride (K*N, 1, K)), "
        "or transposed layout 2 (shape (G, N, K), stride (K*N, K, 1)). "
        f"Got shape {tuple(rhs.shape)}, stride {rhs_stride}."
        + (
            " Note: K == N makes non-transposed and transposed layout 2 ambiguous."
            if K == N
            else ""
        )
    )
    is_out_row_major = out.stride() == (N, 1)
    assert is_out_row_major, "out must be row-major."

    is_rhs_transposed = is_rhs_transposed_layout_1 or is_rhs_transposed_layout_2
    # Get rhs leading dimension according to transposition configuration. Both
    # transposed layouts share the same leading dimension because they have the
    # same physical memory ordering.
    ld_rhs = N if is_rhs_not_transposed else K

    return is_rhs_transposed, ld_rhs


# TGMM helpers: tensor generation.
# ------------------------------------------------------------------------------


def gen_tgmm_input(
    M: int,
    K: int,
    N: int,
    G: int,
    preferred_element_type: torch.dtype = DTYPE,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    trans_lhs: bool = TRANS_LHS,
    alt_trans: bool = False,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    assert K > 0, f"Number of lhs rows K must be positive (M = {K})."
    assert M > 0, f"Number of lhs columns / rhs rows M must be positive (K = {M})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."
    check_group_sizes_dtype(group_sizes_dtype)

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    if trans_lhs:
        # Two physically equivalent transposed layouts are supported. They share the
        # same memory ordering (K varies fastest, then M); only the tensor metadata
        # (shape/stride) differs.
        if alt_trans:
            # Transposed layout 2: shape (M, K), stride (K, 1). lhs is row-major over
            # the swapped shape.
            lhs = torch.randn((M, K), dtype=torch.float32, device=device)
        else:
            # Transposed layout 1: shape (K, M), stride (1, K). lhs is column-major.
            lhs = torch.randn((M, K), dtype=torch.float32, device=device).T
    else:
        # alt_trans is ignored when trans_lhs is False; only the non-transposed
        # row-major layout is supported in that case.
        lhs = torch.randn((K, M), dtype=torch.float32, device=device)
    lhs = lhs.to(preferred_element_type)

    rhs = torch.randn((M, N), dtype=torch.float32, device=device)
    rhs = rhs.to(preferred_element_type)

    group_sizes = (
        gen_uniform_group_sizes(
            M,
            G,
            group_sizes_dtype=group_sizes_dtype,
            device=device,
        )
        if unif_group_sizes
        else gen_group_sizes(
            M,
            G,
            group_sizes_dtype=group_sizes_dtype,
            device=device,
            rng_seed=None,
        )
    )

    return lhs, rhs, group_sizes


def gen_tgmm_output(
    K: int,
    N: int,
    G: int,
    preferred_element_type: torch.dtype = DTYPE,
    device: torch.device | str = DEVICE,
) -> Tensor:
    assert K > 0, f"Number of out rows K must be positive (K = {K})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    out = torch.empty((G, K, N), dtype=preferred_element_type, device=device)

    return out


def gen_tgmm_bias_grad(
    K: int,
    G: int,
    device: torch.device | str = DEVICE,
    with_bias_grad: bool = False,
) -> Tensor:
    if with_bias_grad:
        assert K > 0, f"Number of bias_grad rows K must be positive (K = {K})."
        assert G > 0, f"Number of groups G must be positive (G = {G})."
        return torch.empty((G, K), device=device, dtype=torch.float32)
    else:
        # Return dummy pointer when bias_grad is not needed.
        # Must be float32 because atomic_add does not support bf16/fp16,
        # and Triton validates the pointer dtype even in dead branches.
        return torch.tensor([], device=device, dtype=torch.float32)


def gen_tgmm_tensors(
    M: int,
    K: int,
    N: int,
    G: int,
    num_group_sizes: int,
    input_type: torch.dtype = DTYPE,
    output_type: torch.dtype = DTYPE,
    group_sizes_dtype: torch.dtype = GROUP_SIZES_DTYPE,
    device: torch.device | str = DEVICE,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = False,
    alt_trans: bool = False,
    rng_seed: int | None = RNG_SEED,
    unif_group_sizes: bool = False,
    use_bias: bool = False,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor, Tensor | None]:
    lhs, rhs, group_sizes_0 = gen_tgmm_input(
        M,
        K,
        N,
        G,
        preferred_element_type=input_type,
        group_sizes_dtype=group_sizes_dtype,
        device=device,
        trans_lhs=trans_lhs,
        alt_trans=alt_trans,
        rng_seed=rng_seed,
        unif_group_sizes=unif_group_sizes,
    )
    multiple_group_sizes = gen_multiple_group_sizes(
        num_group_sizes,
        M,
        G,
        group_sizes_dtype=group_sizes_dtype,
        device=device,
        rng_seed=None,
        group_sizes_0=group_sizes_0,
    )
    out = gen_tgmm_output(K, N, G, preferred_element_type=output_type, device=device)
    if use_bias:
        bias_grad = gen_tgmm_bias_grad(K, G, device=device, with_bias_grad=True)
    else:
        bias_grad = None
    return lhs, rhs, multiple_group_sizes, out, bias_grad


# TGMM helpers: get information from tensors.
# ------------------------------------------------------------------------------


def get_tgmm_shape(
    lhs: Tensor, rhs: Tensor, group_sizes: Tensor
) -> tuple[int, int, int, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 2, f"rhs must have 2 dimensions (it's {rhs.dim()})."
    assert (
        group_sizes.dim() == 1
    ), f"group_sizes must have 1 dimension (it's {group_sizes.dim()})."

    rhs_m, N = rhs.shape
    M = rhs_m
    G = group_sizes.shape[0]

    # lhs supports three layouts (see ptgmm() / nptgmm() docstring):
    #   * Non-transposed:        shape (K, M), stride (M, 1).
    #   * Transposed (layout 1): shape (K, M), stride (1, K).
    #   * Transposed (layout 2): shape (M, K), stride (K, 1).
    # Non-transposed and transposed layout 1 share shape (K, M), so M is taken
    # from rhs to disambiguate which dimension of lhs is K.
    lhs_d1, lhs_d2 = lhs.shape
    if lhs_d2 == M:
        # Either non-transposed or transposed layout 1: shape (K, M).
        K = lhs_d1
    elif lhs_d1 == M:
        # Transposed layout 2: shape (M, K).
        K = lhs_d2
    else:
        raise AssertionError(
            f"lhs shape {tuple(lhs.shape)} doesn't match M = {M} from rhs"
            f" (expected (K, M) or (M, K))."
        )

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    return M, K, N, G


def get_tgmm_output(
    K: int,
    N: int,
    G: int,
    device: torch.device | str = DEVICE,
    preferred_element_type: torch.dtype = DTYPE,
    existing_out: Tensor | None = None,
) -> Tensor:
    assert K > 0, f"Number of out rows K must be positive (K = {K})."
    assert N > 0, f"Number of out columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if existing_out is not None:
        assert (
            existing_out.device == device
        ), f"Existing output device and provided device don't match (existing = {existing_out.device}, provided = {device})."
        assert (
            existing_out.dtype == preferred_element_type
        ), f"Existing output type and preferred output type don't match (existing = {existing_out.dtype}, preferred = {preferred_element_type})."
        assert existing_out.shape == (
            G,
            K,
            N,
        ), f"Existing output shape and GMM shape don't match (existing = {tuple(existing_out.shape)}, provided = {(G, K, N)})."
        return existing_out

    return gen_tgmm_output(
        K,
        N,
        G,
        device=device,
        preferred_element_type=preferred_element_type,
    )


def get_tgmm_bias_grad(
    K: int,
    G: int,
    device: torch.device | str = DEVICE,
    existing_bias_grad: Tensor | None = None,
) -> Tensor:
    """
    Get or validate bias gradient tensor for TGMM.

    If existing_bias_grad is provided, validates its shape, device, dtype, and stride,
    and always zeros it before returning (since the kernel uses atomic_add).
    If existing_bias_grad is None, returns a dummy tensor (for use when COMPUTE_BIAS_GRAD=False).
    Parameters
    ----------
    K : int
        Number of rows in the bias gradient tensor.
    G : int
        Number of groups.
    device : torch.device or str
        Device for the tensor.
    existing_bias_grad : torch.Tensor or None
        Existing bias gradient tensor to validate and use.
    Returns
    -------
    torch.Tensor
        Valid bias gradient tensor or dummy tensor.
    """
    assert K > 0, f"Number of bias_grad rows K must be positive (K = {K})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."

    if existing_bias_grad is not None:
        # Validate existing bias_grad tensor.
        expected_shape = (G, K)
        assert (
            tuple(existing_bias_grad.shape) == expected_shape
        ), f"bias_grad must have shape {expected_shape}, got {tuple(existing_bias_grad.shape)}."
        assert (
            existing_bias_grad.device == device
        ), f"bias_grad must be on the same device (bias_grad = {existing_bias_grad.device}, device = {device})."
        assert (
            existing_bias_grad.dtype == torch.float32
        ), f"bias_grad must be torch.float32 (kernel uses atomic_add which requires float32), got {existing_bias_grad.dtype}."
        assert existing_bias_grad.stride() == (
            K,
            1,
        ), f"bias_grad must be row-major with stride (K, 1) = ({K}, 1), got {existing_bias_grad.stride()}."

        # Always zero the tensor since bias_grad represents gradients for the current
        # computation and should start fresh. The kernel uses atomic_add which adds to
        # existing values, so we must zero before the kernel runs.
        existing_bias_grad.zero_()

        return existing_bias_grad

    else:
        return gen_tgmm_bias_grad(K, G, device=device, with_bias_grad=False)


def get_tgmm_transposition(lhs: Tensor, rhs: Tensor, out: Tensor) -> tuple[bool, int]:
    assert lhs.dim() == 2, f"lhs must have 2 dimensions (it's {lhs.dim()})."
    assert rhs.dim() == 2, f"rhs must have 2 dimensions (it's {rhs.dim()})."
    assert out.dim() == 3, f"out must have 3 dimensions (it's {out.dim()})."

    rhs_m, rhs_n = rhs.shape
    G, out_k, out_n = out.shape

    assert (
        rhs_n == out_n
    ), f"N dimension of rhs and out don't match (rhs = {rhs_n}, out = {out_n})."
    M = rhs_m
    K = out_k
    N = rhs_n

    # Three lhs layouts are accepted (see ptgmm() / nptgmm() docstring):
    #   * Non-transposed:        shape (K, M), stride (M, 1) -> TRANS_LHS=False.
    #   * Transposed (layout 1): shape (K, M), stride (1, K) -> TRANS_LHS=True.
    #   * Transposed (layout 2): shape (M, K), stride (K, 1) -> TRANS_LHS=True.
    # Both transposed layouts produce identical byte offsets in the kernel's
    # TRANS_LHS branch and therefore execute the same code; the difference is
    # purely metadata.
    lhs_d1, lhs_d2 = lhs.shape
    is_km_shape = (lhs_d1 == K) and (lhs_d2 == M)  # (K, M)
    is_mk_shape = (lhs_d1 == M) and (lhs_d2 == K)  # (M, K)
    assert is_km_shape or is_mk_shape, (
        f"lhs shape {tuple(lhs.shape)} must be (K, M) = ({K}, {M}) or "
        f"(M, K) = ({M}, {K})."
    )

    assert M > 0, f"M must be positive, it's {M}."
    assert K > 0, f"K must be positive, it's {K}."
    assert N > 0, f"N must be positive, it's {N}"
    assert G > 0, f"G must be positive, it's {G}"

    lhs_stride = lhs.stride()
    is_lhs_not_transposed = is_km_shape and lhs_stride == (M, 1)
    is_lhs_transposed_layout_1 = is_km_shape and lhs_stride == (1, K)
    is_lhs_transposed_layout_2 = is_mk_shape and lhs_stride == (K, 1)
    num_matches = (
        int(is_lhs_not_transposed)
        + int(is_lhs_transposed_layout_1)
        + int(is_lhs_transposed_layout_2)
    )
    # When K == M, shape (K, M) and (M, K) are indistinguishable, and so are the
    # strides for non-transposed and transposed layout 2: (M, 1) and (K, 1)
    # collapse to the same tuple. Transposed layout 1 stays unambiguous because
    # its stride pattern (1, K) differs.
    assert num_matches == 1, (
        "lhs must match exactly one supported layout: "
        "non-transposed (shape (K, M), stride (M, 1)), "
        "transposed layout 1 (shape (K, M), stride (1, K)), "
        "or transposed layout 2 (shape (M, K), stride (K, 1)). "
        f"Got shape {tuple(lhs.shape)}, stride {lhs_stride}."
        + (
            " Note: K == M makes non-transposed and transposed layout 2 ambiguous."
            if K == M
            else ""
        )
    )

    is_rhs_row_major = rhs.stride() == (N, 1)
    assert is_rhs_row_major, "rhs must be row-major."
    is_out_row_major = out.stride() == (K * N, N, 1)
    assert is_out_row_major, "out must be row-major."

    is_lhs_transposed = is_lhs_transposed_layout_1 or is_lhs_transposed_layout_2
    # Get lhs leading dimension according to transposition configuration. Both
    # transposed layouts share the same leading dimension because they have the
    # same physical memory ordering.
    ld_lhs = M if is_lhs_not_transposed else K

    return is_lhs_transposed, ld_lhs
