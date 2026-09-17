# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Public Python wrapper for the scattered-pointer Q4_K_M MoE matvec kernel.
# Designed for streaming-MoE inference where expert weights live in K
# discontiguous device buffers rather than a single contiguous (E, N, K)
# tensor.

import torch
import triton
from aiter.ops.triton.utils.logger import AiterTritonLogger

from aiter.ops.triton._triton_kernels.moe.moe_op_q4k_streaming import (
    _moe_q4k_streaming_kernel,
)

# GGUF Q4_K_M super-block constants.
QK_K = 256
BLOCK_BYTES = 144  # 4 (dm) + 12 (scales) + 128 (qs)


_LOGGER = AiterTritonLogger()


def fused_moe_q4k_streaming(
    A: torch.Tensor,
    expert_ptrs: torch.Tensor,
    remap: torch.Tensor,
    C: torch.Tensor,
    n_dim_out: int,
) -> None:
    """Scattered-pointer fused MoE matvec for GGUF Q4_K_M expert weights.

    Each (token, slot) pair selects one expert via ``remap[token, slot]``,
    which indexes into ``expert_ptrs``. The expert's weight matrix is laid
    out row-major in 144-byte Q4_K_M super-blocks (256 elements / block).

    Stride contract:
        The kernel indexes ``remap`` and ``expert_ptrs`` by linear offset and
        walks the last axis of ``A`` and ``C`` with unit stride. Only the
        strides listed as free below are forwarded to the kernel; the rest
        are required to hold and are asserted here.

        - ``A``: ``stride(1)`` must be 1; ``stride(0)`` is free.
        - ``C``: ``stride(2)`` must be 1; ``stride(0)`` and ``stride(1)`` are
          free.
        - ``remap`` and ``expert_ptrs`` must be contiguous.

        A non-contiguous ``remap`` is the easy mistake here: passing
        ``routing_table[:, :k]``, a slice of a wider top-k table, leaves
        ``stride(0)`` at the full width and would otherwise read the wrong
        experts. Call ``.contiguous()`` on such a slice before passing it.

    Args:
        A (torch.Tensor): Input activations, shape ``(n_tokens, n_dim_in)``,
            ``dtype=torch.float32``. ``n_dim_in`` must be a multiple of 256.
        expert_ptrs (torch.Tensor): Absolute device addresses of each unique
            expert weight blob, shape ``(n_unique_experts,)``,
            ``dtype=torch.uint64``. Each blob is a contiguous byte buffer of
            shape ``(n_dim_out, n_blocks * 144)`` where
            ``n_blocks = n_dim_in / 256``.
        remap (torch.Tensor): Per-(token, slot) routing into ``expert_ptrs``,
            shape ``(n_tokens, n_used_per_token)``, ``dtype=torch.int32``.
        C (torch.Tensor): Output, shape ``(n_tokens, n_used_per_token,
            n_dim_out)``, ``dtype=torch.float32``. Filled in place.
        n_dim_out (int): Number of output rows of each expert weight matrix.

    Returns:
        None. Results are written in-place to ``C``.
    """
    assert A.dtype == torch.float32, "A must be fp32"
    assert C.dtype == torch.float32, "C must be fp32"
    assert expert_ptrs.dtype == torch.uint64, "expert_ptrs must be uint64"
    assert remap.dtype == torch.int32, "remap must be int32"

    n_tokens, n_dim_in = A.shape
    assert n_dim_in % QK_K == 0, "n_dim_in must be a multiple of 256"
    n_used_per_token = remap.shape[1]
    assert remap.shape[0] == n_tokens
    assert C.shape == (n_tokens, n_used_per_token, n_dim_out)

    # See "Stride contract" above. The kernel silently reads the wrong
    # addresses if any of these are violated, so they are checked rather than
    # assumed. Nothing here is made contiguous on the caller's behalf: that
    # would hide a per-dispatch copy in an inference hot path.
    assert A.stride(1) == 1, (
        f"A must have unit stride along n_dim_in, got stride {A.stride()}; "
        "pass a row-contiguous activation buffer"
    )
    assert C.stride(2) == 1, (
        f"C must have unit stride along n_dim_out, got stride {C.stride()}; "
        "pass a row-contiguous output buffer"
    )
    assert remap.is_contiguous(), (
        f"remap must be contiguous, got shape {tuple(remap.shape)} stride "
        f"{remap.stride()}; call .contiguous() (a top-k slice is not)"
    )
    assert (
        expert_ptrs.is_contiguous()
    ), f"expert_ptrs must be contiguous, got stride {expert_ptrs.stride()}"

    _LOGGER.info(
        f"MOE_OP_Q4K_STREAMING: A={tuple(A.shape)} C={tuple(C.shape)} "
        f"n_unique_experts={expert_ptrs.numel()} n_used_per_token={n_used_per_token}"
    )

    # BLOCK_SIZE_N is selected by the kernel's autotuner.
    grid = lambda META: (
        triton.cdiv(n_dim_out, META["BLOCK_SIZE_N"]),
        n_used_per_token,
        n_tokens,
    )
    _moe_q4k_streaming_kernel[grid](
        A,
        expert_ptrs,
        remap,
        C,
        n_dim_in,
        n_dim_out,
        n_used_per_token,
        A.stride(0),
        C.stride(0),
        C.stride(1),
        QK_K=QK_K,
        BLOCK_BYTES=BLOCK_BYTES,
    )
