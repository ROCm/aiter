# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 OPUS grouped backward for routed expert projections."""

import torch


def latent_moe_grouped_gemm_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate one grouped linear projection, Y_e = X_e @ W_e.T.

    Rows are already grouped by local expert. For R routed assignments and E
    local experts, x is [R, K], weight is [E, N, K] and dy is [R, N], all
    contiguous BF16. The caller supplies contiguous int32 offsets [E+1] on
    the same device, nondecreasing from 0 to R. Equal offsets mean an empty
    expert. Offset values are a caller precondition to avoid a host sync.

    The outputs are dx [R, K] in BF16 (in the same routed order)
    and dw [E, N, K] in FP32, zero for empty experts. For Kimi gate/up use
    (K, N) = (3584, 3072); for down use (3072, 3584). The caller combines
    the two input gradients for gate/up and undoes the routing permutation.
    """
    if x.ndim != 2 or dy.ndim != 2 or weight.ndim != 3:
        raise ValueError("expected x [R,K], weight [E,N,K], dy [R,N]")
    experts, n, k = weight.shape
    if x.shape[1] != k or dy.shape != (x.shape[0], n):
        raise ValueError("x, weight and dy must agree on R, K and N")
    if expert_offsets.shape != (experts + 1,):
        raise ValueError("expert_offsets must have E+1 entries")
    if not (x.dtype == weight.dtype == dy.dtype == torch.bfloat16):
        raise TypeError("x, weight and dy must be BF16")
    if expert_offsets.dtype != torch.int32:
        raise TypeError("expert_offsets must be int32")
    for tensor in (weight, dy, expert_offsets):
        if tensor.device != x.device:
            raise ValueError("all inputs must share a device")
    if not all(t.is_contiguous() for t in (x, weight, dy, expert_offsets)):
        raise ValueError("all inputs must be contiguous")
    from ..asm.asm_utils import device_guard, get_gfx_from_device

    if (
        x.device.type != "cuda"
        or not torch.version.hip
        or get_gfx_from_device(x.device) != "gfx950"
    ):
        raise NotImplementedError("latent MoE grouped GEMM backward requires gfx950")
    from ._latent_moe_bwd_gfx950 import backward

    with device_guard(x.device.index):
        return backward(x, weight, dy, expert_offsets)
