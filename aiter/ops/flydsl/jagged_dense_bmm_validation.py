# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared input validation for jagged_dense_bmm_broadcast_add (jdbba) wrappers."""

from __future__ import annotations

import torch


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device,
    contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.device.type != "cuda":
        raise ValueError(f"`{name}` must be a CUDA/ROCm tensor, got {tensor.device}")
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"`{name}` must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"`{name}` must be {ndim}D, got shape {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"`{name}` must be contiguous")


def _validate_jdbba_common(
    jagged: torch.Tensor,
    dense: torch.Tensor,
    seq_offsets: torch.Tensor,
    n_groups: int,
) -> tuple[int, int, int]:
    """Reject unsupported jagged/dense/seq_offsets before DLPack / kernel launch."""
    device = jagged.device
    _require_tensor("jagged", jagged, dtype=torch.bfloat16, ndim=2, device=device)
    _require_tensor("dense", dense, dtype=torch.bfloat16, ndim=3, device=device)
    _require_tensor(
        "seq_offsets", seq_offsets, dtype=torch.int32, ndim=1, device=device
    )

    K = dense.shape[1]
    N = dense.shape[2]
    if K != N:
        raise ValueError(
            f"dense must have a square dim (K == N == D); got dense (K={K}, N={N})."
        )
    if dense.shape[0] != n_groups:
        raise ValueError(f"dense.shape[0]={dense.shape[0]} != n_groups={n_groups}.")
    if seq_offsets.numel() != n_groups + 1:
        raise ValueError(
            f"seq_offsets must have n_groups+1={n_groups + 1} entries, got {seq_offsets.numel()}."
        )

    L = int(seq_offsets[-1].item())
    if jagged.shape[0] != L:
        raise ValueError(
            f"`jagged` rows {jagged.shape[0]} != packed L={L} from seq_offsets[-1]"
        )
    if jagged.shape[1] != K:
        raise ValueError(f"`jagged` width {jagged.shape[1]} != K={K}")

    return K, N, L


def validate_jdbba_autograd_inputs(
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    seq_offsets: torch.Tensor,
    n_groups: int,
) -> tuple[int, int, int]:
    """Full autograd entry contract: common tensors plus bias."""
    K, N, L = _validate_jdbba_common(jagged, dense, seq_offsets, n_groups)
    _require_tensor("bias", bias, dtype=torch.bfloat16, ndim=2, device=jagged.device)
    if bias.shape != (n_groups, N):
        raise ValueError(
            f"`bias` must be (n_groups, N)=({n_groups}, {N}); got {tuple(bias.shape)}."
        )
    return K, N, L


def validate_jdbba_bwd_inputs(
    jagged: torch.Tensor,
    dense: torch.Tensor,
    d_out: torch.Tensor,
    seq_offsets: torch.Tensor,
    n_groups: int,
) -> tuple[int, int, int]:
    """Full backward dispatch contract: common tensors plus upstream grad."""
    K, N, L = _validate_jdbba_common(jagged, dense, seq_offsets, n_groups)
    _require_tensor("d_out", d_out, dtype=torch.bfloat16, ndim=2, device=jagged.device)
    if d_out.shape[0] != L:
        raise ValueError(
            f"`d_out` rows {d_out.shape[0]} != packed L={L} from seq_offsets[-1]"
        )
    if d_out.shape[1] != N:
        raise ValueError(f"`d_out` width {d_out.shape[1]} != N={N}")
    return K, N, L
