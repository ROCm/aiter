# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A16W16 exact launch, Torch workspace, and shape-driven compatibility."""

import torch

from csrc.opus_gemm.opus_gemm_common import OpusGemmInstance

from ...jit.core import compile_ops
from ._arch import _device_arch_and_cu
from .launch_plan import _get_cached_a16w16_launch_plan

# ---- Low-level A16W16 backend --------------------------------------------


def _gen_opus_gemm_a16w16_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_launch",
    gen_fake=_gen_opus_gemm_a16w16_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> torch.Tensor: ...


def _launch_a16w16_backend(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    if (
        torch.compiler.is_compiling()
        or XQ.is_meta
        or getattr(XQ, "fake_mode", None) is not None
    ):
        _opus_gemm_a16w16_launch_raw(
            XQ,
            WQ,
            Y,
            bias,
            workspace,
            kid,
            split_k,
        )
        return
    with torch.cuda.device(XQ.device):
        _opus_gemm_a16w16_launch_raw(
            XQ,
            WQ,
            Y,
            bias,
            workspace,
            kid,
            split_k,
        )


def _check_a16w16_launch_layout(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
) -> None:
    """Validate launcher-required 3D shapes and physical strides."""
    for name, tensor in (("XQ", XQ), ("WQ", WQ), ("Y", Y)):
        if tensor.dim() != 3:
            raise ValueError(
                f"opus_gemm_a16w16_launch: {name} must be 3D "
                f"(got {name}.shape={tuple(tensor.shape)}). "
                "The C++ launcher reads size(0) as batch and uses "
                "hardcoded dense batch strides."
            )

    batch, M, K = XQ.shape
    b_w, N, K_w = WQ.shape
    expected_wq = (batch, N, K)
    expected_y = (batch, M, N)
    if (b_w, K_w) != (batch, K):
        raise ValueError(
            "opus_gemm_a16w16_launch: WQ shape mismatch "
            f"(got {tuple(WQ.shape)}, expected {expected_wq}); "
            f"XQ.shape={tuple(XQ.shape)}"
        )
    if tuple(Y.shape) != expected_y:
        raise ValueError(
            "opus_gemm_a16w16_launch: Y shape mismatch "
            f"(got {tuple(Y.shape)}, expected {expected_y})"
        )

    # XQ/WQ allow padded rows but require contiguous K and dense batches.
    for name, tensor, rows in (("XQ", XQ, M), ("WQ", WQ, N)):
        stride_batch, stride_row, stride_k = tensor.stride()
        if (
            stride_k != 1
            or stride_row < K
            or (batch != 1 and stride_batch != rows * stride_row)
        ):
            raise NotImplementedError(
                f"opus_gemm_a16w16_launch: {name} must be K-contiguous "
                "with an optional padded leading dimension; need "
                "stride[2]==1, stride[1]>=K, and "
                "stride[0]==size(1)*stride[1] when batch>1. "
                f"Got {name}.stride()={tuple(tensor.stride())}, "
                f"{name}.shape={tuple(tensor.shape)}. "
                f"Materialize with `{name} = {name}.contiguous()`."
            )

    # Y must match the launcher's contiguous output strides.
    expected_y_stride = (M * N, N, 1)
    if Y.stride() != expected_y_stride:
        raise NotImplementedError(
            "opus_gemm_a16w16_launch: Y must have contiguous strides "
            f"{expected_y_stride} (got {tuple(Y.stride())}, "
            f"Y.shape={tuple(Y.shape)}). "
            "Materialize with `Y = Y.contiguous()`."
        )


def _execute_a16w16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: torch.Tensor | None = None,
    route_arch: str | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Validate, plan, and launch one exact 3D A16W16 operation."""
    _check_a16w16_launch_layout(XQ, WQ, Y)
    batch, M, K = XQ.shape
    N = Y.shape[2]

    use_gfx950_caller_workspace_fast_path = (
        route_arch == "gfx950"
        and workspace is not None
        and split_k > 0
        and instance is not None
        and instance.splitk_workspace_dtype is not None
    )
    if use_gfx950_caller_workspace_fast_path:
        # A caller-owned gfx950 workspace and the public registry route avoid
        # re-reading device metadata. Explicit gfx950 plans do not consult the
        # CU count, so one is a safe cache-key placeholder here.
        arch, cu_num = route_arch, 1
    else:
        arch, cu_num = _device_arch_and_cu(XQ.device)

    plan = _get_cached_a16w16_launch_plan(
        arch,
        M,
        N,
        K,
        batch,
        cu_num,
        bias is not None,
        XQ.dtype,
        Y.dtype,
        int(kid),
        int(split_k),
    )
    workspace_spec = plan.workspace_spec
    if use_gfx950_caller_workspace_fast_path and workspace_spec is None:
        raise RuntimeError(
            f"OPUS gfx950 kid {plan.resolved_kid} unexpectedly has no "
            "caller-workspace plan"
        )
    if workspace_spec is None:
        if workspace is not None:
            raise ValueError(
                "opus_gemm_a16w16_launch: "
                f"kid {plan.resolved_kid} does not use an external workspace"
            )
    elif workspace is None:
        workspace = torch.empty(
            workspace_spec.shape,
            dtype=workspace_spec.dtype,
            device=XQ.device,
        )

    _launch_a16w16_backend(
        XQ,
        WQ,
        Y,
        bias,
        workspace,
        plan.resolved_kid,
        plan.abi_split_k,
    )
    return Y


def _launch_a16w16_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: torch.Tensor | None = None,
    route_arch: str | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` A16W16 GEMM."""
    if instance is None and (XQ.dim() != 2 or WQ.dim() != 2 or Y.dim() != 2):
        raise ValueError(
            "opus_gemm A16W16 expects 2D XQ/WQ/Y; use opus_bmm for "
            "batch-first 3D tensors"
        )
    _execute_a16w16(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        bias,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        route_arch=route_arch,
        instance=instance,
    )
    return Y


def _launch_a16w16_bmm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: torch.Tensor | None = None,
    route_arch: str | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` A16W16 BMM."""
    if instance is None and (XQ.dim() != 3 or WQ.dim() != 3 or Y.dim() != 3):
        raise ValueError(
            "opus_bmm A16W16 expects batch-first 3D XQ/WQ/Y; use "
            "opus_gemm for logical 2D tensors"
        )
    return _execute_a16w16(
        XQ,
        WQ,
        Y,
        bias,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        route_arch=route_arch,
        instance=instance,
    )


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    """Launch the legacy A16W16 GEMM interface by exact kernel id."""
    return _execute_a16w16(
        XQ,
        WQ,
        Y,
        kid=int(kernelId),
        split_k=int(splitK),
    )


def _prepare_shape_driven_a16w16(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Normalize the legacy 2D/3D caller contract to batch-first tensors."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("gemm_a16w16_opus requires Tensor A and B")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise NotImplementedError(
            "gemm_a16w16_opus only supports bf16 A/B "
            f"(got A.dtype={A.dtype}, B.dtype={B.dtype})"
        )
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "gemm_a16w16_opus only supports bf16/fp32 output dtype, "
            f"got {output_dtype}"
        )
    if A.device != B.device:
        raise ValueError(
            f"gemm_a16w16_opus requires A/B on one device; got {A.device}/{B.device}"
        )

    is_gemm = A.dim() == 2
    if is_gemm:
        M, K = map(int, A.shape)
        batch = 1
    elif A.dim() == 3:
        batch, M, K = map(int, A.shape)
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {tuple(A.shape)}")

    if B.dim() == 2:
        N, K_b = map(int, B.shape)
        if batch != 1:
            raise NotImplementedError(
                "gemm_a16w16_opus requires a real 3D [batch,N,K] weight "
                f"when A is batched; got A.shape={tuple(A.shape)}, "
                f"B.shape={tuple(B.shape)}"
            )
    elif B.dim() == 3:
        b_b, N, K_b = map(int, B.shape)
        if b_b != batch:
            raise ValueError(f"B batch mismatch: expected {batch}, got {b_b}")
    else:
        raise ValueError(
            f"B must be 2D [N,K] or 3D [batch,N,K], got shape {tuple(B.shape)}"
        )
    if K_b != K:
        raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")

    Y = out
    if Y is None:
        Y = torch.empty((batch, M, N), dtype=output_dtype, device=A.device)
    elif not isinstance(Y, torch.Tensor):
        raise TypeError(f"gemm_a16w16_opus out must be a Tensor, got {type(Y)!r}")
    elif Y.device != A.device or Y.dtype != output_dtype:
        raise ValueError(
            "gemm_a16w16_opus out must match A.device and dtype; "
            f"got {Y.device}/{Y.dtype}, expected {A.device}/{output_dtype}"
        )

    XQ = A.unsqueeze(0) if is_gemm else A
    WQ = B.unsqueeze(0) if B.dim() == 2 else B
    _check_a16w16_launch_layout(XQ, WQ, Y)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("gemm_a16w16_opus bias must be a Tensor")
        if bias.device != A.device or bias.dtype not in (output_dtype, torch.float32):
            raise ValueError(
                "gemm_a16w16_opus bias must be on A.device and use fp32 or "
                f"the output dtype; got {bias.device}/{bias.dtype}"
            )
        if not bias.is_contiguous():
            raise ValueError("gemm_a16w16_opus bias must be contiguous")
        if tuple(bias.shape) not in ((N,), (batch, N)):
            raise ValueError(
                f"gemm_a16w16_opus bias must have shape [{N}] or [{batch},{N}], "
                f"got shape {tuple(bias.shape)}"
            )

    return XQ, WQ, Y, is_gemm


def gemm_a16w16_opus(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    splitK: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shape-driven A16W16 compatibility API over the exact-kid launchers.

    Explicit ``kernelId`` bypasses policy. Otherwise a present OPUS tuned row
    is attempted as-is, while a tuned miss uses the migrated architecture
    heuristic.
    """
    XQ, WQ, Y, is_gemm = _prepare_shape_driven_a16w16(A, B, bias, dtype, out)
    if kernelId is None:
        from .policy import (
            lookup_a16w16_opus_config,
            resolve_a16w16_heuristic_candidate,
        )

        arch, cu_num = _device_arch_and_cu(A.device)
        batch, M, K = map(int, XQ.shape)
        N = int(WQ.shape[1])
        lookup_args = {
            "arch": arch,
            "cu_num": cu_num,
            "M": M,
            "N": N,
            "K": K,
            "has_bias": bias is not None,
            "input_dtype": A.dtype,
            "output_dtype": Y.dtype,
        }
        config = lookup_a16w16_opus_config(**lookup_args)
        if config is not None:
            kid, split_k = int(config["solidx"]), int(config["splitK"])
        else:
            plan = resolve_a16w16_heuristic_candidate(batch=batch, **lookup_args)
            if plan is None:
                raise RuntimeError(
                    "gemm_a16w16_opus found no valid OPUS kernel for "
                    f"arch={arch}, shape=({batch},{M},{N},{K})"
                )
            kid, split_k = plan.resolved_kid, 0
    else:
        kid = int(kernelId)
        split_k = int(splitK or 0)

    if is_gemm:
        return _launch_a16w16_gemm(
            XQ.squeeze(0),
            WQ.squeeze(0),
            Y.squeeze(0),
            kid=kid,
            bias=bias,
            split_k=split_k,
        )

    return _launch_a16w16_bmm(XQ, WQ, Y, kid=kid, bias=bias, split_k=split_k)


__all__ = ["gemm_a16w16_opus", "opus_gemm_a16w16_tune"]
