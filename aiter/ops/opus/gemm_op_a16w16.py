# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A16W16 exact launch and Torch workspace support."""

import ctypes
from functools import lru_cache, wraps
from threading import local

import torch

from ...jit.core import compile_ops, get_module
from ...jit.utils.torch_guard import torch_compile_guard
from ...utility.dtypes import aiter_tensor_t, torch_to_aiter
from csrc.opus_gemm.opus_gemm_common import OpusGemmInstance

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


# Keep pybind for the normal JIT build and A/B.  The small OPUS-local ctypes
# adapter below reuses that mixed module without changing aiter.jit.core.

_OPUS_A16W16_MODULE = "module_deepgemm_opus"
_opus_a16w16_cabi_primed = False
_NULL_AITER_TENSOR = ctypes.POINTER(aiter_tensor_t)()


class _OpusA16DescriptorPool(local):
    def __init__(self) -> None:
        self.xq = aiter_tensor_t()
        self.wq = aiter_tensor_t()
        self.y = aiter_tensor_t()
        self.bias = aiter_tensor_t()
        self.workspace = aiter_tensor_t()


_OPUS_A16_DESCRIPTOR_POOL = _OpusA16DescriptorPool()


@lru_cache(maxsize=1)
def _load_opus_a16w16_cabi():
    """Load the already-built mixed OPUS module's private A16 C ABI."""
    module = get_module(_OPUS_A16W16_MODULE)
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise RuntimeError(
            f"{_OPUS_A16W16_MODULE} has no shared-library path for its C ABI"
        )

    library = ctypes.CDLL(module_path)
    launch = library.opus_gemm_a16w16_launch_cabi
    tensor_ptr = ctypes.POINTER(aiter_tensor_t)
    launch.argtypes = [
        tensor_ptr,
        tensor_ptr,
        tensor_ptr,
        tensor_ptr,
        tensor_ptr,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int

    abi_version = getattr(library, "aiter_ctypes_abi_version", None)
    if abi_version is None:
        raise RuntimeError(
            f"{_OPUS_A16W16_MODULE} does not export aiter_ctypes_abi_version"
        )
    abi_version.argtypes = []
    abi_version.restype = ctypes.c_int
    version = abi_version()
    if version < 2:
        raise RuntimeError(
            f"{_OPUS_A16W16_MODULE} has unsupported ctypes ABI version {version}"
        )

    get_error = library.aiter_get_last_error
    get_error.argtypes = []
    get_error.restype = ctypes.c_char_p
    clear_error = library.aiter_clear_last_error
    clear_error.argtypes = []
    clear_error.restype = None
    return library, launch, get_error, clear_error


def _invoke_opus_a16w16_cabi(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    _library, launch, get_error, clear_error = _load_opus_a16w16_cabi()
    pool = _OPUS_A16_DESCRIPTOR_POOL
    xq_descriptor = torch_to_aiter(XQ, pool.xq)
    wq_descriptor = torch_to_aiter(WQ, pool.wq)
    y_descriptor = torch_to_aiter(Y, pool.y)
    bias_descriptor = None if bias is None else torch_to_aiter(bias, pool.bias)
    workspace_descriptor = (
        None
        if workspace is None
        else torch_to_aiter(workspace, pool.workspace)
    )
    bias_arg = (
        _NULL_AITER_TENSOR
        if bias_descriptor is None
        else ctypes.byref(bias_descriptor)
    )
    workspace_arg = (
        _NULL_AITER_TENSOR
        if workspace_descriptor is None
        else ctypes.byref(workspace_descriptor)
    )

    stream = ctypes.c_void_p(torch.cuda.current_stream(XQ.device).cuda_stream)
    status = launch(
        ctypes.byref(xq_descriptor),
        ctypes.byref(wq_descriptor),
        ctypes.byref(y_descriptor),
        bias_arg,
        workspace_arg,
        kid,
        split_k,
        stream,
    )
    if status != 0:
        # aiter_safe_call clears the thread-local error before every launch.
        # Avoid two extra C ABI crossings on the successful hot path and only
        # retrieve the message when the status reports a failure.
        raw_error = get_error()
        message = (
            raw_error.decode(errors="replace")
            if raw_error
            else f"ctypes status={status}"
        )
        clear_error()
        raise RuntimeError(f"opus_gemm_a16w16_launch_cabi failed: {message}")


def _gen_a16w16_backend_fake(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    return None


def _register_a16w16_backend_op(func):
    """Expose the unified private backend as a Torch compile-visible op."""
    guarded = torch_compile_guard(
        device="cuda",
        gen_fake=_gen_a16w16_backend_fake,
    )(func)
    return wraps(func)(guarded)


@_register_a16w16_backend_op
def _launch_a16w16_backend(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    """Launch through the unified pybind-primed, C ABI-backed backend.

    The first call uses the existing pybind wrapper so its normal lazy-build,
    rebuild and architecture checks remain authoritative.  Run that priming
    launch on the input tensor's device: unlike the C ABI below, the pybind
    entry does not install its own device guard.  It then loads the C ABI from
    that same module; subsequent calls use the lower-overhead path.
    """
    global _opus_a16w16_cabi_primed
    if not _opus_a16w16_cabi_primed:
        # torch.cuda.device restores the caller's current device on both the
        # success and exception paths.  This is required when XQ lives on a
        # non-current device, because the generated pybind launcher obtains
        # its launch stream from the current device's TLS state.
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
            _load_opus_a16w16_cabi()
        _opus_a16w16_cabi_primed = True
        return None
    _invoke_opus_a16w16_cabi(XQ, WQ, Y, bias, workspace, kid, split_k)
    return None


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


__all__: list[str] = []
