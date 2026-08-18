# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A16W16 exact launch and Torch workspace support."""

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
from threading import local

import torch

from ...jit.core import compile_ops, get_module
from ...jit.utils.torch_guard import torch_compile_guard
from ...utility.dtypes import aiter_tensor_t, torch_to_aiter
from csrc.opus_gemm.opus_gemm_common import (
    BIAS_AWARE_KIDS,
    GFX942_BF16WS_EXACT_N,
    OpusGemmInstance,
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from . import a16w16_policy as _a16w16_policy
from ._arch import _device_arch_and_cu

_WORKSPACE_DTYPES = {
    "bf16_t": torch.bfloat16,
    "fp32_t": torch.float32,
}
_MAX_AUTO_SPLIT_K = 16
_MIN_ITERS_PER_SPLIT = 2
_EVEN_LOOP_SPLITK_TAGS = frozenset(
    {
        "a16w16_kbuf2v_sk",
        "a16w16_kbuf2v_bk128_sk",
        "a16w16_quad_mfma32_kbuf1_sk",
    }
)


def _validate_a16w16_public_contract(
    *,
    kid: int,
    instance: OpusGemmInstance,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
) -> None:
    """Validate A16W16-only options shared by both public routers."""
    if input_dtype != weight_dtype:
        raise ValueError(
            f"OPUS requires matching XQ/WQ dtypes; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"OPUS kid {kid} requires bf16 XQ/WQ; got {input_dtype}"
        )
    if layout != "plain":
        raise ValueError(
            f"OPUS kid {kid} belongs to family a16w16 and requires "
            f"layout='plain'; got {layout!r}"
        )
    if has_x_scale != has_w_scale:
        raise ValueError("OPUS requires x_scale and w_scale together")
    if has_x_scale:
        raise ValueError("OPUS a16w16 does not accept x_scale/w_scale")
    if not _instance_output_compatible(
        instance,
        needs_workspace=instance.splitk_workspace_dtype is not None,
        output_dtype=output_dtype,
    ):
        raise ValueError(
            f"OPUS kid {kid} does not support Y.dtype={output_dtype}"
        )


@dataclass(frozen=True)
class LaunchConfig:
    """One exact A16W16 kid after split-K resolution."""

    arch: str
    family: str
    actual_kid: int
    allocation_split_k: int
    launch_split_k: int

    @property
    def kid(self) -> int:
        return self.actual_kid

    @property
    def split_k(self) -> int:
        return self.launch_split_k


def _output_dtype_name(dtype) -> str:
    value = str(dtype).lower()
    if value in {"bf16", "bfloat16", "bf16_t", "torch.bfloat16"}:
        return "bf16"
    if value in {"fp32", "float", "float32", "fp32_t", "torch.float32"}:
        return "fp32"
    return value


def _instance_output_compatible(
    instance: OpusGemmInstance,
    *,
    needs_workspace: bool,
    output_dtype,
) -> bool:
    # Workspace reducers cast output; direct kernels use registered dtypes.
    if needs_workspace:
        return _output_dtype_name(output_dtype) in {"bf16", "fp32"}
    return f"{_output_dtype_name(output_dtype)}_t" in instance.output_dtypes


def _instance_shape_compatible(
    instance: OpusGemmInstance,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
) -> bool:
    if arch == "gfx1250" and batch != 1:
        return False

    if instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse":
        split_k = int(instance.fuse_split_k)
        n_cluster = int(instance.fuse_m_cluster)
        if K % 2 != 0 or N % instance.B_N != 0:
            return False
        if split_k < 2 or split_k * n_cluster > 16:
            return False
        num_tiles_n = N // instance.B_N
        if num_tiles_n % n_cluster != 0:
            return False
        return split_k <= (K + instance.B_K - 1) // instance.B_K

    if instance.kernel_tag == "a16w16_mono_tile":
        return N % instance.B_N == 0 and K % instance.B_K == 0

    if not instance.has_oob:
        return M % instance.B_M == 0 and N % instance.B_N == 0
    return True


def _resolve_gfx942_split_k(
    instance: OpusGemmInstance,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    requested: int,
) -> tuple[int, int]:
    """Return allocation/effective split-K matching the gfx942 launcher."""
    if requested > 0:
        allocation = requested
    else:
        tiles_mn = (
            (M + instance.B_M - 1)
            // instance.B_M
            * ((N + instance.B_N - 1) // instance.B_N)
            * batch
        )
        tiles_mn = max(1, tiles_mn)
        target_wg = (2 * cu_num) if instance.kernel_tag.endswith("_p1") else cu_num
        allocation = (target_wg + tiles_mn - 1) // tiles_mn
        allocation = min(_MAX_AUTO_SPLIT_K, max(1, allocation))

    total_iters = (K + instance.B_K - 1) // instance.B_K
    if total_iters < _MIN_ITERS_PER_SPLIT:
        raise ValueError(
            f"K={K} is too small for gfx942 kid B_K={instance.B_K}; "
            f"need at least {instance.B_K * _MIN_ITERS_PER_SPLIT}"
        )

    effective = allocation
    require_even = instance.kernel_tag in _EVEN_LOOP_SPLITK_TAGS
    while effective > 1:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        parity_ok = not require_even or (
            iters_full % 2 == 0 and last_loops % 2 == 0
        )
        if (
            iters_full >= _MIN_ITERS_PER_SPLIT
            and last_loops >= _MIN_ITERS_PER_SPLIT
            and parity_ok
        ):
            break
        effective -= 1

    if require_even:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        if iters_full % 2 != 0 or last_loops % 2 != 0:
            raise ValueError(
                f"gfx942 kid {instance.name} needs even loops per split; "
                f"K={K}, split_k={effective}, loops=({iters_full},{last_loops})"
            )
    return allocation, effective


def _resolve_exact_a16w16_config(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype,
    output_dtype,
    kid: int,
    split_k: int,
) -> LaunchConfig:
    """Validate one caller-provided kid without tuned or heuristic selection."""
    arch = str(arch).lower().split(":", 1)[0]
    M, N, K, batch, cu_num = map(int, (M, N, K, batch, cu_num))
    if min(M, N, K, batch, cu_num) <= 0:
        raise ValueError("M, N, K, batch, and cu_num must all be positive")
    try:
        kid = int(kid)
        split_k = int(split_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"OPUS a16w16 kid/split_k must be integers, got {kid!r}/{split_k!r}"
        ) from exc
    if split_k < 0:
        raise ValueError(
            f"OPUS a16w16 split_k must be non-negative, got {split_k}"
        )
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"OPUS a16w16 requires bf16 XQ/WQ, got input dtype {input_dtype}"
        )

    instance = get_kernel_instance(arch, "a16w16", kid)
    if instance is None:
        raise ValueError(
            f"OPUS kid {kid} is not an a16w16 kernel for runtime arch {arch}"
        )
    needs_workspace = kernel_needs_external_workspace(arch, "a16w16", kid)
    if (
        arch == "gfx942"
        and needs_workspace
        and instance.splitk_workspace_dtype == "bf16_t"
        and N not in GFX942_BF16WS_EXACT_N
    ):
        raise ValueError(
            f"gfx942 exact kid {kid} requires N in "
            f"{sorted(GFX942_BF16WS_EXACT_N)}; got N={N}"
        )
    if not _instance_shape_compatible(
        instance, arch=arch, M=M, N=N, K=K, batch=batch
    ):
        raise ValueError(
            f"OPUS kid {kid} is incompatible with "
            f"shape (batch={batch}, M={M}, N={N}, K={K})"
        )
    if not _instance_output_compatible(
        instance,
        needs_workspace=needs_workspace,
        output_dtype=output_dtype,
    ):
        raise ValueError(
            f"OPUS kid {kid} does not support output dtype {output_dtype}"
        )
    if has_bias and kid not in BIAS_AWARE_KIDS:
        raise ValueError(f"OPUS kid {kid} does not support bias")
    if (
        has_bias
        and instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse"
    ):
        raise ValueError(
            "gfx1250 splitk_fuse has a narrower bf16 [N] bias contract than "
            "the public OPUS interfaces can represent"
        )
    if has_bias and arch == "gfx942" and needs_workspace:
        raise ValueError(
            "the current gfx942 a16w16 launch rejects bias on split-K kernels"
        )

    allocation_split_k = 1
    launch_split_k = split_k
    if needs_workspace:
        allocation_split_k = max(1, split_k)
        if instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse":
            allocation_split_k = int(instance.fuse_split_k)
            launch_split_k = allocation_split_k
        elif arch == "gfx942":
            allocation_split_k, launch_split_k = _resolve_gfx942_split_k(
                instance,
                M=M,
                N=N,
                K=K,
                batch=batch,
                cu_num=cu_num,
                requested=split_k,
            )

    return LaunchConfig(
        arch=arch,
        family="a16w16",
        actual_kid=kid,
        allocation_split_k=allocation_split_k,
        launch_split_k=launch_split_k,
    )


# Temporary private-name compatibility for existing in-tree and external
# callers.  New policy callers should import the explicit APIs from
# a16w16_policy instead of reaching through the execution module.
_heuristic_a16w16_kid_gfx950 = (
    _a16w16_policy._heuristic_a16w16_kid_gfx950
)
_heuristic_a16w16_kid_gfx1250 = (
    _a16w16_policy._heuristic_a16w16_kid_gfx1250
)
_gfx942_heuristic_symbol_to_kid = (
    _a16w16_policy._gfx942_heuristic_symbol_to_kid
)
_gfx942_heuristic_kid_for_symbol = (
    _a16w16_policy._gfx942_heuristic_kid_for_symbol
)
_gfx942_heuristic_split_barrier_ok = (
    _a16w16_policy._gfx942_heuristic_split_barrier_ok
)
_gfx942_heuristic_bf16ws_band = (
    _a16w16_policy._gfx942_heuristic_bf16ws_band
)
_gfx942_heuristic_bf16_symbol = (
    _a16w16_policy._gfx942_heuristic_bf16_symbol
)
_heuristic_a16w16_kid_gfx942 = (
    _a16w16_policy._heuristic_a16w16_kid_gfx942
)
_A16W16_HEURISTICS = _a16w16_policy._A16W16_HEURISTICS
_select_a16w16_heuristic_kid = _a16w16_policy.select_a16w16_heuristic_kid
_resolve_a16w16_caller_candidate = (
    _a16w16_policy.resolve_a16w16_caller_candidate
)

# ---- Low-level bindings ---------------------------------------------------


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


def _gen_opus_gemm_a16w16_cabi_fake(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    return None


def _opus_gemm_a16w16_launch_ctypes_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    """Launch through the private C ABI without modifying generic JIT code.

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


_opus_gemm_a16w16_launch_ctypes_raw = wraps(
    _opus_gemm_a16w16_launch_ctypes_raw
)(
    torch_compile_guard(
        device="cuda",
        calling_func_=_opus_gemm_a16w16_launch_ctypes_raw,
        gen_fake=_gen_opus_gemm_a16w16_cabi_fake,
    )(_opus_gemm_a16w16_launch_ctypes_raw)
)


def _raise_a16w16_bad_dim(name: str, tensor: torch.Tensor) -> None:
    raise ValueError(
        f"opus_gemm_a16w16_launch: {name} must be 3D (got "
        f"{name}.shape={tuple(tensor.shape)}). The C++ launcher reads "
        f"`{name}.size(0)` as batch and indexes with hardcoded "
        f"stride_*_batch == size(1)*size(2)."
    )


def _raise_a16w16_bad_input_stride(
    name: str, tensor: torch.Tensor, inner_k: int
) -> None:
    raise NotImplementedError(
        f"opus_gemm_a16w16_launch: {name} must be K-contiguous with an "
        f"optional padded leading dim -- need stride[2]==1, "
        f"stride[1]>={inner_k}, and stride[0]==size(1)*stride[1] (or "
        f"batch==1). Got {name}.stride()={tuple(tensor.stride())}, "
        f"{name}.shape={tuple(tensor.shape)}. Broadcast / transpose / "
        f"non-K-contiguous slices are not supported; materialize with "
        f"`{name} = {name}.contiguous()` before calling."
    )


def _check_a16w16_launch_layout(XQ: torch.Tensor, WQ: torch.Tensor, Y: torch.Tensor):
    """Validate launcher-required 3D shapes and strides."""
    if XQ.dim() != 3:
        _raise_a16w16_bad_dim("XQ", XQ)
    if WQ.dim() != 3:
        _raise_a16w16_bad_dim("WQ", WQ)
    if Y.dim() != 3:
        _raise_a16w16_bad_dim("Y", Y)

    batch, M, K = XQ.shape
    b_w, N, K_w = WQ.shape
    b_y, M_y, N_y = Y.shape
    if (b_w, K_w) != (batch, K):
        raise ValueError(
            f"opus_gemm_a16w16_launch: WQ shape mismatch (got "
            f"WQ.shape={tuple(WQ.shape)}, expected "
            f"({batch}, N, {K})); XQ.shape={tuple(XQ.shape)}"
        )
    if (b_y, M_y, N_y) != (batch, M, N):
        raise ValueError(
            f"opus_gemm_a16w16_launch: Y shape mismatch (got "
            f"Y.shape={tuple(Y.shape)}, expected ({batch}, {M}, {N}))"
        )

    # XQ/WQ allow padded rows but require contiguous K and dense batches.
    x_stride = XQ.stride()
    if (
        x_stride[2] != 1
        or x_stride[1] < K
        or (batch != 1 and x_stride[0] != M * x_stride[1])
    ):
        _raise_a16w16_bad_input_stride("XQ", XQ, K)
    w_stride = WQ.stride()
    if (
        w_stride[2] != 1
        or w_stride[1] < K
        or (batch != 1 and w_stride[0] != N * w_stride[1])
    ):
        _raise_a16w16_bad_input_stride("WQ", WQ, K)
    # Y must match the launcher's contiguous output strides.
    y_want = (M * N, N, 1)
    if Y.stride() != y_want:
        raise NotImplementedError(
            f"opus_gemm_a16w16_launch: Y must have contiguous strides {y_want} "
            f"(got Y.stride()={tuple(Y.stride())}, Y.shape={tuple(Y.shape)}). "
            f"The launcher hardcodes stride_c == N and stride_c_batch == M*N; "
            f"materialize with `Y = Y.contiguous()` before calling."
        )


def _plan_a16w16_workspace(
    config: LaunchConfig,
    *,
    batch: int,
    M: int,
    N: int,
    K: int,
) -> tuple[tuple[int, ...], torch.dtype] | None:
    """Validate and return the immutable workspace shape/dtype plan."""
    instance = get_kernel_instance(config.arch, config.family, config.actual_kid)
    if instance is None:
        raise RuntimeError(
            "opus_gemm_a16w16_launch: resolved launch has no canonical "
            "instance: "
            f"({config.arch}, {config.family}, {config.actual_kid})"
        )

    if not kernel_needs_external_workspace(
        config.arch, config.family, config.actual_kid
    ):
        return None

    is_fused = (
        instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse"
    )
    # Fused split-K comes from the kernel entry; other kernels use exact-id resolution.
    split_k = (
        int(instance.fuse_split_k)
        if is_fused
        else int(config.allocation_split_k)
    )
    if split_k <= 0:
        raise ValueError(
            "opus_gemm_a16w16_launch: allocation split_k must be positive, "
            f"got {split_k}"
        )

    block_m = int(instance.B_M)
    block_n = int(instance.B_N)
    block_k = int(instance.B_K)
    max_useful_split_k = (K + block_k - 1) // block_k
    if split_k > max_useful_split_k:
        raise ValueError(
            "opus_gemm_a16w16_launch: "
            f"allocation split_k={split_k} exceeds the per-kid K-tile limit "
            f"{max_useful_split_k} for K={K}, B_K={block_k}"
        )

    if config.arch == "gfx1250":
        if batch != 1:
            raise ValueError(
                "opus_gemm_a16w16_launch: gfx1250 workspace kids require "
                "batch=1; "
                f"got batch={batch}"
            )
        num_tiles_m = (M + block_m - 1) // block_m
        num_tiles_n = (N + block_n - 1) // block_n
        if is_fused:
            if split_k < 2:
                raise ValueError(
                    "opus_gemm_a16w16_launch: "
                    f"gfx1250 fused kid {config.actual_kid} must declare "
                    f"compile-time SplitK >= 2, got {split_k}"
                )
            # Fused layout: M tile, N tile, published partial,
            # M element, N element.
            shape = (
                num_tiles_m,
                num_tiles_n,
                split_k - 1,
                block_m,
                block_n,
            )
        else:
            padded_m = num_tiles_m * block_m
            padded_n = num_tiles_n * block_n
            shape = (split_k, padded_m, padded_n)
    else:
        padded_m = ((M + block_m - 1) // block_m) * block_m
        padded_n = ((N + block_n - 1) // block_n) * block_n
        shape = (split_k, batch, padded_m, padded_n)

    dtype_token = instance.splitk_workspace_dtype
    try:
        dtype = _WORKSPACE_DTYPES[dtype_token]
    except KeyError as exc:
        raise ValueError(
            "opus_gemm_a16w16_launch: "
            f"workspace kid {config.actual_kid} must declare "
            f"bf16_t or fp32_t storage, got {dtype_token!r}"
        ) from exc

    required_numel = 1
    max_numel = (2**63 - 1) // int(dtype.itemsize)
    for extent in shape:
        if extent <= 0 or required_numel > max_numel // extent:
            raise OverflowError(
                "opus_gemm_a16w16_launch: "
                f"workspace shape {shape} exceeds the supported tensor size "
                f"for dtype {dtype}"
            )
        required_numel *= extent

    return shape, dtype


def _init_a16w16_workspace(
    config: LaunchConfig,
    XQ: torch.Tensor,
    Y: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Prepare workspace for ``config.actual_kid``."""
    batch, M, K = map(int, XQ.shape)
    N = int(Y.shape[2])
    plan = _plan_a16w16_workspace(
        config,
        batch=batch,
        M=M,
        N=N,
        K=K,
    )
    if plan is None:
        if workspace is not None:
            raise ValueError(
                f"opus_gemm_a16w16_launch: kid {config.actual_kid} does not use an "
                "external workspace"
            )
        return None

    if workspace is not None:
        return workspace
    shape, dtype = plan
    return torch.empty(shape, dtype=dtype, device=XQ.device)


@lru_cache(maxsize=256)
def _cached_explicit_a16w16_plan(
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    kid: int,
    split_k: int,
) -> tuple[int, int, tuple[tuple[int, ...], torch.dtype] | None]:
    """Cache only immutable scalar exact-kid and workspace-plan results."""
    config = _resolve_exact_a16w16_config(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        kid=kid,
        split_k=split_k,
    )
    workspace_plan = _plan_a16w16_workspace(
        config,
        batch=batch,
        M=M,
        N=N,
        K=K,
    )
    return int(config.actual_kid), int(config.launch_split_k), workspace_plan


def _launch_a16w16_with_torch_workspace(
    raw_launch: Callable[..., object],
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    config: LaunchConfig,
    *,
    workspace: torch.Tensor | None = None,
    _layout_checked: bool = False,
) -> torch.Tensor:
    """Prepare workspace and launch the resolved kid."""
    if not _layout_checked:
        _check_a16w16_launch_layout(XQ, WQ, Y)
    workspace = _init_a16w16_workspace(config, XQ, Y, workspace)
    raw_launch(
        XQ,
        WQ,
        Y,
        bias,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )
    return Y


def _explicit_a16w16_launch(
    raw_launch: Callable[..., object],
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    kid: int,
    split_k: int,
    *,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve and launch an explicit kid."""
    _check_a16w16_launch_layout(XQ, WQ, Y)
    batch, M, K = XQ.shape
    N = Y.shape[2]
    arch, cu_num = _device_arch_and_cu(XQ.device)
    actual_kid, launch_split_k, workspace_plan = _cached_explicit_a16w16_plan(
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
    if workspace_plan is None:
        if workspace is not None:
            raise ValueError(
                f"opus_gemm_a16w16_launch: kid {actual_kid} does not use an "
                "external workspace"
            )
    elif workspace is None:
        shape, workspace_dtype = workspace_plan
        workspace = torch.empty(shape, dtype=workspace_dtype, device=XQ.device)

    raw_launch(
        XQ,
        WQ,
        Y,
        bias,
        workspace,
        actual_kid,
        launch_split_k,
    )
    return Y


def _launch_a16w16_exact(
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
    """Launch one exact A16W16 kid shared by the GEMM and BMM adapters."""
    if (
        route_arch == "gfx950"
        and workspace is not None
        and split_k > 0
        and instance is not None
        and instance.splitk_workspace_dtype is not None
    ):
        # A caller-owned gfx950 workspace already gives this exact-kid path
        # everything it needs.  Preserve the checked C ABI fast path without
        # re-reading device metadata or re-entering the generic planner.
        _check_a16w16_launch_layout(XQ, WQ, Y)
        batch, M, K = XQ.shape
        N = Y.shape[2]
        actual_kid, launch_split_k, workspace_plan = (
            _cached_explicit_a16w16_plan(
                route_arch,
                M,
                N,
                K,
                batch,
                1,  # Explicit gfx950 plans do not consult the CU count.
                bias is not None,
                XQ.dtype,
                Y.dtype,
                kid,
                split_k,
            )
        )
        if workspace_plan is None:
            raise RuntimeError(
                f"OPUS gfx950 kid {actual_kid} unexpectedly has no "
                "caller-workspace plan"
            )
        _opus_gemm_a16w16_launch_ctypes_raw(
            XQ,
            WQ,
            Y,
            bias,
            workspace,
            actual_kid,
            launch_split_k,
        )
        return Y
    return _explicit_a16w16_launch(
        _opus_gemm_a16w16_launch_ctypes_raw,
        XQ,
        WQ,
        Y,
        bias,
        kid,
        split_k,
        workspace=workspace,
    )


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
    _launch_a16w16_exact(
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
    return _launch_a16w16_exact(
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
