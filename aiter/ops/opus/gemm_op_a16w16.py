# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private A16W16 caller policy, exact launch, and Torch workspace support."""

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, wraps

import torch

from ...jit.core import compile_ops, get_module
from ...jit.utils.torch_guard import torch_compile_guard
from ...utility.dtypes import aiter_tensor_t, torch_to_aiter
from csrc.opus_gemm.opus_gemm_common import (
    BIAS_AWARE_KIDS,
    DEFAULT_COMPILED_KIDS_BY_ARCH,
    GFX942_BF16WS_EXACT_N,
    OpusGemmInstance,
    get_kernel_instance,
    kernel_needs_external_workspace,
)

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
    """Validate A16W16-only options for the unified public router."""
    if input_dtype != weight_dtype:
        raise ValueError(
            f"opus_gemm requires matching XQ/WQ dtypes; got "
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
        raise ValueError("opus_gemm requires x_scale and w_scale together")
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
            "the unified interface can represent"
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


def _heuristic_a16w16_kid_gfx950(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx950 no-tuned-row fallback kid."""
    del batch, output_dtype
    M, N, K = map(int, (M, N, K))
    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0

    if M <= 4:
        if M % 64 == 0 and N % 64 == 0 and K % 128 == 0:
            return 1208
        return 208
    if M <= 64:
        if M % 64 == 0 and N % 32 == 0 and K % 128 == 0:
            return 1206
        return 206
    if M <= 128:
        if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
            return 1200
        return 200
    if split_barrier_ok and not has_bias:
        if M % 256 == 0 and N % 256 == 0 and K % 64 == 0:
            return 1300
        return 300
    if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
        return 1200
    return 200


def _heuristic_a16w16_kid_gfx1250(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx1250 no-tuned-row fallback kid."""
    del K, batch, has_bias, output_dtype
    M, N = map(int, (M, N))
    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005
    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


@lru_cache(maxsize=1)
def _gfx942_heuristic_symbol_to_kid() -> dict[str, int]:
    """Build the canonical gfx942 launcher-symbol mapping from the registry."""
    result: dict[str, int] = {}
    for kid in DEFAULT_COMPILED_KIDS_BY_ARCH["gfx942"]:
        instance = get_kernel_instance("gfx942", "a16w16", kid)
        if instance is None:
            raise RuntimeError(
                f"gfx942 heuristic kid {kid} has no a16w16 instance"
            )
        previous = result.setdefault(instance.name, int(kid))
        if previous != kid:
            raise RuntimeError(
                f"duplicate gfx942 launcher symbol {instance.name!r}: "
                f"kids {previous} and {kid}"
            )
    return result


def _gfx942_heuristic_kid_for_symbol(symbol: str) -> int:
    try:
        return _gfx942_heuristic_symbol_to_kid()[symbol]
    except KeyError as exc:
        raise RuntimeError(
            f"gfx942 heuristic returned unknown launcher symbol {symbol!r}"
        ) from exc


def _gfx942_heuristic_split_barrier_ok(N: int, K: int) -> bool:
    loops = (K + 63) // 64
    return N % 16 == 0 and K % 64 == 0 and loops >= 2 and loops % 2 == 0


def _gfx942_heuristic_bf16ws_band(M: int, N: int, K: int) -> bool:
    return (
        K >= 4096
        and K % 64 == 0
        and 104 <= M <= 608
        and (N == 256 or 512 <= N <= 2048)
    )


def _gfx942_heuristic_bf16_symbol(M: int, N: int, K: int) -> str:
    """Port of the original gfx942 BF16 launcher-choice ordering."""
    k64_ok = K % 64 == 0
    k32_ok = K % 32 == 0
    wkc_bk64_ok = K >= 4096 and K % 512 == 0
    p1_ok = K % 128 == 0
    sb_ok = _gfx942_heuristic_split_barrier_ok(N, K)

    if K == 4096:
        if p1_ok and (M in (48, 64) and N == 1024):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and ((M == 128 and N == 512) or (M == 256 and N == 256)):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and M == 512 and N == 256:
            return "opus_gemm_gfx942_splitk_p1_bk128_256x64x64x128_2x2_16x16x16_0x0x0"
        if M in (48, 64) and 1536 <= N <= 2048:
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (M == 128 and N == 1024) or (M == 256 and N == 512):
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (
            (M == 128 and 1536 <= N <= 2048)
            or (M == 256 and N == 1024)
            or (M == 512 and N == 512)
        ):
            return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if K >= 1024 and k32_ok and N >= 1536 and M <= 32:
        if M <= 4 and N >= 4096:
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        if M <= 16:
            if wkc_bk64_ok:
                return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
            return "opus_gemm_gfx942_wkc_512x16x32x32_1x1_16x16x16_0x0x0"
        if M == 32 and K == 4096 and wkc_bk64_ok:
            return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_256x32x32x64_1x1_16x16x16_0x0x0"

    if K >= 512 and k64_ok and (
        N <= 64 or (M <= 128 and N <= 1024) or (M <= 8 and N <= 1536)
    ):
        if N <= 64 and M > 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if N <= 256 or M <= 8 or (M <= 16 and N <= 800):
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"

    if _gfx942_heuristic_bf16ws_band(M, N, K):
        return "opus_gemm_gfx942_splitk_legacy_bf16ws_512x128x128x64_2x4_16x16x16_0x0x0"

    if N == 384 and K >= 4096:
        if M <= 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if M <= 224:
            return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        if 392 <= M <= 512:
            return "opus_gemm_gfx942_splitk_em3en4_lds1_pgr2_256x128x96x128_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if k64_ok and N >= 4096 and K <= 3200:
        if K <= 640 and M <= 128:
            return "opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"

    if sb_ok and M >= 128:
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"
    if N <= 256 and p1_ok:
        return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
    return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"


def _heuristic_a16w16_kid_gfx942(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: object = "bf16",
) -> int:
    """Return the original gfx942 no-tuned-row fallback kid."""
    del batch
    M, N, K = map(int, (M, N, K))
    if _output_dtype_name(output_dtype) == "bf16" and not has_bias:
        symbol = _gfx942_heuristic_bf16_symbol(M, N, K)
    elif N <= 256 and K % 128 == 0:
        symbol = "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
    else:
        symbol = "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"
    return _gfx942_heuristic_kid_for_symbol(symbol)


_A16W16_HEURISTICS = {
    "gfx942": _heuristic_a16w16_kid_gfx942,
    "gfx950": _heuristic_a16w16_kid_gfx950,
    "gfx1250": _heuristic_a16w16_kid_gfx1250,
}


def _select_a16w16_heuristic_kid(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    has_bias: bool,
    output_dtype: object,
) -> int:
    """Select one baseline-parity A16 kid before the exact public call."""
    arch = str(arch).lower().split(":", 1)[0]
    heuristic = _A16W16_HEURISTICS.get(arch)
    if heuristic is None:
        raise ValueError(f"no OPUS a16w16 heuristic for runtime arch {arch}")
    kid = int(heuristic(M, N, K, batch, has_bias, output_dtype))
    if kid not in DEFAULT_COMPILED_KIDS_BY_ARCH.get(arch, frozenset()):
        raise RuntimeError(
            f"{arch} a16w16 heuristic returned kid {kid}, which is not in "
            "DEFAULT_COMPILED_KIDS_BY_ARCH"
        )
    return kid


def _resolve_a16w16_caller_candidate(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    requested_kid: object | None,
    requested_split_k: object = 0,
) -> LaunchConfig | None:
    """Resolve a tuned or heuristic candidate to the final exact public kid.

    ``requested_kid=None`` selects the per-architecture heuristic.  Invalid
    tuned/heuristic candidates return ``None`` so the upper caller can proceed
    to the next priority or its PyTorch fallback.  The exact public launch path
    never calls this function and therefore never redirects a supplied kid.
    """
    arch = str(arch).lower().split(":", 1)[0]
    try:
        split_k = int(requested_split_k)
        kid = (
            _select_a16w16_heuristic_kid(
                arch=arch,
                M=M,
                N=N,
                K=K,
                batch=batch,
                has_bias=has_bias,
                output_dtype=output_dtype,
            )
            if requested_kid is None
            else int(requested_kid)
        )
    except (TypeError, ValueError):
        return None

    # Resolve the legacy gfx942 host redirects here, before the final integer
    # kid enters the exact public interface.
    if arch == "gfx942" and N not in GFX942_BF16WS_EXACT_N:
        if kid == 10210:
            kid = 10200
        elif kid == 10213:
            kid = 10203
        elif kid == 10216:
            return None

    try:
        return _resolve_exact_a16w16_config(
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
    except ValueError:
        return None


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
    converted = []
    c_args = []
    null_tensor = ctypes.POINTER(aiter_tensor_t)()
    for tensor in (XQ, WQ, Y, bias, workspace):
        if tensor is None:
            c_args.append(null_tensor)
            continue
        descriptor = torch_to_aiter(tensor)
        converted.append(descriptor)
        c_args.append(ctypes.byref(descriptor))

    stream = ctypes.c_void_p(torch.cuda.current_stream(XQ.device).cuda_stream)
    clear_error()
    status = launch(*c_args, kid, split_k, stream)
    raw_error = get_error()
    if status != 0 or raw_error:
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
    rebuild and architecture checks remain authoritative.  It then loads the
    C ABI from that same module; subsequent calls use the lower-overhead path.
    """
    global _opus_a16w16_cabi_primed
    if not _opus_a16w16_cabi_primed:
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


def _launch_a16w16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch one exact A16W16 kid for the unified ``opus_gemm`` entry."""
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


__all__: list[str] = []
