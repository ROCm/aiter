# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable exact-kid contracts and workspace planning for OPUS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from csrc.opus_gemm.opus_gemm_common import (
    BIAS_AWARE_KIDS,
    GFX942_BF16WS_EXACT_N,
    GFX942_EVEN_LOOP_SPLITK_TAGS,
    GFX942_MAX_AUTO_SPLIT_K,
    GFX942_MIN_ITERS_PER_SPLIT,
    SPLITK_KIDS,
    OpusGemmInstance,
    get_kernel_instance,
)

_WORKSPACE_DTYPES = {
    "bf16_t": torch.bfloat16,
    "fp32_t": torch.float32,
}
_GFX1250_FUSED_SPLITK_TAG = "a16w16_clusterlaunch_tdm_splitk_fuse"


@dataclass(frozen=True)
class WorkspaceSpec:
    """Immutable metadata required to materialize one launch workspace."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class A16W16LaunchPlan:
    """One validated exact A16W16 launch with immutable workspace metadata."""

    registry_arch: str
    resolved_kid: int
    workspace_capacity_split_k: int
    abi_split_k: int
    workspace_spec: WorkspaceSpec | None


# ---- A16W16 exact launch planning ---------------------------------------


def _supports_a16w16_shape(
    instance: OpusGemmInstance,
    *,
    registry_arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
) -> bool:
    if registry_arch == "gfx1250" and batch != 1:
        return False

    if instance.kernel_tag == _GFX1250_FUSED_SPLITK_TAG:
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


def _plan_gfx942_split_k(
    instance: OpusGemmInstance,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    requested: int,
) -> tuple[int, int]:
    """Return workspace-capacity and ABI split-K matching the gfx942 launcher."""
    if requested > 0:
        workspace_capacity = requested
    else:
        tiles_mn = (
            (M + instance.B_M - 1)
            // instance.B_M
            * ((N + instance.B_N - 1) // instance.B_N)
            * batch
        )
        tiles_mn = max(1, tiles_mn)
        target_wg = (2 * cu_num) if instance.kernel_tag.endswith("_p1") else cu_num
        workspace_capacity = (target_wg + tiles_mn - 1) // tiles_mn
        workspace_capacity = min(
            GFX942_MAX_AUTO_SPLIT_K, max(1, workspace_capacity)
        )

    total_iters = (K + instance.B_K - 1) // instance.B_K
    if total_iters < GFX942_MIN_ITERS_PER_SPLIT:
        raise ValueError(
            f"K={K} is too small for gfx942 kid B_K={instance.B_K}; "
            f"need at least {instance.B_K * GFX942_MIN_ITERS_PER_SPLIT}"
        )

    abi_split_k = workspace_capacity
    require_even = instance.kernel_tag in GFX942_EVEN_LOOP_SPLITK_TAGS
    while abi_split_k > 1:
        iters_full = (total_iters + abi_split_k - 1) // abi_split_k
        last_loops = total_iters - (abi_split_k - 1) * iters_full
        parity_ok = not require_even or (
            iters_full % 2 == 0 and last_loops % 2 == 0
        )
        if (
            iters_full >= GFX942_MIN_ITERS_PER_SPLIT
            and last_loops >= GFX942_MIN_ITERS_PER_SPLIT
            and parity_ok
        ):
            break
        abi_split_k -= 1

    if require_even:
        iters_full = (total_iters + abi_split_k - 1) // abi_split_k
        last_loops = total_iters - (abi_split_k - 1) * iters_full
        if iters_full % 2 != 0 or last_loops % 2 != 0:
            raise ValueError(
                f"gfx942 kid {instance.name} needs even loops per split; "
                f"K={K}, split_k={abi_split_k}, "
                f"loops=({iters_full},{last_loops})"
            )
    return workspace_capacity, abi_split_k


def _build_a16w16_workspace_spec(
    instance: OpusGemmInstance,
    *,
    registry_arch: str,
    resolved_kid: int,
    workspace_capacity_split_k: int,
    batch: int,
    M: int,
    N: int,
    K: int,
) -> WorkspaceSpec | None:
    """Build workspace metadata from the already-resolved registry instance."""
    needs_workspace = resolved_kid in SPLITK_KIDS
    declares_workspace = instance.splitk_workspace_dtype is not None
    if needs_workspace != declares_workspace:
        raise RuntimeError(
            "inconsistent OPUS a16w16 workspace registry for "
            f"{registry_arch} kid {resolved_kid}"
        )
    if not needs_workspace:
        return None

    is_fused = instance.kernel_tag == _GFX1250_FUSED_SPLITK_TAG
    split_k = (
        int(instance.fuse_split_k)
        if is_fused
        else int(workspace_capacity_split_k)
    )
    if split_k <= 0:
        raise ValueError(
            "opus_gemm_a16w16_launch: workspace capacity split_k must be "
            f"positive, got {split_k}"
        )

    block_m = int(instance.B_M)
    block_n = int(instance.B_N)
    block_k = int(instance.B_K)
    max_useful_split_k = (K + block_k - 1) // block_k
    if split_k > max_useful_split_k:
        raise ValueError(
            "opus_gemm_a16w16_launch: "
            f"workspace capacity split_k={split_k} exceeds the per-kid "
            f"K-tile limit {max_useful_split_k} for K={K}, B_K={block_k}"
        )

    if registry_arch == "gfx1250":
        if batch != 1:
            raise ValueError(
                "opus_gemm_a16w16_launch: gfx1250 workspace kids require "
                f"batch=1; got batch={batch}"
            )
        num_tiles_m = (M + block_m - 1) // block_m
        num_tiles_n = (N + block_n - 1) // block_n
        if is_fused:
            if split_k < 2:
                raise ValueError(
                    "opus_gemm_a16w16_launch: "
                    f"gfx1250 fused kid {resolved_kid} must declare "
                    f"compile-time SplitK >= 2, got {split_k}"
                )
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
            f"workspace kid {resolved_kid} must declare bf16_t or fp32_t "
            f"storage, got {dtype_token!r}"
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

    return WorkspaceSpec(shape=shape, dtype=dtype)


def _build_a16w16_launch_plan(
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
    kid: int,
    split_k: int,
) -> A16W16LaunchPlan:
    """Validate one exact kid and build all immutable launch metadata."""
    registry_arch = str(arch).lower().split(":", 1)[0]
    M, N, K, batch, cu_num = map(int, (M, N, K, batch, cu_num))
    if min(M, N, K, batch, cu_num) <= 0:
        raise ValueError("M, N, K, batch, and cu_num must all be positive")
    try:
        resolved_kid = int(kid)
        requested_split_k = int(split_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"OPUS a16w16 kid/split_k must be integers, got {kid!r}/{split_k!r}"
        ) from exc
    if requested_split_k < 0:
        raise ValueError(
            "OPUS a16w16 split_k must be non-negative, "
            f"got {requested_split_k}"
        )
    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"OPUS a16w16 requires bf16 XQ/WQ, got input dtype {input_dtype}"
        )

    instance = get_kernel_instance(
        registry_arch, "a16w16", resolved_kid, output_dtype
    )
    if instance is None:
        if get_kernel_instance(registry_arch, "a16w16", resolved_kid) is None:
            raise ValueError(
                f"OPUS kid {resolved_kid} is not an a16w16 kernel for "
                f"runtime arch {registry_arch}"
            )
        raise ValueError(
            f"OPUS kid {resolved_kid} does not support output dtype "
            f"{output_dtype}"
        )

    needs_workspace = resolved_kid in SPLITK_KIDS
    if (
        registry_arch == "gfx942"
        and needs_workspace
        and instance.splitk_workspace_dtype == "bf16_t"
        and N not in GFX942_BF16WS_EXACT_N
    ):
        raise ValueError(
            f"gfx942 exact kid {resolved_kid} requires N in "
            f"{sorted(GFX942_BF16WS_EXACT_N)}; got N={N}"
        )
    if registry_arch == "gfx1250" and needs_workspace and batch != 1:
        raise ValueError(
            "opus_gemm_a16w16_launch: gfx1250 workspace kids require "
            f"batch=1; got batch={batch}"
        )
    if not _supports_a16w16_shape(
        instance,
        registry_arch=registry_arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
    ):
        raise ValueError(
            f"OPUS kid {resolved_kid} is incompatible with "
            f"shape (batch={batch}, M={M}, N={N}, K={K})"
        )
    if has_bias and resolved_kid not in BIAS_AWARE_KIDS:
        raise ValueError(f"OPUS kid {resolved_kid} does not support bias")
    if has_bias and instance.kernel_tag == _GFX1250_FUSED_SPLITK_TAG:
        raise ValueError(
            "gfx1250 splitk_fuse has a narrower bf16 [N] bias contract than "
            "the public OPUS interfaces can represent"
        )
    if has_bias and registry_arch == "gfx942" and needs_workspace:
        raise ValueError(
            "the current gfx942 a16w16 launch rejects bias on split-K kernels"
        )

    workspace_capacity_split_k = 1
    abi_split_k = requested_split_k
    if needs_workspace:
        workspace_capacity_split_k = max(1, requested_split_k)
        if instance.kernel_tag == _GFX1250_FUSED_SPLITK_TAG:
            workspace_capacity_split_k = int(instance.fuse_split_k)
            abi_split_k = workspace_capacity_split_k
        elif registry_arch == "gfx942":
            workspace_capacity_split_k, abi_split_k = _plan_gfx942_split_k(
                instance,
                M=M,
                N=N,
                K=K,
                batch=batch,
                cu_num=cu_num,
                requested=requested_split_k,
            )

    workspace_spec = _build_a16w16_workspace_spec(
        instance,
        registry_arch=registry_arch,
        resolved_kid=resolved_kid,
        workspace_capacity_split_k=workspace_capacity_split_k,
        batch=batch,
        M=M,
        N=N,
        K=K,
    )
    return A16W16LaunchPlan(
        registry_arch=registry_arch,
        resolved_kid=resolved_kid,
        workspace_capacity_split_k=workspace_capacity_split_k,
        abi_split_k=abi_split_k,
        workspace_spec=workspace_spec,
    )


@lru_cache(maxsize=256)
def _get_cached_a16w16_launch_plan(
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: object,
    output_dtype: object,
    kid: int,
    split_k: int,
) -> A16W16LaunchPlan:
    """Return a scalar-only cached exact-kid launch plan."""
    return _build_a16w16_launch_plan(
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


# ---- A8W8 contracts and MXFP8 BMM planning ------------------------------

_A8W8_FAMILY = "a8w8"
_A8W8_BLOCKSCALE_FAMILY = "a8w8_blockscale"
_A8W8_BPRESHUFFLE_FAMILY = "a8w8_blockscale_bpreshuffle"
_A8W8_MXSCALE_BMM_FAMILY = "a8w8_mxscale_bmm"

_A8W8_MXSCALE_BMM_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
        "a8w8_mxscale_bmm_minterleave",
        "a8w8_mxscale_bmm_mouter",
        "a8w8_mxscale_bmm_mouter_tunable",
        "a8w8_mxscale_bmm_pipeline",
        "a8w8_mxscale_bmm_wave8n2",
        "a8w8_mxscale_bmm_wave4m2_selfload",
    }
)
_A8W8_MXSCALE_BMM_WORKSPACE_TAGS = frozenset(
    {
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    }
)
_A8W8_FAMILY_BY_TAG = {
    "a8w8": _A8W8_FAMILY,
    "a8w8_scale": _A8W8_BLOCKSCALE_FAMILY,
    "a8w8_blockscale_bpreshuffle_singlebuf": _A8W8_BPRESHUFFLE_FAMILY,
    **{tag: _A8W8_MXSCALE_BMM_FAMILY for tag in _A8W8_MXSCALE_BMM_TAGS},
}
_A8W8_FAMILY_LAYOUT = {
    _A8W8_FAMILY: "plain",
    _A8W8_BLOCKSCALE_FAMILY: "plain",
    _A8W8_BPRESHUFFLE_FAMILY: "bpreshuffle",
    _A8W8_MXSCALE_BMM_FAMILY: "mxscale_bmm",
}
_FP8_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
    )
    if dtype is not None
)
@dataclass(frozen=True)
class A8W8MxscaleBMMPlan:
    """One resolved MXFP8 BMM launch and its optional workspace."""

    registry_arch: str
    resolved_kid: int
    abi_split_k: int
    workspace_spec: WorkspaceSpec | None


def _validate_a8w8_public_contract(
    *,
    kernel_tag: str,
    kid: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
    has_bias: bool,
    has_workspace: bool,
    split_k: int,
) -> str:
    """Validate immutable options for the public A8W8 operation routers."""
    try:
        family = _A8W8_FAMILY_BY_TAG[kernel_tag]
    except KeyError as exc:
        raise ValueError(
            f"OPUS kid {kid} has unsupported registry tag {kernel_tag!r}"
        ) from exc

    if input_dtype != weight_dtype:
        raise ValueError(
            f"OPUS requires matching XQ/WQ dtypes; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if input_dtype not in _FP8_DTYPES:
        raise ValueError(
            f"OPUS kid {kid} requires FP8 XQ/WQ; got {input_dtype}"
        )

    if family == _A8W8_MXSCALE_BMM_FAMILY:
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"OPUS kid {kid} requires BF16 or FP32 Y; got {output_dtype}"
            )
    else:
        expected_output_dtype = (
            torch.bfloat16
            if family == _A8W8_BPRESHUFFLE_FAMILY
            else torch.float32
        )
        if output_dtype != expected_output_dtype:
            raise ValueError(
                f"OPUS kid {kid} does not support Y.dtype={output_dtype}; "
                f"expected {expected_output_dtype}"
            )

    expected_layout = _A8W8_FAMILY_LAYOUT[family]
    if layout != expected_layout:
        raise ValueError(
            f"OPUS kid {kid} belongs to family {family} and requires "
            f"layout={expected_layout!r}; got {layout!r}"
        )
    if has_x_scale != has_w_scale:
        raise ValueError("OPUS requires x_scale and w_scale together")
    if has_bias:
        raise ValueError(f"OPUS family {family} does not support bias")

    if family == _A8W8_MXSCALE_BMM_FAMILY:
        if not has_x_scale:
            raise ValueError("OPUS a8w8_mxscale_bmm requires x_scale and w_scale")
        if split_k == 0 and has_workspace:
            raise ValueError(
                "OPUS a8w8_mxscale_bmm split_k=0 does not use workspace"
            )
        return family

    if has_workspace:
        raise ValueError(f"OPUS family {family} does not use workspace")
    if split_k != 0:
        raise ValueError(f"OPUS family {family} does not accept split_k")
    if family == _A8W8_FAMILY:
        if has_x_scale:
            raise ValueError("OPUS a8w8 kid does not accept scales")
    elif not has_x_scale:
        raise ValueError(f"OPUS family {family} requires x_scale and w_scale")
    return family


@lru_cache(maxsize=None)
def _require_registered_kid_cached(
    arch: str,
    family: str,
    resolved_kid: int,
    output_dtype: torch.dtype,
) -> int:
    """Validate one A8W8 registry entry and cache successful lookups."""
    if get_kernel_instance(arch, family, resolved_kid, output_dtype) is None:
        raise ValueError(
            "no registered OPUS kernel for "
            f"(arch={arch!r}, family={family!r}, kid={resolved_kid}, "
            f"Y.dtype={output_dtype})"
        )
    return resolved_kid


def _require_registered_kid(
    *,
    arch: str,
    family: str,
    kid: object,
    output_dtype: torch.dtype,
) -> int:
    """Normalize a kid and require the matching A8W8 registry entry."""
    try:
        resolved_kid = int(kid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPUS {family} kid must be an integer, got {kid!r}") from exc
    return _require_registered_kid_cached(
        arch, family, resolved_kid, output_dtype
    )


def _build_a8w8_mxscale_bmm_plan(
    *,
    arch: str,
    kid: int,
    output_dtype: torch.dtype,
    M: int,
    batch: int,
    N: int,
    K: int,
    split_k: int,
) -> A8W8MxscaleBMMPlan:
    """Resolve one MXFP8 BMM kid, ABI split-K and FP32 workspace."""
    registry_arch = str(arch).lower().split(":", 1)[0]
    resolved_kid = int(kid)
    M, batch, N, K = map(int, (M, batch, N, K))
    requested_split_k = int(split_k)

    instance = get_kernel_instance(
        registry_arch,
        _A8W8_MXSCALE_BMM_FAMILY,
        resolved_kid,
        output_dtype,
    )
    if instance is None:
        raise ValueError(
            "no registered OPUS kernel for "
            f"(arch={registry_arch!r}, "
            f"family={_A8W8_MXSCALE_BMM_FAMILY!r}, "
            f"kid={resolved_kid}, Y.dtype={output_dtype})"
        )

    tag = instance.kernel_tag
    if tag not in _A8W8_MXSCALE_BMM_TAGS:
        raise ValueError(f"OPUS kid {resolved_kid} is not an MXFP8 BMM kernel")

    abi_split_k = max(1, requested_split_k)
    m_align = max(1, int(instance.m_align))
    if M % m_align:
        raise ValueError(
            f"OPUS BMM kid {resolved_kid} requires M % {m_align} == 0; "
            f"got M={M}"
        )
    if instance.direct_only and abi_split_k != 1:
        raise ValueError(f"OPUS BMM kid {resolved_kid} requires split_k <= 1")

    workspace_numel = 0
    if tag in _A8W8_MXSCALE_BMM_WORKSPACE_TAGS:
        if abi_split_k > 1:
            tiles_m = (M + instance.B_M - 1) // instance.B_M
            tiles_n = (N + instance.B_N - 1) // instance.B_N
            padded_m = tiles_m * instance.B_M
            padded_n = tiles_n * instance.B_N
            partial_numel = abi_split_k * batch * padded_m * padded_n
            if tag == "a8w8_mxscale_bmm_fused":
                counter_offset = (partial_numel * 4 + 255) & ~255
                counter_bytes = batch * tiles_m * tiles_n * 4
                workspace_numel = (counter_offset + counter_bytes + 3) // 4
            else:
                workspace_numel = partial_numel
    elif tag == "a8w8_mxscale_bmm_mouter_tunable":
        pass
    elif abi_split_k != 1:
        raise ValueError(f"OPUS BMM kid {resolved_kid} requires split_k <= 1")

    workspace_spec = (
        WorkspaceSpec(shape=(workspace_numel,), dtype=torch.float32)
        if workspace_numel
        else None
    )
    return A8W8MxscaleBMMPlan(
        registry_arch=registry_arch,
        resolved_kid=resolved_kid,
        abi_split_k=abi_split_k,
        workspace_spec=workspace_spec,
    )


@lru_cache(maxsize=4096)
def _get_cached_a8w8_mxscale_bmm_plan(
    arch: str,
    kid: int,
    output_dtype: torch.dtype,
    M: int,
    batch: int,
    N: int,
    K: int,
    split_k: int,
) -> A8W8MxscaleBMMPlan:
    """Return a scalar-only cached MXFP8 BMM launch plan."""
    return _build_a8w8_mxscale_bmm_plan(
        arch=arch,
        kid=kid,
        output_dtype=output_dtype,
        M=M,
        batch=batch,
        N=N,
        K=K,
        split_k=split_k,
    )


__all__ = [
    "A16W16LaunchPlan",
    "A8W8MxscaleBMMPlan",
    "WorkspaceSpec",
]
