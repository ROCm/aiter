# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A16W16 exact launch, Torch workspace, and shape-driven compatibility."""

from functools import lru_cache

import torch

from aiter import logger
from csrc.opus_gemm.opus_gemm_common import (
    OpusGemmInstance,
    get_kernel_instance,
    get_registry,
)

from ...jit.core import compile_ops
from ._arch import _device_arch_and_cu
from .launch_plan import (
    _GFX950_4WAVE_B_K,
    GFX950_4WAVE_TAGS,
    _get_cached_a16w16_launch_plan,
)

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
    route_cu_num: int | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Plan and launch one exact 3D A16W16 operation (layout checked by _execute_a16w16_<layout>)."""
    batch, M, K = XQ.shape
    N = Y.shape[2]

    use_gfx950_caller_workspace_fast_path = (
        route_arch == "gfx950"
        and route_cu_num is not None
        and workspace is not None
        and split_k > 0
        and instance is not None
        and instance.splitk_workspace_dtype is not None
    )
    if use_gfx950_caller_workspace_fast_path:
        # Reuse the dispatcher's actual device identity for local-kid lookup.
        arch, cu_num = route_arch, route_cu_num
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


# NT / NN / TT / TN: _launch_a16w16_gemm / _bmm pick the layout from the operand strides (batch-first
# XQ [B,M,K] / WQ [B,N,K]); each _execute_a16w16_<layout> checks its contract, then runs _execute_a16w16.
# NN / TT / TN need a kid whose launcher takes M/N-contiguous operands (gfx950 4-wave; it loads that
# layout's image and checks the strides).
#   NT: XQ (K,1)  WQ (K,1)      NN: XQ (K,1)  WQ (1,N)
#   TT: XQ (1,M)  WQ (K,1)      TN: XQ (1,M)  WQ (1,N)
def _a16w16_operand_transposed(tensor: torch.Tensor) -> bool:
    """True if the [.., rows, K] operand is stored [K, rows] (rows contiguous)."""
    rows, K = tensor.shape[-2:]
    stride_row, stride_k = tensor.stride()[-2:]
    return rows > 1 and stride_row == 1 and not (stride_k == 1 and stride_row >= K)


@lru_cache(maxsize=256)
def _get_cached_a16w16_instance(
    arch: str, cu_num: int, kid: int
) -> OpusGemmInstance | None:
    """Registry record of an exact kid (None if absent), looked up once per kid."""
    return get_kernel_instance(arch, "a16w16", kid, cu_num=cu_num)


def _require_layout_kid(
    layout, XQ, *, kid, instance=None, route_arch=None, route_cu_num=None, **_
):
    if instance is None:
        arch, cu_num = route_arch, route_cu_num
        if arch is None or cu_num is None:
            arch, cu_num = _device_arch_and_cu(XQ.device)
        instance = _get_cached_a16w16_instance(arch, cu_num, int(kid))
    if instance is None or instance.kernel_tag not in GFX950_4WAVE_TAGS:
        raise NotImplementedError(
            f"opus_gemm_a16w16_launch: {layout} operands need a gfx950 a16w16 4-wave kid "
            f"(tags {sorted(GFX950_4WAVE_TAGS)}); kid {kid} takes K-contiguous XQ/WQ only"
        )


def _execute_a16w16_nt(XQ, WQ, Y, bias=None, **kwargs):
    """NT: XQ stored [M,K], WQ stored [N,K]; every kid."""
    _check_a16w16_launch_layout(XQ, WQ, Y)
    return _execute_a16w16(XQ, WQ, Y, bias, **kwargs)


def _execute_a16w16_nn(XQ, WQ, Y, bias=None, **kwargs):
    """NN: XQ stored [M,K], WQ the [N,K] view of a [K,N] weight."""
    _require_layout_kid("NN", XQ, **kwargs)
    return _execute_a16w16(XQ, WQ, Y, bias, **kwargs)


def _execute_a16w16_tt(XQ, WQ, Y, bias=None, **kwargs):
    """TT: XQ the [M,K] view of a [K,M] activation, WQ stored [N,K]."""
    _require_layout_kid("TT", XQ, **kwargs)
    return _execute_a16w16(XQ, WQ, Y, bias, **kwargs)


def _execute_a16w16_tn(XQ, WQ, Y, bias=None, **kwargs):
    """TN: XQ / WQ views of [K,M] / [K,N] storage."""
    _require_layout_kid("TN", XQ, **kwargs)
    return _execute_a16w16(XQ, WQ, Y, bias, **kwargs)


def _a16w16_layout_executor(XQ: torch.Tensor, WQ: torch.Tensor):
    """The _execute_a16w16_<layout> of the operand strides."""
    return (
        (_execute_a16w16_nt, _execute_a16w16_nn),
        (_execute_a16w16_tt, _execute_a16w16_tn),
    )[_a16w16_operand_transposed(XQ)][_a16w16_operand_transposed(WQ)]


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
    route_cu_num: int | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` A16W16 GEMM."""
    if instance is None and (XQ.dim() != 2 or WQ.dim() != 2 or Y.dim() != 2):
        raise ValueError(
            "opus_gemm A16W16 expects 2D XQ/WQ/Y; use opus_bmm for "
            "batch-first 3D tensors"
        )
    _a16w16_layout_executor(XQ, WQ)(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        Y.unsqueeze(0),
        bias,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        route_arch=route_arch,
        route_cu_num=route_cu_num,
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
    route_cu_num: int | None = None,
    instance: OpusGemmInstance | None = None,
) -> torch.Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` A16W16 BMM."""
    if instance is None and (XQ.dim() != 3 or WQ.dim() != 3 or Y.dim() != 3):
        raise ValueError(
            "opus_bmm A16W16 expects batch-first 3D XQ/WQ/Y; use "
            "opus_gemm for logical 2D tensors"
        )
    return _a16w16_layout_executor(XQ, WQ)(
        XQ,
        WQ,
        Y,
        bias,
        kid=kid,
        split_k=split_k,
        workspace=workspace,
        route_arch=route_arch,
        route_cu_num=route_cu_num,
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
    return _a16w16_layout_executor(XQ, WQ)(
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


@lru_cache(maxsize=256)
def _warn_invalid_a16w16_tuned_row(message: str) -> None:
    logger.warning(message)


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

    An explicit ``kernelId`` wins over tuned lookup. Explicit and tuned ids
    still pass through legacy requested-to-actual compatibility resolution;
    a missing or invalid tuned row uses the migrated architecture heuristic.
    """
    XQ, WQ, Y, is_gemm = _prepare_shape_driven_a16w16(A, B, bias, dtype, out)
    from .policy import (
        lookup_a16w16_opus_config,
        resolve_a16w16_heuristic_candidate,
        resolve_a16w16_tuned_candidate,
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
    if kernelId is None:
        config = lookup_a16w16_opus_config(**lookup_args)
        plan = None
        if config is not None:
            plan = resolve_a16w16_tuned_candidate(
                batch=batch,
                requested_kid=config.get("solidx"),
                requested_split_k=config.get("splitK"),
                **lookup_args,
            )
            if plan is None:
                _warn_invalid_a16w16_tuned_row(
                    "Ignoring invalid OPUS A16W16 tuned row for "
                    f"gfx={arch}, cu_num={cu_num}, "
                    f"shape=({batch},{M},{N},{K}), "
                    f"kid={config.get('solidx')!r}, "
                    f"splitK={config.get('splitK')!r}; "
                    "using OPUS heuristic fallback"
                )
        if plan is None:
            plan = resolve_a16w16_heuristic_candidate(batch=batch, **lookup_args)
            if plan is None:
                raise RuntimeError(
                    "gemm_a16w16_opus found no valid OPUS kernel for "
                    f"arch={arch}, shape=({batch},{M},{N},{K})"
                )
            kid, split_k = plan.resolved_kid, 0
        else:
            kid = plan.resolved_kid
            split_k = int(config["splitK"])
    else:
        kid = int(kernelId)
        split_k = int(splitK or 0)
        plan = resolve_a16w16_tuned_candidate(
            batch=batch,
            requested_kid=kid,
            requested_split_k=split_k,
            **lookup_args,
        )
        if plan is not None:
            kid = plan.resolved_kid

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


# ---- BF16 NT / NN / TT / TN (gfx950 4-wave CO kids) --------------------------
# XQ [M, K] / WQ [N, K] are logical views of the stored operands; the kid's
# launcher picks the layout image from their strides. a16w16_4wave kids launch
# directly; a16w16_4wave_ws kids are the same tiles with an fp32 split-K
# workspace and take a bf16 [N] bias. Y [M, N] is bf16 in every layout.
#   NT: XQ (K,1)  WQ (K,1)      NN: XQ (K,1)  WQ (1,N)
#   TT: XQ (1,M)  WQ (K,1)      TN: XQ (1,M)  WQ (1,N)
_A16W16_4WAVE_WS_TAG = "a16w16_4wave_ws"


def _a16w16_direction(XQ: torch.Tensor, WQ: torch.Tensor) -> str | None:
    """Storage of logical XQ [M, K] / WQ [N, K] as the kernel names spell it.

    First letter A, second B. A "N" is stored [M, K] (strides (K, 1)), A "T"
    [K, M] (strides (1, M)); B "T" is stored [N, K] (strides (K, 1)), B "N"
    [K, N] (strides (1, N)). A single row is K-contiguous whatever its row
    stride (as the launcher reads it): M == 1 is "N", N == 1 is "T". Returns
    "NN", "TT", "TN", "NT" or None.
    """
    (M, K), N = XQ.shape, WQ.shape[0]
    if XQ.stride() == (K, 1) or (M == 1 and XQ.stride(1) == 1):
        a = "N"
    elif XQ.stride() == (1, M):
        a = "T"
    else:
        return None
    if WQ.stride() == (K, 1) or (N == 1 and WQ.stride(1) == 1):
        b = "T"
    elif WQ.stride() == (1, N):
        b = "N"
    else:
        return None
    return a + b


def _a16w16_single_row_view(tensor: torch.Tensor) -> torch.Tensor:
    """A K-contiguous [1, K] operand with the canonical (K, 1) strides (no copy)."""
    rows, K = tensor.shape
    if rows == 1 and tensor.stride(1) == 1 and tensor.stride(0) != K:
        return tensor[0].unsqueeze(0)
    return tensor


def _a16w16_4wave_split(instance, M: int, N: int, K: int, cu_num: int) -> int:
    """Split of an a16w16_4wave_ws launch: the largest S in {8, 4, 2} with
    tiles * S <= CUs, K / S >= 1024 and an fp32 workspace
    [S][padded_M][padded_N] under 4 GiB; else 1."""
    m_tiles, n_tiles = -(-M // instance.B_M), -(-N // instance.B_N)
    padded = m_tiles * instance.B_M * n_tiles * instance.B_N
    return next(
        (
            s
            for s in (8, 4, 2)
            if m_tiles * n_tiles * s <= cu_num
            and K // s >= 1024
            and s * padded * 4 <= (1 << 32) - 1
        ),
        1,
    )


def a16w16_4wave_kid_runs(
    instance, M: int, N: int, K: int, split_k: int = 0, layout: str = "NT"
) -> bool:
    """Whether gfx950 4-wave kid ``instance`` runs (M, N, K) in ``layout``
    (NT / NN / TT / TN) with ``split_k``: K tiles of 64, at least two per split;
    an A stored [K, M] needs M % 8 and a B stored [K, N] N % 8; split-K kids
    keep the fp32 workspace under 4 GiB."""
    if instance.kernel_tag not in GFX950_4WAVE_TAGS:
        return False
    if K % _GFX950_4WAVE_B_K or K < 2 * _GFX950_4WAVE_B_K:
        return False
    if (layout.upper()[0] == "T" and M % 8) or (layout.upper()[1] == "N" and N % 8):
        return False
    if instance.kernel_tag != _A16W16_4WAVE_WS_TAG:
        return split_k <= 1
    split = max(1, split_k)
    padded = -(-M // instance.B_M) * instance.B_M * -(-N // instance.B_N) * instance.B_N
    return K // _GFX950_4WAVE_B_K >= 2 * split and split * padded * 4 <= (1 << 32) - 1


@lru_cache(maxsize=256)
def _select_a16w16_4wave_kid(
    direction: str,
    M: int,
    N: int,
    K: int,
    arch: str,
    cu_num: int,
    has_bias: bool = False,
) -> int | None:
    """Pick a gfx950 4-wave kid for the shape (fit to a sweep of every kid and
    split in all four layouts on MI355X).

    Each tile, no wider than the smallest of 64 / 128 / 256 covering M (B_M)
    and N (B_N), runs at the split of :func:`_a16w16_4wave_split`, or unsplit
    when some tile alone already has 1.25x as many tiles as CUs. Of the tiles
    whose blocks (tiles * split) reach 0.55 of the CUs the largest wins, else
    the one with the most blocks; ties go to less padding, the squarer tile,
    then the long side on the smaller of M / N. A split above 1 runs the
    a16w16_4wave_ws kid of that tile, as does a bias. Cached per shape: the
    registry is filtered (and its images checked) once per distinct call.
    """
    kids = {
        kid: instance
        for kid, instance in get_registry("a16w16", arch, cu_num).items()
        if a16w16_4wave_kid_runs(instance, M, N, K, 0, direction)
    }

    def cover(x):
        return next((t for t in (64, 128) if x <= t), 256)

    def tile(instance):
        return instance.B_M, instance.B_N

    split_kid = {
        tile(instance): kid
        for kid, instance in kids.items()
        if instance.kernel_tag == _A16W16_4WAVE_WS_TAG
    }
    candidates = []
    for kid, instance in kids.items():
        b_m, b_n = tile(instance)
        if (
            instance.kernel_tag == _A16W16_4WAVE_WS_TAG
            or b_m > cover(M)
            or b_n > cover(N)
        ):
            continue
        m_tiles, n_tiles = -(-M // b_m), -(-N // b_n)
        ws = kids.get(split_kid.get((b_m, b_n)))
        split = _a16w16_4wave_split(ws, M, N, K, cu_num) if ws is not None else 1
        tie = (abs(b_m - b_n), b_m - b_n if M >= N else b_n - b_m, kid)
        # [kid, tile area, tiles, padded output elements, tie-break, split]
        candidates.append(
            [
                kid,
                b_m * b_n,
                m_tiles * n_tiles,
                m_tiles * b_m * n_tiles * b_n,
                tie,
                split,
            ]
        )
    if not candidates:
        return None
    if any(c[2] >= 1.25 * cu_num for c in candidates):
        for c in candidates:
            c[5] = 1
    busy = [c for c in candidates if c[2] * c[5] >= 0.55 * cu_num]
    if busy:
        best = min(busy, key=lambda c: (-c[1], c[3], c[4]))
    else:
        best = min(candidates, key=lambda c: (-c[2] * c[5], c[3], c[4]))
    kid, split = best[0], best[5]
    if split > 1 or has_bias:
        return split_kid.get(tile(kids[kid]))
    return kid


def gemm_a16w16_layout_opus(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """gfx950 BF16 GEMM for NT / NN / TT / TN operand storage.

    Computes Y[M, N] = XQ[M, K] @ WQ[N, K]^T (+ bias) on logical shapes; the
    storage is read from the strides (see ``_a16w16_direction``), so pass the
    transposed view of a [K, M] A or of a [K, N] B, never a copy. K % 64 == 0
    and K >= 128; A stored [K, M] needs M % 8, B stored [K, N] N % 8. bias is a
    contiguous bf16 [N] (split-K kids only). Output is BF16.

    ``kernelId`` runs that gfx950 4-wave kid (1000-1007 direct, 1008-1015
    split-K, which pick their split). Otherwise NT goes through the tuned
    ``gemm_a16w16_opus`` path and NN / TT / TN pick a kid by shape.
    """
    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError("gemm_a16w16_layout_opus expects 2D XQ / WQ")
    if XQ.dtype != torch.bfloat16 or WQ.dtype != torch.bfloat16:
        raise ValueError(
            f"gemm_a16w16_layout_opus reads bf16 XQ / WQ; got {XQ.dtype}/{WQ.dtype}"
        )
    if dtype != torch.bfloat16:
        raise ValueError("gemm_a16w16_layout_opus writes BF16 only")
    XQ, WQ = _a16w16_single_row_view(XQ), _a16w16_single_row_view(WQ)
    direction = _a16w16_direction(XQ, WQ)
    if direction is None:
        raise ValueError(
            "gemm_a16w16_layout_opus serves NT / NN / TT / TN storage; got "
            f"XQ strides {tuple(XQ.stride())}, WQ strides {tuple(WQ.stride())}"
        )
    M, K = map(int, XQ.shape)
    N = int(WQ.shape[0])
    if bias is not None and (
        bias.dtype != torch.bfloat16
        or tuple(bias.shape) != (N,)
        or not bias.is_contiguous()
    ):
        raise ValueError(
            f"bias must be a contiguous bf16 [{N}], got {bias.dtype} {tuple(bias.shape)}"
        )
    if out is None:
        out = torch.empty((M, N), dtype=dtype, device=XQ.device)
    elif tuple(out.shape) != (M, N) or out.dtype != dtype:
        raise ValueError(
            f"out must be {dtype} [{M}, {N}], got {out.dtype} {tuple(out.shape)}"
        )
    if kernelId is None and direction == "NT":
        gemm_a16w16_opus(XQ, WQ, bias, dtype, out=out.unsqueeze(0))
        return out
    arch, cu_num = _device_arch_and_cu(XQ.device)
    if kernelId is None:
        kernelId = _select_a16w16_4wave_kid(
            direction, M, N, K, arch, cu_num, bias is not None
        )
        if kernelId is None:
            raise ValueError(
                f"no {direction} BF16 4-wave kernel supports M={M}, N={N}, K={K} on "
                f"{arch} (K % 64 == 0 and K >= 128; A stored [K, M] needs M % 8, "
                "B stored [K, N] N % 8)"
            )
    instance = _get_cached_a16w16_instance(arch, cu_num, int(kernelId))
    if instance is None or instance.kernel_tag not in GFX950_4WAVE_TAGS:
        raise ValueError(
            f"gemm_a16w16_layout_opus runs the gfx950 4-wave kids, got {kernelId}"
        )
    split_k = (
        _a16w16_4wave_split(instance, M, N, K, cu_num)
        if instance.kernel_tag == _A16W16_4WAVE_WS_TAG
        else 0
    )
    return _launch_a16w16_gemm(XQ, WQ, out, bias, kid=int(kernelId), split_k=split_k)


__all__ = [
    "a16w16_4wave_kid_runs",
    "gemm_a16w16_layout_opus",
    "gemm_a16w16_opus",
    "opus_gemm_a16w16_tune",
]
