# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 Python user-facing API.

Public entry points:

* `gemm_a16w16_opus(A, B, bias=None, dtype=bf16, *, kernelId=None, splitK=None, out=None)`
  Shape-driven wrapper. The typical user writes `gemm_a16w16_opus(A, B)`
  and never sees a kid number. Internal path:

    1. Reshape A/B to 3D, allocate Y, validate (bias unsupported;
       bpreshuffle and non-bf16 A/B also unsupported).
    2. If `kernelId` is given explicitly -> opus_gemm_a16w16_tune.
    3. Otherwise query opus-private tuned CSV via aiter.ops.opus.common;
       on hit -> opus_gemm_a16w16_tune with the tuned (solidx, splitK).
    4. On miss -> optionally autolog the shape
       (AITER_OPUS_LOG_UNTUNED=1) and fall through to the private
       bf16 no-scale fallback binding `_opus_gemm_bf16_dispatch`,
       which forwards to the C++ entry `opus_gemm` whose bf16 branch
       does its own lookup + heuristic dispatch (see
       csrc/opus_gemm/opus_gemm.cu).

* `opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)`
  Low-level pybind binding to the id-based tune dispatcher. Exposes a
  specific kernel instance by `kernelId` plus optional literal KBatch
  via `splitK`. Intended for the tuner, for debugging a specific kid,
  and for aiter-global integrations (e.g. future tuned_gemm.solMap).

All entry points share the JIT module `module_deepgemm_opus`, which
still hosts bindings for other opus kernel families (a8w8 etc.). The
Python surface is deliberately per-dtype: a16w16 here, a8w8 in its own
module when that lands.
"""

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from . import common as _opus_common


# ---- Low-level pybind bindings --------------------------------------------


def _gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


# Raw pybind binding to the C++ id-based dispatcher. We wrap it in a Python
# function below to add a stride-layout guard before the C++ call -- the
# launcher hardcodes stride_b_batch == N*K and reads gpu memory directly,
# so a broadcast / non-contiguous WQ silently corrupts results or faults
# the GPU. Keep `gen_fake` and `fc_name` on the raw binding so dynamo and
# torch.library see the underlying op.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    gen_fake=_gen_opus_gemm_a16w16_tune_fake_tensors,
)
def _opus_gemm_a16w16_tune_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


def _check_a16w16_tune_layout(XQ: torch.Tensor, WQ: torch.Tensor, Y: torch.Tensor):
    """Reject layouts that the opus launcher's hardcoded strides cannot serve.

    Mirrors the kargs setup in csrc/opus_gemm/gen_instances.py
    (_gen_flatmm_splitk_instance et al.):
        kargs.stride_a        = K
        kargs.stride_b        = K
        kargs.stride_c        = N
        kargs.stride_a_batch  = M * K
        kargs.stride_b_batch  = N * K
        kargs.stride_c_batch  = M * N
    The kernel reads memory at `ptr + batch_id * stride_*_batch + ...`
    directly. Any broadcast view (batch stride == 0), transpose, or
    sliced layout will hit garbage / unmapped memory.

    Cheap to run (a handful of integer comparisons); only raised on real
    misuse so the hot path pays nothing.
    """
    for name, t in (("XQ", XQ), ("WQ", WQ), ("Y", Y)):
        if t.dim() != 3:
            raise ValueError(
                f"opus_gemm_a16w16_tune: {name} must be 3D (got "
                f"{name}.shape={tuple(t.shape)}). The C++ launcher reads "
                f"`{name}.size(0)` as batch and indexes with hardcoded "
                f"stride_*_batch == size(1)*size(2)."
            )

    batch, M, K = XQ.shape
    b_w, N, K_w = WQ.shape
    b_y, M_y, N_y = Y.shape
    if (b_w, K_w) != (batch, K):
        raise ValueError(
            f"opus_gemm_a16w16_tune: WQ shape mismatch (got "
            f"WQ.shape={tuple(WQ.shape)}, expected "
            f"({batch}, N, {K})); XQ.shape={tuple(XQ.shape)}"
        )
    if (b_y, M_y, N_y) != (batch, M, N):
        raise ValueError(
            f"opus_gemm_a16w16_tune: Y shape mismatch (got "
            f"Y.shape={tuple(Y.shape)}, expected ({batch}, {M}, {N}))"
        )

    # Strides must match the launcher's hardcoded assumptions.
    expected = {
        "XQ": (XQ, (M * K, K, 1)),
        "WQ": (WQ, (N * K, K, 1)),
        "Y":  (Y,  (M * N, N, 1)),
    }
    for name, (t, want) in expected.items():
        got = tuple(t.stride())
        if got != want:
            raise NotImplementedError(
                f"opus_gemm_a16w16_tune: {name} must have contiguous "
                f"strides {want} (got {name}.stride()={got}, "
                f"{name}.shape={tuple(t.shape)}). The launcher hardcodes "
                f"stride_*_batch and stride_* values; any broadcast / "
                f"transpose / slice produces wrong results or a memory "
                f"access fault. Materialize with `{name} = {name}."
                f"contiguous()` before calling."
            )


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    """Low-level id-based dispatcher (Python guard + C++ launch).

    See module docstring. This Python wrapper checks XQ/WQ/Y layout up
    front (rejecting broadcast / transpose / slice views that the C++
    kernel would happily run with garbage data); on success it forwards
    to the underlying pybind binding.
    """
    _check_a16w16_tune_layout(XQ, WQ, Y)
    return _opus_gemm_a16w16_tune_raw(XQ, WQ, Y, kernelId, splitK)


# Private bf16 no-scale dispatch binding, used only by gemm_a16w16_opus
# as the CSV-miss fallback path. Wraps the same C++ function (opus_gemm)
# that used to be exposed via the legacy aiter.ops.deepgemm.deepgemm_opus
# entry, but deliberately hides its scale / group_layout arguments so
# callers of the a16w16 module do not see FP8-grouped concepts. The C++
# side's bf16 branch handles lookup + heuristic dispatch internally.
#
# Parameter annotations match the C++ signature exactly; torch_library's
# infer_schema requires every parameter be typed even though we always
# pass None for the last three.
def _gen_opus_gemm_bf16_dispatch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group_layout: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm",
    gen_fake=_gen_opus_gemm_bf16_dispatch_fake_tensors,
)
def _opus_gemm_bf16_dispatch(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group_layout: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...


# ---- High-level shape-driven API -----------------------------------------

# splitk kids (200..299) main kernel only has the <fp32_t> instantiation
# (traits static_assert D_C==float, fp32 workspace). The reduce kernel
# (splitk_reduce_kernel) is templated on D_OUT and dispatches to either
# __bf16 or float at launch time based on Y.dtype(), so both bf16 and fp32
# outputs are valid. Kept here only as a documentation anchor; the dispatch
# code below no longer needs to special-case Y.dtype against splitk kids.
_SPLITK_KID_MIN = 200
_SPLITK_KID_MAX = 299


def _validate_and_reshape(A: Tensor, B: Tensor, bias, dtype, out):
    if bias is not None:
        raise NotImplementedError(
            "gemm_a16w16_opus does not currently support bias (opus kernels "
            "compile with HAS_BIAS=false). Pass bias=None."
        )
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"gemm_a16w16_opus only supports bf16 A/B "
            f"(got A.dtype={A.dtype}, B.dtype={B.dtype})."
        )
    if dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(
            f"gemm_a16w16_opus only supports bf16/fp32 output dtype, got {dtype}"
        )

    # Resolve A first so we know `batch`.
    if A.dim() == 2:
        M, K = A.shape
        batch = 1
        XQ = A.unsqueeze(0)
        reshape_out_to_2d = True
    elif A.dim() == 3:
        batch, M, K = A.shape
        XQ = A
        reshape_out_to_2d = False
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {tuple(A.shape)}")

    # B accepted shapes:
    #   * [N, K]                       - allowed only when batch == 1
    #   * [batch, N, K] real-strided   - allowed for any batch
    #
    # The opus a16w16-family launchers hardcode `kargs.stride_b_batch = N * K`
    # (csrc/opus_gemm/gen_instances.py around lines 531/634/735/865) and the
    # device kernel computes `ptr_b + batch_id * stride_b_batch` directly,
    # ignoring the tensor's reported stride. A `B.unsqueeze(0).expand(batch,
    # -1, -1)` view has batch_stride == 0, so the kernel reads garbage past
    # B's real allocation -- this manifests as NaN, large numerical errors,
    # or HIP "Memory access fault by GPU node-1" depending on what the
    # caching allocator parked next to B. Reject the broken case at the
    # Python boundary rather than letting it through.
    if B.dim() == 2:
        N, K_b = B.shape
        if K_b != K:
            raise ValueError(
                f"K dimension mismatch: A has K={K}, B has K={K_b}"
            )
        if batch > 1:
            raise NotImplementedError(
                f"gemm_a16w16_opus: B must be 3D [batch, N, K] when A is "
                f"batched (got A.shape={tuple(A.shape)}, "
                f"B.shape={tuple(B.shape)}). The opus a16w16 launchers "
                f"assume stride_b_batch == N*K (see "
                f"csrc/opus_gemm/gen_instances.py), which is incompatible "
                f"with the batch_stride=0 view a B.unsqueeze(0)."
                f"expand(batch, -1, -1) would produce. Two valid fixes:\n"
                f"  1. Broadcast explicitly:  B = B.expand({batch}, -1, "
                f"-1).contiguous()\n"
                f"  2. Pass a real 3D weight: B with shape ({batch}, N, K)"
            )
        WQ = B.unsqueeze(0)  # batch == 1 here; kernel never reads stride_b_batch.
    elif B.dim() == 3:
        b_b, N, K_b = B.shape
        if K_b != K:
            raise ValueError(
                f"K dimension mismatch: A has K={K}, B has K={K_b}"
            )
        if b_b != batch:
            raise ValueError(
                f"B batch mismatch: A has batch={batch}, B has batch={b_b}"
            )
        # Reject expand-style broadcast views (batch_stride=0) up front. Any
        # other layout (contiguous, transposed N/K, etc.) is still rejected
        # below by the elements-per-row check; the launcher requires
        # B[b].stride(0) == N*K and B[b].stride(1) == K.
        bs0, bs1, bs2 = B.stride()
        if bs0 != N * K or bs1 != K or bs2 != 1:
            raise NotImplementedError(
                f"gemm_a16w16_opus: B must be a contiguous 3D tensor with "
                f"strides (N*K, K, 1) (got B.shape={tuple(B.shape)}, "
                f"B.stride()={tuple(B.stride())}). The opus launchers "
                f"hardcode stride_b_batch == N*K and stride_b == K; any "
                f"non-standard layout (broadcast view, transpose, slice) "
                f"will produce wrong results or a memory access fault. "
                f"Materialize via B = B.contiguous() first."
            )
        WQ = B
    else:
        raise ValueError(
            f"B must be 2D [N, K] or 3D [batch, N, K] (got shape "
            f"{tuple(B.shape)})"
        )

    if out is not None:
        Y = out
    else:
        Y = torch.empty(batch, M, N, dtype=dtype, device=A.device)

    return XQ, WQ, Y, M, N, K, batch, reshape_out_to_2d


def _finalize_output(Y: Tensor, reshape_out_to_2d: bool) -> Tensor:
    return Y.squeeze(0) if reshape_out_to_2d else Y


def gemm_a16w16_opus(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Shape-driven opus a16w16 GEMM.

    Parameters
    ----------
    A : [M, K] or [batch, M, K], bf16
    B : bf16 weight, plain layout (not pre-shuffled). Two accepted shapes:
        * [N, K]            -- requires batch == 1 (i.e. A is 2D, or A is
                               3D with leading dim 1).
        * [batch, N, K]     -- contiguous strides (N*K, K, 1) only.
                               Broadcast views (e.g. ``B.unsqueeze(0).
                               expand(batch, -1, -1)``) are rejected
                               because the opus launcher assumes
                               ``stride_b_batch == N*K``; pass
                               ``.contiguous()`` if you need to broadcast
                               a single-batch weight across A.
    bias : must be None (opus kernels have HAS_BIAS=false)
    dtype : output dtype, bf16 or fp32 (any kernel family supports either)
    kernelId : optional explicit override. When given, bypass CSV / C++
        dispatch and launch this specific tuned instance via
        opus_gemm_a16w16_tune.
    splitK : optional literal KBatch; only honored when kernelId is set.
    out : optional preallocated [batch, M, N] output; reused instead of
        allocating a fresh tensor.

    Returns
    -------
    Tensor with shape [M, N] when A was 2D, [batch, M, N] when A was 3D.
    """
    XQ, WQ, Y, M, N, K, batch, reshape_out_to_2d = _validate_and_reshape(
        A, B, bias, dtype, out
    )

    # 1) Explicit-kid override path.
    if kernelId is not None:
        opus_gemm_a16w16_tune(XQ, WQ, Y, int(kernelId), int(splitK or 0))
        return _finalize_output(Y, reshape_out_to_2d)

    # 2) Default path: opus-private tuned CSV lookup.
    cfg = _opus_common.lookup_tuned(
        M=M,
        N=N,
        K=K,
        bias=(bias is not None),
        dtype=A.dtype,
        outdtype=dtype,
        scaleAB=False,
        bpreshuffle=False,
    )
    if cfg is not None:
        kid = cfg["solidx"]
        # Both bf16 and fp32 Y are now valid for splitk kids (the reduce
        # kernel handles the cast / passthrough), so no Y.dtype gating is
        # needed here -- always honor the tuned winner.
        opus_gemm_a16w16_tune(XQ, WQ, Y, kid, int(cfg["splitK"]))
        return _finalize_output(Y, reshape_out_to_2d)

    # 3) CSV miss (or incompatible tuned winner): optionally record the
    #    shape and delegate to the C++ bf16 dispatcher, which is (or
    #    becomes, pending PR2' C++ changes) a lookup + heuristic branch.
    _opus_common.maybe_log_untuned_shape(
        M=M,
        N=N,
        K=K,
        bias=(bias is not None),
        dtype=A.dtype,
        outdtype=dtype,
        scaleAB=False,
        bpreshuffle=False,
    )
    _opus_gemm_bf16_dispatch(XQ, WQ, Y, None, None, None)
    return _finalize_output(Y, reshape_out_to_2d)


__all__ = ["opus_gemm_a16w16_tune", "gemm_a16w16_opus"]
