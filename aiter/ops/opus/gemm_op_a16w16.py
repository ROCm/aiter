# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 Python user-facing API.

Public entry points:

* `gemm_a16w16_opus(A, B, bias=None, dtype=bf16, *, kernelId=None, splitK=None, out=None)`
  Shape-driven wrapper. The typical user writes `gemm_a16w16_opus(A, B)`
  and never sees a kid number. Internal path:

    1. Reshape A/B to 3D, allocate Y, validate (bias allowed across the
       split-barrier / splitk kid families; bpreshuffle and non-bf16 A/B
       unsupported).
    2. If `kernelId` is given explicitly -> opus_gemm_a16w16_tune (bias
       is forwarded; the C++ dispatcher rejects non-bias-aware kids).
    3. Otherwise query the global aiter BF16 tuned CSVs via
       aiter.ops.opus.common (filtered by `libtype == 'opus'`, key
       includes bias=True/False); on hit -> opus_gemm_a16w16_tune
       with the tuned (solidx, splitK).
    4. On miss -> fall through to the private bf16 no-scale binding
       `_opus_gemm_bf16_dispatch`, which forwards to the C++ entry
       `opus_gemm` whose bf16 branch does its own lookup + heuristic
       dispatch (see csrc/opus_gemm/opus_gemm.cu). bias is forwarded
       through this path: the C++ entry skips its bias-agnostic lookup
       map when bias is present and goes straight to the heuristic
       dispatcher (which always returns a bias-aware kid).

* `opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK)`
  Low-level pybind binding to the id-based tune dispatcher. Exposes a
  specific kernel instance by `kernelId` plus optional literal KBatch
  via `splitK` and an optional bias tensor (D_OUT-typed, [N] or
  [batch, N]; F.linear convention). Intended for the tuner, for debugging a specific kid,
  and for aiter-global integrations (e.g. future tuned_gemm.solMap).

All entry points share the JIT module `module_deepgemm_opus`, which
still hosts bindings for other opus kernel families (a8w8 etc.). The
Python surface is deliberately per-dtype: a16w16 here, a8w8 in its own
module when that lands.
"""

import logging
from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from . import common as _opus_common

logger = logging.getLogger("aiter")

# ---- Low-level pybind bindings --------------------------------------------


def _gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
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
    develop=True,
)
def _opus_gemm_a16w16_tune_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
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

    # XQ / WQ: the K (innermost / contraction) dimension may be padded -- the
    # launcher passes the tensor's real leading stride as kargs.stride_a/stride_b
    # and the kernels use it as the lda for BOTH addressing and the gmem buffer
    # bound, so a row pitch > K (e.g. a 2880-wide tensor stored at lda 3072) is
    # served correctly. We only require:
    #   * innermost stride == 1   (the kernel layout hardcodes the K stride to 1)
    #   * row pitch (stride[1]) >= K
    #   * batch stride == rows * row pitch (or batch == 1) -- rejects broadcast
    #     (stride 0) and transposed / overlapping views.
    for name, t, rows in (("XQ", XQ, M), ("WQ", WQ, N)):
        s0, s1, s2 = t.stride()
        k_inner = t.shape[2]
        ok = s2 == 1 and s1 >= k_inner and (batch == 1 or s0 == rows * s1)
        if not ok:
            raise NotImplementedError(
                f"opus_gemm_a16w16_tune: {name} must be K-contiguous with an "
                f"optional padded leading dim -- need stride[2]==1, "
                f"stride[1]>={k_inner}, and stride[0]==size(1)*stride[1] (or "
                f"batch==1). Got {name}.stride()={tuple(t.stride())}, "
                f"{name}.shape={tuple(t.shape)}. Broadcast / transpose / "
                f"non-K-contiguous slices are not supported; materialize with "
                f"`{name} = {name}.contiguous()` before calling."
            )
    # Y is the output: the launcher hardcodes stride_c == N and
    # stride_c_batch == M*N, so it must be fully contiguous.
    y_want = (M * N, N, 1)
    if tuple(Y.stride()) != y_want:
        raise NotImplementedError(
            f"opus_gemm_a16w16_tune: Y must have contiguous strides {y_want} "
            f"(got Y.stride()={tuple(Y.stride())}, Y.shape={tuple(Y.shape)}). "
            f"The launcher hardcodes stride_c == N and stride_c_batch == M*N; "
            f"materialize with `Y = Y.contiguous()` before calling."
        )


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias=None,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    """Low-level id-based dispatcher (Python guard + C++ launch).

    See module docstring. This Python wrapper checks XQ/WQ/Y layout up
    front (rejecting broadcast / transpose / slice views that the C++
    kernel would happily run with garbage data); on success it forwards
    to the underlying pybind binding.

    Parameters
    ----------
    bias : optional D_OUT-typed bias tensor, accepted shapes:
           [M] (broadcast across batch; requires batch==1) or [batch, M].
           Only honored on bias-aware kid ranges (split-barrier kid 4..9
           and a16w16_flatmm_splitk kid 200..299); the C++ dispatcher
           rejects bias on other kids.

    Backwards-compatibility note
    ----------------------------
    Older callers used ``opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)``
    with positional args (no bias slot). When the 4th positional argument
    is an int, we silently treat it as kernelId and shift remaining args
    accordingly so existing tuner / test scripts keep working without an
    edit. Mixed-style calls (``..., bias=t, kernelId=k``) keep their kwargs
    semantics.
    """
    # Positional-int back-compat: opus_gemm_a16w16_tune(XQ, WQ, Y, kid, splitK).
    # When `bias` arrives as an int (which torch_library would otherwise
    # reject as not Optional[Tensor]), reinterpret as kernelId.
    if isinstance(bias, int) and not isinstance(bias, bool):
        # Positional int means "this was meant to be kernelId"; treat the
        # next positional (kernelId) as splitK and the original splitK
        # (default 0) as truly unset.
        if splitK != 0 and kernelId == 0:
            # Shouldn't happen in old call sites, but be defensive.
            new_splitK = splitK
        else:
            new_splitK = kernelId
        kernelId = bias
        splitK = new_splitK
        bias = None
    _check_a16w16_tune_layout(XQ, WQ, Y)
    # Mono-tile kid guard: the launcher requires N / K to be tile-aligned
    # (the kernel has no N-tail mask and no K-tail mask; M-tail IS handled
    # via the bounded gmem desc). A CSV winner picked through
    # tuned_gemm.get_padded_m can surface a mono kid whose B_N / B_K does
    # not divide the actual N / K -- the launcher would AITER_CHECK abort
    # the process. Reroute to opus's own bf16 heuristic dispatch instead;
    # it never returns a mono kid, so it always picks something that can
    # run the shape.
    _, _, N = Y.shape
    _, _, K = XQ.shape
    if not _opus_common.mono_kid_shape_ok(kernelId, N, K):
        logger.warning(
            "opus_gemm_a16w16_tune: mono-tile kid %d requires N/K aligned "
            "to its tile; got N=%d K=%d -- rerouting to opus bf16 heuristic.",
            kernelId,
            N,
            K,
        )
        _opus_gemm_bf16_dispatch(XQ, WQ, Y, None, None, None, bias)
        return Y
    # C++ launcher is in-place on Y (returns void after PR #2932-style
    # refactor to aiter_tensor_t). Keep the wrapper's `return Y`
    # contract so callers that did `Y = opus_gemm_a16w16_tune(...)`
    # still see the populated Y.
    _opus_gemm_a16w16_tune_raw(XQ, WQ, Y, bias, kernelId, splitK)
    return Y


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
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm",
    gen_fake=_gen_opus_gemm_bf16_dispatch_fake_tensors,
    develop=True,
)
def _opus_gemm_bf16_dispatch(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group_layout: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
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
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")
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
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")
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
            f"B must be 2D [N, K] or 3D [batch, N, K] (got shape " f"{tuple(B.shape)})"
        )

    if out is not None:
        Y = out
    else:
        Y = torch.empty(batch, M, N, dtype=dtype, device=A.device)

    # Bias validation. Bias may be fp32 OR match the output dtype: the gfx1250
    # splitk main kernel always writes an fp32 workspace and the reduce kernel
    # folds bias in fp32 before the final cast to Y, so an fp32 bias is exact
    # and free regardless of Y dtype (the common accuracy-friendly case for a
    # bf16 output). Bias is per-output-feature [N] (F.linear convention):
    #   * [N]          -> stride_bias_batch = 0 (broadcast across batch)
    #   * [batch, N]   -> stride_bias_batch = N
    # Matches the C++-side gfx1250 bias validation in gen_instances_gfx1250.py.
    if bias is not None:
        if bias.dtype not in (dtype, torch.float32):
            raise ValueError(
                f"gemm_a16w16_opus: bias dtype must be fp32 or match output "
                f"dtype (got bias.dtype={bias.dtype}, dtype={dtype})"
            )
        if not bias.is_contiguous():
            raise ValueError(
                f"gemm_a16w16_opus: bias must be contiguous (got "
                f"bias.stride()={tuple(bias.stride())})"
            )
        if bias.dim() == 1:
            if bias.shape[0] != N:
                raise ValueError(
                    f"gemm_a16w16_opus: 1D bias length must equal N (got "
                    f"bias.shape={tuple(bias.shape)}, N={N})"
                )
        elif bias.dim() == 2:
            if tuple(bias.shape) != (batch, N):
                raise ValueError(
                    f"gemm_a16w16_opus: 2D bias must be [batch, N] (got "
                    f"bias.shape={tuple(bias.shape)}, batch={batch}, N={N})"
                )
        else:
            raise ValueError(
                f"gemm_a16w16_opus: bias must be 1D [N] or 2D [batch, N] "
                f"(got bias.shape={tuple(bias.shape)})"
            )

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
    bias : optional per-output-feature bias (F.linear convention), dtype
        must equal `dtype` (match_d_out). Accepted shapes:
        * [N]                  -- broadcast across batch.
        * [batch, N]           -- per-batch bias vector.
        bias is consumed by the a16w16 split-barrier (kid 4..9) and the
        a16w16_flatmm_splitk (kid 200..299) families. CSV-miss requests
        with bias fall back to the C++ heuristic dispatcher (which only
        returns bias-aware kids), so any (M, N, K) is supported even
        without a tuned bias-aware winner -- accuracy is preserved at
        whatever the heuristic kid achieves; performance may not be
        optimal until the shape is re-tuned with `--bias`.
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

    # 1) Explicit-kid override path. The C++ dispatcher gates non-bias-aware
    #    kids when bias is present, so we just forward.
    if kernelId is not None:
        opus_gemm_a16w16_tune(XQ, WQ, Y, bias, int(kernelId), int(splitK or 0))
        return _finalize_output(Y, reshape_out_to_2d)

    # 2) Default path: opus-private tuned CSV lookup. lookup_tuned() keys
    #    on bias=True/False as part of its 9-column tuple, so bias=True
    #    only matches rows that were tuned with the bias path. CSV miss on
    #    bias=True falls through to the explicit error below; we never
    #    silently route bias to the no-bias fallback.
    cfg = _opus_common.lookup_tuned(
        M=M,
        N=N,
        K=K,
        bias=(bias is not None),
        dtype=A.dtype,
        outdtype=dtype,
        scaleAB=False,
        bpreshuffle=False,
        batch=batch,
    )
    if cfg is not None:
        kid = cfg["solidx"]
        # Both bf16 and fp32 Y are now valid for splitk kids (the reduce
        # kernel handles the cast / passthrough), so no Y.dtype gating is
        # needed here -- always honor the tuned winner.
        opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kid, int(cfg["splitK"]))
        return _finalize_output(Y, reshape_out_to_2d)

    # 3) CSV miss: fall through to the C++ heuristic dispatcher via
    #    opus_gemm. Bias is forwarded through; the C++ entry skips its
    #    bias-agnostic lookup map when bias is present and routes
    #    directly to the heuristic (which only ever returns bias-aware
    #    split-barrier / splitk kids).
    #
    #    (Note: this used to call `_opus_common.maybe_log_untuned_shape`
    #    to autolog the missed shape to a private CSV for offline tuning.
    #    The autolog feature has been removed -- collect untuned shapes
    #    via gradlib's standard --input_file flow instead.)
    _opus_gemm_bf16_dispatch(XQ, WQ, Y, None, None, None, bias)
    return _finalize_output(Y, reshape_out_to_2d)


# Per-stream splitk workspace init. Call once inside `with torch.cuda.stream(s):`
# (eagerly, before HIP graph capture) to register a workspace handle for that
# stream. Needed under vLLM/sglang-style TBO where two CPU threads drive two
# streams concurrently -- each captured graph must bake in its own buffer
# pointer; the prior thread_local cache would fail capture on the second
# stream. After init, run the largest expected gemm eagerly on the same
# stream to grow the buffer, then capture.
@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_workspace_init", develop=True)
def opus_gemm_workspace_init() -> None: ...


def _gen_opus_gemm_a16w16_bhsd_fake_tensors(
    A: torch.Tensor,
    W: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_bhsd",
    gen_fake=_gen_opus_gemm_a16w16_bhsd_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_bhsd_raw(
    A: torch.Tensor,
    W: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


def _gen_opus_gemm_a16w16_mmajor_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 208,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


def _gen_opus_gemm_a8w8_scale_mmajor_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return Y


# mmajor fp8 block-scale raw binding: O(A)/Y are [M, batch, *] (dim0=M,
# dim1=batch), x_scale [M, batch, K/GROUP_K] (per-token M); wo_a + w_scale stay
# batch-major. Zero-copy DSV4 wo_a fp8 (no caller-side transpose). Y is fp32.
# See csrc/opus_gemm/opus_gemm.cu :: opus_gemm_a8w8_scale_mmajor.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_scale_mmajor",
    gen_fake=_gen_opus_gemm_a8w8_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_scale_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor: ...


def _gen_opus_gemm_uniform_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor:
    return Y


# batch-major fp8 block-scale UNIFORM raw binding. O/wo_a/Y are
# [batch,M,K]/[batch,N,K]/[batch,M,N]; x_scale [batch,M,K/GROUP_K],
# w_scale [batch,N/GROUP_N,K/GROUP_K]. Y fp32 or bf16.
# See csrc/opus_gemm/opus_gemm.cu :: opus_gemm_uniform_scale.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_uniform_scale",
    gen_fake=_gen_opus_gemm_uniform_scale_fake_tensors,
    develop=True,
)
def _opus_gemm_uniform_scale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor: ...


def _gen_opus_gemm_uniform_scale_mmajor_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor:
    return Y


# mmajor fp8 block-scale UNIFORM (Route B fp8, 4-wave full-tile, direct store)
# raw binding. Same layout contract as _opus_gemm_a8w8_scale_mmajor_raw;
# kernelId selects the uniform_scale tile (700=128x128, 701=256x128). Y is fp32.
# See csrc/opus_gemm/opus_gemm.cu :: opus_gemm_uniform_scale_mmajor.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_uniform_scale_mmajor",
    gen_fake=_gen_opus_gemm_uniform_scale_mmajor_fake_tensors,
    develop=True,
)
def _opus_gemm_uniform_scale_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 700,
) -> torch.Tensor: ...


# mmajor raw binding: A(O)/Y are read with dim0=M, dim1=batch, so
# batch-in-the-middle layouts need no caller-side transpose/permute. Shared by
# wo_a_gemm_opus (DSV4, A=[num_tokens, n_local_groups, K]) and
# batch_gemm_a16w16_bshd_opus. See csrc/opus_gemm/opus_gemm.cu ::
# opus_gemm_a16w16_mmajor.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_mmajor",
    gen_fake=_gen_opus_gemm_a16w16_mmajor_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_mmajor_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 208,
    splitK: int = 0,
) -> torch.Tensor: ...


def batch_gemm_a16w16_bhsd_opus(
    A: Tensor,
    W: Tensor,
    heads_per_group: int,
    *,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
    out: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """BHSD-layout batch GEMM for MLA output projection.

    Fuses the BHSD->BSHD transpose into the A-matrix address calculation,
    avoiding a full HBM read-write pass for the transpose.

    Parameters
    ----------
    A : [B, H, S, D] bf16 -- attention output in BHSD layout.
        H = n_groups * heads_per_group, D = head_dim.
    W : [G, R, K] bf16 -- wo_a weight reshaped to group form.
        G = n_groups, R = o_lora_rank, K = heads_per_group * head_dim.
    heads_per_group : int -- number of heads per group (e.g. 8 for DSv4).

    Returns
    -------
    Tensor [B, S, G, R] -- the output projection result.
    """
    assert A.ndim == 4, f"A must be 4D [B, H, S, D], got {A.shape}"
    assert W.ndim == 3, f"W must be 3D [G, R, K], got {W.shape}"
    B, H, S, D = A.shape
    G = W.shape[0]
    R = W.shape[1]
    K = W.shape[2]
    assert H == G * heads_per_group, f"H={H} must equal G*hpg={G}*{heads_per_group}"
    assert K == heads_per_group * D, f"K={K} must equal hpg*D={heads_per_group}*{D}"

    # Reshape A: [B, H, S, D] -> [B*G, hpg, S, D]
    A_reshaped = A.view(B, G, heads_per_group, S, D).permute(0, 1, 2, 3, 4)
    A_reshaped = A_reshaped.reshape(B * G, heads_per_group, S, D).contiguous()

    # Expand W: [G, R, K] -> [B*G, R, K]
    W_expanded = W.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * G, R, K).contiguous()

    # Allocate output
    if out is not None:
        assert out.shape == (B, S, G, R), f"out shape must be [{B},{S},{G},{R}]"
        Y = out.view(B * G, S, R)
    else:
        Y = torch.empty(B * G, S, R, dtype=dtype, device=A.device)

    # Default kid: 608 (64x64x128 WG=1 splitk) -- good for decode M<=4
    kid = kernelId if kernelId is not None else 608
    sk = splitK if splitK is not None else 0

    _opus_gemm_a16w16_bhsd_raw(A_reshaped, W_expanded, Y, kid, sk)

    return Y.view(B, G, S, R).permute(0, 2, 1, 3)  # [B, S, G, R]


def batch_gemm_a16w16_bshd_opus(
    A: Tensor,
    W: Tensor,
    heads_per_group: int,
    *,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
    out: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    use_standard_pipeline: bool = True,
) -> Tensor:
    """BSHD-layout batch GEMM for MLA output projection.

    A is in BSHD layout [B, S, H, D] (DSv4 default from sparse_attn).
    Avoids the full HBM transpose by passing strided views to the kernel.

    Two pipeline paths (selected by use_standard_pipeline):
      True  -- standard a16w16 batch GEMM (kids 4-9 / 200-223): constructs
              a strided 3D view [G, S, K] where K=hpg*D is contiguous within
              each group. No head remapping needed.
      False -- BHSD a_offset remapping pipeline (kids 600-655): constructs a
              strided 4D view [G, hpg, S, D]. The kernel's a_offset does
              div/mod to remap tile_k, but for BSHD this degenerates to an
              identity (h*D+d == k_abs). Slightly more overhead per tile.

    Parameters
    ----------
    A : [B, S, H, D] bf16 -- attention output in BSHD layout.
    W : [G, R, K] bf16 -- wo_a weight.
    heads_per_group : int -- number of heads per group (e.g. 8 for DSv4).
    use_standard_pipeline : bool -- True uses standard kids, False uses BHSD
        a_offset remapping kids. Default True (simpler, no div/mod).

    Returns
    -------
    Tensor [B, S, G, R] -- the output projection result.
    """
    assert A.ndim == 4, f"A must be 4D [B, S, H, D], got {A.shape}"
    assert W.ndim == 3, f"W must be 3D [G, R, K], got {W.shape}"
    B, S, H, D = A.shape
    G = W.shape[0]
    R = W.shape[1]
    K = W.shape[2]
    assert H == G * heads_per_group, f"H={H} must equal G*hpg={G}*{heads_per_group}"
    assert K == heads_per_group * D, f"K={K} must equal hpg*D={heads_per_group}*{D}"

    if out is not None:
        assert out.shape == (B, S, G, R), f"out shape must be [{B},{S},{G},{R}]"
        Y_full = out
    else:
        Y_full = torch.empty(B, S, G, R, dtype=dtype, device=A.device)

    hpg = heads_per_group

    # flatmm_splitk (200-299, +1000 nooob mirror) and bhsd_splitk (600-649,
    # +1000 nooob mirror) reduce kernels now take explicit stride_c /
    # stride_c_batch (see gen_flatmm_splitk_instance / gen_bhsd_splitk_instance
    # in gen_instances_gfx950.py + splitk_reduce_gfx950.cuh), so they can write
    # straight into a *strided* Y_full view -- no temp Y_b + transpose-copy
    # needed. Every other a16w16 kid family's launcher still hardcodes
    # stride_c=N / stride_c_batch=M*N (ignores Y's real strides), so passing a
    # strided view there would silently corrupt the output; keep the safe
    # temp-buffer path for those.
    def _kid_supports_strided_y(kid: int, splitk_min: int, splitk_max: int) -> bool:
        base = kid % 1000
        return splitk_min <= base < splitk_max

    if use_standard_pipeline:
        # Standard a16w16 batch GEMM via the mmajor primitive (same clean path
        # as wo_a_gemm_opus). A[b].view(S, G, K) is already [M=S, batch=G, K]
        # and Y_full[b] is [M=S, batch=G, N=R] -- exactly the mmajor layout
        # (dim0=M, dim1=batch) the launcher reads via strides. So NO permute to
        # [G, S, K], NO temp buffer, NO Python-layout-check bypass: the mmajor
        # dispatch reads A/Y strides directly for every kid (incl. the splitk
        # reduce, which writes the strided [S, G, R] Y in place).
        kid = kernelId if kernelId is not None else 208
        sk = splitK if splitK is not None else 0

        for b in range(B):
            A_b = A[b].view(S, G, K)  # [S, G, K] = [M, batch, K], zero-copy
            Y_b = Y_full[b]  # [S, G, R] = [M, batch, N], in-place slice
            _opus_gemm_a16w16_mmajor_raw(A_b, W, Y_b, kid, sk)
    else:
        # BHSD a_offset remapping path: pass 4D strided view [G, hpg, S, D].
        kid = kernelId if kernelId is not None else 608
        sk = splitK if splitK is not None else 0
        direct_write = _kid_supports_strided_y(kid, 600, 650)

        for b in range(B):
            A_b = A[b].view(S, G, hpg, D).permute(1, 2, 0, 3)  # [G, hpg, S, D]

            if direct_write:
                Y_b = Y_full[b].permute(1, 0, 2)  # [G, S, R] view, no copy
                _opus_gemm_a16w16_bhsd_raw(A_b, W, Y_b, kid, sk)
            else:
                Y_b = torch.empty(G, S, R, dtype=dtype, device=A.device)
                _opus_gemm_a16w16_bhsd_raw(A_b, W, Y_b, kid, sk)
                Y_full[b] = Y_b.permute(1, 0, 2)  # [S, G, R]

    return Y_full


def wo_a_gemm_opus(
    o: Tensor,
    wo_a: Tensor,
    *,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
    out: Optional[Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """DeepSeek-V4 grouped output-LoRA GEMM -- direct opus replacement for
    the large-num_tokens branch of ATOM/atom/models/deepseek_v4.py:

        o = o.view(num_tokens, n_local_groups, -1)
        wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
        ...
        o = torch.einsum("sgd,grd->sgr", o, wo_a)

    `o`/`wo_a`/the returned tensor are passed through in exactly this shape
    -- no `.transpose()`/`.permute()`/`.contiguous()` at the Python level at
    all, matching how `torch.einsum` consumes `o` natively (batch axis
    `n_local_groups` in the middle, not outermost). The transpose is instead
    fused into the launcher: an "_mmajor" variant reads O/Y with
    dim0=M(num_tokens)/dim1=batch(n_local_groups) -- the opposite of the
    regular launcher's dim0=batch/dim1=M convention -- so it addresses
    memory correctly without the caller ever materializing a transposed
    view. See gen_flatmm_splitk_instance / gen_noscale_instance_gfx950's
    "_mmajor" emits (gen_instances_gfx950.py), opus_gemm.cu ::
    opus_gemm_a16w16_mmajor, and opus_gemm_arch_gfx950.cuh ::
    opus_a16w16_tune_dispatch_mmajor_gfx950.

    Default kid picks between two "_mmajor" families by num_tokens (both
    verified correct; ATT-profiled on MI355X to pick the empirical winner --
    see yzhou_agent chat history for the sweep):
      * kid 9  (a16w16 split-barrier, non-splitk, 512x256x256x64): wins
        decisively from num_tokens ~384 up (e.g. 98us vs kid 208's 151us at
        T=1024; 456us vs 1541us at T=8192) -- its plain double-buffered
        pipeline pays far fewer explicit s_barrier syncs per K-iter than
        flatmm_splitk's producer/consumer warp-specialization, and that gap
        widens with M.
      * kid 208 (a16w16_flatmm_splitk, warp-specialized, 64x64x128 WG=1):
        wins for smaller num_tokens (e.g. 24us vs kid 9's 72us at T=64) --
        kid 9's large 256x256 output tile is underfilled/wasteful below
        ~256-384 tokens.
    Neither fully matches hipBLASLt (`torch.einsum`'s backing GEMM): kid 9
    lands at roughly 1.15-1.4x hipBLASLt's time in the range it wins,
    narrowing as num_tokens grows. Pass `kernelId=` explicitly to override.

    Parameters
    ----------
    o : [num_tokens, n_local_groups, K] bf16 -- already grouped/flattened
        attention output (K = n_heads * head_dim // o_groups), i.e. exactly
        `o.view(num_tokens, n_local_groups, -1)` from the model.
    wo_a : [n_local_groups, o_lora_rank, K] bf16 -- wo_a weight, already
        grouped, i.e. `self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)`.
    out : optional preallocated [num_tokens, n_local_groups, o_lora_rank]
        output; reused instead of allocating a fresh tensor.

    Returns
    -------
    Tensor [num_tokens, n_local_groups, o_lora_rank] -- same layout
    `torch.einsum("sgd,grd->sgr", o, wo_a)` returns.
    """
    assert o.ndim == 3, f"o must be 3D [num_tokens, n_local_groups, K], got {o.shape}"
    assert (
        wo_a.ndim == 3
    ), f"wo_a must be 3D [n_local_groups, o_lora_rank, K], got {wo_a.shape}"
    T, G, K = o.shape
    G_w, R, K_w = wo_a.shape
    assert (
        G == G_w and K == K_w
    ), f"o/wo_a group or K mismatch: o={tuple(o.shape)} wo_a={tuple(wo_a.shape)}"

    # Per-T kernel + split-K selection, tuned on MI355X for the DSV4 wo_a shape
    # (N=o_lora_rank, K=heads_per_group*head_dim, batch=n_local_groups). Stable
    # run_perftest numbers, see docs/dsv4_wo_a_opus_gemm_optimization.md.
    #
    # Latest mmajor tuner sweep (batch=8, N=1024, K=4096):
    #   T=16:    kid200/sk4
    #   T=32:    kid206/sk2
    #   T=64:    kid200/sk4
    #   T=128:   kid1404
    #   T=256:   kid1403
    #   T=512..2048: kid1401
    #   T>=4096: kid9
    # mono_tile is no-OOB in N/K (requires R%tile_N==0 and K%64==0), but M-tail
    # is handled by bounded buffer descriptors. DSV4 wo_a satisfies N=1024,
    # K=4096, so mono_tile can be the default for T>=128. Fall back to the
    # previous persistent/split-barrier choices if a caller supplies an
    # incompatible N/K.
    auto_sk = 0
    if kernelId is not None:
        kid = kernelId
    else:
        cfg = _opus_common.lookup_tuned(
            T,
            R,
            K,
            False,
            o.dtype,
            dtype,
            False,
            False,
            batch=G,
        )
        if cfg is not None and _opus_common.mono_kid_shape_ok(cfg["solidx"], R, K):
            kid = int(cfg["solidx"])
            auto_sk = int(cfg["splitK"])
        else:
            kid = None

    if kid is None and T < 24:
        kid = 200
        auto_sk = 4
    elif kid is None and T < 48:
        kid = 206
        auto_sk = 2
    elif kid is None and T < 128:
        kid = 200
        auto_sk = 4
    elif kid is None and T < 192 and _opus_common.mono_kid_shape_ok(1404, R, K):
        kid = 1404  # mono_tile 64x128x64, best at T=128
    elif kid is None and T < 512 and _opus_common.mono_kid_shape_ok(1403, R, K):
        kid = 1403  # mono_tile 128x128x64, best at T=256
    elif kid is None and T < 4096 and _opus_common.mono_kid_shape_ok(1401, R, K):
        kid = 1401  # mono_tile 128x256x64, best at T=512..2048
    elif kid is None and T <= 1024:
        kid = 301  # persistent fallback for non-mono-compatible N/K
    elif kid is None:
        kid = 9  # large-M split-barrier fallback for non-mono-compatible N/K
    sk = splitK if splitK is not None else auto_sk

    if out is not None:
        assert out.shape == (T, G, R), f"out shape must be [{T},{G},{R}]"
        y = out
    else:
        y = torch.empty(T, G, R, dtype=dtype, device=o.device)

    _opus_gemm_a16w16_mmajor_raw(o, wo_a, y, kid, sk)

    return y


__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
    "opus_gemm_workspace_init",
]
