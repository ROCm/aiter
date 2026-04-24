# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 Python user-facing API.

Two entry points:

* `opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)`
  Low-level pybind binding to the id-based tune dispatcher. Exposes a
  specific kernel instance by `kernelId` plus optional literal KBatch
  via `splitK`. Intended for the tuner, for debugging a specific kid,
  and for aiter-global integrations (e.g. future tuned_gemm.solMap).

* `gemm_a16w16_opus(A, B, bias=None, dtype=bf16, *, kernelId=None, splitK=None, out=None)`
  High-level shape-driven API. The typical user writes
  `gemm_a16w16_opus(A, B)` and never sees a kid number. Internally:

    1. Reshape A/B to 3D, allocate Y, validate (bias is currently
       unsupported by opus; bpreshuffle and non-bf16 A/B likewise).
    2. If `kernelId` is given explicitly -> opus_gemm_a16w16_tune.
    3. Otherwise query opus-private tuned CSV via aiter.ops.opus.common;
       on hit -> opus_gemm_a16w16_tune with the tuned (solidx, splitK).
    4. On miss -> optionally autolog the shape
       (AITER_OPUS_LOG_UNTUNED=1) and fall through to
       `deepgemm_opus(XQ, WQ, Y, ...)`, i.e. the C++ entry `opus_gemm`,
       whose bf16 branch does its own lookup + heuristic dispatch (see
       csrc/opus_gemm/opus_gemm.cu).

Both entries share the JIT module `module_deepgemm_opus`.
"""

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from . import common as _opus_common
from .deepgemm import deepgemm_opus as _opus_gemm_cpp_entry


# ---- Low-level pybind binding --------------------------------------------


def _gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    gen_fake=_gen_opus_gemm_a16w16_tune_fake_tensors,
)
def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


# ---- High-level shape-driven API -----------------------------------------

# splitk kids (200..299) require bf16 output; their C++ traits static_assert
# D_C==float and the reduce kernel casts fp32 workspace down to bf16 Y. The
# wrapper surfaces this as a clear Python exception before launch when we
# know the tuned CSV picked a splitk kid but user requested fp32 Y.
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
    if B.dim() != 2:
        raise ValueError(
            f"B must be 2D [N, K] (got shape {tuple(B.shape)}); opus assumes "
            "B is already in plain [N, K] layout."
        )
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

    N, K_b = B.shape
    if K_b != K:
        raise ValueError(
            f"K dimension mismatch: A has K={K}, B has K={K_b}"
        )

    WQ = B.unsqueeze(0).expand(batch, -1, -1) if batch > 1 else B.unsqueeze(0)

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
    B : [N, K], bf16 (plain layout; not pre-shuffled)
    bias : must be None (opus kernels have HAS_BIAS=false)
    dtype : output dtype, bf16 or fp32; splitk kids force bf16
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
        if _SPLITK_KID_MIN <= kid <= _SPLITK_KID_MAX and dtype is not torch.bfloat16:
            # Tuned winner is a splitk kid but user asked for fp32 Y --
            # splitk's fp32 workspace + bf16 Y cast hard-codes bf16. Fall
            # through to C++ dispatch rather than picking a bad kid.
            pass
        else:
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
    _opus_gemm_cpp_entry(XQ, WQ, Y, None, None, None)
    return _finalize_output(Y, reshape_out_to_2d)


__all__ = ["opus_gemm_a16w16_tune", "gemm_a16w16_opus"]
