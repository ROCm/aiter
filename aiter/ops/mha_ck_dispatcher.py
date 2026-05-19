# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Opt-in FMHA path that routes attention through the Composable Kernel
**dispatcher** (3rdparty/composable_kernel/dispatcher), instead of AITER's
existing CK codegen / ASM dispatch in ``aiter/ops/mha.py``.

Purpose: enable A/B comparison between AITER's current FMHA selection and the
new dispatcher-based selection. This module does **not** modify any existing
AITER behaviour; users must import this module explicitly.

The dispatcher provides its own Python tooling that performs codegen,
JIT-compilation, caching, and execution:

    3rdparty/composable_kernel/dispatcher/python/fmha_utils.py
        - FmhaKernelSpec / FmhaKernelConfig
        - spec_to_config
        - FmhaRegistry              (multi-kernel registry; parallel build)
        - setup_fmha_dispatcher     (single-kernel: codegen + compile + load)
        - FmhaRunner                (.run(Q, K, V, prob, mask_type=...))
        - FmhaProblem               (problem shape descriptor)

This wrapper picks a single sensible kernel preset, builds it once (cached on
disk by the dispatcher), and exposes a function whose signature mirrors the
positional shape of ``aiter.flash_attn_func`` for forward attention only.

Limitations of this first cut (intentional):
    * forward only, non-varlen
    * no bias / alibi / dropout / sliding-window
    * FirstFit selection (no FMHA heuristic model is shipped today)
    * dtype: fp16, bf16
    * single tile preset (128x128, k0=32, qr_async pipeline)
    * data round-trips host<->device through numpy (the dispatcher's
      FmhaRunner does its own hipMalloc/hipMemcpy)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Locate and import the CK dispatcher's Python tooling.
# --------------------------------------------------------------------------- #
_AITER_ROOT = Path(__file__).resolve().parents[2]
_DISPATCHER_ROOT = _AITER_ROOT / "3rdparty" / "composable_kernel" / "dispatcher"
_DISPATCHER_PY = _DISPATCHER_ROOT / "python"

if not _DISPATCHER_PY.is_dir():
    raise ImportError(
        f"CK dispatcher python package not found at {_DISPATCHER_PY}. "
        "Make sure the composable_kernel submodule is checked out."
    )

if str(_DISPATCHER_PY) not in sys.path:
    sys.path.insert(0, str(_DISPATCHER_PY))

# These imports come from the dispatcher's own python/ folder.
from fmha_utils import (  # noqa: E402
    FmhaKernelSpec,
    FmhaProblem,
    FmhaSetupResult,
    detect_gpu_arch,
    setup_fmha_dispatcher,
    spec_to_config,
)

__all__ = [
    "flash_attn_func_ck_dispatcher",
    "DispatcherBuildError",
    "clear_kernel_cache",
]


class DispatcherBuildError(RuntimeError):
    """Raised when the CK dispatcher fails to codegen/compile a kernel."""


# --------------------------------------------------------------------------- #
# dtype + mask mapping
# --------------------------------------------------------------------------- #
_TORCH_TO_DISPATCHER_DTYPE = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# Matches ck_tile mask_enum (see
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/mask.hpp):
#   0 = no_mask, 1 = mask_top_left, 2 = mask_bottom_right, 3 = window_generic
_MASK_NO = 0
_MASK_CAUSAL_TOP_LEFT = 1


# --------------------------------------------------------------------------- #
# Kernel preset + cache
# --------------------------------------------------------------------------- #
# Cache of built kernels keyed by (dtype_str, hdim_q, hdim_v, arch).
# Each entry is a successfully-built FmhaSetupResult whose .runner is alive.
_KERNEL_CACHE: dict = {}


def _default_spec(hdim: int) -> FmhaKernelSpec:
    """Return a single sensible default kernel spec for the given head dim.

    Chosen to match the CK dispatcher's `01_basic_fmha.py` example happy path.
    Users can extend this module later to enumerate more specs.
    """
    return FmhaKernelSpec(
        name=f"aiter_fwd_hd{hdim}_128x128_k32",
        hdim=hdim,
        pipeline="qr_async",
        tile_m0=128,
        tile_n0=128,
        tile_k0=32,
    )


def _get_runner(dtype_str: str, hdim: int, arch: str) -> FmhaSetupResult:
    """Build (or fetch from cache) a dispatcher runner for this configuration."""
    key = (dtype_str, hdim, hdim, arch)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None and cached.success and cached.runner is not None:
        return cached

    spec = _default_spec(hdim)
    cfg = spec_to_config(spec, dtype_str, arch)

    setup = setup_fmha_dispatcher(cfg, verbose=False)
    if not setup.success or setup.runner is None:
        raise DispatcherBuildError(
            f"CK dispatcher failed to build kernel '{spec.name}' "
            f"(dtype={dtype_str}, hdim={hdim}, arch={arch}): {setup.error}"
        )

    _KERNEL_CACHE[key] = setup
    return setup


def clear_kernel_cache() -> None:
    """Release any cached FmhaRunner instances."""
    for setup in _KERNEL_CACHE.values():
        runner = getattr(setup, "runner", None)
        if runner is not None and hasattr(runner, "cleanup"):
            try:
                runner.cleanup()
            except Exception:
                pass
    _KERNEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def flash_attn_func_ck_dispatcher(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    arch: Optional[str] = None,
) -> Tuple[torch.Tensor, float]:
    """Forward attention via the CK dispatcher.

    Signature mirrors the positional shape of ``aiter.flash_attn_func`` for the
    forward pass only. Unsupported features (dropout, bias, alibi, SWA, varlen,
    return_attn_probs, backward) are intentionally absent.

    Args:
        q: [batch, seqlen_q, nhead_q, hdim] -- BSHD, fp16 or bf16
        k: [batch, seqlen_k, nhead_k, hdim]
        v: [batch, seqlen_k, nhead_k, hdim]
        causal: if True, applies top-left causal mask (mask_enum=1).
        softmax_scale: unused; the dispatcher uses 1/sqrt(hdim_q) internally.
            Provided for API symmetry; a non-default value will raise.
        return_lse: not supported in this first cut.
        arch: GPU arch override (e.g. "gfx942"); auto-detected if None.

    Returns:
        (out, time_ms) where ``out`` is a torch tensor in the same layout/dtype
        as ``q``, and ``time_ms`` is the kernel time reported by the dispatcher.
    """
    if return_lse:
        raise NotImplementedError(
            "return_lse is not supported by the CK dispatcher wrapper yet."
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4-D BSHD tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must share dtype")
    if q.dtype not in _TORCH_TO_DISPATCHER_DTYPE:
        raise ValueError(
            f"Unsupported dtype {q.dtype}. "
            f"Supported: {list(_TORCH_TO_DISPATCHER_DTYPE)}"
        )

    batch, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, _ = k.shape
    _, _, _, hdim_v = v.shape
    if hdim_q != hdim_v:
        raise NotImplementedError(
            "Asymmetric hdim_q != hdim_v not supported by this preset."
        )

    if softmax_scale is not None:
        # FmhaProblem.scale is fixed to 1/sqrt(hdim_q); honoring an arbitrary
        # scale would require a new C ABI call. Bail loudly rather than
        # silently producing wrong numbers.
        import math
        if abs(softmax_scale - 1.0 / math.sqrt(hdim_q)) > 1e-6:
            raise NotImplementedError(
                "Custom softmax_scale is not supported by this wrapper; "
                "only the default 1/sqrt(hdim_q) is used."
            )

    dtype_str = _TORCH_TO_DISPATCHER_DTYPE[q.dtype]
    arch = arch or detect_gpu_arch()
    setup = _get_runner(dtype_str, hdim_q, arch)
    runner = setup.runner

    # BSHD (aiter / flash-attn) -> BHSD (dispatcher).
    # numpy() cannot represent torch.bfloat16, so for bf16 we go through
    # float32 on CPU first (the dispatcher's FmhaRunner.run repacks bf16
    # from float32 via _float32_to_bf16).
    if q.dtype == torch.bfloat16:
        q_cpu = q.detach().permute(0, 2, 1, 3).contiguous().to(torch.float32).cpu()
        k_cpu = k.detach().permute(0, 2, 1, 3).contiguous().to(torch.float32).cpu()
        v_cpu = v.detach().permute(0, 2, 1, 3).contiguous().to(torch.float32).cpu()
    else:
        q_cpu = q.detach().permute(0, 2, 1, 3).contiguous().cpu()
        k_cpu = k.detach().permute(0, 2, 1, 3).contiguous().cpu()
        v_cpu = v.detach().permute(0, 2, 1, 3).contiguous().cpu()
    q_np = q_cpu.numpy()
    k_np = k_cpu.numpy()
    v_np = v_cpu.numpy()

    prob = FmhaProblem(
        batch=batch,
        nhead_q=nhead_q,
        nhead_k=nhead_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=hdim_q,
        hdim_v=hdim_v,
    )

    result = runner.run(
        q_np,
        k_np,
        v_np,
        prob,
        mask_type=_MASK_CAUSAL_TOP_LEFT if causal else _MASK_NO,
        api_family="fwd",
        data_type=dtype_str,
    )
    if not result.success:
        raise RuntimeError(f"CK dispatcher run failed: {result.error}")

    # Output comes back as numpy in BHSD with the same dtype family as input.
    out_np = result.output
    if dtype_str == "bf16":
        # FmhaRunner returns bf16 stored as uint16; unpack to float32 then cast.
        from fmha_utils import _bf16_to_float32  # local import; private helper
        out_np = _bf16_to_float32(out_np)

    out = torch.from_numpy(np.ascontiguousarray(out_np))
    out = out.to(device=q.device, dtype=q.dtype)
    # BHSD -> BSHD
    out = out.permute(0, 2, 1, 3).contiguous()

    return out, float(result.time_ms)
