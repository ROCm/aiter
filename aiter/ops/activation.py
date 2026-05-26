# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_activation"

# Arches where the prebuilt HIP CK module 'module_activation' ships matching
# code objects. On any other arch (e.g. gfx1201 RDNA4 in
# rocm/atom-dev:latest) the HIP kernel launch SIGSEGVs with no catchable
# exception, so we must allowlist arches BEFORE calling the HIP path.
# Same set as aiter.ops.gemm_op_a8w8._BLOCKSCALE_HIP_PREBUILT_ARCHES and
# aiter.ops.cache._CACHE_HIP_PREBUILT_ARCHES.
_ACTIVATION_HIP_PREBUILT_ARCHES = frozenset({"gfx940", "gfx941", "gfx942", "gfx950"})


def _hip_activation_supported() -> bool:
    """True when the prebuilt 'module_activation' has a code object for the
    running device. False -> fall back to the triton variant."""
    try:
        from ..jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() in _ACTIVATION_HIP_PREBUILT_ARCHES
    except Exception:
        return False


@compile_ops("module_activation", develop=True)
def _silu_and_mul_ck(out: Tensor, input: Tensor, limit: float = 0.0) -> None: ...


def silu_and_mul(out: Tensor, input: Tensor, limit: float = 0.0) -> None:
    """SwiGLU activation: ``out = silu(input[..., :d]) * input[..., d:]`` where
    ``d = input.size(-1) // 2``.

    On arches with a prebuilt HIP CK code object for module_activation
    (gfx94x / gfx95x), dispatches to the CK kernel. On other arches
    (e.g. gfx1201 RDNA4) falls back to
    aiter.ops.triton.activation.fused_silu_mul, which JIT-compiles for
    any arch.

    Triton-fallback constraints:
      - ``input`` must be contiguous with an even last dim
      - ``limit`` must be 0.0 (the triton variant has no limit clamp)
    """
    if _hip_activation_supported():
        return _silu_and_mul_ck(out, input, limit)
    assert limit == 0.0, (
        "silu_and_mul triton fallback does not support the 'limit' clamp; "
        f"got limit={limit}"
    )
    assert (
        input.is_contiguous()
    ), "silu_and_mul triton fallback requires a contiguous input tensor"
    assert input.size(-1) % 2 == 0, (
        f"silu_and_mul triton fallback requires an even last dim; "
        f"got {input.size(-1)}"
    )
    from .triton.activation import fused_silu_mul as _silu_and_mul_triton

    _silu_and_mul_triton(input, out=out)


@compile_ops("module_activation", develop=True)
def swiglu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def swiglu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_and_mul_bias(
    out: Tensor, input: Tensor, expert_ids: Tensor, bias: Tensor
) -> None: ...


@compile_ops("module_activation", develop=True)
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def silu_and_mul_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    group_size: int,
    limit: float = 0.0,
    shuffle_scale: bool = False,
) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", develop=True)
def gelu_fast(out: Tensor, input: Tensor) -> None: ...
