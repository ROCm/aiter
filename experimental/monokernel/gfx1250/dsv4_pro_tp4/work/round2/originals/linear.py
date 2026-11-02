# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import logging
from collections.abc import Callable
from functools import partial as functools_partial
from typing import ClassVar

import torch
from aiter import (
    QuantType,
    dtypes,
    gemm_a4w4,
    gemm_a8w8,
    gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_bpreshuffle,
    gemm_a8w8_bpreshuffle,
    get_hip_quant,
)

# import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.quant import dynamic_per_group_scaled_quant, per_group_quant_hip
from aiter.tuned_gemm import tgemm
from aiter.utility import fp4_utils
from torch import nn

from atom.config import QuantizationConfig, get_current_atom_config
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.model_ops.utils import (
    atom_parameter,
    normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale,
    shuffle_weights,
)
from atom.quant_spec import (
    NVFP4_DTYPE,
    NVFP4_GROUP_SIZE,
    LayerQuantConfig,
    should_skip_online_quant,
    should_stream_online_quant,
    validate_nvfp4_global_scales,
    validate_nvfp4_online_target,
    will_online_requant,
)
from atom.quantization.quark.utils import (
    dequant_weight_online,
    dequantize_nvfp4,
    quant_weight_online,
)
from atom.utils import envs
from atom.utils.decorators import mark_trace

logger = logging.getLogger("atom")


def use_triton_gemm() -> bool:
    return envs.ATOM_USE_TRITON_GEMM


def use_fp4_non_shuffle_triton_gemm() -> bool:
    return envs.ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM


def _require_mxfp4_preshuffle_triton() -> Callable:
    global gemm_afp4wfp4_preshuffle

    if gemm_afp4wfp4_preshuffle is None:
        try:
            from aiter.ops.triton.gemm_afp4wfp4 import (
                gemm_afp4wfp4_preshuffle as preshuffle_gemm,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The selected preshuffled MXFP4 backend requires "
                "aiter.ops.triton.gemm_afp4wfp4_preshuffle"
            ) from exc
        gemm_afp4wfp4_preshuffle = preshuffle_gemm
    return gemm_afp4wfp4_preshuffle


if use_fp4_non_shuffle_triton_gemm():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4 = None
else:
    gemm_afp4wfp4 = None


if use_triton_gemm():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4_preshuffle = None

    # Plain (non-preshuffle) Triton blockscale GEMM. Consumes an unshuffled
    # (N, K) weight with row-major x_scale (M, scale_k) and w_scale
    # (scale_n, scale_k) -- the same layout the non-preshuffle path produces.
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale as gemm_a8w8_blockscale_triton,
        )
    except ImportError as e:
        logger.warning(f"Triton w8a8 blockscale GEMM not available: {e}")
        gemm_a8w8_blockscale_triton = None

    # Per-tensor / per-token a8w8 (per-row activation x per-column weight scale).
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
            gemm_a8w8 as gemm_a8w8_triton,
        )
    except ImportError as e:
        logger.warning(f"Triton a8w8 GEMM not available: {e}")
        gemm_a8w8_triton = None
else:
    gemm_afp4wfp4_preshuffle = None
    gemm_a8w8_blockscale_triton = None
    gemm_a8w8_triton = None


def use_mxfp4_preshuffle_triton_gemm(
    params_dtype: torch.dtype,
    *,
    arch: str | None = None,
) -> bool:
    """Select the consumer compatible with the stored MXFP4 preshuffle layout."""
    if params_dtype != dtypes.fp4x2 or use_fp4_non_shuffle_triton_gemm():
        return False
    if arch is None:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
    return arch == "gfx1250" or (
        use_triton_gemm() and gemm_afp4wfp4_preshuffle is not None
    )


from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE


def divide(numerator, denominator):
    assert (
        numerator % denominator == 0
    ), f"numerator {numerator} denominator {denominator}"
    return numerator // denominator


def per_1x128_e8m0_quant_fake(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    transpose_scale: bool,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=quant_dtype, device=x.device),
        torch.empty(
            (*x.shape[:-1], x.shape[-1] // 128),
            dtype=dtypes.fp8_e8m0,
            device=x.device,
        ),
    )


# Per-1x128 activation quant with an E8M0 scale, as a functional op. aiter's
# per_group_quant_hip allocates the scale as E8M0 and fills it through a
# mutating op; Inductor declines to decompose an auto_functionalized_v2 that
# touches E8M0 ("auto_functionalized_v2 was not removed"), so the mutation
# must stay inside this op.
@torch_compile_guard(gen_fake=per_1x128_e8m0_quant_fake, mutates_args=[])
def per_1x128_e8m0_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    transpose_scale: bool,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_group_quant_hip(
        x,
        scale=scale,
        quant_dtype=quant_dtype,
        group_size=128,
        transpose_scale=transpose_scale,
        scale_type=dtypes.fp8_e8m0,
    )


def mxfp8_m32k4_quant_fake(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is not None:
        raise ValueError("unsupported: static per token quant")
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty(x.shape, dtype=quant_dtype, device=x.device),
        torch.empty(
            ((rows + 31) // 32 * 32, x.shape[-1] // 32),
            dtype=dtypes.fp8_e8m0,
            device=x.device,
        ),
    )


# Keep the E8M0 out-buffer mutation inside a functional op, as in the regular
# per-1x128 path above. Inductor cannot lower the mutation when it sees it in
# the graph, including when the scale uses AITER's m32k4 layout.
@torch_compile_guard(gen_fake=mxfp8_m32k4_quant_fake, mutates_args=[])
def mxfp8_m32k4_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is not None:
        raise ValueError("unsupported: static per token quant")
    rows = x.numel() // x.shape[-1]
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    out_scale = torch.empty(
        ((rows + 31) // 32 * 32, x.shape[-1] // 32),
        dtype=dtypes.fp8_e8m0,
        device=x.device,
    )
    dynamic_per_group_scaled_quant(out, x, out_scale, 32, scale_layout_m32k4=True)
    return out, out_scale


def gemm_a4w4_quant_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=otype, device=x.device)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=gemm_a4w4_quant_fake, mutates_args=[])
def gemm_a4w4_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    # Non-shuffle FP4 Triton path: keep x/weight/scale in the original MXFP4
    # layout and call the non-preshuffled gemm_afp4wfp4 kernel.
    if params_dtype == dtypes.fp4x2 and use_fp4_non_shuffle_triton_gemm():
        if gemm_afp4wfp4 is None:
            raise RuntimeError(
                "ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1 requires aiter.ops.triton.gemm_afp4wfp4"
            )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=False,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_afp4wfp4(
            x.view(torch.uint8),
            weight.view(torch.uint8),
            x_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            otype,
            y,
        )
    # Consume ATOM's stored preshuffled weight/scale layout directly.
    elif use_mxfp4_preshuffle_triton_gemm(params_dtype):
        preshuffle_gemm = _require_mxfp4_preshuffle_triton()
        m, _ = x.view(-1, x.size(-1)).shape

        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                shuffle=(m >= MXFP4_QUANT_BLOCK_SIZE),
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        if m >= MXFP4_QUANT_BLOCK_SIZE:
            x_scale = x_scale.view(torch.uint8).view(
                x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            )
        else:
            x_scale = x_scale[:m, ...].view(torch.uint8)

        y = preshuffle_gemm(
            x.view(torch.uint8),
            weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
            x_scale,
            weight_scale.view(torch.uint8).view(
                weight_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            y=y,
        )
    # Default AITER path: quantize/shuffle into the layout expected by gemm_a4w4
    # and use the backend ASM implementation.
    else:
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=True,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_a4w4(
            x,
            weight,
            x_scale,
            weight_scale,
            y,
        )

    return y[:m, ...]


def _mxfp8_bpreshuffle_tuned(n: int, k: int) -> bool:
    """Whether aiter has 1x32 MXFP8 GEMM configs for this (N, K)."""
    from aiter.ops.gemm_op_a8w8 import mxfp8_bpreshuffle_tuned as tuned

    return tuned(n, k)


def _blockscale_preshuffle_gemm(mxscale_shuffle: bool):
    """The aiter preshuffle GEMM for 128-block or shuffled 1x32 e8m0 scales."""
    if mxscale_shuffle:
        from aiter import gemm_a8w8_mxfp8_bpreshuffle

        return gemm_a8w8_mxfp8_bpreshuffle
    return gemm_a8w8_blockscale_bpreshuffle


def gemm_a8w8_blockscale_preshuffle_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    mxscale_shuffle: bool = False,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_preshuffle_fake, mutates_args=[])
def gemm_a8w8_blockscale_preshuffle_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    mxscale_shuffle: bool = False,
) -> torch.Tensor:
    return _blockscale_preshuffle_gemm(mxscale_shuffle)(
        x, weight, x_scale, w_scale, dtype
    )


def gemm_a8w8_blockscale_preshuffle_into_output_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    mxscale_shuffle: bool = False,
) -> None:
    return None


@torch_compile_guard(
    gen_fake=gemm_a8w8_blockscale_preshuffle_into_output_fake, mutates_args=["out"]
)
def gemm_a8w8_blockscale_preshuffle_into_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    mxscale_shuffle: bool = False,
) -> None:
    # Same GEMM, but the result lands in the CALLER-owned `out` buffer (a fixed
    # address) and the op returns None — a mutates_args op must NOT also return
    # the mutated tensor (Inductor functionalization then emits a getitem it
    # cannot lower). Lets the downstream attention cudagraph read this output at
    # a stable address with no per-step input copy. Mirrors 035db69's
    # unified_attention_into_output.
    _blockscale_preshuffle_gemm(mxscale_shuffle)(
        x, weight, x_scale, w_scale, dtype, out=out
    )


def gemm_a8w8_blockscale_triton_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_triton_fake, mutates_args=[])
def gemm_a8w8_blockscale_triton_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Wrap the raw Triton launcher in a custom-op boundary so Dynamo treats it
    # as opaque (otherwise tracing into the Triton kernel launch causes a graph
    # break that splits the compiled model into >1 graph).
    return gemm_a8w8_blockscale_triton(x, weight, x_scale, w_scale, dtype)


def gemm_a8w8_per_tensor_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a8w8_per_tensor_fake, mutates_args=[])
def gemm_a8w8_per_tensor_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # The triton a8w8 kernel applies a per-row (activation) and per-column
    # (weight) scale. Per-tensor quantization produces a single scalar scale
    # for each operand, so broadcast them into the (M,) / (N,) vectors the
    # kernel expects before launching.
    M = x.shape[0]
    N = weight.shape[0]
    x_scale_vec = x_scale.reshape(-1)[:1].to(torch.float32).expand(M).contiguous()
    w_scale_vec = w_scale.reshape(-1)[:1].to(torch.float32).expand(N).contiguous()
    return gemm_a8w8_triton(
        x,
        weight,
        x_scale_vec,
        w_scale_vec,
        bias=bias,
        dtype=dtype,
    )


def gemm_a8w8_per_token_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a8w8_per_token_fake, mutates_args=[])
def gemm_a8w8_per_token_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # The triton a8w8 kernel natively applies a per-row (activation) and
    # per-column (weight) scale -- exactly per-token-per-channel. The scales are
    # already (M, 1) / (N, 1); flatten them to the (M,) / (N,) vectors the kernel
    # expects. Unlike the AITER bpreshuffle path this consumes the unshuffled
    # (N, K) weight, so the loader must skip the per_Token weight shuffle when
    # use_triton_gemm() is enabled (see process_weights_after_loading).
    x_scale_vec = x_scale.reshape(-1).to(torch.float32).contiguous()
    w_scale_vec = w_scale.reshape(-1).to(torch.float32).contiguous()
    return gemm_a8w8_triton(
        x,
        weight,
        x_scale_vec,
        w_scale_vec,
        bias=bias,
        dtype=dtype,
    )


def _can_use_a8w8_preshuffle(output_size: int, input_size: int) -> bool:
    """Whether an a8w8 weight can use the AITER bpreshuffle GEMM as-is.

    ``shuffle_weight(..., layout=(16, 16))`` packs the output (N) dim in 16-row
    tiles and the input (K) dim in ``BK = IK * 2 = 32``-col tiles (it asserts
    ``x.shape[-1] % 32 == 0``). So N must be 16-aligned and K must be 32-aligned.
    """
    return output_size % 16 == 0 and input_size % 32 == 0


def _a8w8_preshuffle_output_padding(output_size: int) -> int:
    """Rows needed to pad an a8w8 weight's output dim (N) up to the GEMM's N-tile
    (128). Returns 0 when already tile-aligned. Padding N to 128 also makes it
    16-aligned, so the tuned preshuffle GEMM can run instead of falling back."""
    remainder = output_size % 128
    return 0 if remainder == 0 else 128 - remainder


def weight_is_stored_preshuffled(
    quant_type: QuantType,
    params_dtype: torch.dtype,
    *,
    needs_preshuffled_weight: bool = False,
    native_group_rows: int | None = None,
) -> bool:
    """Whether a quantized 2D GEMM weight of this kind is held preshuffled.

    One answer for both sides of a weight's life: the initial load
    (``LinearBase.process_weights_after_loading``) and an online weight update
    (``WeightUpdaterMixin._post_process_fp8_weight``). Deciding it twice is how
    a synced weight ends up in a layout the loaded one would never have had,
    which the kernel then reads through the wrong permutation.

    ``needs_preshuffled_weight`` is the module's own override: a fused forward
    that calls the *preshuffle* blockscale GEMM directly (DeepSeek's fused
    qkv_a_proj) needs the 16x16-shuffled weight even when the global
    preshuffle path is off, so it is shuffled once at load rather than per
    forward.

    ``native_group_rows`` marks a native group32 weight, otherwise read as
    checkpoint bytes. Only an FP8 32x32 one is shuffled: on gfx950 for AITER's
    preshuffled group32 GEMM (ATOM_GROUP32_WEIGHT_PRESHUFFLE, on by default),
    and always under ``needs_preshuffled_weight`` (V4.1's grouped ``wo_a``).

    Says nothing about rank. Only 2D weights are shuffled -- Qwen3-Next's GDN
    conv1d expands its weight to 3D and must stay row-major -- so the caller
    that holds the tensor checks that.

    Compared by ``.value``, like every comparison in this file that runs after
    the load (``_is_quantized``, ``_is_blockscale``, ``forward``) and like the
    whole of ``weight_updater``. ``QuantType`` is a pybind enum out of the
    compiled ``aiter.jit.module_aiter_core``, so a module whose ``quant_type``
    came from a differently-identified aiter import -- a plugin process, a
    re-import, ``atom.quant_spec``'s lazy proxy -- does not compare equal to
    these members under ``==``. Every branch would then fall through to
    ``return False``: the sync writes the FP8 weight and never re-shuffles it,
    and the preshuffle GEMM reads a row-major weight. Silent, and only after
    the first weight update. The load side reached here from ``==`` and the
    sync side from ``.value``; unifying on ``==`` would have taken the sync
    side backwards.
    """
    if native_group_rows is not None:
        if native_group_rows != 32 or params_dtype != dtypes.fp8:
            return False
        if needs_preshuffled_weight:
            return True
        from aiter.jit.utils.chip_info import get_gfx

        return envs.ATOM_GROUP32_WEIGHT_PRESHUFFLE and get_gfx() == "gfx950"
    quant_value = quant_type.value
    if quant_value == QuantType.per_Token.value:
        # The triton a8w8 per_Token GEMM consumes the unshuffled (N, K)
        # weight; only the AITER bpreshuffle fallback needs the shuffle.
        return params_dtype == dtypes.fp8 and not (
            use_triton_gemm() and gemm_a8w8_triton is not None
        )
    if quant_value == QuantType.per_1x32.value:
        is_fp4_blockscale = params_dtype == dtypes.fp4x2
        return not is_fp4_blockscale or not use_fp4_non_shuffle_triton_gemm()
    if quant_value == QuantType.per_1x128.value:
        return envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE or needs_preshuffled_weight
    return False


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | list[int],
        tp_dim: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
        override_tp_size: int | None = None,
        override_tp_rank: int | None = None,
    ):
        self.prefix = prefix
        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix)
            if quant_config is not None
            else LayerQuantConfig()
        )
        quant_type = layer_quant_config.quant_type
        params_dtype = layer_quant_config.quant_dtype
        self.source_quant_dtype = source_quant_dtype
        self.layer_quant_config = layer_quant_config
        is_nvfp4 = params_dtype == NVFP4_DTYPE
        if is_nvfp4:
            validate_nvfp4_online_target(quant_config, prefix)
        self.quant_config = quant_config
        super().__init__()
        self.reduce_results = reduce_results
        self.input_size = input_size
        self.output_size = (
            output_size if isinstance(output_size, int) else sum(output_size)
        )
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        # Optional effective-TP override: shard on a coarser grid than the global
        # TP group (e.g. DCP query replication uses effective TP = tp/dcp so each
        # rank materializes its whole DCP group's output shard). Since eff_tp is a
        # valid TP size and eff_rank a valid rank within it, all downstream param
        # sizing / weight_loader narrowing is inherited unchanged.
        #
        # Records that this layer went through the override path -- vs. merely
        # being wide enough by coincidence (e.g. at tp == dcp, where
        # override_tp_size == 1 makes the override a no-op) -- so a consumer
        # can check this instead of reverse-engineering it from `output_size`.
        self.effective_tp_overridden = override_tp_size is not None
        if override_tp_size is not None or override_tp_rank is not None:
            assert (
                override_tp_size is not None and override_tp_rank is not None
            ), "override_tp_size and override_tp_rank must be set together"
            assert 0 <= override_tp_rank < override_tp_size
            self.tp_size = override_tp_size
            self.tp_rank = override_tp_rank
        self.output_partition_sizes = (
            output_size if isinstance(output_size, list) else [output_size]
        )
        if tp_dim == 1:
            self.input_size = divide(input_size, self.tp_size)
        elif tp_dim == 0:
            self.output_size = divide(self.output_size, self.tp_size)
            self.output_partition_sizes = [
                divide(s, self.tp_size) for s in self.output_partition_sizes
            ]
        if is_nvfp4 and self.input_size % NVFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"NVFP4 logical K={self.input_size} must be divisible by "
                f"group_size={NVFP4_GROUP_SIZE}."
            )
        # Checked here rather than at conversion: MXFP4 is the only way out of
        # NVFP4, so a 16-aligned K that is not 32-aligned can never run, and
        # finding out after the checkpoint has loaded helps no one.
        if is_nvfp4 and self.input_size % 32 != 0:
            raise ValueError(
                f"{prefix}: NVFP4 logical K={self.input_size} is valid for "
                "the source group size 16 but cannot be converted by the "
                "MXFP4 Linear path, which requires K divisible by 32."
            )

        # Stream eligible source weights through meta storage.
        self._stream_online_quant = self.source_quant_dtype is None and (
            should_stream_online_quant(quant_config, prefix, quant_type, params_dtype)
        )
        # The default device may be reset before loading starts.
        self._load_device = torch.empty(0).device if self._stream_online_quant else None
        param_device = "meta" if self._stream_online_quant else None

        if self.source_quant_dtype is not None:
            weight_size = (self.output_size, self.input_size)
            self.weight = atom_parameter(
                torch.empty(
                    weight_size, dtype=self.source_quant_dtype, device=param_device
                )
            )
        else:
            weight_dtype = torch.uint8 if is_nvfp4 else params_dtype
            weight_size = (
                (self.output_size, self.input_size)
                if params_dtype not in (dtypes.fp4x2, dtypes.i4x2, NVFP4_DTYPE)
                else (self.output_size, self.input_size // 2)
            )
            self.weight = atom_parameter(
                torch.empty(weight_size, dtype=weight_dtype, device=param_device)
            )
        if bias:
            output_type = get_current_atom_config().torch_dtype
            self.bias = atom_parameter(
                torch.empty(self.output_size, dtype=output_type, device=param_device)
            )
            self.bias.weight_loader_process = self.weight_loader_process
        else:
            self.register_parameter("bias", None)
        self.quant_type = quant_type
        self.params_dtype = params_dtype
        # E8M0 rather than FP32 128x128 block scales, for the weight and the
        # activation quantized for it alike (QuantizationConfig decides).
        self.blockscale_e8m0_scale = (
            quant_type.value == QuantType.per_1x128.value
            and quant_config.blockscale_e8m0_scale
        )
        self.native_a8_group_rows = None
        source_block = layer_quant_config.weight_block_size
        native_fp8 = params_dtype == torch.float8_e4m3fn and source_block in (
            (1, 32),
            (32, 32),
        )
        native_w4a8 = (
            params_dtype == torch.float4_e2m1fn_x2
            and layer_quant_config.activation_dtype == torch.float8_e4m3fn
            and source_block == (1, 32)
        )
        # A weight that online quant is about to overwrite is not a native
        # source: the checkpoint bytes this path reads survive only until
        # `online_quantize_weight` replaces them with the online target's
        # format, which has its own quant_type and its own scale geometry.
        # Only (1, 32) steps aside instead of raising -- its per-row scales are
        # shaped exactly like the generic per_1x32 path's, so dropping the
        # native layout lands on parameters this loader already knows how to
        # fill. A (32, 32) source has no such twin and keeps the error.
        requantized_online = will_online_requant(
            quant_config, prefix, quant_type, params_dtype
        )
        falls_back_to_generic_1x32 = requantized_online and source_block == (1, 32)
        if (native_fp8 or native_w4a8) and not falls_back_to_generic_1x32:
            if (
                quant_type != QuantType.per_1x32
                or self.source_quant_dtype is not None
                or not layer_quant_config.is_dynamic
                or requantized_online
            ):
                raise ValueError(
                    "Native group32 A8 requires dynamic activations and native weights"
                )
            self.native_a8_group_rows = layer_quant_config.weight_block_size[0]
            if self.input_size % 32 or any(
                size % self.native_a8_group_rows for size in self.output_partition_sizes
            ):
                raise ValueError(
                    "TP partitions must align with the native source blocks"
                )

        if quant_type != QuantType.No and self.source_quant_dtype is None:
            if is_nvfp4:
                # Quark / NVIDIA ModelOpt NVFP4 wire format:
                #   weight          U8-packed FP4 E2M1, two values per byte
                #   weight_scale    FP8 E4M3, one scale per 16 logical values
                #   weight_scale_2  F32 global weight scale per source projection
                #
                # Merged linears retain one scalar per source projection so no
                # information is lost before the NVFP4 -> MXFP4 online
                # conversion pass.
                self.weight_scale = atom_parameter(
                    torch.empty(
                        self.output_size,
                        self.input_size // NVFP4_GROUP_SIZE,
                        dtype=torch.float8_e4m3fn,
                        device=param_device,
                    )
                )
                # Zeroed, not empty: a checkpoint that carries no
                # *.weight_scale_2 leaves this untouched, and zero is the one
                # value online_quantize_weight can recognize as "never loaded".
                # Uninitialized memory can read back as a plausible scale.
                self.weight_scale_2 = atom_parameter(
                    torch.zeros(
                        len(self.output_partition_sizes),
                        dtype=torch.float32,
                        device=param_device,
                    )
                )
            elif quant_type == QuantType.per_Tensor:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        len(self.output_partition_sizes),
                        1,
                        dtype=dtypes.fp32,
                        device=param_device,
                    )
                )
                if not layer_quant_config.is_dynamic:
                    self.input_scale = atom_parameter(
                        torch.empty(
                            len(self.output_partition_sizes),
                            1,
                            dtype=dtypes.fp32,
                            device=param_device,
                        )
                    )
                    self.input_scale.weight_loader_process = self.weight_loader_process
                    self.input_scale.weight_loader = self.weight_loader
            elif quant_type == QuantType.per_Token:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        self.output_size, 1, dtype=dtypes.fp32, device=param_device
                    )
                )
            elif quant_type == QuantType.per_1x128:
                scale_dtype = (
                    dtypes.fp8_e8m0 if self.blockscale_e8m0_scale else dtypes.fp32
                )
                self.weight_scale = atom_parameter(
                    torch.empty(
                        (self.output_size + 127) // 128,
                        (self.input_size + 127) // 128,
                        dtype=scale_dtype,
                        device=param_device,
                    )
                )
            elif quant_type == QuantType.per_1x32:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        self.output_size // (self.native_a8_group_rows or 1),
                        (self.input_size + 31) // 32,
                        dtype=dtypes.fp8_e8m0,
                        device=param_device,
                    )
                )
            self.weight.weight_loader_process = self.weight_loader_process
            self.weight_scale.weight_loader_process = self.weight_loader_process
        else:
            self.weight.weight_loader_process = self.weight_loader_process
            self.register_parameter("weight_scale", None)
        if getattr(self, "weight_scale_2", None) is None:
            self.register_parameter("weight_scale_2", None)
        # The converted MXFP4 layer quantizes activations dynamically, so no
        # static activation scale is loaded and the slot stays None for
        # streaming to ignore.
        # Quark checkpoints still ship `input_scale_2`; it is intentionally
        # unused, so the loader listing it as a dropped tensor is expected.
        self.register_parameter("input_scale_2", None)
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
        if self.weight_scale is not None:
            self.weight_scale.weight_loader = self.weight_loader
        if self.weight_scale_2 is not None:
            self.weight_scale_2.weight_loader_process = self.weight_loader_process
            # Use one format-specific loader for every Linear subclass. In
            # particular, QKV variants must not reinterpret replicated scalar
            # metadata as tensor-parallel rows in their custom weight loaders.
            self.weight_scale_2.weight_loader = self._load_nvfp4_global_scale
        self.need_normalize_e4m3fn_to_e4m3fnuz = params_dtype == torch.float8_e4m3fnuz
        self.quant_func = get_hip_quant(self.quant_type)
        self.is_output_padded = False
        # 1x32 scales in aiter's shuffled layout for the shapes aiter has tuned;
        # decided before loading so a fused-quant producer can follow it.
        self.use_mxscale_shuffle = (
            envs.ATOM_FP8_MXSCALE_USE_E8M0_SCALE_SHUFFLE
            and self.blockscale_e8m0_scale
            and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
            and quant_type.value == QuantType.per_1x128.value
            and params_dtype == dtypes.fp8
            and self.source_quant_dtype is None
            and not (quant_config is not None and quant_config.online_quant)
            and _mxfp8_bpreshuffle_tuned(self.output_size, self.input_size)
        )

    @property
    def weight_scale_row_group(self) -> int:
        """Number of weight rows represented by one row of source scales."""
        if self.native_a8_group_rows is not None:
            return self.native_a8_group_rows
        return 128 if self.quant_type == QuantType.per_1x128 else 1

    @staticmethod
    def weight_loader_process(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        post_process_func: Callable = lambda a: a,
    ):
        if param.data.dtype != loaded_weight.dtype:
            if param.data.element_size() == loaded_weight.element_size():
                # Same byte-width: use view for raw-bit-compatible pairs
                # (e.g. fp8 variants) but convert for semantically different
                # formats (float16 ↔ bfloat16) where bit reinterpretation
                # would corrupt values.
                incompatible = {torch.float16, torch.bfloat16}
                if {param.data.dtype, loaded_weight.dtype} == incompatible:
                    loaded_weight = loaded_weight.to(param.data.dtype)
                else:
                    param.data = param.data.view(loaded_weight.dtype)
            else:
                loaded_weight = loaded_weight.to(param.data.dtype)
        loaded_weight = post_process_func(loaded_weight)
        if (
            loaded_weight.shape != param.data.shape
            and loaded_weight.numel() == param.data.numel()
        ):
            loaded_weight = loaded_weight.reshape(param.data.shape)
        if param.data.dtype != dtypes.fp4x2:
            param.data.copy_(loaded_weight)
        else:
            param.data.view(torch.uint8).copy_(loaded_weight.view(torch.uint8))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)

    def _load_nvfp4_global_scale(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None = None,
    ) -> None:
        """Load replicated NVFP4 per-tensor global scales.

        These scalars do not follow either tensor-parallel dimension. A merged
        projection owns one value per source projection; ordinary projections
        own one value. Every TP rank receives the same scalar(s).
        """
        param_data = param.data
        values = loaded_weight.reshape(-1)
        if isinstance(loaded_shard_id, str):
            shard_map = getattr(self, "_nvfp4_global_scale_shard_map", None)
            if shard_map is None:
                raise ValueError(
                    f"{self.prefix}: {type(self).__name__} defines no "
                    "_nvfp4_global_scale_shard_map, so NVFP4 global scale shard "
                    f"id {loaded_shard_id!r} has no slot. Add the map to the "
                    "class, mapping each str shard id to its index in "
                    "output_partition_sizes (or a tuple of indices)."
                )
            if loaded_shard_id not in shard_map:
                supported = ", ".join(repr(key) for key in shard_map) or "none"
                raise ValueError(
                    f"{type(self).__name__} does not support NVFP4 global scale "
                    f"shard id {loaded_shard_id!r}; supported ids: {supported}."
                )
            mapped_shard_id = shard_map[loaded_shard_id]
            if isinstance(mapped_shard_id, tuple) and values.numel() == 1:
                values = values.expand(len(mapped_shard_id))
            self._load_nvfp4_global_scale(param, values, mapped_shard_id)
            return
        if isinstance(loaded_shard_id, tuple):
            if values.numel() != len(loaded_shard_id):
                raise RuntimeError(
                    "NVFP4 global scale count does not match merged shard ids: "
                    f"values={values.numel()}, shard_ids={loaded_shard_id}."
                )
            for value, shard_id in zip(values, loaded_shard_id):
                self._load_nvfp4_global_scale(param, value, shard_id)
            return
        if loaded_shard_id is None:
            if values.numel() != param_data.numel():
                raise RuntimeError(
                    "NVFP4 global scale shape mismatch: "
                    f"param={tuple(param_data.shape)}, loaded={tuple(loaded_weight.shape)}."
                )
            param.weight_loader_process(param_data, values.reshape(param_data.shape))
            return
        if not isinstance(loaded_shard_id, int) or not (
            0 <= loaded_shard_id < param_data.numel()
        ):
            raise ValueError(
                f"Invalid NVFP4 global scale shard id {loaded_shard_id!r}; "
                f"expected [0, {param_data.numel() - 1}]."
            )
        if values.numel() != 1:
            raise RuntimeError(
                "Each NVFP4 source projection must provide exactly one global "
                f"scale, got {values.numel()}."
            )
        target = param_data.narrow(0, loaded_shard_id, 1)
        param.weight_loader_process(target, values)

    def _gather_full_weight(self, weight):
        """Gather sharded weight from all TP ranks to reconstruct the full unpartitioned weight."""
        if self.tp_size <= 1 or self.tp_dim is None:
            return weight
        # NCCL cannot all_gather E8M0 scales (MXFP8 source); gather the raw
        # bytes as uint8 and reinterpret afterwards. The gather only moves
        # bytes, so this is bit-exact.
        if weight.dtype == dtypes.fp8_e8m0:
            gathered = get_tp_group().all_gather(
                weight.view(torch.uint8), dim=self.tp_dim
            )
            return gathered.view(dtypes.fp8_e8m0)
        return get_tp_group().all_gather(weight, dim=self.tp_dim)

    def _shard_quantized_weight(self, q_weight, weight_scale):
        """Split the quantized full weight and scale back to this rank's shard."""
        if self.tp_size <= 1 or self.tp_dim is None:
            return q_weight, weight_scale
        # fp4x2 packs 2 values per byte; view as uint8 so narrow() works correctly
        is_mxfp4 = q_weight.dtype == torch.float4_e2m1fn_x2
        if is_mxfp4:
            q_weight = q_weight.view(torch.uint8)
        w_shard = q_weight.shape[self.tp_dim] // self.tp_size
        w_start = self.tp_rank * w_shard
        q_weight_local = q_weight.narrow(self.tp_dim, w_start, w_shard).contiguous()
        if is_mxfp4:
            q_weight_local = q_weight_local.view(torch.float4_e2m1fn_x2)

        # for col linear qkv, w13, [m, n] -> [m // tp, n] -> [m // tp, 1], TP and non-TP are equivalent
        if weight_scale.dim() > self.tp_dim and weight_scale.shape[self.tp_dim] > 1:
            s_shard = weight_scale.shape[self.tp_dim] // self.tp_size
            s_start = self.tp_rank * s_shard
            weight_scale_local = weight_scale.narrow(
                self.tp_dim, s_start, s_shard
            ).contiguous()
        # for row linear o, w2, [m, n] -> [m, n // tp] -> [m, 1], TP and non-TP are not equivalent
        else:
            weight_scale_local = weight_scale

        return q_weight_local, weight_scale_local

    def online_quantize_weight(self):
        """Re-quantize this layer's weight at load time using the online quant config.

        Handles TP gather/shard when local quantization would produce
        different results than quantizing the full unpartitioned weight.
        """
        online_layer_quant_config = self.quant_config.get_layer_quant_config(
            self.prefix, use_online_quant=True
        )
        online_quant_type = online_layer_quant_config.quant_type
        online_quant_dtype = online_layer_quant_config.quant_dtype
        source_is_nvfp4 = self.params_dtype == NVFP4_DTYPE
        if should_skip_online_quant(
            self.quant_type, self.params_dtype, online_layer_quant_config
        ):
            return

        assert online_quant_dtype in [
            dtypes.fp8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float4_e2m1fn_x2,
        ], (
            f"Unsupported online quant: "
            f"dtype={online_quant_dtype}, type={online_quant_type}"
        )
        # Serialized models arrive in several source formats. We can re-quantize
        # any source we know how to dequantize back to float first: unquantized
        # (No), per-tensor FP8 (per_Tensor), per-output-channel FP8 (per_Token /
        # ptpc_fp8), 128x128 block FP8 (per_1x128), MXFP8 (per_1x32) and NVFP4
        # (per_1x32 with NVFP4_DTYPE). Any other source is rejected up front
        # rather than silently producing garbage.
        assert self.quant_type in [
            QuantType.No,
            QuantType.per_Tensor,
            QuantType.per_Token,
            QuantType.per_1x128,
            QuantType.per_1x32,
        ], (
            f"Unsupported source quant_type for online quantization: "
            f"{self.quant_type} (layer={self.prefix}). Supported sources: "
            f"No, per_Tensor, per_Token, per_1x128, per_1x32."
        )
        weight = self.weight.data
        weight_scale = getattr(self, "weight_scale", None)
        if (
            self.tp_size > 1
            and self.tp_dim is not None
            and isinstance(self, ReplicatedLinear)
        ):
            # W and S of kv_a_proj_with_mqa don't match,
            # but it doesn't need to be split.
            return
        # Gather is required whenever local quantization would differ from
        # quantizing the full unpartitioned weight (bit-exact with offline).
        # Streaming tails are collective-free because modules complete in
        # different orders across ranks; they quantize the local shard instead.
        need_gather = False
        if (
            not self._stream_online_quant
            and self.tp_size > 1
            and self.tp_dim is not None
        ):
            # col qkv w13, tp_dim=0, [m, n] -> [m // tp, n] -> [m // tp, 1], don't need gather
            # row o, w2, tp_dim=1, [m, n] -> [m, n //tp] -> [m, 1], need gather
            if online_quant_type == QuantType.per_Token:
                # per_Token: scale = max(|full_row|), tp_dim=1 has partial rows
                need_gather = self.tp_dim == 1
            elif online_quant_type == QuantType.per_1x128:
                # 128×128 blocks: misaligned if partition size % 128 != 0
                if self.tp_dim == 0:
                    need_gather = self.output_size % 128 != 0
                else:
                    need_gather = self.input_size % 128 != 0
            elif online_quant_type == QuantType.per_1x32:
                # 1×32 blocks: row dim is 1 (always aligned), only column matters
                # col qkv w13, tp_dim=0, [m, n] -> [m // tp, n] -> [m // tp, n // 32], don't need gather
                # row o, w2, tp_dim=1, [m, n] -> [m, n //tp] -> [m, n // tp // 32], need gather
                need_gather = self.tp_dim == 1 and self.input_size % 32 != 0
        if need_gather:
            weight = self._gather_full_weight(weight)
            # Per-token scales are shaped (N, 1). In row-parallel layers
            # (tp_dim=1), that size-1 dim is replicated rather than sharded, so
            # gathering it would duplicate the scale columns. Block scales still
            # follow the weight sharding and are gathered here.
            if weight_scale is not None and not (
                self.quant_type == QuantType.per_Token and self.tp_dim == 1
            ):
                weight_scale = self._gather_full_weight(weight_scale)

        # Dequantize the source weight back to float so it can be re-quantized
        # to the online target format (no-op for an unquantized source).
        # output_partition_sizes lets per_Tensor merged layers (qkv/gate_up)
        # apply the right per-partition scale to each output row-range.
        if source_is_nvfp4:
            if weight_scale is None:
                raise RuntimeError(f"{self.prefix}: NVFP4 weight_scale is missing.")
            global_scale = getattr(self, "weight_scale_2", None)
            # A checkpoint without *.weight_scale_2 leaves the zeroed
            # placeholder in place and would dequantize the whole weight to 0.
            validate_nvfp4_global_scales(
                None if global_scale is None else global_scale.data,
                self.prefix,
                "weight_scale_2",
            )
            global_scale = global_scale.data.reshape(-1)
            if global_scale.numel() != len(self.output_partition_sizes):
                raise RuntimeError(
                    f"{self.prefix}: expected {len(self.output_partition_sizes)} "
                    "NVFP4 global weight scales, got "
                    f"{global_scale.numel()}."
                )
            per_row_scale_2 = global_scale.repeat_interleave(
                torch.tensor(
                    self.output_partition_sizes,
                    device=global_scale.device,
                )
            ).view(-1, 1)
            # Decodes to the model dtype, like every other source format here;
            # quant_weight_online would cast an FP32 weight down anyway.
            weight = dequantize_nvfp4(weight, weight_scale, per_row_scale_2)
            q_weight, weight_scale = quant_weight_online(
                weight, online_quant_type, online_quant_dtype
            )
        else:
            weight = dequant_weight_online(
                weight,
                weight_scale,
                self.quant_type,
                self.params_dtype,
                self.output_partition_sizes,
            )
            q_weight, weight_scale = quant_weight_online(
                weight, online_quant_type, online_quant_dtype
            )
        if need_gather:
            q_weight, weight_scale = self._shard_quantized_weight(
                q_weight, weight_scale
            )
        self.weight = nn.Parameter(q_weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        # Both are fresh Parameters, so they carry none of the attributes
        # __init__ hung on the originals. weight_loader() reads
        # weight_loader_process off the parameter it is handed, so a later
        # weight update -- an RLHF rollout sync -- would fail on it.
        self.weight.weight_loader_process = self.weight_loader_process
        self.weight_scale.weight_loader_process = self.weight_loader_process

        # Update quant state
        self.quant_type = online_quant_type
        self.params_dtype = online_quant_dtype
        self.layer_quant_config = online_layer_quant_config
        self._stream_online_quant = False
        self.quant_func = get_hip_quant(online_quant_type)
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            online_quant_dtype == torch.float8_e4m3fnuz
            and online_quant_type == QuantType.per_Token
        )
        if source_is_nvfp4:
            self.weight_scale_2 = None
            self.input_scale_2 = None
        # A dynamic online target (e.g. ptpc per_Token) quantizes activations at
        # runtime. Drop any static input_scale inherited from a static per_Tensor
        # source, otherwise the per-token quant kernel rejects it
        # ("unsupported: static per token quant").
        if (
            online_layer_quant_config.is_dynamic
            and getattr(self, "input_scale", None) is not None
        ):
            self.input_scale = None
        self._online_quant_info = {
            "layer": self.prefix,
            "quant_type": online_quant_type.name,
            "quant_dtype": str(online_quant_dtype),
        }

    def process_weights_after_loading(self):
        """Settle the layout this weight is stored in, once.

        Every `quant_type` test below compares by `.value`, like
        `weight_is_stored_preshuffled` and everything downstream of the load.
        They have to be the SAME comparison: this function asks that helper
        whether to shuffle and then asks itself whether to pad, and a `==` on
        either side answers False for a module whose `quant_type` came from a
        differently-identified aiter import (see that helper's docstring), so
        the two halves of one decision came out inconsistent -- the weight
        shuffled and its N left unpadded, with the RuntimeError that exists to
        catch exactly that skipped along with the padding.
        """
        # A native group32 weight keeps its compact scales, and its bytes too
        # unless it is one the preshuffled group32 GEMM reads.
        if self.native_a8_group_rows is not None:
            if self.weight.numel() and weight_is_stored_preshuffled(
                self.quant_type,
                self.params_dtype,
                needs_preshuffled_weight=getattr(
                    self, "needs_preshuffled_weight", False
                ),
                native_group_rows=self.native_a8_group_rows,
            ):
                shuffle_weights(self.weight)
            return
        if self.weight.numel() == 0:
            return
        # Re-quantize before process_weights if online quantization is enabled
        if self.quant_config is not None and self.quant_config.online_quant:
            self.online_quantize_weight()
        if self.params_dtype == NVFP4_DTYPE:
            raise RuntimeError(
                f"{self.prefix}: NVFP4 weights were not converted to MXFP4 by "
                "online quantization."
            )
        if self.quant_type.value == QuantType.per_Tensor.value and (
            len(self.output_partition_sizes) > 1
            or hasattr(self, "_loaded_weight_scale_for_requant")
            or hasattr(self, "_loaded_weight_scale_for_requant_parts")
        ):
            loaded_weight_scale = getattr(
                self, "_loaded_weight_scale_for_requant", None
            )
            loaded_weight_scale_parts = getattr(
                self, "_loaded_weight_scale_for_requant_parts", None
            )
            if (
                loaded_weight_scale is None
                and loaded_weight_scale_parts is not None
                and all(part is not None for part in loaded_weight_scale_parts)
            ):
                loaded_weight_scale = torch.cat(loaded_weight_scale_parts, dim=0)
            weight_scale_for_requant = (
                loaded_weight_scale
                if loaded_weight_scale is not None
                else self.weight_scale.data
            )
            weight_scale, weight = requantize_with_max_scale(
                weight=self.weight.data,
                weight_scale=weight_scale_for_requant.to(self.weight.device),
                logical_widths=self.output_partition_sizes,
                normalize_e4m3fn_to_e4m3fnuz=self.need_normalize_e4m3fn_to_e4m3fnuz,
            )
            self.weight.data = weight
            self.weight_scale.data = weight_scale.view(-1)
            if hasattr(self, "input_scale"):
                self.input_scale.data = (
                    self.input_scale.data.max() * 2.0
                    if self.need_normalize_e4m3fn_to_e4m3fnuz
                    else self.input_scale.data.max()
                )
        elif self.need_normalize_e4m3fn_to_e4m3fnuz and self.weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            # The flag records how the weight was allocated, not what it holds.
            # An owner that dequantizes its weight in its own post-load hook --
            # parents run before their child linears -- leaves a float weight and
            # no scale here, and re-encoding that pair is not a no-op, it is a
            # crash. Read the tensor rather than make every such owner remember
            # to clear the flag.
            self.weight.data, self.weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                self.weight.data, self.weight_scale.data
            )
        if (
            self.source_quant_dtype == torch.bfloat16
            and self.quant_type.value == QuantType.per_1x32.value
            and self.params_dtype == torch.float4_e2m1fn_x2
        ):
            w_q, w_s = self.quant_func(
                self.weight.data,
                quant_dtype=self.params_dtype,
                shuffle=False,
            )
            self.weight.data = w_q
            self.weight_scale = atom_parameter(w_s)
            # Only quantized 2D GEMM weights use aiter's preshuffle layout.
            # Qwen3-Next/Qwen3.5 GDN conv1d expands its weight to 3D, so FP8/blocked
            # quantized models must keep that tensor unshuffled here.
            if self.weight.dim() == 2 and not use_fp4_non_shuffle_triton_gemm():
                shuffle_weights(self.weight)
            # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        else:
            need_shuffle = weight_is_stored_preshuffled(
                self.quant_type,
                self.params_dtype,
                needs_preshuffled_weight=getattr(
                    self, "needs_preshuffled_weight", False
                ),
            )
            if need_shuffle and self.weight.dim() == 2:
                self.is_output_padded = self._maybe_pad_a8w8_preshuffle_output()
                shuffle_weights(self.weight)
                # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        # shuffle weight scale once so no reshuffling for every gemm
        if self.quant_type.value == QuantType.per_1x32.value and (
            self.params_dtype != dtypes.fp4x2 or not use_fp4_non_shuffle_triton_gemm()
        ):
            self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        if getattr(self, "use_mxscale_shuffle", False):
            self.shuffle_mxscale_weight_scale()

    def shuffle_mxscale_weight_scale(self) -> None:
        """(Re)build the shuffled 1x32 weight scale, in place once it exists.

        `weight_scale` keeps its checkpoint shape for the loaders and weight sync.
        """
        from aiter.ops.shuffle import shuffle_blockscale_to_mxfp8_scale

        ws = shuffle_blockscale_to_mxfp8_scale(
            self.weight_scale.data, self.weight.shape[0]
        )
        if hasattr(self, "weight_scale_mxscale"):
            self.weight_scale_mxscale.copy_(ws)
        else:
            self.register_buffer("weight_scale_mxscale", ws, persistent=False)

    def _maybe_pad_a8w8_preshuffle_output(self) -> bool:
        # The other half of the shuffle decision `process_weights_after_loading`
        # takes, so it answers on the same terms `weight_is_stored_preshuffled`
        # does -- by value.
        if not (
            self.quant_type.value == QuantType.per_Token.value
            and self.params_dtype == dtypes.fp8
        ):
            return False
        if self.weight.dim() != 2:
            return False
        output_size, input_size = self.weight.shape
        padding_size = _a8w8_preshuffle_output_padding(output_size)
        if not _can_use_a8w8_preshuffle(output_size + padding_size, input_size):
            # Padding the output (N) cannot make this weight preshuffle-able, i.e.
            # the input dim K is not 32-aligned. Fail loudly here rather than let
            # shuffle_weights hit its cryptic `x.shape[-1] % 32 == 0` assertion.
            raise RuntimeError(
                f"{self.prefix}: a8w8 bpreshuffle GEMM requires K % 32 == 0, got "
                f"K={input_size}. Align K or run this layer via the triton a8w8 "
                f"path (ATOM_USE_TRITON_GEMM=1)."
            )
        if padding_size == 0:
            return False
        self._output_size_before_padding = output_size
        self.weight.data = torch.nn.functional.pad(
            self.weight.data, (0, 0, 0, padding_size)
        )
        ws = self.weight_scale.data
        self.weight_scale.data = torch.cat(
            [ws, ws.new_ones((padding_size, *ws.shape[1:]))], dim=0
        )
        # Bias is also per-output-channel
        if self.bias is not None:
            b = self.bias.data
            self.bias.data = torch.cat(
                [b, b.new_zeros((padding_size, *b.shape[1:]))], dim=0
            )
        return True

    # linear mark trace shape/dtype helper
    def get_trace_prefix(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor | None = None,
        otype=dtypes.bf16,
    ) -> str:
        k = x.shape[-1]
        m = x.numel() // k
        n = self.output_size
        a_dtype = (
            self.params_dtype
            if self.quant_type.value != QuantType.No.value
            else x.dtype
        )
        w_dtype = self.params_dtype
        o_dtype = otype
        return f"{self.prefix}[M={m},N={n},K={k},a={a_dtype},w={w_dtype},o={o_dtype}]"

    def supports_out(self) -> bool:
        """Whether `forward(out=...)` is wired for this Linear: only the per_1x128
        preshuffle GEMM writes into a caller-owned destination.

        Computed on access rather than cached at construction -- `quant_type` is
        rewritten when online quantisation is configured, well after __init__.
        """
        return self.quant_type.value == QuantType.per_1x128.value and bool(
            envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        )

    @mark_trace
    def forward(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor | None = None,
        otype=dtypes.bf16,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.params_dtype == NVFP4_DTYPE:
            raise RuntimeError(
                f"{self.prefix}: NVFP4 checkpoint weights are loaded, but direct "
                "NVFP4 inference is intentionally blocked. Convert this layer "
                "to MXFP4 during online quantization before forward."
            )
        # A quant path that cannot honour out= must not silently ignore it.
        assert out is None or self.supports_out(), (
            "Linear out= requested but this quant path does not support it "
            f"(quant_type={self.quant_type})."
        )
        if self.native_a8_group_rows is not None:
            from atom.model_ops.blockscale import native_quant_linear

            y = native_quant_linear(
                x,
                self.weight,
                self.weight_scale,
                x_scale=x_scale,
                weight_group_rows=self.native_a8_group_rows,
                weight_preshuffled=getattr(self.weight, "is_shuffled", False),
                dtype=otype,
            )
            if self.bias is not None:
                y += self.bias
        elif self.quant_type.value == QuantType.No.value:
            y = tgemm.mm(
                x,
                self.weight,
                self.bias,
                otype=otype,
            )
        else:
            if x_scale is None:
                quant_func = self.quant_func
                if getattr(self, "use_mxscale_shuffle", False):
                    quant_func = mxfp8_m32k4_quant
                elif self.quant_type.value == QuantType.per_1x128.value:
                    # preshuffle GEMM expects column-major x_scale;
                    # non-preshuffle GEMM expects row-major x_scale
                    quant_func = functools_partial(
                        (
                            per_1x128_e8m0_quant
                            if self.blockscale_e8m0_scale
                            else self.quant_func
                        ),
                        transpose_scale=envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,
                    )
                if self.quant_type.value != QuantType.per_1x32.value:
                    x, x_scale = quant_func(
                        x,
                        quant_dtype=self.params_dtype,
                        scale=getattr(self, "input_scale", None),
                    )
            if self.quant_type.value == QuantType.per_Tensor.value:
                if use_triton_gemm() and gemm_a8w8_triton is not None:
                    y = gemm_a8w8_per_tensor_impl(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        bias=self.bias,
                        dtype=otype,
                    )
                else:
                    y = tgemm.mm(
                        x,
                        self.weight,
                        self.bias,
                        otype=otype,
                        scale_a=x_scale,
                        scale_b=self.weight_scale,
                    )
            elif self.quant_type.value == QuantType.per_Token.value:
                if self.params_dtype == dtypes.i8:
                    y = gemm_a8w8(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        self.bias,
                        dtype=otype,
                    )
                elif use_triton_gemm() and gemm_a8w8_triton is not None:
                    # Triton a8w8 per-token-per-channel GEMM (unshuffled weight).
                    y = gemm_a8w8_per_token_impl(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        bias=self.bias,
                        dtype=otype,
                    )
                else:
                    y = gemm_a8w8_bpreshuffle(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                    )
                    if self.bias is not None:
                        y += self.bias
            elif self.quant_type.value == QuantType.per_1x128.value:
                mxscale = getattr(self, "use_mxscale_shuffle", False)
                weight_scale = (
                    self.weight_scale_mxscale if mxscale else self.weight_scale
                )
                if envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE:
                    if out is not None:
                        # Fixed-address output: write into `out` (returns None),
                        # then use it as y. Separate op because a mutates_args op
                        # must not also return the mutated tensor.
                        gemm_a8w8_blockscale_preshuffle_into_output(
                            x,
                            self.weight,
                            x_scale,
                            weight_scale,
                            out,
                            dtype=otype,
                            prefix=self.prefix,
                            mxscale_shuffle=mxscale,
                        )
                        y = out
                    else:
                        y = gemm_a8w8_blockscale_preshuffle_impl(
                            x,
                            self.weight,
                            x_scale,
                            weight_scale,
                            dtype=otype,
                            prefix=self.prefix,
                            mxscale_shuffle=mxscale,
                        )
                else:
                    assert out is None, (
                        "Linear out= only supported on the per_1x128 preshuffle "
                        "path (ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE)."
                    )
                    if use_triton_gemm() and gemm_a8w8_blockscale_triton is not None:
                        y = gemm_a8w8_blockscale_triton_impl(
                            x,
                            self.weight,
                            x_scale,
                            self.weight_scale,
                            dtype=otype,
                        )
                    else:
                        y = gemm_a8w8_blockscale(
                            x,
                            self.weight,
                            x_scale,
                            self.weight_scale,
                            dtype=otype,
                        )
                if self.bias is not None:
                    y += self.bias
            elif self.quant_type.value == QuantType.per_1x32.value:
                y = gemm_a4w4_quant(
                    x,
                    x_scale,
                    self.weight,
                    otype,
                    self.weight_scale.data,
                    self.params_dtype,
                    getattr(self, "input_scale", None),
                    self.output_size,
                )
                if self.bias is not None:
                    y += self.bias
        if self.is_output_padded:
            # Drop the padded output rows
            y = y[..., : self._output_size_before_padding]
        if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
            y = tensor_model_parallel_all_reduce(y)
        return y

    def make_row_view(self, start: int, length: int) -> "LinearBase":
        """A layer that computes only output rows [start, start+length).

        Motivation: DCP query replication makes q_proj emit the whole DCP group's
        heads so decode can skip its AllGather Q. Prefill needs only this rank's
        heads, and slicing the *output* means the GEMM still did 8x the work
        (measured: +14.8 ms/step of prefill on GLM-5.2 tp8/dcp8). Slicing the
        WEIGHT instead makes prefill cost what it costs without replication.
        """
        import copy

        assert 0 <= start and length > 0 and start + length <= self.weight.shape[0], (
            f"row view [{start}, {start + length}) out of range for "
            f"weight rows {self.weight.shape[0]}"
        )
        if getattr(self.weight, "is_shuffled", False):
            assert start % 16 == 0 and length % 16 == 0, (
                "a shuffled weight may only be row-sliced on 16-row boundaries "
                f"(shuffle block size); got start={start} length={length}"
            )

        view = copy.copy(self)
        # nn.Module bookkeeping is shared by the shallow copy; give the view its
        # own parameter dict so binding weight/scale on the view cannot disturb
        # `self`. It does NOT isolate the bytes and is not meant to: the view
        # narrows `self.weight.data`, so an in-place write to the parent -- a
        # weight sync, or `shuffle_weights`, which no longer rebinds -- is seen
        # through it. `_local_q_proj` is the only caller and rebuilds the view on
        # every call anyway (its `is not` guard compares two `param.data`
        # objects, and that attribute hands back a fresh one each access).
        view._parameters = dict(self._parameters)
        view.weight = nn.Parameter(
            self.weight.data.narrow(0, start, length), requires_grad=False
        )
        view.weight.is_shuffled = getattr(self.weight, "is_shuffled", False)

        ws = getattr(self, "weight_scale", None)
        if ws is not None and ws.data.dim() == 2 and ws.data.shape[0] > 1:
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                assert start % group == 0 and length % group == 0, (
                    f"blockscale row view must be {group}-aligned; got "
                    f"start={start} length={length}"
                )
                view.weight_scale = nn.Parameter(
                    ws.data.narrow(0, start // group, length // group),
                    requires_grad=False,
                )
            elif (
                self.quant_type == QuantType.per_Token or self.native_a8_group_rows == 1
            ):
                view.weight_scale = nn.Parameter(
                    ws.data.narrow(0, start, length), requires_grad=False
                )
            else:
                raise NotImplementedError(
                    f"make_row_view does not handle a per-output-channel scale "
                    f"for quant_type={self.quant_type}"
                )
        # per_Tensor / unquantized scales are shared as-is by the shallow copy.

        if self.bias is not None:
            view.bias = nn.Parameter(
                self.bias.data.narrow(0, start, length), requires_grad=False
            )

        view.output_size = length
        # `forward` trims padded rows via `_output_size_before_padding`. A row
        # view lives entirely inside the real rows, so it must not re-trim; the
        # caller is expected to slice below the padding (only per_Token fp8 pads
        # at all, and it appends at the end).
        if getattr(self, "is_output_padded", False):
            assert (
                start + length <= self._output_size_before_padding
            ), "row view must stay within the unpadded rows"
            view.is_output_padded = False
        view.prefix = f"{getattr(self, 'prefix', '')}[rows {start}:{start + length}]"
        return view


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            tp_dim=None,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        override_tp_size: int | None = None,
        override_tp_rank: int | None = None,
        **kwargs,
    ):
        self.tp_dim = 0
        super().__init__(
            input_size,
            output_size,
            self.tp_dim,
            bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
            override_tp_size=override_tp_size,
            override_tp_rank=override_tp_rank,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MergedColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            output_sizes,
            tp_dim=0,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None = None,
    ):
        # Support loading multiple consecutive shards in a single tensor.
        # This mirrors vLLM's behavior for packed modules like QKV.
        if isinstance(loaded_shard_id, tuple):
            if len(loaded_shard_id) == 0:
                raise ValueError("loaded_shard_id tuple cannot be empty")
            if any(idx < 0 or idx >= len(self.output_sizes) for idx in loaded_shard_id):
                raise ValueError(
                    f"Invalid shard id in {loaded_shard_id}; "
                    f"valid range is [0, {len(self.output_sizes) - 1}]"
                )
            if len(loaded_shard_id) > 1 and any(
                b - a != 1 for a, b in itertools.pairwise(loaded_shard_id)
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {loaded_shard_id}."
                )

            # Split loaded_weight by the requested shard sizes (pre-TP),
            # then load each shard individually.
            shard_sizes = [self.output_sizes[i] for i in loaded_shard_id]
            current_offset = 0
            for shard_id, shard_size in zip(loaded_shard_id, shard_sizes):
                if param is getattr(self, "weight_scale", None) or param is getattr(
                    self, "input_scale", None
                ):
                    if self.quant_type == QuantType.per_Tensor:
                        shard_size //= 128
                    else:
                        shard_size //= self.weight_scale_row_group
                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk
            # Split it and load each shard individually.
            param_data = param.data
            # Check if this is weight or weight_scale
            is_scale_param = param is getattr(
                self, "weight_scale", None
            ) or param is getattr(self, "input_scale", None)

            # For fused weight, need to match param shape
            if param_data.shape == loaded_weight.shape:
                # Shapes match - direct copy
                param.weight_loader_process(param_data, loaded_weight)
                return

            # Otherwise, split the fused weight and load each output shard
            current_offset = 0
            for shard_id, output_size in enumerate(self.output_sizes):
                shard_size = output_size
                if is_scale_param and self.weight_scale_row_group > 1:
                    shard_size //= self.weight_scale_row_group

                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                shard_offset = (shard_offset + group - 1) // group
                shard_size = (shard_size + group - 1) // group
            elif self.quant_type == QuantType.per_Tensor:
                param_data = param_data.narrow(self.tp_dim, loaded_shard_id, 1)
                if (
                    loaded_weight.ndim > self.tp_dim
                    and loaded_weight.shape[self.tp_dim] > 1
                ):
                    local_scale = loaded_weight.chunk(self.tp_size, self.tp_dim)[
                        self.tp_rank
                    ].contiguous()
                    if not hasattr(self, "_loaded_weight_scale_for_requant_parts"):
                        self._loaded_weight_scale_for_requant_parts = [None] * len(
                            self.output_sizes
                        )
                    self._loaded_weight_scale_for_requant_parts[loaded_shard_id] = (
                        local_scale
                    )
                    loaded_weight = local_scale.max().view(1, 1)
                else:
                    loaded_weight = loaded_weight.max().view(1, 1)
                param.weight_loader_process(param_data, loaded_weight)
                return

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class QKVZBAParallelLinear(ColumnParallelLinear):
    """Fused QKVZBA linear with deinterleaved output layout.

    Output layout is always ``[q | k | v | z | b | a]`` contiguous on dim-0,
    so the caller can use ``torch.split`` for zero-copy views.

    The weight_loader deinterleaves the Qwen3-Next interleaved per-k-head-group
    checkpoint layout during loading.  Qwen3.5 checkpoints (which have separate
    per-component weights) are loaded via individual shard_ids and work without
    deinterleaving.
    """

    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        tp_size = get_tp_group().world_size
        self.num_k_heads = divide(self.num_k_heads, tp_size)
        self.num_v_heads = divide(self.num_v_heads, tp_size)
        # Output layout: [q_all | k_all | v_all | z_all | b_all | a_all]
        q_size = self.num_k_heads * head_k_dim
        k_size = self.num_k_heads * head_k_dim
        v_size = self.num_v_heads * head_v_dim
        z_size = self.num_v_heads * head_v_dim
        b_size = self.num_v_heads
        a_size = self.num_v_heads
        self._section_sizes = [q_size, k_size, v_size, z_size, b_size, a_size]
        output_sizes = [s * tp_size for s in self._section_sizes]
        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    # -- helpers for deinterleaving during weight loading --

    def _deinterleave_qkvz(self, param_data: torch.Tensor, loaded_weight: torch.Tensor):
        """Scatter interleaved qkvz checkpoint rows into [q|k|v|z] regions.

        Checkpoint layout per k-head group (``QKVZ_DIM_SIZE`` rows):
            [q(head_k_dim) | k(head_k_dim) | v0..vR(R*head_v_dim) | z0..zR(R*head_v_dim)]
        where R = num_v_heads / num_k_heads (KV_HEAD_RATIO).
        """
        nk = self.num_k_heads
        hk = self.head_k_dim
        hv = self.head_v_dim
        R = self.num_v_heads // nk  # KV_HEAD_RATIO
        group_size = 2 * hk + 2 * hv * R

        q_total = nk * hk
        k_total = nk * hk
        v_total = self.num_v_heads * hv

        # TP shard the source
        src = loaded_weight.narrow(
            self.tp_dim, self.tp_rank * nk * group_size, nk * group_size
        )

        for g in range(nk):
            base = g * group_size
            # q rows
            param_data[g * hk : (g + 1) * hk] = src[base : base + hk]
            # k rows
            param_data[q_total + g * hk : q_total + (g + 1) * hk] = src[
                base + hk : base + 2 * hk
            ]
            # v sub-heads
            for s in range(R):
                v_src_start = base + 2 * hk + s * hv
                v_dst_start = q_total + k_total + (g * R + s) * hv
                param_data[v_dst_start : v_dst_start + hv] = src[
                    v_src_start : v_src_start + hv
                ]
            # z sub-heads
            for s in range(R):
                z_src_start = base + 2 * hk + R * hv + s * hv
                z_dst_start = q_total + k_total + v_total + (g * R + s) * hv
                param_data[z_dst_start : z_dst_start + hv] = src[
                    z_src_start : z_src_start + hv
                ]

    def _deinterleave_ba(self, param_data: torch.Tensor, loaded_weight: torch.Tensor):
        """Scatter interleaved ba checkpoint rows into [b|a] regions.

        Checkpoint layout per k-head group (2*R elements):
            [b_sub0, b_sub1, ..., b_subR-1, a_sub0, a_sub1, ..., a_subR-1]
        where R = num_v_heads / num_k_heads.
        """
        nk = self.num_k_heads
        nv = self.num_v_heads
        R = nv // nk
        ba_total = 2 * nv

        # TP shard the source
        src = loaded_weight.narrow(self.tp_dim, self.tp_rank * ba_total, ba_total)

        qkvz_total = sum(self._section_sizes[:4])
        b_offset = qkvz_total
        a_offset = qkvz_total + nv

        for g in range(nk):
            group_base = g * 2 * R
            # b sub-heads
            for s in range(R):
                param_data[b_offset + g * R + s] = src[group_base + s]
            # a sub-heads
            for s in range(R):
                param_data[a_offset + g * R + s] = src[group_base + R + s]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["qkvz", "ba", "qkv", "z", "b", "a"]

        is_scale = param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        )

        # For interleaved checkpoint shards ("qkvz", "ba"), deinterleave
        # weight rows so output is [q|k|v|z|b|a] contiguous.
        if loaded_shard_id == "qkvz" and not is_scale:
            self._deinterleave_qkvz(param_data, loaded_weight)
            return
        if loaded_shard_id == "ba" and not is_scale:
            self._deinterleave_ba(param_data, loaded_weight)
            return

        # For individual shard_ids or scale params, use offset-based loading.
        q_size, k_size, v_size, z_size, b_size, a_size = self._section_sizes
        if loaded_shard_id == "qkvz":
            shard_size = q_size + k_size + v_size + z_size
            shard_offset = 0
        elif loaded_shard_id == "qkv":
            shard_size = q_size + k_size + v_size
            shard_offset = 0
        elif loaded_shard_id == "z":
            shard_size = z_size
            shard_offset = q_size + k_size + v_size
        elif loaded_shard_id == "ba":
            shard_size = b_size + a_size
            shard_offset = q_size + k_size + v_size + z_size
        elif loaded_shard_id == "b":
            shard_size = b_size
            shard_offset = q_size + k_size + v_size + z_size
        elif loaded_shard_id == "a":
            shard_size = a_size
            shard_offset = q_size + k_size + v_size + z_size + b_size

        if is_scale:
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                shard_offset = (shard_offset + group - 1) // group
                shard_size = (shard_size + group - 1) // group
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["qkvz", "ba", "qkv", "z", "b", "a"].index(
                    loaded_shard_id
                )
                shard_size = 1
        start_idx = self.tp_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class QKVZParallelLinear(ColumnParallelLinear):
    """Deinterleaving linear for the separate ``in_proj_qkvz`` checkpoint.

    The Qwen3-Next checkpoint stores qkvz in per-k-head-group interleaved
    layout::

        [q(hk) | k(hk) | v0..vR(R*hv) | z0..zR(R*hv)]  per group

    This class deinterleaves both weight rows **and** per-1x128 block scales
    during loading so the output layout is ``[q_all | k_all | v_all | z_all]``
    contiguous, enabling zero-copy ``torch.split`` at runtime.
    """

    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        tp_size = get_tp_group().world_size
        self.nk = divide(num_k_heads, tp_size)
        self.nv = divide(num_v_heads, tp_size)
        self.R = num_v_heads // num_k_heads  # v-heads per k-head group
        output_size = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    @staticmethod
    def _deinterleave(param_data, src, nk, R, hk, hv):
        """Scatter interleaved rows into [q|k|v|z] regions."""
        group_size = 2 * hk + 2 * hv * R
        q_total = nk * hk
        k_total = nk * hk
        v_total = nk * R * hv

        for g in range(nk):
            base = g * group_size
            # q
            param_data[g * hk : (g + 1) * hk] = src[base : base + hk]
            # k
            param_data[q_total + g * hk : q_total + (g + 1) * hk] = src[
                base + hk : base + 2 * hk
            ]
            # v sub-heads
            for s in range(R):
                v_src = base + 2 * hk + s * hv
                v_dst = q_total + k_total + (g * R + s) * hv
                param_data[v_dst : v_dst + hv] = src[v_src : v_src + hv]
            # z sub-heads
            for s in range(R):
                z_src = base + 2 * hk + R * hv + s * hv
                z_dst = q_total + k_total + v_total + (g * R + s) * hv
                param_data[z_dst : z_dst + hv] = src[z_src : z_src + hv]

    @staticmethod
    def _match_dtype(param_data, loaded_weight):
        """View param_data as loaded_weight's dtype if they differ but share element size.

        This mirrors ``weight_loader_process`` behaviour for FP8 on ROCm where
        the param is ``float8_e4m3fnuz`` but the checkpoint stores
        ``float8_e4m3fn``.  The normalisation happens later in
        ``process_weights_after_loading``.
        """
        if (
            param_data.dtype != loaded_weight.dtype
            and param_data.element_size() == loaded_weight.element_size()
        ):
            return param_data.view(loaded_weight.dtype)
        return param_data

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Load like ColumnParallelLinear (no deinterleave at load time).
        super().weight_loader(param, loaded_weight)

    def process_weights_after_loading(self):
        nk, R = self.nk, self.R
        hk, hv = self.head_k_dim, self.head_v_dim

        # Deinterleave weight rows: interleaved → [q|k|v|z]
        w = self.weight.data
        dw = torch.empty_like(w)
        self._deinterleave(dw, w, nk, R, hk, hv)
        self.weight.data = dw

        # Deinterleave weight_scale rows (per_1x128 block scale)
        ws = getattr(self, "weight_scale", None)
        if ws is not None:
            hk_s, hv_s = hk // 128, hv // 128
            s = ws.data
            ds = torch.empty_like(s)
            self._deinterleave(ds, s, nk, R, hk_s, hv_s)
            self.weight_scale.data = ds

        super().process_weights_after_loading()


class BAParallelLinear(ColumnParallelLinear):
    """Deinterleaving linear for the separate ``in_proj_ba`` checkpoint.

    The Qwen3-Next checkpoint stores ba in per-k-head-group layout::

        [b_s0..b_sR | a_s0..a_sR]  per group  (R = num_v_heads / num_k_heads)

    This class deinterleaves during loading so the output layout is
    ``[b_all | a_all]`` contiguous, enabling zero-copy ``torch.split``.

    ``in_proj_ba`` is always BF16 (listed in ``modules_to_not_convert``),
    so no weight-scale handling is needed.
    """

    def __init__(
        self,
        input_size: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        tp_size = get_tp_group().world_size
        self.nk = divide(num_k_heads, tp_size)
        self.R = num_v_heads // num_k_heads
        self.nv = self.nk * self.R
        output_size = 2 * num_v_heads
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Load like ColumnParallelLinear (no deinterleave at load time).
        super().weight_loader(param, loaded_weight)

    def process_weights_after_loading(self):
        nk, R, nv = self.nk, self.R, self.nv
        w = self.weight.data
        dw = torch.empty_like(w)
        group_size = 2 * R
        for g in range(nk):
            base = g * group_size
            dw[g * R : (g + 1) * R] = w[base : base + R]
            dw[nv + g * R : nv + (g + 1) * R] = w[base + R : base + 2 * R]
        self.weight.data = dw
        super().process_weights_after_loading()


class QKVGParallelLinear(ColumnParallelLinear):
    """QKV + output-Gate parallel linear.

    Rearranges interleaved Q+Gate weights from HF checkpoint into grouped
    layout [Gate, Q, K, V] during loading, so inference uses a single split().
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,  # Gate
            self.num_heads * self.head_size * tp_size,  # Q
            self.num_kv_heads * self.head_size * tp_size,  # K
            self.num_kv_heads * self.head_size * tp_size,  # V
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def _deinterleave(
        self, weight: torch.Tensor, head_stride: int | None = None
    ) -> torch.Tensor:
        """Rearrange Q+Gate from interleaved [q0,g0,q1,g1,...] to grouped [Q_all, Gate_all].

        Args:
            head_stride: number of elements per head along dim 0.
                         Defaults to self.head_size (weights); use head_size//128
                         for per-1x128 scales.
        """
        hs = head_stride if head_stride is not None else self.head_size
        return (
            weight.view(self.num_heads, 2, hs, -1)
            .transpose(0, 1)
            .reshape(self.num_heads * 2 * hs, -1)
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        # Layout: [Gate, Q, K, V]
        if loaded_shard_id == "q":
            # HF q_proj contains interleaved Q+Gate; deinterleave then
            # write Gate to offset 0 and Q to offset q_size
            shard_size = q_size * 2
            shard_offset = 0  # placeholder, handled below
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = kv_size
            shard_offset = q_size * 2
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = kv_size
            shard_offset = q_size * 2 + kv_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        is_scale = param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        )

        if loaded_shard_id == "q":
            # Q+Gate: deinterleave [q0,g0,...] -> [Q_all, Gate_all], then
            # write Gate to offset 0 and Q to offset q_size → layout [Gate, Q, ...]
            if is_scale and self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                # Gate scale -> slot 0, Q scale -> slot 1
                q_scale = loaded_weight.narrow(self.tp_dim, shard_rank, 1)
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 0, 1), q_scale.clone()
                )
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 1, 1), q_scale
                )
                return

            scale_factor = self.weight_scale_row_group if is_scale else 1
            half = q_size // scale_factor
            start_idx = shard_rank * shard_size // scale_factor
            loaded_weight = loaded_weight.narrow(
                self.tp_dim, start_idx, shard_size // scale_factor
            )
            stride = self.head_size // scale_factor
            loaded_weight = self._deinterleave(
                loaded_weight, head_stride=stride if scale_factor > 1 else None
            )
            q_part = loaded_weight.narrow(self.tp_dim, 0, half)
            gate_part = loaded_weight.narrow(self.tp_dim, half, half)
            q_offset = q_size // scale_factor
            # Gate at offset 0, Q at offset q_size
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, 0, half), gate_part
            )
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, q_offset, half), q_part
            )
        else:
            # K or V: straightforward load
            if is_scale:
                if self.weight_scale_row_group > 1:
                    shard_offset //= self.weight_scale_row_group
                    shard_size //= self.weight_scale_row_group
                elif self.quant_type == QuantType.per_Tensor:
                    loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                    # [Gate, Q, K, V] -> K=2, V=3
                    shard_offset = {"k": 2, "v": 3}[loaded_shard_id]
                    shard_size = 1

            start_idx = shard_rank * shard_size
            param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param.weight_loader_process(param_data, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    _nvfp4_global_scale_shard_map: ClassVar[dict[str, int]] = {
        "q": 0,
        "k": 1,
        "v": 2,
    }

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        v_head_size: int | None = None,
        **kwargs,
    ):
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = self.num_kv_heads * self.v_head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                shard_offset = (shard_offset + group - 1) // group
                shard_size = (shard_size + group - 1) // group
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v"].index(loaded_shard_id)
                shard_size = 1

        start_idx = shard_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MinimaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """QKV projection fused with MiniMax-M3 lightning-indexer projections.

    The sparse attention layers emit ``[q | k | v | index_q | index_k]`` from a
    single column-parallel GEMM. ``index_q`` follows the KV-head sharding and
    replication rules, while ``index_k`` is a single replicated head.
    """

    _nvfp4_global_scale_shard_map: ClassVar[dict[str, int]] = {
        "q": 0,
        "k": 1,
        "v": 2,
        "index_q": 3,
        "index_k": 4,
    }

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        if total_num_index_heads != total_num_kv_heads:
            raise ValueError(
                "MiniMax-M3 index_q must shard like KV heads: "
                "total_num_index_heads must equal total_num_kv_heads."
            )

        self.head_size = head_size
        self.v_head_size = head_size
        self.index_head_size = index_head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads

        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        # Indexer-only CP scores every index head on every rank, so index_q is
        # replicated at full width instead of following the KV-head sharding --
        # the same treatment index_k already gets. index_k, q, k and v are
        # untouched, so this widens the GEMM by 3 head-columns and nothing else.
        from atom.distributed.indexer_cp import indexer_cp_enabled

        self.indexer_cp = indexer_cp_enabled()
        self.num_index_heads = (
            self.total_num_index_heads if self.indexer_cp else self.num_kv_heads
        )

        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
            self.num_index_heads * self.index_head_size * tp_size,
            self.index_head_size * tp_size,
        ]

        ColumnParallelLinear.__init__(
            self,
            hidden_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
            **kwargs,
        )

    def _shard_offset_size(self, loaded_shard_id: str) -> tuple[int, int]:
        h = self.head_size
        ih = self.index_head_size
        nq = self.num_heads
        nkv = self.num_kv_heads
        nidx = self.num_index_heads
        mapping = {
            "q": (0, nq * h),
            "k": (nq * h, nkv * h),
            "v": ((nq + nkv) * h, nkv * h),
            "index_q": ((nq + 2 * nkv) * h, nidx * ih),
            "index_k": ((nq + 2 * nkv) * h + nidx * ih, ih),
        }
        if loaded_shard_id not in mapping:
            raise ValueError(
                "MiniMax-M3 QKV/indexer shard id must be one of "
                "'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id!r}."
            )
        return mapping[loaded_shard_id]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                shard_offset = (shard_offset + group - 1) // group
                shard_size = (shard_size + group - 1) // group
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v", "index_q", "index_k"].index(
                    loaded_shard_id
                )
                shard_size = 1

        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k" or (
            loaded_shard_id == "index_q" and self.indexer_cp
        ):
            # Replicated: shard_size already spans every index head, so there is
            # only one shard to take.
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.tp_dim,
            shard_rank * shard_size,
            shard_size,
        )
        param.weight_loader_process(param_data, loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_rank = get_tp_group().rank_in_group
        super().__init__(
            input_size,
            output_size,
            tp_dim=1,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param is not getattr(self, "bias", None):
            if (
                param is getattr(self, "weight_scale", None)
                or param is getattr(self, "input_scale", None)
            ) and self.quant_type == QuantType.per_Tensor:
                if loaded_weight.ndim > 0 and loaded_weight.shape[0] > 1:
                    self._loaded_weight_scale_for_requant = loaded_weight.contiguous()
                param.weight_loader_process(param_data, loaded_weight.max().view(1, 1))
                return
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.view(1, 1)
            if loaded_weight.ndim <= self.tp_dim:
                # dims < tp_dim (1D per-channel scale with
                # tp_dim=1)
                param.weight_loader_process(param_data, loaded_weight)
                return
            shard_size = param_data.size(self.tp_dim)
            if loaded_weight.size(self.tp_dim) == 1 and self.tp_size > 1:
                loaded_weight = loaded_weight.repeat(1, self.tp_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        else:
            if self.tp_size > 0 and self.tp_rank != 0:
                loaded_weight.zero_()
        param.weight_loader_process(param_data, loaded_weight)


class MergedReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_size
        super().__init__(
            input_size,
            sum(output_size),  # ？
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def shard_view(self, index: int) -> "LinearBase":
        """A layer computing only the rows shard `index` was declared with.

        The fused matrix is the loader's unit; a consumer of one half -- the
        draft's KV projection out of a fused `[wq_a; wkv]` -- wants the rows
        without restating where they start.
        """
        assert 0 <= index < len(self.output_sizes), index
        return self.make_row_view(
            sum(self.output_sizes[:index]), self.output_sizes[index]
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):  # ？
        param_data = param.data
        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.weight_scale_row_group > 1:
                group = self.weight_scale_row_group
                shard_offset = (
                    sum(self.output_sizes[:loaded_shard_id]) + group - 1
                ) // group
                shard_size = (self.output_sizes[loaded_shard_id] + group - 1) // group
            elif self.quant_type == QuantType.per_Tensor:
                shard_offset = loaded_shard_id
                shard_size = 1
            else:
                # Per-channel same layout as weights
                shard_offset = sum(self.output_sizes[:loaded_shard_id])
                shard_size = self.output_sizes[loaded_shard_id]
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param.weight_loader_process(param_data, loaded_weight)
