import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.mxmoe_reduce_ep32 import compile_native_reduce
from .kernels.tensor_shim import ptr_arg
from .moe_kernels import _run_compiled

# Only the native GEMM2 producer and standard Opus aux sort supply these identities.
# Tensor dtype/strides cannot prove weighting or route masking.
_NATIVE_BF16_SOURCE = object()
_NATIVE_FP32_SOURCE = object()
_OPUS_REVERSE_SOURCE = object()


def _validate_native_reduce_inputs(
    x, reverse, weights, mask, ids, valid, out, preweighted
):
    tensors = {
        "x": x,
        "reverse": reverse,
        "weights": weights,
        "mask": mask,
        "ids": ids,
        "valid": valid,
        "out": out,
    }
    if any(not isinstance(t, torch.Tensor) for t in tensors.values()):
        raise ValueError("EP32 native reduce arguments must be tensors")
    if out.ndim != 2 or out.shape[1] != 3584 or out.dtype != torch.bfloat16:
        raise ValueError(
            "EP32 native reduce requires a two-dimensional H3584 BF16 output"
        )
    m = out.shape[0]
    if ids.ndim != 2 or ids.shape[0] != m or ids.shape[1] not in (16, 17):
        raise ValueError(
            "EP32 native reduce IDs must match output tokens and top-k16/17"
        )
    topk = ids.shape[1]
    if reverse.shape not in (ids.shape, (m * topk,)):
        raise ValueError(
            "EP32 native reduce reverse must cover the complete token/top-k table"
        )
    staging_dtype = torch.bfloat16 if preweighted else torch.float32
    if x.ndim != 4 or tuple(x.shape[1:]) != (224, 64, 4) or x.dtype != staging_dtype:
        raise ValueError(
            "EP32 native staging must be [row16,224,lane64,fragment4] with matching dtype"
        )
    expected = {
        "reverse": torch.int32,
        "weights": torch.float32,
        "mask": torch.int32,
        "ids": torch.int32,
        "valid": torch.int32,
    }
    if any(tensors[name].dtype != dtype for name, dtype in expected.items()):
        raise ValueError(
            "EP32 native reduce requires INT32 routes/counts/mask and FP32 weights"
        )
    if weights.ndim != 1 or mask.ndim != 1 or not mask.numel():
        raise ValueError(
            "EP32 native reduce requires one-dimensional weights and a nonempty mask"
        )
    if valid.ndim != 1 or valid.numel() != 2:
        raise ValueError(
            "EP32 native reduce valid must be an INT32 [sorted_rows,valid_tokens] tensor"
        )
    rows = x.shape[0] * 16
    if not preweighted and weights.numel() < rows:
        raise ValueError(
            "EP32 FP32 native staging requires weights covering its row capacity"
        )
    out_start = out.data_ptr()
    out_end = out_start + out.numel() * out.element_size()
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != out.device:
            raise ValueError(f"EP32 native reduce {name} must be on the output GPU")
        if not tensor.is_contiguous():
            raise ValueError(f"EP32 native reduce {name} must be contiguous")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > 0xFFFFFFFF:
            raise ValueError(
                f"EP32 native reduce {name} exceeds the 32-bit buffer address range"
            )
        alignment = 16 if name in ("x", "out") else 4
        start = tensor.data_ptr()
        if start % alignment:
            raise ValueError(
                f"EP32 native reduce {name} requires {alignment}-byte alignment"
            )
        if (
            name != "out"
            and nbytes
            and out_end > out_start
            and start < out_end
            and out_start < start + nbytes
        ):
            raise ValueError(f"EP32 native reduce output overlaps {name}")
    return m, topk, rows


def ep32_native_reduce(
    x,
    reverse,
    weights,
    mask,
    ids,
    valid,
    out,
    *,
    preweighted_bf16=False,
    reverse_masked=False,
    output_zeroed=False,
    _native_source=None,
    _reverse_source=None,
):
    """Reduce a producer-verified native fragment view; decline unverified inputs."""
    if any(
        type(flag) is not bool
        for flag in (preweighted_bf16, reverse_masked, output_zeroed)
    ):
        raise ValueError("EP32 native reduce mode flags must be Python booleans")
    source = _NATIVE_BF16_SOURCE if preweighted_bf16 else _NATIVE_FP32_SOURCE
    if _native_source is not source:
        raise ValueError(
            "EP32 native reduce requires the matching native GEMM2 producer contract"
        )
    if _reverse_source is not None and _reverse_source is not _OPUS_REVERSE_SOURCE:
        raise ValueError("EP32 native reduce received an unknown reverse-sort contract")
    m, topk, rows = _validate_native_reduce_inputs(
        x, reverse, weights, mask, ids, valid, out, preweighted_bf16
    )
    if not m:
        return
    trusted_sort = _reverse_source is _OPUS_REVERSE_SOURCE
    if not trusted_sort:
        # External presort counts have no Opus bounds guarantee. This stays on
        # the GPU during graph capture; the trusted hot path skips the assertion.
        counts_in_bounds = (
            (valid[0] >= 0) & (valid[0] <= rows) & (valid[1] >= 0) & (valid[1] <= m)
        )
        torch._assert_async(
            counts_in_bounds, "EP32 native reduce valid counts exceed buffer capacity"
        )
    masked_routes = trusted_sort and reverse_masked
    optimized_4096 = (
        preweighted_bf16
        and masked_routes
        and output_zeroed
        and m == 4096
        and topk == 16
        and get_gfx_runtime() == "gfx950"
    )
    optimized_8192 = (
        preweighted_bf16
        and masked_routes
        and m == 8192
        and topk == 16
        and get_gfx_runtime() == "gfx950"
    )
    _run_compiled(
        compile_native_reduce(topk, preweighted_bf16, optimized_4096, optimized_8192),
        (
            ptr_arg(x),
            ptr_arg(reverse),
            ptr_arg(weights),
            ptr_arg(mask),
            ptr_arg(ids),
            ptr_arg(valid),
            ptr_arg(out),
            m,
            mask.numel(),
            torch.cuda.current_stream(out.device),
        ),
    )
