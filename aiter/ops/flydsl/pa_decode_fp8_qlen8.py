# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""gfx950 Qlen8/GQA16 FP8 paged full-attention schedule.

This narrow path keeps Dqk=128/192 and Dv=128 separate. The shared FlyDSL
reducer combines normalized partials from the small-grid and wave schedules.
"""

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.pa_decode_fp8_small import compile_pa_decode_fp8_small
from .kernels.pa_decode_fp8_wave import compile_pa_decode_fp8_wave
from .kernels.tensor_shim import _run_compiled


def pa_decode_fp8_qlen8(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    num_partitions: int,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    max_logits: torch.Tensor | None = None,
    exp_sums: torch.Tensor | None = None,
    temporary_output: torch.Tensor | None = None,
) -> None:
    """Write [B*8, 16, 128] BF16 output from page64 vectorized FP8 KV."""
    from .pa_decode import launch_pa_decode_ps_reduce

    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("FP8 Qlen8 decode requires gfx950")
    if context_lengths.ndim != 1 or context_lengths.numel() < 1:
        raise ValueError("context_lengths must be a nonempty [B] vector")
    batch = context_lengths.numel()
    head_dim = query.shape[-1]
    expected_query = (batch * 8, 16, head_dim)
    if (
        head_dim not in (128, 192)
        or query.shape != expected_query
        or query.dtype != torch.bfloat16
        or query.stride(-1) != 1
    ):
        raise ValueError(
            "FP8 Qlen8 requires BF16 query [B*8, 16, Dqk=128/192] "
            f"with a contiguous head axis, got {query.shape}/{query.dtype}"
        )
    if (
        output.shape != (batch * 8, 16, 128)
        or output.dtype != torch.bfloat16
        or output.stride(-1) != 1
    ):
        raise ValueError("FP8 Qlen8 requires BF16 output [B*8, 16, 128]")
    if (
        key_cache.ndim != 5
        or key_cache.shape[1:] != (1, head_dim // 16, 64, 16)
        or key_cache.dtype != torch.float8_e4m3fn
        or not key_cache.is_contiguous()
    ):
        raise ValueError("FP8 Qlen8 requires vectorized page64 FP8 K")
    if (
        value_cache.shape != (key_cache.shape[0], 1, 4, 128, 16)
        or value_cache.dtype != key_cache.dtype
        or not value_cache.is_contiguous()
    ):
        raise ValueError("FP8 Qlen8 requires vectorized page64 FP8 V128")
    if (
        block_tables.shape[0] != batch
        or block_tables.ndim != 2
        or block_tables.dtype != torch.int32
        or not block_tables.is_contiguous()
    ):
        raise ValueError("block_tables must be contiguous int32 [B, max_pages]")
    if context_lengths.dtype != torch.int32 or not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be contiguous int32 [B]")
    if not isinstance(num_partitions, int) or not 1 <= num_partitions <= 64:
        raise ValueError("num_partitions must be in [1, 64]")
    if not 0 < softmax_scale < float("inf"):
        raise ValueError("softmax_scale must be finite and positive")
    device = query.device
    if device.type != "cuda" or any(
        tensor.device != device
        for tensor in (
            output,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
        )
    ):
        raise ValueError("all FP8 Qlen8 decode tensors must be on the query device")
    for name, scale in (("key_scale", key_scale), ("value_scale", value_scale)):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.shape != (1,)
            or scale.dtype != torch.float32
            or scale.device != device
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32 GPU scalar")

    scalar_shape = (batch, 1, num_partitions, 128)
    if num_partitions == 1:
        max_logits = key_scale if max_logits is None else max_logits
        exp_sums = key_scale if exp_sums is None else exp_sums
        temporary_output = output if temporary_output is None else temporary_output
    else:
        supplied = (max_logits, exp_sums, temporary_output)
        if any(tensor is None for tensor in supplied):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("preallocate FP8 Qlen8 partials before graph capture")
            if any(tensor is not None for tensor in supplied):
                raise ValueError("supply all partial buffers, or none")
            max_logits = torch.empty(scalar_shape, dtype=torch.float32, device=device)
            exp_sums = torch.empty_like(max_logits)
            temporary_output = torch.empty(
                (*scalar_shape, 128), dtype=torch.bfloat16, device=device
            )
        if (
            max_logits.shape != scalar_shape
            or exp_sums.shape != scalar_shape
            or temporary_output.shape != (*scalar_shape, 128)
            or max_logits.dtype != torch.float32
            or exp_sums.dtype != torch.float32
            or temporary_output.dtype != torch.bfloat16
            or any(
                not tensor.is_contiguous() or tensor.device != device
                for tensor in (max_logits, exp_sums, temporary_output)
            )
        ):
            raise ValueError(
                "FP8 Qlen8 partial buffers have incompatible shape or dtype"
            )

    compile_kernel = (
        compile_pa_decode_fp8_small
        if batch * num_partitions <= 64
        else compile_pa_decode_fp8_wave
    )
    compiled = compile_kernel(head_dim, num_partitions, softmax_scale)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            compiled["launch"],
            output,
            max_logits.view(-1),
            exp_sums.view(-1),
            temporary_output.view(-1),
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            int(block_tables.shape[1]),
            batch,
            1,
            0,
            0,
            int(query.stride(0)),
            int(query.stride(1)),
            stream,
        )
        if num_partitions > 1:
            output_5d = output.view(batch, 8, 1, 16, 128)
            launch_pa_decode_ps_reduce(
                output_5d,
                exp_sums,
                max_logits,
                temporary_output,
                None,
                output_5d.stride(0),
                output_5d.stride(1),
                output_5d.stride(2),
                output_5d.stride(3),
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                temporary_output.stride(0),
                temporary_output.stride(1),
                temporary_output.stride(2),
                temporary_output.stride(3),
                query_seq_len=8,
                query_group_size=16,
                head_size=128,
                context_partition_num=num_partitions,
                stream=stream,
            )
