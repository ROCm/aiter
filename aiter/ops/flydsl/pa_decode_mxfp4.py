# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged-attention decode on gfx950: K and V are e2m1 with e8m0 scales.

A sibling of ``pa_decode`` that shares its work plan and partition reducer. Each
of K and V reaches the kernel as a data view and a scale view over the same
byte pages (see ``kernels.pa_decode_mxfp4_kernel``). Only the decode shape the
MiniMax-M3 sparse path produces is served: ``query_length == 1`` (every sparse
query is its own row with its own block table) and head_dim 128.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.kernels_common import ceildiv
from .kernels.pa_decode.traits import KV_COMPUTE_BLOCK, max_tiles_per_task
from .kernels.pa_decode_mxfp4_kernel import (
    DATA_BYTES,
    HEAD_DIM,
    MAX_STAGED_PAGES,
    PAGE_STAGE_ALIGN,
    SCALE_BYTES,
    build_pa_decode_mxfp4,
)
from .kernels.pa_decode_plan import PADecodePlan
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .pa_decode import launch_pa_decode_ps_reduce

__all__ = ["pa_decode_mxfp4"]

_BLOCK_SIZES = (16, 64, 128)
_MAX_GROUP_SIZE = 16


def _check_view(name, t, device, tail, block_size, kv_heads):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {t.dtype}")
    if t.device != device:
        raise ValueError(f"{name} must be on {device}, got {t.device}")
    if t.dim() != 4 or t.shape[2] != block_size or t.shape[3] != tail:
        raise ValueError(
            f"{name} must be [pages, kv_heads, {block_size}, {tail}], "
            f"got {tuple(t.shape)}"
        )
    if t.shape[1] != kv_heads:
        raise ValueError(f"{name} kv heads {t.shape[1]} != {kv_heads}")
    if t.stride(3) != 1 or t.stride(2) != tail:
        raise ValueError(
            f"{name} must be token-major (stride(2) == {tail}, stride(3) == 1), "
            f"got strides {t.stride()}"
        )
    if tail == DATA_BYTES and (t.stride(0) % 16 or t.stride(1) % 16):
        raise ValueError(f"{name} page and head strides must be multiples of 16")
    if max(t.stride(0), t.stride(1)) >= 2**31:
        raise ValueError(f"{name} strides must fit int32")


def pa_decode_mxfp4(
    output: torch.Tensor,  # [num_seqs, num_query_heads, 128] bf16/f16
    query: torch.Tensor,  # [num_seqs, num_query_heads, 128] float8_e4m3fn
    k_data: torch.Tensor,  # uint8 [pages, kv_heads, block_size, 64]
    k_scale: torch.Tensor,  # uint8 [pages, kv_heads, block_size, 4]  (e8m0)
    v_data: torch.Tensor,  # uint8 [pages, kv_heads, block_size, 64]
    v_scale: torch.Tensor,  # uint8 [pages, kv_heads, block_size, 4]  (e8m0)
    context_lengths: torch.Tensor,  # [num_seqs] int32
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32
    softmax_scale: float,
    *,
    exp_sums: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    temporary_output: torch.Tensor | None = None,
    work_plan: PADecodePlan | None = None,
    max_context_length: int | None = None,
    max_partitions: int | None = None,
) -> None:
    """Decode one query token per sequence over an MXFP4 paged K/V cache.

    ``query`` is float8_e4m3fn, used as stored (scale 1).

    Each of the four views carries its own page and head stride (bytes); data
    strides must be multiples of 16. Pages outside ``[0, pages)`` are not
    checked. Scale bytes 0 and 255 are not supported (see the kernel docstring).
    Invalid tail tokens of the last page may hold anything.

    ``work_plan`` is built with ``plan_pa_decode`` over ``context_lengths`` and
    ``k_data.shape[1]`` KV heads, exactly as for ``pa_decode``; scratch is
    ``[kv_heads, capacity, group_size(, 128)]``. ``max_context_length`` bounds the
    scheduling, as in ``pa_decode``.

    With ``work_plan=None`` no plan is built: each sequence is one task that
    writes its row directly (``max_partitions`` may only be 1), so nothing is
    specialized on ``num_seqs``. Every context must be at most
    ``max_context_length`` (default: the block-table capacity); a longer one is
    silently truncated, and a context-0 row is zero-filled. With a plan whose
    ``max_partitions`` is 1 the row is written directly too, and a context-0 row
    is left unwritten.
    """
    if work_plan is not None and not isinstance(work_plan, PADecodePlan):
        raise TypeError("work_plan must be a PADecodePlan or None")
    if work_plan is not None and max_partitions is not None:
        raise ValueError("max_partitions applies to the plan-free form only")
    if max_partitions not in (None, 1):
        raise NotImplementedError("the plan-free form runs one task per sequence")
    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise NotImplementedError(f"pa_decode_mxfp4 supports gfx950, got {arch}")
    for name, t in (
        ("output", output),
        ("query", query),
        ("context_lengths", context_lengths),
        ("block_tables", block_tables),
    ):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if query.device.type != "cuda":
        raise ValueError(f"query must be on a CUDA device, got {query.device}")
    dev = query.device
    if query.dim() != 3 or query.shape[2] != HEAD_DIM:
        raise NotImplementedError(
            f"pa_decode_mxfp4 needs query [rows, heads, {HEAD_DIM}], "
            f"got {tuple(query.shape)}"
        )
    if output.shape != query.shape:
        raise ValueError(
            f"output shape {tuple(output.shape)} must match "
            f"query shape {tuple(query.shape)}"
        )
    if output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"output must be bf16/f16, got {output.dtype}")
    if query.dtype != torch.float8_e4m3fn:
        raise TypeError(f"query must be float8_e4m3fn, got {query.dtype}")
    if query.stride(2) != 1 or output.stride(2) != 1:
        raise ValueError("query and output need a contiguous head_dim axis")
    if query.stride(0) % 16 or query.stride(1) % 16:
        raise ValueError("query row and head strides must be multiples of 16 bytes")

    num_seqs, num_q_heads, _ = query.shape
    if k_data.dim() != 4:
        raise ValueError(f"k_data must be 4D, got {tuple(k_data.shape)}")
    block_size = k_data.shape[2]
    if block_size not in _BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {_BLOCK_SIZES}, got {block_size}")
    num_kv_heads = k_data.shape[1]
    if num_kv_heads < 1 or num_q_heads % num_kv_heads:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be a multiple of kv heads ({num_kv_heads})"
        )
    group_size = num_q_heads // num_kv_heads
    if group_size > _MAX_GROUP_SIZE:
        raise NotImplementedError(
            f"group_size {group_size} > {_MAX_GROUP_SIZE}: one M-tile per CTA"
        )
    for name, t, tail in (
        ("k_data", k_data, DATA_BYTES),
        ("v_data", v_data, DATA_BYTES),
        ("k_scale", k_scale, SCALE_BYTES),
        ("v_scale", v_scale, SCALE_BYTES),
    ):
        _check_view(name, t, dev, tail, block_size, num_kv_heads)

    if context_lengths.dtype != torch.int32 or block_tables.dtype != torch.int32:
        raise TypeError("context_lengths and block_tables must be int32")
    if context_lengths.dim() != 1 or context_lengths.shape[0] != num_seqs:
        raise ValueError("context_lengths must be [num_seqs]")
    if block_tables.dim() != 2 or block_tables.shape[0] != num_seqs:
        raise ValueError("block_tables must be [num_seqs, max_blocks_per_seq]")
    if not (context_lengths.is_contiguous() and block_tables.is_contiguous()):
        raise ValueError("context_lengths and block_tables must be contiguous")
    for name, t in (
        ("output", output),
        ("context_lengths", context_lengths),
        ("block_tables", block_tables),
    ):
        if t.device != dev:
            raise ValueError(f"{name} must be on {dev}, got {t.device}")
    if num_seqs == 0:
        return

    max_blocks_per_seq = block_tables.shape[1]
    schedule_context = max_blocks_per_seq * block_size
    if max_context_length is not None:
        if (
            isinstance(max_context_length, bool)
            or not isinstance(max_context_length, int)
            or not 0 <= max_context_length <= schedule_context
        ):
            raise ValueError(
                "max_context_length must be a nonnegative host integer within "
                "block-table capacity"
            )
        schedule_context = ceildiv(max_context_length, block_size) * block_size
    if work_plan is not None:
        work_plan.validate(num_seqs, num_kv_heads, dev)
        if work_plan.sliding_window != 0:
            raise NotImplementedError("pa_decode_mxfp4 has no sliding window")
        capacity = work_plan.capacity
        parts = work_plan.max_partitions
        tile_bound = max_tiles_per_task(
            num_seqs, capacity, work_plan.max_partitions, max(schedule_context, 1)
        )
        task_tiles = 0
    else:
        task_tiles = max(1, ceildiv(schedule_context, KV_COMPUTE_BLOCK))
        parts = 1
        capacity = num_seqs
        tile_bound = task_tiles
    # One task per sequence: the CTA writes the normalized row itself.
    direct_output = parts == 1
    if direct_output and (
        (output.stride(0) * output.element_size()) % 16
        or (output.stride(1) * output.element_size()) % 16
    ):
        raise ValueError("direct_output needs output row/head strides of 16 bytes")

    # Pages one task can own, staged in LDS in 256-entry units.
    pages = max(tile_bound * (KV_COMPUTE_BLOCK // block_size), 1)
    stage_pages = max(
        PAGE_STAGE_ALIGN, ceildiv(pages, PAGE_STAGE_ALIGN) * PAGE_STAGE_ALIGN
    )
    if stage_pages > MAX_STAGED_PAGES:
        raise NotImplementedError(
            f"a task can own {pages} pages; a task may own at most "
            f"{MAX_STAGED_PAGES} (block_size {block_size}); use a plan with more "
            "max_partitions"
        )

    expected = (num_kv_heads, capacity, group_size)
    if not direct_output:  # the direct path has no partials, hence no scratch
        if max_logits is None:
            max_logits = torch.empty(*expected, dtype=torch.float32, device=dev)
        if exp_sums is None:
            exp_sums = torch.empty(*expected, dtype=torch.float32, device=dev)
        if temporary_output is None:
            temporary_output = torch.empty(
                *expected, HEAD_DIM, dtype=output.dtype, device=dev
            )
        for name, t, shape, dtype in (
            ("max_logits", max_logits, expected, torch.float32),
            ("exp_sums", exp_sums, expected, torch.float32),
            ("temporary_output", temporary_output, (*expected, HEAD_DIM), output.dtype),
        ):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} shape {tuple(t.shape)} != {shape}")
            if t.dtype != dtype or t.device != dev or not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype} on {dev}")

    kernel = build_pa_decode_mxfp4(
        group_size=group_size,
        block_size=block_size,
        stage_pages=stage_pages,
        out_f16=output.dtype == torch.float16,
        direct=direct_output,
        tiles_per_task=task_tiles,
    )
    out_dt = fx.Float16 if output.dtype == torch.float16 else fx.BFloat16
    with torch.cuda.device(dev):
        stream = torch.cuda.current_stream(dev)
        _run_compiled(
            kernel,
            ptr_arg(output if direct_output else max_logits, fx.Float32),
            ptr_arg(output if direct_output else exp_sums, fx.Float32),
            ptr_arg(output if direct_output else temporary_output, out_dt),
            ptr_arg(query.view(torch.uint8), fx.Uint8),
            ptr_arg(k_data, fx.Uint8),
            ptr_arg(k_scale, fx.Uint8),
            ptr_arg(v_data, fx.Uint8),
            ptr_arg(v_scale, fx.Uint8),
            ptr_arg(block_tables, fx.Int32),
            ptr_arg(
                block_tables if work_plan is None else work_plan.work_info, fx.Int32
            ),
            max_blocks_per_seq,
            query.stride(0),
            query.stride(1),
            k_data.stride(0) // 4,
            k_data.stride(1) // 4,
            k_scale.stride(0),
            k_scale.stride(1),
            v_data.stride(0) // 4,
            v_data.stride(1) // 4,
            v_scale.stride(0),
            v_scale.stride(1),
            float(softmax_scale),
            ptr_arg(output, out_dt),
            output.stride(0),
            output.stride(1),
            ptr_arg(context_lengths, fx.Int32),
            ptr_arg(block_tables, fx.Int32),  # unused reduce-info slot
            capacity,
            num_kv_heads,
            stream,
        )
        if direct_output:
            return
        output_5d = output.reshape(num_seqs, 1, num_kv_heads, group_size, HEAD_DIM)
        reduce_partitions = min(
            parts,
            capacity,
            max(1, ceildiv(schedule_context, KV_COMPUTE_BLOCK)),
        )
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
            max_logits.stride(0),
            max_logits.stride(1),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            query_seq_len=1,
            query_group_size=group_size,
            head_size=HEAD_DIM,
            max_partitions=reduce_partitions,
            stream=stream,
            reduce_info=work_plan.reduce_info,
        )
