# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.tensor_shim import ptr_arg

__all__ = [
    "flydsl_mla_decode_reduce",
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
]


_DECODE_PAGE_SIZE = 64
_DECODE_NUM_Q_HEADS = (64, 128)
# At or below this the kernel runs two waves that split the PV output dim rather
# than the heads, which leaves each wave holding only part of a head's output
# row. The single-split fast path writes bf16 straight out and has not been
# validated for that, so these head counts always take the split-merge path.
_DECODE_NARROW_HEADS = 32
# Heads one block covers. Head counts above this are spread over the grid, and
# each of those head groups reads the whole KV cache, which is what makes the
# achieved bandwidth of a heads=128 launch twice its unique-byte figure. Read by
# op_tests/_bench_flydsl_vs_gluon.py to model traffic.
_DECODE_HEADS_PER_BLOCK = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_runtime(condition, message):
    if not condition:
        raise RuntimeError(message)


def _require_layout(name, tensor, dtype, shape=None):
    """Check the dtype, shape and contiguity a kernel argument must satisfy.

    ``shape`` entries may be ``None`` to leave a dimension free.
    """
    _require(tensor.dtype == dtype, f"{name}: expected {dtype}, got {tensor.dtype}")
    if shape is not None:
        actual = tuple(tensor.shape)
        ok = len(actual) == len(shape) and all(
            want is None or got == want for got, want in zip(actual, shape)
        )
        _require(ok, f"{name}: expected shape {list(shape)}, got {list(actual)}")
    _require(tensor.is_contiguous(), f"{name}: expected a contiguous tensor")


def _validate_pagesize1_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require_runtime(
        arch == "gfx1250",
        f"expected gfx1250, got {arch or 'unknown'}",
    )
    softmax_scale = float(softmax_scale)
    batch = q.size(0)
    _require_layout("q", q, torch.float8_e4m3fn, (batch, 128, 576))
    _require_layout("kv_buffer", kv_buffer, torch.float8_e4m3fn, (None, 1, 1, 576))
    _require_layout("split_data", split_data, torch.float32, (None, 128, 512))
    _require_layout("split_lse", split_lse, torch.float32, (split_data.size(0), 128))
    _require_layout("kv_page_indices", kv_page_indices, torch.int32)
    _require_layout("work_indptr", work_indptr, torch.int32)
    _require_layout("work_info", work_info, torch.int32, (None, 8))

    properties = torch.cuda.get_device_properties(q.device)
    lds_size = getattr(properties, "shared_memory_per_multiprocessor", None)
    lds_size = int(lds_size) if lds_size is not None else get_lds_capacity_bytes(arch)
    return work_indptr.numel() - 1, lds_size, softmax_scale


def flydsl_mla_pagesize1_fp8_fp8(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    stream=None,
):
    num_cus, lds_size, softmax_scale = _validate_pagesize1_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
    )
    from .kernels.mla_gfx1250.mla_pagesize1_fp8_fp8 import (
        launch_mla_pagesize1_fp8_fp8,
    )

    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    launch_mla_pagesize1_fp8_fp8(
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        flyc.from_c_void_p(fx.BFloat16, None),
        ptr_arg(q, fx.Int8),
        ptr_arg(kv_buffer, fx.Int8),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(work_indptr, fx.Int32),
        ptr_arg(work_info, fx.Int32),
        softmax_scale,
        kv_buffer.size(0),
        num_cus,
        lds_size,
        stream=stream,
    )


def _validate_pagesize64_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    num_splits,
    page_size,
):

    batch = q.size(0)
    fp8 = torch.float8_e4m3fn
    _require_layout("kv_buffer", kv_buffer, fp8, (None, 64 * 576))
    for name, tensor, shape in (
        ("kv_indptr", kv_indptr, (batch + 1,)),
        ("kv_page_indices", kv_page_indices, None),
        ("kv_last_page_lens", kv_last_page_lens, (batch,)),
        ("qo_indptr", qo_indptr, (batch + 1,)),
        ("num_kv_splits_indptr", num_kv_splits_indptr, (batch + 1,)),
        ("q_scale", q_scale, (1,)),
        ("kv_scale", kv_scale, (1,)),
    ):
        dtype = torch.float32 if name.endswith("scale") else torch.int32
        _require_layout(name, tensor, dtype, shape)

    if num_splits == 1:
        _require_layout("split_data", split_data, torch.bfloat16, (batch, 128, 512))
    else:
        _require_layout(
            "split_data", split_data, torch.float32, (batch, num_splits, 128, 512)
        )
    _require_layout("split_lse", split_lse, torch.float32, (batch, num_splits, 128, 1))
    return batch


def flydsl_mla_pagesize64_fp8_fp8(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    *,
    page_size=64,
    stream=None,
):
    batch = _validate_pagesize64_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
        num_splits,
        page_size,
    )
    from .kernels.mla_gfx1250.mla_pagesize64_fp8_fp8 import (
        launch_mla_pagesize64_fp8_fp8,
    )

    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    output_type = fx.BFloat16 if num_splits == 1 else fx.Float32
    launch_mla_pagesize64_fp8_fp8(
        ptr_arg(split_data, output_type),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(q, fx.Int8),
        ptr_arg(kv_buffer, fx.Int8),
        ptr_arg(kv_indptr, fx.Int32),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(kv_last_page_lens, fx.Int32),
        ptr_arg(qo_indptr, fx.Int32),
        ptr_arg(num_kv_splits_indptr, fx.Int32),
        ptr_arg(q_scale, fx.Float32),
        ptr_arg(kv_scale, fx.Float32),
        float(softmax_scale),
        batch,
        num_splits,
        int(num_splits == 1),
        stream=stream,
    )


def flydsl_mla_decode_reduce(
    split_data,  # fp32 [total_tokens, num_splits, num_heads, v_head_dim]
    split_lse,  # fp32 [total_tokens, num_splits, num_heads, 1]
    seqused_k,  # int32 [batch]
    out,  # [total_tokens, num_heads, v_head_dim]
    num_splits,
    num_tokens_per_seq,
    stream=None,
):
    """Merge the per-split partials of :func:`flydsl_mla_decode_fwd` into ``out``.

    Split out as its own entry point so a caller that ran stage 1 with
    ``skip_reduce`` can drive the merge itself, and so the two stages can be
    timed independently.
    """
    from .kernels.mla_reduce import compile_mla_decode_reduce

    total_tokens, _, num_heads, v_head_dim = split_data.shape
    out_dtype = "bf16" if out.dtype == torch.bfloat16 else "fp16"
    launch = compile_mla_decode_reduce(H=num_heads, Dv=v_head_dim, out_dtype=out_dtype)
    if stream is None:
        stream = torch.cuda.current_stream(out.device)
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    launch(
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(seqused_k, fx.Int32),
        ptr_arg(out, out_type),
        total_tokens,
        seqused_k.numel(),
        num_splits,
        num_tokens_per_seq,
        stream=stream,
    )
    return out
