# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = [
    "flydsl_mla_decode_reduce",
    "flydsl_mla_pagesize1_bf16_bf16",
    "flydsl_mla_pagesize1_bf16_fp8",
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
]

_PAGESIZE1_NUM_Q_HEADS = (16, 32, 64, 96, 128)
_PAGESIZE1_PACKED_Q_HEADS = 64


def flydsl_mla_pagesize1_bf16_fp8(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    kv_scale,
    final_lse=None,
    max_seqlen_q=8,
    causal=True,
    q_scale=None,
    qo_indptr=None,
    kv_indptr=None,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    stream=None,
):
    """gfx942/gfx950 BF16-Q / FP8-KV page-size-1 MLA decode stage 1 for low head counts.

    KV is the arch's native E4M3: FNUZ on gfx942, OCP on gfx950.
    Covers the shapes the planner routes here (``mla_flydsl_low_head`` in
    csrc/include/mla_decode_shape.h): ``q.size(1)`` in 1..16 query heads (one KV
    head), ``max_seqlen_q`` in 1..8, 576-wide Q/KV rows and 512-wide outputs.
    Every (query, head) row of a work item is packed into 16-row MFMA tiles, so
    the KV cache is read once per work item whatever the head count.

    Uses the persistent work_info / normalized split-output / natural-log LSE
    contract of ``get_mla_metadata_v1`` built with the unpadded head count.
    Metadata contents are trusted (no host reads): each row's Q interval fits
    ``max_seqlen_q`` and KV/page/partial indices are in bounds. Rows whose
    partial index is negative are written straight to ``final_output`` (and
    ``final_lse``); the others go to ``split_data``/``split_lse`` for
    ``mla_reduce_v1``. No context parallelism or Q scaling.
    """
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from .kernels.mla_gfx942.mla_pagesize1_bf16_fp8 import (
        launch_mla_pagesize1_bf16_fp8,
    )

    arch = get_gfx_runtime()
    _require_runtime(
        (arch == "gfx942") or (arch == "gfx950"), "expected gfx942 or gfx950"
    )
    _require(q.dim() == 3, "q: expected [total_q, num_heads, 576]")
    num_heads = q.size(1)
    _require(1 <= num_heads <= 16, "q: expected 1--16 query heads")
    _require(
        isinstance(max_seqlen_q, int) and 1 <= max_seqlen_q <= 8,
        "max_seqlen_q: expected an int in 1--8",
    )
    _require(cp_world_size == 1 and cp_rank == 0, "context parallelism is unsupported")
    _require(g_kv_indptr is None, "round-robin context parallelism is unsupported")
    _require(q_scale is None, "BF16 Q does not accept q_scale")
    _require_layout("q", q, torch.bfloat16, (None, num_heads, 576))
    _require_layout("kv_buffer", kv_buffer, dtypes.fp8, (None, 1, 1, 576))
    _require_layout("split_data", split_data, torch.float32, (None, num_heads, 512))
    _require_layout(
        "split_lse", split_lse, torch.float32, (split_data.size(0), num_heads)
    )
    _require_layout(
        "final_output", final_output, torch.bfloat16, (q.size(0), num_heads, 512)
    )
    _require_layout("kv_page_indices", kv_page_indices, torch.int32, (None,))
    _require_layout("work_indptr", work_indptr, torch.int32, (None,))
    _require(work_indptr.numel() >= 2, "work_indptr: expected at least two entries")
    _require_layout("work_info", work_info, torch.int32, (None, 8))
    _require(kv_scale is not None, "kv_scale: expected a float32 scalar tensor")
    _require_layout("kv_scale", kv_scale, torch.float32)
    _require(kv_scale.numel() == 1, "kv_scale: expected one element")
    tensors = [
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        kv_scale,
    ]
    if final_lse is not None:
        _require_layout("final_lse", final_lse, torch.float32, (q.size(0), num_heads))
        tensors.append(final_lse)
    for tensor in tensors:
        _require(
            tensor.is_cuda and tensor.device == q.device,
            "all tensors must be on q.device (CUDA)",
        )
        # KV uses 64-bit global gathers; the other tensors still use
        # exact byte-limited buffer descriptors with 32-bit offsets.
        if tensor is not kv_buffer:
            _require(
                tensor.numel() * tensor.element_size() < 2**31,
                "each non-KV buffer must be smaller than 2 GiB",
            )
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    num_cus = work_indptr.numel() - 1
    causal = bool(causal)
    _run_compiled(
        launch_mla_pagesize1_bf16_fp8,
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(final_output, fx.BFloat16),
        (
            flyc.from_c_void_p(fx.Float32, 0)
            if final_lse is None
            else ptr_arg(final_lse, fx.Float32)
        ),
        ptr_arg(q, fx.BFloat16),
        ptr_arg(kv_buffer, fx.Uint8),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(work_indptr, fx.Int32),
        ptr_arg(work_info, fx.Int32),
        ptr_arg(kv_scale, fx.Float32),
        float(softmax_scale),
        q.size(0),
        kv_page_indices.numel(),
        split_data.size(0),
        work_info.size(0),
        num_cus,
        num_heads,
        max_seqlen_q,
        causal,
        final_lse is not None,
        stream,
        specialization_key=(
            q.device,
            num_cus,
            num_heads,
            max_seqlen_q,
            causal,
            final_lse is not None,
            tuple((t.dtype, t.ndim, tuple(t.stride())) for t in tensors),
        ),
    )


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
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    q_scale,
    kv_scale,
    final_lse,
    max_seqlen_q,
    causal,
    qkv_dtype,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require_runtime(
        arch in ("gfx1250", "gfx1250-strict"),
        f"expected gfx1250, got {arch or 'unknown'}",
    )
    softmax_scale = float(softmax_scale)
    max_seqlen_q = int(max_seqlen_q)
    total_q = q.size(0)
    num_q_heads = q.size(1)
    _require(
        num_q_heads in _PAGESIZE1_NUM_Q_HEADS,
        f"q: expected one of {list(_PAGESIZE1_NUM_Q_HEADS)} heads, "
        f"got {num_q_heads}",
    )
    _require(
        max_seqlen_q in (1, 2, 3, 4) or num_q_heads > _PAGESIZE1_PACKED_Q_HEADS,
        f"max_seqlen_q: expected one of [1, 2, 3, 4], got {max_seqlen_q}",
    )
    _require(
        num_q_heads > _PAGESIZE1_PACKED_Q_HEADS
        or num_q_heads * max_seqlen_q <= 2 * _PAGESIZE1_PACKED_Q_HEADS,
        f"q: {num_q_heads} heads support max_seqlen_q up to "
        f"{min(4, 2 * _PAGESIZE1_PACKED_Q_HEADS // num_q_heads)}, got {max_seqlen_q}",
    )
    _require_layout("q", q, qkv_dtype, (total_q, num_q_heads, 576))
    _require_layout("kv_buffer", kv_buffer, qkv_dtype, (None, 1, 1, 576))
    _require_layout("split_data", split_data, torch.float32, (None, num_q_heads, 512))
    _require_layout(
        "split_lse", split_lse, torch.float32, (split_data.size(0), num_q_heads)
    )
    _require_layout(
        "final_output", final_output, torch.bfloat16, (total_q, num_q_heads, 512)
    )
    _require_layout("kv_page_indices", kv_page_indices, torch.int32)
    _require_layout("work_indptr", work_indptr, torch.int32)
    _require_layout("work_info", work_info, torch.int32, (None, 8))
    if qkv_dtype == torch.float8_e4m3fn:
        for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
            _require(scale is not None, f"{name}: expected a float32 scalar tensor")
    # Only the work items the planner left un-split write here; the ones it did
    # split are the reduce's job. None means the caller wants no LSE at all.
    if final_lse is not None:
        _require_layout("final_lse", final_lse, torch.float32, (total_q, num_q_heads))

    properties = torch.cuda.get_device_properties(q.device)
    lds_size = getattr(properties, "shared_memory_per_multiprocessor", None)
    lds_size = int(lds_size) if lds_size is not None else get_lds_capacity_bytes(arch)
    return (
        work_indptr.numel() - 1,
        lds_size,
        softmax_scale,
        num_q_heads,
        max_seqlen_q,
        int(bool(causal)),
        int(final_lse is not None),
    )


def _run_pagesize1(
    launch,
    qkv_dtype,
    qkv_type,
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    q_scale,
    kv_scale,
    final_lse,
    max_seqlen_q,
    causal,
    qo_indptr,
    kv_indptr,
    g_kv_indptr,
    cp_world_size,
    cp_rank,
    stream,
):
    (
        num_cus,
        lds_size,
        softmax_scale,
        num_q_heads,
        max_seqlen_q,
        causal,
        write_final_lse,
    ) = _validate_pagesize1_inputs(
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        q_scale,
        kv_scale,
        final_lse,
        max_seqlen_q,
        causal,
        qkv_dtype,
    )
    cp_world_size = int(cp_world_size)
    cp_rank = int(cp_rank)
    _require(cp_world_size >= 1, f"cp_world_size: expected >= 1, got {cp_world_size}")
    cp_round_robin = int(cp_world_size > 1 and bool(causal))
    if cp_round_robin:
        _require(
            0 <= cp_rank < cp_world_size,
            f"cp_rank: expected in [0, {cp_world_size}), got {cp_rank}",
        )
        _require(
            qo_indptr is not None and kv_indptr is not None and g_kv_indptr is not None,
            "round-robin CP needs qo_indptr, kv_indptr and g_kv_indptr",
        )
        _require_layout("qo_indptr", qo_indptr, torch.int32, (None,))
        _require_layout("kv_indptr", kv_indptr, torch.int32, (qo_indptr.numel(),))
        _require_layout("g_kv_indptr", g_kv_indptr, torch.int32, (qo_indptr.numel(),))
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    buffers = (
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(final_output, fx.BFloat16),
        (
            flyc.from_c_void_p(fx.Float32, 0)
            if final_lse is None
            else ptr_arg(final_lse, fx.Float32)
        ),
        ptr_arg(q, qkv_type),
        ptr_arg(kv_buffer, qkv_type),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(work_indptr, fx.Int32),
        ptr_arg(work_info, fx.Int32),
        *(
            (
                flyc.from_c_void_p(fx.Float32, 0)
                if scale is None
                else ptr_arg(scale, fx.Float32)
            )
            for scale in (q_scale, kv_scale)
        ),
        softmax_scale,
        kv_buffer.size(0),
        kv_page_indices.numel(),
        (
            ptr_arg(qo_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        (
            ptr_arg(kv_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        (
            ptr_arg(g_kv_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        cp_world_size if cp_round_robin else 1,
        cp_rank if cp_round_robin else 0,
    )
    _run_compiled(
        launch,
        *buffers,
        num_q_heads,
        max_seqlen_q,
        causal,
        cp_round_robin,
        write_final_lse,
        num_cus,
        lds_size,
        stream,
        specialization_key=(
            num_q_heads,
            max_seqlen_q,
            causal,
            cp_round_robin,
            write_final_lse,
            num_cus,
            lds_size,
        ),
    )


def flydsl_mla_pagesize1_fp8_fp8(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    q_scale,
    kv_scale,
    final_lse=None,
    max_seqlen_q=1,
    causal=False,
    qo_indptr=None,
    kv_indptr=None,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    stream=None,
):
    from .kernels.mla_gfx1250.mla_pagesize1_fp8_fp8 import (
        launch_mla_pagesize1_fp8_fp8,
    )

    _run_pagesize1(
        launch_mla_pagesize1_fp8_fp8,
        torch.float8_e4m3fn,
        fx.Int8,
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        q_scale,
        kv_scale,
        final_lse,
        max_seqlen_q,
        causal,
        qo_indptr,
        kv_indptr,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        stream,
    )


def flydsl_mla_pagesize1_bf16_bf16(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    final_lse=None,
    max_seqlen_q=1,
    causal=False,
    qo_indptr=None,
    kv_indptr=None,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    stream=None,
):
    """BF16 Q/KV form of :func:`flydsl_mla_pagesize1_fp8_fp8`; no scales."""
    from .kernels.mla_gfx1250.mla_pagesize1_bf16_bf16 import (
        launch_mla_pagesize1_bf16_bf16,
    )

    _run_pagesize1(
        launch_mla_pagesize1_bf16_bf16,
        torch.bfloat16,
        fx.BFloat16,
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        None,
        None,
        final_lse,
        max_seqlen_q,
        causal,
        qo_indptr,
        kv_indptr,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        stream,
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
    _run_compiled(
        launch_mla_pagesize64_fp8_fp8,
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
        stream,
        specialization_key=int(num_splits == 1),
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
    _run_compiled(
        launch,
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(seqused_k, fx.Int32),
        ptr_arg(out, out_type),
        total_tokens,
        seqused_k.numel(),
        num_splits,
        num_tokens_per_seq,
        stream,
    )
    return out
