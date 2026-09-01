# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from .kernels.tensor_shim import ptr_arg
from .utils import addressable_lds_bytes_for_gfx

__all__ = [
    "flydsl_mla_decode_fwd",
    "flydsl_mla_decode_reduce",
    "flydsl_mla_decode_workspace",
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
]


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
    lds_size = (
        int(lds_size) if lds_size is not None else addressable_lds_bytes_for_gfx(arch)
    )
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
    from .kernels.mla_gfx1250.mla_decode_reduce import compile_mla_decode_reduce

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


def _pick_num_splits(
    batch, num_tokens_per_seq, max_seqlen_kv, page_size, requested, num_heads=128
):
    if requested is not None:
        return requested

    from aiter.mla import get_meta_param

    meta_seqlen_q = min(num_tokens_per_seq, max(1, 512 // num_heads))
    num_splits, _ = get_meta_param(
        None,
        batch,
        batch * max_seqlen_kv,
        num_heads,
        meta_seqlen_q,
        torch.float8_e4m3fn,
    )
    # The kernel hands page p to split p % num_splits, so splits beyond the page
    # count would idle; the heuristic works in tokens and does not know that.
    pages_per_seq = max(1, (max_seqlen_kv + page_size - 1) // page_size)
    return max(1, min(num_splits, pages_per_seq))


def flydsl_mla_decode_workspace(
    total_tokens, num_splits, device, num_heads=128, v_head_dim=512
):
    """Allocate the per-split partial buffers for :func:`flydsl_mla_decode_fwd`.

    ``total_tokens`` is ``q.size(0)``, i.e. ``batch * tokens_per_seq`` under MTP.
    Hoisting these out of the call keeps them alive for as long as the caller
    needs and avoids re-allocating tens of megabytes per decode step.
    """
    split_data = torch.empty(
        (total_tokens, num_splits, num_heads, v_head_dim),
        dtype=torch.float32,
        device=device,
    )
    split_lse = torch.empty(
        (total_tokens, num_splits, num_heads, 1), dtype=torch.float32, device=device
    )
    return split_data, split_lse


def flydsl_mla_decode_fwd(
    q,  # [total_tokens, 128, 576] fp8, contiguous
    kv_buffer,  # [num_blocks, 64, 1, 576] fp8, contiguous
    out,  # [total_tokens, 128, 512]
    cu_seqlens_q,  # [batch + 1] int32
    seqused_k,  # [batch] int32
    max_seqlen_kv,
    block_tables,  # [batch, max_pages] int32
    softmax_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    causal,
    q_descale,  # fp32 [1]
    kv_descale,  # fp32 [1]
    num_splits=None,
    skip_reduce=False,
    workspace=None,
    stream=None,
):
    """FlyDSL gfx1250 MLA decode on the signature of ``mla_decode_fwd``.

    Positional arguments mirror ``aiter.ops.triton.attention.mla.mla_decode_fwd``
    up to ``kv_descale`` so the two backends are interchangeable. ``num_splits``
    is an extension: the Triton path derives its segment count internally and
    offers no override, whereas here the split count is a real launch parameter.

    Writes into ``out`` and returns it. With ``skip_reduce`` the merge is left to
    the caller and ``(split_data, split_lse)`` is returned instead.

    ``workspace`` is a ``(split_data, split_lse)`` pair for the per-split
    partials, obtainable from :func:`flydsl_mla_decode_workspace`. Passing one is
    strongly preferred: the stage-1 kernel is launched on a raw stream handle
    that the caching allocator does not track, so partials allocated per call can
    be handed back to the allocator while the kernel is still writing them.
    """
    from .kernels.mla_gfx1250.mla_decode_gfx1250 import (
        launch_mla_decode_pagesize64_fp8_fp8_gluon,
    )

    batch, total_tokens, num_tokens_per_seq = _validate_flydsl_decode_inputs(
        q, kv_buffer, out, seqused_k, kv_lora_rank, qk_rope_head_dim, causal
    )
    resolved_splits = _pick_num_splits(
        batch,
        num_tokens_per_seq,
        max_seqlen_kv,
        kv_buffer.size(1),
        num_splits,
        num_heads=q.size(1),
    )

    device = q.device
    if workspace is None:
        workspace = flydsl_mla_decode_workspace(
            total_tokens,
            resolved_splits,
            device,
            num_heads=q.size(1),
            v_head_dim=out.size(-1),
        )
    split_data, split_lse = workspace
    if resolved_splits == 1:
        # The kernel writes the final bf16 result straight into `out`.
        split_data = out

    if stream is None:
        stream = torch.cuda.current_stream(device)
    output_type = fx.BFloat16 if resolved_splits == 1 else fx.Float32
    launch_mla_decode_pagesize64_fp8_fp8_gluon(
        ptr_arg(split_data, output_type),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(q, fx.Int8),
        ptr_arg(kv_buffer, fx.Int8),
        ptr_arg(block_tables, fx.Int32),
        ptr_arg(seqused_k, fx.Int32),
        ptr_arg(cu_seqlens_q, fx.Int32),
        ptr_arg(q_descale, fx.Float32),
        ptr_arg(kv_descale, fx.Float32),
        float(softmax_scale),
        batch,
        resolved_splits,
        block_tables.stride(0),
        num_tokens_per_seq,
        int(resolved_splits == 1),
        stream=stream,
    )

    if resolved_splits == 1:
        return out
    if skip_reduce:
        return split_data, split_lse

    return flydsl_mla_decode_reduce(
        split_data,
        split_lse,
        seqused_k,
        out,
        resolved_splits,
        num_tokens_per_seq,
        stream=stream,
    )


def _validate_flydsl_decode_inputs(
    q, kv_buffer, out, seqused_k, kv_lora_rank, qk_rope_head_dim, causal
):

    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    _require(
        q.size(-1) == qk_head_dim
        and kv_buffer.size(-1) == qk_head_dim
        and out.size(-1) == kv_lora_rank,
        f"head dims: q/kv last dim {qk_head_dim}, out last dim {kv_lora_rank}; "
        f"got q {list(q.shape)} kv {list(kv_buffer.shape)} out {list(out.shape)}",
    )
    total_tokens = q.size(0)
    batch = seqused_k.numel()
    _require(
        batch > 0 and total_tokens % batch == 0,
        f"q/seqused_k: total_tokens {total_tokens} must be a multiple of batch {batch}",
    )
    return batch, total_tokens, total_tokens // batch
