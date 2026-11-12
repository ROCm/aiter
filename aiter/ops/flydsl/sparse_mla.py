# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 bf16 absorbed sparse MLA for serving (H=16, D=512).

``sparse_mla_fwd`` (aiter.ops.triton.attention.sparse_mla) routes its
matching rope-free calls here when AITER_SPARSE_MLA_FLYDSL=1. Call
``flydsl_sparse_mla_fwd`` directly for dense indices and the ordered routes.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.kernels_common import LOG2E
from .kernels.sparse_mla import (
    HEAD_DIM,
    NUM_HEADS,
    TOKENS_PER_WORKGROUP,
    compile_sparse_mla_combine,
    compile_sparse_mla_fwd,
    compile_sparse_mla_order_indices,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .kernels.topk.topk_launch import run_topk_launcher

__all__ = [
    "SparseMLAOrderPlan",
    "flydsl_sparse_mla_fwd",
    "flydsl_sparse_mla_fwd_supported",
    "order_sparse_mla_indices",
    "prepare_sparse_mla_order",
]

# slot * D and token * H * D are int32 element offsets.
_MAX_TOKENS = (1 << 31) // (NUM_HEADS * HEAD_DIM)
_MAX_KV_ROWS = (1 << 31) // HEAD_DIM


def _require_gfx942():
    gfx = get_gfx_runtime()
    if gfx != "gfx942":
        raise RuntimeError(f"FlyDSL sparse MLA is gfx942-only (64 KiB LDS), got {gfx}")


def _require_contract(q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor):
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("FlyDSL sparse MLA is bf16-only")
    if q.ndim != 3 or kv.ndim != 2:
        raise ValueError(
            f"expected q[T,H,D] kv[N,D], got {tuple(q.shape)} {tuple(kv.shape)}"
        )
    n_tok, num_heads, head_dim = q.shape
    if num_heads != NUM_HEADS or head_dim != HEAD_DIM:
        raise ValueError(
            f"FlyDSL sparse MLA is specialized for H={NUM_HEADS} D={HEAD_DIM}, "
            f"got H={num_heads} D={head_dim}"
        )
    if kv.shape[1] != HEAD_DIM:
        raise ValueError(f"kv last dim must be {HEAD_DIM}, got {kv.shape[1]}")
    if out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError("out must match q shape and dtype")
    if not q.is_contiguous() or not kv.is_contiguous() or not out.is_contiguous():
        raise ValueError("q, kv, and out must be contiguous")
    if n_tok >= _MAX_TOKENS or kv.shape[0] >= _MAX_KV_ROWS:
        raise ValueError(
            f"FlyDSL sparse MLA int32 offsets support T < {_MAX_TOKENS} "
            f"and KV rows < {_MAX_KV_ROWS}"
        )
    return n_tok, kv.shape[0]


def _checked_scale(softmax_scale: float | None) -> float:
    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    try:
        # Match the FP32 ABI and the kernel's scale * FP32(LOG2E), in that order.
        scale = struct.unpack("f", struct.pack("f", scale))[0]
        log2e = struct.unpack("f", struct.pack("f", LOG2E))[0]
        effective = struct.unpack("f", struct.pack("f", scale * log2e))[0]
    except OverflowError as exc:
        raise ValueError(
            "softmax_scale and scale * LOG2E must be finite and positive in FP32"
        ) from exc
    if not (
        math.isfinite(scale)
        and scale > 0
        and math.isfinite(effective)
        and effective > 0
    ):
        raise ValueError(
            "softmax_scale and scale * LOG2E must be finite and positive in FP32"
        )
    return scale


def _split_count(tokens: int, max_length: int) -> int:
    # Algorithm lever: fill the 304-CU grid, but keep at least one 16-key tile.
    tiles = max(1, (max_length + 15) // 16)
    wanted = max(1, (304 + tokens - 1) // tokens)
    return min(128, tiles, wanted)


def flydsl_sparse_mla_fwd_supported(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dot_precision: str = "bf16",
    out: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    kv_splits: int | None = None,
) -> bool:
    """Whether a ``sparse_mla_fwd`` call fits ``flydsl_sparse_mla_fwd``.

    gfx942, rope-free bf16, q [C,16,512], a contiguous bf16 pool that views
    as [slots,512], an exact-shape contiguous out and a 16-entry sink. The
    caller still rules out LSE, skip_reduce, extra segments and epilogues.
    """
    if get_gfx_runtime() != "gfx942":
        return False
    if (kv_lora_rank, qk_rope_head_dim, dot_precision) != (HEAD_DIM, 0, "bf16"):
        return False
    if q.dtype != torch.bfloat16 or kv_buffer.dtype != torch.bfloat16:
        return False
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        return False
    if not q.is_contiguous() or not kv_buffer.is_contiguous():
        return False
    if kv_buffer.shape[-1] != HEAD_DIM:
        return False
    tokens, slots = q.shape[0], kv_buffer.numel() // HEAD_DIM
    if not (0 < tokens < _MAX_TOKENS and 0 < slots < _MAX_KV_ROWS):
        return False
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
    ):
        return False
    if attn_sink is not None and attn_sink.numel() != NUM_HEADS:
        return False
    if kv_indices.numel() * 4 > 0xFFFFFFFF:
        return False
    if kv_splits is not None:
        if not 1 <= kv_splits <= 128:
            return False
        if kv_splits > 1 and tokens * kv_splits * NUM_HEADS * HEAD_DIM * 4 > (
            0xFFFFFFFF
        ):
            return False
    return True


def flydsl_sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor | None = None,
    kv_indptr: torch.Tensor | None = None,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    max_seqlen: int = 2048,
    num_splits: int | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    order_plan=None,
    row_requests: torch.Tensor | None = None,
    ordered_indices: bool = False,
) -> torch.Tensor:
    """Return sparse attention into ``out`` (or a new bf16 tensor).

    Q is contiguous [T,16,512], KV is contiguous [pool,512]. Indices are
    contiguous int32 dense [T,K], or flat CSR with int32 indptr [T+1].
    Dense topk_length [T] limits each row before -1 masking. CSR lengths can
    be zero; -1 is ignored in either format. Empty rows return zero, with
    or without a sink. Valid nonnegative slots must be inside the KV pool.

    Metadata values must be valid: monotone CSR within the index allocation,
    and 0 <= topk_length <= K. Values are not read on the host. max_seqlen
    only plans splits for CSR; it never truncates a row. Batches below 2048
    tokens use split-K plus a combine kernel; larger batches run four tokens
    per workgroup with one full-D wave per token. num_splits overrides this.
    A reusable workspace is (fp32 [T,S,16,512], fp32 [T,S,16,2]). The first
    tensor is opaque scratch, stored in coalesced [T,S,128,16,4] order. Each call
    writes all partials, including empty splits. This is safe for graph replay.

    Index order picks the V^T route of unsplit launches. Results are the
    same either way; only speed changes. The choice is a call-site contract,
    not a runtime probe: detecting order needs a device pass and a host sync,
    which would break graph capture.

    * Default: V^T goes through LDS. Use it for top-k output as produced,
      because top-k writes each row in score order, not KV-pool order.
    * ordered_indices=True: the caller guarantees each row's slots mostly
      ascend through the KV pool, for example sorted logical positions mapped
      through a block table. V^T is built in registers, which is about 10-15%
      faster on such rows and about 15% slower on rows in score order.
    * order_plan (from ``prepare_sparse_mla_order``, once per block-table
      update): reorders dense [T,2048] rows without topk_length by physical
      page on the GPU, then uses the register route. The caller index tensor
      is unchanged. row_requests maps rows to block-table requests when the
      plan has more than one. At T=16384 the reorder costs about 0.14 ms and
      saves about 2.3 ms over the default route on rows in score order.
    """
    if out is None:
        out = torch.empty_like(q)
    tokens, pool = _require_contract(q, kv, out)
    if q.device.type != "cuda" or tokens <= 0 or pool <= 0:
        raise ValueError("positive T and KV pool on a GPU are required")
    for name, value in (
        ("kv", kv),
        ("out", out),
        ("kv_indices", kv_indices),
        ("kv_indptr", kv_indptr),
        ("topk_length", topk_length),
        ("attn_sink", attn_sink),
    ):
        if value is not None and value.device != q.device:
            raise ValueError(f"{name} must be on the same GPU as q")
    for name, value in (
        ("kv_indices", kv_indices),
        ("kv_indptr", kv_indptr),
        ("topk_length", topk_length),
    ):
        if value is not None and (
            value.dtype != torch.int32 or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32")
    if kv_indices.numel() * 4 > 0xFFFFFFFF:
        raise ValueError("index byte span must fit uint32")
    if kv_indptr is not None:
        if (
            kv_indices.ndim != 1
            or kv_indptr.shape != (tokens + 1,)
            or topk_length is not None
        ):
            raise ValueError("CSR needs flat indices, indptr [T+1], and no topk_length")
        index_format, width = "csr", max_seqlen
    else:
        if kv_indices.ndim != 2 or kv_indices.shape[0] != tokens:
            raise ValueError("dense indices must have shape [T,K]")
        if topk_length is not None and topk_length.shape != (tokens,):
            raise ValueError("topk_length must have shape [T]")
        index_format = "dense" if topk_length is None else "dense_lengths"
        width = kv_indices.shape[1]
    if not isinstance(max_seqlen, int) or max_seqlen < 0:
        raise ValueError("max_seqlen must be a nonnegative integer")
    if attn_sink is not None and (
        attn_sink.shape != (16,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous fp32 [16]")
    scale = _checked_scale(softmax_scale)
    splits = (
        (_split_count(tokens, width) if tokens < 2048 else 1)
        if num_splits is None
        else num_splits
    )
    if not isinstance(splits, int) or not 1 <= splits <= 128:
        raise ValueError("num_splits must be an integer in [1,128]")
    if splits > 1 and tokens * splits * 16 * 512 * 4 > 0xFFFFFFFF:
        raise ValueError("partial workspace byte span must fit uint32")
    if order_plan is not None:
        if index_format != "dense" or width != 2048:
            raise ValueError("ordering requires dense [T,2048] without topk_length")
        kv_indices = order_sparse_mla_indices(
            kv_indices, order_plan, row_requests=row_requests
        )
        ordered_indices = True
    register_vt = splits == 1 and bool(ordered_indices)
    dummy = kv_indices
    sink = attn_sink if attn_sink is not None else q
    if splits > 1:
        shapes = ((tokens, splits, 16, 512), (tokens, splits, 16, 2))
        if workspace is None:
            workspace = tuple(
                torch.empty(shape, device=q.device, dtype=torch.float32)
                for shape in shapes
            )
        if len(workspace) != 2:
            raise ValueError("workspace must contain partials and stats")
        for value, shape in zip(workspace, shapes):
            if (
                value.shape != shape
                or value.dtype != torch.float32
                or value.device != q.device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"workspace must contain contiguous fp32 {shape} on q.device"
                )
        partial, stats = workspace
        target, target_type = partial, fx.Float32
    else:
        target, target_type, stats = out, fx.BFloat16, q
    with torch.cuda.device(q.device):
        _require_gfx942()
        stream = fx.Stream(torch.cuda.current_stream(q.device))
        _run_compiled(
            compile_sparse_mla_fwd(
                attn_sink is not None,
                index_format=index_format,
                split_k=splits,
                register_vt=register_vt,
            ),
            ptr_arg(q, fx.BFloat16),
            ptr_arg(kv, fx.BFloat16),
            ptr_arg(kv_indices, fx.Int32),
            ptr_arg(target, target_type),
            tokens,
            width,
            scale,
            ptr_arg(sink, fx.Float32),
            ptr_arg(kv_indptr if kv_indptr is not None else dummy, fx.Int32),
            ptr_arg(topk_length if topk_length is not None else dummy, fx.Int32),
            ptr_arg(stats, fx.Float32),
            tokens if splits > 1 else -(-tokens // TOKENS_PER_WORKGROUP),
            stream,
        )
        if splits > 1:
            _run_compiled(
                compile_sparse_mla_combine(splits, attn_sink is not None),
                ptr_arg(partial, fx.Float32),
                ptr_arg(stats, fx.Float32),
                ptr_arg(sink, fx.Float32),
                ptr_arg(out, fx.BFloat16),
                tokens,
                stream,
            )
    return out


@dataclass(frozen=True)
class SparseMLAOrderPlan:
    page_size: int
    sorted_pages: torch.Tensor
    physical_rank: torch.Tensor
    logical_rank: torch.Tensor


def prepare_sparse_mla_order(
    block_table: torch.Tensor, page_size: int, pool_pages: int
) -> SparseMLAOrderPlan:
    """Prepare once per block-table update, not once per query or layer.

    block_table is [requests, active_pages], without unused/padded entries.
    Only the small page table is sorted, never the [queries, topk] tensor.
    """
    if (
        block_table.ndim != 2
        or not block_table.is_cuda
        or block_table.dtype != torch.int32
    ):
        raise ValueError("block_table must be CUDA int32 [requests, active_pages]")
    if (
        page_size <= 0
        or page_size % 16
        or pool_pages <= 0
        or min(block_table.shape) == 0
    ):
        raise ValueError(
            "page_size must be a positive multiple of 16; pool and table dimensions must be positive"
        )
    if block_table.shape[1] * page_size > 1 << 18:
        raise ValueError(
            "active page domain must not exceed 262144 rows (16 KiB packed counters)"
        )
    if pool_pages * page_size > (1 << 31):
        raise ValueError("physical slots must fit signed int32")
    sorted_pages, order = block_table.sort(dim=1)
    if bool(((sorted_pages < 0) | (sorted_pages >= pool_pages)).any()) or bool(
        (sorted_pages[:, 1:] == sorted_pages[:, :-1]).any()
    ):
        raise ValueError("block_table pages must be distinct and inside the pool")
    sorted_pages = sorted_pages.contiguous()
    ranks = torch.arange(
        block_table.shape[1], dtype=torch.int32, device=block_table.device
    ).expand_as(block_table)
    physical_rank = torch.full(
        (block_table.shape[0], pool_pages),
        -1,
        dtype=torch.int32,
        device=block_table.device,
    )
    physical_rank.scatter_(1, sorted_pages.long(), ranks)
    logical_rank = torch.empty_like(block_table)
    logical_rank.scatter_(1, order, ranks)
    return SparseMLAOrderPlan(page_size, sorted_pages, physical_rank, logical_rank)


def order_sparse_mla_indices(
    indices: torch.Tensor,
    plan: SparseMLAOrderPlan,
    out: torch.Tensor | None = None,
    row_requests: torch.Tensor | None = None,
    *,
    logical: bool = False,
) -> torch.Tensor:
    """Ascending physical 16-row tiles for each [rows, 2048] row.

    Within-tile order is unspecified; the complete input multiset is preserved.

    logical=True fuses logical-position -> physical-slot mapping with ordering;
    otherwise sort already-mapped physical slots. Pass a reusable plan made
    from the active block table. Counter overflow, -1 padding, and physical
    slots outside the plan use an exact signed-int32 bitonic fallback.
    Invalid logical positions map to -1. out=indices is supported. This orders
    indices only, not paired top-k score values. row_requests must contain
    valid request IDs. Negative padding sorts first on fallback: do not feed
    padded results to a CSR compactor that copies only the first length items.
    The intended fast path is full 2048-entry prefill lists.
    """
    if indices.ndim != 2 or indices.shape[1] != 2048:
        raise ValueError("indices must have shape [rows, 2048]")
    if (
        indices.dtype != torch.int32
        or not indices.is_cuda
        or not indices.is_contiguous()
    ):
        raise ValueError("indices must be contiguous CUDA int32")
    if out is None:
        out = torch.empty_like(indices)
    if (
        out.shape != indices.shape
        or out.dtype != indices.dtype
        or out.device != indices.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must match indices shape, dtype, device and layout")
    # Both views are contiguous and equally sized; pointer intervals detect
    # actual overlap without rejecting disjoint views of the same storage.
    distance = abs(out.data_ptr() - indices.data_ptr())
    if 0 < distance < indices.numel() * indices.element_size():
        raise ValueError("out must not partially overlap indices")
    if plan.sorted_pages.device != indices.device:
        raise ValueError("plan and indices must be on the same device")
    if row_requests is None:
        if plan.sorted_pages.shape[0] != 1:
            raise ValueError("row_requests is required for multiple requests")
        # The single-request kernel uses constant request 0; its unused
        # request-ID argument aliases metadata, avoiding a per-call allocation.
        row_requests = plan.sorted_pages
    elif (
        row_requests.shape != (indices.shape[0],)
        or row_requests.dtype != torch.int32
        or row_requests.device != indices.device
        or not row_requests.is_contiguous()
    ):
        raise ValueError("row_requests must be contiguous CUDA int32 [rows]")
    if indices.shape[0]:
        rank = plan.logical_rank if logical else plan.physical_rank
        with torch.cuda.device(indices.device):
            if get_gfx_runtime() != "gfx942":
                raise RuntimeError("order_sparse_mla_indices currently requires gfx942")
            run_topk_launcher(
                compile_sparse_mla_order_indices(
                    plan.page_size,
                    plan.sorted_pages.shape[1],
                    rank.shape[1],
                    logical,
                    plan.sorted_pages.shape[0] == 1,
                ),
                indices,
                out,
                rank,
                plan.sorted_pages,
                row_requests,
                indices.shape[0],
                torch.cuda.current_stream(),
            )
    return out
