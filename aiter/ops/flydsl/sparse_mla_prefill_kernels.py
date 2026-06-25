# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrappers for the FlyDSL sparse MLA prefill kernel (gfx942).

- ``flydsl_sparse_mla_prefill``        : Phase A flat fp8 cache, and (with
  ``packed=True``) the Phase B1 paged ``fp8_ds_mla`` single-region path
  (UE8M0 + sink + block_table).
- ``flydsl_sparse_mla_prefill_2region``: Phase B2 two-region native path
  (compressed OCP pool + SWA fnuz cache, shared online softmax).

Callers must supply tensors in the layout/dtype the kernel expects.  This module
does **not** cast, copy, or allocate per-forward scratch (except one-time launch
stubs for compile-time-disabled optional kernel inputs).
"""

from functools import lru_cache

import torch
from flydsl.expr.typing import Stream

from .kernels.sparse_mla_prefill import compile_sparse_mla_prefill
from .kernels.tensor_shim import _run_compiled

__all__ = ["flydsl_sparse_mla_prefill", "flydsl_sparse_mla_prefill_2region"]

NUM_HEADS = 128
HEAD_DIM = 512
DEFAULT_SOFTMAX_SCALE = HEAD_DIM**-0.5

# Launch stubs for kernel parameters that stay in the ABI but are unused when
# ``single_request=True`` or ``has_sink=False`` at compile time.  Created once.
_STUB: dict[tuple, torch.Tensor] = {}


def _stub_i32(device: torch.device) -> torch.Tensor:
    key = ("i32", str(device))
    if key not in _STUB:
        _STUB[key] = torch.zeros(1, dtype=torch.int32, device=device)
    return _STUB[key]


def _stub_sink(device: torch.device) -> torch.Tensor:
    key = ("sink", str(device))
    if key not in _STUB:
        _STUB[key] = torch.zeros(NUM_HEADS, dtype=torch.float32, device=device)
    return _STUB[key]


@lru_cache(maxsize=64)
def _get_kernel(
    head_dim: int,
    v_dim: int,
    num_regions: int,
    has_sink: bool,
    r0_dtype: str,
    r0_fnuz: bool,
    r1_dtype: str,
    r1_fnuz: bool,
    qk_split: bool,
    block_n: int,
    block_h: int,
    split_kv: bool,
    packed: bool,
    scale_mode: str,
    softmax_scale: float,
    single_request: bool,
):
    return compile_sparse_mla_prefill(
        head_dim=head_dim,
        v_dim=v_dim,
        num_regions=num_regions,
        has_sink=has_sink,
        region0_dtype=r0_dtype,
        region0_is_fnuz=r0_fnuz,
        region1_dtype=r1_dtype,
        region1_is_fnuz=r1_fnuz,
        qk_split=qk_split,
        block_n=block_n,
        block_h=block_h,
        split_kv=split_kv,
        packed=packed,
        scale_mode=scale_mode,
        softmax_scale=softmax_scale,
        single_request=single_request,
    )


def _check_gfx942(device) -> None:
    try:
        arch = torch.cuda.get_device_properties(device.index).gcnArchName
    except Exception:
        arch = ""
    if not arch.lower().split(":")[0].startswith("gfx942"):
        raise ValueError(f"flydsl_sparse_mla_prefill is gfx942-only, got {arch!r}")


def _fx_stream(device, stream):
    launch_stream = torch.cuda.current_stream(device) if stream is None else stream
    return Stream(launch_stream)


def _require_cuda(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_cuda:
            raise ValueError("flydsl_sparse_mla_prefill requires CUDA/HIP tensors")


def _require_int32_contiguous(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype != torch.int32:
        raise TypeError(f"{name} must be int32, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t


def _require_int32_1d(t: torch.Tensor, name: str, *, numel: int | None = None) -> torch.Tensor:
    t = _require_int32_contiguous(t, name)
    if t.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(t.shape)}")
    if numel is not None and t.numel() != numel:
        raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
    return t


def _require_f32_1d(t: torch.Tensor, name: str, *, numel: int) -> torch.Tensor:
    if t.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {t.dtype}")
    if t.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.numel() != numel:
        raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
    return t


def _require_bf16_contiguous(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t


def _flat_bf16(t: torch.Tensor, name: str) -> torch.Tensor:
    return _require_bf16_contiguous(t, name).view(-1)


def _validate_qout(q: torch.Tensor, out: torch.Tensor) -> tuple[int, int, int, int]:
    if q.dim() != 3 or out.dim() != 3:
        raise ValueError(f"q/out must be 3D [s_q, H, D], got q={tuple(q.shape)} out={tuple(out.shape)}")
    num_queries, num_heads, head_dim = q.shape
    out_sq, out_h, v_dim = out.shape
    if (out_sq, out_h) != (num_queries, num_heads):
        raise ValueError("q and out must share (num_queries, num_heads)")
    if num_heads != NUM_HEADS or head_dim != HEAD_DIM or v_dim != HEAD_DIM:
        raise NotImplementedError(
            f"requires H={NUM_HEADS}, head_dim={HEAD_DIM}, v_dim={HEAD_DIM}; "
            f"got H={num_heads}, head_dim={head_dim}, v_dim={v_dim}"
        )
    return num_queries, num_heads, head_dim, v_dim


def _packed_cache_u8(cache: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int, int]:
    if cache.dim() != 3 or cache.shape[-1] != 584:
        raise ValueError(f"packed cache must be [num_blocks, block_size, 584], got {tuple(cache.shape)}")
    num_blocks, blk, _ = cache.shape
    if blk != block_size:
        raise ValueError(f"cache block_size {blk} != block_size arg {block_size}")
    if not cache.is_contiguous():
        raise ValueError("packed cache must be contiguous")
    num_rows = num_blocks * block_size
    cache_u8 = cache.view(torch.uint8).reshape(-1)
    return cache_u8, num_rows, num_blocks


def _resolve_q_req(
    q_req: torch.Tensor | None,
    *,
    num_queries: int,
    single_request: bool,
    device: torch.device,
) -> torch.Tensor:
    if single_request:
        return _stub_i32(device)
    if q_req is None:
        raise ValueError("q_req [num_queries] int32 is required when single_request=False")
    return _require_int32_1d(q_req, "q_req", numel=num_queries)


def _resolve_sink(
    attn_sink: torch.Tensor | None,
    *,
    has_sink: bool,
    device: torch.device,
) -> torch.Tensor:
    if not has_sink:
        return _stub_sink(device)
    if attn_sink is None:
        raise ValueError("attn_sink [128] float32 is required for this kernel specialization")
    return _require_f32_1d(attn_sink, "attn_sink", numel=NUM_HEADS)


def flydsl_sparse_mla_prefill(
    q: torch.Tensor,  # [num_queries, 128, 512] bf16 contiguous
    kv: torch.Tensor,  # flat fp8 [num_kv_rows, 1, 512] (Phase A) OR packed cache (packed=True)
    indices: torch.Tensor,  # flat int32 CSR values, contiguous
    indptr: torch.Tensor,  # [num_queries + 1] int32, contiguous
    out: torch.Tensor,  # [num_queries, 128, 512] bf16 contiguous (in place)
    *,
    attn_sink: torch.Tensor | None = None,  # [128] f32 contiguous (packed + has_sink)
    block_table: torch.Tensor | None = None,  # [num_reqs, max_blocks] int32 contiguous (packed)
    block_size: int = 1,
    packed: bool = False,
    scale_mode: str = "none",  # "none" | "ue8m0"
    q_req: torch.Tensor | None = None,  # [num_queries] int32 (only if single_request=False)
    num_kv_rows: int | None = None,
    single_request: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Run sparse MLA prefill in-place into ``out``.

    Softmax scale is fixed at compile time (``1/sqrt(512)``).  All tensor args
    must already match kernel dtype/layout; this entry point validates only.
    """
    _require_cuda(q, kv, indices, indptr, out)
    _check_gfx942(q.device)
    num_queries, num_heads, head_dim, v_dim = _validate_qout(q, out)
    _require_int32_1d(indptr, "indptr", numel=num_queries + 1)
    indices_i32 = _require_int32_1d(indices, "indices")
    q_flat = _flat_bf16(q, "q")
    out_flat = _flat_bf16(out, "out")

    if not packed:
        if attn_sink is not None:
            raise NotImplementedError("attn_sink is the packed (B1) path; pass packed=True")
        if kv.dim() != 3 or kv.shape[1] != 1 or kv.shape[2] != head_dim:
            raise ValueError(f"flat kv must be [num_kv_rows, 1, {head_dim}], got {tuple(kv.shape)}")
        if not kv.is_contiguous():
            raise ValueError("kv must be contiguous")
        kv_u8 = kv.view(torch.uint8).reshape(-1)
        n_kv_rows = kv.shape[0]
        exe = _get_kernel(
            head_dim=head_dim, v_dim=v_dim, num_regions=1, has_sink=False,
            r0_dtype="fp8", r0_fnuz=True, r1_dtype="fp8", r1_fnuz=True,
            qk_split=False, block_n=32, block_h=16, split_kv=False, packed=False, scale_mode="none",
            softmax_scale=DEFAULT_SOFTMAX_SCALE, single_request=True,
        )
        with torch.cuda.device(q.device.index):
            _run_compiled(
                exe, q_flat, kv_u8, indices_i32, indptr, out_flat,
                int(num_queries), int(n_kv_rows), _fx_stream(q.device, stream),
            )
        return

    if block_table is None:
        raise ValueError("packed=True requires block_table [num_reqs, max_blocks] int32")
    cache_u8, default_rows, _ = _packed_cache_u8(kv, block_size)
    n_kv_rows = int(num_kv_rows) if num_kv_rows is not None else default_rows
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2D [num_reqs, max_blocks], got {tuple(block_table.shape)}")
    bt_flat = _require_int32_contiguous(block_table, "block_table").view(-1)
    max_blocks = block_table.shape[1]
    has_sink = attn_sink is not None
    q_req_t = _resolve_q_req(q_req, num_queries=num_queries, single_request=single_request, device=q.device)
    sink_t = _resolve_sink(attn_sink, has_sink=has_sink, device=q.device)

    exe = _get_kernel(
        head_dim=head_dim, v_dim=v_dim, num_regions=1, has_sink=has_sink,
        r0_dtype="fp8", r0_fnuz=True, r1_dtype="fp8", r1_fnuz=True,
        qk_split=True, block_n=32, block_h=16, split_kv=False, packed=True, scale_mode=scale_mode,
        softmax_scale=DEFAULT_SOFTMAX_SCALE, single_request=single_request,
    )
    with torch.cuda.device(q.device.index):
        _run_compiled(
            exe,
            q_flat,
            cache_u8,
            indices_i32,
            indptr,
            bt_flat,
            cache_u8,
            indices_i32,
            indptr,
            bt_flat,
            q_req_t,
            sink_t,
            out_flat,
            int(num_queries),
            int(n_kv_rows),
            int(n_kv_rows),
            int(block_size),
            int(block_size),
            int(max_blocks),
            int(max_blocks),
            _fx_stream(q.device, stream),
        )


def flydsl_sparse_mla_prefill_2region(
    q: torch.Tensor,
    out: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    main_block_table: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    extra_block_table: torch.Tensor,
    *,
    block_size: int,
    attn_sink: torch.Tensor | None = None,
    extra_block_size: int | None = None,
    main_num_rows: int | None = None,
    extra_num_rows: int | None = None,
    q_req: torch.Tensor | None = None,
    main_is_fnuz: bool = True,
    extra_is_fnuz: bool = False,
    single_request: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Phase B2: two-region native sparse MLA prefill (compressed + SWA)."""
    _require_cuda(q, out, main_cache, extra_cache, main_indices, main_indptr, extra_indices, extra_indptr)
    _check_gfx942(q.device)
    num_queries, num_heads, head_dim, v_dim = _validate_qout(q, out)
    _require_int32_1d(main_indptr, "main_indptr", numel=num_queries + 1)
    _require_int32_1d(extra_indptr, "extra_indptr", numel=num_queries + 1)

    e_block_size = block_size if extra_block_size is None else extra_block_size
    m_cache_u8, m_default_rows, _ = _packed_cache_u8(main_cache, block_size)
    e_cache_u8, e_default_rows, _ = _packed_cache_u8(extra_cache, e_block_size)
    m_rows = int(main_num_rows) if main_num_rows is not None else m_default_rows
    e_rows = int(extra_num_rows) if extra_num_rows is not None else e_default_rows

    m_idx = _require_int32_1d(main_indices, "main_indices")
    m_iptr = _require_int32_1d(main_indptr, "main_indptr", numel=num_queries + 1)
    e_idx = _require_int32_1d(extra_indices, "extra_indices")
    e_iptr = _require_int32_1d(extra_indptr, "extra_indptr", numel=num_queries + 1)
    if main_block_table.dim() != 2 or extra_block_table.dim() != 2:
        raise ValueError("main_block_table and extra_block_table must be 2D [num_reqs, max_blocks]")
    m_bt = _require_int32_contiguous(main_block_table, "main_block_table").view(-1)
    e_bt = _require_int32_contiguous(extra_block_table, "extra_block_table").view(-1)
    m_max_blocks = main_block_table.shape[1]
    e_max_blocks = extra_block_table.shape[1]

    has_sink = attn_sink is not None
    q_req_t = _resolve_q_req(q_req, num_queries=num_queries, single_request=single_request, device=q.device)
    sink_t = _resolve_sink(attn_sink, has_sink=has_sink, device=q.device)
    q_flat = _flat_bf16(q, "q")
    out_flat = _flat_bf16(out, "out")

    exe = _get_kernel(
        head_dim=head_dim, v_dim=v_dim, num_regions=2, has_sink=has_sink,
        r0_dtype="fp8", r0_fnuz=main_is_fnuz, r1_dtype="fp8", r1_fnuz=extra_is_fnuz,
        qk_split=True, block_n=32, block_h=16, split_kv=False, packed=True, scale_mode="none",
        softmax_scale=DEFAULT_SOFTMAX_SCALE, single_request=single_request,
    )
    with torch.cuda.device(q.device.index):
        _run_compiled(
            exe,
            q_flat,
            m_cache_u8,
            m_idx,
            m_iptr,
            m_bt,
            e_cache_u8,
            e_idx,
            e_iptr,
            e_bt,
            q_req_t,
            sink_t,
            out_flat,
            int(num_queries),
            int(m_rows),
            int(e_rows),
            int(block_size),
            int(e_block_size),
            int(m_max_blocks),
            int(e_max_blocks),
            _fx_stream(q.device, stream),
        )


from .kernels.tensor_shim import _run_compiled

__all__ = ["flydsl_sparse_mla_prefill", "flydsl_sparse_mla_prefill_2region"]
