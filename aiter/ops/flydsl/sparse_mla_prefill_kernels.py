# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrappers for the FlyDSL sparse MLA prefill kernel (gfx942).

- ``flydsl_sparse_mla_prefill``        : Phase A flat fp8 cache, and (with
  ``packed=True``) the Phase B1 paged ``fp8_ds_mla`` single-region path
  (UE8M0 + sink + block_table).
- ``flydsl_sparse_mla_prefill_2region``: Phase B2 two-region native path
  (compressed OCP pool + SWA fnuz cache, shared online softmax).

See ``docs/sparse-mla-prefill/`` and ``aiter/ops/flydsl/kernels/sparse_mla_prefill.py``.
"""

from functools import lru_cache

import torch
from flydsl.expr.typing import Stream

from .kernels.sparse_mla_prefill import compile_sparse_mla_prefill
from .kernels.tensor_shim import _run_compiled

__all__ = ["flydsl_sparse_mla_prefill", "flydsl_sparse_mla_prefill_2region"]


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


def _validate_qout(q, out):
    if q.dim() != 3 or out.dim() != 3:
        raise ValueError(f"q/out must be 3D [s_q, H, D], got q={tuple(q.shape)} out={tuple(out.shape)}")
    num_queries, num_heads, head_dim = q.shape
    out_sq, out_h, v_dim = out.shape
    if (out_sq, out_h) != (num_queries, num_heads):
        raise ValueError("q and out must share (num_queries, num_heads)")
    if num_heads != 128 or head_dim != 512 or v_dim != 512:
        raise NotImplementedError(
            f"requires H=128, head_dim=512, v_dim=512; got H={num_heads}, head_dim={head_dim}, v_dim={v_dim}"
        )
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"q/out must be bf16, got q={q.dtype} out={out.dtype}")
    return num_queries, num_heads, head_dim, v_dim


def _packed_cache_meta(cache, block_size):
    """Return (cache_u8_flat, num_rows, num_blocks) for a packed fp8_ds_mla cache."""
    if cache.dim() != 3 or cache.shape[-1] != 584:
        raise ValueError(f"packed cache must be [num_blocks, block_size, 584], got {tuple(cache.shape)}")
    num_blocks, blk, _ = cache.shape
    if blk != block_size:
        raise ValueError(f"cache block_size {blk} != block_size arg {block_size}")
    num_rows = num_blocks * block_size
    cache_u8 = cache.contiguous().view(torch.uint8).reshape(-1)
    return cache_u8, num_rows, num_blocks


def flydsl_sparse_mla_prefill(
    q: torch.Tensor,  # [num_queries, 128, 512] bf16
    kv: torch.Tensor,  # flat fp8 [num_kv_rows, 1, 512] (Phase A) OR packed cache (packed=True)
    indices: torch.Tensor,  # flat int32 CSR values
    indptr: torch.Tensor,  # [num_queries + 1] int32 CSR offsets
    out: torch.Tensor,  # [num_queries, 128, 512] bf16 (in place)
    *,
    scale: float,
    attn_sink: torch.Tensor | None = None,  # [128] f32 (packed path only)
    block_table: torch.Tensor | None = None,  # [num_reqs, max_blocks] int32 (packed)
    block_size: int = 1,
    packed: bool = False,
    scale_mode: str = "none",  # "none" | "ue8m0"
    q_req: torch.Tensor | None = None,  # [num_queries] int32 -> request id
    num_kv_rows: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Run sparse MLA prefill in-place into ``out``.

    Phase A (``packed=False``): flat fp8 e4m3fnuz cache, single region, no sink.
    Phase B1 (``packed=True``): paged ``fp8_ds_mla`` cache, single region,
    optional UE8M0 (``scale_mode="ue8m0"``) and ``attn_sink``.
    """
    if not (q.is_cuda and kv.is_cuda and indices.is_cuda and indptr.is_cuda and out.is_cuda):
        raise ValueError("flydsl_sparse_mla_prefill requires CUDA/HIP tensors")
    _check_gfx942(q.device)
    num_queries, num_heads, head_dim, v_dim = _validate_qout(q, out)
    if indptr.numel() != num_queries + 1:
        raise ValueError(f"indptr must have num_queries+1={num_queries + 1} elems, got {indptr.numel()}")

    indices_i32 = indices.reshape(-1).to(torch.int32).contiguous()
    indptr_i32 = indptr.reshape(-1).to(torch.int32).contiguous()
    scale_t = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)

    if not packed:
        if attn_sink is not None:
            raise NotImplementedError("attn_sink is the packed (B1) path; pass packed=True")
        kv2d = kv.reshape(kv.shape[0], -1)
        if kv2d.shape[1] != head_dim:
            raise ValueError(f"kv last dim must be head_dim={head_dim}, got {kv2d.shape[1]}")
        n_kv_rows = kv2d.shape[0]
        kv_u8 = kv2d.view(torch.uint8) if kv2d.dtype != torch.uint8 else kv2d
        exe = _get_kernel(
            head_dim=head_dim, v_dim=v_dim, num_regions=1, has_sink=False,
            r0_dtype="fp8", r0_fnuz=True, r1_dtype="fp8", r1_fnuz=True,
            qk_split=False, block_n=32, block_h=16, split_kv=False, packed=False, scale_mode="none",
        )
        with torch.cuda.device(q.device.index):
            _run_compiled(
                exe, q.contiguous().reshape(-1), kv_u8.contiguous().reshape(-1),
                indices_i32, indptr_i32, out.reshape(-1), scale_t,
                int(num_queries), int(n_kv_rows), _fx_stream(q.device, stream),
            )
        return

    # ---- packed B1 path ----
    if block_table is None:
        raise ValueError("packed=True requires block_table [num_reqs, max_blocks] int32")
    cache_u8, default_rows, _ = _packed_cache_meta(kv, block_size)
    n_kv_rows = int(num_kv_rows) if num_kv_rows is not None else default_rows
    bt = block_table.to(torch.int32).contiguous()
    max_blocks = bt.shape[1]
    bt_flat = bt.reshape(-1)
    if q_req is None:
        q_req_t = torch.zeros(num_queries, dtype=torch.int32, device=q.device)
    else:
        q_req_t = q_req.to(torch.int32).reshape(-1).contiguous()
    has_sink = attn_sink is not None
    if has_sink:
        sink_t = attn_sink.to(torch.float32).reshape(-1).contiguous()
    else:
        sink_t = torch.zeros(num_heads, dtype=torch.float32, device=q.device)

    exe = _get_kernel(
        head_dim=head_dim, v_dim=v_dim, num_regions=1, has_sink=has_sink,
        r0_dtype="fp8", r0_fnuz=True, r1_dtype="fp8", r1_fnuz=True,
        qk_split=True, block_n=32, block_h=16, split_kv=False, packed=True, scale_mode=scale_mode,
    )
    with torch.cuda.device(q.device.index):
        _run_compiled(
            exe,
            q.contiguous().reshape(-1),
            cache_u8,
            indices_i32, indptr_i32, bt_flat,
            cache_u8, indices_i32, indptr_i32, bt_flat,  # extra_* dummies (unused, NREG==1)
            q_req_t, sink_t,
            out.reshape(-1), scale_t,
            int(num_queries),
            int(n_kv_rows), int(n_kv_rows),
            int(block_size), int(block_size),
            int(max_blocks), int(max_blocks),
            _fx_stream(q.device, stream),
        )


def flydsl_sparse_mla_prefill_2region(
    q: torch.Tensor,  # [num_queries, 128, 512] bf16
    out: torch.Tensor,  # [num_queries, 128, 512] bf16 (in place)
    main_cache: torch.Tensor,  # packed fp8_ds_mla SWA cache (fnuz on gfx942)
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    main_block_table: torch.Tensor,
    extra_cache: torch.Tensor,  # packed fp8_ds_mla compressed pool (OCP)
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    extra_block_table: torch.Tensor,
    *,
    block_size: int,
    scale: float,
    attn_sink: torch.Tensor | None = None,
    extra_block_size: int | None = None,
    main_num_rows: int | None = None,
    extra_num_rows: int | None = None,
    q_req: torch.Tensor | None = None,
    main_is_fnuz: bool = True,
    extra_is_fnuz: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Phase B2: two-region native sparse MLA prefill (compressed + SWA).

    Region 0 (main / SWA) and region 1 (extra / compressed) run through one
    shared online-softmax state. ``main_is_fnuz`` / ``extra_is_fnuz`` select the
    per-region NoPE fp8 convention (gfx942: SWA fnuz, compressed OCP).
    """
    if not (q.is_cuda and out.is_cuda and main_cache.is_cuda and extra_cache.is_cuda):
        raise ValueError("flydsl_sparse_mla_prefill_2region requires CUDA/HIP tensors")
    _check_gfx942(q.device)
    num_queries, num_heads, head_dim, v_dim = _validate_qout(q, out)
    if main_indptr.numel() != num_queries + 1 or extra_indptr.numel() != num_queries + 1:
        raise ValueError("main/extra indptr must have num_queries+1 elems")

    e_block_size = block_size if extra_block_size is None else extra_block_size
    m_cache_u8, m_default_rows, _ = _packed_cache_meta(main_cache, block_size)
    e_cache_u8, e_default_rows, _ = _packed_cache_meta(extra_cache, e_block_size)
    m_rows = int(main_num_rows) if main_num_rows is not None else m_default_rows
    e_rows = int(extra_num_rows) if extra_num_rows is not None else e_default_rows

    m_idx = main_indices.reshape(-1).to(torch.int32).contiguous()
    m_iptr = main_indptr.reshape(-1).to(torch.int32).contiguous()
    e_idx = extra_indices.reshape(-1).to(torch.int32).contiguous()
    e_iptr = extra_indptr.reshape(-1).to(torch.int32).contiguous()
    m_bt = main_block_table.to(torch.int32).contiguous()
    e_bt = extra_block_table.to(torch.int32).contiguous()
    m_max_blocks = m_bt.shape[1]
    e_max_blocks = e_bt.shape[1]

    if q_req is None:
        q_req_t = torch.zeros(num_queries, dtype=torch.int32, device=q.device)
    else:
        q_req_t = q_req.to(torch.int32).reshape(-1).contiguous()
    has_sink = attn_sink is not None
    sink_t = (
        attn_sink.to(torch.float32).reshape(-1).contiguous()
        if has_sink
        else torch.zeros(num_heads, dtype=torch.float32, device=q.device)
    )
    scale_t = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)

    exe = _get_kernel(
        head_dim=head_dim, v_dim=v_dim, num_regions=2, has_sink=has_sink,
        r0_dtype="fp8", r0_fnuz=main_is_fnuz, r1_dtype="fp8", r1_fnuz=extra_is_fnuz,
        qk_split=True, block_n=32, block_h=16, split_kv=False, packed=True, scale_mode="none",
    )
    with torch.cuda.device(q.device.index):
        _run_compiled(
            exe,
            q.contiguous().reshape(-1),
            m_cache_u8, m_idx, m_iptr, m_bt.reshape(-1),
            e_cache_u8, e_idx, e_iptr, e_bt.reshape(-1),
            q_req_t, sink_t,
            out.reshape(-1), scale_t,
            int(num_queries),
            int(m_rows), int(e_rows),
            int(block_size), int(e_block_size),
            int(m_max_blocks), int(e_max_blocks),
            _fx_stream(q.device, stream),
        )
