# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""sparse_mla_fwd on gfx950 with a per-tensor fp8 cache ("fp8_scalar") and fp8 dots:
the kernel in _gluon_kernels/gfx950/attention/sparse_mla_fp8.py. sparse_mla_fwd
calls sparse_mla_fp8_fwd first; calls it cannot address run the generic kernel."""

import math
from typing import NamedTuple

import torch

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _sparse_mla_reduce as _sparse_mla_reduce_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla_fp8 import (
    _sparse_mla_fp8,
)
from aiter.ops.triton.attention.pa_decode_sparse import _launch_splits
from aiter.ops.triton.attention.sparse_mla import _check_out, _mla_num_splits
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms, get_num_xcds


class _Fp8Segments(NamedTuple):
    """The fp8 kernel's paged lists. A map is (row offset, block stride in rows,
    power-of-two block size) from slot ids to rows of the list's base cache.
    extra_kv: the second cache when the lists run in two loops."""

    two_loops: bool
    main_map: tuple
    extra_indptr: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_map: tuple = (0, 1, 1)
    extra_kv: torch.Tensor | None = None
    extra_kv_scale: torch.Tensor | None = None


class _Fp8Route(NamedTuple):
    """How the fp8 kernel reads a call's caches (_fp8_route)."""

    cache: torch.Tensor  # base of the rows
    segs: _Fp8Segments | None  # None: one flat list


def _fp8_paged_map(cache, base_ptr, row):
    """(row offset, block stride in rows, block size) of a [nb, block, row] fp8
    cache from base_ptr, slot s being row offset + (s // block) * stride
    + s % block. None if the fp8 kernel cannot address it (unaligned, padded
    rows, a block size that is not a power of two)."""
    if cache.ndim != 3 or cache.shape[2] != row or cache.element_size() != 1:
        return None
    block = cache.shape[1]
    if cache.stride(2) != 1 or (block > 1 and cache.stride(1) != row):
        return None
    off = cache.data_ptr() - base_ptr
    if off < 0 or off % row or cache.stride(0) % row or block & (block - 1):
        return None
    return off // row, cache.stride(0) // row, block


def _fp8_route(
    *,
    arch,
    fmt,
    fp8_dots,
    skip_reduce,
    has_invalid,
    q,
    out,
    kv_lora_rank,
    qk_rope_head_dim,
    kv_buffer,
    cache,
    block_size,
    kv_scale,
    kv_indices,
    extra_kv,
    extra_kv_scale,
    extra_indptr,
    extra_indices,
):
    """The _Fp8Route if the fp8 kernel (_gluon_kernels/.../sparse_mla_fp8.py)
    takes this call, else None.

    It takes the fp8_scalar cache with fp8 dots and a 512-wide latent, with no
    rope buffer (DeepSeek-V4 / V4.1, GLM-5.3-Flash) or a 64-wide one (GLM-5.2),
    whenever it can address the caches; everything else, skip_reduce and -1
    sentinels (has_invalid) run the generic kernel. Host metadata only (storage
    pointers, strides, scale identity): no device reads, safe under graph capture.

    A flat pool is one list of row ids. A paged one (cache and block_size from
    _cache_pointers, block_size > 1) maps slot ids to rows. With extra_kv (vLLM's
    SWA cache as kv_buffer, the compressed top-k cache as extra_kv), both caches
    in one allocation under one scale are read as one list in one loop, from the
    lower of the two; otherwise in two loops, each from its own cache."""
    if arch != "gfx950":  # the fp8 kernel is written for gfx950
        return None
    _, num_heads, row = q.shape
    # the id lists are read with buffer loads (signed 32-bit offsets)
    ids_ok = all(
        max_addressable_bytes(t) < 2**31 - 1
        for t in (kv_indices, extra_indices)
        if t is not None
    )
    if not (
        fmt == "fp8_scalar"
        and fp8_dots
        and not skip_reduce
        and not has_invalid
        and kv_lora_rank == 512
        and qk_rope_head_dim in (0, 64)
        and row == kv_lora_rank + qk_rope_head_dim
        and num_heads in (8, 16, 32, 64, 128)
        and q.stride(2) == 1
        and (out is None or out.stride(2) == 1)
        and ids_ok
    ):
        return None
    if extra_kv is None:
        if block_size == 1:
            if not (
                cache.ndim == 2 and cache.stride(0) == row and cache.stride(1) == 1
            ):
                return None
            return _Fp8Route(cache, None)
        main = _fp8_paged_map(cache, cache.data_ptr(), row)
        if main is None:
            return None
        return _Fp8Route(cache, _Fp8Segments(False, main))
    main_u8 = kv_buffer.view(torch.uint8)
    x_u8 = extra_kv.view(torch.uint8)
    one_alloc = (
        main_u8.untyped_storage().data_ptr() == x_u8.untyped_storage().data_ptr()
    )
    one_scale = extra_kv_scale.data_ptr() == kv_scale.data_ptr()
    if one_alloc and one_scale:
        # one u8 view over both caches, from the lower one: the base of both lists
        lo = min(main_u8, x_u8, key=torch.Tensor.data_ptr)
        end = max(t.data_ptr() + max_addressable_bytes(t) for t in (main_u8, x_u8))
        base = lo.new_empty(0).set_(
            lo.untyped_storage(),
            lo.data_ptr() - lo.untyped_storage().data_ptr(),
            (end - lo.data_ptr(),),
        )
        main = _fp8_paged_map(main_u8, base.data_ptr(), row)
        extra = _fp8_paged_map(x_u8, base.data_ptr(), row)
        if main is None or extra is None:
            return None
        segs = _Fp8Segments(False, main, extra_indptr, extra_indices, extra)
        return _Fp8Route(base, segs)
    main = _fp8_paged_map(main_u8, main_u8.data_ptr(), row)
    extra = _fp8_paged_map(x_u8, x_u8.data_ptr(), row)
    if main is None or extra is None:
        return None
    segs = _Fp8Segments(
        True,
        main,
        extra_indptr,
        extra_indices,
        extra,
        x_u8,
        None if one_scale else extra_kv_scale.reshape(1).to(torch.float32),
    )
    return _Fp8Route(main_u8, segs)


def _check_epilogue(q, out, inv_rope_positions, inv_rope_cos_sin_cache, out_mxfp8):
    """The fp8 kernel's output epilogue arguments, as pa_decode_sparse takes them."""
    C, H, D = q.shape
    if (inv_rope_positions is None) != (inv_rope_cos_sin_cache is None):
        raise ValueError("inv_rope_positions and inv_rope_cos_sin_cache go together")
    if out_mxfp8 is not None and inv_rope_positions is None:
        raise ValueError("out_mxfp8 needs the inverse-RoPE inputs")
    if inv_rope_positions is not None:
        cs = inv_rope_cos_sin_cache
        if inv_rope_positions.shape != (C,) or inv_rope_positions.stride(0) != 1:
            raise ValueError(f"inv_rope_positions must be a contiguous [{C}]")
        if cs.dtype != torch.float32 or cs.ndim != 2 or cs.shape[1] != 64:
            raise ValueError("inv_rope_cos_sin_cache must be [P, 64] f32 (cos | sin)")
        if cs.stride(1) != 1 or cs.device != q.device:
            raise ValueError(
                "inv_rope_cos_sin_cache rows must be contiguous, on q's device"
            )
    if out_mxfp8 is not None:
        if out is not None:
            raise ValueError("out and out_mxfp8 are mutually exclusive")
        data, scale = out_mxfp8
        if data.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
            raise ValueError("out_mxfp8 is (e4m3 data, uint8 E8M0 scales)")
        if data.shape != (C, H * D) or scale.shape != (C, H * D // 32):
            raise ValueError(
                f"out_mxfp8 must be ([{C}, {H * D}], [{C}, {H * D // 32}]), got "
                f"{tuple(data.shape)}, {tuple(scale.shape)}"
            )
        if data.stride(-1) != 1 or scale.stride(-1) != 1:
            raise ValueError("out_mxfp8 rows must be contiguous")


# How far (log2) a score may beat the fp8 kernel's running max before the max
# moves; most tiles then skip the accumulator rescale. 0 is the exact max.
_FP8_LAZY = 8


def _fp8_plan(num_queries, num_heads, avg_topk, two_loops=False):
    """(heads per program, split count) for the fp8 kernel."""
    num_cus = get_num_sms()
    # 64 heads once that alone fills the CUs. Not for the two-loop kernel at 64
    # heads: it spills there and 32 is faster.
    if (
        num_heads % 64 == 0
        and num_queries * (num_heads // 64) >= num_cus
        and not (two_loops and num_heads == 64)
    ):
        return 64, 1
    # Short rows: fewer heads per program beats splitting, whose partials cost
    # more than the extra gathers.
    if math.ceil(avg_topk / 64) < 16:
        for g in (32, 16, 8):
            if num_heads % g == 0 and num_queries * (num_heads // g) >= num_cus // 2:
                return g, 1
    g = min(32, num_heads)
    return g, _mla_num_splits(num_queries, num_heads // g, avg_topk)


def _sparse_mla_fp8_launch(
    q,
    q_scale,
    route,
    kv_scale,
    kv_indptr,
    kv_indices,
    softmax_scale,
    out,
    *,
    rope,
    kv_splits=None,
    attn_sink=None,
    return_lse=False,
    inv_rope_positions=None,
    inv_rope_cos_sin_cache=None,
    out_mxfp8=None,
):
    """The fp8 kernel's launch: heads per program and split count, the kernel and,
    when split, the generic merge. q [C, H, 512 + rope] fp8 (+ a [1] f32 q_scale)
    or bf16; route from _fp8_route; kv_scale [1] f32. out [C, H, >= 512] bf16, or
    None with out_mxfp8 (e4m3 data [C, H * 512], E8M0 scales [C, H * 16]).
    Returns (out, lse or None)."""
    C, H, _ = q.shape
    segs = route.segs
    avg_topk = kv_indices.numel() / max(1, C)
    if segs is not None and segs.extra_indices is not None:
        avg_topk += segs.extra_indices.numel() / max(1, C)
    block_h, num_splits = _fp8_plan(C, H, avg_topk, segs is not None and segs.two_loops)
    if kv_splits is not None:
        num_splits = max(1, int(kv_splits))
    split = num_splits > 1
    P = _launch_splits(num_splits)
    q_bf16 = q.dtype == torch.bfloat16
    lse = (
        torch.empty((C, H), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    # the kernel's paged-list arguments
    if segs is None:
        lists = {
            "MAIN_BLOCK_LOG2": 0,
            "EXTRA_BLOCK_LOG2": 0,
            "x_indptr_ptr": None,
            "x_idx_ptr": None,
            "main_row0": None,
            "main_stride": None,
            "extra_row0": None,
            "extra_stride": None,
            "x_kv_ptr": None,
            "x_ks_ptr": None,
        }
    else:
        lists = {
            "MAIN_BLOCK_LOG2": segs.main_map[2].bit_length() - 1,
            "EXTRA_BLOCK_LOG2": segs.extra_map[2].bit_length() - 1,
            "x_indptr_ptr": segs.extra_indptr,
            "x_idx_ptr": segs.extra_indices,
            "main_row0": segs.main_map[0],
            "main_stride": segs.main_map[1],
            "extra_row0": segs.extra_map[0],
            "extra_stride": segs.extra_map[1],
            "x_kv_ptr": segs.extra_kv,
            "x_ks_ptr": segs.extra_kv_scale,
        }
    # the output epilogue's arguments, shared with the merge
    inv = inv_rope_positions is not None
    mx = out_mxfp8 is not None
    epi = {
        "pos_ptr": inv_rope_positions,
        "cs_ptr": inv_rope_cos_sin_cache,
        "cs_stride": inv_rope_cos_sin_cache.stride(0) if inv else 0,
        "osc_ptr": out_mxfp8[1] if mx else None,
        "osc_stride": out_mxfp8[1].stride(0) if mx else 0,
    }
    if mx:
        out = out_mxfp8[0].view(C, H, 512)

    if split:
        part_m = torch.empty((C, P, H), dtype=torch.float32, device=q.device)
        part_l = torch.empty_like(part_m)
        part_acc = torch.empty((C, P, H, 512), dtype=torch.bfloat16, device=q.device)
    else:
        part_m = part_l = part_acc = None
    # A row's head blocks run back to back on one XCD (Program.initialize), so
    # the later blocks' gathers hit its L2.
    head_blocks = H // block_h
    xcds = get_num_xcds()
    if head_blocks > 1:
        grid = (-(-C // xcds) * xcds * head_blocks, 1, P)
    else:
        grid = (C, 1, P)
    # buffer loads take signed 32-bit offsets; a cache past them gathers in 64 bit
    x_kv = None if segs is None else segs.extra_kv
    caches = (route.cache,) if x_kv is None else (route.cache, x_kv)
    addr64 = any(max_addressable_bytes(t) >= 2**31 - 1 for t in caches)
    _sparse_mla_fp8[grid](
        q_ptr=q if q_bf16 else q.view(torch.uint8),
        kv_ptr=route.cache,
        out_ptr=None if split else out,
        indptr_ptr=kv_indptr,
        idx_ptr=kv_indices,
        sink_ptr=attn_sink,
        q_scale_ptr=None if q_bf16 else q_scale,
        ks_ptr=kv_scale.reshape(1).to(torch.float32),
        lse_ptr=None if split else lse,
        part_m_ptr=part_m,
        part_l_ptr=part_l,
        part_acc_ptr=part_acc,
        num_queries=C,
        num_heads=H,
        sm_scale=softmax_scale,
        num_splits=num_splits,
        q_stride0=q.stride(0),
        q_stride1=q.stride(1),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        BLOCK_H=block_h,
        ROPE_TAIL=rope,
        LAZY_LOG2=_FP8_LAZY,
        ADDR64=addr64,
        HEAD_BLOCKS=head_blocks,
        NUM_XCDS=xcds,
        **lists,
        **epi,
        # the RoPE tile needs key waves: 64 heads take 8 waves there
        num_warps=8 if (rope and block_h == 64) else 4,
        waves_per_eu=2,
        # no AGPRs: Triton 3.9's LLVM infers it, 3.8's splits the budget in two
        llvm_fn_attrs="amdgpu-agpr-alloc=0",
        enable_fp_fusion=True,
    )
    if split:
        _sparse_mla_reduce_gfx950[(C, H)](
            part_m,
            part_l,
            part_acc,
            attn_sink,
            out,
            out.stride(0),
            out.stride(1),
            part_m.stride(0),
            part_m.stride(1),
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
            H,
            HAS_SINK=attn_sink is not None,
            HEAD_SIZE=512,
            BLOCK_M=1,
            NUM_SPLITS=P,
            HEAD_ALIGNED=True,
            ADAPTIVE_SPLITS=True,
            lse_ptr=lse,
            HAS_LSE=lse is not None,
            pos_ptr=epi["pos_ptr"],
            cos_sin_ptr=epi["cs_ptr"],
            cs_stride=epi["cs_stride"],
            out_scale_ptr=epi["osc_ptr"],
            os_stride0=epi["osc_stride"],
            ROPE_DIM=64,
            INV_ROPE=inv,
            OUT_MXFP8=mx,
            # a 2-split tile spans two warps; more would hold duplicate lanes
            num_warps=min(4, P),
            enable_fp_fusion=True,
        )
    return out, lse


def sparse_mla_fp8_fwd(
    q,
    q_scale,
    kv_buffer,
    cache,
    block_size,
    kv_scale,
    kv_indptr,
    kv_indices,
    softmax_scale,
    out,
    *,
    kv_lora_rank,
    qk_rope_head_dim,
    kv_splits,
    skip_reduce,
    has_invalid,
    attn_sink,
    return_lse,
    extra_kv,
    extra_kv_scale,
    extra_indptr,
    extra_indices,
    inv_rope_positions,
    inv_rope_cos_sin_cache,
    out_mxfp8,
):
    """(out, lse) from the fp8 kernel, or None when it cannot address the call.
    The arguments are sparse_mla_fwd's after its checks; cache, block_size and
    kv_scale come from _cache_pointers."""
    route = _fp8_route(
        arch="gfx950",
        fmt="fp8_scalar",
        fp8_dots=True,
        skip_reduce=skip_reduce,
        has_invalid=has_invalid,
        q=q,
        out=out,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_buffer=kv_buffer,
        cache=cache,
        block_size=block_size,
        kv_scale=kv_scale,
        kv_indices=kv_indices,
        extra_kv=extra_kv,
        extra_kv_scale=extra_kv_scale,
        extra_indptr=extra_indptr,
        extra_indices=extra_indices,
    )
    epilogue = inv_rope_positions is not None or out_mxfp8 is not None
    if epilogue and (route is None or qk_rope_head_dim):
        raise ValueError(
            "the inverse-RoPE / MXFP8 output epilogue needs a DeepSeek-V4 cache, or "
            "an fp8_scalar one with dot_precision='fp8' in the 512-wide rope-inside "
            "geometry"
        )
    if route is None:
        return None
    _check_epilogue(q, out, inv_rope_positions, inv_rope_cos_sin_cache, out_mxfp8)
    if out_mxfp8 is None:
        out = _check_out(out, q, kv_lora_rank)
    return _sparse_mla_fp8_launch(
        q,
        q_scale,
        route,
        kv_scale,
        kv_indptr,
        kv_indices,
        softmax_scale,
        out,
        rope=qk_rope_head_dim,
        kv_splits=kv_splits,
        attn_sink=attn_sink,
        return_lse=return_lse,
        inv_rope_positions=inv_rope_positions,
        inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
        out_mxfp8=out_mxfp8,
    )
