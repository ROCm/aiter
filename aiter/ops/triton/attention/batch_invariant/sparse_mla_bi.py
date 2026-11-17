# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant sparse (top-k) MLA forward, Gluon kernels (gfx950)."""

import functools
import math

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.batch_invariant.sparse_mla_bi import (
    _sparse_mla_bi_combine_kernel,
    _sparse_mla_bi_fused_kernel,
    _sparse_mla_bi_split_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = ["BI_CONTRACT_VERSION", "sparse_mla_bi_fwd"]

# Bump when a change can change the output bits; callers compare it at startup.
BI_CONTRACT_VERSION = 1

KEY_TILE = 64  # keys per tile, in the order of the row's index list
NUM_SPLITS = 8  # split s owns tiles s, s + 8, ...; merged in order 0 .. 7
KV_LORA_RANK = 512
ROPE_DIM = 64
_DSV32_ROW = KV_LORA_RANK + 4 * (KV_LORA_RANK // 128) + 2 * ROPE_DIM  # 656 B
_MAX_HEADS = 128


@functools.cache
def _config() -> dict:
    """Pinned launch values. They do not change the bits."""
    cfg_dir = resolve_config_dir("attention", "SPARSE_MLA_BI", backend="gluon")
    return dict(load_config_json(f"{cfg_dir}/DEFAULT.json")["sparse_mla_bi"])


def _cache_args(kv: torch.Tensor, kv_scale, d_qk: int):
    """-> (fp8, base, bf16 view, f32 view, block_size, page_stride). page_stride
    is in bytes for fp8_dsv32_mla and in bf16 elements for a bf16 cache."""
    if kv_scale is not None:
        raise ValueError(
            "kv_scale must be None: fp8_dsv32_mla carries its own scales and a "
            "bf16 cache has none"
        )
    if kv.dtype == torch.bfloat16:
        if kv.shape[-1] != d_qk or kv.stride(-1) != 1:
            raise ValueError(
                f"a bf16 cache needs contiguous rows of width {d_qk}, got "
                f"{tuple(kv.shape)} with strides {kv.stride()}"
            )
        if kv.ndim == 4 and kv.shape[1] == 1 and kv.shape[2] == 1:
            return False, kv, kv, kv, 1, kv.stride(0)
        if kv.ndim == 2:
            return False, kv, kv, kv, 1, kv.stride(0)
        if kv.ndim == 3:
            if kv.shape[1] > 1 and kv.stride(1) != d_qk:
                raise ValueError("a [nb, block, R] bf16 cache must have packed rows")
            return False, kv, kv, kv, kv.shape[1], kv.stride(0)
        raise ValueError(
            f"bf16 cache must be [slots, R], [slots, 1, 1, R] or [nb, block, R], got "
            f"{tuple(kv.shape)}"
        )
    if kv.element_size() == 1 and kv.ndim == 3 and kv.shape[2] == _DSV32_ROW:
        u8 = kv.view(torch.uint8)
        if u8.stride(2) != 1 or (u8.shape[1] > 1 and u8.stride(1) != _DSV32_ROW):
            raise ValueError("fp8_dsv32_mla rows must be contiguous 656 B records")
        if u8.stride(0) % 4 or u8.storage_offset() % 4:
            raise ValueError(
                "fp8_dsv32_mla pages must start on 4 B boundaries (f32 scales)"
            )
        return (
            True,
            u8,
            u8.view(torch.bfloat16),
            u8.view(torch.float32),
            u8.shape[1],
            u8.stride(0),
        )
    raise ValueError(
        f"unsupported kv_buffer {tuple(kv.shape)} {kv.dtype}: expected a bf16 cache "
        f"with {d_qk}-wide rows or an fp8_dsv32_mla [nb, block, {_DSV32_ROW}] uint8 "
        "cache"
    )


def _int32_1d(t: torch.Tensor, name: str, device) -> torch.Tensor:
    if not torch.is_tensor(t) or t.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
    if t.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32 (int64 accepted), got {t.dtype}")
    if t.device != device:
        raise ValueError(f"{name} is on {t.device}, q is on {device}")
    if t.numel() >= 2**31:
        raise ValueError(f"{name} has {t.numel()} entries; the limit is 2**31 - 1")
    return t.to(torch.int32).contiguous()


def sparse_mla_bi_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_rope_head_dim: int = ROPE_DIM,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    attn_sink: torch.Tensor | None = None,
    _mode: str | None = None,
    _addr64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse (top-k gathered) MLA attention with batch-invariant bits.

    The arguments are the subset of sparse_mla_fwd that Kimi K3.1 uses. Each
    output row depends only on its own query, index list and cache rows: one
    fixed reduction order (64-key tiles, NUM_SPLITS fixed splits, FP32 online
    softmax, splits merged in order), for every batch, packing and token count.

    Args:
        q: [C, H, 576] bf16 (512 latent + 64 rope), H <= 128, last stride 1.
        kv_buffer: fp8_dsv32_mla [nb, block, 656] uint8 records (512 fp8 | 4 f32
            per-128 scales | 64 bf16 rope), or a bf16 cache [slots, 576],
            [slots, 1, 1, 576] or [nb, block, 576]. Slot i is
            (i // block, i % block).
        kv_indptr: [C + 1] int32 offsets of each row's list in kv_indices.
        kv_indices: flat int32 global slot ids. Ids < 0 (-1 sentinels) and ids
            past the cache are masked out and never read.
        softmax_scale: the layer's softmax scale.
        kv_scale: must be None.
        kv_lora_rank, qk_rope_head_dim: must be 512 and 64.
        has_invalid: accepted for sparse_mla_fwd parity; masking is always on.
        dot_precision: must be "bf16". q_scale: ignored (bf16 q).
        out: optional [C, H, >= 512] bf16 destination, last stride 1.
        return_lse: also return the natural-log LSE, [C, H] f32 (-inf when a
            row has no keys).
        attn_sink: optional [H] f32 per-head sink logits.

    Returns:
        (out [C, H, 512] bf16, lse or None). A row without keys gives zeros.
    """
    if get_arch() != "gfx950":
        raise ValueError(f"sparse_mla_bi_fwd supports gfx950 only, got {get_arch()}")
    if q.ndim != 3 or q.dtype != torch.bfloat16:
        raise ValueError(f"q must be [C, H, d_qk] bf16, got {tuple(q.shape)} {q.dtype}")
    if kv_lora_rank != KV_LORA_RANK or qk_rope_head_dim != ROPE_DIM:
        raise ValueError(
            f"kv_lora_rank / qk_rope_head_dim must be {KV_LORA_RANK} / {ROPE_DIM}, "
            f"got {kv_lora_rank} / {qk_rope_head_dim}"
        )
    if dot_precision != "bf16":
        raise ValueError(f"dot_precision must be 'bf16', got {dot_precision!r}")
    C, H, d_qk = q.shape
    if d_qk != KV_LORA_RANK + ROPE_DIM or q.stride(2) != 1:
        raise ValueError(f"q must be [C, H, {KV_LORA_RANK + ROPE_DIM}] with last stride 1")
    if not 1 <= H <= _MAX_HEADS:
        raise ValueError(f"H must be in [1, {_MAX_HEADS}], got {H}")
    if kv_buffer.device != q.device:
        raise ValueError(f"kv_buffer is on {kv_buffer.device}, q is on {q.device}")
    fp8, kv, kv16, kv32, block_size, page_stride = _cache_args(kv_buffer, kv_scale, d_qk)
    num_slots = kv_buffer.shape[0] * block_size
    # Buffer loads take 32-bit byte offsets; a larger cache (or _addr64) uses 64-bit pointers.
    buf = not _addr64 and kv.shape[0] * kv.stride(0) * kv.element_size() < 2**31
    if num_slots >= 2**31:
        raise ValueError(f"the cache has {num_slots} slots; slot ids are int32")
    kv_indptr = _int32_1d(kv_indptr, "kv_indptr", q.device)
    kv_indices = _int32_1d(kv_indices, "kv_indices", q.device)
    if kv_indptr.numel() != C + 1:
        raise ValueError(
            f"kv_indptr must have C + 1 = {C + 1} entries, got {kv_indptr.numel()}"
        )
    if out is None:
        out = torch.empty((C, H, KV_LORA_RANK), dtype=torch.bfloat16, device=q.device)
    elif (
        out.ndim != 3
        or out.shape[0] != C
        or out.shape[1] != H
        or out.shape[2] < KV_LORA_RANK
        or out.stride(2) != 1
        or out.dtype != torch.bfloat16
        or out.device != q.device
    ):
        raise ValueError(f"out must be [{C}, {H}, >= {KV_LORA_RANK}] bf16, last stride 1")
    has_sink = attn_sink is not None
    if has_sink:
        if attn_sink.numel() != H or attn_sink.device != q.device:
            raise ValueError(f"attn_sink must have one entry per head ({H})")
        sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
    else:
        sink = torch.empty(1, dtype=torch.float32, device=q.device)
    lse = torch.empty(
        (C, H) if return_lse else (1,), dtype=torch.float32, device=q.device
    )
    if C == 0:
        return out, (lse if return_lse else None)

    cfg = _config()
    if _mode not in (None, "fused", "split"):
        raise ValueError(f"_mode must be None, 'fused' or 'split', got {_mode!r}")
    # Head tiles hold 16 heads per warp row; the bits do not depend on the tile.
    big = cfg["fused"] if H > cfg["fused_small_max_heads"] else cfg["fused_small"]
    if _mode is None:
        # Split mode wins while the fused grid cannot fill the GPU.
        fused = C * H >= cfg["fused_min_row_heads"]
    else:
        fused = _mode == "fused"
    tile = big if fused else cfg["split"]
    block_h, num_warps = tile["block_h"], tile["num_warps"]
    heads_blocks = triton.cdiv(H, block_h)
    scale_log2 = float(softmax_scale) * math.log2(math.e)
    common = dict(
        FP8=fp8,
        BLOCK_H=block_h,
        KEY_TILE=KEY_TILE,
        D=KV_LORA_RANK,
        R=ROPE_DIM,
        NUM_SPLITS=NUM_SPLITS,
        BUF=buf,
        num_warps=num_warps,
        waves_per_eu=tile.get("waves_per_eu", 0),
        enable_fp_fusion=False,
    )
    if fused:
        xcds = get_num_xcds()
        grid = (triton.cdiv(C, xcds) * xcds * heads_blocks,)
        _sparse_mla_bi_fused_kernel[grid](
            q,
            kv,
            kv16,
            kv32,
            kv_indices,
            kv_indptr,
            sink,
            out,
            lse,
            scale_log2,
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            block_size,
            page_stride,
            num_slots,
            H,
            C,
            heads_blocks,
            HAS_SINK=has_sink,
            HAS_LSE=return_lse,
            NUM_XCDS=xcds,
            **common,
        )
        return out, (lse if return_lse else None)

    part_acc = torch.empty(
        (C, NUM_SPLITS, H, KV_LORA_RANK), dtype=torch.float32, device=q.device
    )
    part_m = torch.empty((C, NUM_SPLITS, H), dtype=torch.float32, device=q.device)
    part_l = torch.empty_like(part_m)
    _sparse_mla_bi_split_kernel[(C, NUM_SPLITS, heads_blocks)](
        q,
        kv,
        kv16,
        kv32,
        kv_indices,
        kv_indptr,
        part_acc,
        part_m,
        part_l,
        scale_log2,
        q.stride(0),
        q.stride(1),
        block_size,
        page_stride,
        num_slots,
        H,
        **common,
    )
    comb = cfg["combine"]
    _sparse_mla_bi_combine_kernel[(C, triton.cdiv(H, comb["block_h"]))](
        part_acc,
        part_m,
        part_l,
        sink,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        H,
        HAS_SINK=has_sink,
        HAS_LSE=return_lse,
        BLOCK_H=comb["block_h"],
        D=KV_LORA_RANK,
        NUM_SPLITS=NUM_SPLITS,
        num_warps=comb["num_warps"],
        enable_fp_fusion=False,
    )
    return out, (lse if return_lse else None)
