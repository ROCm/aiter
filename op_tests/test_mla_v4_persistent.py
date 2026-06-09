# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Reference test for the DeepSeek-V4 (MODEL1_FP8Sparse) MLA decode path,
mirroring op_tests/test_mla_persistent.py but without an aiter v4 kernel
comparison (the aiter v4 kernel only ships the qh64/qseqlen4/gqa16 ASM
variant today; this file establishes the torch reference + metadata call so
the kernel comparison can be wired in once available).

V4 layout per token (logical):
  - nope:  448 elements, FP8 (e4m3fnuz on gfx94x, e4m3fn on gfx95x)
  - scale:   7 E8M0 scales (1 per 64-elt quant tile), stored duplicated
             as 14 uint8 bytes (scale[2i]==scale[2i+1]; kernel reads at
             /32 granularity). Consumed by
             v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4 (byte B -> 2^(B-127)).
  - rope:   64 elements, BF16
  - d_qk = 448 + 64 = 512
  - d_v  = 512   (V is the *whole* d_qk slice -- both nope and rope --
                  unlike v3.2 where d_v = 512 sliced off the rope)
  - QK softmax scale = 1 / sqrt(d_qk) = 1 / sqrt(512)
"""

import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import is_experimental_enabled
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


# ---------------------------------------------------------------------------
# V4 layout constants. From sglang flashmla_tests/quant.py
# (FP8KVCacheLayout.MODEL1_FP8Sparse): (d, d_nope, d_rope, tile_size, num_tiles)
# = (512, 448, 64, 64, 7).
# ---------------------------------------------------------------------------
V4_DIM_NOPE = 448  # FP8 nope elements per token
V4_DIM_ROPE = 64  # BF16 rope elements per token
V4_DIM_QK = V4_DIM_NOPE + V4_DIM_ROPE  # 512
V4_DIM_V = V4_DIM_QK  # PV uses the full nope+rope slice
V4_TILE = 64  # nope elements covered by one ue8m0 scale
V4_NUM_TILES = V4_DIM_NOPE // V4_TILE  # 7   (active scales: one per 64-elt quant tile)
# Storage granularity is half the quant tile -- 32 elts per byte slot -- with
# scale[2i] == scale[2i+1] (the "duplicate" pattern; consumer reads the byte
# at (col/32) and always gets the correct per-64-tile scale). No bpad: the
# 14 = 448/32 byte slots fit exactly, no padding inside the scale region.
V4_DIM_SCALE_DUP = V4_DIM_NOPE // (V4_TILE // 2)  # 14 bytes (post-duplicate_each)
# Packed Q/KV layout the kernel reads (one FP8 byte per element):
#   stride per token = 512 bytes
#   bytes [0   , 448): NOPE FP8 (kDimNope)
#   bytes [448 , 462): 14 duplicated E8M0 scales (one byte per 32-elt sub-tile;
#                      scale[2i] == scale[2i+1] since the actual quant is per-64)
#   bytes [462 , 512): 50 bytes unused trailing pad (contents undefined --
#                      the kernel never reads this region)
V4_DIM_QK_PACKED = 512
V4_PACK_OFF_NOPE = 0
V4_PACK_OFF_SCALE = V4_DIM_NOPE  # 448
V4_PACK_OFF_PAD = V4_DIM_NOPE + V4_DIM_SCALE_DUP  # 462
# FP8 |max| differs between archs: e4m3fn (gfx95x) = 448, e4m3fnuz (gfx94x) = 240.
# The sglang reference uses 448 (assumes e4m3fn); we look it up from torch.finfo
# so the per-tile scale lands inside the representable range on either arch.


# ---------------------------------------------------------------------------
# Metadata dumper (kept identical to test_mla_persistent.dump_mla_metadata_v1_txt
# so the same DUMP_MLA_METADATA env switch works here too).
# ---------------------------------------------------------------------------
def dump_mla_metadata_v1_txt(
    filepath: Union[str, Path],
    *,
    batch: int,
    q_seq_len: int,
    max_num_blocks: int,
    work_q: int,
    work_kv: int,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    col_width: int = 5,
) -> None:
    path = Path(filepath)
    wi = work_indptr.detach().cpu().to(torch.int64).tolist()
    wis = work_info_set.detach().cpu().to(torch.int32)
    total_tgs = len(wi) - 1
    w = col_width

    def tg_first_work_row(tg: int):
        if tg < 0 or tg >= total_tgs:
            return None
        w0 = int(wi[tg])
        w1 = int(wi[tg + 1])
        if w0 >= w1 or w0 >= wis.shape[0]:
            return None
        return wis[w0]

    def line_for(name, pick) -> str:
        parts = []
        for tg in range(total_tgs):
            row = tg_first_work_row(tg)
            parts.append(pick(row) if row is not None else 0)
        nums = " ".join(f"{v:>{w}}" for v in parts)
        return f"{name}:\n    {nums}\n"

    work_ind_line = " ".join(f"{int(v):>{w}}" for v in wi)
    lines = [
        f"batch:{batch}, q_seq_len:{q_seq_len}, max_num_blocks:{max_num_blocks}, "
        f"work_q:{work_q}, work_kv:{work_kv}, total_tgs:{total_tgs}\n",
        line_for("bs_indptr", lambda r: int(r[0].item())),
        line_for("partial_indptr", lambda r: int(r[1].item())),
        line_for("w_q_start", lambda r: int(r[2].item())),
        line_for("w_q_end", lambda r: int(r[3].item())),
        line_for("w_kv_start", lambda r: int(r[4].item())),
        line_for("w_kv_end", lambda r: int(r[5].item())),
        f"work_indptr:\n    {work_ind_line}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# V4 quantization. Per-tile (64-element) ue8m0 scale: amax / FP8_AMAX rounded
# UP to the nearest power of 2. Mirrors sglang flashmla_tests/quant.py
# `quantize_k_cache(MODEL1_FP8Sparse)`.
# ---------------------------------------------------------------------------
def fp32_pow2_to_e8m0(pow2_fp32: torch.Tensor) -> torch.Tensor:
    """
    Pack a power-of-2 fp32 scale into a 1-byte E8M0 exponent
    (byte B encodes 2^(B-127); B=0 -> 0.0, B=255 -> INF). The kernel
    reads these bytes directly via v_mfma_scale_f32_*_f8f6f4.
    """
    safe = torch.where(pow2_fp32 > 0, pow2_fp32, torch.ones_like(pow2_fp32))
    biased = torch.log2(safe).round().to(torch.int32) + 127
    biased = torch.clamp(biased, 0, 254)
    biased = torch.where(pow2_fp32 > 0, biased, torch.zeros_like(biased))
    return biased.to(torch.uint8)


def e8m0_to_fp32(byte: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 -> fp32 scale; mirrors mla_v4.h:54 `fp8e8m0_to_fp32`."""
    b = byte.to(torch.int32)
    out = torch.where(
        b == 0,
        torch.zeros_like(b, dtype=torch.float32),
        torch.where(
            b == 255,
            torch.full_like(b, float("inf"), dtype=torch.float32),
            torch.exp2((b - 127).to(torch.float32)),
        ),
    )
    return out


def cast_scale_inv_to_ue8m0_pow2(scales_inv: torch.Tensor) -> torch.Tensor:
    """amax/FP8_AMAX -> ceil-log2 -> power-of-2 fp32 (intermediate, pre-pack)."""
    return torch.pow(2.0, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(
        torch.float32
    )


def quantize_v4_nope_bpad8(
    nope_fp32: torch.Tensor,  # [..., V4_DIM_NOPE]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tile (64 elt) E8M0 quantization. Returns (nope_fp8, scale_e8m0,
    nope_dq_bf16):
      - nope_fp8: [..., 448]  FP8
      - scale_e8m0: [..., 7]  uint8 E8M0 bytes (one per 64-elt quant tile).
        The packer below duplicates each byte to land 14 bytes in the
        on-disk record (kernel reads at /32 granularity).
      - nope_dq_bf16: [..., 448]  bf16 round-trip the BF16 MFMA actually sees.
    """
    fp8_amax = float(torch.finfo(dtypes.fp8).max)
    leading = nope_fp32.shape[:-1]
    tiled = nope_fp32.reshape(*leading, V4_NUM_TILES, V4_TILE)
    active_scale_pow2 = cast_scale_inv_to_ue8m0_pow2(
        tiled.abs().amax(dim=-1) / fp8_amax
    )  # [..., 7]  fp32 pow2
    nope_fp8 = (
        (tiled / active_scale_pow2.unsqueeze(-1))
        .to(dtypes.fp8)
        .reshape(*leading, V4_DIM_NOPE)
    )

    # Pack to uint8 E8M0 (7 bytes/token, one per quant tile).
    scale_e8m0 = fp32_pow2_to_e8m0(active_scale_pow2)  # [..., 7] uint8

    nope_dq_bf16 = (
        (
            nope_fp8.to(torch.float32).reshape(*leading, V4_NUM_TILES, V4_TILE)
            * active_scale_pow2.unsqueeze(-1)
        )
        .reshape(*leading, V4_DIM_NOPE)
        .to(torch.bfloat16)
    )
    return nope_fp8, scale_e8m0, nope_dq_bf16


def quantize_v4_q(
    q: torch.Tensor,  # [total_q, nhead, V4_DIM_QK]  bf16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize Q the same way the ASM kernel sees it: nope FP8 + bpad8 E8M0
    scales, rope kept BF16. Returns (q_nope_fp8, q_nope_scale_e8m0,
    q_rope_bf16, q_silver_bf16) where q_silver_bf16 is the round-tripped Q
    the BF16 MFMA consumes.
    """
    q_nope_fp32 = q[..., :V4_DIM_NOPE].float()
    q_rope_bf16 = q[..., V4_DIM_NOPE:].to(torch.bfloat16)
    q_nope_fp8, q_nope_scale_e8m0, q_nope_dq_bf16 = quantize_v4_nope_bpad8(q_nope_fp32)
    q_silver_bf16 = torch.cat([q_nope_dq_bf16, q_rope_bf16], dim=-1)
    return q_nope_fp8, q_nope_scale_e8m0, q_rope_bf16, q_silver_bf16


# ---------------------------------------------------------------------------
# Kernel-shaped (packed) Q/KV layout, 512 bytes per token:
#   1. NOPE                              -> 448 bytes
#   2. duplicate_each(scale_e8m0)        -> 14 bytes (7 scales x 2 dup)
#   3. unused trailing pad               -> 50 bytes (contents undefined)
# Total: 448 + 14 + 50 = 512 bytes/token.
# ---------------------------------------------------------------------------
def _duplicate_each_lastdim(x: torch.Tensor) -> torch.Tensor:
    """[..., N] -> [..., 2*N] with each element written twice; mirrors
    mla_v4.h:73 `duplicate_each`."""
    return x.unsqueeze(-1).expand(*x.shape, 2).reshape(*x.shape[:-1], x.shape[-1] * 2)


def pack_v4_nope_scale(
    nope_fp8: torch.Tensor,  # [..., 448]   FP8 (1 byte/elem)
    scale_e8m0: torch.Tensor,  # [..., 7]    uint8 E8M0 (one byte per 64-elt quant tile)
) -> torch.Tensor:
    """Pack NOPE + duplicated E8M0 scale into a single 512-byte per-token FP8
    tensor matching the kernel's read stride. The 7 input scale bytes are
    written twice each (14 bytes total) so the kernel can read at /32
    granularity. Trailing 50 bytes are unused (contents undefined)."""
    leading = nope_fp8.shape[:-1]
    assert nope_fp8.shape[-1] == V4_DIM_NOPE
    assert scale_e8m0.shape[-1] == V4_NUM_TILES
    assert scale_e8m0.shape[:-1] == leading

    packed = torch.zeros(
        (*leading, V4_DIM_QK_PACKED), dtype=torch.uint8, device=nope_fp8.device
    )
    packed[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE] = nope_fp8.view(
        torch.uint8
    )
    packed[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP] = (
        _duplicate_each_lastdim(scale_e8m0)
    )
    # bytes [V4_PACK_OFF_PAD:V4_DIM_QK_PACKED] left uninitialized (50 bytes).
    return packed.view(dtypes.fp8)


def unpack_v4_nope_scale(
    packed: torch.Tensor,  # [..., 512]   FP8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_v4_nope_scale; recovers (nope_fp8, scale_e8m0).
    Reads the *first* of each duplicated scale byte pair."""
    pb = packed.view(torch.uint8)
    nope_fp8 = pb[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE].view(
        packed.dtype
    )
    scale_dup = pb[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP]
    scale_e8m0 = scale_dup.reshape(*scale_dup.shape[:-1], V4_NUM_TILES, 2)[
        ..., 0
    ].contiguous()
    return nope_fp8, scale_e8m0


def init_v4_kv_cache(
    num_page: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a paged KV cache from a single fp32 source. Returns both the
    "golden" pure-bf16 buffer (no fp8 anywhere) and the kernel-shaped
    (FP8 nope + E8M0 scale + BF16 rope) buffers.

    Returns:
      - kv_buffer_bf16      [num_page, page_size, 1, d_qk=512]  golden ref
                                                                (bf16 cast of fp32)
      - kv_nope_fp8         [num_page, page_size, 1, 448]       FP8 nope
      - kv_nope_scale_e8m0  [num_page, page_size, 1, 7]         uint8 E8M0 bytes
                                                                (one per 64-elt quant tile)
      - kv_rope_bf16        [num_page, page_size, 1, 64]        BF16 rope
    """
    nope_fp32 = torch.randn((num_page, page_size, 1, V4_DIM_NOPE), dtype=torch.float32)
    rope_bf16 = torch.randn((num_page, page_size, 1, V4_DIM_ROPE), dtype=torch.bfloat16)

    # Golden: raw bf16 cast of the fp32 source -- no fp8 round-trip.
    kv_buffer_bf16 = torch.cat([nope_fp32.to(torch.bfloat16), rope_bf16], dim=-1)

    # Silver-side buffers: per-tile bpad8 E8M0 quantization of the same source.
    nope_fp8, scale_e8m0, _ = quantize_v4_nope_bpad8(nope_fp32)
    return kv_buffer_bf16, nope_fp8, scale_e8m0, rope_bf16


def dequant_v4_kv(
    nope_fp8: torch.Tensor,  # [num_page, page_size, 1, 448]
    scale_e8m0: torch.Tensor,  # [num_page, page_size, 1, 7]  uint8 E8M0 (per 64-elt tile)
    rope_bf16: torch.Tensor,  # [num_page, page_size, 1, 64]
) -> torch.Tensor:
    """Reassemble [num_page, page_size, 1, d_qk=512] in fp32 from the 3 buffers."""
    num_page, page_size, _, _ = nope_fp8.shape
    active_scale = e8m0_to_fp32(scale_e8m0[..., :V4_NUM_TILES])
    nope_dq = (
        nope_fp8.to(torch.float32).reshape(
            num_page, page_size, 1, V4_NUM_TILES, V4_TILE
        )
        * active_scale.unsqueeze(-1)
    ).reshape(num_page, page_size, 1, V4_DIM_NOPE)
    return torch.cat([nope_dq, rope_bf16.to(torch.float32)], dim=-1)


# ---------------------------------------------------------------------------
# V4 reference attention. Two key differences vs the v3.2 reference in
# test_mla_persistent.ref_masked_attention:
#   1. K is the full d_qk=512 (nope_dq + rope).
#   2. V is *also* the full d_qk=512 (NOT k[..., :d_nope]) -- d_v == d_qk in v4.
# ---------------------------------------------------------------------------
def ref_masked_attention_v4(
    query: torch.Tensor,  # [s_q, h_q, d_qk=512]
    key: torch.Tensor,  # [s_k, h_kv=1, d_qk=512]
    value: torch.Tensor,  # [s_k, h_kv=1, d_v=512]   -- same buffer as key in v4
    scale: float,
    out_dtype: torch.dtype,
    is_causal: bool = True,
    causal_diagonal: int = None,
    attn_sink: torch.Tensor = None,  # optional [h_q] fp32 per-head sink logit
) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q, s_k = query.shape[0], key.shape[0]
        diag = causal_diagonal if causal_diagonal is not None else s_k - s_q
        bias = torch.zeros(s_q, s_k, dtype=torch.float32)
        bias.masked_fill_(
            torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=diag).logical_not(),
            float("-inf"),
        )
        attn = attn + bias

    # Sink: virtual K column with constant logit, zero V. Inflates the
    # softmax denominator but contributes nothing to the V numerator.
    # attn: [h, q, k]; attn_sink: [h] -> broadcast to [h, q, 1].
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).to(attn.device).view(-1, 1, 1)
        attn_aug = torch.cat([attn, sink.expand(-1, attn.shape[1], 1)], dim=-1)
        lse = attn_aug.logsumexp(dim=-1)
        m = attn_aug.max(dim=-1).values
        attn_exp = torch.exp(attn - m.unsqueeze(-1))  # NOTE: only over real K
        sink_exp = torch.exp(sink - m.unsqueeze(-1))  # [h, q, 1]
        l = attn_exp.sum(-1) + sink_exp.squeeze(-1)  # noqa: E741
    else:
        lse = attn.logsumexp(dim=-1)
        m = attn.max(dim=-1).values
        attn_exp = torch.exp(attn - m.unsqueeze(-1))
        l = attn_exp.sum(-1)  # noqa: E741
    out = torch.einsum("hqk,khd->qhd", attn_exp, value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    return out.to(out_dtype), lse


def _v4_dequant_nope_bpad8(
    nope_fp8: torch.Tensor,  # [..., 448]   FP8
    nope_scale_e8m0: torch.Tensor,  # [..., 8]     uint8 E8M0 bpad8
) -> torch.Tensor:
    """fp8 * per-tile E8M0 scale -> bf16. Mirrors mla_v4.h:124
    (`fp8e4m3_mul_fp8e8m0_bpad8_to_bf16`). The kernel does the equivalent
    multiply via v_mfma_scale_f32_*_f8f6f4 reading the same E8M0 bytes."""
    leading = nope_fp8.shape[:-1]
    active_scale = e8m0_to_fp32(nope_scale_e8m0[..., :V4_NUM_TILES])
    return (
        (
            nope_fp8.to(torch.float32).reshape(*leading, V4_NUM_TILES, V4_TILE)
            * active_scale.unsqueeze(-1)
        )
        .reshape(*leading, V4_DIM_NOPE)
        .to(torch.bfloat16)
    )


def torch_mla_extend_v4_silver(
    # Q (per-token, kernel layout): NOPE 448 FP8 + dup-scale 16 + zero pad 112
    q_packed,  # [total_q, nhead, 512]              FP8
    q_rope_bf16,  # [total_q, nhead, 64]               BF16
    # KV (paged, kernel layout)
    kv_packed,  # [num_page, page_size, 1, 512]      FP8
    kv_rope_bf16,  # [num_page, page_size, 1, 64]       BF16
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    out_dtype,
    is_causal: bool = True,
    attn_sink: torch.Tensor = None,
):
    """
    Reference whose inputs match the ASM kernel's exactly: a single 512-byte
    packed FP8 tensor per Q/KV stream (NOPE bytes + duplicated E8M0 scale +
    zero pad) plus a separate BF16 rope tensor. Internally splits the packed
    buffer into (nope_fp8, scale_bpad8) via `unpack_v4_nope_scale`, dequants
    nope per-tile to BF16 (E8M0 byte B -> 2^(B-127)), concats with rope, then
    runs the same BF16 attention as the golden ref. This captures the FP8
    quantization noise the kernel pays via `v_mfma_scale_f32_*_f8f6f4`.
    """
    q_nope_fp8, q_nope_scale_e8m0 = unpack_v4_nope_scale(q_packed)
    q_nope_bf16 = _v4_dequant_nope_bpad8(q_nope_fp8, q_nope_scale_e8m0)
    q_silver_bf16 = torch.cat([q_nope_bf16, q_rope_bf16], dim=-1)

    kv_nope_fp8, kv_nope_scale_e8m0 = unpack_v4_nope_scale(kv_packed)
    kv_nope_bf16 = _v4_dequant_nope_bpad8(kv_nope_fp8, kv_nope_scale_e8m0)
    kv_silver_bf16 = torch.cat([kv_nope_bf16, kv_rope_bf16], dim=-1)

    return torch_mla_extend_v4(
        q_silver_bf16,
        kv_silver_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        out_dtype,
        is_causal=is_causal,
        attn_sink=attn_sink,
    )


def torch_mla_extend_v4(
    q,  # [total_q, nhead, d_qk=512]
    kv_buffer_bf16,  # [num_page, page_size, 1, d_qk=512]   (dequant golden)
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    out_dtype,
    is_causal: bool = True,
    attn_sink: torch.Tensor = None,  # optional [nhead] fp32
):
    """V4 paged-attention reference. K and V are the same tensor (full d_qk slice)."""
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kv_buffer_bf16, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    page_size = kv_buffer_bf16.shape[1]
    bs = qo_indptr.shape[0] - 1

    outs, lses = [], []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_len = (cur_num_page - 1) * page_size + int(kv_last_page_lens[i].item())
        kvi = kvs[i].flatten(0, 1)[:real_kv_len]  # [s_k, 1, d_qk]
        # In v4: K and V both use the full d_qk slice (nope+rope).
        o, lse = ref_masked_attention_v4(
            qs[i],
            kvi,
            kvi,
            sm_scale,
            out_dtype,
            is_causal=is_causal,
            attn_sink=attn_sink,
        )
        outs.append(o)
        lses.append(lse)

    out = torch.concat(outs)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return out, lse


def torch_mla_v4_split_kv(
    q_silver_bf16,
    kv_silver_bf16,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    work_info_set,
    work_indptr,
    is_causal=True,
    attn_sink: torch.Tensor = None,  # only applied on first split of each batch
):
    num_page, page_size, _, d_qk = kv_silver_bf16.shape
    total_q, nheads, _ = q_silver_bf16.shape
    dev = kv_silver_bf16.device

    kvc = torch.index_select(kv_silver_bf16, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    num_works = int(work_indptr[-1].item())

    final_out = torch.empty(total_q, nheads, d_qk, dtype=torch.bfloat16, device=dev)
    final_lse = torch.empty(total_q, nheads, dtype=torch.float32, device=dev)
    partial_os, partial_lses = [], []

    for work_idx in range(num_works):
        row = work_info_set[work_idx]
        batch_idx = int(row[0].item())
        partial_qo_loc = int(row[1].item())
        qo_start = int(row[2].item())
        qo_end = int(row[3].item())
        kv_start = int(row[4].item())
        kv_offset = int(row[6].item())

        cur_num_page = kvs[batch_idx].shape[0]
        cur_real_kv_seq_len = (cur_num_page - 1) * page_size + int(
            kv_last_page_lens[batch_idx].item()
        )
        real_sum_kv_seq_len = (
            int(kv_indptr[batch_idx].item()) * page_size + cur_real_kv_seq_len
        )

        slice_k = kvc.flatten(0, 1)[
            kv_start * page_size : real_sum_kv_seq_len - kv_offset
        ]
        slice_q = q_silver_bf16[qo_start:qo_end]

        out_dtype = torch.float32 if partial_qo_loc != -1 else torch.bfloat16

        causal_diagonal = None
        if is_causal:
            q_local_start = qo_start - int(qo_indptr[batch_idx].item())
            kv_local_start = (kv_start - int(kv_indptr[batch_idx].item())) * page_size
            total_q_len = int(qo_indptr[batch_idx + 1].item()) - int(
                qo_indptr[batch_idx].item()
            )
            causal_diagonal = (
                q_local_start - kv_local_start + cur_real_kv_seq_len - total_q_len
            )

        # Sink: same fold-rule as the kernel -- apply on OutputFinal
        # (partial_qo_loc == -1, single split) OR on the LAST split of this
        # batch element. kv_offset == 0 iff this split's kv_end coincides
        # with the batch tail (planner sets
        # kv_offset = curr_kv_end - work_info.kv_end). Last-vs-first is
        # mathematically equivalent for the reducer combine; last-split is
        # cheaper on the kernel side (no extra kv_indptr load).
        is_last_split = kv_offset == 0
        work_sink = (
            attn_sink
            if (attn_sink is not None and (partial_qo_loc == -1 or is_last_split))
            else None
        )

        o, lse = ref_masked_attention_v4(
            slice_q,
            slice_k,
            slice_k,
            sm_scale,
            out_dtype,
            is_causal=is_causal,
            causal_diagonal=causal_diagonal,
            attn_sink=work_sink,
        )

        if partial_qo_loc == -1:
            final_out[qo_start:qo_end, :, :] = o
            final_lse[qo_start:qo_end, :] = lse.transpose(0, 1)
        else:
            partial_os.append(o)
            partial_lses.append(lse)

    partial_o = (
        torch.concat(partial_os)
        if partial_os
        else torch.empty(0, nheads, d_qk, dtype=torch.float32, device=dev)
    )
    partial_lse = (
        torch.concat(partial_lses, dim=1).transpose(0, 1)
        if partial_lses
        else torch.empty(0, nheads, dtype=torch.float32, device=dev)
    )
    partial_o = torch.where(
        torch.isnan(partial_o), torch.zeros_like(partial_o), partial_o
    )
    partial_lse = torch.where(
        torch.isnan(partial_lse),
        torch.full_like(partial_lse, float("-inf")),
        partial_lse,
    )
    return partial_o, partial_lse, final_out, final_lse


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def test_mla_v4(
    ctx_lens,
    batch_size,
    nhead,
    page_size,
    varlen,
    decode_qlen,
    max_split_per_batch,
    use_attn_sink: bool = False,
):
    gfx = get_gfx()
    if gfx not in ["gfx950"]:
        print(
            f"skip test_mla_v4(b={batch_size}, c={ctx_lens}, n={nhead}, "
            f"ql={decode_qlen}): unsupported on {gfx}"
        )
        return None
    if not is_experimental_enabled():
        print(
            f"skip test_mla_v4(b={batch_size}, c={ctx_lens}, n={nhead}, "
            f"ql={decode_qlen}): requires AITER_ENABLE_EXPERIMENTAL=1"
        )
        return None

    ret = {}
    out_dtype = torch.bfloat16

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)

    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = random.uniform(5, ctx_lens)
            seq_lens_qo[i] = max(
                min(int(random.normalvariate(ctx_lens, ctx_lens / 2)), ctx_lens), 1
            )
            kv_block_nums[i] = (seq_lens_kv[i] + page_size - 1) // page_size
            kv_last_page_lens[i] = (
                page_size
                if seq_lens_kv[i] % page_size == 0
                else seq_lens_kv[i] % page_size
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
        kv_block_nums.fill_((ctx_lens + page_size - 1) // page_size)
        kv_last_page_lens.fill_(
            page_size if ctx_lens % page_size == 0 else ctx_lens % page_size
        )

    kv_indptr[1 : batch_size + 1] = torch.cumsum(kv_block_nums, dim=0)
    num_page = int(kv_indptr[-1].item())
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = int(seq_lens_qo.max().item())

    # ---- decode-only path (matches test_mla_persistent.test_mla) ----
    seq_lens_qo.fill_(decode_qlen)
    max_seqlen_qo = int(seq_lens_qo.max().item())
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())

    # V4 buffers
    (
        kv_buffer_bf16,  # golden ref (pure bf16, no fp8)
        kv_nope_fp8,  # FP8 nope                       (silver)
        kv_nope_scale_e8m0,  # uint8 E8M0 bpad8 scales        (silver)
        kv_rope_bf16,  # BF16 rope                      (silver)
    ) = init_v4_kv_cache(num_page, page_size)

    q = torch.randn((total_q, nhead, V4_DIM_QK), dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(V4_DIM_QK)  # = 1/sqrt(512)
    nhead_kv = 1

    # Silver Q: FP8 nope + E8M0 bpad8 scale + BF16 rope (the kernel's input layout).
    q_nope_fp8, q_nope_scale_e8m0, q_rope_bf16, _ = quantize_v4_q(q)
    q_rope_bf16 = q_rope_bf16.contiguous()

    # Pack Q/KV into the 512-byte/token kernel layout (NOPE + dup-scale + unused pad
    # pad). This is the exact byte stream mla.py will hand to the ASM kernel
    # once the v4 wrapper lands; build it here so the silver path already
    # consumes the same bytes (it splits NOPE/scale back out internally).
    q_packed = pack_v4_nope_scale(q_nope_fp8, q_nope_scale_e8m0)
    kv_packed = pack_v4_nope_scale(kv_nope_fp8, kv_nope_scale_e8m0)

    # ---- golden reference (Q & KV both pure BF16, no FP8 anywhere) ----
    out_ref, lse_ref = torch_mla_extend_v4(
        q,
        kv_buffer_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        out_dtype=out_dtype,
        is_causal=True,
    )

    # ---- attention sink (optional, per-head bias logit) ----
    # Random small-magnitude sink so the augmented denominator differs
    # meaningfully from the no-sink one but doesn't dominate (which would
    # collapse the V output toward zero and trivialize the check).
    attn_sink = None
    if use_attn_sink:
        attn_sink = torch.randn(nhead, dtype=torch.float32) * 0.5

    # ---- silver reference (kernel-shaped inputs: 512-byte packed FP8 + BF16 rope) ----
    out_silver, lse_silver = torch_mla_extend_v4_silver(
        q_packed,
        q_rope_bf16,
        kv_packed,
        kv_rope_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        out_dtype=out_dtype,
        is_causal=True,
        attn_sink=attn_sink,
    )

    # Quantization-induced drift between golden and silver -- this is the
    # noise floor a real fp8 kernel will sit on top of.
    out_drift_max = (out_ref.float() - out_silver.float()).abs().max().item()
    out_drift_mean = (out_ref.float() - out_silver.float()).abs().mean().item()

    aiter.logger.info(
        "v4 golden vs silver drift: max_abs=%.4f mean_abs=%.5f",
        out_drift_max,
        out_drift_mean,
    )

    # ---- metadata (same v1 API as test_mla_persistent) ----
    if nhead >= 128:
        gpu = torch.cuda.current_device()
        cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
        max_split_per_batch = min(
            (cu_num + batch_size - 1) // batch_size, max_split_per_batch
        )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    work_meta_data = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device="cuda"
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(
        work_info_set_size, dtype=work_info_set_type, device="cuda"
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
    )

    if os.environ.get("DUMP_MLA_METADATA", ""):
        kv_gran = max(page_size, 16)
        max_num_blocks = max(
            (int(seq_lens_kv[i].item()) + kv_gran - 1) // kv_gran
            for i in range(batch_size)
        )
        num_works = int(work_indptr[-1].item())
        if num_works > 0:
            r0 = work_info_set[0, :6].detach().cpu()
            hdr_work_q = int(r0[3].item() - r0[2].item())
        else:
            hdr_work_q = int(max_seqlen_qo)
        dump_mla_metadata_v1_txt(
            os.environ.get("MLA_METADATA_DUMP_PATH", "mla_v4_metadata_dump.txt"),
            batch=batch_size,
            q_seq_len=int(max_seqlen_qo),
            max_num_blocks=max_num_blocks,
            work_q=hdr_work_q,
            work_kv=kv_gran,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
        )

    num_works = int(work_indptr[-1].item())
    aiter.logger.info(
        "v4 ref ok: batch=%d ctx=%d nhead=%d decode_qlen=%d "
        "out=%s lse=%s num_works=%d max_split=%d",
        batch_size,
        ctx_lens,
        nhead,
        decode_qlen,
        tuple(out_ref.shape),
        tuple(lse_ref.shape),
        num_works,
        max_split_per_batch,
    )

    # ---- V4.0 decode kernel (router; HK is the only backend today) ----
    # Packed FP8 (NOPE+dup-scale+pad) Q/KV + BF16 RoPE Q/KV; output BF16.
    # mla_v40_decode_fwd raises NotImplementedError for shapes the router
    # can't dispatch yet, so we only invoke it when the HK constraint
    # (nhead*decode_qlen)==128 is satisfied.
    if max_seqlen_qo * nhead == 128:
        out_v40 = torch.empty((total_q, nhead, V4_DIM_V), dtype=out_dtype)
        (v40_logits, _v40_final_lse), us_v40_decode = run_perftest(
            aiter.mla.mla_v40_decode_fwd,
            q_packed,
            q_rope_bf16,
            kv_packed,
            kv_rope_bf16,
            out_v40,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
        )
        ret["v40_us"] = us_v40_decode
        err = checkAllclose(
            out_silver.to(out_dtype),
            out_v40,
            msg=(
                f"mla_v40_decode    [silver vs aiter_v40]: "
                f"b={batch_size} c={ctx_lens} n={nhead} ql={decode_qlen}"
            ),
        )
        ret["v40_err"] = err

        q_nope_bf16 = _v4_dequant_nope_bpad8(q_nope_fp8, q_nope_scale_e8m0)
        q_silver_bf16 = torch.cat([q_nope_bf16, q_rope_bf16], dim=-1)
        kv_nope_bf16 = _v4_dequant_nope_bpad8(kv_nope_fp8, kv_nope_scale_e8m0)
        kv_silver_bf16 = torch.cat([kv_nope_bf16, kv_rope_bf16], dim=-1)

        partial_out_ref, partial_lse_ref, _, _ = torch_mla_v4_split_kv(
            q_silver_bf16,
            kv_silver_bf16,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            work_info_set,
            work_indptr,
            is_causal=True,
            attn_sink=attn_sink,
        )

        if partial_out_ref.shape[0] > 0:
            v40_logits_flat = v40_logits[: partial_out_ref.shape[0]].flatten(0, 1)
            checkAllclose(
                partial_out_ref,
                v40_logits_flat,
                msg=(
                    f"mla_v40_decode    [partial_out_ref vs attn_logits]: "
                    f"{us_v40_decode:>8.2f} us......"
                ),
            )

        total_kv = int(seq_lens_kv.sum().item())
        flops = decode_qlen * total_kv * nhead * (V4_DIM_QK + V4_DIM_V) * 2

        kv_bytes = (
            total_kv
            * nhead_kv
            * (
                V4_DIM_NOPE * 1  # FP8 NoPE
                + V4_NUM_TILES * 1  # E8M0 scales actually consumed (7 B/token)
                + V4_DIM_ROPE * 2  # BF16 RoPE
            )
        )
        q_bytes = (
            total_q * nhead * (V4_DIM_NOPE * 1 + V4_NUM_TILES * 1 + V4_DIM_ROPE * 2)
        )
        out_bytes = total_q * nhead * V4_DIM_V * out_v40.element_size()
        bytes_v40 = kv_bytes + q_bytes + out_bytes

        ret["v40_flops"] = flops
        ret["v40_bytes"] = bytes_v40
        ret["v40_TFLOPS"] = flops / us_v40_decode / 1e6
        ret["v40_TB/s"] = bytes_v40 / us_v40_decode / 1e6

    ret["batch"] = batch_size
    ret["ctx_lens"] = ctx_lens
    ret["nhead"] = nhead
    ret["decode_qlen"] = decode_qlen
    ret["max_split_per_batch"] = max_split_per_batch
    ret["num_works"] = num_works
    ret["out_shape"] = tuple(out_ref.shape)
    ret["lse_shape"] = tuple(lse_ref.shape)
    return ret


# ---------------------------------------------------------------------------
# argparse driver (matches test_mla_persistent.py flag names)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="DSv4 MLA reference test (torch ref + metadata only).",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="Page size. e.g.: -blk 1",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[64, 256, 1200, 8192],
    help="Context length(s). e.g.: -c 64 256",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 16, 64],
    help="Batch size(s). e.g.: -b 1 16",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="*",
    const=None,
    default=[(16, 4), (128, 1)],  # v4 nm shipped variant: (16, 4) -> 16*4=64
    help="(num_heads, decode_qlen) tuples. e.g.: -n 16,4",
)
parser.add_argument(
    "-ms",
    "--max_split_per_batch",
    type=int,
    nargs="*",
    default=[32],
    help="Max KV splits per batch. e.g.: -ms 32",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="Variable kv seqlens. Default: False",
)
parser.add_argument(
    "--attn_sink",
    action="store_true",
    help="Test attention sink (per-head bias logit). Default: False",
)

args = parser.parse_args()

for nhead, decode_qlen in args.nhead:
    df = []
    for ctx_len, batch_size, max_split_per_batch in itertools.product(
        args.ctxLen, args.batchSize, args.max_split_per_batch
    ):
        ret = test_mla_v4(
            ctx_len,
            batch_size,
            nhead,
            args.block_size,
            varlen=args.varlen,
            decode_qlen=decode_qlen,
            max_split_per_batch=max_split_per_batch,
            use_attn_sink=args.attn_sink,
        )
        if ret is None:
            continue
        df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_v4_persistent summary (markdown):\n%s", df_md)
