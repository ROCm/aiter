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
  - scale:   7 active E8M0 scales (1 per 64 nope elements) padded to 8
             uint8 bytes (bpad8). The kernel reads these bytes directly via
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
from aiter.test_common import checkAllclose

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
V4_NUM_TILES = V4_DIM_NOPE // V4_TILE  # 7   (active scales)
# ASM kernel stores scales bpad8: kDimScale = kDimNope/64 + 1 = 8 slots per
# token, with the last slot zero-padded. See /jruan/doc/hw_ss_team_test/kl/mla/
# mla_v4.h:17 (`kDimScale = 8`) and :146 (`(kDimNope/64 + 1)` indexing).
V4_DIM_SCALE = V4_NUM_TILES + 1  # 8   (storage slots, "bpad8")
# Packed Q/KV layout the ASM kernel actually reads (one FP8 byte per element):
#   stride per token = args.dim(512) + args.k_rotary(64) = 576 bytes
#   bytes [0   , 448): NOPE FP8 (kDimNope)
#   bytes [448 , 464): E8M0 scales duplicated (kDimScale*2 = 16; each bpad8 byte
#                      written twice -- mirrors poc_kl `duplicate_each`)
#   bytes [464 , 512): zero pad (48 bytes; fills out kDimNope+kDimRope=512)
#   bytes [512 , 576): zero pad (64 bytes; "over-copy" region from poc_kl's
#                      hipMemcpy past buf_size_Q -- we zero-fill for determinism)
# See /jruan/doc/hw_ss_team_test/kl/mla/mla_v4.h:245 (`duplicate_each`) and :290
# (`concat_buffers_fast(... kDimScale*2)`); also op_tests/test_mla_v4_nm.py
# (DIM_QK_PACKED=576) and op_tests/test_mla_v4_nm_golden.py docstring.
V4_DIM_QK_PACKED = 576
V4_DIM_SCALE_DUP = V4_DIM_SCALE * 2  # 16 bytes (post-duplicate_each)
V4_PACK_OFF_NOPE = 0
V4_PACK_OFF_SCALE = V4_DIM_NOPE  # 448
V4_PACK_OFF_PAD = V4_DIM_NOPE + V4_DIM_SCALE_DUP  # 464
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
    Per-tile (64 elt) E8M0 quantization with bpad8 storage. Returns
    (nope_fp8, scale_e8m0_bpad8, nope_dq_bf16):
      - nope_fp8: [..., 448]  FP8
      - scale_e8m0_bpad8: [..., 8]  uint8 E8M0 bytes (7 active + 1 zero pad);
        the kernel feeds these directly to v_mfma_scale_f32_*_f8f6f4.
      - nope_dq_bf16: [..., 448]  bf16 round-trip the BF16 MFMA actually sees.
    Mirrors `fp8e4m3_mul_fp8e8m0_bpad8_to_bf16` in mla_v4.h:124.
    """
    fp8_amax = float(torch.finfo(dtypes.fp8).max)
    leading = nope_fp32.shape[:-1]
    tiled = nope_fp32.reshape(*leading, V4_NUM_TILES, V4_TILE)
    active_scale_pow2 = cast_scale_inv_to_ue8m0_pow2(
        tiled.abs().amax(dim=-1) / fp8_amax
    )  # [..., 7]  fp32 pow2
    if os.environ.get("V40_FORCE_SCALE127", "0") == "1":
        active_scale_pow2 = torch.ones_like(active_scale_pow2)
    nope_fp8 = (
        (tiled / active_scale_pow2.unsqueeze(-1))
        .to(dtypes.fp8)
        .reshape(*leading, V4_DIM_NOPE)
    )

    # Pack to uint8 E8M0 with bpad8 (8 bytes/token, last slot = 0).
    active_scale_e8m0 = fp32_pow2_to_e8m0(active_scale_pow2)  # [..., 7] uint8
    scale_e8m0 = torch.zeros(
        (*leading, V4_DIM_SCALE), dtype=torch.uint8, device=nope_fp32.device
    )
    scale_e8m0[..., :V4_NUM_TILES] = active_scale_e8m0

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
# Kernel-shaped (packed) Q/KV layout. Mirrors poc_kl
# `v4_detail::init_host_buffers` (mla_v4.h:250) which builds the 576-byte/token
# tensor the ASM kernel reads via:
#   1. duplicate_each(descale_*_new) -> descale_*_450  (8 -> 16 bytes/token)
#   2. concat(NOPE, descale_*_450)                     -> 448 + 16 = 464 bytes
#   3. concat(step2, zeros)                            -> 464 + 48 = 512 bytes
#   4. allocate full 576-byte stride; trailing 64 bytes are unused over-copy
# ---------------------------------------------------------------------------
def _duplicate_each_lastdim(x: torch.Tensor) -> torch.Tensor:
    """[..., N] -> [..., 2*N] with each element written twice; mirrors
    mla_v4.h:73 `duplicate_each`."""
    return x.unsqueeze(-1).expand(*x.shape, 2).reshape(*x.shape[:-1], x.shape[-1] * 2)


def pack_v4_nope_scale(
    nope_fp8: torch.Tensor,  # [..., 448]   FP8 (1 byte/elem)
    scale_e8m0_bpad8: torch.Tensor,  # [..., 8]     uint8 E8M0 (bpad8)
) -> torch.Tensor:
    """Pack NOPE + duplicated E8M0 scale + zero pad into a single 576-byte
    per-token FP8 tensor matching the ASM kernel's read stride."""
    leading = nope_fp8.shape[:-1]
    assert nope_fp8.shape[-1] == V4_DIM_NOPE
    assert scale_e8m0_bpad8.shape[-1] == V4_DIM_SCALE
    assert scale_e8m0_bpad8.shape[:-1] == leading

    packed = torch.zeros(
        (*leading, V4_DIM_QK_PACKED), dtype=torch.uint8, device=nope_fp8.device
    )
    packed[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE] = nope_fp8.view(
        torch.uint8
    )
    packed[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP] = (
        _duplicate_each_lastdim(scale_e8m0_bpad8)
    )
    # bytes [V4_PACK_OFF_PAD:V4_DIM_QK_PACKED] left zero (48 + 64 over-copy).
    return packed.view(dtypes.fp8)


def unpack_v4_nope_scale(
    packed: torch.Tensor,  # [..., 576]   FP8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_v4_nope_scale; recovers (nope_fp8, scale_e8m0_bpad8).
    Reads the *first* of each duplicated scale byte pair."""
    pb = packed.view(torch.uint8)
    nope_fp8 = pb[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE].view(
        packed.dtype
    )
    scale_dup = pb[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP]
    scale_e8m0_bpad8 = scale_dup.reshape(*scale_dup.shape[:-1], V4_DIM_SCALE, 2)[
        ..., 0
    ].contiguous()
    return nope_fp8, scale_e8m0_bpad8


def init_v4_kv_cache(
    num_page: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a paged KV cache from a single fp32 source. Returns both the
    "golden" pure-bf16 buffer (no fp8 anywhere) and the kernel-shaped
    (FP8 nope + bpad8 scale + BF16 rope) buffers.

    Returns:
      - kv_buffer_bf16      [num_page, page_size, 1, d_qk=512]  golden ref
                                                                (bf16 cast of fp32)
      - kv_nope_fp8         [num_page, page_size, 1, 448]       FP8 nope
      - kv_nope_scale_e8m0  [num_page, page_size, 1, 8]         uint8 E8M0 bytes
                                                                (bpad8: 7 active,
                                                                 last slot = 0)
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
    scale_e8m0: torch.Tensor,  # [num_page, page_size, 1, 8]  uint8 (bpad8)
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
    q_packed,  # [total_q, nhead, 576]              FP8
    q_rope_bf16,  # [total_q, nhead, 64]               BF16
    # KV (paged, kernel layout)
    kv_packed,  # [num_page, page_size, 1, 576]      FP8
    kv_rope_bf16,  # [num_page, page_size, 1, 64]       BF16
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    out_dtype,
    is_causal: bool = True,
):
    """
    Reference whose inputs match the ASM kernel's exactly: a single 576-byte
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
            qs[i], kvi, kvi, sm_scale, out_dtype, is_causal=is_causal
        )
        outs.append(o)
        lses.append(lse)

    out = torch.concat(outs)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return out, lse


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
):
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

    # Pack Q/KV into the 576-byte/token kernel layout (NOPE + dup-scale + zero
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

    # ---- silver reference (kernel-shaped inputs: 576-byte packed FP8 + BF16 rope) ----
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
    )

    # Quantization-induced drift between golden and silver -- this is the
    # noise floor a real fp8 kernel will sit on top of.
    out_drift_max = (out_ref.float() - out_silver.float()).abs().max().item()
    out_drift_mean = (out_ref.float() - out_silver.float()).abs().mean().item()

    if os.environ.get("V40_PROBE_P", ""):
        # Recompute silver internals for head=0 batch=0 to inspect P / V values.
        q_nope_fp8_, q_nope_scale_e8m0_ = unpack_v4_nope_scale(q_packed)
        q_nope_bf16_ = _v4_dequant_nope_bpad8(q_nope_fp8_, q_nope_scale_e8m0_)
        q_silver_bf16_ = torch.cat([q_nope_bf16_, q_rope_bf16], dim=-1)
        kv_nope_fp8_, kv_nope_scale_e8m0_ = unpack_v4_nope_scale(kv_packed)
        kv_nope_bf16_ = _v4_dequant_nope_bpad8(kv_nope_fp8_, kv_nope_scale_e8m0_)
        kv_silver_bf16_ = torch.cat([kv_nope_bf16_, kv_rope_bf16], dim=-1)
        kvi_full = kv_silver_bf16_.flatten(0, 1)  # [num_page*page_size, 1, 512]
        kv0 = kvi_full[:int(seq_lens_kv[0].item())]  # [s_k, 1, 512]
        q0 = q_silver_bf16_[:int(seq_lens_qo[0].item())]  # [s_q, h, 512]
        # P scores for head 0, q-token 0
        scores = torch.einsum("qhd,khd->hqk", q0.float(), kv0.float()) * sm_scale
        sm = torch.softmax(scores, dim=-1)
        print(f"[V40_PROBE_P] sm_scale = {sm_scale:.6f}")
        print(f"[V40_PROBE_P] scores[h=0, q=0, :] = {scores[0, 0].cpu().tolist()}")
        print(f"[V40_PROBE_P] P[h=0, q=0, :] = {sm[0, 0].cpu().tolist()}")
        # NoPE-only and RoPE-only score components
        sc_nope = torch.einsum("qhd,khd->hqk",
                                q0[..., :V4_DIM_NOPE].float(),
                                kv0[..., :V4_DIM_NOPE].float()) * sm_scale
        sc_rope = torch.einsum("qhd,khd->hqk",
                                q0[..., V4_DIM_NOPE:].float(),
                                kv0[..., V4_DIM_NOPE:].float()) * sm_scale
        print(f"[V40_PROBE_P] scores_nope[h=0, q=0, :] = {sc_nope[0, 0].cpu().tolist()}")
        print(f"[V40_PROBE_P] scores_rope[h=0, q=0, :] = {sc_rope[0, 0].cpu().tolist()}")
        # Hypothesis: v40 may compute score on partial dim. Try score_nope_first_256
        sc_nope_256 = torch.einsum("qhd,khd->hqk",
                                q0[..., :256].float(),
                                kv0[..., :256].float()) * sm_scale
        sc_nope_256_448 = torch.einsum("qhd,khd->hqk",
                                q0[..., 256:V4_DIM_NOPE].float(),
                                kv0[..., 256:V4_DIM_NOPE].float()) * sm_scale
        print(f"[V40_PROBE_P] scores_nope_0_256[h=0, q=0, :] = {sc_nope_256[0, 0].cpu().tolist()}")
        print(f"[V40_PROBE_P] scores_nope_256_448[h=0, q=0, :] = {sc_nope_256_448[0, 0].cpu().tolist()}")
        for k in range(min(4, kv0.shape[0])):
            print(f"[V40_PROBE_P] V[k={k}, h=0, 0:8] = "
                  f"{kv0[k, 0, :8].cpu().tolist()}")
        # Hand-compute O[h=0, q=0, 0:8] = sum_k P[0,0,k] * V[k, 0, 0:8]
        O_check = torch.einsum("k,kd->d", sm[0, 0].float(),
                               kv0[:, 0, :].float())
        print(f"[V40_PROBE_P] O_check[h=0, q=0, 0:8] = {O_check[:8].cpu().tolist()}")
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
        v40_logits, _v40_final_lse = aiter.mla.mla_v40_decode_fwd(
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
        )
        err = checkAllclose(
            out_silver.to(out_dtype),
            out_v40,
            msg=(
                f"mla_v40_decode    [silver vs aiter_v40]: "
                f"b={batch_size} c={ctx_lens} n={nhead} ql={decode_qlen}"
            ),
        )
        ret["v40_err"] = err

        if os.environ.get("V40_DUMP", ""):
            silver_b = out_silver.to(out_dtype)
            diff = (silver_b.float() - out_v40.float()).abs()
            match_mask = diff < 0.01
            # Per-(head, col-tile) match rate (32-col tiles, matches output_to_vram chunking)
            mr_per_head_coltile = match_mask.reshape(
                total_q, nhead, V4_DIM_V // 32, 32
            ).float().mean(dim=-1)
            # Per col-tile, averaged across heads + tokens
            mr_per_coltile = mr_per_head_coltile.mean(dim=(0, 1))
            # Per head, averaged across col-tiles + tokens
            mr_per_head = mr_per_head_coltile.mean(dim=(0, 2))
            # Per warp (16 heads per warp)
            mr_per_warp = mr_per_head.reshape(8, 16).mean(dim=-1)
            print(f"[V40_DUMP] match_rate per col_tile (16 tiles of 32 cols): "
                  f"{mr_per_coltile.cpu().tolist()}")
            print(f"[V40_DUMP] match_rate per warp (8 warps of 16 heads): "
                  f"{mr_per_warp.cpu().tolist()}")
            print(f"[V40_DUMP] sample silver[0, 0, :8] = "
                  f"{silver_b[0, 0, :8].cpu().tolist()}")
            print(f"[V40_DUMP] sample v40   [0, 0, :8] = "
                  f"{out_v40[0, 0, :8].cpu().tolist()}")
            # Probe-window dump: when HKMLA_V40_PROBE_KV_LDS is on in the kernel,
            # out[0, 0, 0..15]  = LDS bytes  0.. 31 of p_lds_kv_curr (K[k=0, col=0..15])
            # out[0, 0, 16..31] = LDS bytes 64.. 95 of p_lds_kv_curr (K[k=1, col=0..15])
            print(f"[V40_DUMP] v40 PROBE  K[k=0, 0..15] = "
                  f"{out_v40[0, 0, :16].cpu().tolist()}")
            print(f"[V40_DUMP] v40 PROBE  K[k=1, 0..15] = "
                  f"{out_v40[0, 0, 16:32].cpu().tolist()}")
            # Re-derive K[k=0, h=0, 0..31] from the same packed bytes the
            # kernel reads (so the comparison is apples-to-apples even if
            # quant pack/unpack changes).
            try:
                kv_nope_fp8_p, kv_nope_scale_e8m0_p = unpack_v4_nope_scale(kv_packed)
                kv_nope_bf16_p = _v4_dequant_nope_bpad8(kv_nope_fp8_p, kv_nope_scale_e8m0_p)
                kv_silver_bf16_p = torch.cat([kv_nope_bf16_p, kv_rope_bf16], dim=-1)
                kvflat = kv_silver_bf16_p.flatten(0, 1)  # [num_page*page_size, 1, 512]
                # Dump first 4 KV tokens (k=0..3 for c=4) cols 0..31
                for k in range(min(4, kvflat.shape[0])):
                    print(f"[V40_DUMP] silver K  [k={k}, h=0, 0..15]  = "
                          f"{kvflat[k, 0, :16].cpu().tolist()}")
                    print(f"[V40_DUMP] silver K  [k={k}, h=0, 16..31] = "
                          f"{kvflat[k, 0, 16:32].cpu().tolist()}")
                # Also try with index from kv_indices to see actual mapping
                ki = int(kv_indices[0].item()) if hasattr(kv_indices, 'shape') else 0
                kv_first_token = kv_silver_bf16_p[ki, 0, 0]  # token 0 of mapped page
                print(f"[V40_DUMP] kv_indices[0]={ki}, silver K[page={ki},tok=0,h=0,0..15] = "
                      f"{kv_first_token[:16].cpu().tolist()}")
            except Exception as _e:
                print(f"[V40_DUMP] (could not derive silver K from kv_packed: {_e})")
            print(f"[V40_DUMP] sample silver[0, 0, 32:40] = "
                  f"{silver_b[0, 0, 32:40].cpu().tolist()}")
            print(f"[V40_DUMP] sample v40   [0, 0, 32:40] = "
                  f"{out_v40[0, 0, 32:40].cpu().tolist()}")
            for cb in (448, 464, 480, 496):
                print(f"[V40_DUMP] silver[0, 0, {cb}:{cb+8}] = "
                      f"{silver_b[0, 0, cb:cb+8].cpu().tolist()}")
                print(f"[V40_DUMP] v40   [0, 0, {cb}:{cb+8}] = "
                      f"{out_v40 [0, 0, cb:cb+8].cpu().tolist()}")

        # ------------------------------------------------------------------
        # PMFMA probe comparison: when HKMLA_V40_PROBE_PMFMA is enabled in the
        # kernel, warp 0 writes its p_mfma (post-softmax, pre-normalization
        # exp(scaled_score - row_max), bf16-packed) into out[0, 0, lane*8 +
        # slot] for lane 0..63, slot 0..7. Decode:
        #   lane = g*16 + l   (g = 0..3 = lane group, l = 0..15 = head idx)
        #   slot s in [0..3]: K = g*4 + s
        #   slot s in [4..7]: K = g*4 + (s-4) + 16
        # head index m = l (in decode mode, qlen=1, kTileM=16 = 16 heads/warp).
        if os.environ.get("V40_PROBE_PMFMA", ""):
            print("\n[V40_PROBE_PMFMA] ===== silver vs kernel p_mfma =====")
            # Build silver scores for warp 0 (heads 0..15, q=0, K=0..min(kv_len,32)-1)
            q_nope_fp8_, q_nope_scale_e8m0_ = unpack_v4_nope_scale(q_packed)
            q_nope_bf16_ = _v4_dequant_nope_bpad8(q_nope_fp8_, q_nope_scale_e8m0_)
            q_silver_bf16_ = torch.cat([q_nope_bf16_, q_rope_bf16], dim=-1)
            kv_nope_fp8_, kv_nope_scale_e8m0_ = unpack_v4_nope_scale(kv_packed)
            kv_nope_bf16_ = _v4_dequant_nope_bpad8(kv_nope_fp8_, kv_nope_scale_e8m0_)
            kv_silver_bf16_ = torch.cat([kv_nope_bf16_, kv_rope_bf16], dim=-1)
            # Apply kv_indices remap (kernel reads kv_buffer[kv_indices[i], ...]
            # for batch 0's i-th token). Without this, "silver" sees random tokens.
            kv_indices_b0 = kv_indices[
                int(kv_indptr[0].item()) : int(kv_indptr[1].item())
            ].long().cpu()
            kv_remap = kv_silver_bf16_[kv_indices_b0]  # [num_pages_b0, page_size, 1, 512]
            kvi_full = kv_remap.flatten(0, 1)  # [num_pages*page_size, 1, 512]
            kv_len_b0 = int(seq_lens_kv[0].item())
            kv0 = kvi_full[:kv_len_b0]  # [s_k, 1, 512]
            q0 = q_silver_bf16_[0:1]    # [1, h, 512] (q-token 0)
            # scores[h, q, K] over heads 0..15 (warp 0), q=0, K=0..kv_len_b0-1
            scores = torch.einsum("qhd,khd->hqk",
                                  q0.float(), kv0.float()) * sm_scale  # [h, 1, K]
            row_max = scores.max(dim=-1, keepdim=True).values  # [h, 1, 1]
            silver_exp = torch.exp(scores - row_max)  # [h, 1, K], in [0, 1], max=1.0
            silver_exp_bf16 = silver_exp.to(torch.bfloat16)
            # Build kernel-dump matrix indexed by (head, K_actual) for K in [0..31]
            # via the (g, slot) decode. Fill missing K with NaN.
            kBlockN_v40 = 32
            kern = torch.full((16, kBlockN_v40), float('nan'),
                              dtype=torch.float32, device='cpu')
            v40_cpu = out_v40[0, 0].cpu().float()  # [512]
            for g in range(4):
                for l in range(16):
                    lane = g * 16 + l
                    base = lane * 8
                    for s in range(8):
                        if s < 4:
                            K = g * 4 + s
                        else:
                            K = g * 4 + (s - 4) + 16
                        if K < kBlockN_v40:
                            kern[l, K] = v40_cpu[base + s].item()
            # Print silver vs kernel for heads 0..15, K=0..min(kv_len, 4)
            K_show = min(kv_len_b0, 4)
            print(f"[V40_PROBE_PMFMA] sm_scale = {sm_scale:.6f}, kv_len = {kv_len_b0}")
            print(f"[V40_PROBE_PMFMA] silver_exp = exp(scaled_score - row_max),"
                  f" rounded to bf16; nan in kernel = K not covered for this g.")
            print(f"{'head':>4} | "
                  + " | ".join(f"K={k:>2}(silv,kern)" for k in range(K_show)))
            for h in range(16):
                cells = []
                for k in range(K_show):
                    sv = silver_exp_bf16[h, 0, k].item() if k < silver_exp.shape[-1] else float('nan')
                    kv_ = kern[h, k].item()
                    cells.append(f"({sv:+.4f},{kv_:+.4f})")
                print(f"{h:>4} | " + " | ".join(cells))
            # Also dump max-abs delta across heads + first K columns
            sil_t = silver_exp_bf16[:, 0, :K_show].float()  # [16, K_show]
            ker_t = kern[:, :K_show]
            valid = ~torch.isnan(ker_t)
            diff = (sil_t - ker_t).abs()
            diff_v = diff[valid]
            if diff_v.numel() > 0:
                print(f"[V40_PROBE_PMFMA] max|silver - kern| over heads x K[0..{K_show-1}] = "
                      f"{diff_v.max().item():.5f}, mean = {diff_v.mean().item():.5f}")
            # Also check: for K >= kv_len, are kernel slots 0 (masked)?
            if kv_len_b0 < kBlockN_v40:
                masked_K = kern[:, kv_len_b0:]
                masked_valid = ~torch.isnan(masked_K)
                if masked_valid.any():
                    masked_vals = masked_K[masked_valid]
                    print(f"[V40_PROBE_PMFMA] max|kern| at masked K >= {kv_len_b0} = "
                          f"{masked_vals.abs().max().item():.6f} "
                          f"(should be 0.0)")

        # PCOMP probe: when HKMLA_V40_PROBE_PCOMP is enabled in the kernel,
        # warp 0 writes raw p_comp (8 fp32/lane, BEFORE softmax, the raw QK
        # output) into out[0, 0, lane*16 .. lane*16+15] (= bytes lane*32..+31).
        # Layout matches PMFMA: lane (g, l) p_comp[s=0..3] = scaled_score[
        # head=l, K=g*4+s] (first 16-col tile), p_comp[s=4..7] = scaled_score[
        # head=l, K=g*4+16..g*4+19] (second tile).
        if os.environ.get("V40_PROBE_PCOMP", ""):
            print("\n[V40_PROBE_PCOMP] ===== silver vs kernel raw p_comp (pre-softmax) =====")
            q_nope_fp8_p, q_nope_scale_e8m0_p = unpack_v4_nope_scale(q_packed)
            q_nope_bf16_p = _v4_dequant_nope_bpad8(q_nope_fp8_p, q_nope_scale_e8m0_p)
            q_silver_bf16_p = torch.cat([q_nope_bf16_p, q_rope_bf16], dim=-1)
            kv_nope_fp8_p, kv_nope_scale_e8m0_p = unpack_v4_nope_scale(kv_packed)
            kv_nope_bf16_p = _v4_dequant_nope_bpad8(kv_nope_fp8_p, kv_nope_scale_e8m0_p)
            kv_silver_bf16_p = torch.cat([kv_nope_bf16_p, kv_rope_bf16], dim=-1)
            kv_indices_b0_p = kv_indices[
                int(kv_indptr[0].item()) : int(kv_indptr[1].item())
            ].long().cpu()
            kv_remap_p = kv_silver_bf16_p[kv_indices_b0_p]
            kvi_full_p = kv_remap_p.flatten(0, 1)
            kv_len_b0_p = int(seq_lens_kv[0].item())
            kv0_p = kvi_full_p[:kv_len_b0_p]
            q0_p = q_silver_bf16_p[0:1]
            scores_p = torch.einsum("qhd,khd->hqk",
                                    q0_p.float(), kv0_p.float()) * sm_scale  # [h, 1, K]

            kBlockN_v40 = 32
            kern_p = torch.full((16, kBlockN_v40), float('nan'),
                                dtype=torch.float32, device='cpu')
            # Reinterpret out's first 2048 bytes as fp32 -> 512 fp32 total.
            # lane T writes 8 fp32 at byte v_off = T*32 -> fp32 index T*8.
            # 64 lanes * 32 B = 2048 B spans out[0, 0:2, :] (2 head-rows of 1024 B).
            v40_out_fp32 = (
                out_v40[0, 0:2].cpu().contiguous().reshape(-1).view(torch.uint8)
                .view(torch.float32)
            )  # [512]
            for g in range(4):
                for l in range(16):
                    lane = g * 16 + l
                    base = lane * 8
                    for s in range(8):
                        if s < 4:
                            K = g * 4 + s
                        else:
                            K = g * 4 + (s - 4) + 16
                        if K < kBlockN_v40:
                            kern_p[l, K] = v40_out_fp32[base + s].item()

            K_show = min(kv_len_b0_p, 4)
            print(f"[V40_PROBE_PCOMP] sm_scale = {sm_scale:.6f}, kv_len = {kv_len_b0_p}")
            print(f"[V40_PROBE_PCOMP] silver/kern shown as RAW QK (silver pre-scaled = silver_/sm_scale)")
            print(f"{'head':>4} | "
                  + " | ".join(f"K={k:>2}(silv,kern)" for k in range(K_show)))
            for h in range(16):
                cells = []
                for k in range(K_show):
                    sv = (scores_p[h, 0, k].item() / sm_scale) if k < scores_p.shape[-1] else float('nan')
                    kv_ = kern_p[h, k].item()
                    cells.append(f"({sv:+.4f},{kv_:+.4f})")
                print(f"{h:>4} | " + " | ".join(cells))

            # scores_p shape: [n_heads_total=128, 1, K]. Restrict to warp 0's
            # heads 0..15. Kernel p_comp is RAW QK (no sm_scale), but our silver
            # multiplied by sm_scale -- divide silver by sm_scale to get raw QK.
            sil_full = torch.full((16, kBlockN_v40), float('nan'),
                                  dtype=torch.float32, device='cpu')
            K_actual = min(scores_p.shape[-1], kBlockN_v40)
            sil_full[:, :K_actual] = (scores_p[:16, 0, :K_actual] / sm_scale).cpu()
            valid = ~torch.isnan(kern_p) & ~torch.isnan(sil_full)
            diff = (sil_full - kern_p).abs()
            diff_v = diff[valid]
            if diff_v.numel() > 0:
                print(f"[V40_PROBE_PCOMP] max|silver - kern| over heads x K = "
                      f"{diff_v.max().item():.5f}, mean = {diff_v.mean().item():.5f}")
            # Locate worst cell
            if diff_v.numel() > 0:
                # mask out invalid
                diff_masked = diff.clone()
                diff_masked[~valid] = -1.0
                flat_idx = int(diff_masked.argmax().item())
                h_w = flat_idx // kBlockN_v40
                k_w = flat_idx % kBlockN_v40
                print(f"[V40_PROBE_PCOMP] worst cell: head={h_w}, K={k_w}, "
                      f"silver={sil_full[h_w, k_w].item():+.5f}, "
                      f"kern={kern_p[h_w, k_w].item():+.5f}")

        # KVTOP probe: kernel writes kv_top (4 dwords/lane = 8 bf16/lane) to
        # split_output (v40_logits) as a raw byte stream at offset lane*16. We
        # write to split_output (not out) because the kernel's epilogue still
        # writes to `out` after the probe fires (Phase A iter is mid-compute).
        # Expected mfma_f32_16x16x32_bf16 A-operand layout: lane (g, l) holds
        # K[K_token=l, feat=g*8 + s] for s in 0..7. So at kColOffset = iter*32:
        #   lane (g, l) bf16[0..7] = K[K=l, feat = kColOffset + g*8 .. + g*8+7]
        if os.environ.get("V40_PROBE_KVTOP", ""):
            iter_idx = int(os.environ.get("V40_PROBE_KVTOP_ITER", "0"))
            col_off = iter_idx * 32
            print(f"\n[V40_PROBE_KVTOP] ===== kv_top dump (iter={iter_idx}, kColOffset={col_off}) =====")
            try:
                kv_nope_fp8_p, kv_nope_scale_e8m0_p = unpack_v4_nope_scale(kv_packed)
                kv_nope_bf16_p = _v4_dequant_nope_bpad8(kv_nope_fp8_p, kv_nope_scale_e8m0_p)
                kv_silver_p = torch.cat([kv_nope_bf16_p, kv_rope_bf16], dim=-1)
                kvflat_p = kv_silver_p.flatten(0, 1)  # [num_tokens, 1, 512]
            except Exception as _e:
                print(f"[V40_PROBE_KVTOP] could not derive silver K: {_e}")
                kvflat_p = None
            # v40_logits is fp32 [reduce_partial_map.size(0)*max_seqlen_q, 1, nhead, v_head_dim].
            # Probe writes 16 bytes/lane = 4 dwords starting at offset 0 -> first 256 fp32
            # elements of v40_logits contain (as raw byte stream) 64*16 = 1024 bytes = 512 bf16.
            # Reinterpret first 1024 bytes as bf16.
            v40_logits_flat = v40_logits.flatten()[:256].contiguous().cpu().view(torch.bfloat16)
            for lane in (0, 1, 16, 17, 32, 33, 48, 49):
                g = lane // 16
                l = lane % 16
                kern_vals = v40_logits_flat[lane * 8 : lane * 8 + 8].tolist()
                feat_start = col_off + g * 8
                if kvflat_p is not None and l < kvflat_p.shape[0]:
                    silv_vals = kvflat_p[l, 0, feat_start : feat_start + 8].tolist()
                else:
                    silv_vals = [float("nan")] * 8
                print(f"lane={lane:>2} (g={g}, l={l:>2}) K[k={l},feat={feat_start:>3}..{feat_start+7:>3}]")
                print(f"  kern : {[round(v, 4) for v in kern_vals]}")
                print(f"  silv : {[round(v, 4) for v in silv_vals]}")
                ok = all(abs(a - b) < 1e-3 for a, b in zip(kern_vals, silv_vals))
                print(f"  match: {ok}")

        # Q VGPR / Q LDS probes (Phase A pinned q_vgpr or Phase B q_lds load).
        # Both share the same B-operand layout:
        #   lane (g, l) bf16[0..7] = Q[head = qo_start + l, feat = base + g*8 .. g*8+7]
        # Phase A: base = ITER*32 (covers Q[:, 0..256])
        # Phase B: base = 256 + ITER*32 (covers Q[:, 256..512])
        # Probe writes to split_output as raw byte stream at lane*16.
        for env_name, base_offset_fn, header in (
            ("V40_PROBE_QVGPR",
             lambda it: it * 32,
             "[V40_PROBE_QVGPR] q_vgpr (Phase A, pinned)"),
            ("V40_PROBE_QLDS",
             lambda it: 256 + it * 32,
             "[V40_PROBE_QLDS] q_lds  (Phase B, from LDS)"),
        ):
            if not os.environ.get(env_name, ""):
                continue
            iter_idx = int(os.environ.get(env_name + "_ITER", "0"))
            base_feat = base_offset_fn(iter_idx)
            print(f"\n{header} ===== iter={iter_idx}, base_feat={base_feat} =====")
            try:
                q_nope_fp8_p, q_nope_scale_e8m0_p = unpack_v4_nope_scale(q_packed)
                q_nope_bf16_p = _v4_dequant_nope_bpad8(q_nope_fp8_p, q_nope_scale_e8m0_p)
                # q_silver: [total_q, nhead, 512]
                q_silver = torch.cat([q_nope_bf16_p, q_rope_bf16], dim=-1)
            except Exception as _e:
                print(f"{header} could not derive silver Q: {_e}")
                q_silver = None
            v40_logits_flat = v40_logits.flatten()[:256].contiguous().cpu().view(torch.bfloat16)
            # warp 0 covers q-token 0, heads [qo_start ... qo_start+15]
            qo_start = 0
            mismatch_count = 0
            mismatch_heads = []
            for lane in range(64):
                g = lane // 16
                l = lane % 16
                head_idx = qo_start + l
                feat_start = base_feat + g * 8
                kern_vals = v40_logits_flat[lane * 8 : lane * 8 + 8].tolist()
                if q_silver is not None and head_idx < q_silver.shape[1]:
                    silv_vals = q_silver[0, head_idx, feat_start : feat_start + 8].tolist()
                else:
                    silv_vals = [float("nan")] * 8
                ok = all(abs(a - b) < 1e-3 for a, b in zip(kern_vals, silv_vals))
                if not ok:
                    mismatch_count += 1
                    if l not in mismatch_heads:
                        mismatch_heads.append(l)
                # only print first 4 mismatches and a few sample matches
                if not ok and mismatch_count <= 6:
                    print(f"  MISMATCH lane={lane:>2} (g={g}, l={l:>2}) "
                          f"head={head_idx}, feat={feat_start:>3}..{feat_start+7:>3}")
                    print(f"    kern: {[round(v, 4) for v in kern_vals]}")
                    print(f"    silv: {[round(v, 4) for v in silv_vals]}")
                if ok and lane in (0, 1, 16, 17):
                    print(f"  OK lane={lane:>2} head={head_idx} feat={feat_start:>3}..")
            print(f"{header} total mismatch lanes: {mismatch_count}/64")
            print(f"{header} affected heads (l-values): {sorted(mismatch_heads)}")

        # P1 staging raw fp8 dump probe -- isolation re-issue of chunk's
        # vmem->staging, dumps raw 16 fp8 per lane (bytes 0..8 + bytes 32..40
        # of staging row, both as u8 view). Used to determine if vmem->LDS
        # primitive is correct independent of cvt + production interleaving.
        if os.environ.get("V40_PROBE_P1_STAGING", ""):
            chunk_idx = int(os.environ.get("V40_PROBE_P1_STAGING_CHUNK", "0"))
            print(f"\n[V40_PROBE_P1_STAGING] chunk={chunk_idx} (cols [{chunk_idx*64}, {chunk_idx*64+64}))")
            try:
                q_nope_fp8_p, _q_nope_scale_e8m0_p = unpack_v4_nope_scale(q_packed)
                # q_nope_fp8_p has shape [total_q, nhead, 448] as fp8 byte view.
                # We need raw fp8 bytes for [token=0, head=h, cols[chunk_base+col_base..]]
                raw_bytes = q_nope_fp8_p[0].view(torch.uint8)  # [nhead, 448]
            except Exception as _e:
                print(f"  could not derive raw fp8: {_e}")
                raw_bytes = None
            # split_output bytes for warp 0: 64 lanes * 16 B = 1024 B = 256 dwords
            dump_u8 = v40_logits.flatten()[:256].contiguous().cpu().view(torch.uint8)
            mismatch = 0
            for lane in range(64):
                g = lane >> 4
                l = lane & 15
                # Per probe layout: bytes [0,8) = staging off:0 = head l cols g*8..g*8+7
                #                   bytes [8,16) = staging off:32 = head l cols g*8+32..g*8+39
                kern_iter0 = dump_u8[lane*16 : lane*16+8].tolist()
                kern_iter1 = dump_u8[lane*16+8 : lane*16+16].tolist()
                if raw_bytes is not None and l < raw_bytes.shape[0]:
                    chunk_col_base = chunk_idx * 64
                    exp_iter0 = raw_bytes[l, chunk_col_base + g*8 : chunk_col_base + g*8 + 8].tolist()
                    exp_iter1 = raw_bytes[l, chunk_col_base + g*8 + 32 : chunk_col_base + g*8 + 40].tolist()
                else:
                    exp_iter0 = [0]*8
                    exp_iter1 = [0]*8
                ok0 = kern_iter0 == exp_iter0
                ok1 = kern_iter1 == exp_iter1
                if not (ok0 and ok1) and mismatch < 6:
                    print(f"  lane={lane:>2} (g={g}, l={l:>2})")
                    print(f"    iter0(off:0)  kern={kern_iter0}  exp={exp_iter0}  match={ok0}")
                    print(f"    iter1(off:32) kern={kern_iter1}  exp={exp_iter1}  match={ok1}")
                if not (ok0 and ok1):
                    mismatch += 1
                elif lane in (0, 1, 16, 17):
                    print(f"  OK lane={lane:>2} (g={g}, l={l:>2})  iter0={kern_iter0[:4]}.. iter1={kern_iter1[:4]}..")
            print(f"  total mismatch lanes: {mismatch}/64")

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
        )
        df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_v4_persistent summary (markdown):\n%s", df_md)
