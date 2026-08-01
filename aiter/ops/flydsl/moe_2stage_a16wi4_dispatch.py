# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production dispatch for the a16wi4 (bf16 A x int4 W) MoE path.

Routes the per_1x32 + int4-weight (i4x2) case of
:func:`aiter.fused_moe.fused_moe_` to the shared FlyDSL a16w-mix kernel package
(:mod:`aiter.ops.flydsl.kernels.moe_2stage_a16wmix`, ``w_dtype="int4"``), unifying
the a16w4 (mxfp4) and a16wi4 (int4) MoE families onto one kernel. On-par with
aiter's own int4 kernel (both numerically correct); this consolidates the family.

Weight layout: the a16wi4 caller contract (shared with aiter's own int4 kernel)
delivers packed-i4x2 weights ``pack_int8_to_packed_int4(shuffle_weight(w, (16,16)))``
and the groupwise bf16 scale in aiter's ``shuffle_scale_for_int4`` ``(E, G//2, N, 2)``
layout. The ported kernel wants a different packing (contiguous 2-nibbles/byte +
standard ``shuffle_weight``) and the ``(E, N, G//2, 2)`` scale layout, so we invert
aiter's packing/shuffle back to the raw signed-int4 weights and ``[E, G, N]`` scale
(pure permutations + nibble repack, verified round-trip), then re-apply the layout
the kernel reads. Keeping the caller contract intact means no ripple in fused_moe's
i4x2 dtype routing / int4 width-adjust.

Tiles: gemm1/gemm2 tiles come from aiter's tuned int4 CSV
(``kimik2_i4_tuned_fmoe.csv``) via ``resolve_a16w4_gemm{1,2}_config`` (aiter's
grid split-K is mapped onto the kernel's intra-block ``k_wave``). block_m follows
``a16wi4_recommend_block_m``: keep the CSV tile_m (16 at the mid-band) and only
bump to 64 at the tok~2048 W1-reuse fill point.
"""

import torch

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    a16wi4_recommend_block_m,
    a16wi4_scale_to_kernel_layout,
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    resolve_a16w4_gemm1_config,
    resolve_a16w4_gemm2_config,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import pack_uint4

_A16WI4_GROUP = 32
_A16WI4_CSV = "kimik2_i4_tuned_fmoe.csv"


def _int4_csv_path():
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(root, "configs", "model_configs", _A16WI4_CSV)
    return p if os.path.isfile(p) else None


# shuffle_weight (16,16) int8 constants (aiter/ops/shuffle.py, non-int4 branch):
# BN=16, BK=IK*2=32, Kel=16//itemsize(int8)=16 -> BK//Kel=2.
_SW_BN, _SW_BK, _SW_KEL = 16, 32, 16


def _unpack_aiter_int4_to_raw(packed_i4x2, rows, K):
    """Invert ``pack_int8_to_packed_int4(shuffle_weight(raw_i8, (16,16)))`` back to the
    raw signed-int4 weight ``[rows, K]`` (int8 container, values in [-7, 7])."""
    flat = packed_i4x2.view(torch.int8).contiguous().view(-1)
    # invert pack_int8_to_packed_int4: 4 bytes -> 8 nibbles, sign-extend 4-bit.
    b = flat.view(-1, 4).to(torch.int16) & 0xFF
    v = torch.empty((b.shape[0], 8), dtype=torch.int16, device=b.device)
    for i in range(4):
        v[:, i] = b[:, i] & 0xF
        v[:, i + 4] = (b[:, i] >> 4) & 0xF
    shuf = torch.where(v >= 8, v - 16, v).to(torch.int8).view(rows, K)
    # invert shuffle_weight(16,16): post-permute 6D -> inverse permute (0,1,4,2,3,5).
    s = shuf.view(1, rows // _SW_BN, K // _SW_BK, _SW_BK // _SW_KEL, _SW_BN, _SW_KEL)
    return s.permute(0, 1, 4, 2, 3, 5).contiguous().view(rows, K)


def _pack_shuffle_int4(raw_i8):
    """Raw signed-int4 ``[rows, K]`` (int8) -> the kernel's packed+shuffled bytes
    (contiguous 2 nibbles/byte + standard ``shuffle_weight``), flattened."""
    u = (raw_i8.to(torch.int16) & 0xF).to(torch.uint8).contiguous()
    packed = pack_uint4(u)  # [rows, K//2] uint8 (low=even K, high=odd K)
    return (
        shuffle_weight(packed.view(torch.float4_e2m1fn_x2))
        .view(torch.uint8)
        .contiguous()
        .view(-1)
    )


def _int4_scale_kernel_layout(scale_aiter_flat, E, N, K):
    """aiter ``shuffle_scale_for_int4`` flat ``(E, G//2, N, 2)`` -> the kernel's
    ``(E, N, G//2, 2)`` bf16-pair layout, flattened.

    First recover the raw ``[E, G, N]`` groupwise scale, then apply the kernel layout.
    """
    G = K // _A16WI4_GROUP
    s = scale_aiter_flat.view(E, G // 2, N, 2)
    # invert shuffle_scale_for_int4: (E,G//2,N,2).permute(0,1,3,2) -> (E,G//2,2,N) = [E,G,N]
    scale_gn = s.permute(0, 1, 3, 2).contiguous().view(E, G, N)
    sc_ng = scale_gn.float().permute(0, 2, 1).contiguous()  # [E,G,N] -> [E,N,G]
    return a16wi4_scale_to_kernel_layout(sc_ng).view(-1).contiguous()


def _g1_config(model_dim, inter_dim, E, topk, tokens):
    """gemm1 (tile_n, tile_k, k_wave) from the int4 CSV; adaptive default otherwise."""
    cfg = resolve_a16w4_gemm1_config(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tokens=tokens,
        csv_path=_int4_csv_path(),
    )
    if cfg is not None:
        return cfg
    tile_n = 128 if inter_dim % 128 == 0 else 64
    tile_k = 128 if model_dim % 128 == 0 else 64
    return {"tile_m": 16, "tile_n": tile_n, "tile_k": tile_k, "k_wave": 1}


def _g2_config(model_dim, inter_dim, E, topk, tokens):
    """gemm2 (tile_n, tile_k) from the int4 CSV; adaptive default otherwise."""
    cfg = resolve_a16w4_gemm2_config(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tokens=tokens,
        csv_path=_int4_csv_path(),
    )
    if cfg is not None:
        return cfg
    tile_n = 128 if model_dim % 128 == 0 else 64
    tile_k = 128 if inter_dim % 128 == 0 else 64
    return {"tile_n": tile_n, "tile_k": tile_k}


def fused_moe_a16wi4_flydsl(
    hidden_states,
    w1,  # [E, 2*inter_dim, model_dim//2] i4x2, aiter packed+shuffled int4
    w2,  # [E, model_dim, inter_dim//2] i4x2, aiter packed+shuffled int4
    topk_weight,
    topk_ids,
    *,
    E,
    model_dim,
    inter_dim,
    topk,
    dtype,
    w1_scale,  # aiter shuffle_scale_for_int4 flat (E, (model_dim//32)//2, 2*inter_dim, 2)
    w2_scale,  # aiter shuffle_scale_for_int4 flat (E, (inter_dim//32)//2, model_dim, 2)
    activation="silu",
    kernel_bench_callable=None,
):
    """Run the shared FlyDSL a16wi4 (bf16 A x int4 W) 2-stage MoE; returns
    [tokens, model_dim]. Inverts aiter's int4 weight/scale layout to the kernel's."""
    dev = hidden_states.device
    tokens = hidden_states.shape[0]
    N_OUT = 2 * inter_dim

    g1c = _g1_config(model_dim, inter_dim, E, topk, tokens)
    g2c = _g2_config(model_dim, inter_dim, E, topk, tokens)

    # block_m: keep the CSV tile_m (16 at the mid-band); only bump to 64 at the
    # tok~2048 W1-reuse fill point. block_m sizes moe_sorting padding, so the SAME
    # BM drives routing AND gemm1/gemm2 tile_m.
    BM = int(g1c["tile_m"])
    if a16wi4_recommend_block_m(tokens, E, topk) >= 64:
        BM = 64

    # aiter packed-i4x2 layout -> raw signed int4 -> the kernel's pack+shuffle.
    w1_raw = _unpack_aiter_int4_to_raw(w1, E * N_OUT, model_dim)
    w2_raw = _unpack_aiter_int4_to_raw(w2, E * model_dim, inter_dim)
    w1_shuf = _pack_shuffle_int4(w1_raw).contiguous()
    w2_shuf = _pack_shuffle_int4(w2_raw).contiguous()
    w1_scale_k = _int4_scale_kernel_layout(w1_scale, E, N_OUT, model_dim)
    w2_scale_k = _int4_scale_kernel_layout(w2_scale, E, model_dim, inter_dim)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, E, model_dim, dtype, BM
    )
    sorted_size = int(sorted_expert_ids.shape[0]) * BM
    cumsum = num_valid_ids.to(torch.int32).contiguous()
    m_indices = sorted_ids.to(torch.int32).contiguous()
    x_bf16 = hidden_states.to(torch.bfloat16).contiguous()
    inter_sorted = torch.zeros(sorted_size, inter_dim, dtype=torch.bfloat16, device=dev)

    g1_tile_n = int(g1c["tile_n"])
    if inter_dim % g1_tile_n != 0:
        g1_tile_n = 128 if inter_dim % 128 == 0 else 64
    g1_tile_k = int(g1c["tile_k"])
    if model_dim % g1_tile_k != 0:
        g1_tile_k = 128 if model_dim % 128 == 0 else 64
    g1_k_wave = int(g1c.get("k_wave", 1))

    def g1():
        flydsl_a16w4_gemm1(
            a_bf16=x_bf16,
            w1_u8=w1_shuf,
            w1_scale_u8=w1_scale_k,
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            m_indices=m_indices,
            inter_sorted_bf16=inter_sorted,
            n_tokens=tokens,
            NE=E,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=BM,
            tile_n=g1_tile_n,
            tile_k=g1_tile_k,
            k_wave=g1_k_wave,
            act=activation,
            w_dtype="int4",
        )

    out_buf = torch.zeros(tokens * model_dim, dtype=torch.bfloat16, device=dev)

    g2_tile_n = int(g2c["tile_n"])
    if model_dim % g2_tile_n != 0:
        g2_tile_n = 128 if model_dim % 128 == 0 else 64
    g2_tile_k = int(g2c["tile_k"])
    if inter_dim % g2_tile_k != 0:
        g2_tile_k = 128 if inter_dim % 128 == 0 else 64

    def g2():
        out_buf.zero_()  # atomic scatter accumulates; zero before each launch
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=inter_sorted,
            w2_u8=w2_shuf,
            w2_scale_u8=w2_scale_k,
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            flat_out=out_buf,
            M_logical=tokens,
            max_sorted=sorted_size,
            NE=E,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=BM,
            tile_n=g2_tile_n,
            tile_k=g2_tile_k,
            w_dtype="int4",
        )

    g1()
    torch.cuda.synchronize()
    g2()
    torch.cuda.synchronize()
    out = out_buf.view(tokens, model_dim)

    if kernel_bench_callable is not None:
        kernel_bench_callable.append(("stage1", g1))
        kernel_bench_callable.append(("stage2", g2))
    return out
