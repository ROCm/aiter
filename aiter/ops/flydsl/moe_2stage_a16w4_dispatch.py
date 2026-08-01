# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production dispatch for the a16w4 (bf16 A x MXFP4 W) SiTUv2 MoE path.

Routes the a16w4 SiTUv2 case of :func:`aiter.fused_moe.fused_moe_` to the ported
FlyDSL kernel package (:mod:`aiter.ops.flydsl.kernels.moe_2stage_a16wmix`), which
is numerically correct where aiter's former ``compile_mixed_moe_gemm{1,2}_a16w4``
was not (that kernel fails aiter's own strict accuracy gate).

Weight layout: the ported kernel wants the STANDARD ``shuffle_weight`` (16,16) /
``e8m0_shuffle`` layout, but the a16w4 caller contract delivers the
gate/up-interleaved (guinterleave / GUGU) layout produced by
``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` -- the same layout shared with
the a8w4 / mxfp8 paths. To keep that shared contract intact, we invert the
guinterleave permutation back to raw quantized bytes and re-apply the standard
shuffle. Both directions are pure byte permutations (verified round-trip), so no
numeric error is introduced.

Tile config: gemm2 tiles come from aiter's tuned CSV via
``resolve_a16w4_gemm2_config`` (on-par-to-faster vs the kernel default). gemm1
keeps the kernel's own tuned default (``tile_n=None`` -> tile_n=128 + the M>=16
TILE_K=128/xcd lever): the CSV gemm1 tiles regress mid/high-M by 1.1x-2.0x, so we
do not take them.
"""

import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    resolve_a16w4_gemm2_config,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

# guinterleave fp4x2 weight layout constants (aiter/ops/shuffle.py shuffle_weight,
# is_guinterleave=True) and n32k4 e8m0 scale constants (shuffle_scale).
_NLANE, _KPACK = 16, 16
_KLANE = 64 // _NLANE  # 4
_S_NPACK, _S_KPACK, _S_NLANE = 2, 2, 16
_S_KLANE = 64 // _S_NLANE  # 4


def _unshuffle_guinterleave_weight(w_gui, gate_up):
    """Invert ``shuffle_weight(is_guinterleave=True)`` -> raw fp4x2 ``[E, N, K_pk]``."""
    b = w_gui.view(torch.uint8).contiguous()
    E, N, K_pk = b.shape
    K0 = K_pk // (_KLANE * _KPACK)
    if gate_up:
        # fwd view(E,2,N0,NLane,K0,KLane,KPack).permute(0,2,1,4,5,3,6); inverse below.
        N0 = (N // 2) // _NLANE
        g = b.reshape(E, N0, 2, K0, _KLANE, _NLANE, _KPACK)
        raw = g.permute(0, 2, 1, 5, 3, 4, 6).contiguous().reshape(E, N, K_pk)
    else:
        # fwd view(E,N0,NLane,K0,KLane,KPack).permute(0,1,3,4,2,5); inverse below.
        N0 = N // _NLANE
        g = b.reshape(E, N0, K0, _KLANE, _NLANE, _KPACK)
        raw = g.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(E, N, K_pk)
    return raw.view(dtypes.fp4x2)


def _unshuffle_guinterleave_scale(s_gui, E, N, Kg, gate_up):
    """Invert ``shuffle_scale_a16w4`` (n32k4) -> raw e8m0 ``[E*N, Kg]``.

    ``shuffle_scale_a16w4`` pads ``Kg = K//32`` up to a multiple of 8; invert on the
    padded width and strip the padding.
    """
    b = s_gui.view(torch.uint8).contiguous()
    Kg_pad = (Kg + 7) // 8 * 8
    K1 = Kg_pad // _S_KPACK // _S_KLANE
    N1 = N // _S_NLANE // _S_NPACK
    g = b.reshape(E, N1, K1, _S_KLANE, _S_NLANE, _S_KPACK, _S_NPACK)
    if gate_up:  # inverse of permute (0,2,4,6,3,5,1)
        raw = g.permute(0, 6, 1, 4, 2, 5, 3).contiguous().reshape(E * N, Kg_pad)
    else:  # inverse of permute (0,1,4,6,3,5,2)
        raw = g.permute(0, 1, 6, 4, 2, 5, 3).contiguous().reshape(E * N, Kg_pad)
    return raw[:, :Kg].contiguous()


def fused_moe_a16w4_flydsl(
    hidden_states,
    w1,  # [E, 2*inter_dim, model_dim//2] fp4x2, guinterleave (gate_up) shuffled
    w2,  # [E, model_dim, inter_dim//2] fp4x2, guinterleave shuffled
    topk_weight,
    topk_ids,
    *,
    E,
    model_dim,
    inter_dim,
    topk,
    dtype,
    w1_scale,  # [E*2*inter_dim, model_dim//32] e8m0, guinterleave (gate_up)
    w2_scale,  # [E*model_dim, inter_dim//32] e8m0, guinterleave
    block_size_M,
    kernel_bench_callable=None,
):
    """Run the ported FlyDSL a16w4 SiTUv2 2-stage MoE; returns [tokens, model_dim].

    SiTUv2 beta/linear_beta are baked at 1.0 in the ported kernel; the caller
    (fused_moe_) gates non-default betas / bias / EP away from this path.
    """
    dev = hidden_states.device
    tokens = hidden_states.shape[0]
    N_OUT = 2 * inter_dim
    BM = int(block_size_M) if block_size_M else 32

    # guinterleave -> raw -> standard shuffle_weight / e8m0_shuffle
    w1_shuf = (
        shuffle_weight(_unshuffle_guinterleave_weight(w1, True))
        .view(torch.uint8)
        .contiguous()
    )
    w2_shuf = (
        shuffle_weight(_unshuffle_guinterleave_weight(w2, False))
        .view(torch.uint8)
        .contiguous()
    )
    w1_scale_1d = (
        e8m0_shuffle(
            _unshuffle_guinterleave_scale(w1_scale, E, N_OUT, model_dim // 32, True)
        )
        .view(torch.uint8)
        .contiguous()
    )
    w2_scale_1d = (
        e8m0_shuffle(
            _unshuffle_guinterleave_scale(
                w2_scale, E, model_dim, inter_dim // 32, False
            )
        )
        .view(torch.uint8)
        .contiguous()
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, E, model_dim, dtype, BM
    )
    sorted_size = int(sorted_expert_ids.shape[0]) * BM
    cumsum = num_valid_ids.to(torch.int32).contiguous()
    m_indices = sorted_ids.to(torch.int32).contiguous()
    x_bf16 = hidden_states.to(torch.bfloat16).contiguous()
    inter_sorted = torch.zeros(sorted_size, inter_dim, dtype=torch.bfloat16, device=dev)

    def g1():
        # gemm1 keeps the kernel's tuned default (tile_n=None); CSV tiles regress here.
        flydsl_a16w4_gemm1(
            a_bf16=x_bf16,
            w1_u8=w1_shuf,
            w1_scale_u8=w1_scale_1d,
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
            tile_n=None,
            act="situv2",
        )

    out_buf = torch.zeros(tokens * model_dim, dtype=torch.bfloat16, device=dev)

    # gemm2 uses aiter's tuned CSV tiles (on-par-to-faster); None -> kernel default.
    g2c = (
        resolve_a16w4_gemm2_config(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tokens=tokens,
        )
        or {}
    )
    g2_tile_n = g2c.get("tile_n", 256)
    if model_dim % g2_tile_n != 0:
        g2_tile_n = 256 if model_dim % 256 == 0 else 128
    g2_tile_k = g2c.get("tile_k", 256)
    if inter_dim % g2_tile_k != 0:
        g2_tile_k = 128 if inter_dim % 128 == 0 else 64

    def g2():
        out_buf.zero_()  # atomic scatter accumulates; zero before each launch
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=inter_sorted,
            w2_u8=w2_shuf,
            w2_scale_u8=w2_scale_1d,
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
            b_nt=g2c.get("b_nt"),
            waves_per_eu=g2c.get("waves_per_eu"),
            xcd_swizzle=g2c.get("xcd_swizzle", 1),
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
