# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production dispatch for the a16w4 (bf16 A x MXFP4 W) SiTUv2 SEPARATED MoE path.

Routes the a16w4 SiTUv2 SEPARATED case of :func:`aiter.fused_moe.fused_moe_` to
the ported FlyDSL kernel package
(:mod:`aiter.ops.flydsl.kernels.moe_2stage_a16wmix`), which is numerically
correct where aiter's previous ``compile_mixed_moe_gemm{1,2}_a16w4`` kernel was
not (the old kernel fails aiter's own strict accuracy gate).

The ported kernel expects the STANDARD ``shuffle_weight`` (16,16) weight layout
and ``e8m0_shuffle`` scale layout, whereas the a16w4 fused-MoE caller contract
delivers the gate/up-interleaved (guinterleave / GUGU) layout produced by
``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` -- the same layout shared with
the a8w4 / mxfp8 paths. To keep that shared caller contract intact, this module
inverts the incoming guinterleave permutation back to the raw quantized bytes and
re-applies the standard shuffle the ported kernel wants. Both directions are pure
byte permutations (verified round-trip), so no numeric error is introduced.
"""

import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    pick_a16w4_config,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

# guinterleave (shuffle_weight is_guinterleave=True) fp4x2 weight constants,
# mirroring aiter/ops/shuffle.py shuffle_weight lines 174-190.
_NLANE = 16
_KPACK = 16
_KLANE = 64 // _NLANE  # 4

# guinterleave n32k4 scale constants, mirroring aiter/ops/shuffle.py
# shuffle_scale lines 383-409 (MXFP4 e8m0).
_S_NPACK = 2
_S_KPACK = 2
_S_NLANE = 16
_S_KLANE = 64 // _S_NLANE  # 4


def _unshuffle_guinterleave_weight(w_gui, gate_up):
    """Invert ``shuffle_weight(is_guinterleave=True)`` back to raw fp4x2 bytes.

    ``w_gui``: guinterleave fp4x2 weight ``[E, N, K_pk]`` (uint8-viewable).
    Returns the raw ``[E, N, K_pk]`` fp4x2 tensor (contiguous, standard-order).
    """
    b = w_gui.view(torch.uint8).contiguous()
    E, N, K_pk = b.shape
    K0 = K_pk // (_KLANE * _KPACK)
    if gate_up:
        # fwd: view(E,2,N0,NLane,K0,KLane,KPack).permute(0,2,1,4,5,3,6)
        # fwd out 7D shape: [E, N0, 2, K0, KLane, NLane, KPack]
        N0 = (N // 2) // _NLANE
        g = b.reshape(E, N0, 2, K0, _KLANE, _NLANE, _KPACK)
        # inverse of permute (0,2,1,4,5,3,6) is (0,2,1,5,3,4,6)
        raw = g.permute(0, 2, 1, 5, 3, 4, 6).contiguous().reshape(E, N, K_pk)
    else:
        # fwd: view(E,N0,NLane,K0,KLane,KPack).permute(0,1,3,4,2,5)
        # fwd out 6D shape: [E, N0, K0, KLane, NLane, KPack]
        N0 = N // _NLANE
        g = b.reshape(E, N0, K0, _KLANE, _NLANE, _KPACK)
        # inverse of permute (0,1,3,4,2,5) is (0,1,4,2,3,5)
        raw = g.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(E, N, K_pk)
    return raw.view(dtypes.fp4x2)


def _unshuffle_guinterleave_scale(s_gui, E, N, Kg, gate_up):
    """Invert ``shuffle_scale_a16w4`` (n32k4 guinterleave) back to raw e8m0.

    ``s_gui``: guinterleave e8m0 scale, ``[E*N, Kg_padded]`` (uint8-viewable),
    where ``Kg = K//32`` and ``Kg_padded = round_up(Kg, 8)`` matches the host-side
    padding ``shuffle_scale_a16w4`` applies when ``Kg % 8 != 0``. Returns the raw
    ``[E*N, Kg]`` uint8 scale (standard row-major order, padding stripped).
    """
    b = s_gui.view(torch.uint8).contiguous()
    Kg_pad = (Kg + 7) // 8 * 8  # host pads K//32 up to a multiple of 8
    K1 = Kg_pad // _S_KPACK // _S_KLANE
    N1 = N // _S_NLANE // _S_NPACK
    # fwd 7D out: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
    g = b.reshape(E, N1, K1, _S_KLANE, _S_NLANE, _S_KPACK, _S_NPACK)
    if gate_up:
        # inverse of permute (0,2,4,6,3,5,1) is (0,6,1,4,2,5,3)
        raw = g.permute(0, 6, 1, 4, 2, 5, 3).contiguous().reshape(E * N, Kg_pad)
    else:
        # inverse of permute (0,1,4,6,3,5,2) is (0,1,6,4,2,5,3)
        raw = g.permute(0, 1, 6, 4, 2, 5, 3).contiguous().reshape(E * N, Kg_pad)
    return raw[:, :Kg].contiguous()  # strip K//32 padding


def _cfg(model_dim, inter_dim, E, topk, tokens, stage):
    """aiter tuned tile config for (shape, token, stage); {} if none / no CSV."""
    try:
        c = pick_a16w4_config(
            None,  # host auto-discovers aiter's tuned fp4 fmoe CSV
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tokens=tokens,
            stage=stage,
        )
    except Exception:  # noqa: BLE001 - CSV lookup failure -> host defaults
        c = None
    return c or {}


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
    """Run the ported FlyDSL a16w4 (bf16 A x MXFP4 W) SiTUv2 SEPARATED 2-stage MoE.

    Consumes the guinterleave (a8w4/mxfp8-shared) weight/scale layout the caller
    provides, converts it to the ported kernel's standard shuffle_weight/e8m0
    layout, and returns the final ``[tokens, model_dim]`` output. SiTUv2
    beta/linear_beta are baked at 1.0 in the ported kernel; the caller gates
    non-default betas away from this path.
    """
    dev = hidden_states.device
    tokens = hidden_states.shape[0]
    N_OUT = 2 * inter_dim

    g1c = _cfg(model_dim, inter_dim, E, topk, tokens, 1)
    g2c = _cfg(model_dim, inter_dim, E, topk, tokens, 2)
    BM = int(block_size_M) if block_size_M else int(g1c.get("tile_m", 32))

    # guinterleave -> raw -> standard shuffle_weight (what the ported kernel wants)
    w1_raw = _unshuffle_guinterleave_weight(w1, gate_up=True)
    w2_raw = _unshuffle_guinterleave_weight(w2, gate_up=False)
    w1_shuf = shuffle_weight(w1_raw).view(torch.uint8).contiguous()
    w2_shuf = shuffle_weight(w2_raw).view(torch.uint8).contiguous()

    # guinterleave n32k4 scale -> raw e8m0 -> e8m0_shuffle
    w1_scale_raw = _unshuffle_guinterleave_scale(
        w1_scale, E, N_OUT, model_dim // 32, gate_up=True
    )
    w2_scale_raw = _unshuffle_guinterleave_scale(
        w2_scale, E, model_dim, inter_dim // 32, gate_up=False
    )
    w1_scale_1d = e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
    w2_scale_1d = e8m0_shuffle(w2_scale_raw).view(torch.uint8).contiguous()

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, E, model_dim, dtype, BM
    )
    sorted_size = int(sorted_expert_ids.shape[0]) * BM
    cumsum = num_valid_ids.to(torch.int32).contiguous()
    m_indices = sorted_ids.to(torch.int32).contiguous()
    x_bf16 = hidden_states.to(torch.bfloat16).contiguous()

    inter_sorted = torch.zeros(sorted_size, inter_dim, dtype=torch.bfloat16, device=dev)

    def g1():
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
            tile_n=None,  # host picks the tuned default (mxfp4 occupancy fix)
            act="situv2",
        )

    out_buf = torch.zeros(tokens * model_dim, dtype=torch.bfloat16, device=dev)

    # gemm2: tile_n tiles model_dim (N); tile_k tiles inter_dim (K contraction).
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
