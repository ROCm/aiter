# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import _to_raw as _raw
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int8, T

from .mxmoe_gemm_v2 import (
    BK,
    BN,
    HIDDEN_MAX_DEFAULT,
    INTER_DEFAULT,
    INTER_MAX_DEFAULT,
    MAX_M,
    NE,
    TOPK_DEFAULT,
    gemm1_body_v2,
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
    lds_bytes_for,
)
from .tensor_shim import _run_compiled as run_compiled

__all__ = [
    "compile_gemm1_a4w4_port",
    "compile_gemm2_a4w4_port",
    "gemm1_grid",
    "mxfp4_moe_gemm1",
    "mxfp4_moe_gemm2",
    "select_pipe_config",
    "gemm1_use_nt",
    "gemm2_use_nt",
]


def _get_cu_num() -> int:
    """CU count for the persistent-m fixed grid (env CU_NUM override, else device props)."""
    env = os.environ.get("CU_NUM")
    if env:
        return int(env)
    try:
        import torch

        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    except Exception:
        return 304


# aiter-tuned dispatch: family (model_dim,inter_dim,experts) -> {tokens: (block_m_csv, epilog)}; select_pipe_config clamps block_m to {32,64}, snaps tokens. (aiter-tuned-config-map.md)
_AITER_PIPE_TABLE = {
    # DeepSeekV3 fp4 (7168/256/E257/k9) — reduce-dominant cold at every token; BM clamped <=64.
    (7168, 256, 257): {
        1: (32, "atomic"),
        2: (32, "reduce"),
        4: (32, "reduce"),
        8: (32, "reduce"),
        16: (32, "reduce"),
        32: (32, "reduce"),
        64: (32, "reduce"),
        128: (64, "reduce"),
        256: (32, "reduce"),
        512: (64, "reduce"),
        1024: (64, "reduce"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
    },
    # KimiK2 fp4 (7168/256/E384/k8) — reduce-dominant cold; atomic only <=8 tokens, BM32<=1024 else BM64.
    (7168, 256, 384): {
        1: (32, "atomic"),
        2: (32, "atomic"),
        4: (32, "atomic"),
        8: (32, "atomic"),
        16: (32, "reduce"),
        32: (32, "reduce"),
        64: (32, "reduce"),
        128: (32, "reduce"),
        256: (32, "reduce"),
        512: (32, "reduce"),
        1024: (32, "reduce"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
    },
    # DeepSeekV4 a8w4 (7168/512/E384/k6) — BM32 atomic now beats reduce-TOTAL (gemm2+topk-reduce)
    # at every token after the padding-row atomic gate fix (self-contained, no extra reduce kernel):
    # atomic vs reduce-TOTAL same-tile BM32 M1 -37%, M8 -15%, M64 -7%, M256 -6%, M1024 -9%. The
    # BM64 buckets (>=2048) keep reduce (BM64 atomic not re-validated post-fix). BM32<=1024 else BM64.
    (7168, 512, 384): {
        1: (32, "atomic"),
        2: (32, "atomic"),
        4: (32, "atomic"),
        8: (32, "atomic"),
        16: (32, "atomic"),
        32: (32, "atomic"),
        64: (32, "atomic"),
        128: (32, "atomic"),
        256: (32, "atomic"),
        512: (32, "atomic"),
        1024: (32, "atomic"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
        131072: (64, "reduce"),
    },
    # GPT-OSS (3072/3072/E128/k4) swiglu — atomic-dominant (large inter -> low contention); BM64 atomic fastest.
    (3072, 3072, 128): {
        256: (64, "atomic"),
        512: (64, "atomic"),
        1024: (64, "atomic"),
        2048: (64, "atomic"),
        4096: (64, "atomic"),
        8192: (64, "atomic"),
        16384: (64, "atomic"),
        32768: (64, "atomic"),
    },
}


def _norm_sbm(SBM, BM):
    """Resolve SBM (sort_block_m): None -> SBM==BM (byte-identical); shared by cache key + compile path."""
    return BM if SBM is None else SBM


def _nearest_token_key(tok_map, tokens):
    """Pick the table token bucket nearest (<=) the requested token count; fall back to the min key."""
    keys = sorted(tok_map)
    chosen = keys[0]
    for k in keys:
        if k <= tokens:
            chosen = k
        else:
            break
    return chosen


# stage1-only BM128: family sig -> min tokens for a gemm1 BM128 tile (SBM=128, gemm2 stays <=64); allow_bm128-gated.
_STAGE1_BM128_MIN_TOK = {
    (7168, 256, 257): 4096,  # DeepSeekV3 fp4
    (7168, 256, 384): 4096,  # KimiK2 fp4
    (7168, 512, 384): 4096,  # DeepSeekV4 a8w4
}

# persist (aiter `_persist`): family sig -> min tokens; fp4 only (fp8-A persist known-broken, fail-fast in compile).
_PERSIST_MIN_TOK = {
    (7168, 256, 257): 4096,  # DeepSeekV3 fp4
    (7168, 256, 384): 4096,  # KimiK2 fp4
}

# tiny-M gemm1 BN=64+k_wave=4: family sig -> max tokens for block-count-bound fp4 families (bn64-kw4.md).
_TINYM_BN64_MAX_TOK = {
    (7168, 256, 257): 2,  # DeepSeekV3 fp4
    (7168, 256, 384): 2,  # KimiK2 fp4
}

# fp8-A (a8w4) tiny/low-M gemm1 N-tiling: sig -> {tokens: (bn, k_wave)}. BN=64 gives finer
# N-blocks (16 vs 4 for id=512) to fill the grid when wave-starved; BN64 needs k_wave>=2 and
# the fp8-A path forces cached B (see mxfp4_moe_gemm1). Above the listed tokens the grid
# saturates and k_wave's redundant B-loads lose -> fall back to bn256/kw1.
_A8W4_BN64_TABLE = {
    (7168, 512, 384): {1: (64, 4), 2: (64, 4), 4: (64, 2), 8: (64, 2)},  # DeepSeekV4 a8w4
}

# fp8-A (a8w4) mid-M gemm1 BN=128 (k_wave=1) N-tiling: sig -> {tokens: (bn, k_wave)}. BN=128 gives
# 8 N-blocks (id=512, N_OUT=1024) between BN=256 (4, wave-starved) and BN=64 (16, too fine), matching
# baseline tile_n=128 with no k_wave overhead; unlike BN=64 it is nt-SAFE (validated: no nt miscompile,
# cos=100 over repeated runs), so it uses the normal reuse-aware nt policy. Measured at the PRODUCTION
# per-bucket (BM_stage1, nt): BN128 gives a small consistent win only in the BM32 mid-M dense-grid band
# 512/1024 (-0.5..-1.2% vs BN256, 2-run stable). Elsewhere it is parity (M<=256) or loses: the BM64
# buckets (M>=2048) and M>=8192 re-favour BN=256's coarser tiling (BN128 +5..+20% at BM64), so those
# stay bn256/kw1. Keys on the pipe-table bucket the token snaps to.
_A8W4_BN128_TABLE = {
    (7168, 512, 384): {512: (128, 1), 1024: (128, 1)},  # DeepSeekV4 a8w4 (512/1024-token buckets, BM32)
}

# fp8-A (a8w4) stage1-only BM=16 (stage2 keeps BM32): sig -> max tokens. BM16 halves the A-LDS
# tile + C/VGPR footprint -> the gemm1 kernel drops 168->104 VGPR (measured, 0 spill), i.e. 4 vs 3
# waves/SIMD on gfx950. At the M>=16 HBM-weight-BW-bound floor this lifts achieved BW (occupancy),
# and at tiny-M it halves the per-expert sort-padding floor (E=384, few tokens/expert). Measured vs
# the current BM32 prod config (2-run stable, cos=100.0 nt on/off): M1 -22%, M2 -9%, M4-8 -6..-9%,
# M16-1024 -1.2..-3.1%. Above ~1024 the grid saturates and BM32's larger tile amortizes fixed
# overhead, so stay BM32. BM per K/N tiling (bn,k_wave) is unchanged (BN64/kw for M<=8, else BN256/
# BN128); BM16 only shrinks the M-tile. stage2 tile is untouched (BM16 is gemm1-only here).
_A8W4_STAGE1_BM16_MAX_TOK = {
    (7168, 512, 384): 1024,  # DeepSeekV4 a8w4
}


def select_pipe_config(model_dim, inter_dim, experts, topk, tokens, allow_bm128=False):
    """Host-side per-(shape,token) config picker -> (BM, epilog, bm_stage1, persist, bn, k_wave); unlisted families -> default (32,'atomic',32,False,256,1)."""
    sig = (model_dim, inter_dim, experts)
    fam = _AITER_PIPE_TABLE.get(sig)
    if fam is None:
        return 32, "atomic", 32, False, 256, 1
    bm_csv, epilog = fam[_nearest_token_key(fam, tokens)]
    bm = 64 if bm_csv >= 128 else bm_csv  # stage2 tile caps at 64 (BM128 is stage1-only)
    if bm not in (32, 64):
        bm = 32
    bm_stage1 = bm
    s1_min = _STAGE1_BM128_MIN_TOK.get(sig)
    if allow_bm128 and s1_min is not None and tokens >= s1_min:
        bm_stage1 = 128
    # fp8-A (a8w4) low/mid-M: shrink the stage1 M-tile to BM16 (occupancy + padding win). Disjoint
    # from the BM128 window (BM128 >= 4096 tok, BM16 <= 1024 tok); BM16 only overrides the stage1 tile.
    s1_bm16_max = _A8W4_STAGE1_BM16_MAX_TOK.get(sig)
    if s1_bm16_max is not None and tokens <= s1_bm16_max:
        bm_stage1 = 16
    p_min = _PERSIST_MIN_TOK.get(sig)
    persist = p_min is not None and tokens >= p_min
    a8_bn = _A8W4_BN64_TABLE.get(sig)
    a8_bn128 = _A8W4_BN128_TABLE.get(sig)
    t_max = _TINYM_BN64_MAX_TOK.get(sig)
    # BN128 large-M window keys on the SAME pipe-table bucket the token snaps to (so tokens in the
    # 4096-bucket route to BN128; the 8192+ buckets stay BN256). Checked before the BN64 tiny-M table
    # (ranges are disjoint: BN64 <=8 tok, BN128 >=4096 tok).
    bucket = _nearest_token_key(fam, tokens)
    if a8_bn128 is not None and bucket in a8_bn128:
        bn, k_wave = a8_bn128[bucket]
    elif a8_bn is not None and tokens <= max(a8_bn):
        bn, k_wave = a8_bn[_nearest_token_key(a8_bn, tokens)]
    elif t_max is not None and tokens <= t_max:
        bn, k_wave = 64, 4
    else:
        bn, k_wave = 256, 1
    return bm, epilog, bm_stage1, persist, bn, k_wave


def gemm1_use_nt(experts, topk, tokens, bm_stage1):
    """Reuse-aware gemm1 B-weight cache policy: nt (stream) when <=1 m-block per active expert, else cached."""
    slots = tokens * topk
    active = min(slots, experts)
    if active <= 0:
        return False
    rows_per_expert = (slots + active - 1) // active
    m_blocks_per_expert = (rows_per_expert + bm_stage1 - 1) // bm_stage1
    return m_blocks_per_expert <= 1


def gemm2_use_nt(experts, topk, tokens, bm_stage2):
    """Reuse-aware gemm2 w2 cache policy; identical reuse metric to gemm1, keyed on bm_stage2."""
    return gemm1_use_nt(experts, topk, tokens, bm_stage2)


# ---- gemm1 (up/gate-proj) compile ----
def gemm1_grid(n_tokens, BM=32, NE=NE, TOPK=TOPK_DEFAULT, INTER=INTER_DEFAULT):
    """Host-side grid size (BM=32 active-experts bound)."""
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * ((2 * INTER) // 256)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    HIDDEN_MAX=HIDDEN_MAX_DEFAULT,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    has_pad=False,
):
    # SBM (sort padding unit): None -> SBM==BM (byte-identical); else a multiple of BM.
    SBM = _norm_sbm(SBM, BM)
    b_nontemporal = use_nt  # B-load cache policy: True -> non-temporal, False -> cached
    if BM not in (16, 32, 64, 128):
        raise AssertionError(f"mxfp4_moe_gemm1 supports only BM in {{16,32,64,128}}, got BM={BM}")
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if out_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"out_dtype must be 'fp4' or 'fp8', got {out_dtype!r}")
    if act not in ("silu", "swiglu"):
        raise AssertionError(f"act must be 'silu' or 'swiglu', got {act!r}")

    # Contraction K = model_dim/hidden is a runtime arg (multiple of BK=256, <= HIDDEN_MAX which caps the B-view / A-scale-LDS bounds); K%k_wave / K%BK checks are host-side in mxfp4_moe_gemm1.
    assert HIDDEN_MAX % BK == 0, f"HIDDEN_MAX must be a multiple of {BK}, got {HIDDEN_MAX}"

    # k_wave (intra-block K-slice) guards: k_wave in {1,2,4}, K%k_wave==0 (host-side), per-K-wave A + slabs fit LDS (sized by K_TILES_MAX).
    LDS_LIMIT = 160 * 1024  # gfx950
    if k_wave not in (1, 2, 4):
        raise AssertionError(f"k_wave must be in {{1,2,4}} (4-wave block), got {k_wave}")
    if k_wave > 1:
        if not interleave:
            raise AssertionError("k_wave>1 requires interleave gate mode")
        if (HIDDEN_MAX // BK) % k_wave != 0:
            raise AssertionError(f"HIDDEN_MAX/BK ({HIDDEN_MAX // BK}) must be divisible by k_wave ({k_wave})")
    # BN (fused gate|up N-tile) in {64,128,256}; BN<256 needs interleave and an even NJ>=2 (gate+up pair). BN=64 needs k_wave in {2,4}; BN=128 is legal at k_wave=1 (NJ=2).
    if BN not in (64, 128, 256):
        raise AssertionError(f"BN must be in {{64, 128, 256}}, got {BN}")
    if BN != 256:
        if not interleave:
            raise AssertionError("BN != 256 requires interleave gate mode")
        num_n_waves = 4 // k_wave
        nj = (BN // num_n_waves) // 16
        if nj < 2 or nj % 2 != 0:
            raise AssertionError(
                f"BN={BN} with k_wave={k_wave} (num_n_waves={num_n_waves}) yields NJ={nj}; each N-wave "
                f"needs an even NJ>=2 (gate+up pair). BN=64 needs k_wave in {{2,4}}; BN=128 allows k_wave=1."
            )

    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    lds_bytes = lds_bytes_for(
        HIDDEN_MAX // BK, KH_TILE_A, BM=BM, k_wave=k_wave, BN=BN
    )  # K_TILES_MAX sizes A LDS (inter-independent)
    if lds_bytes > LDS_LIMIT:
        raise AssertionError(f"k_wave LDS {lds_bytes} > {LDS_LIMIT} (BM={BM}, k_wave={k_wave})")

    # Kernel-name tags empty on the default so its name/IR stays byte-identical (each non-default is a distinct variant).
    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    a_tag = "a8" if a_dtype == "fp8" else "a4"
    o_tag = "o8" if out_dtype == "fp8" else "o4"
    act_tag = "" if act == "silu" else f"_swiglu{swiglu_limit:g}"
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    kw_tag = "" if k_wave == 1 else f"_kw{k_wave}"
    bn_tag = "" if BN == 256 else f"_bn{BN}"
    pad_tag = "_pad" if has_pad else ""  # has_pad adds the runtime pad kernarg + weight-OOB pad-skip
    name_suffix = (
        f"hmax{HIDDEN_MAX}_bm{BM}_{bnt_tag}_{gu_tag}_{a_tag}{o_tag}{act_tag}{sbm_tag}{kw_tag}{bn_tag}{pad_tag}_v2"
    )

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _gemm1_kernel_body(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_cumsum,
        arg_sti,
        arg_aqout,
        arg_ascaleout,
        bx_i32,
        lane,
        wave,
        i32_ntok,
        i32_inter,
        i32_hidden,
        i32_kpad,
        i32_npad,
    ):
        # Shared body for both has_pad variants (@flyc.jit -> rewriter recurses the scf if); default passes i32_kpad/i32_npad=0 (no kernarg), folding pad math away.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = cumsum0 // fx.Int32(BM)
        num_n_blocks = (fx.Int32(i32_inter) * fx.Int32(2)) // fx.Int32(BN)  # N_OUT // BN
        bound = total_m_blocks * num_n_blocks
        if fx.Int32(bx_i32) < bound:
            gemm1_body_v2(
                lds_base_i32,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                total_m_blocks,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                BM=BM,
                HIDDEN_MAX=HIDDEN_MAX,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
                a_dtype=a_dtype,
                out_dtype=out_dtype,
                act=act,
                swiglu_limit=swiglu_limit,
                has_pad=has_pad,
                SBM=SBM,
                k_wave=k_wave,
                BN=BN,
            )

    if not has_pad:

        @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
        def gemm1_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_sti: fx.Int64,
            i32_ntok: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
        ):
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = arith.index_cast(T.i32, tx)
            bx_i32 = arith.index_cast(T.i32, bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            _gemm1_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                i32_inter,
                i32_hidden,
                fx.Int32(0),
                fx.Int32(0),
            )

        @flyc.jit
        def launch_gemm1(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_sti: fx.Int64,
            i32_ntok: fx.Int32,
            i32_grid: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
            stream: fx.Stream,
        ):
            grid_x = arith.index_cast(T.index, i32_grid)
            gemm1_kernel(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                i32_ntok,
                i32_inter,
                i32_hidden,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
            ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    else:

        @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
        def gemm1_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_sti: fx.Int64,
            i32_ntok: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
        ):
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = arith.index_cast(T.i32, tx)
            bx_i32 = arith.index_cast(T.i32, bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            _gemm1_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
            )

        @flyc.jit
        def launch_gemm1(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_sti: fx.Int64,
            i32_ntok: fx.Int32,
            i32_grid: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
            arg_aqout: fx.Int64,
            arg_ascaleout: fx.Int64,
            arg_hidden: fx.Int64,
            stream: fx.Stream,
        ):
            grid_x = arith.index_cast(T.index, i32_grid)
            gemm1_kernel(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                i32_ntok,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
            ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1


# ---- gemm2 (down-proj) compile ----
def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """ck_tile GemmSpatiallyLocalTilePartitioner::GetOutputTileIndex: 1D block id -> spatially-local (m_block_idx, n_block_idx). block_1d_id/M0 runtime; N0/group_num/m01 compile-time."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x <= big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)

    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = idx_local // M01_adapt
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    HIDDEN_MAX=HIDDEN_MAX_DEFAULT,
    MAX_M=MAX_M,
    epilog="atomic",
    INTER_MAX=INTER_MAX_DEFAULT,
    a_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    has_pad=False,
    g2_kstages=None,
    g2_bhoist=None,
    g2_ascale_pf=None,
    g2_spart=None,
):
    """Compile gemm2 a4w4 down-proj; epilog 'atomic' (weighted atomic-fadd) or 'reduce' (store into out[token_id*topk+slot]). inter_dim runtime; SBM None -> SBM==BM byte-identical."""
    SBM = _norm_sbm(SBM, BM)
    if BM not in (16, 32, 64, 128) or epilog not in ("atomic", "reduce"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{16,32,64,128}}, epilog in {{'atomic','reduce'}}); "
            f"got (BM={BM}, epilog={epilog})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    use_reduce = epilog == "reduce"
    # gemm2 perf knobs (default ON; env override, explicit arg wins): kstages=2 double-buffers B one tile ahead; bhoist hoists that prefetch above the LDS barrier; ascale_pf prefetches A-scale; spart = SpatiallyLocalTilePartitioner remap GroupNum*100+M01 (402; 0=naive).
    if g2_kstages is None:
        g2_kstages = int(os.environ.get("MXFP4_G2_KSTAGES", "2"))
    if g2_kstages not in (1, 2):
        raise AssertionError(f"g2_kstages must be 1 or 2, got {g2_kstages}")
    if g2_bhoist is None:
        g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_bhoist = bool(g2_bhoist)
    if g2_ascale_pf is None:
        g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    g2_ascale_pf = bool(g2_ascale_pf)
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    if g2_spart > 0 and (g2_group_num < 1 or g2_m01 < 1):
        raise AssertionError(f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}, got {INTER_MAX}"
    is_f8 = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    aStages = 3  # runtime K-loop: triple-buffered A LDS (handles both K_TILES==2 and larger)
    lds_bytes = max(BM * BN * 4, aStages * slot_bytes)
    # N_OUT = model_dim/hidden is a runtime arg (i32_hidden); num_n_blocks = N_OUT//256 is computed runtime in the body/launch (HIDDEN_MAX only caps host checks).
    assert HIDDEN_MAX % BK == 0, f"HIDDEN_MAX must be a multiple of {BK}, got {HIDDEN_MAX}"

    # Kernel-name tags empty on the default so its name/IR stays byte-identical (each variant distinct).
    atag = "_a8" if is_f8 else ""
    etag = "atomic" if not use_reduce else f"reduce_tk{topk}"
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    if persist and is_f8:
        # fp8-A gemm2 persist is a known-broken F2 combo (cos=0 at large M); fail fast.
        raise AssertionError(
            "a8w4/fp8-A gemm2 persist is not supported (known-broken F2 path: cos=0 at large M). "
            "Use persist only with a_dtype='fp4', or run a8w4 with persist=False."
        )
    persist_tag = "" if not persist else f"_persist_cu{cu_num}"
    pad_tag = "_pad" if has_pad else ""  # has_pad adds the runtime pad kernarg + weight-OOB pad-skip
    ks_tag = "" if g2_kstages == 1 else f"_g2ks{g2_kstages}"
    bh_tag = "_bhoist" if g2_bhoist else ""
    apf_tag = "_apf" if g2_ascale_pf else ""
    spart_tag = f"_spart{g2_group_num}x{g2_m01}" if g2_spart > 0 else ""
    tag = f"hmax{HIDDEN_MAX}_imax{INTER_MAX}_bm{BM}{'_nt' if use_nt else ''}_{etag}{atag}{sbm_tag}{persist_tag}{pad_tag}{ks_tag}{bh_tag}{apf_tag}{spart_tag}_v2"
    name = f"gemm2_a4w4_port_{tag}"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _gemm2_kernel_body(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_cumsum,
        arg_stids,
        arg_sweights,
        arg_out,
        bx_i32,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_inter,
        i32_hidden,
        i32_kpad,
        i32_npad,
    ):
        # Shared body for both has_pad variants (@flyc.jit -> rewriter recurses scf if / grid-stride); default passes i32_kpad/i32_npad=0 (no kernarg), folding pad math away.
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(256)  # N_OUT//256 runtime (i32_hidden = model_dim)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)  # A row stride bytes (runtime)
        aq_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(fx.Int32(BM) * k_bytes)
        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_aq)), num_records_bytes=aq_num)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        # Preload the first kStages K-tiles (the streaming prologue).
        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(
                    arg_aq,
                    aq_num,
                    lds_base_i32,
                    slot,
                    slot,
                    m_row0,
                    wave,
                    lane,
                    is_f8,
                    KH_TILE_A,
                    k_bytes,
                    BM=BM,
                )

        # One (m_block, n_block) unit for a synthesized unit_bx; non-persist calls once, persist per m-tile.
        def run_unit(unit_bx):
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                unit_bx,
                lane,
                wave,
                aq_rsrc,
                arg_aq,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                BM=BM,
                use_nt=use_nt,
                INTER_MAX=INTER_MAX,
                aStages=aStages,
                a_dtype=a_dtype,
                use_reduce=use_reduce,
                topk=topk,
                has_pad=has_pad,
                SBM=SBM,
                g2_kstages=g2_kstages,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
            )

        if const_expr(not persist and g2_spart <= 0):
            # One-shot naive linear block->(m,n): issue A->LDS before the cumsum load (latency overlap).
            issue_all_a_loads((bx_i32 // num_n_blocks) * fx.Int32(BM))
            rocdl.sched_barrier(0)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                run_unit(bx_i32)
        elif const_expr(not persist):
            # One-shot with spatial-partitioner remap (g2_spart>0): needs M0=total_m_blocks so cumsum is read FIRST.
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                m_block_idx, n_block_idx = _spart_output_tile_index(
                    bx_i32, total_m_blocks, num_n_blocks, g2_group_num, g2_m01
                )
                unit_bx = m_block_idx * fx.Int32(num_n_blocks) + n_block_idx
                issue_all_a_loads(m_block_idx * fx.Int32(BM))
                rocdl.sched_barrier(0)
                run_unit(unit_bx)
        else:
            # Persistent-m: fixed cu_num*num_n_blocks grid; each block grid-strides m-tiles by cu_num (aiter `_persist`).
            m_tile0 = bx_i32 // fx.Int32(num_n_blocks)
            n_block = bx_i32 - m_tile0 * fx.Int32(num_n_blocks)
            c_stride = fx.Int32(cu_num)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            # ceil((total_m_blocks - m_tile0) / cu_num), clamped to 0 when m_tile0 >= total_m_blocks.
            diff = total_m_blocks - m_tile0
            rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
            n_iters = (rem + c_stride - fx.Int32(1)) // c_stride
            for _it in range(fx.Index(0), ArithValue(_raw(n_iters)).index_cast(T.index), fx.Index(1)):
                m_block = m_tile0 + fx.Int32(_it) * c_stride
                unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                issue_all_a_loads(m_block * fx.Int32(BM))
                rocdl.sched_barrier(0)
                if fx.Int32(m_block) < total_m_blocks:
                    run_unit(unit_bx)

    if not has_pad:

        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def gemm2_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
        ):
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = fx.Int32(tx)
            bx_i32 = fx.Int32(bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            _gemm2_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_out,
                bx_i32,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                fx.Int32(0),
                fx.Int32(0),
            )

        @flyc.jit
        def launch_gemm2(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_grid_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,
            stream: fx.Stream,
        ):
            # i32_max_m_blocks sizes buffer resources; i32_grid_blocks bounds the launch to real m-blocks; num_n_blocks = N_OUT//256 runtime.
            num_n_blocks = arith.index_cast(T.index, _raw(fx.Int32(i32_hidden) // fx.Int32(256)))
            grid_x = arith.index_cast(T.index, i32_grid_blocks) * num_n_blocks
            gemm2_kernel(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                arg_out,
                arg_out_scale,
            ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    else:

        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def gemm2_kernel(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
        ):
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = fx.Int32(tx)
            bx_i32 = fx.Int32(bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            _gemm2_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_out,
                bx_i32,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
            )

        @flyc.jit
        def launch_gemm2(
            arg_aq: fx.Int64,
            arg_ascale: fx.Int64,
            arg_bq: fx.Int64,
            arg_bscale: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_stids: fx.Int64,
            arg_sweights: fx.Int64,
            i32_M: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            i32_grid_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_hidden: fx.Int32,
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,
            stream: fx.Stream,
        ):
            num_n_blocks = arith.index_cast(T.index, _raw(fx.Int32(i32_hidden) // fx.Int32(256)))
            grid_x = arith.index_cast(T.index, i32_grid_blocks) * num_n_blocks
            gemm2_kernel(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                arg_out,
                arg_out_scale,
            ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G1_CACHE = {}
G2_CACHE = {}


def get_g1(
    BM,
    use_nt,
    HIDDEN_MAX,
    interleave,
    a_dtype,
    out_dtype,
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    has_pad=False,
):
    # Cache key = compile-time dims only; inter_dim + model_dim/hidden are runtime (HIDDEN_MAX caps K), NE/topk host-only.
    SBM = _norm_sbm(SBM, BM)
    key = (BM, use_nt, HIDDEN_MAX, interleave, a_dtype, out_dtype, act, swiglu_limit, SBM, k_wave, BN, has_pad)
    launch = G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            HIDDEN_MAX=HIDDEN_MAX,
            interleave=interleave,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            swiglu_limit=swiglu_limit,
            SBM=SBM,
            k_wave=k_wave,
            BN=BN,
            has_pad=has_pad,
        )
        G1_CACHE[key] = launch
    return launch


def get_g2(
    BM, use_nt, HIDDEN_MAX, epilog, INTER_MAX, a_dtype, topk=1, SBM=None, persist=False, cu_num=0, has_pad=False
):
    # Cache key = compile-time dims; inter_dim + model_dim/hidden runtime (INTER_MAX/HIDDEN_MAX cap them), topk keyed only for reduce.
    SBM = _norm_sbm(SBM, BM)
    topk_key = topk if epilog == "reduce" else 1
    cu_key = cu_num if persist else 0
    # gemm2 perf knobs enter the key; defaults ON (env override), matching compile_gemm2_a4w4_port.
    g2_kstages = int(os.environ.get("MXFP4_G2_KSTAGES", "2"))
    g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    key = (
        BM,
        use_nt,
        HIDDEN_MAX,
        epilog,
        INTER_MAX,
        a_dtype,
        topk_key,
        SBM,
        persist,
        cu_key,
        has_pad,
        g2_kstages,
        g2_bhoist,
        g2_ascale_pf,
        g2_spart,
    )
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            HIDDEN_MAX=HIDDEN_MAX,
            epilog=epilog,
            INTER_MAX=INTER_MAX,
            a_dtype=a_dtype,
            topk=topk_key,
            SBM=SBM,
            persist=persist,
            cu_num=cu_key,
            has_pad=has_pad,
            g2_kstages=g2_kstages,
            g2_bhoist=g2_bhoist,
            g2_ascale_pf=g2_ascale_pf,
            g2_spart=g2_spart,
        )
        G2_CACHE[key] = launch
    return launch


def mxfp4_moe_gemm1(
    *,
    a_quant,
    a_scale_sorted_shuffled,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    hidden_states,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM=32,
    use_nt=False,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    n_sorted_padded=None,
    model_dim_pad=0,
    inter_dim_pad=0,
    stream=None,
):
    """Stage-1 up/gate gemm: A_q x w1 -> inter (packed MXFP4/MXFP8, sorted). model_dim_pad/inter_dim_pad>0 enable has_pad pad-skip (both 0 -> byte-identical); n_sorted_padded bounds the grid (None -> worst case)."""
    import torch

    has_pad = model_dim_pad > 0 or inter_dim_pad > 0
    # model_dim/hidden (contraction K) is a runtime arg; validate it host-side (K is not compile-time).
    if D_HIDDEN % BK != 0:
        raise AssertionError(f"D_HIDDEN (K) must be a multiple of {BK}, got {D_HIDDEN}")
    if D_HIDDEN > HIDDEN_MAX_DEFAULT:
        raise AssertionError(f"D_HIDDEN ({D_HIDDEN}) exceeds compile cap HIDDEN_MAX ({HIDDEN_MAX_DEFAULT})")
    if k_wave > 1 and (D_HIDDEN // BK) % k_wave != 0:
        raise AssertionError(f"D_HIDDEN/BK ({D_HIDDEN // BK}) must be divisible by k_wave ({k_wave})")
    # fp8-A + BN<256: the non-temporal B-weight load miscompiles under the tight BN64 register
    # allocation (WAR hazard: nt loads complete out-of-order vs the vmcnt the pipeliner assumes,
    # so a prefetch clobbers a still-live B fragment -> nondeterministic zero accm on whole
    # n_blocks). Cached B is correct AND faster here (weights reuse across the K-waves), so force
    # it. fp4 BN64 is unaffected (uses fx.gemm, different reg schedule); BN256 has reg headroom.
    if a_dtype == "fp8" and BN == 64 and use_nt:
        use_nt = False
    launch = get_g1(
        BM,
        use_nt,
        HIDDEN_MAX_DEFAULT,
        interleave,
        a_dtype,
        out_dtype,
        act,
        swiglu_limit,
        SBM=SBM,
        k_wave=k_wave,
        BN=BN,
        has_pad=has_pad,
    )
    sbm = _norm_sbm(SBM, BM)
    num_n_blocks = (2 * D_INTER) // BN
    if n_sorted_padded is None:
        # E-based worst-case grid: sort padding per SBM, compute grid in BM blocks (SBM==BM -> gemm1_grid).
        active = min(n_tokens * topk, NE)
        padded_rows = ((n_tokens * topk + active * (sbm - 1) + sbm - 1) // sbm) * sbm
        grid = (padded_rows // BM) * num_n_blocks
    else:
        grid = (n_sorted_padded // BM) * num_n_blocks
    # has_pad threads the runtime i32_kpad (model_dim_pad) + i32_npad (inter_dim_pad) after i32_inter.
    pad_args = (int(model_dim_pad), int(inter_dim_pad)) if has_pad else ()
    run_compiled(
        launch,
        a_quant.data_ptr(),
        a_scale_sorted_shuffled.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        n_tokens,
        grid,
        D_INTER,
        D_HIDDEN,
        *pad_args,
        inter_sorted_quant.data_ptr(),
        inter_sorted_shuffled_scale.data_ptr(),
        hidden_states.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_quant, inter_sorted_shuffled_scale


def mxfp4_moe_gemm2(
    *,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM=32,
    use_nt=False,
    a_dtype="fp4",
    epilog="atomic",
    SBM=None,
    persist=False,
    cu_num=0,
    n_sorted_padded=None,
    inter_dim_pad=0,
    model_dim_pad=0,
    stream=None,
):
    """Stage-2 down-proj gemm; epilog 'atomic' (weighted atomic.fadd) or 'reduce' (store into out[token_id*topk+slot]). inter_dim_pad/model_dim_pad>0 enable has_pad pad-skip (both 0 -> byte-identical); persist = fixed cu_num m-slot grid (default OFF)."""
    import torch

    if persist and cu_num <= 0:
        cu_num = _get_cu_num()
    has_pad = inter_dim_pad > 0 or model_dim_pad > 0
    # model_dim/hidden (gemm2 N-output) is a runtime arg; validate host-side (not compile-time).
    if D_HIDDEN % 256 != 0:
        raise AssertionError(f"D_HIDDEN (N_OUT) must be a multiple of 256, got {D_HIDDEN}")
    if D_HIDDEN > HIDDEN_MAX_DEFAULT:
        raise AssertionError(f"D_HIDDEN ({D_HIDDEN}) exceeds compile cap HIDDEN_MAX ({HIDDEN_MAX_DEFAULT})")
    launch = get_g2(
        BM,
        use_nt,
        HIDDEN_MAX_DEFAULT,
        epilog,
        INTER_MAX_DEFAULT,
        a_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
        has_pad=has_pad,
    )
    if D_INTER > INTER_MAX_DEFAULT:
        raise AssertionError(f"D_INTER ({D_INTER}) exceeds compile cap INTER_MAX ({INTER_MAX_DEFAULT})")
    max_m_blocks = (max_sorted + BM - 1) // BM
    if persist:
        # Fixed grid: cu_num m-slots; each block loops over its m-tiles.
        grid_blocks = cu_num
    else:
        grid_blocks = max_m_blocks if n_sorted_padded is None else (n_sorted_padded // BM)
    out_scale = out  # unused by the atomic epilog; any valid device ptr is fine
    # has_pad threads the runtime i32_kpad (inter_dim_pad) + i32_npad (model_dim_pad) after i32_inter.
    pad_args = (int(inter_dim_pad), int(model_dim_pad)) if has_pad else ()
    run_compiled(
        launch,
        inter_sorted_quant.data_ptr(),
        inter_sorted_shuffled_scale.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        M_logical,
        max_m_blocks,
        grid_blocks,
        D_INTER,
        D_HIDDEN,
        *pad_args,
        out.data_ptr(),
        out_scale.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
