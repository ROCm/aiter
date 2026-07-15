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
    INTER_MAX_DEFAULT,
    NE,
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
)
from .tensor_shim import _run_compiled as run_compiled

__all__ = [
    "compile_gemm2_a4w4_port",
    "mxfp4_moe_gemm2",
    "select_gemm2_config",
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


def _norm_sbm(SBM, BM):
    """Resolve SBM (sort_block_m): None -> SBM==BM."""
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


# gemm2-only tuned overrides: sig (model_dim,inter_dim,experts) -> {tokens: (bm_s2, epilog,
# persist, use_nt)}. Only gemm2-side knobs are stored here.
_GEMM2_TUNED_TABLE = {
    # NOTE: one inner dict per sig; do NOT repeat the sig key (dup keys overwrite).
    (6144, 512, 257): {  # (bm_s2, epilog, persist, use_nt)
        # 1: (32, 'reduce', False, True),  # 4.985us ok100% sbm32
        # 2: (32, 'atomic', False, False),  # 5.946us ok100% sbm32
        # 4: (32, 'reduce', False, True),  # 12.726us ok100% sbm32
        # 8: (32, 'reduce', False, False),  # 20.457us ok100% sbm32
        # 16: (32, 'atomic', False, False),  # 36.226us ok100% sbm32
        # 32: (32, 'atomic', False, True),  # 64.799us ok100% sbm32
        # 64: (64, 'reduce', False, True),  # 83.815us ok100% sbm64
        # 128: (32, 'atomic', False, True),  # 70.371us ok100% sbm32
        # 256: (64, 'atomic', False, True),  # 90.692us ok100% sbm64
        # 512: (32, 'atomic', True, True),  # 83.007us ok100% sbm32
        # 1024: (64, 'reduce', True, True),  # 104.510us ok100% sbm64
        # 2048: (64, 'reduce', True, False),  # 170.830us ok100% sbm128
        # 4096: (64, 'reduce', True, False),  # 248.961us ok100% sbm64
        # 8192: (64, 'reduce', True, False),  # 410.807us ok100% sbm64
        # 16384: (64, 'reduce', True, False),  # 789.103us ok100% sbm128
        # 32768: (64, 'reduce', True, False),  # 1484.276us ok100% sbm128
        1: (32, 'atomic', False, False),  # 4.318us ok100% sbm32
        2: (32, 'atomic', False, False),  # 6.305us ok100% sbm32
        4: (32, 'atomic', False, True),  # 12.615us ok100% sbm32
        8: (32, 'atomic', False, False),  # 20.526us ok100% sbm32
        16: (32, 'atomic', False, False),  # 36.722us ok100% sbm32
        32: (32, 'atomic', False, True),  # 65.429us ok100% sbm32
        64: (32, 'atomic', False, True),  # 68.020us ok100% sbm32
        128: (32, 'atomic', False, True),  # 68.822us ok100% sbm32
        256: (64, 'reduce', False, True),  # 89.203us ok100% sbm64
        512: (32, 'atomic', False, True),  # 80.653us ok100% sbm32
        1024: (64, 'atomic', False, True),  # 117.670us ok100% sbm64
        2048: (64, 'reduce', True, False),  # 202.098us ok100% sbm128
        4096: (64, 'reduce', True, False),  # 311.413us ok100% sbm64
        8192: (64, 'reduce', True, False),  # 539.506us ok100% sbm64
        16384: (64, 'reduce', True, False),  # 1089.024us ok100% sbm128
        32768: (64, 'reduce', True, False),  # 1999.693us ok100% sbm128
    },
}


def _default_gemm2_use_nt(experts, topk, tokens, bm_stage2):
    """Reuse-aware gemm2 w2 cache policy."""
    slots = tokens * topk
    active = min(slots, experts)
    if active <= 0:
        return False
    rows_per_expert = (slots + active - 1) // active
    m_blocks_per_expert = (rows_per_expert + bm_stage2 - 1) // bm_stage2
    return m_blocks_per_expert <= 1


def select_gemm2_config(model_dim, inter_dim, experts, topk, tokens):
    """gemm2 deploy config -> (bm_s2, epilog, persist, use_nt)."""
    sig = (model_dim, inter_dim, experts)
    tuned = _GEMM2_TUNED_TABLE.get(sig)
    if tuned is not None:
        return tuned[_nearest_token_key(tuned, tokens)]
    bm, epilog, persist = 32, "atomic", False
    return bm, epilog, persist, _default_gemm2_use_nt(experts, topk, tokens, bm)


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
    g2_bf16_lds=None,
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
    if g2_bf16_lds is None:
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", "1") == "1" and use_reduce
    g2_bf16_lds = bool(g2_bf16_lds) and use_reduce
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    aStages = 2 if g2_bf16_lds else 3
    c_lds_bytes = BM * BN * (2 if g2_bf16_lds else 4)
    lds_bytes = max(c_lds_bytes, aStages * slot_bytes)
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
    bf16lds_tag = "_bf16lds" if g2_bf16_lds else ""
    tag = f"hmax{HIDDEN_MAX}_imax{INTER_MAX}_bm{BM}{'_nt' if use_nt else ''}_{etag}{atag}{sbm_tag}{persist_tag}{pad_tag}{ks_tag}{bh_tag}{apf_tag}{spart_tag}{bf16lds_tag}_v2"
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
                g2_bf16_lds=g2_bf16_lds,
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
G2_CACHE = {}


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
    g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", "1") == "1" and epilog == "reduce"
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
        g2_bf16_lds,
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
            g2_bf16_lds=g2_bf16_lds,
        )
        G2_CACHE[key] = launch
    return launch


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
    inter_max = 512 if D_INTER == 512 else INTER_MAX_DEFAULT
    launch = get_g2(
        BM,
        use_nt,
        HIDDEN_MAX_DEFAULT,
        epilog,
        inter_max,
        a_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
        has_pad=has_pad,
    )
    if D_INTER > inter_max:
        raise AssertionError(f"D_INTER ({D_INTER}) exceeds compile cap INTER_MAX ({inter_max})")
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
