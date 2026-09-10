# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from aiter.jit.utils.chip_info import get_cu_num

from .mxfp4_gemm_common import (
    kStages,
)
from .mxmoe_g2_atoms import issue_a_load_lds_dt
from .mxmoe_g2_kloop import gemm2_body_v2
from .mxmoe_g2_scheduler import g2_launch_grid_x, schedule_g2_tiles
from .tensor_shim import _run_compiled as run_compiled

__all__ = [
    "compile_gemm2_a4w4_port",
    "mxfp4_moe_gemm2",
]


def _norm_sbm(SBM, BM):
    """Resolve SBM (sort_block_m): None -> SBM==BM."""
    return BM if SBM is None else SBM


def _active_m_blocks_upper_bound(M_logical, topk, NE, BM, SBM):
    """Host-side upper bound for non-persistent GEMM2 M tiles."""
    routes = M_logical * topk
    active_experts = min(routes, NE)
    sort_blocks = (routes + active_experts * (SBM - 1) + SBM - 1) // SBM
    return sort_blocks * (SBM // BM)


def _validate_v2_gemm2_dtypes(a_dtype: str, b_dtype: str) -> None:
    if (a_dtype, b_dtype) not in {
        ("fp4", "fp4"),
        ("fp8", "fp4"),
        ("fp8", "fp8"),
    }:
        raise AssertionError(f"unsupported v2 GEMM2 dtype pair {(a_dtype, b_dtype)!r}")


# ---- gemm2 (down-proj) compile ----
def _pick_epi_lanes(BM, BN, route_out_fp8, g2_scale_blk, nthreads=256):
    if not route_out_fp8:
        return None
    order = (32, 16, 8) if BN >= 512 else (16, 8, 32)
    for lanes in order:
        epi_rows = nthreads // lanes
        route_vec = BN // lanes
        if BM % epi_rows or BN % lanes or route_vec % 4:
            continue
        if g2_scale_blk in (route_vec, 2 * route_vec, 4 * route_vec):
            return lanes
    return None


def compile_gemm2_a4w4_port(
    BM=32,
    BN=256,
    BK=256,
    use_nt=False,
    D_HIDDEN=8192,
    epilog="atomic",
    D_INTER=8192,
    a_dtype="fp4",
    b_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    g2_bhoist=None,
    g2_ascale_pf=None,
    g2_spart=None,
    g2_bf16_lds=None,
    g2_kstatic=False,
    out_dtype="bf16",
    enable_bias=False,
):
    """Compile gemm2 a4w4 down-proj.

    epilog:
      atomic  — weighted atomic-fadd into out[token]
      reduce  — store out[token*topk+slot], host ``_run_moe_reduction``
      scatter — store flat_out[sorted_row], host ``mxfp4_moe_scatter_reduce``
    D_HIDDEN/D_INTER compile-time; SBM None -> SBM==BM.
    """
    SBM = _norm_sbm(SBM, BM)
    if BM not in (16, 32, 64, 128) or epilog not in ("atomic", "reduce", "scatter"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{16,32,64,128}}, "
            f"epilog in {{'atomic','reduce','scatter'}}); "
            f"got (BM={BM}, epilog={epilog})"
        )
    if BN not in (128, 256, 512) or BK not in (128, 256):
        raise AssertionError(
            "mxfp4_moe_gemm2 supports only "
            f"(BN in {{128,256,512}}, BK in {{128,256}}); got (BN={BN}, BK={BK})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    use_reduce = epilog == "reduce"
    use_scatter = epilog == "scatter"
    out_dtype = str(out_dtype).strip().lower()
    if out_dtype not in ("bf16", "fp8"):
        raise AssertionError(f"out_dtype must be 'bf16' or 'fp8', got {out_dtype!r}")
    route_out_fp8 = out_dtype == "fp8"
    if route_out_fp8 and not use_reduce:
        raise AssertionError("out_dtype='fp8' is supported only with epilog='reduce'")
    g2_kstatic = bool(g2_kstatic)
    if g2_kstatic and route_out_fp8:
        from .mxfp4_gemm_common import FP8OUT_PITCH_ALIGN, FP8OUT_SCALE_BLK

        g2_defer_weight = True
        g2_out_pitch_align = FP8OUT_PITCH_ALIGN
        g2_scale_blk = FP8OUT_SCALE_BLK
    else:
        g2_defer_weight = False
        g2_out_pitch_align = 0
        g2_scale_blk = 8
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
        raise AssertionError(
            f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)"
        )
    _validate_v2_gemm2_dtypes(a_dtype, b_dtype)
    assert D_INTER % BK == 0, f"D_INTER must be a multiple of {BK}, got {D_INTER}"
    is_f8 = a_dtype == "fp8"
    if g2_bf16_lds is None:
        default_bf16_lds = "1" if g2_kstatic else "0"
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", default_bf16_lds) == "1"
    g2_bf16_lds = bool(g2_bf16_lds)
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    if use_scatter:
        aStages = kStages
        c_lds_bytes = 0
    else:
        c_lds_bytes = BM * BN * (2 if g2_bf16_lds else 4)
        # aStages must exceed kStages: the K-loop ds_reads slot kt%aStages then
        # prefetches kt+kStages into (kt+kStages)%aStages, so equal counts make that
        # DMA rewrite the slot being read (cross-wave: waves DMA their own rows but
        # ds_read all BM rows). Only bump to 3 when the C region already covers it,
        # so lds_bytes and occupancy are unchanged; otherwise keep 2 and let
        # a_slot_alias fence the prefetch instead.
        aStages = 3 if (not g2_bf16_lds or 3 * slot_bytes <= c_lds_bytes) else 2
    a_slot_alias = aStages <= kStages
    lds_bytes = max(c_lds_bytes, aStages * slot_bytes)
    K_TILES = D_INTER // BK
    g2_apre = g2_kstatic and aStages >= K_TILES
    a_preload = min(aStages, K_TILES) if g2_apre else kStages
    assert D_HIDDEN % BN == 0, f"D_HIDDEN must be a multiple of {BN}, got {D_HIDDEN}"

    # Kernel-name tags empty on the default so its name/IR stays byte-identical (each variant distinct).
    atag = "_a8" if is_f8 else ""
    btag = "_w8" if b_dtype == "fp8" else ""
    if use_scatter:
        etag = "scatter"
    elif use_reduce:
        etag = f"reduce_tk{topk}"
    else:
        etag = "atomic"
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    # Scatter selects persist-flat. persist still means persist-M; persist-M will
    # be removed later and persist will then select persist-flat.
    persist_flat = epilog == "scatter"
    persist = bool(persist or persist_flat)
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    if persist and is_f8 and not persist_flat:
        # fp8-A gemm2 persist-M is a known-broken F2 combo (cos=0 at large M); fail fast.
        # persist_flat remaps (M,N) tiles and is allowed for fp8-A.
        raise AssertionError(
            "a8w4/fp8-A gemm2 persist is not supported (known-broken F2 path: cos=0 at large M). "
            "Use persist only with a_dtype='fp4', persist_flat, or run a8w4 with persist=False."
        )
    persist_tag = "" if not persist else f"_persist_cu{cu_num}"
    persist_tag += "_pflat" if persist_flat else ""
    bh_tag = "_bhoist" if g2_bhoist else ""
    apf_tag = "_apf" if g2_ascale_pf else ""
    spart_tag = f"_spart{g2_group_num}x{g2_m01}" if g2_spart > 0 else ""
    bf16lds_tag = "_bf16lds" if g2_bf16_lds else ""
    dw_tag = "_dw" if g2_defer_weight else ""
    kst_tag = "_kst" if g2_kstatic else ""
    pitch_tag = (
        f"_pa{g2_out_pitch_align}" if (route_out_fp8 and g2_out_pitch_align) else ""
    )
    sblk_tag = f"_sblk{g2_scale_blk}" if (route_out_fp8 and g2_scale_blk != 8) else ""
    out_tag = "_fp8out" if route_out_fp8 else ""
    tile_tag = "" if (BN, BK) == (256, 256) else f"_bn{BN}_bk{BK}"
    bias_tag = "_bias" if enable_bias else ""
    g2_epi_lanes = _pick_epi_lanes(BM, BN, route_out_fp8, g2_scale_blk)
    tag = f"h{D_HIDDEN}_i{D_INTER}_bm{BM}{tile_tag}{'_nt' if use_nt else ''}_{etag}{atag}{btag}{sbm_tag}{persist_tag}{bh_tag}{apf_tag}{spart_tag}{bf16lds_tag}{dw_tag}{kst_tag}{pitch_tag}{sblk_tag}{out_tag}{bias_tag}_v2_biasabi7"
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
        arg_bias,
        arg_out,
        bx_i32,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_grid_blocks,
    ):
        num_n_blocks = D_HIDDEN // BN
        k_bytes = D_INTER // (1 if is_f8 else 2)
        aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * k_bytes)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(a_preload):
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

        def run_unit(unit_bx, mn_idx=None):
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_bias,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                unit_bx,
                lane,
                wave,
                arg_aq,
                BM=BM,
                BN=BN,
                BK=BK,
                use_nt=use_nt,
                D_INTER=D_INTER,
                D_HIDDEN=D_HIDDEN,
                g2_kstatic=g2_kstatic,
                aStages=aStages,
                a_slot_alias=a_slot_alias,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                use_reduce=use_reduce,
                topk=topk,
                SBM=SBM,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
                g2_bf16_lds=g2_bf16_lds,
                g2_defer_weight=g2_defer_weight,
                g2_out_pitch_align=g2_out_pitch_align,
                g2_scale_blk=g2_scale_blk,
                route_out_fp8=route_out_fp8,
                g2_epi_lanes=g2_epi_lanes,
                g2_apre=g2_apre,
                enable_bias=enable_bias,
                nonatomic=use_scatter,
                mn_idx=mn_idx,
            )

        def issue_and_run(unit_bx, m_block, mn_idx=None):
            issue_all_a_loads(m_block * fx.Int32(BM))
            rocdl.sched_barrier(0)
            run_unit(unit_bx, mn_idx)

        schedule_g2_tiles(
            bx_i32,
            arg_cumsum,
            num_n_blocks,
            persist_flat=persist_flat,
            persist=persist,
            g2_spart=g2_spart,
            BM=BM,
            cu_num=cu_num,
            g2_group_num=g2_group_num,
            g2_m01=g2_m01,
            issue_all_a_loads=issue_all_a_loads,
            run_unit=run_unit,
            issue_and_run=issue_and_run,
        )

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
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
        i32_grid_blocks: fx.Int32,
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
            arg_bias,
            arg_out,
            bx_i32,
            lane,
            wave,
            i32_M,
            i32_max_m_blocks,
            i32_grid_blocks,
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
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        # i32_max_m_blocks sizes buffer resources; i32_grid_blocks bounds the launch to real m-blocks.
        num_n_blocks = D_HIDDEN // BN
        grid_x = g2_launch_grid_x(
            persist_flat, i32_max_m_blocks, i32_grid_blocks, num_n_blocks, cu_num
        )
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_bias,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_out_scale,
            i32_grid_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G2_CACHE = {}


def get_g2(
    BM,
    BN,
    BK,
    use_nt,
    D_HIDDEN,
    epilog,
    D_INTER,
    a_dtype,
    b_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    out_dtype="bf16",
    g2_spart=None,
    g2_kstatic=False,
    enable_bias=False,
):
    # D_HIDDEN/D_INTER specialize the kernel (N/K addressing, views, grid).
    # g2_kstatic still selects the unrolled K-loop (and related opts).
    SBM = _norm_sbm(SBM, BM)
    out_dtype = str(out_dtype).strip().lower()
    topk_key = topk if epilog == "reduce" else 1
    persist_flat = epilog == "scatter"
    persist = bool(persist or persist_flat)
    cu_key = cu_num if persist else 0
    # gemm2 perf knobs enter the key; defaults ON (env override), matching compile_gemm2_a4w4_port.
    g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_kstatic = bool(g2_kstatic)
    default_bf16_lds = "1" if g2_kstatic else "0"
    g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", default_bf16_lds) == "1"
    key = (
        BM,
        BN,
        BK,
        use_nt,
        D_HIDDEN,
        epilog,
        D_INTER,
        a_dtype,
        b_dtype,
        topk_key,
        SBM,
        persist,
        cu_key,
        g2_bhoist,
        g2_ascale_pf,
        g2_spart,
        g2_bf16_lds,
        g2_kstatic,
        out_dtype,
        enable_bias,
    )
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            BN=BN,
            BK=BK,
            use_nt=use_nt,
            D_HIDDEN=D_HIDDEN,
            epilog=epilog,
            D_INTER=D_INTER,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            topk=topk_key,
            SBM=SBM,
            persist=persist,
            cu_num=cu_key,
            g2_bhoist=g2_bhoist,
            g2_ascale_pf=g2_ascale_pf,
            g2_spart=g2_spart,
            g2_bf16_lds=g2_bf16_lds,
            g2_kstatic=g2_kstatic,
            out_dtype=out_dtype,
            enable_bias=enable_bias,
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
    BN=256,
    BK=256,
    use_nt=False,
    a_dtype="fp4",
    b_dtype="fp4",
    epilog="atomic",
    SBM=None,
    persist=False,
    cu_num=0,
    n_sorted_padded=None,
    out_dtype="bf16",
    g2_spart=None,
    stream=None,
    bias=None,
):
    """Stage-2 down-proj gemm for unpadded dimensions."""
    import torch

    _validate_v2_gemm2_dtypes(a_dtype, b_dtype)
    persist = bool(persist or epilog == "scatter")
    if persist and cu_num <= 0:
        cu_num = get_cu_num()
    SBM = _norm_sbm(SBM, BM)
    if BN not in (128, 256, 512):
        raise AssertionError(f"BN must be one of (128, 256, 512), got {BN}")
    if BK not in (128, 256):
        raise AssertionError(f"BK must be one of (128, 256), got {BK}")
    if D_HIDDEN % BN != 0:
        raise AssertionError(
            f"D_HIDDEN (N_OUT) must be a multiple of BN ({BN}), got {D_HIDDEN}"
        )
    if D_INTER % BK != 0:
        raise AssertionError(
            f"D_INTER (K) must be a multiple of BK ({BK}), got {D_INTER}"
        )
    if (
        str(out_dtype).strip().lower() == "bf16"
        and getattr(out, "dtype", None) != torch.bfloat16
    ):
        raise TypeError(
            "FlyDSL v2 GEMM2 supports only torch.bfloat16 output, "
            f"got {getattr(out, 'dtype', None)}"
        )
    if sorted_weights is None:
        raise NotImplementedError(
            "FlyDSL v2 GEMM2 requires sorted_weights; "
            "doweight_stage1=True is not supported"
        )
    _kstatic = os.environ.get("MXFP4_G2_KSTATIC", "1") == "1"
    if bias is not None:
        if bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        if not bias.is_contiguous():
            bias = bias.contiguous()
    launch = get_g2(
        BM,
        BN,
        BK,
        use_nt,
        D_HIDDEN,
        epilog,
        D_INTER,
        a_dtype,
        g2_kstatic=_kstatic,
        b_dtype=b_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
        out_dtype=out_dtype,
        g2_spart=g2_spart,
        enable_bias=bias is not None,
    )
    max_m_blocks = (max_sorted + BM - 1) // BM
    if persist:
        # Fixed grid: cu_num m-slots; each block loops over its m-tiles.
        grid_blocks = cu_num
    elif n_sorted_padded is not None:
        grid_blocks = n_sorted_padded // BM
    else:
        grid_blocks = min(
            max_m_blocks,
            _active_m_blocks_upper_bound(M_logical, topk, NE, BM, SBM),
        )
    out_scale = out  # unused by the atomic epilog; any valid device ptr is fine
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
        (bias if bias is not None else out).data_ptr(),
        M_logical,
        max_m_blocks,
        grid_blocks,
        out.data_ptr(),
        out_scale.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
