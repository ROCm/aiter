# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.expr.typing import as_ir_value as _raw_v

from aiter.jit.utils.chip_info import get_cu_num

from .mxfp4_gemm_common import _udiv
from .mxmoe_gemm_v2 import (
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
)
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
def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01, nmajor=False):
    """ck_tile GemmSpatiallyLocalTilePartitioner::GetOutputTileIndex: 1D block id -> spatially-local (m_block_idx, n_block_idx). block_1d_id/M0 runtime; N0/group_num/m01 compile-time."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = _udiv(mn + gn - fx.Int32(1), gn)
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = _udiv(block_1d_id, gn)
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x <= big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)

    if nmajor:
        if m01 != 1:
            raise AssertionError("nmajor requires m01==1")
        idx_N0 = _udiv(remap, M0)
        return remap - idx_N0 * M0, idx_N0

    idx_M0 = _udiv(remap, n0)
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = _udiv(M0, m01c)
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = _udiv(idx_M0, m01c)
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = _udiv(idx_local, M01_adapt)
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def _pick_epi_lanes(BM, BN, route_out_fp8, g2_scale_blk, nthreads=256):
    """Epilogue lanes-per-output-row for this tile, or 0 to keep the module default.

    Fewer lanes per row means each lane owns more columns, so the per-route
    address block (token_id -> out_row -> row_base_addr) is paid fewer times and
    the value store widens. The constraint is the mxfp8 block amax, which folds
    across lanes with at most a quad DPP: a scale block may span 1, 2 or 4 lanes.
    Measured on gfx950 at BM64/BN256/topk16/M=32768: 16 lanes 1.07x, 8 lanes
    1.06x, 32 lanes (the old fixed value) 1.00x -- hence the 16, 8, 32 order.

    That ordering is BN=256's answer and it is WRONG for the BN=512 tile this
    now ships. Re-measured there on an idle gfx950, duplicated arms, bit-exact
    at every shape (cos=1.000000, maxdiff=0.000e+00):
        t= 4096   16 lanes 218.9 us   32 lanes 214.1 us   x1.022
        t=16384   16 lanes 493.5      32 lanes 486.8      x1.014
        t=32768   16 lanes 911.5      32 lanes 905.9      x1.006
    At BN=512, 16 lanes gives route_vec=32 == g2_scale_blk (no cross-lane amax
    fold) while 32 lanes gives route_vec=16 and needs a pair DPP fold -- and the
    fold is cheaper than the wider per-lane work it buys. So the preference is
    BN-dependent, not universal; only BN>=512 is flipped, leaving BN=256 on the
    order its own measurement established.
    """
    if not route_out_fp8:
        return 0
    order = (32, 16, 8) if BN >= 512 else (16, 8, 32)
    for lanes in order:
        epi_rows = nthreads // lanes
        route_vec = BN // lanes
        if epi_rows == 0 or BM % epi_rows or route_vec == 0 or BN % lanes:
            continue
        if g2_scale_blk in (route_vec, 2 * route_vec, 4 * route_vec):
            return lanes
    return 0


def compile_gemm2_a4w4_port(
    BM=32,
    BN=256,
    BK=256,
    use_nt=False,
    HIDDEN_MAX=8192,
    epilog="atomic",
    INTER_MAX=8192,
    a_dtype="fp4",
    b_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    has_pad=False,
    g2_bhoist=None,
    g2_ascale_pf=None,
    g2_spart=None,
    g2_bf16_lds=None,
    g2_cpad=None,
    g2_swz=None,
    g2_pk8=None,
    g2_xcd=None,
    g2_tr=None,
    g2_serp=None,
    g2_maxnreg=None,
    g2_kstatic=False,
    out_dtype="bf16",
    g2_wpeu=None,
    g2_off32=False,
    g2_epi_lanes=0,
    g2_bfull=None,
    g2_split_lds=None,
    g2_wave_epi=None,
    g2_noepi=None,
    g2_sched=None,
    g2_schedmask=None,
    g2_mfmarep=None,
    g2_prio=None,
    g2_apf2=None,
    g2_bsingle=None,
    g2_epirep=None,
    g2_agpr=None,
    g2_swapab=None,
    g2_cbase=None,
    g2_earlybar=None,
    g2_interleave=None,
    g2_noppad=None,
    g2_barpad=None,
    g2_dspad=None,
    g2_klnop=None,
    g2_epinop=None,
    g2_illag=None,
    g2_asplit=None,
    g2_bspread=None,
    g2_nwaves=None,
    g2_nrep=None,
    g2_apre=None,
    g2_kmerge=None,
    g2_skippad=None,
    g2_brstore=None,
):
    """Compile gemm2 a4w4 down-proj; epilog 'atomic' (weighted atomic-fadd) or 'reduce' (store into out[token_id*topk+slot]). inter_dim runtime; SBM None -> SBM==BM byte-identical."""
    SBM = _norm_sbm(SBM, BM)
    if BM not in (16, 32, 64, 128) or epilog not in ("atomic", "reduce"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{16,32,64,128}}, epilog in {{'atomic','reduce'}}); "
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
    if g2_wpeu is None:
        g2_wpeu = int(os.environ.get("MXFP4_G2_WPEU", "0"))
    g2_wpeu = int(g2_wpeu)
    _validate_v2_gemm2_dtypes(a_dtype, b_dtype)
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}, got {INTER_MAX}"
    is_f8 = a_dtype == "fp8"
    if g2_bf16_lds is None:
        default_bf16_lds = "1" if g2_kstatic else "0"
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", default_bf16_lds) == "1"
    g2_bf16_lds = bool(g2_bf16_lds)
    if g2_cpad is None:
        g2_cpad = int(os.environ.get("MXFP4_G2_CPAD", "0"))
    g2_cpad = int(g2_cpad) if g2_bf16_lds else 0
    if g2_swz is None:
        g2_swz = int(os.environ.get("MXFP4_G2_SWZ", "0"))
    g2_swz = int(g2_swz) if g2_bf16_lds else 0
    if g2_pk8 is None:
        g2_pk8 = int(os.environ.get("MXFP4_G2_PK8", "1"))
    g2_pk8 = int(g2_pk8) if g2_bf16_lds else 0
    if g2_tr is None:
        g2_tr = int(os.environ.get("MXFP4_G2_TR", "0"))
    g2_tr = int(g2_tr) if g2_bf16_lds else 0
    if g2_serp is None:
        g2_serp = int(os.environ.get("MXFP4_G2_SERP", "0"))
    g2_serp = int(g2_serp)
    if g2_maxnreg is None:
        g2_maxnreg = int(os.environ.get("MXFP4_G2_MAXNREG", "0"))
    g2_maxnreg = int(g2_maxnreg)
    if g2_xcd is None:
        g2_xcd = int(os.environ.get("MXFP4_G2_XCD", "0"))
    g2_xcd = int(g2_xcd)
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    c_lds_bytes = (
        BN * (BM + 4) * 2
        if g2_tr
        else BM * ((BN // 16) * 18 + 4) * 2
        if g2_swz
        else BM * (BN + g2_cpad) * (2 if g2_bf16_lds else 4)
    )
    # aStages must exceed kStages: the K-loop ds_reads slot kt%aStages then
    # prefetches kt+kStages into (kt+kStages)%aStages, so equal counts make that
    # DMA rewrite the slot being read (cross-wave: waves DMA their own rows but
    # ds_read all BM rows). Only bump to 3 when the C region already covers it,
    # so lds_bytes and occupancy are unchanged; otherwise keep 2 and let
    # a_slot_alias fence the prefetch instead.
    aStages = 3 if (not g2_bf16_lds or 3 * slot_bytes <= c_lds_bytes) else 2
    a_slot_alias = aStages <= kStages
    K_TILES_RT_MAX = INTER_MAX // BK
    if g2_apre is None:
        g2_apre = aStages >= K_TILES_RT_MAX
    g2_apre = bool(g2_apre) and g2_kstatic
    if g2_skippad is None:
        g2_skippad = os.environ.get("MXFP4_G2_SKIPPAD", "0") == "1"
    g2_skippad = bool(g2_skippad)
    apre_tag = "_apre" if g2_apre else ""
    if g2_kmerge is None:
        g2_kmerge = int(os.environ.get("MXFP4_G2_KMERGE", "1"))
    _gridmul = int(os.environ.get("MXFP4_G2_GRIDMUL", "1"))
    g2_kmerge = int(g2_kmerge)
    kmg_tag = f"_kmg{g2_kmerge}" if g2_kmerge != 1 else ""
    if g2_brstore is None:
        g2_brstore = os.environ.get("MXFP4_G2_BRSTORE", "1") == "1"
    g2_brstore = bool(g2_brstore)
    skippad_tag = "_skpad" if g2_skippad else ""
    brs_tag = "" if g2_brstore else "_nobrs"
    cpad_tag = f"_cpad{g2_cpad}" if g2_cpad else ""
    swz_tag = "_swz" if g2_swz else ""
    pk8_tag = "_pk8" if g2_pk8 else ""
    tr_tag = "_tr" if g2_tr else ""
    serp_tag = "_serp" if g2_serp else ""
    mnr_tag = f"_mnr{g2_maxnreg}" if g2_maxnreg else ""
    xcd_tag = f"_xcd{g2_xcd}" if g2_xcd else ""
    if g2_split_lds is None:
        g2_split_lds = os.environ.get("MXFP4_G2_SPLITLDS", "0") == "1"
    g2_split_lds = bool(g2_split_lds)
    lds_bytes = (
        aStages * slot_bytes + c_lds_bytes
        if (g2_split_lds or g2_asplit)
        else max(c_lds_bytes, aStages * slot_bytes)
    )
    if g2_split_lds and lds_bytes > 160 * 1024:
        raise AssertionError(
            f"g2_split_lds needs {lds_bytes} B of LDS, over the 160 KiB gfx950 limit"
        )
    g2_ldspad = int(os.environ.get("MXFP4_G2_LDSPAD", "0"))
    if g2_ldspad:
        lds_bytes += g2_ldspad * 1024
        if lds_bytes > 160 * 1024:
            raise AssertionError(
                f"MXFP4_G2_LDSPAD={g2_ldspad} gives {lds_bytes} B, over 160 KiB"
            )
    ldspad_tag = f"_ldspad{g2_ldspad}" if g2_ldspad else ""
    # N_OUT = model_dim/hidden is runtime; HIDDEN_MAX is a compile/cache bucket
    # so different runtime hidden sizes can reuse one compiled launcher.
    assert (
        HIDDEN_MAX % BN == 0
    ), f"HIDDEN_MAX must be a multiple of {BN}, got {HIDDEN_MAX}"

    # Kernel-name tags empty on the default so its name/IR stays byte-identical (each variant distinct).
    atag = "_a8" if is_f8 else ""
    btag = "_w8" if b_dtype == "fp8" else ""
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
    pad_tag = (
        "_pad" if has_pad else ""
    )  # has_pad adds the runtime pad kernarg + weight-OOB pad-skip
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
    wpeu_tag = f"_wpeu{g2_wpeu}" if g2_wpeu > 0 else ""
    off32_tag = "_o32" if g2_off32 else ""
    if g2_nwaves is None:
        g2_nwaves = int(os.environ.get("MXFP4_G2_NWAVES", "4"))
    g2_nwaves = int(g2_nwaves)
    if BN % (g2_nwaves * 16) or BM % g2_nwaves:
        raise AssertionError(f"g2_nwaves={g2_nwaves} does not tile BM={BM}/BN={BN}")
    nw_tag = f"_nw{g2_nwaves}" if g2_nwaves != 4 else ""
    block_threads = g2_nwaves * 64
    g2_epi_lanes = int(g2_epi_lanes or os.environ.get("MXFP4_G2_EPI_LANES", "0"))
    if g2_epi_lanes == 0:
        g2_epi_lanes = _pick_epi_lanes(
            BM, BN, route_out_fp8, g2_scale_blk, nthreads=block_threads
        )
    epil_tag = f"_epil{g2_epi_lanes}" if g2_epi_lanes else ""
    if g2_bfull is None:
        g2_bfull = os.environ.get("MXFP4_G2_BFULL", "0") == "1"
    g2_bfull = bool(g2_bfull) and g2_kstatic
    bfull_tag = "_bfull" if g2_bfull else ""
    split_tag = "_splds" if g2_split_lds else ""
    if g2_wave_epi is None:
        g2_wave_epi = os.environ.get("MXFP4_G2_WAVE_EPI", "0") == "1"
    g2_wave_epi = bool(g2_wave_epi) and route_out_fp8 and use_reduce
    wepi_tag = "_wepi" if g2_wave_epi else ""
    if g2_noepi is None:
        g2_noepi = os.environ.get("MXFP4_G2_NOEPI", "0") == "1"
    g2_noepi = bool(g2_noepi)
    noepi_tag = "_NOEPIPROBE" if g2_noepi else ""
    if g2_sched is None:
        g2_sched = os.environ.get("MXFP4_G2_SCHED", "1") == "1"
    g2_sched = bool(g2_sched)
    sched_tag = "" if g2_sched else "_nosched"
    if g2_schedmask is None:
        g2_schedmask = int(os.environ.get("MXFP4_G2_SCHEDMASK", "0"), 0)
    g2_schedmask = int(g2_schedmask)
    schedmask_tag = f"_sm{g2_schedmask:x}" if g2_schedmask else ""
    if g2_mfmarep is None:
        g2_mfmarep = int(os.environ.get("MXFP4_G2_MFMAREP", "1"))
    g2_mfmarep = max(1, int(g2_mfmarep))
    mfmarep_tag = f"_mrep{g2_mfmarep}" if g2_mfmarep != 1 else ""
    if g2_prio is None:
        g2_prio = int(os.environ.get("MXFP4_G2_PRIO", "1"))
    g2_prio = int(g2_prio)
    prio_tag = "" if g2_prio else "_noprio"
    if g2_apf2 is None:
        g2_apf2 = int(os.environ.get("MXFP4_G2_APF2", "0"))
    g2_apf2 = int(g2_apf2)
    apf2_tag = f"_apf2x{g2_apf2}" if g2_apf2 else ""
    if g2_bsingle is None:
        g2_bsingle = os.environ.get("MXFP4_G2_BSINGLE", "0") == "1"
    g2_bsingle = bool(g2_bsingle)
    bsingle_tag = "_b1" if g2_bsingle else ""
    if g2_epirep is None:
        g2_epirep = int(os.environ.get("MXFP4_G2_EPIREP", "1"))
    g2_epirep = max(1, int(g2_epirep))
    epirep_tag = f"_erep{g2_epirep}" if g2_epirep != 1 else ""
    if g2_agpr is None:
        g2_agpr = os.environ.get("MXFP4_G2_AGPR", "0") == "1"
    g2_agpr = bool(g2_agpr)
    agpr_tag = "_agpr" if g2_agpr else ""
    if g2_swapab is None:
        g2_swapab = int(os.environ.get("MXFP4_G2_SWAPAB", "0"))
    # DEFAULT OFF -- NOT CORRECT YET. Compiled from a cold cache, swapab=1 drops
    # 56448 of 524288 route-out rows (10.8%) at token=32768; an earlier
    # "bit-identical" result was a stale-cache artifact and no cpad value fixes it.
    # The gate below is still required on top: only the reduce path's store_one_mr
    # understands the swapped lane->(row,col) mapping, the atomic epilogue derives
    # its route weight from the unswapped one (NaN), and tr/swz/wave-epi stage C
    # differently.
    g2_swapab = (
        int(g2_swapab)
        if (
            g2_bf16_lds
            and route_out_fp8
            and use_reduce
            and not g2_wave_epi
            and not g2_tr
            and not g2_swz
        )
        else 0
    )
    swapab_tag = f"_swab{g2_swapab}" if g2_swapab else ""
    if g2_cbase is None:
        g2_cbase = int(os.environ.get("MXFP4_G2_CBASE", "0"))
    g2_cbase = int(g2_cbase)
    cbase_tag = f"_cb{g2_cbase}" if g2_cbase else ""
    if g2_earlybar is None:
        g2_earlybar = os.environ.get("MXFP4_G2_EARLYBAR", "0") == "1"
    g2_earlybar = bool(g2_earlybar)
    ebar_tag = "_ebar" if g2_earlybar else ""
    if g2_interleave is None:
        g2_interleave = os.environ.get("MXFP4_G2_INTERLEAVE", "1") == "1"
    g2_interleave = bool(g2_interleave)
    if g2_noepi:
        g2_interleave = False
    if g2_interleave:
        g2_earlybar = True          # the C/A barrier must precede the final cluster
        ebar_tag = "_ebar"
    il_tag = "_il" if g2_interleave else ""
    if g2_noppad is None:
        g2_noppad = int(os.environ.get("MXFP4_G2_NOPPAD", "0"))
    g2_noppad = int(g2_noppad)
    nop_tag = f"_nop{g2_noppad}" if g2_noppad else ""
    if g2_barpad is None:
        g2_barpad = int(os.environ.get("MXFP4_G2_BARPAD", "0"))
    g2_barpad = int(g2_barpad)
    barpad_tag = f"_bar{g2_barpad}" if g2_barpad else ""
    if g2_dspad is None:
        g2_dspad = int(os.environ.get("MXFP4_G2_DSPAD", "0"))
    g2_dspad = int(g2_dspad)
    dsp_tag = f"_dsp{g2_dspad}" if g2_dspad else ""
    if g2_bspread is None:
        g2_bspread = os.environ.get("MXFP4_G2_BSPREAD", "0") == "1"
    g2_bspread = bool(g2_bspread)
    bsp_tag = "_bsp" if g2_bspread else ""
    if g2_klnop is None:
        g2_klnop = int(os.environ.get("MXFP4_G2_KLNOP", "0"))
    g2_klnop = int(g2_klnop)
    kln_tag = f"_kln{g2_klnop}" if g2_klnop else ""
    if g2_epinop is None:
        g2_epinop = int(os.environ.get("MXFP4_G2_EPINOP", "0"))
    g2_epinop = int(g2_epinop)
    epn_tag = f"_epn{g2_epinop}" if g2_epinop else ""
    if g2_illag is None:
        g2_illag = int(os.environ.get("MXFP4_G2_ILLAG", "1"))
    g2_illag = int(g2_illag)
    illag_tag = f"_lag{g2_illag}" if (g2_interleave and g2_illag != 1) else ""
    if g2_asplit is None:
        g2_asplit = os.environ.get("MXFP4_G2_ASPLIT", "0") == "1"
    g2_asplit = bool(g2_asplit) and not g2_split_lds
    asp_tag = "_asp" if g2_asplit else ""
    if g2_cbase and not g2_split_lds:
        lds_bytes += g2_cbase
    if g2_swapab and g2_cpad == 0:
        # Fastest of the pads tried, but note it does NOT make swapab correct
        # (see the row-drop note above); it only avoids the unpadded case where
        # all 16 lanes of a write group land in the same LDS bank.
        g2_cpad = 40
        cpad_tag = f"_cpad{g2_cpad}"
    if g2_nrep is None:
        g2_nrep = int(os.environ.get("MXFP4_G2_NREP", "1"))
    g2_nrep = int(g2_nrep)
    if g2_nrep != 1 and not (g2_kstatic and route_out_fp8 and use_reduce):
        g2_nrep = 1  # the n-block pipeline is only wired for the fp8 route path
    nrep_tag = ("" if g2_nrep == 1 else
                f"_nrep{'m' if g2_nrep < 0 else ''}{abs(g2_nrep)}")
    tag = f"hmax{HIDDEN_MAX}_imax{INTER_MAX}_bm{BM}{tile_tag}{'_nt' if use_nt else ''}_{etag}{atag}{btag}{sbm_tag}{persist_tag}{pad_tag}{bh_tag}{apf_tag}{spart_tag}{bf16lds_tag}{dw_tag}{kst_tag}{pitch_tag}{sblk_tag}{out_tag}{off32_tag}{epil_tag}{bfull_tag}{split_tag}{wepi_tag}{noepi_tag}{sched_tag}{nrep_tag}{apre_tag}{skippad_tag}{brs_tag}{cpad_tag}{swz_tag}{pk8_tag}{tr_tag}{serp_tag}{mnr_tag}{xcd_tag}{wpeu_tag}{ldspad_tag}{schedmask_tag}{mfmarep_tag}{prio_tag}{apf2_tag}{bsingle_tag}{epirep_tag}{agpr_tag}{swapab_tag}{cbase_tag}{ebar_tag}{il_tag}{nop_tag}{barpad_tag}{dsp_tag}{bsp_tag}{kln_tag}{epn_tag}{illag_tag}{asp_tag}{nw_tag}{kmg_tag}_v2"
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
        i32_grid_blocks,
    ):
        # Shared body for both has_pad variants (@flyc.jit -> rewriter recurses scf if / grid-stride); default passes i32_kpad/i32_npad=0 (no kernarg), folding pad math away.
        num_n_blocks = _udiv(i32_hidden, BN)
        k_bytes = _udiv(i32_inter, 1 if is_f8 else 2)
        aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * k_bytes)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        _a_preload = min(aStages, K_TILES_RT_MAX) if g2_apre else kStages

        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(_a_preload):
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
        def run_unit(unit_bx, mn_idx=None):
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
                arg_aq,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                BM=BM,
                BN=BN,
                BK=BK,
                use_nt=use_nt,
                INTER_MAX=INTER_MAX,
                g2_kstatic=g2_kstatic,
                aStages=aStages,
                a_slot_alias=a_slot_alias,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                use_reduce=use_reduce,
                topk=topk,
                has_pad=has_pad,
                SBM=SBM,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
                g2_bf16_lds=g2_bf16_lds,
                g2_cpad=g2_cpad,
                g2_swz=g2_swz,
                g2_pk8=g2_pk8,
                g2_tr=g2_tr,
                g2_serp=g2_serp,
                g2_defer_weight=g2_defer_weight,
                g2_out_pitch_align=g2_out_pitch_align,
                g2_scale_blk=g2_scale_blk,
                route_out_fp8=route_out_fp8,
                g2_off32=g2_off32,
                g2_epi_lanes=g2_epi_lanes or None,
                g2_bfull=g2_bfull,
                g2_split_lds=g2_split_lds,
                g2_wave_epi=g2_wave_epi,
                g2_noepi=g2_noepi,
                g2_sched=g2_sched,
                g2_schedmask=g2_schedmask,
                g2_mfmarep=g2_mfmarep,
                g2_prio=g2_prio,
                g2_apf2=g2_apf2,
                g2_bsingle=g2_bsingle,
                g2_epirep=g2_epirep,
                g2_swapab=g2_swapab,
                g2_cbase=g2_cbase,
                g2_earlybar=g2_earlybar,
                g2_interleave=g2_interleave,
                g2_noppad=g2_noppad,
                g2_barpad=g2_barpad,
                g2_dspad=g2_dspad,
                g2_klnop=g2_klnop,
                g2_epinop=g2_epinop,
                g2_illag=g2_illag,
                g2_asplit=g2_asplit,
                g2_bspread=g2_bspread,
                g2_nwaves=g2_nwaves,
                g2_nrep=g2_nrep,
                g2_apre=g2_apre,
                g2_kmerge=g2_kmerge,
                g2_brstore=g2_brstore,
                mn_idx=mn_idx,
            )

        if const_expr(not persist and g2_spart <= 0):
            # One-shot naive linear block->(m,n): issue A->LDS before the cumsum load (latency overlap).
            issue_all_a_loads(_udiv(bx_i32, num_n_blocks) * fx.Int32(BM))
            rocdl.sched_barrier(0)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                run_unit(bx_i32)
        elif const_expr(not persist):
            # One-shot with spatial-partitioner remap (g2_spart>0): needs M0=total_m_blocks so cumsum is read FIRST.
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = _udiv(cumsum0, BM)
            n_groups = _udiv(num_n_blocks, fx.Int32(abs(g2_nrep)))
            bound = total_m_blocks * n_groups

            _bx_eff = bx_i32
            if const_expr(g2_xcd > 0):
                _nx = fx.Int32(g2_xcd)
                _xcd = fx.Int32(bx_i32) % _nx
                _intra = fx.Int32(bx_i32) // _nx
                _base = bound // _nx
                _extra = bound % _nx
                _bx_eff = (
                    _xcd * _base
                    + (_xcd < _extra).select(_xcd, _extra)
                    + _intra
                )

            if fx.Int32(bx_i32) < bound:
                m_block_idx, n_block_idx = _spart_output_tile_index(
                    _bx_eff,
                    total_m_blocks,
                    num_n_blocks // abs(g2_nrep) if isinstance(num_n_blocks, int) else n_groups,
                    g2_group_num,
                    g2_m01,
                )
                unit_bx = m_block_idx * fx.Int32(num_n_blocks) + n_block_idx
                _m_row0 = m_block_idx * fx.Int32(BM)
                _tok0 = (
                    global_typed_ptr(arg_stids, T.i32)[_m_row0] & fx.Int32(0x00FFFFFF)
                )
                _tok0 = fx.Int32(rocdl.readfirstlane(T.i32, _raw_v(_tok0)))
                if const_expr(not g2_skippad):
                    issue_all_a_loads(_m_row0)
                    rocdl.sched_barrier(0)
                    run_unit(unit_bx, mn_idx=(m_block_idx, n_block_idx))
                elif _tok0 < i32_M:
                    issue_all_a_loads(_m_row0)
                    rocdl.sched_barrier(0)
                    run_unit(unit_bx, mn_idx=(m_block_idx, n_block_idx))
        else:
            # Persistent-m: fixed cu_num*num_n_blocks grid; each block grid-strides m-tiles by cu_num (aiter `_persist`).
            m_tile0 = _udiv(bx_i32, num_n_blocks)
            n_block = bx_i32 - m_tile0 * fx.Int32(num_n_blocks)
            c_stride = fx.Int32(cu_num)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = _udiv(cumsum0, BM)
            # ceil((total_m_blocks - m_tile0) / cu_num), clamped to 0 when m_tile0 >= total_m_blocks.
            diff = total_m_blocks - m_tile0
            rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
            n_iters = _udiv(rem + c_stride - fx.Int32(1), c_stride)
            for _it in range(
                fx.Int32(0),
                n_iters,
                fx.Int32(1),
            ):
                m_block = m_tile0 + fx.Int32(_it) * c_stride
                unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                gpu.barrier()  # persist: separate prev-iter epilog C-slab LDS reads from this iter's A-load into the shared LDS union
                issue_all_a_loads(m_block * fx.Int32(BM))
                rocdl.sched_barrier(0)
                if fx.Int32(m_block) < total_m_blocks:
                    run_unit(unit_bx)

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
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
        # i32_max_m_blocks sizes buffer resources; i32_grid_blocks bounds the launch to real m-blocks.
        num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(BN))
        grid_x = i32_grid_blocks * (num_n_blocks // fx.Int32(abs(g2_nrep)))
        if _gridmul > 1:
            grid_x = grid_x * fx.Int32(_gridmul)
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
            i32_grid_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

    if g2_wpeu > 0:
        launch_gemm2.compile_hints["waves_per_eu"] = g2_wpeu
    _func_attrs = {}
    if g2_maxnreg > 0:
        _func_attrs["amdgpu-num-vgpr"] = str(g2_maxnreg)

    if g2_agpr:
        launch_gemm2.compile_hints["llvm_options"] = {
            "amdgpu-mfma-vgpr-form": False
        }
        _func_attrs["amdgpu-agpr-alloc"] = os.environ.get("MXFP4_G2_AGPRN", "128")
    if _func_attrs:
        launch_gemm2.compile_hints["func_attrs"] = _func_attrs
    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G2_CACHE = {}


def get_g2(
    BM,
    BN,
    BK,
    use_nt,
    HIDDEN_MAX,
    epilog,
    INTER_MAX,
    a_dtype,
    b_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    has_pad=False,
    out_dtype="bf16",
    g2_bf16_lds=None,
    g2_spart=None,
    g2_kstatic=False,
    g2_off32=False,
):
    g2_epi_lanes = int(os.environ.get("MXFP4_G2_EPI_LANES", "0"))
    g2_bfull = os.environ.get("MXFP4_G2_BFULL", "0") == "1"
    g2_split_lds = os.environ.get("MXFP4_G2_SPLITLDS", "0") == "1"
    g2_wave_epi = os.environ.get("MXFP4_G2_WAVE_EPI", "0") == "1"
    g2_noepi = os.environ.get("MXFP4_G2_NOEPI", "0") == "1"
    g2_sched = os.environ.get("MXFP4_G2_SCHED", "1") == "1"
    g2_nrep = int(os.environ.get("MXFP4_G2_NREP", "1"))
    _apre_env = os.environ.get("MXFP4_G2_APRE")
    g2_apre = None if _apre_env is None else (_apre_env == "1")
    g2_kmerge = int(os.environ.get("MXFP4_G2_KMERGE", "1"))
    g2_skippad = os.environ.get("MXFP4_G2_SKIPPAD", "0") == "1"
    g2_brstore = os.environ.get("MXFP4_G2_BRSTORE", "1") == "1"
    # Cache key uses compile-time buckets; runtime inter_dim/model_dim share a
    # launcher while remaining within their respective caps.
    SBM = _norm_sbm(SBM, BM)
    out_dtype = str(out_dtype).strip().lower()
    topk_key = topk if epilog == "reduce" else 1
    cu_key = cu_num if persist else 0
    # gemm2 perf knobs enter the key; defaults ON (env override), matching compile_gemm2_a4w4_port.
    g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_kstatic = bool(g2_kstatic)
    if g2_bf16_lds is None:
        default_bf16_lds = "1" if g2_kstatic else "0"
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", default_bf16_lds) == "1"
    g2_bf16_lds = bool(g2_bf16_lds)
    g2_wpeu = int(os.environ.get("MXFP4_G2_WPEU", "0"))
    g2_cpad = int(os.environ.get("MXFP4_G2_CPAD", "0"))
    g2_swz = int(os.environ.get("MXFP4_G2_SWZ", "0"))
    g2_pk8 = int(os.environ.get("MXFP4_G2_PK8", "1"))
    g2_xcd = int(os.environ.get("MXFP4_G2_XCD", "0"))
    g2_tr = int(os.environ.get("MXFP4_G2_TR", "0"))
    g2_serp = int(os.environ.get("MXFP4_G2_SERP", "0"))
    g2_maxnreg = int(os.environ.get("MXFP4_G2_MAXNREG", "0"))
    g2_ldspad = int(os.environ.get("MXFP4_G2_LDSPAD", "0"))
    g2_schedmask = int(os.environ.get("MXFP4_G2_SCHEDMASK", "0"), 0)
    g2_mfmarep = int(os.environ.get("MXFP4_G2_MFMAREP", "1"))
    g2_prio = int(os.environ.get("MXFP4_G2_PRIO", "1"))
    g2_apf2 = int(os.environ.get("MXFP4_G2_APF2", "0"))
    g2_bsingle = os.environ.get("MXFP4_G2_BSINGLE", "0") == "1"
    g2_epirep = int(os.environ.get("MXFP4_G2_EPIREP", "1"))
    g2_agpr = os.environ.get("MXFP4_G2_AGPR", "0") == "1"
    g2_swapab = int(os.environ.get("MXFP4_G2_SWAPAB", "0"))
    g2_cbase = int(os.environ.get("MXFP4_G2_CBASE", "0"))
    g2_earlybar = os.environ.get("MXFP4_G2_EARLYBAR", "0") == "1"
    g2_interleave = os.environ.get("MXFP4_G2_INTERLEAVE", "1") == "1"
    g2_noppad = int(os.environ.get("MXFP4_G2_NOPPAD", "0"))
    g2_barpad = int(os.environ.get("MXFP4_G2_BARPAD", "0"))
    g2_dspad = int(os.environ.get("MXFP4_G2_DSPAD", "0"))
    g2_bspread = os.environ.get("MXFP4_G2_BSPREAD", "0") == "1"
    g2_nwaves = int(os.environ.get("MXFP4_G2_NWAVES", "4"))
    key = (
        BM,
        BN,
        BK,
        use_nt,
        HIDDEN_MAX,
        epilog,
        INTER_MAX,
        a_dtype,
        b_dtype,
        topk_key,
        SBM,
        persist,
        cu_key,
        has_pad,
        g2_bhoist,
        g2_ascale_pf,
        g2_spart,
        g2_bf16_lds,
        g2_kstatic,
        out_dtype,
        g2_wpeu,
        g2_off32,
        g2_epi_lanes,
        g2_bfull,
        g2_split_lds,
        g2_wave_epi,
        g2_noepi,
        g2_sched,
        g2_nrep,
        g2_apre,
        g2_kmerge,
        g2_skippad,
        g2_brstore,
        g2_cpad,
        g2_swz,
        g2_pk8,
        g2_xcd,
        g2_tr,
        g2_serp,
        g2_maxnreg,
        g2_ldspad,
        g2_schedmask,
        g2_mfmarep,
        g2_prio,
        g2_apf2,
        g2_bsingle,
        g2_epirep,
        g2_agpr,
        g2_swapab,
        g2_cbase,
        g2_earlybar,
        g2_interleave,
        g2_noppad,
        g2_barpad,
        g2_dspad,
        g2_bspread,
        g2_nwaves,
    )
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            BN=BN,
            BK=BK,
            use_nt=use_nt,
            HIDDEN_MAX=HIDDEN_MAX,
            epilog=epilog,
            INTER_MAX=INTER_MAX,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            topk=topk_key,
            SBM=SBM,
            persist=persist,
            cu_num=cu_key,
            has_pad=has_pad,
            g2_bhoist=g2_bhoist,
            g2_ascale_pf=g2_ascale_pf,
            g2_spart=g2_spart,
            g2_bf16_lds=g2_bf16_lds,
            g2_kstatic=g2_kstatic,
            out_dtype=out_dtype,
            g2_wpeu=g2_wpeu,
            g2_off32=g2_off32,
            g2_epi_lanes=g2_epi_lanes,
            g2_bfull=g2_bfull,
            g2_split_lds=g2_split_lds,
            g2_wave_epi=g2_wave_epi,
            g2_noepi=g2_noepi,
            g2_sched=g2_sched,
            g2_nrep=g2_nrep,
            g2_apre=g2_apre,
            g2_kmerge=g2_kmerge,
            g2_skippad=g2_skippad,
            g2_brstore=g2_brstore,
            g2_cpad=g2_cpad,
            g2_swz=g2_swz,
            g2_pk8=g2_pk8,
            g2_xcd=g2_xcd,
            g2_schedmask=g2_schedmask,
            g2_mfmarep=g2_mfmarep,
            g2_prio=g2_prio,
            g2_apf2=g2_apf2,
            g2_bsingle=g2_bsingle,
            g2_epirep=g2_epirep,
            g2_agpr=g2_agpr,
            g2_swapab=g2_swapab,
            g2_cbase=g2_cbase,
            g2_earlybar=g2_earlybar,
            g2_interleave=g2_interleave,
            g2_noppad=g2_noppad,
            g2_barpad=g2_barpad,
            g2_dspad=g2_dspad,
            g2_bspread=g2_bspread,
            g2_nwaves=g2_nwaves,
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
    inter_dim_pad=0,
    model_dim_pad=0,
    out_dtype="bf16",
    HIDDEN_MAX=8192,
    INTER_MAX=8192,
    g2_bf16_lds=None,
    g2_spart=None,
    stream=None,
):
    """Stage-2 down-proj gemm; epilog 'atomic' (weighted atomic.fadd) or 'reduce' (store into out[token_id*topk+slot]). inter_dim_pad/model_dim_pad>0 enable has_pad pad-skip (both 0 -> byte-identical); persist = fixed cu_num m-slot grid (default OFF)."""
    import torch

    _validate_v2_gemm2_dtypes(a_dtype, b_dtype)
    if persist and cu_num <= 0:
        cu_num = get_cu_num()
    SBM = _norm_sbm(SBM, BM)
    has_pad = inter_dim_pad > 0 or model_dim_pad > 0
    if BN not in (128, 256, 512):
        raise AssertionError(f"BN must be one of (128, 256, 512), got {BN}")
    if BK not in (128, 256):
        raise AssertionError(f"BK must be one of (128, 256), got {BK}")
    # model_dim/hidden (gemm2 N-output) is a runtime arg; validate host-side (not compile-time).
    if D_HIDDEN % BN != 0:
        raise AssertionError(
            f"D_HIDDEN (N_OUT) must be a multiple of BN ({BN}), got {D_HIDDEN}"
        )
    if D_INTER % BK != 0:
        raise AssertionError(
            f"D_INTER (K) must be a multiple of BK ({BK}), got {D_INTER}"
        )
    if D_HIDDEN > HIDDEN_MAX:
        raise AssertionError(
            f"D_HIDDEN ({D_HIDDEN}) exceeds compile cap HIDDEN_MAX ({HIDDEN_MAX})"
        )
    if D_INTER > INTER_MAX:
        raise AssertionError(
            f"D_INTER ({D_INTER}) exceeds compile cap INTER_MAX ({INTER_MAX})"
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
    if _kstatic:
        INTER_MAX = D_INTER
    _off32 = out.numel() * out.element_size() <= 0x7FFFFFFF
    if os.environ.get("MXFP4_G2_OFF32", "1") != "1":
        _off32 = False
    launch = get_g2(
        BM,
        BN,
        BK,
        use_nt,
        HIDDEN_MAX,
        epilog,
        INTER_MAX,
        a_dtype,
        g2_kstatic=_kstatic,
        b_dtype=b_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
        has_pad=has_pad,
        out_dtype=out_dtype,
        g2_bf16_lds=g2_bf16_lds,
        g2_spart=g2_spart,
        g2_off32=_off32,
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
    # i32_kpad (inter_dim_pad) + i32_npad (model_dim_pad) are always threaded after
    # i32_hidden; when has_pad is False they are 0 and the kernel folds pad math away.
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
        int(inter_dim_pad),
        int(model_dim_pad),
        out.data_ptr(),
        out_scale.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
