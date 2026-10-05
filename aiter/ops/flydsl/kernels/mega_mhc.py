# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-launch Mega-mHC seam for DeepSeek-V4.1 (delayed / Single-Pass mHC), gfx950.

For T tokens, 4 residual streams and hidden size H, one launch computes

    R'_j   = post_j * y + sum_h comb[h][j] * R_h            -> bf16, new residual
    mixes  = flatten(R') @ fn^T, rstd = rsqrt(mean(R'^2) + rms_eps)
    pre'   = sigmoid(mixes[0:4]  * rstd * s0 + b) + hc_pre_eps
    post'  = sigmoid(mixes[4:8]  * rstd * s1 + b) * hc_post_mult
    comb'  = Sinkhorn(mixes[8:24] * rstd * s2 + b)
    x1     = sum_j pre_j * R'_j                             -> bf16
    out    = x1 * rsqrt(mean(x1^2) + norm_eps) * w          -> bf16, or FP8 e4m3 group-32

with the same math and rounding points as the Triton seam
(``aiter/ops/triton/fusions/mhc_fused_post_pre_delayed_rmsnorm.py``).

Work split. A workgroup owns ``BLOCK_M`` tokens and ``H / NUM_KSPLIT`` columns of
all four streams. Its warps split either those columns (``WARP_SPLIT="cols"``,
reduced through LDS) or those tokens (``"tokens"``, sharing each k-step's fn tile
through LDS). A warp walks its columns in k-steps of ``TILE_K`` for 16 tokens.

Layouts. The streams (R, y, R', x1, FP8 out) are read and written in an
elementwise layout: lane ``l`` owns token ``l // 4`` and 8 columns ``(l % 4) * 8`` of
every 32-column chunk for all four streams, so 4 lanes cover 64 contiguous bytes of
a row, and the post-mix, the collapse and the FP8 group amax are lane-local (plus
two shuffles). The bf16 R' tile is moved through LDS into the A layout of
``mfma_f32_16x16x32_bf16`` (row = ``l % 16``) and multiplied against ``fn`` split into
bf16 hi + lo, pre-packed by the wrapper in B-register order: 3 B operands per
(chunk, stream), rows 16..23 hi and lo packed into one N tile (see ``N_FN_OPS``).

Finish. With ``NUM_KSPLIT == 1`` the workgroup finishes its own tokens. Otherwise
each split writes one fp32 partial row per token ``[0:24] mixes, [24] sum R'^2,
[25] sum x1^2`` and bumps the token block's counter; the last split to arrive sums
the rows, computes the gates (16 lanes per token, Sinkhorn by row/column shuffles),
rescales the staged bf16 ``x1`` in place (or only the FP8 scales) and re-arms the
counter. ``COHERENCE`` makes the other splits' writes visible to it:
  "xcd"   all splits of a token block run on one XCD (shared L2); L1-bypass reads.
  "agent" splits anywhere; agent-scope release/acquire, L1+L2-bypass reads.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_ir
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops as bops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

WAVE = 64
N_STREAMS = 4
N_MIX = 24
PSLOT = 32  # partial row: [0:24] mixes, [24] sum R'^2, [25] sum x1^2
FP8_GROUP = 32
LOG2E = 1.4426950408889634

# buffer cache policy bits (gfx950): sc0 = 1, nt = 2, sc1 = 16
CM_NT = 2
CM_L1_BYPASS = 1  # sc0: miss the CU's L1, hit the XCD's L2
CM_L2_BYPASS = 17  # sc0 sc1: system scope, past this XCD's L2

WARP_SPLITS = ("cols", "tokens")


def _put(t, idx, val):
    """``t[idx] = val`` outside the kernel AST, so the frontend does not read the
    subscript store as a rebinding of ``t`` inside runtime branches."""
    t[idx] = val


def _aux(cm):
    if bops._RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE:
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm)
    return fx.Int32(cm).ir_value()


def _ld(rsrc, voff, soff, n_dw, cm):
    """``n_dw`` dwords at byte offset voff (VGPR) + soff (SGPR); raw i32 vector."""
    ty = T.i32 if n_dw == 1 else T.vec(n_dw, T.i32)
    return Vec(
        rocdl_ir.RawPtrBufferLoadOp(
            ty, rsrc, fx.Int32(voff).ir_value(), fx.Int32(soff).ir_value(), aux=_aux(cm)
        ).result
    )


def _st(val, rsrc, voff, soff, cm):
    """Store a scalar / vector at byte offset voff (VGPR) + soff (SGPR)."""
    rocdl_ir.RawPtrBufferStoreOp(
        val.ir_value(),
        rsrc,
        fx.Int32(voff).ir_value(),
        fx.Int32(soff).ir_value(),
        aux=_aux(cm),
    )


COHERENCE_MODES = ("none", "xcd", "agent")


def check_config(H: int, cfg: dict) -> None:
    """Raise a ValueError naming the first violated legality rule of a knob set."""
    bm, ws, w = cfg["BLOCK_M"], cfg["WARP_SPLIT"], cfg["WARPS_PER_WG"]
    ks, tk = cfg["NUM_KSPLIT"], cfg["TILE_K"]
    coh = cfg["COHERENCE"]
    if ws not in WARP_SPLITS:
        raise ValueError(f"WARP_SPLIT must be one of {WARP_SPLITS}, got {ws!r}")
    if coh not in COHERENCE_MODES:
        raise ValueError(f"COHERENCE must be one of {COHERENCE_MODES}, got {coh!r}")
    if bm % 16:
        raise ValueError(f"BLOCK_M={bm} must be a multiple of the MFMA M (16)")
    if tk % 32:
        raise ValueError(f"TILE_K={tk} must be a multiple of the MFMA K (32)")
    if w not in (1, 2, 4, 8, 16):
        raise ValueError(f"WARPS_PER_WG={w} must be a power of two <= 16")
    if cfg["WARPS_PER_SIMD"] not in (1, 2, 3, 4):
        raise ValueError(f"WARPS_PER_SIMD={cfg['WARPS_PER_SIMD']} must be in 1..4")
    if ws == "cols":
        if H % (ks * w * tk):
            raise ValueError(
                f"WARP_SPLIT=cols needs H % (NUM_KSPLIT*WARPS_PER_WG*TILE_K) == 0: "
                f"{H} % ({ks}*{w}*{tk}) != 0"
            )
    else:
        if bm < 16 * w or bm % (16 * w):
            raise ValueError(
                f"WARP_SPLIT=tokens needs BLOCK_M a multiple of 16*WARPS_PER_WG: "
                f"BLOCK_M={bm}, WARPS_PER_WG={w}"
            )
        if H % (ks * tk):
            raise ValueError(
                f"WARP_SPLIT=tokens needs H % (NUM_KSPLIT*TILE_K) == 0: "
                f"{H} % ({ks}*{tk}) != 0"
            )
    if bm % (4 * w) and (4 * w) % bm:
        raise ValueError(
            f"the finish handles 4 tokens per warp: BLOCK_M={bm} and "
            f"4*WARPS_PER_WG={4 * w} must divide one another"
        )
    if (bm * PSLOT) % (WAVE * w):
        raise ValueError(f"BLOCK_M*32={bm * PSLOT} must be a multiple of the WG size")
    if ks == 1 and coh != "none":
        raise ValueError("NUM_KSPLIT=1 needs no coherence mode (use 'none')")
    if ks > 1 and coh == "none":
        raise ValueError("NUM_KSPLIT>1 needs COHERENCE 'xcd' or 'agent'")


def kernel_name(cfg: dict, has_post: bool, identity_pre: bool, out_fp8: bool) -> str:
    mode = "post" if has_post else "nopost"
    if identity_pre:
        mode += "_idpre"
    return (
        f"mega_mhc_bm{cfg['BLOCK_M']}_{cfg['WARP_SPLIT']}_w{cfg['WARPS_PER_WG']}"
        f"_s{cfg['WARPS_PER_SIMD']}_k{cfg['NUM_KSPLIT']}_t{cfg['TILE_K']}"
        f"_{cfg['COHERENCE']}_nt{int(cfg['NT_STREAMS'])}"
        f"_pk{int(cfg['FN_PREPACKED'])}_rcp{int(cfg['SINKHORN_RCP'])}"
        f"_{mode}_{'fp8' if out_fp8 else 'bf16'}"
    )


def grid_size(T: int, cfg: dict) -> tuple[int, int]:
    """(number of token blocks, number of workgroups) for T tokens."""
    nblk = -(-T // cfg["BLOCK_M"])
    ks = cfg["NUM_KSPLIT"]
    if cfg["COHERENCE"] == "xcd":
        return nblk, -(-nblk // 8) * 8 * ks
    return nblk, nblk * ks


@functools.cache
def compile_mega_mhc(
    *,
    H: int,
    BLOCK_M: int,
    WARP_SPLIT: str,
    WARPS_PER_WG: int,
    WARPS_PER_SIMD: int,
    NUM_KSPLIT: int,
    TILE_K: int,
    COHERENCE: str,
    NT_STREAMS: bool,
    FN_PREPACKED: bool,
    HAS_POST: bool,
    IDENTITY_PRE: bool,
    OUT_FP8: bool,
    SINKHORN_RCP: bool,
    SINKHORN_ITERS: int,
    FP8_MAX: float = 448.0,
):
    """Compile the Mega-mHC kernel for one knob set; returns the ``@flyc.jit`` launcher."""
    cfg = {
        "BLOCK_M": BLOCK_M,
        "WARP_SPLIT": WARP_SPLIT,
        "WARPS_PER_WG": WARPS_PER_WG,
        "WARPS_PER_SIMD": WARPS_PER_SIMD,
        "NUM_KSPLIT": NUM_KSPLIT,
        "TILE_K": TILE_K,
        "COHERENCE": COHERENCE,
        "NT_STREAMS": NT_STREAMS,
        "FN_PREPACKED": FN_PREPACKED,
        "SINKHORN_RCP": SINKHORN_RCP,
    }
    check_config(H, cfg)
    assert H % FP8_GROUP == 0 and SINKHORN_ITERS >= 1

    W = WARPS_PER_WG
    THREADS = WAVE * W
    KS = NUM_KSPLIT
    BM = BLOCK_M
    NC = TILE_K // 32  # 32-column chunks per k-step
    if WARP_SPLIT == "cols":
        M_WARPS, K_WARPS = 1, W
    else:
        M_WARPS, K_WARPS = W, 1
    MT = BM // (16 * M_WARPS)  # m-tiles per warp
    COLS_WG = H // KS
    COLS_W = COLS_WG // K_WARPS
    NK = COLS_W // TILE_K  # k-steps per warp
    K4 = N_STREAMS * H
    H8 = H // 8
    NG = H // FP8_GROUP
    RED = K_WARPS * BM * PSLOT  # LDS floats for the cross-warp reduce
    # The last-WG partial sum: thread groups each sum a residue class of splits.
    UNITS = BM * PSLOT // 4  # 16 B units of the BM x 32 partial block
    KGROUPS = max(1, min(THREADS // UNITS, KS))
    UPASS = max(1, UNITS // THREADS)
    assert UNITS % THREADS == 0 or THREADS % UNITS == 0
    FIN_PASSES = max(1, BM // (4 * W))
    FIN_WARPS = min(W, BM // 4)  # warps that hold tokens in the finish
    cm_stream = CM_NT if NT_STREAMS else 0
    cm_fin = CM_L2_BYPASS if COHERENCE == "agent" else CM_L1_BYPASS
    # Prepacked fn: [H/32 chunk][stream][op][lane][8] bf16, one 1 KiB B operand per
    # (chunk, stream, op): op 0/1 = rows 0..15 hi/lo, op 2 = rows 16..23 hi in
    # columns 0..7 and lo in columns 8..15 of the second N tile, folded by one
    # shuffle after the k-loop. 3 B operands and MFMAs per (chunk, stream), not 4.
    N_FN_OPS = 3
    NOPS = NC * N_STREAMS * N_FN_OPS  # 1 KiB B operands per k-step
    # Token-split warps share their columns, so a k-step's fn tile is loaded once
    # per WG into a double-buffered LDS slot instead of once per warp.
    FN_LDS = WARP_SPLIT == "tokens" and WARPS_PER_WG > 1 and FN_PREPACKED
    FN_PER_WARP = -(-NOPS // WARPS_PER_WG)
    fn_units = (H // 32) * N_STREAMS * N_FN_OPS * WAVE
    fn32_units = N_MIX * K4 // 4  # fp32 fn, 16 B units
    name = kernel_name(cfg, HAS_POST, IDENTITY_PRE, OUT_FP8)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, max(RED, KGROUPS * UNITS * 4), 16]
        fin: fx.Array[fx.Float32, BM * PSLOT, 16]
        rstdn: fx.Array[fx.Float32, BM, 16]
        flag: fx.Array[fx.Int32, 4, 16]
        # per-wave R' bf16 tile in elementwise lane order, read back as MFMA A
        xpose: fx.Array[fx.Int32, W * N_STREAMS * WAVE * 4, 16]
        fnbuf: fx.Array[fx.Int32, (2 * NOPS * WAVE * 4) if FN_LDS else 4, 16]

    F32, BF16, I32 = fx.Float32, fx.BFloat16, fx.Int32

    def mfma(a_bf16, b_bf16, acc):
        return Vec(
            rocdl.mfma_f32_16x16x32_bf16(
                T.vec(4, T.f32),
                [a_bf16.ir_value(), b_bf16.ir_value(), acc.ir_value(), 0, 0, 0],
            )
        )

    def bf16x8(v_i32x4):
        return Vec(v_i32x4).bitcast(BF16)

    def as_i32x4(v_bf16x8):
        return Vec(v_bf16x8).bitcast(I32)

    def fma(a, b, c):
        return F32(fx.fma(F32(a), F32(b), F32(c)))

    def exp(x):
        return F32(rocdl.exp2(T.f32, (x * F32(LOG2E)).ir_value()))

    def rcp(x):
        return F32(rocdl.rcp(T.f32, F32(x).ir_value()))

    def sigmoid(x):
        return F32(1.0) / (F32(1.0) + exp(F32(0.0) - x))

    def div(x, y):
        if fx.const_expr(SINKHORN_RCP):
            return x * rcp(y)
        return x / y

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mega_mhc_kernel(
        residual: fx.Pointer,  # (T, 4, H) bf16
        sublayer: fx.Pointer,  # (T, H) bf16                [HAS_POST]
        post_mix: fx.Pointer,  # (T, 4) fp32                [HAS_POST]
        comb_mix: fx.Pointer,  # (T, 4, 4) fp32 [h][j]      [HAS_POST]
        pre_mix: fx.Pointer,  # (T, 4) fp32                [not IDENTITY_PRE]
        fn: fx.Pointer,  # (2, 24, 4H) bf16 hi/lo, or (24, 4H) fp32
        hc_scale: fx.Pointer,  # (3,) fp32
        hc_base: fx.Pointer,  # (24,) fp32
        norm_w: fx.Pointer,  # (H,) bf16
        residual_out: fx.Pointer,  # (T, 4, H) bf16            [HAS_POST]
        out: fx.Pointer,  # (T, H) bf16, or fp8 bytes
        out_scale: fx.Pointer,  # (T, H/32) fp32              [OUT_FP8]
        post_out: fx.Pointer,  # (T, 4) fp32
        comb_out: fx.Pointer,  # (T, 16) fp32
        pre_out: fx.Pointer,  # (T, 4) fp32
        partials: fx.Pointer,  # (T, KS, 32) fp32            [KS > 1]
        counters: fx.Pointer,  # (nblk * 32,) int32         [KS > 1]
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        rms_eps: fx.Float32,
        hc_pre_eps: fx.Float32,
        hc_sinkhorn_eps: fx.Float32,
        hc_post_mult: fx.Float32,
        norm_eps: fx.Float32,
    ):
        tid = I32(fx.thread_idx.x)
        wi = tid // WAVE
        lane = tid % WAVE
        row = lane % 16  # MFMA layout: token row / fn row of a 16-tile
        kg = lane // 16  # MFMA layout: 8-column group
        erow = lane // 4  # elementwise layout: token row
        ekg = lane % 4  # elementwise layout: 8-column group (4 lanes = 64 B)
        w_id = I32(fx.block_idx.x)
        if fx.const_expr(COHERENCE == "xcd"):
            j = w_id // 8
            blk = (j // KS) * 8 + w_id % 8
            ks = j % KS
        else:
            blk = w_id // KS
            ks = w_id % KS
        tok0 = blk * BM

        nt64 = fx.Int64(n_tok)
        # Exact descriptor sizes: token rows >= T read 0 and drop their stores.
        post_t = ptr_buf_tensor(post_mix, F32, num_records_bytes=nt64 * 16)
        comb_t = ptr_buf_tensor(comb_mix, F32, num_records_bytes=nt64 * 64)
        pre_t = ptr_buf_tensor(pre_mix, F32, num_records_bytes=nt64 * 16)
        scale_t = ptr_buf_tensor(hc_scale, F32, num_records_bytes=12)
        base_t = ptr_buf_tensor(hc_base, F32, num_records_bytes=N_MIX * 4)
        w_t = ptr_buf_tensor(norm_w, I32, unit_elems=4, num_records_bytes=H * 2)
        if fx.const_expr(OUT_FP8):
            osc_t = ptr_buf_tensor(out_scale, F32, num_records_bytes=nt64 * (NG * 4))
        else:
            out_t = ptr_buf_tensor(
                out, I32, unit_elems=4, num_records_bytes=nt64 * (H * 2)
            )
        posto_t = ptr_buf_tensor(post_out, F32, num_records_bytes=nt64 * 16)
        combo_t = ptr_buf_tensor(comb_out, F32, num_records_bytes=nt64 * 64)
        preo_t = ptr_buf_tensor(pre_out, F32, num_records_bytes=nt64 * 16)
        part_t = ptr_buf_tensor(
            partials, F32, num_records_bytes=nt64 * (KS * PSLOT * 4)
        )
        part4_t = ptr_buf_tensor(
            partials, F32, unit_elems=4, num_records_bytes=nt64 * (KS * PSLOT * 4)
        )
        cnt_t = ptr_buf_tensor(counters, I32, num_records_bytes=fx.Int64(n_blk) * 128)

        def body():
            # LDS views are built here so they dominate every use in the body
            lds = fx.SharedAllocator().allocate(Smem).peek()
            red = lds.red
            fin = lds.fin
            rstdn = lds.rstdn
            flag = lds.flag
            xpose = lds.xpose
            fnbuf = lds.fnbuf

            # ---------------------------------------------------------- main pass
            # Hot-loop addressing: a few per-lane byte voffsets (token row, column
            # group) and wave-uniform soffsets (stream, column, k-step) on descriptors
            # based at this WG's first token, sized to the tokens that exist.
            wi_u = I32(rocdl.readfirstlane(T.i32, wi.ir_value()))
            if fx.const_expr(WARP_SPLIT == "cols"):
                tok_w = tok0
                col_w = ks * COLS_WG + wi_u * COLS_W
                red_slot = wi
            else:
                tok_w = tok0 + wi_u * (MT * 16)
                col_w = ks * COLS_WG
                red_slot = I32(0)
            rows_left = fx.Int64((n_tok > tok_w).select(n_tok - tok_w, I32(0)))

            def rs_rows(ptr, row_bytes):
                base = buf_base_i64(ptr) + fx.Int64(tok_w) * row_bytes
                return bops.create_buffer_resource_from_addr(
                    base.ir_value(),
                    num_records_bytes=(rows_left * row_bytes).ir_value(),
                )

            def rs_flat(ptr, nbytes):
                return bops.create_buffer_resource_from_addr(
                    buf_base_i64(ptr).ir_value(), num_records_bytes=nbytes
                )

            rs_res = rs_rows(residual, K4 * 2)
            rs_rout = rs_rows(residual_out, K4 * 2)
            rs_y = rs_rows(sublayer, H * 2)
            if fx.const_expr(OUT_FP8):
                rs_q = rs_rows(out, H)
                rs_sc = rs_rows(out_scale, NG * 4)
            else:
                rs_x1 = rs_rows(out, H * 2)
            rs_fn = rs_flat(fn, fn_units * 16 if FN_PREPACKED else fn32_units * 16)
            rs_w = rs_flat(norm_w, H * 2)

            # The streams (R, y, R', x1, fp8 out) use the elementwise layout: 4
            # consecutive lanes cover 64 contiguous bytes of one token row, so a
            # dwordx4 is 16 coalesced 64 B segments. Only the bf16 R' MFMA A operand
            # is moved to the MFMA layout (row = lane % 16), through LDS.
            OOB = I32(0x7FFFFFFF)
            v_r = erow * (K4 * 2) + ekg * 16  # residual / residual_out
            v_y = erow * (H * 2) + ekg * 16  # sublayer / staged x1
            if fx.const_expr(FN_PREPACKED):
                v_fn0 = lane * 16
                v_fn1 = lane * 16
            else:
                v_fn0 = row * (K4 * 4) + kg * 32
                v_fn1 = (16 + row % 8) * (K4 * 4) + kg * 32
            v_q = erow * H + ekg * 8
            v_sc = (ekg == 0).select(erow * (NG * 4), OOB)
            v_w = ekg * 16
            # LDS transpose: lane e writes slot e, MFMA lane m reads slot (m%16)*4 + m//16
            xp_base = wi * (N_STREAMS * WAVE * 4)
            xp_wr = xp_base + lane * 4
            xp_rd = xp_base + (row * 4 + kg) * 4

            def to_mfma_a(rn):
                for s in range_constexpr(N_STREAMS):
                    fx.ptr_store(
                        as_i32x4(rn[s]).ir_value(),
                        fx.add_offset(xpose.ptr, xp_wr + s * WAVE * 4),
                    )
                return [
                    Vec(
                        fx.ptr_load(
                            fx.add_offset(xpose.ptr, xp_rd + s * WAVE * 4),
                            T.vec(4, T.i32),
                        )
                    ).bitcast(BF16)
                    for s in range_constexpr(N_STREAMS)
                ]

            def mt_token(mt):
                return tok_w + mt * 16 + erow

            def load_gates(mt):
                t = mt_token(mt)
                g = {}
                if fx.const_expr(HAS_POST):
                    g["post"] = [post_t[t * 4 + jj] for jj in range_constexpr(4)]
                    g["comb"] = [comb_t[t * 16 + e] for e in range_constexpr(16)]
                if fx.const_expr(not IDENTITY_PRE):
                    g["pre"] = [pre_t[t * 4 + jj] for jj in range_constexpr(4)]
                return g

            gates = [load_gates(mt) for mt in range_constexpr(MT)]

            def chunk_col(iv, c):
                """wave-uniform first column of chunk c of k-step iv"""
                return col_w + iv * TILE_K + c * 32

            def load_tile(iv, dead):
                """R (and y) 16 B units of k-step iv; ``dead`` turns them into OOB no-ops."""
                vr = dead.select(OOB, v_r)
                vy = dead.select(OOB, v_y)
                vals = []
                for mt in range_constexpr(MT):
                    for c in range_constexpr(NC):
                        cb = chunk_col(iv, c) * 2
                        for s in range_constexpr(N_STREAMS):
                            vals.append(
                                _ld(
                                    rs_res,
                                    vr,
                                    cb + (mt * 16 * K4 + s * H) * 2,
                                    4,
                                    cm_stream,
                                )
                            )
                        if fx.const_expr(HAS_POST):
                            vals.append(
                                _ld(rs_y, vy, cb + mt * 16 * H * 2, 4, cm_stream)
                            )
                return vals

            PER_C = N_STREAMS + (1 if HAS_POST else 0)

            def load_fn(iv, c, s, dead):
                """the N_FN_OPS bf16x8 B operands of stream s, chunk c (raw i32x4)"""
                if fx.const_expr(FN_PREPACKED):
                    v = dead.select(OOB, v_fn0)
                    so = ((chunk_col(iv, c) // 32) * N_STREAMS + s) * (
                        N_FN_OPS * WAVE * 16
                    )
                    return [
                        _ld(rs_fn, v, so + op * WAVE * 16, 4, 0)
                        for op in range_constexpr(N_FN_OPS)
                    ]
                ops = []
                for nt in range_constexpr(2):
                    v = dead.select(OOB, v_fn0 if nt == 0 else v_fn1)
                    so = (chunk_col(iv, c) + s * H) * 4
                    f0 = Vec(_ld(rs_fn, v, so, 4, 0)).bitcast(F32)
                    f1 = Vec(_ld(rs_fn, v, so + 16, 4, 0)).bitcast(F32)
                    f = Vec.from_elements(
                        [f0[i] for i in range_constexpr(4)]
                        + [f1[i] for i in range_constexpr(4)],
                        F32,
                    )
                    hi = as_i32x4(f.to(BF16))
                    lo = as_i32x4((f - f.to(BF16).to(F32)).to(BF16))
                    if fx.const_expr(nt == 0):
                        ops += [hi, lo]
                    else:
                        ops.append((row < 8).select(hi, lo))
                return ops

            def load_fn_all(iv, dead, n_chunks=NC):
                """B operands of the first n_chunks chunks of k-step iv, flat [c][s][op]"""
                flat = []
                for c in range_constexpr(n_chunks):
                    for s in range_constexpr(N_STREAMS):
                        flat += load_fn(iv, c, s, dead)
                return flat

            def step(iv, tiles, fns, acc, sqr, sqx):
                """One k-step in three phases: the VALU work of every chunk, then
                the stores with the 32-column halves of a 128 B line back to back
                (so L2 merges them into one line write), then the MFMAs. ``fns``
                holds chunk 0's B operands (all chunks with FN_LDS); later chunks
                are loaded when their MFMAs are reached."""
                acc = list(acc)
                sqr = list(sqr)
                sqx = list(sqx)
                rn_all = {}  # (c, mt) -> 4 bf16x8 R' streams
                st_r = {}  # (mt, jj) -> [c] R' units to store
                st_x = {}  # mt -> [c] x1 / fp8 units to store
                for c in range_constexpr(NC):
                    cb = chunk_col(iv, c)
                    if fx.const_expr(OUT_FP8):
                        wv = bf16x8(_ld(rs_w, v_w, cb * 2, 4, 0)).to(F32)
                    for mt in range_constexpr(MT):
                        g = gates[mt]
                        base = (mt * NC + c) * PER_C
                        # Scalar fp32 FMAs on scalar gates: packed v_pk_* math needs a
                        # 2-wide splat of every gate, which doubles the live gate VGPRs.
                        r_bf = [
                            bf16x8(tiles[base + s]) for s in range_constexpr(N_STREAMS)
                        ]
                        if fx.const_expr(HAS_POST):
                            yv = bf16x8(tiles[base + N_STREAMS]).to(F32)
                            rf = [r_bf[h].to(F32) for h in range_constexpr(N_STREAMS)]
                        # Each new stream is folded into the square sums and the
                        # collapse as soon as it is rounded, so only one fp32 R' stream
                        # is live at a time.
                        rn = []
                        sq_acc = None
                        x1_el = None
                        for jj in range_constexpr(N_STREAMS):
                            if fx.const_expr(HAS_POST):
                                el = []
                                for e in range_constexpr(8):
                                    v = yv[e] * g["post"][jj]
                                    for h in range_constexpr(N_STREAMS):
                                        v = fma(rf[h][e], g["comb"][h * 4 + jj], v)
                                    el.append(v)
                                v_bf = Vec.from_elements(el, F32).to(BF16)
                                st_r.setdefault((mt, jj), []).append(as_i32x4(v_bf))
                            else:
                                v_bf = r_bf[jj]
                            rn.append(v_bf)
                            gj = v_bf.to(F32)
                            for e in range_constexpr(8):
                                sq_acc = (
                                    gj[e] * gj[e]
                                    if sq_acc is None
                                    else fma(gj[e], gj[e], sq_acc)
                                )
                            if fx.const_expr(not IDENTITY_PRE):
                                if fx.const_expr(jj == 0):
                                    x1_el = [
                                        gj[e] * g["pre"][0] for e in range_constexpr(8)
                                    ]
                                else:
                                    x1_el = [
                                        fma(gj[e], g["pre"][jj], x1_el[e])
                                        for e in range_constexpr(8)
                                    ]
                        rn_all[(c, mt)] = rn
                        sqr[mt] = sqr[mt] + sq_acc
                        if fx.const_expr(IDENTITY_PRE):
                            x1_bf = rn[0]
                        else:
                            x1_bf = Vec.from_elements(x1_el, F32).to(BF16)
                        x1f = x1_bf.to(F32)
                        a_x = x1f[0] * x1f[0]
                        for e in range_constexpr(1, 8):
                            a_x = fma(x1f[e], x1f[e], a_x)
                        sqx[mt] = sqx[mt] + a_x
                        if fx.const_expr(OUT_FP8):
                            v = x1f * wv
                            amax = F32(0.0)
                            for e in range_constexpr(8):
                                amax = fx.maximumf(amax, fx.absf(v[e]))
                            amax = fx.maximumf(amax, amax.shuffle_xor(1, WAVE))
                            amax = fx.maximumf(amax, amax.shuffle_xor(2, WAVE))
                            zero = amax == F32(0.0)
                            scale = zero.select(F32(1.0), amax * F32(1.0 / FP8_MAX))
                            inv = zero.select(F32(0.0), F32(FP8_MAX) * rcp(amax))
                            q = v * inv
                            dw = []
                            for d in range_constexpr(2):
                                pk = I32(0).ir_value()
                                pk = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    q[4 * d].ir_value(),
                                    q[4 * d + 1].ir_value(),
                                    pk,
                                    0,
                                )
                                pk = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    q[4 * d + 2].ir_value(),
                                    q[4 * d + 3].ir_value(),
                                    pk,
                                    1,
                                )
                                dw.append(I32(pk))
                            st_x.setdefault(mt, []).append(
                                (Vec.from_elements(dw, I32), scale)
                            )
                        else:
                            st_x.setdefault(mt, []).append(as_i32x4(x1_bf))
                cb0 = chunk_col(iv, 0)
                for mt in range_constexpr(MT):
                    if fx.const_expr(HAS_POST):
                        for jj in range_constexpr(N_STREAMS):
                            for c in range_constexpr(NC):
                                _st(
                                    st_r[(mt, jj)][c],
                                    rs_rout,
                                    v_r,
                                    (cb0 + c * 32) * 2 + (mt * 16 * K4 + jj * H) * 2,
                                    cm_stream,
                                )
                    for c in range_constexpr(NC):
                        cb = cb0 + c * 32
                        if fx.const_expr(OUT_FP8):
                            qv, scale = st_x[mt][c]
                            _st(qv, rs_q, v_q, cb + mt * 16 * H, 0)
                            _st(
                                scale,
                                rs_sc,
                                v_sc,
                                (cb // FP8_GROUP) * 4 + mt * 16 * NG * 4,
                                0,
                            )
                        else:
                            _st(st_x[mt][c], rs_x1, v_y, (cb + mt * 16 * H) * 2, 0)
                for c in range_constexpr(NC):
                    if fx.const_expr(FN_LDS or c == 0):
                        ops_c = fns[
                            c * N_STREAMS * N_FN_OPS : (c + 1) * N_STREAMS * N_FN_OPS
                        ]
                    else:
                        ops_c = []
                        for s in range_constexpr(N_STREAMS):
                            ops_c += load_fn(iv, c, s, never_oob)
                    for mt in range_constexpr(MT):
                        a_ops = to_mfma_a(rn_all[(c, mt)])
                        for s in range_constexpr(N_STREAMS):
                            for op in range_constexpr(N_FN_OPS):
                                nt = 0 if op < 2 else 1
                                acc[mt * 2 + nt] = mfma(
                                    a_ops[s],
                                    Vec(ops_c[s * N_FN_OPS + op]).bitcast(BF16),
                                    acc[mt * 2 + nt],
                                )
                return acc, sqr, sqx

            acc0 = [Vec.filled(4, 0.0, F32) for _ in range_constexpr(2 * MT)]
            sq0 = [F32(0.0) for _ in range_constexpr(MT)]
            never_oob = I32(0) != I32(0)  # a "dead" flag that is always false
            tiles0 = load_tile(I32(0), never_oob)
            n_acc, n_sq, n_tiles = 2 * MT, MT, len(tiles0)

            def pack(acc, sqr, sqx, tiles):
                return (
                    [a.ir_value() for a in acc]
                    + [F32(x).ir_value() for x in sqr]
                    + [F32(x).ir_value() for x in sqx]
                    + [Vec(t).ir_value() for t in tiles]
                )

            def unpack(state):
                acc = [Vec(state[i]) for i in range_constexpr(n_acc)]
                o = n_acc
                sqr = [F32(state[o + i]) for i in range_constexpr(n_sq)]
                o += n_sq
                sqx = [F32(state[o + i]) for i in range_constexpr(n_sq)]
                o += n_sq
                tiles = [Vec(state[o + i]) for i in range_constexpr(n_tiles)]
                return acc, sqr, sqx, tiles

            def fn_glb(k, dead):
                """this warp's share of k-step k's fn tile: ops wi, wi + W, ..."""
                chunk0 = chunk_col(k, 0) // 32
                regs = []
                for j in range_constexpr(FN_PER_WARP):
                    o = wi_u + j * W
                    v = (dead | (o >= I32(NOPS))).select(OOB, v_fn0)
                    regs.append(
                        _ld(rs_fn, v, (chunk0 * NOPS // NC + o) * (WAVE * 16), 4, 0)
                    )
                return regs

            def fn_to_lds(regs, buf):
                for j in range_constexpr(FN_PER_WARP):
                    o = wi_u + j * W
                    ptr = fx.add_offset(
                        fnbuf.ptr, (buf * NOPS + o) * (WAVE * 4) + lane * 4
                    )
                    if fx.const_expr(NOPS % W == 0):
                        fx.ptr_store(Vec(regs[j]).ir_value(), ptr)
                    else:
                        if o < I32(NOPS):
                            fx.ptr_store(Vec(regs[j]).ir_value(), ptr)

            def fn_from_lds(buf):
                return [
                    Vec(
                        fx.ptr_load(
                            fx.add_offset(
                                fnbuf.ptr, (buf * NOPS + o) * (WAVE * 4) + lane * 4
                            ),
                            T.vec(4, T.i32),
                        )
                    )
                    for o in range_constexpr(NOPS)
                ]

            # The current k-step's fn operands are issued before the next k-step's
            # R/y prefetch: vmcnt is in order, so waiting for fn then also covers the
            # current tiles while the prefetch stays in flight. The loop is unrolled
            # by two with A/B tile buffers: a single carried buffer interferes with
            # its own prefetch, and the back-edge copy then waits for that prefetch.
            # With FN_LDS the next k-step's fn tile goes global -> regs -> LDS buffer
            # 1 - buf behind one barrier per k-step.
            def half(k, cur, acc, sqr, sqx, buf=0):
                if fx.const_expr(FN_LDS):
                    g = fn_glb(k + 1, k + 1 == I32(NK))
                    nxt = load_tile(k + 1, k + 1 == I32(NK))
                    fns = fn_from_lds(buf)
                    acc, sqr, sqx = step(k, cur, fns, acc, sqr, sqx)
                    fn_to_lds(g, 1 - buf)
                    gpu.barrier()
                else:
                    fns = load_fn_all(k, never_oob, 1)
                    nxt = load_tile(k + 1, k + 1 == I32(NK))
                    acc, sqr, sqx = step(k, cur, fns, acc, sqr, sqx)
                return nxt, acc, sqr, sqx

            if fx.const_expr(FN_LDS):
                fn_to_lds(fn_glb(I32(0), never_oob), 0)
                gpu.barrier()
            init = pack(acc0, sq0, sq0, tiles0)
            results = init
            for ip, state in range(I32(0), I32(NK // 2), I32(1), init=init):
                acc, sqr, sqx, tiles_a = unpack(state)
                k0 = I32(ip) * 2
                tiles_b, acc, sqr, sqx = half(k0, tiles_a, acc, sqr, sqx, 0)
                tiles_a, acc, sqr, sqx = half(k0 + 1, tiles_b, acc, sqr, sqx, 1)
                results = yield pack(acc, sqr, sqx, tiles_a)
            if fx.const_expr(NK % 2):
                acc, sqr, sqx, tiles_a = unpack(results)
                _, acc, sqr, sqx = half(I32(NK - 1), tiles_a, acc, sqr, sqx, 0)
                results = pack(acc, sqr, sqx, tiles_a)
            acc, sqr, sqx, _ = unpack(results)

            # per-warp partials -> LDS red[slot][token][32]
            for mt in range_constexpr(MT):
                s_r = sqr[mt]
                s_x = sqx[mt]
                for off in (1, 2):
                    s_r = s_r + s_r.shuffle_xor(off, WAVE)
                    s_x = s_x + s_x.shuffle_xor(off, WAVE)
                tl_base = tok_w - tok0 + mt * 16
                for nt in range_constexpr(2):
                    n = nt * 16 + row
                    for i in range_constexpr(4):
                        tl = tl_base + kg * 4 + i
                        v = acc[mt * 2 + nt][i]
                        if fx.const_expr(nt == 1):
                            # rows 16..23: hi part in columns 0..7 + lo part in 8..15
                            v = (row < 8).select(v + v.shuffle_xor(8, WAVE), F32(0.0))
                            src = (
                                kg * 4 + i
                            ) * 4  # token t's sums sit on lanes 4t..4t+3
                            r_t = F32(gpu.shuffle(s_r, src, WAVE, mode="idx"))
                            x_t = F32(gpu.shuffle(s_x, src, WAVE, mode="idx"))
                            v = (n == N_MIX).select(
                                r_t, (n == N_MIX + 1).select(x_t, v)
                            )
                        _put(red, (red_slot * BM + tl) * PSLOT + n, v)
            gpu.barrier()

            # --------------------------------------------------------- finish
            def finish_gates_body():
                for p in range_constexpr(FIN_PASSES):
                    tl = p * (4 * W) + wi * 4 + kg
                    tok = tok0 + tl
                    e = row
                    fb = tl * PSLOT
                    m_e = fin[fb + e]
                    m_c = fin[fb + 8 + e]
                    s_r = fin[fb + N_MIX]
                    s_x = fin[fb + N_MIX + 1]
                    rstd = fx.rsqrt(s_r * F32(1.0 / K4) + rms_eps)
                    sc = scale_t[(e < 8).select(e // 4, I32(0))]
                    gate = sigmoid(m_e * rstd * sc + base_t[(e < 8).select(e, I32(0))])
                    is_pre = e < 4
                    is_post = (e >= 4) & (e < 8)
                    n4 = n_tok * 4
                    _put(preo_t, is_pre.select(tok * 4 + e, n4), gate + hc_pre_eps)
                    _put(
                        posto_t,
                        is_post.select(tok * 4 + e - 4, n4),
                        gate * hc_post_mult,
                    )
                    a = m_c * rstd * scale_t[2] + base_t[8 + e]
                    mx = a
                    for off in (1, 2):
                        mx = fx.maximumf(mx, mx.shuffle_xor(off, WAVE))
                    P = exp(a - mx)
                    rs = P
                    for off in (1, 2):
                        rs = rs + rs.shuffle_xor(off, WAVE)
                    P = div(P, rs) + hc_sinkhorn_eps
                    cs = P
                    for off in (4, 8):
                        cs = cs + cs.shuffle_xor(off, WAVE)
                    P = div(P, cs + hc_sinkhorn_eps)
                    for _ in range_constexpr(SINKHORN_ITERS - 1):
                        rs = P
                        for off in (1, 2):
                            rs = rs + rs.shuffle_xor(off, WAVE)
                        P = div(P, rs + hc_sinkhorn_eps)
                        cs = P
                        for off in (4, 8):
                            cs = cs + cs.shuffle_xor(off, WAVE)
                        P = div(P, cs + hc_sinkhorn_eps)
                    _put(combo_t, tok * 16 + e, P)
                    _put(rstdn, tl, fx.rsqrt(s_x * F32(1.0 / H) + norm_eps))

            RS_G = 4  # rescale units in flight per thread (16 measured slower)
            # the finish only walks the token rows that exist (decode: T < BLOCK_M)
            rows_valid = (n_tok - tok0 < I32(BM)).select(n_tok - tok0, I32(BM))

            def finish_rescale(rtid, NTHR):
                if fx.const_expr(OUT_FP8):
                    n_el = rows_valid * NG
                    for r_it in range(
                        I32(0), (n_el + NTHR * RS_G - 1) // (NTHR * RS_G), I32(1)
                    ):
                        r_idx, r_tl, r_sv = [], [], []
                        for g in range_constexpr(RS_G):
                            r_u = (I32(r_it) * RS_G + g) * NTHR + rtid
                            r_live = r_u < n_el
                            tl_g = r_live.select(r_u // NG, I32(0))
                            r_tl.append(tl_g)
                            r_idx.append(
                                r_live.select((tok0 + tl_g) * NG + r_u % NG, n_tok * NG)
                            )
                            r_sv.append(
                                buf_copy_load(
                                    osc_t, r_idx[g], F32, 1, cache_modifier=cm_fin
                                )
                            )
                        for g in range_constexpr(RS_G):
                            _put(osc_t, r_idx[g], r_sv[g] * rstdn[r_tl[g]])
                else:
                    n_u = rows_valid * H8
                    for r_it in range(
                        I32(0), (n_u + NTHR * RS_G - 1) // (NTHR * RS_G), I32(1)
                    ):
                        r_idx, r_tl, r_c8, r_xv = [], [], [], []
                        for g in range_constexpr(RS_G):
                            r_u = (I32(r_it) * RS_G + g) * NTHR + rtid
                            r_live = r_u < n_u
                            tl_g = r_live.select(r_u // H8, I32(0))
                            r_tl.append(tl_g)
                            r_c8.append(r_u % H8)
                            r_idx.append(
                                r_live.select((tok0 + tl_g) * H8 + r_c8[g], n_tok * H8)
                            )
                            r_xv.append(
                                buf_copy_load(
                                    out_t, r_idx[g], I32, 4, cache_modifier=cm_fin
                                )
                            )
                        for g in range_constexpr(RS_G):
                            r_wv = bf16x8(buf_copy_load(w_t, r_c8[g], I32, 4)).to(F32)
                            r_o = (bf16x8(r_xv[g]).to(F32) * rstdn[r_tl[g]]) * r_wv
                            buf_copy_store(
                                out_t, r_idx[g], as_i32x4(r_o.to(BF16)), I32, 4
                            )

            def finish():
                if fx.const_expr(FIN_WARPS < W):
                    if wi < I32(FIN_WARPS):
                        finish_gates_body()
                else:
                    finish_gates_body()
                gpu.barrier()
                finish_rescale(tid, THREADS)

            if fx.const_expr(KS == 1):
                for it in range_constexpr(BM * PSLOT // THREADS):
                    e = it * THREADS + tid
                    v = red[e]
                    for s in range_constexpr(1, K_WARPS):
                        v = v + red[s * BM * PSLOT + e]
                    _put(fin, e, v)
                gpu.barrier()
                finish()
            else:
                for it in range_constexpr(BM * PSLOT // THREADS):
                    e = it * THREADS + tid
                    v = red[e]
                    for s in range_constexpr(1, K_WARPS):
                        v = v + red[s * BM * PSLOT + e]
                    tl = e // PSLOT
                    _put(part_t, ((tok0 + tl) * KS + ks) * PSLOT + e % PSLOT, v)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == I32(0):
                    if fx.const_expr(COHERENCE == "agent"):
                        fx.llvm.memory_fence(
                            syncscope=rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Release,
                        )
                    cnt_ptr = fx.inttoptr(
                        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                        buf_base_i64(counters) + fx.Int64(blk) * 128,
                    )
                    old = fx.llvm.atomic_add(
                        cnt_ptr,
                        I32(1),
                        syncscope=rocdl.SyncScope.Agent,
                        ordering=fx.AtomicOrdering.Monotonic,
                    )
                    _put(flag, 0, old)
                gpu.barrier()
                if flag[0] == I32(KS - 1):
                    if fx.const_expr(COHERENCE == "agent"):
                        fx.llvm.memory_fence(
                            syncscope=rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Acquire,
                        )
                    # thread (g, u): sums splits k = g, g + KGROUPS, ... of unit u
                    g_id = tid // UNITS
                    u_lo = tid % UNITS
                    for up in range_constexpr(UPASS):
                        u = u_lo + up * THREADS
                        tl = u // (PSLOT // 4)
                        q4 = u % (PSLOT // 4)
                        if g_id < I32(KGROUPS):
                            acc4 = Vec.filled(4, 0.0, F32)
                            for k in range_constexpr(-(-KS // KGROUPS)):
                                kk = g_id + k * KGROUPS
                                live = kk < I32(KS)
                                uidx = ((tok0 + tl) * KS + kk) * (PSLOT // 4) + q4
                                uidx = live.select(uidx, n_tok * (KS * PSLOT // 4))
                                acc4 = acc4 + buf_copy_load(
                                    part4_t, uidx, F32, 4, cache_modifier=cm_fin
                                )
                            for i in range_constexpr(4):
                                _put(red, (g_id * UNITS + u) * 4 + i, acc4[i])
                    gpu.barrier()
                    for it in range_constexpr(BM * PSLOT // THREADS):
                        e = it * THREADS + tid
                        v = red[e]
                        for gg in range_constexpr(1, KGROUPS):
                            v = v + red[gg * UNITS * 4 + e]
                        _put(fin, e, v)
                    gpu.barrier()
                    finish()
                    if tid == I32(0):
                        _put(cnt_t, blk * 32, I32(0))

        if fx.const_expr(COHERENCE == "xcd"):
            # the XCD mapping pads the grid to whole groups of 8 token blocks
            if tok0 < n_tok:
                body()
        else:
            body()

    @flyc.jit
    def launch_mega_mhc(
        residual: fx.Pointer,
        sublayer: fx.Pointer,
        post_mix: fx.Pointer,
        comb_mix: fx.Pointer,
        pre_mix: fx.Pointer,
        fn: fx.Pointer,
        hc_scale: fx.Pointer,
        hc_base: fx.Pointer,
        norm_w: fx.Pointer,
        residual_out: fx.Pointer,
        out: fx.Pointer,
        out_scale: fx.Pointer,
        post_out: fx.Pointer,
        comb_out: fx.Pointer,
        pre_out: fx.Pointer,
        partials: fx.Pointer,
        counters: fx.Pointer,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        n_wg: fx.Int32,
        rms_eps: fx.Float32,
        hc_pre_eps: fx.Float32,
        hc_sinkhorn_eps: fx.Float32,
        hc_post_mult: fx.Float32,
        norm_eps: fx.Float32,
        stream: fx.Stream,
    ):
        mega_mhc_kernel(
            residual,
            sublayer,
            post_mix,
            comb_mix,
            pre_mix,
            fn,
            hc_scale,
            hc_base,
            norm_w,
            residual_out,
            out,
            out_scale,
            post_out,
            comb_out,
            pre_out,
            partials,
            counters,
            n_tok,
            n_blk,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult,
            norm_eps,
            value_attrs={"rocdl.waves_per_eu": int(WARPS_PER_SIMD)},
        ).launch(grid=(n_wg, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch_mega_mhc.compile_hints = {
        "waves_per_eu": int(WARPS_PER_SIMD),
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
            # packed v_pk_fma_f32 needs a 2-wide splat per gate (and LLVM keeps
            # one per use), which spills; scalar FMAs issue the same count
            "slp-threshold": 100000,
        },
    }
    launch_mega_mhc.kernel_name = name
    return launch_mega_mhc
