"""FlyDSL grouped GEMM for MXFP6 MoE on gfx950: MXFP6 weights (W6, E2M3 / E3M2) x MXFP6 or MXFP8
activations (A6 / A8), E8M0 scale per 32 along K, V_MFMA_SCALE_F32_16X16X128_F8F6F4.

    C[m, n] = sum_k A[row(m), k] * B[e(tile(m)), n, k]

Storage formats are produced by aiter.ops.flydsl.mxfp6_moe_quant (packed 6-bit weights preshuffled into
MFMA fragments; activation rows in MFMA K order; scales packed 4 K-steps per dword) and by
mxfp6_moe_aux.quant_rows.

Kernel (compile_gg6e): 4 waves, wave tile 128x128 / 128x64 / 64x128 / 64x64 (FM x FN 16x16 fragments),
3-stage LDS ring filled by async buffer->LDS DMA (16-B pieces, dense layout), scales DMA'd per 4 K-steps,
DMA issue interleaved over the MFMA groups, XCD-aware grouped tile order over the used tiles, LDS-staged
coalesced epilogue. Epilogues: "store" (bf16 C), "swiglu" (bf16 silu(gate)*up; B N-fragments gate/up
interleaved), "swiglu_q" (SwiGLU + MX quantization of the result in the stage-2 A layout).
All per-expert / per-tile bases are 64-bit (rebased buffer descriptors): tensors may exceed 4 GiB.

Measured operand contract of the scaled MFMA (lane l: row = l % 16, g = l // 16):
  FP6 / FP4: lane holds K[32g, 32g+32) contiguous (element j at bits [6j, 6j+6) / [4j, 4j+4), LE dwords).
  FP8: two K=64 halves: bytes 0-15 = K[16g, 16g+16), bytes 16-31 = K[64+16g, 64+16g+16).
  E8M0 scale of (row r, K-block b of 32) in lane 16b + r; acc[i] of lane l = C[4*(l//16) + i, l % 16].
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_store, ptr_arg, ptr_buf_tensor
from aiter.ops.flydsl.kernels.act import silu_mul_batch
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _lds_ptr3

from .mxfp6_moe_aux import mx_quant_store

FMT_CODE = {"e4m3": 0, "e5m2": 1, "e2m3": 2, "e3m2": 3, "e2m1": 4}
BITS = {"e4m3": 8, "e2m3": 6, "e3m2": 6}


def _uni(x):
    """Assert wave-uniformity of an i32 (v_readfirstlane) so derived values / descriptors live in SGPRs."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(x).ir_value()))


def _v(x):
    return fx.Int32(x).ir_value()


def _i64(p):
    return fx.Int64(fx.ptrtoint(p))


def _ptr(t: torch.Tensor):
    return ptr_arg(t)


def _rsrc(addr_i64):
    """Buffer descriptor (V#) of a raw 64-bit address, for the raw async buffer->LDS DMA below."""
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(addr_i64)))


# ----------------------------------------------------------------------------------------------
# compile_gg6e: dense LDS (16-B DMA pieces, no padding), 3-stage ring: DMA for step s+2 is issued while
# step s computes; end of step waits only for step s+1 (vmcnt = #DMAs of one step).
#   A in LDS: [BM rows][RB = 16*abits B] row-major (a row's 128-K chunk is contiguous in global)
#   B in LDS: [BN/16 frags][1536 B] (fragment order, as preshuffled)
# Loads past the last step are clamped to the last step (harmless rewrite of an unused buffer),
# so every step issues the same DMAs and the partial vmcnt is uniform.
# ----------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def compile_gg6e(N: int, K: int, a_fmt: str, b_fmt: str, epi: str, gather: bool,
                 FM: int = 8, FN: int = 8, WGM: int = 2, WGN: int = 2, NSTAGE: int = 3, ablate: str = "",
                 GM: int = 0, dev: int = 0):
    BM_, BN_ = 16 * FM * WGM, 16 * FN * WGN
    NW = WGM * WGN
    NT = 64 * NW
    assert N % BN_ == 0 and K % 128 == 0 and (epi == "store" or FN % 2 == 0)
    KS = K // 128
    KQ = (KS + 3) // 4
    KQF, KT = KS // 4, KS % 4
    abits = BITS[a_fmt]
    UA = 4 * abits  # bytes per (row, group) unit
    RB = 4 * UA  # A bytes per row per 128-step
    PPR = RB // 16  # 16-B pieces per row-step
    a_row_bytes = K * abits // 8
    b_frag_bytes = 1536
    b_expert_bytes = (N // 16) * KS * b_frag_bytes
    bs_expert_bytes = (N // 16) * KQ * 64 * 4
    SWI = epi in ("swiglu", "swiglu_q")
    n_out = N // 2 if SWI else N
    # swiglu_q: h is quantized in the epilogue to MX a_fmt (stage-2 "A" layout, see aiter.ops.flydsl.mxfp6_moe_quant)
    q_row_bytes = n_out * abits // 8
    KQ2 = (n_out // 128 + 3) // 4
    cbsz, blgp = FMT_CODE[a_fmt], FMT_CODE[b_fmt]
    BF = BN_ // 16
    A_BYTES, B_BYTES = BM_ * RB, BF * b_frag_bytes
    IA, IB = A_BYTES // 1024, B_BYTES // 1024  # DMA instructions per step (block)
    assert A_BYTES % 1024 == 0 and B_BYTES % 1024 == 0 and IA % NW == 0 and IB % NW == 0
    IAW, IBW = IA // NW, IB // NW
    STAGE = A_BYTES + B_BYTES
    SC_A, SC_B = BM_ * 16, BF * 256  # one quad of scales: A [rows][4 b] dwords, B [frags][64 lanes]
    SLOT = SC_A + SC_B
    SC_BASE = NSTAGE * STAGE
    lds_bytes = SC_BASE + 2 * SLOT
    BNO = BN_ // 2 if epi != "store" else BN_  # output columns per block
    OSTR = BNO * 2 + 16  # padded LDS row stride (bytes) of the staged output tile
    assert BM_ * OSTR <= lds_bytes, (BM_ * OSTR, lds_bytes)
    assert lds_bytes <= 160 * 1024, lds_bytes
    SBP = 16 if (SC_B // 1024) % NW == 0 else 4  # B-scale DMA piece bytes
    ISA_, ISB_ = SC_A // 256, SC_B // (64 * SBP)  # scale DMA instrs (4-B A pieces, SBP-B B pieces)
    assert ISA_ % NW == 0 and ISB_ % NW == 0, (ISA_, ISB_, NW)
    ISAW, ISBW = ISA_ // NW, ISB_ // NW
    name = (f"gg6e_{a_fmt}_{b_fmt}_{epi}_g{int(gather)}_n{N}_k{K}_f{FM}x{FN}_w{WGM}x{WGN}_s{NSTAGE}"
            + (f"_x{ablate}" if ablate else "") + (f"_gm{GM}" if GM else ""))
    GX = N // BN_

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def gg6e(
        A: fx.Pointer, AS: fx.Pointer, RIDX: fx.Pointer, TEXP: fx.Pointer,
        B: fx.Pointer, BS: fx.Pointer, C: fx.Pointer, CS: fx.Pointer, GY: fx.Int32,
    ):
        lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_raw))
        lds64 = fx.recast_iter(fx.Int64, lds_raw)
        tid = fx.Int32(gpu.thread_id("x"))
        nb = fx.Int32(gpu.block_id("x"))
        mb = fx.Int32(gpu.block_id("y"))
        if const_expr(GM > 0):
            # XCD-contiguous remap (dispatch round-robins blocks over 8 XCDs), then grouped
            # order: GM m-tiles x all n-tiles, m fastest, so co-resident blocks share A and B in L2
            # TEXP[GY] = number of used m-tiles: blocks past it exit (first-dispatched blocks are
            # the used ones, so the 8 XCDs stay balanced); the remap runs over used tiles only
            nvalid = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), _i64(TEXP))[GY]
            total = nvalid * fx.Int32(GX)
            L = mb * fx.Int32(GX) + nb
            L = (L < total).select(L, fx.Int32(-1))
            per = total // fx.Int32(8)
            lr = (total % fx.Int32(8) == fx.Int32(0)).select((L % fx.Int32(8)) * per + L // fx.Int32(8), L)
            grp = lr // fx.Int32(GM * GX)
            w = lr % fx.Int32(GM * GX)
            rem = nvalid - grp * fx.Int32(GM)
            gm_eff = (rem < fx.Int32(GM)).select(rem, fx.Int32(GM))
            gm_eff = (gm_eff < fx.Int32(1)).select(fx.Int32(1), gm_eff)
            mb = (L >= fx.Int32(0)).select(grp * fx.Int32(GM) + w % gm_eff, GY)
            nb = w // gm_eff
        i32pt = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        e = fx.inttoptr(i32pt, _i64(TEXP))[mb]
        if const_expr(GM > 0):
            e = (mb < GY).select(e, fx.Int32(-1))
        if e >= fx.Int32(0):
            lane = tid % fx.Int32(64)
            wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // fx.Int32(64)).ir_value()))
            wm = wave // fx.Int32(WGN)
            wn = wave % fx.Int32(WGN)
            r = lane % fx.Int32(16)
            g = lane // fx.Int32(16)

            a_rsrc = _rsrc(_i64(A))
            as_rsrc = _rsrc(_i64(AS))
            ridx = fx.inttoptr(i32pt, _i64(RIDX))
            b_base = _i64(B) + fx.Int64(e) * fx.Int64(b_expert_bytes)
            bs_base = _i64(BS) + fx.Int64(e) * fx.Int64(bs_expert_bytes)
            b_rsrc = _rsrc(b_base)
            bs_rsrc = _rsrc(bs_base)
            nfb = nb * fx.Int32(BF)

            a_src = []
            for t in range_constexpr(IAW):
                P = (wave + fx.Int32(t * NW)) * fx.Int32(64) + lane
                m = mb * fx.Int32(BM_) + P // fx.Int32(PPR)
                row = ridx[m] if const_expr(gather) else m
                a_src.append(row * fx.Int32(a_row_bytes) + (P % fx.Int32(PPR)) * fx.Int32(16))
            b_src = []
            for t in range_constexpr(IBW):
                P = (wave + fx.Int32(t * NW)) * fx.Int32(64) + lane
                f = P // fx.Int32(96)
                b_src.append((nfb + f) * fx.Int32(KS * b_frag_bytes) + (P % fx.Int32(96)) * fx.Int32(16))

            sa_src = []
            for t in range_constexpr(ISAW):
                D = (wave + fx.Int32(t * NW)) * fx.Int32(64) + lane
                m = mb * fx.Int32(BM_) + D // fx.Int32(4)
                row = ridx[m] if const_expr(gather) else m
                sa_src.append(((row * fx.Int32(4) + D % fx.Int32(4)) * fx.Int32(KQ)) * fx.Int32(4))
            sb_src = []
            for t in range_constexpr(ISBW):
                tt = wave + fx.Int32(t * NW)
                LPF = 256 // SBP  # lanes per fragment
                f = tt * fx.Int32(64 // LPF) + lane // fx.Int32(LPF)
                sb_src.append((nfb + f) * fx.Int32(KQ * 256) + (lane % fx.Int32(LPF)) * fx.Int32(SBP))

            def dma_scales(q_clamped, slot_off):
                for t in range_constexpr(ISAW):
                    dst = slot_off + (wave + fx.Int32(t * NW)) * fx.Int32(256)
                    rocdl.raw_ptr_buffer_load_async_lds(
                        as_rsrc, _lds_ptr3(lds_i32, dst), _v(4), _v(sa_src[t] + q_clamped * fx.Int32(4)), _v(0), _v(0))
                for t in range_constexpr(ISBW):
                    dst = slot_off + fx.Int32(SC_A) + (wave + fx.Int32(t * NW)) * fx.Int32(64 * SBP)
                    rocdl.raw_ptr_buffer_load_async_lds(
                        bs_rsrc, _lds_ptr3(lds_i32, dst), _v(SBP), _v(sb_src[t] + q_clamped * fx.Int32(256)), _v(0), _v(0))

            # Raw rocdl.raw_ptr_buffer_load_async_lds: every lane DMAs a 16-B piece from its own gathered
            # voffset into a wave-uniform LDS base (M0); the async form keeps LLVM from inserting
            # vmcnt(0) before the ds_reads of the current stage (the synchronous buffer->LDS copy did,
            # which serialized the 3-stage pipeline: 1463 -> 1849 TF/s on MI355X when switched).
            # TODO: evaluate fx.rocdl.BufferLoadAsyncLDS128b copy atoms with a gathered source layout.
            def dma(s_clamped, buf_off):
                for t in range_constexpr(IAW):
                    dst = buf_off + (wave + fx.Int32(t * NW)) * fx.Int32(1024)
                    rocdl.raw_ptr_buffer_load_async_lds(
                        a_rsrc, _lds_ptr3(lds_i32, dst), _v(16), _v(a_src[t] + s_clamped * fx.Int32(RB)), _v(0), _v(0))
                for t in range_constexpr(IBW):
                    dst = buf_off + fx.Int32(A_BYTES) + (wave + fx.Int32(t * NW)) * fx.Int32(1024)
                    rocdl.raw_ptr_buffer_load_async_lds(
                        b_rsrc, _lds_ptr3(lds_i32, dst), _v(16), _v(b_src[t] + s_clamped * fx.Int32(b_frag_bytes)), _v(0), _v(0))

            def l_frag(off_q, q):
                vals = [fx.ptr_load(fx.add_offset(lds64, off_q + fx.Int32(qq))) for qq in range_constexpr(q)]
                return fx.Vector.from_elements(vals, fx.Int64).bitcast(fx.Int32)

            # per-fragment LDS offsets as one base + compile-time constants (fold into ds_read immediates)
            a_lds0 = ((wm * fx.Int32(16 * FM) + r) * fx.Int32(RB) + g * fx.Int32(UA)) // fx.Int32(8)
            b_lds0 = wn * fx.Int32(FN * 192) + lane * fx.Int32(3)
            a_lds = [a_lds0 + fx.Int32(mi * 16 * RB // 8) for mi in range_constexpr(FM)]
            b_lds = [b_lds0 + fx.Int32(A_BYTES // 8 + ni * 192) for ni in range_constexpr(FN)]

            l32 = fx.recast_iter(fx.Int32, lds_raw)
            sa_lds0 = (wm * fx.Int32(16 * FM) + r) * fx.Int32(4) + g
            sb_lds0 = wn * fx.Int32(FN * 64) + lane
            sa_lds = [sa_lds0 + fx.Int32(mi * 64) for mi in range_constexpr(FM)]
            sb_lds = [sb_lds0 + fx.Int32(SC_A // 4 + ni * 64) for ni in range_constexpr(FN)]

            def load_scales(qd):
                so = fx.Int32(SC_BASE // 4) + (qd % fx.Int32(2)) * fx.Int32(SLOT // 4)
                a_sc = [fx.ptr_load(fx.add_offset(l32, so + sa_lds[mi])) for mi in range_constexpr(FM)]
                b_sc = [fx.ptr_load(fx.add_offset(l32, so + sb_lds[ni])) for ni in range_constexpr(FN)]
                return a_sc, b_sc

            def stage_off(s):
                return (s % fx.Int32(NSTAGE)) * fx.Int32(STAGE)

            def step(accs, s, j, sc):
                # stage s landed (all but the newest NSTAGE-2 groups) and every wave finished
                # reading stage s-1, whose buffer the DMA below overwrites
                if const_expr("bar" not in ablate):
                    # sched_barrier(0) on both sides: s_barrier has no memory semantics, so without
                    # these the scheduler may move this step's ds_reads / next DMAs across it
                    rocdl.sched_barrier(0)
                    rocdl.wait_asyncmark(NSTAGE - 2)
                    rocdl.s_barrier()
                    rocdl.sched_barrier(0)
                nxt = s + fx.Int32(NSTAGE - 1)
                nxt = (nxt < fx.Int32(KS)).select(nxt, fx.Int32(KS - 1))
                nbuf = stage_off(s + fx.Int32(NSTAGE - 1))
                # this step's DMA instructions as thunks, so they can be spread over the MFMA groups
                thunks = []
                if const_expr("dma" not in ablate):
                    for t in range_constexpr(IAW):
                        thunks.append(lambda t=t: rocdl.raw_ptr_buffer_load_async_lds(
                            a_rsrc, _lds_ptr3(lds_i32, nbuf + (wave + fx.Int32(t * NW)) * fx.Int32(1024)), _v(16),
                            _v(a_src[t] + nxt * fx.Int32(RB)), _v(0), _v(0)))
                    for t in range_constexpr(IBW):
                        thunks.append(lambda t=t: rocdl.raw_ptr_buffer_load_async_lds(
                            b_rsrc, _lds_ptr3(lds_i32, nbuf + fx.Int32(A_BYTES) + (wave + fx.Int32(t * NW)) * fx.Int32(1024)),
                            _v(16), _v(b_src[t] + nxt * fx.Int32(b_frag_bytes)), _v(0), _v(0)))
                    if const_expr((j + NSTAGE - 1) % 4 == 0):
                        def _sc():
                            qn = (s + fx.Int32(NSTAGE - 1)) // fx.Int32(4)
                            qn = (qn < fx.Int32(KQ)).select(qn, fx.Int32(KQ - 1))
                            dma_scales(qn, fx.Int32(SC_BASE) + (qn % fx.Int32(2)) * fx.Int32(SLOT))
                        thunks.append(_sc)
                IL = "noil" not in ablate  # interleave DMA issue over the MFMA groups (ATT: issue stalls)
                if const_expr(not IL):
                    for th in thunks:
                        th()
                    rocdl.asyncmark()
                if const_expr(j == 0):
                    sc = load_scales(s // fx.Int32(4))
                a_sc, b_sc = sc
                cur = stage_off(s) // fx.Int32(8)
                bv = [l_frag(cur + b_lds[ni], 3) for ni in range_constexpr(FN)]
                out = list(accs)
                av_next = l_frag(cur + a_lds[0], UA // 8)
                nth = len(thunks)
                for mi in range_constexpr(FM):
                    av = av_next
                    if const_expr(IL):
                        rocdl.sched_barrier(0)
                        for th in thunks[mi * nth // FM:(mi + 1) * nth // FM]:
                            th()
                    if const_expr(mi + 1 < FM):
                        av_next = l_frag(cur + a_lds[mi + 1], UA // 8)
                    for ni in range_constexpr(FN):
                        # Raw scaled MFMA: the A and B formats (cbsz / blgp: FP6 E2M3 / E3M2, FP8 E4M3) and the
                        # E8M0 scale-byte select j vary per call site; operands are raw 6/8-dword vectors read
                        # from LDS. TODO: switch to fx.rocdl.cdna4.MFMA_Scale atoms (one per opsel pair).
                        out[mi * FN + ni] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [av, bv[ni], accs[mi * FN + ni], cbsz, blgp, j, a_sc[mi], j, b_sc[ni]])
                if const_expr(IL):
                    rocdl.sched_barrier(0)
                    rocdl.asyncmark()
                return out, sc

            for p in range_constexpr(NSTAGE - 1):
                dma(fx.Int32(min(p, KS - 1)), fx.Int32(p * STAGE))
                if const_expr(p == 0):
                    dma_scales(fx.Int32(0), fx.Int32(SC_BASE))
                rocdl.asyncmark()
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            accs = [zero for _ in range(FM * FN)]
            if const_expr(KQF > 0):
                for qd, state in range(fx.Int32(0), fx.Int32(KQF), fx.Int32(1), init=accs):
                    cur_acc = list(state)
                    qi = fx.Int32(qd)
                    sc = None
                    for j in range_constexpr(4):
                        cur_acc, sc = step(cur_acc, qi * fx.Int32(4) + fx.Int32(j), j, sc)
                    results = yield cur_acc
                accs = list(results)
            if const_expr(KT > 0):
                sc = None
                for j in range_constexpr(KT):
                    accs, sc = step(accs, fx.Int32(KQF * 4 + j), j, sc)
            rocdl.wait_asyncmark(0)

            # ---- epilogue: stage the bf16 tile in LDS (padded rows), then 16-B coalesced stores ----
            # every wave is done reading the last K stage. gpu.barrier (fenced), not rocdl.s_barrier:
            # the raw barrier has no memory semantics and LLVM hoisted the epilogue ds_writes above it
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=0)  # wait_asyncmark(0) alone left LDS DMAs in flight (racy 256x128 runs)
            gpu.barrier()
            l16 = fx.recast_iter(fx.BFloat16, lds_raw)
            m_loc = wm * fx.Int32(16 * FM)
            for mi in range_constexpr(FM):
                for i in range_constexpr(4):
                    rl = m_loc + fx.Int32(mi * 16 + i) + g * fx.Int32(4)
                    rbase = rl * fx.Int32(OSTR // 2)
                    if const_expr(SWI):
                        for p in range_constexpr(FN // 2):
                            gv = fx.Vector(accs[mi * FN + 2 * p])[i]
                            uv = fx.Vector(accs[mi * FN + 2 * p + 1])[i]
                            h = silu_mul_batch([gv], [uv])[0]
                            col = (wn * fx.Int32(FN // 2) + fx.Int32(p)) * fx.Int32(16) + r
                            fx.ptr_store(fx.BFloat16(h), fx.add_offset(l16, rbase + col))
                    else:
                        for ni in range_constexpr(FN):
                            v = fx.Vector(accs[mi * FN + ni])[i]
                            col = (wn * fx.Int32(FN) + fx.Int32(ni)) * fx.Int32(16) + r
                            fx.ptr_store(fx.BFloat16(v), fx.add_offset(l16, rbase + col))
            gpu.barrier()
            if const_expr(epi == "swiglu_q"):
                y_base = _i64(C) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(q_row_bytes)
                s_base = _i64(CS) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(16 * KQ2)
                BPR = BNO // 32  # 32-element blocks per row of the tile
                for it in range_constexpr(BM_ * BPR // NT):
                    u = tid + fx.Int32(it * NT)
                    row = u // fx.Int32(BPR)
                    cb = u % fx.Int32(BPR)
                    vals = []
                    for c in range_constexpr(4):
                        v = fx.Vector(fx.ptr_load(fx.add_offset(l32, row * fx.Int32(OSTR // 4) + cb * fx.Int32(16) + fx.Int32(4 * c)),
                                                  result_type=T.vec(4, T.i32)))
                        for k in range_constexpr(4):
                            wd = fx.Int32(v[k])
                            vals.append((wd << fx.Int32(16)).bitcast(fx.Float32))
                            vals.append((wd & fx.Int32(-65536)).bitcast(fx.Float32))
                    mx_quant_store(vals, a_fmt, y_base, row * fx.Int32(q_row_bytes), nb * fx.Int32(BPR) + cb,
                                   s_base, row * fx.Int32(16 * KQ2), KQ2)
            c_tile_base = _i64(C) + fx.Int64(mb * fx.Int32(BM_)) * fx.Int64(n_out * 2)
            ct = ptr_buf_tensor(c_tile_base, fx.Int32, unit_elems=4, unit_stride=1)
            CPR = BNO // 8  # 16-B chunks per row
            for it in range_constexpr(0 if epi == "swiglu_q" else BM_ * CPR // NT):
                c = tid + fx.Int32(it * NT)
                row = c // fx.Int32(CPR)
                cc = c % fx.Int32(CPR)
                v = fx.ptr_load(fx.add_offset(l32, row * fx.Int32(OSTR // 4) + cc * fx.Int32(4)), result_type=T.vec(4, T.i32))
                buf_copy_store(ct, (row * fx.Int32(n_out) + nb * fx.Int32(BNO) + cc * fx.Int32(8)) // fx.Int32(2), v, fx.Int32, 4)

    @flyc.jit
    def launch(A: fx.Pointer, AS: fx.Pointer, RIDX: fx.Pointer, TEXP: fx.Pointer,
               B: fx.Pointer, BS: fx.Pointer, C: fx.Pointer, CS: fx.Pointer, m_tiles: fx.Int32, stream: fx.Stream):
        gg6e(A, AS, RIDX, TEXP, B, BS, C, CS, m_tiles).launch(
            grid=(N // BN_, m_tiles, 1), block=(NT, 1, 1), stream=stream
        )

    return launch


def run_gg6e(a_rows, a_scales, ridx, texp, b_frags, b_scales, out, *, N, K, a_fmt, b_fmt="e2m3",
             epi="store", gather=True, cfg=(8, 8, 2, 2), nstage=3, ablate="", gm=0, out_scales=None):
    """Launch compile_gg6e. texp: [m_tiles + 1] int32, tile -> expert (-1 unused), texp[m_tiles] = used tiles.

    All tensors must be contiguous and on one device; the launcher cache is keyed by the device index."""
    dev = out.device.index if out.device.index is not None else torch.cuda.current_device()
    exe = compile_gg6e(N, K, a_fmt, b_fmt, epi, gather, *cfg, NSTAGE=nstage, ablate=ablate, GM=gm, dev=dev)
    _run_compiled(exe, _ptr(a_rows), _ptr(a_scales), _ptr(ridx), _ptr(texp), _ptr(b_frags), _ptr(b_scales),
                  _ptr(out), _ptr(out_scales if out_scales is not None else out), texp.numel() - 1,
                  torch.cuda.current_stream(out.device))
    return out
