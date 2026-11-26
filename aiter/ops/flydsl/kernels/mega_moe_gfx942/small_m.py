# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The small path's kernel (gfx942): the EP4 / EP8 a8w4 MoE layer for decode-sized
batches as ONE kernel launch per call: x quant + dispatch, gemm1, gemm2 and the combine.

Design: aiter ``ops/flydsl/kernels/mega_moe/mega_moe_small_m.py`` (the MegaMoEV2
small-input path, gfx950), on the GEMM tiles of ``kloop._make_core`` (fd2 mxfp4 W
streamed into VGPRs, fp8 A through LDS):

- Fixed slots: every local expert owns TMAX rows on its rank (TMAX >= the batch over all
  ranks; a token sends at most one route to an expert), so a source writes a route
  straight into its expert's next row on the destination: one system-scope atomic on the
  destination's ``run[expert]`` gives the row. No count exchange, no plan kernel, no sort.
- Producers (workgroups [0, P)): one wave per token (tokens striped over the producers
  first): the bf16 row -> fp8 + f32 scales (QX 1: one per 128 columns, QX 0: one per row)
  in registers while the token's routes take their rows (lanes [0, topk), one round
  trip), then the row, scales, weight and source id go to each destination (written
  through). Each producer, once its stores completed, adds DONE_UNIT / P to ``done[me]``
  on every rank.
- Plan: every workgroup waits for ``done`` of all ranks (epoch x DONE_UNIT), reads the
  ``run`` counts of its E_LOCAL experts and derives the same tile list (BM-row m-tiles,
  expert order); nothing is published.
- Units: XCD x takes, from its own counter, its eighth of the gemm1 units (m-tile x 128
  inter cols, m-major) and then its eighth of the gemm2 units (m-tile x NO out cols): the
  same m-tiles, so gemm2 mostly reads a2 rows its own XCD wrote (a2 is written through:
  at the slice ends, and for units taken from another XCD's counter once the own one is
  empty, another XCD reads them). On MI308X's 4 XCDs each XCD takes two of the eight
  slices (``atomic._home_queue``). gemm1 = the main kernel's (TR epilogue: act, fp8 a2 + one scale
  per (row, 128 cols); QX 1 rescales the accumulator at every 128-K group, QX 0 applies
  the row scale in the epilogue); a gemm2 unit waits for the I/128 gemm1 units of its
  m-tile (W prefetch already in flight), then writes every row (times the route weight,
  bf16) into ``cin[token, slot]`` of the route's source rank: one wave per row.
- Combine: the last workgroup to finish (two-level count) resets the counters and sets
  ``cflag[me]`` = epoch on every rank; workgroup b < max(n, 1) (n: this rank's tokens)
  waits for ``cflag`` of all ranks and sums the topk rows of tokens b, b + GRID, ... into
  ``out`` (f32, slot order; a negative id contributes nothing).

Epoch: ``ctl[SC_EPOCH] + 1``, stored back by the last workgroup: the same on every rank
as long as every rank makes the same calls, right in CUDA graph replays. A rank enters
call N + 1 only after all ranks' rows of call N reached it, i.e. after every rank finished
reading its slots and reset its ``run`` counts. Everything a peer reads lives in the
uncached MORI heap and is written with system-scope stores; a flag is stored only after
the data stores completed (``s_waitcnt vmcnt(0)``): no L2 write-back or invalidation.

Needs the whole GPU: the grid (``NUM_CU`` x ``WPC`` workgroups, ``NUM_CU`` the GPU's CUs)
must be resident at once (the plan wait and the combine wait spin).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..act import gate_up_act, situ_params
from .atomic import (
    _atomic_add_at,
    _atomic_load_at,
    _home_queue,
    _inv_l1,
    _sys_load_at,
    _sys_store_at,
)
from .cvt import _abs_bits, _amax_floor, _channel_factor, _fp8x4_pack
from .hw import CM_NT, CM_SC1, CM_SYS, LDS_BYTES_MAX, _opaque_tx, _u
from .kloop import _dispatch_rows, _make_core
from .mem import (
    _buffer_i32_scalar_read,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _ld,
    _lds_f32_store_masked,
    _raw,
    _rsrc,
    _store,
    buffer_load,
    buffer_store,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from .protocol import (
    BUFFER_BYTES_MAX,
    SC_EPOCH,
    SC_HEAD,
    SC_SUB,
    SC_TILE,
    SC_TOP,
    SMALL_LINE,
    small_max_tiles,
    small_sym_layout,
)

P2P = CM_SYS | CM_NT  # stores into a rank's (uncached) heap: written through
WT = CM_SC1  # a2: written through (a gemm2 unit on another XCD may read it)


def _bcast_lane(v, src_lane):
    """i32 ``v`` of lane ``src_lane`` (per lane) -> every lane (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, _raw(src_lane * fx.Int32(4)), _raw(fx.Int32(v)))
    )


def _readlane(v, k):
    return fx.Int32(rocdl.readlane(T.i32, _raw(fx.Int32(v)), _raw(fx.Int32(k))))


def _addr(lo, hi):
    return fx.Int64(
        (fx.Uint64(fx.Uint32(hi)) << fx.Uint64(32)) | fx.Uint64(fx.Uint32(lo))
    )


def ep_mega_small_kernel(
    *,
    H,
    I,
    TOPK,
    E_LOCAL,
    TMAX,
    NUM_CU,
    XCDS,
    NPES,
    BM=64,
    NO=256,
    ARB=256,
    PF=2,
    P=16,
    WPC=1,
    QX=1,
    ACT="silu",
    BCM1=2,
    BCM2=2,
):
    """The small megamoe kernel for (H, I, top-k), ``E_LOCAL`` experts on each of ``NPES``
    ranks (4 or 8) and batches of <= TMAX tokens over all
    ranks (fixed slots of TMAX rows per expert). ``NUM_CU`` / ``XCDS``: CUs and XCDs of
    the GPU, 304 / 8 (MI300X) or 80 / 4 (MI308X). ``BM``: rows of an m-tile (row variants
    16 .. BM); ``NO``: gemm2 output cols per unit; ``ARB`` / ``PF``: K bytes per A tile /
    W prefetch distance of both GEMMs; ``P``: producer workgroups; ``WPC``: workgroups per
    CU; ``QX``: x scales, 1 one per (row, 128 cols), 0 one per row; ``BCM1`` / ``BCM2``:
    gemm1 / gemm2 W load cache modifier (2 = nt)."""
    assert ACT in ("silu", "silu_clamped", "situv2") and QX in (0, 1), (ACT, QX)
    NW = 8
    NT = 64 * NW
    NI = 128
    N2 = 2 * I
    NG, ND = I // NI, H // NO  # gemm1 / gemm2 units per m-tile
    G1N, G2N = H // 128, I // 128  # 128-K scale groups of gemm1 / gemm2
    NIW, NOW = NI // NW, NO // NW
    NC1, NC2 = NIW // 16, NOW // 16
    GPW, WPG = NI // 128, NW // (
        NI // 128
    )  # gemm1 epilogue: groups per tile, waves per group
    GRID = NUM_CU * WPC
    E_GLOBAL = NPES * E_LOCAL
    CAP = TMAX
    ROWS = E_LOCAL * CAP  # fixed-slot rows of a rank
    ROWV = tuple(r for r in (16, 32, 64) if r <= BM)
    MAXT = small_max_tiles(TOPK, TMAX, BM, E_LOCAL)
    A2_ROWS = MAXT * BM
    CH = H // 1024  # producer: 1 KB fp8 chunks of a row, 16 B per lane
    VW = NO // 128  # gemm2 scatter: i32 per lane of a row's NO bf16
    UNITS = H // 8  # combine: a bf16 row in 16 B units, NT per pass
    LINE = SMALL_LINE
    off, _ = small_sym_layout(H, TOPK, TMAX, E_LOCAL, NPES)
    O_RX, O_RS, O_RW, O_RID = off["rx"], off["rs"], off["rw"], off["rid"]
    O_RUN, O_DONE, O_CIN, O_CFLAG = off["run"], off["done"], off["cin"], off["cflag"]
    assert BM in (16, 32, 64) and CAP % BM == 0 and TMAX <= 1 << 16, (BM, TMAX)
    assert H % 1024 == 0 and G1N == 8 * CH and I % NI == 0 and NIW % 16 == 0
    assert H % NO == 0 and NO % 128 == 0 and NOW % 16 == 0, NO
    # a lane per row scale (producers) in one wave; the plan takes the local experts 64
    # (a wave) at a time
    assert G1N <= 64, G1N
    PLAN_PASSES = -(-E_LOCAL // 64)
    assert 1 <= TOPK <= 8 and 0 < P <= GRID and WPC in (1, 2), (TOPK, P, WPC)
    # done[s] on every rank: each of source s's P producers adds DONE_UNIT / P, so a call
    # adds DONE_UNIT whatever P (the call with epoch k is complete at k DONE_UNIT)
    DONE_UNIT = 64
    assert DONE_UNIT % P == 0, f"producers {P} must divide {DONE_UNIT}"
    RATIO = (
        2 * BM * ARB
    )  # f32 ratio tables [G][BM] + fin [BM] after the A double buffer
    LDS_GEMM = max(
        RATIO + (G1N + 1) * BM * 4, RATIO + (G2N + 1) * BM * 4, BM * NO * 2, BM * NW * 4
    )
    L_CNT = LDS_GEMM // 4  # i32: rows per local expert [E_LOCAL]
    L_TB = (
        L_CNT + E_LOCAL
    )  # first m-tile of each expert [E_LOCAL], then the m-tiles [1]
    L_BC = L_TB + E_LOCAL + 1  # broadcast words [4]
    L_PEER = -(-(L_BC + 4) // 2) * 2  # every rank's heap base (i64 [NPES])
    LDS_BYTES = -(-(L_PEER + 2 * NPES) * 4 // 16) * 16
    assert LDS_BYTES <= LDS_BYTES_MAX // WPC, LDS_BYTES
    kname = (
        f"a8w4_megamoe_small_h{H}_i{I}_k{TOPK}_e{E_LOCAL}_t{TMAX}_bm{BM}_no{NO}"
        f"_a{ARB}_pf{PF}_p{P}_w{WPC}"
        + ("" if (BCM1, BCM2) == (2, 2) else f"_bcm{BCM1}{BCM2}")
        + ("" if QX else "_xrow")
        + ("" if ACT == "silu" else f"_{ACT}")
        + ("" if (NUM_CU, XCDS) == (304, 8) else f"_cu{NUM_CU}_x{XCDS}")
        + ("" if NPES == 8 else f"_n{NPES}")
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    NAMES64 = (
        "x",
        "ids",
        "tw",
        "out",
        "sym",
        "stab",
        "ctl",
        "a2",
        "a2s",
        "w1",
        "w1s",
        "w1e",
        "w2",
        "w2s",
        "w2e",
    )
    NAMES32 = ("n", "me")
    NAMESF = ("sb", "sbr", "lb", "lbr", "lim")
    N64 = len(NAMES64)

    @flyc.kernel(name=kname, known_block_size=[NT, 1, 1])
    def kernel(
        x: fx.Int64,
        ids: fx.Int64,
        tw: fx.Int64,
        out: fx.Int64,
        sym: fx.Int64,
        stab: fx.Int64,
        ctl: fx.Int64,
        a2: fx.Int64,
        a2s: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w1e: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        w2e: fx.Int64,
        n: fx.Int32,
        me: fx.Int32,
        sb: fx.Float32,
        sbr: fx.Float32,
        lb: fx.Float32,
        lbr: fx.Float32,
        lim: fx.Float32,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds = lds_typed_ptr(lds_i32, T.i32)
        lds_f32 = lds_typed_ptr(lds_i32, T.f32)
        tx0 = fx.Int32(gpu.thread_id("x"))
        lane0 = tx0 % fx.Int32(64)
        wave0 = _u(tx0 // fx.Int32(64))
        bid = fx.Int32(gpu.block_id("x"))
        ctl0 = fx.Int64(ctl)
        sym0 = fx.Int64(sym)
        epoch = _u(_sys_load_at(ctl0 + fx.Int64(SC_EPOCH * 4))) + fx.Int32(1)

        def ctl_at(i):
            return ctl0 + fx.Int64(fx.Int32(i) * fx.Int32(4))

        def lane0_add(addr, value):
            """Atomic add by lane 0 only (old value in lane 0). The address goes through an
            opaque VGPR zero: at a uniform address LLVM's atomic optimizer would sum the
            lanes in a 64-step loop."""
            taken = fx.Int32(0)
            if lane0 == fx.Int32(0):
                zero = fx.Int32(
                    llvm.inline_asm(
                        T.i32, [], "v_mov_b32 $0, 0", "=v", has_side_effects=True
                    )
                )
                taken = _atomic_add_at(
                    fx.Int64(addr) + fx.Int64(zero), value, "agent-one-as"
                )
            return taken

        def to_all(v, slot):
            """wave 0's uniform ``v`` -> every wave (LDS word ``slot``)."""
            if wave0 == fx.Int32(0):
                _store(lds, fx.Int32(L_BC + slot), v)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            r = _u(lds[fx.Int32(L_BC + slot)])
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            return r

        def wait_flags(base, target):
            """wave 0: until the NPES i32 words at base + s * LINE all hold ``target``."""
            if wave0 == fx.Int32(0):
                addr = base + fx.Int64(
                    fx.min(lane0, fx.Int32(NPES - 1)) * fx.Int32(LINE)
                )
                seen = _sys_load_at(addr)
                pending = fx.Int64(
                    rocdl.ballot(T.i64, _raw(seen != target))
                ) != fx.Int64(0)
                while pending:
                    rocdl.s_sleep(2)
                    seen = _sys_load_at(addr)
                    pending = fx.Int64(
                        rocdl.ballot(T.i64, _raw(seen != target))
                    ) != fx.Int64(0)
            gpu.barrier()

        def reload():
            """Kernel args read again from the kernarg segment through an opaque base: with
            loop-invariant args LLVM hoists every unit kind's address math out of the unit
            loop and runs out of registers (as the main kernel)."""
            ptr4 = ir.Type.parse("!llvm.ptr<4>")
            seg = llvm.call_intrinsic(
                ptr4, "llvm.amdgcn.kernarg.segment.ptr", [], [], []
            )
            base = fx.Int64(
                llvm.inline_asm(
                    T.i64,
                    [llvm.ptrtoint(T.i64, seg)],
                    "s_mov_b64 $0, $1",
                    "=s,s",
                    has_side_effects=True,
                )
            )

            def load(off, ty, align):
                return llvm.LoadOp(
                    ty,
                    llvm.inttoptr(ptr4, (base + fx.Int64(off)).ir_value()),
                    alignment=align,
                ).result

            a = {k: fx.Int64(load(8 * i, T.i64, 8)) for i, k in enumerate(NAMES64)}
            a.update(
                {
                    k: fx.Int32(load(8 * N64 + 4 * j, T.i32, 4))
                    for j, k in enumerate(NAMES32)
                }
            )
            a.update(
                {
                    k: fx.Float32(load(8 * N64 + 4 * (len(NAMES32) + j), T.f32, 4))
                    for j, k in enumerate(NAMESF)
                }
            )
            return a

        # -------------------------------------------------------------- producers
        def load_row(x_rs, t, lane):
            """bf16 x row t: per lane CH x 2 x 16 B (elements c*1024 + lane*16 .. + 16)."""
            return [
                [
                    Vec(
                        _ld(
                            x_rs,
                            t * fx.Int32(H // 2)
                            + fx.Int32(c * 512 + h * 4)
                            + lane * fx.Int32(8),
                            4,
                        )
                    )
                    for h in range(2)
                ]
                for c in range(CH)
            ]

        def quant_row(raw, lane):
            """``load_row``'s chunks -> per lane: CH x 16 fp8 (i32x4) and, in lanes
            [0, H/128), the f32 scale of 128-col group ``lane`` (max(amax, 2^-117) / 240,
            1 for an all-zero group; as mega_quant_kernel); QX 0: ``quant_row_xrow``."""
            if const_expr(not QX):
                return quant_row_xrow(raw)
            qs, ss = [], []
            for c in range_constexpr(CH):
                vals = []
                for h in range_constexpr(2):
                    f = Vec(raw[c][h].bitcast(fx.BFloat16)).to(fx.Float32)
                    vals.extend(fx.Float32(f[i]) for i in range(8))
                am = fx.Int32(0)
                for v in vals:
                    am = fx.max(am, _abs_bits(v))
                for s in (1, 2, 4):  # 8 lanes = one 128-col group
                    am = fx.max(am, am.shuffle_xor(s, 64))
                sc = (am == 0).select(
                    fx.Float32(1.0), _amax_floor(am) / fx.Float32(240.0)
                )
                inv = fx.Float32(1.0) / sc
                qs.append(
                    Vec.from_elements(
                        [
                            _fp8x4_pack(*[vals[4 * j + i] * inv for i in range(4)])
                            for j in range(4)
                        ],
                        fx.Int32,
                    )
                )
                ss.append(sc.bitcast(fx.Int32))
            src = (lane % fx.Int32(8)) * fx.Int32(
                8
            )  # group g: chunk g // 8, lane (g % 8) * 8
            sv = fx.Int32(0)
            for c in range_constexpr(CH):
                sv = (lane // fx.Int32(8) == fx.Int32(c)).select(
                    _bcast_lane(ss[c], src), sv
                )
            return qs, sv

        def quant_row_xrow(raw):
            """QX 0: ``load_row``'s chunks -> per lane: CH x 16 fp8 (i32x4) and the row's f32
            scale (max(amax, 2^-117) / 240 over the row, 1 for an all-zero row) in every
            lane."""
            vals = []
            am = fx.Int32(0)
            for c in range_constexpr(CH):
                vc = []
                for h in range_constexpr(2):
                    f = Vec(raw[c][h].bitcast(fx.BFloat16)).to(fx.Float32)
                    vc.extend(fx.Float32(f[i]) for i in range(8))
                for v in vc:
                    am = fx.max(am, _abs_bits(v))
                vals.append(vc)
            for s in range_constexpr(6):
                am = fx.max(am, am.shuffle_xor(1 << s, 64))
            sc = (am == 0).select(fx.Float32(1.0), _amax_floor(am) / fx.Float32(240.0))
            inv = fx.Float32(1.0) / sc
            qs = [
                Vec.from_elements(
                    [
                        _fp8x4_pack(*[vals[c][4 * j + i] * inv for i in range(4)])
                        for j in range(4)
                    ],
                    fx.Int32,
                )
                for c in range(CH)
            ]
            return qs, sc.bitcast(fx.Int32)

        def produce(a, lane, wave):
            """Wave ``wave`` of producer ``bid``: tokens bid + P wave, + 8 P, ... (a small
            batch spreads over the producers: one CU's remote stores are slow)."""
            n_ = a["n"]
            tab_rs = _rsrc(a["stab"], NPES * 8)
            peer = (lane & fx.Int32(NPES - 1)) * fx.Int32(2)
            base_lo = fx.Int32(_ld(tab_rs, peer))  # lane: rank lane % 8's heap base
            base_hi = fx.Int32(_ld(tab_rs, peer + fx.Int32(1)))
            x_rs = _rsrc(a["x"], fx.Int64(n_) * fx.Int64(H * 2))
            ids_rs = _rsrc(a["ids"], fx.Int64(n_) * fx.Int64(TOPK * 4))
            tw_rs = _rsrc(a["tw"], fx.Int64(n_) * fx.Int64(TOPK * 4))
            kl = lane < fx.Int32(TOPK)
            t = bid + wave * fx.Int32(P)
            while t < n_:
                ri = t * fx.Int32(TOPK) + lane
                g = kl.select(fx.Int32(_ld(ids_rs, ri, mask=kl)), fx.Int32(-1))
                wbits = fx.Int32(_ld(tw_rs, ri, mask=kl))
                raw = load_row(
                    x_rs, t, lane
                )  # in flight during the atomic's round trip
                valid = (g >= fx.Int32(0)) & (g < fx.Int32(E_GLOBAL))
                gs = valid.select(g, fx.Int32(0))
                dest = gs // fx.Int32(E_LOCAL)
                e = gs % fx.Int32(E_LOCAL)
                lo = _bcast_lane(base_lo, dest)
                hi = _bcast_lane(base_hi, dest)
                slot = fx.Int32(0)
                if valid:
                    slot = _atomic_add_at(
                        _addr(lo, hi) + fx.Int64(O_RUN) + fx.Int64(e * fx.Int32(LINE)),
                        1,
                        "one-as",
                    )
                ok = (valid & (slot < fx.Int32(CAP))).select(fx.Int32(1), fx.Int32(0))
                row = e * fx.Int32(CAP) + slot
                qs, sv = quant_row(raw, lane)
                for k in range_constexpr(TOPK):
                    if _readlane(ok, k) != fx.Int32(0):
                        rk = fx.Int64(_readlane(row, k))
                        db = _addr(_readlane(lo, k), _readlane(hi, k))
                        rx_rs = _rsrc(db + fx.Int64(O_RX) + rk * fx.Int64(H), H)
                        for c in range_constexpr(CH):
                            buffer_store(
                                qs[c],
                                rx_rs,
                                fx.Int32(c * 256) + lane * fx.Int32(4),
                                cache_modifier=P2P,
                            )
                        buffer_store(
                            sv,
                            _rsrc(
                                db + fx.Int64(O_RS) + rk * fx.Int64(G1N * 4), G1N * 4
                            ),
                            lane,
                            mask=lane < fx.Int32(G1N if QX else 1),
                            cache_modifier=P2P,
                        )
                        sid = fx.Int32(k << 24) | (a["me"] << fx.Int32(16)) | t
                        first = lane == fx.Int32(0)
                        buffer_store(
                            _readlane(wbits, k),
                            _rsrc(db + fx.Int64(O_RW) + rk * fx.Int64(4), 4),
                            fx.Int32(0),
                            mask=first,
                            cache_modifier=P2P,
                        )
                        buffer_store(
                            sid,
                            _rsrc(db + fx.Int64(O_RID) + rk * fx.Int64(4), 4),
                            fx.Int32(0),
                            mask=first,
                            cache_modifier=P2P,
                        )
                t = t + fx.Int32(P * NW)

        # -------------------------------------------------------------- plan
        def plan():
            """wave 0: rows of each local expert -> LDS cnt[e], tb[e] (its first m-tile),
            tb[E_LOCAL] (m-tiles in all). Lane l takes experts l, l + 64, ...: a scan per
            64 experts, each pass offset by the m-tiles of the passes before."""
            if wave0 == fx.Int32(0):
                before = fx.Int32(0)
                for p in range_constexpr(PLAN_PASSES):
                    le = lane0 + fx.Int32(p * 64)
                    e = fx.min(le, fx.Int32(E_LOCAL - 1))
                    c = _sys_load_at(
                        sym0 + fx.Int64(O_RUN) + fx.Int64(e * fx.Int32(LINE))
                    )
                    c = (le < fx.Int32(E_LOCAL)).select(
                        fx.min(c, fx.Int32(CAP)), fx.Int32(0)
                    )
                    nt = (c + fx.Int32(BM - 1)) // fx.Int32(BM)
                    inc = nt
                    for bit in range_constexpr(6):
                        d = 1 << bit
                        other = _bcast_lane(
                            inc, fx.max(lane0 - fx.Int32(d), fx.Int32(0))
                        )
                        inc = inc + (lane0 >= fx.Int32(d)).select(other, fx.Int32(0))
                    if le < fx.Int32(E_LOCAL):
                        _store(lds, fx.Int32(L_CNT) + le, c)
                        _store(lds, fx.Int32(L_TB) + le, before + inc - nt)
                    if le == fx.Int32(E_LOCAL - 1):
                        _store(lds, fx.Int32(L_TB + E_LOCAL), before + inc)
                    if const_expr(p + 1 < PLAN_PASSES):
                        before = before + _readlane(inc, 63)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        def locate(tile):
            """m-tile -> (local expert, valid rows, its first fixed-slot row)."""
            e = fx.Int32(0)
            for k in range_constexpr(1, E_LOCAL):
                e = e + (lds[fx.Int32(L_TB + k)] <= tile).select(
                    fx.Int32(1), fx.Int32(0)
                )
            e = _u(e)
            j = tile - _u(lds[fx.Int32(L_TB) + e])
            rows = fx.min(_u(lds[fx.Int32(L_CNT) + e]) - j * fx.Int32(BM), fx.Int32(BM))
            return e, rows, e * fx.Int32(CAP) + j * fx.Int32(BM)

        def ratio_issue(base, nbytes, G, R, srow_of, tx):
            """Loads of the A scales of rows [0, R) (s[r][g], s[r][g-1]; ``srow_of(r)``:
            row r's first scale), issued for ``ratio_finish``."""
            tiles = _global_i32_buffer_tiles(base, nbytes, 1)
            atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            out = []
            for p in range_constexpr(-(-(R * G) // NT)):
                idx = tx + fx.Int32(p * NT)
                r = idx // fx.Int32(G)
                g = idx % fx.Int32(G)
                ok = r < fx.Int32(R)
                rr = ok.select(r, fx.Int32(0))
                srow = srow_of(rr)
                out.append(
                    (
                        rr,
                        g,
                        ok,
                        _buffer_i32_scalar_read(tiles, srow + g, atom),
                        _buffer_i32_scalar_read(
                            tiles, srow + fx.max(g - fx.Int32(1), fx.Int32(0)), atom
                        ),
                    )
                )
            return out

        def ratio_finish(loaded, G):
            """LDS: ratio[g][r] = s[r][g-1] / s[r][g] (ratio[0] = 1), fin[r] = s[r][G-1]; a
            zero scale (a padding row) acts as 1. Visible after the K loop's first barrier.
            """
            one = fx.Float32(1.0)
            for rr, g, ok, cur, prv in loaded:
                cur = cur.bitcast(fx.Float32)
                prv = prv.bitcast(fx.Float32)
                cur = (cur == fx.Float32(0.0)).select(one, cur)
                prv = (prv == fx.Float32(0.0)).select(one, prv)
                ratio = (g == fx.Int32(0)).select(one, prv / cur)
                _lds_f32_store_masked(
                    lds_f32, fx.Int32(RATIO // 4) + g * fx.Int32(BM) + rr, ratio, ok
                )
                _lds_f32_store_masked(
                    lds_f32,
                    fx.Int32(RATIO // 4 + G * BM) + rr,
                    cur,
                    ok & (g == fx.Int32(G - 1)),
                )

        # -------------------------------------------------------------- gemm1
        def gemm1_unit(a, u, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            tile = u // fx.Int32(NG)
            nb = u - tile * fx.Int32(NG)
            e, n_valid, row0 = locate(tile)
            arow = tile * fx.Int32(BM)
            col0 = nb * fx.Int32(NI) + wave * fx.Int32(NIW)
            cols = []
            for gu in range_constexpr(2):
                for c in range_constexpr(NC1):
                    n0 = e * fx.Int32(N2) + fx.Int32(gu * I) + col0 + fx.Int32(c * 16)
                    cols.append(((gu, c), n0, (gu, c // 2)))
            exps = {
                (gu, c, v): fx.Int32(
                    _global_i32_at(
                        a["w1e"],
                        e * fx.Int32(N2)
                        + fx.Int32(gu * I)
                        + col0
                        + fx.Int32(c * 16)
                        + lane_div_16 * fx.Int32(4)
                        + fx.Int32(v),
                    )
                )
                for gu in range(2)
                for c in range(NC1)
                for v in range(4)
            }
            rx = a["sym"] + fx.Int64(O_RX)
            core = _make_core(
                K=H,
                W_ROWS=E_LOCAL * N2,
                BM=BM,
                PF=PF,
                bcm=BCM1,
                cols=cols,
                lds_ptr=lds_ptr,
                LDS_BYTES=LDS_GEMM,
                lane=lane,
                tx=tx,
                NT=NT,
                arg_w=a["w1"],
                arg_ws=a["w1s"],
                arg_a=rx,
                a_num_bytes=fx.Int64(ROWS * H),
                ARB=ARB,
                acm=2,
                TR=True,
            )

            def row_base16(
                row,
            ):  # rows past the m-tile's valid ones read past the end: 0
                return (row < n_valid).select(
                    (row0 + row) * fx.Int32(H // 16), fx.Int32(ROWS * (H // 16))
                )

            def scale_row(rr):  # padding rows read past the end: 0 (acts as 1)
                return (rr < n_valid).select(
                    (row0 + rr) * fx.Int32(G1N), fx.Int32(ROWS * G1N)
                )

            def hook(acc, kt, s):
                kk = s * 32  # K offset of MFMA step s in the ARB-K tile
                if const_expr(kk % 128 != 0):
                    return
                gidx = kt * fx.Int32(ARB // 128) + fx.Int32(kk // 128)
                for key in acc:
                    for mi in range_constexpr(
                        len(acc[key])
                    ):  # TR: lane = row mi*16 + L%16
                        rt = lds_f32[
                            fx.Int32(RATIO // 4 + mi * 16)
                            + gidx * fx.Int32(BM)
                            + lane_mod_16
                        ]
                        a4 = Vec(acc[key][mi].load())
                        acc[key][mi].store(
                            Vec.from_elements(
                                [a4[v] * rt for v in range(4)], fx.Float32
                            )
                        )

            def epilogue(R, acc):
                """dequant -> act(gate)*up in registers -> row amax over the lane's columns,
                the 4 lane groups and the NW waves (LDS [row][NW]) -> fp8 a2 + f32 scale per
                (row, 128 cols), written through."""
                MR = R // 16
                rows = [fx.Int32(mi * 16) + lane_mod_16 for mi in range(MR)]
                valid = [r < n_valid for r in rows]
                fac = {k: _channel_factor(v) for k, v in exps.items()}
                rsc = [
                    fx.Float32(lds_f32[fx.Int32(RATIO // 4 + G1N * BM) + r])
                    for r in rows
                ]
                situ = None
                if const_expr(ACT != "silu"):
                    situ = situ_params(*[fx.Float32(a[k]) for k in NAMESF])
                ys = {}
                for mi in range_constexpr(MR):
                    for c in range_constexpr(NC1):
                        g4 = Vec(fx.memref_load_vec(acc[0, c][mi]))
                        u4 = Vec(fx.memref_load_vec(acc[1, c][mi]))
                        gv = [
                            fx.Float32(g4[v]) * (rsc[mi] * fac[0, c, v])
                            for v in range(4)
                        ]
                        uv = [
                            fx.Float32(u4[v]) * (rsc[mi] * fac[1, c, v])
                            for v in range(4)
                        ]
                        ys[mi, c] = gate_up_act(ACT, gv, uv, situ)
                col = (
                    nb * fx.Int32(NI) + wave * fx.Int32(NIW) + lane_div_16 * fx.Int32(4)
                )
                for mi in range_constexpr(MR):
                    am = fx.Int32(0)
                    for c in range_constexpr(NC1):
                        for v in range_constexpr(4):
                            am = fx.max(am, _abs_bits(ys[mi, c][v]))
                    am = fx.max(am, am.shuffle_xor(16, 64))
                    am = fx.max(am, am.shuffle_xor(32, 64))
                    lds[rows[mi] * fx.Int32(NW) + wave] = am  # 4 lanes, same value
                gpu.barrier()
                out_rs = _rsrc(a["a2"], fx.Int64(arow + fx.Int32(BM)) * fx.Int64(I))
                os_rs = _rsrc(
                    a["a2s"], fx.Int64(arow + fx.Int32(BM)) * fx.Int64(G2N * 4)
                )
                grp = wave // fx.Int32(WPG)
                for mi in range_constexpr(MR):
                    pv = Vec(
                        lds_vec_load(
                            lds_i32,
                            (rows[mi] * fx.Int32(NW) + grp * fx.Int32(WPG))
                            * fx.Int32(4),
                            Vec.make_type(WPG, fx.Int32),
                            fx.Int32,
                            align=4 * WPG,
                        )
                    )
                    am = fx.Int32(pv[0])
                    for j in range_constexpr(1, WPG):
                        am = fx.max(am, fx.Int32(pv[j]))
                    s = (am == 0).select(
                        fx.Float32(1.0), _amax_floor(am) * fx.Float32(1.0 / 240.0)
                    )
                    inv = fx.Float32(rocdl.rcp(T.f32, _raw(s)))
                    srow = arow + rows[mi]
                    for c in range_constexpr(NC1):
                        q = _fp8x4_pack(*[ys[mi, c][v] * inv for v in range(4)])
                        buffer_store(
                            q,
                            out_rs,
                            (srow * fx.Int32(I) + col + fx.Int32(c * 16))
                            // fx.Int32(4),
                            mask=valid[mi],
                            cache_modifier=WT,
                        )
                    buffer_store(
                        s,
                        os_rs,
                        srow * fx.Int32(G2N) + nb * fx.Int32(GPW) + grp,
                        mask=valid[mi]
                        & (lane_div_16 == fx.Int32(0))
                        & (wave % fx.Int32(WPG) == fx.Int32(0)),
                        cache_modifier=WT,
                    )

            def run_rows(R):
                loaded = []

                def pre_a():  # the W prefetch is in flight
                    if const_expr(QX):
                        loaded.extend(
                            ratio_issue(
                                a["sym"] + fx.Int64(O_RS),
                                ROWS * G1N * 4,
                                G1N,
                                R,
                                scale_row,
                                tx,
                            )
                        )
                    else:  # the rows' x scales (QX 0: the first of a row's G1N slots)
                        rs_rs = _rsrc(a["sym"] + fx.Int64(O_RS), ROWS * G1N * 4)
                        loaded.append(
                            fx.Int32(_ld(rs_rs, scale_row(tx), mask=tx < n_valid))
                        )

                def post_a():  # the first A tile is in flight too
                    if const_expr(QX):
                        ratio_finish(loaded, G1N)
                    else:  # LDS fin[r]: row r's scale (padding rows: 1)
                        sc = (tx < n_valid).select(
                            loaded[0].bitcast(fx.Float32), fx.Float32(1.0)
                        )
                        _lds_f32_store_masked(
                            lds_f32,
                            fx.Int32(RATIO // 4 + G1N * BM) + tx,
                            sc,
                            tx < fx.Int32(R),
                        )

                acc = core(
                    R, row_base16, hook if QX else None, pre_a=pre_a, post_a=post_a
                )
                epilogue(R, acc)
                rocdl.s_waitcnt(vmcnt=0)  # a2 written through
                gpu.barrier()
                if wave == fx.Int32(0):
                    lane0_add(ctl_at(fx.Int32(SC_TILE) + tile * fx.Int32(16)), 1)

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # -------------------------------------------------------------- gemm2
        def gemm2_unit(a, v, tx, lane, wave):
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            tile = v // fx.Int32(ND)
            nb = v - tile * fx.Int32(ND)
            e, n_valid, row0 = locate(tile)
            arow = tile * fx.Int32(BM)
            col0 = nb * fx.Int32(NO) + wave * fx.Int32(NOW)
            cols = [
                ((0, c), e * fx.Int32(H) + col0 + fx.Int32(c * 16), (0, c // 2))
                for c in range(NC2)
            ]
            exps = [
                fx.Int32(
                    _global_i32_at(
                        a["w2e"],
                        e * fx.Int32(H) + col0 + fx.Int32(c * 16) + lane_mod_16,
                    )
                )
                for c in range(NC2)
            ]
            core = _make_core(
                K=I,
                W_ROWS=E_LOCAL * H,
                BM=BM,
                PF=PF,
                bcm=BCM2,
                cols=cols,
                lds_ptr=lds_ptr,
                LDS_BYTES=LDS_GEMM,
                lane=lane,
                tx=tx,
                NT=NT,
                arg_w=a["w2"],
                arg_ws=a["w2s"],
                arg_a=a["a2"],
                a_num_bytes=fx.Int64(A2_ROWS * I),
                ARB=ARB,
                acm=2,
                TR=False,
            )

            def row_base16(row):
                return (arow + row) * fx.Int32(I // 16)

            def scale_row(rr):
                return (arow + rr) * fx.Int32(G2N)

            def ratio4(base_f32, mi):
                """this lane's 4 rows (L//16)*4 + v of 16-row block mi from an f32 [BM] row."""
                off = (
                    base_f32 + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                ) * fx.Int32(4)
                return Vec(
                    lds_vec_load(
                        lds_i32, off, Vec.make_type(4, fx.Float32), fx.Float32, align=16
                    )
                )

            def hook(acc, kt, s):
                kk = s * 32
                if const_expr(kk % 128 != 0):
                    return
                gidx = kt * fx.Int32(ARB // 128) + fx.Int32(kk // 128)
                for mi in range_constexpr(len(acc[0, 0])):
                    rt = ratio4(fx.Int32(RATIO // 4) + gidx * fx.Int32(BM), mi)
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        acc[0, c][mi].store(
                            Vec.from_elements(
                                [a4[v] * rt[v] for v in range(4)], fx.Float32
                            )
                        )

            def route_loads(R):
                """Issued before the m-tile wait: the weights of this lane's epilogue rows
                (L//16)*4 + v of each 16-row block, the source ids of this wave's scatter rows
                (w, w + NW, ...); producers wrote them before the plan."""
                sw_tiles = _global_i32_buffer_tiles(
                    a["sym"] + fx.Int64(O_RW), ROWS * 4, 1
                )
                sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
                wts = [
                    [
                        _buffer_i32_scalar_read(
                            sw_tiles,
                            row0 + fx.Int32(mi * 16 + v) + lane_div_16 * fx.Int32(4),
                            sw_atom,
                        )
                        for v in range(4)
                    ]
                    for mi in range(R // 16)
                ]
                rid_rs = _rsrc(a["sym"] + fx.Int64(O_RID), ROWS * 4)
                sids = [
                    fx.Int32(_ld(rid_rs, row0 + fx.Int32(p * NW) + wave))
                    for p in range(-(-R // NW))
                ]
                return wts, sids

            def scatter(R, sids):
                """bf16 LDS tile [R, NO] -> row r into cin[token, slot] of its source rank:
                wave w sends rows w, w + NW, ... (64 lanes x VW i32 per row; a padding row's
                stores fall outside an empty range)."""
                for p in range_constexpr(-(-R // NW)):
                    r = fx.Int32(p * NW) + wave
                    sid = _u(sids[p])
                    src = (sid >> fx.Int32(16)) & fx.Int32(NPES - 1)
                    slot = (sid & fx.Int32(0xFFFF)) * fx.Int32(TOPK) + (
                        (sid >> fx.Int32(24)) & fx.Int32(0xFF)
                    )
                    base = _addr(
                        lds[fx.Int32(L_PEER) + src * fx.Int32(2)],
                        lds[fx.Int32(L_PEER + 1) + src * fx.Int32(2)],
                    )
                    dst = (
                        base
                        + fx.Int64(O_CIN)
                        + (fx.Int64(slot) * fx.Int64(H) + fx.Int64(nb * fx.Int32(NO)))
                        * fx.Int64(2)
                    )
                    pk = lds_vec_load(
                        lds_i32,
                        (r * fx.Int32(NO) + lane * fx.Int32(2 * VW)) * fx.Int32(2),
                        Vec.make_type(VW, fx.Int32) if VW > 1 else T.i32,
                        fx.Int32,
                        align=4 * VW,
                    )
                    buffer_store(
                        pk,
                        _rsrc(dst, (r < n_valid).select(fx.Int32(NO * 2), fx.Int32(0))),
                        lane * fx.Int32(VW),
                        cache_modifier=P2P,
                    )

            def epilogue(R, acc, wts_raw, sids):
                """acc * fin * channel factor * route weight -> bf16 LDS tile [R, NO] -> scatter."""
                MR = R // 16
                fac = [_channel_factor(x) for x in exps]
                lds_bf16 = lds_typed_ptr(lds_i32, T.bf16)
                vals = {}
                for mi in range_constexpr(MR):
                    fin = ratio4(fx.Int32(RATIO // 4 + G2N * BM), mi)
                    wts = [
                        wts_raw[mi][v].bitcast(fx.Float32) * fin[v] for v in range(4)
                    ]
                    for c in range_constexpr(NC2):
                        a4 = Vec(acc[0, c][mi].load())
                        vals[mi, c] = [a4[v] * (wts[v] * fac[c]) for v in range(4)]
                gpu.barrier()  # ratio table / A region free for the staging tile
                for mi in range_constexpr(MR):
                    rows4 = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                    for c in range_constexpr(NC2):
                        col = wave * fx.Int32(NOW) + fx.Int32(c * 16) + lane_mod_16
                        bf4 = Vec.from_elements(vals[mi, c], fx.Float32).to(fx.BFloat16)
                        for r in range_constexpr(4):
                            lds_bf16[(rows4 + fx.Int32(r)) * fx.Int32(NO) + col] = bf4[
                                r
                            ]
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                scatter(R, sids)

            def run_rows(R):
                routes, loaded = [], []

                def pre_a():  # the W prefetch is in flight
                    routes.extend(route_loads(R))
                    if wave == fx.Int32(0):
                        cnt = ctl_at(fx.Int32(SC_TILE) + tile * fx.Int32(16))
                        seen = _atomic_load_at(cnt, "agent-one-as")
                        while seen < fx.Int32(NG):
                            rocdl.s_sleep(2)
                            seen = _atomic_load_at(cnt, "agent-one-as")
                        _inv_l1()
                    gpu.barrier()
                    loaded.extend(
                        ratio_issue(a["a2s"], A2_ROWS * G2N * 4, G2N, R, scale_row, tx)
                    )

                def post_a():  # the first A tile is in flight too
                    ratio_finish(loaded, G2N)

                acc = core(R, row_base16, hook, pre_a=pre_a, post_a=post_a)
                epilogue(R, acc, *routes)

            _dispatch_rows(ROWV, [n_valid <= fx.Int32(r) for r in ROWV[:-1]], run_rows)

        # -------------------------------------------------------------- units
        def run_unit(a, i, n1, b1, b2):
            tx, lane, wave = _opaque_tx(tx0)
            if i < n1:
                gemm1_unit(a, b1 + i, tx, lane, wave)
            else:
                gemm2_unit(a, b2 + i - n1, tx, lane, wave)

        def produce_if(a):
            if bid < fx.Int32(P):
                _tx, lane, wave = _opaque_tx(tx0)
                produce(a, lane, wave)
                rocdl.s_waitcnt(vmcnt=0)  # this wave's P2P stores landed
                gpu.barrier()
                # this producer's share of done[me] on every rank (nested runtime ifs:
                # FlyDSL lowers each to its own branch, as the kernel was validated)
                if wave0 == fx.Int32(0):  # noqa: SIM102
                    if lane0 < fx.Int32(NPES):
                        tab = global_typed_ptr(a["stab"], T.i64, align=8)
                        _atomic_add_at(
                            fx.Int64(tab[lane0])
                            + fx.Int64(O_DONE)
                            + fx.Int64(a["me"] * fx.Int32(LINE)),
                            DONE_UNIT // P,
                            "one-as",
                        )

        def finish(a, units_t):
            """count this workgroup out; the last one resets the counters and tells every
            rank that its rows are sent."""
            top = fx.Int32(-1)
            if wave0 == fx.Int32(0):
                g = bid % fx.Int32(8)
                n_in_g = (fx.Int32(GRID + 7) - g) // fx.Int32(8)
                old = _u(lane0_add(ctl_at(fx.Int32(SC_SUB) + g * fx.Int32(16)), 1))
                if old == n_in_g - fx.Int32(1):
                    top = _u(lane0_add(ctl_at(SC_TOP), 1))
            if to_all(top, 2) == fx.Int32(7):
                for i in range(tx0, units_t, NT):
                    _sys_store_at(
                        ctl_at(fx.Int32(SC_TILE) + i * fx.Int32(16)), fx.Int32(0)
                    )
                if tx0 < fx.Int32(8):
                    _sys_store_at(
                        ctl_at(fx.Int32(SC_HEAD) + tx0 * fx.Int32(16)), fx.Int32(0)
                    )
                    _sys_store_at(
                        ctl_at(fx.Int32(SC_SUB) + tx0 * fx.Int32(16)), fx.Int32(0)
                    )
                if tx0 < fx.Int32(E_LOCAL):
                    _sys_store_at(
                        sym0 + fx.Int64(O_RUN) + fx.Int64(tx0 * fx.Int32(LINE)),
                        fx.Int32(0),
                    )
                if tx0 == fx.Int32(0):
                    _sys_store_at(ctl_at(SC_TOP), fx.Int32(0))
                    _sys_store_at(ctl_at(SC_EPOCH), epoch)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tx0 < fx.Int32(NPES):
                    tab = global_typed_ptr(a["stab"], T.i64, align=8)
                    _sys_store_at(
                        fx.Int64(tab[tx0])
                        + fx.Int64(O_CFLAG)
                        + fx.Int64(a["me"] * fx.Int32(LINE)),
                        epoch,
                    )

        def combine(a):
            """this rank's tokens b, b + GRID, ...: out[t] = sum of its valid slots' rows.
            Workgroup 0 waits for every rank's flag even without tokens: a call must not
            end before all ranks finished it, or this rank's next call adds to ``done``
            on a rank whose plan still waits for this call's count."""
            if bid < fx.max(a["n"], fx.Int32(1)):
                wait_flags(sym0 + fx.Int64(O_CFLAG), epoch)
                cin_rs = _rsrc(sym0 + fx.Int64(O_CIN), TMAX * TOPK * H * 2)
                out_rs = _rsrc(a["out"], fx.Int64(a["n"]) * fx.Int64(H * 2))
                ids_p = global_typed_ptr(a["ids"], T.i32)
                t = bid
                while t < a["n"]:
                    for p in range_constexpr(-(-UNITS // NT)):
                        # this thread's 16 B of the row; a pass past its end is masked
                        u = tx0 if p == 0 else tx0 + fx.Int32(p * NT)
                        full = (p + 1) * NT <= UNITS
                        acc = Vec.filled(8, 0.0, fx.Float32)
                        for k in range_constexpr(TOPK):
                            g = _u(fx.Int32(ids_p[t * fx.Int32(TOPK) + fx.Int32(k)]))
                            ok = (g >= fx.Int32(0)) & (g < fx.Int32(E_GLOBAL))
                            if const_expr(not full):
                                ok = ok & (u < fx.Int32(UNITS))
                            v = Vec(
                                buffer_load(
                                    cin_rs,
                                    (t * fx.Int32(TOPK) + fx.Int32(k))
                                    * fx.Int32(H // 2)
                                    + u * fx.Int32(4),
                                    vec_width=4,
                                    dtype=T.i32,
                                    mask=ok,
                                    cache_modifier=CM_SYS,
                                )
                            )
                            acc = acc + v.bitcast(fx.BFloat16).to(fx.Float32)
                        if const_expr(full):
                            buffer_store(
                                acc.to(fx.BFloat16).bitcast(fx.Int32),
                                out_rs,
                                t * fx.Int32(H // 2) + u * fx.Int32(4),
                            )
                        else:
                            buffer_store(
                                acc.to(fx.BFloat16).bitcast(fx.Int32),
                                out_rs,
                                t * fx.Int32(H // 2) + u * fx.Int32(4),
                                mask=u < fx.Int32(UNITS),
                            )
                    t = t + fx.Int32(GRID)

        a0 = reload()
        if tx0 < fx.Int32(2 * NPES):  # every rank's heap base -> LDS (gemm2's scatter)
            _store(
                lds,
                fx.Int32(L_PEER) + tx0,
                fx.Int32(_ld(_rsrc(a0["stab"], NPES * 8), tx0)),
            )
        produce_if(a0)
        wait_flags(sym0 + fx.Int64(O_DONE), epoch * fx.Int32(DONE_UNIT))
        plan()
        tiles = _u(lds[fx.Int32(L_TB + E_LOCAL)])
        home = _u(_home_queue(XCDS))
        u1 = tiles * fx.Int32(NG)
        u2 = tiles * fx.Int32(ND)
        # slice c (XCD c's on 8 XCDs): gemm1 units [c u1 / 8, (c + 1) u1 / 8), then gemm2
        # units [c u2 / 8, (c + 1) u2 / 8) on its counter; the own slice first, then the
        # others' leftovers (a plain load first: an atomic per leftover workgroup slows the
        # units still running)
        q = fx.Int32(0)
        while q < fx.Int32(8):
            cand = (home + q) & fx.Int32(7)
            b1 = cand * u1 // fx.Int32(8)
            n1 = (cand + fx.Int32(1)) * u1 // fx.Int32(8) - b1
            b2 = cand * u2 // fx.Int32(8)
            nall = n1 + (cand + fx.Int32(1)) * u2 // fx.Int32(8) - b2
            head = ctl_at(fx.Int32(SC_HEAD) + cand * fx.Int32(16))
            got = nall
            if wave0 == fx.Int32(0):
                take = q == fx.Int32(0)  # the own slice: no probe
                if q != fx.Int32(0):
                    take = _u(_atomic_load_at(head, "agent-one-as")) < nall
                if take:
                    got = _u(lane0_add(head, 1))
            i = to_all(got, 0)
            if i < nall:
                run_unit(reload(), i, n1, b1, b2)
                gpu.barrier()  # the unit's LDS reads are done
            else:
                q = q + fx.Int32(1)
        rocdl.s_waitcnt(vmcnt=0)  # this workgroup's rows are sent
        gpu.barrier()
        a1 = reload()
        finish(a1, tiles)
        combine(a1)

    @flyc.jit
    def launch(
        x: fx.Int64,
        ids: fx.Int64,
        tw: fx.Int64,
        out: fx.Int64,
        sym: fx.Int64,
        stab: fx.Int64,
        ctl: fx.Int64,
        a2: fx.Int64,
        a2s: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w1e: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        w2e: fx.Int64,
        n: fx.Int32,
        me: fx.Int32,
        sb: fx.Float32,
        sbr: fx.Float32,
        lb: fx.Float32,
        lbr: fx.Float32,
        lim: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            x,
            ids,
            tw,
            out,
            sym,
            stab,
            ctl,
            a2,
            a2s,
            w1,
            w1s,
            w1e,
            w2,
            w2s,
            w2e,
            n,
            me,
            sb,
            sbr,
            lb,
            lbr,
            lim,
            value_attrs={"rocdl.waves_per_eu": 2 * WPC},
        ).launch(grid=(GRID, 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =============================================================================
# host
# =============================================================================

# Kernel keys: m-tile rows, gemm2 cols per unit, A bytes per K tile, W prefetch depth,
# producer workgroups, workgroups per CU, gemm1 / gemm2 W cache modifier. Only BM moves
# the time beyond +-2% (8x MI300X sweep of each key).
SMALL_KEY = {
    "BM": 64,
    "NO": 256,
    "ARB": 256,
    "PF": 2,
    "P": 16,
    "WPC": 1,
    "BCM1": 2,
    "BCM2": 2,
}


class SmallMRuntime:
    """The small path of one rank: ONE kernel per call for batches of at most ``tmax``
    tokens over all ranks (``tmax`` rows of fixed slots per local expert, the combine
    ``tmax`` x ``topk`` rows: ~196 MB of the MORI heap at tmax 1024, H 4096, top-8).
    Negative ids are no routes (their rows stay 0). ``num_cu`` / ``xcds``: CUs and XCDs
    of the GPU (the kernel's grid has ``WPC`` workgroups per CU)."""

    @staticmethod
    def check_capacity(*, world_size, model_dim, inter_dim, experts, topk, tmax):
        """Refuses a ``tmax`` whose buffers the kernel cannot address: its fixed slots
        (``rx``, gemm1's A through one buffer descriptor), the combine rows (``cin``) and
        gemm2's A (``a2``) each stay below 2 GiB (i32 byte offsets, and a masked lane's
        offset 0x7FFFFFFF must be out of range), and tmax <= 2^16."""
        bm = SMALL_KEY["BM"]
        H, I, topk = int(model_dim), int(inter_dim), int(topk)
        e_local = int(experts) // int(world_size)

        def too_big(tmax):
            a2_rows = max(
                small_max_tiles(topk, tmax, b, e_local) * b for b in (16, 32, 64)
            )
            sizes = {
                "rx": e_local * tmax * H,
                "cin": tmax * topk * H * 2,
                "a2": a2_rows * I,
            }
            return {k: v for k, v in sizes.items() if v >= BUFFER_BYTES_MAX}

        tmax = -(-int(tmax) // bm) * bm
        big = too_big(tmax)
        if big or tmax > 1 << 16:
            fits = min(tmax, 1 << 16)
            while fits and too_big(fits):
                fits -= bm
            raise ValueError(
                f"small_M at {tmax // world_size} tokens per rank: "
                + ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in big.items())
                + f"; the kernel addresses buffers below 2 GiB, at most "
                f"{fits // world_size} tokens per rank here"
            )

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        tmax,
        act,
        x_quant,
        num_cu,
        xcds,
    ):
        from . import host
        from .protocol import SC_TILE, qx_of

        self.check_capacity(
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tmax=tmax,
        )
        # An expert's fixed slots are whole m-tiles of BM rows: tmax rounds up to them
        # (the caller's limit on the tokens of a call stays tmax).
        bm = SMALL_KEY["BM"]
        tmax = -(-int(tmax) // bm) * bm
        self.rank, self.topk, self.tmax = int(rank), int(topk), tmax
        H, I = int(model_dim), int(inter_dim)
        e_local = int(experts) // int(world_size)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.sym = host.symmetric(
            (small_sym_layout(H, topk, tmax, e_local, int(world_size))[1],), torch.uint8
        )
        torch.cuda.synchronize()
        import mori.shmem as ms

        ms.shmem_barrier_all()
        self.stab = host.peer_table(self.sym, self.rank, world_size)
        bms = (16, 32, 64)  # a2 rows / tile counters for every m-tile size
        rows = max(small_max_tiles(topk, tmax, bm, e_local) * bm for bm in bms)
        tiles = max(small_max_tiles(topk, tmax, bm, e_local) for bm in bms)
        self.ctl = torch.zeros(SC_TILE + 16 * tiles, dtype=torch.int32, device=dev)
        self.a2 = torch.empty((rows, I), dtype=torch.uint8, device=dev)
        self.a2s = torch.empty((rows, I // 128), dtype=torch.float32, device=dev)
        self.out = torch.empty((tmax, H), dtype=torch.bfloat16, device=dev)
        self.kernel = host.launcher(
            ep_mega_small_kernel,
            dev.index,
            H=H,
            I=I,
            TOPK=self.topk,
            E_LOCAL=e_local,
            TMAX=self.tmax,
            NUM_CU=int(num_cu),
            XCDS=int(xcds),
            NPES=int(world_size),
            ACT=act,
            QX=qx_of(x_quant),
            **SMALL_KEY,
        )

    def forward(self, x, ids, wts, W, situ):
        """x bf16 [T, H], ids int32 [T, topk], wts f32 [T, topk]; ``W`` the weights
        dict, ``situ`` the five activation scalars. Returns bf16 [T, H], a view of the
        output buffer (valid until the next call)."""
        from . import host

        T = x.shape[0]
        host.launch(
            self.kernel,
            x.contiguous(),
            ids,
            wts,
            self.out,
            self.sym,
            self.stab,
            self.ctl,
            self.a2,
            self.a2s,
            W["w1"],
            W["w1_scale"],
            W["w1_exp"],
            W["w2"],
            W["w2_scale"],
            W["w2_exp"],
            T,
            self.rank,
            *situ,
        )
        return self.out[:T]
