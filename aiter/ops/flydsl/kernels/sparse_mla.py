# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA forward: bf16, H=16, D=512.

Each wave owns one query token and the full D=512 of its output, so waves
never exchange partial scores or synchronize. An unsplit workgroup runs four
waves on four tokens. A split-K launch runs one wave per (token, split) and
writes FP32 partials that ``compile_sparse_mla_combine`` merges.

Per 16-key tile a wave:

* gathers the KV rows into one of two fixed register slots, four adjacent
  lanes per 1 KiB row, so every gather has a full tile of compute to land;
* moves the gathered dwords into MFMA operand order with ``ds_bpermute``,
  interleaved with the previous tile's PV MFMAs and this tile's QK MFMAs
  (PV first, which keeps the LDS FIFO ahead of the QK dependency);
* computes S^T = K Q^T (token on M, head on N): each lane owns one head, the
  softmax reduces across lanes 16 and 32 apart, and P^T is already the PV B
  operand. The online softmax rescales only when the running max grows by
  more than RESCALE_TAU (log2 units);
* accumulates out^T = V^T P^T, with V^T from one of two routes:

  - LDS (default, and every split launch): each wave stores its tile to a
    private, swizzled 16 KiB LDS region and reads V^T back.
  - register (``register_vt``, unsplit only): an identity MFMA transposes the
    K operand in registers and ``v_perm`` packs its bf16 halves; no LDS.
    It is faster when each row's slots ascend through the KV pool and slower
    on rows in top-k score order, where it raises VMEM issue stalls.

One coalesced 32-index fetch serves two tiles through lane transfers. Tile
counts round up to pairs; out-of-range tiles and -1 slots have zero weight.

``v_mfma_f32_16x16x16_bf16`` fragment map:

    C[i] = (A @ B^T)[lane_k + i, lane_row]
    lane_row = lane % 16, lane_k = (lane // 16) * 4

``compile_sparse_mla_order_indices`` builds the optional pass that sorts
each [T,2048] index row by 16-row physical KV tile for the register route.

Credits: derived from Zhuo Su's MIT-licensed kernel in hellozhuo-amd/aiter,
4d33e0e171f8c9e8a1e2388c0a3fd3ffa1e408a2 through
e569a2290f25561ff80bed2f0634a517e8d96a0b: the V^T/P^T MFMA mapping,
wave-private LDS V^T, V^T from identity MFMAs, and adjacent-lane row gathers
moved into MFMA order with ``ds_bpermute`` interleaved with the MFMAs. The
S^T layout, lazy rescale and static-layout helpers come from Amir Ghamarian's
earlier gfx942 kernel.
"""

# NOTE: do NOT add `from __future__ import annotations`: fx.struct reads its
# field types from raw __annotations__, which PEP 563 turns into strings.

from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E, atomic_add_i32
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_atom,
)
from aiter.ops.flydsl.kernels.topk.radix_topk_one_block import (
    _build_bitonic_schedule,
)

WAVE = 64
HEAD_DIM = 512
NUM_HEADS = 16
VEC = 8
NEG_INF = float("-inf")
RESCALE_TAU = 8.0
TOKENS_PER_WORKGROUP = 4
BLOCK_K = 16
INDEX_FORMATS = ("dense", "csr", "dense_lengths")
# Identity MFMAs issued ahead of their V^T pack; 2 adds a loop-top vmcnt(0).
IDENTITY_MFMA_LEAD = 1


def _mfma_ssa(mma, a, b, acc):
    out = fly_dialect.mma_atom_call_ssa(
        [fx.Vector.make_type(4, fx.Float32)],
        mma,
        a,
        b,
        acc,
    )
    if isinstance(out, (list, tuple)):
        out = out[0]
    return fx.Vector(out)


def _element(acc4, i):
    return fx.Vector(acc4)[i]


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _to_bf16(v):
    bits = v.bitcast(fx.Uint32) + fx.Uint32(0x00008000)
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _scale_acc(acc4, alphas):
    return fx.Vector.from_elements(
        [_element(acc4, i) * alphas[i] for i in range_constexpr(4)],
        fx.Float32,
    )


def _head_max(m, head_xor):
    # max over the four lanes (one per k-group) that hold one head
    for sh in head_xor:
        peer = m.shuffle_xor(sh, WAVE)
        m = (m > peer).select(m, peer)
    return m


def _rescale_all(acc, alpha):
    return [_scale_acc(a, [alpha] * 4) for a in acc]


# Trace-time layout helpers: loops walk static layouts, never index arithmetic.
def _static_int(x):
    """A static layout value (index, coordinate, extent) as Python ints."""
    return x.to_py_value()


def _mode_extent(layout, mode):
    """Extent of one top-level mode of a static layout."""
    return _static_int(fx.size(fx.get_(fx.get_shape(layout), mode)))


def _rowmajor(*shape):
    return fx.make_ordered_layout(shape, tuple(reversed(range(len(shape)))))


def _colmajor(*shape):
    return fx.make_ordered_layout(shape, tuple(range(len(shape))))


def _halves(v8, halves):
    # 8 bf16 -> MFMA operands for tiles 2p (low half) and 2p+1 (high half); halves: (e, half) -> element
    v8 = fx.Vector(v8)
    return [
        fx.Vector.from_elements(
            [
                v8[_static_int(fx.crd2idx((e, h), halves))]
                for e in range_constexpr(_mode_extent(halves, 0))
            ],
            fx.BFloat16,
        )
        for h in range_constexpr(_mode_extent(halves, 1))
    ]


def _bpermute_mfma_schedule(pv_blocks, qk_pairs, halves):
    """Interleave dword lane transfers with PV/QK; satisfy each QK dependency."""
    acts, bi, nbp = [], 0, 4 * qk_pairs
    mm = [("pv",) + b for b in pv_blocks]
    mm += [("qk", pr, h) for pr in range(qk_pairs) for h in range(halves)]
    for it in mm:
        if bi < nbp:
            acts.append(("bp", bi))
            bi += 1
        need = 0 if it[0] == "pv" else 4 * it[1] + 4
        while bi < need:
            acts.append(("bp", bi))
            bi += 1
        acts.append(it)
    while bi < nbp:
        acts.append(("bp", bi))
        bi += 1
    return acts


@lru_cache(maxsize=64)
def compile_sparse_mla_fwd(
    has_sink=False, index_format="dense", split_k=1, register_vt=False
):
    """Compile one specialization of the sparse MLA forward kernel.

    index_format: "dense" [T,K] rows, "csr" flat rows with indptr [T+1], or
    "dense_lengths" [T,K] rows limited by topk_length [T]. -1 slots are
    ignored in every format.
    split_k > 1 launches grid [T,S] with one wave per workgroup and writes
    FP32 partials and (max, sum) stats; otherwise four tokens share a
    workgroup and the kernel writes bf16 output. register_vt selects the
    register V^T route for unsplit launches; split launches ignore it.
    """
    if index_format not in INDEX_FORMATS:
        raise ValueError(f"index_format must be one of {INDEX_FORMATS}")
    CSR = index_format == "csr"
    LENGTHS = index_format == "dense_lengths"
    SPLIT = split_k > 1
    REGISTER_VT = bool(register_vt) and not SPLIT
    LDS_VT = not REGISTER_VT
    NUM_WAVES = 1 if SPLIT else TOKENS_PER_WORKGROUP
    BLOCK_THREADS = NUM_WAVES * WAVE
    HAS_SINK = bool(has_sink) and not SPLIT
    vt_route = "register_vt" if REGISTER_VT else "lds_vt"
    kernel_name = (
        f"sparse_mla_fwd_{index_format}_{vt_route}"
        f"_sink{int(HAS_SINK)}_split{split_k}"
    )
    UNBOUNDED = 1 << 24
    CHUNKS_PER_ROW = HEAD_DIM // VEC
    WAVE_REGION = BLOCK_K * HEAD_DIM  # elements of one wave's private LDS region

    def _lds_load(view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), view, r)
        return r.load()

    def _lds_store(vec, view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        r.store(vec)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), r, view)

    def _swizzle(layout):
        # the 16-B chunk (offset bits 3..5) XORed with row & 7 (row pitch 512: bits 9..11)
        return fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 6)), layout)

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def sparse_mla_fwd_kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        index_width: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
        indptr_ptr: fx.Pointer,
        lengths_ptr: fx.Pointer,
        stats_ptr: fx.Pointer,
    ):
        # ---- static tilings (full D per wave)
        n_d_tiles = _mode_extent(
            fx.zipped_divide(fx.make_layout(HEAD_DIM, 1), (16,)), 1
        )
        qk_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (2,))
        pv_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (4,))
        QK_PAIRS = _mode_extent(qk_tiles, 1)
        PV_GROUPS = _mode_extent(pv_tiles, 1)
        HALVES = _mode_extent(qk_tiles, 0)
        H4 = _mode_extent(pv_tiles, 0)
        halves = fx.zipped_divide(fx.make_layout(VEC, 1), (H4,))
        acc_l = _colmajor(H4, PV_GROUPS)
        N_ACC = _static_int(fx.size(acc_l))

        tid = fx.Int32(gpu.thread_id("x"))
        tc = fx.idx2crd(tid, _colmajor(WAVE, NUM_WAVES))
        lane = fx.get_(tc, 0).unpack()
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.get_(tc, 1).unpack()))
        # wave -> token slot
        slot = warp
        lanes = _colmajor(16, 4)
        lc = fx.idx2crd(lane, lanes)
        lane_row = fx.get_(lc, 0).unpack()
        lane_g = fx.get_(lc, 1).unpack()
        if const_expr(SPLIT):
            ld_row, ld_chunk = lane_row, lane_g
        else:
            ld_row = lane // fx.Int32(4)
            ld_chunk = lane % fx.Int32(4)
        # MFMA lane r + 16g consumes gather lane 4r + g (byte address).
        perm_addr = (lane_row * fx.Int32(4) + lane_g) * fx.Int32(4)
        head_xor = [_static_int(fx.crd2idx((0, b), lanes)) for b in (1, 2)]
        block = fx.Int32(gpu.block_id("x"))
        t_idx = fx.get_scalar(
            fx.crd2idx((slot, block), _colmajor(NUM_WAVES, UNBOUNDED))
        )
        n_tok_i = fx.Int32(n_tok)
        live = t_idx < n_tok_i
        q_row = live.select(t_idx, fx.Int32(0))

        # ---- global views (elements)
        q_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(q_ptr),
            _rowmajor(UNBOUNDED, NUM_HEADS, QK_PAIRS, 4, VEC),
        )
        if const_expr(REGISTER_VT):
            out_l = _rowmajor(UNBOUNDED, NUM_HEADS, QK_PAIRS, 4, VEC)
        else:
            out_l = _rowmajor(UNBOUNDED, NUM_HEADS, PV_GROUPS, 4, 4, H4)
        out_g = fx.make_view(fx.rocdl.make_buffer_ptr(out_ptr), out_l)
        split = fx.Int32(gpu.block_id("y"))
        width = fx.Int32(index_width)
        start = q_row * width
        length = width
        if const_expr(CSR):
            ip = fx.make_view(
                fx.rocdl.make_buffer_ptr(indptr_ptr), fx.make_layout(UNBOUNDED, 1)
            )
            start = fx.Int32(fx.rocdl.readfirstlane(T.i32, ip[q_row]))
            end = fx.Int32(fx.rocdl.readfirstlane(T.i32, ip[q_row + 1]))
            length = end - start
        if const_expr(LENGTHS):
            lp = fx.make_view(
                fx.rocdl.make_buffer_ptr(lengths_ptr), fx.make_layout(UNBOUNDED, 1)
            )
            length = fx.Int32(fx.rocdl.readfirstlane(T.i32, lp[q_row]))
        total_tiles = (length + BLOCK_K - 1) // BLOCK_K
        per_split = total_tiles // split_k
        extra = total_tiles % split_k
        first_tile = split * per_split + (split < extra).select(split, extra)
        count = per_split + (split < extra).select(fx.Int32(1), fx.Int32(0))
        paired_count = ((count + 1) // 2) * 2
        n_kt = (paired_count > 0).select(paired_count, fx.Int32(2))
        kv_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(kv_ptr), _rowmajor(UNBOUNDED, CHUNKS_PER_ROW, VEC)
        )
        # Adjacent groups of four lanes read one row's four 16-byte chunks.
        g_chunks = _rowmajor(QK_PAIRS, 4)

        # ---- LDS V^T route: one private region [16 rows][512 d] per wave, swizzled
        if const_expr(LDS_VT):

            @fx.struct
            class SharedStorage:
                kv: fx.Array[fx.BFloat16, NUM_WAVES * WAVE_REGION, 16]

            storage = fx.SharedAllocator().allocate(SharedStorage).peek()
            regions = _colmajor(WAVE_REGION, NUM_WAVES)
            own_base = fx.get_scalar(fx.crd2idx((0, warp), regions))
            own_ptr = fx.add_offset(storage.kv.ptr, own_base)
            tile_l = fx.make_layout((BLOCK_K, HEAD_DIM // VEC, VEC), (HEAD_DIM, VEC, 1))
            own_tile = own_ptr.view(_swizzle(tile_l))
            rows = fx.make_layout(BLOCK_K, HEAD_DIM)
            vt_tok = fx.flat_divide(rows, (4,))
            vt_rows = _swizzle(
                fx.make_layout(
                    (fx.get_shape(vt_tok), 16, 4), (fx.get_stride(vt_tok), 4, 1)
                )
            )

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        if const_expr(REGISTER_VT):
            # B = I: lane (n, g) holds B[4g + j][n].
            identity = fx.Vector.from_elements(
                [
                    _to_bf16(
                        (lane_row == lane_g * fx.Int32(4) + fx.Int32(j)).select(
                            one_f, zero_f
                        )
                    )
                    for j in range_constexpr(4)
                ],
                fx.BFloat16,
            )
            # v_perm selector: bytes 2,3 then 6,7 are the two f32 high halves.
            high_halves = fx.Int32(0x07060302)

        q_frags = [
            _halves(q_g[q_row, lane_row, pr, lane_g, None].load(), halves)
            for pr in range_constexpr(QK_PAIRS)
        ]

        def _load_idx(kt):
            pos = (first_tile + kt) * BLOCK_K + ld_row
            index = fx.Int32(-1)
            if live & (kt < count) & (pos < length):
                iv = fx.make_view(
                    fx.rocdl.make_buffer_ptr(idx_ptr),
                    fx.make_layout(UNBOUNDED * 128, 1),
                )
                index = fx.Int32(iv[start + pos])
            return index

        def _load_idx_pair(kt):
            # One fetch of 32 index positions serves tiles kt and kt + 1.
            lane_pos = lane % fx.Int32(2 * BLOCK_K)
            pos = (first_tile + kt) * BLOCK_K + lane_pos
            index = fx.Int32(-1)
            if live & (kt + lane_pos // BLOCK_K < count) & (pos < length):
                iv = fx.make_view(
                    fx.rocdl.make_buffer_ptr(idx_ptr),
                    fx.make_layout(UNBOUNDED * 128, 1),
                )
                index = fx.Int32(iv[start + pos])
            return index

        def _pick_idx(pair, slot):
            src_lane = ld_row + fx.Int32(slot * BLOCK_K)
            return fx.Int32(fx.rocdl.ds_bpermute(T.i32, src_lane * fx.Int32(4), pair))

        def _chunk(pr):
            return fx.get_scalar(fx.crd2idx((pr, ld_chunk), g_chunks))

        def _gather(idx_v):
            raw_idx = idx_v
            idx_v = (idx_v >= 0).select(idx_v, fx.Int32(0))
            frags = [
                kv_g[idx_v, _chunk(pr), None].load() for pr in range_constexpr(QK_PAIRS)
            ]
            frags.append(fx.Vector.from_elements([raw_idx], fx.Int32))
            return frags

        def _store_tile(frags):
            lane_chunks = _rowmajor(QK_PAIRS, 4)
            # Predicated stores avoid keeping a second, zero-masked KV copy.
            if fx.Int32(frags[-1][0]) >= 0:
                for pr in range_constexpr(QK_PAIRS):
                    c = fx.get_scalar(fx.crd2idx((pr, lane_g), lane_chunks))
                    _lds_store(fx.Vector(frags[pr]), own_tile[lane_row, c, None], VEC)
            else:
                z = fx.Vector.filled(VEC, 0.0, fx.Float32).to(fx.BFloat16)
                for pr in range_constexpr(QK_PAIRS):
                    c = fx.get_scalar(fx.crd2idx((pr, lane_g), lane_chunks))
                    _lds_store(z, own_tile[lane_row, c, None], VEC)

        def _read_vt():
            # V^T A operands of tiles (h, g): row r is d = 64 g + 4 r + h
            ops = []
            for g in range_constexpr(PV_GROUPS):
                vt_op = fx.add_offset(own_ptr, g * 64).view(vt_rows)
                rows_g = [
                    fx.Vector(_lds_load(vt_op[(e, lane_g), lane_row, None], 4))
                    for e in range_constexpr(4)
                ]
                ops += [
                    fx.Vector.from_elements(
                        [rows_g[e][h] for e in range_constexpr(4)], fx.BFloat16
                    )
                    for h in range_constexpr(H4)
                ]
            return ops

        def _identity_mfma(kop):
            # D = K_op x I is V^T in A-operand order: tile t = 2 pr + hh, row n is
            # d = 32 pr + 8 (n // 4) + 4 hh + n % 4, column k is token k (exact f32).
            return _mfma_ssa(mma, kop, identity, _zero4())

        def _pv_packed(acc_t, d, p):
            # D is exact bf16 in f32: its high halves are already the bf16 A operand
            # (lane (n, g) holds tokens 4g..4g+3, matching the QK S^T order of p).
            w = fx.Vector(d).bitcast(fx.Int32)
            lo = fx.rocdl.perm_b32(fx.Int32(w[1]), fx.Int32(w[0]), high_halves)
            hi = fx.rocdl.perm_b32(fx.Int32(w[3]), fx.Int32(w[2]), high_halves)
            a = fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)
            return _mfma_ssa(mma, a.bitcast(fx.BFloat16), p, acc_t)

        def _mask_vt(vt, valid_bits):
            # Invalid K rows may hold NaN and NaN * 0 survives the MFMAs: zero them.
            full = fx.Int32(0xFFFF)
            if (valid_bits & full) != full:
                keep = ((valid_bits >> lane_row) & fx.Int32(1)) != fx.Int32(0)
                masked = []
                for t in range_constexpr(N_ACC):
                    w = fx.Vector(vt[t]).bitcast(fx.Int32)
                    masked.append(
                        fx.Vector.from_elements(
                            [
                                keep.select(fx.Int32(w[j]), fx.Int32(0))
                                for j in range_constexpr(2)
                            ],
                            fx.Int32,
                        ).bitcast(fx.BFloat16)
                    )
                vt = masked
            return vt

        def _pv(acc, vt, p_b):
            acc = list(acc)
            if const_expr(LDS_VT):
                for g in range_constexpr(PV_GROUPS):
                    for h in range_constexpr(H4):
                        t = _static_int(fx.crd2idx((h, g), acc_l))
                        acc[t] = _mfma_ssa(mma, vt[t], p_b, acc[t])
            else:
                d = _identity_mfma(vt[0])
                for t in range_constexpr(N_ACC):
                    d_next = None
                    if const_expr(t + 1 < N_ACC):
                        d_next = _identity_mfma(vt[t + 1])
                    acc[t] = _pv_packed(acc[t], d, p_b)
                    d = d_next
            return acc

        def _qk(frags):
            s_accs = [_zero4() for _ in range_constexpr(HALVES)]
            for pr in range_constexpr(QK_PAIRS):
                k = _halves(frags[pr], halves)
                for h in range_constexpr(HALVES):
                    s_accs[h] = _mfma_ssa(mma, k[h], q_frags[pr][h], s_accs[h])
            s_acc = s_accs[0]
            for a in s_accs[1:]:
                s_acc = fx.Vector.from_elements(
                    [_element(s_acc, i) + _element(a, i) for i in range_constexpr(4)],
                    fx.Float32,
                )
            return s_acc

        def _mfma_phase(acc, vt, p_b, frags, idx_s, prefetch):
            """PV of the previous tile and QK of this tile, at raised priority."""
            if const_expr(SPLIT):
                # Preserve the original decode/split-K instruction schedule.
                fx.rocdl.s_setprio(1)
                acc = _pv(acc, vt, p_b)
                s_acc = _qk(frags)
                fx.rocdl.s_setprio(0)
                return acc, s_acc, frags, frags, None
            else:
                fx.rocdl.s_setprio(1)
                new_frags = list(frags)
                next_idx = idx_s
                idx_s = (idx_s >= 0).select(idx_s, fx.Int32(0))
                dw = [
                    fx.Vector(frags[pr]).bitcast(fx.Int32)
                    for pr in range_constexpr(QK_PAIRS)
                ]
                moved = [[None] * 4 for _ in range_constexpr(QK_PAIRS)]
                kops = [None] * QK_PAIRS
                acc = list(acc)
                s_accs = [_zero4() for _ in range_constexpr(HALVES)]
                # Raw validity must follow the same row permutation as the KV payload.
                raw_idx = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32, perm_addr, fx.Int32(frags[-1][0])
                            )
                        )
                    ],
                    fx.Int32,
                )
                # Register V^T: the permuted K operands are the next tile's V^T
                # source. PV transposes them with an identity MFMA ahead of the
                # packed bf16 PV MFMAs that use them.
                new_vt = [None] * N_ACC
                transposed = {}
                sched = _bpermute_mfma_schedule(
                    [(g, h) for g in range(PV_GROUPS) for h in range(H4)],
                    QK_PAIRS,
                    HALVES,
                )
                pv_ts = [
                    _static_int(fx.crd2idx((a[2], a[1]), acc_l))
                    for a in sched
                    if a[0] == "pv"
                ]
                pv_n = 0
                for n in range_constexpr(len(sched)):
                    act = sched[n]
                    if const_expr(act[0] == "bp"):
                        pr, i = act[1] // 4, act[1] % 4
                        moved[pr][i] = fx.Int32(
                            fx.rocdl.ds_bpermute(T.i32, perm_addr, fx.Int32(dw[pr][i]))
                        )
                        if const_expr(prefetch and i == 3):
                            # Raw registers are free after their fourth transfer.
                            # Preserve the PV-first FIFO lookahead before QK use.
                            fx.rocdl.sched_barrier(0)
                            new_frags[pr] = kv_g[idx_s, _chunk(pr), None].load()
                    elif const_expr(act[0] == "pv"):
                        t = pv_ts[pv_n]
                        if const_expr(LDS_VT):
                            acc[t] = _mfma_ssa(mma, vt[t], p_b, acc[t])
                        else:
                            if const_expr(pv_n == 0):
                                for j in range_constexpr(
                                    min(IDENTITY_MFMA_LEAD, len(pv_ts))
                                ):
                                    transposed[pv_ts[j]] = _identity_mfma(vt[pv_ts[j]])
                            if const_expr(pv_n + IDENTITY_MFMA_LEAD < len(pv_ts)):
                                u = pv_ts[pv_n + IDENTITY_MFMA_LEAD]
                                transposed[u] = _identity_mfma(vt[u])
                            acc[t] = _pv_packed(acc[t], transposed.pop(t), p_b)
                        pv_n += 1
                    else:
                        pr, h = act[1], act[2]
                        if const_expr(kops[pr] is None):
                            kops[pr] = _halves(
                                fx.Vector.from_elements(moved[pr], fx.Int32).bitcast(
                                    fx.BFloat16
                                ),
                                halves,
                            )
                        s_accs[h] = _mfma_ssa(
                            mma, kops[pr][h], q_frags[pr][h], s_accs[h]
                        )
                        new_vt[pr * HALVES + h] = kops[pr][h]
                    fx.rocdl.sched_barrier(0)
                if const_expr(prefetch):
                    new_frags[-1] = fx.Vector.from_elements([next_idx], fx.Int32)
                s_acc = s_accs[0]
                for a in s_accs[1:]:
                    s_acc = fx.Vector.from_elements(
                        [
                            _element(s_acc, i) + _element(a, i)
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )
                frags = []
                if const_expr(LDS_VT):
                    # Converted operands are also V: the LDS route stores them.
                    frags = [
                        fx.Vector.from_elements(moved[pr], fx.Int32).bitcast(
                            fx.BFloat16
                        )
                        for pr in range_constexpr(QK_PAIRS)
                    ]
                frags.append(raw_idx)
                fx.rocdl.s_setprio(0)
                return acc, s_acc, frags, new_frags, new_vt

        def _ballot_valid(frags):
            return fx.Int32(
                fx.Int64(fx.rocdl.ballot(T.i64, fx.Int32(frags[-1][0]) >= 0))
            )

        def _softmax(s_acc, m_i, l_i, acc, valid_bits):
            scores = []
            for i in range_constexpr(4):
                valid = live & (
                    ((valid_bits >> (lane_g * 4 + i)) & fx.Int32(1)) != fx.Int32(0)
                )
                scores.append(valid.select(_element(s_acc, i), neg_inf))
            m_w = scores[0]
            for i in range_constexpr(1, 4):
                m_w = (m_w > scores[i]).select(m_w, scores[i])
            # Lazy rescale: keep the stale max until some head grows past RESCALE_TAU.
            moved = fx.Int64(
                fx.rocdl.ballot(
                    fx.Int64.ir_type, (m_w - m_i) * qk_scale > fx.Float32(RESCALE_TAU)
                )
            )
            m_n = m_i
            alpha = one_f
            if moved != fx.Int64(0):
                m_h = _head_max(m_w, head_xor)
                m_n = (m_h > m_i).select(m_h, m_i)
                alpha = (m_n == neg_inf).select(zero_f, _exp2((m_i - m_n) * qk_scale))
                acc = _rescale_all(acc, alpha)
            empty = m_n == neg_inf
            p_t = []
            for i in range_constexpr(4):
                p = empty.select(zero_f, _exp2((scores[i] - m_n) * qk_scale))
                p_t.append(live.select(p, zero_f))
            l_i = l_i * alpha + ((p_t[0] + p_t[1]) + (p_t[2] + p_t[3]))
            p_b = fx.Vector.from_elements([_to_bf16(p) for p in p_t], fx.BFloat16)
            return m_n, l_i, acc, p_b

        def _softmax_phase(s_acc, m_i, l_i, acc, frags, idx_s, prefetch, new_frags, vt):
            """Softmax of this tile and its V^T for the next PV."""
            # Prefetch is static here: a runtime skip would merge a gather-free
            # path into loop waitcnt state and drain the next gather early.
            if const_expr(LDS_VT):
                _store_tile(frags)
                if const_expr(SPLIT and prefetch):
                    new_frags = _gather(idx_s)
                    fx.rocdl.sched_barrier(0)
                valid_bits = _ballot_valid(frags)
                m_i, l_i, acc, p_b = _softmax(s_acc, m_i, l_i, acc, valid_bits)
                vt = _read_vt()
                fx.rocdl.s_waitcnt(lgkmcnt=0)
            else:
                valid_bits = _ballot_valid(frags)
                m_i, l_i, acc, p_b = _softmax(s_acc, m_i, l_i, acc, valid_bits)
                vt = _mask_vt(vt, valid_bits)
            return m_i, l_i, acc, p_b, vt, new_frags

        # Full-D ownership: no peer exchange, one softmax and two fixed gather slots.
        m_i = neg_inf
        l_i = zero_f
        acc = [_zero4() for _ in range_constexpr(N_ACC)]
        vt = [
            fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
            for _ in range_constexpr(N_ACC)
        ]
        p_b = fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
        gather0 = _gather(_load_idx(fx.Int32(0)))
        gather1 = _gather(_load_idx(fx.Int32(1)))
        pair = _load_idx_pair(fx.Int32(2))
        idx0 = _pick_idx(pair, 0)
        idx1 = _pick_idx(pair, 1)
        loop_init = [m_i, l_i, p_b, idx0, idx1] + acc + vt + gather0 + gather1

        # KV fragments plus the raw index that marks the row valid or masked.
        N_BUF = QK_PAIRS + 1

        def _unpack_state(state):
            m, l, p, i0, i1 = state[:5]
            rest = state[5:]
            return (
                fx.Float32(m),
                fx.Float32(l),
                fx.Vector(p),
                fx.Int32(i0),
                fx.Int32(i1),
                [fx.Vector(a) for a in rest[:N_ACC]],
                [fx.Vector(a) for a in rest[N_ACC : 2 * N_ACC]],
                [fx.Vector(a) for a in rest[2 * N_ACC : 2 * N_ACC + N_BUF]],
                [fx.Vector(a) for a in rest[2 * N_ACC + N_BUF :]],
            )

        # Two static slots: each next gather has a full tile of compute before use.
        # Ranges round up to a pair; the final dummy tile has zero weight.
        for kt, state in range(
            fx.Int32(0), n_kt - fx.Int32(2), fx.Int32(2), init=loop_init
        ):
            m_i, l_i, p_b, idx0, idx1, acc, vt, gather0, gather1 = _unpack_state(state)
            acc, s_acc, gather0, next_frags, new_vt = _mfma_phase(
                acc, vt, p_b, gather0, idx0, True
            )
            pair = _load_idx_pair(fx.Int32(kt) + fx.Int32(4))
            fx.rocdl.sched_barrier(0)
            m_i, l_i, acc, p_b, vt, gather0 = _softmax_phase(
                s_acc, m_i, l_i, acc, gather0, idx0, True, next_frags, new_vt
            )
            acc, s_acc, gather1, next_frags, new_vt = _mfma_phase(
                acc, vt, p_b, gather1, idx1, True
            )
            m_i, l_i, acc, p_b, vt, gather1 = _softmax_phase(
                s_acc, m_i, l_i, acc, gather1, idx1, True, next_frags, new_vt
            )
            # Carry completed lane transfers, not pending global-load copies.
            idx0 = _pick_idx(pair, 0)
            idx1 = _pick_idx(pair, 1)
            results = yield [m_i, l_i, p_b, idx0, idx1] + acc + vt + gather0 + gather1
        m_i, l_i, p_b, idx0, idx1, acc, vt, gather0, gather1 = _unpack_state(
            list(results)
        )
        acc, s_acc, gather0, next_frags, new_vt = _mfma_phase(
            acc, vt, p_b, gather0, idx0, False
        )
        m_i, l_i, acc, p_b, vt, gather0 = _softmax_phase(
            s_acc, m_i, l_i, acc, gather0, idx0, False, next_frags, new_vt
        )
        acc, s_acc, gather1, next_frags, new_vt = _mfma_phase(
            acc, vt, p_b, gather1, idx1, False
        )
        m_i, l_i, acc, p_b, vt, gather1 = _softmax_phase(
            s_acc, m_i, l_i, acc, gather1, idx1, False, next_frags, new_vt
        )
        acc = _pv(acc, vt, p_b)

        for sh in head_xor:
            l_i = l_i + l_i.shuffle_xor(sh, WAVE)
        if const_expr(SPLIT):
            partial = fx.make_view(
                fx.rocdl.make_buffer_ptr(out_ptr),
                _rowmajor(UNBOUNDED, split_k, PV_GROUPS, 4, 4, NUM_HEADS, H4),
            )
            stats = fx.make_view(
                fx.rocdl.make_buffer_ptr(stats_ptr),
                _rowmajor(UNBOUNDED, split_k, NUM_HEADS, 2),
            )
            if live:
                if lane_g == fx.Int32(0):
                    stats[t_idx, split, lane_row, 0] = m_i * fx.Float32(scale)
                    stats[t_idx, split, lane_row, 1] = l_i
                for g in range_constexpr(PV_GROUPS):
                    for i in range_constexpr(4):
                        vals = [
                            _element(acc[_static_int(fx.crd2idx((h, g), acc_l))], i)
                            for h in range_constexpr(H4)
                        ]
                        partial[t_idx, split, g, lane_g, i, lane_row, None].store(
                            fx.Vector.from_elements(vals, fx.Float32)
                        )
        else:
            tiny = fx.Float32(1.0e-30)
            head = lane_row
            denom = l_i
            alpha_q = fx.Float32(1.0)
            if const_expr(HAS_SINK):
                # The sink is one extra logit per head: it only enters the denominator.
                sink = fx.make_view(
                    fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(NUM_HEADS, 1)
                )[head]
                m_scaled = m_i * fx.Float32(scale)
                m_final = (sink > m_scaled).select(sink, m_scaled)
                empty = m_final == neg_inf
                alpha_q = empty.select(
                    zero_f, _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
                )
                sink_term = empty.select(
                    zero_f, _exp2((sink - m_final) * fx.Float32(LOG2E))
                )
                denom = denom * alpha_q + sink_term
            denom = (denom > tiny).select(denom, tiny)
            inv = alpha_q / denom
            if live:
                if const_expr(REGISTER_VT):
                    for pr in range_constexpr(QK_PAIRS):
                        vals = [
                            _to_bf16(_element(acc[pr * HALVES + hh], i) * inv)
                            for hh in range_constexpr(HALVES)
                            for i in range_constexpr(4)
                        ]
                        out_g[t_idx, head, pr, lane_g, None].store(
                            fx.Vector.from_elements(vals, fx.BFloat16)
                        )
                else:
                    for g in range_constexpr(PV_GROUPS):
                        for i in range_constexpr(4):
                            vals = [
                                _to_bf16(
                                    _element(
                                        acc[_static_int(fx.crd2idx((h, g), acc_l))], i
                                    )
                                    * inv
                                )
                                for h in range_constexpr(H4)
                            ]
                            out_g[t_idx, head, g, lane_g, i, None].store(
                                fx.Vector.from_elements(vals, fx.BFloat16)
                            )

    @flyc.jit
    def launch_sparse_mla_fwd(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        index_width: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
        indptr_ptr: fx.Pointer,
        lengths_ptr: fx.Pointer,
        stats_ptr: fx.Pointer,
        n_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        sparse_mla_fwd_kernel(
            q_ptr,
            kv_ptr,
            idx_ptr,
            out_ptr,
            n_tok,
            index_width,
            scale,
            sink_ptr,
            indptr_ptr,
            lengths_ptr,
            stats_ptr,
        ).launch(
            grid=(fx.Int64(n_blocks), split_k, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_sparse_mla_fwd.compile_hints = {
        "waves_per_eu": 1,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_sparse_mla_fwd


@lru_cache(maxsize=32)
def compile_sparse_mla_combine(splits, has_sink):
    @flyc.kernel(
        name=f"sparse_mla_combine_split{splits}_sink{int(has_sink)}",
        known_block_size=[256, 1, 1],
    )
    def combine(
        part_ptr: fx.Pointer,
        stats_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
    ):
        t = fx.Int32(gpu.block_id("x"))
        head = fx.Int32(gpu.block_id("y"))
        tid = fx.Int32(gpu.thread_id("x"))
        part = fx.make_view(
            fx.rocdl.make_buffer_ptr(part_ptr),
            _rowmajor(1 << 20, splits, 128, 16, 2, 2),
        )
        stats = fx.make_view(
            fx.rocdl.make_buffer_ptr(stats_ptr), _rowmajor(1 << 20, splits, 16, 2)
        )
        out = fx.make_view(
            fx.rocdl.make_buffer_ptr(out_ptr), _rowmajor(1 << 20, 16, 256, 2)
        )
        m = fx.Float32(float("-inf"))
        l = fx.Float32(0.0)
        a0 = fx.Float32(0.0)
        a1 = fx.Float32(0.0)
        if const_expr(has_sink):
            sink = fx.make_view(
                fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(16, 1)
            )
            m = fx.Float32(sink[head])
            l = fx.Float32(1.0)
        for s in range(splits):
            ms = fx.Float32(stats[t, s, head, 0])
            ls = fx.Float32(stats[t, s, head, 1])
            mn = (ms > m).select(ms, m)
            empty = mn == fx.Float32(float("-inf"))
            alpha = empty.select(fx.Float32(0.0), _exp2((m - mn) * LOG2E))
            beta = (ls > 0).select(_exp2((ms - mn) * LOG2E), fx.Float32(0.0))
            v = fx.Vector(part[t, s, tid // 2, head, tid % 2, None].load())
            a0 = a0 * alpha + v[0] * beta
            a1 = a1 * alpha + v[1] * beta
            l = l * alpha + ls * beta
            m = mn
        inv = (l > 0).select(fx.Float32(1.0) / l, fx.Float32(0.0))
        out[t, head, tid, None].store(
            fx.Vector.from_elements(
                [_to_bf16(a0 * inv), _to_bf16(a1 * inv)], fx.BFloat16
            )
        )

    @flyc.kernel(
        name=f"sparse_mla_combine_parallel_split{splits}_sink{int(has_sink)}",
        known_block_size=[256, 1, 1],
    )
    def parallel(
        part_ptr: fx.Pointer,
        stats_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
    ):
        t = fx.Int32(gpu.block_id("x"))
        db = fx.Int32(gpu.block_id("y"))
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 64
        wave = tid // 64
        head = lane // 4
        dim = lane % 4
        part = fx.make_view(
            fx.rocdl.make_buffer_ptr(part_ptr), _rowmajor(1 << 20, splits, 128, 16, 4)
        )
        stats = fx.make_view(
            fx.rocdl.make_buffer_ptr(stats_ptr), _rowmajor(1 << 20, splits, 16, 2)
        )
        out = fx.make_view(
            fx.rocdl.make_buffer_ptr(out_ptr), _rowmajor(1 << 20, 16, 128, 4)
        )
        m = fx.Float32(float("-inf"))
        l = fx.Float32(0.0)
        acc = fx.Float32(0.0)
        for chunk in range((splits + 3) // 4):
            s = wave + chunk * 4
            safe = (s < splits).select(s, fx.Int32(0))
            ls = (s < splits).select(
                fx.Float32(stats[t, safe, head, 1]), fx.Float32(0.0)
            )
            ms = (ls > 0).select(
                fx.Float32(stats[t, safe, head, 0]), fx.Float32(float("-inf"))
            )
            mn = (ms > m).select(ms, m)
            empty = mn == fx.Float32(float("-inf"))
            alpha = empty.select(fx.Float32(0.0), _exp2((m - mn) * LOG2E))
            beta = (ls > 0).select(_exp2((ms - mn) * LOG2E), fx.Float32(0.0))
            v = fx.Float32(part[t, safe, db, head, dim])
            acc = acc * alpha + v * beta
            l = l * alpha + ls * beta
            m = mn

        @fx.struct
        class SharedStorage:
            data: fx.Array[fx.Float32, 3 * 256, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        shared = storage.data.ptr.view(_rowmajor(3, 4, 64))
        shared[0, wave, lane] = m
        shared[1, wave, lane] = l
        shared[2, wave, lane] = acc
        gpu.barrier()
        if wave == fx.Int32(0):
            m = fx.Float32(float("-inf"))
            l = fx.Float32(0.0)
            acc = fx.Float32(0.0)
            if const_expr(has_sink):
                sink = fx.make_view(
                    fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(16, 1)
                )
                m = fx.Float32(sink[head])
                l = fx.Float32(1.0)
            for w in range_constexpr(4):
                ms = fx.Float32(shared[0, w, lane])
                ls = fx.Float32(shared[1, w, lane])
                mn = (ms > m).select(ms, m)
                empty = mn == fx.Float32(float("-inf"))
                alpha = empty.select(fx.Float32(0.0), _exp2((m - mn) * LOG2E))
                beta = (ls > 0).select(_exp2((ms - mn) * LOG2E), fx.Float32(0.0))
                acc = acc * alpha + fx.Float32(shared[2, w, lane]) * beta
                l = l * alpha + ls * beta
                m = mn
            inv = (l > 0).select(fx.Float32(1.0) / l, fx.Float32(0.0))
            out[t, head, db, dim] = _to_bf16(acc * inv)

    @flyc.jit
    def launch(
        part_ptr: fx.Pointer,
        stats_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        if const_expr(splits >= 32):
            parallel(part_ptr, stats_ptr, sink_ptr, out_ptr).launch(
                grid=(tokens, 128, 1), block=(256, 1, 1), stream=stream
            )
        else:
            combine(part_ptr, stats_ptr, sink_ptr, out_ptr).launch(
                grid=(tokens, 16, 1), block=(256, 1, 1), stream=stream
            )

    launch.compile_hints = {"waves_per_eu": 2}
    return launch


@cache
def compile_sparse_mla_order_indices(
    page_size, pages, lookup_pages, logical=False, single_request=True
):
    if page_size % 16:
        raise ValueError("page_size must be a multiple of 16 for tile ordering")
    threads, items = 256, 8
    words_per_thread = ((pages * page_size + 63) // 64 + threads - 1) // threads
    num_words = words_per_thread * threads
    sizes, strides = _build_bitonic_schedule(2048)
    block_threads = threads
    Scan = fx.coop.BlockScan[fx.Int32, threads]

    @fx.struct
    class Storage:
        scratch: fx.Array[fx.Int32, max(num_words, 2048)]
        scan: Scan.SharedStorage

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def sparse_mla_order_indices_kernel(
        src: fx.Tensor,
        dst: fx.Tensor,
        page_rank: fx.Tensor,
        sorted_pages: fx.Tensor,
        row_requests: fx.Tensor,
    ):
        row, tid = fx.block_idx.x, fx.thread_idx.x
        req = fx.Int32(0) if const_expr(single_request) else row_requests[row]
        storage = fx.SharedAllocator().allocate(Storage)
        scratch = storage.scratch.peek().view(fx.make_layout(max(num_words, 2048), 1))
        scan = storage.scan.peek()
        regs = fx.make_rmem_tensor(items, fx.Int32)
        tmp = fx.make_rmem_tensor(items, fx.Int32)
        counts = fx.make_rmem_tensor(words_per_thread, fx.Int32)
        compact_regs = fx.make_rmem_tensor(items, fx.Int32)
        rank_regs = fx.make_rmem_tensor(items, fx.Int32)
        offsets = fx.make_rmem_tensor(items, fx.Int32)
        within = fx.make_rmem_tensor(items, fx.Int32)
        for w in range_constexpr(words_per_thread):
            scratch[tid + w * threads] = fx.Int32(0)
        gpu.barrier()
        src_vec = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(fx.slice(src, (row, None)), max_size=False),
            fx.make_layout(4, 1),
        )
        dst_vec = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(fx.slice(dst, (row, None)), max_size=False),
            fx.make_layout(4, 1),
        )
        rank_row = fx.rocdl.make_buffer_tensor(
            fx.slice(page_rank, (req, None)), max_size=False
        )
        pages_row = fx.rocdl.make_buffer_tensor(
            fx.slice(sorted_pages, (req, None)), max_size=False
        )
        atom = buf_copy_atom(16, fx.Int32)
        for v in range_constexpr(items // 4):
            part = fx.slice(src_vec, (None, tid + v * threads))
            frag = fx.make_fragment_like(part)
            fx.copy(atom, part, frag)
            vec = fx.Vector(fx.memref_load_vec(frag))
            for j in range_constexpr(4):
                regs[v * 4 + j] = vec[j]
        # Issue every page-rank lookup before consuming any of them. Safe
        # clamping removes per-key control flow without reading outside a LUT.
        for item in range_constexpr(items):
            value = regs[item]
            unsigned = value.bitcast(fx.Uint32)
            page = unsigned // fx.Uint32(page_size)
            valid = page < fx.Uint32(lookup_pages)
            safe_page = valid.select(page.to(fx.Int32), fx.Int32(0))
            rank = rank_row[safe_page]
            offsets[item] = (unsigned - page * fx.Uint32(page_size)).to(fx.Int32)
            rank_regs[item] = valid.select(rank, fx.Int32(-1))
        for item in range_constexpr(items):
            value = regs[item]
            rank = rank_regs[item]
            offset = offsets[item]
            compact = fx.Int32(0)
            within[item] = fx.Int32(0)
            if rank >= 0:
                compact = rank * page_size + offset
                shift = ((compact // 16) % 4) * 8
                ticket = atomic_add_i32(
                    scratch, fx.Int32(1) << shift, compact // 64, "workgroup"
                )
                within[item] = (ticket >> shift) & fx.Int32(255)
            compact_regs[item] = compact
            if const_expr(logical):
                regs[item] = (rank >= 0).select(compact, fx.Int32(-1))
        gpu.barrier()
        for item in range_constexpr(items):
            compact = compact_regs[item]
            shift = ((compact // 16) % 4) * 8
            below = (fx.Int32(1) << shift) - fx.Int32(1)
            packed = scratch[compact // 64] & below
            pairs = (packed & fx.Int32(0x00FF00FF)) + (
                (packed >> 8) & fx.Int32(0x00FF00FF)
            )
            within[item] = (
                within[item]
                + (pairs & fx.Int32(65535))
                + ((pairs >> 16) & fx.Int32(65535))
            )
        count = fx.Int32(0)
        for w in range_constexpr(words_per_thread):
            word = scratch[tid * words_per_thread + w]
            pairs = (word & fx.Int32(0x00FF00FF)) + ((word >> 8) & fx.Int32(0x00FF00FF))
            n = (pairs & fx.Int32(65535)) + ((pairs >> 16) & fx.Int32(65535))
            counts[w] = n
            count = count + n
        prefix, total = Scan.exclusive_with_aggregate(
            count, fx.ReductionOp.ADD, storage=scan
        )
        gpu.barrier()
        if total == 2048:
            # Prefix four byte-sized tile counts per word, then rank only
            # the selected inputs. Keep values in registers and reuse LDS
            # for coalesced output. No comparison network on the hot path.
            for w in range_constexpr(words_per_thread):
                scratch[tid * words_per_thread + w] = prefix
                prefix = prefix + counts[w]
            gpu.barrier()
            for item in range_constexpr(items):
                within[item] = scratch[compact_regs[item] // 64] + within[item]
            gpu.barrier()
            for item in range_constexpr(items):
                scratch[within[item]] = regs[item]
            gpu.barrier()
            for v in range_constexpr(items // 4):
                part = fx.slice(dst_vec, (None, tid + v * threads))
                frag = fx.make_fragment_like(part)
                values = []
                for j in range_constexpr(4):
                    value = scratch[(tid + v * threads) * 4 + j]
                    if const_expr(logical):
                        value = pages_row[
                            (value.bitcast(fx.Uint32) // fx.Uint32(page_size)).to(
                                fx.Int32
                            )
                        ] * page_size + (
                            value.bitcast(fx.Uint32) % fx.Uint32(page_size)
                        ).to(
                            fx.Int32
                        )
                    values.append(value)
                vec = fx.Vector.from_elements(values, fx.Int32)
                fx.memref_store_vec(vec, frag)
                fx.copy(atom, frag, part)
        else:
            # Counter overflow, padding or an out-of-plan physical slot.
            # Every carry loses 255 (or 256 at the top byte) from the sum
            # of byte counts, so total==2048 proves no counter overflow.
            # Preserve the full signed-int32 multiset on the exact fallback.
            if const_expr(logical):
                for item in range_constexpr(items):
                    compact = regs[item]
                    mapped = (
                        sorted_pages[
                            req,
                            (compact >= 0).select(compact // page_size, fx.Int32(0)),
                        ]
                        * page_size
                        + compact % page_size
                    )
                    regs[item] = (compact >= 0).select(mapped, fx.Int32(-1))
            for stage in range_constexpr(len(sizes)):
                size = sizes[stage]
                stride = strides[stage]
                if const_expr(256 <= stride < block_threads * 4):
                    for item in range_constexpr(items):
                        scratch[(tid + (item // 4) * block_threads) * 4 + item % 4] = (
                            regs[item]
                        )
                    gpu.barrier()
                for item in range_constexpr(items):
                    pos = (tid + (item // 4) * block_threads) * 4 + item % 4
                    left = regs[item]
                    if const_expr(stride < 4):
                        right = regs[item ^ stride]
                    elif const_expr(stride < 256):
                        right = left.shuffle_xor(fx.Int32(stride // 4), fx.Int32(64))
                    elif const_expr(stride >= block_threads * 4):
                        right = regs[item ^ (stride // block_threads)]
                    else:
                        right = scratch[pos ^ fx.Int32(stride)]
                    take_min = ((pos & fx.Int32(size)) == 0) == (
                        (pos & fx.Int32(stride)) == 0
                    )
                    swap = take_min.select(left > right, left < right)
                    tmp[item] = swap.select(right, left)
                if const_expr(256 <= stride < block_threads * 4):
                    gpu.barrier()
                for item in range_constexpr(items):
                    regs[item] = tmp[item]
            for item in range_constexpr(items):
                dst[row, (tid + (item // 4) * threads) * 4 + item % 4] = regs[item]

    @flyc.jit
    def launch(
        src: fx.Tensor,
        dst: fx.Tensor,
        page_rank: fx.Tensor,
        sorted_pages: fx.Tensor,
        row_requests: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        sparse_mla_order_indices_kernel(
            src, dst, page_rank, sorted_pages, row_requests
        ).launch(grid=(rows, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch
