# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flash Attention forward for gfx11 (RDNA3 / RDNA3.5), bf16/f16.

Computed transposed so that every lane owns one query row:

  S^T[key, q] = K[key, :] . Q[q, :]          (A = K from LDS, B = Q in VGPRs)
  O^T[d, q]  += V^T[d, key] . P^T[key, q]    (A = V^T from LDS, B = P^T in VGPRs)

gfx11 WMMA takes v16 A/B operands (lanes 16-31 replicate lanes 0-15) and puts
lane L's accumulator element ``si`` at D[2*si + L/16][L%16]. The query is
therefore L%16 in both lane halves: the row max needs one xor-16 exchange and
the row sum stays per lane until the epilogue.

Q/K/V/O are contiguous BSHD. Batch and both sequence lengths are runtime
values. Every batch slice gets its own bounds-checked buffer descriptor, so
loads past a sequence end return zero and stores past it are dropped; keys past
the end are additionally masked to -inf when the KV length can have a tail.
``head_dim`` only needs to be a multiple of 8: it is zero-padded to the WMMA K
of 16 inside the kernel.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .kernels_common import LOG2E as _LOG2E
from .tensor_shim import (
    _run_compiled,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_arg,
    ptr_buf_tensor,
)

KERNEL_NAME = "flash_attn_func_gfx11_kernel"

WAVE_SIZE = 32
WMMA_MNK = 16
LDS_LIMIT_BYTES = 65536
NEG_BIG = -1.0e30  # finite: NEG_BIG - m stays finite, exp2 of it is exactly 0

# K rows and V^T rows are padded by 8 elements: keeps every v8 LDS access
# 16-byte aligned and makes the 16 row reads of one WMMA operand bank-conflict free.
LDS_PAD = 8


def lds_bytes(head_dim, block_n, double_buffer, elem_bytes=2):
    """LDS footprint of one configuration (K tile + V^T tile, times buffers)."""
    dp = -(-head_dim // WMMA_MNK) * WMMA_MNK
    one = (block_n * (dp + LDS_PAD) + dp * (block_n + LDS_PAD)) * elem_bytes
    return one * (2 if double_buffer else 1)


def build_flash_attn_func_module_gfx11(
    num_heads,
    head_dim,
    causal=False,
    dtype_str="bf16",
    sm_scale=None,
    kv_tail=True,
    num_waves=16,
    block_n=32,
    vt_rows=2,
    double_buffer=True,
    head_group=None,
    waves_per_eu=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
):
    """Build the gfx11 Flash Attention forward kernel.

    Args:
        num_heads, head_dim: compile-time head geometry (``head_dim % 8 == 0``).
        causal: top-left aligned causal mask; the caller guarantees Lq == Lk.
        kv_tail: mask keys at or past ``seq_kv``. Pass False only when the KV
            length is a multiple of ``block_n``.
        num_waves: waves per workgroup; each wave owns 16 query rows.
        block_n: keys per main-loop tile (multiple of 16).
        vt_rows: keys packed per LDS write in the V -> V^T transpose (2, 4, 8).
        double_buffer: ping-pong the K/V^T tile in LDS (one barrier per tile).
            Dropped automatically when two buffers do not fit in LDS.
        head_group: heads walked fastest in the block order (``None`` = all of
            them, 1 = one head at a time). See the block mapping in the kernel.
        waves_per_eu: ``rocdl.waves_per_eu`` occupancy hint.

    Returns:
        ``launch(Q, K, V, O, batch, seq_q, seq_kv, stream=...)``.
    """
    if head_dim % 8 != 0 or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive multiple of 8, got {head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be bf16 or f16, got {dtype_str!r}")
    if block_n % WMMA_MNK != 0 or vt_rows not in (2, 4, 8) or block_n % vt_rows != 0:
        raise ValueError(f"unsupported block_n={block_n} / vt_rows={vt_rows}")

    H = num_heads
    D = head_dim
    if head_group is None:
        head_group = num_heads
    HEAD_GROUP = head_group if head_group >= 1 and num_heads % head_group == 0 else 1
    DP = -(-D // WMMA_MNK) * WMMA_MNK
    ND = DP // WMMA_MNK
    BLOCK_M = WMMA_MNK * num_waves
    BLOCK_N = block_n
    NK = BLOCK_N // WMMA_MNK
    THREADS = num_waves * WAVE_SIZE
    TOK = H * D  # elements between consecutive tokens (BSHD)
    is_bf16 = dtype_str == "bf16"
    elem = fx.BFloat16 if is_bf16 else fx.Float16

    if lds_bytes(D, BLOCK_N, False) > LDS_LIMIT_BYTES:
        # A launch over the LDS limit leaves the HIP context unusable; refuse here.
        raise ValueError(
            f"head_dim={D}, block_n={BLOCK_N} exceeds {LDS_LIMIT_BYTES} B of LDS"
        )
    double_buffer = double_buffer and lds_bytes(D, BLOCK_N, True) <= LDS_LIMIT_BYTES
    NBUF = 2 if double_buffer else 1

    K_ROW = DP + LDS_PAD
    V_ROW = BLOCK_N + LDS_PAD
    K_SIZE = BLOCK_N * K_ROW
    BUF = K_SIZE + DP * V_ROW

    K_CHUNKS = BLOCK_N * (D // 8)  # one 8-element chunk per (key, d0)
    K_PER_THR = -(-K_CHUNKS // THREADS)
    DBLKS = D // 8
    V_BLOCKS = (BLOCK_N // vt_rows) * DBLKS  # vt_rows keys x 8 d per block
    V_PER_THR = -(-V_BLOCKS // THREADS)

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(D)
    SCALE_LOG2 = float(sm_scale) * _LOG2E
    CONCAT16 = list(range(16))

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem, BUF * NBUF, 16]

    def _wmma(a, b, acc):
        if const_expr(is_bf16):
            return fx.rocdl.wmma_f32_16x16x16_bf16(
                acc.type, a.bitcast(fx.Int16), b.bitcast(fx.Int16), acc
            ).result
        return fx.rocdl.wmma_f32_16x16x16_f16(acc.type, a, b, acc).result

    def _lds_barrier():
        # Waits on LDS only. gpu.barrier() also drains vmcnt, which would block on
        # the next tile's global prefetch issued at the top of the iteration.
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
            constraints="",
            has_side_effects=True,
        )

    def _exp2(x):
        # Arguments are <= 0 (<= -huge for masked keys); a flushed denormal is fine.
        return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        seq_q: fx.Int32,
        seq_kv: fx.Int32,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().kv.ptr

        def lds_view(off, n):
            p = fx.recast_iter(
                elem, fx.add_offset(lds_ptr, fx.make_int_tuple(fx.Int32(off)))
            )
            return fx.make_view(p, fx.make_layout(n, 1))

        def lds_v16(off):
            return Vec(lds_view(off, 8).load()).shuffle(
                Vec(lds_view(off + 8, 8).load()), CONCAT16
            )

        tid = fx.thread_idx.x
        bat = fx.block_idx.z
        n_qblk = (seq_q + (BLOCK_M - 1)) // BLOCK_M
        # Linear block id -> (head, qblk), HEAD_GROUP heads fastest. In BSHD one
        # token row holds every head, so workgroups resident together that cover
        # several heads of the same tokens share pages; walking one head at a time
        # instead sweeps the whole tensor touching 2*D bytes per row, which on
        # gfx1151 measured up to 4x slower (address translation bound) once a batch
        # slice passed ~128 MiB.
        lin = fx.block_idx.x
        per_group = n_qblk * HEAD_GROUP
        grp = lin // per_group
        rem = lin - grp * per_group
        head = grp * HEAD_GROUP + rem % HEAD_GROUP
        qblk = rem // HEAD_GROUP
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        l16 = lane % 16
        hh = lane // 16
        is_h0 = hh == 0

        n_kv_tiles = (seq_kv + (BLOCK_N - 1)) // BLOCK_N
        if const_expr(causal):
            # Longest rows first: the diagonal blocks have the most KV tiles.
            qblk = n_qblk - 1 - qblk

        def batch_buf(ptr, seqlen):
            # Bounded per batch slice: the kernel relies on OOB loads returning 0
            # (rows past seq_q / seq_kv) and OOB stores being dropped.
            slice_bytes = fx.Int64(seqlen) * (TOK * 2)
            base = buf_base_i64(ptr) + fx.Int64(bat) * slice_bytes
            return ptr_buf_tensor(
                base,
                elem=elem,
                unit_elems=8,
                unit_stride=1,
                num_records_bytes=slice_bytes,
            )

        q_buf = batch_buf(Q, seq_q)
        k_buf = batch_buf(K, seq_kv)
        v_buf = batch_buf(V, seq_kv)
        o_buf = batch_buf(O, seq_q)

        def g_load8(buf, off):
            return buf_copy_load(buf, off, elem=elem, unit_elems=8)

        def tok_off(tok):
            return tok * TOK + head * D

        # ---- Q -> VGPRs as WMMA B operands (zero past seq_q / past head_dim) ----
        zero_e8 = Vec.filled(8, 0.0, elem)
        q_row = qblk * BLOCK_M + wave * 16 + l16
        q_base = tok_off(q_row)
        q_vecs = []
        for c in range_constexpr(ND):
            lo = zero_e8
            hi = zero_e8
            if const_expr(16 * c < D):
                lo = g_load8(q_buf, q_base + 16 * c)
            if const_expr(16 * c + 8 < D):
                hi = g_load8(q_buf, q_base + 16 * c + 8)
            q_vecs.append(Vec(lo).shuffle(Vec(hi), CONCAT16))

        if const_expr(causal):
            n_tiles = ((qblk + 1) * BLOCK_M + (BLOCK_N - 1)) // BLOCK_N
            n_tiles = (n_tiles < n_kv_tiles).select(n_tiles, n_kv_tiles)
        else:
            n_tiles = n_kv_tiles

        # ---- K / V tile staging: global -> VGPRs -> LDS (K row-major, V^T) ----
        def k_chunk(i):
            chunk = tid + i * THREADS
            if const_expr(K_PER_THR * THREADS > K_CHUNKS):
                # Surplus threads redo the last chunk: same data, same address.
                chunk = (chunk < K_CHUNKS).select(chunk, K_CHUNKS - 1)
            return chunk

        def v_block(i):
            blk = tid + i * THREADS
            if const_expr(V_PER_THR * THREADS > V_BLOCKS):
                blk = (blk < V_BLOCKS).select(blk, V_BLOCKS - 1)
            return blk

        def gload_tile(tile):
            key0 = tile * BLOCK_N
            kregs = []
            for i in range_constexpr(K_PER_THR):
                chunk = k_chunk(i)
                row = chunk // DBLKS
                kregs.append(g_load8(k_buf, tok_off(key0 + row) + (chunk % DBLKS) * 8))
            vregs = []
            for i in range_constexpr(V_PER_THR):
                blk = v_block(i)
                kb0 = key0 + (blk // DBLKS) * vt_rows
                d0 = (blk % DBLKS) * 8
                vregs.append(
                    [
                        g_load8(v_buf, tok_off(kb0 + r) + d0)
                        for r in range_constexpr(vt_rows)
                    ]
                )
            return kregs, vregs

        def store_tile(regs, bb):
            kregs, vregs = regs
            for i in range_constexpr(K_PER_THR):
                chunk = k_chunk(i)
                lds_view(bb + (chunk // DBLKS) * K_ROW + (chunk % DBLKS) * 8, 8).store(
                    Vec(kregs[i])
                )
            for i in range_constexpr(V_PER_THR):
                blk = v_block(i)
                krow0 = (blk // DBLKS) * vt_rows
                d0 = (blk % DBLKS) * 8
                # Dword t of a row holds d0+2t (low half) and d0+2t+1 (high half);
                # pair up consecutive keys of one d into dwords along V^T rows.
                words = [Vec(r).bitcast(fx.Int32) for r in vregs[i]]
                for t in range_constexpr(4):
                    lo_w = []
                    hi_w = []
                    for pr in range_constexpr(vt_rows // 2):
                        a = fx.Int32(words[2 * pr][t])
                        b = fx.Int32(words[2 * pr + 1][t])
                        lo_w.append((a & 0xFFFF) | (b << 16))
                        hi_w.append(a.shrui(fx.Int32(16)) | (b & fx.Int32(-65536)))
                    for half, ws in ((0, lo_w), (1, hi_w)):
                        dst = bb + K_SIZE + (d0 + 2 * t + half) * V_ROW + krow0
                        lds_view(dst, vt_rows).store(
                            Vec.from_elements(ws, fx.Int32).bitcast(elem)
                        )

        if const_expr(DP != D):
            # K pad columns [D, DP) feed the S^T WMMA but are never loaded. LDS is
            # uninitialized and NaN * 0 = NaN, so zero them once per buffer.
            for i in range_constexpr(-(-BLOCK_N // THREADS)):
                zrow = tid + i * THREADS
                zrow = (zrow < BLOCK_N).select(zrow, BLOCK_N - 1)
                for b in range_constexpr(NBUF):
                    lds_view(b * BUF + zrow * K_ROW + D, 8).store(zero_e8)

        # ---- P^T operand assembly ----
        def pack2(a, b):
            """Two f32 P values in [0, 1] -> one dword of elem, a in the low half."""
            if const_expr(is_bf16):
                # Round-to-nearest-even without the NaN guard (P is never NaN):
                # gfx11 has no f32 -> bf16 convert instruction.
                ra = a.bitcast(fx.Int32)
                rb = b.bitcast(fx.Int32)
                ra = ra + 0x7FFF + (ra.shrui(fx.Int32(16)) & 1)
                rb = rb + 0x7FFF + (rb.shrui(fx.Int32(16)) & 1)
                return fx.Int32(fx.rocdl.perm_b32(rb, ra, fx.Int32(0x07060302)))
            return Vec.from_elements([a.to(elem), b.to(elem)], elem).bitcast(fx.Int32)[
                0
            ]

        # Lane L holds keys 2*si + L/16 of its query; the B operand wants all 16 keys
        # in order. After swapping packed pairs with lane L^16, one v_perm_b32 per
        # dword interleaves own/partner halves (selector depends on the lane half).
        sel_even = is_h0.select(fx.Int32(0x05040100), fx.Int32(0x01000504))
        sel_odd = is_h0.select(fx.Int32(0x07060302), fx.Int32(0x03020706))

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        init = [fx.Float32(NEG_BIG), fx.Float32(0.0)] + [
            zero8 for _ in range_constexpr(ND)
        ]

        store_tile(gload_tile(fx.Int32(0)), 0)
        _lds_barrier()

        for j, state in range(0, n_tiles, 1, init=init):
            m_old = fx.Float32(state[0])
            l_old = fx.Float32(state[1])
            o_acc = [Vec(state[2 + i]) for i in range_constexpr(ND)]
            j32 = fx.Int32(j)
            cur = 0
            if const_expr(double_buffer):
                cur = (j32 & 1) * BUF

            nxt = j32 + 1
            nxt = (nxt < n_tiles).select(nxt, j32)
            next_regs = gload_tile(nxt)
            # Keep the prefetch at the top of the body; the machine scheduler would
            # otherwise sink it next to its LDS store and lose the overlap.
            fx.rocdl.sched_barrier(0)

            # ---- S^T = K Q^T; c outer so consecutive WMMAs hit independent accs ----
            s_accs = [zero8 for _ in range_constexpr(NK)]
            for c in range_constexpr(ND):
                for kb in range_constexpr(NK):
                    a = lds_v16(cur + (kb * 16 + l16) * K_ROW + 16 * c)
                    s_accs[kb] = _wmma(a, q_vecs[c], s_accs[kb])

            s_vals = []
            for kb in range_constexpr(NK):
                acc = Vec(s_accs[kb])
                row = []
                for si in range_constexpr(8):
                    x = fx.Float32(acc[si])
                    if const_expr(causal or kv_tail):
                        key = j32 * BLOCK_N + kb * 16 + 2 * si + hh
                    if const_expr(causal):
                        x = (key > q_row).select(fx.Float32(NEG_BIG), x)
                    if const_expr(kv_tail):
                        x = (key >= seq_kv).select(fx.Float32(NEG_BIG), x)
                    row.append(x)
                s_vals.append(row)

            # ---- online softmax; m in raw S units, the scale folds into one fma ----
            m_loc = s_vals[0][0]
            for kb in range_constexpr(NK):
                for si in range_constexpr(8):
                    if const_expr(kb != 0 or si != 0):
                        m_loc = fx.maxnumf(m_loc, s_vals[kb][si])
            m_loc = fx.maxnumf(m_loc, m_loc.shuffle_xor(16, WAVE_SIZE))
            m_new = fx.maxnumf(m_old, m_loc)
            alpha = _exp2((m_old - m_new) * SCALE_LOG2)
            neg_mc = m_new * (-SCALE_LOG2)
            l_new = l_old * alpha
            o_acc = [
                Vec.from_elements(
                    [fx.Float32(o[si]) * alpha for si in range_constexpr(8)], fx.Float32
                )
                for o in o_acc
            ]

            for kb in range_constexpr(NK):
                own = []
                for si in range_constexpr(8):
                    p = _exp2(fx.fma(s_vals[kb][si], fx.Float32(SCALE_LOG2), neg_mc))
                    l_new = l_new + p
                    own.append(p)
                ownp = [
                    pack2(own[2 * j2], own[2 * j2 + 1]) for j2 in range_constexpr(4)
                ]
                partp = [
                    ownp[j2].shuffle_xor(16, WAVE_SIZE) for j2 in range_constexpr(4)
                ]
                dwords = []
                for j2 in range_constexpr(4):
                    dwords.append(
                        fx.Int32(fx.rocdl.perm_b32(partp[j2], ownp[j2], sel_even))
                    )
                    dwords.append(
                        fx.Int32(fx.rocdl.perm_b32(partp[j2], ownp[j2], sel_odd))
                    )
                p_vec = Vec.from_elements(dwords, fx.Int32).bitcast(elem)
                for dc in range_constexpr(ND):
                    a = lds_v16(cur + K_SIZE + (dc * 16 + l16) * V_ROW + kb * 16)
                    o_acc[dc] = Vec(_wmma(a, p_vec, o_acc[dc]))

            if const_expr(double_buffer):
                # The other buffer was last read in iteration j-1 and every wave has
                # passed that iteration's barrier, so it is free to overwrite now.
                store_tile(next_regs, BUF - cur)
                _lds_barrier()
            else:
                _lds_barrier()
                store_tile(next_regs, 0)
                _lds_barrier()

            results = yield [m_new, l_new] + o_acc

        # ---- epilogue: O = O^T / l, 128-bit stores ----
        l_fin = fx.Float32(results[1])
        l_fin = l_fin + l_fin.shuffle_xor(16, WAVE_SIZE)
        inv = fx.Float32(1.0) / l_fin
        # Recomputed rather than kept live across the main loop (VGPR pressure).
        q_base = tok_off(qblk * BLOCK_M + (tid // WAVE_SIZE) * 16 + (tid % 16))
        for dc in range_constexpr(ND):
            if const_expr(dc * 16 < D):
                o = Vec(results[2 + dc])
                # Lane L holds d = 16*dc + 2*si + L/16. After a swap with lane L^16
                # both lanes see all 16 d of the query; lanes 0-15 store d 0..7 and
                # lanes 16-31 store d 8..15.
                own = [fx.Float32(o[si]) * inv for si in range_constexpr(8)]
                part = [own[si].shuffle_xor(16, WAVE_SIZE) for si in range_constexpr(8)]
                ev = [is_h0.select(own[si], part[si]) for si in range_constexpr(8)]
                od = [is_h0.select(part[si], own[si]) for si in range_constexpr(8)]
                out = []
                for si in range_constexpr(4):
                    out.append(is_h0.select(ev[si], ev[si + 4]).to(elem))
                    out.append(is_h0.select(od[si], od[si + 4]).to(elem))
                off = q_base + dc * 16 + hh * 8
                if const_expr(dc * 16 + 8 >= D):
                    # D % 16 == 8: the upper half is padding. Point it past the
                    # descriptor bound so the hardware drops the store.
                    off = is_h0.select(off, 0x3FFFFFFF)
                buf_copy_store(
                    o_buf, off, Vec.from_elements(out, elem), elem=elem, unit_elems=8
                )

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        batch_size: fx.Int32,
        seq_q: fx.Int32,
        seq_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        ctx = CompilationContext.get_current()
        n_qblk = (seq_q + (BLOCK_M - 1)) // BLOCK_M
        launcher = flash_attn_func_kernel(Q, K, V, O, seq_q, seq_kv)
        launcher.launch(
            grid=(n_qblk * H, 1, batch_size), block=(THREADS, 1, 1), stream=stream
        )

        if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1):
            # The gpu.func only exists once launch() has emitted it; an attribute
            # set before that point finds no op and is silently dropped.
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )

    launch_flash_attn_func.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(Q, K, V, O, batch_size, seq_q, seq_kv, stream=None):
        _run_compiled(
            launch_flash_attn_func,
            ptr_arg(Q),
            ptr_arg(K),
            ptr_arg(V),
            ptr_arg(O),
            batch_size,
            seq_q,
            seq_kv,
            fx.Stream(None) if stream is None else stream,
        )

    _launch.block_m = BLOCK_M
    _launch.block_n = BLOCK_N
    _launch.double_buffer = double_buffer
    return _launch
