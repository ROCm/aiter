# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-KV decode attention for gfx11 (RDNA3 / RDNA3.5), bf16/f16: the split-KV
half of the ``unified_attention`` form, for sequences of a few queries (decode,
speculative decode). ``flash_attn_paged_gfx11.py`` takes the long ones.

The 16 rows of the WMMA tile are the ``GROUP`` query heads of one KV head x
``16 / GROUP`` query tokens of one sequence, so a decode token's heads share
every K / V byte the workgroup reads. The workgroup walks one segment of the
sequence's keys (grid y = segment): K rows go straight from the paged cache
into the WMMA A operand (lane L%16 holds key L%16 of the tile), V rows are
transposed through LDS as in the varlen kernel. It writes either the normalized
output (one segment) or the segment's unnormalized O, running max and exp-sum
for the merge kernel: segment s of a sequence with ``n`` keys covers tiles
``[s * t, (s + 1) * t)`` of ``BLOCK_N`` keys, ``t = ceil(n / (segments *
BLOCK_N))``; a segment past the keys writes (-inf, 0) and the merge skips it.

The head dim is split over ``d_split`` waves (1, 2 or 4): wave w loads and
multiplies its ``head_dim / d_split`` columns of K, Q and V and holds that slice
of O^T, which keeps a wave's registers at head_dim 256 (16 x 256 fp32 of O alone
fill half of them). The waves add up their partial S^T through LDS once per tile
(one barrier), then run the same softmax on the same numbers.

Same transposed core as the other gfx11 kernels:

  S^T[key, q] = K[key, :] . Q[q, :]          (A = K from global, B = Q in VGPRs)
  O^T[d, q]  += V^T[d, key] . P^T[key, q]    (A = V^T from LDS, B = P^T in VGPRs)

Causal is bottom-right aligned: query row r of a sequence attends keys
``<= r + seq_kv - seq_q``; a decode row sees the whole context.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .flash_attn_func_gfx11 import LDS_LIMIT_BYTES, LDS_PAD, WAVE_SIZE, WMMA_MNK
from .flash_attn_varlen_func_gfx11 import NEG_BIG, POS_BIG
from .kernels_common import LOG2E as _LOG2E
from .tensor_shim import (
    _run_compiled,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_arg,
    ptr_buf_tensor,
)

KERNEL_NAME = "flash_attn_paged_decode_gfx11_kernel"
NEG_INF = float("-inf")


def build_flash_attn_paged_decode_module_gfx11(
    num_heads,
    head_dim,
    page_size,
    dtype_str="bf16",
    sm_scale=None,
    block_n=16,
    vt_rows=2,
    d_split=1,
    q_reload=False,
    direct=False,
    waves_per_eu=None,
    num_kv_heads=None,
    fast_fp_math=True,
    unsafe_fp_math=True,
):
    """Build the gfx11 paged-KV decode forward and return its launcher.

    ``launch(q, k_cache, v_cache, out, segm_output, segm_max, segm_expsum,
    cu_seqlens_q, seqused_k, block_table, batch_size, max_seqlen_q,
    num_segments, stream=None)``: ``q`` / ``out`` are packed THD
    ``[total_q, num_heads, head_dim]``; the caches ``[num_blocks, page_size,
    num_kv_heads, head_dim]``; ``segm_output`` fp32 ``[batch * max_seqlen_q,
    num_heads, num_segments, next_pow2(head_dim)]``, ``segm_max`` /
    ``segm_expsum`` fp32 ``[batch * max_seqlen_q, num_heads, num_segments]``,
    row ``b * max_seqlen_q + t`` for token ``t`` of sequence ``b`` (unused with
    ``direct``, pass ``out``); ``cu_seqlens_q`` ``batch + 1`` int32 entries,
    ``seqused_k`` ``batch`` entries, ``block_table`` int32 ``[batch,
    max_blocks]``. Sequences with more than ``max_seqlen_q`` queries are left
    alone (the prefill kernel's); ``max_seqlen_q`` is at most ``16 / GROUP`` x
    the number of query tiles the grid covers.

    Args:
        num_heads, head_dim: compile-time head geometry (``head_dim % 8 == 0``).
        page_size: cache rows per block (a multiple of 8).
        block_n: keys per tile (16 or 32).
        vt_rows: keys packed per LDS write in the V -> V^T transpose (2, 4, 8).
        d_split: waves per workgroup, each with ``head_dim / d_split`` columns
            (1, 2 or 4; ``head_dim`` a multiple of ``16 * d_split`` when > 1).
        q_reload: read the Q operand from the (cached) global tensor on every
            tile instead of keeping it in VGPRs.
        direct: one segment: normalize and write ``out`` (bf16 / f16) instead
            of the segment partials.
        num_kv_heads: K / V heads; ``num_heads / num_kv_heads`` must divide 16.
    """
    if head_dim % 8 != 0 or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive multiple of 8, got {head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be bf16 or f16, got {dtype_str!r}")
    if block_n not in (16, 32) or vt_rows not in (2, 4, 8) or block_n % vt_rows != 0:
        raise ValueError(f"unsupported block_n={block_n} / vt_rows={vt_rows}")
    if page_size <= 0 or page_size % 8 != 0:
        raise ValueError(f"page_size must be a positive multiple of 8, got {page_size}")

    H = num_heads
    D = head_dim
    PAGE = int(page_size)
    HKV = num_heads if num_kv_heads is None else int(num_kv_heads)
    if HKV < 1 or H % HKV != 0:
        raise ValueError(f"num_heads={H} must be a multiple of num_kv_heads={HKV}")
    GROUP = H // HKV  # query heads per K/V head = heads per WMMA row group
    if GROUP > 16 or 16 % GROUP != 0:
        raise ValueError(f"num_heads / num_kv_heads must divide 16, got {GROUP}")
    TQ = 16 // GROUP  # query tokens per WMMA tile
    DP = -(-D // WMMA_MNK) * WMMA_MNK
    ND = DP // WMMA_MNK
    DPAD2 = 1 << (D - 1).bit_length()  # the merge's padded row length
    NWD = int(d_split)
    if NWD not in (1, 2, 4) or (NWD > 1 and D % (WMMA_MNK * NWD) != 0):
        raise ValueError(f"d_split={d_split} does not divide head_dim={D} into 16s")
    NDW = ND // NWD  # 16-column chunks per wave
    DW = NDW * WMMA_MNK  # columns per wave
    BLOCK_N = block_n
    NK = BLOCK_N // WMMA_MNK
    THREADS = WAVE_SIZE * NWD
    TOK = H * D  # elements between consecutive Q / O tokens (THD)
    TOK_KV = HKV * D  # elements of one K / V cache row
    PAGE_ELEMS = PAGE * TOK_KV
    is_bf16 = dtype_str == "bf16"
    elem = fx.BFloat16 if is_bf16 else fx.Float16

    V_ROW = BLOCK_N + LDS_PAD
    VT_SIZE = DW * V_ROW  # elements of one wave's V^T slice in LDS
    DBLKS = DW // 8  # 8-column blocks of a wave's slice
    V_BLOCKS = (BLOCK_N // vt_rows) * DBLKS  # vt_rows keys x 8 d per block
    V_PER_THR = -(-V_BLOCKS // WAVE_SIZE)  # each wave stages its own slice
    # Partial S^T exchange (NWD > 1): two parity slots x NWD waves x the
    # lane's NK x 8 fp32, in `elem` units.
    S_LANE = NK * 8  # fp32 per lane
    S_SLOT = S_LANE * WAVE_SIZE  # fp32 per wave slot
    S_ELEMS = 2 * NWD * S_SLOT * 2 if NWD > 1 else 0
    S_BASE = NWD * VT_SIZE  # elem offset of the exchange area (even)
    LDS_ELEMS = S_BASE + S_ELEMS
    if LDS_ELEMS * 2 > LDS_LIMIT_BYTES:
        raise ValueError(
            f"head_dim={D}, block_n={BLOCK_N}, d_split={NWD} exceeds "
            f"{LDS_LIMIT_BYTES} B of LDS"
        )

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(D)
    SCALE_LOG2 = float(sm_scale) * _LOG2E
    CONCAT16 = list(range(16))

    @fx.struct
    class SharedStorage:
        vt: fx.Array[elem, LDS_ELEMS, 16]

    def _wmma(a, b, acc):
        if const_expr(is_bf16):
            return fx.rocdl.wmma_f32_16x16x16_bf16(
                acc.type, a.bitcast(fx.Int16), b.bitcast(fx.Int16), acc
            ).result
        return fx.rocdl.wmma_f32_16x16x16_f16(acc.type, a, b, acc).result

    def _lds_wait():
        # Within one wave: its LDS reads and writes complete in order; wait for
        # them before its V^T tile is overwritten or read back.
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)",
            constraints="",
            has_side_effects=True,
        )

    def _barrier():
        # Across the waves: LDS traffic done, then meet.
        _llvm.inline_asm(
            res=None,
            operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
            constraints="",
            has_side_effects=True,
        )

    def _exp2(x):
        return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def flash_attn_paged_decode_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        SEGM_O: fx.Pointer,
        SEGM_M: fx.Pointer,
        SEGM_L: fx.Pointer,
        CU_Q: fx.Pointer,
        SEQUSED_K: fx.Pointer,
        BLOCK_TABLE: fx.Pointer,
        batch_size: fx.Int32,
        bt_stride: fx.Int32,
        num_segments: fx.Int32,
        q_small: fx.Int32,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().vt.ptr

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
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        l16 = lane % 16
        hh = lane // 16
        is_h0 = hh == 0
        is_w0 = wave == 0
        vt_base = wave * VT_SIZE  # this wave's V^T slice in LDS
        col0 = wave * DW  # this wave's first head-dim column
        bat = fx.block_idx.z
        seg = fx.block_idx.y
        kvh = fx.block_idx.x % HKV
        qt = fx.block_idx.x // HKV

        def table_scalar(ptr, idx, entries):
            rsrc = buffer_ops.create_buffer_resource_from_addr(
                buf_base_i64(ptr), num_records_bytes=fx.Int64(entries) * 4
            )
            return fx.Int32(
                buffer_ops.buffer_load(rsrc, fx.Int32(idx), vec_width=1, is_scalar=True)
            )

        q_start = table_scalar(CU_Q, bat, fx.Int32(batch_size) + 1)
        q_end = table_scalar(CU_Q, bat + 1, fx.Int32(batch_size) + 1)
        seq_q = q_end - q_start
        # Only the short sequences are this kernel's; the others count as empty.
        seq_q = (seq_q <= q_small).select(seq_q, fx.Int32(0))
        seq_kv = table_scalar(SEQUSED_K, bat, batch_size)
        diag = seq_kv - seq_q
        n_kv_tiles = (seq_kv + (BLOCK_N - 1)) // BLOCK_N
        # This segment's tiles, the merge's partition; capped by the causal
        # limit of the tile's last row.
        tiles_per_seg = (seq_kv + (num_segments * BLOCK_N - 1)) // (
            num_segments * BLOCK_N
        )
        tile_lo = seg * tiles_per_seg
        tile_hi = tile_lo + tiles_per_seg
        tile_hi = (tile_hi < n_kv_tiles).select(tile_hi, n_kv_tiles)
        last_vis = (qt + 1) * TQ + diag  # one past the last visible key
        last_vis = (last_vis > 0).select(last_vis, fx.Int32(0))
        cap = (last_vis + (BLOCK_N - 1)) // BLOCK_N
        tile_hi = (tile_hi < cap).select(tile_hi, cap)
        n_tiles = tile_hi - tile_lo
        n_tiles = (n_tiles > 0).select(n_tiles, fx.Int32(0))
        n_tiles = (seq_q > 0).select(n_tiles, fx.Int32(0))
        has_keys = n_tiles > 0

        # Rows: GROUP heads of this KV head, heads fastest, then tokens.
        tok = qt * TQ + l16 // GROUP
        head = kvh * GROUP + l16 % GROUP
        row_ok = tok < seq_q
        q_lim = tok + diag  # keys <= q_lim are visible to this row

        def seq_buf(ptr, seqlen, start):
            slice_bytes = fx.Int64(seqlen) * TOK * 2
            slice_bytes = (seqlen > 0).select(slice_bytes, fx.Int64(0))
            return ptr_buf_tensor(
                buf_base_i64(ptr) + fx.Int64(start) * 2,
                elem=elem,
                unit_elems=8,
                unit_stride=1,
                num_records_bytes=slice_bytes,
            )

        q_buf = seq_buf(Q, seq_q, fx.Int64(q_start) * TOK)
        # This sequence's block-table row; empty (block 0) for a sequence
        # without keys, and a page index past it reads block 0.
        bt_bytes = (seq_kv > 0).select(fx.Int64(bt_stride) * 4, fx.Int64(0))
        bt_buf = ptr_buf_tensor(
            buf_base_i64(BLOCK_TABLE) + fx.Int64(bat) * fx.Int64(bt_stride) * 4,
            elem=fx.Int32,
            unit_elems=1,
            num_records_bytes=bt_bytes,
        )
        last_key = seq_kv - 1
        k_base = buf_base_i64(K) + fx.Int64(kvh) * (D * 2)
        v_base = buf_base_i64(V) + fx.Int64(kvh) * (D * 2)
        GPTR16 = fx.PointerType.get(
            elem.ir_type, address_space=fx.AddressSpace.Global, alignment=16
        )

        def kv_row(base, key, align=1):
            """64-bit address of row `key` of this sequence's K or V, through the
            block table. A key past the last one is clamped onto the last key
            (valid memory; the caller masks or zeroes the row), rounded down to
            a multiple of `align` so that `align` consecutive rows from it stay
            inside one page (`align` divides the page)."""
            if const_expr(align > 1):
                last = (last_key // align) * align
            else:
                last = last_key
            key_c = (key < last).select(key, last)
            key_c = (key_c > 0).select(key_c, fx.Int32(0))
            page = key_c // PAGE
            blk = fx.Int32(buf_copy_load(bt_buf, page, elem=fx.Int32, unit_elems=1))
            off = fx.Int64(blk) * PAGE_ELEMS + fx.Int64((key_c % PAGE) * TOK_KV)
            return base + off * 2

        def load8_at(addr, off):
            """8 elements at element offset `off` from the 64-bit address."""
            p = fx.inttoptr(GPTR16, addr + off * 2)
            return fx.make_view(p, fx.make_layout(8, 1)).load()

        def g_load8(buf, off):
            return buf_copy_load(buf, off, elem=elem, unit_elems=8)

        # ---- Q -> WMMA B operands (zero past seq_q / past head_dim) ----
        zero_e8 = Vec.filled(8, 0.0, elem)
        q_off = tok * TOK + head * D + col0

        def load_q():
            vecs = []
            for c in range_constexpr(NDW):
                lo = zero_e8
                hi = zero_e8
                if const_expr(16 * c < D):  # a partial last chunk only at NWD 1
                    lo = g_load8(q_buf, q_off + 16 * c)
                if const_expr(16 * c + 8 < D):
                    hi = g_load8(q_buf, q_off + 16 * c + 8)
                vecs.append(Vec(lo).shuffle(Vec(hi), CONCAT16))
            return vecs

        if const_expr(not q_reload):
            q_vecs = load_q()

        # ---- K rows straight into the A operand: lane L%16 holds key L%16 ----
        def k_vec(addr, c):
            """Columns col0 + 16c .. col0 + 16c + 15 of the K row at `addr`."""
            lo = zero_e8
            hi = zero_e8
            if const_expr(16 * c < D):
                lo = load8_at(addr, col0 + 16 * c)
            if const_expr(16 * c + 8 < D):
                hi = load8_at(addr, col0 + 16 * c + 8)
            return Vec(lo).shuffle(Vec(hi), CONCAT16)

        # ---- V slice staging: global (gathered) -> VGPRs -> LDS as V^T ----
        def v_block(i):
            blk = lane + i * WAVE_SIZE
            if const_expr(V_PER_THR * WAVE_SIZE > V_BLOCKS):
                blk = (blk < V_BLOCKS).select(blk, V_BLOCKS - 1)
            return blk

        def gload_v(tile):
            key0 = tile * BLOCK_N
            vregs = []
            for i in range_constexpr(V_PER_THR):
                blk = v_block(i)
                kb0 = key0 + (blk // DBLKS) * vt_rows
                d0 = (blk % DBLKS) * 8
                # The vt_rows rows of a block start at a multiple of vt_rows, so
                # they share a page: one block-table lookup for all of them.
                addr0 = kv_row(v_base, kb0, vt_rows)
                rows = []
                for r in range_constexpr(vt_rows):
                    v = Vec(load8_at(addr0, r * TOK_KV + col0 + d0))
                    # A row past the keys would multiply P = 0 by whatever the
                    # clamped page holds (NaN * 0 = NaN): zero it.
                    rows.append((kb0 + r < seq_kv).select(v, zero_e8))
                vregs.append(rows)
            return vregs

        def store_v(vregs):
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
                        dst = vt_base + (d0 + 2 * t + half) * V_ROW + krow0
                        lds_view(dst, vt_rows).store(
                            Vec.from_elements(ws, fx.Int32).bitcast(elem)
                        )

        # ---- partial S^T exchange between the d_split waves ----
        if const_expr(NWD > 1):
            s_ptr = fx.recast_iter(fx.Float32, lds_ptr)

            def s_view(off_f32, n):
                return fx.make_view(
                    fx.add_offset(s_ptr, fx.make_int_tuple(fx.Int32(off_f32))),
                    fx.make_layout(n, 1),
                )

            def s_slot(parity, w):
                return S_BASE // 2 + (parity * NWD + w) * S_SLOT + lane * S_LANE

        # ---- P^T operand assembly ----
        def pack2(a, b):
            """Two f32 P values in [0, 1] -> one dword of elem, a in the low half."""
            if const_expr(is_bf16):
                ra = a.bitcast(fx.Int32)
                rb = b.bitcast(fx.Int32)
                ra = ra + 0x7FFF + (ra.shrui(fx.Int32(16)) & 1)
                rb = rb + 0x7FFF + (rb.shrui(fx.Int32(16)) & 1)
                return fx.Int32(fx.rocdl.perm_b32(rb, ra, fx.Int32(0x07060302)))
            return Vec.from_elements([a.to(elem), b.to(elem)], elem).bitcast(fx.Int32)[
                0
            ]

        sel_even = is_h0.select(fx.Int32(0x05040100), fx.Int32(0x01000504))
        sel_odd = is_h0.select(fx.Int32(0x07060302), fx.Int32(0x03020706))

        zero8 = Vec.filled(8, 0.0, fx.Float32)
        # A row with no visible key from the segment's first tile on starts at
        # +BIG: every p is exactly 0, l and O stay 0 (see the varlen kernel).
        m0 = (q_lim < tile_lo * BLOCK_N).select(
            fx.Float32(POS_BIG), fx.Float32(NEG_BIG)
        )
        init = [m0, fx.Float32(0.0)] + [zero8 for _ in range_constexpr(NDW)]

        store_v(gload_v(tile_lo))
        _lds_wait()

        def tile_step(j32, state):
            m_old = fx.Float32(state[0])
            l_old = fx.Float32(state[1])
            o_acc = [Vec(state[2 + i]) for i in range_constexpr(NDW)]
            tile = tile_lo + j32
            key0 = tile * BLOCK_N

            nxt = j32 + 1
            nxt = (nxt < n_tiles).select(nxt, j32)
            next_v = gload_v(tile_lo + nxt)
            fx.rocdl.sched_barrier(0)
            if const_expr(q_reload):
                qv = load_q()
            else:
                qv = q_vecs

            # ---- S^T = K Q^T over this wave's columns, K rows from the cache ----
            s_acc = []
            for kb in range_constexpr(NK):
                acc = zero8
                k_addr = kv_row(k_base, key0 + kb * 16 + l16)
                for c in range_constexpr(NDW):
                    acc = _wmma(k_vec(k_addr, c), qv[c], acc)
                s_acc.append(Vec(acc))

            if const_expr(NWD > 1):
                # Sum the waves' partial S^T: each writes its slot, all read.
                parity = j32 % 2
                mine = s_slot(parity, wave)
                for kb in range_constexpr(NK):
                    s_view(mine + kb * 8, 8).store(s_acc[kb])
                _barrier()
                for w in range_constexpr(NWD):
                    if const_expr(w == 0):
                        continue
                    other = s_slot(parity, (wave + w) % NWD)
                    for kb in range_constexpr(NK):
                        o_s = Vec(s_view(other + kb * 8, 8).load())
                        s_acc[kb] = Vec.from_elements(
                            [
                                fx.Float32(s_acc[kb][si]) + fx.Float32(o_s[si])
                                for si in range_constexpr(8)
                            ],
                            fx.Float32,
                        )

            s_vals = []
            for kb in range_constexpr(NK):
                row = []
                for si in range_constexpr(8):
                    x = fx.Float32(s_acc[kb][si])
                    key = key0 + kb * 16 + 2 * si + hh
                    x = (key > q_lim).select(fx.Float32(NEG_BIG), x)
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
                for dc in range_constexpr(NDW):
                    a = lds_v16(vt_base + (dc * 16 + l16) * V_ROW + kb * 16)
                    o_acc[dc] = Vec(_wmma(a, p_vec, o_acc[dc]))

            # The V^T reads above complete before the next tile overwrites them.
            _lds_wait()
            store_v(next_v)
            _lds_wait()
            return [m_new, l_new] + o_acc

        for j, state in range(0, n_tiles, 1, init=init):
            results = yield tile_step(fx.Int32(j), state)

        # ---- epilogue: every wave writes its columns, wave 0 the statistics ----
        l_fin = fx.Float32(results[1])
        l_fin = l_fin + l_fin.shuffle_xor(16, WAVE_SIZE)
        # An element offset past every descriptor, so the store is dropped. Its
        # byte offset plus a 16-byte access must not wrap past 2^32 (that would
        # land at the start of the buffer), and nothing is added to it after
        # the gating: every store offset is gated as a whole.
        OOB = fx.Int32(0x3FFFFFF0)

        def gated(off, lane_ok=None):
            """off for rows with keys and a token, OOB otherwise."""
            off = row_ok.select(off, OOB)
            off = has_keys.select(off, OOB)
            if lane_ok is not None:
                off = lane_ok.select(off, OOB)
            return off

        if const_expr(direct):
            # One segment: normalized output, as the varlen kernel's epilogue.
            inv = (l_fin > 0.0).select(fx.Float32(1.0) / l_fin, fx.Float32(0.0))
            o_buf = seq_buf(O, seq_q, fx.Int64(q_start) * TOK)
            for dc in range_constexpr(NDW):
                if const_expr(dc * 16 < D):
                    o = Vec(results[2 + dc])
                    own = [fx.Float32(o[si]) * inv for si in range_constexpr(8)]
                    part = [
                        own[si].shuffle_xor(16, WAVE_SIZE) for si in range_constexpr(8)
                    ]
                    ev = [is_h0.select(own[si], part[si]) for si in range_constexpr(8)]
                    od = [is_h0.select(part[si], own[si]) for si in range_constexpr(8)]
                    out = []
                    for si in range_constexpr(4):
                        out.append(is_h0.select(ev[si], ev[si + 4]).to(elem))
                        out.append(is_h0.select(od[si], od[si + 4]).to(elem))
                    off = q_off + dc * 16 + hh * 8
                    if const_expr(dc * 16 + 8 >= D):
                        # D % 16 == 8: the upper half is padding; drop it.
                        off = is_h0.select(off, OOB)
                    # A row without keys in a one-segment launch has O = 0.
                    off = row_ok.select(off, OOB)
                    buf_copy_store(
                        o_buf,
                        off,
                        Vec.from_elements(out, elem),
                        elem=elem,
                        unit_elems=8,
                    )
        else:
            # Segment partials for the merge kernel, row b * q_small + t (a
            # sequence's slots past its tokens stay unwritten and unread); a row
            # with no visible key in this segment gets max = -inf (O, sum 0).
            nseg64 = fx.Int64(num_segments)
            rows64 = fx.Int64(batch_size) * fx.Int64(q_small) * H
            o_tab = ptr_buf_tensor(
                buf_base_i64(SEGM_O),
                elem=fx.Float32,
                unit_elems=4,
                unit_stride=1,
                num_records_bytes=rows64 * nseg64 * (DPAD2 * 4),
            )
            m_tab = ptr_buf_tensor(
                buf_base_i64(SEGM_M),
                elem=fx.Float32,
                unit_elems=1,
                num_records_bytes=rows64 * nseg64 * 4,
            )
            l_tab = ptr_buf_tensor(
                buf_base_i64(SEGM_L),
                elem=fx.Float32,
                unit_elems=1,
                num_records_bytes=rows64 * nseg64 * 4,
            )
            row_seg = ((bat * q_small + tok) * H + head) * num_segments + seg
            m_log2 = fx.Float32(results[0]) * fx.Float32(SCALE_LOG2)
            m_out = (l_fin > 0.0).select(m_log2, fx.Float32(NEG_INF))
            # A segment without keys for this row still writes its statistics,
            # (-inf, 0), so the merge needs no knowledge of the partition; its O
            # slot stays unwritten and the merge skips it by the zero exp-sum.
            m_out = has_keys.select(m_out, fx.Float32(NEG_INF))
            l_out = has_keys.select(l_fin, fx.Float32(0.0))
            stat_off = row_ok.select(row_seg, OOB)
            if const_expr(NWD > 1):
                stat_off = is_w0.select(stat_off, OOB)
            stat_off = is_h0.select(stat_off, OOB)
            buf_copy_store(m_tab, stat_off, m_out, elem=fx.Float32, unit_elems=1)
            buf_copy_store(l_tab, stat_off, l_out, elem=fx.Float32, unit_elems=1)
            o_row = row_seg * DPAD2 + col0
            for dc in range_constexpr(NDW):
                o = Vec(results[2 + dc])
                own = [fx.Float32(o[si]) for si in range_constexpr(8)]
                part = [own[si].shuffle_xor(16, WAVE_SIZE) for si in range_constexpr(8)]
                ev = [is_h0.select(own[si], part[si]) for si in range_constexpr(8)]
                od = [is_h0.select(part[si], own[si]) for si in range_constexpr(8)]
                vals = []
                for si in range_constexpr(4):
                    vals.append(is_h0.select(ev[si], ev[si + 4]))
                    vals.append(is_h0.select(od[si], od[si + 4]))
                for h4 in range_constexpr(2):
                    buf_copy_store(
                        o_tab,
                        gated(o_row + dc * 16 + hh * 8 + 4 * h4),
                        Vec.from_elements(vals[4 * h4 : 4 * h4 + 4], fx.Float32),
                        elem=fx.Float32,
                        unit_elems=4,
                    )

    @flyc.jit
    def launch_flash_attn_paged_decode(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        SEGM_O: fx.Pointer,
        SEGM_M: fx.Pointer,
        SEGM_L: fx.Pointer,
        CU_Q: fx.Pointer,
        SEQUSED_K: fx.Pointer,
        BLOCK_TABLE: fx.Pointer,
        batch_size: fx.Int32,
        n_qtile: fx.Int32,
        bt_stride: fx.Int32,
        num_segments: fx.Int32,
        q_small: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        flash_attn_paged_decode_kernel(
            Q,
            K,
            V,
            O,
            SEGM_O,
            SEGM_M,
            SEGM_L,
            CU_Q,
            SEQUSED_K,
            BLOCK_TABLE,
            batch_size,
            bt_stride,
            num_segments,
            q_small,
            value_attrs={
                "rocdl.waves_per_eu": (
                    int(waves_per_eu)
                    if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1)
                    else None
                ),
            },
        ).launch(
            grid=(n_qtile * HKV, num_segments, batch_size),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_flash_attn_paged_decode.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(
        Q,
        K,
        V,
        O,
        segm_output,
        segm_max,
        segm_expsum,
        cu_seqlens_q,
        seqused_k,
        block_table,
        batch_size,
        max_seqlen_q,
        num_segments,
        stream=None,
    ):
        """max_seqlen_q: the longest sequence this kernel handles (its q_small);
        the grid covers ceil(max_seqlen_q / TQ) query tiles per sequence."""
        if direct and int(num_segments) != 1:
            raise ValueError("built with direct=True: num_segments must be 1")
        n_qtile = -(-int(max_seqlen_q) // TQ)
        _run_compiled(
            launch_flash_attn_paged_decode,
            ptr_arg(Q),
            ptr_arg(K),
            ptr_arg(V),
            ptr_arg(O),
            ptr_arg(segm_output),
            ptr_arg(segm_max),
            ptr_arg(segm_expsum),
            ptr_arg(cu_seqlens_q, fx.Int32),
            ptr_arg(seqused_k, fx.Int32),
            ptr_arg(block_table, fx.Int32),
            batch_size,
            n_qtile,
            block_table.stride(0),
            int(num_segments),
            int(max_seqlen_q),
            fx.Stream(None) if stream is None else stream,
        )

    _launch.tq = TQ
    _launch.block_n = BLOCK_N
    _launch.d_split = NWD
    _launch.dpad2 = DPAD2
    _launch.direct = direct
    return _launch
