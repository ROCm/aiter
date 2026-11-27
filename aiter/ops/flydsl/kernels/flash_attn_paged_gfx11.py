# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-KV flash attention forward for gfx11 (RDNA3 / RDNA3.5), bf16/f16: the
``unified_attention`` form (vLLM's paged attention) of ``flash_attn_varlen_func_gfx11.py``.

Same core as the varlen kernel, computed transposed so that every lane owns one
query row:

  S^T[key, q] = K[key, :] . Q[q, :]          (A = K from LDS, B = Q in VGPRs)
  O^T[d, q]  += V^T[d, key] . P^T[key, q]    (A = V^T from LDS, B = P^T in VGPRs)

Queries are packed THD (``[total_q, heads, head_dim]`` + ``cu_seqlens_q``). Keys
and values live in a paged cache ``[num_blocks, page, kv_heads, head_dim]``:
sequence ``b`` has ``seqused_k[b]`` keys, key ``t`` is row ``t % page`` of block
``block_table[b, t // page]``. A K / V tile therefore gathers its rows: each
thread reads the block id of the page its rows are on (one dword from the
sequence's block-table row, bounded) and forms a 64-bit address, since a KV
cache can exceed the 4 GiB a buffer descriptor spans. Rows past ``seqused_k``
are clamped onto the sequence's last page (a valid address), masked out of the
softmax on the last tile, and zeroed before they reach the V^T operand.

Causal is bottom-right aligned as in flash-attn and ``unified_attention``: query
row r of a sequence attends keys ``<= r + seq_kv - seq_q``, so a chunk of queries
over a longer context (chunked prefill, prefix caching, speculative decode) sees
its whole prefix, and decode rows (one query over the context) see everything.
A query block past a shorter sequence's end runs no KV tile and its stores fall
outside its descriptor. ``head_dim`` only needs to be a multiple of 8: it is
zero-padded to the WMMA K of 16 inside the kernel.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .flash_attn_func_gfx11 import (
    LDS_LIMIT_BYTES,
    LDS_PAD,
    WAVE_SIZE,
    WMMA_MNK,
    lds_bytes,
)
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

KERNEL_NAME = "flash_attn_paged_gfx11_kernel"


def build_flash_attn_paged_module_gfx11(
    num_heads,
    head_dim,
    page_size,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    num_waves=8,
    block_n=32,
    vt_rows=2,
    double_buffer=True,
    head_group=None,
    waves_per_eu=None,
    num_kv_heads=None,
    fast_fp_math=True,
    unsafe_fp_math=True,
):
    """Build the gfx11 paged-KV forward and return its launcher.

    ``launch(q, k_cache, v_cache, out, cu_seqlens_q, seqused_k, block_table,
    batch_size, max_seqlen_q, stream=None)``: ``q`` / ``out`` are packed THD
    ``[total_q, num_heads, head_dim]``; ``k_cache`` / ``v_cache`` are
    ``[num_blocks, page_size, num_kv_heads, head_dim]``; ``cu_seqlens_q`` has
    ``batch + 1`` int32 entries, ``seqused_k`` ``batch`` entries (keys per
    sequence, cache rows already written), ``block_table`` is int32
    ``[batch, max_blocks]`` (row stride ``max_blocks``). All on the device.

    Args:
        num_heads, head_dim: compile-time head geometry (``head_dim % 8 == 0``).
        page_size: cache rows per block (a multiple of 8).
        causal: bottom-right aligned causal mask (see the module docstring);
            the only form ``unified_attention`` takes.
        num_waves: waves per workgroup; each wave owns 16 query rows.
        block_n: keys per main-loop tile (multiple of 16).
        vt_rows: keys packed per LDS write in the V -> V^T transpose (2, 4, 8).
        double_buffer: ping-pong the K/V^T tile in LDS (one barrier per tile).
            Dropped automatically when two buffers do not fit in LDS.
        head_group: heads walked together by consecutive workgroups (divides
            num_heads); None = all heads.
        waves_per_eu: ``rocdl.waves_per_eu`` occupancy hint.
        num_kv_heads: K / V heads (GQA / MQA); None = num_heads.

    The builder does not check launch arguments. The caller must guarantee
    ``max_seqlen_q >= 1``, ``sm_scale > 0``, every sequence of Q / O below 2 GiB
    (32-bit buffer offsets), block ids below ``num_blocks`` for the pages a
    sequence uses, 16-byte aligned tensors, and Q / O as plain THD. Sequences
    may be empty (no queries or no keys).
    """
    if head_dim % 8 != 0 or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive multiple of 8, got {head_dim}")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be bf16 or f16, got {dtype_str!r}")
    if block_n % WMMA_MNK != 0 or vt_rows not in (2, 4, 8) or block_n % vt_rows != 0:
        raise ValueError(f"unsupported block_n={block_n} / vt_rows={vt_rows}")
    if page_size <= 0 or page_size % 8 != 0:
        raise ValueError(f"page_size must be a positive multiple of 8, got {page_size}")

    H = num_heads
    D = head_dim
    PAGE = int(page_size)
    HKV = num_heads if num_kv_heads is None else int(num_kv_heads)
    if HKV < 1 or H % HKV != 0:
        raise ValueError(f"num_heads={H} must be a multiple of num_kv_heads={HKV}")
    GROUP = H // HKV  # query heads per K/V head
    if head_group is None:
        head_group = num_heads
    HEAD_GROUP = head_group if head_group >= 1 and num_heads % head_group == 0 else 1
    DP = -(-D // WMMA_MNK) * WMMA_MNK
    ND = DP // WMMA_MNK
    BLOCK_M = WMMA_MNK * num_waves
    BLOCK_N = block_n
    NK = BLOCK_N // WMMA_MNK
    THREADS = num_waves * WAVE_SIZE
    TOK = H * D  # elements between consecutive Q / O tokens (THD)
    TOK_KV = HKV * D  # elements of one K / V cache row
    PAGE_ELEMS = PAGE * TOK_KV  # elements of one cache block
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

    DBLKS = D // 8  # 8-element chunks per row
    K_CHUNKS = BLOCK_N * DBLKS
    K_PER_THR = -(-K_CHUNKS // THREADS)
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
    def flash_attn_paged_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        CU_Q: fx.Pointer,
        SEQUSED_K: fx.Pointer,
        BLOCK_TABLE: fx.Pointer,
        batch_size: fx.Int32,
        n_qblk_max: fx.Int32,
        bt_stride: fx.Int32,
        q_small: fx.Int32,
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
        # Sequence slowest (grid z): consecutive blocks then work on the same
        # tokens of one sequence (several heads of the same rows share cache
        # lines).
        lin = fx.block_idx.x
        bat = fx.block_idx.z

        # This sequence's query range and key count, as scalar (SMEM) loads: the
        # lengths stay in SGPRs, so loop bounds and descriptors are uniform. The
        # table descriptors are bounded, so an index past them reads 0.
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
        # Sequences of up to q_small queries (decode rows) belong to the split-KV
        # kernel: treat them as empty here, so this block loads nothing and
        # every store falls outside its (empty) descriptor.
        seq_q = (seq_q > q_small).select(seq_q, fx.Int32(0))
        seq_kv = table_scalar(SEQUSED_K, bat, batch_size)
        # Causal diagonal offset (bottom-right alignment): row r sees keys
        # <= r + diag. 0 for equal lengths, the context length for decode rows.
        diag = seq_kv - seq_q

        # The whole body as a closure under a wave-uniform guard (scf.if): a block
        # of a sequence this kernel does not handle (none, or left to the decode
        # kernel) exits after the three scalar loads instead of streaming a KV tile
        # and issuing dropped stores. In a mixed batch those idle blocks are most
        # of the grid.
        def _body():
            n_qblk = (seq_q + (BLOCK_M - 1)) // BLOCK_M
            # Linear block id -> (head, qblk), HEAD_GROUP heads fastest (see the
            # varlen kernel). The grid is sized for n_qblk_max query blocks.
            per_group = n_qblk_max * HEAD_GROUP
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
            # A block past this sequence's last query block does nothing: it walks no
            # KV tile and its rows are past seq_q, so the Q loads return 0 and the
            # O stores are dropped by the descriptor.
            active = qblk < n_qblk
            if const_expr(causal):
                # Longest rows first: the diagonal blocks have the most KV tiles.
                qblk = active.select(n_qblk - 1 - qblk, qblk)

            def seq_buf(ptr, seqlen, start):
                # Q / O: bounded per sequence, up to the end of its last token row, so
                # loads past seq_q return 0 and stores past it are dropped. An empty
                # sequence gets an empty descriptor.
                slice_bytes = fx.Int64(seqlen) * TOK * 2
                slice_bytes = (seqlen > 0).select(slice_bytes, fx.Int64(0))
                base = buf_base_i64(ptr) + fx.Int64(start) * 2
                return ptr_buf_tensor(
                    base,
                    elem=elem,
                    unit_elems=8,
                    unit_stride=1,
                    num_records_bytes=slice_bytes,
                )

            q_buf = seq_buf(Q, seq_q, fx.Int64(q_start) * TOK)
            o_buf = seq_buf(O, seq_q, fx.Int64(q_start) * TOK)
            # This sequence's block-table row; a page index past it reads block 0.
            bt_buf = ptr_buf_tensor(
                buf_base_i64(BLOCK_TABLE) + fx.Int64(bat) * fx.Int64(bt_stride) * 4,
                elem=fx.Int32,
                unit_elems=1,
                num_records_bytes=fx.Int64(bt_stride) * 4,
            )
            # Rows past seq_kv are clamped onto the sequence's last key (its page).
            last_key = seq_kv - 1

            def g_load8(buf, off):
                return buf_copy_load(buf, off, elem=elem, unit_elems=8)

            def tok_off(tok):
                return tok * TOK + head * D

            kv_head = head // GROUP if GROUP > 1 else head  # GROUP is a Python int
            k_base = buf_base_i64(K) + fx.Int64(kv_head * D) * 2
            v_base = buf_base_i64(V) + fx.Int64(kv_head * D) * 2
            # 16-byte aligned global pointer type for the gathered K / V loads.
            GPTR16 = fx.PointerType.get(
                elem.ir_type, address_space=fx.AddressSpace.Global, alignment=16
            )

            def kv_row(base, key, align=1):
                """64-bit address of row `key` of this sequence's K or V, through
                the block table. A key past the last one is clamped onto the last
                key (valid memory; the caller masks or zeroes the row), rounded
                down to a multiple of `align` so that `align` consecutive rows
                from it stay inside one page (`align` divides the page)."""
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
                # Keys this block's last row can see: its row index + diag, clamped
                # at 0 so a block with no visible keys walks no tile.
                last_vis = (qblk + 1) * BLOCK_M + diag
                last_vis = (last_vis > 0).select(last_vis, fx.Int32(0))
                n_tiles = (last_vis + (BLOCK_N - 1)) // BLOCK_N
                n_tiles = (n_tiles < n_kv_tiles).select(n_tiles, n_kv_tiles)
                q_lim = q_row + diag
            else:
                n_tiles = n_kv_tiles
            n_tiles = active.select(n_tiles, fx.Int32(0))

            # ---- K / V tile staging: global (gathered) -> VGPRs -> LDS (K rows, V^T) ----
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
                    d0 = (chunk % DBLKS) * 8
                    kregs.append(load8_at(kv_row(k_base, key0 + row), d0))
                vregs = []
                for i in range_constexpr(V_PER_THR):
                    blk = v_block(i)
                    kb0 = key0 + (blk // DBLKS) * vt_rows
                    d0 = (blk % DBLKS) * 8
                    # The vt_rows rows of a block start at a multiple of vt_rows,
                    # so they share a page: one block-table lookup for all of them.
                    addr0 = kv_row(v_base, kb0, vt_rows)
                    rows = []
                    for r in range_constexpr(vt_rows):
                        v = Vec(load8_at(addr0, r * TOK_KV + d0))
                        # A row past the keys would multiply P = 0 by whatever the
                        # clamped page holds (NaN * 0 = NaN): zero it.
                        rows.append((kb0 + r < seq_kv).select(v, zero_e8))
                    vregs.append(rows)
                return kregs, vregs

            def store_tile(regs, bb):
                kregs, vregs = regs
                for i in range_constexpr(K_PER_THR):
                    chunk = k_chunk(i)
                    dst = bb + (chunk // DBLKS) * K_ROW + (chunk % DBLKS) * 8
                    lds_view(dst, 8).store(Vec(kregs[i]))
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
                return Vec.from_elements([a.to(elem), b.to(elem)], elem).bitcast(
                    fx.Int32
                )[0]

            # Lane L holds keys 2*si + L/16 of its query; the B operand wants all 16 keys
            # in order. After swapping packed pairs with lane L^16, one v_perm_b32 per
            # dword interleaves own/partner halves (selector depends on the lane half).
            sel_even = is_h0.select(fx.Int32(0x05040100), fx.Int32(0x01000504))
            sel_odd = is_h0.select(fx.Int32(0x07060302), fx.Int32(0x03020706))

            zero8 = Vec.filled(8, 0.0, fx.Float32)
            m0 = fx.Float32(NEG_BIG)
            if const_expr(causal):
                # A row that can see no key at all: with m = NEG_BIG its masked x - m
                # would be rounding noise and exp2 of it 1 or inf. Starting such a
                # row at +BIG makes every p exactly 0, so l and O stay 0.
                m0 = (q_lim < 0).select(fx.Float32(POS_BIG), m0)
            init = [m0, fx.Float32(0.0)] + [zero8 for _ in range_constexpr(ND)]

            store_tile(gload_tile(fx.Int32(0)), 0)
            _lds_barrier()

            def tile_step(j32, state, mask_tail):
                """One KV tile: S^T, online softmax, O^T update, next tile into LDS.
                mask_tail (compile-time): mask keys at or past seq_kv; only the last
                KV tile can hold such keys, so the main loop runs without it."""
                m_old = fx.Float32(state[0])
                l_old = fx.Float32(state[1])
                o_acc = [Vec(state[2 + i]) for i in range_constexpr(ND)]
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
                        if const_expr(causal or mask_tail):
                            key = j32 * BLOCK_N + kb * 16 + 2 * si + hh
                        if const_expr(causal):
                            x = (key > q_lim).select(fx.Float32(NEG_BIG), x)
                        if const_expr(mask_tail):
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
                        [fx.Float32(o[si]) * alpha for si in range_constexpr(8)],
                        fx.Float32,
                    )
                    for o in o_acc
                ]

                for kb in range_constexpr(NK):
                    own = []
                    for si in range_constexpr(8):
                        p = _exp2(
                            fx.fma(s_vals[kb][si], fx.Float32(SCALE_LOG2), neg_mc)
                        )
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

                return [m_new, l_new] + o_acc

            # Only the last KV tile can hold keys at or past seq_kv, and only when the
            # tiles walked here reach it (causal rows may stop earlier). The tiles
            # before it run without the tail mask; the partial one gets it.
            partial = ((seq_kv % BLOCK_N) > 0).select(fx.Int32(1), fx.Int32(0))
            n_full = n_tiles - (n_tiles == n_kv_tiles).select(partial, fx.Int32(0))

            for j, state in range(0, n_full, 1, init=init):
                results = yield tile_step(fx.Int32(j), state, False)

            for j, state in range(n_full, n_tiles, 1, init=results):
                results = yield tile_step(fx.Int32(j), state, True)

            # ---- epilogue: O = O^T / l, 128-bit stores ----
            l_fin = fx.Float32(results[1])
            l_fin = l_fin + l_fin.shuffle_xor(16, WAVE_SIZE)
            # l is 0 only when no key was seen (a sequence without keys): O = 0 then.
            inv = (l_fin > 0.0).select(fx.Float32(1.0) / l_fin, fx.Float32(0.0))
            # Recomputed rather than kept live across the main loop (VGPR pressure).
            q_base = tok_off(qblk * BLOCK_M + (tid // WAVE_SIZE) * 16 + (tid % 16))
            for dc in range_constexpr(ND):
                if const_expr(dc * 16 < D):
                    o = Vec(results[2 + dc])
                    # Lane L holds d = 16*dc + 2*si + L/16. After a swap with lane L^16
                    # both lanes see all 16 d of the query; lanes 0-15 store d 0..7 and
                    # lanes 16-31 store d 8..15.
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
                    off = q_base + dc * 16 + hh * 8
                    if const_expr(dc * 16 + 8 >= D):
                        # D % 16 == 8: the upper half is padding. Point it past the
                        # descriptor bound so the hardware drops the store.
                        off = is_h0.select(off, 0x3FFFFFFF)
                    buf_copy_store(
                        o_buf,
                        off,
                        Vec.from_elements(out, elem),
                        elem=elem,
                        unit_elems=8,
                    )

        if seq_q > 0:
            _body()

    @flyc.jit
    def launch_flash_attn_paged(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        CU_Q: fx.Pointer,
        SEQUSED_K: fx.Pointer,
        BLOCK_TABLE: fx.Pointer,
        batch_size: fx.Int32,
        max_seqlen_q: fx.Int32,
        bt_stride: fx.Int32,
        q_small: fx.Int32,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: default is evaluated once at import on purpose
            None
        ),
    ):
        # The grid covers the longest sequence.
        n_qblk = (max_seqlen_q + (BLOCK_M - 1)) // BLOCK_M
        flash_attn_paged_kernel(
            Q,
            K,
            V,
            O,
            CU_Q,
            SEQUSED_K,
            BLOCK_TABLE,
            batch_size,
            n_qblk,
            bt_stride,
            q_small,
            value_attrs={
                "rocdl.waves_per_eu": (
                    int(waves_per_eu)
                    if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1)
                    else None
                ),
            },
        ).launch(grid=(n_qblk * H, 1, batch_size), block=(THREADS, 1, 1), stream=stream)

    launch_flash_attn_paged.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(
        Q,
        K,
        V,
        O,
        cu_seqlens_q,
        seqused_k,
        block_table,
        batch_size,
        max_seqlen_q,
        stream=None,
        q_small=0,
    ):
        """q_small: sequences with at most this many queries are skipped (left
        to the split-KV decode kernel); 0 handles every sequence here."""
        _run_compiled(
            launch_flash_attn_paged,
            ptr_arg(Q),
            ptr_arg(K),
            ptr_arg(V),
            ptr_arg(O),
            # int32 tables: the kernel indexes them in elements
            ptr_arg(cu_seqlens_q, fx.Int32),
            ptr_arg(seqused_k, fx.Int32),
            ptr_arg(block_table, fx.Int32),
            batch_size,
            max_seqlen_q,
            block_table.stride(0),
            int(q_small),
            fx.Stream(None) if stream is None else stream,
        )

    _launch.block_m = BLOCK_M
    _launch.block_n = BLOCK_N
    _launch.double_buffer = double_buffer
    _launch.page_size = PAGE
    return _launch
