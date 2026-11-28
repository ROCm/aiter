from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _v4_load_q_regs,
    _v4_qk_mx_split_regs,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.constexpr_function
def _slot_layout(block_k, num_warps, split, cga):
    per = block_k // split
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i] for i in range(per.bit_length() - 1)],
        lane_bases=[[0]] * 5,
        warp_bases=[
            [per << i if 1 << i < split else 0]
            for i in range(num_warps.bit_length() - 1)
        ],
        block_bases=cga,
        shape=[block_k],
    )


@gluon.jit
def _pv_scale_table(tbuf, sc_tab, ks_shared: gl.constexpr, BLOCK_K: gl.constexpr):
    """sc_tab[j, k] = bf16 2^(e[k, j] - 127) for the tile's keys.

    One key per lane reads its record's scale dwords (bytes 576..591, dword w =
    [e_2w, e_2w, e_2w+1, e_2w+1]) and stores eight bf16 values (bits e << 7), so
    PV reads each block's scale vector already packed for v_pk_mul_bf16."""
    KS_LANE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [1, 1], [1, 0])
    d = tbuf.reinterpret(gl.int32, [BLOCK_K, 32], ks_shared).slice(16, 4, 1)
    d = d.load(KS_LANE)
    v = gl.reshape(gl.join((d & 0xFF) << 7, (d >> 9) & 0x7F80), [BLOCK_K, 8])
    dst = sc_tab.reinterpret(shape=[8, BLOCK_K], layout=sc_tab.layout)
    dst.permute([1, 0]).store(v.to(gl.int16).to(gl.bfloat16, bitcast=True))


@gluon.jit
def _pv(acc, p_bf, kbuf, v, sc_tab, P_L: gl.constexpr, V_L: gl.constexpr):
    """acc[J] += (P * 2^(e_J - 127)) @ V[:, 64J:64J+64] for the eight 64-column
    blocks; block J+1's V read is in flight while block J computes. ``v`` is
    block 0, already read."""
    for J in gl.static_range(8):
        s = sc_tab.index(J).reshape([kbuf.shape[0]]).load(gl.SliceLayout(0, P_L))
        if J < 7:
            v_next = kbuf.slice(64 * (J + 1), 64, dim=1).load(V_L)
            gl.amd.hint.sched_barrier()
        pv = gl.amd.gfx1250.wmma(p_bf * s[None, :], v.to(gl.bfloat16), acc[J])
        acc = acc[:J] + (pv,) + acc[J + 1 :]
        if J < 7:
            v = v_next
    return acc


@gluon.jit
def _slot_rows(
    slot_buf,
    blk_rows,
    num_slots,
    ROW_LAYOUT: gl.constexpr,
    BS: gl.constexpr,
    IDENT_ROWS: gl.constexpr,
    LANE_MAP: gl.constexpr,
):
    """Slot vector -> descriptor rows, wave-uniform as TDM wants them.

    The unsigned min keeps every row inside the descriptor (a negative slot
    becomes num_slots - 1), so no gather relies on TDM out-of-bounds handling.
    LANE_MAP maps one slot per lane and broadcasts through LDS, which wins for
    unsplit 16-head blocks; the 32-head kernel and split-K decode prefer a
    scalar chain per slot."""
    BLOCK_K: gl.constexpr = slot_buf.shape[1]
    if LANE_MAP:
        layout: gl.constexpr = gl.BlockedLayout([BLOCK_K // 32], [32], [1], [0])
    else:
        layout: gl.constexpr = ROW_LAYOUT
    slot = slot_buf.reshape([BLOCK_K]).load(layout).to(gl.uint32, bitcast=True)
    row = gl.minimum(slot, (num_slots - 1).to(gl.uint32))
    if not IDENT_ROWS:
        LOG2_BS: gl.constexpr = BS.bit_length() - 1
        row = (row >> LOG2_BS) * blk_rows + (row & (BS - 1))
    return gl.convert_layout(row.to(gl.int32, bitcast=True), ROW_LAYOUT)


@gluon.jit
def _req_lookup(slot_buf, bt_row, max_blocks, BS: gl.constexpr):
    BLOCK_K: gl.constexpr = slot_buf.shape[1]
    pos = slot_buf.reshape([BLOCK_K]).load(gl.BlockedLayout([1], [32], [1], [0]))
    blk = pos >> (BS.bit_length() - 1)
    ok = (pos >= 0) & (blk < max_blocks)
    return pos, gl.amd.cdna4.buffer_load(bt_row, blk, mask=ok, other=0)


@gluon.jit
def _req_rows(pend, max_blocks, blk_rows, BS: gl.constexpr, ROW_LAYOUT: gl.constexpr):
    pos, base = pend
    ok = (pos >= 0) & ((pos >> (BS.bit_length() - 1)) < max_blocks)
    row = gl.where(ok, base * blk_rows + (pos & (BS - 1)), 0)
    return gl.convert_layout(row, ROW_LAYOUT)


@gluon.jit
def _gather(j, row, num_iters, kv_desc, tail_desc, kv_bufs, tail_bufs):
    """Gather tile j's records (NoPE + tail) into ring slot j % NBUF; predicated
    off past the split's last tile."""
    live = (j < num_iters).to(gl.int32)
    nb = j % kv_bufs.shape[0]
    kv = tdm.update_tensor_descriptor(kv_desc, pred=live)
    tail = tdm.update_tensor_descriptor(tail_desc, pred=live)
    tdm.async_gather(kv, row, kv_bufs.index(nb))
    tdm.async_gather(tail, row, tail_bufs.index(nb))


_pa_decode_sparse_fp8_repr = make_kernel_repr(
    "_pa_decode_sparse_fp8",
    ["BLOCK_H", "BLOCK_K", "KV_SPLITS", "NBUF", "HAS_INVALID", "HAS_LSE", "REQ_TOPK"],
)


@gluon.jit(repr=_pa_decode_sparse_fp8_repr)
def _pa_decode_sparse_fp8(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    attn_sink_ptr,
    out_ptr,
    lse_ptr,
    total_rows,
    blk_rows,
    num_slots,
    req_id_ptr,
    block_table_ptr,
    bt_stride,
    max_blocks,
    topk_stride,
    q_stride_t: gl.constexpr,
    q_stride_h: gl.constexpr,
    mp_stride_t: gl.constexpr,
    mp_stride_k: gl.constexpr,
    mp_stride_h: gl.constexpr,
    ap_stride_t: gl.constexpr,
    ap_stride_k: gl.constexpr,
    ap_stride_h: gl.constexpr,
    ap_stride_d: gl.constexpr,
    out_stride_t: gl.constexpr,
    out_stride_h: gl.constexpr,
    out_stride_d: gl.constexpr,
    H: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    MAIN_BLOCK_SIZE: gl.constexpr,
    HAS_INVALID: gl.constexpr,
    HAS_LSE: gl.constexpr,
    NBUF: gl.constexpr,
    LAZY_TH: gl.constexpr,
    IDENT_ROWS: gl.constexpr,
    REQ_TOPK: gl.constexpr,
    num_warps: gl.constexpr,
):
    """fp8_k3_mla_640 sparse MLA, one wave per (query, head block, KV split).

    Per 32-key tile: TDM-gather the 640 B records (512 B NoPE + 128 B tail) two
    tiles ahead into an NBUF ring, QK as five FP8 MX WMMAs (Q in VGPRs), base-2
    online softmax with lazy rescaling, PV straight from the fp8 tile. With
    KV_SPLITS > 1 every split, idle ones too, writes (m, l, acc) partials.
    """
    gl.static_assert(num_warps == 1 and BLOCK_K == 32 and NBUF >= 3)
    gl.static_assert(BLOCK_H == 16 or BLOCK_H == 32)
    NOPE: gl.constexpr = 512
    ROPE: gl.constexpr = 64
    TAIL: gl.constexpr = 128
    LOG2E: gl.constexpr = 1.4426950408889634
    NSLOT: gl.constexpr = 8 if REQ_TOPK else 4  # REQ_TOPK fetches one more ahead

    QK_L: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [], instr_shape=[16, 16, 128])
    Q_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=QK_L, k_width=16)
    K_L: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=QK_L, k_width=16)
    QS_L: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(Q_L, [BLOCK_H, 4], 32)
    KS_L: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(K_L, [BLOCK_K, 4], 32)
    PV_L: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [], instr_shape=[16, 16, 32])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=PV_L, k_width=8)
    V_L: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=PV_L, k_width=8)
    SLOT_L: gl.constexpr = _slot_layout(BLOCK_K, 1, 1, [])
    COL_L: gl.constexpr = gl.SliceLayout(0, QK_L)
    ROW_L: gl.constexpr = gl.SliceLayout(1, QK_L)
    OUT_L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [1, 1], [1, 0])
    OROW_L: gl.constexpr = gl.SliceLayout(1, OUT_L)

    # [rows, width] row-major, 16 elements of pad per row
    kv_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NOPE, 16]], [BLOCK_K, NOPE], [1, 0]
    )
    tail_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TAIL, 16]], [BLOCK_K, TAIL], [1, 0]
    )
    ks_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TAIL // 4, 4]], [BLOCK_K, TAIL // 4], [1, 0]
    )
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    t = gl.num_programs(0) - 1 - gl.program_id(0)
    pid_h = gl.program_id(1)
    pid_k = gl.program_id(2)
    h0 = pid_h * BLOCK_H
    h_o = h0 + gl.arange(0, BLOCK_H, layout=OROW_L)
    m_ptr = m_partial_ptr + t * mp_stride_t + pid_k * mp_stride_k
    l_ptr = l_partial_ptr + t * mp_stride_t + pid_k * mp_stride_k
    ml_offs = (h_o * mp_stride_h).to(gl.int32)

    kv_start = gl.load(kv_indptr_ptr + t)
    kv_len = gl.load(kv_indptr_ptr + t + 1) - kv_start
    num_tiles = gl.cdiv(kv_len, BLOCK_K)
    tiles_per_segment = gl.cdiv(num_tiles, KV_SPLITS)
    if pid_k * tiles_per_segment >= num_tiles:
        zero = gl.zeros([BLOCK_H], gl.float32, OROW_L)
        if KV_SPLITS == 1:  # an empty row: zero output, LSE -inf
            d = gl.arange(0, NOPE, layout=gl.SliceLayout(0, OUT_L))
            offs = h_o[:, None] * out_stride_h + d[None, :] * out_stride_d
            zeros = gl.zeros([BLOCK_H, NOPE], out_ptr.dtype.element_ty, OUT_L)
            gl.amd.cdna4.buffer_store(
                zeros, out_ptr + t * out_stride_t, offs.to(gl.int32), (h_o < H)[:, None]
            )
            if HAS_LSE:
                gl.amd.cdna4.buffer_store(
                    zero - float("inf"), lse_ptr + t * H, h_o.to(gl.int32), h_o < H
                )
        else:  # an idle split: m = -inf, l = 0, so the reduce needs no kv_indptr
            gl.amd.cdna4.buffer_store(zero - float("inf"), m_ptr, ml_offs, h_o < H)
            gl.amd.cdna4.buffer_store(zero, l_ptr, ml_offs, h_o < H)
        return
    tile_start = pid_k * tiles_per_segment
    num_iters = gl.minimum(tile_start + tiles_per_segment, num_tiles) - tile_start

    kv_bufs = gl.allocate_shared_memory(
        kv_ptr.dtype.element_ty, [NBUF, BLOCK_K, NOPE], kv_shared
    )
    tail_bufs = gl.allocate_shared_memory(
        kv_ptr.dtype.element_ty, [NBUF, BLOCK_K, TAIL], tail_shared
    )
    slot_bufs = gl.allocate_shared_memory(gl.int32, [NSLOT, 1, BLOCK_K], plain)
    sc_tab = gl.allocate_shared_memory(gl.bfloat16, [8, 1, BLOCK_K], plain)
    kv_desc = tdm.make_tensor_descriptor(
        kv_ptr, [total_rows, NOPE], [640, 1], [BLOCK_K, NOPE], kv_shared
    )
    tail_desc = tdm.make_tensor_descriptor(
        kv_ptr + NOPE, [total_rows, TAIL], [640, 1], [BLOCK_K, TAIL], tail_shared
    )
    n = gl.maximum(kv_len, 1)
    bt_row = block_table_ptr
    if REQ_TOPK:  # kv_indices_ptr is the [rows, top-k] request-level index
        kv_start = t * topk_stride
        bt_row = block_table_ptr + gl.load(req_id_ptr + t) * bt_stride
    slot_desc = tdm.make_tensor_descriptor(
        kv_indices_ptr + kv_start, [1, n], [n, 1], [1, BLOCK_K], plain
    )

    q_smem = kv_bufs.index(NBUF - 1).reinterpret(
        shape=[BLOCK_H, NOPE],
        layout=gl.PaddedSharedLayout.with_identity_for(
            [[NOPE, 16]], [BLOCK_H, NOPE], [1, 0]
        ),
    )
    qt_smem = tail_bufs.index(NBUF - 1).reinterpret(
        shape=[BLOCK_H, TAIL],
        layout=gl.PaddedSharedLayout.with_identity_for(
            [[TAIL, 16]], [BLOCK_H, TAIL], [1, 0]
        ),
    )
    q_row = q_ptr + t * q_stride_t
    q_desc = tdm.make_tensor_descriptor(
        q_row, [H, NOPE], [q_stride_h, 1], [BLOCK_H, NOPE], q_smem.layout
    )
    qt_desc = tdm.make_tensor_descriptor(
        q_row + NOPE, [H, TAIL], [q_stride_h, 1], [BLOCK_H, TAIL], qt_smem.layout
    )
    tdm.async_load(q_desc, [h0, 0], q_smem)
    tdm.async_load(qt_desc, [h0, 0], qt_smem)
    for j in gl.static_range(NBUF + REQ_TOPK):
        tdm.async_load(slot_desc, [0, (tile_start + j) * BLOCK_K], slot_bufs.index(j))
    tdm.async_wait(0)
    LANE_MAP: gl.constexpr = BLOCK_H == 16 and KV_SPLITS == 1
    for j in gl.static_range(NBUF - 1):
        if REQ_TOPK:
            pend = _req_lookup(slot_bufs.index(j), bt_row, max_blocks, MAIN_BLOCK_SIZE)
            row = _req_rows(pend, max_blocks, blk_rows, MAIN_BLOCK_SIZE, SLOT_L)
        else:
            row = _slot_rows(
                slot_bufs.index(j), blk_rows, num_slots, SLOT_L, MAIN_BLOCK_SIZE,
                IDENT_ROWS, LANE_MAP,
            )  # fmt: skip
        _gather(j, row, num_iters, kv_desc, tail_desc, kv_bufs, tail_bufs)
    if REQ_TOPK:
        pend = _req_lookup(
            slot_bufs.index(NBUF - 1), bt_row, max_blocks, MAIN_BLOCK_SIZE
        )

    qs_smem = qt_smem.reinterpret(dtype=gl.uint8).slice(ROPE, 32, dim=1)
    rope_q = gl.arange(0, TAIL, layout=gl.SliceLayout(0, Q_L)) < ROPE
    q_r = gl.where(rope_q[None, :], qt_smem.load(Q_L), 0.0)
    q_r_sc = qs_smem.slice(16, 4, dim=1).load(QS_L)
    q = _v4_load_q_regs(q_smem, qs_smem, Q_L, QS_L, NOPE, 128)
    rope_k = gl.arange(0, TAIL, layout=gl.SliceLayout(1, K_L)) < ROPE
    # The Q reads must land before gather(NBUF - 1) overwrites their ring slot.
    gl.barrier()

    qk_scale = softmax_scale * LOG2E
    h_rows = h0 + gl.arange(0, BLOCK_H, layout=ROW_L)
    row_live = h_rows < H
    if KV_SPLITS == 1 and not HAS_LSE:
        sink = gl.amd.cdna4.buffer_load(
            attn_sink_ptr, h_rows.to(gl.int32), mask=row_live, other=float("-inf")
        )
        m_i = sink.to(gl.float32) * LOG2E
    else:
        m_i = gl.full([BLOCK_H], float("-inf"), gl.float32, layout=ROW_L)
    l_i = gl.full([BLOCK_H], 1.0, gl.float32, layout=ROW_L)
    z = gl.zeros([BLOCK_H, 64], gl.float32, layout=PV_L)
    acc = (z, z, z, z, z, z, z, z)
    k_offs = gl.arange(0, BLOCK_K, layout=COL_L)

    gl.assume(num_iters >= 1)
    for i in range(num_iters):
        # Everything but the newest gather is done: tile i and slots(i + NBUF - 1).
        tdm.async_wait(2)
        jn = i + NBUF - 1
        if REQ_TOPK:
            row = _req_rows(pend, max_blocks, blk_rows, MAIN_BLOCK_SIZE, SLOT_L)
            nxt = slot_bufs.index((jn + 1) % NSLOT)
            pend = _req_lookup(nxt, bt_row, max_blocks, MAIN_BLOCK_SIZE)
        else:
            row = _slot_rows(
                slot_bufs.index(jn % NSLOT), blk_rows, num_slots, SLOT_L,
                MAIN_BLOCK_SIZE, IDENT_ROWS, LANE_MAP,
            )  # fmt: skip
        jf = jn + 1 + REQ_TOPK
        tdm.async_load(
            slot_desc, [0, (tile_start + jf) * BLOCK_K], slot_bufs.index(jf % NSLOT)
        )
        _gather(jn, row, num_iters, kv_desc, tail_desc, kv_bufs, tail_bufs)

        kbuf = kv_bufs.index(i % NBUF)
        tbuf = tail_bufs.index(i % NBUF)
        sbuf = tbuf.reinterpret(dtype=gl.uint8).slice(ROPE, 32, dim=1)
        s = _v4_qk_mx_split_regs(
            q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], kbuf, sbuf, K_L, KS_L,
            QK_L, BLOCK_H, BLOCK_K, 128,
        )  # fmt: skip
        kr = gl.where(rope_k[:, None], tbuf.permute([1, 0]).load(K_L), 0.0)
        kr_sc = sbuf.slice(16, 4, dim=1).load(KS_L)
        s = gl.amd.gfx1250.wmma_scaled(q_r, q_r_sc, "e4m3", kr, kr_sc, "e4m3", s)
        s = qk_scale * s
        if HAS_INVALID:
            slot = slot_bufs.index(i % NSLOT).reshape([BLOCK_K]).load(SLOT_L)
            k = (tile_start + i) * BLOCK_K + gl.arange(0, BLOCK_K, layout=SLOT_L)
            if REQ_TOPK:  # the convert's -1: a negative position or past the table
                lim = max_blocks << (MAIN_BLOCK_SIZE.bit_length() - 1)
            else:
                lim = num_slots
            valid = (k < kv_len) & (slot >= 0) & (slot < lim)
            s += gl.where(gl.convert_layout(valid, COL_L), 0.0, float("-inf"))[None, :]
        elif (tile_start + i + 1) * BLOCK_K > kv_len:
            in_range = (tile_start + i) * BLOCK_K + k_offs < kv_len
            s += gl.where(in_range, 0.0, float("-inf"))[None, :]

        _pv_scale_table(tbuf, sc_tab, ks_shared, BLOCK_K)
        v0 = kbuf.slice(0, 64, dim=1).load(V_L)
        gl.amd.hint.sched_barrier()

        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        if gl.max(m_new - m_i, axis=0) > LAZY_TH:
            alpha = gl.exp2(m_i - gl.where(m_new == float("-inf"), 0.0, m_new))
            l_i = l_i * alpha
            alpha = gl.convert_layout(alpha[:, None], layout=PV_L)
            for J in gl.static_range(8):
                acc = acc[:J] + (acc[J] * alpha,) + acc[J + 1 :]
            m_i = m_new
        m_use = gl.where(m_i == float("-inf"), 0.0, m_i)
        p = gl.exp2(s - gl.where(row_live, m_use, float("inf"))[:, None])
        l_i = l_i + gl.sum(p, axis=1)
        p_bf = gl.convert_layout(p.to(gl.bfloat16), P_L)
        acc = _pv(acc, p_bf, kbuf, v0, sc_tab, P_L, V_L)

    tdm.async_wait(0)
    if KV_SPLITS == 1:
        l_div = gl.where(m_i == float("-inf"), 1.0, l_i) if HAS_LSE else l_i
        inv = gl.convert_layout(1.0 / l_div[:, None], layout=PV_L)
        oty: gl.constexpr = out_ptr.dtype.element_ty
        o_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[NOPE // 2, 8]], [BLOCK_H, NOPE // 2], [1, 0]
        )
        out_desc = tdm.make_tensor_descriptor(
            out_ptr + t * out_stride_t,
            [H, NOPE],
            [out_stride_h, out_stride_d],
            [BLOCK_H, NOPE // 2],
            o_shared,
        )
        # Two half-width pieces: the first one's TDM store overlaps the second.
        o0 = gl.allocate_shared_memory(oty, [BLOCK_H, NOPE // 2], o_shared)
        o1 = gl.allocate_shared_memory(oty, [BLOCK_H, NOPE // 2], o_shared)
        for J in gl.static_range(8):
            o = o0 if J < 4 else o1
            o.slice(64 * (J % 4), 64, dim=1).store((acc[J] * inv).to(oty))
            if J % 4 == 3:
                tdm.async_store(out_desc, [h0, (NOPE // 2) * (J // 4)], o)
        tdm.async_wait(0)
        if HAS_LSE:
            lse = gl.convert_layout((m_i + gl.log2(l_i)) * 0.6931471805599453, OROW_L)
            gl.amd.cdna4.buffer_store(lse, lse_ptr + t * H, h_o.to(gl.int32), h_o < H)
    else:
        gl.amd.cdna4.buffer_store(
            gl.convert_layout(m_i, OROW_L), m_ptr, ml_offs, h_o < H
        )
        gl.amd.cdna4.buffer_store(
            gl.convert_layout(l_i, OROW_L), l_ptr, ml_offs, h_o < H
        )
        ab = acc_partial_ptr + t * ap_stride_t + pid_k * ap_stride_k
        hp = h0 + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, PV_L))
        dp = gl.arange(0, 64, layout=gl.SliceLayout(0, PV_L))
        offs = (hp[:, None] * ap_stride_h + dp[None, :] * ap_stride_d).to(gl.int32)
        for J in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                acc[J], ab, offs + J * 64 * ap_stride_d, mask=(hp < H)[:, None]
            )
