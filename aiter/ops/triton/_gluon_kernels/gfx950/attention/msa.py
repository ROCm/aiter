# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax sparse attention (MSA), Gluon, gfx950.

msa_gfx950_gluon_kernel        k4v8 cache (MXFP4 K + fp8 V): BLOCK_M = 16 * BLOCK_Q
                               rows per program; BLOCK_Q == 1 walks a row's own
                               picks, BLOCK_Q > 1 a row group's page union
msa_gfx950_gluon_k4v4_kernel   k4v4 cache (K and V MXFP4): transposed form,
                               4 waves with independent softmax state

The cache layouts are documented in aiter/ops/triton/_triton_kernels/attention/msa.py.
K (and the k4v8 V) are stored in aiter's (16, 16) preshuffle, the MFMA operand
order, so a page loads from HBM straight into dot-operand registers.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

PAGE = gl.constexpr(128)
D = gl.constexpr(128)
HG = gl.constexpr(16)
K_BYTES = gl.constexpr(128 * 128 // 2)
KS_BYTES = gl.constexpr(128 * 128 // 32)
NUM_SCALES = gl.constexpr(128 // 32)


@gluon.jit
def _shuffled(offs_k, offs_n, K_EXTENT: gl.constexpr):
    """Byte of (k container, n) in a (16, 16)-preshuffled [N, K_EXTENT] tile."""
    return (
        (offs_k % 16)
        + (offs_k // 16) * 256
        + (offs_n % 16) * 16
        + (offs_n // 16) * (16 * K_EXTENT)
    )


@gluon.jit
def _page_at(uq, i, u1, last):
    """Page id of union slot i, 0 past the walk (a harmless in-bounds page)."""
    return gl.where(i < u1, gl.load(uq + gl.minimum(i, last)), 0)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _nth_valid(rank, blks, pages, j, nvalid):
    """(block, page) of the row's j-th valid pick; (-1, 0) once j >= nvalid."""
    sel = rank == j
    blk = gl.sum(gl.where(sel, blks, 0), axis=0)
    page = gl.sum(gl.where(sel, pages, 0), axis=0)
    live = j < nvalid
    return gl.where(live, blk, -1), gl.where(live, page, 0)


@gluon.jit
def _barrier():
    gl.barrier()


@gluon.jit
def _copy_k(k_smem, buf, u8_ptr, base, k_src):
    """Async-copy a page's K block into LDS buffer buf (one commit group)."""
    gl.amd.cdna4.async_copy.global_load_to_shared(k_smem.index(buf), u8_ptr + base + k_src)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _copy_v(v_smem, buf, f8_ptr, base, v_src):
    """Async-copy a page's V block into LDS buffer buf (one commit group)."""
    gl.amd.cdna4.async_copy.global_load_to_shared(v_smem.index(buf), f8_ptr + base + v_src)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _k_view(kb):
    """K page in HBM order [t / 16, (b / 16, t % 16, b % 16)] -> K^T [64 B, 128 keys]."""
    return (
        kb.reshape([PAGE // 16, D // 2 // 16, 16, 16])
        .permute((1, 3, 0, 2))
        .reshape([D // 2, PAGE])
    )


@gluon.jit
def _v_view(vb):
    """V page in HBM order [(t / 16, d / 64), ((d / 16) % 4, d % 16, t % 16)]
    -> V [128 keys, d]."""
    return (
        vb.reshape([PAGE // 16, D // 64, 4, 16, 16])
        .permute((0, 4, 1, 2, 3))
        .reshape([PAGE, D])
    )


_msa_gfx950_gluon_kernel_repr = make_kernel_repr(
    "msa_gfx950_gluon_kernel",
    ['BLOCK_M', 'NSEG', 'DECODE_QL', 'QT', 'KV_LDS', 'TOPK', 'NUM_HEADS', 'NUM_WARPS'],
)


@gluon.jit(repr=_msa_gfx950_gluon_kernel_repr)
def msa_gfx950_gluon_kernel(
    q8_ptr,  # e4m3 [rows, NUM_HEADS, QT, D] (unscaled MXFP8 terms)
    qs_ptr,  # u8 E8M0 [rows, NUM_HEADS, QT, D / 32]
    cache_u8_ptr,
    cache_f8_ptr,
    topk_ptr,  # BLOCK_Q == 1: [Hkv, rows, TOPK] int32 picks
    block_table_ptr,
    seq_lens_ptr,
    query_req_ptr,
    query_pos_ptr,
    uniq_ptr,  # BLOCK_Q > 1: the union pre-pass (msa_union_kernel)
    count_ptr,
    tokblk_ptr,
    pos_ptr,
    out_ptr,
    part_o_ptr,
    part_ml_ptr,
    seg_count_ptr,  # int32 [Hkv, groups]: the program's page count (segment 0 writes it)
    v_scale_ptr,
    qk_scale_log2,
    num_rows,
    seg_pages,  # pages a segment covers (split rebalance)
    stride_page,
    stride_head,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    stride_ot,
    stride_oh,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NSEG: gl.constexpr,
    DECODE_QL: gl.constexpr,
    QT: gl.constexpr,
    P_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    KV_LDS: gl.constexpr,
):
    """MSA over a k4v8 cache: BLOCK_M = 16 * BLOCK_Q rows (BLOCK_Q consecutive
    query rows x the 16 heads of one KV head) x one split.

      S = Q K^T   mfma_scaled 16x16x128, A = MXFP8 Q, B = MXFP4 K
      O += P V    mfma_scaled 16x16x128, A = fp8 P,   B = fp8 V

    Warps split the keys for QK and the head dim for PV, so each warp loads
    only its own K columns and V columns of a page.

    Split rebalance (as PR 7421's): segment sid covers pages
    [sid * seg_pages, (sid + 1) * seg_pages) of the program's compacted page
    list, so every live segment does the same work and a row (group) with more
    valid pages gets more segments; segments past its list exit with nothing to
    do. NSEG = 1 writes the output; otherwise segments write partials that
    msa_gfx950_gluon_reduce_split_rebalance_kernel merges, reading only the
    live ones (segment 0 records the page count).

    The two BLOCK_Q cases differ only in where the pages come from and how
    rows are masked:
      BLOCK_Q == 1  the row's own picks, read from topk and compacted (invalid
                    picks dropped); every page is unmasked except the
                    diagonal one (the block holding the row's position),
                    visited last.
      BLOCK_Q > 1   the union of the rows' picks (msa_union_kernel), so a page
                    several rows picked loads once; each row masks every page
                    to the blocks it picked, up to its position.
    K / V loading (page ids two steps ahead; per-lane global addresses, as page
    ids come out of vector reductions and a scalar buffer base would waterfall):
      KV_LDS = False  each lane loads its operand chunks straight into MFMA
                      registers (the preshuffle makes each 16-lane tile one
                      contiguous 256 B run), step j + 1's loads in flight
                      during step j's math
      KV_LDS = True   K and V of a page (each one contiguous run) are async-
                      copied into double-buffered LDS and unshuffled by the LDS
                      view (the PR 7421 structure, cdna4 async copy in place of
                      gfx1250's TDM). Each wave copies exactly the K rows / V
                      columns its MFMAs read, so wave-local waits suffice:
                        issue K(i+1), V(i+1); wait_group(3): K(i) landed;
                        QK(i); wait_group(2): V(i) landed; PV(i)
                      K scales (128 B a wave) stay register loads, one ahead.
    """
    BLOCK_Q: gl.constexpr = BLOCK_M // HG
    C: gl.constexpr = BLOCK_Q * TOPK
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NUM_WARPS])
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MFMA, 16)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MFMA, 16)
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [BLOCK_M, NUM_SCALES])
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B_OP, [PAGE, NUM_SCALES])
    ROW: gl.constexpr = gl.SliceLayout(1, MFMA)
    COL: gl.constexpr = gl.SliceLayout(0, MFMA)
    VEC: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])

    grp = gl.program_id(0)
    g = gl.program_id(1)
    sid = gl.program_id(2)
    r0 = grp * BLOCK_Q

    # Q of the BLOCK_M rows (row m: query row r0 + m // HG, head g * HG + m % HG).
    qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, A_OP))[:, None]
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, A_OP))[None, :]
    q_offs = ((r0 + qm // HG).to(gl.int64) * NUM_HEADS + g * HG + qm % HG) * (QT * D) + qd
    sm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_SCL))[:, None]
    ss = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL))[None, :]
    qs_offs = ((r0 + sm // HG).to(gl.int64) * NUM_HEADS + g * HG + sm % HG) * (
        QT * NUM_SCALES
    ) + ss
    if BLOCK_Q == 1:  # the program's one row always exists
        q_hi = gl.load(q8_ptr + q_offs)
        qs_hi = gl.load(qs_ptr + qs_offs)
        if QT == 2:
            q_lo = gl.load(q8_ptr + q_offs + D)
            qs_lo = gl.load(qs_ptr + qs_offs + NUM_SCALES)
    else:
        q_ok = (r0 + qm // HG) < num_rows
        s_ok = (r0 + sm // HG) < num_rows
        q_hi = gl.load(q8_ptr + q_offs, mask=q_ok, other=0.0)
        qs_hi = gl.load(qs_ptr + qs_offs, mask=s_ok, other=127)
        if QT == 2:
            q_lo = gl.load(q8_ptr + q_offs + D, mask=q_ok, other=0.0)
            qs_lo = gl.load(qs_ptr + qs_offs + NUM_SCALES, mask=s_ok, other=127)

    # register path: operand offsets inside one (page, KV head) record
    kb = gl.arange(0, D // 2, layout=gl.SliceLayout(1, B_OP))[:, None]
    kn = gl.arange(0, PAGE, layout=gl.SliceLayout(0, B_OP))[None, :]
    k_offs = _shuffled(kb, kn, D // 2)
    st = gl.arange(0, PAGE, layout=gl.SliceLayout(1, K_SCL))[:, None]
    sg = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, K_SCL))[None, :]
    ks_offs = K_BYTES + st * NUM_SCALES + sg
    vt = gl.arange(0, PAGE, layout=gl.SliceLayout(1, B_OP))[:, None]
    vd = gl.arange(0, D, layout=gl.SliceLayout(0, B_OP))[None, :]
    v_offs = K_BYTES + KS_BYTES + (vt // 16) * (D * 16) + vd * 16 + vt % 16
    # LDS path. The MFMA hands key tile i (QK's N) and head-dim tile i (PV's
    # N) to wave i % 4, so wave w reads K rows (16-key tiles) w and w + 4 and
    # V head-dim tiles w and w + 4: the copy layouts give wave w exactly those
    # bytes (K: [8 tiles, 1024 B] with the warps on the tile; V: [t / 16 and
    # d / 64, (d-tile % 4, 256 B)] with the warps on the 256 B d-tile), 16 B a lane.
    FLAT: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    K_CP: gl.constexpr = gl.BlockedLayout([1, 16], [1, 64], [NUM_WARPS, 1], [1, 0])
    # V: logical [rows, d-tile % 4, 256 B]; in LDS the d-tile is outermost, so
    # a wave's chunks are one contiguous run (a direct-to-LDS copy writes each
    # 64-lane instruction to a contiguous 1 KB)
    V_LDS: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 0, 1])
    V_CP: gl.constexpr = gl.BlockedLayout([1, 1, 16], [4, 1, 16], [1, NUM_WARPS, 1], [2, 0, 1])
    NBUF: gl.constexpr = 2 if KV_LDS else 1
    K_R: gl.constexpr = PAGE // 16 if KV_LDS else 1
    V_R: gl.constexpr = (PAGE // 16) * (D // 64) if KV_LDS else 1
    k_smem = gl.allocate_shared_memory(gl.uint8, [NBUF, K_R, (D // 2) * 16], FLAT)
    v_smem = gl.allocate_shared_memory(gl.float8e4nv, [NBUF, V_R, 4, 256], V_LDS)
    k_src = (
        gl.arange(0, K_R, layout=gl.SliceLayout(1, K_CP))[:, None] * ((D // 2) * 16)
        + gl.arange(0, (D // 2) * 16, layout=gl.SliceLayout(0, K_CP))[None, :]
    )
    v_src = K_BYTES + KS_BYTES + (
        gl.arange(0, V_R, layout=gl.SliceLayout(1, gl.SliceLayout(2, V_CP)))[:, None, None] * 1024
        + gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, V_CP)))[None, :, None] * 256
        + gl.arange(0, 256, layout=gl.SliceLayout(0, gl.SliceLayout(1, V_CP)))[None, None, :]
    )
    tok_n = gl.arange(0, PAGE, layout=COL)
    head_off = g * stride_head

    if BLOCK_Q == 1:
        # One row: its picks for this split, compacted; safe ones first, the
        # diagonal one (the block holding pos) last.
        if DECODE_QL > 0:
            pos = gl.load(seq_lens_ptr + grp // DECODE_QL) - DECODE_QL + grp % DECODE_QL
            req = (grp // DECODE_QL).to(gl.int64)
        else:
            pos = gl.load(query_pos_ptr + grp)
            req = gl.load(query_req_ptr + grp).to(gl.int64)
        jj = gl.arange(0, TOPK, layout=VEC)
        blks = gl.load(topk_ptr + g * stride_tk_h + grp.to(gl.int64) * stride_tk_t + jj)
        valid = (blks >= 0) & (blks * PAGE <= pos)
        pages = gl.load(
            block_table_ptr + req * stride_bt + gl.where(valid, blks, 0), mask=valid, other=0
        )
        diag = valid & (blks == pos // PAGE)
        safe = valid & (blks != pos // PAGE)
        n_safe = gl.sum(safe.to(gl.int32), axis=0)
        rank = gl.where(
            safe,
            gl.associative_scan(safe.to(gl.int32), 0, _add) - 1,
            gl.where(diag, n_safe, -1),
        )
        n_pages = n_safe + gl.sum(diag.to(gl.int32), axis=0)
        # this segment: compacted picks [j0, j_end)
        j0 = sid * seg_pages
        j_end = gl.minimum(n_pages, j0 + seg_pages)
        n_iter = gl.maximum(j_end - j0, 0)
        b_cur, p_cur = _nth_valid(rank, blks, pages, j0, j_end)
        b_nxt, p_nxt = _nth_valid(rank, blks, pages, j0 + 1, j_end)
    else:
        # A row group: this segment's pages [u0, u1) of the group's page union.
        ubase = (g * gl.num_programs(0) + grp).to(gl.int64)
        row_t = gl.arange(0, BLOCK_M, layout=ROW) // HG
        row_pos = gl.load(pos_ptr + r0 + row_t, mask=(r0 + row_t) < num_rows, other=-1)
        n_pages = gl.load(count_ptr + ubase)
        u0 = sid * seg_pages
        u1 = gl.minimum(n_pages, u0 + seg_pages)
        last = gl.maximum(u1 - 1, 0)
        uq = uniq_ptr + ubase * C
        tb = tokblk_ptr + ubase * C * BLOCK_Q + row_t
        n_iter = gl.maximum(u1 - u0, 0)
        p_cur = _page_at(uq, u0, u1, last)
        p_nxt = _page_at(uq, u0 + 1, u1, last)
        rb_c = gl.load(tb + gl.minimum(u0, last) * BLOCK_Q)

    base = p_cur.to(gl.int64) * stride_page + head_off
    if KV_LDS:
        _copy_k(k_smem, 0, cache_u8_ptr, base, k_src)
        _copy_v(v_smem, 0, cache_f8_ptr, base, v_src)
        ks_c = gl.load(cache_u8_ptr + base + ks_offs)
    else:
        k_c = gl.load(cache_u8_ptr + base + k_offs)
        ks_c = gl.load(cache_u8_ptr + base + ks_offs)
        v_c = gl.load(cache_f8_ptr + base + v_offs)
    buf = 0

    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, ROW)
    l_i = gl.zeros([BLOCK_M], gl.float32, ROW)
    acc = gl.zeros([BLOCK_M, D], gl.float32, MFMA)
    for j in range(n_iter):
        # Issue step j + 1's loads (and step j + 2's page id) before step j's math.
        if BLOCK_Q == 1:
            b_nxt2, p_nxt2 = _nth_valid(rank, blks, pages, j0 + j + 2, j_end)
        else:
            p_nxt2 = _page_at(uq, u0 + j + 2, u1, last)
            rb_n = gl.load(tb + gl.minimum(u0 + j + 1, last) * BLOCK_Q)
        nbase = p_nxt.to(gl.int64) * stride_page + head_off
        if KV_LDS:
            # Step j + 1's copies go first (into the region of the other buffer
            # this wave read at step j - 1, its reads long consumed), then wait
            # for step j's K only.
            _copy_k(k_smem, 1 - buf, cache_u8_ptr, nbase, k_src)
            _copy_v(v_smem, 1 - buf, cache_f8_ptr, nbase, v_src)
            ks_n = gl.load(cache_u8_ptr + nbase + ks_offs)
            gl.amd.cdna4.async_copy.wait_group(3)  # K(j) in; V(j), K / V(j + 1) in flight
            k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(_k_view(k_smem.index(buf)), B_OP)
        else:
            k_n = gl.load(cache_u8_ptr + nbase + k_offs)
            ks_n = gl.load(cache_u8_ptr + nbase + ks_offs)
            v_n = gl.load(cache_f8_ptr + nbase + v_offs)

        s = gl.zeros([BLOCK_M, PAGE], gl.float32, MFMA)
        s = gl.amd.cdna4.mfma_scaled(q_hi, qs_hi, "e4m3", k_c, ks_c, "e2m1", s)
        if QT == 2:
            s = gl.amd.cdna4.mfma_scaled(q_lo, qs_lo, "e4m3", k_c, ks_c, "e2m1", s)
        if BLOCK_Q == 1:
            if j0 + j == n_safe:  # the diagonal block: mask the keys after pos
                s = gl.where((b_cur * PAGE + tok_n <= pos)[None, :], s, float("-inf"))
            # every visited block has a visible key: the max is finite after step 0
            m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale_log2)
            m_use = m_new
        else:
            ok = (rb_c[:, None] >= 0) & (
                rb_c[:, None] * PAGE + tok_n[None, :] <= row_pos[:, None]
            )
            s = gl.where(ok, s, float("-inf"))
            # a row that did not pick this page stays at -inf: exponentiate against 0
            m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale_log2)
            m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_use)
        p8 = (gl.exp2(s * qk_scale_log2 - m_use[:, None]) * P_SCALE).to(gl.float8e4nv)
        l_i = l_i * alpha + gl.sum(p8.to(gl.float32), axis=1)
        acc = acc * alpha[:, None]
        if KV_LDS:
            gl.amd.cdna4.async_copy.wait_group(2)  # V(j) in; K / V(j + 1) in flight
            v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(_v_view(v_smem.index(buf)), B_OP)
        acc = gl.amd.cdna4.mfma_scaled(
            gl.convert_layout(p8, A_OP), None, "e4m3", v_c, None, "e4m3", acc
        )
        m_i = m_new

        if KV_LDS:
            buf = 1 - buf
            ks_c = ks_n
        else:
            k_c = k_n
            ks_c = ks_n
            v_c = v_n
        if BLOCK_Q == 1:
            b_cur = b_nxt
            b_nxt = b_nxt2
        else:
            rb_c = rb_n
        p_nxt = p_nxt2

    if KV_LDS:
        gl.amd.cdna4.async_copy.wait_group(0)  # the last (unused) prefetch
    om = gl.arange(0, BLOCK_M, layout=ROW)[:, None]
    od = gl.arange(0, D, layout=COL)[None, :]
    o_row = r0 + om // HG
    pair = (g * gl.num_programs(0) + grp).to(gl.int64)
    if NSEG == 1:
        v_scale = gl.load(v_scale_ptr)
        inv_l = gl.where(l_i > 0, v_scale / l_i, 0.0)
        o_ptrs = out_ptr + o_row.to(gl.int64) * stride_ot + (g * HG + om % HG) * stride_oh + od
        if BLOCK_Q == 1:
            gl.store(o_ptrs, (acc * inv_l[:, None]).to(gl.bfloat16))
        else:
            gl.store(o_ptrs, (acc * inv_l[:, None]).to(gl.bfloat16), mask=o_row < num_rows)
    else:
        if sid == 0:
            gl.store(seg_count_ptr + pair, n_pages)
        if n_iter > 0:  # a live segment: its partial (O, max, sum)
            part = pair * NSEG + sid
            gl.store(part_o_ptr + part * (BLOCK_M * D) + om * D + od, acc)
            mm = gl.arange(0, BLOCK_M, layout=ROW)
            gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2, m_i)
            gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2 + 1, l_i)




_msa_gfx950_gluon_reduce_split_rebalance_kernel_repr = make_kernel_repr(
    "msa_gfx950_gluon_reduce_split_rebalance_kernel",
    ['BLOCK_M', 'NSEG', 'NUM_WARPS'],
)


@gluon.jit(repr=_msa_gfx950_gluon_reduce_split_rebalance_kernel_repr)
def msa_gfx950_gluon_reduce_split_rebalance_kernel(
    part_o_ptr,  # fp32 [Hkv, groups, NSEG, BLOCK_M, D]
    part_ml_ptr,  # fp32 [Hkv, groups, NSEG, BLOCK_M, 2] (running max, sum)
    seg_count_ptr,  # int32 [Hkv, groups]: pages of each program's list
    out_ptr,
    v_scale_ptr,
    num_rows,
    seg_pages,
    stride_ot,
    stride_oh,
    BLOCK_M: gl.constexpr,
    NSEG: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """The split-rebalance reduce of msa_gfx950_gluon_kernel: one program per
    (row group, KV head) merges its live segments, cdiv(pages, seg_pages) of
    them, in segment order (deterministic) with the next segment's partial in
    flight; segments past a short list were never written and are not read.
    Row m of a group is query row grp * BLOCK_M / 16 + m // 16, head
    g * 16 + m % 16; a row group with no pages gives zeros."""
    L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NUM_WARPS, 1], [1, 0])
    R: gl.constexpr = gl.SliceLayout(1, L)
    grp = gl.program_id(0)
    g = gl.program_id(1)
    pair = (g * gl.num_programs(0) + grp).to(gl.int64)
    live = (gl.load(seg_count_ptr + pair) + seg_pages - 1) // seg_pages
    om = gl.arange(0, BLOCK_M, layout=R)
    od = gl.arange(0, D, layout=gl.SliceLayout(0, L))[None, :]
    o_offs = om[:, None] * D + od
    base0 = pair * NSEG
    m_n = gl.load(part_ml_ptr + base0 * (BLOCK_M * 2) + om * 2, mask=(om >= 0) & (live > 0), other=float("-inf"))
    l_n = gl.load(part_ml_ptr + base0 * (BLOCK_M * 2) + om * 2 + 1, mask=(om >= 0) & (live > 0), other=0.0)
    o_n = gl.load(part_o_ptr + base0 * (BLOCK_M * D) + o_offs, mask=(o_offs >= 0) & (live > 0), other=0.0)
    m_t = gl.full([BLOCK_M], float("-inf"), gl.float32, R)
    l_t = gl.zeros([BLOCK_M], gl.float32, R)
    acc = gl.zeros([BLOCK_M, D], gl.float32, L)
    for sgm in range(live):
        m_s = m_n
        l_s = l_n
        o_s = o_n
        nxt = base0 + gl.minimum(sgm + 1, NSEG - 1)
        ok = sgm + 1 < live
        m_n = gl.load(part_ml_ptr + nxt * (BLOCK_M * 2) + om * 2, mask=(om >= 0) & ok, other=float("-inf"))
        l_n = gl.load(part_ml_ptr + nxt * (BLOCK_M * 2) + om * 2 + 1, mask=(om >= 0) & ok, other=0.0)
        o_n = gl.load(part_o_ptr + nxt * (BLOCK_M * D) + o_offs, mask=(o_offs >= 0) & ok, other=0.0)
        m_new = gl.maximum(m_t, m_s)
        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        a = gl.exp2(m_t - m_use)
        b = gl.exp2(m_s - m_use)
        acc = acc * a[:, None] + o_s * b[:, None]
        l_t = l_t * a + l_s * b
        m_t = m_new
    v_scale = gl.load(v_scale_ptr)
    inv_l = gl.where(l_t > 0, v_scale / l_t, 0.0)
    row = grp * (BLOCK_M // HG) + om // HG
    gl.store(
        out_ptr + row.to(gl.int64)[:, None] * stride_ot + (g * HG + om % HG)[:, None] * stride_oh + od,
        (acc * inv_l[:, None]).to(gl.bfloat16),
        mask=(row < num_rows)[:, None],
    )


V4_BYTES = gl.constexpr(128 * 128 // 2)


_msa_gfx950_gluon_k4v4_kernel_repr = make_kernel_repr(
    "msa_gfx950_gluon_k4v4_kernel",
    ['SPLITS', 'DECODE_QL', 'TOPK', 'NUM_HEADS'],
)


@gluon.jit(repr=_msa_gfx950_gluon_k4v4_kernel_repr)
def msa_gfx950_gluon_k4v4_kernel(
    q8_ptr,  # e4m3 [rows, NUM_HEADS, D] (one MXFP8 term, unscaled)
    qs_ptr,  # u8 E8M0 [rows, NUM_HEADS, D / 32]
    cache_u8_ptr,
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    query_req_ptr,
    query_pos_ptr,
    out_ptr,
    part_o_ptr,
    part_ml_ptr,
    qk_scale_log2,
    stride_page,
    stride_head,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    stride_ot,
    stride_oh,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    SPLITS: gl.constexpr,
    DECODE_QL: gl.constexpr,
):
    """MSA, one query row x the 16 heads of one KV head x one split, 4 waves
    with independent softmax state (the native NVFP4 kernel's structure):

      S^T = K Q^T          mfma_scaled 16x16x128, keys on M, warps along keys:
                           wave w computes its own 32 keys (64 r + 16 w + ...)
      O^T_w += V_w^T P_w^T batched mfma 16x16x32 over the wave dim, P^T taken
                           from the accumulators as they are (asserted)

    Each wave keeps its own running max / sum / O^T over its keys of every
    page; the waves combine once, after the walk. Nothing crosses waves per
    page. k4v4: K and V MXFP4; V converts with a unit scale (the hardware
    scaled convert takes one scale per byte, and a byte holds two keys with
    their own E8M0 scales), times its key's scale; P is bf16.
    """
    NW: gl.constexpr = 4
    QK: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [NW, 1])
    PV3: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [NW, 1, 1])
    KA: gl.constexpr = gl.DotOperandLayout(0, QK, 16)
    QB: gl.constexpr = gl.DotOperandLayout(1, QK, 16)
    VA3: gl.constexpr = gl.DotOperandLayout(0, PV3, 8)
    VSRC3: gl.constexpr = gl.DotOperandLayout(0, PV3, 4)
    PB3: gl.constexpr = gl.DotOperandLayout(1, PV3, 8)
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(KA, [PAGE, NUM_SCALES])
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(QB, [HG, NUM_SCALES])
    S3: gl.constexpr = (
        gl.zeros([PAGE, HG], gl.float32, QK)
        .reshape([2, NW, 4, 4, HG]).permute((1, 2, 0, 3, 4)).reshape([NW, 32, HG]).type.layout
    )
    WH: gl.constexpr = gl.SliceLayout(1, S3)  # [wave, head]
    OWH: gl.constexpr = gl.SliceLayout(1, PV3)
    V4: gl.constexpr = gl.zeros([NW, D, 32], gl.bfloat16, VA3).reshape([NW, 4, 32, 32]).type.layout
    SC: gl.constexpr = gl.SliceLayout(2, V4)  # [wave, scale group, kappa]
    VEC: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    PER: gl.constexpr = TOPK // SPLITS

    row = gl.program_id(0)
    g = gl.program_id(1)
    sid = gl.program_id(2)
    if DECODE_QL > 0:
        req = row // DECODE_QL
        pos = gl.load(seq_lens_ptr + req) - DECODE_QL + row % DECODE_QL
    else:
        req = gl.load(query_req_ptr + row)
        pos = gl.load(query_pos_ptr + row)
    req = req.to(gl.int64)

    head0 = row.to(gl.int64) * NUM_HEADS + g * HG
    qd = gl.arange(0, D, layout=gl.SliceLayout(1, QB))[:, None]
    qn = gl.arange(0, HG, layout=gl.SliceLayout(0, QB))[None, :]
    q_t = gl.load(q8_ptr + (head0 + qn) * D + qd)
    sn = gl.arange(0, HG, layout=gl.SliceLayout(1, Q_SCL))[:, None]
    sg = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL))[None, :]
    q_sc = gl.load(qs_ptr + (head0 + sn) * NUM_SCALES + sg)

    kn = gl.arange(0, PAGE, layout=gl.SliceLayout(1, KA))[:, None]
    kb = gl.arange(0, D // 2, layout=gl.SliceLayout(0, KA))[None, :]
    k_offs = _shuffled(kb, kn, D // 2)
    kst = gl.arange(0, PAGE, layout=gl.SliceLayout(1, K_SCL))[:, None]
    ksg = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, K_SCL))[None, :]
    ks_offs = K_BYTES + kst * NUM_SCALES + ksg
    V_OFF: gl.constexpr = K_BYTES + KS_BYTES
    vw = gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(2, VSRC3)))[:, None, None]
    vdd = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, VSRC3)))[None, :, None]
    vc = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, VSRC3)))[None, None, :]
    v_offs = V_OFF + (vw * D + vdd) * 16 + vc
    cw = gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(2, SC)))[:, None, None]
    cg = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, SC)))[None, :, None]
    ck = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, SC)))[None, None, :]
    vs_offs = V_OFF + V4_BYTES + (cw * 4 + cg) * 32 + ck
    unit_sc = gl.full([NW, D, 32], 127, gl.uint8, VA3)
    tok = gl.arange(0, PAGE, layout=gl.SliceLayout(1, QK))
    head_off = g * stride_head

    jj = gl.arange(0, PER, layout=VEC)
    blks = gl.load(topk_ptr + g * stride_tk_h + row.to(gl.int64) * stride_tk_t + sid * PER + jj)
    valid = (blks >= 0) & (blks * PAGE <= pos)
    pages = gl.load(block_table_ptr + req * stride_bt + gl.where(valid, blks, 0), mask=valid, other=0)
    diag = valid & (blks == pos // PAGE)
    safe = valid & (blks != pos // PAGE)
    n_safe = gl.sum(safe.to(gl.int32), axis=0)
    rank = gl.where(
        safe,
        gl.associative_scan(safe.to(gl.int32), 0, _add) - 1,
        gl.where(diag, n_safe, -1),
    )
    nvalid = n_safe + gl.sum(diag.to(gl.int32), axis=0)
    b_cur, p_cur = _nth_valid(rank, blks, pages, 0, nvalid)
    b_nxt, p_nxt = _nth_valid(rank, blks, pages, 1, nvalid)
    base = p_cur.to(gl.int64) * stride_page + head_off
    k_c = gl.load(cache_u8_ptr + base + k_offs)
    ks_c = gl.load(cache_u8_ptr + base + ks_offs)
    v_c = gl.load(cache_u8_ptr + base + v_offs)
    vs_c = gl.load(cache_u8_ptr + base + vs_offs)

    m3 = gl.full([NW, HG], float("-inf"), gl.float32, WH)
    l3 = gl.zeros([NW, HG], gl.float32, WH)
    o3 = gl.zeros([NW, D, HG], gl.float32, PV3)
    for j in range(nvalid):
        b_nxt2, p_nxt2 = _nth_valid(rank, blks, pages, j + 2, nvalid)
        nbase = p_nxt.to(gl.int64) * stride_page + head_off
        k_n = gl.load(cache_u8_ptr + nbase + k_offs)
        ks_n = gl.load(cache_u8_ptr + nbase + ks_offs)
        v_n = gl.load(cache_u8_ptr + nbase + v_offs)
        vs_n = gl.load(cache_u8_ptr + nbase + vs_offs)

        s = gl.zeros([PAGE, HG], gl.float32, QK)
        s = gl.amd.cdna4.mfma_scaled(k_c, ks_c, "e2m1", q_t, q_sc, "e4m3", s)
        if j == n_safe:  # the diagonal block: mask keys after pos
            s = gl.where((b_cur * PAGE + tok <= pos)[:, None], s, float("-inf"))
        # wave w's 32 keys, in the kappa order its PV consumes
        s3 = s.reshape([2, NW, 4, 4, HG]).permute((1, 2, 0, 3, 4)).reshape([NW, 32, HG])
        m_new = gl.maximum(m3, gl.max(s3, axis=1) * qk_scale_log2)
        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m3 - m_use)
        p3 = gl.exp2(s3 * qk_scale_log2 - m_use[:, None, :])
        p3 = p3.to(gl.bfloat16)
        l3 = l3 * alpha + gl.sum(p3.to(gl.float32), axis=1)
        v3 = gl.amd.cdna4.scaled_upcast(v_c, unit_sc, gl.bfloat16, axis=2)
        v3 = gl.convert_layout(v3, VA3, assert_trivial=True)
        sc = (vs_c.to(gl.int16) << 7).to(gl.bfloat16, bitcast=True)  # 2^(s - 127)
        v3 = (v3.reshape([NW, 4, 32, 32]) * gl.expand_dims(sc, 2)).reshape([NW, D, 32])
        v3 = gl.convert_layout(v3, VA3, assert_trivial=True)
        p3 = gl.convert_layout(p3, PB3, assert_trivial=True)
        o3 = o3 * gl.convert_layout(alpha, OWH)[:, None, :]
        o3 = gl.amd.cdna4.mfma(v3, p3, o3)
        m3 = m_new

        k_c = k_n
        ks_c = ks_n
        v_c = v_n
        vs_c = vs_n
        b_cur = b_nxt
        b_nxt = b_nxt2
        p_nxt = p_nxt2

    # Combine the waves once: rescale each to the common max and sum.
    m_all = gl.max(m3, axis=0)
    m_all_u = gl.where(m_all == float("-inf"), 0.0, m_all)
    wgt = gl.exp2(m3 - m_all_u[None, :])
    l_all = gl.sum(wgt * l3, axis=0)
    o = gl.sum(o3 * gl.convert_layout(wgt, OWH)[:, None, :], axis=0)  # [D, HG]
    OL: gl.constexpr = gl.SliceLayout(0, PV3)
    od = gl.arange(0, D, layout=gl.SliceLayout(1, OL))[:, None]
    on = gl.arange(0, HG, layout=gl.SliceLayout(0, OL))[None, :]
    l_o = gl.convert_layout(l_all, gl.SliceLayout(0, OL))
    if SPLITS == 1:
        inv_l = gl.where(l_o > 0, 1.0 / l_o, 0.0)
        gl.store(
            out_ptr + row.to(gl.int64) * stride_ot + (g * HG + on) * stride_oh + od,
            (o * inv_l[None, :]).to(gl.bfloat16),
        )
    else:
        part = (g * gl.num_programs(0) + row).to(gl.int64) * SPLITS + sid
        gl.store(part_o_ptr + part * (HG * D) + on * D + od, o)
        hh = gl.arange(0, HG, layout=gl.SliceLayout(0, OL))
        gl.store(part_ml_ptr + part * (HG * 2) + hh * 2, gl.convert_layout(m_all, gl.SliceLayout(0, OL)))
        gl.store(part_ml_ptr + part * (HG * 2) + hh * 2 + 1, l_o)
