# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax sparse attention (MSA), Gluon, gfx950.

msa_gfx950_gluon_kernel        k4v8 cache (MXFP4 K + fp8 V): BLOCK_M = 16 * BLOCK_Q
                               rows per program; BLOCK_Q == 1 walks a row's own
                               picks, BLOCK_Q > 1 a row group's page sort_topk
msa_gfx950_gluon_k4v4pt_kernel k4v4pt cache (MXFP4 K + e2m1 V with an fp32 scale
                               per token), decode: one row per program; needs a
                               Triton with gl.amd.cdna4.load_shared_fp4_repacked

The cache layouts are documented in aiter/ops/triton/_triton_kernels/attention/msa.py.
K (and the k4v8 V) are stored in aiter's (16, 16) preshuffle, the MFMA operand
order, so a page loads from HBM straight into dot-operand registers.
"""

from triton.experimental import gluon
from triton.runtime.jit import constexpr_function
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@constexpr_function
def triton_next_pow2(n):
    return 1 << (int(n) - 1).bit_length()

PAGE = gl.constexpr(128)
D = gl.constexpr(128)
HG = gl.constexpr(16)
K_BYTES = gl.constexpr(128 * 128 // 2)
KS_BYTES = gl.constexpr(128 * 128 // 32)
RECORD_BYTES = gl.constexpr(128 * 128 // 2 + 128 * 128 // 32 + 128 * 128)  # K + K scales + V
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
    """Page id of sort_topk slot i, 0 past the walk (a harmless in-bounds page)."""
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
def _copy_ks(ks_smem, buf, u8_ptr, base, ks_src):
    """Async-copy a page's K scales into LDS buffer buf (no commit: it joins the
    K group that follows): 1 KB, the 512 B of scales then 512 B of V, so the 4
    waves' 64 lanes x 4 B cover it exactly (a replicated lane does not lower)."""
    gl.amd.cdna4.async_copy.global_load_to_shared(ks_smem.index(buf), u8_ptr + base + ks_src)


@gluon.jit
def _ks_view(kb):
    """The [128 keys, scales] K scales at the head of a 1 KB K-scale buffer."""
    return kb.reshape([2 * PAGE, NUM_SCALES]).slice(0, PAGE)


@gluon.jit
def _ks_lds(ks_smem, buf, K_SCL: gl.constexpr):
    gl.barrier()  # every wave's part of the K-scale copy (each waited for its own)
    return gl.amd.cdna4.async_copy.load_shared_relaxed(_ks_view(ks_smem.index(buf)), K_SCL)


@gluon.jit
def _ws_row(q8_ptr, qs_ptr, pos_ptr, r, num_rows, g, qm16, qd16, sm16, ss16, NUM_HEADS: gl.constexpr):
    """Per-wave key split: query row r's 16-head Q (MXFP8 A operand), scales and position."""
    ok = r < num_rows
    q = gl.load(q8_ptr + ((r.to(gl.int64) * NUM_HEADS + qm16) * D + qd16) + g * (HG * D),
                mask=ok & (qm16 >= 0), other=0.0)
    qs = gl.load(qs_ptr + ((r.to(gl.int64) * NUM_HEADS + sm16) * NUM_SCALES + ss16) + g * (HG * NUM_SCALES),
                 mask=ok & (sm16 >= 0), other=127)
    pos = gl.load(pos_ptr + r, mask=ok, other=-1)
    return q, qs, pos


@gluon.jit
def _ws_qk_softmax(q, qs, k_c, ks_c, rb, pos, tok_n, m3, l3, acc3, qk_scale_log2,
                   P_SCALE: gl.constexpr, MFMA: gl.constexpr, MF3: gl.constexpr,
                   NUM_WARPS: gl.constexpr):  # fmt: skip
    """Per-wave key split, one query row (16 heads) on a page it picked: QK (2-D, wave
    w computes keys r * 64 + w * 16 + c), S viewed as [wave, head, its 32 keys] (the
    same registers), each wave's online softmax. Returns (fp8 P, m, l, rescaled O)."""
    s = gl.zeros([HG, PAGE], gl.float32, MFMA)
    s = gl.amd.cdna4.mfma_scaled(q, qs, "e4m3", k_c, ks_c, "e2m1", s)
    s = s * qk_scale_log2
    if rb == pos // PAGE:  # only the row's own block has keys past its position
        s = gl.where((rb * PAGE + tok_n <= pos)[None, :], s, float("-inf"))
    s3 = gl.convert_layout(
        s.reshape([HG, 2, NUM_WARPS, 16]).permute((2, 0, 1, 3)).reshape([NUM_WARPS, HG, 32]), MF3)
    m_new = gl.maximum(m3, gl.max(s3, axis=2))
    # a wave's 32 keys can all lie past the row's position: exponentiate against 0
    m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
    al = gl.exp2(m3 - m_use)
    p3 = (gl.exp2(s3 - m_use[:, :, None]) * P_SCALE).to(gl.float8e4nv)
    l3 = l3 * al + gl.sum(p3.to(gl.float32), axis=2)
    # (O is rescaled every page: skipping it when no max grew needs a CTA-wide test,
    # a cross-wave reduction per row and page, +24..29 us on the AgentX dumps)
    return p3, m_new, l3, acc3 * al[:, :, None]


@gluon.jit
def _ws_pv(p3, v3, acc3, A3: gl.constexpr):
    """PV of one row: rank-3 fp8 MFMA over each wave's 32 keys."""
    return gl.amd.cdna4.mfma(gl.convert_layout(p3, A3), v3, acc3)


@gluon.jit
def _ws_v_lds(vw_smem, buf, B3: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Each wave's V [32 keys, D] of page buf from LDS, read just before a row's PV
    (not held across rows); waits for the page's V group (this wave's blocks)."""
    gl.amd.cdna4.async_copy.wait_group(2)
    return gl.amd.cdna4.async_copy.load_shared_relaxed(
        vw_smem.index(buf).permute((1, 0, 3, 2)).reshape([NUM_WARPS, 32, D]), B3)


@gluon.jit
def _ws_merge_store(mrg, t: gl.constexpr, m3, l3, acc3):
    """One row's waves (m, l, O) -> its normalized per-wave partials (fp16, bounded
    by |V|) in rows [16 t, 16 t + 16) of the merge buffer [wave, row, d]."""
    m_w = gl.max(m3, axis=0)
    m_wu = gl.where(m_w == float("-inf"), 0.0, m_w)
    wt = gl.exp2(m3 - m_wu[None, :])
    l_w = gl.sum(wt * l3, axis=0)
    sc = wt / gl.where(l_w > 0, l_w, 1.0)[None, :]
    mrg.slice(t * HG, HG, dim=1).store((acc3 * sc[:, :, None]).to(gl.float16))


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
    ['BLOCK_M', 'NSEG', 'ATOMIC', 'OUT_MX', 'DECODE_QL', 'QT', 'KV_LDS', 'WSPLIT', 'TOPK', 'NUM_HEADS', 'NUM_WARPS'],
)


@gluon.jit(repr=_msa_gfx950_gluon_kernel_repr)
def msa_gfx950_gluon_kernel(
    q8_ptr,  # e4m3 [rows, NUM_HEADS, QT, D] (unscaled MXFP8 terms)
    qs_ptr,  # u8 E8M0 [rows, NUM_HEADS, QT, D / 32]
    cache_u8_ptr,  # k4v8 records [pages, Hkv, RECORD_BYTES] (fp8 V: a bitcast view)
    topk_ptr,  # BLOCK_Q == 1: [Hkv, rows, TOPK] int32 picks (row stride TOPK)
    block_table_ptr,
    meta0_ptr,  # decode: seq_lens; prefill: query_req
    meta1_ptr,  # prefill: query_pos (decode: unused)
    st_ptr,  # BLOCK_Q > 1: the sort_topk, one int32 buffer [uniq | count | tokblk | pos]
    out_ptr,
    ws_ptr,  # NSEG > 1: fp32 [part_o | part_ml | seg_count (int32)]
    counter_ptr,  # int32 [>= Hkv * groups], zeroed: segment arrivals (ATOMIC)
    mxq_ptr,  # OUT_MX: e4m3 [rows, NUM_HEADS * D], o_proj's MXFP8 input
    mxs_ptr,  # OUT_MX: E8M0 [rows, NUM_HEADS * D / 32]
    v_scale_ptr,
    qk_scale_log2,
    num_rows,
    seg_pages,  # pages a segment covers (split rebalance)
    stride_page,
    stride_tk_h,
    stride_bt,
    stride_ot,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NSEG: gl.constexpr,
    ATOMIC: gl.constexpr,
    OUT_MX: gl.constexpr,
    DECODE_QL: gl.constexpr,
    QT: gl.constexpr,
    P_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    KV_LDS: gl.constexpr,
    WSPLIT: gl.constexpr = False,
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
      BLOCK_Q > 1   the sort_topk of the rows' picks (msa_sort_topk_kernel), so a page
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
    # Views and strides the launch no longer passes (fixed by the layouts the
    # API validates): the fp8 V view of the cache, the row metadata slots, the
    # sort_topk and split-workspace regions, the dense strides.
    cache_f8_ptr = cache_u8_ptr.to(gl.pointer_type(gl.float8e4nv), bitcast=True)
    seq_lens_ptr = meta0_ptr
    query_req_ptr = meta0_ptr
    query_pos_ptr = meta1_ptr
    stride_head: gl.constexpr = RECORD_BYTES
    stride_tk_t: gl.constexpr = TOPK
    stride_oh: gl.constexpr = D
    stride_mq: gl.constexpr = NUM_HEADS * D
    stride_ms: gl.constexpr = NUM_HEADS * NUM_SCALES
    n_prog = gl.num_programs(0).to(gl.int64) * gl.num_programs(1)
    uniq_ptr = st_ptr
    count_ptr = st_ptr + n_prog * C
    tokblk_ptr = count_ptr + n_prog
    pos_ptr = tokblk_ptr + n_prog * (C * BLOCK_Q)
    n_part = n_prog * NSEG
    part_o_ptr = ws_ptr
    part_ml_ptr = ws_ptr + n_part * (BLOCK_M * D)
    seg_count_ptr = (part_ml_ptr + n_part * (BLOCK_M * 2)).to(gl.pointer_type(gl.int32), bitcast=True)
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NUM_WARPS])
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MFMA, 16)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MFMA, 16)
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [BLOCK_M, NUM_SCALES])
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B_OP, [PAGE, NUM_SCALES])
    ROW: gl.constexpr = gl.SliceLayout(1, MFMA)
    COL: gl.constexpr = gl.SliceLayout(0, MFMA)
    VEC: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])

    # Launch the last row groups first (as gfx1250 unified_attention_2d): under
    # causal masking the late rows pick the most pages, so they should not be
    # the ones left running at the tail.
    grp = gl.num_programs(0) - 1 - gl.program_id(0)
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
    V_R: gl.constexpr = (PAGE // 16) * (D // 64) if (KV_LDS and not WSPLIT) else 1
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
    # K scales (WSPLIT with KV_LDS: async-copied with K): per buffer [2, 512] u8,
    # row 0 the page's scales (row 1: V bytes, copied only for exact lane coverage,
    # as a replicated lane does not lower). Not wave-local: a barrier after the K
    # wait makes every wave's part visible. Elsewhere: register loads one page ahead.
    KS_CP: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [2, NUM_WARPS // 2], [1, 0])
    ks_smem = gl.allocate_shared_memory(gl.uint8, [2 if (KV_LDS and WSPLIT) else 1, 2, 512], FLAT)
    ks_src = K_BYTES + (
        gl.arange(0, 2, layout=gl.SliceLayout(1, KS_CP))[:, None] * 512
        + gl.arange(0, 512, layout=gl.SliceLayout(0, KS_CP))[None, :]
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
        # A row group: this segment's pages [u0, u1) of the group's page sort_topk.
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

    if WSPLIT:
        # Per-wave key split (BLOCK_Q = 2 / 4 prefill, NSEG = 1). Each query row
        # (16 heads) has its own per-wave state (m, l, O) over the wave's 32 keys
        # of a page and runs QK / softmax / PV only on the pages it picked (a
        # CTA-uniform branch; the loop has no cross-wave reduction, so no barrier
        # inside one). K, K scales and V are loaded once a page for all rows.
        # KV_LDS: K as the 1-tile path (async copy, wave-local); V async-copied
        # wave-local too: wave w copies the two 2 KB key blocks it reads (keys
        # [16 w, 16 w + 16) and [64 + 16 w, ...)); the K scales async-copied with
        # K (not wave-local: a barrier before their read, see ks_smem). The rows'
        # waves merge once, after the loop.
        gl.static_assert((BLOCK_Q == 2 or BLOCK_Q == 4) and QT == 1 and NUM_WARPS == 4 and NSEG == 1)
        MF3: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [NUM_WARPS, 1, 1])
        A3: gl.constexpr = gl.DotOperandLayout(0, MF3, 8)
        B3: gl.constexpr = gl.DotOperandLayout(1, MF3, 8)
        ST3: gl.constexpr = gl.SliceLayout(2, MF3)
        # V_w [32 keys, D] of wave w: key = (k // 16) * 64 + w * 16 + k % 16
        w3v = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, B3)))[:, None, None]
        k3v = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, B3)))[None, :, None]
        d3v = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(1, B3)))[None, None, :]
        v3_offs = K_BYTES + KS_BYTES + ((k3v // 16) * NUM_WARPS + w3v) * (D * 16) + d3v * 16 + k3v % 16
        base = p_cur.to(gl.int64) * stride_page + head_off
        if KV_LDS:
            # [buf, r, w, d, key % 16]: key block r * 4 + w (2 KB, HBM order d, key % 16)
            VW_CP: gl.constexpr = gl.BlockedLayout([1, 1, 1, 16], [1, 1, 64, 1], [1, NUM_WARPS, 1, 1], [3, 2, 1, 0])
            VW_SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [3, 2, 1, 0])
            vw_smem = gl.allocate_shared_memory(gl.float8e4nv, [2, 2, NUM_WARPS, D, 16], VW_SH)
            vw_r = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, VW_CP))))
            vw_w = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, VW_CP))))
            vw_d = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, VW_CP))))
            vw_c = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, VW_CP))))
            vw_src = K_BYTES + KS_BYTES + (
                (vw_r[:, None, None, None] * NUM_WARPS + vw_w[None, :, None, None]) * (D * 16)
                + vw_d[None, None, :, None] * 16 + vw_c[None, None, None, :]
            )
            _copy_ks(ks_smem, 0, cache_u8_ptr, base, ks_src)
            _copy_k(k_smem, 0, cache_u8_ptr, base, k_src)
            _copy_v(vw_smem, 0, cache_f8_ptr, base, vw_src)
        else:
            k_c = gl.load(cache_u8_ptr + base + k_offs)
            ks_c = gl.load(cache_u8_ptr + base + ks_offs)
            v3_c = gl.load(cache_f8_ptr + base + v3_offs)
        # the rows: Q, position, row block of the current page
        Q_SCL16: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [HG, NUM_SCALES])
        qm16 = gl.arange(0, HG, layout=gl.SliceLayout(1, A_OP))[:, None]
        qd16 = gl.arange(0, D, layout=gl.SliceLayout(0, A_OP))[None, :]
        sm16 = gl.arange(0, HG, layout=gl.SliceLayout(1, Q_SCL16))[:, None]
        ss16 = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL16))[None, :]
        tbs = tokblk_ptr + ubase * (C * BLOCK_Q)
        slot = gl.minimum(u0, last) * BLOCK_Q
        q_t0, qs_t0, pos_t0 = _ws_row(q8_ptr, qs_ptr, pos_ptr, r0, num_rows, g, qm16, qd16, sm16, ss16, NUM_HEADS)
        q_t1, qs_t1, pos_t1 = _ws_row(q8_ptr, qs_ptr, pos_ptr, r0 + 1, num_rows, g, qm16, qd16, sm16, ss16, NUM_HEADS)
        rb_t0 = gl.load(tbs + slot)
        rb_t1 = gl.load(tbs + slot + 1)
        m_t0 = gl.full([NUM_WARPS, HG], float("-inf"), gl.float32, ST3)
        l_t0 = gl.zeros([NUM_WARPS, HG], gl.float32, ST3)
        acc_t0 = gl.zeros([NUM_WARPS, HG, D], gl.float32, MF3)
        m_t1 = gl.full([NUM_WARPS, HG], float("-inf"), gl.float32, ST3)
        l_t1 = gl.zeros([NUM_WARPS, HG], gl.float32, ST3)
        acc_t1 = gl.zeros([NUM_WARPS, HG, D], gl.float32, MF3)
        if BLOCK_Q == 4:
            q_t2, qs_t2, pos_t2 = _ws_row(q8_ptr, qs_ptr, pos_ptr, r0 + 2, num_rows, g, qm16, qd16, sm16, ss16, NUM_HEADS)
            q_t3, qs_t3, pos_t3 = _ws_row(q8_ptr, qs_ptr, pos_ptr, r0 + 3, num_rows, g, qm16, qd16, sm16, ss16, NUM_HEADS)
            rb_t2 = gl.load(tbs + slot + 2)
            rb_t3 = gl.load(tbs + slot + 3)
            m_t2 = gl.full([NUM_WARPS, HG], float("-inf"), gl.float32, ST3)
            l_t2 = gl.zeros([NUM_WARPS, HG], gl.float32, ST3)
            acc_t2 = gl.zeros([NUM_WARPS, HG, D], gl.float32, MF3)
            m_t3 = gl.full([NUM_WARPS, HG], float("-inf"), gl.float32, ST3)
            l_t3 = gl.zeros([NUM_WARPS, HG], gl.float32, ST3)
            acc_t3 = gl.zeros([NUM_WARPS, HG, D], gl.float32, MF3)
        buf = 0
        for j in range(n_iter):
            p_nxt2 = _page_at(uq, u0 + j + 2, u1, last)
            nslot = gl.minimum(u0 + j + 1, last) * BLOCK_Q
            rb_n0 = gl.load(tbs + nslot)
            rb_n1 = gl.load(tbs + nslot + 1)
            if BLOCK_Q == 4:
                rb_n2 = gl.load(tbs + nslot + 2)
                rb_n3 = gl.load(tbs + nslot + 3)
            nbase = p_nxt.to(gl.int64) * stride_page + head_off
            if KV_LDS:
                _copy_ks(ks_smem, 1 - buf, cache_u8_ptr, nbase, ks_src)
                _copy_k(k_smem, 1 - buf, cache_u8_ptr, nbase, k_src)
                _copy_v(vw_smem, 1 - buf, cache_f8_ptr, nbase, vw_src)
                # page j's K in; its V (read per row, just before that row's PV) and
                # page j + 1's K / V in flight
                gl.amd.cdna4.async_copy.wait_group(3)
                ks_c = _ks_lds(ks_smem, buf, K_SCL)
                k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(_k_view(k_smem.index(buf)), B_OP)
            else:
                k_n = gl.load(cache_u8_ptr + nbase + k_offs)
                ks_n = gl.load(cache_u8_ptr + nbase + ks_offs)
                v3_n = gl.load(cache_f8_ptr + nbase + v3_offs)
            if rb_t0 >= 0:
                p3, m_t0, l_t0, acc_t0 = _ws_qk_softmax(q_t0, qs_t0, k_c, ks_c, rb_t0, pos_t0, tok_n, m_t0,
                                                            l_t0, acc_t0, qk_scale_log2, P_SCALE, MFMA, MF3, NUM_WARPS)
                if KV_LDS:
                    v3_c = _ws_v_lds(vw_smem, buf, B3, NUM_WARPS)
                acc_t0 = _ws_pv(p3, v3_c, acc_t0, A3)
            if rb_t1 >= 0:
                p3, m_t1, l_t1, acc_t1 = _ws_qk_softmax(q_t1, qs_t1, k_c, ks_c, rb_t1, pos_t1, tok_n, m_t1,
                                                            l_t1, acc_t1, qk_scale_log2, P_SCALE, MFMA, MF3, NUM_WARPS)
                if KV_LDS:
                    v3_c = _ws_v_lds(vw_smem, buf, B3, NUM_WARPS)
                acc_t1 = _ws_pv(p3, v3_c, acc_t1, A3)
            if BLOCK_Q == 4:
                if rb_t2 >= 0:
                    p3, m_t2, l_t2, acc_t2 = _ws_qk_softmax(q_t2, qs_t2, k_c, ks_c, rb_t2, pos_t2, tok_n, m_t2,
                                                                l_t2, acc_t2, qk_scale_log2, P_SCALE, MFMA, MF3, NUM_WARPS)
                    if KV_LDS:
                        v3_c = _ws_v_lds(vw_smem, buf, B3, NUM_WARPS)
                    acc_t2 = _ws_pv(p3, v3_c, acc_t2, A3)
                if rb_t3 >= 0:
                    p3, m_t3, l_t3, acc_t3 = _ws_qk_softmax(q_t3, qs_t3, k_c, ks_c, rb_t3, pos_t3, tok_n, m_t3,
                                                                l_t3, acc_t3, qk_scale_log2, P_SCALE, MFMA, MF3, NUM_WARPS)
                    if KV_LDS:
                        v3_c = _ws_v_lds(vw_smem, buf, B3, NUM_WARPS)
                    acc_t3 = _ws_pv(p3, v3_c, acc_t3, A3)
                rb_t2 = rb_n2
                rb_t3 = rb_n3
            if KV_LDS:
                buf = 1 - buf
            else:
                k_c = k_n
                ks_c = ks_n
                v3_c = v3_n
            rb_t0 = rb_n0
            rb_t1 = rb_n1
            p_nxt = p_nxt2
        # Merge each row's waves: normalized per-wave partials (fp16) into one LDS
        # buffer [wave, row, d], one barrier, then a read with the wave axis in each
        # thread's registers (an in-thread sum). The output is normalized (l = 1).
        MRG_SH: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[D, 8]], [NUM_WARPS, BLOCK_M, D], [2, 1, 0])
        MRG_LD: gl.constexpr = gl.BlockedLayout([NUM_WARPS, 1, 8], [1, 4, 16], [1, NUM_WARPS, 1], [2, 1, 0])
        # The merge buffer reuses the K / V buffers' LDS: drain this wave's last
        # (unused) prefetch and wait for every wave to leave the page loop first.
        if KV_LDS:
            gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()
        mrg = gl.allocate_shared_memory(gl.float16, [NUM_WARPS, BLOCK_M, D], MRG_SH)
        _ws_merge_store(mrg, 0, m_t0, l_t0, acc_t0)
        _ws_merge_store(mrg, 1, m_t1, l_t1, acc_t1)
        if BLOCK_Q == 4:
            _ws_merge_store(mrg, 2, m_t2, l_t2, acc_t2)
            _ws_merge_store(mrg, 3, m_t3, l_t3, acc_t3)
        gl.barrier()
        acc = gl.convert_layout(gl.sum(mrg.load(MRG_LD).to(gl.float32), axis=0), MFMA)
        m_i = gl.zeros([BLOCK_M], gl.float32, ROW)
        l_i = gl.full([BLOCK_M], 1.0, gl.float32, ROW)
    else:
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
        if OUT_MX:
            _store_out_mxfp8(
                acc * inv_l[:, None], r0, g, num_rows, mxq_ptr, mxs_ptr, stride_mq, stride_ms,
                BLOCK_M, NUM_WARPS,
            )  # fmt: skip
        else:
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
            mm = gl.arange(0, BLOCK_M, layout=ROW)
            if ATOMIC:
                # write-through: complete at the coherence point once vmcnt drains
                gl.store(part_o_ptr + part * (BLOCK_M * D) + om * D + od, acc, cache_modifier=".wt")
                gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2, m_i, cache_modifier=".wt")
                gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2 + 1, l_i, cache_modifier=".wt")
            else:
                gl.store(part_o_ptr + part * (BLOCK_M * D) + om * D + od, acc)
                gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2, m_i)
                gl.store(part_ml_ptr + part * (BLOCK_M * 2) + mm * 2 + 1, l_i)
        if ATOMIC:
            # Every segment program takes a ticket once its partial is out; the
            # last to arrive merges the live ones in segment order (as the reduce
            # kernel would, so no reduce launch) and resets the counter. As the
            # native NVFP4 decode: write-through partials, the barrier waits for
            # them (vmcnt), a relaxed ticket, and the merge reads past L2 (.cv),
            # so no L2 writeback / invalidate fence in every program.
            gl.barrier()
            ticket = gl.atomic_add(counter_ptr + pair, 1, sem="relaxed", scope="gpu")
            if ticket == NSEG - 1:
                gl.store(counter_ptr + pair, 0)
                _merge_segments(
                    part_o_ptr, part_ml_ptr, out_ptr, v_scale_ptr, pair, grp, g,
                    (n_pages + seg_pages - 1) // seg_pages, num_rows, stride_ot, stride_oh,
                    mxq_ptr, mxs_ptr, stride_mq, stride_ms,
                    BLOCK_M, NSEG, NUM_WARPS, OUT_MX, ".cv",
                )  # fmt: skip




@gluon.jit
def _store_out_mxfp8(
    o, r0, g, num_rows, mxq_ptr, mxs_ptr, stride_mq, stride_ms,
    BLOCK_M: gl.constexpr, NUM_WARPS: gl.constexpr,
):  # fmt: skip
    """o (fp32 [BLOCK_M, D], normalized, any layout) as o_proj's MXFP8 input:
    e4m3 [rows, H * D] + E8M0 [rows, H * D / 32], row-major, bit-exact with
    vLLM's mxfp8_e4m3_quantize of the bf16 output (scale 2^(sb - 127) with
    sb = clamp(ceil(log2(amax / 448)) + 127, 0, 254), values times
    2^(127 - sb)). Row m of the group is query row r0 + m // 16, head
    g * 16 + m % 16. Each thread holds one 32-dim MX block of a row."""
    LX: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [NUM_WARPS, 1], [1, 0])
    x = gl.convert_layout(o.to(gl.bfloat16), LX).to(gl.float32)
    om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LX))
    od = gl.arange(0, D, layout=gl.SliceLayout(0, LX))[None, :]
    row = r0 + om // HG
    hd = g * HG + om % HG
    ok = row < num_rows
    blk = od // 32
    ax = gl.abs(x)
    rs = gl.zeros([BLOCK_M, D], gl.float32, LX)
    for j in gl.static_range(D // 32):
        amax = gl.maximum(gl.max(gl.where(blk == j, ax, 0.0), axis=1), 1.1754943508222875e-38)
        sb = gl.minimum(gl.maximum(gl.ceil(gl.log2(amax / 448.0)) + 127.0, 0.0), 254.0)
        rs = gl.where(blk == j, gl.exp2(127.0 - sb)[:, None], rs)
        gl.store(mxs_ptr + row.to(gl.int64) * stride_ms + hd * (D // 32) + j, sb.to(gl.uint8), mask=ok)
    gl.store(
        mxq_ptr + row.to(gl.int64)[:, None] * stride_mq + hd[:, None] * D + od,
        (x * rs).to(gl.float8e4nv),
        mask=ok[:, None],
    )


@gluon.jit
def _merge_segments(
    part_o_ptr, part_ml_ptr, out_ptr, v_scale_ptr, pair, grp, g, live, num_rows,
    stride_ot, stride_oh, mxq_ptr, mxs_ptr, stride_mq, stride_ms,
    BLOCK_M: gl.constexpr, NSEG: gl.constexpr, NUM_WARPS: gl.constexpr,
    OUT_MX: gl.constexpr = False, LOAD_CM: gl.constexpr = "",
):  # fmt: skip
    """Merge a row group's first `live` segment partials and write the output.
    All partials load at once (the segment axis in each thread's registers),
    then: the max over segments, each segment's weight, and the weighted sums
    over segments in segment order -- deterministic, the same arithmetic for
    msa_gfx950_gluon_reduce_split_rebalance_kernel and for the last segment
    program to arrive (msa_gfx950_gluon_kernel, ATOMIC; LOAD_CM ".cv")."""
    NS: gl.constexpr = triton_next_pow2(NSEG)
    L3: gl.constexpr = gl.BlockedLayout([NS, 1, 8], [1, 4, 16], [1, NUM_WARPS, 1], [2, 1, 0])
    L: gl.constexpr = gl.SliceLayout(0, L3)
    LS: gl.constexpr = gl.SliceLayout(2, L3)
    R: gl.constexpr = gl.SliceLayout(1, L)
    sg = gl.arange(0, NS, layout=gl.SliceLayout(1, LS))[:, None]  # [NS, 1]
    mr = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, LS))[None, :]  # [1, BLOCK_M]
    ok2 = (sg < live) & (mr >= 0)
    seg = pair * NSEG + sg
    m2 = gl.load(part_ml_ptr + seg * (BLOCK_M * 2) + mr * 2, mask=ok2, other=float("-inf"), cache_modifier=LOAD_CM)
    l2 = gl.load(part_ml_ptr + seg * (BLOCK_M * 2) + mr * 2 + 1, mask=ok2, other=0.0, cache_modifier=LOAD_CM)
    s3 = gl.arange(0, NS, layout=gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    m3 = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    d3 = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
    ok3 = (s3 < live) & (m3 >= 0) & (d3 >= 0)
    o3 = gl.load(part_o_ptr + (pair * NSEG + s3) * (BLOCK_M * D) + m3 * D + d3, mask=ok3, other=0.0, cache_modifier=LOAD_CM)
    m_t = gl.max(m2, axis=0)
    m_use = gl.where(m_t == float("-inf"), 0.0, m_t)
    w = gl.exp2(m2 - m_use[None, :])  # 0 for dead segments (m = -inf)
    l_t = gl.sum(w * l2, axis=0)
    acc = gl.sum(gl.convert_layout(w, gl.SliceLayout(2, L3))[:, :, None] * o3, axis=0)
    v_scale = gl.load(v_scale_ptr)
    inv_l = gl.convert_layout(gl.where(l_t > 0, v_scale / l_t, 0.0), R)
    if OUT_MX:
        _store_out_mxfp8(
            acc * inv_l[:, None], grp * (BLOCK_M // HG), g, num_rows, mxq_ptr, mxs_ptr,
            stride_mq, stride_ms, BLOCK_M, NUM_WARPS,
        )  # fmt: skip
    else:
        om = gl.arange(0, BLOCK_M, layout=R)
        od = gl.arange(0, D, layout=gl.SliceLayout(0, L))[None, :]
        row = grp * (BLOCK_M // HG) + om // HG
        gl.store(
            out_ptr + row.to(gl.int64)[:, None] * stride_ot + (g * HG + om % HG)[:, None] * stride_oh + od,
            (acc * inv_l[:, None]).to(gl.bfloat16),
            mask=(row < num_rows)[:, None],
        )


_msa_gfx950_gluon_reduce_split_rebalance_kernel_repr = make_kernel_repr(
    "msa_gfx950_gluon_reduce_split_rebalance_kernel",
    ['BLOCK_M', 'NSEG', 'NUM_HEADS', 'NUM_WARPS', 'OUT_MX'],
)


@gluon.jit(repr=_msa_gfx950_gluon_reduce_split_rebalance_kernel_repr)
def msa_gfx950_gluon_reduce_split_rebalance_kernel(
    ws_ptr,  # fp32 [part_o [Hkv, groups, NSEG, BLOCK_M, D] | part_ml [.., BLOCK_M, 2] | seg_count]
    out_ptr,
    mxq_ptr,  # OUT_MX: e4m3 [rows, H * D] + E8M0 [rows, H * D / 32]
    mxs_ptr,
    v_scale_ptr,
    num_rows,
    seg_pages,
    stride_ot,
    BLOCK_M: gl.constexpr,
    NSEG: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    OUT_MX: gl.constexpr,
):
    """The split-rebalance reduce of msa_gfx950_gluon_kernel: one program per
    (row group, KV head) merges its live segments, cdiv(pages, seg_pages) of
    them, in segment order (deterministic) with the next segment's partial in
    flight; segments past a short list were never written and are not read.
    Row m of a group is query row grp * BLOCK_M / 16 + m // 16, head
    g * 16 + m % 16; a row group with no pages gives zeros."""
    grp = gl.program_id(0)
    g = gl.program_id(1)
    n_part = gl.num_programs(0).to(gl.int64) * gl.num_programs(1) * NSEG
    part_ml_ptr = ws_ptr + n_part * (BLOCK_M * D)
    seg_count_ptr = (part_ml_ptr + n_part * (BLOCK_M * 2)).to(gl.pointer_type(gl.int32), bitcast=True)
    pair = (g * gl.num_programs(0) + grp).to(gl.int64)
    live = (gl.load(seg_count_ptr + pair) + seg_pages - 1) // seg_pages
    _merge_segments(
        ws_ptr, part_ml_ptr, out_ptr, v_scale_ptr, pair, grp, g, live, num_rows,
        stride_ot, D, mxq_ptr, mxs_ptr, NUM_HEADS * D, NUM_HEADS * NUM_SCALES,
        BLOCK_M, NSEG, NUM_WARPS, OUT_MX,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# k4v4pt decode. Record per (page, KV head), 17,408 B (block 128, head dim 128):
#   [K e2m1 128 x 64 B, preshuffled as k4v8 | Ks E8M0 128 x 4 | V e2m1 128 x 64 B
#    packed along D (low nibble first) | Vs fp32 128, one per token]
# QK = mfma_scaled(MXFP8 Q, Qs, MXFP4 K, Ks). P is scaled by Vs along the keys
# while fp32, then e4m3: P' = P * Vs[t] * (P_SCALE / max_t Vs) (the per-page
# factor max Vs / P_SCALE keeps e4m3 in range and rides on O's per-row
# rescale). PV = mfma_scaled(e4m3 P', e2m1 V) with no scales, V read from LDS
# packed along the keys by load_shared_fp4_repacked (ds_read_b64_tr_b4).
# ---------------------------------------------------------------------------

K4V4PT_V_OFF = gl.constexpr(K_BYTES + KS_BYTES)  # 8704
K4V4PT_VS_OFF = gl.constexpr(K_BYTES + KS_BYTES + 128 * 128 // 2)  # 16896
K4V4PT_RECORD_BYTES = gl.constexpr(K_BYTES + KS_BYTES + 128 * 128 // 2 + 128 * 4)  # 17408


_msa_gfx950_gluon_k4v4pt_kernel_repr = make_kernel_repr(
    "msa_gfx950_gluon_k4v4pt_kernel",
    ["NSEG", "DECODE_QL", "TOPK", "NUM_HEADS", "NUM_WARPS"],
)


@gluon.jit(repr=_msa_gfx950_gluon_k4v4pt_kernel_repr)
def msa_gfx950_gluon_k4v4pt_kernel(
    q8_ptr,  # e4m3 [rows, NUM_HEADS, D] (MXFP8 terms)
    qs_ptr,  # u8 E8M0 [rows, NUM_HEADS, D / 32]
    cache_ptr,  # u8 k4v4pt records [pages, Hkv, K4V4PT_RECORD_BYTES]
    topk_ptr,  # [Hkv, rows, TOPK] int32 logical blocks, -1 = none
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,  # bf16 [rows, NUM_HEADS, D] (NSEG = 1)
    ws_ptr,  # NSEG > 1: fp32 [part_o | part_ml | seg_count (int32)], as k4v8's
    qk_scale_log2,
    seg_pages,  # pages a segment covers (split rebalance)
    stride_page,
    stride_tk_h,
    stride_bt,
    stride_ot,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    NSEG: gl.constexpr,
    DECODE_QL: gl.constexpr,
    P_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """One program: a decode row's 16 heads of one KV head, one segment of its
    compacted picks (split rebalance, as k4v8; NSEG > 1 writes partials for
    msa_gfx950_gluon_reduce_split_rebalance_kernel with v_scale 1).

    Every K / Ks / V / Vs byte goes HBM -> LDS by async copy into double buffers;
    two commit groups a page, [K, Ks + Vs] then [V]. Page j:
      wait_group(1): page j's K / scales in   barrier (every wave's copies of it;
        every wave done with page j - 1, so its buffer can be refilled)
      issue page j + 1's two groups
      LDS: K (unshuffling view), Ks, Vs; QK, mask, softmax, P' (e4m3)
      wait_group(2): page j's V in   barrier
      LDS: V packed along the keys (ds_read_b64_tr_b4); PV
    The copies are not wave-local (the transposed V read takes 8 B of every
    token's row), hence the barriers."""
    BLOCK_M: gl.constexpr = 16
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NUM_WARPS])
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MFMA, 16)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MFMA, 16)
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [BLOCK_M, NUM_SCALES])
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B_OP, [PAGE, NUM_SCALES])
    ROW: gl.constexpr = gl.SliceLayout(1, MFMA)
    COL: gl.constexpr = gl.SliceLayout(0, MFMA)
    VEC: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    FLAT2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    CP16: gl.constexpr = gl.BlockedLayout([1, 16], [1, 64], [NUM_WARPS, 1], [1, 0])  # [8, 1024], 16 B lanes
    CP4: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [2, NUM_WARPS // 2], [1, 0])  # [2, 512], 4 B lanes

    grp = gl.num_programs(0) - 1 - gl.program_id(0)
    g = gl.program_id(1)
    sid = gl.program_id(2)

    qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, A_OP))[:, None]
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, A_OP))[None, :]
    q = gl.load(q8_ptr + (grp.to(gl.int64) * NUM_HEADS + g * HG + qm) * D + qd)
    sm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_SCL))[:, None]
    ss = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL))[None, :]
    qs = gl.load(qs_ptr + (grp.to(gl.int64) * NUM_HEADS + g * HG + sm) * NUM_SCALES + ss)

    # copy sources (byte offsets inside a record) and the LDS double buffers
    blk_src = (
        gl.arange(0, 8, layout=gl.SliceLayout(1, CP16))[:, None] * 1024
        + gl.arange(0, 1024, layout=gl.SliceLayout(0, CP16))[None, :]
    )
    srow = gl.arange(0, 2, layout=gl.SliceLayout(1, CP4))[:, None]
    s_src = gl.where(srow == 0, K_BYTES, K4V4PT_VS_OFF) + gl.arange(0, 512, layout=gl.SliceLayout(0, CP4))[None, :]
    k_smem = gl.allocate_shared_memory(gl.uint8, [2, 8, 1024], FLAT2)
    v_smem = gl.allocate_shared_memory(gl.uint8, [2, 8, 1024], FLAT2)
    s_smem = gl.allocate_shared_memory(gl.uint8, [2, 2, 512], FLAT2)  # [buf, Ks | Vs]
    tok_n = gl.arange(0, PAGE, layout=COL)
    head_off = g * K4V4PT_RECORD_BYTES

    # the row's picks for this segment, compacted: safe ones first, the diagonal last
    pos = gl.load(seq_lens_ptr + grp // DECODE_QL) - DECODE_QL + grp % DECODE_QL
    req = (grp // DECODE_QL).to(gl.int64)
    jj = gl.arange(0, TOPK, layout=VEC)
    blks = gl.load(topk_ptr + g * stride_tk_h + grp.to(gl.int64) * TOPK + jj)
    valid = (blks >= 0) & (blks * PAGE <= pos)
    pages = gl.load(block_table_ptr + req * stride_bt + gl.where(valid, blks, 0), mask=valid, other=0)
    diag = valid & (blks == pos // PAGE)
    safe = valid & (blks != pos // PAGE)
    n_safe = gl.sum(safe.to(gl.int32), axis=0)
    rank = gl.where(safe, gl.associative_scan(safe.to(gl.int32), 0, _add) - 1, gl.where(diag, n_safe, -1))
    n_pages = n_safe + gl.sum(diag.to(gl.int32), axis=0)
    j0 = sid * seg_pages
    j_end = gl.minimum(n_pages, j0 + seg_pages)
    n_iter = gl.maximum(j_end - j0, 0)
    b_cur, p_cur = _nth_valid(rank, blks, pages, j0, j_end)
    b_nxt, p_nxt = _nth_valid(rank, blks, pages, j0 + 1, j_end)

    # page j0 into buffer 0: [K, Ks + Vs], then [V]
    base = p_cur.to(gl.int64) * stride_page + head_off
    gl.amd.cdna4.async_copy.global_load_to_shared(k_smem.index(0), cache_ptr + base + blk_src)
    gl.amd.cdna4.async_copy.global_load_to_shared(s_smem.index(0), cache_ptr + base + s_src)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.global_load_to_shared(v_smem.index(0), cache_ptr + base + K4V4PT_V_OFF + blk_src)
    gl.amd.cdna4.async_copy.commit_group()

    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, ROW)
    l_i = gl.zeros([BLOCK_M], gl.float32, ROW)
    acc = gl.zeros([BLOCK_M, D], gl.float32, MFMA)
    f_acc = 1.0  # O is held in units of the last page's max Vs / P_SCALE
    buf = 0
    for j in range(n_iter):
        b_nxt2, p_nxt2 = _nth_valid(rank, blks, pages, j0 + j + 2, j_end)
        gl.amd.cdna4.async_copy.wait_group(1)  # page j's K / scales (its V may still be in flight)
        gl.barrier()  # every wave's: K / scales of page j landed, page j - 1 consumed
        nbase = p_nxt.to(gl.int64) * stride_page + head_off
        gl.amd.cdna4.async_copy.global_load_to_shared(k_smem.index(1 - buf), cache_ptr + nbase + blk_src)
        gl.amd.cdna4.async_copy.global_load_to_shared(s_smem.index(1 - buf), cache_ptr + nbase + s_src)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.global_load_to_shared(v_smem.index(1 - buf), cache_ptr + nbase + K4V4PT_V_OFF + blk_src)
        gl.amd.cdna4.async_copy.commit_group()

        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(_k_view(k_smem.index(buf)), B_OP)
        sb = s_smem.index(buf)
        ks_c = sb.reinterpret(gl.uint8, [PAGE, NUM_SCALES], FLAT2).load(K_SCL)
        vs_c = sb.reinterpret(gl.float32, [2, PAGE], FLAT1).index(1).load(COL)

        s = gl.zeros([BLOCK_M, PAGE], gl.float32, MFMA)
        s = gl.amd.cdna4.mfma_scaled(q, qs, "e4m3", k_c, ks_c, "e2m1", s)
        if j0 + j == n_safe:  # the diagonal block: mask the keys after pos
            s = gl.where((b_cur * PAGE + tok_n <= pos)[None, :], s, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale_log2)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(s * qk_scale_log2 - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        # P * Vs along the keys while fp32, in units of the page's max Vs / P_SCALE
        vmax = gl.max(vs_c, axis=0)
        p8 = (p * (vs_c * (P_SCALE / vmax))[None, :]).to(gl.float8e4nv)
        f_new = vmax / P_SCALE
        acc = acc * (alpha * (f_acc / f_new))[:, None]

        gl.amd.cdna4.async_copy.wait_group(2)  # page j's V (page j + 1's two groups in flight)
        gl.barrier()  # every wave's V copy of page j
        v_op = gl.amd.cdna4.load_shared_fp4_repacked(v_smem.index(buf).reshape([PAGE, D // 2]), B_OP)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(p8, A_OP), None, "e4m3", v_op, None, "e2m1", acc)
        f_acc = f_new
        m_i = m_new
        buf = 1 - buf
        b_cur = b_nxt
        b_nxt = b_nxt2
        p_nxt = p_nxt2
    gl.amd.cdna4.async_copy.wait_group(0)  # the last (unused) prefetch

    om = gl.arange(0, BLOCK_M, layout=ROW)[:, None]
    od = gl.arange(0, D, layout=COL)[None, :]
    if NSEG == 1:
        o = acc * gl.where(l_i > 0, f_acc / l_i, 0.0)[:, None]
        gl.store(out_ptr + grp.to(gl.int64) * stride_ot + (g * HG + om) * D + od, o.to(gl.bfloat16))
    else:
        n_prog = gl.num_programs(0).to(gl.int64) * gl.num_programs(1)
        n_part = n_prog * NSEG
        part_ml = ws_ptr + n_part * (BLOCK_M * D)
        seg_count = (part_ml + n_part * (BLOCK_M * 2)).to(gl.pointer_type(gl.int32), bitcast=True)
        pair = (g * gl.num_programs(0) + grp).to(gl.int64)
        if sid == 0:
            gl.store(seg_count + pair, n_pages)
        if n_iter > 0:
            part = pair * NSEG + sid
            mm = gl.arange(0, BLOCK_M, layout=ROW)
            gl.store(ws_ptr + part * (BLOCK_M * D) + om * D + od, acc * f_acc)
            gl.store(part_ml + part * (BLOCK_M * 2) + mm * 2, m_i)
            gl.store(part_ml + part * (BLOCK_M * 2) + mm * 2 + 1, l_i)
