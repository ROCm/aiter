# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cascade (shared-prefix) decode attention kernels.

Companion kernels to the FlyDSL ``pa_decode`` op. A cascade decode step
splits each grouped request's attention into

* one **shared pass** (``_cascade_shared_kernel``): one work item per
  item of up to ``MEMBERS`` sibling requests streams the block-aligned
  shared prefix once, packing every member's GQA rows into a single
  MFMA tile; for occupancy, each item's prefix splits across
  ``A_SEGMENTS`` segments, each writing ``(m, l, acc)`` partials per
  member; and
* a **tail pass**: the stock FlyDSL ``pa_decode`` over each member's
  private tail (block table shifted by the item's shared block count),
  with partition partials staged in the usual
  ``exp_sums``/``max_logits``/``temporary_output`` buffers,

and merges all partials in log-sum-exp space
(``_cascade_fold_kernel``), adding optional attention sinks exactly once.

The merge is exact: softmax attention over a concatenation ``[S | T]``
equals the LSE combine of the two passes -- the same identity split-KV
decode relies on. Partials use natural-log statistics throughout
(``m`` in natural log, unnormalised ``acc``, non-normalised ``l``).
"""

import triton
import triton.language as tl

_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _cascade_shared_kernel(
    # Q of the full decode batch: [rows, num_q_heads, head_dim], dim contiguous.
    q_ptr,
    # K/V caches, 5D packed BF16 layouts (contiguous):
    #   K: [num_blocks, num_kv_heads, head_dim // 8, page, 8]
    #   V: [num_blocks, num_kv_heads, page // 8, head_dim, 8]
    k_ptr,
    v_ptr,
    # Block table of the full batch: [rows, max_blocks_per_seq] int32,
    # contiguous rows (bt_row_stride == max_blocks_per_seq).
    bt_ptr,
    # Item table: [max_items, 2 + MEMBERS] int32 rows laid out as
    #   (sblk, count, member_row[0], ..., member_row[MEMBERS-1])
    # with count <= 0 marking a padded (no-op) item.
    items_ptr,
    # A partials: am/al [max_items, A_SEGMENTS, MEMBERS, num_q_heads] fp32,
    # aacc [max_items, A_SEGMENTS, MEMBERS, num_q_heads, head_dim] fp32.
    am_ptr,
    al_ptr,
    aacc_ptr,
    # Runtime strides (elements).
    bt_row_stride,
    am_item_stride,
    am_seg_stride,
    am_mem_stride,
    aa_item_stride,
    aa_seg_stride,
    aa_mem_stride,
    softmax_scale_log2,  # softmax_scale * log2(e)
    PAGE: tl.constexpr,  # tokens per KV block
    GQA: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,  # power of two
    MEMBERS: tl.constexpr,  # members per item
    BLOCK_MG: tl.constexpr,  # next_pow2(MEMBERS * GQA)
    Q_ROW_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    K_BLK_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_BLK_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
):
    item = tl.program_id(0)
    seg = tl.program_id(1)
    kvh = tl.program_id(2)
    A_SEGMENTS: tl.constexpr = tl.num_programs(1)

    meta0 = items_ptr + item * (2 + MEMBERS)
    sblk = tl.load(meta0 + 0)
    count = tl.load(meta0 + 1)
    if count <= 0:
        # Padded item: keep the launch grid static (CUDA-graph friendly).
        return

    member = tl.arange(0, BLOCK_MG) // GQA
    g = tl.arange(0, BLOCK_MG) % GQA
    member_ok = member < count
    member_row = tl.load(meta0 + 2 + member, mask=member_ok, other=0).to(tl.int64)
    head = kvh * GQA + g
    d = tl.arange(0, HEAD_DIM)

    # i64 row offsets: row strides times row indices can exceed 2**31
    # elements on large caches/batches.
    q = tl.load(
        q_ptr
        + member_row[:, None] * Q_ROW_STRIDE
        + head[:, None] * Q_HEAD_STRIDE
        + d[None, :],
        mask=member_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    # Scale Q into log2 units once, so scores are directly exp2-compatible.
    q = (q * softmax_scale_log2).to(tl.bfloat16)

    # Any member row serves the shared block range; use member 0.
    bt_row = tl.load(meta0 + 2).to(tl.int64)

    m_i = tl.full([BLOCK_MG], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_MG], tl.float32)
    acc = tl.zeros([BLOCK_MG, HEAD_DIM], tl.float32)
    tt = tl.arange(0, PAGE)

    # Segment-local shared range [first_blk, last_blk). Segments are
    # block-aligned, so every iteration covers one full 16-token block:
    # a scalar i64 block base plus fully affine per-element offsets keeps
    # address math in 32-bit registers and the loads vectorized.
    first_blk = tl.minimum(sblk * seg // A_SEGMENTS, sblk)
    last_blk = tl.minimum(sblk * (seg + 1) // A_SEGMENTS, sblk)
    base_ptr = bt_ptr + bt_row * bt_row_stride
    for j in range(first_blk, last_blk):
        bid = tl.load(base_ptr + j).to(tl.int64)
        k = tl.load(
            k_ptr
            + bid * K_BLK_STRIDE
            + kvh * K_HEAD_STRIDE
            + tt[:, None] * 8
            + (d[None, :] // 8) * (PAGE * 8)
            + (d[None, :] % 8)
        )

        s = tl.dot(q, tl.trans(k))  # fp32, log2 units
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        alpha = tl.where(m_new == float("-inf"), 0.0, alpha)
        alpha = tl.where(m_i == float("-inf"), 1.0, alpha)
        p = tl.exp2(s - m_new[:, None])

        v = tl.load(
            v_ptr
            + bid * V_BLK_STRIDE
            + kvh * V_HEAD_STRIDE
            + (tt[:, None] // 8) * (HEAD_DIM * 8)
            + d[None, :] * 8
            + (tt[:, None] % 8)
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    m_nat = m_i * _LN2
    am_base = item * am_item_stride + seg * am_seg_stride + member * am_mem_stride
    tl.store(am_ptr + am_base + head, m_nat, mask=member_ok)
    tl.store(al_ptr + am_base + head, l_i, mask=member_ok)
    tl.store(
        aacc_ptr
        + item * aa_item_stride
        + seg * aa_seg_stride
        + member[:, None] * aa_mem_stride
        + head[:, None] * HEAD_DIM
        + d[None, :],
        acc,
        mask=member_ok[:, None],
    )


@triton.jit
def _cascade_fold_kernel(
    # Final output of the full decode batch: [rows, num_q_heads, head_dim].
    out_ptr,
    # A partials from the shared pass (see _cascade_shared_kernel).
    am_ptr,
    al_ptr,
    aacc_ptr,
    # Member bookkeeping (device, int32):
    #   b_map[i]   original batch row of B-pass row i
    #   b_item[i]  item index of B-pass row i
    #   b_slot[i]  member slot of B-pass row i within its item
    #   n_ptr[0]   number of live B-pass rows (padded rows exit early)
    b_map_ptr,
    b_item_ptr,
    b_slot_ptr,
    n_ptr,
    # B-pass staging written by FlyDSL pa_decode's tile kernel:
    #   pmax [num_kv_heads, capacity, rows] fp32, natural-log partition max
    #   psum [num_kv_heads, capacity, rows] fp32, partition sum-exp
    #   pout [num_kv_heads, capacity, rows, head_dim] out dtype,
    #        per-partition NORMALIZED output (acc / l)
    pmax_ptr,
    psum_ptr,
    pout_ptr,
    # B-pass plan's [n_b, 2] int32: (first packed task, task count) per row.
    reduce_info_ptr,
    # Optional attention sinks [num_q_heads], unscaled natural-log logits.
    sinks_ptr,
    # Strides (elements); pout's group stride is HEAD_DIM.
    out_row_stride,
    out_head_stride,
    am_item_stride,
    am_seg_stride,
    am_mem_stride,
    aa_item_stride,
    aa_seg_stride,
    aa_mem_stride,
    ps_head_stride,
    ps_part_stride,
    po_head_stride,
    po_part_stride,
    GQA: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,
    A_SEGMENTS: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    KERNEL_MAX_PARTITIONS: tl.constexpr,  # B-pass plan's max_partitions
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    n = tl.load(n_ptr)
    if b >= n:
        return
    kvh = head // GQA
    g = head % GQA
    d = tl.arange(0, HEAD_DIM)

    item = tl.load(b_item_ptr + b)
    slot = tl.load(b_slot_ptr + b)
    # i64 bases: the staging head strides span capacity * rows * head_dim
    # elements and can exceed 2**31 on large B batches with many heads.
    kvh64 = kvh.to(tl.int64)

    sink = tl.load(sinks_ptr + head) if HAS_SINKS else float("-inf")
    m_a = tl.load(am_ptr + item * am_item_stride + slot * am_mem_stride + head)
    big_m = tl.maximum(m_a, sink)
    for sgm in range(1, A_SEGMENTS):
        m_s = tl.load(
            am_ptr
            + item * am_item_stride
            + sgm * am_seg_stride
            + slot * am_mem_stride
            + head
        )
        big_m = tl.maximum(big_m, m_s)

    first = tl.load(reduce_info_ptr + b * 2)
    count = tl.load(reduce_info_ptr + b * 2 + 1)
    for p in range(KERNEL_MAX_PARTITIONS):
        if p < count:
            m_p = tl.load(
                pmax_ptr + kvh64 * ps_head_stride + (first + p) * ps_part_stride + g
            )
            big_m = tl.maximum(big_m, m_p)

    # An entirely empty row (no shared, no tail, -inf or absent sink) has no
    # finite mass: store zeros rather than NaNs.
    if big_m == float("-inf"):
        row = tl.load(b_map_ptr + b).to(tl.int64)
        tl.store(
            out_ptr + row * out_row_stride + head * out_head_stride + d,
            tl.zeros([HEAD_DIM], tl.float32).to(out_ptr.dtype.element_ty),
        )
        return

    # exp(sink - big_m), guarding -inf/inf collisions (mirror the stock
    # reducer's "+inf sinks suppress the head, -inf contributes nothing").
    sink_mass = tl.exp(tl.where(sink == big_m, 0.0, sink - big_m))
    sink_mass = tl.where(sink == float("-inf"), 0.0, sink_mass)

    num = tl.zeros([HEAD_DIM], tl.float32)
    den = sink_mass
    for sgm in range(A_SEGMENTS):
        m_s = tl.load(
            am_ptr
            + item * am_item_stride
            + sgm * am_seg_stride
            + slot * am_mem_stride
            + head
        )
        l_s = tl.load(
            al_ptr
            + item * am_item_stride
            + sgm * am_seg_stride
            + slot * am_mem_stride
            + head
        )
        a_s = tl.load(
            aacc_ptr
            + item * aa_item_stride
            + sgm * aa_seg_stride
            + slot * aa_mem_stride
            + head * HEAD_DIM
            + d
        )
        w_s = tl.exp(m_s - big_m)
        num += a_s * w_s
        den += l_s * w_s

    for p in range(KERNEL_MAX_PARTITIONS):
        if p < count:
            m_p = tl.load(
                pmax_ptr + kvh64 * ps_head_stride + (first + p) * ps_part_stride + g
            )
            l_p = tl.load(
                psum_ptr + kvh64 * ps_head_stride + (first + p) * ps_part_stride + g
            )
            po = tl.load(
                pout_ptr
                + kvh64 * po_head_stride
                + (first + p) * po_part_stride
                + g * HEAD_DIM
                + d
            ).to(tl.float32)
            w_p = l_p * tl.exp(m_p - big_m)
            num += po * w_p
            den += w_p

    row = tl.load(b_map_ptr + b).to(tl.int64)
    tl.store(
        out_ptr + row * out_row_stride + head * out_head_stride + d,
        (num / den).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _cascade_scatter_kernel(
    # Copy singles' pa_decode output rows into the full-batch output.
    src_ptr,  # [max_singles, num_q_heads, head_dim]
    dst_ptr,  # [rows, num_q_heads, head_dim]
    rows_ptr,  # [max_singles] int32 destination rows (b_map style)
    n_ptr,
    src_row_stride,
    dst_row_stride,
    dst_head_stride,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    n = tl.load(n_ptr)
    if b >= n:
        return
    row = tl.load(rows_ptr + b).to(tl.int64)
    d = tl.arange(0, HEAD_DIM)
    val = tl.load(src_ptr + b * src_row_stride + head * HEAD_DIM + d)
    tl.store(dst_ptr + row * dst_row_stride + head * dst_head_stride + d, val)
