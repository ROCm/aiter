# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TDM load / WMMA consume helpers for gfx1250 Gluon kernels.

The ring buffer holds NUM_BUFFERS slots. A producer index issues the TDM copies, a consumer index
drains them and multiplies. Both indices are plain counters, so the slot is index % NUM_BUFFERS.
"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon


@gluon.jit
def issue_loads(
    producer,
    a_d,
    b_d,
    a_buffer,
    b_buffer,
    BLOCK_K: gl.constexpr,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    TRANSPOSE_A: gl.constexpr = False,
):
    """Start the TDM copies for one K step, then step the descriptors one K tile.

    The descriptors are carried, so a load needs no offset maths: the caller placed them on the
    tile once and each step adds BLOCK_K. Rebuilding them from the base every load instead cost
    a multiply and a bounds update for each load, 0.211 scalar ALU ops for each wmma against
    0.078 in the reference kernel.

    clamp_bounds keeps the descriptor bounds with the moved offsets. Without it the last K tile
    of a K that does not divide BLOCK_K reads past K instead of zero-filling, and every such
    shape is wrong: 64x64x288 gave a relative error of 0.94. Compiling it out when BLOCK_K
    divides K takes the loop's scalar ALU from 48 to 20 ops for each 256 wmma, matching the
    reference kernel exactly, and measures flat: that axis is not the gap.

    A single CTA has no multicast, so it issues both loads itself. One fused op then costs one
    TDM slot instead of two, which is worth 3% on the small-M buckets. A cluster shares the
    loads already, where fusing measures flat, so it keeps the two separate copies.
    """
    num_ctas: gl.constexpr = gl.num_ctas()
    slot = producer % NUM_BUFFERS
    if num_ctas == 1:
        # Split the warps between the two members; the hints must be disjoint bitmasks.
        half: gl.constexpr = gl.num_warps() // 2
        hint_a: gl.constexpr = (1 << half) - 1
        hint_b: gl.constexpr = ((1 << gl.num_warps()) - 1) ^ hint_a
        gl.amd.cdna5.tdm.async_load_fused(
            [(a_d, a_buffer.index(slot), hint_a), (b_d, b_buffer.index(slot), hint_b)]
        )
    else:
        gl.amd.cdna5.tdm.async_load(a_d, [0, 0], a_buffer.index(slot))
        gl.amd.cdna5.tdm.async_load(b_d, [0, 0], b_buffer.index(slot))
    # A transposed is [BK, BM] with M innermost, so K steps the first axis, as for native B.
    if TRANSPOSE_A:
        a_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            a_d, add_offsets=[BLOCK_K, 0], clamp_bounds=True
        )
    else:
        a_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            a_d, add_offsets=[0, BLOCK_K], clamp_bounds=True
        )
    if TRANSPOSE_B:
        b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            b_d, add_offsets=[0, BLOCK_K], clamp_bounds=True
        )
    else:
        b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            b_d, add_offsets=[BLOCK_K, 0], clamp_bounds=True
        )
    return producer + 1, a_d, b_d


@gluon.constexpr_function
def tokens_per_step(num_ctas):
    """TDM tokens one K step leaves outstanding. A fused load retires as one, not two."""
    return 1 if num_ctas == 1 else 2


@gluon.jit
def _cluster_gate(wait_producers_cnt):
    """Drain this CTA's loads, then align the cluster so the multicast stays in step.

    The async_wait must not sit between arrive and wait: a CTA that arrives and then stalls on its
    load counter inside the barrier window can hang on hardware.
    """
    num_ctas: gl.constexpr = gl.num_ctas()
    gl.amd.cdna5.tdm.async_wait(wait_producers_cnt)
    if num_ctas > 1:
        gl.amd.cdna5.cluster.arrive()
        gl.amd.cdna5.cluster.wait()


@gluon.jit
def issue_wmma(
    consumer,
    a_buffer,
    a_layout: gl.constexpr,
    b_buffer,
    b_layout: gl.constexpr,
    accumulator,
    wait_producers_cnt,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    TRANSPOSE_A: gl.constexpr = False,
):
    """Consume one ring slot: wait, align the cluster, read LDS, then multiply."""
    _cluster_gate(wait_producers_cnt)

    slot = consumer % NUM_BUFFERS
    # Relaxed async LDS to register reads overlap with the matrix unit.
    # wmma wants A as [BM, BK]; a transposed ring holds [BK, BM], so permute it.
    if TRANSPOSE_A:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(slot).permute([1, 0]), a_layout
        )
    else:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffer.index(slot), a_layout)
    if TRANSPOSE_B:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).permute([1, 0]), b_layout
        )
    else:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buffer.index(slot), b_layout)

    return consumer + 1, gl.amd.cdna5.wmma(a, b, accumulator)


@gluon.jit
def issue_wmma_ktail(
    consumer,
    a_buffer,
    a_layout: gl.constexpr,
    b_buffer,
    b_layout: gl.constexpr,
    accumulator,
    wait_producers_cnt,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    KTAIL: gl.constexpr,
    TRANSPOSE_A: gl.constexpr = False,
):
    """Narrow form of issue_wmma for the last K step, when K does not fill BLOCK_K.

    The TDM descriptor clamps out-of-range K, so the tail slot is zero-filled and a full-width
    wmma throws away the fill. KTAIL is the real tail width, rounded up to the wmma K step, so
    only the useful part is multiplied. Only the wmma narrows; the load already happened.
    """
    _cluster_gate(wait_producers_cnt)

    slot = consumer % NUM_BUFFERS
    if TRANSPOSE_A:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(slot).slice(0, KTAIL, dim=0).permute([1, 0]), a_layout
        )
    else:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(slot).slice(0, KTAIL, dim=1), a_layout
        )
    if TRANSPOSE_B:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).slice(0, KTAIL, dim=1).permute([1, 0]), b_layout
        )
    else:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).slice(0, KTAIL, dim=0), b_layout
        )

    return consumer + 1, gl.amd.cdna5.wmma(a, b, accumulator)


@gluon.jit
def compute_tile(
    a_desc,
    b_desc,
    c_desc,
    c_ptr,
    off_m,
    off_n,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    OPERAND_A: gl.constexpr,
    OPERAND_B: gl.constexpr,
    WMMA: gl.constexpr,
    SHARED_C: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    TRANSPOSE_B: gl.constexpr,
    ACC_DTYPE: gl.constexpr,
    C_ALIAS: gl.constexpr,
    C_OFFSET_64: gl.constexpr,
    KTAIL: gl.constexpr,
    TOKENS: gl.constexpr,
    TRANSPOSE_A: gl.constexpr = False,
):
    """Compute one BLOCK_M x BLOCK_N tile over the whole K and write it to C.

    Phase 1 and the StreamK remainder both call this. They differ only in the tile size and the
    layouts, which arrive as constexprs.
    """
    # Allocate the rings here, not before the persistent tile loop. Every tile refills the
    # pipeline below, so the ring contents are never carried between tiles and a per-tile live
    # range is correct. It is also what lets the allocator place the C stage on the dead rings:
    # a ring allocated outside the loop is live in every iteration and interferes with C.
    a_buffer = gl.allocate_shared_memory(
        a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout
    )
    b_buffer = gl.allocate_shared_memory(
        b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout
    )
    producer = 0
    consumer = 0
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE, layout=WMMA)

    # Place the descriptors on this tile once. The K loop then only adds BLOCK_K.
    if TRANSPOSE_A:
        a_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, off_m], clamp_bounds=True
        )
    else:
        a_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[off_m, 0], clamp_bounds=True
        )
    if TRANSPOSE_B:
        b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[off_n, 0], clamp_bounds=True
        )
    else:
        b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, off_n], clamp_bounds=True
        )

    for _ in gl.static_range(NUM_BUFFERS - 1):
        producer, a_d, b_d = issue_loads(
            producer,
            a_d,
            b_d,
            a_buffer,
            b_buffer,
            BLOCK_K,
            TRANSPOSE_B,
            NUM_BUFFERS,
            TRANSPOSE_A,
        )

    # Steady state: one load issue and one consume per step keep the ring full.
    for _ in range(gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        producer, a_d, b_d = issue_loads(
            producer,
            a_d,
            b_d,
            a_buffer,
            b_buffer,
            BLOCK_K,
            TRANSPOSE_B,
            NUM_BUFFERS,
            TRANSPOSE_A,
        )
        consumer, acc = issue_wmma(
            consumer,
            a_buffer,
            OPERAND_A,
            b_buffer,
            OPERAND_B,
            acc,
            (NUM_BUFFERS - 1) * TOKENS,
            TRANSPOSE_B,
            NUM_BUFFERS,
            TRANSPOSE_A,
        )

    # KTAIL is a host constexpr decided from the real K, so only one drain path is compiled.
    if KTAIL:
        for i in gl.static_range(NUM_BUFFERS - 2):
            consumer, acc = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_A,
                b_buffer,
                OPERAND_B,
                acc,
                (NUM_BUFFERS - 2 - i) * TOKENS,
                TRANSPOSE_B,
                NUM_BUFFERS,
                TRANSPOSE_A,
            )
        consumer, acc = issue_wmma_ktail(
            consumer,
            a_buffer,
            OPERAND_A,
            b_buffer,
            OPERAND_B,
            acc,
            0,
            TRANSPOSE_B,
            NUM_BUFFERS,
            KTAIL,
            TRANSPOSE_A,
        )
    else:
        for i in gl.static_range(NUM_BUFFERS - 1):
            consumer, acc = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_A,
                b_buffer,
                OPERAND_B,
                acc,
                (NUM_BUFFERS - 2 - i) * TOKENS,
                TRANSPOSE_B,
                NUM_BUFFERS,
                TRANSPOSE_A,
            )

    if C_ALIAS:
        c_stage = gl.allocate_shared_memory(
            c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], SHARED_C
        )
        gl.barrier()  # every wmma read of a ring must retire before the C stage reuses it
        c_stage.store(acc.to(c_ptr.type.element_ty))
        gl.barrier()  # the LDS writes must be visible to the TDM engine
        gl.amd.cdna5.tdm.async_store(c_desc, [off_m, off_n], c_stage)
        # The TDM engine must finish reading LDS before the next tile refills the ring.
        gl.amd.cdna5.tdm.async_wait(0)
    else:
        rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA))
        cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA))
        mask_c = ((off_m + rows)[:, None] < M) & ((off_n + cols)[None, :] < N)
        if C_OFFSET_64:
            # C holds more than 2**31 elements, so an absolute index overflows int32 and drops
            # every store past it. Rebase to this tile in int64 and index inside it in int32.
            # The host gates this: 64-bit address maths costs up to 18% on this store, so a C
            # that cannot overflow must keep the int32 form.
            c_base = (
                c_ptr + off_m.to(gl.int64) * stride_cm + off_n.to(gl.int64) * stride_cn
            )
            offs_c = stride_cm * rows[:, None] + stride_cn * cols[None, :]
            gl.store(c_base + offs_c, acc.to(c_ptr.type.element_ty), mask=mask_c)
        else:
            offs_c = (
                stride_cm * (off_m + rows)[:, None]
                + stride_cn * (off_n + cols)[None, :]
            )
            gl.store(c_ptr + offs_c, acc.to(c_ptr.type.element_ty), mask=mask_c)
