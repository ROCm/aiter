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
    a_desc,
    b_desc,
    off_m,
    off_n,
    a_buffer,
    b_buffer,
    BLOCK_K: gl.constexpr,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """Start the TDM copies for one K step into the next ring slot.

    A single CTA has no multicast, so it issues both loads itself. One fused op then costs one TDM
    slot instead of two, which is worth 3% on the small-M buckets. A cluster shares the loads
    already, where fusing measures flat, so it keeps the two separate copies.
    """
    num_ctas: gl.constexpr = gl.num_ctas()
    slot = producer % NUM_BUFFERS
    off_k = producer * BLOCK_K
    if num_ctas == 1:
        # Split the warps between the two members; the hints must be disjoint bitmasks.
        half: gl.constexpr = gl.num_warps() // 2
        hint_a: gl.constexpr = (1 << half) - 1
        hint_b: gl.constexpr = ((1 << gl.num_warps()) - 1) ^ hint_a
        a_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[off_m, off_k]
        )
        if TRANSPOSE_B:
            b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[off_n, off_k]
            )
        else:
            b_d = gl.amd.cdna5.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[off_k, off_n]
            )
        gl.amd.cdna5.tdm.async_load_fused(
            [(a_d, a_buffer.index(slot), hint_a), (b_d, b_buffer.index(slot), hint_b)]
        )
    else:
        gl.amd.cdna5.tdm.async_load(a_desc, [off_m, off_k], a_buffer.index(slot))
        if TRANSPOSE_B:
            gl.amd.cdna5.tdm.async_load(b_desc, [off_n, off_k], b_buffer.index(slot))
        else:
            gl.amd.cdna5.tdm.async_load(b_desc, [off_k, off_n], b_buffer.index(slot))
    return producer + 1


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
):
    """Consume one ring slot: wait, align the cluster, read LDS, then multiply."""
    _cluster_gate(wait_producers_cnt)

    slot = consumer % NUM_BUFFERS
    # Relaxed async LDS to register reads overlap with the matrix unit.
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffer.index(slot), a_layout)
    if TRANSPOSE_B:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).permute([1, 0]), b_layout
        )
    else:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buffer.index(slot), b_layout)

    return consumer + 1, gl.amd.cdna5.wmma(a, b, accumulator)


@gluon.jit
def issue_wmma_khalf(
    consumer,
    a_buffer,
    a_layout: gl.constexpr,
    b_buffer,
    b_layout: gl.constexpr,
    accumulator,
    wait_producers_cnt,
    TRANSPOSE_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Half-K form of issue_wmma for the last K step when K leaves exactly BLOCK_K / 2.

    The TDM descriptor clamps out-of-range K, so the tail slot is zero-filled and a full-width wmma
    throws away half its work. Only the wmma is halved; the load already happened.
    """
    _cluster_gate(wait_producers_cnt)

    HALF: gl.constexpr = BLOCK_K // 2
    slot = consumer % NUM_BUFFERS
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(slot).slice(0, HALF, dim=1), a_layout
    )
    if TRANSPOSE_B:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).slice(0, HALF, dim=1).permute([1, 0]), b_layout
        )
    else:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(slot).slice(0, HALF, dim=0), b_layout
        )

    return consumer + 1, gl.amd.cdna5.wmma(a, b, accumulator)


@gluon.jit
def compute_tile(
    a_desc,
    b_desc,
    c_desc,
    a_buffer,
    b_buffer,
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
    KTAIL: gl.constexpr,
    TOKENS: gl.constexpr,
):
    """Compute one BLOCK_M x BLOCK_N tile over the whole K and write it to C.

    Phase 1 and the StreamK remainder both call this. They differ only in the tile size and the
    layouts, which arrive as constexprs.
    """
    producer = 0
    consumer = 0
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE, layout=WMMA)

    for _ in gl.static_range(NUM_BUFFERS - 1):
        producer = issue_loads(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            a_buffer,
            b_buffer,
            BLOCK_K,
            TRANSPOSE_B,
            NUM_BUFFERS,
        )

    # Steady state: one load issue and one consume per step keep the ring full.
    for _ in range(gl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
        producer = issue_loads(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            a_buffer,
            b_buffer,
            BLOCK_K,
            TRANSPOSE_B,
            NUM_BUFFERS,
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
            )
        consumer, acc = issue_wmma_khalf(
            consumer,
            a_buffer,
            OPERAND_A,
            b_buffer,
            OPERAND_B,
            acc,
            0,
            TRANSPOSE_B,
            NUM_BUFFERS,
            BLOCK_K,
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
            )

    if C_ALIAS:
        c_stage = b_buffer.reinterpret(
            c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], SHARED_C
        )
        gl.barrier()  # every wmma read of the B ring must retire before the overwrite
        c_stage.store(acc.to(c_ptr.type.element_ty))
        gl.barrier()  # the LDS writes must be visible to the TDM engine
        gl.amd.cdna5.tdm.async_store(c_desc, [off_m, off_n], c_stage)
        # The TDM engine must finish reading LDS before the next tile refills the ring.
        gl.amd.cdna5.tdm.async_wait(0)
    else:
        offs_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA))
        offs_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA))
        offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        gl.store(c_ptr + offs_c, acc.to(c_ptr.type.element_ty), mask=mask_c)
