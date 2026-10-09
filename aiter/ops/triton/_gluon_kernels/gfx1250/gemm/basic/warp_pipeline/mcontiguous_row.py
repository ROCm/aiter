# SPDX-License-Identifier: MIT
# Native K128 A-scale preshuffle with 64-bit LDS loads, derived from Triton CDNA5 examples.
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import tdm

from .helpers import chiplet_transform


@gluon.jit
def _get_xcd_swizzled_pids(
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = chiplet_transform(pid, GRID_MN, NUM_XCDS)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return (pid_m, pid_n)


@gluon.jit
def _cluster_sync():
    gl.amd.cdna5.cluster.arrive()
    gl.amd.cdna5.cluster.wait()


@gluon.jit
def _tdm_store_full_tile(
    c_ptr,
    pid_m,
    pid_n,
    stride_cm,
    stride_cn,
    M,
    N,
    acc,
    STORE_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CTA_N: gl.constexpr,
):
    c_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N // CTA_N, 8]], [BLOCK_M, BLOCK_N], [1, 0], STORE_LAYOUT.cga_layout
    )
    c_shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [BLOCK_M, BLOCK_N], c_shared_layout
    )
    c_shared.store(acc.to(c_ptr.type.element_ty))
    c_desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=c_shared_layout,
    )
    tdm.async_store(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_shared)
    tdm.async_wait(0)


@triton.constexpr_function
def _output_cga_layout(cga):
    return tuple(tuple(b) + (0, 0) for b in cga)


@triton.constexpr_function
def _strip_output_cga(cga):
    return tuple(tuple(b) + (0,) for b in cga)


@gluon.jit
def _tdm_store_row_grid(
    c_ptr,
    pid_m,
    pid_n,
    stride_cm,
    stride_cn,
    M,
    N,
    output,
    OUTPUT_CGA: gl.constexpr,
    CM: gl.constexpr,
    CN: gl.constexpr,
    cta_m: gl.constexpr,
    cta_n: gl.constexpr,
):
    if cta_m == 64:
        column_shape: gl.constexpr = [CM, CN, cta_m, cta_n // 32, 16]
        column_cga: gl.constexpr = _strip_output_cga(OUTPUT_CGA)
        column_layout: gl.constexpr = gl.SwizzledSharedLayout(
            1, 1, 1, [4, 3, 2, 1, 0], column_cga
        )
        c0 = gl.allocate_shared_memory(
            c_ptr.type.element_ty, column_shape, column_layout
        )
        c1 = gl.allocate_shared_memory(
            c_ptr.type.element_ty, column_shape, column_layout
        )
        column_output = output.reshape((CM, CN, cta_m, cta_n // 32, 2, 16))
        column_slice: gl.constexpr = [CM, CN, cta_m, cta_n // 32, 1, 16]
        o0 = gl.amd.slice(column_output, column_slice, [0, 0, 0, 0, 0, 0]).reshape(
            column_shape
        )
        o1 = gl.amd.slice(column_output, column_slice, [0, 0, 0, 0, 1, 0]).reshape(
            column_shape
        )
        desc_c = tdm.make_tensor_descriptor(
            base=c_ptr,
            shape=(M // cta_m, N // cta_n, cta_m, cta_n // 32, 32),
            strides=(
                cta_m * stride_cm,
                cta_n * stride_cn,
                stride_cm,
                32 * stride_cn,
                stride_cn,
            ),
            block_shape=column_shape,
            layout=column_layout,
        )
        c0.store(o0)
        tdm.async_store(desc_c, [pid_m * CM, pid_n * CN, 0, 0, 0], c0)
        c1.store(o1)
        tdm.async_store(desc_c, [pid_m * CM, pid_n * CN, 0, 0, 16], c1)
        tdm.async_wait(0)
        return
    else:
        STRIP: gl.constexpr = 16
        output_cga: gl.constexpr = _strip_output_cga(OUTPUT_CGA)
        shape: gl.constexpr = [CM, CN, cta_m // (2 * STRIP), STRIP, cta_n]
        layout: gl.constexpr = gl.SwizzledSharedLayout(
            1, 1, 1, [4, 3, 2, 1, 0], output_cga
        )
        shared0 = gl.allocate_shared_memory(c_ptr.type.element_ty, shape, layout)
        shared1 = gl.allocate_shared_memory(c_ptr.type.element_ty, shape, layout)
        output = output.reshape((CM, CN, cta_m // (2 * STRIP), 2, STRIP, cta_n))
        slice_shape: gl.constexpr = [CM, CN, cta_m // (2 * STRIP), 1, STRIP, cta_n]
        output0 = gl.amd.slice(output, slice_shape, [0, 0, 0, 0, 0, 0]).reshape(shape)
        output1 = gl.amd.slice(output, slice_shape, [0, 0, 0, 1, 0, 0]).reshape(shape)
        desc = tdm.make_tensor_descriptor(
            base=c_ptr,
            shape=(M // cta_m, N // cta_n, cta_m // (2 * STRIP), 2 * STRIP, cta_n),
            strides=(
                cta_m * stride_cm,
                cta_n * stride_cn,
                2 * STRIP * stride_cm,
                stride_cm,
                stride_cn,
            ),
            block_shape=shape,
            layout=layout,
        )
        shared0.store(output0)
        tdm.async_store(desc, [pid_m * CM, pid_n * CN, 0, 0, 0], shared0)
        shared1.store(output1)
        tdm.async_store(desc, [pid_m * CM, pid_n * CN, 0, STRIP, 0], shared1)
        tdm.async_wait(0)


@gluon.jit
def _tdm_store_tiled_two_buffers(
    c_ptr,
    pid_m,
    pid_n,
    stride_cm,
    stride_cn,
    M,
    N,
    acc,
    STORE_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    cta_m: gl.constexpr = BLOCK_M // CLUSTER
    cta_n: gl.constexpr = BLOCK_N // CLUSTER
    output = (
        acc.to(c_ptr.type.element_ty)
        .reshape((CLUSTER, cta_m, CLUSTER, cta_n))
        .permute((0, 2, 1, 3))
    )
    _tdm_store_row_grid(
        c_ptr,
        pid_m,
        pid_n,
        stride_cm,
        stride_cn,
        M,
        N,
        output,
        _output_cga_layout(STORE_LAYOUT.cga_layout),
        CLUSTER,
        CLUSTER,
        cta_m,
        cta_n,
    )


@gluon.jit
def _tdm_store_split_n2(
    c_ptr,
    pid_m,
    pid_n,
    stride_cm,
    stride_cn,
    M,
    N,
    acc0,
    acc1,
    OUTPUT_CGA_LAYOUT: gl.constexpr,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    TWO_BUFFERS: gl.constexpr = False,
):
    if TWO_BUFFERS:
        o0 = (
            acc0.to(c_ptr.type.element_ty)
            .reshape((CTA_M, 256, CTA_N, 128))
            .permute((0, 2, 1, 3))
        )
        o1 = (
            acc1.to(c_ptr.type.element_ty)
            .reshape((CTA_M, 256, CTA_N, 128))
            .permute((0, 2, 1, 3))
        )
        output = (
            gl.join(o0, o1).permute((0, 1, 2, 4, 3)).reshape((CTA_M, CTA_N, 256, 256))
        )
        _tdm_store_row_grid(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            output,
            OUTPUT_CGA_LAYOUT,
            CTA_M,
            CTA_N,
            256,
            256,
        )
        return
    cta_m: gl.constexpr = 256
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[half_n, 8]], [CTA_M, CTA_N, cta_m, half_n], [3, 2, 1, 0], OUTPUT_CGA_LAYOUT
    )
    shared = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout
    )
    if TWO_BUFFERS:
        shared1 = gl.allocate_shared_memory(
            c_ptr.type.element_ty, [CTA_M, CTA_N, cta_m, half_n], shared_layout
        )
    output0 = acc0.reshape((CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    output1 = acc1.reshape((CTA_M, cta_m, CTA_N, half_n)).permute((0, 2, 1, 3))
    desc = tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M // cta_m, N // cta_n, cta_m, cta_n),
        strides=(cta_m * stride_cm, cta_n * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, cta_m, half_n),
        layout=shared_layout,
    )
    base_m = pid_m * CTA_M
    base_n = pid_n * CTA_N
    shared.store(output0.to(c_ptr.type.element_ty))
    tdm.async_store(desc, [base_m, base_n, 0, 0], shared)
    if TWO_BUFFERS:
        shared1.store(output1.to(c_ptr.type.element_ty))
        tdm.async_store(desc, [base_m, base_n, 0, half_n], shared1)
    else:
        tdm.async_wait(0)
        shared.store(output1.to(c_ptr.type.element_ty))
        tdm.async_store(desc, [base_m, base_n, 0, half_n], shared)
    tdm.async_wait(0)


@triton.constexpr_function
def _native_scale_broadcast_layout(layout, row_repeat, rows):
    def split(bases):
        return [[b[0] // row_repeat, b[0] % row_repeat, b[1]] for b in bases]

    return gl.DistributedLinearLayout(
        split(layout.reg_bases),
        split(layout.lane_bases),
        split(layout.warp_bases),
        split(layout.block_bases),
        [rows, row_repeat, 4],
    )


@gluon.jit
def _load_mcontiguous_a(
    buf, slot, start_k: gl.constexpr, target_layout: gl.constexpr, rows: gl.constexpr
):
    value = (
        buf.index(slot)
        .slice(start_k // 4, 1, 0)
        .permute((1, 0))
        .load(layout=target_layout)
    )
    return value + gl.full((rows, 4), 0, gl.uint8, target_layout)


@triton.constexpr_function
def _transpose_scale_cga(layout):
    return tuple((b[1], b[0]) for b in layout.cga_layout)


@gluon.jit
def _mxfp8_load_native_scale(
    buf, slot, start_k: gl.constexpr, target_layout: gl.constexpr, rows: gl.constexpr
):
    mcontiguous_rows: gl.constexpr = buf.type.shape[1]
    row_repeat: gl.constexpr = rows // mcontiguous_rows
    expanded_layout: gl.constexpr = _native_scale_broadcast_layout(
        target_layout, row_repeat, mcontiguous_rows
    )
    load_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    value = buf.index(slot).slice(start_k // 4, 1, 1).load(layout=load_layout)
    value = value[:, None, :]
    value = value + gl.full(
        (mcontiguous_rows, row_repeat, 4), 0, gl.uint8, expanded_layout
    )
    return gl.convert_layout(
        value.reshape((rows, 4)), target_layout, assert_trivial=True
    )


@gluon.jit
def _mxfp8_native_scale_buffers(
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_as,
    stride_bs,
    pid_m,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BK: gl.constexpr,
    SLOTS: gl.constexpr,
    SHARED_AS: gl.constexpr,
    SHARED_BS: gl.constexpr,
):
    AS_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 1, [1, 0], _transpose_scale_cga(SHARED_AS)
    )
    as_buf = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty, [SLOTS, BK // 128, BLOCK_M], AS_LAYOUT
    )
    bs_buf = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty, [SLOTS, BLOCK_N // 128, BK // 128], SHARED_BS
    )
    as_desc = tdm.make_tensor_descriptor(
        base=a_scale_ptr + pid_m * BLOCK_M,
        shape=(K // 128, M),
        strides=(stride_as, 1),
        block_shape=(BK // 128, BLOCK_M),
        layout=AS_LAYOUT,
    )
    bs_desc = tdm.make_tensor_descriptor(
        base=b_scale_ptr + pid_n * (BLOCK_N // 128) * stride_bs,
        shape=(N // 128, K // 128),
        strides=(stride_bs, 1),
        block_shape=(BLOCK_N // 128, BK // 128),
        layout=SHARED_BS,
    )
    return as_buf, bs_buf, as_desc, bs_desc


@gluon.jit
def _mxfp8_load_scale(
    scale_buffer,
    slot,
    start_k: gl.constexpr,
    LAYOUT: gl.constexpr,
    BLOCK_NONK: gl.constexpr,
    BK_SCALE: gl.constexpr,
    SUBTILE_SCALE_K: gl.constexpr,
):
    if (
        scale_buffer.type.shape[1] == BK_SCALE // 4
        and scale_buffer.type.shape[-1] == BLOCK_NONK
    ):
        return _load_mcontiguous_a(scale_buffer, slot, start_k, LAYOUT, BLOCK_NONK)
    elif scale_buffer.type.shape[-1] == 2:
        return _mxfp8_load_native_scale(scale_buffer, slot, start_k, LAYOUT, BLOCK_NONK)
    else:
        scale_slice = (
            scale_buffer.index(slot)
            .reshape((BLOCK_NONK // 128, BK_SCALE // 4, 32, 4, 4))
            .permute((0, 3, 2, 1, 4))
            .reshape((BLOCK_NONK, BK_SCALE))
        )
        return (
            scale_slice.slice(0, BLOCK_NONK, 0)
            .slice(start_k, SUBTILE_SCALE_K, 1)
            .load(layout=LAYOUT)
        )


@gluon.jit
def _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=15)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=15)


@gluon.jit
def _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, slot):
    _mxfp8_issue_b2_parent_data_without_update(a_desc, b_desc, a_buf, b_buf, slot)
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
    return (a_desc, b_desc)


@gluon.jit
def _mxfp8_load_b2_parent(
    buffer,
    slot,
    start_n: gl.constexpr,
    start_k: gl.constexpr,
    B2_LOAD_LAYOUT: gl.constexpr,
    DOT_B: gl.constexpr,
):
    rank3 = (
        buffer.index(slot)
        .slice(start_n, 128, 1)
        .slice(start_k, 128, 2)
        .permute([2, 0, 1])
        .load(layout=B2_LOAD_LAYOUT)
    )
    rank2 = rank3.reshape((128, buffer.type.shape[1] * 128))
    return gl.convert_layout(rank2, DOT_B)


@gluon.jit
def _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, slot):
    tdm.async_load_fused(
        [(as_desc, as_buf.index(slot), 3), (bs_desc, bs_buf.index(slot), 12)]
    )
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[as_buf.type.shape[1], 0]
    )
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, bs_buf.type.shape[-1]]
    )
    return (as_desc, bs_desc)


@gluon.jit
def _mxfp8_bk256_b2_consume_and_refill(
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    data_slot,
    scale_slot,
    refill_scale_slot,
    a_desc,
    b_desc,
    as_desc,
    bs_desc,
    acc0,
    acc1,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    B2_LOAD_LAYOUT: gl.constexpr,
    SCALE_A_LAYOUT: gl.constexpr,
    SCALE_A_LOAD: gl.constexpr,
    SCALE_B_LAYOUT: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CTA_N: gl.constexpr,
    REFILL_SCALE: gl.constexpr,
    DATA_SCALE_WAITCNT: gl.constexpr,
):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    tdm.async_wait(DATA_SCALE_WAITCNT)
    next_a_desc = a_desc
    next_b_desc = b_desc
    next_as_desc = as_desc
    next_bs_desc = bs_desc
    with gl.amd.warp_pipeline_stage("mxfp8_stage0_load_low", phase_gap=1):
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        a0 = (
            a_buf.index(data_slot)
            .slice(0, BLOCK_M, 0)
            .slice(0, 128, 1)
            .load(layout=DOT_A)
        )
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage("mxfp8_stage1_compute_low"):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("mxfp8_stage2_load_high_advance"):
        a1 = (
            a_buf.index(data_slot)
            .slice(0, BLOCK_M, 0)
            .slice(128, 128, 1)
            .load(layout=DOT_A)
        )
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
        gl.amd.cdna5.cluster.arrive()
        next_a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
        if REFILL_SCALE:
            next_as_desc = tdm.update_tensor_descriptor(
                as_desc, add_offsets=[as_buf.type.shape[1], 0]
            )
            next_bs_desc = tdm.update_tensor_descriptor(
                bs_desc, add_offsets=[0, bs_buf.type.shape[-1]]
            )
    with gl.amd.warp_pipeline_stage("mxfp8_stage3_compute_high"):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("mxfp8_stage4_wait_refill"):
        gl.amd.cdna5.cluster.wait()
        _mxfp8_issue_b2_parent_data_without_update(
            a_desc, b_desc, a_buf, b_buf, data_slot
        )
        if REFILL_SCALE:
            tdm.async_load_fused(
                [
                    (as_desc, as_buf.index(refill_scale_slot), 3),
                    (bs_desc, bs_buf.index(refill_scale_slot), 12),
                ]
            )
        a_desc = next_a_desc
        b_desc = next_b_desc
        as_desc = next_as_desc
        bs_desc = next_bs_desc
    return (a_desc, b_desc, as_desc, bs_desc, acc0, acc1)


@gluon.jit
def _mxfp8_bk256_b2_consume_tail(
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    data_slot,
    scale_slot,
    acc0,
    acc1,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    B2_LOAD_LAYOUT: gl.constexpr,
    SCALE_A_LAYOUT: gl.constexpr,
    SCALE_A_LOAD: gl.constexpr,
    SCALE_B_LAYOUT: gl.constexpr,
    SCALE_B_LOAD: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CTA_N: gl.constexpr,
):
    cta_n: gl.constexpr = 256
    half_n: gl.constexpr = 128
    packed_n: gl.constexpr = BLOCK_N // 2
    with gl.amd.warp_pipeline_stage("b2_tail_load_low"):
        a0 = (
            a_buf.index(data_slot)
            .slice(0, BLOCK_M, 0)
            .slice(0, 128, 1)
            .load(layout=DOT_A)
        )
        as0 = _mxfp8_load_scale(as_buf, scale_slot, 0, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as0 = gl.convert_layout(as0, SCALE_A_LAYOUT, assert_trivial=True)
        bs0 = _mxfp8_load_scale(bs_buf, scale_slot, 0, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs0_grid = bs0.reshape((CTA_N, cta_n, 4))
        bs0_lo = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        bs0_hi = gl.convert_layout(
            gl.amd.slice(bs0_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        b0_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 0, B2_LOAD_LAYOUT, DOT_B)
        b0_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 0, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage("b2_tail_compute_low"):
        acc0 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_lo, bs0_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0_hi, bs0_hi, "e4m3", acc1)
    with gl.amd.warp_pipeline_stage("b2_tail_load_high"):
        a1 = (
            a_buf.index(data_slot)
            .slice(0, BLOCK_M, 0)
            .slice(128, 128, 1)
            .load(layout=DOT_A)
        )
        as1 = _mxfp8_load_scale(as_buf, scale_slot, 4, SCALE_A_LOAD, BLOCK_M, 8, 4)
        as1 = gl.convert_layout(as1, SCALE_A_LAYOUT, assert_trivial=True)
        bs1 = _mxfp8_load_scale(bs_buf, scale_slot, 4, SCALE_B_LOAD, BLOCK_N, 8, 4)
        bs1_grid = bs1.reshape((CTA_N, cta_n, 4))
        bs1_lo = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, 0, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        bs1_hi = gl.convert_layout(
            gl.amd.slice(bs1_grid, [CTA_N, half_n, 4], [0, half_n, 0]).reshape(
                (packed_n, 4)
            ),
            SCALE_B_LAYOUT,
            assert_trivial=True,
        )
        b1_lo = _mxfp8_load_b2_parent(b_buf, data_slot, 0, 128, B2_LOAD_LAYOUT, DOT_B)
        b1_hi = _mxfp8_load_b2_parent(b_buf, data_slot, 128, 128, B2_LOAD_LAYOUT, DOT_B)
    with gl.amd.warp_pipeline_stage("b2_tail_compute_high"):
        acc0 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_lo, bs1_lo, "e4m3", acc0)
        acc1 = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1_hi, bs1_hi, "e4m3", acc1)
    return (acc0, acc1)


@gluon.jit
def mxfp8_gemm_warp_pipeline(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_a,
    stride_scale_b,
    GRID_MN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B2: gl.constexpr,
    SHARED_SCALE_A: gl.constexpr,
    SHARED_SCALE_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    LOAD_WMMA_LAYOUT: gl.constexpr,
    B2_LOAD_LAYOUT: gl.constexpr,
    OUTPUT_CGA_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4]
    )
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2
    )
    scale_slots: gl.constexpr = 3
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A,
    )
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_desc = tdm.make_tensor_descriptor(
        base=b_base,
        shape=(CTA_N, N // CTA_N, K),
        strides=(256 * stride_bn, stride_bn, stride_bk),
        block_shape=(CTA_N, 256, 256),
        layout=SHARED_LAYOUT_B2,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_scale_a,
            stride_scale_b,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            256,
            scale_slots,
            SHARED_SCALE_A,
            SHARED_SCALE_B,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_M // 128, 1024],
            SHARED_SCALE_A,
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_N // 128, 1024],
            SHARED_SCALE_B,
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M // 128 * stride_scale_a,
            shape=(M // 128, K // 32 * 128),
            strides=(stride_scale_a, 1),
            block_shape=(BLOCK_M // 128, 1024),
            layout=SHARED_SCALE_A,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale_b,
            shape=(N // 128, K // 32 * 128),
            strides=(stride_scale_b, 1),
            block_shape=(BLOCK_N // 128, 1024),
            layout=SHARED_SCALE_B,
        )
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    iter_max = gl.cdiv(K, 256)
    gl.assume(iter_max >= 4)
    gl.assume(iter_max % 2 == 0)
    data_scale_waitcnt: gl.constexpr = 4
    if NATIVE_SCALE128:
        # Six tiles cover both data-slot (2) and scale-slot (3) cycles.
        # Keep scale loads local to each memory stage instead of carrying pairs.
        for _ in range((iter_max - 4) // 6):
            for inner in gl.static_range(6):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    _mxfp8_bk256_b2_consume_and_refill(
                        a_buf,
                        b_buf,
                        as_buf,
                        bs_buf,
                        inner % 2,
                        inner % 3,
                        inner % 3,
                        a_desc,
                        b_desc,
                        as_desc,
                        bs_desc,
                        acc0,
                        acc1,
                        dot_a,
                        dot_b,
                        B2_LOAD_LAYOUT,
                        scale_a_layout,
                        scale_a_layout,
                        scale_b_layout,
                        scale_b_load,
                        BLOCK_M,
                        BLOCK_N,
                        CTA_N,
                        True,
                        data_scale_waitcnt,
                    )
                )
        for inner in gl.static_range(6):
            if inner < (iter_max - 4) % 6:
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    _mxfp8_bk256_b2_consume_and_refill(
                        a_buf,
                        b_buf,
                        as_buf,
                        bs_buf,
                        inner % 2,
                        inner % 3,
                        inner % 3,
                        a_desc,
                        b_desc,
                        as_desc,
                        bs_desc,
                        acc0,
                        acc1,
                        dot_a,
                        dot_b,
                        B2_LOAD_LAYOUT,
                        scale_a_layout,
                        scale_a_layout,
                        scale_b_layout,
                        scale_b_load,
                        BLOCK_M,
                        BLOCK_N,
                        CTA_N,
                        True,
                        data_scale_waitcnt,
                    )
                )
    else:
        for _ in range((iter_max - 4) // 6):
            for inner in gl.static_range(6):
                a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                    _mxfp8_bk256_b2_consume_and_refill(
                        a_buf,
                        b_buf,
                        as_buf,
                        bs_buf,
                        inner % 2,
                        inner % 3,
                        inner % 3,
                        a_desc,
                        b_desc,
                        as_desc,
                        bs_desc,
                        acc0,
                        acc1,
                        dot_a,
                        dot_b,
                        B2_LOAD_LAYOUT,
                        scale_a_layout,
                        scale_a_layout,
                        scale_b_layout,
                        scale_b_load,
                        BLOCK_M,
                        BLOCK_N,
                        CTA_N,
                        True,
                        data_scale_waitcnt,
                    )
                )
        for cleanup_pair in range((iter_max - 4) % 6 // 2):
            cleanup_tile = cleanup_pair * 2
            s0 = cleanup_tile % 3
            s1 = (cleanup_tile + 1) % 3
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                _mxfp8_bk256_b2_consume_and_refill(
                    a_buf,
                    b_buf,
                    as_buf,
                    bs_buf,
                    0,
                    s0,
                    s0,
                    a_desc,
                    b_desc,
                    as_desc,
                    bs_desc,
                    acc0,
                    acc1,
                    dot_a,
                    dot_b,
                    B2_LOAD_LAYOUT,
                    scale_a_layout,
                    scale_a_layout,
                    scale_b_layout,
                    scale_b_load,
                    BLOCK_M,
                    BLOCK_N,
                    CTA_N,
                    True,
                    data_scale_waitcnt,
                )
            )
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
                _mxfp8_bk256_b2_consume_and_refill(
                    a_buf,
                    b_buf,
                    as_buf,
                    bs_buf,
                    1,
                    s1,
                    s1,
                    a_desc,
                    b_desc,
                    as_desc,
                    bs_desc,
                    acc0,
                    acc1,
                    dot_a,
                    dot_b,
                    B2_LOAD_LAYOUT,
                    scale_a_layout,
                    scale_a_layout,
                    scale_b_layout,
                    scale_b_load,
                    BLOCK_M,
                    BLOCK_N,
                    CTA_N,
                    True,
                    data_scale_waitcnt,
                )
            )
    residual = iter_max - 4
    s0 = residual % 3
    s1 = (residual + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        0,
        s0,
        s0,
        a_desc,
        b_desc,
        as_desc,
        bs_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
        True,
        data_scale_waitcnt,
    )
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        1,
        s1,
        s1,
        a_desc,
        b_desc,
        as_desc,
        bs_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
        False,
        data_scale_waitcnt,
    )
    tdm.async_wait(3)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        0,
        (iter_max - 2) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
    )
    tdm.async_wait(0)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        1,
        (iter_max - 1) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_split_n2(
        c_ptr,
        pid_m,
        pid_n,
        stride_cm,
        stride_cn,
        M,
        N,
        acc0,
        acc1,
        OUTPUT_CGA_LAYOUT,
        CTA_M,
        CTA_N,
        TWO_BUFFER_OUTPUT,
    )


@gluon.jit
def mxfp8_gemm_warp_pipeline_flat(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale_a,
    stride_scale_b,
    GRID_MN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B2: gl.constexpr,
    SHARED_SCALE_A: gl.constexpr,
    SHARED_SCALE_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    LOAD_WMMA_LAYOUT: gl.constexpr,
    B2_LOAD_LAYOUT: gl.constexpr,
    OUTPUT_CGA_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 16)
    dot_b_load: gl.constexpr = gl.DotOperandLayout(1, LOAD_WMMA_LAYOUT, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N // 2, 4]
    )
    scale_b_load: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b_load, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 256], SHARED_LAYOUT_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, CTA_N, 256, 256], SHARED_LAYOUT_B2
    )
    scale_slots: gl.constexpr = 3
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 256),
        layout=SHARED_LAYOUT_A,
    )
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_desc = tdm.make_tensor_descriptor(
        base=b_base,
        shape=(CTA_N, N // CTA_N, K),
        strides=(256 * stride_bn, stride_bn, stride_bk),
        block_shape=(CTA_N, 256, 256),
        layout=SHARED_LAYOUT_B2,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_scale_a,
            stride_scale_b,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            256,
            scale_slots,
            SHARED_SCALE_A,
            SHARED_SCALE_B,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_M // 128, 1024],
            SHARED_SCALE_A,
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [scale_slots, BLOCK_N // 128, 1024],
            SHARED_SCALE_B,
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M // 128 * stride_scale_a,
            shape=(M // 128, K // 32 * 128),
            strides=(stride_scale_a, 1),
            block_shape=(BLOCK_M // 128, 1024),
            layout=SHARED_SCALE_A,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N // 128 * stride_scale_b,
            shape=(N // 128, K // 32 * 128),
            strides=(stride_scale_b, 1),
            block_shape=(BLOCK_N // 128, 1024),
            layout=SHARED_SCALE_B,
        )
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 0)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 0)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 1)
    a_desc, b_desc = _mxfp8_issue_b2_parent_data(a_desc, b_desc, a_buf, b_buf, 1)
    as_desc, bs_desc = _mxfp8_issue_leading4_scale(as_desc, bs_desc, as_buf, bs_buf, 2)
    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), dtype=gl.float32, layout=WMMA_LAYOUT)
    # Expand every K tile before warp-pipeline lowering, including the drain below.
    iter_max: gl.constexpr = K // 256
    gl.static_assert(iter_max >= 4 and iter_max % 2 == 0)
    data_scale_waitcnt: gl.constexpr = 4
    for tile in gl.static_range(iter_max - 4):
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = (
            _mxfp8_bk256_b2_consume_and_refill(
                a_buf,
                b_buf,
                as_buf,
                bs_buf,
                tile % 2,
                tile % 3,
                tile % 3,
                a_desc,
                b_desc,
                as_desc,
                bs_desc,
                acc0,
                acc1,
                dot_a,
                dot_b,
                B2_LOAD_LAYOUT,
                scale_a_layout,
                scale_a_layout,
                scale_b_layout,
                scale_b_load,
                BLOCK_M,
                BLOCK_N,
                CTA_N,
                True,
                data_scale_waitcnt,
            )
        )
    residual = iter_max - 4
    s0 = residual % 3
    s1 = (residual + 1) % 3
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        0,
        s0,
        s0,
        a_desc,
        b_desc,
        as_desc,
        bs_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
        True,
        data_scale_waitcnt,
    )
    a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _mxfp8_bk256_b2_consume_and_refill(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        1,
        s1,
        s1,
        a_desc,
        b_desc,
        as_desc,
        bs_desc,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
        False,
        data_scale_waitcnt,
    )
    tdm.async_wait(3)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        0,
        (iter_max - 2) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
    )
    tdm.async_wait(0)
    acc0, acc1 = _mxfp8_bk256_b2_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        1,
        (iter_max - 1) % 3,
        acc0,
        acc1,
        dot_a,
        dot_b,
        B2_LOAD_LAYOUT,
        scale_a_layout,
        scale_a_layout,
        scale_b_layout,
        scale_b_load,
        BLOCK_M,
        BLOCK_N,
        CTA_N,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    _tdm_store_split_n2(
        c_ptr,
        pid_m,
        pid_n,
        stride_cm,
        stride_cn,
        M,
        N,
        acc0,
        acc1,
        OUTPUT_CGA_LAYOUT,
        CTA_M,
        CTA_N,
        TWO_BUFFER_OUTPUT,
    )


@gluon.jit
def _mxfp8_tiled_issue_refill(
    a_desc,
    b_desc,
    as_desc,
    bs_desc,
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    slot,
    BLOCK_K: gl.constexpr,
    SCALE_STEP: gl.constexpr,
):
    tdm.async_load(a_desc, dest=a_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load(b_desc, dest=b_buf.index(slot), warp_used_hint=0b00001111)
    tdm.async_load_fused(
        [
            (as_desc, as_buf.index(slot), 0b00000011),
            (bs_desc, bs_buf.index(slot), 0b00001100),
        ]
    )
    a_desc = tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    b_desc = tdm.update_tensor_descriptor(b_desc, add_offsets=[0, BLOCK_K])
    as_desc = tdm.update_tensor_descriptor(
        as_desc, add_offsets=[as_buf.type.shape[1], 0]
    )
    bs_desc = tdm.update_tensor_descriptor(
        bs_desc, add_offsets=[0, bs_buf.type.shape[-1]]
    )
    return a_desc, b_desc, as_desc, bs_desc


@gluon.jit
def _mxfp8_tiled_load_scale(
    scale_buf, slot, start_k: gl.constexpr, layout: gl.constexpr, rows: gl.constexpr
):
    if scale_buf.type.shape[-1] >= 256:
        return _load_mcontiguous_a(scale_buf, slot, start_k, layout, rows)
    elif scale_buf.type.shape[-1] <= 8:
        return _mxfp8_load_native_scale(scale_buf, slot, start_k, layout, rows)
    else:
        return scale_buf.index(slot).slice(start_k, 4, 1).load(layout=layout)


@gluon.jit
def _mxfp8_tiled_consume_tail(
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    slot,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    scale_a_layout: gl.constexpr,
    scale_b_layout: gl.constexpr,
    block_k: gl.constexpr,
):
    for k in gl.static_range(0, block_k, 128):
        with gl.amd.warp_pipeline_stage("tiled_tail_load"):
            a = a_buf.index(slot).slice(k, 128, 1).load(layout=dot_a)
            a_scale = _mxfp8_tiled_load_scale(
                as_buf, slot, k // 32, scale_a_layout, a_buf.type.shape[1]
            )
            b = b_buf.index(slot).slice(k, 128, 1).permute([1, 0]).load(layout=dot_b)
            b_scale = _mxfp8_tiled_load_scale(
                bs_buf, slot, k // 32, scale_b_layout, b_buf.type.shape[1]
            )
        with gl.amd.warp_pipeline_stage("tiled_tail_compute"):
            acc = gl.amd.cdna5.wmma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
    return acc


@gluon.jit
def _mxfp8_tiled_bk512_five(
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    slot,
    a_desc,
    b_desc,
    as_desc,
    bs_desc,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    scale_a_layout: gl.constexpr,
    scale_b_layout: gl.constexpr,
):
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage("tiled_stage0_load_k0", phase_gap=1):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = _mxfp8_tiled_load_scale(
            as_buf, slot, 0, scale_a_layout, a_buf.type.shape[1]
        )
        bs0 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 0, scale_b_layout, b_buf.type.shape[1]
        )
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = _mxfp8_tiled_load_scale(
            as_buf, slot, 4, scale_a_layout, a_buf.type.shape[1]
        )
        bs1 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 4, scale_b_layout, b_buf.type.shape[1]
        )
        b0 = b_buf.index(slot).slice(0, 128, 1).permute([1, 0]).load(layout=dot_b)
        b1 = b_buf.index(slot).slice(128, 128, 1).permute([1, 0]).load(layout=dot_b)
    with gl.amd.warp_pipeline_stage("tiled_stage1_compute_k0"):
        acc = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0, bs0, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tiled_stage2_load_k2"):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
        as2 = _mxfp8_tiled_load_scale(
            as_buf, slot, 8, scale_a_layout, a_buf.type.shape[1]
        )
        b2 = b_buf.index(slot).slice(256, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs2 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 8, scale_b_layout, b_buf.type.shape[1]
        )
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
        as3 = _mxfp8_tiled_load_scale(
            as_buf, slot, 12, scale_a_layout, a_buf.type.shape[1]
        )
        b3 = b_buf.index(slot).slice(384, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs3 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 12, scale_b_layout, b_buf.type.shape[1]
        )
        gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("tiled_stage3_compute_k2"):
        acc = gl.amd.cdna5.wmma_scaled(a2, as2, "e4m3", b2, bs2, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("tiled_stage4_wait_refill"):
        gl.amd.cdna5.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            512,
            16,
        )
    return (a_desc, b_desc, as_desc, bs_desc, acc)


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk512(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as,
    stride_bs,
    GRID_MN: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    SHARED_AS: gl.constexpr,
    SHARED_BS: gl.constexpr,
    WMMA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CLUSTER: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 512], SHARED_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_N, 512], SHARED_B
    )
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 512),
        layout=SHARED_A,
    )
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, 512),
        layout=SHARED_B,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_as,
            stride_bs,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            512,
            2,
            SHARED_AS,
            SHARED_BS,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [2, BLOCK_M, 16], SHARED_AS
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [2, BLOCK_N, 16], SHARED_BS
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * stride_as,
            shape=(M, K // 32),
            strides=(stride_as, 1),
            block_shape=(BLOCK_M, 16),
            layout=SHARED_AS,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_bs,
            shape=(N, K // 32),
            strides=(stride_bs, 1),
            block_shape=(BLOCK_N, 16),
            layout=SHARED_BS,
        )
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            512,
            16,
        )
    iter_max = gl.cdiv(K, 512)
    gl.assume(iter_max >= 3)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for _ in range((iter_max - 2) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(
                a_buf,
                b_buf,
                as_buf,
                bs_buf,
                inner % 2,
                a_desc,
                b_desc,
                as_desc,
                bs_desc,
                acc,
                dot_a,
                dot_b,
                scale_a_layout,
                scale_b_layout,
            )
    for tile in range((iter_max - 2) // 6 * 6, iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            tile % 2,
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            acc,
            dot_a,
            dot_b,
            scale_a_layout,
            scale_b_layout,
        )
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 2) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        512,
    )
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 1) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        512,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if TWO_BUFFER_OUTPUT:
        _tdm_store_tiled_two_buffers(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )
    else:
        _tdm_store_full_tile(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk512_flat(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as,
    stride_bs,
    GRID_MN: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    SHARED_AS: gl.constexpr,
    SHARED_BS: gl.constexpr,
    WMMA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CLUSTER: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 512], SHARED_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_N, 512], SHARED_B
    )
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 512),
        layout=SHARED_A,
    )
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, 512),
        layout=SHARED_B,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_as,
            stride_bs,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            512,
            2,
            SHARED_AS,
            SHARED_BS,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [2, BLOCK_M, 16], SHARED_AS
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [2, BLOCK_N, 16], SHARED_BS
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * stride_as,
            shape=(M, K // 32),
            strides=(stride_as, 1),
            block_shape=(BLOCK_M, 16),
            layout=SHARED_AS,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_bs,
            shape=(N, K // 32),
            strides=(stride_bs, 1),
            block_shape=(BLOCK_N, 16),
            layout=SHARED_BS,
        )
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            512,
            16,
        )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    iter_max: gl.constexpr = K // 512
    gl.static_assert(K % 512 == 0 and iter_max >= 3)
    for tile in gl.static_range(iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk512_five(
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            tile % 2,
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            acc,
            dot_a,
            dot_b,
            scale_a_layout,
            scale_b_layout,
        )
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 2) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        512,
    )
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 1) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        512,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if TWO_BUFFER_OUTPUT:
        _tdm_store_tiled_two_buffers(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )
    else:
        _tdm_store_full_tile(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )


@gluon.jit
def _mxfp8_tiled_bk1024_nine(
    a_buf,
    b_buf,
    as_buf,
    bs_buf,
    slot,
    a_desc,
    b_desc,
    as_desc,
    bs_desc,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    scale_a_layout: gl.constexpr,
    scale_b_layout: gl.constexpr,
):
    # A tile has eight K128 products. Read every slice before reusing its slot.
    # The two-slot pipeline retains the same three async operations per refill
    # as BK512: A, B, and one fused A/B scale load.
    tdm.async_wait(3)
    with gl.amd.warp_pipeline_stage("bk1024_stage0_load_k0", phase_gap=1):
        a0 = a_buf.index(slot).slice(0, 128, 1).load(layout=dot_a)
        as0 = _mxfp8_tiled_load_scale(
            as_buf, slot, 0, scale_a_layout, a_buf.type.shape[1]
        )
        b0 = b_buf.index(slot).slice(0, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs0 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 0, scale_b_layout, b_buf.type.shape[1]
        )
        a1 = a_buf.index(slot).slice(128, 128, 1).load(layout=dot_a)
        as1 = _mxfp8_tiled_load_scale(
            as_buf, slot, 4, scale_a_layout, a_buf.type.shape[1]
        )
        b1 = b_buf.index(slot).slice(128, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs1 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 4, scale_b_layout, b_buf.type.shape[1]
        )
    with gl.amd.warp_pipeline_stage("bk1024_stage1_compute_k0"):
        acc = gl.amd.cdna5.wmma_scaled(a0, as0, "e4m3", b0, bs0, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a1, as1, "e4m3", b1, bs1, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk1024_stage2_load_k2"):
        a2 = a_buf.index(slot).slice(256, 128, 1).load(layout=dot_a)
        as2 = _mxfp8_tiled_load_scale(
            as_buf, slot, 8, scale_a_layout, a_buf.type.shape[1]
        )
        b2 = b_buf.index(slot).slice(256, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs2 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 8, scale_b_layout, b_buf.type.shape[1]
        )
        a3 = a_buf.index(slot).slice(384, 128, 1).load(layout=dot_a)
        as3 = _mxfp8_tiled_load_scale(
            as_buf, slot, 12, scale_a_layout, a_buf.type.shape[1]
        )
        b3 = b_buf.index(slot).slice(384, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs3 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 12, scale_b_layout, b_buf.type.shape[1]
        )
    with gl.amd.warp_pipeline_stage("bk1024_stage3_compute_k2"):
        acc = gl.amd.cdna5.wmma_scaled(a2, as2, "e4m3", b2, bs2, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a3, as3, "e4m3", b3, bs3, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk1024_stage4_load_k4"):
        a4 = a_buf.index(slot).slice(512, 128, 1).load(layout=dot_a)
        as4 = _mxfp8_tiled_load_scale(
            as_buf, slot, 16, scale_a_layout, a_buf.type.shape[1]
        )
        b4 = b_buf.index(slot).slice(512, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs4 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 16, scale_b_layout, b_buf.type.shape[1]
        )
        a5 = a_buf.index(slot).slice(640, 128, 1).load(layout=dot_a)
        as5 = _mxfp8_tiled_load_scale(
            as_buf, slot, 20, scale_a_layout, a_buf.type.shape[1]
        )
        b5 = b_buf.index(slot).slice(640, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs5 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 20, scale_b_layout, b_buf.type.shape[1]
        )
    with gl.amd.warp_pipeline_stage("bk1024_stage5_compute_k4"):
        acc = gl.amd.cdna5.wmma_scaled(a4, as4, "e4m3", b4, bs4, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a5, as5, "e4m3", b5, bs5, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk1024_stage6_load_k6"):
        a6 = a_buf.index(slot).slice(768, 128, 1).load(layout=dot_a)
        as6 = _mxfp8_tiled_load_scale(
            as_buf, slot, 24, scale_a_layout, a_buf.type.shape[1]
        )
        b6 = b_buf.index(slot).slice(768, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs6 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 24, scale_b_layout, b_buf.type.shape[1]
        )
        a7 = a_buf.index(slot).slice(896, 128, 1).load(layout=dot_a)
        as7 = _mxfp8_tiled_load_scale(
            as_buf, slot, 28, scale_a_layout, a_buf.type.shape[1]
        )
        b7 = b_buf.index(slot).slice(896, 128, 1).permute([1, 0]).load(layout=dot_b)
        bs7 = _mxfp8_tiled_load_scale(
            bs_buf, slot, 28, scale_b_layout, b_buf.type.shape[1]
        )
        gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("bk1024_stage7_compute_k6"):
        acc = gl.amd.cdna5.wmma_scaled(a6, as6, "e4m3", b6, bs6, "e4m3", acc)
        acc = gl.amd.cdna5.wmma_scaled(a7, as7, "e4m3", b7, bs7, "e4m3", acc)
    with gl.amd.warp_pipeline_stage("bk1024_stage8_wait_refill"):
        gl.amd.cdna5.cluster.wait()
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            1024,
            32,
        )
    return (a_desc, b_desc, as_desc, bs_desc, acc)


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk1024(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as,
    stride_bs,
    GRID_MN: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    SHARED_AS: gl.constexpr,
    SHARED_BS: gl.constexpr,
    WMMA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CLUSTER: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 1024], SHARED_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_N, 1024], SHARED_B
    )
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 1024),
        layout=SHARED_A,
    )
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, 1024),
        layout=SHARED_B,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_as,
            stride_bs,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            1024,
            2,
            SHARED_AS,
            SHARED_BS,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [2, BLOCK_M, 32], SHARED_AS
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [2, BLOCK_N, 32], SHARED_BS
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * stride_as,
            shape=(M, K // 32),
            strides=(stride_as, 1),
            block_shape=(BLOCK_M, 32),
            layout=SHARED_AS,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_bs,
            shape=(N, K // 32),
            strides=(stride_bs, 1),
            block_shape=(BLOCK_N, 32),
            layout=SHARED_BS,
        )
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            1024,
            32,
        )
    iter_max = gl.cdiv(K, 1024)
    gl.assume(iter_max >= 3)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for _ in range((iter_max - 2) // 6):
        for inner in gl.static_range(6):
            a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk1024_nine(
                a_buf,
                b_buf,
                as_buf,
                bs_buf,
                inner % 2,
                a_desc,
                b_desc,
                as_desc,
                bs_desc,
                acc,
                dot_a,
                dot_b,
                scale_a_layout,
                scale_b_layout,
            )
    for tile in range((iter_max - 2) // 6 * 6, iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk1024_nine(
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            tile % 2,
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            acc,
            dot_a,
            dot_b,
            scale_a_layout,
            scale_b_layout,
        )
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 2) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        1024,
    )
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 1) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        1024,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if TWO_BUFFER_OUTPUT:
        _tdm_store_tiled_two_buffers(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )
    else:
        _tdm_store_full_tile(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )


@gluon.jit
def mxfp8_gemm_warp_pipeline_bk1024_flat(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as,
    stride_bs,
    GRID_MN: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    SHARED_AS: gl.constexpr,
    SHARED_BS: gl.constexpr,
    WMMA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CLUSTER: gl.constexpr,
    TWO_BUFFER_OUTPUT: gl.constexpr = False,
    NATIVE_SCALE128: gl.constexpr = False,
):
    gl.static_assert(gl.num_ctas() == CLUSTER * CLUSTER)
    pid_m, pid_n = _get_xcd_swizzled_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, 8, 4)
    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA, 16)
    scale_a_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_a, [BLOCK_M, 4]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna5.get_wmma_scale_layout(
        dot_b, [BLOCK_N, 4]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, 1024], SHARED_A
    )
    b_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_N, 1024], SHARED_B
    )
    a_desc = tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, 1024),
        layout=SHARED_A,
    )
    b_desc = tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, 1024),
        layout=SHARED_B,
    )
    if NATIVE_SCALE128:
        as_buf, bs_buf, as_desc, bs_desc = _mxfp8_native_scale_buffers(
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            K,
            stride_as,
            stride_bs,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            1024,
            2,
            SHARED_AS,
            SHARED_BS,
        )
    else:
        as_buf = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty, [2, BLOCK_M, 32], SHARED_AS
        )
        bs_buf = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty, [2, BLOCK_N, 32], SHARED_BS
        )
        as_desc = tdm.make_tensor_descriptor(
            base=a_scale_ptr + pid_m * BLOCK_M * stride_as,
            shape=(M, K // 32),
            strides=(stride_as, 1),
            block_shape=(BLOCK_M, 32),
            layout=SHARED_AS,
        )
        bs_desc = tdm.make_tensor_descriptor(
            base=b_scale_ptr + pid_n * BLOCK_N * stride_bs,
            shape=(N, K // 32),
            strides=(stride_bs, 1),
            block_shape=(BLOCK_N, 32),
            layout=SHARED_BS,
        )
    for slot in gl.static_range(2):
        a_desc, b_desc, as_desc, bs_desc = _mxfp8_tiled_issue_refill(
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            slot,
            1024,
            32,
        )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    iter_max: gl.constexpr = K // 1024
    gl.static_assert(K % 1024 == 0 and iter_max >= 3)
    for tile in gl.static_range(iter_max - 2):
        a_desc, b_desc, as_desc, bs_desc, acc = _mxfp8_tiled_bk1024_nine(
            a_buf,
            b_buf,
            as_buf,
            bs_buf,
            tile % 2,
            a_desc,
            b_desc,
            as_desc,
            bs_desc,
            acc,
            dot_a,
            dot_b,
            scale_a_layout,
            scale_b_layout,
        )
    tdm.async_wait(3)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 2) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        1024,
    )
    tdm.async_wait(0)
    acc = _mxfp8_tiled_consume_tail(
        a_buf,
        b_buf,
        as_buf,
        bs_buf,
        (iter_max - 1) % 2,
        acc,
        dot_a,
        dot_b,
        scale_a_layout,
        scale_b_layout,
        1024,
    )
    _cluster_sync()
    a_buf._keep_alive()
    b_buf._keep_alive()
    as_buf._keep_alive()
    bs_buf._keep_alive()
    if TWO_BUFFER_OUTPUT:
        _tdm_store_tiled_two_buffers(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )
    else:
        _tdm_store_full_tile(
            c_ptr,
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            M,
            N,
            acc,
            WMMA,
            BLOCK_M,
            BLOCK_N,
            CLUSTER,
        )


def validate_gemm_shape(dtype, M, N, K, tile=None, cluster=None):
    """Apply the same dimension constraints to CLI and Python callers."""
    if dtype == "mxfp8":
        return mxfp8_config(M, N, K, tile, cluster)
    constraints = {"bf16": (256, 256), "mxfp4": (1024, 2048), "fp8_mxfp4": (256, 1024)}
    if dtype not in constraints:
        raise ValueError(f"unsupported GEMM dtype: {dtype}")
    if M <= 0 or N <= 0 or M % 1024 or N % 1024:
        raise ValueError("M and N must be positive multiples of 1024")
    multiple, minimum = constraints[dtype]
    if K < minimum or K % multiple:
        raise ValueError(f"{dtype} requires K >= {minimum} and divisible by {multiple}")


MXFP8_TILES = {
    "256x256x256": (256, 256),
    "128x128x512": (128, 512),
    "64x64x512": (64, 512),
    "64x64x1024": (64, 1024),
}


def mxfp8_config(M, N, K, tile, cluster):
    """Validate the user-selected CTA tile, cluster and matrix dimensions."""
    if tile is None or cluster is None:
        raise ValueError("MXFP8 requires explicit --mxfp8-tile and --mxfp8-cluster")
    if tile not in MXFP8_TILES:
        raise ValueError(
            f"unsupported MXFP8 tile: {tile}; choose from {', '.join(MXFP8_TILES)}"
        )
    if cluster not in ("2x2", "4x4"):
        raise ValueError(f"unsupported MXFP8 cluster: {cluster}; choose 2x2 or 4x4")
    cta_tile, block_k = MXFP8_TILES[tile]
    width = int(cluster[0])
    block = cta_tile * width
    minimum = 1024 if block_k == 256 else 3 * block_k
    k_multiple = 512 if block_k == 256 else block_k
    if M <= 0 or N <= 0 or M % block or N % block:
        raise ValueError(
            f"MXFP8 {tile} / {width}x{width} requires positive M/N divisible by {block}"
        )
    if K < minimum or K % k_multiple:
        raise ValueError(
            f"MXFP8 {tile} requires K >= {minimum} and divisible by {k_multiple}"
        )
    return cta_tile, block_k, width
