# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math

import triton
import triton.experimental.gluon.language as gl
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    _store_tile_from_registers,
)
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp4 import (
    cluster_shape,
)
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "NUM_BUFFERS",
    "NUM_CTAS",
    "GROUP_SIZE_M",
    "FUSED_LOAD",
    "REGISTER_STORE",
    "NUM_KSPLIT",
    "SCALE_TDM",
    "GRID_2D",
    "EVEN_K",
]

_gemm_a8w8_ptpc_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_a8w8_ptpc_gfx1250_bandwidth_bound_kernel", _REPR_KEYS + ["NUM_KSUB"]
)
_gemm_a8w8_ptpc_pipelined_repr = make_kernel_repr(
    "_gemm_a8w8_ptpc_gfx1250_pipelined_kernel", _REPR_KEYS + ["SLICED_K", "NUM_KSUB", "B2B_WMMA"]
)
_gemm_a8w8_ptpc_persistent_repr = make_kernel_repr(
    "_gemm_a8w8_ptpc_gfx1250_persistent_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "NUM_BUFFERS", "NUM_CTAS", "FUSED_LOAD", "REGISTER_STORE", "ADD_BIAS", "B2B_WMMA", "GRID"],
)

_WMMA_INSTR_SHAPE = [16, 16, 128]
_MAX_PAD_INTERVAL_BYTES = 1024


@functools.cache
def create_ptpc_layouts(
    kernel_type,
    num_warps,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_ctas=1,
    warps_n=2,
    register_store=0,
    ctas_m=0,
):
    """BLOCK_M and BLOCK_N are cluster tiles. ctas_m is the CTA count along M (0 balances the tile)."""
    if ctas_m:
        assert num_ctas % ctas_m == 0, "CTAS_M must divide num_ctas"
        ctas_n = num_ctas // ctas_m
    else:
        ctas_m, ctas_n = cluster_shape(num_ctas, BLOCK_M, BLOCK_N)
    cga = []
    if num_ctas > 1:
        cga = make_cga_layout([ctas_m, ctas_n], [ctas_m, ctas_n], [0, 1])

    warps_n = min(warps_n, num_warps)
    warp_bases = [(0, 1 << i) for i in range(int(math.log2(warps_n)))]
    warp_bases += [(1 << i, 0) for i in range(int(math.log2(num_warps // warps_n)))]
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=_WMMA_INSTR_SHAPE,
        cga_layout=cga,
    )
    cga_a = gl.DotOperandLayout(0, wmma_layout, k_width=16).cga_layout
    cga_b = gl.DotOperandLayout(1, wmma_layout, k_width=16).cga_layout

    # A and B multicast to the CTAs that share their rows; a 16-byte pad avoids bank conflicts on dual-address loads.
    pad_interval = min(max(BLOCK_K, 256), _MAX_PAD_INTERVAL_BYTES)
    layouts = {
        "SHARED_LAYOUT_A": gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, 16]], [BLOCK_M, BLOCK_K], [1, 0], cga_a
        ),
        "SHARED_LAYOUT_B": gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, 16]],
            [BLOCK_N, BLOCK_K],
            [1, 0],
            [[n, k] for k, n in cga_b],
        ),
        "SHARED_LAYOUT_C": gl.SwizzledSharedLayout(1, 1, 1, [1, 0], cga_layout=cga),
        "STORE_LAYOUT": None,
        "SCALE_LAYOUT_A": gl.SwizzledSharedLayout(
            1, 1, 1, [1, 0], cga_layout=[[0, m] for m, _ in cga]
        ),
        "SCALE_LAYOUT_B": gl.SwizzledSharedLayout(
            1, 1, 1, [1, 0], cga_layout=[[0, n] for _, n in cga]
        ),
    }

    if kernel_type == "sliced_k":
        cga_3d = [[0, m, n] for m, n in cga]
        wmma_layout = gl.amd.AMDWMMALayout(
            version=3,
            transposed=True,
            warp_bases=[[1 << i, 0, 0] for i in range(int(math.log2(num_warps)))],
            reg_bases=[],
            instr_shape=_WMMA_INSTR_SHAPE,
            cga_layout=cga_3d,
            rank=3,
        )
        cta_n = BLOCK_N // ctas_n
        threads_n = min(32, cta_n // 4)
        reduce_reg = gl.BlockedLayout(
            [num_warps, 1, 4],
            [1, 32 // threads_n, threads_n],
            [1, num_warps, 1],
            [2, 1, 0],
            cga_3d,
        )
        layouts["REDUCE_SHARED_LAYOUT"] = gl.PaddedSharedLayout.with_identity_for(
            [[cta_n, 4]], [num_warps, BLOCK_M, BLOCK_N], [2, 1, 0], cga_3d
        )
        layouts["REDUCE_LAYOUT"] = reduce_reg
        layouts["OUT_LAYOUT"] = gl.SliceLayout(0, reduce_reg)
    else:
        layouts["OUT_LAYOUT"] = wmma_layout
        if register_store in (2, 3):
            cta_n = BLOCK_N // ctas_n
            vec = min(8, cta_n)
            threads_n = min(32, cta_n // vec)
            layouts["STORE_LAYOUT"] = gl.BlockedLayout(
                [1, vec], [32 // threads_n, threads_n], [num_warps, 1], [1, 0], cga
            )
        if register_store == 3:
            layouts["STAGE_LAYOUT"] = gl.PaddedSharedLayout.with_identity_for(
                [[cta_n, 8]], [BLOCK_M, BLOCK_N], [1, 0], cga
            )

    # The second half of K negates the smaller operand of the CTA in LDS, as int32 words,
    # in K chunks of at most 16 words for each thread.
    neg_b = BLOCK_N // ctas_n < BLOCK_M // ctas_m
    cta_rows = BLOCK_N // ctas_n if neg_b else BLOCK_M // ctas_m
    neg_layout = layouts["SHARED_LAYOUT_B" if neg_b else "SHARED_LAYOUT_A"]
    assert all(k == 0 for _, k in neg_layout.cga_layout), "the CGA must not split K"
    words = BLOCK_K // 4
    chunk = min(words, max(4, 16 * 32 * num_warps // cta_rows))
    threads_k = min(32, chunk // 4)
    layouts["NEG_B"] = neg_b
    layouts["NEG_WORDS"] = chunk
    layouts["NEG_SHARED_LAYOUT"] = gl.PaddedSharedLayout.with_identity_for(
        [[pad_interval // 4, 4]],
        [neg_layout.shape[0], words],
        [1, 0],
        neg_layout.cga_layout,
    )
    layouts["NEG_LAYOUT"] = gl.BlockedLayout(
        [1, 4],
        [32 // threads_k, threads_k],
        [num_warps, 1],
        [1, 0],
        neg_layout.cga_layout,
    )
    layouts["WMMA_LAYOUT"] = wmma_layout
    layouts["OPERAND_LAYOUT_A"] = gl.DotOperandLayout(0, wmma_layout, k_width=16)
    layouts["OPERAND_LAYOUT_B"] = gl.DotOperandLayout(1, wmma_layout, k_width=16)
    return layouts


def _even_k(args):
    split = args.get("SPLITK_BLOCK_SIZE", 0) if args.get("NUM_KSPLIT", 1) > 1 else 0
    if split:
        return split % args["BLOCK_K"] == 0 and args["K"] % split == 0
    return args["K"] % args["BLOCK_K"] == 0


@gluon.jit
def _issue_stage(
    a_desc,
    b_desc,
    a_slot,
    b_slot,
    BLOCK_K: gl.constexpr,
    EVEN_K: gl.constexpr,
    FUSED_LOAD: gl.constexpr,
):
    if FUSED_LOAD:
        W: gl.constexpr = ((1 << gl.num_warps()) - 1) // 0b11
        gl.amd.gfx1250.tdm.async_load_fused(
            [(a_desc, a_slot, W), (b_desc, b_slot, W << 1)]
        )
    else:
        gl.amd.gfx1250.tdm.async_load(a_desc, dest=a_slot)
        gl.amd.gfx1250.tdm.async_load(b_desc, dest=b_slot)
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, BLOCK_K], clamp_bounds=not EVEN_K
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, BLOCK_K], clamp_bounds=not EVEN_K
    )
    return a_desc, b_desc


@gluon.jit
def _negate_slot(
    a_slot,
    b_slot,
    NEG_B: gl.constexpr,
    NEG_WORDS: gl.constexpr,
    NEG_SHARED_LAYOUT: gl.constexpr,
    NEG_LAYOUT: gl.constexpr,
):
    """Flip the sign bits of the FP8 slot of A, or of B, in LDS.

    NEG_WORDS == 0: the second half of K of A or of B is already negated in memory.
    """
    if NEG_WORDS > 0:
        if NEG_B:
            slot = b_slot
        else:
            slot = a_slot
        view = slot.reinterpret(
            gl.int32, [slot.shape[0], slot.shape[1] // 4], NEG_SHARED_LAYOUT
        )
        for c in gl.static_range(view.shape[1] // NEG_WORDS):
            part = view.slice(c * NEG_WORDS, NEG_WORDS, 1)
            part.store(part.load(NEG_LAYOUT) ^ -2139062144)
        gl.barrier()


@gluon.jit
def _read_stage(
    a_slot,
    b_slot,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    SLICED_K: gl.constexpr,
):
    if SLICED_K:
        KS: gl.constexpr = gl.num_warps()
        a_view = a_slot.reshape([a_slot.shape[0], KS, a_slot.shape[1] // KS]).permute(
            [1, 0, 2]
        )
        b_view = b_slot.reshape([b_slot.shape[0], KS, b_slot.shape[1] // KS]).permute(
            [1, 2, 0]
        )
    else:
        a_view = a_slot
        b_view = b_slot.permute([1, 0])
    stage_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_view, OPERAND_LAYOUT_A)
    stage_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_view, OPERAND_LAYOUT_B)
    return stage_a, stage_b


@gluon.jit
def _compute_stage_in_place(
    a_slot,
    b_slot,
    accumulator,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    NUM_KSUB: gl.constexpr,
):
    KSUB: gl.constexpr = a_slot.shape[1] // NUM_KSUB
    for sub in gl.static_range(NUM_KSUB):
        sub_a, sub_b = _read_stage(
            a_slot.slice(sub * KSUB, KSUB, 1),
            b_slot.slice(sub * KSUB, KSUB, 1),
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            False,
        )
        accumulator = gl.amd.gfx1250.wmma(sub_a, sub_b, accumulator)
        if NUM_KSUB > 1:
            gl.amd.hint.sched_barrier()
    return accumulator


@gluon.jit
def _issue_scale_loads(
    a_scale_ptr,
    b_scale_ptr,
    pid_m,
    pid_n,
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SCALE_LAYOUT_A: gl.constexpr,
    SCALE_LAYOUT_B: gl.constexpr,
):
    as_buffer = gl.allocate_shared_memory(
        gl.float32, [1, 1, BLOCK_M], SCALE_LAYOUT_A
    ).index(0)
    bs_buffer = gl.allocate_shared_memory(
        gl.float32, [1, 1, BLOCK_N], SCALE_LAYOUT_B
    ).index(0)
    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N
    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr + m_off,
        shape=(1, M - m_off),
        strides=(M, 1),
        block_shape=(1, BLOCK_M),
        layout=SCALE_LAYOUT_A,
    )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr + n_off,
        shape=(1, N - n_off),
        strides=(N, 1),
        block_shape=(1, BLOCK_N),
        layout=SCALE_LAYOUT_B,
    )
    if gl.num_warps() >= 2:
        gl.amd.gfx1250.tdm.async_load_fused(
            [(as_desc, as_buffer, 0b01), (bs_desc, bs_buffer, 0b10)]
        )
    else:
        gl.amd.gfx1250.tdm.async_load(as_desc, dest=as_buffer)
        gl.amd.gfx1250.tdm.async_load(bs_desc, dest=bs_buffer)
    return as_buffer, bs_buffer


@gluon.jit
def _init_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    GRID_2D: gl.constexpr,
):
    k_start = 0
    K_local = K
    if GRID_2D:
        pid_m = gl.program_id(axis=0)
        pid_n = gl.program_id(axis=1)
    else:
        pid = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        if NUM_KSPLIT > 1:
            num_tiles = num_pid_m * gl.cdiv(N, BLOCK_N)
            pid_k = pid // num_tiles
            pid = pid % num_tiles
            k_start = pid_k * SPLITK_BLOCK_SIZE
            K_local = gl.minimum(K - k_start, SPLITK_BLOCK_SIZE)
            c_ptr = c_ptr + pid_k.to(gl.int64) * stride_ck
        if GROUP_SIZE_M == 0:
            pid_m = pid % num_pid_m
            pid_n = pid // num_pid_m
        else:
            pid_m, pid_n = pid_grid(pid, num_pid_m, gl.cdiv(N, BLOCK_N), GROUP_SIZE_M)

    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + m_off.to(gl.int64) * stride_am + k_start * stride_ak,
        shape=(M - m_off, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + n_off.to(gl.int64) * stride_bn + k_start * stride_bk,
        shape=(N - n_off, K_local),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
        layout=SHARED_LAYOUT_B,
    )
    return (
        pid_m,
        pid_n,
        K_local,
        c_ptr,
        a_desc,
        b_desc,
        a_buffer,
        b_buffer,
    )


@gluon.jit
def _load_scales(
    a_scale_ptr,
    b_scale_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_a_scale,
    stride_b_scale,
    bias_ptr,
    stride_bias,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    as_buffer,
    bs_buffer,
    SCALE_TDM: gl.constexpr,
):
    offs_m = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, OUT_LAYOUT)
    )
    offs_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, OUT_LAYOUT)
    )
    if SCALE_TDM:
        gl.barrier()
        a_scale = as_buffer.permute((1, 0)).load(layout=OUT_LAYOUT)
        b_scale = bs_buffer.load(layout=OUT_LAYOUT)
    else:
        a_scale = gl.load(
            a_scale_ptr + offs_m.to(gl.int64) * stride_a_scale,
            mask=offs_m < M,
            other=0.0,
        )
        b_scale = gl.load(
            b_scale_ptr + offs_n.to(gl.int64) * stride_b_scale,
            mask=offs_n < N,
            other=0.0,
        )
        a_scale = a_scale[:, None]
        b_scale = b_scale[None, :]
    if ADD_BIAS:
        bias = gl.load(
            bias_ptr + offs_n.to(gl.int64) * stride_bias, mask=offs_n < N, other=0.0
        ).to(gl.float32)[None, :]
    else:
        bias = b_scale
    return a_scale, b_scale, bias


@gluon.jit
def _scale_and_store(
    c_buffer,
    accumulator,
    a_scale,
    b_scale,
    bias,
    c_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    REGISTER_STORE: gl.constexpr,
    STORE_LAYOUT: gl.constexpr,
    STAGE_LAYOUT: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """Apply the scales and the bias to the fp32 tile and store it, clamped to (M, N).

    REGISTER_STORE: 0 TDM store, 1 register store, 2 coalesced register store, 3 staged store.
    """
    gl.static_assert(
        (REGISTER_STORE >= 0) & (REGISTER_STORE <= 3),
        "REGISTER_STORE must be 0, 1, 2 or 3",
    )
    c_tile = accumulator * a_scale * b_scale
    if ADD_BIAS:
        c_tile = c_tile + bias
    c_tile = c_tile.to(c_ptr.type.element_ty)

    if REGISTER_STORE == 0:
        c_buffer.store(c_tile)
        gl.barrier()
        if NUM_CTAS > 1:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
        c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=SHARED_LAYOUT_C,
        )
        gl.amd.gfx1250.tdm.async_store(
            c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_buffer
        )
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        STORE: gl.constexpr = OUT_LAYOUT if REGISTER_STORE == 1 else STORE_LAYOUT
        if REGISTER_STORE == 3:
            gl.barrier()
            stage = gl.allocate_shared_memory(
                c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N], layout=STAGE_LAYOUT
            )
            stage.store(c_tile)
            gl.barrier()
            c_tile = stage.load(layout=STORE)
        elif REGISTER_STORE == 2:
            c_tile = gl.convert_layout(c_tile, STORE)
        _store_tile_from_registers(
            c_ptr,
            c_tile,
            pid_m,
            pid_n,
            M,
            N,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            STORE,
        )


@triton.heuristics({"EVEN_K": _even_k})
@gluon.jit(repr=_gemm_a8w8_ptpc_bandwidth_bound_repr)
def _gemm_a8w8_ptpc_bandwidth_bound_kernel(
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
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_ck,
    stride_a_scale,
    stride_b_scale,
    bias_ptr,
    stride_bias,
    first_tiles,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EVEN_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    SCALE_LAYOUT_A: gl.constexpr,
    SCALE_LAYOUT_B: gl.constexpr,
    STORE_LAYOUT: gl.constexpr = None,
    STAGE_LAYOUT: gl.constexpr = None,
    NUM_CTAS: gl.constexpr = 1,
    GROUP_SIZE_M: gl.constexpr = 0,
    FUSED_LOAD: gl.constexpr = 0,
    REGISTER_STORE: gl.constexpr = 1,
    ADD_BIAS: gl.constexpr = False,
    NUM_KSPLIT: gl.constexpr = 1,
    SPLITK_BLOCK_SIZE: gl.constexpr = 0,
    SCALE_TDM: gl.constexpr = 0,
    GRID_2D: gl.constexpr = False,
    NUM_KSUB: gl.constexpr = 1,
    NEG_B: gl.constexpr = False,
    NEG_WORDS: gl.constexpr = 4,
    NEG_SHARED_LAYOUT: gl.constexpr = None,
    NEG_LAYOUT: gl.constexpr = None,
):
    """C = (A x B^T) * a_scale[:, None] * b_scale[None, :] with A [M, K] and B [N, K] in FP8.

    Each stage is read from LDS in place, in NUM_KSUB K steps.
    """
    # A cluster with one LDS slot hangs.
    gl.static_assert(
        NUM_CTAS == 1 or NUM_BUFFERS >= 2, "a cluster needs NUM_BUFFERS >= 2"
    )
    gl.static_assert(
        BLOCK_K % (128 * NUM_KSUB) == 0, "each K step must be a multiple of 128"
    )
    TDM_OPS: gl.constexpr = 1 if FUSED_LOAD else 2
    pid_m, pid_n, K_local, c_ptr, a_desc, b_desc, a_buffer, b_buffer = _init_tile(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_ck,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        GROUP_SIZE_M,
        NUM_KSPLIT,
        SPLITK_BLOCK_SIZE,
        GRID_2D,
    )
    c_buffer = None
    if REGISTER_STORE == 0:
        # C has its own LDS: a C buffer that reused the operand LDS gave intermittent wrong tiles.
        c_buffer = gl.allocate_shared_memory(
            c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N], layout=SHARED_LAYOUT_C
        )
    as_buffer = None
    bs_buffer = None
    if SCALE_TDM:
        as_buffer, bs_buffer = _issue_scale_loads(
            a_scale_ptr,
            b_scale_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            SCALE_LAYOUT_A,
            SCALE_LAYOUT_B,
        )
    load_idx = 0
    compute_idx = 0
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Prologue: fill NUM_BUFFERS - 1 slots.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = load_idx % NUM_BUFFERS
        a_desc, b_desc = _issue_stage(
            a_desc,
            b_desc,
            a_buffer.index(slot),
            b_buffer.index(slot),
            BLOCK_K,
            EVEN_K,
            FUSED_LOAD,
        )
        load_idx += 1
    if NUM_CTAS > 1:
        gl.amd.gfx1250.cluster.arrive()

    # The FP8 WMMA truncates its products toward -inf. The second half of K runs with one operand
    # and the accumulator negated, so that half truncates toward +inf and the two biases cancel.
    num_main = gl.cdiv(K_local, BLOCK_K) - (NUM_BUFFERS - 1)
    # first_tiles is the K stage where a weight with a pre-negated K tail changes sign.
    half = gl.minimum(gl.cdiv(K_local, BLOCK_K) // 2, first_tiles)
    num_first = gl.maximum(gl.minimum(half, num_main), 0)
    # Main loop. The cluster barrier stops a multicast load from filling a slot that a CTA still reads.
    for phase in gl.static_range(2):
        if phase == 1:
            accumulator = -accumulator
        for _ in range(num_first * phase, num_first + (num_main - num_first) * phase):
            if NUM_CTAS > 1:
                gl.amd.gfx1250.cluster.wait()
            slot = load_idx % NUM_BUFFERS
            a_desc, b_desc = _issue_stage(
                a_desc,
                b_desc,
                a_buffer.index(slot),
                b_buffer.index(slot),
                BLOCK_K,
                EVEN_K,
                FUSED_LOAD,
            )
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS)
            load_idx += 1
            slot = compute_idx % NUM_BUFFERS
            if phase == 1:
                _negate_slot(
                    a_buffer.index(slot),
                    b_buffer.index(slot),
                    NEG_B,
                    NEG_WORDS,
                    NEG_SHARED_LAYOUT,
                    NEG_LAYOUT,
                )
            accumulator = _compute_stage_in_place(
                a_buffer.index(slot),
                b_buffer.index(slot),
                accumulator,
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                NUM_KSUB,
            )
            compute_idx += 1
            if NUM_CTAS > 1:
                gl.amd.gfx1250.cluster.arrive()

    if NUM_CTAS > 1:
        gl.amd.gfx1250.cluster.wait()

    # The scale loads run under the last WMMAs.
    a_scale, b_scale, bias = _load_scales(
        a_scale_ptr,
        b_scale_ptr,
        pid_m,
        pid_n,
        M,
        N,
        stride_a_scale,
        stride_b_scale,
        bias_ptr,
        stride_bias,
        BLOCK_M,
        BLOCK_N,
        OUT_LAYOUT,
        ADD_BIAS,
        as_buffer,
        bs_buffer,
        SCALE_TDM,
    )

    # Drain: no more loads.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * TDM_OPS)
        slot = compute_idx % NUM_BUFFERS
        _negate_slot(
            a_buffer.index(slot),
            b_buffer.index(slot),
            NEG_B,
            NEG_WORDS,
            NEG_SHARED_LAYOUT,
            NEG_LAYOUT,
        )
        accumulator = _compute_stage_in_place(
            a_buffer.index(slot),
            b_buffer.index(slot),
            accumulator,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            NUM_KSUB,
        )
        compute_idx += 1
    # The last sign change goes into the row scale: fewer instructions than a negated accumulator.
    a_scale = -a_scale

    _scale_and_store(
        c_buffer,
        accumulator,
        a_scale,
        b_scale,
        bias,
        c_ptr,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        NUM_CTAS,
        OUT_LAYOUT,
        SHARED_LAYOUT_C,
        REGISTER_STORE,
        STORE_LAYOUT,
        STAGE_LAYOUT,
        ADD_BIAS,
    )


@gluon.jit
def _enable_b2b_wmma():
    """Set HW_REG_WAVE_SCHED_MODE bit 2: the wave issues WMMAs back to back."""
    gl.inline_asm_elementwise(
        "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1; v_mov_b32 $0, 0;",
        constraints="=v",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _pipelined_ksub2_loop(
    accumulator,
    a_desc,
    b_desc,
    a_buffer,
    b_buffer,
    a_scale_ptr,
    b_scale_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_a_scale,
    stride_b_scale,
    bias_ptr,
    stride_bias,
    as_buffer,
    bs_buffer,
    num_first,
    num_main,
    load_idx,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EVEN_K: gl.constexpr,
    FUSED_LOAD: gl.constexpr,
    NUM_CTAS: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    SCALE_TDM: gl.constexpr,
):
    """K loop with each stage in two K sub-steps of BLOCK_K // 2; the NUM_BUFFERS stages are issued.

    The TDM issue and the cluster barrier run once per two sub-steps. A CTA arrives at the cluster
    barrier as soon as it has issued the last reads of a slot (during the first sub-step), a whole
    sub-step before the wait that guards the refill of that slot. The scheduling barrier before the
    wait keeps the register use below 1024 VGPRs at BLOCK 1024x1024 with 16 CTAs.
    Needs the K tail negated in memory (no LDS negation).
    """
    TDM_OPS: gl.constexpr = 1 if FUSED_LOAD else 2
    KSUB: gl.constexpr = BLOCK_K // 2
    compute_idx = 0
    cur_a, cur_b = _read_stage(
        a_buffer.index(0).slice(0, KSUB, 1),
        b_buffer.index(0).slice(0, KSUB, 1),
        OPERAND_LAYOUT_A,
        OPERAND_LAYOUT_B,
        False,
    )
    for phase in gl.static_range(2):
        if phase == 1:
            accumulator = -accumulator
        for i in range(num_first * phase, num_first + (num_main - num_first) * phase):
            slot = compute_idx % NUM_BUFFERS
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a, cur_b = _read_stage(
                a_buffer.index(slot).slice(KSUB, KSUB, 1),
                b_buffer.index(slot).slice(KSUB, KSUB, 1),
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                False,
            )
            if NUM_CTAS > 1:
                gl.amd.gfx1250.cluster.arrive()
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            if NUM_CTAS > 1:
                gl.amd.hint.sched_barrier()
                gl.amd.gfx1250.cluster.wait()
            slot = load_idx % NUM_BUFFERS
            a_desc, b_desc = _issue_stage(
                a_desc,
                b_desc,
                a_buffer.index(slot),
                b_buffer.index(slot),
                BLOCK_K,
                EVEN_K,
                FUSED_LOAD,
            )
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS)
            load_idx += 1
            slot = (compute_idx + 1) % NUM_BUFFERS
            cur_a, cur_b = _read_stage(
                a_buffer.index(slot).slice(0, KSUB, 1),
                b_buffer.index(slot).slice(0, KSUB, 1),
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                False,
            )
            compute_idx += 1
    if NUM_CTAS > 1:
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
    # Drain: no more loads.
    for st in gl.static_range(NUM_BUFFERS):
        slot = compute_idx % NUM_BUFFERS
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        cur_a, cur_b = _read_stage(
            a_buffer.index(slot).slice(KSUB, KSUB, 1),
            b_buffer.index(slot).slice(KSUB, KSUB, 1),
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            False,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        if st < NUM_BUFFERS - 1:
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - st) * TDM_OPS)
            slot = (compute_idx + 1) % NUM_BUFFERS
            cur_a, cur_b = _read_stage(
                a_buffer.index(slot).slice(0, KSUB, 1),
                b_buffer.index(slot).slice(0, KSUB, 1),
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                False,
            )
            compute_idx += 1
    # The scales are read after the drain: fewer live registers under the last WMMAs.
    a_scale, b_scale, bias = _load_scales(
        a_scale_ptr,
        b_scale_ptr,
        pid_m,
        pid_n,
        M,
        N,
        stride_a_scale,
        stride_b_scale,
        bias_ptr,
        stride_bias,
        BLOCK_M,
        BLOCK_N,
        OUT_LAYOUT,
        ADD_BIAS,
        as_buffer,
        bs_buffer,
        SCALE_TDM,
    )
    return accumulator, a_scale, b_scale, bias


@triton.heuristics({"EVEN_K": _even_k})
@gluon.jit(repr=_gemm_a8w8_ptpc_pipelined_repr)
def _gemm_a8w8_ptpc_pipelined_kernel(
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
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_ck,
    stride_a_scale,
    stride_b_scale,
    bias_ptr,
    stride_bias,
    first_tiles,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EVEN_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    SCALE_LAYOUT_A: gl.constexpr,
    SCALE_LAYOUT_B: gl.constexpr,
    STORE_LAYOUT: gl.constexpr = None,
    STAGE_LAYOUT: gl.constexpr = None,
    NUM_CTAS: gl.constexpr = 1,
    GROUP_SIZE_M: gl.constexpr = 0,
    FUSED_LOAD: gl.constexpr = 0,
    REGISTER_STORE: gl.constexpr = 1,
    ADD_BIAS: gl.constexpr = False,
    NUM_KSPLIT: gl.constexpr = 1,
    SPLITK_BLOCK_SIZE: gl.constexpr = 0,
    SCALE_TDM: gl.constexpr = 0,
    GRID_2D: gl.constexpr = False,
    SLICED_K: gl.constexpr = False,
    REDUCE_SHARED_LAYOUT: gl.constexpr = None,
    REDUCE_LAYOUT: gl.constexpr = None,
    NEG_B: gl.constexpr = False,
    NEG_WORDS: gl.constexpr = 4,
    NEG_SHARED_LAYOUT: gl.constexpr = None,
    NEG_LAYOUT: gl.constexpr = None,
    NUM_KSUB: gl.constexpr = 1,
    B2B_WMMA: gl.constexpr = 0,
):
    """Same result as the bandwidth_bound kernel; stage i + 1 is read from LDS during the WMMA of stage i.

    SLICED_K=True gives each warp one K slice of each stage and adds the warp partials through LDS.
    NUM_KSUB=2 runs each stage as two K sub-steps (see _pipelined_ksub2_loop).
    B2B_WMMA=1 lets each wave issue WMMAs back to back (HW_REG_WAVE_SCHED_MODE bit 2).
    """
    gl.static_assert(NUM_BUFFERS >= 2, "the pipelined kernel needs NUM_BUFFERS >= 2")
    if SLICED_K:
        gl.static_assert(
            BLOCK_K % (128 * gl.num_warps()) == 0,
            "sliced_k needs BLOCK_K % (128 * num_warps) == 0",
        )
    gl.static_assert(
        (NUM_KSUB == 1) | ((NUM_KSUB == 2) & (NEG_WORDS == 0) & (not SLICED_K)),
        "NUM_KSUB=2 needs a K tail negated in memory and no sliced_k",
    )
    if B2B_WMMA:
        _enable_b2b_wmma()
    TDM_OPS: gl.constexpr = 1 if FUSED_LOAD else 2
    pid_m, pid_n, K_local, c_ptr, a_desc, b_desc, a_buffer, b_buffer = _init_tile(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_ck,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        GROUP_SIZE_M,
        NUM_KSPLIT,
        SPLITK_BLOCK_SIZE,
        GRID_2D,
    )
    c_buffer = None
    if REGISTER_STORE == 0:
        # C has its own LDS: a C buffer that reused the operand LDS gave intermittent wrong tiles.
        c_buffer = gl.allocate_shared_memory(
            c_ptr.type.element_ty, shape=[BLOCK_M, BLOCK_N], layout=SHARED_LAYOUT_C
        )
    as_buffer = None
    bs_buffer = None
    if SCALE_TDM:
        as_buffer, bs_buffer = _issue_scale_loads(
            a_scale_ptr,
            b_scale_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            SCALE_LAYOUT_A,
            SCALE_LAYOUT_B,
        )
    load_idx = 0
    compute_idx = 0
    if SLICED_K:
        accumulator = gl.zeros(
            (gl.num_warps(), BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT
        )
    else:
        accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Prologue: fill all slots, then read stage 0 into registers.
    for _ in gl.static_range(NUM_BUFFERS):
        slot = load_idx % NUM_BUFFERS
        a_desc, b_desc = _issue_stage(
            a_desc,
            b_desc,
            a_buffer.index(slot),
            b_buffer.index(slot),
            BLOCK_K,
            EVEN_K,
            FUSED_LOAD,
        )
        load_idx += 1
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS)
    slot = compute_idx % NUM_BUFFERS
    num_main = gl.cdiv(K_local, BLOCK_K) - NUM_BUFFERS
    # first_tiles is the K stage where a weight with a pre-negated K tail changes sign.
    half = gl.minimum(gl.cdiv(K_local, BLOCK_K) // 2, first_tiles)
    num_first = gl.maximum(gl.minimum(half, num_main), 0)
    if num_first == 0:
        _negate_slot(
            a_buffer.index(slot),
            b_buffer.index(slot),
            NEG_B,
            NEG_WORDS,
            NEG_SHARED_LAYOUT,
            NEG_LAYOUT,
        )
    if NUM_KSUB == 1:
        cur_a, cur_b = _read_stage(
            a_buffer.index(slot),
            b_buffer.index(slot),
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            SLICED_K,
        )
        if NUM_CTAS > 1:
            gl.amd.gfx1250.cluster.arrive()

        # Main loop: WMMA of cur, refill its slot, wait, read the next stage.
        for phase in gl.static_range(2):
            if phase == 1:
                # See the bandwidth_bound kernel. cur was read negated in the last phase-0 iteration.
                accumulator = -accumulator
            for i in range(num_first * phase, num_first + (num_main - num_first) * phase):
                accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
                # Each CTA has read the slot of cur before a multicast load fills it.
                if NUM_CTAS > 1:
                    gl.amd.gfx1250.cluster.wait()
                slot = load_idx % NUM_BUFFERS
                a_desc, b_desc = _issue_stage(
                    a_desc,
                    b_desc,
                    a_buffer.index(slot),
                    b_buffer.index(slot),
                    BLOCK_K,
                    EVEN_K,
                    FUSED_LOAD,
                )
                gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS)
                load_idx += 1
                slot = (compute_idx + 1) % NUM_BUFFERS
                if phase == 1 or i == num_first - 1:
                    _negate_slot(
                        a_buffer.index(slot),
                        b_buffer.index(slot),
                        NEG_B,
                        NEG_WORDS,
                        NEG_SHARED_LAYOUT,
                        NEG_LAYOUT,
                    )
                cur_a, cur_b = _read_stage(
                    a_buffer.index(slot),
                    b_buffer.index(slot),
                    OPERAND_LAYOUT_A,
                    OPERAND_LAYOUT_B,
                    SLICED_K,
                )
                compute_idx += 1
                if NUM_CTAS > 1:
                    gl.amd.gfx1250.cluster.arrive()

        if NUM_CTAS > 1:
            gl.amd.gfx1250.cluster.wait()

        # The scale loads run under the last WMMAs.
        a_scale, b_scale, bias = _load_scales(
            a_scale_ptr,
            b_scale_ptr,
            pid_m,
            pid_n,
            M,
            N,
            stride_a_scale,
            stride_b_scale,
            bias_ptr,
            stride_bias,
            BLOCK_M,
            BLOCK_N,
            OUT_LAYOUT,
            ADD_BIAS,
            as_buffer,
            bs_buffer,
            SCALE_TDM,
        )

        # Drain: no more loads.
        for i in gl.static_range(NUM_BUFFERS - 1):
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * TDM_OPS)
            slot = (compute_idx + 1) % NUM_BUFFERS
            _negate_slot(
                a_buffer.index(slot),
                b_buffer.index(slot),
                NEG_B,
                NEG_WORDS,
                NEG_SHARED_LAYOUT,
                NEG_LAYOUT,
            )
            cur_a, cur_b = _read_stage(
                a_buffer.index(slot),
                b_buffer.index(slot),
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                SLICED_K,
            )
            compute_idx += 1
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
    else:
        accumulator, a_scale, b_scale, bias = _pipelined_ksub2_loop(
            accumulator,
            a_desc,
            b_desc,
            a_buffer,
            b_buffer,
            a_scale_ptr,
            b_scale_ptr,
            pid_m,
            pid_n,
            M,
            N,
            stride_a_scale,
            stride_b_scale,
            bias_ptr,
            stride_bias,
            as_buffer,
            bs_buffer,
            num_first,
            num_main,
            load_idx,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
            EVEN_K,
            FUSED_LOAD,
            NUM_CTAS,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            OUT_LAYOUT,
            ADD_BIAS,
            SCALE_TDM,
        )
    # The last sign change goes into the row scale: fewer instructions than a negated accumulator.
    a_scale = -a_scale

    if SLICED_K:
        # Add the K slice partials through LDS. The operand slots are dead here.
        gl.barrier()
        partials = gl.allocate_shared_memory(
            gl.float32, [gl.num_warps(), BLOCK_M, BLOCK_N], REDUCE_SHARED_LAYOUT
        )
        partials.store(accumulator)
        gl.barrier()
        accumulator = gl.sum(partials.load(REDUCE_LAYOUT), axis=0)

    _scale_and_store(
        c_buffer,
        accumulator,
        a_scale,
        b_scale,
        bias,
        c_ptr,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        NUM_CTAS,
        OUT_LAYOUT,
        SHARED_LAYOUT_C,
        REGISTER_STORE,
        STORE_LAYOUT,
        STAGE_LAYOUT,
        ADD_BIAS,
    )


@gluon.jit
def _persistent_tile(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    tile,
    k0,
    M: gl.constexpr,
    N: gl.constexpr,
    K,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
):
    """Tile -> (pid_m, pid_n) and operand descriptors from K offset k0.

    Tiles walk M first; the N columns are rotated by one, so the first num_pid_m tiles (one per
    cluster) are the last, partial N column: the clusters that start there run out of phase with
    the others, and their output stores overlap the K loops of the others.
    """
    num_pid_m: gl.constexpr = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = tile % num_pid_m
    pid_n = (tile // num_pid_m + num_pid_n - 1) % num_pid_n
    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + m_off.to(gl.int64) * stride_am + k0,
        shape=(M - m_off, K - k0),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + n_off.to(gl.int64) * stride_bn + k0,
        shape=(N - n_off, K - k0),
        strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    return pid_m, pid_n, a_desc, b_desc


@gluon.jit
def _persistent_prologue(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    tile,
    M: gl.constexpr,
    N: gl.constexpr,
    K,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    SCALE_LAYOUT_A: gl.constexpr,
    SCALE_LAYOUT_B: gl.constexpr,
    a_buffer,
    b_buffer,
    as_buffer,
    bs_buffer,
    NUM_BUFFERS: gl.constexpr,
    FUSED_LOAD: gl.constexpr,
):
    """Issue the scale loads and the first NUM_BUFFERS stages of a tile."""
    pid_m, pid_n, a_desc, b_desc = _persistent_tile(
        a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, tile, 0, M, N, K, stride_am, stride_bn,
        BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B,
    )
    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N
    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr + m_off, shape=(1, M - m_off), strides=(M, 1), block_shape=(1, BLOCK_M), layout=SCALE_LAYOUT_A
    )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr + n_off, shape=(1, N - n_off), strides=(N, 1), block_shape=(1, BLOCK_N), layout=SCALE_LAYOUT_B
    )
    gl.amd.gfx1250.tdm.async_load_fused([(as_desc, as_buffer, 0b01), (bs_desc, bs_buffer, 0b10)])
    for st in gl.static_range(NUM_BUFFERS):
        a_desc, b_desc = _issue_stage(
            a_desc, b_desc, a_buffer.index(st), b_buffer.index(st), BLOCK_K, True, FUSED_LOAD
        )


@gluon.jit(repr=_gemm_a8w8_ptpc_persistent_repr)
def _gemm_a8w8_ptpc_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K,
    stride_am: gl.constexpr,
    stride_bn: gl.constexpr,
    stride_cm: gl.constexpr,
    first_tiles: gl.constexpr,
    num_tiles: gl.constexpr,
    GRID: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    OUT_LAYOUT: gl.constexpr,
    SCALE_LAYOUT_A: gl.constexpr,
    SCALE_LAYOUT_B: gl.constexpr,
    STORE_LAYOUT: gl.constexpr = None,
    STAGE_LAYOUT: gl.constexpr = None,
    NUM_CTAS: gl.constexpr = 16,
    FUSED_LOAD: gl.constexpr = 1,
    REGISTER_STORE: gl.constexpr = 2,
    ADD_BIAS: gl.constexpr = False,
    B2B_WMMA: gl.constexpr = 0,
):
    """Persistent compute_bound kernel: GRID clusters walk the tiles tile, tile + GRID, ...

    Each stage runs as two K sub-steps with the early cluster arrive of _pipelined_ksub2_loop.
    The scales and the first NUM_BUFFERS stages of the next tile are issued before the output store
    of the current tile. x and w need the K tail negated in memory; first_tiles is the phase change.
    M, N and the strides are constexpr: with runtime values the kernel spills. Needs M % BLOCK_M == 0
    (a partial M tile spills).
    """
    gl.static_assert(NUM_CTAS > 1, "the persistent kernel needs a cluster")
    if B2B_WMMA:
        _enable_b2b_wmma()
    TDM_OPS: gl.constexpr = 1 if FUSED_LOAD else 2
    NB: gl.constexpr = NUM_BUFFERS
    KSUB: gl.constexpr = BLOCK_K // 2
    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty, shape=[NB, BLOCK_M, BLOCK_K], layout=SHARED_LAYOUT_A
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty, shape=[NB, BLOCK_N, BLOCK_K], layout=SHARED_LAYOUT_B
    )
    as_buffer = gl.allocate_shared_memory(gl.float32, [1, 1, BLOCK_M], SCALE_LAYOUT_A).index(0)
    bs_buffer = gl.allocate_shared_memory(gl.float32, [1, 1, BLOCK_N], SCALE_LAYOUT_B).index(0)
    start = gl.program_id(axis=0)
    _persistent_prologue(
        a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, start, M, N, K, stride_am, stride_bn,
        BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B, SCALE_LAYOUT_A, SCALE_LAYOUT_B,
        a_buffer, b_buffer, as_buffer, bs_buffer, NB, FUSED_LOAD,
    )
    num_main = gl.cdiv(K, BLOCK_K) - NB
    num_first = gl.maximum(gl.minimum(first_tiles, num_main), 0)
    for tile in range(start, num_tiles, GRID):
        pid_m, pid_n, a_desc, b_desc = _persistent_tile(
            a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, tile, NB * BLOCK_K, M, N, K, stride_am, stride_bn,
            BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B,
        )
        # The scale load was issued before the stages.
        gl.amd.gfx1250.tdm.async_wait((NB - 1) * TDM_OPS)
        compute_idx = 0
        load_idx = NB
        accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
        cur_a, cur_b = _read_stage(
            a_buffer.index(0).slice(0, KSUB, 1),
            b_buffer.index(0).slice(0, KSUB, 1),
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
            False,
        )
        for phase in gl.static_range(2):
            if phase == 1:
                accumulator = -accumulator
            for i in range(num_first * phase, num_first + (num_main - num_first) * phase):
                slot = compute_idx % NB
                accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
                cur_a, cur_b = _read_stage(
                    a_buffer.index(slot).slice(KSUB, KSUB, 1),
                    b_buffer.index(slot).slice(KSUB, KSUB, 1),
                    OPERAND_LAYOUT_A,
                    OPERAND_LAYOUT_B,
                    False,
                )
                gl.amd.gfx1250.cluster.arrive()
                accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
                gl.amd.hint.sched_barrier()
                gl.amd.gfx1250.cluster.wait()
                lslot = load_idx % NB
                a_desc, b_desc = _issue_stage(
                    a_desc, b_desc, a_buffer.index(lslot), b_buffer.index(lslot), BLOCK_K, True, FUSED_LOAD
                )
                gl.amd.gfx1250.tdm.async_wait((NB - 1) * TDM_OPS)
                load_idx += 1
                slot = (compute_idx + 1) % NB
                cur_a, cur_b = _read_stage(
                    a_buffer.index(slot).slice(0, KSUB, 1),
                    b_buffer.index(slot).slice(0, KSUB, 1),
                    OPERAND_LAYOUT_A,
                    OPERAND_LAYOUT_B,
                    False,
                )
                compute_idx += 1
        for st in gl.static_range(NB):
            slot = compute_idx % NB
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a, cur_b = _read_stage(
                a_buffer.index(slot).slice(KSUB, KSUB, 1),
                b_buffer.index(slot).slice(KSUB, KSUB, 1),
                OPERAND_LAYOUT_A,
                OPERAND_LAYOUT_B,
                False,
            )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            if st < NB - 1:
                gl.amd.gfx1250.tdm.async_wait((NB - 2 - st) * TDM_OPS)
                slot = (compute_idx + 1) % NB
                cur_a, cur_b = _read_stage(
                    a_buffer.index(slot).slice(0, KSUB, 1),
                    b_buffer.index(slot).slice(0, KSUB, 1),
                    OPERAND_LAYOUT_A,
                    OPERAND_LAYOUT_B,
                    False,
                )
                compute_idx += 1
        a_scale, b_scale, bias = _load_scales(
            a_scale_ptr, b_scale_ptr, pid_m, pid_n, M, N, 1, 1, bias_ptr, 1,
            BLOCK_M, BLOCK_N, OUT_LAYOUT, ADD_BIAS, as_buffer, bs_buffer, 1,
        )
        # Every CTA of the cluster has read its slots and scales: the next tile's multicast loads may start.
        gl.amd.gfx1250.cluster.arrive()
        gl.amd.gfx1250.cluster.wait()
        nxt = tile + GRID
        if nxt < num_tiles:
            _persistent_prologue(
                a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, nxt, M, N, K, stride_am, stride_bn,
                BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B, SCALE_LAYOUT_A, SCALE_LAYOUT_B,
                a_buffer, b_buffer, as_buffer, bs_buffer, NB, FUSED_LOAD,
            )
        # The last sign change goes into the row scale.
        _scale_and_store(
            None, accumulator, -a_scale, b_scale, bias, c_ptr, pid_m, pid_n, M, N, stride_cm, 1,
            BLOCK_M, BLOCK_N, NUM_CTAS, OUT_LAYOUT, SHARED_LAYOUT_C, REGISTER_STORE, STORE_LAYOUT,
            STAGE_LAYOUT, ADD_BIAS,
        )


_KERNEL_MAP = {
    "bandwidth_bound": _gemm_a8w8_ptpc_bandwidth_bound_kernel,
    "compute_bound": _gemm_a8w8_ptpc_pipelined_kernel,
    "sliced_k": _gemm_a8w8_ptpc_pipelined_kernel,
    "compute_bound_persistent": _gemm_a8w8_ptpc_persistent_kernel,
}

# Smallest NUM_BUFFERS of each kernel type.
_MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2, "sliced_k": 2, "compute_bound_persistent": 2}
