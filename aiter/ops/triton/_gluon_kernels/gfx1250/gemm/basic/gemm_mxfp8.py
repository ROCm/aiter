# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid

# wmma_scaled always consumes one e8m0 byte per 32 elements along K, whatever
# granularity the scales were produced at. Coarser scales are broadcast on load.
SCALE_GROUP_SIZE = 32
# B (weights) are blockscale: one e8m0 byte per 128(N) x 128(K) block.
B_SCALE_N_GROUP = 128
B_SCALE_K_GROUP = 128

_PRESHUFFLE_GLUON_REPR_KEYS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "A_SCALE_K_GROUP",
    "NUM_KSPLIT",
    "SPLITK_BLOCK_SIZE",
    "EVEN_K",
    "num_warps",
    "waves_per_eu",
    "cache_modifier",
    "NUM_BUFFERS",
]

_gemm_mxfp8_preshuffle_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel",
    _PRESHUFFLE_GLUON_REPR_KEYS,
)


@gluon.jit
def depreshuffle_b(
    smem_b_raw,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
):
    """Unshuffle a preshuffled weight tile in shared memory.

    Host shuffle (aiter.ops.shuffle.shuffle_weight, layout=(16, 16)):
        (N//16, 16, K//32, 2, 16) -> permute(0, 2, 3, 1, 4) -> (N//16, K*16)
    Inverse:
        (N//16, K//32, 2, 16, 16) -> permute(0, 3, 1, 2, 4)
        -> (N, K) then transpose to (K, N)

    Elements are 1 byte for fp8, so the element and byte views coincide and this
    is identical to the a8w8 blockscale unshuffle. Pure reindexing of the LDS
    view -- no data movement.
    """
    return (
        smem_b_raw.reshape((BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 32, 2, 16, 16))
        .permute((0, 3, 1, 2, 4))
        .reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
        .permute((1, 0))
    )


@gluon.jit
def _load_scale_tile(
    scale_ptr,
    tile_idx,
    k_split_offset,
    K,
    row_off,
    offs_kg,
    stride_k,
    BLOCK_SIZE_K: gl.constexpr,
    SCALE_K_GROUP: gl.constexpr,
    out_layout: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """Load a (BLOCK, BLOCK_SIZE_K // 32) e8m0 scale tile in the wmma scale layout.

    ``SCALE_K_GROUP`` is how many K elements share one scale byte in the source
    tensor (32 for MX scales, 128 for blockscale). When it exceeds 32 the same
    byte is fetched by several of the 32-element groups -- redundant but
    cache-resident, and exactly what the triton kernel's scale pointer
    arithmetic does. ``row_off`` is the already-strided row offset, so the
    caller decides whether the source is row-major or transposed.

    The K index is clamped to the last valid group rather than masked: TDM
    zero-fills the operand past K, so the tail group contributes nothing and the
    clamp keeps every address in bounds.
    """
    SCALE_GROUP_SIZE: gl.constexpr = 32

    k_base = k_split_offset + tile_idx * BLOCK_SIZE_K
    kg_max = (K + SCALE_K_GROUP - 1) // SCALE_K_GROUP - 1
    kg_idx = (k_base + offs_kg * SCALE_GROUP_SIZE) // SCALE_K_GROUP
    kg_idx = gl.minimum(kg_idx, kg_max)

    ptrs = scale_ptr + row_off[:, None] + kg_idx[None, :] * stride_k
    return gl.convert_layout(gl.load(ptrs, cache_modifier=cache_modifier), out_layout)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@gluon.jit(repr=_gemm_mxfp8_preshuffle_bandwidth_bound_repr)
def _gemm_mxfp8_preshuffle_bandwidth_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    A_SCALE_K_GROUP: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    waves_per_eu: gl.constexpr = 0,
):
    """
    Gluon gfx1250 kernel for FP8 x FP8 GEMM with preshuffled weights.

    A is fp8 e4m3 (M, K). Its e8m0 scales cover ``A_SCALE_K_GROUP`` elements
    along K -- 32 for MX activations (``(M, K//32)``) or 128 for blockscale
    activations (``(M, K//128)``). The wrapper folds a transposed scale buffer
    into ``stride_asm`` / ``stride_ask``, so both layouts land here identically.

    B is fp8 e4m3, preshuffled on the host into (N // 16, K * 16), with coarse
    128x128 e8m0 scales stored compact as (N // 128, K // 128).

    Both scale operands ride inside gl.amd.gfx1250.wmma_scaled, so unlike the
    a8w8 blockscale kernel there is no per-K-tile `acc += res * scale` multiply
    and no reason to cap BLOCK_SIZE_N / BLOCK_SIZE_K at the scale group size.

    Pipelining follows the a8w8 blockscale bandwidth_bound kernel: TDM streams
    A and B NUM_BUFFERS deep; the prologue fills NUM_BUFFERS - 1 slots, the main
    loop runs NUM_K_ITER - (NUM_BUFFERS - 1) times, then the epilogue drains.
    Requires NUM_K_ITER >= NUM_BUFFERS - 1 (the wrapper clamps NUM_BUFFERS).

    waves_per_eu is deliberately unread in the body. It is a HIPOptions field,
    so triton forwards it to the AMD backend, which emits it as the
    amdgpu-waves-per-eu LLVM function attribute (an occupancy hint); 0 means no
    attribute is emitted. Declaring it as a constexpr as well puts it in the
    specialization key and in the kernel name via make_kernel_repr, so tuned
    variants stay distinguishable in traces and caches.
    """
    SCALE_GROUP_SIZE: gl.constexpr = 32
    B_SCALE_N_GROUP: gl.constexpr = 128
    B_SCALE_K_GROUP: gl.constexpr = 128
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

    # The wmma instruction shape is [16, 16, 128], so a K tile must be a whole
    # number of k-steps; 128 also keeps K_GROUPS aligned to the 32-element MX
    # groups and to the 128-element scale blocks.
    gl.static_assert(BLOCK_SIZE_K % 128 == 0)
    gl.static_assert(BLOCK_SIZE_N % 16 == 0)
    gl.static_assert(NUM_BUFFERS >= 2)
    gl.static_assert(A_SCALE_K_GROUP % SCALE_GROUP_SIZE == 0)

    # ---- program setup: split-K decomposition ----
    pid_unified = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_m * num_pid_n
    pid_k = pid_unified // GRID_MN
    pid = pid_unified % GRID_MN

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    k_split_offset = pid_k * SPLITK_BLOCK_SIZE
    K_local = K - k_split_offset
    if NUM_KSPLIT > 1:
        K_local = SPLITK_BLOCK_SIZE

    NUM_K_ITER = gl.cdiv(K_local, BLOCK_SIZE_K)

    # ---- layouts ----
    # fp8 operands: 1 byte each, so both sides use the k=128 wmma instruction
    # shape with k_width=16 (matching the fp8 side of the a8w4 MoE kernel).
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=16
    )
    a_scale_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a_layout, [BLOCK_SIZE_M, K_GROUPS], scale_factor=SCALE_GROUP_SIZE
    )
    b_scale_layout: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_layout, [BLOCK_SIZE_N, K_GROUPS], scale_factor=SCALE_GROUP_SIZE
    )

    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )

    # ---- scale addressing (see _load_scale_tile) ----
    as_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, K_GROUPS], [32, 1], [num_warps, 1], [1, 0]
    )
    bs_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, K_GROUPS], [32, 1], [num_warps, 1], [1, 0]
    )
    # `% M` rather than a mask: TDM zero-fills the A tile past M, so a wrapped
    # scale row multiplies zero data.
    offs_as_m = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, as_load_layout))
    ) % M
    offs_as_kg = gl.arange(0, K_GROUPS, layout=gl.SliceLayout(0, as_load_layout))
    as_row_off = offs_as_m * stride_asm

    offs_bs_n = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, bs_load_layout))
    ) % N
    offs_bs_kg = gl.arange(0, K_GROUPS, layout=gl.SliceLayout(0, bs_load_layout))
    bs_row_off = (offs_bs_n // B_SCALE_N_GROUP) * stride_bsn

    # ---- TDM descriptors ----
    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bn_tdm = pid_n * (BLOCK_SIZE_N // 16)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + k_split_offset * stride_ak,
        shape=(M, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        layout=tdm_shared_a,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + k_split_offset * 16 * stride_bk,
        shape=(gl.cdiv(N, 16), K_local * 16),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16),
        layout=tdm_shared_b,
    )

    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_K],
        layout=tdm_shared_a,
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16],
        layout=tdm_shared_b,
    )

    num_loads = 0
    num_computes = 0

    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_layout)

    # ---------------- Prologue ----------------
    for _ in gl.static_range(NUM_BUFFERS - 1):
        slot = num_loads % NUM_BUFFERS
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [off_am_tdm, num_loads * BLOCK_SIZE_K], tdm_smem_a.index(slot)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16], tdm_smem_b.index(slot)
        )
        num_loads += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    slot_c = num_computes % NUM_BUFFERS
    cur_a = tdm_smem_a.index(slot_c).load(layout=dot_a_layout)
    cur_b = depreshuffle_b(
        tdm_smem_b.index(slot_c),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    ).load(layout=dot_b_layout)

    # ---------------- Main loop ----------------
    for _ in range(NUM_K_ITER - (NUM_BUFFERS - 1)):
        cur_as = _load_scale_tile(
            a_scale_ptr,
            num_computes,
            k_split_offset,
            K,
            as_row_off,
            offs_as_kg,
            stride_ask,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            SCALE_K_GROUP=A_SCALE_K_GROUP,
            out_layout=a_scale_layout,
            cache_modifier=cache_modifier,
        )
        cur_bs = _load_scale_tile(
            b_scale_ptr,
            num_computes,
            k_split_offset,
            K,
            bs_row_off,
            offs_bs_kg,
            stride_bsk,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            SCALE_K_GROUP=B_SCALE_K_GROUP,
            out_layout=b_scale_layout,
            cache_modifier=cache_modifier,
        )
        acc = gl.amd.gfx1250.wmma_scaled(
            cur_a, cur_as, "e4m3", cur_b, cur_bs, "e4m3", acc
        )

        slot = num_loads % NUM_BUFFERS
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_a.index(slot),
            pred=1,
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16],
            tdm_smem_b.index(slot),
            pred=1,
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1

        next_slot = (num_computes + 1) % NUM_BUFFERS
        cur_a = tdm_smem_a.index(next_slot).load(layout=dot_a_layout)
        cur_b = depreshuffle_b(
            tdm_smem_b.index(next_slot),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)
        num_computes += 1

    # ---------------- Epilogue ----------------
    for i in gl.static_range(NUM_BUFFERS - 2):
        cur_as = _load_scale_tile(
            a_scale_ptr,
            num_computes,
            k_split_offset,
            K,
            as_row_off,
            offs_as_kg,
            stride_ask,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            SCALE_K_GROUP=A_SCALE_K_GROUP,
            out_layout=a_scale_layout,
            cache_modifier=cache_modifier,
        )
        cur_bs = _load_scale_tile(
            b_scale_ptr,
            num_computes,
            k_split_offset,
            K,
            bs_row_off,
            offs_bs_kg,
            stride_bsk,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            SCALE_K_GROUP=B_SCALE_K_GROUP,
            out_layout=b_scale_layout,
            cache_modifier=cache_modifier,
        )
        acc = gl.amd.gfx1250.wmma_scaled(
            cur_a, cur_as, "e4m3", cur_b, cur_bs, "e4m3", acc
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)

        next_slot = (num_computes + 1) % NUM_BUFFERS
        cur_a = tdm_smem_a.index(next_slot).load(layout=dot_a_layout)
        cur_b = depreshuffle_b(
            tdm_smem_b.index(next_slot),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)
        num_computes += 1

    # ---------------- Final WMMA ----------------
    cur_as = _load_scale_tile(
        a_scale_ptr,
        num_computes,
        k_split_offset,
        K,
        as_row_off,
        offs_as_kg,
        stride_ask,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SCALE_K_GROUP=A_SCALE_K_GROUP,
        out_layout=a_scale_layout,
        cache_modifier=cache_modifier,
    )
    cur_bs = _load_scale_tile(
        b_scale_ptr,
        num_computes,
        k_split_offset,
        K,
        bs_row_off,
        offs_bs_kg,
        stride_bsk,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SCALE_K_GROUP=B_SCALE_K_GROUP,
        out_layout=b_scale_layout,
        cache_modifier=cache_modifier,
    )
    acc = gl.amd.gfx1250.wmma_scaled(cur_a, cur_as, "e4m3", cur_b, cur_bs, "e4m3", acc)

    # ---------------- Store ----------------
    # c_ptr is the (M, N) output for NUM_KSPLIT == 1, or the fp32 partial slab
    # c_ptr + pid_k * stride_ck otherwise (a downstream reduce sums the slabs).
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_k * stride_ck,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


_PRESHUFFLE_KERNEL_MAP = {
    "bandwidth_bound": _gemm_mxfp8_preshuffle_bandwidth_bound_kernel,
}
