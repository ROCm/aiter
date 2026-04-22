# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

SCALE_GROUP_ELEMS = 32
FP4_ELEMS_PER_BYTE = 2


def get_gemm_mxfp4_nopad_layouts(
    num_warps: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    physical_kn: bool = True,
):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS
    BLOCK_K_BYTES = BLOCK_K // FP4_ELEMS_PER_BYTE

    # Non-preshuffled scales: tiles_per_warp=1 pattern from the gfx1250
    # reference (triton-dev .../examples/gluon/mxfp_gemm_gfx1250.py). The
    # preshuffle variant uses tiles_per_warp=2 (warp_bases doubled, reg_bases
    # non-empty) to match scales packed in 32-row groups; using that pattern
    # with raw scales gives the wrong thread→scale mapping.
    if num_warps == 2:
        warp_bases = [[0, 1]]
        reg_bases = []
    elif num_warps == 4:
        warp_bases = [[0, 1], [1, 0]]
        reg_bases = []
    else:
        warp_bases = [[0, 1], [0, 2], [1, 0]]
        reg_bases = []

    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 64],
    )
    wmma_acc_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 128],
    )

    PAD_A = 256 if BLOCK_K_BYTES <= 256 else BLOCK_K_BYTES
    shared_A = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_A, 16]], [BLOCK_M, BLOCK_K_BYTES], [1, 0]
    )
    if physical_kn:
        # B is (K_bytes, N) with N as the fastest (contiguous) dim.
        shared_B = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K_BYTES, BLOCK_N], [1, 0]
        )
    else:
        # B is laid out as (N, K_bytes) with K_bytes as the fastest dim —
        # i.e. w.T of a (N, K_bytes) K-contiguous weight.  Mirror the A pad.
        PAD_B = 256 if BLOCK_K_BYTES <= 256 else BLOCK_K_BYTES
        shared_B = gl.PaddedSharedLayout.with_identity_for(
            [[PAD_B, 16]], [BLOCK_N, BLOCK_K_BYTES], [1, 0]
        )
    shared_S = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    dot_a_layout = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=16)
    dot_b_layout = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=16)

    a_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a_layout, [BLOCK_M, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS
    )
    b_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_layout, [BLOCK_N, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS
    )

    return {
        "wmma_layout": wmma_layout,
        "wmma_acc_layout": wmma_acc_layout,
        "shared_A": shared_A,
        "shared_B": shared_B,
        "shared_S": shared_S,
        "dot_a_layout": dot_a_layout,
        "dot_b_layout": dot_b_layout,
        "a_scale_layout": a_scale_layout,
        "b_scale_layout": b_scale_layout,
    }


_GLUON_REPR_KEYS = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "NUM_BUFFERS",
    "PHYSICAL_KN",
    "num_warps",
]

_gemm_mxfp4_nopad_lds_pipeline_repr = make_kernel_repr(
    "_gemm_mxfp4_nopad_gfx1250_lds_pipeline_kernel", _GLUON_REPR_KEYS
)


@gluon.jit(repr=_gemm_mxfp4_nopad_lds_pipeline_repr, loop_carried_load_percent=0)
def gemm_mxfp4_nopad_gfx1250(
    a_fp4_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K_elems,
    stride_a_m,
    stride_a_kbytes,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    stride_as_m,
    stride_as_k,
    stride_bs_n,
    stride_bs_k,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    num_warps: gl.constexpr,
    wmma_layout: gl.constexpr,
    wmma_acc_layout: gl.constexpr,
    shared_A: gl.constexpr,
    shared_B: gl.constexpr,
    shared_S: gl.constexpr,
    dot_a_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    """Local-load pipelining across K-tiles for mxfp4.

    Mirrors the a16w16 lds_pipeline kernel: manually places
    load_shared_relaxed for tile i+1 *before* the wmma_scaled for tile i so
    the hardware LDS unit and matrix unit can run in parallel.

    Each K-tile transfers 4 TDM streams (A fp4, B fp4, A scales, B scales),
    so the `async_wait` counters use `(NUM_BUFFERS-2)*4` in the steady state.

    Requires NUM_BUFFERS >= 2; >= 3 recommended.
    """
    FP4_ELEMS_PER_BYTE: gl.constexpr = 2
    SCALE_GROUP_ELEMS: gl.constexpr = 32
    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // FP4_ELEMS_PER_BYTE
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_ELEMS

    gl.static_assert(BLOCK_SIZE_K % 32 == 0)
    gl.static_assert(
        NUM_BUFFERS >= 2, "lds_pipeline kernel requires NUM_BUFFERS >= 2"
    )

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    K_bytes = K_elems // FP4_ELEMS_PER_BYTE

    # Bias each descriptor's base pointer by this block's (M, N) offset so
    # subsequent async_loads use [0, 0] and advance only along K.
    a_base = a_fp4_ptr + pid_m * BLOCK_SIZE_M * stride_a_m
    b_base = b_ptr + pid_n * BLOCK_SIZE_N * stride_b_n
    # Scales are preshuffled (factor 32): one preshuffled row packs 32 logical
    # rows, so the per-block M (or N) bias is scaled by BLOCK / PF.
    SCALE_PRESHUFFLE_FACTOR: gl.constexpr = 32
    SCALE_KWIDTH: gl.constexpr = 4
    as_base = a_scale_ptr + (
        pid_m * BLOCK_SIZE_M // SCALE_PRESHUFFLE_FACTOR
    ) * stride_as_m
    bs_base = b_scale_ptr + (
        pid_n * BLOCK_SIZE_N // SCALE_PRESHUFFLE_FACTOR
    ) * stride_bs_n

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_base,
        shape=(M - pid_m * BLOCK_SIZE_M, K_bytes),
        strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_SIZE_M, BLOCK_K_BYTES),
        layout=shared_A,
    )
    if PHYSICAL_KN:
        # B is (K_bytes, N) with N as the contiguous dim.
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(K_bytes, N - pid_n * BLOCK_SIZE_N),
            strides=(stride_b_k, stride_b_n),
            block_shape=(BLOCK_K_BYTES, BLOCK_SIZE_N),
            layout=shared_B,
        )
    else:
        # B is viewed as (N, K_bytes) with K_bytes as the contiguous dim
        # (w.T of a (N, K_bytes) K-contiguous weight — no physical copy).
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(N - pid_n * BLOCK_SIZE_N, K_bytes),
            strides=(stride_b_n, stride_b_k),
            block_shape=(BLOCK_SIZE_N, BLOCK_K_BYTES),
            layout=shared_B,
        )
    # Scales: preshuffled (factor 32), loaded directly from global into the
    # register layout wmma_scaled consumes — no LDS round-trip. The offset
    # tensor uses SliceLayout(axis, {a,b}_scale_layout) so the load result is
    # already in the right register layout.
    #
    # Preshuffled memory layout: scale at logical (m, k) lives at
    #   (m // PF) * stride_{as,bs}_m                 -- preshuffled row
    # + (k // KW) * (PF * KW)                        -- k-chunk within row
    # + (m %  PF//4) * (4 * KW)                      -- 8-way micro-row
    # + ((m % PF) // (PF // 4)) * KW                 -- 4-way lane
    # + (k %  KW)
    # where PF = SCALE_PRESHUFFLE_FACTOR (32), KW = SCALE_KWIDTH (4).
    PF: gl.constexpr = SCALE_PRESHUFFLE_FACTOR
    KW: gl.constexpr = SCALE_KWIDTH
    offs_as_m = gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, a_scale_layout)
    )
    offs_as_k = gl.arange(
        0, K_GROUPS, layout=gl.SliceLayout(0, a_scale_layout)
    )
    offs_as_tile = (
        (offs_as_m[:, None] // PF) * stride_as_m
        + (offs_as_k[None, :] // KW) * (PF * KW)
        + (offs_as_m[:, None] % (PF // 4)) * (4 * KW)
        + ((offs_as_m[:, None] % PF) // (PF // 4)) * KW
        + (offs_as_k[None, :] % KW)
    )
    offs_bs_n = gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, b_scale_layout)
    )
    offs_bs_k = gl.arange(
        0, K_GROUPS, layout=gl.SliceLayout(0, b_scale_layout)
    )
    offs_bs_tile = (
        (offs_bs_n[:, None] // PF) * stride_bs_n
        + (offs_bs_k[None, :] // KW) * (PF * KW)
        + (offs_bs_n[:, None] % (PF // 4)) * (4 * KW)
        + ((offs_bs_n[:, None] % PF) // (PF // 4)) * KW
        + (offs_bs_k[None, :] % KW)
    )
    # Per-tile K advance in preshuffled memory: K_GROUPS logical columns
    # step = (K_GROUPS / KW) k-chunks * (PF * KW) = K_GROUPS * PF.
    as_k_step: gl.constexpr = K_GROUPS * PF
    bs_k_step: gl.constexpr = K_GROUPS * PF

    a_buffer = gl.allocate_shared_memory(
        a_fp4_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_K_BYTES],
        layout=shared_A,
    )
    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K_BYTES, BLOCK_SIZE_N],
            layout=shared_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_SIZE_N, BLOCK_K_BYTES],
            layout=shared_B,
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_acc_layout
    )

    # Issue scale buffer_loads for tile 0 UP FRONT, before the TDM prologue.
    # The TDM prologue + async_wait below (and the tile-0 ds_read) give the
    # long-latency scale loads time to complete before the peeled WMMA.
    cur_as = gl.amd.gfx1250.buffer_load(as_base, offs_as_tile)
    cur_bs = gl.amd.gfx1250.buffer_load(bs_base, offs_bs_tile)

    # TDM prologue: fill the pipeline with NUM_BUFFERS-1 tiles.
    # Only A/B go through TDM now — scales are pulled directly from global at
    # WMMA time, so 2 TDM ops per tile (not 4).
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )

        a_desc = gl.amd.gfx1250.tdm.advance(
            a_desc, [0, BLOCK_K_BYTES], update_bounds=False
        )
        if PHYSICAL_KN:
            b_desc = gl.amd.gfx1250.tdm.advance(
                b_desc, [BLOCK_K_BYTES, 0], update_bounds=False
            )
        else:
            b_desc = gl.amd.gfx1250.tdm.advance(
                b_desc, [0, BLOCK_K_BYTES], update_bounds=False
            )

        load_idx += 1

    num_k_tiles = gl.cdiv(K_bytes, BLOCK_K_BYTES)

    # Register pre-load prologue: wait for tile 0 then read A/B from LDS and
    # scales directly from global via buffer_load.
    # 2 TDM ops/tile in flight, so we wait on (NUM_BUFFERS-2)*2 to let tile 0
    # complete.
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(compute_idx % NUM_BUFFERS), dot_a_layout
    )
    if PHYSICAL_KN:
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS), dot_b_layout
        )
    else:
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
            dot_b_layout,
        )
    # cur_as / cur_bs were pre-issued before the TDM prologue.

    # ---- Peeled first iteration ----
    # Issue NEXT tile's scale buffer_loads BEFORE the WMMA so memory latency
    # overlaps with the matrix unit's execution of the current WMMA.
    next_as = gl.amd.gfx1250.buffer_load(
        as_base, offs_as_tile + (compute_idx + 1) * as_k_step
    )
    next_bs = gl.amd.gfx1250.buffer_load(
        bs_base, offs_bs_tile + (compute_idx + 1) * bs_k_step
    )

    # WMMA for the current tile — uses operands pre-loaded above so no
    # ds_read stall before the matrix op.
    accumulator = gl.amd.gfx1250.wmma_scaled(
        cur_a, cur_as, "e2m1", cur_b, cur_bs, "e2m1", accumulator
    )

    # Issue TDM for the tile that is (NUM_BUFFERS-1) steps ahead.
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )

    a_desc = gl.amd.gfx1250.tdm.advance(
        a_desc, [0, BLOCK_K_BYTES], update_bounds=False
    )
    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.advance(
            b_desc, [BLOCK_K_BYTES, 0], update_bounds=False
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.advance(
            b_desc, [0, BLOCK_K_BYTES], update_bounds=False
        )

    # 2 TDM ops/tile; after the new batch there are (NUM_BUFFERS-1)*2
    # in-flight, waiting for (NUM_BUFFERS-2)*2 lands tile compute_idx+1.
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    load_idx += 1

    # Pre-load the NEXT tile's A/B operands into registers *before* the WMMA
    # in the next iteration. (Scales were already issued above.)
    next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_a_layout
    )
    if PHYSICAL_KN:
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_b_layout
        )
    else:
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
            dot_b_layout,
        )

    cur_a = next_a
    cur_b = next_b
    cur_as = next_as
    cur_bs = next_bs
    compute_idx += 1

    # ---- Remaining main-loop iterations ----
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1) - 1):

        # Issue next tile's scale loads first so their memory latency
        # overlaps with the current WMMA and the subsequent TDM/wait.
        next_as = gl.amd.gfx1250.buffer_load(
            as_base, offs_as_tile + (compute_idx + 1) * as_k_step
        )
        next_bs = gl.amd.gfx1250.buffer_load(
            bs_base, offs_bs_tile + (compute_idx + 1) * bs_k_step
        )

        accumulator = gl.amd.gfx1250.wmma_scaled(
            cur_a, cur_as, "e2m1", cur_b, cur_bs, "e2m1", accumulator
        )

        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )

        a_desc = gl.amd.gfx1250.tdm.advance(
            a_desc, [0, BLOCK_K_BYTES], update_bounds=False
        )
        if PHYSICAL_KN:
            b_desc = gl.amd.gfx1250.tdm.advance(
                b_desc, [BLOCK_K_BYTES, 0], update_bounds=False
            )
        else:
            b_desc = gl.amd.gfx1250.tdm.advance(
                b_desc, [0, BLOCK_K_BYTES], update_bounds=False
            )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

        load_idx += 1

        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_a_layout
        )
        if PHYSICAL_KN:
            next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_b_layout
            )
        else:
            next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
                dot_b_layout,
            )

        cur_a = next_a
        cur_b = next_b
        cur_as = next_as
        cur_bs = next_bs
        compute_idx += 1

    # Epilogue: no more TDM loads; drain the remaining NUM_BUFFERS-1 tiles.
    # The first NUM_BUFFERS-2 iterations still use the pre-load / WMMA pattern.
    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)

        # Issue next tile's scale loads early to overlap memory latency with
        # the WMMA and ds_reads below.
        next_as = gl.amd.gfx1250.buffer_load(
            as_base, offs_as_tile + (compute_idx + 1) * as_k_step
        )
        next_bs = gl.amd.gfx1250.buffer_load(
            bs_base, offs_bs_tile + (compute_idx + 1) * bs_k_step
        )

        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_a_layout
        )
        if PHYSICAL_KN:
            next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index((compute_idx + 1) % NUM_BUFFERS), dot_b_layout
            )
        else:
            next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
                dot_b_layout,
            )

        accumulator = gl.amd.gfx1250.wmma_scaled(
            cur_a, cur_as, "e2m1", cur_b, cur_bs, "e2m1", accumulator
        )

        cur_a = next_a
        cur_b = next_b
        cur_as = next_as
        cur_bs = next_bs
        compute_idx += 1

    # Final WMMA for the last pre-loaded tile.
    accumulator = gl.amd.gfx1250.wmma_scaled(
        cur_a, cur_as, "e2m1", cur_b, cur_bs, "e2m1", accumulator
    )

    if NUM_BUFFERS > 2:
        gl.amd.sched_barrier(0)

    # C store: build the offset tensor in the accumulator's own layout so
    # buffer_store doesn't have to convert between layouts. Using
    # SliceLayout(axis, wmma_acc_layout) keeps offs_c aligned with the
    # accumulator's distribution.
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_acc_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_acc_layout)
    )
    offs_c = stride_c_m * offs_cm[:, None] + stride_c_n * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty), c_ptr, offs_c, mask=mask_c
    )
