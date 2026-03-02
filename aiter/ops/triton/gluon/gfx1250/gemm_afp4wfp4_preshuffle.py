from typing import Optional

import torch

from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

_LOGGER = AiterTritonLogger()

SCALE_GROUP_ELEMS = 32


def get_gemm_afp4wfp4_preshuffle_layouts(
    num_warps: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS

    blocked_mk = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    blocked_kn = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 4],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    # Raw LDS -> reg layouts (must be DistributedLayout)
    #   B_raw:   (BLOCK_N//16, BLOCK_K_BYTES*16)
    #   AS_raw:  (BLOCK_M//32, K_GROUPS*32)
    #   BS_raw:  (BLOCK_N//32, K_GROUPS*32)

    b_raw_reg_layout = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )

    # e2m1 uses instr_shape [16,16,64] for operands
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [1, 0]],
        reg_bases=[],
        instr_shape=[16, 16, 64],
    )
    # scaled WMMA accumulator must be [16,16,128]
    wmma_acc_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [1, 0]],
        reg_bases=[],
        instr_shape=[16, 16, 128],
    )

    # LDS layouts (shared memory layouts). These must be SharedLayout types.
    shared_A = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0])
    shared_B = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[0, 1])
    shared_S = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    # Dot operand layouts (register layouts expected by WMMA)
    dot_a_layout = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=16)
    dot_b_layout = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=16)

    # Register layouts for scales used by wmma_scaled
    a_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a_layout, [BLOCK_M, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS
    )
    b_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_layout, [BLOCK_N, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS
    )

    return {
        "blocked_mk": blocked_mk,
        "blocked_kn": blocked_kn,
        "b_raw_reg_layout": b_raw_reg_layout,
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


@gluon.jit
def unshuffle_a_scales(
    a_scales_shuffled,
    BLOCK_M: gl.constexpr,
    K_GROUPS: gl.constexpr,
):
    # Inverse of shuffle_scales: (BLOCK_M//32, K_GROUPS*32) -> (BLOCK_M, K_GROUPS)
    return (
        a_scales_shuffled.reshape(BLOCK_M // 32, K_GROUPS // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(BLOCK_M, K_GROUPS)
    )


@gluon.jit
def depreshuffle_b_raw_to_kn(
    b_raw,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_K_BYTES: gl.constexpr,
):
    # raw -> logical [BLOCK_K_BYTES, BLOCK_N]
    return (
        b_raw.reshape(1, BLOCK_N // 16, BLOCK_K // 64, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_N, BLOCK_K_BYTES)
        .trans(1, 0)
    )


@gluon.jit
def unshuffle_b_scales(
    b_scales_shuffled,
    BLOCK_N: gl.constexpr,
    K_GROUPS: gl.constexpr,
):
    # Inverse of shuffle_scales: (BLOCK_N//32, K_GROUPS*32) -> (BLOCK_N, K_GROUPS)
    # Matches basic kernel: reshape(n//32, k//8, 4, 16, 2, 2, 1).permute(0,5,3,1,4,2,6).reshape(n, k)
    return (
        b_scales_shuffled.reshape(BLOCK_N // 32, K_GROUPS // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(BLOCK_N, K_GROUPS)
    )


@gluon.jit
def store_c_tile(
    c_ptr,
    tile_m,
    tile_n,
    split_k_id,
    M,
    N,
    stride_c_k,
    stride_c_m,
    stride_c_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    acc,
):
    out_m = tile_m * BLOCK_M + gl.arange(0, BLOCK_M).to(gl.int64)
    out_n = tile_n * BLOCK_N + gl.arange(0, BLOCK_N).to(gl.int64)

    mask = (out_m[:, None] < M) & (out_n[None, :] < N)
    c_offsets = (
        out_m[:, None] * stride_c_m
        + out_n[None, :] * stride_c_n
        + split_k_id * stride_c_k
    ).to(gl.int32)

    gl.amd.gfx1250.buffer_store(
        stored_value=acc.to(c_ptr.type.element_ty),
        ptr=c_ptr,
        offsets=c_offsets,
        mask=mask,
    )


@gluon.jit
def gemm_afp4wfp4_preshuffle_gfx1250(
    a_fp4_ptr,
    b_preshuf_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K_elems,
    stride_a_m,
    stride_a_kbytes,
    stride_b_n16,
    stride_b_kshuf,
    stride_c_k,
    stride_c_m,
    stride_c_n,
    stride_as_m,
    stride_as_k,
    stride_bs_n,
    stride_bs_k,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    cache_modifier: gl.constexpr,
    blocked_mk: gl.constexpr,
    blocked_kn: gl.constexpr,
    b_raw_reg_layout: gl.constexpr,
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
    # Compile-time constants
    FP4_ELEMS_PER_BYTE: gl.constexpr = 2
    SCALE_GROUP_ELEMS: gl.constexpr = 32

    BLOCK_K_BYTES: gl.constexpr = BLOCK_K // FP4_ELEMS_PER_BYTE
    SPLITK_BYTES: gl.constexpr = SPLITK_BLOCK // FP4_ELEMS_PER_BYTE
    K_GROUPS: gl.constexpr = BLOCK_K // SCALE_GROUP_ELEMS

    gl.static_assert(BLOCK_K % 32 == 0)
    gl.static_assert(K_GROUPS * 32 == BLOCK_K)

    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_N)

    split_k_id = pid % NUM_KSPLIT
    tile_linear = pid // NUM_KSPLIT
    tile_m = tile_linear // tiles_n
    tile_n = tile_linear - tile_m * tiles_n

    # split-k bounds
    K_bytes = K_elems // FP4_ELEMS_PER_BYTE
    split_k0_bytes = split_k_id * SPLITK_BYTES
    if split_k0_bytes >= K_bytes:
        return

    k_tiles: gl.constexpr = (SPLITK_BYTES + BLOCK_K_BYTES - 1) // BLOCK_K_BYTES

    # LDS allocations:
    #   - A is staged into LDS
    #   - A nad B scales are staged into LDS
    #   - B is NOT staged into LDS (bypass)
    smem_A = gl.allocate_shared_memory(
        a_fp4_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K_BYTES],
        layout=shared_A,
    )
    smem_B = gl.allocate_shared_memory(
        b_preshuf_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_N // 16, BLOCK_K_BYTES * 16],
        # [NUM_BUFFERS, BLOCK_K_BYTES, BLOCK_N],
        layout=shared_B,
    )
    # scales: raw shuffled blocks into LDS via TDM, then unshuffle and re-store into LDS for layouted loads
    smem_ASraw = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M // 32, K_GROUPS * 32],
        layout=shared_S,
    )
    smem_BSraw = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_N // 32, K_GROUPS * 32],
        layout=shared_S,
    )

    # Base pointers for this split-K slice; advance by k_tile each iteration
    split_k0_groups = split_k_id * (SPLITK_BLOCK // 32)

    # -------------------- TDM descriptors --------------------
    grid_m = gl.cdiv(M, BLOCK_M) * BLOCK_M
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_fp4_ptr,
        shape=(grid_m, K_bytes),
        strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_M, BLOCK_K_BYTES),
        layout=shared_A,
    )

    grid_n16 = gl.cdiv(N, 16)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_preshuf_ptr,
        shape=(grid_n16, K_bytes * 16),
        strides=(stride_b_n16, stride_b_kshuf),
        block_shape=(BLOCK_N // 16, BLOCK_K_BYTES * 16),
        layout=shared_B,
    )

    # shuffled scales are:
    #   A_scales: (M//32, K_groups_total*32)
    #   B_scales: (N//32, K_groups_total*32)
    grid_m32 = gl.cdiv(M, 32)
    grid_n32 = gl.cdiv(N, 32)

    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr,
        shape=(grid_m32, K_elems),
        strides=(stride_as_m, stride_as_k),
        block_shape=(BLOCK_M // 32, K_GROUPS * 32),
        layout=shared_S,
    )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr,
        shape=(grid_n32, K_elems),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_N // 32, K_GROUPS * 32),
        layout=shared_S,
    )

    producer = 0
    consumer = 0

    # ---- Prologue ---- stage (NUM_BUFFERS - 1) K-tiles into LDS
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if producer < k_tiles:
            slot_p = producer
            k_tile_p = producer

            # A/B offsets (bytes-domain for A_fp4 and B_preshuf raw)
            a_offs = [tile_m * BLOCK_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
            b_offs = [
                tile_n * (BLOCK_N // 16),
                (split_k0_bytes + k_tile_p * BLOCK_K_BYTES) * 16,
            ]

            # Scale offsets are in groups-domain -> element-domain (groups*32)
            g0 = split_k0_groups + k_tile_p * K_GROUPS
            as_offs = [tile_m * (BLOCK_M // 32), g0 * 32]
            bs_offs = [tile_n * (BLOCK_N // 32), g0 * 32]

            gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_ASraw.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BSraw.index(slot_p), pred=1)

        producer += 1
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()

    # accumulator is in vGPR for the whole C tile
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=wmma_acc_layout)

    # ---- Main pipeline ----
    main_iters: gl.constexpr = k_tiles - (NUM_BUFFERS - 1)
    for _ in gl.static_range(main_iters):
        # Producer: advance pointers for this k_tile, reuse precomputed offsets
        # HBM -> vGPR -> LDS
        slot_p = producer % NUM_BUFFERS
        k_tile_p = producer

        a_offs = [tile_m * BLOCK_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
        b_offs = [
            tile_n * (BLOCK_N // 16),
            (split_k0_bytes + k_tile_p * BLOCK_K_BYTES) * 16,
        ]
        g0 = split_k0_groups + k_tile_p * K_GROUPS
        as_offs = [tile_m * (BLOCK_M // 32), g0 * 32]
        bs_offs = [tile_n * (BLOCK_N // 32), g0 * 32]

        gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
        gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
        gl.amd.gfx1250.tdm.async_load(
            as_desc, as_offs, smem_ASraw.index(slot_p), pred=1
        )
        gl.amd.gfx1250.tdm.async_load(
            bs_desc, bs_offs, smem_BSraw.index(slot_p), pred=1
        )

        producer += 1
        # Consumer: wait for data we’re about to use
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()

        slot_c = consumer % NUM_BUFFERS
        # k_tile_c = consumer

        # LDS -> vGPR
        A = smem_A.index(slot_c).load(layout=dot_a_layout)

        # B operand (raw in LDS -> depreshuffle -> logical)
        B_raw = smem_B.index(slot_c).load(layout=b_raw_reg_layout)
        B = gl.convert_layout(
            depreshuffle_b_raw_to_kn(
                B_raw, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_K_BYTES=BLOCK_K_BYTES
            ),
            layout=dot_b_layout,
        )
        # scales: raw shuffled -> unshuffle -> load with wmma scale layouts
        # AS_raw = smem_ASraw.index(slot_c).load(layout=as_raw_reg_layout)

        as_view = smem_ASraw.index(slot_c)
        as_view = (
            as_view.reshape((BLOCK_M // 32, K_GROUPS // 8, 4, 16, 2, 2, 1))
            .permute((0, 5, 3, 1, 4, 2, 6))
            .reshape((BLOCK_M, K_GROUPS))
        )
        AS = as_view.load(layout=a_scale_layout)

        BS_raw = smem_BSraw.index(slot_c)
        BS_raw = (
            BS_raw.reshape(
                (
                    BLOCK_N // 32,  # nblocks
                    K_GROUPS // 8,  # kgroups/8
                    4,
                    16,
                    2,
                    2,
                    1,
                )
            )
            .permute((0, 5, 3, 1, 4, 2, 6))
            .reshape((BLOCK_N, K_GROUPS))
        )

        # gl.barrier()

        # AS = gl.convert_layout(unshuffle_a_scales(AS_raw, BLOCK_M=BLOCK_M, K_GROUPS=K_GROUPS), layout=a_scale_layout)
        BS = BS_raw.load(layout=b_scale_layout)

        acc = gl.amd.gfx1250.wmma_scaled(A, AS, "e2m1", B, BS, "e2m1", acc)
        consumer += 1

    # ---- Drain ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if consumer < k_tiles:
            slot_c = consumer % NUM_BUFFERS

            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()

            slot_c = consumer % NUM_BUFFERS

            A = smem_A.index(slot_c).load(layout=dot_a_layout)

            B_raw = smem_B.index(slot_c).load(layout=b_raw_reg_layout)
            B = gl.convert_layout(
                depreshuffle_b_raw_to_kn(
                    B_raw, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_K_BYTES=BLOCK_K_BYTES
                ),
                layout=dot_b_layout,
            )
            as_view = smem_ASraw.index(slot_c)
            as_view = (
                as_view.reshape((BLOCK_M // 32, K_GROUPS // 8, 4, 16, 2, 2, 1))
                .permute((0, 5, 3, 1, 4, 2, 6))
                .reshape((BLOCK_M, K_GROUPS))
            )
            AS = as_view.load(layout=a_scale_layout)

            BS_raw = smem_BSraw.index(slot_c)
            BS_raw = (
                BS_raw.reshape(
                    (
                        BLOCK_N // 32,  # nblocks
                        K_GROUPS // 8,  # kgroups/8
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                )
                .permute((0, 5, 3, 1, 4, 2, 6))
                .reshape((BLOCK_N, K_GROUPS))
            )

            # gl.barrier()

            # AS = gl.convert_layout(unshuffle_a_scales(AS_raw, BLOCK_M=BLOCK_M, K_GROUPS=K_GROUPS), layout=a_scale_layout)
            BS = BS_raw.load(layout=b_scale_layout)

            acc = gl.amd.gfx1250.wmma_scaled(A, AS, "e2m1", B, BS, "e2m1", acc)

        consumer += 1

    # Store C tile
    store_c_tile(
        c_ptr=c_ptr,
        tile_m=tile_m,
        tile_n=tile_n,
        split_k_id=split_k_id,
        M=M,
        N=N,
        stride_c_k=stride_c_k,
        stride_c_m=stride_c_m,
        stride_c_n=stride_c_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        acc=acc,
    )


def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-AFP4WFP4_PRESHUFFLED", M, N, K)
    return config, is_tuned


def gemm_afp4wfp4_preshuffled_gfx1250(
    x_fp4: torch.Tensor,
    w_preshuf: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    M, K_bytes = x_fp4.shape
    n16, _ = w_preshuf.shape
    N = n16 * 16
    K_elems = K_bytes * 2

    if config is None:
        config, _ = _get_config(M, N, K_elems)

    BLOCK_M = int(config.get("BLOCK_SIZE_M", 32))
    BLOCK_N = int(config.get("BLOCK_SIZE_N", 64))
    BLOCK_K = int(config.get("BLOCK_SIZE_K", 256))
    NUM_KSPLIT = int(config.get("NUM_KSPLIT", 1))

    if NUM_KSPLIT == 1:
        if y is None:
            y = torch.empty((M, N), device=x_fp4.device, dtype=dtype or torch.bfloat16)
        stride_c_k = 0
        SPLITK_BLOCK = K_elems
    else:
        SPLITK_BLOCK = triton.cdiv(K_elems, NUM_KSPLIT)
        SPLITK_BLOCK = triton.cdiv(SPLITK_BLOCK, BLOCK_K) * BLOCK_K
        if y is None:
            y = torch.empty((NUM_KSPLIT, M, N), device=x_fp4.device, dtype=torch.float32)
        stride_c_k = y.stride(0)

    num_warps = config.get("num_warps", 4)
    layouts = get_gemm_afp4wfp4_preshuffle_layouts(num_warps, BLOCK_M, BLOCK_N, BLOCK_K)

    grid = (NUM_KSPLIT * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    gemm_afp4wfp4_preshuffle_gfx1250[grid](
        x_fp4,
        w_preshuf,
        y,
        x_scales,
        w_scales,
        M,
        N,
        K_elems,
        x_fp4.stride(0),
        x_fp4.stride(1),
        w_preshuf.stride(0),
        w_preshuf.stride(1),
        stride_c_k,
        y.stride(-2),
        y.stride(-1),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_WARPS=num_warps,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK=SPLITK_BLOCK,
        NUM_BUFFERS=2,
        cache_modifier=config.get("cache_modifier", ".ca"),
        **layouts,
    )
    return y
