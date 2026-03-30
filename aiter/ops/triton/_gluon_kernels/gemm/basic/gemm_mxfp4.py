from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

SCALE_GROUP_ELEMS = 32


def get_gemm_afp4wfp4_preshuffle_layouts(
    num_warps: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS

    if num_warps == 2:
        warp_bases= [[0, 1]]
        reg_bases= []
    elif num_warps == 4:
        warp_bases= [[0, 2], [2, 0]]
        reg_bases= [[1,0],[0,1]]
    else:
        warp_bases = [[0, 1], [0, 2], [1, 0]]
        reg_bases = []
        

    # e2m1 uses instr_shape [16,16,64] for operands
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 64],
    )
    # scaled WMMA accumulator must be [16,16,128]
    wmma_acc_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 128],
    )

    # LDS layouts (shared memory layouts). These must be SharedLayout types.
    shared_A = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0])
    shared_B = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0])
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
def depreshuffle_b_raw_to_kn(
    b_raw,
    BLOCK_N: gl.constexpr,
    BLOCK_K_BYTES: gl.constexpr,
):
    # raw -> logical [BLOCK_K_BYTES, BLOCK_N]
    return (
        b_raw.reshape((BLOCK_N // 16, BLOCK_K_BYTES // 32, 2, 16, 16))
        .permute((0, 3, 1, 2, 4))
        .reshape((BLOCK_N, BLOCK_K_BYTES))
        .permute((1, 0))
    )


@gluon.jit
def unswizzle_mx_scale_gfx1250(
    scale_buffer_slice,
    BLOCK_N: gl.constexpr,
    MX_SCALE_BLOCK_K: gl.constexpr,
    PRESHUFFLE_FACTOR: gl.constexpr,
    SCALE_KWIDTH: gl.constexpr,
):
    return (
        scale_buffer_slice.reshape(
            (
                BLOCK_N // PRESHUFFLE_FACTOR,
                MX_SCALE_BLOCK_K // SCALE_KWIDTH,
                PRESHUFFLE_FACTOR // 4,
                4,
                SCALE_KWIDTH,
            )
        )
        .permute((0, 3, 2, 1, 4))
        .reshape((BLOCK_N, MX_SCALE_BLOCK_K))
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

_gemm_mxfp4_preshuffle_gfx1250_repr = make_kernel_repr(
    "_gemm_mxfp4_preshuffle_gfx1250_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)

@gluon.jit(repr=_gemm_mxfp4_preshuffle_gfx1250_repr)
def gemm_mxfp4_preshuffle_gfx1250(
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
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    SPLITK_BLOCK: gl.constexpr,
    num_warps: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    cache_modifier: gl.constexpr,
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

    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // FP4_ELEMS_PER_BYTE
    SPLITK_BYTES: gl.constexpr = SPLITK_BLOCK // FP4_ELEMS_PER_BYTE
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_ELEMS

    gl.static_assert(BLOCK_SIZE_K % 32 == 0)
    gl.static_assert(K_GROUPS * 32 == BLOCK_SIZE_K)

    # Scale preshuffle constants for gfx1250
    PRESHUFFLE_FACTOR: gl.constexpr = 32
    SCALE_KWIDTH: gl.constexpr = 4 if K_GROUPS >= 4 else K_GROUPS

    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_SIZE_N)

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
    # Base pointers for this split-K slice; advance by k_tile each iteration
    split_k0_groups = split_k_id * (SPLITK_BLOCK // 32)

    # LDS allocations:
    #   - A is staged into LDS
    #   - A nad B scales are staged into LDS
    smem_A = gl.allocate_shared_memory(
        a_fp4_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_K_BYTES],
        layout=shared_A,
    )
    smem_B = gl.allocate_shared_memory(
        b_preshuf_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N // 16, BLOCK_K_BYTES * 16],
        layout=shared_B,
    )
    # A scales: M>=32 uses preshuffled (M//PRESHUFFLE_FACTOR, K*PRESHUFFLE_FACTOR) layout;
    #           M<32 uses (M, K//32) per row
    if BLOCK_SIZE_M < 32:
        smem_ASraw = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [NUM_BUFFERS, BLOCK_SIZE_M, K_GROUPS],
            layout=shared_S,
        )
    else:
        smem_ASraw = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [NUM_BUFFERS, BLOCK_SIZE_M // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR],
            layout=shared_S,
        )
    smem_BSraw = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR],
        layout=shared_S,
    )

    # -------------------- TDM descriptors --------------------
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_fp4_ptr,
        shape=(M, K_bytes),
        strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_SIZE_M, BLOCK_K_BYTES),
        layout=shared_A,
    )

    grid_n16 = gl.cdiv(N, 16)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_preshuf_ptr,
        shape=(grid_n16, K_bytes * 16),
        strides=(stride_b_n16, stride_b_kshuf),
        block_shape=(BLOCK_SIZE_N // 16, BLOCK_K_BYTES * 16),
        layout=shared_B,
    )

    grid_m_as = gl.cdiv(M, PRESHUFFLE_FACTOR)
    grid_n_bs = gl.cdiv(N, PRESHUFFLE_FACTOR)
    k_scale_cols = K_elems // SCALE_GROUP_ELEMS

    if BLOCK_SIZE_M < 32:
        as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_scale_ptr,
            shape=(M, k_scale_cols),
            strides=(stride_as_m, stride_as_k),
            block_shape=(BLOCK_SIZE_M, K_GROUPS),
            layout=shared_S,
        )
    else:
        as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_scale_ptr,
            shape=(grid_m_as, k_scale_cols * PRESHUFFLE_FACTOR),
            strides=(stride_as_m, stride_as_k),
            block_shape=(BLOCK_SIZE_M // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR),
            layout=shared_S,
        )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr,
        shape=(grid_n_bs, k_scale_cols * PRESHUFFLE_FACTOR),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_SIZE_N // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR),
        layout=shared_S,
    )

    k_tile_load_idx = 0
    k_tile_compute_idx = 0

    # ---- Prologue ---- stage (NUM_BUFFERS - 1) K-tiles into LDS
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if k_tile_load_idx < k_tiles:
            slot_p = k_tile_load_idx
            k_tile_p = k_tile_load_idx

            # A/B offsets (bytes-domain for A_fp4 and B_preshuf raw)
            a_offs = [tile_m * BLOCK_SIZE_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
            b_offs = [
                tile_n * (BLOCK_SIZE_N // 16),
                (split_k0_bytes + k_tile_p * BLOCK_K_BYTES) * 16,
            ]

            # Scale offsets in preshuffled domain
            g0 = split_k0_groups + k_tile_p * K_GROUPS
            if BLOCK_SIZE_M < 32:
                as_offs = [tile_m * BLOCK_SIZE_M, g0]
            else:
                as_offs = [tile_m * (BLOCK_SIZE_M // PRESHUFFLE_FACTOR), g0 * PRESHUFFLE_FACTOR]
            bs_offs = [tile_n * (BLOCK_SIZE_N // PRESHUFFLE_FACTOR), g0 * PRESHUFFLE_FACTOR]

            gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_ASraw.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BSraw.index(slot_p), pred=1)

        k_tile_load_idx += 1

    # accumulator is in vGPR for the whole C tile
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_acc_layout)

    # ---- Main pipeline ----
    main_iters: gl.constexpr = k_tiles - (NUM_BUFFERS - 1)
    for _ in range(main_iters):
        # Load: advance pointers for this k_tile
        # HBM -> vGPR -> LDS
        slot_p = k_tile_load_idx % NUM_BUFFERS
        k_tile_p = k_tile_load_idx

        a_offs = [tile_m * BLOCK_SIZE_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
        b_offs = [
            tile_n * (BLOCK_SIZE_N // 16),
            (split_k0_bytes + k_tile_p * BLOCK_K_BYTES) * 16,
        ]
        g0 = split_k0_groups + k_tile_p * K_GROUPS
        if BLOCK_SIZE_M < 32:
            as_offs = [tile_m * BLOCK_SIZE_M, g0]
        else:
            as_offs = [tile_m * (BLOCK_SIZE_M // PRESHUFFLE_FACTOR), g0 * PRESHUFFLE_FACTOR]
        bs_offs = [tile_n * (BLOCK_SIZE_N // PRESHUFFLE_FACTOR), g0 * PRESHUFFLE_FACTOR]

        gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
        gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
        gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_ASraw.index(slot_p), pred=1)
        gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BSraw.index(slot_p), pred=1)

        k_tile_load_idx += 1
        # Compute: wait for data we’re about to use
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

        slot_c = k_tile_compute_idx % NUM_BUFFERS

        # LDS -> vGPR
        A = smem_A.index(slot_c).load(layout=dot_a_layout)

        # B operand (raw unshuffle -> logical)
        B = depreshuffle_b_raw_to_kn(
                smem_B.index(slot_c), BLOCK_N=BLOCK_SIZE_N, BLOCK_K_BYTES=BLOCK_K_BYTES
            ).load(layout=dot_b_layout)

        # scales: unswizzle -> load with wmma scale layouts
        if BLOCK_SIZE_M < 32:
            AS = smem_ASraw.index(slot_c).load(layout=a_scale_layout)
        else:
            AS = unswizzle_mx_scale_gfx1250(
                smem_ASraw.index(slot_c), BLOCK_SIZE_M, K_GROUPS,
                PRESHUFFLE_FACTOR, SCALE_KWIDTH,
            ).load(layout=a_scale_layout)
        BS = unswizzle_mx_scale_gfx1250(
            smem_BSraw.index(slot_c), BLOCK_SIZE_N, K_GROUPS,
            PRESHUFFLE_FACTOR, SCALE_KWIDTH,
        ).load(layout=b_scale_layout)

        acc = gl.amd.gfx1250.wmma_scaled(A, AS, "e2m1", B, BS, "e2m1", acc)
        k_tile_compute_idx += 1

    # ---- Drain ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if k_tile_compute_idx < k_tiles:
            slot_c = k_tile_compute_idx % NUM_BUFFERS

            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

            slot_c = k_tile_compute_idx % NUM_BUFFERS

            A = smem_A.index(slot_c).load(layout=dot_a_layout)

            B = depreshuffle_b_raw_to_kn(
                smem_B.index(slot_c), BLOCK_N=BLOCK_SIZE_N, BLOCK_K_BYTES=BLOCK_K_BYTES
            ).load(layout=dot_b_layout)

            if BLOCK_SIZE_M < 32:
                AS = smem_ASraw.index(slot_c).load(layout=a_scale_layout)
            else:
                AS = unswizzle_mx_scale_gfx1250(
                    smem_ASraw.index(slot_c), BLOCK_SIZE_M, K_GROUPS,
                    PRESHUFFLE_FACTOR, SCALE_KWIDTH,
                ).load(layout=a_scale_layout)
            BS = unswizzle_mx_scale_gfx1250(
                smem_BSraw.index(slot_c), BLOCK_SIZE_N, K_GROUPS,
                PRESHUFFLE_FACTOR, SCALE_KWIDTH,
            ).load(layout=b_scale_layout)

            acc = gl.amd.gfx1250.wmma_scaled(A, AS, "e2m1", B, BS, "e2m1", acc)

        k_tile_compute_idx += 1

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
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        acc=acc,
    )