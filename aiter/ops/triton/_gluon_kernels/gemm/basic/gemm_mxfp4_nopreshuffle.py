
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

SCALE_GROUP_ELEMS = 32


def get_gemm_mxfp4_nopreshuffle_layouts(
    num_warps: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS
    BLOCK_K_BYTES = BLOCK_K // 2

    if num_warps == 2:
        warp_bases = [[0, 1]]
        reg_bases = []
    elif num_warps == 4:
        warp_bases = [[0, 2], [2, 0]]
        reg_bases = [[1, 0], [0, 1]]
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

    # PaddedSharedLayout for A and B
    # W stored as (N, K_bytes) in LDS, loaded with .permute((1,0))
    PAD_A = 256 if BLOCK_K_BYTES <= 256 else BLOCK_K_BYTES
    shared_A = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_A, 16]], [BLOCK_M, BLOCK_K_BYTES], [1, 0])
    PAD_B = 256 if BLOCK_K_BYTES <= 256 else BLOCK_K_BYTES
    shared_B = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_B, 16]], [BLOCK_N, BLOCK_K_BYTES], [1, 0])

    dot_a_layout = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=16)
    dot_b_layout = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=16)

    a_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_a_layout, [BLOCK_M, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS)
    b_scale_layout = gl.amd.gfx1250.get_wmma_scale_layout(
        dot_b_layout, [BLOCK_N, K_GROUPS], scale_factor=SCALE_GROUP_ELEMS)

    return {
        "wmma_layout": wmma_layout,
        "wmma_acc_layout": wmma_acc_layout,
        "shared_A": shared_A,
        "shared_B": shared_B,
        "dot_a_layout": dot_a_layout,
        "dot_b_layout": dot_b_layout,
        "a_scale_layout": a_scale_layout,
        "b_scale_layout": b_scale_layout,
    }

_repr = make_kernel_repr(
    "_gemm_mxfp4_nopreshuffle_gfx1250_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
     "num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim",
     "cache_modifier", "NUM_BUFFERS"],
)


@gluon.jit(repr=_repr)
def gemm_mxfp4_nopreshuffle_gfx1250(
    a_fp4_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M, N, K_elems,
    stride_a_m, stride_a_kbytes,
    stride_b_n, stride_b_k,
    stride_c_m, stride_c_n,
    stride_as_m, stride_as_k,
    stride_bs_n, stride_bs_k,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    num_warps: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    waves_per_eu: gl.constexpr,
    num_stages: gl.constexpr,
    matrix_instr_nonkdim: gl.constexpr,
    cache_modifier: gl.constexpr,
    wmma_layout: gl.constexpr,
    wmma_acc_layout: gl.constexpr,
    shared_A: gl.constexpr,
    shared_B: gl.constexpr,
    dot_a_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    FP4_ELEMS_PER_BYTE: gl.constexpr = 2
    SCALE_GROUP_ELEMS: gl.constexpr = 32

    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // FP4_ELEMS_PER_BYTE
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_ELEMS

    gl.static_assert(BLOCK_SIZE_K % 32 == 0)
    gl.static_assert(NUM_BUFFERS >= 2)

    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_SIZE_N)
    tile_m = pid // tiles_n
    tile_n = pid - tile_m * tiles_n

    K_bytes = K_elems // FP4_ELEMS_PER_BYTE
    k_scale_cols = K_elems // SCALE_GROUP_ELEMS
    k_tiles = gl.cdiv(K_bytes, BLOCK_K_BYTES)

    # LDS — all operands through LDS via TDM (4 ops per tile)
    # B stored as (N, K_bytes), loaded with .permute((1,0))
    # Scales stored as (M/N, K_GROUPS)
    SHARED_LAYOUT_S: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    smem_A = gl.allocate_shared_memory(
        a_fp4_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_K_BYTES], layout=shared_A)
    smem_B = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N, BLOCK_K_BYTES], layout=shared_B)
    smem_AS = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_M, K_GROUPS], layout=SHARED_LAYOUT_S)
    smem_BS = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N, K_GROUPS], layout=SHARED_LAYOUT_S)

    # TDM descriptors
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_fp4_ptr, shape=(M, K_bytes),
        strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_SIZE_M, BLOCK_K_BYTES), layout=shared_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr, shape=(N, K_bytes),
        strides=(stride_b_n, stride_b_k),
        block_shape=(BLOCK_SIZE_N, BLOCK_K_BYTES), layout=shared_B)
    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr, shape=(M, k_scale_cols),
        strides=(stride_as_m, stride_as_k),
        block_shape=(BLOCK_SIZE_M, K_GROUPS), layout=SHARED_LAYOUT_S)
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr, shape=(N, k_scale_cols),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_SIZE_N, K_GROUPS), layout=SHARED_LAYOUT_S)

    load_idx = 0
    compute_idx = 0
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_acc_layout)

    NUM_TDM_OPS: gl.constexpr = 4 

    # ---- TDM Prologue ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if load_idx < k_tiles:
            slot = load_idx % NUM_BUFFERS
            a_offs = [tile_m * BLOCK_SIZE_M, load_idx * BLOCK_K_BYTES]
            b_offs = [tile_n * BLOCK_SIZE_N, load_idx * BLOCK_K_BYTES]
            as_offs = [tile_m * BLOCK_SIZE_M, load_idx * K_GROUPS]
            bs_offs = [tile_n * BLOCK_SIZE_N, load_idx * K_GROUPS]

            gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot), pred=1)
            gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot), pred=1)
            gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_AS.index(slot), pred=1)
            gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BS.index(slot), pred=1)
        load_idx += 1

    # Register pre-load prologue: wait for tile 0 then read into cur
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * NUM_TDM_OPS)

    cur_A = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_A.index(compute_idx % NUM_BUFFERS), layout=dot_a_layout)
    cur_B = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_B.index(compute_idx % NUM_BUFFERS).permute([1, 0]), layout=dot_b_layout)
    cur_AS = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_AS.index(compute_idx % NUM_BUFFERS), layout=a_scale_layout)
    cur_BS = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_BS.index(compute_idx % NUM_BUFFERS), layout=b_scale_layout)

    # ---- Main pipeline loop ----
    for _ in range(k_tiles - (NUM_BUFFERS - 1)):

        # WMMA for the current tile — operands pre-loaded last iteration
        acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, "e2m1", cur_B, cur_BS, "e2m1", acc)

        # Issue TDM for the tile (NUM_BUFFERS-1) steps ahead
        a_offs = [tile_m * BLOCK_SIZE_M, load_idx * BLOCK_K_BYTES]
        b_offs = [tile_n * BLOCK_SIZE_N, load_idx * BLOCK_K_BYTES]
        as_offs = [tile_m * BLOCK_SIZE_M, load_idx * K_GROUPS]
        bs_offs = [tile_n * BLOCK_SIZE_N, load_idx * K_GROUPS]

        gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(load_idx % NUM_BUFFERS), pred=1)
        gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(load_idx % NUM_BUFFERS), pred=1)
        gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_AS.index(load_idx % NUM_BUFFERS), pred=1)
        gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BS.index(load_idx % NUM_BUFFERS), pred=1)

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * NUM_TDM_OPS)

        load_idx += 1

        # Pre-load NEXT tile
        next_slot = (compute_idx + 1) % NUM_BUFFERS
        cur_A = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_A.index(next_slot), layout=dot_a_layout)
        cur_B = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_B.index(next_slot).permute([1, 0]), layout=dot_b_layout)
        cur_AS = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_AS.index(next_slot), layout=a_scale_layout)
        cur_BS = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_BS.index(next_slot), layout=b_scale_layout)
        compute_idx += 1

    # ---- Epilogue ---- no more TDM; drain remaining NUM_BUFFERS-1 tiles.
    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * NUM_TDM_OPS)

        next_A = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_A.index((compute_idx + 1) % NUM_BUFFERS), layout=dot_a_layout)
        next_B = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_B.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]), layout=dot_b_layout)
        next_AS = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_AS.index((compute_idx + 1) % NUM_BUFFERS), layout=a_scale_layout)
        next_BS = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_BS.index((compute_idx + 1) % NUM_BUFFERS), layout=b_scale_layout)

        acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, "e2m1", cur_B, cur_BS, "e2m1", acc)

        cur_A = next_A
        cur_B = next_B
        cur_AS = next_AS
        cur_BS = next_BS
        compute_idx += 1

    # Final WMMA for the last pre-loaded tile
    acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, "e2m1", cur_B, cur_BS, "e2m1", acc)

    out_m = tile_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M).to(gl.int64)
    out_n = tile_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N).to(gl.int64)
    mask = (out_m[:, None] < M) & (out_n[None, :] < N)
    c_offsets = (out_m[:, None] * stride_c_m + out_n[None, :] * stride_c_n).to(gl.int32)
    
    gl.amd.gfx1250.buffer_store(
        stored_value=acc.to(c_ptr.type.element_ty),
        ptr=c_ptr, offsets=c_offsets, mask=mask,
    )