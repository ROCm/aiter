
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

SCALE_GROUP_ELEMS = 32


def get_gemm_mxfp4_nopad_layouts(
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

    # PaddedSharedLayout for both A and B — avoids bank conflicts via padding
    PAD_A = 256 if BLOCK_K_BYTES <= 256 else BLOCK_K_BYTES
    shared_A = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_A, 16]], [BLOCK_M, BLOCK_K_BYTES], [1, 0])
    shared_B = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_K_BYTES, BLOCK_N], [1, 0])
    shared_S = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

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
        "shared_S": shared_S,
        "dot_a_layout": dot_a_layout,
        "dot_b_layout": dot_b_layout,
        "a_scale_layout": a_scale_layout,
        "b_scale_layout": b_scale_layout,
    }


@gluon.jit
def store_c_tile(
    c_ptr, tile_m, tile_n, split_k_id, M, N,
    stride_c_k, stride_c_m, stride_c_n,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, acc,
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
        ptr=c_ptr, offsets=c_offsets, mask=mask,
    )


_repr = make_kernel_repr(
    "_gemm_mxfp4_nopad_gfx1250_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
     "num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim",
     "cache_modifier", "NUM_KSPLIT", "NUM_BUFFERS"],
)


@gluon.jit(repr=_repr, loop_carried_load_percent=0)
def gemm_mxfp4_nopad_gfx1250(
    a_fp4_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M, N, K_elems,
    stride_a_m, stride_a_kbytes,
    stride_b_k, stride_b_n,
    stride_c_k, stride_c_m, stride_c_n,
    stride_as_m, stride_as_k,
    stride_bs_n, stride_bs_k,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    SPLITK_BLOCK: gl.constexpr,
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
    shared_S: gl.constexpr,
    dot_a_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    FP4_ELEMS_PER_BYTE: gl.constexpr = 2
    SCALE_GROUP_ELEMS: gl.constexpr = 32

    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K // FP4_ELEMS_PER_BYTE
    SPLITK_BYTES: gl.constexpr = SPLITK_BLOCK // FP4_ELEMS_PER_BYTE
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_ELEMS

    gl.static_assert(BLOCK_SIZE_K % 32 == 0)
    gl.static_assert(NUM_BUFFERS >= 2)

    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_SIZE_N)
    split_k_id = pid % NUM_KSPLIT
    tile_linear = pid // NUM_KSPLIT
    tile_m = tile_linear // tiles_n
    tile_n = tile_linear - tile_m * tiles_n

    K_bytes = K_elems // FP4_ELEMS_PER_BYTE
    split_k0_bytes = split_k_id * SPLITK_BYTES
    if split_k0_bytes >= K_bytes:
        return

    k_tiles: gl.constexpr = (SPLITK_BYTES + BLOCK_K_BYTES - 1) // BLOCK_K_BYTES
    split_k0_groups = split_k_id * (SPLITK_BLOCK // 32)

    # LDS — simple 2D shapes with PaddedSharedLayout, no preshuffle
    smem_A = gl.allocate_shared_memory(
        a_fp4_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_K_BYTES], layout=shared_A)
    smem_B = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_K_BYTES, BLOCK_SIZE_N], layout=shared_B)
    smem_AS = gl.allocate_shared_memory(
        a_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_M, K_GROUPS], layout=shared_S)
    smem_BS = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N, K_GROUPS], layout=shared_S)

    # TDM descriptors — straightforward 2D
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_fp4_ptr, shape=(M, K_bytes),
        strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_SIZE_M, BLOCK_K_BYTES), layout=shared_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr, shape=(K_bytes, N),
        strides=(stride_b_k, stride_b_n),
        block_shape=(BLOCK_K_BYTES, BLOCK_SIZE_N), layout=shared_B)

    k_scale_cols = K_elems // SCALE_GROUP_ELEMS
    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr, shape=(M, k_scale_cols),
        strides=(stride_as_m, stride_as_k),
        block_shape=(BLOCK_SIZE_M, K_GROUPS), layout=shared_S)
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr, shape=(N, k_scale_cols),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_SIZE_N, K_GROUPS), layout=shared_S)

    load_idx = 0
    compute_idx = 0
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_acc_layout)

    # ---- TDM Prologue ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if load_idx < k_tiles:
            slot_p = load_idx % NUM_BUFFERS
            k_tile_p = load_idx
            a_offs = [tile_m * BLOCK_SIZE_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
            b_offs = [split_k0_bytes + k_tile_p * BLOCK_K_BYTES, tile_n * BLOCK_SIZE_N]
            g0 = split_k0_groups + k_tile_p * K_GROUPS
            as_offs = [tile_m * BLOCK_SIZE_M, g0]
            bs_offs = [tile_n * BLOCK_SIZE_N, g0]

            gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_AS.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BS.index(slot_p), pred=1)
        load_idx += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 4)

    # Pre-load tile 0
    cur_A = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_A.index(0), layout=dot_a_layout)
    cur_B = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_B.index(0), layout=dot_b_layout)
    cur_AS = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_AS.index(0), layout=a_scale_layout)
    cur_BS = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_BS.index(0), layout=b_scale_layout)

    # ---- Unified loop: k_tiles - 1 iterations ----
    # Merges main loop + epilogue into one loop with conditional TDM.
    # Only 2 WMMA sites: one here, one final after the loop.
    for _ in range(k_tiles - 1):
        acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, "e2m1", cur_B, cur_BS, "e2m1", acc)

        # Issue TDM only while there are still tiles to fetch
        if load_idx < k_tiles:
            slot_p = load_idx % NUM_BUFFERS
            k_tile_p = load_idx
            a_offs = [tile_m * BLOCK_SIZE_M, split_k0_bytes + k_tile_p * BLOCK_K_BYTES]
            b_offs = [split_k0_bytes + k_tile_p * BLOCK_K_BYTES, tile_n * BLOCK_SIZE_N]
            g0 = split_k0_groups + k_tile_p * K_GROUPS
            as_offs = [tile_m * BLOCK_SIZE_M, g0]
            bs_offs = [tile_n * BLOCK_SIZE_N, g0]

            gl.amd.gfx1250.tdm.async_load(a_desc, a_offs, smem_A.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(b_desc, b_offs, smem_B.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(as_desc, as_offs, smem_AS.index(slot_p), pred=1)
            gl.amd.gfx1250.tdm.async_load(bs_desc, bs_offs, smem_BS.index(slot_p), pred=1)
            load_idx += 1
            gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 4)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)

        # Pre-load next tile
        next_slot = (compute_idx + 1) % NUM_BUFFERS
        cur_A = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_A.index(next_slot), layout=dot_a_layout)
        cur_B = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_B.index(next_slot), layout=dot_b_layout)
        cur_AS = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_AS.index(next_slot), layout=a_scale_layout)
        cur_BS = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_BS.index(next_slot), layout=b_scale_layout)
        compute_idx += 1

    # Final WMMA for the last pre-loaded tile
    acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, "e2m1", cur_B, cur_BS, "e2m1", acc)

    if NUM_BUFFERS > 2:
        gl.amd.sched_barrier(0)

    store_c_tile(
        c_ptr=c_ptr, tile_m=tile_m, tile_n=tile_n,
        split_k_id=split_k_id, M=M, N=N,
        stride_c_k=stride_c_k, stride_c_m=stride_c_m, stride_c_n=stride_c_n,
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, acc=acc,
    )
