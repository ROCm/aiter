from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import torch

SCALE_GROUP_ELEMS = 32


def get_gemm_afp4wfp4_preshuffle_layouts(
    num_warps: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
):
    K_GROUPS = BLOCK_K // SCALE_GROUP_ELEMS

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
    BLOCK_K: gl.constexpr,
    BLOCK_K_BYTES: gl.constexpr,
):
    # raw -> logical [BLOCK_K_BYTES, BLOCK_N]
    return (
        b_raw.reshape((1, BLOCK_N // 16, BLOCK_K // 64, 2, 16, 16))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_N, BLOCK_K_BYTES))
        .permute((1, 0))
    )


@gluon.jit
def unshuffle_scales_32(
    scales_shuf,
    BLOCK_X: gl.constexpr,
    K_GROUPS: gl.constexpr,
):
    # One shared unshuffle for A/B scales
    return (
        scales_shuf.reshape((BLOCK_X // 32, K_GROUPS // 8, 4, 16, 2, 2, 1))
        .permute((0, 5, 3, 1, 4, 2, 6))
        .reshape((BLOCK_X, K_GROUPS))
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
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
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
    # A scales: M>=32 uses preshuffled (M//32, K) layout; M<32 uses (M, K//32) per row
    if BLOCK_SIZE_M < 32:
        smem_ASraw = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [NUM_BUFFERS, BLOCK_SIZE_M, K_GROUPS],
            layout=shared_S,
        )
    else:
        smem_ASraw = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [NUM_BUFFERS, BLOCK_SIZE_M // 32, K_GROUPS * 32],
            layout=shared_S,
        )
    smem_BSraw = gl.allocate_shared_memory(
        b_scale_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_SIZE_N // 32, K_GROUPS * 32],
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

    grid_m32 = gl.cdiv(M, 32)
    grid_n32 = gl.cdiv(N, 32)
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
            shape=(grid_m32, K_elems),
            strides=(stride_as_m, stride_as_k),
            block_shape=(BLOCK_SIZE_M // 32, K_GROUPS * 32),
            layout=shared_S,
        )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr,
        shape=(grid_n32, K_elems),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_SIZE_N // 32, K_GROUPS * 32),
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

            # Scale offsets are in groups-domain -> element-domain (groups*32)
            g0 = split_k0_groups + k_tile_p * K_GROUPS
            if BLOCK_SIZE_M < 32:
                as_offs = [tile_m * BLOCK_SIZE_M, g0]
            else:
                as_offs = [tile_m * (BLOCK_SIZE_M // 32), g0 * 32]
            bs_offs = [tile_n * (BLOCK_SIZE_N // 32), g0 * 32]

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
            as_offs = [tile_m * (BLOCK_SIZE_M // 32), g0 * 32]
        bs_offs = [tile_n * (BLOCK_SIZE_N // 32), g0 * 32]

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
                smem_B.index(slot_c), BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K, BLOCK_K_BYTES=BLOCK_K_BYTES
            ).load(layout=dot_b_layout)

        # scales: unshuffle -> load with wmma scale layouts
        if BLOCK_SIZE_M < 32:
            AS = smem_ASraw.index(slot_c).load(layout=a_scale_layout)
        else:
            AS = unshuffle_scales_32(
                smem_ASraw.index(slot_c), BLOCK_X=BLOCK_SIZE_M, K_GROUPS=K_GROUPS
            ).load(layout=a_scale_layout)
        BS = unshuffle_scales_32(
            smem_BSraw.index(slot_c), BLOCK_X=BLOCK_SIZE_N, K_GROUPS=K_GROUPS
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
                smem_B.index(slot_c), BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K, BLOCK_K_BYTES=BLOCK_K_BYTES
            ).load(layout=dot_b_layout)

            if BLOCK_SIZE_M < 32:
                AS = smem_ASraw.index(slot_c).load(layout=a_scale_layout)
            else:
                AS = unshuffle_scales_32(
                    smem_ASraw.index(slot_c), BLOCK_X=BLOCK_SIZE_M, K_GROUPS=K_GROUPS
                ).load(layout=a_scale_layout)
            BS = unshuffle_scales_32(
                smem_BSraw.index(slot_c), BLOCK_X=BLOCK_SIZE_N, K_GROUPS=K_GROUPS
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


def _shuffle_scales_for_main(scales: torch.Tensor) -> torch.Tensor:
    """Shuffle scales for preshuffled layout (matches test_gemm_afp4wfp4)."""
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def _generate_inputs_for_main(M: int, N: int, K: int, dtype: torch.dtype):
    """Generate preshuffled inputs for gemm_afp4wfp4 (matches test layout)."""
    from aiter.ops.shuffle import shuffle_weight

    scale_group_size = 32
    # x: [M, K//2] uint8 packed fp4
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x = (x_high << 4 | x_low).to(device="cuda")
    # w: [N, K//2] uint8 packed fp4 on GPU
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    # scales: e8m0, shape (M_pad, K//32) and (N, K//32) then transposed for API
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(
        124, 128, (K // scale_group_size, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // scale_group_size, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    x_scales_shuffled = _shuffle_scales_for_main(x_scales) if M >= 32 else x_scales.contiguous()
    w_scales_shuffled = _shuffle_scales_for_main(w_scales)
    w_preshuf = shuffle_weight(w, layout=(16, 16), use_int4=False).reshape(
        w.shape[0] // 16, w.shape[1] * 16
    )
    return (
        x,
        w_preshuf,
        x_scales_shuffled[:M],
        w_scales_shuffled,
        dtype,
    )


if __name__ == "__main__":
    import triton

    import aiter.ops.triton.utils._triton.arch_info as arch_info

    if arch_info.get_arch() != "gfx1250":
        raise RuntimeError("gemm_afp4wfp4_preshuffle_gfx1250 is only supported on gfx1250")

    # Use a single (M, N, K) so roccap can capture kernel "gemm_afp4wfp4_preshuffle_gfx1250/0"
    # N % 32 == 0 and K % 256 == 0 required for preshuffled kernel
    # (2048, 7168, 4096)
    # (32, 7168, 4096)
    M, N, K = 2048, 7168, 4096
    dtype = torch.float16
    x, w_preshuf, x_scales_shuffled, w_scales_shuffled, _ = _generate_inputs_for_main(M, N, K, dtype)
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    K_elems = K
    if M >= 32:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 512, 512
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 64, 256
    NUM_WARPS = 4
    NUM_KSPLIT = 1
    SPLITK_BLOCK = K_elems
    layouts = get_gemm_afp4wfp4_preshuffle_layouts(
        NUM_WARPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_mxfp4_preshuffle_gfx1250[grid](
        x,
        w_preshuf,
        y,
        x_scales_shuffled,
        w_scales_shuffled,
        M,
        N,
        K_elems,
        x.stride(0),
        x.stride(1),
        w_preshuf.stride(0),
        w_preshuf.stride(1),
        0 if NUM_KSPLIT == 1 else y.stride(0),
        y.stride(-2),
        y.stride(-1),
        x_scales_shuffled.stride(0),
        x_scales_shuffled.stride(1),
        w_scales_shuffled.stride(0),
        w_scales_shuffled.stride(1),
        NUM_BUFFERS=2,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_WARPS=NUM_WARPS,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK=SPLITK_BLOCK,
        cache_modifier=None,
        **layouts,
    )
    torch.cuda.synchronize()
    print(f"gemm_mxfp4_preshuffle_gfx1250 done: output shape {y.shape}")
