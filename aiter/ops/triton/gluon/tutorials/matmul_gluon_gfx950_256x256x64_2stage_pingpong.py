import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


@triton.heuristics(
    {
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr, GRID_MN: gl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = gl.program_id(axis=0)
    # not remap 940t vs remap 890t when K = 4096
    # pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    gl.assume(pid_m >= 0)
    gl.assume(pid_n >= 0)
    gl.assume(stride_am > 0)
    gl.assume(stride_ak > 0)
    gl.assume(stride_bn > 0)
    gl.assume(stride_bk > 0)
    gl.assume(stride_cm > 0)
    gl.assume(stride_cn > 0)

    # Add user set layout
    blocked_a: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,1),(0,2), (0,4),(8,0), (128, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (16, 0), (32, 0), (64, 0)),
        warp_bases=((1, 0), (2, 0), (4, 0)),
        block_bases=[],
        shape=[256, 64],
    )

    blocked_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1,0),(2,0),(4,0),(0,8), (0, 128)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 16), (0, 32), (0, 64)),
        warp_bases=((0, 1), (0, 2), (0, 4)),
        block_bases=[],
        shape=[64, 256],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs = [[512,16]],
        offset_bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16,0], [32,0], [64,0], [1,0], [2,0], [4,0], [8,0], [128,0]],
        block_bases = [],
        shape = [256, 64]
    )
    shared_b: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs = [[512,16]],
        offset_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128]],
        block_bases = [],
        shape = [64, 256]
    )

    a_bufs = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    b_bufs = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [2, BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # compute offsets
    offs_am = (pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_b))
    a_offs = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offs = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

    num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)


    # prologue
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_bufs.index(0), a_ptr, a_offs)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_bufs.index(0), b_ptr, b_offs)
    gl.amd.cdna4.async_copy.commit_group()
    a_offs += BLOCK_SIZE_K * stride_ak
    b_offs += BLOCK_SIZE_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna3.s_barrier()

    if gl.amd.cdna3.warp_id() >= 4:
        gl.amd.cdna3.s_barrier()

    gl.assume(num_k_iter > 2)
    buf_idx = 0
    # mainloop
    for k in range(num_k_iter-1):
        async_idx = (buf_idx + 1) % 2
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_bufs.index(async_idx), a_ptr, a_offs)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_bufs.index(async_idx), b_ptr, b_offs)
        gl.amd.cdna4.async_copy.commit_group()

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_bufs.index(buf_idx), dot_a_layout)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_bufs.index(buf_idx), dot_b_layout)
        #cur_a = a_bufs.index(buf_idx).load(layout=dot_a_layout)
        #cur_b = b_bufs.index(buf_idx).load(layout=dot_b_layout)

        gl.amd.cdna3.sched_barrier(0)
        gl.amd.cdna4.async_copy.wait_group(0)
        #gl.amd.cdna3.s_waitcnt(lgkmcnt, 0)
        gl.amd.cdna3.s_barrier()
        gl.amd.cdna3.sched_barrier(0)

        accumulator = gl.amd.cdna4.mfma(cur_a, cur_b, accumulator)

        a_offs += BLOCK_SIZE_K * stride_ak
        b_offs += BLOCK_SIZE_K * stride_bk
        buf_idx = (buf_idx + 1) % 2
        gl.amd.cdna3.sched_barrier(0)
        gl.amd.cdna3.s_barrier()

    # epilogue
    if gl.amd.cdna3.warp_id() < 4:
        gl.amd.cdna3.s_barrier()
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna3.s_barrier()
    cur_a = a_bufs.index(buf_idx).load(layout=dot_a_layout)
    cur_b = b_bufs.index(buf_idx).load(layout=dot_b_layout)
    accumulator = gl.amd.cdna4.mfma(cur_a, cur_b, accumulator)

    # store c
    c = accumulator.to(gl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout))
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout))
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_store(stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask)
