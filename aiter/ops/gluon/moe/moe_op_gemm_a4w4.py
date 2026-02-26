import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: gl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        gl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: gl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@gluon.jit
def clip(x, limit, clip_lower: gl.constexpr):
    res = gl.minimum(x, limit)
    if clip_lower:
        res = gl.maximum(-limit, res)
    return res


@gluon.jit
def _swiglu(input, alpha, limit):
    gelu, linear = gl.split(gl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(gl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(gl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + gl.exp2(-1.44269504089 * alpha * gelu))
    return gl.fma(s, linear, s)  # (s * (linear + 1))


@gluon.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(gl.float32)
    tensor = tensor / scale
    tensor = tensor.to(gl.float8e4nv)
    return tensor


@gluon.jit
def _reduce_grouped(
    X,
    stride_xb: gl.uint64,
    stride_xm: gl.uint64,
    stride_xn,  #
    Out,
    stride_om: gl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    N,  #
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    pid_t = gl.program_id(1)
    pid_n = gl.program_id(0)

    BLOCK_N_OUT: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in gl.static_range(0, K):
            indxs = indxs + (gl.load(InIndx + start + i),)
    XPtrs = X + (pid_n * BLOCK_N + gl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + gl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = gl.zeros([BLOCK_N_OUT], dtype=gl.float32)
    x_n_mask = pid_n * BLOCK_N + gl.arange(0, BLOCK_N) < N
    # accumulate contributions for this tile
    for i in gl.static_range(0, K):
        curr = gl.zeros([BLOCK_N], dtype=gl.float32)
        # iterate over split_k partial values
        for b in gl.range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
            if EVEN_N:
                vals = gl.load(x_row_ptr)
            else:
                vals = gl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(gl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        curr = gl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr

    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        gl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + gl.arange(0, BLOCK_N_OUT) < Nrem
        gl.store(out_ptr, acc, mask=out_n_mask)


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype: bool = None,
):
    """
    In-place grouped row reduction.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]` and overwrites the row corresponding to the first
    valid (non-negative) index with the per-group sum. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Behavior and edge cases
    - Invalid (-1) entries are skipped during accumulation and do not generate
      memory traffic. If a group has no valid entries, nothing is written for
      that group.
    - Reduction is performed tile-by-tile along the N dimension within a single
      kernel launch (persistent along N) to minimize launch overhead.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The input tensor `x` (modified in place).
    """
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0
    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)

    _reduce_grouped[(num_blocks, num_groups)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),  #
        out,
        out.stride(0),
        out.stride(1),  #
        indx,  #
        x.shape[0],
        x.shape[-1],  #
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,  #
        num_warps=2,  #
    )
    return out


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2

    split_k = 1
    if block_m == 16:
        block_n = 128
        block_k = 256
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    else:
        # for scale preshuffling
        block_n = 512
        block_k = 256
        num_warps = 8

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }

    return ret


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    N = w.shape[-1]
    M = x.shape[-2]
    if gather_indx is not None:
        M = gather_indx.shape[0]

    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = scatter_indx.shape[0] // routing_data.n_expts_act

    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)

    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None

    return matmul_output, final_output


@gluon.constexpr_function
def get_wmma_layout(num_warps, packed, scale_preshuffle):
    assert num_warps in (4, 8)
    if scale_preshuffle:
        reg_bases = [[0, 1], [1, 0]]
        tiles_per_warp = 2
    else:
        reg_bases = []
        tiles_per_warp = 1

    # [NUM_WARPS // 2, 2]
    if num_warps == 4:
        warp_bases = [[0, tiles_per_warp], [tiles_per_warp, 0]]
    else:
        warp_bases = [[0, tiles_per_warp], [0, tiles_per_warp * 2], [tiles_per_warp, 0]]

    instr_shape = [16, 16, 64] if packed else [16, 16, 128]

    return gl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


@gluon.jit
def _moe_gemm_a4w4_gfx1250(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    X_static_scale,  # NOTE: not supported
    Quant_static_scale,
    # bias
    B,
    stride_b_e,
    Gammas,
    # shapes
    num_tokens,
    N,
    K,
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    MASK_K_LIMIT: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    WMMA_LAYOUT_PACKED: gl.constexpr,
    IDX_LAYOUT: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    gl.assume(stride_y_k >= 0)
    gl.assume(stride_y_m >= 0)
    gl.assume(stride_y_n >= 0)
    gl.assume(stride_x_m >= 0)
    gl.assume(stride_x_k >= 0)
    gl.assume(stride_w_e >= 0)
    gl.assume(stride_w_k >= 0)
    gl.assume(stride_w_n >= 0)
    if stride_x_mx_m is not None:
        gl.assume(stride_x_mx_m >= 0)
    if stride_x_mx_k is not None:
        gl.assume(stride_x_mx_k >= 0)
    if stride_w_mx_e is not None:
        gl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        gl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        gl.assume(stride_w_mx_n >= 0)
    if B is not None:
        gl.assume(stride_b_e >= 0)
    gl.assume(grid_m >= 0)
    gl.assume(grid_n >= 0)

    NUM_BUFFERS: gl.constexpr = 2

    MX_PACK_DIVISOR: gl.constexpr = 32
    gl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )

    is_x_microscaled: gl.constexpr = XMxScale is not None
    NUM_LOADS_IN_BATCH: gl.constexpr = 4 if is_x_microscaled else 3
    w_type: gl.constexpr = W.dtype.element_ty
    gl.static_assert(w_type == gl.uint8, "mx_weight_ptr must be uint8 or fp8")
    gl.static_assert(
        WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
    )
    x_type: gl.constexpr = X.dtype.element_ty
    if is_x_microscaled:
        gl.static_assert(x_type == gl.uint8, "mx_act_ptr must be uint8")
        gl.static_assert(
            XMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8"
        )

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    # get program id
    pid = gl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: gl.constexpr = 0

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    # get unpadded grid size
    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)

    # for split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k

    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # constants
    X_M_DIVISOR: gl.constexpr = 1
    X_K_DIVISOR: gl.constexpr = 2
    W_K_DIVISOR: gl.constexpr = 2
    W_N_DIVISOR: gl.constexpr = 1
    PACKED_BLOCK_M_X: gl.constexpr = BLOCK_M // X_M_DIVISOR
    PACKED_BLOCK_K_X: gl.constexpr = BLOCK_K // X_K_DIVISOR
    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: gl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: gl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    # A pointers
    offs_x_m = PACKED_BLOCK_M_X * block_id
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        offs_x_m = PACKED_BLOCK_M_X * block_id + gl.arange(
            0, PACKED_BLOCK_M_X, layout=IDX_LAYOUT
        )
        GatherIndx += start_m
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT

    # B scale pointers
    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
        gl.static_assert(stride_w_mx_k is not None)
        gl.static_assert(stride_w_mx_n is not None)
        SCALE_KWIDTH: gl.constexpr = 4 if MX_SCALE_BLOCK_K >= 4 else MX_SCALE_BLOCK_K
        PRESHUFFLE_FACTOR: gl.constexpr = 128
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K * PRESHUFFLE_FACTOR
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
    else:
        PRESHUFFLE_FACTOR: gl.constexpr = 1
        PACKED_MX_BLOCK: gl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: gl.constexpr = BLOCK_N
    offs_w_n_scale = pid_n * SCALE_BLOCK_N

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W
    W += expt_id * stride_w_e

    # A scale pointers
    if is_x_microscaled and GatherIndx is None:
        XMxScale += start_m * stride_x_mx_m

    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PACKED_BLOCK_K_X, 16]], [PACKED_BLOCK_M_X, PACKED_BLOCK_K_X], [1, 0]
    )
    SHARED_LAYOUT_W: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PACKED_BLOCK_K_W, 16]], [PACKED_BLOCK_N_W, PACKED_BLOCK_K_W], [1, 0]
    )
    SHARED_LAYOUT_X_SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K], [1, 0]
    )
    SHARED_LAYOUT_W_SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [SCALE_BLOCK_N, PACKED_MX_BLOCK], [1, 0]
    )

    if GatherIndx is None:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(M, K // X_K_DIVISOR),
            strides=(stride_x_m, stride_x_k),
            block_shape=(PACKED_BLOCK_M_X, PACKED_BLOCK_K_X),
            layout=SHARED_LAYOUT_X,
        )
    else:
        x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(num_tokens, K // X_K_DIVISOR),
            strides=(stride_x_m, stride_x_k),
            block_shape=(PACKED_BLOCK_M_X, PACKED_BLOCK_K_X),
            layout=SHARED_LAYOUT_X,
        )
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=W,
        shape=(N, K // W_K_DIVISOR),
        strides=(stride_w_n, stride_w_k),
        block_shape=(PACKED_BLOCK_N_W, PACKED_BLOCK_K_W),
        layout=SHARED_LAYOUT_W,
    )
    if is_x_microscaled:
        if GatherIndx is None:
            x_scales_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=XMxScale,
                shape=(M, gl.cdiv(K, MX_PACK_DIVISOR)),
                strides=(stride_x_mx_m, stride_x_mx_k),
                block_shape=(PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K),
                layout=SHARED_LAYOUT_X_SCALES,
            )
        else:
            x_scales_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=XMxScale,
                shape=(num_tokens, gl.cdiv(K, MX_PACK_DIVISOR)),
                strides=(stride_x_mx_m, stride_x_mx_k),
                block_shape=(PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K),
                layout=SHARED_LAYOUT_X_SCALES,
            )
    w_scales_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=WMxScale,
        shape=(N // PRESHUFFLE_FACTOR, gl.cdiv(K, MX_PACK_DIVISOR) * PRESHUFFLE_FACTOR),
        strides=(stride_w_mx_n, stride_w_mx_k),
        block_shape=(SCALE_BLOCK_N, PACKED_MX_BLOCK),
        layout=SHARED_LAYOUT_W_SCALES,
    )

    DOT_LAYOUT_X: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT_PACKED, k_width=16
    )
    DOT_LAYOUT_W: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT_PACKED, k_width=16
    )
    DOT_LAYOUT_X_SCALES: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        DOT_LAYOUT_X, [PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K]
    )
    DOT_LAYOUT_W_SCALES: gl.constexpr = gl.amd.gfx1250.get_wmma_scale_layout(
        DOT_LAYOUT_W, [PACKED_BLOCK_N_W, MX_SCALE_BLOCK_K]
    )

    x_buffer = gl.allocate_shared_memory(
        x_desc.dtype, shape=[NUM_BUFFERS] + x_desc.block_shape, layout=x_desc.layout
    )
    w_buffer = gl.allocate_shared_memory(
        w_desc.dtype, shape=[NUM_BUFFERS] + w_desc.block_shape, layout=w_desc.layout
    )
    if is_x_microscaled:
        x_scales_buffer = gl.allocate_shared_memory(
            x_scales_desc.dtype,
            shape=[NUM_BUFFERS] + x_scales_desc.block_shape,
            layout=x_scales_desc.layout,
        )
    w_scales_buffer = gl.allocate_shared_memory(
        w_scales_desc.dtype,
        shape=[NUM_BUFFERS] + w_scales_desc.block_shape,
        layout=w_scales_desc.layout,
    )

    load_idx = 0
    wmma_idx = 0

    # prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc,
                [offs_x_m, load_idx * PACKED_BLOCK_K_X],
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc,
                offs_x_m,
                load_idx * PACKED_BLOCK_K_X,
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        gl.amd.gfx1250.tdm.async_load(
            w_desc,
            [offs_w_n, load_idx * PACKED_BLOCK_K_W],
            w_buffer.index(load_idx % NUM_BUFFERS),
        )
        if is_x_microscaled:
            if GatherIndx is None:
                gl.amd.gfx1250.tdm.async_load(
                    x_scales_desc,
                    [offs_x_m, load_idx * MX_SCALE_BLOCK_K],
                    x_scales_buffer.index(load_idx % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_gather(
                    x_scales_desc,
                    offs_x_m,
                    load_idx * MX_SCALE_BLOCK_K,
                    x_scales_buffer.index(load_idx % NUM_BUFFERS),
                )
        gl.amd.gfx1250.tdm.async_load(
            w_scales_desc,
            [offs_w_n_scale, load_idx * PACKED_MX_BLOCK],
            w_scales_buffer.index(load_idx % NUM_BUFFERS),
        )
        load_idx += 1

    # compute output
    num_k_iter = gl.cdiv(K, BLOCK_K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for k in range(num_k_iter - (NUM_BUFFERS - 1)):
        if GatherIndx is None:
            gl.amd.gfx1250.tdm.async_load(
                x_desc,
                [offs_x_m, load_idx * PACKED_BLOCK_K_X],
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_gather(
                x_desc,
                offs_x_m,
                load_idx * PACKED_BLOCK_K_X,
                x_buffer.index(load_idx % NUM_BUFFERS),
            )
        gl.amd.gfx1250.tdm.async_load(
            w_desc,
            [offs_w_n, load_idx * PACKED_BLOCK_K_W],
            w_buffer.index(load_idx % NUM_BUFFERS),
        )
        if is_x_microscaled:
            if GatherIndx is None:
                gl.amd.gfx1250.tdm.async_load(
                    x_scales_desc,
                    [offs_x_m, load_idx * MX_SCALE_BLOCK_K],
                    x_scales_buffer.index(load_idx % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_gather(
                    x_scales_desc,
                    offs_x_m,
                    load_idx * MX_SCALE_BLOCK_K,
                    x_scales_buffer.index(load_idx % NUM_BUFFERS),
                )
        gl.amd.gfx1250.tdm.async_load(
            w_scales_desc,
            [offs_w_n_scale, load_idx * PACKED_MX_BLOCK],
            w_scales_buffer.index(load_idx % NUM_BUFFERS),
        )
        load_idx += 1

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)

        x = x_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        w = (
            w_buffer.index(wmma_idx % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=DOT_LAYOUT_W)
        )
        if is_x_microscaled:
            x_scales_buffer_slice = x_scales_buffer.index(wmma_idx % NUM_BUFFERS)
        w_scales_buffer_slice = w_scales_buffer.index(wmma_idx % NUM_BUFFERS)
        if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
            w_scales_buffer_slice = (
                w_scales_buffer_slice.reshape(
                    (
                        SCALE_BLOCK_N,
                        MX_SCALE_BLOCK_K // SCALE_KWIDTH,
                        PRESHUFFLE_FACTOR // 4,
                        4,
                        SCALE_KWIDTH,
                    )
                )
                .permute((0, 3, 2, 1, 4))
                .reshape((BLOCK_N, MX_SCALE_BLOCK_K))
            )
        if is_x_microscaled:
            x_scales = x_scales_buffer_slice.load(layout=DOT_LAYOUT_X_SCALES)
        else:
            x_scales = gl.full(
                (PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K), 127, dtype=gl.uint8
            )
        w_scales = w_scales_buffer_slice.load(layout=DOT_LAYOUT_W_SCALES)

        acc = gl.amd.gfx1250.wmma_scaled(x, x_scales, "e2m1", w, w_scales, "e2m1", acc)
        wmma_idx += 1

    # epilogue
    for k_ep in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - k_ep) * NUM_LOADS_IN_BATCH)

        x = x_buffer.index(wmma_idx % NUM_BUFFERS).load(layout=DOT_LAYOUT_X)
        w = (
            w_buffer.index(wmma_idx % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=DOT_LAYOUT_W)
        )
        if is_x_microscaled:
            x_scales_buffer_slice = x_scales_buffer.index(wmma_idx % NUM_BUFFERS)
        w_scales_buffer_slice = w_scales_buffer.index(wmma_idx % NUM_BUFFERS)
        if SWIZZLE_MX_SCALE == "GFX1250_SCALE":
            w_scales_buffer_slice = (
                w_scales_buffer_slice.reshape(
                    (
                        SCALE_BLOCK_N,
                        MX_SCALE_BLOCK_K // SCALE_KWIDTH,
                        PRESHUFFLE_FACTOR // 4,
                        4,
                        SCALE_KWIDTH,
                    )
                )
                .permute((0, 3, 2, 1, 4))
                .reshape((BLOCK_N, MX_SCALE_BLOCK_K))
            )
        if is_x_microscaled:
            x_scales = x_scales_buffer_slice.load(layout=DOT_LAYOUT_X_SCALES)
        else:
            x_scales = gl.full(
                (PACKED_BLOCK_M_X, MX_SCALE_BLOCK_K), 127, dtype=gl.uint8
            )
        w_scales = w_scales_buffer_slice.load(layout=DOT_LAYOUT_W_SCALES)

        acc = gl.amd.gfx1250.wmma_scaled(x, x_scales, "e2m1", w, w_scales, "e2m1", acc)

    # scalar fp8 scale
    if X_static_scale is not None:
        # should not go in here since static scale fp4 is disabled
        gl.static_assert(
            X_static_scale is None,
            f"Static scale is disabled for fp4 precision. got {X_static_scale}",
        )

    # bias
    offs_m = BLOCK_M * block_id + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_y_n = BLOCK_N * pid_n + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e
        bias = gl.amd.gfx1250.buffer_load(BPtrs, offs_y_n, mask=mask_n)
        acc = acc + bias[None, :]

    # apply activation function
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(
            0, OUT_BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        mask_n = offs_y_n < yN
    else:
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    # apply gammas
    if Gammas is not None:
        gammas = gl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, gl.load(Quant_static_scale))

    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    offs_y = (
        offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    if Quant_static_scale is None:
        out = out.to(gl.bfloat16)
    gl.amd.gfx1250.buffer_store(out, Y, offs_y, mask=mask)


def moe_gemm_a4w4_gfx1250(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None,  # NOTE: not supported
    quant_static_scale=None,
    bias=None,
    routing_data=None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,  # TODO: add support for swizzle
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    unpadded_N=None,
    unpadded_K=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx:
        assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"

    # determine shapes
    num_tokens = x.shape[-2]
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1] * 2, w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    # compute optimization flags
    config = get_kernel_config(M, N, K, routing_data)
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1

    # allocate output memory
    y, y_final = allocate_output(
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
    )

    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    # spmd grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    # layouts
    WMMA_LAYOUT = get_wmma_layout(
        config["num_warps"], False, swizzle_mx_scale == "GFX1250_SCALE"
    )
    WMMA_LAYOUT_PACKED = get_wmma_layout(
        config["num_warps"], True, swizzle_mx_scale == "GFX1250_SCALE"
    )
    IDX_LAYOUT = gl.BlockedLayout([config["block_m"]], [32], [config["num_warps"]], [0])

    # launch kernel
    _moe_gemm_a4w4_gfx1250[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x_scales,
        x_scales.stride(0) if x_has_mx else 0,
        x_scales.stride(1) if x_has_mx else 0,
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scales,
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        x_static_scale,
        quant_static_scale,
        bias,
        bias.stride(0) if bias is not None else 0,
        gammas,
        num_tokens,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        routing_data.n_expts_act,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["group_m"],
        XCD_SWIZZLE=config["xcd_swizzle"],
        SWIZZLE_MX_SCALE=swizzle_mx_scale,
        SPLIT_K=config["split_k"],
        EVEN_K=K % config["block_k"] == 0,
        MASK_K_LIMIT=K % config["block_k"],
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        WMMA_LAYOUT=WMMA_LAYOUT,
        WMMA_LAYOUT_PACKED=WMMA_LAYOUT_PACKED,
        IDX_LAYOUT=IDX_LAYOUT,
        UPCAST_INDICES=should_upcast_indices(x, w, y),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        kpack=config["kpack"],
    )

    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
    )

    return y_final
