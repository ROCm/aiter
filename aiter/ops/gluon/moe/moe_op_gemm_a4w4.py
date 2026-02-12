import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

from aiter.ops.triton.moe.quant_moe import DequantScaleRoundingMode
from aiter.ops.triton.moe.moe_routing.routing import RoutingData


@gluon.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: ttgl.constexpr = 1):
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
        ttgl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: ttgl.constexpr):
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
def clip(x, limit, clip_lower: ttgl.constexpr):
    res = ttgl.minimum(x, limit)
    if clip_lower:
        res = ttgl.maximum(-limit, res)
    return res


@gluon.jit
def _swiglu(input, alpha, limit):
    gelu, linear = ttgl.split(ttgl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(ttgl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(ttgl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + ttgl.exp2(-1.44269504089 * alpha * gelu))
    return ttgl.fma(s, linear, s)  # (s * (linear + 1))


@gluon.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(ttgl.float32)
    tensor = tensor / scale
    tensor = tensor.to(ttgl.float8e4nv)
    return tensor


@gluon.jit
def _reduce_grouped(
    X,
    stride_xb: ttgl.uint64,
    stride_xm: ttgl.uint64,
    stride_xn,  #
    Out,
    stride_om: ttgl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    N,  #
    # fused activation function
    APPLY_SWIGLU: ttgl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: ttgl.constexpr,
    K: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    EVEN_N: ttgl.constexpr,
):
    pid_t = ttgl.program_id(1)
    pid_n = ttgl.program_id(0)

    BLOCK_N_OUT: ttgl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in ttgl.static_range(0, K):
            indxs = indxs + (ttgl.load(InIndx + start + i),)
    XPtrs = X + (pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + ttgl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = ttgl.zeros([BLOCK_N_OUT], dtype=ttgl.float32)
    x_n_mask = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N) < N
    # accumulate contributions for this tile
    for i in ttgl.static_range(0, K):
        curr = ttgl.zeros([BLOCK_N], dtype=ttgl.float32)
        # iterate over split_k partial values
        for b in ttgl.range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
            if EVEN_N:
                vals = ttgl.load(x_row_ptr)
            else:
                vals = ttgl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(ttgl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        curr = ttgl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr

    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        ttgl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + ttgl.arange(0, BLOCK_N_OUT) < Nrem
        ttgl.store(out_ptr, acc, mask=out_n_mask)


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
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )

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
    assert (num_warps in (4, 8))
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

    return ttgl.amd.AMDWMMALayout(3, True, warp_bases, reg_bases, instr_shape)


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
    X_static_scale, # NOTE: not supported
    Quant_static_scale,
    # bias
    B,
    stride_b_e,
    Gammas,
    # shapes
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
    APPLY_SWIGLU: ttgl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: ttgl.constexpr,
    # MoE config
    N_EXPTS_ACT: ttgl.constexpr,
    # optimization config
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    GROUP_M: ttgl.constexpr,
    XCD_SWIZZLE: ttgl.constexpr,
    SWIZZLE_MX_SCALE: ttgl.constexpr, # TODO: add support for swizzle
    EVEN_K: ttgl.constexpr,
    MASK_K_LIMIT: ttgl.constexpr,
    SPLIT_K: ttgl.constexpr,
    W_CACHE_MODIFIER: ttgl.constexpr,
    # layouts
    WMMA_LAYOUT: ttgl.constexpr,
    SHARED_LAYOUT_X: ttgl.constexpr,
    SHARED_LAYOUT_W: ttgl.constexpr,
    SHARED_LAYOUT_X_SCALES: ttgl.constexpr,
    SHARED_LAYOUT_W_SCALES: ttgl.constexpr,
    DOT_LAYOUT_X: ttgl.constexpr,
    DOT_LAYOUT_W: ttgl.constexpr,
    LAYOUT_X_SCALES: ttgl.constexpr,
    LAYOUT_W_SCALES: ttgl.constexpr,
    BLOCKED_LAYOUT: ttgl.constexpr,
    UPCAST_INDICES: ttgl.constexpr = False,
):
    ttgl.assume(stride_y_k >= 0)
    ttgl.assume(stride_y_m >= 0)
    ttgl.assume(stride_y_n >= 0)
    ttgl.assume(stride_x_m >= 0)
    ttgl.assume(stride_x_k >= 0)
    ttgl.assume(stride_w_e >= 0)
    ttgl.assume(stride_w_k >= 0)
    ttgl.assume(stride_w_n >= 0)
    if stride_x_mx_m is not None:
        ttgl.assume(stride_x_mx_m >= 0)
    if stride_x_mx_k is not None:
        ttgl.assume(stride_x_mx_k >= 0)
    if stride_w_mx_e is not None:
        ttgl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        ttgl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        ttgl.assume(stride_w_mx_n >= 0)
    if B is not None:
        ttgl.assume(stride_b_e >= 0)
    ttgl.assume(grid_m >= 0)
    ttgl.assume(grid_n >= 0)

    MX_PACK_DIVISOR: ttgl.constexpr = 32
    ttgl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )

    is_x_microscaled: ttgl.constexpr = XMxScale is not None
    w_type: ttgl.constexpr = W.dtype.element_ty
    ttgl.static_assert(w_type == ttgl.uint8, "mx_weight_ptr must be uint8 or fp8")
    ttgl.static_assert(
        WMxScale.dtype.element_ty == ttgl.uint8, "mx_scale_ptr must be uint8"
    )
    x_type: ttgl.constexpr = X.dtype.element_ty
    if is_x_microscaled:
        ttgl.static_assert(x_type == ttgl.uint8, "mx_act_ptr must be uint8")
        ttgl.static_assert(
            XMxScale.dtype.element_ty == ttgl.uint8, "mx_scale_ptr must be uint8"
        )

    OUT_BLOCK_N: ttgl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = ttgl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - ttgl.load(ExptOffsSum)
    else:
        padding_m: ttgl.constexpr = 0

    index_type: ttgl.constexpr = ttgl.int64 if UPCAST_INDICES else ttgl.int32

    unpadded_m = grid_m - padding_m
    ttgl.assume(unpadded_m >= 0)
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

    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k

    # unpack expert data
    expt_data = ttgl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = ttgl.load(ExptHist + expt_id)
    start_m = ttgl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    X_M_DIVISOR: ttgl.constexpr = 1
    X_K_DIVISOR: ttgl.constexpr = 2
    W_K_DIVISOR: ttgl.constexpr = 2
    W_N_DIVISOR: ttgl.constexpr = 1
    PACKED_BLOCK_M_X: ttgl.constexpr = BLOCK_M // X_M_DIVISOR
    PACKED_BLOCK_K_X: ttgl.constexpr = BLOCK_K // X_K_DIVISOR
    PACKED_BLOCK_K_W: ttgl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: ttgl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: ttgl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    # A pointers: compute offs_x_k first (Gluon pass can infer make_range type when it's not the first in the block)
    offs_x_k = PACKED_BLOCK_K_X * pid_k + ttgl.arange(0, PACKED_BLOCK_K_X, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_x_m = PACKED_BLOCK_M_X * block_id + ttgl.arange(0, PACKED_BLOCK_M_X, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_x_m = ttgl.max_contiguous(
        ttgl.multiple_of(offs_x_m % (M // X_M_DIVISOR), PACKED_BLOCK_M_X),
        PACKED_BLOCK_M_X,
    )
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = ttgl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    # B scale pointers
    WMxScale += expt_id * stride_w_mx_e
    PACKED_MX_BLOCK: ttgl.constexpr = MX_SCALE_BLOCK_K
    SCALE_BLOCK_N: ttgl.constexpr = BLOCK_N
    offs_w_n_scale = (pid_n * SCALE_BLOCK_N + ttgl.arange(0, SCALE_BLOCK_N, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))) % N
    offs_w_n_scale = ttgl.max_contiguous(
        ttgl.multiple_of(offs_w_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N
    )
    # K dimension must be the last dimension for the scales
    offs_w_k_scale = PACKED_MX_BLOCK * pid_k + ttgl.arange(0, PACKED_MX_BLOCK, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    WMxScalePtrs = (
        WMxScale
        + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k
        + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
    )

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W + ttgl.arange(0, PACKED_BLOCK_N_W, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs_w_n = ttgl.max_contiguous(
        ttgl.multiple_of(offs_w_n % (N // W_N_DIVISOR), PACKED_BLOCK_N_W),
        PACKED_BLOCK_N_W,
    )
    offs_w_k = PACKED_BLOCK_K_W * pid_k + ttgl.arange(0, PACKED_BLOCK_K_W, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    # A scales pointers
    if is_x_microscaled:
        if GatherIndx is None:
            XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = MX_SCALE_BLOCK_K * pid_k + ttgl.arange(0, MX_SCALE_BLOCK_K, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
        XMxScalePtrs = (
            XMxScale
            + offs_x_m.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

    # determine number of k iterations
    num_k_iter = ttgl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    # compute output
    acc = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT)
    for k in range(num_k_iter):
        x = ttgl.load(XPtrs)
        w = ttgl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)

        x = ttgl.convert_layout(x, DOT_LAYOUT_X)
        w = ttgl.convert_layout(w, DOT_LAYOUT_W)

        if is_x_microscaled:
            x_scales = ttgl.load(XMxScalePtrs)
        else:
            x_scales = ttgl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=ttgl.uint8)
        w_scales = ttgl.load(WMxScalePtrs)

        x_scales = ttgl.convert_layout(x_scales, LAYOUT_X_SCALES)
        w_scales = ttgl.convert_layout(w_scales, LAYOUT_W_SCALES)

        acc = ttgl.amd.gfx1250.wmma_scaled(
            x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc,
        )

        WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
        if is_x_microscaled:
            XMxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_x_mx_k

        XPtrs += (PACKED_BLOCK_K_X * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < (MASK_K_LIMIT // X_K_DIVISOR)
        mask_w_k = offs_w_k < (MASK_K_LIMIT // W_K_DIVISOR)
        if SWIZZLE_MX_SCALE is None:
            mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
        if is_x_microscaled:
            mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

        x = ttgl.load(XPtrs, mask=mask_x_k[None, :], other=0)
        w = ttgl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )

        x = ttgl.convert_layout(x, DOT_LAYOUT_X)
        w = ttgl.convert_layout(w, DOT_LAYOUT_W)

        if is_x_microscaled:
            x_scales = ttgl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
        else:
            x_scales = ttgl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=ttgl.uint8)
        w_scales = ttgl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

        x_scales = ttgl.convert_layout(x_scales, LAYOUT_X_SCALES)
        w_scales = ttgl.convert_layout(w_scales, LAYOUT_W_SCALES)

        acc = ttgl.amd.gfx1250.wmma_scaled(
            x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc,
        )

    # scalar fp8 scale
    if X_static_scale is not None:
        # should not go in here since static scale fp4 is disabled
        ttgl.static_assert(
            X_static_scale is None,
            f"Static scale is disabled for fp4 precision. got {X_static_scale}",
        )

    # bias
    offs_m = BLOCK_M * block_id + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
    offs_y_n = BLOCK_N * pid_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = ttgl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = ttgl.full([BLOCK_N], 0, dtype=ttgl.float32, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        acc = acc + bias[None, :]

    # apply activation function
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit)
        ttgl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + ttgl.arange(0, OUT_BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        mask_n = offs_y_n < yN
    else:
        ttgl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    # apply gammas
    if Gammas is not None:
        gammas = ttgl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, ttgl.load(Quant_static_scale))

    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    ttgl.store(YPtrs, out, mask=mask)


def moe_gemm_a4w4_gfx1250(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None, # NOTE: not supported
    quant_static_scale=None,
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None, # TODO: add support for swizzle
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
    num_warps = config["num_warps"]
    BLOCK_M = config["block_m"]
    BLOCK_N = config["block_n"]
    BLOCK_K = config["block_k"]

    BLOCK_K_SCALE = BLOCK_K // 32
    SCALE_KWIDTH = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
    PRESHUFFLE_FACTOR = 1 # TODO: add support for preshuffling

    BLOCK_M_PRESHUFFLED = BLOCK_M // PRESHUFFLE_FACTOR
    BLOCK_N_PRESHUFFLED = BLOCK_N // PRESHUFFLE_FACTOR
    BLOCK_K_PRESHUFFLED = BLOCK_K_SCALE * PRESHUFFLE_FACTOR
    
    WMMA_LAYOUT = get_wmma_layout(num_warps, False, False)
    WMMA_LAYOUT_PACKED = get_wmma_layout(num_warps, True, False)

    DOT_LAYOUT_X = ttgl.DotOperandLayout(operand_index=0, parent=WMMA_LAYOUT_PACKED, k_width=16)
    DOT_LAYOUT_W = ttgl.DotOperandLayout(operand_index=1, parent=WMMA_LAYOUT_PACKED, k_width=16)
    LAYOUT_X_SCALES = ttgl.amd.gfx1250.get_wmma_scale_layout(DOT_LAYOUT_X, [BLOCK_M, BLOCK_K_SCALE])
    LAYOUT_W_SCALES = ttgl.amd.gfx1250.get_wmma_scale_layout(DOT_LAYOUT_W, [BLOCK_N, BLOCK_K_SCALE])

    BLOCK_K_PACKED = BLOCK_K // 2
    SHARED_LAYOUT_X = ttgl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K_PACKED, 16]], [BLOCK_M, BLOCK_K_PACKED], [1, 0])
    SHARED_LAYOUT_W = ttgl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K_PACKED, 16]], [BLOCK_N, BLOCK_K_PACKED], [1, 0])

    SHARED_LAYOUT_X_SCALES = ttgl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [BLOCK_M_PRESHUFFLED, BLOCK_K_PRESHUFFLED], [1, 0])
    SHARED_LAYOUT_W_SCALES = ttgl.PaddedSharedLayout.with_identity_for(
        [[256, 16]], [BLOCK_N_PRESHUFFLED, BLOCK_K_PRESHUFFLED], [1, 0])

    BLOCKED_LAYOUT = ttgl.BlockedLayout([1, 8], [1, 32], [num_warps // 2, 2], [1, 0])

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
        SHARED_LAYOUT_X=SHARED_LAYOUT_X,
        SHARED_LAYOUT_W=SHARED_LAYOUT_W,
        SHARED_LAYOUT_X_SCALES=SHARED_LAYOUT_X_SCALES,
        SHARED_LAYOUT_W_SCALES=SHARED_LAYOUT_W_SCALES,
        DOT_LAYOUT_X=DOT_LAYOUT_X,
        DOT_LAYOUT_W=DOT_LAYOUT_W,
        LAYOUT_X_SCALES=LAYOUT_X_SCALES,
        LAYOUT_W_SCALES=LAYOUT_W_SCALES,
        BLOCKED_LAYOUT=BLOCKED_LAYOUT,
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


if __name__ == "__main__":
    from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import (
        init_routing_data,
        init_compute_data,
        dtype_str_to_torch,
        assert_close,
    )
    from aiter.ops.triton.moe.quant_moe import (
        downcast_to_mxfp,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        mxfp4_quant,
        moe_gemm_a4w4,
    )

    # test the kernel
    device = torch.device("cuda")
    m = 32
    n = 32
    k = 32
    n_expts_tot = 8
    n_expts_act = 2
    do_gather = False
    do_scatter = False
    has_y_gammas = False
    fused_quant = False
    swizzle_mx_scale = None
    apply_swiglu = False

    act_mxfp4 = "mxfloat4_e2m1"
    weight_mxfp4 = "mxfloat4_e2m1"
    weight_dtype_str = weight_mxfp4[2:]
    weight_dtype = dtype_str_to_torch(weight_dtype_str)

    print(f"Configuration:")
    print(f"  Tokens: {m}")
    print(f"  Total Experts: {n_expts_tot}")
    print(f"  Active Experts: {n_expts_act}")
    print(f"  K: {k}, N: {n}")
    print()

    # initialize routing data
    m, routing_data, gather_idx, scatter_idx = init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device)
    print(f"  Routing block_m: {routing_data.block_m}")
    print(f"  Expert histogram: {routing_data.expt_hist.tolist()}")
    print()

    # initialize compute data
    x, w, bias, gamma = init_compute_data(m, n, k, gather_idx, scatter_idx, n_expts_tot, n_expts_act, torch.bfloat16, torch.bfloat16, has_y_gammas, device=device)
    x, x_scales = mxfp4_quant(x)
    w, w_scales = downcast_to_mxfp(w, torch.uint8, axis=1)
    x_static_scale = None
    out_dtype = torch.bfloat16
    print(f"  X shape: {x.shape} {x.dtype}")
    print(f"  W shape: {w.shape} {w.dtype}")
    print(f"  X scales shape: {x_scales.shape} {x_scales.dtype}")
    print(f"  W scales shape: {w_scales.shape} {w_scales.dtype}")
    print(f"  Bias shape: {bias.shape} {bias.dtype}")
    print()

    # run gluon kernel
    if not act_mxfp4 and fused_quant:
        quant_static_scale = x.abs().max().float() / 448.0
    else:
        quant_static_scale = None
    y_gluon = moe_gemm_a4w4_gfx1250(
        x,
        w,
        x_scales,
        w_scales,
        x_static_scale,
        quant_static_scale,
        bias,
        routing_data,
        gather_idx,
        scatter_idx,
        gamma,
        swizzle_mx_scale,
        out_dtype,
        apply_swiglu,
    )
    print("========= y_gluon =========")
    print(y_gluon)

    # run triton kernel
    y_triton = moe_gemm_a4w4(
        x,
        w,
        x_scales,
        w_scales,
        x_static_scale,
        quant_static_scale,
        bias,
        routing_data,
        gather_idx,
        scatter_idx,
        gamma,
        swizzle_mx_scale,
        out_dtype,
        apply_swiglu,
    )
    print("========= y_triton =========")
    print(y_triton)

    # compute error
    assert_close(y_gluon, y_triton)