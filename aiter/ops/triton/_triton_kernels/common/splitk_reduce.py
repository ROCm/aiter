import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gemm_splitk_reduce_repr = make_kernel_repr(
    "_gemm_splitk_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS",
        "activation",
        "use_activation",
    ],
    name_key="KERNEL_NAME",
)


@triton.jit(repr=_gemm_splitk_reduce_repr)
def _gemm_splitk_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    bias_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    KERNEL_NAME: tl.constexpr = "_gemm_splitk_reduce_kernel",
):
    tl.assume(stride_c_in_k > 0)
    tl.assume(stride_c_in_m > 0)
    tl.assume(stride_c_in_n > 0)
    tl.assume(stride_c_out_m > 0)
    tl.assume(stride_c_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tell the AMD backend pid * stride stays non-negative so it can lower
    # the loads/stores to buffer ops instead of generic global ops.
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None].to(tl.int64) * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        # Async copy ignores a mask that has no lower bound.
        k_mask = (offs_k >= 0) & (offs_k < ACTUAL_KSPLIT)
        c = tl.load(c_in_ptrs, mask=k_mask[:, None, None], other=0.0)
    c = tl.sum(c, axis=0)

    if ADD_BIAS:
        acc_dtype = tl.float32 if c_in_ptr.type.element_ty != tl.int8 else tl.int32
        bias = tl.load(bias_ptr + offs_n).to(dtype=acc_dtype)
        bias = tl.broadcast_to(bias[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c += bias

    if use_activation:
        c = activation(c)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


_batched_gemm_splitk_reduce_repr = make_kernel_repr(
    "_batched_gemm_splitk_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS",
        "activation",
        "use_activation",
    ],
    name_key="KERNEL_NAME",
)


@triton.jit(repr=_batched_gemm_splitk_reduce_repr)
def _batched_gemm_splitk_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    bias_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_b,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_b,
    stride_c_out_m,
    stride_c_out_n,
    stride_biasb,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    KERNEL_NAME: tl.constexpr = "_batched_gemm_splitk_reduce_kernel",
):
    """Reduce the split-K partial sums for a batched GEMM.

    Mirrors ``_gemm_splitk_reduce_kernel`` with an added leading batch
    dimension. ``c_in_ptr`` points at the partial sums with shape
    (MAX_KSPLIT, B, M, N) and ``c_out_ptr`` at the reduced output (B, M, N).
    """
    tl.assume(stride_c_in_k > 0)
    tl.assume(stride_c_in_b > 0)
    tl.assume(stride_c_in_m > 0)
    tl.assume(stride_c_in_n > 0)
    tl.assume(stride_c_out_b > 0)
    tl.assume(stride_c_out_m > 0)
    tl.assume(stride_c_out_n > 0)

    batch_id = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Tell the AMD backend pid * stride stays non-negative so it can lower
    # the loads/stores to buffer ops instead of generic global ops.
    tl.assume(batch_id >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow.
    batch_id = tl.cast(batch_id, tl.int64)
    stride_c_in_b = tl.cast(stride_c_in_b, tl.int64)
    stride_c_out_b = tl.cast(stride_c_out_b, tl.int64)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + batch_id * stride_c_in_b
        + (offs_k[:, None, None].to(tl.int64) * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)

    if ADD_BIAS:
        acc_dtype = tl.float32 if c_in_ptr.type.element_ty != tl.int8 else tl.int32
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_n).to(dtype=acc_dtype)
        bias = tl.broadcast_to(bias[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c += bias

    if use_activation:
        c = activation(c)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + batch_id * stride_c_out_b
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


@triton.jit
def _tile_on_xcd(N: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, SPLITS: tl.constexpr):
    """(pid_m, pid_n, tile, split) of a 1D grid, all splits of a tile on one XCD.

    CTAs go to XCDs by pid % 8, a multiple of every gfx950 XCD count.
    """
    pid = tl.program_id(0)
    tile = (pid // 8 // SPLITS) * 8 + pid % 8
    grid_n: tl.constexpr = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    return tile // grid_n, tile % grid_n, tile, pid // 8 % SPLITS


@triton.jit
def _sum_splits_on_xcd(
    partial, c_ptrs, mask, slot, counter, split, SPLITS: tl.constexpr
):
    """The tile's last CTA to arrive sums its splits in split order.

    The splits share one XCD's L2, so a completed store is visible without an
    agent-scope release and its L2 writeback.
    """
    BM: tl.constexpr = partial.shape[0]
    BN: tl.constexpr = partial.shape[1]
    local = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(slot + split * BM * BN + local, partial)
    # debug_barrier is a bare s_barrier: drain this wave's stores first.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(counter, 1, sem="acq_rel", scope="cta") == SPLITS - 1:
        total = tl.zeros((BM, BN), tl.float32)
        for s in tl.static_range(SPLITS):
            # .cv skips this CU's L1, which never saw the other splits.
            total += tl.load(slot + s * BM * BN + local, cache_modifier=".cv")
        tl.store(c_ptrs, total.to(c_ptrs.dtype.element_ty), mask)
        tl.store(counter, 0)
