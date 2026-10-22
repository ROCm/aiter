import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_reduce_grouped_repr = make_kernel_repr(
    "_reduce_grouped",
    [
        "BLOCK_N",
        "EVEN_N",
        "K",
        "APPLY_SWIGLU",
        "ACTIVATION_REDUCTION_N",
        "SWIGLU_ADD_RESIDUAL",
        "USE_TDM",
        "HAS_EXT_RESIDUAL",
    ],
)


@triton.jit
def _scatter_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,
    Out,
    stride_om: tl.uint64,
    stride_on,
    # (M,) int32: destination row of Out for sorted row m, negative to skip.
    DstRow,
    B,
    M,
    N,
    num_blocks,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Route each matmul row to an arbitrary row of `Out` instead of reducing.

    The mirror image of ``_reduce_grouped``: that one gathers a token's K sorted
    rows and sums them locally, this one leaves the sum to somebody else and just
    places each row where that somebody expects it. It exists for expert-parallel
    combine, where the K rows of a token live on K different ranks, so the sum
    cannot happen until every rank has delivered its row -- and once delivery is
    a row-granular scatter, the local reduce has nothing left to do.

    Rows are written verbatim: the matmul epilogue already applied the route
    weight (``Gammas``, indexed by the same sorted row), so the consumer's sum is
    unweighted.

    `Out` is addressed purely through ``stride_om``, so it may be a strided view
    over a peer-mapped symmetric window whose row pitch exceeds N -- which is the
    point, since a combine staging slot is typically padded to a power of two.
    ``DstRow`` is indexed by sorted row and pre-filled negative, so dead rows
    (expert-parallel gates this rank does not own, or slots past the received
    count) skip the load entirely rather than scattering garbage.
    """
    pid = tl.program_id(0)
    pid_m = pid // num_blocks
    pid_n = pid % num_blocks

    # Scalar load -> CTA-uniform branch: a skipped row costs no row traffic.
    dst = tl.load(DstRow + pid_m)
    if dst >= 0:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptr = X + pid_m * stride_xm + offs_n * stride_xn
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for b in tl.range(0, B):
            if EVEN_N:
                vals = tl.load(x_ptr + b * stride_xb)
            else:
                vals = tl.load(x_ptr + b * stride_xb, mask=offs_n < N, other=0.0)
            acc += vals.to(tl.float32)
        # int64: dst * stride_om spans the whole symmetric window (every peer's
        # slot region), which overflows int32 at realistic hidden sizes.
        out_ptr = Out + dst.to(tl.int64) * stride_om + offs_n * stride_on
        if EVEN_N:
            tl.store(out_ptr, acc)
        else:
            tl.store(out_ptr, acc, mask=offs_n < N)


@triton.jit(repr=_reduce_grouped_repr)
def _reduce_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,
    Out,
    stride_om: tl.uint64,
    stride_on,  # output tensor
    InIndx,
    # Optional per-gate validity, same [num_groups, K] shape as InIndx. Needed
    # when only some of a group's K slots were produced by the matmul -- e.g.
    # expert-parallel routing, where a token's non-local experts are computed on
    # another rank, so those slots of X are never written. Without this the
    # reduce sums uninitialized memory.
    InIndxValid,
    B,
    M,
    N,
    num_blocks,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    K: tl.constexpr,
    HAS_INDX_VALID: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    USE_TDM: tl.constexpr,
    # Step 9: external residual fold-in. When HAS_EXT_RESIDUAL=True,
    # Residual[token, :] is added to `acc` before the writeback.
    Residual,
    stride_extres_m: tl.uint64,
    stride_extres_n,
    HAS_EXT_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_t = pid // num_blocks
    pid_n = pid % num_blocks

    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i),)
    if HAS_INDX_VALID:
        # Clamp masked slots to row 0 so the pointer stays in bounds even on the
        # path that skips the load. The skip itself is the point: under expert
        # parallelism only ~n_local/n_global of a token's K slots are live (1.5 of
        # 6 for DeepSeek-V4 at ep8), so loading every slot and zeroing it after
        # issues 4x the row reads this kernel actually needs.
        valids = ()
        for i in tl.static_range(0, K):
            valids = valids + (tl.load(InIndxValid + start + i) != 0,)
        safe_indxs = ()
        for i in tl.static_range(0, K):
            safe_indxs = safe_indxs + (tl.where(valids[i], indxs[i], 0),)
        indxs = safe_indxs
    XPtrs = X + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
    x_n_mask = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) < N
    if USE_TDM and EVEN_N:
        x_desc = tl.make_tensor_descriptor(
            base=X, shape=(B * M, N), strides=(N, 1), block_shape=(1, BLOCK_N)
        )
    # accumulate contributions for this tile
    for i in tl.static_range(0, K):
        curr = tl.zeros([BLOCK_N], dtype=tl.float32)
        # `valids[i]` comes from a scalar load, so it is CTA-uniform and this is a
        # uniform branch, not a divergent one -- the whole workgroup skips the row
        # together. A dead slot leaves `curr` at zero, which is what the
        # `tl.where` used to produce after paying for the load.
        if (not HAS_INDX_VALID) or valids[i]:
            # iterate over split_k partial values
            for b in tl.range(0, B):
                if USE_TDM and EVEN_N:
                    row = b * M + indxs[i]
                    vals = tl.reshape(x_desc.load([row, pid_n * BLOCK_N]), (BLOCK_N,))
                else:
                    x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                    if EVEN_N:
                        vals = tl.load(x_row_ptr)
                    else:
                        vals = tl.load(x_row_ptr, mask=x_n_mask, other=0.0)
                vals = vals.to(tl.float32)
                curr += vals

        if HAS_INDX_VALID:
            # Redundant now that the load is skipped -- kept because it also
            # holds the invariant the activation below depends on: a masked slot
            # must contribute exactly 0, not swiglu(0), which is not the same
            # value once SWIGLU_ADD_RESIDUAL is on.
            curr = tl.where(valids[i], curr, 0.0)

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(
                curr[None, :], alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL
            )
        curr = tl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # Step 9: optional external residual fold-in: load residual at this
    # tile and add to acc before writeback. Same per-token-row layout as Out.
    if HAS_EXT_RESIDUAL:
        res_offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        res_ptr = Residual + pid_t * stride_extres_m + res_offs_n * stride_extres_n
        if EVEN_N:
            res = tl.load(res_ptr).to(tl.float32)
            acc = acc + res
        else:
            res_mask = res_offs_n < Nrem
            res = tl.load(res_ptr, mask=res_mask, other=0.0).to(tl.float32)
            acc = acc + res
    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        tl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT) < Nrem
        tl.store(out_ptr, acc, mask=out_n_mask)
