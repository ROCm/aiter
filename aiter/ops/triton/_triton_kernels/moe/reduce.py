import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu


@triton.jit
def _reduce_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,  #
    Out,
    stride_om: tl.uint64,
    stride_on,  # output tensor
    InIndx,
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
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
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

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
        curr = tl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # Step 9: optional external residual fold-in: load residual at this
    # tile and add to acc before writeback. Same per-token-row layout as Out.
    if HAS_EXT_RESIDUAL:
        res_offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        res_ptr = (
            Residual + pid_t * stride_extres_m + res_offs_n * stride_extres_n
        )
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
