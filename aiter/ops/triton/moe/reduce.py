import torch
import triton
from aiter.ops.triton._triton_kernels.moe.reduce import _reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype=None,
    add_residual: bool = True,
    # Step 9: optional external residual folded into the writeback.
    residual: "torch.Tensor | None" = None,
):
    """
    Grouped row reduction used during moe scatter and also compatible with split-k reduce.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]`. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The output tensor `out`.
    """

    if indx is None and x.shape[0] == 1:
        assert residual is None, (
            "reduce_grouped early-return path can't apply external residual; "
            "either rebuild routing with K>=1 or skip residual fold for this call"
        )
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

    # Step 9: prep external residual buffer + strides for the kernel.
    if residual is not None:
        assert (
            residual.shape == out.shape
        ), f"residual.shape {tuple(residual.shape)} must match out.shape {tuple(out.shape)}"
        res_stride_m = residual.stride(0)
        res_stride_n = residual.stride(1)
        has_ext_residual = True
    else:
        res_stride_m = 0
        res_stride_n = 0
        has_ext_residual = False
    _reduce_grouped[(num_blocks * num_groups,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),  #
        out,
        out.stride(0),
        out.stride(1),  #
        indx,  #
        x.shape[0],
        x.shape[1],
        x.shape[2],
        num_blocks,
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,  #
        ADD_RESIDUAL=add_residual,
        USE_TDM=is_tdm_avail(),
        Residual=residual,
        stride_extres_m=res_stride_m,
        stride_extres_n=res_stride_n,
        HAS_EXT_RESIDUAL=has_ext_residual,
        num_warps=2,  #
    )
    return out
