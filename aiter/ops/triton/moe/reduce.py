from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.reduce import (
    _reduce_grouped,
    _scatter_grouped,
)
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.reduce import (
        reduce_grouped_gluon as _reduce_grouped_gluon,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.reduce import (
        reduce_grouped_gluon_num_warps as _reduce_grouped_gluon_num_warps,
    )
except (ImportError, ModuleNotFoundError):
    _reduce_grouped_gluon = None
    _reduce_grouped_gluon_num_warps = None


@dataclass(frozen=True)
class EpCombineScatter:
    """Where a grouped GEMM's un-reduced rows go, in place of a local reduce.

    ``out`` is the destination the rows are placed in -- for expert-parallel
    combine, a strided view over the symmetric combine-staging window covering
    every peer's slot region, so one row index selects both the peer and the slot
    within it. ``dst_row[m]`` is that row index for sorted row ``m``, negative
    where the row must not be delivered at all.

    Handing both to the GEMM (rather than reducing and letting the caller scatter)
    is what removes a full pass over the output: the rows are already in registers
    when their destination is known.

    ``fused`` asks the GEMM epilogue to place the rows itself, so no
    ``_scatter_grouped`` launch happens at all. It needs a kernel with the
    EP_SCATTER epilogue (currently the gfx1250 gluon a8w4 prefill/decode
    kernels); every other path ignores it and runs the standalone scatter, which
    produces the same bytes either way. Set it False to A/B the two.
    """

    out: torch.Tensor
    dst_row: torch.Tensor
    fused: bool = True

    def __post_init__(self):
        if self.out.ndim != 2:
            raise ValueError(f"out must be 2-D, got shape {tuple(self.out.shape)}")
        if self.out.stride(-1) != 1:
            raise ValueError(
                f"out must be contiguous along its last dim, got strides "
                f"{tuple(self.out.stride())}"
            )
        if self.dst_row.dtype != torch.int32 or not self.dst_row.is_contiguous():
            raise ValueError("dst_row must be contiguous int32")


def scatter_grouped(
    x: torch.Tensor,
    dst_row: torch.Tensor,
    out: torch.Tensor,
):
    """Place each row of `x` at ``out[dst_row[m]]``; skip rows with dst_row < 0.

    The expert-parallel counterpart of :func:`reduce_grouped` -- see
    ``_scatter_grouped`` for why the reduce goes away rather than moving. `x` is
    the matmul output, ``[split_k, M, N]``; split-k partials are summed on the way
    out, so the destination sees one finished row.

    Returns `out`.
    """
    assert x.ndim == 3, f"x must be [split_k, M, N], got {tuple(x.shape)}"
    m_rows = x.shape[1]
    assert dst_row.numel() >= m_rows, (
        f"dst_row has {dst_row.numel()} entries but the matmul produced "
        f"{m_rows} rows"
    )
    assert (
        x.shape[-1] == out.shape[-1]
    ), f"row width mismatch: x {x.shape[-1]} vs out {out.shape[-1]}"

    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)
    _scatter_grouped[(num_blocks * m_rows,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        dst_row,
        x.shape[0],
        m_rows,
        x.shape[-1],
        num_blocks,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        num_warps=2,
    )
    return out


def validate_reduce_out(out, shape, dtype, device):
    """Accept a caller-provided `out` buffer for reduce_grouped, or allocate one.

    Lets a caller hand in a *slice* of a larger tensor so the reduction writes
    its final rows in place, instead of writing a fresh buffer that the caller
    then copies. The reduction reads `out.stride(0)` / `out.stride(1)`, so any
    row-major-rows layout works -- `stride(0)` need not equal the row width, and
    that is exactly what a `big[:n]` view gives when `big` is wider in rows only.

    Asserts rather than silently allocating on a mismatch: a wrong shape means
    the caller's model of the kernel's output geometry has broken, and taking
    the fresh buffer would hide that behind a missing in-place write.
    """
    if out is None:
        return torch.empty(shape, device=device, dtype=dtype)
    assert tuple(out.shape) == tuple(shape), (
        f"provided output buffer has shape {tuple(out.shape)}, "
        f"but this call produces {tuple(shape)}"
    )
    assert out.dtype == dtype, (
        f"provided output buffer has dtype {out.dtype}, but this call "
        f"produces {dtype}"
    )
    assert out.device == torch.device(device), (
        f"provided output buffer is on {out.device}, but this call runs on " f"{device}"
    )
    # Only the trailing dim must be packed; stride(0) is free (slice of a
    # taller tensor).
    assert out.stride(-1) == 1, (
        f"provided output buffer must be contiguous along its last dim, "
        f"got strides {tuple(out.stride())}"
    )
    return out


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype=None,
    swiglu_add_residual: bool = True,
    residual: torch.Tensor | None = None,
    indx_valid: torch.Tensor | None = None,
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

    # Gluon path on gfx1250 for the plain grouped combine; swiglu-fused (MoE1 split-k) reductions, reduction_n != 1, and non-contiguous inputs stay on the Triton _reduce_grouped.
    use_gluon = (
        is_tdm_avail()
        # the gluon reduce has no per-gate validity support
        and indx_valid is None
        and indx is not None
        and not apply_swiglu
        and reduction_n == 1
        and x.ndim == 3
        and x.is_contiguous()
        and indx.is_contiguous()
    )
    if use_gluon:
        B, M, N = x.shape[0], x.shape[1], x.shape[2]
        npad = triton.next_power_of_2(N)
        has_ext_residual = residual is not None
        if has_ext_residual:
            assert residual.shape == out.shape, (
                f"residual.shape {tuple(residual.shape)} must match "
                f"out.shape {tuple(out.shape)}"
            )
        gluon_num_warps = _reduce_grouped_gluon_num_warps(npad)
        _reduce_grouped_gluon[(num_groups,)](
            X=x,
            Out=out,
            InIndx=indx,
            Residual=residual if has_ext_residual else out,
            stride_xm=x.stride(1),
            stride_om=out.stride(0),
            stride_on=out.stride(1),
            stride_res_m=residual.stride(0) if has_ext_residual else 0,
            stride_res_n=residual.stride(1) if has_ext_residual else 0,
            M=M,
            N=N,
            NPAD=npad,
            B=B,
            K=K,
            NUM_WARPS=gluon_num_warps,
            HAS_EXT_RESIDUAL=has_ext_residual,
            num_warps=gluon_num_warps,
        )
        return out

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
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        indx_valid,
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
        K=K,
        HAS_INDX_VALID=indx_valid is not None,
        SWIGLU_ADD_RESIDUAL=swiglu_add_residual,
        USE_TDM=is_tdm_avail(),
        Residual=residual,
        stride_extres_m=res_stride_m,
        stride_extres_n=res_stride_n,
        HAS_EXT_RESIDUAL=has_ext_residual,
        num_warps=2,
    )
    return out
