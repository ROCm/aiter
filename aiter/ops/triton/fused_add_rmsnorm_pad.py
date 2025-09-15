import torch
import triton
import triton.language as tl


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.jit
def _fused_add_rmsnorm_pad(
    x_ptr,
    res_ptr,
    out_ptr,
    res_out_ptr,
    weight_ptr,
    eps,
    M,
    N,
    N_OUT,
    x_stride_m,
    x_stride_n,
    res_stride_m,
    res_stride_n,
    out_stride_m,
    out_stride_n,
    res_out_stride_m,
    res_out_stride_n,
    HAS_RES: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.assume(x_stride_m > 0)
    tl.assume(x_stride_n > 0)
    tl.assume(res_stride_m > 0)
    tl.assume(res_stride_n > 0)
    tl.assume(out_stride_m > 0)
    tl.assume(out_stride_n > 0)

    pid_m = tl.program_id(0)
    tl.assume(pid_m >= 0)

    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < N
    x = tl.load(
        x_ptr + pid_m * x_stride_m + n_offs * x_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    if HAS_RES:
        res = tl.load(
            res_ptr + pid_m * res_stride_m + n_offs * res_stride_n,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x = x + res

    w = tl.load(
        weight_ptr + n_offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = _rmsmorm_op(x, w, N, eps).to(out_ptr.dtype.element_ty)

    tl.store(
        out_ptr + pid_m * out_stride_m + n_offs * out_stride_n,
        out,
        mask=(n_offs < N_OUT),
    )
    if HAS_RES:
        tl.store(
            res_out_ptr + pid_m * res_out_stride_m + n_offs * res_out_stride_n,
            x,
            mask=mask,
        )


def fused_add_rmsnorm_pad(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor = None,
    x_pad_to_multiple: int = 0,
):
    M, N = x.shape

    if x_pad_to_multiple > 0:
        N_out = triton.cdiv(N, x_pad_to_multiple) * x_pad_to_multiple
    else:
        N_out = N
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)

    res_out = None
    if res is not None:
        M2, N2 = res.shape
        assert M == M2, "Shape error!"
        assert N == N2, "Shape error!"
        res_out = torch.empty((M, N), dtype=res.dtype, device=res.device)
    BLOCK_SIZE_N = triton.next_power_of_2(N_out)
    _fused_add_rmsnorm_pad[(M,)](
        x,
        res,
        out,
        res_out,
        weight,
        epsilon,
        M,
        N,
        N_out,
        x.stride(0),
        x.stride(1),
        res.stride(0) if res is not None else 0,
        res.stride(1) if res is not None else 0,
        out.stride(0),
        out.stride(1),
        res_out.stride(0) if res is not None else 0,
        res_out.stride(1) if res is not None else 0,
        HAS_RES=(res is not None),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    if res is not None:
        return out, res_out
    return out
