import torch
import triton
from aiter.ops.triton._triton_kernels.fused_add_rmsnorm_pad import (
    _fused_add_rmsnorm_pad,
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
    )

    if res is not None:
        return out, res_out
    return out
