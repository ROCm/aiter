from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.normalization.fused_add_rmsnorm_pad import (
    _fused_add_rmsnorm_pad,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_add_rmsnorm_pad(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor = None,
    x_pad_to_multiple: int = 0,
    kernel_type: str = "tdm",
    backend: Optional[str] = None,
):
    M, N = x.shape

    if backend is None:
        backend = "gluon" if get_arch() == "gfx1250" else "triton"

    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert get_arch() == "gfx1250", "Gluon kernel is only supported on gfx1250"
        from aiter.ops.triton._gluon_kernels.gfx1250.norm.fused_add_rmsnorm_pad import (
            _KERNEL_MAP,
        )

        assert (
            kernel_type in _KERNEL_MAP
        ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"

        _LOGGER.info(
            f"FUSED_ADD_RMSNORM_PAD [gluon/gfx1250]: x={tuple(x.shape)} weight={tuple(weight.shape)} "
            f"kernel={kernel_type}"
        )

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

        _KERNEL_MAP[kernel_type][(M,)](
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
