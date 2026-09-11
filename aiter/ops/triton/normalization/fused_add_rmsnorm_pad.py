import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.norm.fused_add_rmsnorm_pad import (
    _gluon_fused_add_rmsnorm_pad_kernel,
)
from aiter.ops.triton._triton_kernels.normalization.fused_add_rmsnorm_pad import (
    _fused_add_rmsnorm_pad,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _get_config(block_size_n: int, backend: str) -> dict:
    config_dir = resolve_config_dir(
        "normalization", "FUSED_ADD_RMSNORM_PAD", backend=backend
    )
    raw = load_config_json(f"{config_dir}/DEFAULT.json")
    for bound in sorted(int(k[len("N_LEQ_") :]) for k in raw if k.startswith("N_LEQ_")):
        if block_size_n <= bound:
            return dict(raw[f"N_LEQ_{bound}"])
    return dict(raw["any"])


def fused_add_rmsnorm_pad(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor = None,
    x_pad_to_multiple: int = 0,
    backend: str | None = None,
):
    """
    Fuses rmnsnorm, add, and padding into a single kernel.

    Parameters:
        x (torch.Tensor): Input tensor of shape (M, N)
        weight (torch.Tensor): Weight tensor of shape (N, )
        epsilon (float): Epsilon value for rmsnorm
        res (torch.Tensor): Residual tensor of shape (M, N) (optional)
        x_pad_to_multiple (int): Pad x to multiple of this value and return output of shape (M, N_out) (optional)
        backend (str): Decides which kernel to use from the available options (triton or gluon) (optional)

    Returns:
        x (torch.Tensor): Output tensor of shape (M, N) if x_pad_to_multiple is 0, otherwise shape (M, N_out)
        res_out (torch.Tensor): Residual output tensor of shape (M, N) (optional)

    """

    M, N = x.shape

    if backend is None:
        backend = "gluon" if get_arch() == "gfx1250" else "triton"

    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        if get_arch() == "gfx1250":

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
            config = _get_config(BLOCK_SIZE_N, "gluon")
            NUM_WARPS = config["num_warps"]
            _gluon_fused_add_rmsnorm_pad_kernel[(M,)](
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
                num_warps=NUM_WARPS,
            )

            if res is not None:
                return out, res_out
            return out
        else:
            _LOGGER.warning(
                "Gluon kernel is only supported on gfx1250, using triton kernel instead"
            )
            backend = "triton"

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
