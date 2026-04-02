import functools
import torch

from aiter.jit.utils.torch_guard import torch_compile_guard


def _to_flydsl_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"unsupported dtype: {dtype}")


def flydsl_rmsnorm_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@functools.cache
def _get_rmsnorm_launch(M: int, N: int, dtype_str: str):
    from .kernels.rmsnorm_kernel import build_rmsnorm_module
    return build_rmsnorm_module(M, N, dtype_str)


@torch._dynamo.disable
def _flydsl_rmsnorm_impl(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("FlyDSL RMSNorm requires CUDA tensors")
    if gamma.device != x.device:
        raise RuntimeError("gamma must be on same device as x")
    if eps != 1e-5:
        raise RuntimeError(f"FlyDSL RMSNorm currently expects eps=1e-5, got {eps}")

    orig_shape = x.shape
    N = x.shape[-1]

    x_2d = x.reshape(-1, N).contiguous()

    # FlyDSL JIT cannot export tensors requiring grad.
    gamma_launch = gamma.detach().to(
        device=x_2d.device,
        dtype=x_2d.dtype,
    ).contiguous()

    M = x_2d.shape[0]
    dtype_str = _to_flydsl_dtype(x_2d.dtype)

    launch_fn = _get_rmsnorm_launch(M, N, dtype_str)

    out = torch.empty_like(x_2d)
    stream = torch.cuda.current_stream()
    launch_fn(x_2d, gamma_launch, out, M, stream=stream)

    return out.reshape(orig_shape)


@torch_compile_guard(gen_fake=flydsl_rmsnorm_fake)
def flydsl_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _flydsl_rmsnorm_impl(x, gamma, eps)


