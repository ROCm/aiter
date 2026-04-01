import functools
import torch


def _to_flydsl_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"unsupported dtype: {dtype}")


@functools.cache
def _get_rmsnorm_launch(M: int, N: int, dtype_str: str):
    from kernels.rmsnorm_kernel import build_rmsnorm_module

    return build_rmsnorm_module(M, N, dtype_str)


@torch._dynamo.disable
def flydsl_rmsnorm(x, gamma, eps):
    if not x.is_cuda:
        raise RuntimeError("FlyDSL RMSNorm requires CUDA tensors")

    if gamma.device != x.device:
        raise RuntimeError("gamma must be on same device as x")

    orig_shape = x.shape
    N = x.shape[-1]

    x_2d = x.reshape(-1, N).contiguous()
    gamma = gamma.to(dtype=x_2d.dtype).contiguous()

    M = x_2d.shape[0]
    dtype_str = _to_flydsl_dtype(x_2d.dtype)

    launch_fn = _get_rmsnorm_launch(M, N, dtype_str)

    out = torch.empty_like(x_2d)

    stream = torch.cuda.current_stream()
    launch_fn(x_2d, gamma, out, M, stream=stream)

    return out.reshape(orig_shape)
