import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.swiglu import _swiglu_kernel
from aiter.ops.triton.utils.types import str_to_torch_dtype


def _run_swiglu_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch the Triton SwiGLU kernel and return the output tensor."""
    assert a.shape == b.shape
    assert a.device.type == "cuda" and b.device.type == "cuda"

    m, n = a.shape
    out = torch.empty_like(a)

    max_fused_size = 65536 // a.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(n))
    block_size = max(block_size, 1)

    grid = lambda meta: (m,)  # noqa: E731
    _swiglu_kernel[grid](
        out,
        a,
        b,
        out.stride(0),
        a.stride(0),
        b.stride(0),
        m,
        n,
        BLOCK_SIZE=block_size,
    )
    return out


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "m, n",
    [
        (1823, 781),
        (1, 1),
        (128, 1),
        (1, 128),
        (8192, 8192),
        (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 65536),
        (1, 131072),
        (64, 4096),
    ],
)
def test_swiglu(m, n, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)

    a = torch.randn((m, n), dtype=dtype, device="cuda")
    b = torch.randn((m, n), dtype=dtype, device="cuda")

    y_triton = _run_swiglu_kernel(a, b)
    y_torch = torch.nn.functional.silu(b).to(dtype) * a

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
