import pytest
import torch

from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.utils.types import str_to_torch_dtype


# pytest
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1823, 781),
        (1, 1),
        (128, 1),
        (1, 128),
        (8192, 8192),
        (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 131072),
        (1, 89999),
    ],
)
def test_softmax(M, N, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("layout", ["col_step", "transposed"])
def test_softmax_non_contiguous(layout, dtype):
    # Inputs with a column stride other than 1.
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    if layout == "col_step":
        x = torch.randn(4, 200, dtype=dtype, device="cuda")[:, ::2]
    else:
        x = torch.randn(100, 4, dtype=dtype, device="cuda").t()
    y_triton = softmax(x)
    y_torch = torch.softmax(x.float(), dim=1).to(dtype)
    torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=1e-2)
