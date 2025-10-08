# rn this is ripped from regular softmax file

import torch
import pytest
from aiter.ops.triton.swiglu_am import swiglu
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
def test_swiglu(M, N, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    y_triton = swiglu(x)
    y_torch = torch.swiglu(x, axis=1)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    print("swiglu asserted")
