import torch
import pytest
from aiter.ops.triton.softmax import softmax,softmax2
from aiter.ops.triton.utils.types import str_to_torch_dtype


dtype_test_cases = ["fp32", "fp16", "bf16"]
shape_test_cases = [
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
    ]
# pytest
@pytest.mark.parametrize("dtype", dtype_test_cases)
@pytest.mark.parametrize("M, N", shape_test_cases)

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

@pytest.mark.parametrize("dtype", dtype_test_cases)
@pytest.mark.parametrize("M, N", shape_test_cases)
def test_softmax2(M, N, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    y_torch = torch.softmax(x, axis=1)
    y_triton2 = softmax2(x)
    
    #If tests need to be made with original softmax
    y_triton = softmax(x)
    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(y_torch, y_triton2, atol=atol, rtol=rtol)
    