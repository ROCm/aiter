import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.swiglu import swiglu,swiglu_ref
from aiter.ops.triton.utils.types import str_to_torch_dtype

@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("M, K, N",
    [
        (32, 64, 128),
        (16, 32, 32),
        (128, 16, 64),
        (8, 8, 8),
        (64, 128, 32),
    ]
)

def test_swiglu(M, K, N, dtype):
    dtype = str_to_torch_dtype[dtype]
    torch.manual_seed(0)
    x = torch.randn(M, 2*K, dtype=dtype,device="cuda")
    W = torch.randn(K, N, dtype=dtype,device="cuda")
    V = torch.randn(K, N, dtype=dtype,device="cuda")
    b = torch.randn(N, dtype=dtype,device="cuda")
    c = torch.randn(N, dtype=dtype,device="cuda")
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16

    y_triton = swiglu(x, W, V, b, c, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    y_ref = swiglu_ref(x, W, V, b, c)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(y_triton, y_ref, atol=atol, rtol=rtol)
    
