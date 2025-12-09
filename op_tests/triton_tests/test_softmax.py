import torch
import pytest
import triton
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.test_common import perftest, checkAllclose

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

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)

@perftest()
def run_torch(input, dim=1):
    """Run torch softmax with perftest timing"""
    output = torch.softmax(input, dim=dim)
    return output

@perftest()
def run_triton(input, dim=1):
    """Run triton softmax with perftest timing"""
    output = softmax(input)
    return output

def benchmark_softmax_perftest(M, N, dtype_str="bf16"):
    """
    Benchmark softmax using perftest (consistent with CK kernel testing)

    Args:
        M: Number of rows
        N: Number of columns
        dtype_str: Data type string ("fp32", "fp16", "bf16")
    """
    dtype = str_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    # Run benchmarks with perftest
    (a, *_), avg_torch = run_torch(x)
    (b, *_), avg_triton = run_triton(x)

    # Calculate speedup
    speedup = avg_torch / avg_triton

    # Check correctness and print results
    msg = (f"[perf] Shape: ({M}, {N}), dtype: {dtype_str}, "
           f"Torch avg: {avg_torch:<8.2f} us, Triton avg: {avg_triton:<8.2f} us, "
           f"Speedup: {speedup:<5.2f}x")

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    checkAllclose(a, b, rtol=rtol, atol=atol, msg=msg)

    return avg_triton, avg_torch, speedup

benchmark_softmax_perftest(32768, 8192, "bf16")

test_softmax(32768, 8192, "bf16")
