import pytest
import torch
from aiter.ops.triton.topk import topk as triton_topk
DEVICE = "cuda"

# FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
FLOAT_DTYPES = [torch.float32]
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2 = [16, 128256]
K = [2, 8]

def _to_cpu(res: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Move `res` to CPU so it matches `ref`’s device."""
    if res.device.type != "cpu":
        res = res.cpu()
    return res

def _assert_close(
    res: torch.Tensor,
    ref: torch.Tensor,
    dtype: torch.dtype,
    *,
    equal_nan: bool = False,
    reduce_dim: int = 1,
) -> None:
    res = _to_cpu(res, ref)
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)

def _assert_equal(res: torch.Tensor, ref: torch.Tensor, *, equal_nan: bool = False) -> None:
    res = _to_cpu(res, ref)
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)

TEST_assert_close = lambda *a, **kw: _assert_close(*a, **kw)
TEST_assert_equal = lambda *a, **kw: _assert_equal(*a, **kw)

# ─────────────────────────────── tests ──────────────────────────────────────
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", DIM2)
@pytest.mark.parametrize("topk", K)
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_topk(batch_size, hiddensize, topk, largest, dtype):
    """Correctness check against torch.topk on small inputs."""
    x = torch.arange(hiddensize, dtype=dtype, device=DEVICE).repeat(batch_size, 1)

    # Per-row shuffle so every row has a distinct permutation
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize, device=DEVICE)]

    ref_value, ref_index = torch.topk(x, topk, largest=largest)
    res_value, res_index = triton_topk(x, topk, largest=largest)

    TEST_assert_close(res_value, ref_value.cpu(), dtype)
    TEST_assert_equal(res_index.cpu(), ref_index.cpu())
