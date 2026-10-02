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
    """Move `res` to CPU so it matches `ref`'s device."""
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


def _assert_equal(
    res: torch.Tensor, ref: torch.Tensor, *, equal_nan: bool = False
) -> None:
    res = _to_cpu(res, ref)
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)


def TEST_assert_close(*a, **kw):
    return _assert_close(*a, **kw)


def TEST_assert_equal(*a, **kw):
    return _assert_equal(*a, **kw)


# ------------------------------- tests --------------------------------------
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", DIM2)
@pytest.mark.parametrize("topk", K)
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_topk(batch_size, hiddensize, topk, largest, dtype):
    """Correctness check against torch.topk on small inputs."""
    torch.manual_seed(0)
    x = torch.arange(hiddensize, dtype=dtype, device=DEVICE).repeat(batch_size, 1)

    # Per-row shuffle so every row has a distinct permutation
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize, device=DEVICE)]

    ref_value, ref_index = torch.topk(x, topk, largest=largest)
    res_value, res_index = triton_topk(x, topk, largest=largest)

    TEST_assert_close(res_value, ref_value.cpu(), dtype)
    TEST_assert_equal(res_index.cpu(), ref_index.cpu())


def _rows_with_inf_and_extrema(batch_size, hiddensize, topk, case, dtype):
    """Rows whose top-k has to include -inf or the dtype's extrema.

    Row r has ``r % (2 * topk)`` "high" entries at random positions, so many
    rows have fewer than ``topk`` of them; every other entry is a "low" value.
    "masked": high entries are randn, low entries are -inf (masked logits).
    "extrema": high entries are drawn from {finfo.min, -1, -0.0, 0.0, 1,
    finfo.max, +inf}, low entries from {-inf, finfo.min}.
    "signed_zero": high entries are randn, low entries are -0.0 or +0.0.
    """
    n_high = torch.arange(batch_size, device=DEVICE) % (2 * topk)
    rank = torch.rand(batch_size, hiddensize, device=DEVICE).argsort(dim=1)
    rank = rank.argsort(dim=1)  # rank[r, c]: position of column c in a random order
    is_low = rank >= n_high[:, None]
    shape = (batch_size, hiddensize)
    if case == "masked":
        high = torch.randn(shape, device=DEVICE)
        low = torch.full(shape, float("-inf"), device=DEVICE)
    elif case == "signed_zero":
        high = torch.randn(shape, device=DEVICE)
        pool = torch.tensor([-0.0, 0.0], device=DEVICE)
        low = pool[torch.randint(0, pool.numel(), shape, device=DEVICE)]
    else:
        finfo = torch.finfo(dtype)
        pool = torch.tensor(
            [finfo.min, -1.0, -0.0, 0.0, 1.0, finfo.max, float("inf")], device=DEVICE
        )
        high = pool[torch.randint(0, pool.numel(), shape, device=DEVICE)]
        pool = torch.tensor([float("-inf"), finfo.min], device=DEVICE)
        low = pool[torch.randint(0, pool.numel(), shape, device=DEVICE)]
    return torch.where(is_low, low, high).to(dtype)


@pytest.mark.parametrize(
    "batch_size, hiddensize, topk",
    [
        (4, 16, 8),  # 1-stage, minimal
        (64, 1000, 8),  # 1-stage, padded lanes (1000 < BLOCK=1024)
        (64, 1000, 50),
        (8, 4097, 8),  # 2-stage, last chunk has fewer than k entries
        (64, 32000, 50),  # 2-stage, vocab-sized rows
        (64, 128256, 8),
    ],
)
@pytest.mark.parametrize("case", ["masked", "extrema", "signed_zero"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_topk_inf_and_extrema(batch_size, hiddensize, topk, case, dtype):
    """-inf, finfo extrema and -0.0 must not be confused with the kernel's padding.

    Tie order is unspecified, so instead of comparing indices with torch.topk
    check that the indices are in range, distinct per row, point at the
    returned values, and that the values equal torch.topk's.
    """
    torch.manual_seed(0)
    x = _rows_with_inf_and_extrema(batch_size, hiddensize, topk, case, dtype)

    ref_value, _ = torch.topk(x.float(), topk)
    res_value, res_index = triton_topk(x, topk)

    assert res_index.shape == (batch_size, topk)
    assert bool(((res_index >= 0) & (res_index < hiddensize)).all())
    sorted_index = res_index.sort(dim=1).values
    assert bool((sorted_index[:, 1:] != sorted_index[:, :-1]).all())
    TEST_assert_equal(res_value.float(), torch.gather(x, 1, res_index).float().cpu())
    TEST_assert_equal(res_value.float(), ref_value.cpu())
