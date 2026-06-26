# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

if not is_flydsl_available():
    pytest.skip("FlyDSL is not available on this device.", allow_module_level=True)

from aiter.ops.flydsl import flydsl_top_k_per_row_decode  # noqa: E402


def _decode_row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
    row_ids = torch.arange(num_rows, device=seq_lens.device, dtype=torch.int32)
    seq_rows = row_ids // next_n
    slots = row_ids % next_n
    return seq_lens[seq_rows] - next_n + slots + 1


def test_flydsl_top_k_per_row_decode_direct_fill_k2048():
    torch.manual_seed(0)
    next_n = 2
    k = 2048
    seq_lens = torch.tensor([64, 128], device="cuda", dtype=torch.int32)
    num_rows = int(seq_lens.numel()) * next_n
    logits = torch.randn((num_rows, 128), device="cuda", dtype=torch.float32)
    indices = torch.empty((num_rows, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    torch.cuda.synchronize()

    row_ends = _decode_row_ends(seq_lens, next_n, num_rows).cpu().tolist()
    for row, row_end in enumerate(row_ends):
        expected_valid = torch.arange(row_end, dtype=torch.int32)
        torch.testing.assert_close(indices[row, :row_end].cpu(), expected_valid)
        torch.testing.assert_close(
            indices[row, row_end:k].cpu(),
            torch.full((k - row_end,), -1, dtype=torch.int32),
        )


def test_flydsl_top_k_per_row_decode_long_row_k2048_matches_torch_topk_set():
    torch.manual_seed(1)
    next_n = 1
    k = 2048
    seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    logits = torch.randn((1, 4096), device="cuda", dtype=torch.float32)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        1,
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    torch.cuda.synchronize()

    _assert_topk_set_equivalent(logits[0], indices[0], k)


def test_flydsl_top_k_per_row_decode_long_row_k512_matches_torch_topk_set():
    torch.manual_seed(2)
    next_n = 1
    k = 512
    seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    logits = torch.randn((1, 4096), device="cuda", dtype=torch.float32)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        1,
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    torch.cuda.synchronize()

    _assert_topk_set_equivalent(logits[0], indices[0], k)


def _assert_topk_set_equivalent(logits_row, actual_row, k):
    """Assert ``actual_row`` is a Top-K *set* of ``logits_row`` (ties by value)."""
    L = int(logits_row.numel())
    kk = min(k, L)
    actual = actual_row[:kk].cpu()
    assert not ((actual < 0) | (actual >= L)).any(), f"invalid indices: {actual.tolist()[:8]}"
    assert len(set(actual.tolist())) == len(actual.tolist()), "duplicate indices"
    expected = torch.topk(logits_row[:L], kk).indices.cpu()
    aset, eset = set(actual.tolist()), set(expected.tolist())
    if aset == eset:
        return
    actual_only = list(aset - eset)
    expected_only = list(eset - aset)
    assert len(actual_only) == len(expected_only), "different topk set sizes"
    av = torch.tensor([logits_row[i].item() for i in actual_only]).sort().values
    ev = torch.tensor([logits_row[i].item() for i in expected_only]).sort().values
    torch.testing.assert_close(av, ev, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("k", [512, 2048, 256, 1024])
def test_flydsl_top_k_per_row_decode_unordered_set_matches_torch_topk(k):
    torch.manual_seed(3)
    next_n = 1
    seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    logits = torch.randn((1, 4096), device="cuda", dtype=torch.float32)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, 1,
        logits.stride(0), logits.stride(1), k=k, ordered=False,
    )
    torch.cuda.synchronize()
    _assert_topk_set_equivalent(logits[0], indices[0], k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("dist", ["ties", "10LSBits"])
def test_flydsl_top_k_per_row_decode_unordered_edge_distributions(k, dist):
    torch.manual_seed(11)
    next_n = 1
    L = 4096
    seq_lens = torch.tensor([L], device="cuda", dtype=torch.int32)
    if dist == "ties":
        logits = torch.randint(-16, 16, (1, L), dtype=torch.int32, device="cuda").to(
            torch.float32
        )
    else:  # 10LSBits: identical top 22 bits, random low 10 bits -> radix worst case
        rb = torch.randint(0, 2**10, (1, L), dtype=torch.int32, device="cuda")
        bits = (0x3F900000 & 0xFFFFFC00) | (rb & 0x000003FF)
        logits = bits.view(torch.float32)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, 1,
        logits.stride(0), logits.stride(1), k=k, ordered=False,
    )
    torch.cuda.synchronize()
    _assert_topk_set_equivalent(logits[0], indices[0], k)


def test_flydsl_top_k_per_row_decode_ties_k512_matches_torch_topk_set():
    next_n = 1
    k = 512
    seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    logits = torch.zeros((1, 4096), device="cuda", dtype=torch.float32)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        1,
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    torch.cuda.synchronize()

    _assert_topk_set_equivalent(logits[0], indices[0], k)
