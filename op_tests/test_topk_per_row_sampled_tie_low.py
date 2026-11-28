# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""top_k_per_row_prefill_sampled(tie_low=True): ties at the k-th score keep the
lowest column ids, the set the stable one-block kernel keeps.

Covers both emit paths of the tie-low select:
  - rows rounded to a few levels (1/64 and 1/8 steps): many copies of the pivot,
    selected in LDS (block_gather_topk_low);
  - constant rows: the radix fallback's crossing bucket is a single value, so the
    row is emitted by the stream path (block_gather_stream_low).
Gaussian rows (practically tie-free) check that tie_low changes nothing there.

    python op_tests/test_topk_per_row_sampled_tie_low.py
    pytest op_tests/test_topk_per_row_sampled_tie_low.py
"""

import pytest
import torch

K = 2048
SHAPES = [(64, 81920), (32, 163840)]
KINDS = ["gaussian", "ties64", "ties8", "const"]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="top_k_per_row_prefill_sampled needs a GPU"
)


def _supported():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() in ("gfx942", "gfx950")


def _twiddle(bits: int) -> int:
    bits &= 0xFFFFFFFF
    return bits ^ (0 if (bits >> 31) else 0x7FFFFFFF)


def _cpu_ref(row: torch.Tensor, k: int) -> list:
    """Bit-faithful reference: rank by the kernel's key, ties to the smaller column."""
    raw = row.detach().cpu().contiguous().view(torch.int32).numpy()
    order = sorted(range(len(raw)), key=lambda i: (_twiddle(int(raw[i])), i))
    return sorted(order[:k])


def _logits(kind, m, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, dtype=torch.float32, device="cuda", generator=g)
    if kind == "ties64":
        x = (x * 64).round() / 64
    elif kind == "ties8":
        x = (x * 8).round() / 8
    elif kind == "const":
        x = torch.zeros_like(x)
    return x


def _run(x, rs, re, tie_low):
    from aiter.ops.topk import top_k_per_row_prefill, top_k_per_row_prefill_sampled

    m = x.shape[0]
    idx = torch.empty(m, K, dtype=torch.int32, device="cuda")
    if tie_low is None:
        top_k_per_row_prefill(
            x, rs, re, idx, None, m, x.stride(0), x.stride(1), k=K, stable=True
        )
    else:
        top_k_per_row_prefill_sampled(
            x, rs, re, idx, None, m, x.stride(0), x.stride(1), k=K, tie_low=tie_low
        )
    torch.cuda.synchronize()
    return torch.sort(idx, dim=1).values


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ragged", [False, True])
def test_tie_low_matches_stable(m, n, kind, ragged):
    if not _supported():
        pytest.skip("sampled top-k is built for gfx942 / gfx950")
    x = _logits(kind, m, n, m * 131 + n + KINDS.index(kind))
    rs = torch.zeros(m, dtype=torch.int32, device="cuda")
    if ragged:
        re = (n - torch.arange(m, device="cuda", dtype=torch.int32) * 37).clamp_min(
            K + 1
        )
    else:
        re = torch.full((m,), n, dtype=torch.int32, device="cuda")
    want = _run(x, rs, re, None)
    got = _run(x, rs, re, True)
    assert torch.equal(
        got, want
    ), f"{int((got != want).any(1).sum())} of {m} rows differ from the stable kernel"
    assert torch.equal(
        _run(x, rs, re, True), got
    ), "tie_low is not run-to-run identical"
    for r in (0, m - 1):
        assert got[r].cpu().tolist() == _cpu_ref(x[r, : int(re[r])], K), f"row {r}"


def test_tie_low_rejects_values_and_nan_high():
    if not _supported():
        pytest.skip("sampled top-k is built for gfx942 / gfx950")
    from aiter.ops.topk import top_k_per_row_prefill_sampled

    m, n = 4, 81920
    x = _logits("ties8", m, n, 1)
    rs = torch.zeros(m, dtype=torch.int32, device="cuda")
    re = torch.full((m,), n, dtype=torch.int32, device="cuda")
    idx = torch.empty(m, K, dtype=torch.int32, device="cuda")
    vals = torch.empty(m, K, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        top_k_per_row_prefill_sampled(x, rs, re, idx, vals, m, n, 1, k=K, tie_low=True)
    with pytest.raises(ValueError):
        top_k_per_row_prefill_sampled(
            x, rs, re, idx, None, m, n, 1, k=K, tie_low=True, nan_high=True
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
