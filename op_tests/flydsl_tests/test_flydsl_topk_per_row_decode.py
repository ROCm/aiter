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

SUPPORTED_KS = (256, 512, 1024, 2048)
DISTRIBUTIONS = ("random", "ties", "10LSBits")

# Lengths chosen to cross every tier of the K=2048 kernel
# (short <= 16384, mid <= 65536, long > 65536) and to include the ``+1``/``+3``
# unaligned lengths that exercise the vectorized-load tail masking. L == 2048
# hits the direct-fill boundary at k == 2048.
TIER_LENGTHS = (2048, 2049, 8192, 16384, 32768, 32769, 65536, 120000, 120003)

# Physical width for padded (poison-tail) runs. Wide enough that a whole-row
# over-scan is caught, small enough to keep per-case allocation cheap.
_PAD_WIDTH = 200_000
_POISON = 1e30
_SEED = 1234


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _decode_row_ends(
    seq_lens: torch.Tensor,
    next_n: int,
    num_rows: int,
) -> torch.Tensor:
    row_ids = torch.arange(num_rows, device=seq_lens.device, dtype=torch.int32)
    seq_rows = row_ids // next_n
    slots = row_ids % next_n
    return (seq_lens[seq_rows] - next_n + slots + 1).clamp(min=0)


def _fill_distribution(shape, dist: str, device) -> torch.Tensor:
    if dist == "random":
        return torch.randn(shape, dtype=torch.float32, device=device)

    if dist == "ties":
        # Small integer range -> many exactly-equal keys.
        return torch.randint(-16, 16, shape, dtype=torch.int32, device=device).to(
            torch.float32
        )

    if dist == "10LSBits":
        # Identical top 22 bits, random low 10 bits: radix worst case
        low = torch.randint(0, 2**10, shape, dtype=torch.int32, device=device)
        bits = (0x3F900000 & 0xFFFFFC00) | (low & 0x000003FF)
        return bits.view(torch.float32)

    if dist == "constant":
        # Every value identical -> the whole row is one tie group -> the boundary
        # bucket holds all elements and the back-fill must pick exactly k.
        return torch.full(shape, 3.5, dtype=torch.float32, device=device)

    raise ValueError(f"unknown distribution: {dist}")


def _build_case(num_rows, width, next_n, seq_len, dist, poison, stride_pad=0):
    """Return (logits, seq_lens, row_ends) for one shape."""
    device = torch.device("cuda")
    torch.manual_seed(_SEED)

    n_seqs = num_rows // next_n
    seq_vals = (
        list(seq_len) if isinstance(seq_len, (list, tuple)) else [seq_len] * n_seqs
    )
    seq_lens = torch.tensor(seq_vals, dtype=torch.int32, device=device)
    row_ends = _decode_row_ends(seq_lens, next_n, num_rows)
    phys = _fill_distribution((num_rows, width + stride_pad), dist, device)

    logits = phys[:, :width] if stride_pad else phys
    if poison is not None:
        for row, end in enumerate(row_ends.tolist()):
            if end < width:
                logits[row, end:] = poison

    return logits, seq_lens, row_ends


def _run(logits, seq_lens, num_rows, next_n, k):
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
        ordered=False,
    )
    torch.cuda.synchronize()
    return indices


def _assert_row_topk_set(logits_row, actual_row, k, row_len):
    """Assert ``actual_row`` is a Top-K *set* of ``logits_row[:row_len]``"""
    valid = min(k, row_len)
    a = actual_row[:valid]
    bad = (a < 0) | (a >= row_len)
    assert not bad.any(), f"invalid/poison indices: {a[bad][:8].tolist()}"
    assert len(set(a.tolist())) == len(a.tolist()), "duplicate indices"

    if row_len < k:
        pad = actual_row[row_len:k]
        assert (pad == -1).all(), (
            f"expected -1 padding, got {pad[pad != -1][:8].tolist()}"
        )

    expected = torch.topk(logits_row[:row_len], valid).indices
    a_set, e_set = set(a.tolist()), set(expected.tolist())
    if a_set == e_set:
        return

    a_only = sorted(a_set - e_set)
    e_only = sorted(e_set - a_set)
    assert len(a_only) == len(e_only), (
        f"set size mismatch: {len(a_only)} extra vs {len(e_only)} missing"
    )

    av = torch.tensor([logits_row[i].item() for i in a_only]).sort().values
    ev = torch.tensor([logits_row[i].item() for i in e_only]).sort().values
    torch.testing.assert_close(av, ev, rtol=0, atol=0)


def _assert_batch(logits, indices, row_ends, k):
    """Assert Top-K set-equivalence for every row of a launch."""
    logits_cpu = logits.detach().cpu()
    actual = indices.detach().cpu()
    for row, end in enumerate(row_ends.tolist()):
        _assert_row_topk_set(logits_cpu[row], actual[row], k, int(end))


def _check_set_equivalence(k, num_rows, next_n, seq_len, dist, padded):
    """Build one shape, run the kernel, assert Top-K set-equivalence per row."""
    width = max(_PAD_WIDTH, seq_len) if padded else seq_len
    poison = _POISON if padded else None
    logits, seq_lens, row_ends = _build_case(
        num_rows, width, next_n, seq_len, dist, poison
    )
    indices = _run(logits, seq_lens, num_rows, next_n, k)
    _assert_batch(logits, indices, row_ends, k)


# --------------------------------------------------------------------------- #
# Top-K set-equivalence vs torch.topk across every
# tier/boundary (L) and radix worst case (dist), at single-row and batched
# (num_rows 1, 8) cooperation. Every row carries a poison tail past row_len, so
# any over-scan pulls a huge value into the result and fails.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
@pytest.mark.parametrize("L", TIER_LENGTHS)
@pytest.mark.parametrize("num_rows", [1, 8])
def test_k2048_poison_tail_matrix(num_rows, L, dist):
    _check_set_equivalence(2048, num_rows, 1, L, dist, padded=True)


# --------------------------------------------------------------------------- #
# Curated set-equivalence scenarios: the coverage the K=2048 padded matrix does
# not reach -- unpadded (contiguous) shapes, K != 2048 across tiers, and
# next_n > 1 (MTP) with per-slot differing row lengths.
# --------------------------------------------------------------------------- #
# id -> (k, num_rows, next_n, seq_len, dist, padded)
_EQUIVALENCE_SCENARIOS = {
    # unpadded (width == L): contiguous full-width path, unaligned lengths
    "unpadded-k2048-short-random-rows1": (2048, 1, 1, 8192, "random", False),
    "unpadded-k2048-short+1-random-rows8": (2048, 8, 1, 2049, "random", False),
    "unpadded-k2048-mid-ties-rows8": (2048, 8, 1, 32769, "ties", False),
    "unpadded-k2048-long-10lsb-rows1": (2048, 1, 1, 120003, "10LSBits", False),
    # every supported K != 2048 across short / mid / long tiers
    "k256-short-random": (256, 1, 1, 4096, "random", True),
    "k256-mid-ties": (256, 1, 1, 32769, "ties", True),
    "k256-long-10lsb": (256, 1, 1, 120003, "10LSBits", True),
    "k512-short-ties": (512, 1, 1, 4096, "ties", True),
    "k512-mid-10lsb": (512, 1, 1, 32769, "10LSBits", True),
    "k512-long-random": (512, 1, 1, 120003, "random", True),
    "k1024-short-10lsb": (1024, 1, 1, 4096, "10LSBits", True),
    "k1024-mid-random": (1024, 1, 1, 32769, "random", True),
    "k1024-long-ties": (1024, 1, 1, 120003, "ties", True),
    # next_n > 1 (MTP): rows of differing length within one launch
    "nextn4-k512-short-rows8": (512, 8, 4, 4096, "random", True),
    "nextn4-k2048-mid-rows8": (2048, 8, 4, 32769, "random", True),
    "nextn2-k1024-long-rows4": (1024, 4, 2, 120003, "ties", True),
    # arbitrary (non-power-of-2) K -- the kernel is k-independent; tiny / odd /
    # >2048 K across short, long, and direct-fill (L <= k) tiers.
    "arbitrary-k1-long": (1, 1, 1, 120003, "random", True),
    "arbitrary-k7-short": (7, 1, 1, 8192, "random", True),
    "arbitrary-k777-long": (777, 1, 1, 120003, "random", True),
    "arbitrary-k3000-long": (3000, 1, 1, 120003, "random", True),
    "arbitrary-k3000-directfill": (3000, 1, 1, 2000, "random", True),
}


@pytest.mark.parametrize(
    ("k", "num_rows", "next_n", "seq_len", "dist", "padded"),
    list(_EQUIVALENCE_SCENARIOS.values()),
    ids=list(_EQUIVALENCE_SCENARIOS.keys()),
)
def test_set_equivalence_scenarios(k, num_rows, next_n, seq_len, dist, padded):
    _check_set_equivalence(k, num_rows, next_n, seq_len, dist, padded)


# --------------------------------------------------------------------------- #
# Ragged batches, edge shapes, and workspace reuse -- the tiered kernel's core
# job (rows of different length -> different tiers -> in one launch) plus edges
# the uniform-length matrix never reaches.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [512, 2048])
def test_ragged_batch_mixed_tiers(k):
    """One launch whose sequences span direct-fill / short / mid / long tiers."""
    seq_lens = [1024, 4096, 20001, 70000, 150003]
    width = max(_PAD_WIDTH, max(seq_lens))
    n = len(seq_lens)
    logits, sl, row_ends = _build_case(n, width, 1, seq_lens, "random", _POISON)
    indices = _run(logits, sl, n, 1, k)
    _assert_batch(logits, indices, row_ends, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_empty_row_in_batch(k):
    """seq_len 0 -> row_len 0 -> all -1, alongside a normal selection row."""
    logits, sl, row_ends = _build_case(
        2, max(_PAD_WIDTH, 4096), 1, [0, 4096], "random", _POISON
    )
    indices = _run(logits, sl, 2, 1, k)
    actual = indices.cpu()
    assert (actual[0] == -1).all(), "empty row (row_len 0) must be all -1"
    _assert_row_topk_set(logits.cpu()[1], actual[1], k, int(row_ends[1]))


@pytest.mark.parametrize("k", SUPPORTED_KS)
def test_all_identical_values(k):
    """Every value equal: whole row is one tie group, back-fill must pick k."""
    L = 8192
    logits, sl, row_ends = _build_case(1, max(_PAD_WIDTH, L), 1, L, "constant", _POISON)
    indices = _run(logits, sl, 1, 1, k)
    _assert_batch(logits, indices, row_ends, k)


@pytest.mark.parametrize("L", [8192, 120003], ids=["short", "long"])
@pytest.mark.parametrize("k", [256, 2048])
def test_non_finite_never_selected(k, L):
    """+inf / -inf / NaN in the valid region must never be selected"""
    device = torch.device("cuda")
    torch.manual_seed(_SEED)
    logits = torch.randn((1, L), dtype=torch.float32, device=device)
    # Scatter non-finite through the valid region. +inf/NaN are the ones a raw-bit
    # radix-select wrongly ranks at the very top; -inf should sort out naturally.
    bad = {
        10: float("inf"),
        100: float("nan"),
        L // 3: float("inf"),
        L // 2: float("-inf"),
        L - 5: float("nan"),
    }
    for pos, val in bad.items():
        logits[0, pos] = val
    sl = torch.tensor([L], dtype=torch.int32, device=device)

    indices = _run(logits, sl, 1, 1, k)
    got = indices[0].to(device).long()

    # (a) no selected index points to a non-finite value
    sel_vals = logits[0, got]
    assert torch.isfinite(sel_vals).all(), (
        f"selected non-finite values: "
        f"{sel_vals[~torch.isfinite(sel_vals)][:8].tolist()}"
    )
    # (b) the selected set is the top-k of the finite values (non-finite -> -inf)
    sanitized = torch.where(
        torch.isfinite(logits[0]),
        logits[0],
        torch.full_like(logits[0], float("-inf")),
    )
    _assert_row_topk_set(sanitized.cpu(), indices[0].cpu(), k, L)


@pytest.mark.parametrize("k", [512, 2048])
def test_row_strided_logits(k):
    """logits.stride(0) > width (row-slice of a wider buffer): exercises the
    row_base = row * stride0 addressing across multiple distinct rows."""
    L = 32769
    width = max(_PAD_WIDTH, L)
    logits, sl, row_ends = _build_case(
        4, width, 1, L, "random", _POISON, stride_pad=257
    )
    assert logits.stride(0) == width + 257  # sanity: rows are non-contiguous
    indices = _run(logits, sl, 4, 1, k)
    _assert_batch(logits, indices, row_ends, k)


def test_repeated_calls_reuse_workspace():
    """A single caller-provided workspace reused across back-to-back calls must
    stay correct -- guards the on-stream conditional-zero (a stale workspace only
    surfaces on the 2nd+ call). Long length so the multi-block zero is in play."""
    from aiter.ops.flydsl import flydsl_top_k_per_row_decode_workspace_size

    k, L = 2048, 120003
    logits, sl, row_ends = _build_case(1, max(_PAD_WIDTH, L), 1, L, "random", _POISON)
    size = flydsl_top_k_per_row_decode_workspace_size(1, logits.shape[1])
    ws = torch.empty(size, dtype=torch.int32, device="cuda")
    for _ in range(3):
        indices = torch.empty((1, k), device="cuda", dtype=torch.int32)
        flydsl_top_k_per_row_decode(
            logits,
            1,
            sl,
            indices,
            1,
            logits.stride(0),
            logits.stride(1),
            k=k,
            ordered=False,
            workspace=ws,
        )
        torch.cuda.synchronize()
        _assert_batch(logits, indices, row_ends, k)


def test_undersized_workspace_rejected():
    from aiter.ops.flydsl import flydsl_top_k_per_row_decode_workspace_size

    k, L = 2048, 120003
    logits, sl, _ = _build_case(1, max(_PAD_WIDTH, L), 1, L, "random", _POISON)
    size = flydsl_top_k_per_row_decode_workspace_size(1, logits.shape[1])
    ws = torch.empty(size - 1, dtype=torch.int32, device="cuda")
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        flydsl_top_k_per_row_decode(
            logits,
            1,
            sl,
            indices,
            1,
            logits.stride(0),
            logits.stride(1),
            k=k,
            ordered=False,
            workspace=ws,
        )


def test_num_rows_subset_leaves_extra_untouched():
    """numRows < physical rows: only the first numRows rows are processed; the
    trailing output rows must be left untouched."""
    k, L = 512, 8192
    n_phys, n_proc = 4, 2
    width = max(_PAD_WIDTH, L)
    logits, sl, row_ends = _build_case(n_phys, width, 1, L, "random", _POISON)
    indices = torch.full((n_phys, k), -7, device="cuda", dtype=torch.int32)
    flydsl_top_k_per_row_decode(
        logits,
        1,
        sl,
        indices,
        n_proc,
        logits.stride(0),
        logits.stride(1),
        k=k,
        ordered=False,
    )
    torch.cuda.synchronize()
    actual = indices.cpu()
    for row in range(n_proc):
        _assert_row_topk_set(logits.cpu()[row], actual[row], k, int(row_ends[row]))
    assert (actual[n_proc:] == -7).all(), "rows beyond numRows must be untouched"


# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("logits_dtype", "call_kwargs", "exc"),
    [
        (torch.float32, {"k": 512, "ordered": True}, ValueError),
        (torch.float32, {"k": 0}, ValueError),
        (torch.float16, {"k": 512}, TypeError),
    ],
    ids=["ordered-true", "nonpositive-k", "fp16-logits"],
)
def test_invalid_args_rejected(logits_dtype, call_kwargs, exc):
    k = call_kwargs["k"]
    seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    logits = torch.randn((1, 4096), device="cuda", dtype=logits_dtype)
    indices = torch.empty((1, k), device="cuda", dtype=torch.int32)
    with pytest.raises(exc):
        flydsl_top_k_per_row_decode(
            logits,
            1,
            seq_lens,
            indices,
            1,
            logits.stride(0),
            logits.stride(1),
            **call_kwargs,
        )
