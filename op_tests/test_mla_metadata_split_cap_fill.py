# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The planner must fit in the buffers `get_mla_metadata_info_v1` sizes.

`test_mla_metadata_split_cap.py` checks the sizing arithmetic, but it can only
compare the formula against itself: it cannot say whether the bound is one the
planner actually reaches, or whether a tight cap sizes *too* tightly. That needs
running the planner and looking at how much it filled.

So this allocates exactly what the sizing returns, runs `get_mla_metadata_v1`
with the same `max_split_per_batch`, and checks the populated extents fit.

Shapes are chosen from a sweep, not by guesswork. Two traps:

  * a short decode, or any cap of 1, writes **zero** partials -- `0 <= bound`
    then passes for any bound at all and the row proves nothing. Every row here
    is asserted to be non-vacuous;
  * at large batch with uniform KV the planner does not split either. Batch 512
    needs jittered lengths before it fills.

Calling convention follows `test_metadata.py` so both drive the same planner.
"""

import random

import pytest
import torch

import aiter
from aiter import dtypes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU to run the planner"
)

KIMI_NHEAD_KV = 1  # MLA: a single latent KV head
PAGE_SIZE = 1
KV_GRANULARITY = max(PAGE_SIZE, 16)
NHEAD = 128
MAX_SEQLEN_QO = 4  # max_qo_tiles_per_batch = 4 on gfx950 fp8
UNI_SEQLEN_QO = 4
IS_CAUSAL = True

_BUFFERS = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _kv_lens(batch_size, ctx_len, jitter):
    if not jitter:
        return [ctx_len] * batch_size
    rng = random.Random(0xA17E)
    return [rng.randint(max(1, ctx_len // 2), ctx_len) for _ in range(batch_size)]


def _plan(
    batch_size, cap, ctx_len, jitter, slack=1, nhead=NHEAD, qlen=MAX_SEQLEN_QO, topk=-1
):
    """Allocate from the sizing, run the planner, return the buffers.

    `slack` multiplies every buffer so a tight allocation can be compared
    against a roomy one. `qlen` selects the planner: v1.2 takes its parallel
    path only at max_seqlen_qo == 1 (v1_2_device.cuh:757-760). `topk >= 0`
    switches both calls to the sparse path -- the C++ derives is_sparse from
    topk, so sizing and fill must agree on it or they describe different shapes.
    """
    is_sparse = topk >= 0
    sizes = aiter.get_mla_metadata_info_v1(
        batch_size,
        qlen,
        nhead,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=is_sparse,
        fast_mode=True,
        max_split_per_batch=cap,
    )
    outs = {}
    for name, (size, dtype) in zip(_BUFFERS, sizes):
        shape = (size,) if isinstance(size, int) else tuple(size)
        if slack != 1:
            shape = (shape[0] * slack,) + shape[1:]
        outs[name] = torch.zeros(shape, dtype=dtype, device="cuda")

    kv_lens = _kv_lens(batch_size, ctx_len, jitter)
    qo_indptr = torch.arange(
        0,
        (batch_size + 1) * qlen,
        qlen,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(kv_lens, dtype=torch.int32, device="cuda").cumsum(0)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // KIMI_NHEAD_KV,
        KIMI_NHEAD_KV,
        IS_CAUSAL,
        outs["work_meta_data"],
        outs["work_info_set"],
        outs["work_indptr"],
        outs["reduce_indptr"],
        outs["reduce_final_map"],
        outs["reduce_partial_map"],
        page_size=PAGE_SIZE,
        kv_granularity=KV_GRANULARITY,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=cap,
        topk=topk,
        # MUST match the dtypes the sizing call above used. Omitting these makes
        # the C++ planner default BOTH to bf16 (csrc/kernels/mla/metadata.cu:89),
        # and the native-support gate is dtype-dependent: on gfx942 nhead=128 is
        # native in fp8 but folds to 16 heads in bf16, so a sized-for-fp8 buffer
        # would be filled by a planner using an 8x larger effective batch.
        dtype_q_nope=dtypes.fp8,
        dtype_kv_nope=dtypes.fp8,
    )
    torch.cuda.synchronize()
    return outs


# Rows the planner actually splits on, from a sweep over batch x cap x ctx.
# Excluded deliberately: cap=1 at any batch (writes 0 partials -- the cap
# forbids the extra splits that produce them), and large batch with uniform KV
# (batch alone supplies the parallelism, so nothing splits).
ROWS = [
    pytest.param(1, 256, 65536, False, id="b1-cap256-tightest"),
    pytest.param(1, -1, 65536, False, id="b1-nocap"),
    pytest.param(8, 256, 65536, False, id="b8-cap256"),
    pytest.param(8, -1, 65536, False, id="b8-nocap"),
    pytest.param(512, 256, 65536, True, id="b512-cap256-jitter"),
    pytest.param(512, -1, 65536, True, id="b512-nocap-jitter"),
]


@pytest.mark.parametrize("batch_size,cap,ctx_len,jitter", ROWS)
def test_the_planner_fits_the_sized_buffers(batch_size, cap, ctx_len, jitter):
    outs = _plan(batch_size, cap, ctx_len, jitter)

    partials = int(outs["reduce_indptr"][-1])
    work = int(outs["work_indptr"][-1])

    assert partials <= outs["reduce_partial_map"].numel(), (
        f"batch={batch_size} cap={cap}: planner wrote {partials} partials into "
        f"a {outs['reduce_partial_map'].numel()}-entry reduce_partial_map"
    )
    assert work <= outs["work_info_set"].size(0), (
        f"batch={batch_size} cap={cap}: planner wrote {work} work entries into "
        f"a {outs['work_info_set'].size(0)}-row work_info_set"
    )

    indptr = outs["reduce_indptr"]
    assert int(indptr[0]) == 0
    assert bool(torch.all(indptr[1:] >= indptr[:-1])), "reduce_indptr not sorted"

    # Keep filled-vs-bound in the CI log, not only in a comment: the headroom
    # is the evidence for the sizing change and it is CU-count dependent.
    print(
        f"\n  batch={batch_size:>4} cap={cap:>4} jitter={int(jitter)}  "
        f"partials={partials:>5}/{outs['reduce_partial_map'].numel():<5} "
        f"work={work:>5}/{outs['work_info_set'].size(0):<5}"
    )


def test_the_tightest_allocation_does_not_overflow():
    """cap=1 is excluded from ROWS because it writes zero partials -- the cap
    forbids the extra splits that produce them. But 5 entries is the smallest
    allocation this change produces (against 1024 uncapped), so it is the one
    most likely to overflow if the capped bound were wrong. Vacuous for
    tightness; not vacuous for safety."""
    outs = _plan(1, 1, 65536, jitter=False)
    filled = int(outs["reduce_indptr"][-1])
    bound = outs["reduce_partial_map"].numel()
    assert filled <= bound, f"filled={filled} overflowed bound={bound}"
    assert int(outs["work_indptr"][-1]) <= outs["work_info_set"].size(0)


@pytest.mark.parametrize("batch_size,cap,ctx_len,jitter", ROWS)
def test_the_fit_check_is_not_vacuous(batch_size, cap, ctx_len, jitter):
    """`0 <= bound` holds for any bound, so a row that never fills proves
    nothing. Fail loudly rather than passing silently."""
    outs = _plan(batch_size, cap, ctx_len, jitter)
    assert int(outs["reduce_indptr"][-1]) > 0, (
        f"batch={batch_size} cap={cap} ctx={ctx_len} jitter={jitter}: planner "
        "wrote no partials, so this row asserts nothing"
    )


@pytest.mark.parametrize("batch_size,cap,ctx_len,jitter", ROWS)
def test_a_tight_buffer_matches_an_oversized_one(batch_size, cap, ctx_len, jitter):
    """An extent check is necessary but not sufficient: HIP does not reliably
    trap an out-of-bounds write, so a buffer could be overrun while the reported
    extent still looks fine. Plan twice -- once into exactly-sized buffers, once
    into 4x-sized ones -- and compare the populated prefix."""
    tight = _plan(batch_size, cap, ctx_len, jitter, slack=1)
    roomy = _plan(batch_size, cap, ctx_len, jitter, slack=4)

    n = int(tight["reduce_indptr"][-1])
    assert n == int(roomy["reduce_indptr"][-1]), "planner disagreed on extent"
    assert torch.equal(
        tight["reduce_partial_map"][:n], roomy["reduce_partial_map"][:n]
    ), f"batch={batch_size} cap={cap}: tight buffer differs from the roomy one"


def main():
    """aiter's CI runs each op_tests module with `python3`, not pytest."""
    if not torch.cuda.is_available():
        aiter.logger.warning("no GPU available; skipping metadata fill tests")
        return
    for row in ROWS:
        batch_size, cap, ctx_len, jitter = row.values
        test_the_planner_fits_the_sized_buffers(batch_size, cap, ctx_len, jitter)
        test_the_fit_check_is_not_vacuous(batch_size, cap, ctx_len, jitter)
        test_a_tight_buffer_matches_an_oversized_one(batch_size, cap, ctx_len, jitter)
    test_the_tightest_allocation_does_not_overflow()
    for nhead, cap in ((48, 16), (48, 256)):
        test_a_folded_head_count_still_fits(nhead, cap)
    for batch_size, cap, jitter in (
        (1, 256, False),
        (8, 256, False),
        (64, 256, False),
        (512, 256, True),
    ):
        test_the_parallel_planner_fits_at_qlen_one(batch_size, cap, jitter)
    for batch_size, cap, topk in ((1, 256, 2048), (8, 256, 2048)):
        test_a_sparse_shape_fits_the_capped_bound(batch_size, cap, topk)
    aiter.logger.info("mla metadata split-cap fill tests: all passed")


# gfx950 fp8 serves 32/64/128 heads natively but NOT 48, so the planner folds 48
# to 16 and triples its batch count before applying the cap
# (v1_2_device.cuh:910-928). Only cap=256 is used: cap 1 and 4 write ZERO
# partials at this shape, so `0 <= bound` would pass for any bound at all.
#
# cap=16 is the row that DISCRIMINATES the fold: the planner writes 48 partials
# there (exactly cap x qk_batch_ratio), against an unfolded bound of 28, so
# dropping the fold overflows. cap=256 does not discriminate -- per_tile_cap
# saturates at max_splits either way -- but it is kept as the non-overflow check
# at the largest capped shape.
@pytest.mark.parametrize("nhead,cap", [(48, 16), (48, 256)])
def test_a_folded_head_count_still_fits(nhead, cap):
    outs = _plan(1, cap, 65536, jitter=False, nhead=nhead)
    filled = int(outs["reduce_indptr"][-1])
    bound = outs["reduce_partial_map"].numel()
    assert filled <= bound, (
        f"nhead={nhead} cap={cap}: planner wrote {filled} partials into a "
        f"{bound}-entry reduce_partial_map -- the qk_batch_ratio fold was not "
        f"applied to the sizing"
    )
    print(f"\n  nhead={nhead:>4} cap={cap:>4}  partials={filled:>5}/{bound:<5}")


# v1.2 selects its parallel planner only at max_seqlen_qo == 1
# (v1_2_device.cuh:757-760), which every other row here misses because they all
# use qlen 4. It also packs the bound far tighter: batch 8 fills 262 of 263, a
# single entry of headroom, against 256/260 on the qlen=4 path. An off-by-one in
# the cap-aware bound shows up here and nowhere else.
@pytest.mark.parametrize(
    "batch_size,cap,jitter",
    [(1, 256, False), (8, 256, False), (64, 256, False), (512, 256, True)],
)
def test_the_parallel_planner_fits_at_qlen_one(batch_size, cap, jitter):
    outs = _plan(batch_size, cap, 65536, jitter=jitter, qlen=1)
    filled = int(outs["reduce_indptr"][-1])
    bound = outs["reduce_partial_map"].numel()
    assert filled > 0, f"batch={batch_size} qlen=1 wrote no partials -- vacuous row"
    assert filled <= bound, (
        f"batch={batch_size} cap={cap} qlen=1: planner wrote {filled} partials "
        f"into a {bound}-entry reduce_partial_map"
    )
    assert int(outs["work_indptr"][-1]) <= outs["work_info_set"].size(0)
    print(
        f"\n  qlen=1 batch={batch_size:>4} cap={cap:>4}  partials={filled:>5}/{bound:<5}"
    )


# The cap-aware branch is reachable with is_sparse=True as well, and sizing
# there uses the raw KV batch count rather than the sparse-expanded one
# (v1_2_device.cuh:860).
#
# NOTE this proves the sparse bound is not *under*-sized, but it cannot catch
# the reverse: sizing from the expanded batch over-reserves, so nothing
# overflows and this still passes. Pinning that would mean asserting an exact
# size, i.e. restating the formula the test is meant to check independently.
# The raw-count requirement rests on v1_2_device.cuh:860. Drives both calls from one topk so they describe the
# same shape. batch 64 and small caps are excluded: measured, they write zero
# partials, and `0 <= bound` proves nothing.
@pytest.mark.parametrize("batch_size,cap,topk", [(1, 256, 2048), (8, 256, 2048)])
def test_a_sparse_shape_fits_the_capped_bound(batch_size, cap, topk):
    outs = _plan(batch_size, cap, 65536, jitter=False, topk=topk)
    filled = int(outs["reduce_indptr"][-1])
    bound = outs["reduce_partial_map"].numel()
    assert filled > 0, f"sparse batch={batch_size} wrote no partials -- vacuous row"
    assert filled <= bound, (
        f"sparse batch={batch_size} cap={cap} topk={topk}: planner wrote "
        f"{filled} partials into a {bound}-entry reduce_partial_map -- the cap "
        f"must be derived from the raw KV batch count, not the expanded one"
    )
    assert int(outs["work_indptr"][-1]) <= outs["work_info_set"].size(0)
    print(
        f"\n  sparse batch={batch_size:>4} cap={cap:>4}  partials={filled:>5}/{bound:<5}"
    )


if __name__ == "__main__":
    main()
