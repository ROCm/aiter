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


def _plan(batch_size, cap, ctx_len, jitter, slack=1):
    """Allocate from the sizing, run the planner, return the buffers.

    `slack` multiplies every buffer so a tight allocation can be compared
    against a roomy one.
    """
    sizes = aiter.get_mla_metadata_info_v1(
        batch_size,
        MAX_SEQLEN_QO,
        NHEAD,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
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
        (batch_size + 1) * MAX_SEQLEN_QO,
        MAX_SEQLEN_QO,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(
        kv_lens, dtype=torch.int32, device="cuda"
    ).cumsum(0)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        NHEAD // KIMI_NHEAD_KV,
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
        max_seqlen_qo=MAX_SEQLEN_QO,
        uni_seqlen_qo=UNI_SEQLEN_QO,
        fast_mode=True,
        max_split_per_batch=cap,
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



