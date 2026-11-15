# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""An empty work item adds nothing to a persistent MLA decode.

Plan get_mla_metadata_v1 once for every request at a fixed KV bound, so each
request gets several KV splits, then shrink each request's work items to the
keys it really has, as a caller reusing one plan across layers does. A request
with fewer keys than splits keeps work items with no KV. The persistent kernel
writes no partial for those, so mla_reduce_v1 must give them no weight; the
partial buffers are filled with NaN first, so a stale slot cannot pass.
"""

import itertools
import sys

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import get_mla_decode_shape_support
from aiter.test_common import assertAllclose

KV_LORA_RANK = 512
DIM = KV_LORA_RANK + 64
SM_SCALE = 1.0 / DIM**0.5
BOUND = 2048
# Keys each request really has: none, fewer than its splits, and the bound.
KV_LENS = [0, 1, 3, 5, 17, 300, BOUND, 7]
MAX_SPLIT_PER_BATCH = 16
PLAN_ORDER = (
    "work_meta_data",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
INFO_ORDER = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="needs a GPU",
)


def _i32(values):
    return torch.tensor(list(values), dtype=torch.int32, device="cuda")


def _bound_plan(bs, n, qd):
    kw = {"fast_mode": True, "intra_batch_mode": False}
    info = aiter.get_mla_metadata_info_v1(
        bs,
        1,
        n,
        qd,
        qd,
        is_sparse=False,
        num_kv_splits=MAX_SPLIT_PER_BATCH,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        **kw,
    )
    meta = {
        name: torch.zeros(shape, dtype=t, device="cuda")
        for name, (shape, t) in zip(INFO_ORDER, info)
    }
    aiter.get_mla_metadata_v1(
        _i32(range(bs + 1)),
        _i32(range(0, (bs + 1) * BOUND, BOUND)),
        _i32([1] * bs),
        n,
        1,
        True,
        *(meta[k] for k in PLAN_ORDER),
        page_size=1,
        kv_granularity=32 if n == 64 and qd == dtypes.fp8 else 16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        dtype_q_nope=qd,
        dtype_kv_nope=qd,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        **kw,
    )
    return meta


def _shrink_to_owned(meta, starts, fold):
    """Scale each work item's KV range from [0, BOUND) to its request's keys;
    return the fewest splits a (pseudo-)request has."""
    n_work = int(meta["work_indptr"][-1].item())
    info = meta["work_info_set"][:n_work].cpu()
    for item in info:
        b = int(item[0]) // fold
        begin, length = starts[b], starts[b + 1] - starts[b]
        for col in (4, 5):
            item[col] = begin + (int(item[col]) - b * BOUND) * length // BOUND
        item[6] = begin + length - int(item[5])  # kv_offset: keys after the item
    meta["work_info_set"][:n_work].copy_(info)
    splits = info[:, 0].tolist()
    return min(splits.count(fb) for fb in range((len(starts) - 1) * fold))


@requires_gpu
@pytest.mark.parametrize("n", (16, 96, 128))
@pytest.mark.parametrize("d", ["fp8", "bf16"])
def test_empty_work_item_adds_nothing(d, n):
    qd = dtypes.fp8 if d == "fp8" else torch.bfloat16
    s = get_mla_decode_shape_support(n, 1, qd, qd, return_lse=True)
    if s.supported is not True:
        pytest.skip(f"decode {s.backend.value} supported={s.supported}")
    bs = len(KV_LENS)
    starts = [0, *itertools.accumulate(KV_LENS)]
    g = torch.Generator().manual_seed(518 + n)
    q = torch.randn(bs, n, DIM, generator=g).to("cuda").to(qd)
    kv = torch.randn(starts[-1], 1, 1, DIM, generator=g).to("cuda").to(qd)
    kv_indices = torch.randperm(starts[-1], generator=g).to(torch.int32).to("cuda")

    meta = _bound_plan(bs, n, qd)
    splits = _shrink_to_owned(meta, starts, s.fold_factor)
    assert splits > max(length for length in KV_LENS if length < 16), splits

    o = torch.empty(bs, n, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
    args = (q, kv, o, _i32(range(bs + 1)), _i32(starts), kv_indices, _i32([1] * bs), 1)
    kw = {}
    if qd == dtypes.fp8:
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    # Leave NaN in the blocks the decode's partial buffers are about to take.
    slots = meta["reduce_partial_map"].size(0)
    for shape in ((slots, 1, n, KV_LORA_RANK), (slots, 1, n, 1)):
        torch.full(shape, float("nan"), device="cuda")
    _, lse = aiter.mla.mla_decode_fwd(
        *args,
        page_size=1,
        nhead_kv=1,
        sm_scale=SM_SCALE,
        num_kv_splits=MAX_SPLIT_PER_BATCH,
        return_lse=True,
        empty_work_items=True,
        **meta,
        **kw,
    )
    o, lse = o.float().cpu(), lse.reshape(bs, n).float().cpu()

    kvf = kv[:, 0, 0].float().cpu()
    for b, length in enumerate(KV_LENS):
        tag = f"{d} N={n} request {b} with {length} keys:"
        if length == 0:
            # Every split has LSE -inf; the reduce maps that empty sum to +inf.
            assert bool(torch.isposinf(lse[b]).all()), f"{tag} LSE {lse[b].unique()}"
            assert bool((o[b] == 0).all()), f"{tag} O {o[b].unique()[:4]}"
            continue
        k = kvf[kv_indices[starts[b] : starts[b + 1]].long().cpu()]
        scores = (q[b].float().cpu() @ k.T) * SM_SCALE
        o_ref = scores.softmax(-1) @ k[:, :KV_LORA_RANK]
        assert bool(o[b].isfinite().all()), tag
        if qd == dtypes.fp8:
            x, y = o[b].double(), o_ref.double()
            cos_diff = 1 - 2 * (x * y).sum() / (x * x + y * y).sum()
            assert cos_diff < 3e-2, f"{tag} cos_diff {cos_diff}"
        else:
            assertAllclose(o[b], o_ref, msg=f"{tag} O")
        assertAllclose(lse[b], scores.logsumexp(-1), msg=f"{tag} LSE")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))
