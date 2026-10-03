# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for non-uniform MLA query offsets after head folding.

gfx942 has no native bf16 persistent kernel for 96 heads, so metadata
rewrites each request as ``nhead / 16`` pseudo-batches. ``seqlens_qo_indptr``
still has one entry per original request. ``QoState::get_begin/get_end`` must
map the pseudo-batch back onto that table. Indexing it directly reads past
the buffer and the decode kernel faults.
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MLA metadata tests need a GPU"
)

_D = 576
_DV = 512


def _fold_ratio(nhead: int) -> int:
    """Host-side mirror of the bf16 branch in get_mla_metadata_v1_2_device."""
    gfx = get_gfx()
    if gfx == "gfx950":
        return 1
    if gfx == "gfx942":
        return 1 if nhead == 16 else nhead // 16
    pytest.skip(f"folded-qo metadata test is not defined for {gfx}")


def _expected_ranges(qo_lens, ratio: int):
    """(begin, end) of every pseudo-batch in the folded Q layout."""
    indptr = [0]
    for length in qo_lens:
        indptr.append(indptr[-1] + length)
    ranges = []
    for batch, length in enumerate(qo_lens):
        begin = indptr[batch]
        for subgroup in range(ratio):
            folded_begin = (begin - indptr[0]) * ratio + subgroup * length
            ranges.append((folded_begin, folded_begin + length))
    return ranges


def _alloc_metadata(batch_size: int, max_seqlen_qo: int, nhead: int):
    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    info = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtypes.bf16,
        dtypes.bf16,
        is_sparse=False,
        fast_mode=True,
    )
    return {
        name: torch.empty(size, dtype=dtype, device="cuda")
        for name, (size, dtype) in zip(names, info)
    }


def _works(buffers):
    n_works = int(buffers["work_indptr"][-1].item())
    work_info = buffers["work_info_set"].view(-1, 8)
    assert 0 < n_works <= work_info.shape[0]
    return work_info[:n_works].cpu()


def _build_metadata(qo_lens, kv_lens, nhead: int, uni_seqlen_qo: int):
    device = "cuda"
    qo_indptr = torch.tensor(
        [0, *torch.tensor(qo_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    kv_indptr = torch.zeros(len(kv_lens) + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.tensor(kv_lens, dtype=torch.int32, device=device).cumsum(0)
    last = torch.ones(len(kv_lens), dtype=torch.int32, device=device)
    buffers = _alloc_metadata(len(qo_lens), max(qo_lens), nhead)
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last,
        nhead,
        1,
        True,
        buffers["work_meta_data"],
        buffers["work_info_set"],
        buffers["work_indptr"],
        buffers["reduce_indptr"],
        buffers["reduce_final_map"],
        buffers["reduce_partial_map"],
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=max(qo_lens),
        uni_seqlen_qo=uni_seqlen_qo,
        fast_mode=True,
        dtype_q=dtypes.bf16,
        dtype_kv=dtypes.bf16,
    )
    torch.cuda.synchronize()
    return {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "last": last,
        "buffers": buffers,
        "works": _works(buffers),
    }


def _assert_query_ranges(works, qo_lens, ratio: int):
    expected = _expected_ranges(qo_lens, ratio)
    total_rows = sum(length * ratio for length in qo_lens)
    covered = {span: [] for span in expected if span[1] > span[0]}

    for row in works.tolist():
        qo_start, qo_end = row[2], row[3]
        assert 0 <= qo_start <= qo_end <= total_rows, (
            f"query range [{qo_start}, {qo_end}) is outside the {total_rows}-row "
            f"folded Q tensor"
        )
        if qo_start == qo_end:
            continue
        owners = [
            span
            for span in covered
            if span[0] <= qo_start and qo_end <= span[1]
        ]
        assert len(owners) == 1, (
            f"query range [{qo_start}, {qo_end}) does not sit inside exactly one "
            f"folded pseudo-batch: {owners}"
        )
        covered[owners[0]].append((qo_start, qo_end))

    for span, pieces in covered.items():
        assert pieces, f"pseudo-batch {span} produced no query work"
        merged = []
        for start, end in sorted(pieces):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        assert merged == [span], f"pseudo-batch {span} was covered by {merged}"


@pytest.mark.parametrize(
    "nhead,qo_lens",
    [
        (96, [1, 1, 1, 1, 1, 1, 1, 0]),
        (96, [2, 1, 3]),
        (16, [2, 1, 3]),
    ],
)
def test_nonuniform_query_ranges_stay_inside_folded_q(nhead, qo_lens):
    ratio = _fold_ratio(nhead)
    # KV length only affects how the query range is split across CUs.
    kv_lens = [32 + 16 * i for i in range(len(qo_lens))]
    if qo_lens[-1] == 0:
        kv_lens[-1] = 0
    meta = _build_metadata(qo_lens, kv_lens, nhead, uni_seqlen_qo=-1)
    _assert_query_ranges(meta["works"], qo_lens, ratio)


def _decode_reference(q, kv, kv_indptr, kv_indices, scale):
    outs = []
    for token in range(q.shape[0]):
        begin = int(kv_indptr[token].item())
        end = int(kv_indptr[token + 1].item())
        keys = kv[kv_indices[begin:end]].float()
        scores = torch.matmul(q[token].float(), keys.transpose(0, 1)) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs, keys[:, :_DV]))
    return torch.stack(outs)


def test_folded_nonuniform_decode_matches_reference():
    """The GSM8K tail: 7 real decode tokens plus one empty graph-padding row."""
    if get_gfx() != "gfx942":
        pytest.skip("the 96-head fold is the gfx942 bf16 path")

    nhead = 96
    qo_lens = [1, 1, 1, 1, 1, 1, 1, 0]
    kv_lens = [48, 64, 33, 80, 17, 96, 40, 0]
    ratio = _fold_ratio(nhead)
    device = "cuda"
    scale = _D**-0.5

    meta = _build_metadata(qo_lens, kv_lens, nhead, uni_seqlen_qo=-1)
    _assert_query_ranges(meta["works"], qo_lens, ratio)
    buffers = meta["buffers"]

    total_kv = int(sum(kv_lens))
    kv = torch.randn(total_kv + 8, _D, device=device, dtype=torch.bfloat16)
    kv_indices = torch.randperm(kv.shape[0], device=device)[:total_kv].to(torch.int32)
    ntokens = sum(qo_lens)
    q = torch.randn(ntokens, nhead, _D, device=device, dtype=torch.bfloat16)
    out = torch.empty(ntokens, nhead, _DV, device=device, dtype=torch.bfloat16)
    mla_decode_fwd(
        q,
        kv.view(-1, 1, 1, _D),
        out,
        meta["qo_indptr"],
        meta["kv_indptr"],
        kv_indices,
        meta["last"],
        1,
        page_size=1,
        sm_scale=scale,
        return_lse=False,
        work_meta_data=buffers["work_meta_data"],
        work_indptr=buffers["work_indptr"],
        work_info_set=buffers["work_info_set"],
        reduce_indptr=buffers["reduce_indptr"],
        reduce_final_map=buffers["reduce_final_map"],
        reduce_partial_map=buffers["reduce_partial_map"],
    )
    torch.cuda.synchronize()
    reference = _decode_reference(q, kv, meta["kv_indptr"], kv_indices, scale)
    torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)
