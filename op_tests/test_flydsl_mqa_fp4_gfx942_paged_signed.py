#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP4 MQA scorers: signed scores, in-place paged KV, row-packed prefill.

Each new path is checked bit-exactly against the existing dense path:
- ``relu=False`` against the head-weighted signed reference,
- an interleaved ``[data | scales]`` page allocation read through
  ``kv_page_stride`` / ``kv_scale_page_stride`` / ``kv_scale_offset`` against
  the dense ``kv_cache`` / ``kv_scale`` pair (decode, long-context, prefill),
- ``flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942`` against the 64-token
  ``amax`` of single-row ``relu=False`` prefill logits.

Usage:
    python op_tests/test_flydsl_mqa_fp4_gfx942_paged_signed.py
"""

import pytest
import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx942", reason="gfx942-only kernels")

DEV = "cuda"
PAGE = 64
HEAD_DIM = 128
DATA_BYTES = PAGE * HEAD_DIM // 2
SCALE_BYTES = PAGE * HEAD_DIM // 32
SLOT_BYTES = DATA_BYTES + SCALE_BYTES


def _fp4_tensors():
    from op_tests.test_flydsl_pa_mqa_logits_fp4_gfx942 import (
        create_paged_preshuffle_kv_fp4,
        fp4_quant_e2m1_with_e8m0,
    )

    return create_paged_preshuffle_kv_fp4, fp4_quant_e2m1_with_e8m0


def _paged_kv(batch, ctx, num_pages, seed):
    """Dense preshuffled KV, the same pages interleaved in one allocation, and
    a shuffled page table."""
    create_kv, _ = _fp4_tensors()
    g = torch.Generator(device=DEV).manual_seed(seed)
    t_max = (ctx + PAGE - 1) // PAGE * PAGE
    kv = torch.randn(batch, t_max, HEAD_DIM, device=DEV, generator=g)
    perm = torch.randperm(num_pages, device=DEV, generator=g)
    table = perm[: batch * t_max // PAGE].view(batch, -1).to(torch.int32)
    kv_cache, kv_scale, _, _ = create_kv(kv.bfloat16(), PAGE, num_pages, table)
    slots = torch.empty((num_pages, SLOT_BYTES), dtype=torch.uint8, device=DEV)
    slots[:, :DATA_BYTES] = kv_cache.view(num_pages, -1)
    slots[:, DATA_BYTES:] = kv_scale.view(num_pages, -1)
    paged = {
        "kv_page_stride": SLOT_BYTES,
        "kv_scale_page_stride": SLOT_BYTES,
        "kv_scale_offset": DATA_BYTES,
    }
    return kv_cache, kv_scale, slots, paged, table


def _query(rows, heads, seed, one_hot=False):
    """Packed FP4 Q in the scorer's preshuffled scale ABI, and head weights."""
    _, quant = _fp4_tensors()
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(rows, heads, HEAD_DIM, device=DEV, generator=g)
    q_fp4, q_e8m0 = quant(q.reshape(-1, HEAD_DIM))
    q_fp4 = q_fp4.view(rows, heads, HEAD_DIM // 2)
    scale = q_e8m0.view(rows, heads // 16, 16, 1, 4).permute(0, 3, 4, 2, 1)
    q_scale = torch.nn.functional.pad(scale.contiguous(), (0, 3)).contiguous()
    w = torch.randn(rows, heads, device=DEV, generator=g).bfloat16()
    if one_hot:
        w = torch.zeros(rows, heads, device=DEV, dtype=torch.bfloat16)
        w[:, 0] = 1
    return q_fp4, q_scale, w


def _same(a, b):
    return torch.equal(
        torch.nan_to_num(a, neginf=-3e38), torch.nan_to_num(b, neginf=-3e38)
    )


@pytest.mark.parametrize("batch,ctx,next_n", [(1, 3000, 1), (4, 8193, 1), (2, 6000, 4)])
def test_decode_signed_in_place(batch, ctx, next_n):
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_gfx942

    heads = 64
    kv_cache, kv_scale, slots, paged, table = _paged_kv(batch, ctx, 600, seed=1)
    q, qs, w = _query(batch * next_n, heads, seed=2)
    # ReLU acts per head before the weighted sum; positive weights expose it.
    w = w.abs()
    q = q.view(batch, next_n, heads, -1)
    qs = qs.view(batch, next_n, *qs.shape[1:])
    lens = torch.full((batch,), ctx, dtype=torch.int32, device=DEV)

    def run(kv, kvs, relu, **kw):
        return flydsl_pa_mqa_logits_fp4_gfx942(
            q,
            qs,
            kv,
            kvs,
            table,
            w,
            lens,
            ctx,
            weight_scale=0.5,
            next_n=next_n,
            relu=relu,
            **kw,
        )

    relu = run(kv_cache, kv_scale, True)
    signed = run(kv_cache, kv_scale, False)
    assert _same(run(slots, slots, False, **paged), signed)
    assert _same(run(slots, slots, True, **paged), relu)
    finite = torch.isfinite(signed)
    assert (signed[finite] < 0).any()
    assert (relu[finite] >= 0).all()


def test_longctx_signed_in_place():
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4_gfx942,
        flydsl_pa_mqa_logits_fp4_gfx942_longctx,
        recommended_gfx942_fp4_longctx_config,
    )

    batch, ctx, heads = 2, 70000, 16
    kv_cache, kv_scale, slots, paged, table = _paged_kv(batch, ctx, 2400, seed=3)
    q, qs, w = _query(batch, heads, seed=4, one_hot=True)
    q, qs = q.view(batch, 1, heads, -1), qs.view(batch, 1, *qs.shape[1:])
    lens = torch.tensor([ctx, ctx - 777], dtype=torch.int32, device=DEV)
    cfg = recommended_gfx942_fp4_longctx_config(ctx, batch_size=batch, next_n=1)

    ref = flydsl_pa_mqa_logits_fp4_gfx942(
        q, qs, kv_cache, kv_scale, table, w, lens, ctx, relu=False
    )
    got = flydsl_pa_mqa_logits_fp4_gfx942_longctx(
        q,
        qs,
        slots,
        slots,
        table,
        w,
        lens,
        ctx,
        relu=False,
        validate_q_scale=False,
        **paged,
        **cfg,
    )
    assert _same(got, ref)


def _prefill_rows(lens, q_lens):
    rows = sum(q_lens)
    row_to_batch = torch.repeat_interleave(
        torch.arange(len(lens), device=DEV, dtype=torch.int32),
        torch.tensor(q_lens, device=DEV),
    )
    ends = torch.cat(
        [torch.arange(n - q + 1, n + 1, device=DEV) for n, q in zip(lens, q_lens)]
    ).to(torch.int32)
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=DEV)
    return rows, row_to_batch, torch.zeros_like(ends), ends, cu.to(torch.int32)


@pytest.mark.parametrize(
    "lens,q_lens", [([37], [37]), ([5000, 129, 900], [300, 16, 47]), ([70000], [128])]
)
def test_rowpack_page_max_matches_single_row_signed(lens, q_lens):
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4_prefill_gfx942,
        flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942,
        rowpack_prefill_inputs_gfx942,
    )

    batch, max_len = len(lens), max(lens)
    kv_cache, kv_scale, slots, paged, table = _paged_kv(batch, max_len, 3000, seed=5)
    rows, row_to_batch, starts, ends, cu = _prefill_rows(lens, q_lens)
    q, qs, w = _query(rows, 16, seed=6, one_hot=True)
    row_scale = qs[:, 0, :, 0, 0].contiguous()

    ref = flydsl_pa_mqa_logits_fp4_prefill_gfx942(
        q,
        qs,
        kv_cache,
        kv_scale,
        table,
        w,
        row_to_batch,
        starts,
        ends,
        max_len,
        weight_scale=0.25,
        relu=False,
    )
    pages = (max_len + PAGE - 1) // PAGE
    ref = torch.nn.functional.pad(ref, (0, pages * PAGE - max_len), value=-torch.inf)
    ref = ref.view(rows, pages, PAGE).amax(-1)

    inputs = rowpack_prefill_inputs_gfx942(
        q[:, 0].contiguous(), row_scale, cu, row_to_batch, starts, ends
    )
    for kv, kvs, kw in ((kv_cache, kv_scale, {}), (slots, slots, paged)):
        out = torch.full((rows, pages), -torch.inf, device=DEV)
        flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
            *inputs[:2],
            kv,
            kvs,
            table,
            *inputs[2:],
            max_len,
            out,
            weight_scale=0.25,
            **kw,
        )
        assert _same(out, ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
