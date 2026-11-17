# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 fp8_mqa_logits at 32 heads x 128 (the tuned table): row counts that take the
row-group kernel must match a reference, and every call mode must return the same bits.
"""

import pytest
import torch

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils._triton import arch_info

E4M3 = torch.float8_e4m3fn


def _case(rows, keys, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 32, 128, device="cuda", generator=g).to(E4M3)
    kv = torch.randn(keys, 128, device="cuda", generator=g).to(E4M3)
    scales = torch.rand(keys, device="cuda", generator=g) + 0.5
    weights = torch.randn(rows, 32, device="cuda", generator=g)
    starts = torch.randint(0, keys // 2 + 1, (rows,), device="cuda", generator=g)
    span = torch.randint(0, keys + 1, (rows,), device="cuda", generator=g)
    ends = torch.minimum(starts + span, torch.full_like(starts, keys))
    return q, kv, scales, weights, starts.int(), ends.int()


def _reference(q, kv, scales, weights, starts, ends):
    s = torch.einsum("rhd,kd->rhk", q.float(), kv.float()).relu()
    out = (s * weights[:, :, None]).sum(1) * scales[None, :]
    cols = torch.arange(kv.shape[0], device="cuda")
    inside = (cols[None, :] >= starts[:, None]) & (cols[None, :] < ends[:, None])
    return out, inside


@pytest.mark.skipif(
    not arch_info.get_arch().startswith("gfx1250"), reason="gfx1250 tuned path"
)
@pytest.mark.parametrize("rows,keys", [(2048, 1025), (2051, 3001), (4097, 777)])
@torch.inference_mode()
def test_rowgroup_modes_bit_identical(rows, keys):
    q, kv, scales, weights, starts, ends = _case(rows, keys, rows + keys)
    ref, inside = _reference(q, kv, scales, weights, starts, ends)

    dense = fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits=False)
    err = (dense[inside] - ref[inside]).abs().max() / ref[inside].abs().max()
    assert err < 1e-3, f"max-abs relative error {err:.2e}"

    clean = fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits=True)
    assert torch.equal(clean[inside], dense[inside])
    assert torch.isneginf(clean[~inside]).all()

    out = torch.full((rows, keys), float("nan"), device="cuda")
    fp8_mqa_logits(q, kv, scales, weights, starts, ends, clean_logits=False, out=out)
    assert torch.equal(out[inside], dense[inside])

    lens = (ends - starts).clamp(min=0).long()
    offsets = torch.zeros_like(lens)
    offsets[1:] = lens.cumsum(0)[:-1]
    packed = torch.full((int(lens.sum()) + 8,), float("nan"), device="cuda")
    fp8_mqa_logits(
        q, kv, scales, weights, starts, ends, out=packed[4:-4], out_offsets=offsets
    )
    row_ids = torch.repeat_interleave(torch.arange(rows, device="cuda"), lens)
    first = lens.cumsum(0) - lens
    columns = torch.arange(int(lens.sum()), device="cuda") - first[row_ids]
    assert torch.equal(
        packed[4:-4][offsets[row_ids] + columns],
        dense[row_ids, starts[row_ids] + columns],
    )
    assert torch.isnan(packed[:4]).all() and torch.isnan(packed[-4:]).all()
