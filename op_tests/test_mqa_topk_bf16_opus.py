# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for aiter.ops.opus.mqa_topk_bf16: an FP32 torch reference, the output contract and the
op on packed causal windows of one context-parallel rank.

    python3 op_tests/test_mqa_topk_bf16_opus.py [--shape small|glm|kimi31] [--cp 4]
"""

import argparse

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.mqa_topk_bf16 import (
    HEAD_DIM,
    HEADS,
    mqa_topk_bf16,
    mqa_topk_bf16_config,
)

PENDING = "gfx950 OPUS mqa_topk_bf16 kernels are pending"
# (document lengths, topk): one CP rank holds 1 / cp of the rows, keys are the whole pack.
SHAPES = {
    "small": ([256, 512, 64, 1024, 128], 64),
    "glm": ([32768, 16384, 8192, 8192, 4096], 2048),  # 69,632 keys, GLM-5.3 Lumen#37
    "kimi31": ([16384, 16384], 2048),  # Kimi-K3.1 Lumen#35
}


def make_windows(doc_lens, cp_size=1, cp_rank=0):
    """Rows of one CP rank over packed causal documents, in global key-row space.

    Each document is cut into 2 * cp_size chunks; the rank holds chunks cp_rank and
    2 * cp_size - 1 - cp_rank. Returns int32 (row_starts, row_ends) and the key count.
    """
    starts, ends, base = [], [], 0
    for n in doc_lens:
        assert n % (2 * cp_size) == 0, "pad documents to a multiple of 2 * cp"
        c = n // (2 * cp_size)
        for j in (cp_rank, 2 * cp_size - 1 - cp_rank):
            for p in range(base + j * c, base + (j + 1) * c):
                starts.append(base)
                ends.append(p + 1)
        base += n
    i32 = torch.int32
    return torch.tensor(starts, dtype=i32), torch.tensor(ends, dtype=i32), base


def reference(q, k, weights, row_starts, row_ends, topk, relu=True, chunk=256):
    """FP32 scores, then the top-W keys by (score desc, index asc), stored ascending."""
    rows, keys = q.shape[0], k.shape[0]
    width = min(topk, keys)
    kf = k.float().t()
    cols = torch.arange(keys, device=q.device)
    indices = torch.empty(rows, width, dtype=torch.int32, device=q.device)
    values = torch.empty(rows, width, dtype=torch.float32, device=q.device)
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        s = torch.zeros(r1 - r0, keys, device=q.device)
        for h in range(q.shape[1]):
            d = q[r0:r1, h].float() @ kf
            s += weights[r0:r1, h, None] * (d.relu() if relu else d)
        lo = row_starts[r0:r1, None].long().clamp(0, keys)
        hi = torch.maximum(row_ends[r0:r1, None].long().clamp(max=keys), lo)
        s = s.masked_fill((cols < lo) | (cols >= hi), float("-inf"))
        val, idx = s.sort(dim=1, descending=True, stable=True)
        keep = torch.arange(width, device=q.device) < (hi - lo).clamp(max=width)
        idx = torch.where(keep, idx[:, :width], keys)
        idx, order = idx.sort(dim=1)
        val = val[:, :width].gather(1, order)
        keep = idx < keys
        indices[r0:r1] = torch.where(keep, idx, -1)
        values[r0:r1] = torch.where(keep, val, float("-inf"))
    return indices, values


def check_contract(indices, values, row_starts, row_ends, keys, topk):
    """Valid prefix of min(n, W) ascending rows inside [lo, hi), then -1; -inf values there."""
    rows, width = row_starts.shape[0], min(topk, keys)
    assert indices.dtype == torch.int32 and tuple(indices.shape) == (rows, width)
    lo = row_starts.long().clamp(0, keys)[:, None]
    hi = torch.maximum(row_ends.long().clamp(max=keys)[:, None], lo)
    valid = torch.arange(width, device=indices.device) < (hi - lo).clamp(max=width)
    idx = indices.long()
    assert torch.equal(idx >= 0, valid), "valid entries first, then -1"
    assert ((idx >= lo) & (idx < hi))[valid].all(), "an index lies outside its window"
    ascending = idx[:, 1:] > idx[:, :-1]
    assert ascending[valid[:, 1:]].all(), "valid entries ascend, no duplicates"
    if values is not None:
        assert torch.isneginf(values[~valid]).all(), "-1 slots hold -inf"
        assert torch.isfinite(values[valid]).all(), "valid slots hold finite scores"


def make_inputs(doc_lens, cp_size, cp_rank, device, seed=0):
    row_starts, row_ends, keys = make_windows(doc_lens, cp_size, cp_rank)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = row_starts.shape[0]
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g).bfloat16()
    k = torch.randn(keys, HEAD_DIM, generator=g).bfloat16()
    w = torch.randn(rows, HEADS, generator=g) * (HEADS * HEAD_DIM) ** -0.5
    return [t.to(device) for t in (q, k, w, row_starts, row_ends)]


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 1), (4, 0), (4, 3)])
def test_reference_contract(cp_size, cp_rank):
    q, k, w, rs, re = make_inputs([64, 256, 32, 512], cp_size, cp_rank, "cpu")
    idx, val = reference(q, k, w, rs, re, topk=48)
    check_contract(idx, val, rs, re, k.shape[0], 48)


def test_reference_ties():
    # Equal keys give equal scores: the smallest indices of each window win.
    q, k, w, rs, re = make_inputs([256], 1, 0, "cpu")
    k = k[:1].expand_as(k).contiguous()
    idx, _ = reference(q, k, w, rs, re, topk=16)
    want = rs[:, None] + torch.arange(16, dtype=torch.int32)
    want = torch.where(want < re[:, None], want, -1)
    assert torch.equal(idx, want)


def run_op(shape="small", cp_size=4, cp_rank=0, seed=0):
    """The op against the reference; returns False while the kernels are pending."""
    doc_lens, topk = SHAPES[shape]
    q, k, w, rs, re = make_inputs(doc_lens, cp_size, cp_rank, "cuda", seed)
    cfg = mqa_topk_bf16_config(
        q.device, topk=topk, max_rows=q.shape[0], max_keys=k.shape[0]
    )
    try:
        idx, val = mqa_topk_bf16(q, k, w, rs, re, cfg=cfg, return_values=True)
    except NotImplementedError as exc:
        if str(exc) != PENDING:
            raise
        return False
    check_contract(idx, val, rs, re, k.shape[0], topk)
    ref_idx, _ = reference(q, k, w, rs, re, topk)
    hits = sum(
        torch.isin(a[a >= 0], b[b >= 0]).sum().item() for a, b in zip(idx, ref_idx)
    )
    recall = hits / max(int((ref_idx >= 0).sum()), 1)
    assert recall >= 0.9995, f"recall {recall:.6f} against the FP32 reference"
    return True


@pytest.mark.parametrize("shape", ["small", "kimi31"])
def test_mqa_topk_bf16(shape):
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        pytest.skip("requires gfx950")
    if not run_op(shape):
        pytest.skip(PENDING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="small")
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    test_reference_ties()
    for cp_size, cp_rank in ((1, 0), (4, 0), (4, 3)):
        test_reference_contract(cp_size, cp_rank)
    print("reference contract: ok")
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        print("mqa_topk_bf16: skipped (requires gfx950)")
        return
    ran = run_op(args.shape, args.cp, args.cp_rank, args.seed)
    print(f"mqa_topk_bf16 {args.shape}: {'ok' if ran else 'skipped, ' + PENDING}")


if __name__ == "__main__":
    main()
