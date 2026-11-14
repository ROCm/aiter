# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.step5_paged_mqa_logits import step5_paged_mqa_logits
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx942", reason="launch config published for gfx942 only"
)

HEAD_DIM = 256
FP8_NEG_ZERO = 0x80


def _fp8_bytes(shape):
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    raw = x.to(torch.float8_e4m3fn).view(torch.uint8)
    # -0.0 in e4m3fn has the NaN encoding of e4m3fnuz.
    neg_zero = torch.rand(shape, dtype=torch.float32, device="cuda") < 0.01
    return raw.masked_fill(neg_zero, FP8_NEG_ZERO)


def make_inputs(batch, next_n, heads, page_size, max_seq_len, layout, seed=0):
    """Make random inputs in the packed page layout of the wrapper.

    ``layout`` sets the page mapping and the lengths:

    - ``"shared"``: all requests use the same pages, with causal lengths, as
      the rows of one prefill chunk do.
    - ``"mixed"``: two owners, so that one row block maps to two page sets.
    - ``"distinct"``: each request has its own pages and a random length.
    """
    torch.manual_seed(seed)
    pages_per_req = -(-max_seq_len // page_size)
    if layout == "shared":
        owners = torch.zeros(batch, dtype=torch.int64, device="cuda")
    elif layout == "mixed":
        owners = torch.arange(batch, device="cuda") * 2 // max(batch, 1)
    else:
        owners = torch.arange(batch, device="cuda")
    num_owners = {"shared": 1, "mixed": 2}.get(layout, batch)
    num_pages = num_owners * pages_per_req
    owner_pages = torch.randperm(num_pages, device="cuda").view(
        num_owners, pages_per_req
    )
    block_tables = owner_pages[owners].to(torch.int32)

    if layout == "distinct":
        seq_lens = torch.randint(
            0, max_seq_len + 1, (batch,), dtype=torch.int32, device="cuda"
        )
    else:
        seq_lens = torch.linspace(
            1, max_seq_len, batch, dtype=torch.float32, device="cuda"
        ).to(torch.int32)

    keys = _fp8_bytes((num_pages, page_size * HEAD_DIM))
    scales = torch.rand((num_pages, page_size), dtype=torch.float32, device="cuda")
    kv_fused = torch.cat([keys, scales.view(torch.uint8)], dim=1).view(
        num_pages, page_size, 1, HEAD_DIM + 4
    )
    q = _fp8_bytes((batch, next_n, heads, HEAD_DIM)).view(torch.float8_e4m3fn)
    weights = torch.randn((batch * next_n, heads), dtype=torch.float32, device="cuda")
    return q, kv_fused, weights, block_tables, seq_lens


def ref_step5_paged_mqa_logits(
    q, kv_fused, weights, block_tables, seq_lens, max_seq_len
):
    batch, next_n, heads, _ = q.shape
    num_pages, page_size = kv_fused.shape[:2]
    packed = kv_fused.view(num_pages, -1)
    keys = packed[:, : page_size * HEAD_DIM].view(torch.float8_e4m3fn).double()
    keys = keys.view(num_pages, page_size, HEAD_DIM)
    scales = packed[:, page_size * HEAD_DIM :].view(torch.float32).double()

    k = torch.arange(max_seq_len, device=q.device)
    page = block_tables.long()[:, k // page_size]
    key = keys[page, k % page_size]
    scale = scales[page, k % page_size]

    dots = torch.einsum("bnhd,bkd->bnhk", q.double(), key).relu()
    w = weights.double().reshape(batch, next_n, heads, 1)
    logits = (dots * w).sum(dim=2) * scale[:, None, :]
    visible = k[None, None, :] < seq_lens[:, None, None]
    logits = logits.masked_fill(~visible, float("-inf"))
    return logits.reshape(batch * next_n, max_seq_len).float()


def check_logits(logits, ref):
    assert logits.shape == ref.shape
    torch.testing.assert_close(logits, ref, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("page_size", [16, 32, 48, 256])
@pytest.mark.parametrize(
    "batch, next_n, max_seq_len, layout",
    [
        (64, 1, 1500, "shared"),
        (37, 1, 300, "mixed"),
        (19, 1, 777, "distinct"),
        (9, 2, 1100, "distinct"),
    ],
)
@torch.inference_mode()
def test_step5_paged_mqa_logits(heads, page_size, batch, next_n, max_seq_len, layout):
    inputs = make_inputs(batch, next_n, heads, page_size, max_seq_len, layout)

    logits = step5_paged_mqa_logits(*inputs, max_seq_len)

    check_logits(logits, ref_step5_paged_mqa_logits(*inputs, max_seq_len))


@torch.inference_mode()
def test_step5_paged_mqa_logits_strided():
    batch, groups, heads, max_seq_len = 40, 2, 8, 1000
    q, kv_fused, weights, block_tables, seq_lens = make_inputs(
        batch, groups, heads, 32, max_seq_len, "shared"
    )
    # One query group of a [tokens, groups, heads, dim] tensor, as the model
    # passes it.
    q = q[:, 1:]
    weights = weights.view(batch, groups, heads)[:, 1]
    inputs = (q, kv_fused, weights, block_tables, seq_lens)

    logits = step5_paged_mqa_logits(*inputs, max_seq_len)

    check_logits(logits, ref_step5_paged_mqa_logits(*inputs, max_seq_len))


@torch.inference_mode()
def test_step5_paged_mqa_logits_e4m3fn_path(monkeypatch):
    inputs = make_inputs(21, 1, 16, 32, 900, "mixed")
    ref = ref_step5_paged_mqa_logits(*inputs, 900)
    check_logits(step5_paged_mqa_logits(*inputs, 900), ref)

    # Run the e4m3fn path of the other archs. Triton emulates it on gfx942.
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx950")
    check_logits(step5_paged_mqa_logits(*inputs, 900), ref)


@torch.inference_mode()
def test_step5_paged_mqa_logits_empty():
    inputs = make_inputs(0, 1, 16, 32, 64, "distinct")

    logits = step5_paged_mqa_logits(*inputs, 64)

    assert logits.shape == (0, 64)
