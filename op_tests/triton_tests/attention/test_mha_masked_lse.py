# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LSE of fully masked query rows.

With causal attention and seqlen_q > seqlen_k the mask is aligned to the bottom right, so the
first (seqlen_q - seqlen_k) query rows attend to no keys and their LSE is log(0) = -inf. A
finite value gives those rows a non-zero weight in a downstream chunked-prefill merge while
their output is all zeros, which pulls the merged result toward zero.

test_mha.py compares LSE only where the reference is finite, so these rows were never covered.
"""

import pytest
import torch

from aiter.ops.triton.attention.mha import flash_attn_func


def _merge_attn_states(out_a, lse_a, out_b, lse_b):
    """Reference chunked-prefill merge, in plain torch.

    Written out here rather than imported because the consumer of this LSE lives outside AITER;
    the property under test is a property of the value this kernel stores.
    """
    max_lse = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - max_lse).unsqueeze(-1)
    wb = torch.exp(lse_b - max_lse).unsqueeze(-1)
    return (wa * out_a + wb * out_b) / (wa + wb)


def _fwd(seqlen_q, seqlen_k, sink=None, batch=2, nheads=4, head_sz=128):
    torch.manual_seed(20)
    shape_q = (batch, seqlen_q, nheads, head_sz)
    shape_k = (batch, seqlen_k, nheads, head_sz)
    q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape_k, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape_k, device="cuda", dtype=torch.bfloat16)
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True, sink=sink)
    # lse is (batch, nheads, seqlen_q); line it up with out, which is (batch, seqlen_q, nheads).
    return out, lse.transpose(1, 2)


# 512/128 masks whole BLOCK_M blocks, which take the kernel's early-return path. 300/128 puts
# causal_start_idx at 172, inside a block, which takes the post-loop path instead. Both sites
# wrote 0.0, so both are covered.
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k", [(512, 128), (300, 128), (256, 64), (129, 1)]
)
def test_fully_masked_rows_have_neg_inf_lse(seqlen_q, seqlen_k):
    _, lse = _fwd(seqlen_q, seqlen_k)
    causal_start = seqlen_q - seqlen_k

    masked = lse[:, :causal_start, :]
    attended = lse[:, causal_start:, :]

    assert torch.isneginf(masked).all(), (
        f"rows before causal_start_idx={causal_start} attend to no keys, so their LSE must be "
        f"-inf; got min={masked.min().item()} max={masked.max().item()}"
    )
    assert torch.isfinite(attended).all(), (
        "rows at or after causal_start_idx attend to at least one key, so their LSE must be "
        "finite"
    )


def test_masked_block_is_identity_under_merge():
    """The reason the value matters: merging a fully masked chunk must be a no-op."""
    seqlen_q, seqlen_k = 512, 128
    out, lse = _fwd(seqlen_q, seqlen_k)
    causal_start = seqlen_q - seqlen_k

    masked_out = out[:, :causal_start, :, :]
    masked_lse = lse[:, :causal_start, :]
    assert torch.count_nonzero(masked_out) == 0, "a fully masked row must output zeros"

    # Stand-in for the partial result this chunk would be merged with.
    torch.manual_seed(21)
    other_out = torch.randn_like(masked_out)
    other_lse = (
        torch.randn_like(masked_lse).abs() + 0.5
    )  # finite and well away from zero

    merged = _merge_attn_states(masked_out, masked_lse, other_out, other_lse)
    assert torch.equal(
        merged, other_out
    ), "merging a fully masked chunk must reproduce the other chunk exactly"

    # Guard the regression directly: with the old 0.0 the merge shrinks the result, so if this
    # ever stops holding the test above has stopped testing anything.
    shrunk = _merge_attn_states(
        masked_out, torch.zeros_like(masked_lse), other_out, other_lse
    )
    assert not torch.equal(
        shrunk, other_out
    ), "0.0 LSE is expected to corrupt the merge; if it no longer does, this test is vacuous"


# Both shapes are needed for the same reason as above: 512/128 only reaches the early-return
# guard, so the post-loop one would still pass with its ENABLE_SINK check removed.
@pytest.mark.parametrize("seqlen_q, seqlen_k", [(512, 128), (300, 128)])
def test_sink_path_is_unchanged(seqlen_q, seqlen_k):
    """Regression guard, not a correctness claim.

    With a sink, m_i starts at the sink logit rather than -inf, so a row that attends to no keys
    still has a genuine finite LSE and the -inf fix does not apply. This pins the sink path so
    the fix cannot leak into it. Whether the value the sink path stores is itself the right one
    is a separate question and deliberately not settled here.
    """
    batch, nheads = 2, 4
    sink = torch.randn((nheads,), device="cuda", dtype=torch.float32)
    _, lse = _fwd(seqlen_q, seqlen_k, sink=sink, batch=batch, nheads=nheads)

    assert torch.isfinite(lse).all(), "the sink path must not produce -inf"
