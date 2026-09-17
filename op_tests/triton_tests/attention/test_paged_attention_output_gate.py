# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for paged_attention_output_gate_group_fp8_quant.

The reference is deliberately naive fp32 PyTorch rather than a second optimized
path, so a bug shared between implementations cannot pass.

Coverage is chosen around two things the op claims and a table-based dispatch
could not:

  * **Non-power-of-two batch.** A CUDA graph capture asks for decode batches of
    12, 24, 40, 48, 56 ... and dispatch happens once at capture, so a miss is
    permanent. These sizes are in the parametrization deliberately.
  * **Context length independent of any compile-time bound.** Short and long
    contexts run the same binary; the merge width is chosen device-side.
"""

import pytest
import torch

from aiter.ops.triton.attention.paged_attention_output_gate import (
    paged_attention_output_gate_group_fp8_quant,
    paged_attention_output_gate_supported,
)
from aiter.ops.triton.utils._triton import arch_info

HEAD_DIM = 256
QUANT_GROUP = 128

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx950",
    reason="paged_attention_output_gate has a gfx950 Gluon body only",
)


def _make_inputs(rows, ctx_len, heads, dtype, device="cuda", seed=0):
    """One decode step: `rows` sequences each holding `ctx_len` tokens.

    Page size is 1, so one KV page is one token. `kv_indices` is a permutation of
    the pool so the gather is scattered the way a real paged cache is.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    pages = rows * ctx_len
    q = torch.randn(
        rows, heads, HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device
    )
    # 0.25 keeps the fp8 cast away from saturation. A saturated fixture makes
    # every implementation agree trivially and hides real divergence.
    kc = (torch.randn(pages, 1, HEAD_DIM, generator=g, device=device) * 0.25).to(dtype)
    vc = (torch.randn(pages, 1, HEAD_DIM, generator=g, device=device) * 0.25).to(dtype)
    indptr = torch.arange(0, rows + 1, dtype=torch.int32, device=device) * ctx_len
    indices = torch.randperm(pages, generator=g, device=device).to(torch.int32)
    gate = torch.randn(
        rows, heads * HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device
    )
    return q, kc, vc, indptr.contiguous(), indices.contiguous(), gate


@torch.no_grad()
def _reference(q, kc, vc, indptr, indices, gate, scale, k_scale, v_scale, quant_dtype):
    """fp32 reference, reproducing the three BF16 rounding boundaries exactly."""
    rows, heads, _ = q.shape
    kf = kc.float().squeeze(1)
    vf = vc.float().squeeze(1)
    if k_scale is not None:
        kf = kf * k_scale.float().reshape(())
    if v_scale is not None:
        vf = vf * v_scale.float().reshape(())

    out = torch.empty(rows, heads, HEAD_DIM, dtype=torch.float32, device=q.device)
    ip = indptr.tolist()
    for r in range(rows):
        sel = indices[ip[r] : ip[r + 1]].long()
        logits = (q[r].float() @ kf[sel].T) * scale
        out[r] = torch.softmax(logits, dim=-1) @ vf[sel]

    # Contract: attention -> bf16 -> fp32, sigmoid -> bf16 -> fp32, product ->
    # bf16, and only then the group absmax. Keeping these in fp32 changes the
    # emitted FP8 codes.
    flat = out.reshape(rows, heads * HEAD_DIM).to(torch.bfloat16).float()
    sig = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
    gated = (flat * sig).to(torch.bfloat16)
    if quant_dtype is None:
        return gated, None, None

    fp8_max = torch.finfo(quant_dtype).max
    grouped = gated.float().reshape(rows, -1, QUANT_GROUP)
    qscale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(1e-10)
    quantized = (grouped / qscale).clamp(-fp8_max, fp8_max).to(quant_dtype)
    return (
        gated,
        quantized.reshape(rows, heads * HEAD_DIM),
        qscale.reshape(rows, heads * 2),
    )


@pytest.mark.parametrize(
    "rows",
    # powers of two, plus the non-power-of-two sizes a graph capture asks for
    [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 128],
)
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("ctx_len", [1, 129, 1024])
def test_matches_reference(rows, heads, ctx_len):
    dtype = torch.float8_e4m3fn
    q, kc, vc, indptr, indices, gate = _make_inputs(
        rows, ctx_len, heads, dtype, seed=rows * 31 + heads
    )
    scale = HEAD_DIM**-0.5
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    gated, quantized, scales = paged_attention_output_gate_group_fp8_quant(
        q,
        kc,
        vc,
        indptr,
        indices,
        gate,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_dtype=dtype,
    )
    ref_gated, ref_q, ref_s = _reference(
        q, kc, vc, indptr, indices, gate, scale, k_scale, v_scale, dtype
    )

    # bf16 output: rtol covers the split-K reassociation, which reorders the
    # softmax accumulation relative to the reference's single pass.
    torch.testing.assert_close(gated.float(), ref_gated.float(), rtol=2e-2, atol=2e-2)
    # Compare the fp8 dequantized, not the codes: a legitimate one-ulp rounding
    # difference at 2^-3 relative would fail a bitwise comparison.
    torch.testing.assert_close(
        quantized.float() * scales.repeat_interleave(QUANT_GROUP, dim=1).float(),
        ref_q.float() * ref_s.repeat_interleave(QUANT_GROUP, dim=1).float(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(scales, ref_s, rtol=2e-2, atol=1e-6)


@pytest.mark.parametrize("rows", [1, 16])
def test_quant_dtype_none_skips_epilogue(rows):
    """quant_dtype=None must return the bf16 output and nothing else."""
    dtype = torch.float8_e4m3fn
    q, kc, vc, indptr, indices, gate = _make_inputs(rows, 512, 4, dtype, seed=7)
    gated, quantized, scales = paged_attention_output_gate_group_fp8_quant(
        q, kc, vc, indptr, indices, gate, scale=HEAD_DIM**-0.5
    )
    assert quantized is None and scales is None
    ref_gated, _, _ = _reference(
        q, kc, vc, indptr, indices, gate, HEAD_DIM**-0.5, None, None, None
    )
    torch.testing.assert_close(gated.float(), ref_gated.float(), rtol=2e-2, atol=2e-2)


def test_gate_row_stride_is_honored():
    """A non-contiguous gate must be read through its stride, not assumed packed.

    The Artemis original folds row and head into one index and silently requires
    `gate.stride(0) == heads*HEAD_DIM`; this asserts we do not.
    """
    dtype = torch.float8_e4m3fn
    rows, heads = 8, 4
    q, kc, vc, indptr, indices, gate = _make_inputs(rows, 512, heads, dtype, seed=11)
    # A wider buffer whose rows are strided, sliced back to the right shape.
    padded = torch.randn(
        rows, heads * HEAD_DIM * 2, dtype=torch.bfloat16, device="cuda"
    )
    padded[:, : heads * HEAD_DIM] = gate
    strided = padded[:, : heads * HEAD_DIM]
    assert strided.stride(0) != heads * HEAD_DIM

    packed, _, _ = paged_attention_output_gate_group_fp8_quant(
        q, kc, vc, indptr, indices, gate, scale=HEAD_DIM**-0.5
    )
    loose, _, _ = paged_attention_output_gate_group_fp8_quant(
        q, kc, vc, indptr, indices, strided, scale=HEAD_DIM**-0.5
    )
    torch.testing.assert_close(loose.float(), packed.float(), rtol=0, atol=0)


def test_supported_predicate_rejects_and_explains():
    """The predicate must decline with a reason, never raise, so callers can
    fall back."""
    dtype = torch.float8_e4m3fn
    q, kc, vc, _, _, gate = _make_inputs(4, 256, 4, dtype, seed=3)

    ok, reason = paged_attention_output_gate_supported(q, kc, vc, gate, dtype)
    assert ok and reason == ""

    # bf16 KV cache is not supported by this body.
    ok, reason = paged_attention_output_gate_supported(
        q, kc.to(torch.bfloat16), vc.to(torch.bfloat16), gate, dtype
    )
    assert not ok and "fp8" in reason

    # More heads than HEAD_TILE.
    wide_q = torch.randn(4, 17, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    wide_gate = torch.randn(4, 17 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ok, reason = paged_attention_output_gate_supported(wide_q, kc, vc, wide_gate, dtype)
    assert not ok and "heads" in reason

    ok, reason = paged_attention_output_gate_supported(q, kc, vc, gate, torch.float16)
    assert not ok and "quant_dtype" in reason
