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
  * **Context length independent of any compile-time bound.** `max_context` is
    a capacity hint that picks a body and a launch geometry; it never clamps the
    live length, which comes from `kv_indptr`. Every case therefore runs with a
    hint on each side of the crossover, and one case deliberately runs a context
    far longer than the hint.
"""

import pytest
import torch

from aiter.ops.triton.attention.paged_attention_output_gate import (
    _SHORT_CONTEXT_MAX,
    _short_body_selected,
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
# Qwen3-Next has 16 query heads, so a TP degree of 1/2/4/8/16 gives a rank
# 16/8/4/2/1 of them. All five must work: the support predicate admits
# 0 < heads <= 16, so anything it admits has to be covered here or the
# predicate is lying. 3 is included as a non-power-of-two the predicate also
# admits.
@pytest.mark.parametrize("heads", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("ctx_len", [1, 129, 1024])
# One hint on each side of the crossover, so both bodies see every shape.
@pytest.mark.parametrize("max_context", [_SHORT_CONTEXT_MAX, _SHORT_CONTEXT_MAX * 2])
def test_matches_reference(rows, heads, ctx_len, max_context):
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
        max_context=max_context,
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
@pytest.mark.parametrize("max_context", [None, _SHORT_CONTEXT_MAX * 2])
def test_quant_dtype_none_skips_epilogue(rows, max_context):
    """quant_dtype=None must return the bf16 output and nothing else.

    Not just a missing return value: the short body stages `sigmoid(gate)` in
    the `gated` buffer and the launcher has no FP8 buffers to point the epilogue
    at, so the FP8 stores have to be compiled out rather than aliased onto it.
    A live-but-aliased store would corrupt `gated` here.
    """
    dtype = torch.float8_e4m3fn
    q, kc, vc, indptr, indices, gate = _make_inputs(rows, 512, 4, dtype, seed=7)
    gated, quantized, scales = paged_attention_output_gate_group_fp8_quant(
        q, kc, vc, indptr, indices, gate, scale=HEAD_DIM**-0.5, max_context=max_context
    )
    assert quantized is None and scales is None
    ref_gated, _, _ = _reference(
        q, kc, vc, indptr, indices, gate, HEAD_DIM**-0.5, None, None, None
    )
    torch.testing.assert_close(gated.float(), ref_gated.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rows", [1, 3, 12])
def test_max_context_hint_never_clamps(rows):
    """A context far longer than the hint must still be attended in full.

    The Artemis short-context body this was ported from clamps the live length
    to a compile-time capacity, which silently truncates exactly this case. The
    hint here selects a body and a launch geometry and nothing else.
    """
    dtype = torch.float8_e4m3fn
    ctx_len = 5 * _SHORT_CONTEXT_MAX // 4
    q, kc, vc, indptr, indices, gate = _make_inputs(rows, ctx_len, 4, dtype, seed=17)
    assert _short_body_selected(4, kc, vc, 1024), "meant to exercise the short body"

    gated, _, _ = paged_attention_output_gate_group_fp8_quant(
        q, kc, vc, indptr, indices, gate, scale=HEAD_DIM**-0.5, max_context=1024
    )
    ref_gated, _, _ = _reference(
        q, kc, vc, indptr, indices, gate, HEAD_DIM**-0.5, None, None, None
    )
    torch.testing.assert_close(gated.float(), ref_gated.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rows", [1, 8, 32])
# Qwen3-Next has 16 query heads, so a TP degree of 1/2/4/8/16 gives a rank
# 16/8/4/2/1 of them. All five must work: the support predicate admits
# 0 < heads <= 16, so anything it admits has to be covered here or the
# predicate is lying. 3 is included as a non-power-of-two the predicate also
# admits.
@pytest.mark.parametrize("heads", [1, 2, 3, 4, 8, 16])
def test_bodies_agree(rows, heads):
    """The two bodies are a performance choice, so they must agree numerically.

    Not bit-exact: they reassociate the split-K sum differently. The tolerance
    is the same one each is held to against the fp32 reference.
    """
    dtype = torch.float8_e4m3fn
    q, kc, vc, indptr, indices, gate = _make_inputs(rows, 1024, heads, dtype, seed=23)
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    args = (q, kc, vc, indptr, indices, gate)
    kwargs = {
        "scale": HEAD_DIM**-0.5,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "quant_dtype": dtype,
    }
    if not _short_body_selected(heads, kc, vc, _SHORT_CONTEXT_MAX):
        # Head counts outside _SHORT_HEADS never reach the short body, so there
        # is no second body to compare against. That is a documented property,
        # asserted in test_body_selection_contract; skipping here keeps this
        # test about numerical agreement rather than re-testing selection.
        pytest.skip(f"heads={heads} has no short-body path")
    assert not _short_body_selected(heads, kc, vc, _SHORT_CONTEXT_MAX + 1)

    short_gated, short_q, short_s = paged_attention_output_gate_group_fp8_quant(
        *args, max_context=_SHORT_CONTEXT_MAX, **kwargs
    )
    long_gated, long_q, long_s = paged_attention_output_gate_group_fp8_quant(
        *args, max_context=_SHORT_CONTEXT_MAX + 1, **kwargs
    )
    torch.testing.assert_close(
        short_gated.float(), long_gated.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(short_s, long_s, rtol=2e-2, atol=1e-6)
    torch.testing.assert_close(
        short_q.float() * short_s.repeat_interleave(QUANT_GROUP, dim=1).float(),
        long_q.float() * long_s.repeat_interleave(QUANT_GROUP, dim=1).float(),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize("max_context", [None, _SHORT_CONTEXT_MAX * 2])
def test_gate_row_stride_is_honored(max_context):
    """A non-contiguous gate must be read through its stride, not assumed packed.

    The Artemis long-context original folds row and head into one index and
    silently requires `gate.stride(0) == heads*HEAD_DIM`; this asserts neither
    body does.
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
        q, kc, vc, indptr, indices, gate, scale=HEAD_DIM**-0.5, max_context=max_context
    )
    loose, _, _ = paged_attention_output_gate_group_fp8_quant(
        q,
        kc,
        vc,
        indptr,
        indices,
        strided,
        scale=HEAD_DIM**-0.5,
        max_context=max_context,
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

    # fnuz is the gfx942 e4m3 encoding. The same bytes denote different values
    # under the OCP encoding this gfx950 body reads, so accepting it would
    # dequantize silently wrong rather than fail -- reject both as cache and as
    # output dtype.
    ok, reason = paged_attention_output_gate_supported(
        q,
        kc.view(torch.float8_e4m3fnuz),
        vc.view(torch.float8_e4m3fnuz),
        gate,
        dtype,
    )
    assert not ok and "e4m3fn" in reason
    ok, reason = paged_attention_output_gate_supported(
        q, kc, vc, gate, torch.float8_e4m3fnuz
    )
    assert not ok and "quant_dtype" in reason

    # The long body adds the head-dim offset unscaled, so a non-unit query dim
    # stride would read the wrong elements. Build one by transposing a wider
    # buffer, so stride(2) != 1 while the shape stays valid.
    strided_q = torch.randn(
        4, HEAD_DIM, 4, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)
    assert strided_q.shape == q.shape and strided_q.stride(2) != 1
    ok, reason = paged_attention_output_gate_supported(strided_q, kc, vc, gate, dtype)
    assert not ok and "query" in reason and "stride" in reason


def test_csr_inputs_must_be_contiguous():
    """A strided CSR view must fail loudly.

    Both bodies walk these as flat pointer offsets, so a non-contiguous view
    would read the wrong row boundaries or page slots and return a plausible
    but wrong answer. That is worse than raising.
    """
    dtype = torch.float8_e4m3fn
    q, kc, vc, indptr, indices, gate = _make_inputs(4, 256, 4, dtype, seed=11)

    # Interleave, then take every other element: same values, stride 2.
    strided_indptr = torch.stack([indptr, indptr], dim=1).flatten()[::2]
    assert not strided_indptr.is_contiguous()
    assert torch.equal(strided_indptr, indptr)
    with pytest.raises(ValueError, match="contiguous"):
        paged_attention_output_gate_group_fp8_quant(
            q, kc, vc, strided_indptr, indices, gate,
            scale=1.0, quant_dtype=dtype,
        )

    strided_indices = torch.stack([indices, indices], dim=1).flatten()[::2]
    assert not strided_indices.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        paged_attention_output_gate_group_fp8_quant(
            q, kc, vc, indptr, strided_indices, gate,
            scale=1.0, quant_dtype=dtype,
        )


def test_body_selection_contract():
    """Pin *which* body runs, not just that the answer is right.

    Body choice is a performance contract with three independent conditions,
    and nothing else in this file would notice if any of them silently changed:

      * `max_context > 32768`     -> long body
      * `heads not in (4,8,16)`   -> long body REGARDLESS of context, because
        the short body's QK MFMA tiles 4x64x64 over the head axis
      * cache not 16-element aligned -> long body, whose gather has no such
        requirement

    The head-count one is a performance cliff worth stating out loud: a TP8
    deployment of Qwen3-Next has 2 heads per rank and therefore never gets the
    short body, even at short context where it is ~24% faster. That is
    correctness-neutral -- body choice never changes the answer, only the speed
    (see `test_matches_reference`, which covers heads 1..16) -- but it should
    not be able to change without someone noticing.
    """
    from aiter.ops.triton.attention import paged_attention_output_gate as mod

    dtype = torch.float8_e4m3fn
    _, kc, vc, _, _, _ = _make_inputs(4, 256, 4, dtype, seed=99)
    short_ctx = mod._SHORT_CONTEXT_MAX
    long_ctx = mod._SHORT_CONTEXT_MAX * 2

    for heads in mod._SHORT_HEADS:
        assert mod._short_body_selected(
            heads, kc, vc, short_ctx
        ), f"heads={heads} at max_context={short_ctx} should use the short body"
        assert not mod._short_body_selected(
            heads, kc, vc, long_ctx
        ), f"heads={heads} above the context threshold should use the long body"

    for heads in (1, 2, 3, 5, 6):
        assert heads not in mod._SHORT_HEADS
        assert not mod._short_body_selected(heads, kc, vc, short_ctx), (
            f"heads={heads} is not a short-body head count and must fall to the "
            "long body even at short context"
        )

    # A misaligned cache must fall to the long body rather than being rejected.
    # Slicing rows (kc[1:]) gives an offset of 256 elements, which is still
    # 16-aligned; the offset has to be built element-wise to be misaligned.
    pages = kc.shape[0]
    flat = torch.empty(pages * HEAD_DIM + 1, dtype=kc.dtype, device=kc.device)
    mis = flat[1:].view(pages, 1, HEAD_DIM)
    assert mis.storage_offset() % 16 != 0
    assert not mod._short_body_selected(4, mis, mis, short_ctx)
