# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused Qwen3-Next GDN decode kernel.

Fuses, in one launch: packed qkvz/ba split, depthwise causal conv1d (with bias)
+ SiLU, delta-rule gating, the fp32 recurrent state update, gated RMSNorm with a
SiLU gate, and an optional per-head group FP8 quantization epilogue.

This is a *sibling* of ``fused_kda_decode``, not a variant of it. KDA uses a
per-channel lower-bounded sigmoid gate, a sigmoid output gate, uniform head
counts and no conv bias. Qwen3-Next GDN uses a per-head
``exp(-exp(A_log) * softplus(a + dt_bias))`` decay, a SiLU output gate,
``num_v_heads == 2 * num_k_heads`` and a biased conv.
"""

import pytest
import torch

device = "cuda"

# Matching tolerances: the kernel rounds the recurrence output to bf16 before
# the norm, and quantizes the bf16-rounded normalized value, so the reference
# must do the same or the comparison is meaningless.
ATOL = 0.05
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

# One batch per dispatch branch in the wrapper, rather than a scaled ladder --
# sizes within a branch are duplicates of each other:
#   1    -> _fused_decode,       grid (batch, k_heads)
#   32   -> _fused_decode_tiled, B32=True, flattened 1-D grid
#   64   -> _fused_decode,       transposed grid (k_heads, batch)
#   127  -> _fused_decode_tiled, B32=False, 2-D grid (also INDEX64=True)
#   128  -> _decode_group,       4 warps
#   256  -> _decode_group,       8 warps
_BATCHES = [1, 32, 64, 127, 128, 256]


def split_qkvz(projected_qkvz, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    """Unpack the interleaved ``in_proj_qkvz`` output.

    Per k-head group the layout is ``[q(KD) | k(KD) | v(ratio*VD) | z(ratio*VD)]``.
    """
    T = projected_qkvz.shape[0]
    ratio = num_v_heads // num_k_heads
    group_width = 2 * head_k_dim + 2 * ratio * head_v_dim
    grouped = projected_qkvz.view(T, num_k_heads, group_width)

    q = grouped[:, :, :head_k_dim]
    k = grouped[:, :, head_k_dim : 2 * head_k_dim]
    vz = grouped[:, :, 2 * head_k_dim :].reshape(T, num_k_heads, 2, ratio, head_v_dim)
    v = vz[:, :, 0].reshape(T, num_v_heads, head_v_dim)
    z = vz[:, :, 1].reshape(T, num_v_heads, head_v_dim)
    return q, k, v, z


def pack_conv_channels(q, k, v):
    """Interleave q/k/v into the ``[all q | all k | all v]`` conv channel order."""
    T = q.shape[0]
    return torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], dim=-1)


def ref_gdn_decode(
    projected_qkvz,
    projected_ba,
    conv_state,
    ssm_state,
    ssm_state_indices,
    conv_weight,
    conv_bias,
    A_log,
    dt_bias,
    norm_weight,
    scale,
    norm_eps,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    quant_dtype=None,
    quant_group_size=128,
    pad_slot_id=-1,
):
    """Pure-torch fp32 reference. Mutates ``conv_state`` and ``ssm_state``."""
    T = projected_qkvz.shape[0]
    ratio = num_v_heads // num_k_heads
    width = conv_weight.shape[-1]

    q, k, v, z = split_qkvz(
        projected_qkvz.float(), num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    x = pack_conv_channels(q, k, v)  # [T, channels]
    channels = x.shape[-1]

    out = torch.zeros(T, num_v_heads, head_v_dim, dtype=torch.float32, device=device)

    for t in range(T):
        slot = int(ssm_state_indices[t].item())
        if slot == pad_slot_id:
            # Padded CUDA-graph row: no state is read and none is written.
            continue

        # --- depthwise causal conv1d over the rolling window, + bias, + SiLU ---
        history = conv_state[slot].float()  # [channels, width-1]
        acc = (
            conv_bias.float().clone()
            if conv_bias is not None
            else torch.zeros(channels, dtype=torch.float32, device=device)
        )
        for tap in range(width - 1):
            acc += history[:, tap] * conv_weight[:, tap].float()
        acc += x[t] * conv_weight[:, width - 1].float()
        conv_out = acc * torch.sigmoid(acc)

        # Roll the window: drop the oldest, append the raw (pre-activation) input.
        shifted = torch.empty_like(history)
        shifted[:, : width - 2] = history[:, 1:]
        shifted[:, width - 2] = x[t]
        conv_state[slot] = shifted.to(conv_state.dtype)

        qk_split = 2 * num_k_heads * head_k_dim
        q_t = conv_out[: qk_split // 2].view(num_k_heads, head_k_dim)
        k_t = conv_out[qk_split // 2 : qk_split].view(num_k_heads, head_k_dim)
        v_t = conv_out[qk_split:].view(num_v_heads, head_v_dim)

        # --- QK L2 norm. Epsilon is inside the rsqrt, on the sum of squares. ---
        q_t = q_t * torch.rsqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6) * scale
        k_t = k_t * torch.rsqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)

        for h in range(num_v_heads):
            g = h // ratio
            # --- GDN gate ---
            ba_base = 2 * ratio * g
            b_val = projected_ba[t, ba_base + (h % ratio)].float()
            a_val = projected_ba[t, ba_base + ratio + (h % ratio)].float()
            sp_arg = a_val + dt_bias[h].float()
            softplus = torch.where(
                sp_arg <= 20.0, torch.log1p(torch.exp(sp_arg)), sp_arg
            )
            decay = torch.exp(-torch.exp(A_log[h].float()) * softplus)
            # The kernel rounds beta through bf16 before using it.
            beta = torch.sigmoid(b_val).to(torch.bfloat16).float()

            # --- delta rule on the fp32 state [VD, KD] ---
            state = ssm_state[slot, h].float()
            kv = k_t[g]
            decayed = state * decay
            predicted = (decayed * kv[None, :]).sum(-1)
            residual = (v_t[h] - predicted) * beta
            updated = decayed + residual[:, None] * kv[None, :]
            ssm_state[slot, h] = updated.to(ssm_state.dtype)
            out[t, h] = (updated * q_t[g][None, :]).sum(-1)

    # --- gated RMSNorm. The kernel rounds the core output to bf16 first. ---
    core = out.to(torch.bfloat16).float()
    rms = torch.rsqrt((core * core).sum(-1, keepdim=True) / head_v_dim + norm_eps)
    gate = z.float()
    normalized = (
        core * rms * norm_weight.float()[None, None, :] * gate * torch.sigmoid(gate)
    ).to(torch.bfloat16)

    if quant_dtype is None:
        return normalized, None, None

    # --- group FP8 quantization, consuming the bf16-rounded value ---
    assert quant_group_size == head_v_dim, "one scale per (token, v_head)"
    values = normalized.float()
    absmax = values.abs().amax(-1).clamp_min(1e-10)
    scales = absmax / FP8_MAX
    quantized = (values / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(quant_dtype)
    return normalized, quantized.view(-1, num_v_heads * head_v_dim), scales


def make_inputs(
    batch, num_k_heads=4, head_dim=128, num_slots=None, width=4, seed=0, ratio=2
):
    """Build a self-consistent input set for the Qwen3-Next GDN decode contract.

    Every live row gets a *distinct* slot. Duplicate slots would let two rows
    read-modify-write the same state concurrently, which is order-dependent in
    the kernel and sequential in the reference — a mismatch that says nothing
    about correctness. SGLang's own validator enforces distinct live slots.
    """
    torch.manual_seed(seed)
    # Slot 0 is reserved, so N must exceed the batch to give everyone a slot.
    num_slots = num_slots if num_slots is not None else batch + 1
    num_v_heads = ratio * num_k_heads
    group_width = 2 * head_dim + 2 * ratio * head_dim
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim

    return {
        "projected_qkvz": torch.randn(
            batch, num_k_heads * group_width, dtype=torch.bfloat16, device=device
        ),
        "projected_ba": torch.randn(
            batch, 2 * num_v_heads, dtype=torch.bfloat16, device=device
        ),
        "conv_state": torch.randn(
            num_slots, channels, width - 1, dtype=torch.bfloat16, device=device
        ),
        "ssm_state": torch.randn(
            num_slots,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        ),
        "ssm_state_indices": torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        ),
        "conv_weight": torch.randn(
            channels, width, dtype=torch.bfloat16, device=device
        ),
        "conv_bias": torch.randn(channels, dtype=torch.bfloat16, device=device),
        "A_log": torch.randn(num_v_heads, dtype=torch.float32, device=device),
        "dt_bias": torch.randn(num_v_heads, dtype=torch.bfloat16, device=device),
        "norm_weight": torch.randn(head_dim, dtype=torch.bfloat16, device=device),
        "scale": head_dim**-0.5,
        "norm_eps": 1e-6,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_dim,
        "head_v_dim": head_dim,
    }


def to_flat_layout(inp):
    """Re-pack interleaved ``projected_qkvz`` / ``projected_ba`` (per key-head
    group ``[q | k | v*ratio | z*ratio]``, ``[b*ratio | a*ratio]``) into the flat
    Qwen3.5/3.8 layout ``[q_all | k_all | v_all | z_all]``, ``[b_all | a_all]``.
    Returns a copy of ``inp`` with the two tensors replaced."""
    kh, vh = inp["num_k_heads"], inp["num_v_heads"]
    kd, vd = inp["head_k_dim"], inp["head_v_dim"]
    ratio = vh // kh
    qkvz = inp["projected_qkvz"].view(-1, kh, 2 * kd + 2 * ratio * vd)
    q, k, v, z = torch.split(qkvz, [kd, kd, ratio * vd, ratio * vd], dim=-1)
    flat_qkvz = torch.cat([t.reshape(t.shape[0], -1) for t in (q, k, v, z)], dim=-1)
    ba = inp["projected_ba"].view(-1, kh, 2 * ratio)
    b, a = torch.split(ba, [ratio, ratio], dim=-1)
    flat_ba = torch.cat([b.reshape(b.shape[0], -1), a.reshape(a.shape[0], -1)], dim=-1)
    out = dict(inp)
    out["projected_qkvz"] = flat_qkvz.contiguous()
    out["projected_ba"] = flat_ba.contiguous()
    return out


def _import_op():
    try:
        import triton.experimental.gluon  # noqa: F401
    except ImportError:
        pytest.skip("Triton Gluon not available")
    # Deliberately not guarded: an ImportError raised from inside the op module
    # is a broken op, and must fail rather than skip every kernel test below.
    from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
        fused_gdn_decode_qkvz,
    )

    return fused_gdn_decode_qkvz


def _requires_gfx950():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip(f"Gluon GDN decode kernel is gfx950-only, got {get_arch()}")


# --------------------------------------------------------------------------
# The reference is itself under test: these run without the kernel and pin the
# contract (layout, gate algebra, pad-slot semantics) independently of any port.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", _BATCHES)
def test_reference_shapes_and_finiteness(batch):
    inp = make_inputs(batch)
    normalized, quantized, scales = ref_gdn_decode(
        **inp, quant_dtype=FP8_DTYPE, quant_group_size=128
    )
    nv, hd = inp["num_v_heads"], inp["head_v_dim"]
    assert normalized.shape == (batch, nv, hd)
    assert normalized.dtype == torch.bfloat16
    assert quantized.shape == (batch, nv * hd)
    assert quantized.dtype == FP8_DTYPE
    assert scales.shape == (batch, nv)
    assert torch.isfinite(normalized.float()).all()
    assert torch.isfinite(scales).all()
    assert (scales > 0).all()


def test_reference_pad_slot_leaves_state_untouched():
    """PAD_SLOT_ID rows must not read or write conv/ssm state.

    This is the contract SGLang's replay_state_indices_validator documents and
    the one aiter's other GDN ops already honour. Artemis instead redirects
    padded rows to reserved pool slot 0 via a monkeypatched _replay_metadata;
    adopting the sentinel is what makes this a self-contained library op.
    """
    inp = make_inputs(4)
    inp["ssm_state_indices"] = torch.tensor(
        [1, -1, 2, -1], dtype=torch.int32, device=device
    )
    conv_before = inp["conv_state"].clone()
    ssm_before = inp["ssm_state"].clone()

    normalized, _, _ = ref_gdn_decode(**inp, pad_slot_id=-1)

    # Rows 1 and 3 are padding: their slots must be untouched and output zero.
    assert torch.equal(inp["conv_state"][3:], conv_before[3:])
    assert torch.equal(inp["ssm_state"][3:], ssm_before[3:])
    assert not torch.equal(inp["conv_state"][1], conv_before[1])
    assert torch.equal(normalized[1], torch.zeros_like(normalized[1]))
    assert torch.equal(normalized[3], torch.zeros_like(normalized[3]))


def test_reference_quant_roundtrip():
    """Dequantizing must recover the bf16 output to within fp8 group error."""
    inp = make_inputs(8)
    normalized, quantized, scales = ref_gdn_decode(**inp, quant_dtype=FP8_DTYPE)
    nv, hd = inp["num_v_heads"], inp["head_v_dim"]
    dequant = quantized.float().view(-1, nv, hd) * scales[:, :, None]
    rel = (dequant - normalized.float()).abs() / normalized.float().abs().clamp_min(
        1e-3
    )
    # e4m3 has ~3 mantissa bits; 12.5% relative error is the group-quant bound.
    assert rel.median() < 0.125


def test_reference_bf16_and_fp8_are_consistent():
    """The fp8 output must be a quantization of the returned bf16, not of fp32."""
    inp = make_inputs(4)
    inp_copy = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    bf16_only, q_none, s_none = ref_gdn_decode(**inp, quant_dtype=None)
    assert q_none is None and s_none is None
    normalized, _, _ = ref_gdn_decode(**inp_copy, quant_dtype=FP8_DTYPE)
    assert torch.equal(bf16_only, normalized)


# --------------------------------------------------------------------------
# Kernel tests. Skipped until the op lands; they are the acceptance criteria.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", _BATCHES)
def test_fused_gdn_decode_correctness(batch):
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(batch)
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, ref_quant, ref_scales = ref_gdn_decode(**ref_inp, quant_dtype=FP8_DTYPE)

    out, quant, scales = fused_gdn_decode_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        norm_eps=inp["norm_eps"],
        quant_dtype=FP8_DTYPE,
    )

    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(scales, ref_scales, atol=ATOL, rtol=ATOL)
    # Compare FP8 in value space, not code space: a 1-ULP e4m3 code difference
    # near the top of the range is huge as a raw integer and meaningless as a
    # value. Dequantize with each side's own scale.
    nv, hd = inp["num_v_heads"], inp["head_v_dim"]
    torch.testing.assert_close(
        quant.float().view(-1, nv, hd) * scales[:, :, None],
        ref_quant.float().view(-1, nv, hd) * ref_scales[:, :, None],
        atol=ATOL,
        rtol=0.13,  # e4m3 has ~3 mantissa bits; this is the group-quant bound
    )
    torch.testing.assert_close(
        inp["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        inp["conv_state"].float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )


# Head topologies served by the ratio-generic kernel at every batch size:
# (16, 3) is Qwen3.5/3.8-27B (48 value heads over 16 key heads, 10240 conv
# channels); (8, 1) and (4, 4) bound the supported ratio range.
_RATIO_BATCHES = [1, 16, 64, 128]
_RATIO_TOPOLOGIES = [(16, 3), (8, 1), (4, 4)]


@pytest.mark.parametrize("batch", _RATIO_BATCHES)
@pytest.mark.parametrize("num_k_heads,ratio", _RATIO_TOPOLOGIES)
def test_fused_gdn_decode_correctness_ratio(batch, num_k_heads, ratio):
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(batch, num_k_heads=num_k_heads, ratio=ratio)
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, ref_quant, ref_scales = ref_gdn_decode(**ref_inp, quant_dtype=FP8_DTYPE)

    out, quant, scales = fused_gdn_decode_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        norm_eps=inp["norm_eps"],
        quant_dtype=FP8_DTYPE,
    )

    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(scales, ref_scales, atol=ATOL, rtol=ATOL)
    nv, hd = inp["num_v_heads"], inp["head_v_dim"]
    torch.testing.assert_close(
        quant.float().view(-1, nv, hd) * scales[:, :, None],
        ref_quant.float().view(-1, nv, hd) * ref_scales[:, :, None],
        atol=ATOL,
        rtol=0.13,
    )
    torch.testing.assert_close(
        inp["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        inp["conv_state"].float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("num_k_heads,ratio", [(16, 3), (4, 2), (8, 1), (4, 4)])
def test_fused_gdn_decode_correctness_flat_layout(batch, num_k_heads, ratio):
    """Flat ``[q_all | k_all | v_all | z_all]`` packing (Qwen3.5/3.8 checkpoints)
    must match the reference run on the equivalent interleaved packing; ratio 2
    takes the ratio-generic kernel here instead of the generated pair kernels."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(batch, num_k_heads=num_k_heads, ratio=ratio)
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, _, _ = ref_gdn_decode(**ref_inp, quant_dtype=None)
    flat = to_flat_layout(inp)
    out, quant, scales = fused_gdn_decode_qkvz(
        flat["projected_qkvz"],
        flat["projected_ba"],
        flat["conv_state"],
        flat["ssm_state"],
        flat["ssm_state_indices"],
        flat["conv_weight"],
        flat["conv_bias"],
        flat["A_log"],
        flat["dt_bias"],
        flat["norm_weight"],
        scale=flat["scale"],
        norm_eps=flat["norm_eps"],
        quant_dtype=None,
        qkvz_layout="flat",
    )
    assert quant is None and scales is None
    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(
        flat["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        flat["conv_state"].float(),
        ref_inp["conv_state"].float(),
        atol=ATOL,
        rtol=ATOL,
    )


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("conv_layout", ["channel_major", "width_major"])
@pytest.mark.parametrize(
    "num_k_heads,ratio,qkvz_layout", [(16, 3, "flat"), (4, 2, "interleaved")]
)
def test_fused_gdn_decode_slot_strided_state_pools(
    batch, conv_layout, num_k_heads, ratio, qkvz_layout
):
    """Serving engines carve the conv and recurrent states of a block out of one
    page, so the state tensors are strided along the slot dimension, and store
    the conv window either channel-major ([N, channels, 3], vLLM "DS") or
    width-major ([N, 3, channels] handed over as a transpose, vLLM "SD"). The
    ratio-generic kernel must read and write them in place through those views."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(batch, num_k_heads=num_k_heads, ratio=ratio)
    slots = inp["conv_state"].shape[0]
    channels, width = inp["conv_state"].shape[1:]
    vh, vd, kd = inp["ssm_state"].shape[1:]
    conv_pool = torch.randn(
        slots, channels * width + 4096, dtype=torch.bfloat16, device=device
    )
    ssm_pool = torch.randn(
        slots, vh * vd * kd + 1024, dtype=torch.float32, device=device
    )
    if conv_layout == "channel_major":
        conv_view = conv_pool[:, : channels * width].unflatten(1, (channels, width))
    else:
        conv_view = (
            conv_pool[:, : channels * width]
            .unflatten(1, (width, channels))
            .transpose(1, 2)
        )
    ssm_view = ssm_pool[:, : vh * vd * kd].unflatten(1, (vh, vd, kd))
    assert not conv_view.is_contiguous() and not ssm_view.is_contiguous()
    conv_view.copy_(inp["conv_state"])
    ssm_view.copy_(inp["ssm_state"])
    pool_tail_conv = conv_pool[:, channels * width :].clone()
    pool_tail_ssm = ssm_pool[:, vh * vd * kd :].clone()

    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, _, _ = ref_gdn_decode(**ref_inp, quant_dtype=None)
    ops = to_flat_layout(inp) if qkvz_layout == "flat" else inp
    # Ratio 2 interleaved reaches the ratio-generic kernel only with query_start_loc.
    qsl = (
        torch.arange(batch + 1, dtype=torch.int32, device=device)
        if qkvz_layout == "interleaved"
        else None
    )
    out, _, _ = fused_gdn_decode_qkvz(
        ops["projected_qkvz"],
        ops["projected_ba"],
        conv_view,
        ssm_view,
        ops["ssm_state_indices"],
        ops["conv_weight"],
        ops["conv_bias"],
        ops["A_log"],
        ops["dt_bias"],
        ops["norm_weight"],
        scale=ops["scale"],
        norm_eps=ops["norm_eps"],
        quant_dtype=None,
        qkvz_layout=qkvz_layout,
        query_start_loc=qsl,
    )
    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(ssm_view, ref_inp["ssm_state"], atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(
        conv_view.float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )
    # Nothing outside the state views was touched.
    assert torch.equal(conv_pool[:, channels * width :], pool_tail_conv)
    assert torch.equal(ssm_pool[:, vh * vd * kd :], pool_tail_ssm)


@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize(
    "num_k_heads,ratio,qkvz_layout", [(16, 3, "flat"), (4, 2, "interleaved")]
)
def test_fused_gdn_decode_query_start_loc_padding(
    batch, num_k_heads, ratio, qkvz_layout
):
    """CUDA-graph replay pads the batch: rows past the real requests have an
    empty query segment but their slot index is the null block or a stale live
    slot. With ``query_start_loc`` those rows must not touch any state."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(batch, num_k_heads=num_k_heads, ratio=ratio)
    live = batch - 3
    # Padded rows alias live slots (worst case) and the null slot 0.
    idx = inp["ssm_state_indices"].clone()
    idx[live] = idx[0]
    idx[live + 1] = idx[1]
    idx[live + 2] = 0
    qsl = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    qsl[1 : live + 1] = torch.arange(1, live + 1, dtype=torch.int32, device=device)
    qsl[live + 1 :] = live
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    for k in ("projected_qkvz", "projected_ba", "ssm_state_indices"):
        ref_inp[k] = ref_inp[k][:live]
    ref_norm, _, _ = ref_gdn_decode(**ref_inp, quant_dtype=None)
    ops = to_flat_layout(inp) if qkvz_layout == "flat" else inp
    out, _, _ = fused_gdn_decode_qkvz(
        ops["projected_qkvz"],
        ops["projected_ba"],
        ops["conv_state"],
        ops["ssm_state"],
        idx,
        ops["conv_weight"],
        ops["conv_bias"],
        ops["A_log"],
        ops["dt_bias"],
        ops["norm_weight"],
        scale=ops["scale"],
        norm_eps=ops["norm_eps"],
        quant_dtype=None,
        qkvz_layout=qkvz_layout,
        query_start_loc=qsl,
    )
    torch.testing.assert_close(
        out[:live].float(), ref_norm.float(), atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        ops["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        ops["conv_state"].float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )


def test_fused_gdn_decode_supported_rejects_cpu_optional_operands():
    """``out`` and ``query_start_loc`` are kernel operands like the rest, so CPU
    tensors with valid shapes and dtypes must not pass the support check."""
    _import_op()
    _requires_gfx950()
    from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
        fused_gdn_decode_qkvz_supported,
    )

    batch = 4
    ops = to_flat_layout(make_inputs(batch, num_k_heads=16, ratio=3))
    args = [
        ops[k]
        for k in (
            "projected_qkvz",
            "projected_ba",
            "conv_state",
            "ssm_state",
            "ssm_state_indices",
            "conv_weight",
            "conv_bias",
            "A_log",
            "dt_bias",
            "norm_weight",
        )
    ]
    vh, vd = ops["num_v_heads"], ops["head_v_dim"]
    qsl = torch.arange(batch + 1, dtype=torch.int32)
    out = torch.empty(batch, vh, vd, dtype=torch.bfloat16)

    assert fused_gdn_decode_qkvz_supported(
        *args, qkvz_layout="flat", query_start_loc=qsl.to(device), out=out.to(device)
    ) == (True, "")
    for kwargs in ({"query_start_loc": qsl}, {"out": out}):
        ok, reason = fused_gdn_decode_qkvz_supported(
            *args, qkvz_layout="flat", **kwargs
        )
        assert not ok and "device" in reason, kwargs.keys()


def test_fused_gdn_decode_preallocated_out():
    """``out=`` writes the normalised output into the caller's buffer in place
    (a slice of a larger activation buffer) and returns that same tensor."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    batch = 16
    inp = make_inputs(batch, num_k_heads=16, ratio=3)
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, _, _ = ref_gdn_decode(**ref_inp, quant_dtype=None)
    flat = to_flat_layout(inp)
    buf = torch.full(
        (batch + 8, inp["num_v_heads"], inp["head_v_dim"]),
        7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    out = buf[:batch]
    ret, _, _ = fused_gdn_decode_qkvz(
        flat["projected_qkvz"],
        flat["projected_ba"],
        flat["conv_state"],
        flat["ssm_state"],
        flat["ssm_state_indices"],
        flat["conv_weight"],
        flat["conv_bias"],
        flat["A_log"],
        flat["dt_bias"],
        flat["norm_weight"],
        scale=flat["scale"],
        norm_eps=flat["norm_eps"],
        quant_dtype=None,
        qkvz_layout="flat",
        out=out,
    )
    assert ret is out
    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    assert torch.all(buf[batch:] == 7.0)


def test_fused_gdn_decode_ratio3_bf16_only_and_pad_slot():
    """Ratio 3: ``quant_dtype=None`` returns bf16 only, and PAD_SLOT_ID rows leave
    both state pools untouched."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(8, num_k_heads=16, ratio=3)
    inp["ssm_state_indices"][3] = -1  # PAD_SLOT_ID
    pad_slot_ssm = inp["ssm_state"].clone()
    pad_slot_conv = inp["conv_state"].clone()
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    ref_norm, _, _ = ref_gdn_decode(**ref_inp, quant_dtype=None)
    out, quant, scales = fused_gdn_decode_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        norm_eps=inp["norm_eps"],
        quant_dtype=None,
    )
    assert quant is None and scales is None
    live = inp["ssm_state_indices"] != -1
    torch.testing.assert_close(
        out[live].float(), ref_norm[live].float(), atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        inp["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    # The slot the padded row would have used (slot 4 = row 3 + 1) is bit-unchanged.
    assert torch.equal(inp["ssm_state"][4], pad_slot_ssm[4])
    assert torch.equal(inp["conv_state"][4], pad_slot_conv[4])


def test_fused_gdn_decode_bf16_only():
    """quant_dtype=None must still produce the bf16 output (the 79% path).

    An unquantized out_proj cannot consume an (fp8, scales) tuple, so the fp8
    epilogue has to be optional for the op to be generally useful.
    """
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(32)
    ref_inp = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
    }
    out, quant, scales = fused_gdn_decode_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        norm_eps=inp["norm_eps"],
        quant_dtype=None,
    )
    assert out.dtype == torch.bfloat16
    assert quant is None and scales is None

    # HAS_FP8=False is a separate compiled specialization, so its bf16 output
    # and state mutations need checking too -- not just that it returns None.
    ref_norm, ref_q, ref_s = ref_gdn_decode(**ref_inp, quant_dtype=None)
    assert ref_q is None and ref_s is None
    torch.testing.assert_close(out.float(), ref_norm.float(), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(
        inp["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        inp["conv_state"].float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )


def test_fused_gdn_decode_pad_slot():
    """PAD_SLOT_ID (-1) rows are skipped without modifying state."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(4)
    inp["ssm_state_indices"] = torch.tensor(
        [-1, -1, -1, -1], dtype=torch.int32, device=device
    )
    conv_before = inp["conv_state"].clone()
    ssm_before = inp["ssm_state"].clone()

    fused_gdn_decode_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        norm_eps=inp["norm_eps"],
    )
    assert torch.equal(
        inp["ssm_state"], ssm_before
    ), "SSM state modified for PAD_SLOT_ID"
    assert torch.equal(
        inp["conv_state"], conv_before
    ), "Conv state modified for PAD_SLOT_ID"


def test_fused_gdn_decode_determinism():
    """Same inputs, same state, twice: bit-identical outputs."""
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(32)
    args = (
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"].clone(),
        inp["ssm_state"].clone(),
        inp["ssm_state_indices"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
    )
    kwargs = {
        "scale": inp["scale"],
        "norm_eps": inp["norm_eps"],
        "quant_dtype": FP8_DTYPE,
    }
    first = fused_gdn_decode_qkvz(
        *args[:2], *[a.clone() for a in args[2:4]], *args[4:], **kwargs
    )
    second = fused_gdn_decode_qkvz(
        *args[:2], *[a.clone() for a in args[2:4]], *args[4:], **kwargs
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1].float(), second[1].float())
    assert torch.equal(first[2], second[2])
