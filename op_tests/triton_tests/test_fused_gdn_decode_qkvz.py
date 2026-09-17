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


def make_inputs(batch, num_k_heads=4, head_dim=128, num_slots=None, width=4, seed=0):
    """Build a self-consistent input set for the Qwen3-Next GDN decode contract.

    Every live row gets a *distinct* slot. Duplicate slots would let two rows
    read-modify-write the same state concurrently, which is order-dependent in
    the kernel and sequential in the reference — a mismatch that says nothing
    about correctness. SGLang's own validator enforces distinct live slots.
    """
    torch.manual_seed(seed)
    # Slot 0 is reserved, so N must exceed the batch to give everyone a slot.
    num_slots = num_slots if num_slots is not None else batch + 1
    num_v_heads = 2 * num_k_heads
    ratio = num_v_heads // num_k_heads
    group_width = 2 * head_dim + 2 * ratio * head_dim
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim

    return dict(
        projected_qkvz=torch.randn(
            batch, num_k_heads * group_width, dtype=torch.bfloat16, device=device
        ),
        projected_ba=torch.randn(
            batch, 2 * num_v_heads, dtype=torch.bfloat16, device=device
        ),
        conv_state=torch.randn(
            num_slots, channels, width - 1, dtype=torch.bfloat16, device=device
        ),
        ssm_state=torch.randn(
            num_slots,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        ),
        # Slot 0 is reserved for CUDA-graph dummy traffic; allocate from 1 up.
        ssm_state_indices=torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        # Deployed dtypes: conv/norm/dt parameters arrive as bf16 model weights,
        # A_log as fp32, and the recurrent state as fp32. Two of the seven
        # Artemis variants (m16, m64) assert exactly this; the rest accept
        # whatever the pointer says, which is why the contract is worth pinning.
        conv_weight=torch.randn(channels, width, dtype=torch.bfloat16, device=device),
        conv_bias=torch.randn(channels, dtype=torch.bfloat16, device=device),
        A_log=torch.randn(num_v_heads, dtype=torch.float32, device=device),
        dt_bias=torch.randn(num_v_heads, dtype=torch.bfloat16, device=device),
        norm_weight=torch.randn(head_dim, dtype=torch.bfloat16, device=device),
        scale=head_dim**-0.5,
        norm_eps=1e-6,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
    )


def _import_op():
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz import (
            fused_gdn_decode_qkvz,
        )
    except ImportError:
        pytest.skip("fused_gdn_decode_qkvz not available")
    return fused_gdn_decode_qkvz


def _requires_gfx950():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip(f"Gluon GDN decode kernel is gfx950-only, got {get_arch()}")


# --------------------------------------------------------------------------
# The reference is itself under test: these run without the kernel and pin the
# contract (layout, gate algebra, pad-slot semantics) independently of any port.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64, 128, 192, 256])
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


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64, 128, 192, 256])
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
    torch.testing.assert_close(
        inp["ssm_state"], ref_inp["ssm_state"], atol=ATOL, rtol=ATOL
    )
    torch.testing.assert_close(
        inp["conv_state"].float(), ref_inp["conv_state"].float(), atol=ATOL, rtol=ATOL
    )


def test_fused_gdn_decode_bf16_only():
    """quant_dtype=None must still produce the bf16 output (the 79% path).

    An unquantized out_proj cannot consume an (fp8, scales) tuple, so the fp8
    epilogue has to be optional for the op to be generally useful.
    """
    fused_gdn_decode_qkvz = _import_op()
    _requires_gfx950()

    inp = make_inputs(32)
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
    kwargs = dict(scale=inp["scale"], norm_eps=inp["norm_eps"], quant_dtype=FP8_DTYPE)
    first = fused_gdn_decode_qkvz(
        *args[:2], *[a.clone() for a in args[2:4]], *args[4:], **kwargs
    )
    second = fused_gdn_decode_qkvz(
        *args[:2], *[a.clone() for a in args[2:4]], *args[4:], **kwargs
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1].float(), second[1].float())
    assert torch.equal(first[2], second[2])
