# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused Qwen3-Next GDN *prefill* kernel.

Fuses, in a tight set of Gluon launches (not a single launch): packed qkvz/ba
split, depthwise causal conv1d (with bias) + SiLU, delta-rule gating, the fp32
chunked delta-rule scan over ragged sequences, gated RMSNorm with a SiLU gate,
and a per-head group-128 FP8 quantization epilogue.

This is the *prefill sibling* of ``fused_gdn_decode_qkvz``. The per-token math is
identical; prefill carries the conv window and the fp32 recurrent state
*sequentially within each ragged sequence* (``cu_seqlens`` + ``has_initial_state``)
and writes the final window/state back to the pools, instead of doing a single
step per slot.

The reference here is a pure-torch fp32 sequential scan -- deliberately *not* the
chunked algorithm the kernel uses, so agreement is meaningful. It mirrors the
bf16 rounding boundaries the kernel commits to (conv output, beta, core output,
normalized value) or the tolerance comparison would be vacuous.
"""

import pytest
import torch

# Inputs are CUDA-only; skip the whole module on CPU-only workers before any
# allocation runs (matches the sibling fused_kda decode test).
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

device = "cuda"

ATOL = 0.05
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 128  # one FP8 scale per (token, v_head)


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


def ref_gdn_prefill(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    cu_seqlens,
    has_initial_state,
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
    quant=True,
):
    """Pure-torch fp32 sequential reference. Mutates ``conv_state``/``delta_state``.

    Returns ``(normalized_bf16, quantized_fp8, scales)`` matching the kernel's
    ``(normalized, quantized, scales)`` outputs (the two mutated pools are checked
    in place by the caller).
    """
    T = projected_qkvz.shape[0]
    ratio = num_v_heads // num_k_heads
    width = conv_weight.shape[-1]
    batch = cache_indices.numel()

    q, k, v, z = split_qkvz(
        projected_qkvz.float(), num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    x = pack_conv_channels(q, k, v)  # [T, channels]
    channels = x.shape[-1]
    cw = conv_weight.float()
    cb = conv_bias.float()
    a_log_f = A_log.float()
    dt_f = dt_bias.float()

    out = torch.zeros(T, num_v_heads, head_v_dim, dtype=torch.float32, device=device)
    qk_split = 2 * num_k_heads * head_k_dim

    for b in range(batch):
        slot = int(cache_indices[b].item())
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())

        # Kernel semantics: has_initial_state gates ONLY the conv history (the
        # scan is called without it). The recurrent state is ALWAYS seeded from
        # the pool; the caller zeroes fresh slots. Match that here.
        state = delta_state[slot].float().clone()  # [v_heads, VD, KD]
        if bool(has_initial_state[b].item()):
            history = conv_state[slot].float().clone()  # [channels, width-1]
        else:
            history = torch.zeros(
                channels, width - 1, dtype=torch.float32, device=device
            )

        for t in range(start, end):
            # --- depthwise causal conv1d + bias + SiLU -----------------------
            acc = cb.clone()
            for tap in range(width - 1):
                acc += history[:, tap] * cw[:, tap]
            acc += x[t] * cw[:, width - 1]
            conv_out = acc * torch.sigmoid(acc)

            shifted = torch.empty_like(history)
            shifted[:, : width - 2] = history[:, 1:]
            shifted[:, width - 2] = x[t]
            history = shifted

            q_t = conv_out[: qk_split // 2].view(num_k_heads, head_k_dim)
            k_t = conv_out[qk_split // 2 : qk_split].view(num_k_heads, head_k_dim)
            v_t = conv_out[qk_split:].view(num_v_heads, head_v_dim)

            # --- QK L2 norm (eps inside rsqrt, on the sum of squares) --------
            q_t = q_t * torch.rsqrt((q_t * q_t).sum(-1, keepdim=True) + 1e-6) * scale
            k_t = k_t * torch.rsqrt((k_t * k_t).sum(-1, keepdim=True) + 1e-6)

            # --- GDN gate, vectorized over v_heads ---------------------------
            g_idx = torch.arange(num_v_heads, device=device) // ratio  # [VH]
            r_idx = torch.arange(num_v_heads, device=device) % ratio
            ba_base = 2 * ratio * g_idx
            b_val = projected_ba[t, ba_base + r_idx].float()
            a_val = projected_ba[t, ba_base + ratio + r_idx].float()
            sp_arg = a_val + dt_f
            softplus = torch.where(
                sp_arg <= 20.0, torch.log1p(torch.exp(sp_arg)), sp_arg
            )
            decay = torch.exp(-torch.exp(a_log_f) * softplus)  # [VH]
            beta = torch.sigmoid(b_val).to(torch.bfloat16).float()  # [VH]

            # --- delta rule on fp32 state [VH, VD, KD] -----------------------
            kv = k_t[g_idx]  # [VH, KD]
            decayed = state * decay[:, None, None]
            predicted = (decayed * kv[:, None, :]).sum(-1)  # [VH, VD]
            residual = (v_t - predicted) * beta[:, None]
            state = decayed + residual[:, :, None] * kv[:, None, :]
            out[t] = (state * q_t[g_idx][:, None, :]).sum(-1)  # [VH, VD]

        conv_state[slot] = history.to(conv_state.dtype)
        delta_state[slot] = state.to(delta_state.dtype)

    # --- gated RMSNorm (kernel rounds the core output to bf16 first) ---------
    core = out.to(torch.bfloat16).float()
    rms = torch.rsqrt((core * core).sum(-1, keepdim=True) / head_v_dim + norm_eps)
    gate = z.float()
    normalized = (
        core * rms * norm_weight.float()[None, None, :] * gate * torch.sigmoid(gate)
    ).to(torch.bfloat16)

    if not quant:
        return normalized, None, None

    # --- per-head group-128 FP8 quantization, on the bf16-rounded value ------
    assert GROUP == head_v_dim, "one scale per (token, v_head)"
    values = normalized.float()
    absmax = values.abs().amax(-1).clamp_min(1e-10)
    scales = absmax / FP8_MAX
    quantized = (values / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return normalized, quantized.view(-1, num_v_heads * head_v_dim), scales


def make_inputs(seqlens, num_k_heads=4, head_dim=128, width=4, seed=0):
    """Build a self-consistent Qwen3-Next GDN prefill input set.

    ``seqlens`` is the per-sequence token count list; every sequence gets a
    distinct slot (duplicate live slots would race the pooled state).
    """
    torch.manual_seed(seed)
    batch = len(seqlens)
    num_v_heads = 2 * num_k_heads
    ratio = num_v_heads // num_k_heads
    group_width = 2 * head_dim + 2 * ratio * head_dim
    channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    m = int(sum(seqlens))
    num_slots = batch + 1  # slot 0 reserved for CUDA-graph dummy traffic

    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)

    return {
        "projected_qkvz": (
            torch.randn(
                m, num_k_heads * group_width, dtype=torch.bfloat16, device=device
            )
            * 0.1
        ),
        "projected_ba": torch.randn(
            m, 2 * num_v_heads, dtype=torch.bfloat16, device=device
        )
        * 0.1,
        "conv_state": torch.randn(
            num_slots, channels, width - 1, dtype=torch.bfloat16, device=device
        )
        * 0.1,
        "delta_state": torch.randn(
            num_slots,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.1,
        "cache_indices": torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        "cu_seqlens": cu,
        "has_initial_state": torch.ones(batch, dtype=torch.bool, device=device),
        "conv_weight": torch.randn(channels, width, dtype=torch.bfloat16, device=device)
        * 0.1,
        "conv_bias": torch.randn(channels, dtype=torch.bfloat16, device=device) * 0.1,
        "A_log": torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1,
        "dt_bias": torch.randn(num_v_heads, dtype=torch.bfloat16, device=device) * 0.1,
        "norm_weight": torch.ones(head_dim, dtype=torch.bfloat16, device=device),
        "scale": head_dim**-0.5,
        "norm_eps": 1e-6,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_dim,
        "head_v_dim": head_dim,
    }


def _require_supported(inp):
    """Skip unless the op reports this exact call as supported (arch + Triton)."""
    from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
        fused_gdn_prefill_qkvz_supported,
    )

    ok, reason = fused_gdn_prefill_qkvz_supported(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["delta_state"],
        inp["cache_indices"],
        inp["cu_seqlens"],
        inp["has_initial_state"],
        inp["conv_weight"],
        inp["conv_bias"],
        FP8_DTYPE,
    )
    if not ok:
        pytest.skip(f"unsupported here: {reason}")


def _run_kernel(inp):
    from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
        fused_gdn_prefill_qkvz,
    )

    return fused_gdn_prefill_qkvz(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["delta_state"],
        inp["cache_indices"],
        inp["cu_seqlens"],
        inp["has_initial_state"],
        inp["conv_weight"],
        inp["conv_bias"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_weight"],
        scale=inp["scale"],
        eps=inp["norm_eps"],
    )


def _split_ref_kwargs(inp):
    return {
        k: inp[k]
        for k in (
            "cache_indices",
            "cu_seqlens",
            "has_initial_state",
            "scale",
            "norm_eps",
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
        )
    }


# Each case names the tile it must dispatch to (see the wrapper's profile map).
# One case per tile so every dispatched schedule is checked against the single
# reference below. The two 12289-token cases are heavier (the sequential
# reference is O(tokens)) but are deliberately left unmarked: aiter CI runs
# ``pytest op_tests/triton_tests/`` with no marker filter, so an unconditional
# param is what guarantees they actually execute in CI rather than silently rot.
_CASES = [
    pytest.param([1024], "m1024_3071", id="m1024_b1"),
    pytest.param([1024, 1024], "m1024_3071", id="m2048_b2"),
    pytest.param([3072], "m3072_16384", id="m3072_b1"),
    pytest.param([2048, 2048, 2048], "m3072_16384", id="m6144_b3"),
    pytest.param([12289], "m12289_16384_b1_5", id="m12289_b1"),
    pytest.param([1600] * 8, "m12289_16384_b6_15", id="m12800_b8"),
]


@pytest.mark.parametrize("seqlens,tile", _CASES)
def test_matches_reference(seqlens, tile):
    from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import _select_tile_key

    inp = make_inputs(seqlens)
    _require_supported(inp)

    m = int(sum(seqlens))
    assert _select_tile_key(m, len(seqlens)) == tile, "dispatched to the wrong tile"

    # Reference mutates its own copies of the pools.
    ref_conv = inp["conv_state"].clone()
    ref_delta = inp["delta_state"].clone()
    ref_norm, ref_q, ref_sc = ref_gdn_prefill(
        inp["projected_qkvz"],
        inp["projected_ba"],
        ref_conv,
        ref_delta,
        conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        norm_weight=inp["norm_weight"],
        **_split_ref_kwargs(inp),
    )

    norm, conv_out, delta_out, quant, scales = _run_kernel(inp)

    torch.testing.assert_close(norm.float(), ref_norm.float(), atol=ATOL, rtol=0)
    torch.testing.assert_close(conv_out.float(), ref_conv.float(), atol=ATOL, rtol=0)
    torch.testing.assert_close(delta_out.float(), ref_delta.float(), atol=ATOL, rtol=0)
    # Dequantized FP8 activations must match the reference within group-scale error.
    deq = quant.float().view(-1, inp["num_v_heads"], GROUP) * scales[:, :, None]
    ref_deq = ref_q.float().view(-1, inp["num_v_heads"], GROUP) * ref_sc[:, :, None]
    torch.testing.assert_close(deq, ref_deq, atol=2 * ATOL, rtol=0)


def test_no_initial_state_zeroes_carry():
    """has_initial_state=False must ignore whatever is in the pools."""
    inp = make_inputs([2048])
    _require_supported(inp)
    inp["has_initial_state"] = torch.zeros(1, dtype=torch.bool, device=device)
    # has_initial_state gates ONLY the conv history: poison conv_state and a
    # correct kernel must ignore it. delta_state is always read (caller zeroes
    # fresh slots), so leave it at its realistic small values.
    inp["conv_state"].fill_(9.0)

    ref_conv = inp["conv_state"].clone()
    ref_delta = inp["delta_state"].clone()
    ref_norm, _, _ = ref_gdn_prefill(
        inp["projected_qkvz"],
        inp["projected_ba"],
        ref_conv,
        ref_delta,
        conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        norm_weight=inp["norm_weight"],
        **_split_ref_kwargs(inp),
    )
    norm, *_ = _run_kernel(inp)
    torch.testing.assert_close(norm.float(), ref_norm.float(), atol=ATOL, rtol=0)


def test_determinism():
    inp = make_inputs([2048, 1024])
    _require_supported(inp)

    def once():
        i = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}
        return _run_kernel(i)[0]

    torch.testing.assert_close(once().float(), once().float(), atol=0, rtol=0)


def test_unsupported_shape_reports_reason():
    """Below the smallest tile -> not supported, with a reason (no crash)."""
    from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
        fused_gdn_prefill_qkvz_supported,
    )

    inp = make_inputs([512])  # 512 tokens < 1024 floor
    ok, reason = fused_gdn_prefill_qkvz_supported(
        inp["projected_qkvz"],
        inp["projected_ba"],
        inp["conv_state"],
        inp["delta_state"],
        inp["cache_indices"],
        inp["cu_seqlens"],
        inp["has_initial_state"],
        inp["conv_weight"],
        inp["conv_bias"],
        FP8_DTYPE,
    )
    # On gfx950+Triton3.8 this is the tile-coverage reason; elsewhere it is the
    # arch/Triton gate. Either way it is a clean (False, reason), never a raise.
    assert ok is False and reason
