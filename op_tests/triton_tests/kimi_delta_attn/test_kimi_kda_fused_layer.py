# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused Kimi-K3 KDA whole-layer ops (gfx950, TP8).

The dispatch-table and fail-closed gate tests run anywhere (no GPU). The
run/correctness tests require a gfx950 device with Triton >= 3.8 and are
skipped otherwise.
"""

import pytest
import torch

from aiter.ops.triton.gated_delta_net.causal_conv1d_prefill import (
    causal_conv1d_split_qkv_triton_fn,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode
from aiter.ops.triton.kimi_delta_attn import (
    chunk_kimi_delta_attn,
    kimi_kda_fused_layer_decode,
    kimi_kda_fused_layer_decode_supported,
    kimi_kda_fused_layer_prefill,
    kimi_kda_fused_layer_prefill_supported,
)
from aiter.ops.triton.kimi_delta_attn import fused_layer_decode as _decode
from aiter.ops.triton.kimi_delta_attn import fused_layer_prefill as _prefill
from aiter.ops.triton.utils._triton import arch_info

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32


def _gfx950():
    try:
        return arch_info.get_arch() == "gfx950"
    except Exception:  # noqa: BLE001 - any probe failure => not gfx950
        return False


on_device = pytest.mark.skipif(
    not (torch.cuda.is_available() and _gfx950()),
    reason="requires a gfx950 device",
)


# --- dispatch table (pure, no GPU) ---------------------------------------
@pytest.mark.parametrize(
    "m,key",
    [
        (1, "m64"),
        (2, "m64"),
        (32, "m64"),
        (64, "m64"),
        (72, "m256"),
        (96, "m256"),
        (128, "m256"),
        (256, "m256"),
        (0, None),
        (257, None),
        (65, None),
        (70, None),
        (100, None),  # m>64 not divisible by the group
    ],
)
def test_select_tile_key(m, key):
    assert _decode._select_tile_key(m) == key


def test_prefill_coverage():
    assert _prefill._covered(1024, 1)
    assert _prefill._covered(8192, 32)
    assert not _prefill._covered(1023, 1)
    assert not _prefill._covered(8193, 1)
    assert not _prefill._covered(1024, 33)


# --- fail-closed gates (no GPU) ------------------------------------------
def test_decode_supported_fails_closed_on_junk():
    ok, reason = kimi_kda_fused_layer_decode_supported(*([None] * 12))
    assert ok is False and reason


def test_prefill_supported_fails_closed_on_junk():
    ok, reason = kimi_kda_fused_layer_prefill_supported(*([None] * 14))
    assert ok is False and reason


# --- on-device run / in-place / determinism ------------------------------
def _rnd(shape, dtype, dev):
    if dtype == I32:
        return torch.zeros(shape, dtype=I32, device=dev)
    return (torch.randn(shape, device=dev, dtype=torch.float32) * 0.05).to(dtype)


def _decode_inputs(m, dev, seed=0):
    torch.manual_seed(seed)
    slots = max(m, 64)
    return {
        "x": _rnd((m, 7168), BF16, dev),
        "qkvg_weight": _rnd((6144, 7168), BF16, dev),
        "beta_forget_weight": _rnd((144, 7168), BF16, dev),
        "output_weight": _rnd((7168, 1536), BF16, dev),
        "forget_weight": _rnd((1536, 128), BF16, dev),
        "conv_weight": _rnd((4608, 4), F32, dev),
        "a_log": _rnd((12,), F32, dev),
        "dt_bias": _rnd((1536,), F32, dev),
        "norm_weight": _rnd((128,), BF16, dev),
        "conv_state": _rnd((slots, 3, 4608), BF16, dev),
        "state": _rnd((slots, 12, 128, 128), F32, dev),
        "state_indices": torch.arange(m, dtype=I32, device=dev),
    }


def _ref_decode(ins, lower_bound, norm_eps):
    """Independent reference for the fused decode layer.

    Composes the native aiter KDA ops the fused kernel replaces: the QKVG and
    beta/forget-A projections and the O projection are plain matmuls, and the
    conv + single-token gated-delta recurrence + gated RMSNorm core is the
    native ``fused_kda_decode``. Returns ``(o_ref, conv_ref, state_ref)`` so the
    test checks both the output and the in-place cache updates. The projection
    layout mirrors the fused decode launcher: ``mixed_qkv`` and the output gate
    are the two halves of QKVG, the decay gate is the forget-B projection of
    forget-A, and ``beta`` is the 12 heads that follow forget-A.
    """
    m = ins["x"].shape[0]
    x = ins["x"].float()
    qkvg = (x @ ins["qkvg_weight"].float().T).to(BF16)  # (m, 6144)
    beta_forget = (x @ ins["beta_forget_weight"].float().T).to(BF16)  # (m, 144)
    mixed_qkv = qkvg[:, :4608].contiguous()
    out_gate = qkvg[:, 4608:6144].contiguous()
    forget_a = beta_forget[:, :128]
    beta = beta_forget[:, 128:140].reshape(1, m, 12).contiguous()
    gate = (
        (forget_a.float() @ ins["forget_weight"].float().T)
        .to(BF16)
        .reshape(1, m, 12, 128)
        .contiguous()
    )
    # The native op wants conv_state as [N, 3*lp, STATE_LEN]; the fused cache is
    # [N, STATE_LEN, 3*lp]. Clone first: the native op updates it in place.
    conv_ref = ins["conv_state"].clone().transpose(1, 2).contiguous()
    state_ref = ins["state"].clone()
    cu_seqlens = torch.arange(0, m + 1, dtype=torch.int64, device=ins["x"].device)
    core = fused_kda_decode(
        mixed_qkv,
        conv_ref,
        ins["conv_weight"],
        gate,
        beta,
        out_gate,
        ins["a_log"],
        ins["dt_bias"],
        state_ref,
        ins["state_indices"],
        cu_seqlens,
        ins["norm_weight"],
        norm_eps,
        head_dim=128,
        num_local_heads=12,
        lower_bound=lower_bound,
    )
    o_ref = (core.float() @ ins["output_weight"].float().T).to(BF16)
    return o_ref, conv_ref.transpose(1, 2).contiguous(), state_ref


@on_device
@pytest.mark.parametrize("m", [1, 8, 32, 64, 96, 128, 192, 256])
def test_decode_matches_native_reference(m):
    """The fused decode layer matches the native KDA decode path it replaces.

    Checks correctness against an independent reference (projections + native
    ``fused_kda_decode`` + O matmul), plus the wrapper contract: the output
    lands in ``output_tensor`` and the conv/recurrent caches update in place.
    """
    dev = "cuda"
    ins = _decode_inputs(m, dev, seed=m)
    ok, reason = kimi_kda_fused_layer_decode_supported(**ins)
    assert ok, reason

    o_ref, conv_ref, state_ref = _ref_decode(ins, lower_bound=-5.0, norm_eps=1e-5)

    conv_before = ins["conv_state"].clone()
    state_before = ins["state"].clone()
    out = torch.empty((m, 7168), dtype=BF16, device=dev)
    o, conv, state = kimi_kda_fused_layer_decode(
        **ins, lower_bound=-5.0, norm_eps=1e-5, output_tensor=out
    )

    # wrapper contract: shape/dtype, output written in place, caches aliased
    assert o.shape == (m, 7168) and o.dtype == BF16
    assert o.data_ptr() == out.data_ptr()
    assert conv.data_ptr() == ins["conv_state"].data_ptr()
    assert state.data_ptr() == ins["state"].data_ptr()
    assert not torch.equal(conv, conv_before)
    assert not torch.equal(state, state_before)

    # correctness against the native reference
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(conv.float(), conv_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state, state_ref, atol=1e-2, rtol=1e-2)


@on_device
def test_decode_rejects_bad_output_tensor():
    dev = "cuda"
    ins = _decode_inputs(8, dev)
    with pytest.raises(ValueError):
        kimi_kda_fused_layer_decode(
            **ins, output_tensor=torch.empty((8, 7168), dtype=F32, device=dev)
        )


def _prefill_inputs(m, n, dev, seed=0, initial=False):
    torch.manual_seed(seed)
    slots = max(n, 4)
    base = m // n
    lens = [base] * n
    lens[-1] += m - base * n
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=I32, device=dev
    )
    has_initial_state = torch.zeros(n, dtype=torch.bool, device=dev)
    if initial:
        has_initial_state[::2] = True  # read initial states for the even sequences
    return {
        "qkv": _rnd((m, 4608), BF16, dev),
        "gate": _rnd((m, 1536), BF16, dev),
        "forget_a": _rnd((m, 128), BF16, dev),
        "beta": _rnd((m, 12), BF16, dev),
        "forget_weight": _rnd((1536, 128), BF16, dev),
        "conv_weight": _rnd((4608, 4), F32, dev),
        "a_log": _rnd((12,), F32, dev),
        "dt_bias": _rnd((1536,), F32, dev),
        "norm_weight": _rnd((128,), BF16, dev),
        "conv_state": _rnd((slots, 3, 4608), BF16, dev),
        "state": _rnd((slots, 12, 128, 128), F32, dev),
        "state_indices": torch.arange(n, dtype=I32, device=dev),
        "cu_seqlens": cu,
        "has_initial_state": has_initial_state,
    }


def _ref_prefill(ins, lower_bound, norm_eps):
    """Independent reference for the fused prefill core.

    Composes the native aiter ops the fused kernel replaces: native causal
    conv1d + SiLU (``causal_conv1d_split_qkv_triton_fn``), the chunked KDA mixer
    (``chunk_kimi_delta_attn``, which fuses the forget-B decay, beta sigmoid, q/k
    L2-norm and the fp32 chunked gated-delta scan), and a torch gated RMSNorm.
    The recurrent cache is V-first ``[N, H, V, K]`` -- the same layout the native
    decode op uses -- so the scan runs with ``state_v_first=True``. Returns
    ``(o_ref, state_ref)``: the ``(m, 1536)`` core and the cache with each
    sequence's final state scattered back into its slot.
    """
    dev = ins["qkv"].device
    m = ins["qkv"].shape[0]
    n = ins["state_indices"].numel()
    cu = ins["cu_seqlens"]
    lens = (cu[1:] - cu[:-1]).tolist()
    sidx = ins["state_indices"]
    has_init = ins["has_initial_state"]

    # native causal conv1d + SiLU over the ragged batch; x is channel-major and
    # the native op splits the 4608 channels into q/k/v and updates its cache.
    x = ins["qkv"].float().T.contiguous().to(BF16)
    conv_ref = ins["conv_state"].clone().transpose(1, 2).contiguous()
    q, k, v = causal_conv1d_split_qkv_triton_fn(
        x,
        ins["conv_weight"].to(BF16),
        None,
        conv_ref,
        cu,
        lens,
        k_dim=1536,
        v_dim=1536,
        cache_indices=sidx,
        has_initial_state=has_init,
        activation="silu",
    )
    q = q.reshape(1, m, 12, 128)
    k = k.reshape(1, m, 12, 128)
    v = v.reshape(1, m, 12, 128)

    # forget-B decay gate + the chunked gated-delta scan (gate/beta/l2norm fused)
    g = (ins["forget_a"].float() @ ins["forget_weight"].float().T).to(BF16)
    g = g.reshape(1, m, 12, 128)
    beta = ins["beta"].reshape(1, m, 12).contiguous()
    h0 = torch.zeros(n, 12, 128, 128, device=dev, dtype=F32)
    for i in range(n):
        if bool(has_init[i]):
            h0[i] = ins["state"][int(sidx[i])]
    core, final = chunk_kimi_delta_attn(
        q,
        k,
        v,
        g,
        beta,
        A_log=ins["a_log"],
        dt_bias=ins["dt_bias"],
        scale=128**-0.5,
        initial_state=h0,
        output_final_state=True,
        state_v_first=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu.long(),
    )

    # gated RMSNorm over the 128-wide head
    cf = core.reshape(m, 12, 128).float()
    inv = torch.rsqrt((cf * cf).mean(-1, keepdim=True) + norm_eps)
    gate = torch.sigmoid(ins["gate"].reshape(m, 12, 128).float())
    o_ref = (cf * inv * ins["norm_weight"].float() * gate).reshape(m, 1536).to(BF16)

    state_ref = ins["state"].clone()
    for i in range(n):
        state_ref[int(sidx[i])] = final[i]
    return o_ref, state_ref


@on_device
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("m,n", [(1024, 1), (2048, 4), (8192, 32)])
def test_prefill_matches_native_reference(m, n, initial):
    """The fused prefill core matches the native KDA prefill path it replaces.

    Checks correctness against an independent reference (native conv + chunked
    KDA mixer + gated RMSNorm), plus the wrapper contract: the conv/recurrent
    caches update in place. Covers both the no-initial-state and the
    read-initial-state ragged paths.
    """
    dev = "cuda"
    ins = _prefill_inputs(m, n, dev, seed=m + n, initial=initial)
    ok, reason = kimi_kda_fused_layer_prefill_supported(**ins)
    assert ok, reason

    o_ref, state_ref = _ref_prefill(ins, lower_bound=-5.0, norm_eps=1e-5)

    conv_before = ins["conv_state"].clone()
    state_before = ins["state"].clone()
    o, conv, state = kimi_kda_fused_layer_prefill(
        **ins, lower_bound=-5.0, norm_eps=1e-5
    )

    # wrapper contract: shape/dtype, caches aliased and updated in place
    assert o.shape == (m, 1536) and o.dtype == BF16
    assert conv.data_ptr() == ins["conv_state"].data_ptr()
    assert state.data_ptr() == ins["state"].data_ptr()
    assert not torch.equal(conv, conv_before)
    assert not torch.equal(state, state_before)

    # correctness against the native reference
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state, state_ref, atol=2e-2, rtol=2e-2)
