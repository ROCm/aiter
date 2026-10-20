# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused Kimi-K3 KDA whole-layer ops (gfx950, TP8).

The dispatch-table and fail-closed gate tests run anywhere (no GPU). The
run/correctness tests require a gfx950 device with Triton >= 3.8 and are
skipped otherwise.
"""

import pytest
import torch

from aiter.ops.triton.kimi_delta_attn import fused_layer_decode as _decode
from aiter.ops.triton.kimi_delta_attn import fused_layer_prefill as _prefill
from aiter.ops.triton.kimi_delta_attn import (
    kimi_kda_fused_layer_decode,
    kimi_kda_fused_layer_decode_supported,
    kimi_kda_fused_layer_prefill,
    kimi_kda_fused_layer_prefill_supported,
)
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


@on_device
@pytest.mark.parametrize("m", [1, 8, 32, 64, 96, 128, 192, 256])
def test_decode_runs_inplace_deterministic(m):
    dev = "cuda"
    ins = _decode_inputs(m, dev, seed=m)
    ok, reason = kimi_kda_fused_layer_decode_supported(**ins)
    assert ok, reason
    conv_before = ins["conv_state"].clone()
    state_before = ins["state"].clone()
    out = torch.empty((m, 7168), dtype=BF16, device=dev)
    o, conv, state = kimi_kda_fused_layer_decode(
        **ins, lower_bound=-5.0, norm_eps=1e-5, output_tensor=out
    )
    assert o.shape == (m, 7168) and o.dtype == BF16
    assert o.data_ptr() == out.data_ptr()
    assert conv.data_ptr() == ins["conv_state"].data_ptr()
    assert state.data_ptr() == ins["state"].data_ptr()
    # the conv/recurrent caches are updated in place
    assert not torch.equal(conv, conv_before)
    assert not torch.equal(state, state_before)
    ins2 = _decode_inputs(m, dev, seed=m)
    out2 = torch.empty((m, 7168), dtype=BF16, device=dev)
    o2, _, _ = kimi_kda_fused_layer_decode(
        **ins2, lower_bound=-5.0, norm_eps=1e-5, output_tensor=out2
    )
    assert torch.equal(o, o2)


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


@on_device
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("m,n", [(1024, 1), (2048, 4), (8192, 32)])
def test_prefill_runs_inplace_deterministic(m, n, initial):
    dev = "cuda"
    ins = _prefill_inputs(m, n, dev, seed=m + n, initial=initial)
    ok, reason = kimi_kda_fused_layer_prefill_supported(**ins)
    assert ok, reason
    conv_before = ins["conv_state"].clone()
    state_before = ins["state"].clone()
    o, conv, state = kimi_kda_fused_layer_prefill(
        **ins, lower_bound=-5.0, norm_eps=1e-5
    )
    assert o.shape == (m, 1536) and o.dtype == BF16
    assert conv.data_ptr() == ins["conv_state"].data_ptr()
    assert state.data_ptr() == ins["state"].data_ptr()
    # the conv/recurrent caches are updated in place
    assert not torch.equal(conv, conv_before)
    assert not torch.equal(state, state_before)
    ins2 = _prefill_inputs(m, n, dev, seed=m + n, initial=initial)
    o2, _, _ = kimi_kda_fused_layer_prefill(**ins2, lower_bound=-5.0, norm_eps=1e-5)
    assert torch.equal(o, o2)
