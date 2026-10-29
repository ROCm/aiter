# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for causal_conv1d_split_qkv_triton_tile_fn on x with unit channel stride (the sliding-window path).

x is a column slice of a row-major projection output, as in Kimi-K3 and Qwen3-Next prefill.
"""

import itertools

import pytest
import torch

import aiter.ops.triton.gated_delta_net.causal_conv1d_prefill as conv_prefill
from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_triton_tile_fn

PAD_SLOT_ID = -1
WIDTH = 4


def _reference(
    x, weight, bias, conv_states, qsl, cache_indices, has_init, k_dim, v_dim, activation
):
    """fp32 reference. Returns q, k, v, the new conv states and the mask of rows that have a defined output."""
    dim, T = x.shape
    sl = WIDTH - 1
    w = weight.float()
    out = torch.zeros(T, dim, dtype=torch.float32, device=x.device)
    new_states = conv_states.clone()
    valid = torch.zeros(T, dtype=torch.bool, device=x.device)
    starts, slots, inits = qsl.tolist(), cache_indices.tolist(), has_init.tolist()
    for s in range(len(starts) - 1):
        a, b = starts[s], starts[s + 1]
        if b <= a or slots[s] == PAD_SLOT_ID:
            continue
        prior = (
            conv_states[slots[s], :, :sl].float()
            if inits[s]
            else torch.zeros(dim, sl, dtype=torch.float32, device=x.device)
        )
        xx = torch.cat([prior, x[:, a:b].float()], dim=1)
        acc = sum(w[:, j : j + 1] * xx[:, j : j + b - a] for j in range(WIDTH))
        if bias is not None:
            acc = acc + bias.float()[:, None]
        if activation in ("silu", "swish"):
            acc = acc * torch.sigmoid(acc)
        out[a:b] = acc.t()
        valid[a:b] = True
        new_states[slots[s], :, :sl] = xx[:, -sl:].to(conv_states.dtype)
    q = out[:, :k_dim]
    k = out[:, k_dim : 2 * k_dim]
    v = out[:, 2 * k_dim : 2 * k_dim + v_dim]
    return q, k, v, new_states, valid


def _make_inputs(
    seqlens,
    has_init,
    k_dim,
    v_dim,
    row_extra,
    state_cols,
    sd_layout,
    n_slots,
    pad_seqs,
    w_dtype,
    state_dtype,
    seed,
):
    torch.manual_seed(seed)
    dim = 2 * k_dim + v_dim
    T = sum(seqlens)
    proj = torch.randn((T, dim + row_extra), dtype=torch.bfloat16, device="cuda")
    x = proj[:, :dim].t()
    weight = torch.randn((dim, WIDTH), dtype=w_dtype, device="cuda") * 0.5
    if sd_layout:
        conv_states = torch.randn(
            (n_slots, state_cols, dim), dtype=state_dtype, device="cuda"
        ).transpose(1, 2)
    else:
        conv_states = torch.randn(
            (n_slots, dim, state_cols), dtype=state_dtype, device="cuda"
        )
    qsl = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device="cuda"
    )
    cache_indices = torch.randperm(n_slots, device="cuda")[: len(seqlens)].to(
        torch.int32
    )
    for i in pad_seqs:
        cache_indices[i] = PAD_SLOT_ID
    has_initial_state = torch.tensor(has_init, dtype=torch.bool, device="cuda")
    return x, weight, conv_states, qsl, cache_indices, has_initial_state


def _check(
    seqlens,
    has_init,
    k_dim=1536,
    v_dim=1536,
    row_extra=1680,
    state_cols=7,
    sd_layout=False,
    n_slots=None,
    pad_seqs=(),
    w_dtype=torch.float32,
    state_dtype=torch.bfloat16,
    preserve_input_dtype=True,
    with_bias=False,
    activation="silu",
    strided_ci=False,
    seed=0,
):
    n_slots = n_slots or max(16, 2 * len(seqlens))
    x, weight, conv_states, qsl, ci, hi = _make_inputs(
        seqlens,
        has_init,
        k_dim,
        v_dim,
        row_extra,
        state_cols,
        sd_layout,
        n_slots,
        pad_seqs,
        w_dtype,
        state_dtype,
        seed,
    )
    if strided_ci:
        # Odd entries hold other sequences' slots, so a unit-stride read is caught
        ci = torch.stack([ci, ci.roll(1)], dim=1).flatten()[::2]
    bias = (
        torch.randn((x.shape[0],), dtype=torch.float32, device="cuda") * 0.1
        if with_bias
        else None
    )
    q_r, k_r, v_r, st_r, valid = _reference(
        x, weight, bias, conv_states, qsl, ci, hi, k_dim, v_dim, activation
    )
    q, k, v = causal_conv1d_split_qkv_triton_tile_fn(
        x,
        weight,
        bias,
        conv_states,
        qsl,
        k_dim,
        v_dim,
        cache_indices=ci,
        has_initial_state=hi,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
        block_m=8,
        preserve_input_dtype=preserve_input_dtype,
    )
    out_dtype = x.dtype if preserve_input_dtype else conv_states.dtype
    for got, ref, width in ((q, q_r, k_dim), (k, k_r, k_dim), (v, v_r, v_dim)):
        assert (
            got.shape == (x.shape[1], width)
            and got.is_contiguous()
            and got.dtype == out_dtype
        )
        torch.testing.assert_close(got.float()[valid], ref[valid], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv_states, st_r, atol=0, rtol=0)


class _LaunchSpy:
    """Counts the launches of a kernel and forwards them."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.launches += 1
            return launch(*args, **kwargs)

        return run


@pytest.mark.parametrize(
    "seqlens,has_init",
    [
        ([7680], [True]),  # one continuing prefill chunk
        ([5010, 2680], [True, False]),
        ([1112] * 7, [False] * 7),
        ([1, 2, 3, 4, 5, 7, 8, 9], [True, False, True, True, False, True, False, True]),
        (
            [30, 33, 31, 34, 62, 3, 29, 100],
            [True, True, False, True, True, True, False, True],
        ),
        ([63, 64, 65, 127, 128, 129, 255, 257], [True, False] * 4),
        ([1] * 16, [True] * 8 + [False] * 8),
        ([10, 0, 5, 0, 40], [True] * 5),  # zero-length sequences
    ],
)
@pytest.mark.parametrize(
    "row_extra", [1680, 0]
)  # x row stride 6288 (projection view) or 4608 (gather)
def test_kimi_k3_shapes(seqlens, has_init, row_extra):
    _check(seqlens, has_init, row_extra=row_extra)


@pytest.mark.parametrize("n_seq", [17, 40, 100, 300])
def test_many_sequences(n_seq):
    # 300 sequences is more than the largest sequence-start vector: the kernel uses a binary search.
    seqlens = [(7 * i) % 23 + 1 for i in range(n_seq)]
    _check(seqlens, [i % 3 != 0 for i in range(n_seq)], n_slots=2 * n_seq)


@pytest.mark.parametrize("sd_layout", [False, True])
@pytest.mark.parametrize("state_cols", [3, 7])
def test_state_layouts(sd_layout, state_cols):
    _check(
        [1000, 1, 2, 3, 517],
        [True, True, False, True, True],
        sd_layout=sd_layout,
        state_cols=state_cols,
    )


def test_pad_slots():
    _check([40, 2, 70, 1, 33], [True] * 5, pad_seqs=(0, 3))


@pytest.mark.parametrize("k_dim,v_dim", [(256, 512), (2048, 4096)])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_qwen3_next_dims(k_dim, v_dim, w_dtype, with_bias):
    _check(
        [300, 2, 1, 129],
        [True, True, True, False],
        k_dim=k_dim,
        v_dim=v_dim,
        row_extra=64,
        w_dtype=w_dtype,
        with_bias=with_bias,
    )


@pytest.mark.parametrize("window", ["1", "0"])
def test_strided_cache_indices(monkeypatch, window):
    # cache_indices with stride 2 (e.g. indices[::2]) on both kernels.
    monkeypatch.setenv("AITER_CONV1D_SPLIT_QKV_WINDOW", window)
    _check([257, 3, 40, 1], [True, False, True, True], strided_ci=True)


@pytest.mark.parametrize("activation", [None, "silu"])
def test_activation(activation):
    _check([257, 3], [True, False], activation=activation)


@pytest.mark.parametrize("preserve_input_dtype", [True, False])
def test_fp32_conv_state(preserve_input_dtype):
    # bf16 x with an fp32 conv state (vLLM --mamba-cache-dtype float32).
    _check(
        [700, 2, 33],
        [True, True, False],
        state_dtype=torch.float32,
        preserve_input_dtype=preserve_input_dtype,
    )


@pytest.mark.parametrize(
    "seqlens,has_init,row_extra",
    [
        ([7680], [True], 1680),
        ([5010, 2680], [True, False], 0),
        ([1112] * 7, [False] * 7, 0),
        (
            [1, 2, 3, 4, 5, 7, 8, 9],
            [True, False, True, True, False, True, False, True],
            0,
        ),
    ],
)
def test_2d_tiled_kernel_kimi_k3_shapes(monkeypatch, seqlens, has_init, row_extra):
    # The 2D-tiled kernel with the vLLM launch (block_m=8): derived block_n and one store per program.
    monkeypatch.setenv("AITER_CONV1D_SPLIT_QKV_WINDOW", "0")
    _check(seqlens, has_init, row_extra=row_extra)


@pytest.mark.parametrize("opt_out", [False, True])
def test_window_kernel_selection(monkeypatch, opt_out):
    # x with unit channel stride selects the sliding-window kernel; AITER_CONV1D_SPLIT_QKV_WINDOW=0 turns it off.
    monkeypatch.setenv("AITER_CONV1D_SPLIT_QKV_WINDOW", "0" if opt_out else "1")
    spy = _LaunchSpy(conv_prefill._causal_conv1d_fwd_split_qkv_window_kernel)
    monkeypatch.setattr(conv_prefill, "_causal_conv1d_fwd_split_qkv_window_kernel", spy)
    _check([700, 33], [True, False])
    # Archs without a window config (e.g. gfx942) use the 2D-tiled kernel
    has_config = conv_prefill.has_conv_config(conv_prefill._WINDOW_CONFIG_NAME)
    assert spy.launches == (0 if opt_out or not has_config else 1)


def test_token_contiguous_x_uses_2d_kernel(monkeypatch):
    # x with unit token stride is not for the sliding-window kernel.
    spy = _LaunchSpy(conv_prefill._causal_conv1d_fwd_split_qkv_window_kernel)
    monkeypatch.setattr(conv_prefill, "_causal_conv1d_fwd_split_qkv_window_kernel", spy)
    torch.manual_seed(0)
    k_dim = v_dim = 256
    dim = 3 * k_dim
    x = torch.randn((dim, 138), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((dim, WIDTH), dtype=torch.float32, device="cuda") * 0.5
    conv_states = torch.randn((8, dim, 3), dtype=torch.bfloat16, device="cuda")
    qsl = torch.tensor([0, 33, 38, 138], dtype=torch.int32, device="cuda")
    ci = torch.tensor([3, 1, 6], dtype=torch.int32, device="cuda")
    hi = torch.tensor([True, False, True], dtype=torch.bool, device="cuda")
    q_r, k_r, v_r, st_r, valid = _reference(
        x, weight, None, conv_states, qsl, ci, hi, k_dim, v_dim, "silu"
    )
    q, k, v = causal_conv1d_split_qkv_triton_tile_fn(
        x,
        weight,
        None,
        conv_states,
        qsl,
        k_dim,
        v_dim,
        cache_indices=ci,
        has_initial_state=hi,
        preserve_input_dtype=True,
    )
    assert spy.launches == 0
    for got, ref in ((q, q_r), (k, k_r), (v, v_r)):
        torch.testing.assert_close(got.float()[valid], ref[valid], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv_states, st_r, atol=0, rtol=0)
