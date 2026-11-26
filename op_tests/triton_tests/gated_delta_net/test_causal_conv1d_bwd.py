# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for causal_conv1d_split_qkv_bwd_triton_fn against an fp32 autograd reference.

Kimi-K3 KDA short convs (https://github.com/ZhangDanyang-AMD/Lumen/issues/35):
q, k and v each have a depthwise weight ``[12288, 1, 4]``. The TP8 shard has
1536 channels per tensor. x is a column slice of a row-major projection output.
"""

import itertools
import types

import pytest
import torch

from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_bwd_triton_fn

PAD_SLOT_ID = -1
KIMI_K3_DIM = 12288
KIMI_K3_DIM_TP8 = 1536


def _reference(x, weight, bias, dout, qsl, states, cache_indices, has_init, activation):
    """fp32 autograd reference. dout is [cu_seqlen, dim]. Returns (dx, dw, db, dinit)."""
    dim, T = x.shape
    width = weight.shape[1]
    sl = width - 1
    w = weight.float().detach().requires_grad_(True)
    b = None if bias is None else bias.float().detach().requires_grad_(True)
    dx = torch.zeros((dim, T), dtype=torch.float32, device=x.device)
    starts, slots = qsl.tolist(), cache_indices.tolist()
    inits = has_init.tolist() if has_init is not None else [False] * len(slots)
    dinit = torch.zeros((len(slots), dim, sl), dtype=torch.float32, device=x.device)
    for s in range(len(starts) - 1):
        lo, hi = starts[s], starts[s + 1]
        if hi <= lo or slots[s] == PAD_SLOT_ID:
            continue
        xs = x[:, lo:hi].float().detach().requires_grad_(True)
        prior = (
            states[slots[s], :, :sl].float().detach().requires_grad_(True)
            if inits[s]
            else torch.zeros((dim, sl), dtype=torch.float32, device=x.device)
        )
        xx = torch.cat([prior, xs], dim=1)
        acc = sum(w[:, j : j + 1] * xx[:, j : j + hi - lo] for j in range(width))
        if b is not None:
            acc = acc + b[:, None]
        if activation in ("silu", "swish"):
            acc = torch.nn.functional.silu(acc)
        acc.backward(dout[lo:hi].float().t())
        dx[:, lo:hi] = xs.grad
        if inits[s]:
            dinit[s] = prior.grad
    dw = w.grad if w.grad is not None else torch.zeros_like(w)
    db = None
    if b is not None:
        db = b.grad if b.grad is not None else torch.zeros_like(b)
    return dx, dw, db, dinit


def _assert_close(got, ref, tol, name):
    err = (got.float() - ref).abs().max().item() if ref.numel() else 0.0
    scale = max(1.0, ref.abs().max().item() if ref.numel() else 0.0)
    assert err <= tol * scale, f"{name}: max err {err:.4g} > {tol} * {scale:.4g}"


def _make_inputs(
    seqlens,
    has_init=None,
    k_dim=KIMI_K3_DIM_TP8,
    v_dim=KIMI_K3_DIM_TP8,
    width=4,
    row_extra=1680,
    x_dtype=torch.bfloat16,
    w_dtype=torch.float32,
    with_bias=False,
    state_cols=None,
    sd_layout=False,
    pad_seqs=(),
    packed_dout=False,
    channel_major=False,
    nan_pad=False,
    tail=0,
    seed=0,
):
    """Inputs of causal_conv1d_split_qkv_bwd_triton_fn, as a dict of its argument names."""
    torch.manual_seed(seed)
    dim = 2 * k_dim + v_dim
    T = sum(seqlens) + tail
    n_seqs = len(seqlens)
    proj = torch.randn((T, dim + row_extra), dtype=x_dtype, device="cuda")
    x = proj[:, :dim].t()
    weight = torch.randn((dim, width), dtype=w_dtype, device="cuda") * 0.5
    bias = (
        torch.randn((dim,), dtype=w_dtype, device="cuda") * 0.1 if with_bias else None
    )
    dout = torch.randn((T, dim), dtype=x_dtype, device="cuda")
    if nan_pad:
        # Padded sequences hold garbage; their gradients must still be exactly zero.
        starts = [0, *itertools.accumulate(seqlens)]
        for i in pad_seqs:
            proj[starts[i] : starts[i + 1]] = float("nan")
            dout[starts[i] : starts[i + 1]] = float("nan")
    if packed_dout:
        dq, dk, dv = dout[:, :k_dim], dout[:, k_dim : 2 * k_dim], dout[:, 2 * k_dim :]
    else:
        dq = dout[:, :k_dim].contiguous()
        dk = dout[:, k_dim : 2 * k_dim].contiguous()
        dv = dout[:, 2 * k_dim :].contiguous()
    if channel_major:
        # Token-contiguous x and gradients: no unit channel stride anywhere.
        x = x.contiguous()
        dq, dk, dv = (g.t().contiguous().t() for g in (dq, dk, dv))
    qsl = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device="cuda"
    )
    n_slots = max(16, 2 * n_seqs)
    cache_indices = torch.randperm(n_slots, device="cuda")[:n_seqs].to(torch.int32)
    for i in pad_seqs:
        cache_indices[i] = PAD_SLOT_ID
    states = None
    has_initial_state = None
    if has_init is not None:
        cols = state_cols or width - 1
        if sd_layout:
            states = torch.randn(
                (n_slots, cols, dim), dtype=x_dtype, device="cuda"
            ).transpose(1, 2)
        else:
            states = torch.randn((n_slots, dim, cols), dtype=x_dtype, device="cuda")
        has_initial_state = torch.tensor(has_init, dtype=torch.bool, device="cuda")
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "x": x,
        "weight": weight,
        "bias": bias,
        "query_start_loc": qsl,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "initial_states": states,
        "cache_indices": cache_indices,
        "has_initial_state": has_initial_state,
        "dout": dout,
    }


def _check(
    seqlens,
    has_init=None,
    width=4,
    x_dtype=torch.bfloat16,
    w_dtype=torch.float32,
    with_bias=False,
    activation="silu",
    **kwargs,
):
    inp = _make_inputs(
        seqlens,
        has_init,
        width=width,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        with_bias=with_bias,
        **kwargs,
    )
    dout = inp.pop("dout")
    x, weight, bias = inp["x"], inp["weight"], inp["bias"]
    dim, n_seqs = x.shape[0], len(seqlens)
    dx_r, dw_r, db_r, dinit_r = _reference(
        x,
        weight,
        bias,
        dout,
        inp["query_start_loc"],
        inp["initial_states"],
        inp["cache_indices"],
        inp["has_initial_state"],
        activation,
    )
    dx, dw, db, dinit = causal_conv1d_split_qkv_bwd_triton_fn(
        **inp, activation=activation, pad_slot_id=PAD_SLOT_ID
    )
    assert dx.shape == x.shape and dx.dtype == x.dtype
    assert dw.shape == weight.shape and dw.dtype == weight.dtype
    tol = 2e-2 if x_dtype == torch.bfloat16 else 1e-4
    _assert_close(dx, dx_r, tol, "dx")
    _assert_close(dw, dw_r, 1e-2 if w_dtype == torch.bfloat16 else 1e-4, "dweight")
    if with_bias:
        assert db.shape == bias.shape and db.dtype == bias.dtype
        _assert_close(db, db_r, 1e-2 if w_dtype == torch.bfloat16 else 1e-4, "dbias")
    else:
        assert db is None
    if has_init is not None:
        assert dinit.shape == (n_seqs, dim, width - 1)
        _assert_close(dinit, dinit_r, tol, "dinitial_states")
    else:
        assert dinit is None


@pytest.mark.parametrize(
    "seqlens",
    [
        [7680],
        [5010, 2680],
        [1112] * 7,
        [1, 2, 3, 4, 5, 7, 8, 9],
        [63, 64, 65, 127, 128, 129, 255, 257],
        [10, 0, 5, 0, 40],  # zero-length sequences
    ],
)
@pytest.mark.parametrize("row_extra", [1680, 0])
def test_kimi_k3_tp8_shapes(seqlens, row_extra):
    _check(seqlens, row_extra=row_extra)


@pytest.mark.parametrize("seqlens", [[4096], [1024, 3, 1000]])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_kimi_k3_full_dim(seqlens, w_dtype):
    # q, k and v with 12288 channels each (TP1).
    _check(
        seqlens,
        k_dim=KIMI_K3_DIM,
        v_dim=KIMI_K3_DIM,
        row_extra=0,
        w_dtype=w_dtype,
        with_bias=True,
    )


@pytest.mark.parametrize("sd_layout", [False, True])
@pytest.mark.parametrize("state_cols", [3, 7])
def test_initial_states(sd_layout, state_cols):
    _check(
        [1000, 1, 2, 3, 517],
        has_init=[True, True, False, True, True],
        sd_layout=sd_layout,
        state_cols=state_cols,
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_width_activation_bias(width, activation, with_bias):
    _check(
        [257, 3, 64, 1],
        has_init=[True, False, True, True],
        width=width,
        activation=activation,
        with_bias=with_bias,
    )


def test_pad_slots():
    _check([40, 2, 70, 1, 33], has_init=[True] * 5, pad_seqs=(0, 3))


def test_pad_slots_nan():
    # NaN in the x and dout rows of padded sequences must not reach any gradient.
    _check(
        [40, 2, 70, 1, 33],
        has_init=[True] * 5,
        pad_seqs=(0, 3),
        nan_pad=True,
        with_bias=True,
    )


def test_packed_dout_and_tail_tokens():
    # dq, dk, dv as column views of one [T, dim] buffer; tokens after the last sequence get zero.
    _check([300, 2, 129], packed_dout=True, tail=17)


def test_channel_major_strides():
    _check([300, 2, 129], has_init=[True, False, True], channel_major=True)


def test_large_channel_stride():
    # Token-contiguous x inside a [36864, 65536] buffer: channel offsets from 32768
    # on pass 2**31 elements, so they need int64 arithmetic.
    k_dim = v_dim = KIMI_K3_DIM
    dim, row, T = 3 * k_dim, 65536, 256
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free device memory")
    torch.manual_seed(0)
    buf = torch.empty((dim, row), dtype=torch.bfloat16, device="cuda")
    x = buf[:, :T]
    x.normal_()
    weight = torch.randn((dim, 4), dtype=torch.float32, device="cuda") * 0.5
    dout = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda")
    qsl = torch.tensor([0, 100, T], dtype=torch.int32, device="cuda")
    slots = torch.arange(2, dtype=torch.int32, device="cuda")
    dx_r, dw_r, _, _ = _reference(x, weight, None, dout, qsl, None, slots, None, "silu")
    dx, dw, _, _ = causal_conv1d_split_qkv_bwd_triton_fn(
        dout[:, :k_dim].contiguous(),
        dout[:, k_dim : 2 * k_dim].contiguous(),
        dout[:, 2 * k_dim :].contiguous(),
        x,
        weight,
        None,
        qsl,
        k_dim,
        v_dim,
    )
    _assert_close(dx, dx_r, 2e-2, "dx")
    _assert_close(dw, dw_r, 1e-4, "dweight")


@pytest.mark.parametrize("k_dim,v_dim", [(256, 512), (100, 260)])
def test_unaligned_split(k_dim, v_dim):
    # Channel blocks that straddle the q/k/v boundaries.
    _check([300, 2, 1, 129], k_dim=k_dim, v_dim=v_dim, row_extra=64, with_bias=True)


def test_fp32():
    _check(
        [513, 7],
        has_init=[True, False],
        x_dtype=torch.float32,
        w_dtype=torch.float32,
        with_bias=True,
    )


def test_deterministic():
    torch.manual_seed(0)
    k_dim = v_dim = KIMI_K3_DIM_TP8
    dim, T = 3 * k_dim, 4096
    x = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda").t()
    weight = torch.randn((dim, 4), dtype=torch.float32, device="cuda")
    dout = torch.randn((T, dim), dtype=torch.bfloat16, device="cuda")
    qsl = torch.tensor([0, 1000, T], dtype=torch.int32, device="cuda")
    args = (dout[:, :k_dim], dout[:, k_dim : 2 * k_dim], dout[:, 2 * k_dim :], x)
    first = causal_conv1d_split_qkv_bwd_triton_fn(
        *args, weight, None, qsl, k_dim, v_dim
    )
    second = causal_conv1d_split_qkv_bwd_triton_fn(
        *args, weight, None, qsl, k_dim, v_dim
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_nums_dict_schedule():
    # The second call reuses the schedule the first one stores, with no host sync.
    # Tokens after the last sequence must still get zero dx, whatever memory dx reuses.
    tail = 17
    inp = _make_inputs([700, 33, 1], has_init=[True, False, True], tail=tail)
    inp.pop("dout")
    ref = causal_conv1d_split_qkv_bwd_triton_fn(**inp)
    metadata = types.SimpleNamespace(nums_dict={})
    causal_conv1d_split_qkv_bwd_triton_fn(**inp, metadata=metadata)
    assert len(metadata.nums_dict) == 1
    poison = torch.full_like(ref[0].t(), float("nan"))
    del poison
    got = causal_conv1d_split_qkv_bwd_triton_fn(**inp, metadata=metadata)
    assert not got[0][:, -tail:].any()
    for g, r in zip(got, ref):
        assert (g is None and r is None) or torch.equal(g, r)
