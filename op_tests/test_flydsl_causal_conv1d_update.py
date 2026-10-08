# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for FlyDSL causal-conv1d update."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import NamedTuple

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.causal_conv1d_update_kernels import (
    _SGLANG_WIDTHS,
    _SUPPORTED_DTYPES,
    _VLLM_WIDTHS,
    NULL_BLOCK_ID,
    _causal_conv1d_update_sglang_flydsl_supported,
    causal_conv1d_update_flydsl,
    causal_conv1d_update_sglang_flydsl,
)
from aiter.ops.triton.conv.causal_conv1d import (
    causal_conv1d_update as causal_conv1d_update_triton,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.triton_tests.utils.causal_conv1d_update_refs import (
    causal_conv1d_update_sglang as causal_conv1d_update_sglang_upstream,
)
from op_tests.triton_tests.utils.causal_conv1d_update_refs import (
    causal_conv1d_update_vllm as causal_conv1d_update_vllm_upstream,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

SUPPORTED_GFX = ("gfx942", "gfx950")
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="FlyDSL causal_conv1d update requires gfx942 or gfx950",
)

_DTYPE_EPS = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_ERR_FACTOR = 8.0


# -- inputs ---------------------------------------------------------------


def _make_inputs(batch, dim, width, seqlen, *, spec, seed, has_bias=True, dtype=DTYPE):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    state_len = (width - 1 + (seqlen - 1)) if spec else (width - 1)
    return {
        "x": torch.randn(
            (batch, dim, seqlen), generator=gen, device=DEVICE, dtype=dtype
        ),
        "conv_state": torch.randn(
            (batch + 1, dim, state_len), generator=gen, device=DEVICE, dtype=dtype
        ),
        "weight": torch.randn((dim, width), generator=gen, device=DEVICE, dtype=dtype),
        "bias": (
            torch.randn((dim,), generator=gen, device=DEVICE, dtype=dtype)
            if has_bias
            else None
        ),
        "num_accepted": (
            torch.randint(
                1, seqlen + 1, (batch,), dtype=torch.int32, device=DEVICE, generator=gen
            )
            if spec
            else None
        ),
    }


def _eagle_tree_links(batch, seqlen):
    nxt = torch.full((batch, seqlen), -1, dtype=torch.int32, device=DEVICE)
    sib = torch.full((batch, seqlen), -1, dtype=torch.int32, device=DEVICE)
    for b in range(batch):
        for i in range(seqlen - 1):
            nxt[b, i] = i + 1
            if i + 2 < seqlen:
                sib[b, i + 1] = i + 2
    return nxt, sib


# -- the upstream kernels, as oracle and as baseline ----------------------


class _Oracle(NamedTuple):

    spec: torch.Tensor
    scale: torch.Tensor
    state: torch.Tensor
    window: torch.Tensor | None
    parents: torch.Tensor | None


def _torch_ref(t, mode):
    varlen = "x_dense" in t
    x = t["x_dense"].float() if varlen else t["x"].float()
    state = t["conv_state"][t["indices"].long()].float()
    weight = t["weight"].float()
    width = weight.shape[1]
    accepted = t.get("num_accepted", t.get("nacc"))
    offsets = (
        accepted.long() - 1
        if accepted is not None
        else torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
    )
    history = torch.stack(
        [
            state[row, :, offset : offset + width - 1]
            for row, offset in enumerate(offsets)
        ]
    )

    parents = None
    if mode == "sglang_verify_tree":
        parents = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.long, device=DEVICE)
        for row in range(x.shape[0]):
            for token in range(x.shape[-1]):
                child = int(t["next_token"][row, token])
                sibling = int(t["next_sibling"][row, token])
                if child >= 0:
                    parents[row, child] = token
                if sibling >= 0:
                    parents[row, sibling] = parents[row, token]

    steps = []
    out = torch.empty_like(x, dtype=torch.float32)
    rows = torch.arange(x.shape[0], device=DEVICE)
    for token in range(x.shape[-1]):
        if parents is not None and token:
            history = torch.stack(steps, dim=1)[rows, parents[:, token]]
        window = torch.cat((history, x[:, :, token, None]), dim=-1)
        value = (window * weight).sum(-1)
        if t["bias"] is not None:
            value += t["bias"].float()
        out[:, :, token] = torch.nn.functional.silu(value)
        history = window[:, :, 1:]
        steps.append(history)

    if mode == "vllm_decode":
        return out.squeeze(-1)
    if mode == "vllm_verify":
        return out.transpose(1, 2).reshape(-1, out.shape[1])
    if varlen:
        return torch.cat(
            [out[row, :, :length].T for row, length in enumerate(t["lens"])],
            dim=0,
        )
    return out


def _oracle_vllm(
    x,
    conv_state,
    weight,
    bias,
    conv_state_indices,
    num_accepted_tokens,
    *,
    null_block_id=NULL_BLOCK_ID,
    query_start_loc=None,
    max_query_len=-1,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
):
    kw = {
        "conv_state_indices": conv_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "null_block_id": null_block_id,
        "query_start_loc": query_start_loc,
        "max_query_len": max_query_len,
        "block_idx_last_scheduled_token": block_idx_last_scheduled_token,
        "initial_state_idx": initial_state_idx,
    }
    state = conv_state.clone()
    causal_conv1d_update_vllm_upstream(
        x.clone(), state, weight, bias=bias, activation="silu", **kw
    )
    spec = causal_conv1d_update_vllm_upstream(
        x.float(),
        conv_state.float(),
        weight.float(),
        bias=None if bias is None else bias.float(),
        activation="silu",
        **kw,
    )
    scale = causal_conv1d_update_vllm_upstream(
        x.float().abs(),
        conv_state.float().abs(),
        weight.float().abs(),
        bias=None if bias is None else bias.float().abs(),
        activation=None,
        **kw,
    )
    return _Oracle(spec.double(), scale.double(), state, None, None)


def _oracle_sglang(
    x,
    conv_state,
    weight,
    bias,
    conv_state_indices,
    num_accept_tokens,
    *,
    pad_slot_id=-1,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
):
    kw = {
        "conv_state_indices": conv_state_indices,
        "num_accept_tokens": num_accept_tokens,
        "intermediate_state_indices": intermediate_state_indices,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "pad_slot_id": pad_slot_id,
    }
    win = intermediate_conv_window

    def parent_buf():
        return (
            None
            if retrieve_next_token is None
            else torch.zeros_like(retrieve_next_token)
        )

    state = conv_state.clone()
    window = None if win is None else win.clone()
    parents = parent_buf()
    causal_conv1d_update_sglang_upstream(
        x.clone(),
        state,
        weight,
        bias=bias,
        activation="silu",
        intermediate_conv_window=window,
        retrieve_parent_token=parents,
        **kw,
    )
    spec = causal_conv1d_update_sglang_upstream(
        x.float(),
        conv_state.float(),
        weight.float(),
        bias=None if bias is None else bias.float(),
        activation="silu",
        intermediate_conv_window=None if win is None else win.float(),
        retrieve_parent_token=parent_buf(),
        **kw,
    )
    scale = causal_conv1d_update_sglang_upstream(
        x.float().abs(),
        conv_state.float().abs(),
        weight.float().abs(),
        bias=None if bias is None else bias.float().abs(),
        activation=None,
        intermediate_conv_window=None if win is None else win.float(),
        retrieve_parent_token=parent_buf(),
        **kw,
    )
    return _Oracle(spec.double(), scale.double(), state, window, parents)


# -- shared assertions ----------------------------------------------------


def _assert_tracks_spec(label, actual, spec, scale, factor=_ERR_FACTOR):
    name = _DTYPE_NAME[actual.dtype]
    err = (actual.double() - spec).abs()
    budget = factor * _DTYPE_EPS[actual.dtype] * scale
    ratio = err / budget.clamp_min(torch.finfo(torch.float64).tiny)
    worst = ratio.argmax()
    assert (err <= budget).all(), (
        f"{label}: output exceeds {factor} {name} steps of its own conditioning; "
        f"worst element {worst.item()}: got {actual.flatten()[worst].item():.6g}, "
        f"spec {spec.flatten()[worst].item():.6g}, "
        f"error {err.flatten()[worst].item():.3e} > budget "
        f"{budget.flatten()[worst].item():.3e} "
        f"(term magnitude {scale.flatten()[worst].item():.3e})"
    )


def _assert_matches_oracle(label, actual, ref):
    _assert_tracks_spec(label, actual, ref.spec, ref.scale)


def _assert_bit_exact(label, what, actual, expected):
    assert torch.equal(actual, expected), (
        f"{label}: {what} is not bit-exact "
        f"(max|delta|={(actual.float() - expected.float()).abs().max():.3e}); "
        "it is assembled from copies, never computed, so this is not rounding"
    )


# -- vLLM interface -------------------------------------------------------

_VLLM_CASES = [
    (1, 128, 4, 1, False),
    (4, 128, 4, 2, False),
    (8, 256, 4, 2, True),
    (16, 256, 4, 8, True),
]


def _check_vllm_case(batch, dim, width, seqlen, spec, *, seed, dtype=DTYPE):
    t = _make_inputs(batch, dim, width, seqlen, spec=spec, seed=seed, dtype=dtype)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    label = f"vllm b{batch} d{dim} w{width} {_DTYPE_NAME[dtype]} s{seqlen} spec={spec}"

    state_fly = t["conv_state"].clone()
    out_fly = causal_conv1d_update_flydsl(
        t["x"].clone(),
        state_fly,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=indices,
        num_accepted_tokens=t["num_accepted"],
    )

    ref = _oracle_vllm(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        indices,
        t["num_accepted"],
    )
    t["indices"] = indices
    ref = ref._replace(spec=_torch_ref(t, "vllm_dense").double())

    _assert_matches_oracle(label, out_fly, ref)
    _assert_bit_exact(label, "conv_state roll-back", state_fly, ref.state)


@pytest.mark.parametrize("batch,dim,width,seqlen,spec", _VLLM_CASES)
def test_vllm_matches_upstream_vllm(batch, dim, width, seqlen, spec):
    _check_vllm_case(batch, dim, width, seqlen, spec, seed=batch * 100 + seqlen)


_WIDTH_DTYPE_CASES = [
    (2, torch.bfloat16, 1, False),
    (2, torch.float16, 4, True),
    (3, torch.bfloat16, 4, True),
    (3, torch.float16, 1, False),
    (4, torch.float16, 4, True),
]
_VLLM_WIDTH_DTYPE_CASES = _WIDTH_DTYPE_CASES + [
    (5, torch.bfloat16, 1, False),
    (6, torch.float16, 1, False),
]


@pytest.mark.parametrize("width,dtype,seqlen,spec", _VLLM_WIDTH_DTYPE_CASES)
def test_vllm_matches_upstream_across_widths_and_dtypes(width, dtype, seqlen, spec):
    assert width in _VLLM_WIDTHS and dtype in _SUPPORTED_DTYPES
    _check_vllm_case(4, 256, width, seqlen, spec, seed=width * 17 + seqlen, dtype=dtype)


def test_vllm_skips_the_null_block():
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=11)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)

    state = t["conv_state"].clone()
    x_in = t["x"].clone()
    out = causal_conv1d_update_flydsl(
        x_in.clone(),
        state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=indices,
        null_block_id=0,
    )

    assert torch.equal(out[0], x_in[0]), "null-block sequence had its output written"
    assert torch.equal(
        state[0], t["conv_state"][0]
    ), "null-block sequence had its cache line rolled"
    assert not torch.equal(
        state[1], t["conv_state"][1]
    ), "a live sequence was skipped as well, so nothing was actually tested"


# -- SGLang interface -----------------------------------------------------

_SGLANG_CASES = [
    (4, 256, 4, 1, False, False, False),
    (32, 384, 4, 4, True, False, False),
    (8, 256, 4, 4, True, True, False),
    (4, 128, 4, 4, True, True, True),
    (8, 256, 4, 8, False, True, True),
]


def _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree, dtype=DTYPE):
    t = _make_inputs(
        batch, dim, width, seqlen, spec=spec, seed=batch * 31 + seqlen, dtype=dtype
    )
    t["indices"] = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    t["window"] = None
    if save_inter:
        gen = torch.Generator(device=DEVICE).manual_seed(5)
        t["window"] = torch.randn(
            (t["conv_state"].shape[0], seqlen, dim, width - 1),
            generator=gen,
            device=DEVICE,
            dtype=dtype,
        )
    t["next_token"], t["next_sibling"] = (
        _eagle_tree_links(batch, seqlen) if tree else (None, None)
    )
    return t


def _run_sglang(fn, t, batch, seqlen, tree, save_inter):
    state = t["conv_state"].clone()
    window = t["window"].clone() if save_inter else None
    parents = (
        torch.zeros((batch, seqlen), dtype=torch.int32, device=DEVICE) if tree else None
    )
    out = fn(
        t["x"].clone(),
        state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["indices"],
        num_accept_tokens=t["num_accepted"],
        intermediate_conv_window=window,
        intermediate_state_indices=t["indices"] if save_inter else None,
        retrieve_next_token=t["next_token"],
        retrieve_next_sibling=t["next_sibling"],
        retrieve_parent_token=parents,
    )
    return out, state, window, parents


def _check_sglang_case(batch, dim, width, seqlen, spec, save_inter, tree, dtype=DTYPE):
    t = _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree, dtype)
    label = (
        f"sglang b{batch} d{dim} w{width} {_DTYPE_NAME[dtype]} s{seqlen} "
        f"spec={spec} inter={save_inter} tree={tree}"
    )

    out, state, window, parents = _run_sglang(
        causal_conv1d_update_sglang_flydsl, t, batch, seqlen, tree, save_inter
    )
    ref = _oracle_sglang(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        intermediate_conv_window=t["window"],
        intermediate_state_indices=t["indices"] if save_inter else None,
        retrieve_next_token=t["next_token"],
        retrieve_next_sibling=t["next_sibling"],
    )
    mode = "sglang_verify_tree" if tree else "sglang_verify"
    ref = ref._replace(spec=_torch_ref(t, mode).double())

    _assert_matches_oracle(label, out, ref)
    _assert_bit_exact(label, "conv_state roll-back", state, ref.state)
    if save_inter:
        _assert_bit_exact(label, "intermediate_conv_window", window, ref.window)
    if tree:
        _assert_bit_exact(label, "retrieve_parent_token map", parents, ref.parents)


@pytest.mark.parametrize("batch,dim,width,seqlen,spec,save_inter,tree", _SGLANG_CASES)
def test_sglang_matches_upstream_sglang(
    batch, dim, width, seqlen, spec, save_inter, tree
):
    _check_sglang_case(batch, dim, width, seqlen, spec, save_inter, tree)


@pytest.mark.parametrize("width,dtype,seqlen,spec", _WIDTH_DTYPE_CASES)
def test_sglang_matches_upstream_across_widths_and_dtypes(width, dtype, seqlen, spec):
    assert width in _SGLANG_WIDTHS and dtype in _SUPPORTED_DTYPES
    _check_sglang_case(
        4, 256, width, seqlen, spec, save_inter=spec, tree=False, dtype=dtype
    )


# -- the aiter dispatch seam ----------------------------------------------


_SEAM_ENV = "AITER_CONV1D_UPDATE_FLYDSL"


def _call_triton_entry(t, batch, seqlen, save_inter, *, flydsl):
    previous = os.environ.get(_SEAM_ENV)
    os.environ[_SEAM_ENV] = "1" if flydsl else "0"
    try:
        state = t["conv_state"].clone()
        window = t["window"].clone() if save_inter else None
        x_in = t["x"].clone()
        out = causal_conv1d_update_triton(
            x_in,
            state,
            t["weight"],
            bias=t["bias"],
            activation="silu",
            conv_state_indices=t["indices"],
            num_accepted_tokens=t["num_accepted"],
            intermediate_conv_window=window,
        )
        return out, state, window, x_in
    finally:
        if previous is None:
            del os.environ[_SEAM_ENV]
        else:
            os.environ[_SEAM_ENV] = previous


_SEAM_CASES = [
    (8, 256, 4, 1, False, False, True),
    (8, 256, 4, 4, True, True, False),
]


@pytest.mark.parametrize(
    "batch,dim,width,seqlen,spec,save_inter,decode_2d", _SEAM_CASES
)
def test_dispatch_seam_routes_to_flydsl(
    batch, dim, width, seqlen, spec, save_inter, decode_2d
):
    t = _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree=False)
    if decode_2d:
        t["x"] = t["x"].squeeze(-1)
    original = t["x"].clone()
    label = f"seam b{batch} d{dim} w{width} s{seqlen} spec={spec} inter={save_inter}"
    assert _causal_conv1d_update_sglang_flydsl_supported(
        t["x"],
        t["conv_state"],
        t["weight"],
        num_accept_tokens=t["num_accepted"],
        intermediate_conv_window=t["window"] if save_inter else None,
    ), f"{label}: expected this case to be in the port's scope"

    out_off, state_off, window_off, _ = _call_triton_entry(
        t, batch, seqlen, save_inter, flydsl=False
    )
    out_on, state_on, window_on, x_on = _call_triton_entry(
        t, batch, seqlen, save_inter, flydsl=True
    )

    assert out_on.data_ptr() == x_on.data_ptr(), (
        f"{label}: the seam broke the write-over-x contract -- callers that read "
        "x after the call would silently see stale activations"
    )
    if decode_2d:
        assert (
            out_on.shape == original.shape
        ), f"{label}: changed rank to {out_on.shape}"
        assert not torch.equal(x_on, original), f"{label}: x was never written"
        _assert_bit_exact(label, "2D output", out_on, out_off)
        _assert_bit_exact(label, "2D conv_state", state_on, state_off)
        return

    ref = _oracle_sglang(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        intermediate_conv_window=t["window"],
        intermediate_state_indices=t["indices"] if save_inter else None,
    )
    ref = ref._replace(spec=_torch_ref(t, "sglang_verify").double())
    _assert_matches_oracle(label, out_on, ref)
    _assert_bit_exact(label, "conv_state roll-back", state_on, ref.state)
    _assert_bit_exact(f"{label} (vs seam off)", "conv_state", state_on, state_off)
    if save_inter:
        _assert_bit_exact(label, "intermediate_conv_window", window_on, ref.window)
        _assert_bit_exact(
            f"{label} (vs seam off)",
            "intermediate_conv_window",
            window_on,
            window_off,
        )


def test_dispatch_seam_falls_through_when_out_of_scope():
    from aiter.ops.triton.conv.causal_conv1d import _flydsl_conv1d_update_enabled

    previous = os.environ.pop(_SEAM_ENV, None)
    try:
        assert not _flydsl_conv1d_update_enabled()
    finally:
        if previous is not None:
            os.environ[_SEAM_ENV] = previous

    t = _sglang_problem(8, 256, 4, 4, spec=True, save_inter=False, tree=False)
    t["x"] = t["x"].float()
    t["conv_state"] = t["conv_state"].float()
    t["weight"] = t["weight"].float()
    t["bias"] = t["bias"].float()

    assert not _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accept_tokens=t["num_accepted"]
    ), "fp32 is supposed to be outside the port's scope"

    out_off, state_off, _, _ = _call_triton_entry(t, 8, 4, False, flydsl=False)
    out_on, state_on, _, _ = _call_triton_entry(t, 8, 4, False, flydsl=True)
    _assert_bit_exact("seam fp32", "output (fell through)", out_on, out_off)
    _assert_bit_exact("seam fp32", "conv_state (fell through)", state_on, state_off)

    t = _sglang_problem(2, 64, 4, 1, spec=False, save_inter=False, tree=False)
    t["indices"] = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)
    assert not _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], conv_state_indices=t["indices"]
    )
    out_off, state_off, _, _ = _call_triton_entry(t, 2, 1, False, flydsl=False)
    out_on, state_on, _, _ = _call_triton_entry(t, 2, 1, False, flydsl=True)
    _assert_bit_exact("seam int64 index", "output", out_on, out_off)
    _assert_bit_exact("seam int64 index", "conv_state", state_on, state_off)


@pytest.mark.parametrize(
    "interface,name,dims",
    [
        ("sglang", "conv_state_indices", 1),
        ("sglang", "num_accept_tokens", 1),
        ("sglang", "intermediate_state_indices", 1),
        ("sglang", "retrieve_next_token", 2),
        ("sglang", "retrieve_next_sibling", 2),
        ("sglang", "retrieve_parent_token", 2),
        ("vllm", "conv_state_indices", 1),
        ("vllm", "num_accepted_tokens", 1),
        ("vllm", "query_start_loc", 1),
        ("vllm", "block_idx_last_scheduled_token", 1),
        ("vllm", "initial_state_idx", 1),
    ],
)
def test_flydsl_rejects_non_int32_indices(interface, name, dims):
    t = _make_inputs(2, 64, 4, 2, spec=True, seed=37)
    good = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    kwargs = {name: torch.zeros((2,) * dims, dtype=torch.int64, device=DEVICE)}
    if name == "query_start_loc":
        kwargs["conv_state_indices"] = good
    elif name == "block_idx_last_scheduled_token":
        kwargs["initial_state_idx"] = good
    elif name == "initial_state_idx":
        kwargs["block_idx_last_scheduled_token"] = good
    fn = (
        causal_conv1d_update_sglang_flydsl
        if interface == "sglang"
        else causal_conv1d_update_flydsl
    )
    with pytest.raises(ValueError, match=name):
        fn(t["x"], t["conv_state"], t["weight"], **kwargs)


_VARLEN_CASES = [
    (4, 3, True, (3, 1, 2, 3)),
    (4, 4, True, (4, 2, 1, 4, 3)),
    (4, 2, False, (2, 1, 2)),
    (4, 3, True, (3, 0, 2, 1)),
    (4, 4, True, (0, 4)),
]


def _pack(x_dense, lens):
    rows = [x_dense[i, :, : lens[i]].T for i in range(len(lens))]
    return torch.cat(rows, dim=0).contiguous()


def _varlen_problem(width, seqlen, spec, lens):
    batch, dim = len(lens), 128
    gen = torch.Generator(device=DEVICE).manual_seed(sum(lens) * 17 + width * 7 + batch)
    state_len = (width - 1 + (seqlen - 1)) if spec else (width - 1)
    qsl = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(lens, dtype=torch.int32, device=DEVICE).cumsum(0)
    return {
        "x_dense": torch.randn(
            (batch, dim, seqlen), generator=gen, device=DEVICE, dtype=DTYPE
        ),
        "conv_state": torch.randn(
            (batch + 1, dim, state_len), generator=gen, device=DEVICE, dtype=DTYPE
        ),
        "weight": torch.randn((dim, width), generator=gen, device=DEVICE, dtype=DTYPE),
        "bias": torch.randn((dim,), generator=gen, device=DEVICE, dtype=DTYPE),
        "indices": torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE),
        "nacc": (
            torch.tensor([max(1, n) for n in lens], dtype=torch.int32, device=DEVICE)
            if spec
            else None
        ),
        "qsl": qsl,
        "lens": lens,
    }


def _run_varlen(fn, t, seqlen):
    state = t["conv_state"].clone()
    out = fn(
        _pack(t["x_dense"], t["lens"]),
        state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["indices"],
        num_accepted_tokens=t["nacc"],
        query_start_loc=t["qsl"],
        max_query_len=seqlen,
    )
    return out, state


@pytest.mark.parametrize("width,seqlen,spec,lens", _VARLEN_CASES)
def test_vllm_varlen_matches_upstream_vllm(width, seqlen, spec, lens):
    t = _varlen_problem(width, seqlen, spec, lens)
    label = f"varlen w{width} s{seqlen} spec={spec} lens={lens}"

    got, got_state = _run_varlen(causal_conv1d_update_flydsl, t, seqlen)
    ref = _oracle_vllm(
        _pack(t["x_dense"], t["lens"]),
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["nacc"],
        query_start_loc=t["qsl"],
        max_query_len=seqlen,
    )
    ref = ref._replace(spec=_torch_ref(t, "vllm_varlen").double())

    _assert_bit_exact(label, "conv_state", got_state, ref.state)
    _assert_matches_oracle(label, got, ref)


# -- call-site input layouts ----------------------------------------------

_QKVZ_Z_RATIO = 0.5


def _alloc_x(layout, shape, gen, dtype=DTYPE):
    kw = {"generator": gen, "device": DEVICE, "dtype": dtype}
    if layout == "contiguous":
        return torch.randn(shape, **kw)
    if layout == "qkvz_slice":
        tokens, dim = shape
        z_size = max(1, int(dim * _QKVZ_Z_RATIO))
        return torch.randn((tokens, dim + z_size), **kw)[:, :dim]
    if layout == "verify_view":
        batch, dim, tokens = shape
        return torch.randn((batch, tokens, dim), **kw).transpose(1, 2)
    raise ValueError(f"unknown layout {layout!r}")


def _assert_layout(layout, view):
    assert view.stride(1) == 1, (
        f"{layout}: the channel axis is not contiguous (strides {view.stride()}), "
        "which is the one thing both upstream kernels assert about x"
    )
    tokens = view.shape[0] if view.dim() == 2 else view.shape[2]
    if layout != "contiguous" and tokens > 1:
        assert not view.is_contiguous(), (
            f"{layout}: came back contiguous (strides {view.stride()}), so the "
            "strided path is not exercised"
        )
    return view


def _assert_same_run(label, run, baseline, variant, result_names):
    expected = run(baseline)
    actual = run(variant)
    for what, got, want in zip(result_names, actual, expected):
        _assert_bit_exact(label, what, got, want)


BENCH_MODES = ("vllm_decode", "vllm_verify", "sglang_verify", "sglang_verify_tree")

_MODE_LAYOUTS = {
    "vllm_decode": ("contiguous", "qkvz_slice"),
    "vllm_verify": ("contiguous", "qkvz_slice"),
    "sglang_verify": ("verify_view",),
    "sglang_verify_tree": ("verify_view",),
}

_SGLANG_MODES = ("sglang_verify", "sglang_verify_tree")
BENCH_LAYOUTS = ("contiguous", "qkvz_slice", "verify_view")


def _bench_problem(batch, dim, width, seqlen, mode, layout, dtype):
    spec = mode == "vllm_verify"
    t = _make_inputs(
        batch, dim, width, seqlen, spec=spec, seed=batch * 31 + seqlen, dtype=dtype
    )
    gen = torch.Generator(device=DEVICE).manual_seed(batch * 31 + seqlen + 7)
    t["indices"] = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    call = {
        "weight": t["weight"],
        "bias": t["bias"],
        "activation": "silu",
        "conv_state_indices": t["indices"],
    }

    if mode in _SGLANG_MODES:
        x = _alloc_x(layout, (batch, dim, seqlen), gen, dtype)
        dense = x
        t["window"] = torch.randn(
            (t["conv_state"].shape[0], seqlen, dim, width - 1),
            generator=gen,
            device=DEVICE,
            dtype=dtype,
        )
        call["intermediate_conv_window"] = t["window"]
        if mode == "sglang_verify_tree":
            t["next_token"], t["next_sibling"] = _eagle_tree_links(batch, seqlen)
            call["retrieve_next_token"] = t["next_token"]
            call["retrieve_next_sibling"] = t["next_sibling"]
    elif mode == "vllm_decode":
        x = _alloc_x(layout, (batch, dim), gen, dtype)
        dense = x.unsqueeze(-1)
    else:
        x = _alloc_x(layout, (batch * seqlen, dim), gen, dtype)
        dense = x.unflatten(0, (batch, seqlen)).transpose(1, 2)
        call["num_accepted_tokens"] = t["num_accepted"]
        call["query_start_loc"] = torch.arange(
            0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=DEVICE
        )
        call["max_query_len"] = seqlen

    call["x"] = _assert_layout(layout, x)
    t["x"] = dense

    def new_x():
        dup = _alloc_x(layout, tuple(x.shape), gen, dtype)
        dup.copy_(x)
        return dup

    return t, call, new_x


def _bench_bytes(t, batch, dim, seqlen, width):
    esz = t["x"].element_size()
    state_len = t["conv_state"].shape[-1]
    elems = 2 * batch * seqlen * dim + 2 * batch * dim * state_len
    elems += t["weight"].numel() + (0 if t["bias"] is None else t["bias"].numel())
    if t.get("window") is not None:
        elems += batch * seqlen * dim * (width - 1)
    return elems * esz


@benchmark()
def test_causal_conv1d_update_perf(
    batch: int = 8,
    dim: int = 512,
    width: int = 4,
    seqlen: int = 1,
    mode: str = "vllm_decode",
    layout: str = "contiguous",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    t, call, new_x = _bench_problem(batch, dim, width, seqlen, mode, layout, dtype)
    label = f"{mode}/{layout} b{batch} d{dim} w{width} s{seqlen}"
    ref = _torch_ref(t, mode)

    if mode in _SGLANG_MODES:
        candidates = {
            "flydsl": (causal_conv1d_update_sglang_flydsl, {}),
            "sglang": (
                causal_conv1d_update_sglang_upstream,
                {"intermediate_state_indices": t["indices"]},
            ),
        }
        if mode == "sglang_verify":
            candidates["triton"] = (causal_conv1d_update_triton, {})
    else:
        candidates = {
            "flydsl": (causal_conv1d_update_flydsl, {}),
            "vllm": (causal_conv1d_update_vllm_upstream, {}),
        }
        if mode == "vllm_decode":
            candidates["triton"] = (causal_conv1d_update_triton, {})

    def fresh(extra):
        args = dict(call, x=new_x(), conv_state=t["conv_state"].clone(), **extra)
        if "intermediate_conv_window" in args:
            args["intermediate_conv_window"] = t["window"].clone()
        if "retrieve_next_token" in args:
            args["retrieve_parent_token"] = torch.zeros(
                (batch, seqlen), dtype=torch.int32, device=DEVICE
            )
        return args

    tokens = batch * seqlen
    flops = 2 * width * tokens * dim
    nbytes = _bench_bytes(t, batch, dim, seqlen, width)

    ret = {"gfx": get_gfx(), "dtype": _DTYPE_NAME[dtype]}
    for name, (fn, extra) in candidates.items():
        out, _ = run_perftest(
            fn,
            **fresh(extra),
            num_iters=1,
            num_warmup=0,
            num_rotate_args=1,
            use_cuda_event=True,
        )
        _, us = run_perftest(fn, **fresh(extra))
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: {label}",
        )
    return ret


test_causal_conv1d_update_perf.__test__ = False


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl causal_conv1d_update is unsupported on %s; skipping", gfx
        )
        return 0

    parser = argparse.ArgumentParser(
        description="FlyDSL causal_conv1d update correctness and perf"
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 32, 128])
    parser.add_argument("--dim", type=int, nargs="*", default=[1024, 2048, 4096, 8192])
    parser.add_argument("-w", "--width", type=int, nargs="*", default=[4])
    parser.add_argument("--seqlen", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,"
    )
    parser.add_argument(
        "--mode", type=str, nargs="*", default=list(BENCH_MODES), choices=BENCH_MODES
    )
    parser.add_argument(
        "--layout",
        type=str,
        nargs="*",
        default=list(BENCH_LAYOUTS),
        choices=BENCH_LAYOUTS,
    )
    args = parser.parse_args()

    aiter.logger.info("causal_conv1d_update: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "causal_conv1d_update: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1

    rows = []
    for mode, layout, dtype, width, seqlen, dim, batch in itertools.product(
        args.mode,
        args.layout,
        args.dtype,
        args.width,
        args.seqlen,
        args.dim,
        args.batch,
    ):
        if layout not in _MODE_LAYOUTS[mode] or (seqlen == 1) != (
            mode == "vllm_decode"
        ):
            continue
        rows.append(
            test_causal_conv1d_update_perf(
                batch, dim, width, seqlen, mode, layout, dtype
            )
        )
    aiter.logger.info(
        "flydsl causal_conv1d_update summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
