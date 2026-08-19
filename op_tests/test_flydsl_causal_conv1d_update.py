# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parity tests for the FlyDSL causal_conv1d update kernels (decode + verify).

Two kernels are under test, one per upstream interface. They are the same
algorithm behind two different shells, and both are maintained:

* ``causal_conv1d_update_flydsl`` -- vLLM's interface (skip sentinel
  ``null_block_id``, default ``0``).
* ``causal_conv1d_update_sglang_flydsl`` -- SGLang's interface (skip sentinel
  ``pad_slot_id``, default ``-1``), which adds per-step
  ``intermediate_conv_window`` snapshots and an EAGLE **tree** path where the
  convolution walks each candidate token's parent chain instead of its linear
  predecessor.

Both are measured against one high-precision torch spec (``run_torch``) that
covers every path including the packed layout, the prefix-caching copy, the tree
and the snapshots. The vLLM cases are additionally compared with aiter's own
Triton ``causal_conv1d_update``, and ``test_dispatch_seam_routes_to_flydsl``
exercises the opt-in seam that routes that Triton entry point to the FlyDSL port.

Nothing here imports ``vllm`` or ``sglang``. The upstream behaviour those
packages would have supplied is written into ``run_torch`` instead, against the
revisions named below, which is how the rest of aiter treats upstream references
(``op_tests/test_causal_conv1d_update.py`` carries Tri Dao's, and
``op_tests/triton_tests/utils/mla_extend_ref.py`` SGLang's). The cost of that
choice is real: a spec written from the same reading as the kernel cannot catch a
misreading of the contract, so the pieces that are transcribed rather than
derived are marked in place and carry the upstream revision they were read from.

How the numeric bound is set
----------------------------
A bf16 result is *not* required to agree bit-for-bit with any other bf16 kernel:
they round differently, and against fp64 the FlyDSL kernels are the more accurate
of the pair, so demanding agreement would pin the less accurate one's error.
Instead each element gets a budget derived from its own conditioning --
``factor * 2**-8 * sum_j |tap_j * w_j|`` -- because the error of a dot product is
bounded by the sum of the *term* magnitudes, not by the magnitude of the result.
A budget scaled to the result would be vacuous where the taps cancel and the
output lands near zero, which is exactly where a bf16 kernel is least accurate.
The spec returns that per-element sum alongside the output. For calibration, a
pinned copy of SGLang's own Triton kernel landed at 0.07-0.19x of this budget
across every case below, i.e. the factor of 8 leaves about 5x of headroom.

``conv_state``, the snapshots and the tree's parent map are compared **bit-exactly**
in every case: they are assembled from copies of ``x`` and of the previous state
with no arithmetic applied, so any difference there is a state-management bug
rather than rounding.

Sequence indices deliberately start at 1 in the vLLM cases: that wrapper's
``null_block_id`` sentinel is ``0`` (block 0 is the reserved null block), so a
plain ``arange(batch)`` would make sequence 0 a null block and silently skip it.
``test_vllm_skips_the_null_block`` covers that behavior on purpose.

Upstream revisions transcribed
------------------------------
vLLM ``63a9a5010`` and SGLang ``18107e38d2``, both read on 2026-08-14. Re-read
them before changing ``run_torch``: the packed layout's window revision and the
prefix-caching read/write split are transcriptions of those kernels, not
consequences of the convolution, so they cannot be re-derived from first
principles if they drift.

Usage:
    HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/test_flydsl_causal_conv1d_update.py

    # or, the way CI invokes it -- the same checks with no pytest dependency,
    # followed by the perf sweep:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_causal_conv1d_update.py

    # perf only, one mode:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_causal_conv1d_update.py \
        --run perf --mode varlen -b 128 -s 4
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL causal_conv1d update tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.causal_conv1d_update_kernels import (
        NULL_BLOCK_ID,
        _causal_conv1d_update_flydsl_supported,
        _causal_conv1d_update_sglang_flydsl_supported,
        causal_conv1d_update_flydsl,
        causal_conv1d_update_sglang_flydsl,
    )
    from aiter.ops.triton.conv.causal_conv1d import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import the FlyDSL causal_conv1d update kernels: {exc}",
        allow_module_level=True,
    )


DEVICE = "cuda"
DTYPE = torch.bfloat16

#: The kernels are built for CDNA3/CDNA4 and the launch policy is tuned on them.
SUPPORTED_GFX = ("gfx942", "gfx950")

#: One mantissa step of the storage dtype, the relative cost of storing any
#: single value. fp16 keeps three more significand bits than bf16, so it has to
#: be held to the tighter budget or its cases pass on bf16's slack.
_DTYPE_EPS = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}

#: How many of those steps a result may accumulate over the whole convolution,
#: scaled per element by the conditioning (see the module docstring). Covers the
#: width taps, the bias, the silu approximation and the store.
_ERR_FACTOR = 8.0


# -- inputs ---------------------------------------------------------------


def _make_inputs(batch, dim, width, seqlen, *, spec, seed, has_bias=True, dtype=DTYPE):
    """Random problem plus a spare cache line, so indices can start at 1."""
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
    """A left-deep candidate tree with one sibling branch per level.

    ``retrieve_next_token[b, i]`` is token ``i``'s first child and
    ``retrieve_next_sibling[b, i]`` its next sibling, matching SGLang's EAGLE
    layout. Shape matters more than realism here: what has to be exercised is a
    parent chain that is not the linear predecessor.
    """
    nxt = torch.full((batch, seqlen), -1, dtype=torch.int32, device=DEVICE)
    sib = torch.full((batch, seqlen), -1, dtype=torch.int32, device=DEVICE)
    for b in range(batch):
        for i in range(seqlen - 1):
            nxt[b, i] = i + 1
            if i + 2 < seqlen:
                sib[b, i + 1] = i + 2
    return nxt, sib


# -- high-precision spec --------------------------------------------------


def run_torch(
    x,
    conv_state,
    weight,
    bias,
    conv_state_indices,
    num_accepted,
    *,
    skip_line,
    lens=None,
    write_indices=None,
    window=None,
    inter_indices=None,
    next_token=None,
    next_sibling=None,
):
    """fp64 spec of the update, covering both upstreams' STEP 1-5.

    The reference for every case in this file, named for the aiter convention
    (``op_tests`` reference implementations are ``run_torch``) and kept at module
    level so other tests can import it.

    Returns ``(out, conv_state, window, parents, scale)``. Only ``out`` is fp64,
    as the yardstick for either bf16 kernel; the rolled state, the snapshots and
    the parent map are copies of input values, so they come back in their own
    dtype to be compared bit-exactly. ``scale`` is the per-element sum of the
    convolution's term magnitudes, which is what the error budget is scaled by.

    ``skip_line`` is the caller's sentinel: a sequence whose cache line equals it
    is left completely alone (vLLM spells it ``null_block_id`` and defaults to the
    valid index ``0``; SGLang spells it ``pad_slot_id`` and uses ``-1``).

    ``x`` stays dense ``(batch, dim, seqlen)`` even for the packed layout, so
    there is one spec and not two. ``lens`` then gives each sequence's real token
    count and ``seqlen`` is the budget the caller compiled for; only the first
    ``lens[i]`` columns of ``out`` are meaningful. ``write_indices`` splits the
    cache line the window is written to from the one the history is read from,
    which is what Automatic Prefix Caching does.

    The chain path accumulates as one fp64 reduction over the width taps rather
    than the kernel's sequential FMA chain, so agreement is a numerical
    cross-check and not a restatement of the implementation.
    """
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    is_spec = num_accepted is not None
    state_len = (width - 1 + (seqlen - 1)) if is_spec else (width - 1)
    tree = next_token is not None
    if inter_indices is None:
        inter_indices = conv_state_indices

    out = x.double().clone()
    scale = torch.zeros_like(out)
    new_state = conv_state.clone()
    new_window = window.clone() if window is not None else None
    parents = (
        torch.zeros((batch, seqlen), dtype=torch.int32, device=x.device)
        if tree
        else None
    )
    w = weight.double()
    b = (
        bias.double()
        if bias is not None
        else torch.zeros(dim, dtype=torch.float64, device=x.device)
    )

    for i in range(batch):
        line = int(conv_state_indices[i])
        if line >= conv_state.shape[0] or (skip_line is not None and line == skip_line):
            continue
        # Upstream revises both lengths by the padding this sequence did not use,
        # and returns before any store when it owns no tokens at all. Transcribed
        # from vLLM's kernel rather than derived: without a rollback point the
        # revision pulls the window below width - 1, which no dense call can be
        # asked to reproduce.
        tokens = seqlen if lens is None else int(lens[i])
        if tokens == 0:
            continue
        seq_state_len = state_len - (seqlen - tokens)
        val = seq_state_len - tokens
        write_line = line if write_indices is None else int(write_indices[i])
        win_line = int(inter_indices[i])
        raw = conv_state[line]
        state = raw.double()
        offset = (int(num_accepted[i]) - 1) if is_spec else 0
        # STEP 1: the width-1 history columns, oldest first (col0 .. col_{W-2}).
        cols = [state[:, offset + k] for k in range(width - 1)]
        cols_raw = [raw[:, offset + k] for k in range(width - 1)]

        if tree:
            for token in range(tokens):
                # STEP 3: the parent map is built in token order exactly as the
                # kernel does -- a child's parent is the current token, and a
                # sibling inherits the current token's parent.
                child = int(next_token[i, token])
                if child != -1:
                    parents[i, child] = token
                sibling = int(next_sibling[i, token])
                if sibling != -1:
                    parents[i, sibling] = parents[i, token]

                # STEP 5: convolve along the parent chain, newest tap first.
                cur = token
                xv, xv_raw = x[i, :, cur].double(), x[i, :, cur]
                acc, mag = b.clone(), b.abs().clone()
                for j in range(width):
                    if new_window is not None and (width - j - 2) >= 0:
                        new_window[win_line, token, :, width - j - 2] = xv_raw
                    term = xv * w[:, width - 1 - j]
                    acc, mag = acc + term, mag + term.abs()
                    if cur > 0:
                        cur = int(parents[i, cur])
                        xv, xv_raw = x[i, :, cur].double(), x[i, :, cur]
                    else:
                        # Walked off the chunk: keep going back through history.
                        k = max(0, width - 2 + cur)
                        xv, xv_raw = cols[k], cols_raw[k]
                        cur -= 1
                out[i, :, token] = acc * torch.sigmoid(acc)
                scale[i, :, token] = mag
        else:
            win, win_raw = list(cols), list(cols_raw)
            for token in range(tokens):
                taps = torch.stack(win + [x[i, :, token].double()], dim=1)
                acc = (taps * w).sum(dim=1) + b
                out[i, :, token] = acc * torch.sigmoid(acc)
                scale[i, :, token] = (taps * w).abs().sum(dim=1) + b.abs()
                # The register window shifts, then the snapshot records it.
                win = win[1:] + [x[i, :, token].double()]
                win_raw = win_raw[1:] + [x[i, :, token]]
                if new_window is not None:
                    for k in range(width - 1):
                        new_window[win_line, token, :, k] = win_raw[k]

        # STEP 2: slide the stored window and blend in the new tokens. The slide
        # is one column when a speculative rollback point is in play and the
        # whole chunk otherwise, which only differ once 1 < tokens < width - 1.
        # Only the revised window is written, so whatever the destination line
        # held past it stays -- that is upstream's mask, not an omission here.
        shift = 1 if is_spec else tokens
        rolled = [
            (
                raw[:, offset + shift + t]
                if (t + tokens) < seq_state_len
                else x[i, :, t - val]
            )
            for t in range(seq_state_len)
        ]
        new_state[write_line, :, :seq_state_len] = torch.stack(rolled, dim=1)

    return out, new_state, new_window, parents, scale


# -- shared assertions ----------------------------------------------------


def _assert_tracks_spec(label, actual, spec, scale, factor=_ERR_FACTOR):
    """``actual`` must sit inside the conditioning-scaled error budget."""
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


def _assert_no_worse_than(label, flydsl, other, spec, scale, other_name):
    """FlyDSL must track the spec at least as closely as ``other`` does.

    The two backends round differently, so requiring them to agree would pin the
    less accurate one's error. One step of the element's own conditioning is
    allowed as slack, so a rounding tie cannot fail this while a real regression
    still does.
    """
    err_fly = (flydsl.double() - spec).abs()
    err_other = (other.double() - spec).abs()
    slack = _DTYPE_EPS[flydsl.dtype] * scale
    regressed = err_fly > err_other + slack
    assert not regressed.any(), (
        f"{label}: FlyDSL is further from the spec than {other_name} on "
        f"{int(regressed.sum())} of {regressed.numel()} elements "
        f"(max excess {(err_fly - err_other - slack).max().item():.3e})"
    )


def _assert_bit_exact(label, what, actual, expected):
    assert torch.equal(actual, expected), (
        f"{label}: {what} is not bit-exact "
        f"(max|delta|={(actual.float() - expected.float()).abs().max():.3e}); "
        "it is assembled from copies, never computed, so this is not rounding"
    )


# -- vLLM interface -------------------------------------------------------

# (batch, dim, width, seqlen, spec); spec=True enables the chain rollback.
_VLLM_CASES = [
    (1, 128, 4, 1, False),
    (4, 256, 4, 1, False),
    (64, 512, 4, 1, False),
    (2, 256, 3, 1, False),
    # Multi-token without a rollback point, in the one band where the slide
    # length is observable: 1 < seqlen < width - 1 keeps part of the rolled
    # window coming from conv_state, and it then slides by the chunk length
    # rather than by one column. At seqlen >= width - 1 every slot comes from x
    # and the distinction vanishes, which is why width 4 / seqlen 2 is the only
    # cell of this kind the Triton oracle (widths 2-4) can witness.
    (4, 128, 4, 2, False),
    (8, 256, 4, 2, True),
    (32, 384, 4, 4, True),
    (16, 256, 4, 8, True),
    (8, 256, 3, 4, True),
]

#: Every shape in bf16, plus fp16 on a representative few. fp16 is the same
#: algorithm but a separately compiled specialization, so it needs its own cases;
#: these four span both widths, both rollback modes and the narrow-window band
#: without doubling the table, and they are what would catch the error budget
#: being left at bf16's three-bits-looser step.
_VLLM_FP16_CASES = [
    (4, 256, 4, 1, False),
    (4, 128, 4, 2, False),
    (32, 384, 4, 4, True),
    (8, 256, 3, 4, True),
]


def _vllm_param(batch, dim, width, seqlen, spec, dtype):
    """Spell the dtype in the test id; pytest would render it as ``dtype3``."""
    return pytest.param(
        batch,
        dim,
        width,
        seqlen,
        spec,
        dtype,
        id=f"b{batch}-d{dim}-w{width}-s{seqlen}"
        f"-{'spec' if spec else 'nospec'}-{_DTYPE_NAME[dtype]}",
    )


_VLLM_DTYPE_CASES = [_vllm_param(*c, torch.bfloat16) for c in _VLLM_CASES] + [
    _vllm_param(*c, torch.float16) for c in _VLLM_FP16_CASES
]


@pytest.mark.parametrize("batch,dim,width,seqlen,spec,dtype", _VLLM_DTYPE_CASES)
def test_vllm_against_spec_and_aiter_triton(batch, dim, width, seqlen, spec, dtype):
    """Drop-in for aiter's Triton causal_conv1d_update, and at least as accurate.

    Checks three things at once: the output stays inside its error budget, it is
    no further from the spec than the Triton kernel is, and the ``conv_state``
    roll-back matches both the spec and Triton bit-exactly.
    """
    t = _make_inputs(
        batch, dim, width, seqlen, spec=spec, seed=batch * 100 + seqlen, dtype=dtype
    )
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    label = f"vllm b{batch} d{dim} w{width} s{seqlen} spec={spec} {_DTYPE_NAME[dtype]}"

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

    # The Triton entry point spells the sentinel `pad_slot_id` and defaults it to
    # -1; the indices above avoid both sentinels either way.
    state_tri = t["conv_state"].clone()
    out_tri = causal_conv1d_update_triton(
        t["x"].clone(),
        state_tri,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=indices,
        num_accepted_tokens=t["num_accepted"],
    )

    out_spec, state_spec, _, _, scale = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        indices,
        t["num_accepted"],
        skip_line=0,
    )

    _assert_tracks_spec(label, out_fly, out_spec, scale)
    _assert_no_worse_than(
        label, out_fly, out_tri, out_spec, scale, "aiter's Triton kernel"
    )
    _assert_bit_exact(
        f"{label} (vs spec)", "conv_state roll-back", state_fly, state_spec
    )
    _assert_bit_exact(
        f"{label} (vs Triton)", "conv_state roll-back", state_fly, state_tri
    )


@pytest.mark.parametrize("channels_per_thread", [1, 2])
def test_vllm_channels_per_thread_agree(channels_per_thread):
    """CPT is an occupancy knob only: both mappings must produce one answer."""
    t = _make_inputs(32, 512, 4, 4, spec=True, seed=99)
    indices = torch.arange(1, 33, dtype=torch.int32, device=DEVICE)

    states, outs = [], []
    for cpt in (1, channels_per_thread):
        state = t["conv_state"].clone()
        outs.append(
            causal_conv1d_update_flydsl(
                t["x"].clone(),
                state,
                t["weight"],
                bias=t["bias"],
                activation="silu",
                conv_state_indices=indices,
                num_accepted_tokens=t["num_accepted"],
                channels_per_thread=cpt,
            )
        )
        states.append(state)

    assert torch.equal(outs[0], outs[1]), "CPT changed the output"
    _assert_bit_exact(
        f"cpt={channels_per_thread}", "conv_state roll-back", states[0], states[1]
    )


def test_cpt_policy_backs_off_for_long_speculative_windows():
    """The channels-per-thread default has to track S, not just the batch.

    Paired A/B measured +4.1% at S=4 but -2.4% at S=8, and the EAGLE tree path
    runs at S=8, so a policy that only looked at the batch would hand the tree a
    knob setting known to cost it time. Pinned here because the reversal is not
    something the code can derive.
    """
    from aiter.ops.flydsl.causal_conv1d_update_kernels import _CPT_MAX, _pick_cpt

    device = torch.device(DEVICE)
    big_enough = 256  # batch past the occupancy target at dim=4096

    assert _pick_cpt(big_enough, 4096, device, seqlen=4) == _CPT_MAX
    assert _pick_cpt(big_enough, 4096, device, seqlen=5) == 1
    assert _pick_cpt(big_enough, 4096, device, seqlen=8) == 1
    # A small batch stays at 1 whatever the window: it is occupancy-bound first.
    assert _pick_cpt(1, 256, device, seqlen=1) == 1


def test_vllm_skips_the_null_block():
    """A sequence pointing at the null block keeps its output and cache line.

    This is the sentinel trap the wrapper's module docstring warns about, pinned
    as behavior: with ``null_block_id=0``, cache line 0 is skipped.
    """
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


def test_vllm_out_parameter_leaves_x_untouched():
    """With ``out=``, ``x`` survives; without it, upstream overwrites ``x``."""
    t = _make_inputs(4, 256, 4, 2, spec=True, seed=13)
    indices = torch.arange(1, 5, dtype=torch.int32, device=DEVICE)
    x_in = t["x"].clone()

    out = torch.empty_like(x_in)
    state = t["conv_state"].clone()
    returned = causal_conv1d_update_flydsl(
        x_in,
        state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=indices,
        num_accepted_tokens=t["num_accepted"],
        out=out,
    )

    assert torch.equal(x_in, t["x"]), "`out=` was given but `x` was overwritten"
    assert torch.equal(returned, out), "the returned tensor is not `out`"


# -- SGLang interface -----------------------------------------------------

# (batch, dim, width, seqlen, spec, save_intermediate, tree)
_SGLANG_CASES = [
    (4, 256, 4, 1, False, False, False),
    (1, 128, 3, 1, False, False, False),
    (32, 384, 4, 4, True, False, False),
    (16, 256, 4, 8, True, False, False),
    (8, 256, 4, 4, True, True, False),
    (8, 256, 4, 8, True, False, True),
    (4, 128, 4, 4, True, True, True),
    (4, 128, 2, 4, True, True, True),
    (4, 128, 3, 4, True, True, True),
    # Tree links with `num_accept_tokens` left out. This is the shape SGLang's
    # own target-verify sites send (gdn / kda / mamba2 all pass the tree and the
    # snapshot but no accept count), and it is a distinct path: the accept count
    # is what turns on the rollback and the wider state, so upstream and this
    # port both fall to `offset = 0` and `state_len = width - 1` here and let the
    # parent links carry the chain instead.
    (8, 256, 4, 8, False, True, True),
    (4, 128, 3, 4, False, False, True),
]


def _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree):
    t = _make_inputs(batch, dim, width, seqlen, spec=spec, seed=batch * 31 + seqlen)
    t["indices"] = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    t["window"] = None
    if save_inter:
        gen = torch.Generator(device=DEVICE).manual_seed(5)
        t["window"] = torch.randn(
            (t["conv_state"].shape[0], seqlen, dim, width - 1),
            generator=gen,
            device=DEVICE,
            dtype=DTYPE,
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


@pytest.mark.parametrize("batch,dim,width,seqlen,spec,save_inter,tree", _SGLANG_CASES)
def test_sglang_against_spec(batch, dim, width, seqlen, spec, save_inter, tree):
    """Covers SGLang's decode, chain-verify, SAVE_INTERMEDIATE and tree paths."""
    t = _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree)
    label = (
        f"sglang b{batch} d{dim} w{width} s{seqlen} spec={spec} "
        f"inter={save_inter} tree={tree}"
    )

    out, state, window, parents = _run_sglang(
        causal_conv1d_update_sglang_flydsl, t, batch, seqlen, tree, save_inter
    )
    out_spec, state_spec, window_spec, parents_spec, scale = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        skip_line=-1,
        window=t["window"],
        inter_indices=t["indices"] if save_inter else None,
        next_token=t["next_token"],
        next_sibling=t["next_sibling"],
    )

    _assert_tracks_spec(label, out, out_spec, scale)
    _assert_bit_exact(label, "conv_state roll-back", state, state_spec)
    if save_inter:
        _assert_bit_exact(label, "intermediate_conv_window", window, window_spec)
    if tree:
        _assert_bit_exact(label, "retrieve_parent_token map", parents, parents_spec)


# -- the aiter dispatch seam ----------------------------------------------


_SEAM_ENV = "AITER_CONV1D_UPDATE_FLYDSL"


def _call_triton_entry(t, batch, seqlen, save_inter, *, flydsl):
    """aiter's Triton entry point, with the FlyDSL seam on or off."""
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


# (batch, dim, width, seqlen, spec, save_intermediate); the seam has no tree path.
_SEAM_CASES = [
    (8, 256, 4, 1, False, False),
    (8, 256, 4, 4, True, False),
    (8, 256, 4, 4, True, True),
    (8, 256, 3, 4, True, False),
]


@pytest.mark.parametrize("batch,dim,width,seqlen,spec,save_inter", _SEAM_CASES)
def test_dispatch_seam_routes_to_flydsl(batch, dim, width, seqlen, spec, save_inter):
    """``AITER_CONV1D_UPDATE_FLYDSL=1`` must be a transparent swap.

    The seam has to preserve this entry point's contract, which is not the same as
    SGLang's: the output is written *over* ``x`` and returned.
    """
    t = _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree=False)
    label = f"seam b{batch} d{dim} w{width} s{seqlen} spec={spec} inter={save_inter}"
    assert _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accept_tokens=t["num_accepted"]
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

    _, _, _, _, scale = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        skip_line=-1,
    )
    out_spec, state_spec, window_spec, _, _ = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        skip_line=-1,
        window=t["window"],
        inter_indices=t["indices"] if save_inter else None,
    )
    _assert_tracks_spec(label, out_on, out_spec, scale)
    _assert_no_worse_than(
        label, out_on, out_off, out_spec, scale, "the Triton path (seam off)"
    )
    _assert_bit_exact(label, "conv_state roll-back", state_on, state_spec)
    _assert_bit_exact(f"{label} (vs seam off)", "conv_state", state_on, state_off)
    if save_inter:
        _assert_bit_exact(label, "intermediate_conv_window", window_on, window_spec)
        _assert_bit_exact(
            f"{label} (vs seam off)",
            "intermediate_conv_window",
            window_on,
            window_off,
        )


def test_dispatch_seam_falls_through_when_out_of_scope():
    """A problem the port does not cover must come back on the Triton path.

    fp32 is the trigger here: the Triton kernel is dtype-generic, while the port
    only specializes bf16 and fp16. With the seam on, the answer therefore has to
    be the Triton one, bit for bit.
    """
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


def test_triton_update_refuses_unimplemented_width():
    """Widths past 4 must fail loudly on the Triton path, as they do on HIP.

    That kernel loads a 4th history column at width 5 but never loads a 5th weight
    column, and its tap loop only branches on 2/3/4, so before this guard it
    returned a silently wrong result. The FlyDSL vLLM-interface kernel does
    implement widths up to 6, which is why only the Triton side is refused here.
    """
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=31)
    wide = torch.randn((256, 5), device=DEVICE, dtype=DTYPE)
    state = torch.randn((5, 256, 4), device=DEVICE, dtype=DTYPE)
    indices = torch.arange(4, dtype=torch.int32, device=DEVICE)

    with pytest.raises(AssertionError, match="width must be 2, 3 or 4"):
        causal_conv1d_update_triton(
            t["x"].clone(), state, wide, activation="silu", conv_state_indices=indices
        )
    # The FlyDSL kernel covers this width, so it must not have been made to refuse.
    assert _causal_conv1d_update_flydsl_supported(t["x"], state, wide)


def test_dispatch_seam_preserves_2d_decode_shape():
    """The production decode shape is ``(batch, dim)``, with no token axis.

    The seam returns before this entry point's own 2D handling, so the wrapper's
    squeeze has to restore the caller's rank -- and still write through to ``x``,
    which is a view of the buffer the kernel wrote.
    """
    t = _sglang_problem(8, 256, 4, 1, spec=False, save_inter=False, tree=False)
    t["x"] = t["x"].squeeze(-1)
    original = t["x"].clone()

    out_off, state_off, _, _ = _call_triton_entry(t, 8, 1, False, flydsl=False)
    out_on, state_on, _, x_on = _call_triton_entry(t, 8, 1, False, flydsl=True)

    assert out_on.shape == original.shape, f"seam changed the rank: {out_on.shape}"
    assert out_on.data_ptr() == x_on.data_ptr(), "the 2D path stopped writing over x"
    assert not torch.equal(x_on, original), "x was never written"
    _assert_bit_exact("seam 2D", "output", out_on, out_off)
    _assert_bit_exact("seam 2D", "conv_state", state_on, state_off)


def test_dispatch_seam_is_off_by_default():
    """No env var, no behavior change: the default path must remain Triton."""
    from aiter.ops.triton.conv.causal_conv1d import _flydsl_conv1d_update_enabled

    previous = os.environ.pop(_SEAM_ENV, None)
    try:
        assert not _flydsl_conv1d_update_enabled()
    finally:
        if previous is not None:
            os.environ[_SEAM_ENV] = previous


# -- support predicates ---------------------------------------------------


def test_predicates_accept_the_supported_slice():
    t = _make_inputs(4, 256, 4, 4, spec=True, seed=17)
    assert _causal_conv1d_update_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accepted_tokens=t["num_accepted"]
    )
    assert _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accept_tokens=t["num_accepted"]
    )


def test_vllm_predicate_accepts_varlen_and_apc():
    """Both of vLLM's optional modes are in scope now, so neither may be refused."""
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=19)
    marker = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    packed = t["x"].squeeze(-1) if t["x"].dim() == 3 else t["x"]

    assert _causal_conv1d_update_flydsl_supported(
        t["x"],
        t["conv_state"],
        t["weight"],
        block_idx_last_scheduled_token=marker,
        initial_state_idx=marker,
    )
    assert _causal_conv1d_update_flydsl_supported(
        packed,
        t["conv_state"],
        t["weight"],
        query_start_loc=torch.zeros(5, dtype=torch.int32, device=DEVICE),
        max_query_len=1,
    )
    # Packing without the token budget cannot be sized, so it stays out of scope.
    assert not _causal_conv1d_update_flydsl_supported(
        packed,
        t["conv_state"],
        t["weight"],
        query_start_loc=torch.zeros(5, dtype=torch.int32, device=DEVICE),
    )


def test_vllm_varlen_needs_indices_and_a_token_budget():
    """Packed x hides the batch and the per-sequence budget; both must be given."""
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=31)
    packed = t["x"].squeeze(-1)
    qsl = torch.arange(5, dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError):
        causal_conv1d_update_flydsl(
            packed, t["conv_state"], t["weight"], query_start_loc=qsl, max_query_len=1
        )
    with pytest.raises(ValueError):
        causal_conv1d_update_flydsl(
            packed,
            t["conv_state"],
            t["weight"],
            conv_state_indices=torch.arange(4, dtype=torch.int32, device=DEVICE),
            query_start_loc=qsl,
        )


def test_vllm_apc_needs_both_index_tensors():
    """Upstream keys APC off one tensor then dereferences the other regardless."""
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=29)
    marker = torch.zeros(4, dtype=torch.int32, device=DEVICE)

    indices = torch.arange(4, dtype=torch.int32, device=DEVICE).unsqueeze(1)
    with pytest.raises(ValueError):
        causal_conv1d_update_flydsl(
            t["x"],
            t["conv_state"],
            t["weight"],
            conv_state_indices=indices,
            block_idx_last_scheduled_token=marker,
        )


#: (width, max_query_len, spec, per-sequence token counts).
_VARLEN_CASES = [
    (4, 1, False, (1, 1, 1, 1)),
    (4, 3, True, (3, 1, 2, 3)),
    (4, 4, True, (4, 2, 1, 4, 3)),
    (3, 2, True, (2, 1)),
    # 1 < seqlen < width - 1 again, this time reached through the packed layout.
    (4, 2, False, (2, 1, 2)),
    # A slot with no tokens at all: upstream returns before writing anything, so
    # its cache line has to come back untouched.
    (4, 3, True, (3, 0, 2, 1)),
    (4, 4, True, (0, 4)),
]


#: The subset whose sequences are each equivalent to a dense problem. With a
#: rollback point the window is history + accepted drafts, whose length tracks the
#: sequence, and at max_query_len == 1 there is no padding to revise away. Without
#: either, upstream narrows a short sequence's window below width - 1, which the
#: dense entry point cannot express, so only the spec reaches those.
_VARLEN_DENSE_EQUIVALENT = [c for c in _VARLEN_CASES if c[2] or c[1] == 1]


def _pack(x_dense, lens):
    """Dense ``(batch, dim, seqlen)`` -> vLLM's packed ``(cu_tokens, dim)``."""
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
        # An empty slot never reaches the accept count upstream, so any value in
        # it is legal; 1 keeps it inside the tensor's own range.
        "nacc": (
            torch.tensor([max(1, n) for n in lens], dtype=torch.int32, device=DEVICE)
            if spec
            else None
        ),
        "qsl": qsl,
        "lens": lens,
    }


def _run_varlen(fn, t, seqlen, **extra):
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
        **extra,
    )
    return out, state


@pytest.mark.parametrize("width,seqlen,spec,lens", _VARLEN_DENSE_EQUIVALENT)
def test_vllm_varlen_matches_the_dense_path(width, seqlen, spec, lens):
    """Packing is a layout change, not a different computation.

    For these cases each packed sequence is the dense problem at its own length,
    so run every sequence on its own through the dense path -- already pinned
    against Triton and the fp64 spec -- and require the packed batch to reproduce
    it bit for bit. An empty slot must come back untouched, which is what upstream
    does by returning before any store.
    """
    t = _varlen_problem(width, seqlen, spec, lens)
    label = f"varlen w{width} s{seqlen} spec={spec} lens={lens}"
    packed_out, packed_state = _run_varlen(causal_conv1d_update_flydsl, t, seqlen)

    for i, n in enumerate(lens):
        line = int(t["indices"][i])
        if n == 0:
            _assert_bit_exact(
                label,
                f"seq {i} empty slot untouched",
                packed_state[line],
                t["conv_state"][line],
            )
            continue
        one_state = t["conv_state"][
            line : line + 1, :, : (width - 1 + (n - 1)) if spec else (width - 1)
        ].clone()
        one_out = causal_conv1d_update_flydsl(
            t["x_dense"][i : i + 1, :, :n].clone(),
            one_state,
            t["weight"],
            bias=t["bias"],
            activation="silu",
            conv_state_indices=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            num_accepted_tokens=None if t["nacc"] is None else t["nacc"][i : i + 1],
            null_block_id=None,
        )
        got = packed_out[int(t["qsl"][i]) : int(t["qsl"][i]) + n].T.unsqueeze(0)
        _assert_bit_exact(label, f"seq {i} output", got.contiguous(), one_out)
        _assert_bit_exact(
            label,
            f"seq {i} rolled window",
            packed_state[line, :, : one_state.shape[2]],
            one_state[0],
        )


@pytest.mark.parametrize("width,seqlen,spec,lens", _VARLEN_CASES)
def test_vllm_varlen_matches_the_spec(width, seqlen, spec, lens):
    """Every packed case against the spec, narrowed windows included.

    The test above only reaches the cases each of whose sequences is a dense
    problem. This one reaches all of them, because the spec is handed the
    per-sequence token counts and performs upstream's window revision itself
    rather than inferring it, which is the part the dense entry point cannot
    express. ``conv_state`` must match bit for bit -- it is copies all the way
    down, including the slots past a narrowed window that must keep whatever the
    cache line already held.
    """
    t = _varlen_problem(width, seqlen, spec, lens)
    label = f"varlen-spec w{width} s{seqlen} spec={spec} lens={lens}"

    got, got_state = _run_varlen(causal_conv1d_update_flydsl, t, seqlen)
    out_spec, state_spec, _, _, scale = run_torch(
        t["x_dense"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["nacc"],
        skip_line=NULL_BLOCK_ID,
        lens=lens,
    )

    _assert_bit_exact(label, "conv_state", got_state, state_spec)
    for i, n in enumerate(lens):
        if n == 0:
            continue
        start = int(t["qsl"][i])
        _assert_tracks_spec(
            f"{label} seq {i}",
            got[start : start + n].T.contiguous(),
            out_spec[i, :, :n],
            scale[i, :, :n],
        )


def _with_apc_blocks(t):
    """Give a varlen problem two disjoint cache blocks per sequence."""
    batch = len(t["lens"])
    gen = torch.Generator(device=DEVICE).manual_seed(batch * 13 + 5)
    state_len = t["conv_state"].shape[-1]
    dim = t["conv_state"].shape[1]
    t["conv_state"] = torch.randn(
        (2 * batch + 1, dim, state_len), generator=gen, device=DEVICE, dtype=DTYPE
    )
    blocks = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    t["read_blocks"] = 1 + 2 * blocks
    t["write_blocks"] = 2 + 2 * blocks
    t["indices"] = torch.stack((t["read_blocks"], t["write_blocks"]), dim=1)
    t["init_col"] = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    t["last_col"] = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    return t


@pytest.mark.parametrize("width,seqlen,spec,lens", _VARLEN_CASES)
def test_vllm_varlen_with_apc_copies_instead_of_clobbering(width, seqlen, spec, lens):
    """The two optional modes compose, which is how vLLM's MTP sites call this.

    Packing decides which tokens a sequence owns and APC decides which cache
    block its window lands on; neither should disturb the other, so the packed run
    with APC must reproduce the packed run reading the same blocks without it, and
    the block the history came from must survive. Only the slots a sequence
    actually owns are compared: packing narrows a short sequence's window, and
    whatever the destination block held beyond it is left in place -- which the
    last assertion pins.
    """
    t = _with_apc_blocks(_varlen_problem(width, seqlen, spec, lens))
    label = f"varlen+apc w{width} s{seqlen} spec={spec} lens={lens}"
    original = t["conv_state"].clone()
    val = original.shape[-1] - seqlen  # slots still fed by conv_state

    plain_out, plain_state = _run_varlen(
        causal_conv1d_update_flydsl, {**t, "indices": t["read_blocks"]}, seqlen
    )
    apc_out, apc_state = _run_varlen(
        causal_conv1d_update_flydsl,
        t,
        seqlen,
        block_idx_last_scheduled_token=t["last_col"],
        initial_state_idx=t["init_col"],
    )

    _assert_bit_exact(label, "output", apc_out, plain_out)
    for i, n in enumerate(lens):
        read, write = int(t["read_blocks"][i]), int(t["write_blocks"][i])
        owned = val + n if n else 0
        _assert_bit_exact(
            label,
            f"seq {i} rolled window on the scheduled block",
            apc_state[write, :, :owned],
            plain_state[read, :, :owned],
        )
        _assert_bit_exact(
            label,
            f"seq {i} slots beyond the narrowed window",
            apc_state[write, :, owned:],
            original[write, :, owned:],
        )
        _assert_bit_exact(
            label, f"seq {i} source block left alone", apc_state[read], original[read]
        )


def _channel_major(x):
    """``mamba_mixer``'s layout: tokens adjacent, one channel a token count apart.

    Its decode path projects into a channel-major buffer and hands over
    ``hidden_states_BC_d.transpose(0, 1)`` without making it contiguous.
    """
    return x.t().contiguous().t()


def _column_slice(x, lead=3, trail=5):
    """``mamba_mixer2``'s layout: a channel window of a wider fused projection.

    Its input is a column slice of the packed QKV-style projection, so a token's
    channels are adjacent but consecutive tokens are the whole projection apart.
    """
    tokens, dim = x.shape
    wide = torch.empty(
        (tokens, lead + dim + trail), device=x.device, dtype=x.dtype
    ).normal_()
    view = wide[:, lead : lead + dim]
    view.copy_(x)
    return view


def _assert_strided(view, source):
    """Guard the guard: a relayout that came back contiguous tests nothing."""
    assert not view.is_contiguous() and torch.equal(view, source), (
        f"relayout produced a {'contiguous' if view.is_contiguous() else 'differing'} "
        f"tensor (strides {view.stride()}), so the strided path is not exercised"
    )
    return view


@pytest.mark.parametrize("relayout", [_channel_major, _column_slice])
def test_vllm_reads_x_through_its_strides(relayout):
    """Neither mixer makes ``x`` contiguous, so both layouts must give one answer.

    The kernel addresses ``x`` through the strides it is handed, and a wrong
    assumption there is invisible on a contiguous tensor -- both call sites pass
    a view. Same values, different layout, so this is bit-exact rather than a
    tolerance: dense decode covers ``mamba_mixer`` and packed varlen covers
    ``mamba_mixer2``.
    """
    label = f"strided x via {relayout.__name__}"

    # Dense decode, x as (batch, dim).
    t = _make_inputs(8, 256, 4, 1, spec=False, seed=4242)
    flat = t["x"].squeeze(-1)
    indices = torch.arange(1, 9, dtype=torch.int32, device=DEVICE)
    runs = []
    for variant in (flat, _assert_strided(relayout(flat), flat)):
        state = t["conv_state"].clone()
        out = causal_conv1d_update_flydsl(
            variant,
            state,
            t["weight"],
            bias=t["bias"],
            activation="silu",
            conv_state_indices=indices,
        )
        runs.append((out, state))
    _assert_bit_exact(f"{label} (dense)", "output", runs[1][0], runs[0][0])
    _assert_bit_exact(f"{label} (dense)", "conv_state", runs[1][1], runs[0][1])

    # Packed varlen, x as (cu_tokens, dim).
    v = _varlen_problem(4, 4, True, (4, 2, 1, 4))
    packed = _pack(v["x_dense"], v["lens"])
    runs = []
    for variant in (packed, _assert_strided(relayout(packed), packed)):
        state = v["conv_state"].clone()
        out = causal_conv1d_update_flydsl(
            variant,
            state,
            v["weight"],
            bias=v["bias"],
            activation="silu",
            conv_state_indices=v["indices"],
            num_accepted_tokens=v["nacc"],
            query_start_loc=v["qsl"],
            max_query_len=4,
        )
        runs.append((out, state))
    _assert_bit_exact(f"{label} (varlen)", "output", runs[1][0], runs[0][0])
    _assert_bit_exact(f"{label} (varlen)", "conv_state", runs[1][1], runs[0][1])


#: (batch, dim, width, seqlen, spec). The first is vLLM's decode + APC call
#: (``mamba_mixer``), the second its speculative one (``mamba_mixer2``) minus the
#: varlen packing that site also uses.
_APC_CASES = [
    (4, 256, 4, 1, False),
    (8, 256, 4, 4, True),
    (4, 128, 3, 1, False),
    (2, 128, 2, 2, True),
]


def _apc_problem(batch, dim, width, seqlen, spec, seed):
    """An APC problem whose read and write blocks are disjoint.

    Two cache lines per sequence, laid out so nothing aliases, and line 0 left
    unused because the default ``null_block_id`` is the valid index ``0``.
    ``conv_state_indices`` becomes ``(batch, 2)``: column 0 is the block the
    state was last computed into, column 1 the block scheduled for this step.
    """
    t = _make_inputs(batch, dim, width, seqlen, spec=spec, seed=seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    state_len = t["conv_state"].shape[-1]
    t["conv_state"] = torch.randn(
        (2 * batch + 1, dim, state_len), generator=gen, device=DEVICE, dtype=DTYPE
    )
    blocks = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    t["read_blocks"] = 1 + 2 * blocks
    t["write_blocks"] = 2 + 2 * blocks
    t["indices2d"] = torch.stack((t["read_blocks"], t["write_blocks"]), dim=1)
    t["init_col"] = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    t["last_col"] = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    return t


@pytest.mark.parametrize("batch,dim,width,seqlen,spec", _APC_CASES)
def test_vllm_apc_copies_instead_of_clobbering(batch, dim, width, seqlen, spec):
    """APC only moves where the state is read from and written to.

    So the output must be bit-identical to the plain path reading the same block,
    the rolled window must land on the scheduled block, and the block it was read
    from must come back untouched -- that last part is the whole point of the
    mode, and the one a wrong write address would break silently.
    """
    t = _apc_problem(batch, dim, width, seqlen, spec, seed=batch * 41 + seqlen)
    label = f"apc b{batch} d{dim} w{width} s{seqlen} spec={spec}"

    plain_state = t["conv_state"].clone()
    plain_out = causal_conv1d_update_flydsl(
        t["x"].clone(),
        plain_state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["read_blocks"],
        num_accepted_tokens=t["num_accepted"],
    )

    apc_state = t["conv_state"].clone()
    apc_out = causal_conv1d_update_flydsl(
        t["x"].clone(),
        apc_state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["indices2d"],
        num_accepted_tokens=t["num_accepted"],
        block_idx_last_scheduled_token=t["last_col"],
        initial_state_idx=t["init_col"],
    )

    _assert_bit_exact(label, "output", apc_out, plain_out)
    _assert_bit_exact(
        label,
        "rolled window on the scheduled block",
        apc_state[t["write_blocks"]],
        plain_state[t["read_blocks"]],
    )
    _assert_bit_exact(
        label,
        "source block left alone",
        apc_state[t["read_blocks"]],
        t["conv_state"][t["read_blocks"]],
    )

    # The three assertions above are differential: they would all still hold if
    # both runs read and wrote the wrong block in the same way. The spec pins the
    # absolute addresses, so it is what fails if the pair of them agree on a
    # wrong one.
    out_spec, state_spec, _, _, scale = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["read_blocks"],
        t["num_accepted"],
        skip_line=NULL_BLOCK_ID,
        write_indices=t["write_blocks"],
    )
    _assert_tracks_spec(label, apc_out, out_spec, scale)
    _assert_bit_exact(label, "conv_state against the spec", apc_state, state_spec)


def test_vllm_apc_reads_the_selected_column():
    """The read column is ``initial_state_idx``, not just "the first one".

    Pointing both columns at the same block turns APC into an in-place update, so
    the result has to match the plain path exactly. Getting the column arithmetic
    wrong (say ignoring the index, or swapping the two) shows up here.
    """
    t = _apc_problem(4, 256, 4, 4, True, seed=97)
    # Swap the columns and the index tensors with them: same read/write blocks.
    swapped = torch.stack((t["write_blocks"], t["read_blocks"]), dim=1)

    want_state = t["conv_state"].clone()
    want = causal_conv1d_update_flydsl(
        t["x"].clone(),
        want_state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["indices2d"],
        num_accepted_tokens=t["num_accepted"],
        block_idx_last_scheduled_token=t["last_col"],
        initial_state_idx=t["init_col"],
    )
    got_state = t["conv_state"].clone()
    got = causal_conv1d_update_flydsl(
        t["x"].clone(),
        got_state,
        t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=swapped,
        num_accepted_tokens=t["num_accepted"],
        block_idx_last_scheduled_token=t["init_col"],
        initial_state_idx=t["last_col"],
    )
    _assert_bit_exact("apc swapped columns", "output", got, want)
    _assert_bit_exact("apc swapped columns", "conv_state", got_state, want_state)


def test_sglang_predicate_refuses_circular_buffer():
    """``cache_seqlens`` is unimplemented here, as it is in SGLang's own kernel."""
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=23)
    seqlens = torch.zeros(4, dtype=torch.int32, device=DEVICE)

    assert not _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], cache_seqlens=seqlens
    )
    with pytest.raises(NotImplementedError):
        causal_conv1d_update_sglang_flydsl(
            t["x"], t["conv_state"], t["weight"], cache_seqlens=seqlens
        )


def test_predicates_refuse_unsupported_shapes():
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=29)

    # fp32 is not one of the specializations, and would otherwise be dispatched
    # as fp16 by the dtype selection in the wrapper.
    assert not _causal_conv1d_update_flydsl_supported(
        t["x"].float(), t["conv_state"].float(), t["weight"].float()
    )
    # SGLang's kernel covers widths 2/3/4 only; the vLLM one goes to 6.
    wide = torch.randn((256, 5), device=DEVICE, dtype=DTYPE)
    state_wide = torch.randn((5, 256, 4), device=DEVICE, dtype=DTYPE)
    assert _causal_conv1d_update_flydsl_supported(t["x"], state_wide, wide)
    assert not _causal_conv1d_update_sglang_flydsl_supported(t["x"], state_wide, wide)
    with pytest.raises(NotImplementedError):
        causal_conv1d_update_sglang_flydsl(t["x"], state_wide, wide)
    # A conv_state too short for the requested window.
    short = torch.randn((5, 256, 1), device=DEVICE, dtype=DTYPE)
    assert not _causal_conv1d_update_flydsl_supported(t["x"], short, t["weight"])


# -- performance ----------------------------------------------------------

#: ``decode`` is one token per sequence, ``verify`` a speculative window over the
#: dense layout and ``varlen`` the same window packed the way vLLM hands it over.
#: aiter's Triton kernel has no packed entry point, so it only runs the first two.
BENCH_MODES = ("decode", "verify", "varlen")


def _bench_problem(batch, dim, width, seqlen, mode, dtype):
    """Inputs for one benchmark row: the dense problem plus the call it needs."""
    spec = mode != "decode"
    t = _make_inputs(
        batch, dim, width, seqlen, spec=spec, seed=batch * 31 + seqlen, dtype=dtype
    )
    t["indices"] = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    call = dict(
        weight=t["weight"],
        bias=t["bias"],
        activation="silu",
        conv_state_indices=t["indices"],
        num_accepted_tokens=t["num_accepted"],
    )
    if mode == "varlen":
        call["x"] = _pack(t["x"], (seqlen,) * batch)
        call["query_start_loc"] = torch.arange(
            0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=DEVICE
        )
        call["max_query_len"] = seqlen
    else:
        call["x"] = t["x"]
    return t, call


def _bench_bytes(t, batch, dim, seqlen):
    """Bytes the kernel has to move, at best.

    Every touched cache line is both read (the history taps) and written (the
    rolled window), so ``conv_state`` counts twice. The weights and bias are one
    row each and are reread by every workgroup, but they are counted once: what
    this bounds is the traffic the kernel cannot avoid.
    """
    esz = t["x"].element_size()
    state_len = t["conv_state"].shape[-1]
    elems = 2 * batch * seqlen * dim + 2 * batch * dim * state_len
    elems += t["weight"].numel() + (0 if t["bias"] is None else t["bias"].numel())
    return elems * esz


@benchmark()
def bench_causal_conv1d_update(
    batch: int = 64,
    dim: int = 4096,
    width: int = 4,
    seqlen: int = 1,
    mode: str = "decode",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """One row of the perf table: FlyDSL against aiter's Triton kernel.

    Not named ``test_*``: it returns a row instead of asserting, which pytest
    warns about today and rejects from pytest 9. The test below covers it.
    """
    t, call = _bench_problem(batch, dim, width, seqlen, mode, dtype)
    tokens = batch * seqlen

    # Both entry points overwrite x and conv_state, as upstream does, so every
    # call gets its own copies and the answer is taken from a clean one: over
    # run_perftest's ~100 replays the input feeds on its own output and reaches
    # inf, which costs nothing in time (measured at 0.99-1.00x against a pristine
    # input) but is not what the spec describes.
    def fresh():
        return dict(call, x=call["x"].clone(), conv_state=t["conv_state"].clone())

    out_fly = causal_conv1d_update_flydsl(**fresh())
    _, us_fly = run_perftest(causal_conv1d_update_flydsl, **fresh())
    if mode == "varlen":
        us_tri = float("nan")  # aiter's Triton kernel has no packed entry point
    else:
        _, us_tri = run_perftest(causal_conv1d_update_triton, **fresh())

    out_spec, _, _, _, _ = run_torch(
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["indices"],
        t["num_accepted"],
        skip_line=0,
    )
    if mode == "varlen":
        out_fly = out_fly.view(batch, seqlen, dim).transpose(1, 2)
    err = checkAllclose(
        out_fly.float(),
        out_spec.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{mode} b{batch} d{dim} w{width} s{seqlen} out",
    )

    nbytes = _bench_bytes(t, batch, dim, seqlen)
    return {
        "gfx": get_gfx(),
        "dtype": _DTYPE_NAME[dtype],
        "us": us_fly,
        "triton_us": us_tri,
        "TB/s": nbytes / us_fly / 1e6,
        # width multiply-adds per output element; the bias and the silu are not
        # counted, so this is the convolution proper.
        "TFLOPS": 2 * width * tokens * dim / us_fly / 1e6,
        "err_pct": err,
    }


@pytest.mark.parametrize("mode", BENCH_MODES)
def test_perf_row_agrees_with_the_spec(mode):
    """Keep the timing path inside the suite, one small row per mode.

    Nothing else here calls ``run_perftest``, so the harness is the easiest thing
    in the file to break without noticing -- and ``checkAllclose`` only logs, so
    the row's error column has to be asserted on to mean anything.
    """
    row = bench_causal_conv1d_update(
        batch=8, dim=512, seqlen=1 if mode == "decode" else 2, mode=mode
    )
    assert row["us"] > 0, f"{mode}: no timing recorded"
    assert row["err_pct"] == 0, (
        f"{mode}: {row['err_pct']:.1%} of the output is outside checkAllclose's "
        "1e-2 window against the fp64 spec"
    )


# -- CI entry point -------------------------------------------------------


_DTYPE_BY_NAME = {name: dt for dt, name in _DTYPE_NAME.items()}


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Correctness and perf for the FlyDSL causal_conv1d update kernels",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="*",
        default=["correctness", "perf"],
        choices=["correctness", "perf"],
        help="""Which halves to run. Only correctness sets the exit code.
        e.g.: --run perf""",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 32, 128, 512])
    parser.add_argument("--dim", type=int, nargs="*", default=[2048, 4096])
    parser.add_argument("-w", "--width", type=int, nargs="*", default=[4])
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="""Tokens per sequence. 1 is decode; the rest are the speculative
        window, so they are skipped in decode mode and vice versa.""",
    )
    parser.add_argument(
        "-d", "--dtype", type=str, nargs="*", default=["bf16"], choices=["bf16", "fp16"]
    )
    parser.add_argument(
        "--mode", type=str, nargs="*", default=list(BENCH_MODES), choices=BENCH_MODES
    )
    return parser.parse_args()


def _run_perf_sweep(args):
    rows = []
    # Sweep order, slowest to fastest changing: mode, dtype, width, seqlen, dim.
    for mode, dtype, width, seqlen, dim, batch in itertools.product(
        args.mode, args.dtype, args.width, args.seqlen, args.dim, args.batch
    ):
        # A one-token speculative window is the decode row under another name.
        if (seqlen == 1) != (mode == "decode"):
            continue
        rows.append(
            bench_causal_conv1d_update(
                batch=batch,
                dim=dim,
                width=width,
                seqlen=seqlen,
                mode=mode,
                dtype=_DTYPE_BY_NAME[dtype],
            )
        )
    if rows:
        aiter.logger.info(
            "flydsl causal_conv1d_update perf (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )


def main():
    """Run every check without pytest, then the perf sweep.

    Exits non-zero if any check fails. CI shards ``op_tests/test_*.py`` by
    invoking ``python3 <file>``, which would otherwise merely define the test
    functions above and report success. The exit code matters: a harness that
    prints failures but exits 0 leaves the job green while the kernel is broken.
    """
    args = _parse_args()
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl causal_conv1d_update is unsupported on %s; skipping", get_gfx()
        )
        return 0

    failures = 0
    if "correctness" in args.run:
        failures = _run_correctness()
    if "perf" in args.run:
        _run_perf_sweep(args)
    return 1 if failures else 0


def _run_correctness():
    torch.manual_seed(0)
    cases = [
        (
            test_vllm_against_spec_and_aiter_triton,
            [p.values for p in _VLLM_DTYPE_CASES],
        ),
        (test_vllm_channels_per_thread_agree, [(1,), (2,)]),
        (test_cpt_policy_backs_off_for_long_speculative_windows, [()]),
        (test_vllm_skips_the_null_block, [()]),
        (test_vllm_out_parameter_leaves_x_untouched, [()]),
        (test_sglang_against_spec, _SGLANG_CASES),
        (test_dispatch_seam_routes_to_flydsl, _SEAM_CASES),
        (test_dispatch_seam_falls_through_when_out_of_scope, [()]),
        (test_triton_update_refuses_unimplemented_width, [()]),
        (test_dispatch_seam_preserves_2d_decode_shape, [()]),
        (test_dispatch_seam_is_off_by_default, [()]),
        (test_predicates_accept_the_supported_slice, [()]),
        (test_vllm_predicate_accepts_varlen_and_apc, [()]),
        (test_vllm_varlen_needs_indices_and_a_token_budget, [()]),
        (test_vllm_varlen_matches_the_dense_path, _VARLEN_DENSE_EQUIVALENT),
        (test_vllm_varlen_matches_the_spec, _VARLEN_CASES),
        (test_vllm_varlen_with_apc_copies_instead_of_clobbering, _VARLEN_CASES),
        (test_vllm_reads_x_through_its_strides, [(_channel_major,), (_column_slice,)]),
        (test_vllm_apc_needs_both_index_tensors, [()]),
        (test_vllm_apc_copies_instead_of_clobbering, _APC_CASES),
        (test_vllm_apc_reads_the_selected_column, [()]),
        (test_sglang_predicate_refuses_circular_buffer, [()]),
        (test_predicates_refuse_unsupported_shapes, [()]),
        (test_perf_row_agrees_with_the_spec, [(m,) for m in BENCH_MODES]),
    ]

    failures = 0
    for fn, arg_sets in cases:
        for args in arg_sets:
            name = f"{fn.__name__}{args if args else ''}"
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 - report and keep going
                failures += 1
                # Unbuffered: redirected to a file these would otherwise all
                # land after the perf table, which logs to stderr.
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
            else:
                print(f"ok   {name}", flush=True)

    print(f"\n{failures} failure(s)", flush=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
