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

The reference is the upstream kernel itself
-------------------------------------------
Each port is measured against **the kernel it claims to replace**: vLLM's own
``causal_conv1d_update`` and SGLang's own Triton one, vendored verbatim into
``op_tests/triton_tests/utils/causal_conv1d_update_refs.py`` and called from
``_oracle_vllm`` and ``_oracle_sglang``. Nothing in this file re-derives what
the convolution should do.

That is deliberate, and it replaces an earlier pair of hand-written torch
references. A transcription can only ever encode one reading of a contract, and
it is the same reading the port was written from, so the two agree on a
misreading and the reference reports success. It also cannot be *timed*: the
question "is the port faster than what ships today" needs the shipping kernel in
the room, not a torch loop that is orders of magnitude slower. Vendoring answers
both at once -- the same code is the oracle in the correctness tests and a
candidate in the perf table.

Nothing here imports ``vllm`` or ``sglang``: the suite must not depend on either
being installed, and a live import would silently re-point the oracle whenever
the installed version moved. Both copies carry their upstream revision and the
exact line range they came from, and are re-synced mechanically; see that file's
module docstring. This is how the rest of aiter treats upstream references
(``op_tests/test_causal_conv1d_update.py`` carries Tri Dao's, and
``op_tests/triton_tests/utils/mla_extend_ref.py`` SGLang's).

Sharing a file does not make them one kernel, upstream or here. The contracts
overlap but are not the same one: vLLM has the packed layout and no snapshots,
SGLang has the EAGLE tree and the snapshots and no packing, and their bodies
share barely a dozen lines. Note also that aiter's own Triton
kernel is a *fork* of SGLang's with the tree path and PDL stripped out, so it is
not interchangeable with either -- it appears in the perf table as the incumbent
and at the dispatch seam, but never as the oracle.

How the numeric bound is set
----------------------------
A bf16 result is *not* required to agree bit-for-bit with the upstream bf16
kernel: they round differently, and the FlyDSL kernels are frequently the more
accurate of the pair, so demanding agreement would pin the less accurate one's
error. Instead each upstream kernel is run three times per case:

* at **bf16** -- the baseline. Its ``conv_state``, snapshots and parent map are
  the bit-exact expectation, and its output is what the port must be no further
  from the spec than.
* at **fp32** -- the spec. The same code on the same inputs with ~16 more
  mantissa bits. Against an fp64 torch evaluation this lands at ~1e-6 while the
  bf16 answers land at ~4e-2, so as a yardstick for bf16 it is exact for free.
* at **fp32 on absolute values**, bias absolute and activation off -- the
  conditioning scale. The convolution is a dot product, so feeding ``|taps|`` and
  ``|w|`` makes every term positive and the result is exactly
  ``sum_j |tap_j * w_j|``.

Each element then gets a budget of ``factor * 2**-8 * sum_j |tap_j * w_j|``,
because the error of a dot product is bounded by the sum of the *term*
magnitudes, not by the magnitude of the result: a budget scaled to the result
would be vacuous where the taps cancel and the output lands near zero, which is
exactly where a bf16 kernel is least accurate. Deriving the scale from upstream
rather than from a hand-written gather is the point -- *which* taps a token has
is the contract-dependent part (rollback offset, packed window revision, parent
chain), and it is now not duplicated anywhere.

``conv_state``, the snapshots and the tree's parent map are compared
**bit-exactly** against the upstream bf16 run in every case: they are assembled
from copies of ``x`` and of the previous state with no arithmetic applied, so any
difference there is a state-management bug rather than rounding.

The perf table pairs each port with its own upstream on the same buffers, which
is the comparison the port exists to win: ``flydsl`` next to ``vllm`` on the
decode and varlen rows, next to ``sglang`` on the verify rows, and ``triton``
(aiter's current kernel) wherever it can express the call at all.

Sequence indices deliberately start at 1 in the vLLM cases: that wrapper's
``null_block_id`` sentinel is ``0`` (block 0 is the reserved null block), so a
plain ``arange(batch)`` would make sequence 0 a null block and silently skip it.
``test_vllm_skips_the_null_block`` covers that behavior on purpose.

Scope: Qwen, and only Qwen
--------------------------
This port serves Qwen's GDN linear attention, so the shapes, dtypes and memory
layouts are taken from Qwen's own call sites and from nowhere else. See the block
above ``_alloc_x`` for each layout and the line it comes from; both upstreams
assert ``x.stride(1) == 1``, so the channel axis is always contiguous and what
varies is the token stride.

The kernels are wider than that, and the difference is deliberately **not**
covered here. Each of these is a real capability of the port with no Qwen caller,
so the cost of leaving it out is that a regression in it would land silently:

* **Automatic Prefix Caching** -- the 2D ``conv_state_indices`` with a separate
  read and write block. No Qwen call site passes it.
* **fp16** -- a separately compiled specialization; Qwen serves bf16.
* **Widths other than 4** -- the port compiles 2-6 and Qwen's
  ``linear_conv_kernel_dim`` is 4, its default and its only value.
* **``cache_seqlens``** (the circular buffer) and the ``out=`` parameter, neither
  of which any Qwen site passes.

Upstream revisions vendored
---------------------------
vLLM ``63a9a5010`` and SGLang ``18107e38d2``. Bumping either means re-extracting
the copy, not editing it; the recipe is in each vendored file's docstring. A bump
that changes an answer is upstream changing its contract, which is worth seeing
as a test failure rather than absorbing into a hand-edited reference.

Usage:
    HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/test_flydsl_causal_conv1d_update.py

    # or, the way CI invokes it -- the same checks with no pytest dependency,
    # followed by the perf sweep:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_causal_conv1d_update.py

    # narrow the perf sweep:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_causal_conv1d_update.py \
        --mode varlen -l qkvz_slice -b 128 -s 4
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback
from typing import NamedTuple

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

# CI runs this as `python3 op_tests/<file>`, which puts op_tests/ on sys.path
# rather than the repo root, so the vendored upstream kernels below would not
# resolve. Same line as op_tests/test_gemm_a8w8_blockscale.py:9.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.triton_tests.utils.causal_conv1d_update_refs import (
    causal_conv1d_update_sglang as causal_conv1d_update_sglang_upstream,
)
from op_tests.triton_tests.utils.causal_conv1d_update_refs import (
    causal_conv1d_update_vllm as causal_conv1d_update_vllm_upstream,
)

_SKIP_REASON = None
if not torch.cuda.is_available():
    _SKIP_REASON = "ROCm is not available"
elif not is_flydsl_available():
    _SKIP_REASON = "flydsl is not installed"
else:
    try:
        from aiter.ops.flydsl.causal_conv1d_update_kernels import (
            NULL_BLOCK_ID,
            _causal_conv1d_update_flydsl_supported,
            _causal_conv1d_update_sglang_flydsl_supported,
            _is_dedup_conv_window,
            causal_conv1d_update_flydsl,
            causal_conv1d_update_sglang_flydsl,
        )
        from aiter.ops.triton.conv.causal_conv1d import (
            causal_conv1d_update as causal_conv1d_update_triton,
        )
    except ImportError as exc:
        _SKIP_REASON = f"the FlyDSL causal_conv1d update kernels do not import ({exc})"

# Only pytest can be told to skip a module. CI shards op_tests with
# `python3 <file>`, where a module-level skip raises Skipped with nobody to catch
# it and the shard reports a failure instead of a skip, so main() handles that
# case and this stays out of its way.
if _SKIP_REASON is not None and __name__ != "__main__":
    pytest.skip(
        f"{_SKIP_REASON}. Skipping the FlyDSL causal_conv1d update tests.",
        allow_module_level=True,
    )


DEVICE = "cuda"
DTYPE = torch.bfloat16

#: The kernels are built for CDNA3/CDNA4 and the launch policy is tuned on them.
SUPPORTED_GFX = ("gfx942", "gfx950")

#: One mantissa step of the storage dtype, the relative cost of storing any
#: single value. Only bf16 appears here because that is what Qwen serves; the
#: port also specializes fp16, which would need its own tighter step (2**-11) as
#: well as its own cases, and is out of this file's scope.
_DTYPE_EPS = {torch.bfloat16: 2.0**-8}
_DTYPE_NAME = {torch.bfloat16: "bf16"}

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


# -- the upstream kernels, as oracle and as baseline ----------------------
#
# Both references are vLLM's and SGLang's own kernels, vendored verbatim into
# op_tests/triton_tests/utils/causal_conv1d_update_refs.py and called here.
# Nothing in this file re-derives what the convolution should do.
#
# Each is run three times per case, which is what lets one kernel serve as the
# whole yardstick:
#
# 1. at the tested dtype -- the *baseline*. This is the number the port has to
#    beat on speed and match on accuracy, and its conv_state / snapshots / parent
#    map are the bit-exact expectation, since those are assembled from copies.
# 2. at fp32 -- the *spec*. Same code, same inputs, ~16 more mantissa bits.
#    Measured against an fp64 torch evaluation this lands at ~1e-6 while the bf16
#    answers land at ~4e-2, so as a yardstick for bf16 it is exact for free.
# 3. at fp32 on absolute values, bias absolute, activation off -- the
#    *conditioning scale*. The convolution is a dot product, so feeding |taps|
#    and |w| makes every term positive and the sum of products is exactly
#    sum_j |tap_j * w_j|, which is what the error budget is scaled by. Taking it
#    from upstream rather than from a hand-written gather matters: which taps a
#    token even has is the contract-dependent part (rollback offset, packed
#    window revision, parent chain), and it is now not duplicated anywhere.
#
# The two are deliberately not factored together. The contracts overlap but are
# not the same one -- vLLM has the packed layout and no snapshots; SGLang has the
# EAGLE tree and the snapshots and no packing -- and they are different files
# upstream for that reason.


class _Oracle(NamedTuple):
    """What one upstream kernel says about a problem.

    ``spec`` and ``scale`` are fp64 views of the fp32 runs. ``out``, ``state``,
    ``window`` and ``parents`` come from the run at the tested dtype and are the
    baseline; the last three are compared bit-exactly.
    """

    spec: torch.Tensor
    scale: torch.Tensor
    out: torch.Tensor
    state: torch.Tensor
    window: torch.Tensor | None
    parents: torch.Tensor | None


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
):
    """vLLM's own ``causal_conv1d_update`` (rev ``63a9a5010``), run as the oracle.

    Takes the packed arguments too: upstream handles the packed layout itself, so
    unlike a dense reference this needs no separate story for it. Note the
    wrapper defaults ``out`` to ``x`` and writes in place, hence the copies.
    """
    kw = {
        "conv_state_indices": conv_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "null_block_id": null_block_id,
        "query_start_loc": query_start_loc,
        "max_query_len": max_query_len,
    }
    state = conv_state.clone()
    out = causal_conv1d_update_vllm_upstream(
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
    return _Oracle(spec.double(), scale.double(), out, state, None, None)


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
    """SGLang's Triton ``causal_conv1d_update`` (rev ``18107e38d2``), as the oracle.

    Parameter names are upstream's (``num_accept_tokens``, not vLLM's
    ``num_accepted_tokens``). Upstream allocates its output with ``empty_like``,
    so a sequence skipped via ``pad_slot_id`` gets undefined values rather than
    ``x``'s -- no case here uses a pad slot, but a comparison over one would have
    to mask those rows.
    """
    kw = {
        "conv_state_indices": conv_state_indices,
        "num_accept_tokens": num_accept_tokens,
        "intermediate_state_indices": intermediate_state_indices,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "pad_slot_id": pad_slot_id,
    }
    win = intermediate_conv_window

    # The parent map is an output the kernel also reads back while walking, so
    # every run needs its own; the tree links alone determine it, so all three
    # runs produce the same map.
    def parent_buf():
        return (
            None
            if retrieve_next_token is None
            else torch.zeros_like(retrieve_next_token)
        )

    state = conv_state.clone()
    window = None if win is None else win.clone()
    parents = parent_buf()
    out = causal_conv1d_update_sglang_upstream(
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
    return _Oracle(spec.double(), scale.double(), out, state, window, parents)


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
# Width is 4 throughout because that is Qwen's `linear_conv_kernel_dim`, its
# default and its only value; the port compiles widths 2-6 but this file does not
# reach for them (see the module docstring on what is deliberately not covered).
_VLLM_CASES = [
    (1, 128, 4, 1, False),
    (4, 256, 4, 1, False),
    (64, 512, 4, 1, False),
    # Multi-token without a rollback point, in the one band where the slide
    # length is observable: 1 < seqlen < width - 1 keeps part of the rolled
    # window coming from conv_state, and it then slides by the chunk length
    # rather than by one column. At seqlen >= width - 1 every slot comes from x
    # and the distinction vanishes, so width 4 / seqlen 2 is the only cell of
    # this kind reachable at Qwen's width.
    (4, 128, 4, 2, False),
    (8, 256, 4, 2, True),
    (32, 384, 4, 4, True),
    (16, 256, 4, 8, True),
]


@pytest.mark.parametrize("batch,dim,width,seqlen,spec", _VLLM_CASES)
def test_vllm_matches_upstream_vllm(batch, dim, width, seqlen, spec):
    """Drop-in for vLLM's own causal_conv1d_update, and at least as accurate.

    Three things at once: the output stays inside its error budget, it is no
    further from the spec than vLLM's kernel is, and the ``conv_state`` roll-back
    matches vLLM's bit-exactly. The baseline is upstream itself rather than
    aiter's Triton fork, because it is vLLM's contract this entry point claims to
    implement -- aiter's Triton is SGLang-shaped and is measured in the perf table
    and at the dispatch seam instead.
    """
    t = _make_inputs(batch, dim, width, seqlen, spec=spec, seed=batch * 100 + seqlen)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    label = f"vllm b{batch} d{dim} w{width} s{seqlen} spec={spec}"

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

    _assert_tracks_spec(label, out_fly, ref.spec, ref.scale)
    _assert_no_worse_than(
        label, out_fly, ref.out, ref.spec, ref.scale, "vLLM's own kernel"
    )
    _assert_bit_exact(label, "conv_state roll-back", state_fly, ref.state)


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


# -- SGLang interface -----------------------------------------------------

# (batch, dim, width, seqlen, spec, save_intermediate, tree)
_SGLANG_CASES = [
    (4, 256, 4, 1, False, False, False),
    (1, 128, 4, 1, False, False, False),
    (32, 384, 4, 4, True, False, False),
    (16, 256, 4, 8, True, False, False),
    (8, 256, 4, 4, True, True, False),
    (8, 256, 4, 8, True, False, True),
    (4, 128, 4, 4, True, True, True),
    # Tree links with `num_accept_tokens` left out. This is the shape SGLang's
    # own target-verify sites send (gdn and kda both pass the tree and the
    # snapshot but no accept count), and it is a distinct path: the accept count
    # is what turns on the rollback and the wider state, so upstream and this
    # port both fall to `offset = 0` and `state_len = width - 1` here and let the
    # parent links carry the chain instead.
    (8, 256, 4, 8, False, True, True),
    (4, 128, 4, 4, False, False, True),
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
def test_sglang_matches_upstream_sglang(
    batch, dim, width, seqlen, spec, save_inter, tree
):
    """Covers SGLang's decode, chain-verify, SAVE_INTERMEDIATE and tree paths.

    The tree cases are the ones where aiter's Triton could never have served as
    the baseline: it is a fork of this kernel with the EAGLE path stripped out, so
    only upstream itself can say what the parent-chain convolution should return.
    """
    t = _sglang_problem(batch, dim, width, seqlen, spec, save_inter, tree)
    label = (
        f"sglang b{batch} d{dim} w{width} s{seqlen} spec={spec} "
        f"inter={save_inter} tree={tree}"
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

    _assert_tracks_spec(label, out, ref.spec, ref.scale)
    _assert_no_worse_than(
        label, out, ref.out, ref.spec, ref.scale, "SGLang's own kernel"
    )
    _assert_bit_exact(label, "conv_state roll-back", state, ref.state)
    if save_inter:
        _assert_bit_exact(label, "intermediate_conv_window", window, ref.window)
    if tree:
        _assert_bit_exact(label, "retrieve_parent_token map", parents, ref.parents)


#: SGLang only enables its deduplicated snapshot layout for CUDA linear draft
#: chains (see ``conv_window_dedup_enabled``): tree ancestors need independent
#: windows, so an aliasing view would have siblings overwrite each other. The
#: widths on either side of Qwen's 4 pin the collapse at more than one ratio,
#: and width 2 pins the opposite -- a single-tap window has nothing to fold.
_DEDUP_CASES = [
    (8, 256, 4, 4),
    (8, 256, 4, 8),
    (4, 128, 3, 4),
    (4, 128, 3, 8),
    (4, 128, 2, 4),
]


@pytest.mark.parametrize("batch,dim,width,seqlen", _DEDUP_CASES)
def test_sglang_dedup_conv_window_matches_dense(batch, dim, width, seqlen):
    """SGLang's overlapping snapshot view gets the same bytes, written once.

    On the linear draft chain SGLang does not allocate the dense
    ``(lines, seqlen, dim, width-1)`` snapshot at all -- it allocates a compact
    ``(lines, dim, seqlen+width-2)`` buffer and hands us an ``as_strided`` view
    of it, so consecutive steps' windows alias and the dense store pattern
    writes every element ``width-1`` times over. The kernel detects that view
    and walks the run once instead. What has to hold is that the caller cannot
    tell: reading the snapshot back through the view must give exactly what the
    dense path would have left there.
    """
    t = _sglang_problem(batch, dim, width, seqlen, True, True, False)
    label = f"dedup b{batch} d{dim} w{width} s{seqlen}"
    lines = t["conv_state"].shape[0]

    def run(window):
        state = t["conv_state"].clone()
        out = causal_conv1d_update_sglang_flydsl(
            t["x"].clone(),
            state,
            t["weight"],
            bias=t["bias"],
            activation="silu",
            conv_state_indices=t["indices"],
            num_accept_tokens=t["num_accepted"],
            intermediate_conv_window=window,
            intermediate_state_indices=t["indices"],
        )
        return out, state

    dense = torch.zeros((lines, seqlen, dim, width - 1), device=DEVICE, dtype=DTYPE)
    out_d, state_d = run(dense)

    phys = torch.zeros((lines, dim, seqlen + width - 2), device=DEVICE, dtype=DTYPE)
    view = phys.as_strided(
        (lines, seqlen, dim, width - 1),
        (phys.stride(0), 1, phys.stride(1), 1),
    )
    assert _is_dedup_conv_window(view, width) == (width > 2), "detection misfired"
    assert not _is_dedup_conv_window(dense, width), "dense snapshot must stay dense"
    out_v, state_v = run(view)

    _assert_bit_exact(label, "output", out_v, out_d)
    _assert_bit_exact(label, "conv_state roll-back", state_v, state_d)
    _assert_bit_exact(label, "snapshot read back through the view", view, dense)


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
    (8, 256, 4, 8, True, False),
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

    # aiter's Triton entry point is SGLang-shaped -- it spells the sentinel
    # `pad_slot_id`, takes `intermediate_conv_window` (which vLLM's kernel does
    # not have at all) and falls through to the SGLang-interface port -- so
    # SGLang's kernel is the one that describes it.
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
    _assert_tracks_spec(label, out_on, ref.spec, ref.scale)
    _assert_no_worse_than(
        label, out_on, out_off, ref.spec, ref.scale, "the Triton path (seam off)"
    )
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


def test_predicates_accept_what_the_port_covers():
    """Everything Qwen sends must be in scope, or the seam silently falls back.

    A predicate that refuses a supported case is not a wrong answer, it is a
    wrong *kernel*: the dispatch seam hands the call back to Triton and the port
    is never exercised. So the accept side is worth pinning as tightly as the
    refuse side.
    """
    t = _make_inputs(4, 256, 4, 4, spec=True, seed=17)
    assert _causal_conv1d_update_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accepted_tokens=t["num_accepted"]
    )
    assert _causal_conv1d_update_sglang_flydsl_supported(
        t["x"], t["conv_state"], t["weight"], num_accept_tokens=t["num_accepted"]
    )

    # The packed layout, which is what Qwen's MTP site sends.
    flat = _make_inputs(4, 256, 4, 1, spec=False, seed=19)
    assert _causal_conv1d_update_flydsl_supported(
        flat["x"].squeeze(-1),
        flat["conv_state"],
        flat["weight"],
        query_start_loc=torch.zeros(5, dtype=torch.int32, device=DEVICE),
        max_query_len=1,
    )


def test_out_of_scope_is_refused_rather_than_mishandled():
    """Everything the port does not implement, refused the way the caller expects.

    Two different refusals, and the distinction matters. The predicates return
    False so the dispatch seam can fall through to Triton, which is how an
    unsupported *configuration* is handled. The wrapper raises when the call
    itself is malformed, because there is nothing to fall through to.
    """
    t = _make_inputs(4, 256, 4, 1, spec=False, seed=29)
    qsl = torch.zeros(5, dtype=torch.int32, device=DEVICE)
    packed = t["x"].squeeze(-1)

    # -- refused by predicate: fall through to Triton --
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
    # A conv_state too short for the requested window.
    short = torch.randn((5, 256, 1), device=DEVICE, dtype=DTYPE)
    assert not _causal_conv1d_update_flydsl_supported(t["x"], short, t["weight"])
    # Packing without the token budget cannot be sized.
    assert not _causal_conv1d_update_flydsl_supported(
        packed, t["conv_state"], t["weight"], query_start_loc=qsl
    )

    # -- refused by raising: the call cannot be completed at all --
    with pytest.raises(NotImplementedError):
        causal_conv1d_update_sglang_flydsl(t["x"], state_wide, wide)
    # Packed x hides both the batch and the per-sequence budget, so dropping
    # either one leaves the kernel with no way to size the launch.
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


#: (width, max_query_len, spec, per-sequence token counts).
_VARLEN_CASES = [
    (4, 1, False, (1, 1, 1, 1)),
    (4, 3, True, (3, 1, 2, 3)),
    (4, 4, True, (4, 2, 1, 4, 3)),
    (4, 2, True, (2, 1)),
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
#: dense entry point cannot express, so only the packed comparison reaches those.
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


@pytest.mark.parametrize("width,seqlen,spec,lens", _VARLEN_DENSE_EQUIVALENT)
def test_vllm_varlen_matches_the_dense_path(width, seqlen, spec, lens):
    """Packing is a layout change, not a different computation.

    For these cases each packed sequence is the dense problem at its own length,
    so run every sequence on its own through the dense path -- already pinned
    against vLLM's own kernel -- and require the packed batch to reproduce
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
def test_vllm_varlen_matches_upstream_vllm(width, seqlen, spec, lens):
    """Every packed case against vLLM's kernel, narrowed windows included.

    The test above only reaches the cases each of whose sequences is a dense
    problem. This one reaches all of them by handing upstream the same packed
    buffer, so the window revision that a short sequence triggers is performed by
    the kernel that defines it rather than inferred. ``conv_state`` must match bit
    for bit -- it is copies all the way down, including the slots past a narrowed
    window that must keep whatever the cache line already held.
    """
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

    _assert_bit_exact(label, "conv_state", got_state, ref.state)
    # Every row of the packed buffer belongs to some sequence, so the whole thing
    # is meaningful and there is nothing to mask off.
    _assert_tracks_spec(label, got, ref.spec, ref.scale)
    _assert_no_worse_than(label, got, ref.out, ref.spec, ref.scale, "vLLM's own kernel")


# -- the layouts Qwen hands the kernel ------------------------------------
#
# Both upstreams assert ``x.stride(1) == 1`` under ``validate_data``
# (vLLM ``causal_conv1d.py:1153``, SGLang ``causal_conv1d_triton.py``), and
# Qwen's decode site passes ``validate_data=True``, so the channel axis is always
# the contiguous one. What differs between call sites is the *token* stride, and
# none of them is a fresh contiguous tensor of the shape the kernel sees -- every
# one is a view into a buffer some projection wrote:
#
# ``contiguous``   vLLM Qwen3-Next: ``mixed_qkv = torch.cat((query, key, value),
#                  dim=-1)`` (``qwen_gdn_linear_attn.py:920``). The speculative
#                  site's ``mixed_qkv.index_select(0, spec_token_indx)`` (:1320)
#                  lands here too -- ``index_select`` materializes a fresh
#                  contiguous tensor whatever it was handed.
# ``qkvz_slice``   vLLM Qwen3.5: ``mixed_qkv, z = mixed_qkvz.split([qkv_size,
#                  z_size], dim=-1)`` (:925), the leading column window of the
#                  fused qkvz projection. A token's channels stay adjacent, but
#                  consecutive tokens are the whole projection apart.
# ``verify_view``  SGLang GDN target-verify: ``mixed_qkv.view(batch, draft, -1)
#                  .transpose(1, 2)`` (``gdn_backend.py:545``). The dense 3D
#                  window is a transposed view -- a *contiguous* (batch, dim,
#                  tokens) has ``stride(1) == tokens`` and so is the one shape
#                  upstream's own assertion rejects.
#
# A channel-major decode input is deliberately absent: no Qwen path produces it,
# and with ``stride(1) == tokens`` it is outside the contract above.

#: ``z_size / qkv_size`` for the qkvz slice. Qwen3-Next defaults to 16 key heads
#: and 32 value heads of 128 (``configs/qwen3_next.py:205-208``), so the conv dim
#: is ``2 * 2048 + 4096 = 8192`` and the trailing z is ``value_dim = 4096``.
_QKVZ_Z_RATIO = 0.5


def _alloc_x(layout, shape, gen, dtype=DTYPE):
    """Allocate ``x`` as a view into the buffer the projection would have written.

    Not a contiguous tensor relaid out afterwards: the container is the real one
    and the kernel gets a view into it, which is both what the model does and one
    copy fewer. ``shape`` is the view's shape -- ``(tokens, dim)`` for the 2D
    layouts, ``(batch, dim, tokens)`` for ``verify_view``. The columns outside the
    view are randomized as well, so a kernel that strays past its window reads
    plausible values instead of zeros and still fails the comparison.
    """
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
    """Both halves of the contract: channel-contiguous, and actually strided."""
    assert view.stride(1) == 1, (
        f"{layout}: the channel axis is not contiguous (strides {view.stride()}), "
        "which is the one thing both upstream kernels assert about x"
    )
    # A single token makes the token stride unobservable -- a one-row column
    # slice really is contiguous -- so the layouts genuinely coincide there and
    # there is nothing to guard. Past that they must not.
    tokens = view.shape[0] if view.dim() == 2 else view.shape[2]
    if layout != "contiguous" and tokens > 1:
        assert not view.is_contiguous(), (
            f"{layout}: came back contiguous (strides {view.stride()}), so the "
            "strided path is not exercised"
        )
    return view


def test_vllm_reads_x_through_its_strides():
    """Qwen3.5 slices ``x`` out of the fused projection, so strides must be read.

    The kernel addresses ``x`` through the strides it is handed, and a wrong
    assumption there is invisible on the contiguous tensor Qwen3-Next passes.
    Same values at a different token stride, so this is bit-exact rather than a
    tolerance -- only the addresses differ. Dense decode and packed varlen both,
    because they take different paths to the same load.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(4242)

    # Dense decode, x as (batch, dim).
    t = _make_inputs(8, 256, 4, 1, spec=False, seed=4242)
    flat = t["x"].squeeze(-1)
    sliced = _assert_layout("qkvz_slice", _alloc_x("qkvz_slice", flat.shape, gen))
    sliced.copy_(flat)
    indices = torch.arange(1, 9, dtype=torch.int32, device=DEVICE)
    runs = []
    for variant in (flat, sliced):
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
    _assert_bit_exact("qkvz_slice (dense)", "output", runs[1][0], runs[0][0])
    _assert_bit_exact("qkvz_slice (dense)", "conv_state", runs[1][1], runs[0][1])

    # Packed varlen, x as (cu_tokens, dim).
    v = _varlen_problem(4, 4, True, (4, 2, 1, 4))
    packed = _pack(v["x_dense"], v["lens"])
    sliced = _assert_layout("qkvz_slice", _alloc_x("qkvz_slice", packed.shape, gen))
    sliced.copy_(packed)
    runs = []
    for variant in (packed, sliced):
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
    _assert_bit_exact("qkvz_slice (varlen)", "output", runs[1][0], runs[0][0])
    _assert_bit_exact("qkvz_slice (varlen)", "conv_state", runs[1][1], runs[0][1])


def test_sglang_reads_the_transposed_verify_window():
    """The 3D window SGLang verifies with is a transposed view, never contiguous.

    ``mixed_qkv.view(batch, draft, -1).transpose(1, 2)`` puts the channel axis at
    stride 1 and the token axis at ``dim``, which is the opposite of a contiguous
    ``(batch, dim, tokens)``. Both must give one answer; the contiguous tensor is
    only here as the value baseline, and is itself a shape upstream's
    ``x.stride(1) == 1`` assertion would refuse.
    """
    t = _sglang_problem(8, 256, 4, 4, spec=True, save_inter=True, tree=False)
    view = _assert_layout(
        "verify_view",
        _alloc_x(
            "verify_view", t["x"].shape, torch.Generator(device=DEVICE).manual_seed(77)
        ),
    )
    view.copy_(t["x"])

    runs = []
    for variant in (t["x"], view):
        probe = dict(t, x=variant)
        runs.append(
            _run_sglang(
                causal_conv1d_update_sglang_flydsl,
                probe,
                8,
                4,
                tree=False,
                save_inter=True,
            )
        )
    out_c, state_c, window_c, _ = runs[0]
    out_v, state_v, window_v, _ = runs[1]
    _assert_bit_exact("verify_view", "output", out_v, out_c)
    _assert_bit_exact("verify_view", "conv_state", state_v, state_c)
    _assert_bit_exact("verify_view", "intermediate_conv_window", window_v, window_c)


# -- performance ----------------------------------------------------------

#: Each mode is one upstream call site, reproduced as the model makes it.
#:
#: ``decode``  vLLM Qwen GDN, one token per sequence, ``x`` as ``(batch, dim)``
#:             and no accept count (``qwen_gdn_linear_attn.py:1366``, ``:1662``).
#: ``varlen``  vLLM Qwen GDN speculative decode -- the MTP site. The draft window
#:             is packed as ``(cu_tokens, dim)`` and comes with
#:             ``query_start_loc`` + ``num_accepted_tokens`` (``:1332``, ``:1712``).
#: ``verify``  SGLang GDN target-verify: the draft window as a transposed 3D view,
#:             through SGLang's interface, with the per-step snapshot buffer and
#:             *no* accept count -- that site passes neither ``num_accept_tokens``
#:             nor ``cache_seqlens`` (``gdn_backend.py:549``).
#: ``verify_tree``
#:             The same site once ``speculative_eagle_topk > 1``: the draft is a
#:             candidate *tree*, so the convolution walks each token's parent
#:             chain instead of its linear predecessor and the kernel builds
#:             ``retrieve_parent_token`` on the way (``gdn_backend.py:558``). Only
#:             the ReplaySSM fast path is restricted to a linear chain
#:             (``server_args.py:6315``); the tree falls back to this verify, so
#:             it is a live path and gets its own rows.
BENCH_MODES = ("decode", "verify", "verify_tree", "varlen")

#: The x layouts each call site can produce; see the block above `_alloc_x`.
#: Both verify modes' windows are transposed views and cannot be anything else --
#: a contiguous ``(batch, dim, tokens)`` is the one shape upstream rejects -- and
#: the 2D sites are contiguous on Qwen3-Next and a qkvz column slice on Qwen3.5.
_MODE_LAYOUTS = {
    "decode": ("contiguous", "qkvz_slice"),
    "varlen": ("contiguous", "qkvz_slice"),
    "verify": ("verify_view",),
    "verify_tree": ("verify_view",),
}

#: The two SGLang target-verify rows, which differ only by the tree links.
_VERIFY_MODES = ("verify", "verify_tree")

BENCH_LAYOUTS = ("contiguous", "qkvz_slice", "verify_view")


def _bench_problem(batch, dim, width, seqlen, mode, layout, dtype):
    """Inputs for one benchmark row: the call the model makes, and its dense form.

    ``x`` is allocated directly in the layout its call site produces rather than
    built contiguous and relaid out afterwards, so the buffer behind it is the one
    the projection would have written. The reference's dense
    ``(batch, dim, seqlen)`` is then a *view* of that same storage: the timed call
    and the checked values are one tensor, not a copy of one.
    """
    spec = mode == "varlen"
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

    if mode in _VERIFY_MODES:
        x = _alloc_x(layout, (batch, dim, seqlen), gen, dtype)
        dense = x
        # GDN always hands target-verify a snapshot buffer. Its companion
        # `intermediate_state_indices` is not in the shared call: aiter's Triton
        # has no such parameter, so passing it to every candidate would drop that
        # one. SGLang's own kernel dereferences it whenever a snapshot buffer is
        # present, so it gets it as a per-candidate extra below; the port falls
        # back to `conv_state_indices`. It selects a destination row and changes
        # no work, so it cannot move the timing either way.
        t["window"] = torch.randn(
            (t["conv_state"].shape[0], seqlen, dim, width - 1),
            generator=gen,
            device=DEVICE,
            dtype=dtype,
        )
        call["intermediate_conv_window"] = t["window"]
        if mode == "verify_tree":
            t["next_token"], t["next_sibling"] = _eagle_tree_links(batch, seqlen)
            call["retrieve_next_token"] = t["next_token"]
            call["retrieve_next_sibling"] = t["next_sibling"]
    elif mode == "decode":
        x = _alloc_x(layout, (batch, dim), gen, dtype)
        dense = x.unsqueeze(-1)
    else:
        # Every sequence gets the full draft window, packed end to end.
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
        """A fresh ``x`` holding the same values in the same layout.

        ``.clone()`` will not do: on a column slice it comes back *contiguous*,
        which is the very thing being measured, so the row would report the
        contiguous number under the strided layout's name.
        """
        dup = _alloc_x(layout, tuple(x.shape), gen, dtype)
        dup.copy_(x)
        return dup

    return t, call, new_x


def _bench_bytes(t, batch, dim, seqlen, width):
    """Bytes the kernel has to move, at best.

    Every touched cache line is both read (the history taps) and written (the
    rolled window), so ``conv_state`` counts twice. The weights and bias are one
    row each and are reread by every workgroup, but they are counted once: what
    this bounds is the traffic the kernel cannot avoid. The snapshot buffer is
    write-only and only exists on the verify modes, where it is by far the
    largest term -- ``width - 1`` values per token rather than one.
    """
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
    mode: str = "decode",
    layout: str = "contiguous",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """One row of the perf table: every candidate timed and checked on one shape.

    The defaults are a small decode row, so importing this under pytest costs one
    cheap launch; ``main()`` sweeps the real shapes.
    """
    t, call, new_x = _bench_problem(batch, dim, width, seqlen, mode, layout, dtype)
    label = f"{mode}/{layout} b{batch} d{dim} w{width} s{seqlen}"

    # Each mode is measured through the interface its own call site uses, and
    # against the upstream that defines that interface -- which is the point of
    # the table: `flydsl` in a decode or varlen row is the vLLM-shaped entry
    # point and its neighbour is vLLM's own kernel, and in a verify row it is the
    # SGLang-shaped one next to SGLang's. `triton` is aiter's current kernel, the
    # incumbent both ports have to beat. Each candidate is timed on the same
    # buffers, and a name absent from a row is a kernel that cannot express it.
    if mode in _VERIFY_MODES:
        candidates = {
            "flydsl": (causal_conv1d_update_sglang_flydsl, {}),
            "sglang": (
                causal_conv1d_update_sglang_upstream,
                {"intermediate_state_indices": t["indices"]},
            ),
        }
        if mode == "verify":
            # aiter's Triton is a fork of SGLang's with the tree path removed.
            candidates["triton"] = (causal_conv1d_update_triton, {})
        ref = _oracle_sglang(
            call["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            t["indices"],
            None,
            intermediate_conv_window=t["window"],
            intermediate_state_indices=t["indices"],
            retrieve_next_token=call.get("retrieve_next_token"),
            retrieve_next_sibling=call.get("retrieve_next_sibling"),
        )
    else:
        candidates = {
            "flydsl": (causal_conv1d_update_flydsl, {}),
            "vllm": (causal_conv1d_update_vllm_upstream, {}),
        }
        if mode == "decode":
            # aiter's Triton has no packed entry point, so it sits out varlen.
            candidates["triton"] = (causal_conv1d_update_triton, {})
        ref = _oracle_vllm(
            call["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            t["indices"],
            call.get("num_accepted_tokens"),
            query_start_loc=call.get("query_start_loc"),
            max_query_len=call.get("max_query_len", -1),
        )

    # Every entry point overwrites x and conv_state, as upstream does, so each
    # candidate gets its own copies. The answer comes from a clean call: over
    # run_perftest's ~100 replays the input feeds on its own output and reaches
    # inf, which costs nothing in time (measured at 0.99-1.00x against a pristine
    # input) but is not what the spec describes.
    def fresh(extra):
        args = dict(call, x=new_x(), conv_state=t["conv_state"].clone(), **extra)
        if "intermediate_conv_window" in args:
            args["intermediate_conv_window"] = t["window"].clone()
        if "retrieve_next_token" in args:
            # The parent map is an output: the kernel fills it in.
            args["retrieve_parent_token"] = torch.zeros(
                (batch, seqlen), dtype=torch.int32, device=DEVICE
            )
        return args

    tokens = batch * seqlen
    # width multiply-adds per output element; the bias and the silu are not
    # counted, so this is the convolution proper.
    flops = 2 * width * tokens * dim
    nbytes = _bench_bytes(t, batch, dim, seqlen, width)

    ret = {"gfx": get_gfx(), "dtype": _DTYPE_NAME[dtype]}
    for name, (fn, extra) in candidates.items():
        out = fn(**fresh(extra))
        _, us = run_perftest(fn, **fresh(extra))
        # No reshaping: the oracle was handed the same `x` the candidates get, so
        # its answer already has the call site's own shape, packed or 2D.
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            out.float(),
            ref.spec.float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: {label}",
        )
    return ret


@pytest.mark.parametrize("mode", BENCH_MODES)
def test_perf_row_agrees_with_the_spec(mode):
    """Keep the timing path inside the suite, one small row per mode.

    Nothing else here calls ``run_perftest``, so the harness is the easiest thing
    in the file to break without noticing -- and ``checkAllclose`` only logs, so
    the row's error column has to be asserted on to mean anything.
    """
    row = test_causal_conv1d_update_perf(
        mode=mode,
        seqlen=1 if mode == "decode" else 2,
        # The strided layout, so the row that guards the harness is the one whose
        # addressing can actually go wrong.
        layout=_MODE_LAYOUTS[mode][-1],
    )
    assert row["flydsl us"] > 0, f"{mode}: no timing recorded"
    assert row["flydsl err"] == 0, (
        f"{mode}: {row['flydsl err']:.1%} of the output is outside checkAllclose's "
        "1e-2 window against upstream run at fp32"
    )


# -- CI entry point -------------------------------------------------------


_DTYPE_BY_NAME = {name: dt for dt, name in _DTYPE_NAME.items()}


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Correctness and perf for the FlyDSL causal_conv1d update kernels",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 32, 128, 512])
    parser.add_argument(
        "--dim",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192],
        help="""Conv channels. Qwen3-Next's GDN uses 16 key heads and 32 value
        heads of 128, so the conv dim is 2 * 2048 + 4096 = 8192 at tp1; the
        defaults are that divided by tp = 1, 2, 4, 8.""",
    )
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
        "-d", "--dtype", type=str, nargs="*", default=["bf16"], choices=["bf16"]
    )
    parser.add_argument(
        "--mode", type=str, nargs="*", default=list(BENCH_MODES), choices=BENCH_MODES
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        default=list(BENCH_LAYOUTS),
        choices=BENCH_LAYOUTS,
        help="""How x is laid out. `contiguous` is what Qwen3-Next's torch.cat
        produces and `qkvz_slice` what Qwen3.5 slices out of the fused qkvz
        projection; `verify_view` is SGLang's transposed 3D window and is the only
        layout its mode accepts. Combinations a call site cannot produce are
        skipped.""",
    )
    return parser.parse_args()


def _run_perf_sweep(args):
    rows = []
    # Sweep order, slowest to fastest changing: mode, layout, dtype, width, ...
    for mode, layout, dtype, width, seqlen, dim, batch in itertools.product(
        args.mode,
        args.layout,
        args.dtype,
        args.width,
        args.seqlen,
        args.dim,
        args.batch,
    ):
        # A one-token speculative window is the decode row under another name.
        if (seqlen == 1) != (mode == "decode"):
            continue
        # Only the layouts this call site can actually hand over.
        if layout not in _MODE_LAYOUTS[mode]:
            continue
        rows.append(
            test_causal_conv1d_update_perf(
                batch=batch,
                dim=dim,
                width=width,
                seqlen=seqlen,
                mode=mode,
                layout=layout,
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
    if _SKIP_REASON is not None:
        aiter.logger.warning(
            "%s; skipping the FlyDSL causal_conv1d update tests", _SKIP_REASON
        )
        return 0
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl causal_conv1d_update is unsupported on %s; skipping", get_gfx()
        )
        return 0

    failures = _run_correctness()
    _run_perf_sweep(args)
    return 1 if failures else 0


def _run_correctness():
    torch.manual_seed(0)
    cases = [
        (test_vllm_matches_upstream_vllm, _VLLM_CASES),
        (test_vllm_channels_per_thread_agree, [(1,), (2,)]),
        (test_cpt_policy_backs_off_for_long_speculative_windows, [()]),
        (test_vllm_skips_the_null_block, [()]),
        (test_sglang_matches_upstream_sglang, _SGLANG_CASES),
        (test_sglang_dedup_conv_window_matches_dense, _DEDUP_CASES),
        (test_dispatch_seam_routes_to_flydsl, _SEAM_CASES),
        (test_dispatch_seam_falls_through_when_out_of_scope, [()]),
        (test_triton_update_refuses_unimplemented_width, [()]),
        (test_dispatch_seam_preserves_2d_decode_shape, [()]),
        (test_dispatch_seam_is_off_by_default, [()]),
        (test_vllm_varlen_matches_the_dense_path, _VARLEN_DENSE_EQUIVALENT),
        (test_vllm_varlen_matches_upstream_vllm, _VARLEN_CASES),
        (test_vllm_reads_x_through_its_strides, [()]),
        (test_sglang_reads_the_transposed_verify_window, [()]),
        (test_predicates_accept_what_the_port_covers, [()]),
        (test_out_of_scope_is_refused_rather_than_mishandled, [()]),
        (test_perf_row_agrees_with_the_spec, [(m,) for m in BENCH_MODES]),
    ]

    # CI runs this file, not pytest, so a test that never makes it into `cases`
    # is a test CI does not have. Compare by identity: `@benchmark` returns a
    # bare closure, so the wrapped perf fn's ``__name__`` is not its own.
    reached = {id(fn) for fn, _ in cases} | {id(test_causal_conv1d_update_perf)}
    unreached = sorted(
        name
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj) and id(obj) not in reached
    )
    assert not unreached, f"never run by the CI entry point: {unreached}"

    # Passing cases say nothing a reader needs -- the count and the exit code
    # already do -- and there are enough of them to bury the perf table that is
    # the point of a sweep run. AITER_LOG_MORE, which aiter's own perftest
    # already honours, brings the roll call back for when a run dies partway and
    # the question is which case it died on.
    verbose = bool(int(os.environ.get("AITER_LOG_MORE", "0")))
    failures = cases_run = 0
    for fn, arg_sets in cases:
        for args in arg_sets:
            name = f"{fn.__name__}{args if args else ''}"
            cases_run += 1
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 - report and keep going
                failures += 1
                # Unbuffered: redirected to a file these would otherwise all
                # land after the perf table, which logs to stderr.
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
            else:
                if verbose:
                    print(f"ok   {name}", flush=True)

    print(f"{cases_run} case(s), {failures} failure(s)", flush=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
