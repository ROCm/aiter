# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parity tests for the FlyDSL gated delta rule MTP kernels.

One kernel builder, two upstream contracts: ``flydsl_gdr_mtp`` is vLLM's linear
chain and ``flydsl_gdr_mtp_sglang`` is SGLang's snapshot buffer, with an EAGLE
tree on top of it. A verify pass has to be able to undo the tokens the target
model rejects, and what each upstream keeps where is documented on
``create_vk_gdr_mtp_kernel``.

The reference is the upstream kernel itself
-------------------------------------------
Each contract is checked against the kernel it claims to replace: vLLM's fused
gating plus its ``fused_recurrent_gated_delta_rule``, and SGLang's
``fused_sigmoid_gating_delta_rule_update``, vendored verbatim into
``op_tests/triton_tests/utils/gdr_mtp_refs.py``. Nothing here re-derives what
the recurrence should do.

Nothing imports ``vllm`` or ``sglang``: the suite must not depend on either
being installed, and a live import would re-point the oracle whenever the
installed version moved.

aiter's own Triton kernel implements the chain contract only, so it is a
candidate in the perf table and the incumbent at the dispatch seam, but never
the oracle.

Each upstream is run three times per case -- at the tested dtype, at fp32 for
the spec, and at fp32 on absolute values for a conditioning scale. ``_Oracle``
holds the three, and ``_magnitudes``, ``_assert_tracks_spec`` and
``_assert_no_worse_than`` each say what their own bound is worth.

Bit-exactness is asserted only where one state is reached by two address
computations, which is what makes a wrapped offset observable at all; see
``test_chain_checkpoints_and_snapshots_are_the_same_state`` and the
``test_*_past_2gib_elements`` pair.

Usage:
    HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/test_flydsl_gdr_mtp.py

    # or, the way CI invokes it -- the same checks with no pytest dependency,
    # followed by the perf sweep:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_gdr_mtp.py

    # narrow the perf sweep:
    HIP_VISIBLE_DEVICES=7 python3 op_tests/test_flydsl_gdr_mtp.py \
        --mode vllm_chain -b 32 -s 4
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import statistics
import sys
import traceback
from typing import NamedTuple

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

# CI runs this as `python3 op_tests/<file>`, which puts op_tests/ on sys.path
# rather than the repo root, so the vendored upstream kernels below would not
# resolve. Same line as op_tests/test_gemm_a8w8_blockscale.py:9.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.triton_tests.utils.gdr_mtp_refs import (
    fused_gdn_gating_vllm,
    fused_recurrent_gated_delta_rule_vllm,
    fused_sigmoid_gating_delta_rule_update_sglang,
)

_SKIP_REASON = None
if not torch.cuda.is_available():
    _SKIP_REASON = "ROCm is not available"
else:
    # `aiter.ops.flydsl` raises on a flydsl that is missing or too old, so the
    # import below is also the availability check.
    try:
        from aiter.ops.flydsl.kernels.gdr_decode import (
            MTP_MODE_SNAPSHOT,
            create_vk_gdr_mtp_kernel,
        )
        from aiter.ops.flydsl.kernels.tensor_shim import get_dtype_str
        from aiter.ops.flydsl.linear_attention_kernels import (
            _SUPPORTED_DTYPES,
            _SUPPORTED_STATE_DTYPES,
            _flydsl_gdr_mtp_sglang_supported,
            flydsl_gdr_mtp,
            flydsl_gdr_mtp_sglang,
        )
        from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
            _flydsl_gdr_enabled,
            _uniform_draft_window,
            fused_rearrange_sigmoid_gated_delta_rule,
        )
    except ImportError as exc:
        _SKIP_REASON = f"the FlyDSL GDR MTP kernels do not import ({exc})"

# Only under pytest. CI also shards op_tests with `python3 <file>`, where a
# module-level skip raises Skipped with nobody to catch it and the shard reports
# a failure; main() handles that case.
if _SKIP_REASON is not None and __name__ != "__main__":
    pytest.skip(
        f"{_SKIP_REASON}. Skipping the FlyDSL GDR MTP tests.",
        allow_module_level=True,
    )


DEVICE = "cuda"
DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32

#: Targets the kernels are built for; anything else skips.
SUPPORTED_GFX = ("gfx942", "gfx950")

#: Head counts and widths of a Qwen3-Next-shaped model, which is the deployment
#: target. The correctness cases shrink the head counts to keep the pool small
#: -- a slot is ``num_v_heads * head_v_dim * head_k_dim`` and the chain wants one
#: per draft token -- but never the head widths, which drive the tiling.
HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 2
NUM_V_HEADS = 4

#: The gating constants both upstreams hard-code at the call site.
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

#: One mantissa step of the storage dtype: 2**-(mantissa bits + 1).
_DTYPE_EPS = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_DTYPE_BY_NAME = {"bf16": torch.bfloat16, "fp16": torch.float16}

#: The state dtype is the one that moves the bytes: at the large-batch rows the
#: pool is most of the traffic.
_STATE_DTYPE_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16"}
_STATE_DTYPE_BY_NAME = {v: k for k, v in _STATE_DTYPE_NAME.items()}

#: How many of those steps a result may accumulate, scaled per element by the
#: conditioning bound from `_magnitudes`. Loose because the recurrence compounds
#: over the draft window; the binding check is `_assert_no_worse_than`.
_ERR_FACTOR = 64.0

#: vLLM's wrapper reads slot 0 as "no state", so live slots start at 1.
NULL_BLOCK_ID = 0


# -- inputs ---------------------------------------------------------------


class _Problem(NamedTuple):
    """One MTP problem, in the shapes both entry points want."""

    q: torch.Tensor  # [B, T, H, K]
    k: torch.Tensor
    v: torch.Tensor  # [B, T, HV, V]
    a: torch.Tensor  # [B, T, HV]
    b: torch.Tensor
    dt_bias: torch.Tensor  # [HV]
    A_log: torch.Tensor
    pool: torch.Tensor  # [slots, HV, V, K]
    chain_indices: torch.Tensor  # [B, T]  vLLM: a slot per draft token
    seq_indices: torch.Tensor  # [B]     SGLang: one slot per sequence
    num_accepted: torch.Tensor  # [B]
    cu_seqlens: torch.Tensor  # [B + 1]
    batch: int
    seqlen: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int


def _make_problem(
    batch,
    seqlen,
    *,
    seed,
    accepted="full",
    dtype=DTYPE,
    state_dtype=STATE_DTYPE,
    num_k_heads=NUM_K_HEADS,
    num_v_heads=NUM_V_HEADS,
    head_k_dim=HEAD_K_DIM,
    head_v_dim=HEAD_V_DIM,
    pool_slots=None,
):
    """A random problem plus the index vectors both contracts need.

    The chain contract needs a distinct slot per draft token, so the pool is
    ``batch * seqlen`` slots and one spare: slot 0 is vLLM's null block and
    handing it out as a live slot would silently skip a sequence.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    tokens = batch * seqlen

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=dt)

    slots = pool_slots if pool_slots is not None else tokens + 1
    if accepted == "full":
        nacc = torch.full((batch,), seqlen, device=DEVICE, dtype=torch.int32)
    elif accepted == "first":
        nacc = torch.ones(batch, device=DEVICE, dtype=torch.int32)
    elif accepted == "mixed":
        # Every sequence rolls back to a different point, which is the case
        # that tells a per-sequence read apart from a shared one.
        nacc = (torch.arange(batch, device=DEVICE, dtype=torch.int32) % seqlen) + 1
    else:
        raise ValueError(f"unknown accepted pattern {accepted!r}")

    return _Problem(
        q=rnd(batch, seqlen, num_k_heads, head_k_dim),
        k=rnd(batch, seqlen, num_k_heads, head_k_dim),
        v=rnd(batch, seqlen, num_v_heads, head_v_dim),
        a=rnd(batch, seqlen, num_v_heads),
        b=rnd(batch, seqlen, num_v_heads),
        dt_bias=rnd(num_v_heads),
        A_log=rnd(num_v_heads, dt=torch.float32),
        pool=rnd(slots, num_v_heads, head_v_dim, head_k_dim, dt=state_dtype),
        chain_indices=torch.arange(
            1, tokens + 1, device=DEVICE, dtype=torch.int32
        ).view(batch, seqlen),
        seq_indices=torch.arange(1, batch + 1, device=DEVICE, dtype=torch.int32),
        num_accepted=nacc,
        cu_seqlens=torch.arange(
            0, (batch + 1) * seqlen, seqlen, device=DEVICE, dtype=torch.int32
        ),
        batch=batch,
        seqlen=seqlen,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )


def _eagle_tree(batch, seqlen):
    """A parent map that is a tree rather than a chain, for ``seqlen >= 4``.

    Token 0 is the root and its parent entry is unread. Tokens 1 and 2 both
    hang off the root, which is what makes it a tree: token 2 must restart from
    the root's snapshot rather than from token 1. The rest chain onto token 1,
    so the map is neither a pure chain nor a pure star.
    """
    parents = [0] * seqlen
    for t in range(1, seqlen):
        parents[t] = 0 if t <= 2 else t - 2
    return torch.tensor([parents] * batch, device=DEVICE, dtype=torch.int32)


def _chain_parents(batch, seqlen):
    """The parent map that makes SGLang's tree path compute the chain."""
    parents = [max(t - 1, 0) for t in range(seqlen)]
    return torch.tensor([parents] * batch, device=DEVICE, dtype=torch.int32)


# -- conditioning ---------------------------------------------------------


def _magnitudes(p: _Problem, *, init_slots, use_qk_l2norm=True, parents=None):
    """The size of the terms the recurrence sums, in fp64.

    An error budget has to be scaled by how big the intermediate terms are, not
    by how big the answer is: the delta rule subtracts ``k @ h`` from ``v``, so
    an output near zero can be the difference of two large numbers and a budget
    read off the output would be far too tight there.

    Re-running upstream on ``abs()``-ed inputs does not give that number.
    ``g = -exp(A_log) * softplus(a + dt_bias)`` is negative by construction, so
    ``abs(A_log)`` strengthens the decay and reports magnitudes below the true
    ones. The magnitudes are therefore propagated here alongside the real
    recurrence, with the real ``g`` and ``beta`` and every signed accumulation
    replaced by an absolute one. Returns the bound for the outputs and for the
    state after each token.
    """
    B, T = p.batch, p.seqlen
    HV, K = p.num_v_heads, p.head_k_dim
    rep = HV // p.num_k_heads

    q, k = p.q.double(), p.k.double()
    if use_qk_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    # Grouped-value attention repeats each k head over its v heads, which is the
    # `i_h = i_hv // (HV // H)` the upstream kernels index with.
    q = (q * K**-0.5).abs().repeat_interleave(rep, dim=2)
    k = k.abs().repeat_interleave(rep, dim=2)
    v = p.v.double().abs()

    x = p.a.double() + p.dt_bias.double()
    softplus = torch.where(
        SOFTPLUS_BETA * x <= SOFTPLUS_THRESHOLD,
        torch.log1p(torch.exp(SOFTPLUS_BETA * x)) / SOFTPLUS_BETA,
        x,
    )
    decay = torch.exp(-torch.exp(p.A_log.double()) * softplus)
    beta = torch.sigmoid(p.b.double())

    mag = p.pool.double().abs()[init_slots.long()]
    steps = torch.empty(
        B, T, HV, p.head_v_dim, K, dtype=torch.float64, device=p.pool.device
    )
    out = torch.empty(B, T, HV, p.head_v_dim, dtype=torch.float64, device=p.pool.device)
    for t in range(T):
        if parents is not None and t > 0:
            mag = steps[torch.arange(B, device=steps.device), parents[:, t].long()]
        mag = mag * decay[:, t][..., None, None]
        u = v[:, t] + torch.einsum("bhvk,bhk->bhv", mag, k[:, t])
        mag = mag + (u * beta[:, t][..., None])[..., None] * k[:, t][:, :, None, :]
        out[:, t] = torch.einsum("bhvk,bhk->bhv", mag, q[:, t])
        steps[:, t] = mag
    return out, steps


# -- oracles --------------------------------------------------------------


class _Oracle(NamedTuple):
    """What one upstream kernel says about a problem.

    The ``spec_`` fields are the fp32 run in fp64 and are what a result is held
    to; the ``scale_`` fields are the conditioning bounds from ``_magnitudes``
    that size its budget. ``out``, ``pool`` and ``inter`` come from the run at
    the tested dtype and are the baseline the port may not do worse than.
    """

    spec: torch.Tensor
    scale: torch.Tensor
    spec_pool: torch.Tensor
    scale_pool: torch.Tensor
    spec_inter: torch.Tensor | None
    scale_inter: torch.Tensor | None
    out: torch.Tensor
    pool: torch.Tensor
    inter: torch.Tensor | None


def _oracle_vllm(p: _Problem, *, use_qk_l2norm=True):
    """vLLM's gating plus its recurrence (rev ``63a9a5010``), as the oracle.

    Upstream splits gating out of the recurrence where the port and SGLang fuse
    it, so both halves are vendored and both run here: comparing only the
    recurrence would leave the port's fused softplus and sigmoid unchecked.

    The recurrence is varlen-only in this configuration, so the batch is packed
    into one row of ``cu_seqlens``-delimited tokens, which is how vLLM's own
    caller hands it over.
    """
    B, T = p.batch, p.seqlen
    H, HV = p.num_k_heads, p.num_v_heads
    K, V = p.head_k_dim, p.head_v_dim
    tokens = B * T
    scale_factor = K**-0.5

    def run(q, k, v, a, b, dt_bias, A_log, pool):
        g, beta = fused_gdn_gating_vllm(
            A_log,
            a.reshape(tokens, HV),
            b.reshape(tokens, HV),
            dt_bias,
            SOFTPLUS_BETA,
            SOFTPLUS_THRESHOLD,
        )
        state = pool.clone()
        out, _ = fused_recurrent_gated_delta_rule_vllm(
            q=q.reshape(1, tokens, H, K),
            k=k.reshape(1, tokens, H, K),
            v=v.reshape(1, tokens, HV, V),
            g=g.view(1, tokens, HV),
            beta=beta.view(1, tokens, HV),
            scale=scale_factor,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=p.cu_seqlens,
            ssm_state_indices=p.chain_indices,
            num_accepted_tokens=p.num_accepted,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        return out.reshape(B, T, HV, V), state

    out, pool = run(p.q, p.k, p.v, p.a, p.b, p.dt_bias, p.A_log, p.pool)
    spec, spec_pool = run(
        p.q.float(),
        p.k.float(),
        p.v.float(),
        p.a.float(),
        p.b.float(),
        p.dt_bias.float(),
        p.A_log.float(),
        p.pool.float(),
    )
    rows = torch.arange(B, device=p.pool.device)
    scale, steps = _magnitudes(
        p,
        init_slots=p.chain_indices[rows, p.num_accepted.long() - 1],
        use_qk_l2norm=use_qk_l2norm,
    )
    # The chain checkpoints token t into its own slot, so the bound on that slot
    # is the bound on the state after token t.
    scale_pool = p.pool.double().abs()
    scale_pool[p.chain_indices.long().flatten()] = steps.flatten(0, 1)
    return _Oracle(
        spec.double(),
        scale,
        spec_pool.double(),
        scale_pool,
        None,
        None,
        out,
        pool,
        None,
    )


def _oracle_sglang(
    p: _Problem,
    *,
    use_qk_l2norm=True,
    save_inter=False,
    parents=None,
    disable_state_update=False,
):
    """SGLang's fused kernel (rev ``18107e38d2``), as the oracle.

    Its snapshot buffer is allocated ``[slot, step, HV, K, V]`` but addressed at
    ``v * K + k``, so the same bytes are the port's ``[slot, step, HV, V, K]``.
    The view, not a transpose: the two agree on the layout and disagree only on
    the name, and copying would hide a real stride bug.
    """
    B, T = p.batch, p.seqlen
    HV, K, V = p.num_v_heads, p.head_k_dim, p.head_v_dim
    tokens = B * T

    def run(q, k, v, a, b, dt_bias, A_log, pool):
        state = pool.clone()
        inter = (
            torch.zeros(B, T, HV, K, V, device=DEVICE, dtype=state.dtype)
            if save_inter
            else None
        )
        out = fused_sigmoid_gating_delta_rule_update_sglang(
            A_log=A_log,
            a=a.reshape(tokens, HV),
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=p.seq_indices,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=inter,
            intermediate_state_indices=(
                torch.arange(B, device=DEVICE, dtype=torch.int32)
                if save_inter
                else None
            ),
            retrieve_parent_token=parents,
        )
        return out, state, (None if inter is None else inter.view(B, T, HV, V, K))

    out, pool, inter = run(p.q, p.k, p.v, p.a, p.b, p.dt_bias, p.A_log, p.pool)
    spec, spec_pool, spec_inter = run(
        p.q.float(),
        p.k.float(),
        p.v.float(),
        p.a.float(),
        p.b.float(),
        p.dt_bias.float(),
        p.A_log.float(),
        p.pool.float(),
    )
    scale, steps = _magnitudes(
        p,
        init_slots=p.seq_indices,
        use_qk_l2norm=use_qk_l2norm,
        parents=parents,
    )
    # A sequence commits its last step back into the slot it started from.
    scale_pool = p.pool.double().abs()
    if not disable_state_update:
        scale_pool[p.seq_indices.long()] = steps[:, -1]
    return _Oracle(
        spec.double(),
        scale,
        spec_pool.double(),
        scale_pool,
        None if spec_inter is None else spec_inter.double(),
        steps if save_inter else None,
        out,
        pool,
        inter,
    )


def _run_flydsl_chain(p: _Problem, *, use_qk_l2norm=True, inplace=False):
    """``inplace`` updates the problem's own pool instead of a copy.

    Only the >2**31 probe wants it, and it wants it because the copy would
    double an already multi-gigabyte allocation.
    """
    pool = p.pool if inplace else p.pool.clone()
    out = torch.empty_like(p.v)
    flydsl_gdr_mtp(
        query=p.q,
        key=p.k,
        value=p.v,
        a=p.a,
        b=p.b,
        dt_bias=p.dt_bias,
        A_log=p.A_log,
        state=pool,
        out=out,
        ssm_state_indices=p.chain_indices,
        num_accepted_tokens=p.num_accepted,
        use_qk_l2norm=use_qk_l2norm,
    )
    return out, pool


def _run_flydsl_sglang(
    p: _Problem,
    *,
    use_qk_l2norm=True,
    save_inter=False,
    parents=None,
    disable_state_update=False,
    inter_indices=None,
    inter=None,
    inplace=False,
):
    pool = p.pool if inplace else p.pool.clone()
    out = torch.empty_like(p.v)
    if save_inter and inter is None:
        inter = torch.zeros(
            p.batch,
            p.seqlen,
            p.num_v_heads,
            p.head_v_dim,
            p.head_k_dim,
            device=DEVICE,
            dtype=pool.dtype,
        )
    if save_inter and inter_indices is None:
        inter_indices = torch.arange(p.batch, device=DEVICE, dtype=torch.int32)
    flydsl_gdr_mtp_sglang(
        query=p.q,
        key=p.k,
        value=p.v,
        a=p.a,
        b=p.b,
        dt_bias=p.dt_bias,
        A_log=p.A_log,
        state=pool,
        out=out,
        initial_state_indices=p.seq_indices,
        intermediate_states_buffer=inter if save_inter else None,
        intermediate_state_indices=inter_indices if save_inter else None,
        retrieve_parent_token=parents,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
    )
    return out, pool, inter


# -- shared assertions ----------------------------------------------------


def _assert_tracks_spec(label, actual, spec, scale, factor=_ERR_FACTOR):
    """``actual`` must sit inside the conditioning-scaled error budget."""
    name = _DTYPE_NAME.get(actual.dtype, str(actual.dtype))
    err = (actual.double() - spec).abs()
    eps = _DTYPE_EPS.get(actual.dtype, 2.0**-24)
    budget = factor * eps * scale.clamp_min(scale.abs().max() * 1e-6)
    worst = (err / budget.clamp_min(torch.finfo(torch.float64).tiny)).argmax()
    assert (err <= budget).all(), (
        f"{label}: output exceeds {factor} {name} steps of its own conditioning; "
        f"worst element {worst.item()}: got {actual.flatten()[worst].item():.6g}, "
        f"spec {spec.flatten()[worst].item():.6g}, "
        f"error {err.flatten()[worst].item():.3e} > budget "
        f"{budget.flatten()[worst].item():.3e} "
        f"(term magnitude {scale.flatten()[worst].item():.3e})"
    )


def _assert_no_worse_than(
    label, flydsl, other, spec, scale, other_name, slack_steps=2.0
):
    """FlyDSL must track the spec at least as closely as ``other`` does.

    The two backends round differently, so requiring them to agree would pin the
    less accurate one's error. A few steps of the element's own conditioning are
    allowed as slack, so a rounding tie cannot fail this while a real regression
    still does.

    ``slack_steps`` is how many of those steps a difference in summation order
    alone can account for. Two is right for a value both sides round to a narrow
    dtype, where that rounding dominates; the states are kept at fp32, where
    reassociating the ``k @ h`` reduction dominates instead, so callers pass
    ``head_k_dim``.
    """
    err_fly = (flydsl.double() - spec).abs()
    err_other = (other.double() - spec).abs()
    eps = _DTYPE_EPS.get(flydsl.dtype, 2.0**-24)
    slack = slack_steps * eps * scale
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
        "both are the same state written by two address computations, so this "
        "is placement, not rounding"
    )


def _touched_slots(p: _Problem):
    """The pool rows the chain contract writes, in ``[B, T, ...]`` order."""
    return p.chain_indices.reshape(-1).long()


# -- vLLM's chain contract ------------------------------------------------

#: (batch, seqlen, accepted). ``accepted`` picks where each sequence rolls back
#: to: the whole window, only the first token, or a different point per sequence.
_CHAIN_CASES = [
    (1, 2, "full"),
    (2, 2, "first"),
    (3, 2, "mixed"),
    (1, 4, "full"),
    (4, 4, "first"),
    (5, 4, "mixed"),
    (8, 4, "full"),
    (3, 8, "mixed"),
]


def _check_chain_case(batch, seqlen, accepted, *, dtype=DTYPE, use_qk_l2norm=True):
    p = _make_problem(
        batch, seqlen, seed=batch * 131 + seqlen * 7 + len(accepted), dtype=dtype
    )
    label = f"chain b{batch} s{seqlen} {accepted} {_DTYPE_NAME[dtype]}"
    ref = _oracle_vllm(p, use_qk_l2norm=use_qk_l2norm)
    out, pool = _run_flydsl_chain(p, use_qk_l2norm=use_qk_l2norm)

    _assert_tracks_spec(f"{label}: out", out, ref.spec, ref.scale)
    _assert_no_worse_than(f"{label}: out", out, ref.out, ref.spec, ref.scale, "vLLM")

    # Only the rows the contract touches: the rest of the pool is untouched in
    # both runs and comparing it would pass on the seed rather than the kernel.
    slots = _touched_slots(p)
    got, want = pool[slots], ref.pool[slots]
    _assert_tracks_spec(
        f"{label}: checkpoints", got, ref.spec_pool[slots], ref.scale_pool[slots]
    )
    _assert_no_worse_than(
        f"{label}: checkpoints",
        got,
        want,
        ref.spec_pool[slots],
        ref.scale_pool[slots],
        "vLLM",
        slack_steps=p.head_k_dim,
    )
    assert torch.isfinite(got).all(), f"{label}: checkpoints have non-finite values"
    _assert_untouched_outside(label, pool, ref.pool, slots)
    return got, want


def _assert_untouched_outside(label, pool, ref_pool, slots):
    """No slot outside the contract's own may be written.

    This is the cheap half of the addressing check and it runs on every case: a
    store that wraps has to land somewhere, and unless it lands on another
    touched slot it lands here.
    """
    mask = torch.ones(pool.shape[0], dtype=torch.bool, device=pool.device)
    mask[slots] = False
    if mask.any():
        _assert_bit_exact(
            label, "pool outside the written slots", pool[mask], ref_pool[mask]
        )


@pytest.mark.parametrize("batch,seqlen,accepted", _CHAIN_CASES)
def test_chain_matches_upstream_vllm(batch, seqlen, accepted):
    """The chain contract, against vLLM's gating and recurrence."""
    _check_chain_case(batch, seqlen, accepted)


@pytest.mark.parametrize("dtype", sorted(_SUPPORTED_DTYPES, key=str))
@pytest.mark.parametrize("use_qk_l2norm", [False, True])
def test_chain_matches_upstream_across_dtypes(dtype, use_qk_l2norm):
    """Both input dtypes and both normalisation settings the port advertises."""
    _check_chain_case(4, 4, "mixed", dtype=dtype, use_qk_l2norm=use_qk_l2norm)


@pytest.mark.parametrize("state_dtype", sorted(_SUPPORTED_STATE_DTYPES, key=str))
def test_chain_matches_upstream_across_state_dtypes(state_dtype):
    """The pool's dtype is the caller's, and it changes the kernel's tiling.

    ``VALUES_PER_THREAD_K`` is 4 for an fp32 pool and 8 for a narrow one, so the
    two are different kernels and the narrow one is not covered by anything
    above.
    """
    p = _make_problem(4, 4, seed=99, accepted="mixed", state_dtype=state_dtype)
    label = f"chain state={state_dtype}"
    ref = _oracle_vllm(p)
    out, pool = _run_flydsl_chain(p)
    _assert_tracks_spec(f"{label}: out", out, ref.spec, ref.scale)
    slots = _touched_slots(p)
    _assert_tracks_spec(
        f"{label}: checkpoints",
        pool[slots],
        ref.spec_pool[slots],
        ref.scale_pool[slots],
    )


def test_chain_rolls_back_to_the_accepted_token():
    """``num_accepted`` must select the slot the next pass resumes from.

    Runs the same window twice from the same pool, once accepting everything and
    once accepting one token, and requires the answers to differ from the second
    token on. Without this a kernel that ignored ``num_accepted`` and always
    read slot ``[n, 0]`` would pass every value check above whenever the accept
    count happened to be 1.
    """
    p = _make_problem(4, 4, seed=7, accepted="full")
    full_out, full_pool = _run_flydsl_chain(p)

    partial = p._replace(
        num_accepted=torch.ones(p.batch, device=DEVICE, dtype=torch.int32)
    )
    part_out, part_pool = _run_flydsl_chain(partial)

    # Token 0 reads its own rollback slot in both runs only when the accept
    # count is 1 there too, so the first token is where they may agree.
    diff = (full_out.float() - part_out.float()).abs().amax(dim=(0, 2, 3))
    assert diff[1:].min() > 0, (
        "rolling back to a different token changed nothing from token 1 on: "
        f"per-token max|delta| = {diff.tolist()}"
    )
    assert not torch.equal(
        full_pool, part_pool
    ), "the checkpoints are identical across two different rollback points"

    # And the value is right, not merely different.
    ref = _oracle_vllm(partial)
    _assert_no_worse_than("rollback", part_out, ref.out, ref.spec, ref.scale, "vLLM")


def test_chain_skips_the_null_block():
    """Slot 0 is vLLM's null block: neither read nor written.

    aiter's Triton and SGLang both use a negative sentinel instead, and the port
    accepts both, so the negative case is covered alongside.
    """
    for sentinel in (NULL_BLOCK_ID, -1):
        p = _make_problem(4, 4, seed=11, accepted="full")
        idx = p.chain_indices.clone()
        idx[1] = sentinel  # sequence 1 is a graph-capture pad row
        p = p._replace(chain_indices=idx)

        out, pool = _run_flydsl_chain(p)
        live = _make_problem(4, 4, seed=11, accepted="full")
        live_out, _ = _run_flydsl_chain(live)

        assert torch.equal(out[0], live_out[0]) and torch.equal(
            out[2], live_out[2]
        ), f"sentinel {sentinel}: skipping a sequence disturbed its neighbours"
        touched = torch.ones(pool.shape[0], dtype=torch.bool, device=DEVICE)
        touched[idx[idx > 0].reshape(-1).long()] = False
        _assert_bit_exact(
            f"sentinel {sentinel}",
            "the skipped sequence's slots",
            pool[touched],
            p.pool[touched],
        )


# -- SGLang's snapshot contract -------------------------------------------

#: (batch, seqlen, save_inter, tree). Without a snapshot buffer the kernel is a
#: straight-through recurrence with one write-back; with one it also records
#: every step; with a tree it reloads a parent's record per token.
_SGLANG_CASES = [
    (1, 2, False, False),
    (2, 2, True, False),
    (3, 4, False, False),
    (4, 4, True, False),
    (4, 4, True, True),
    (8, 4, True, True),
    (3, 8, True, True),
]


def _check_sglang_case(batch, seqlen, save_inter, tree, *, dtype=DTYPE):
    p = _make_problem(batch, seqlen, seed=batch * 17 + seqlen, dtype=dtype)
    parents = _eagle_tree(batch, seqlen) if tree else None
    label = (
        f"sglang b{batch} s{seqlen} "
        f"{'tree' if tree else 'chain'}{' +snap' if save_inter else ''}"
    )
    ref = _oracle_sglang(p, save_inter=save_inter, parents=parents)
    out, pool, inter = _run_flydsl_sglang(p, save_inter=save_inter, parents=parents)

    _assert_tracks_spec(f"{label}: out", out, ref.spec, ref.scale)
    _assert_no_worse_than(f"{label}: out", out, ref.out, ref.spec, ref.scale, "SGLang")

    slots = p.seq_indices.long()
    _assert_tracks_spec(
        f"{label}: state", pool[slots], ref.spec_pool[slots], ref.scale_pool[slots]
    )
    _assert_no_worse_than(
        f"{label}: state",
        pool[slots],
        ref.pool[slots],
        ref.spec_pool[slots],
        ref.scale_pool[slots],
        "SGLang",
        slack_steps=p.head_k_dim,
    )
    _assert_untouched_outside(label, pool, ref.pool, slots)
    if save_inter:
        _assert_tracks_spec(
            f"{label}: snapshots", inter, ref.spec_inter, ref.scale_inter
        )
        _assert_no_worse_than(
            f"{label}: snapshots",
            inter,
            ref.inter,
            ref.spec_inter,
            ref.scale_inter,
            "SGLang",
            slack_steps=p.head_k_dim,
        )
    return out, pool, inter


@pytest.mark.parametrize("batch,seqlen,save_inter,tree", _SGLANG_CASES)
def test_sglang_matches_upstream_sglang(batch, seqlen, save_inter, tree):
    """The snapshot contract, against SGLang's own fused kernel."""
    _check_sglang_case(batch, seqlen, save_inter, tree)


@pytest.mark.parametrize("dtype", sorted(_SUPPORTED_DTYPES, key=str))
def test_sglang_matches_upstream_across_dtypes(dtype):
    _check_sglang_case(4, 4, True, True, dtype=dtype)


def test_sglang_disable_state_update_leaves_the_pool_alone():
    """A verify pass records every step and commits none of them."""
    p = _make_problem(4, 4, seed=23)
    parents = _eagle_tree(4, 4)
    ref = _oracle_sglang(p, save_inter=True, parents=parents, disable_state_update=True)
    out, pool, inter = _run_flydsl_sglang(
        p, save_inter=True, parents=parents, disable_state_update=True
    )

    _assert_bit_exact("disable_state_update", "the pool", pool, p.pool)
    _assert_bit_exact("disable_state_update", "upstream's pool", ref.pool, p.pool)
    _assert_tracks_spec("disable_state_update: out", out, ref.spec, ref.scale)
    _assert_tracks_spec(
        "disable_state_update: snapshots", inter, ref.spec_inter, ref.scale_inter
    )


def test_sglang_tree_is_not_running_a_chain():
    """The tree path must actually branch.

    A tree and a chain over the same tokens differ only from the first token
    whose parent is not its predecessor, so a dead tree path -- one that
    silently ran the chain -- would agree everywhere. Requiring the two to
    *disagree* from that token on is what makes the tree cases above mean
    something.
    """
    p = _make_problem(2, 4, seed=31)
    chain = _chain_parents(2, 4)
    tree = _eagle_tree(2, 4)
    assert torch.equal(chain[:, :2], tree[:, :2]), "the maps must share a prefix"
    assert not torch.equal(chain, tree), "the maps must diverge somewhere"

    out_chain, _, _ = _run_flydsl_sglang(p, save_inter=True, parents=chain)
    out_tree, _, _ = _run_flydsl_sglang(p, save_inter=True, parents=tree)
    per_token = (out_chain.float() - out_tree.float()).abs().amax(dim=(0, 2, 3))

    first_split = min(t for t in range(4) if chain[0, t] != tree[0, t])
    assert per_token[:first_split].max() == 0, (
        "tokens before the parent maps diverge must be identical: "
        f"{per_token.tolist()}"
    )
    assert per_token[first_split:].min() > 0, (
        "the tree path produced the chain's answer, so it is not being taken: "
        f"{per_token.tolist()}"
    )


def test_sglang_tree_without_a_snapshot_slot_runs_the_chain():
    """The second traced body: no snapshot slot means nothing to reload.

    The builder emits the tree loop twice, once reloading parents and once not,
    and picks between them on ``cache_idx`` at kernel entry. The second copy has
    to behave like the plain chain and write no snapshots at all.
    """
    p = _make_problem(2, 4, seed=37)
    tree = _eagle_tree(2, 4)
    no_slot = torch.full((2,), -1, device=DEVICE, dtype=torch.int32)

    out, _, inter = _run_flydsl_sglang(
        p, save_inter=True, parents=tree, inter_indices=no_slot
    )
    plain, _, _ = _run_flydsl_sglang(p, save_inter=False, parents=None)

    _assert_bit_exact(
        "tree without a snapshot slot",
        "the snapshot buffer",
        inter,
        torch.zeros_like(inter),
    )
    _assert_bit_exact("tree without a snapshot slot", "the output", out, plain)


def test_a_snapshot_wider_than_the_state_is_refused_at_every_layer():
    """A snapshot dtype wider than the state's has no store to lower to.

    The lane count is chosen so the *state* vector is 16 bytes -- 4 fp32 or 8
    bf16 -- and the snapshot reuses it with its own element. A bf16 state and an
    fp32 snapshot therefore ask for eight fp32 in one store, twice what a buffer
    op carries, which aborts the compile rather than raising. An aborted process
    cannot be caught, so every layer reachable from outside has to refuse it:
    the dispatch screen by declining, the API and the builder by raising.
    """
    p = _make_problem(2, 4, seed=41, state_dtype=torch.bfloat16)
    wide = torch.zeros(
        p.batch,
        p.seqlen,
        p.num_v_heads,
        p.head_v_dim,
        p.head_k_dim,
        device=DEVICE,
        dtype=torch.float32,
    )
    assert wide.dtype.itemsize > p.pool.dtype.itemsize, "the case under test"

    # The screen the dispatch seam consults: decline, so the caller keeps its
    # own kernel rather than being handed an error.
    assert not _flydsl_gdr_mtp_sglang_supported(
        p.q, p.k, p.v, p.pool, p.seq_indices, wide, p.seq_indices, None
    )

    # The API, which a caller may reach without going through the screen.
    with pytest.raises(ValueError, match="cannot be wider than"):
        _run_flydsl_sglang(p, save_inter=True, parents=_eagle_tree(2, 4), inter=wide)

    # The builder, which is the layer that cannot express the store. Strides
    # come off the real tensors so the call stays valid as the signature moves.
    with pytest.raises(AssertionError, match="cannot be wider than the state"):
        create_vk_gdr_mtp_kernel(
            get_dtype_str(p.q.dtype),
            get_dtype_str(p.A_log.dtype),
            get_dtype_str(p.pool.dtype),
            get_dtype_str(wide.dtype),
            p.seqlen,
            p.num_k_heads,
            p.num_v_heads,
            p.head_k_dim,
            p.head_v_dim,
            p.q.stride(),
            p.k.stride(),
            p.v.stride(),
            p.pool.stride(),
            p.a.stride(),
            p.b.stride(),
            tuple(p.seq_indices.stride()) + (1,),
            tuple(wide.stride()),
            (),
            True,
            MTP_MODE_SNAPSHOT,
        )


# -- the two address computations, against each other ---------------------


@contextlib.contextmanager
def _one_tiling_for_every_contract():
    """Drop every contract onto the same tiling, leaving addressing the only
    difference between them.

    The tuned table is keyed by contract, so at a shape where one contract has
    a row and another does not, the two run different tilings -- and a tiling
    sets how the recurrence is split across lanes, so it sets the order the
    partial products are summed in. Different orders differ by a few ulp, which
    is arithmetic, not placement, and it would sit on top of exactly the signal
    the bit-exact comparison below is reading. Emptying the table drops both
    onto the rule, which does not take a contract and so answers both the same.
    """
    import aiter.ops.flydsl.linear_attention_kernels as lak

    saved = lak.GDR_GLOBAL_CONFIG_MAP
    lak.GDR_GLOBAL_CONFIG_MAP = {}
    # The lookup memoises on the shape, not on the table, so a shape already
    # asked for would come back with the row still applied.
    lak._mtp_kwargs.cache_clear()
    try:
        yield
    finally:
        lak.GDR_GLOBAL_CONFIG_MAP = saved
        lak._mtp_kwargs.cache_clear()


def test_chain_checkpoints_and_snapshots_are_the_same_state():
    """The comparison that catches an addressing bug rather than a value bug.

    Both modes run one identical recurrence over the same tokens. The chain
    writes h after token t into ``state[indices[n, t]]`` through ``_state_at``;
    the snapshot mode writes the same registers into ``inter[cache[n], t]``
    through ``_inter_at``. So they are the same bytes reached two ways, and must
    agree exactly.

    A value check cannot do this job: if an address wraps, the read and the
    write wrap together, the arithmetic still closes, and only the placement is
    wrong. Holding two independent address computations against each other is
    what makes the wrap observable.

    Both runs are held to one tiling so that the comparison stays a comparison
    of addresses; see ``_one_tiling_for_every_contract``.
    """
    for batch, seqlen in ((1, 2), (4, 4), (3, 8)):
        p = _make_problem(batch, seqlen, seed=batch * 5 + seqlen, accepted="first")
        # The chain rolls back to the slot for token 0, which holds the state
        # *after* token 0, not before it. Point the snapshot run at the same
        # place by giving it that slot as its single sequence slot.
        with _one_tiling_for_every_contract():
            chain_out, chain_pool = _run_flydsl_chain(p)

            # A column of the slot map, copied rather than viewed: a length-1
            # slice keeps the row pitch as its stride and torch still calls it
            # contiguous.
            slot0 = torch.empty_like(p.seq_indices).copy_(p.chain_indices[:, 0])
            snap = p._replace(seq_indices=slot0)
            snap_out, _, inter = _run_flydsl_sglang(snap, save_inter=True)

        # Both start from the same state, so token 0's answers agree and every
        # checkpoint does. The chain's slot for token t holds the same value the
        # snapshot's step t does.
        got = chain_pool[_touched_slots(p)].view(
            batch, seqlen, p.num_v_heads, p.head_v_dim, p.head_k_dim
        )
        _assert_bit_exact(
            f"b{batch} s{seqlen}", "chain checkpoints against snapshots", got, inter
        )
        _assert_bit_exact(f"b{batch} s{seqlen}", "the outputs", chain_out, snap_out)


def _skip_unless_free_hbm(need, what):
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(
            f"{what} needs {need / 2**30:.2f} GiB free, "
            f"only {free / 2**30:.2f} GiB is"
        )


#: Elements past which a 32-bit buffer offset wraps. The pool is indexed in
#: elements, so this is the count that matters, not the byte size.
_I32_ELEMS = 2**31


def test_chain_addresses_a_state_pool_past_2gib_elements():
    """The pool term has to be 64-bit, and nothing smaller shows it.

    A slot is ``HV * V * K`` elements, so a pool crosses 2**31 elements at a
    slot count an ordinary serving cache reaches. Past that, a 32-bit offset
    wraps to another slot silently, with no fault: the value written is right
    and the place is not.

    The check is therefore placement, not value. One live sequence is given a
    slot beyond the wrap; the same problem is then run against a small pool, and
    the two must produce the same checkpoints bit for bit.
    """
    HV, V, K = 4, HEAD_V_DIM, HEAD_K_DIM
    per_slot = HV * V * K
    slots = _I32_ELEMS // per_slot + 8
    need = slots * per_slot * 4 + (1 << 28)
    _skip_unless_free_hbm(need, "the >2**31-element pool")

    batch, seqlen = 2, 4
    small = _make_problem(batch, seqlen, seed=41, num_v_heads=HV, accepted="mixed")
    small_out, small_pool = _run_flydsl_chain(small)

    big_pool = torch.zeros(slots, HV, V, K, device=DEVICE, dtype=STATE_DTYPE)
    # Place the live slots at the far end, past where a 32-bit offset wraps.
    base = slots - batch * seqlen - 1
    big_idx = (small.chain_indices + base).to(torch.int32)
    assert (
        big_idx.long() * per_slot
    ).max() > _I32_ELEMS, "the probe did not actually reach past a 32-bit offset"
    for n in range(batch):
        for t in range(seqlen):
            big_pool[big_idx[n, t]] = small.pool[small.chain_indices[n, t]]

    big = small._replace(pool=big_pool, chain_indices=big_idx)
    big_out, big_pool_after = _run_flydsl_chain(big, inplace=True)

    _assert_bit_exact("pool past 2**31 elements", "the output", big_out, small_out)
    _assert_bit_exact(
        "pool past 2**31 elements",
        "the checkpoints",
        big_pool_after[big_idx.reshape(-1).long()],
        small_pool[_touched_slots(small)],
    )
    del big_pool, big_pool_after
    torch.cuda.empty_cache()


def test_sglang_addresses_a_snapshot_buffer_past_2gib_elements():
    """The snapshot buffer wraps sooner than the pool does.

    A snapshot slot spans ``seqlen`` states where a pool slot spans one, so for
    the same sequence count it reaches 2**31 elements ``seqlen`` times sooner.
    It is also a *store*, so a wrap corrupts another sequence's record rather
    than merely reading the wrong one.
    """
    HV, V, K = 4, HEAD_V_DIM, HEAD_K_DIM
    batch, seqlen = 2, 4
    per_step = HV * V * K
    per_slot = seqlen * per_step
    slots = _I32_ELEMS // per_slot + 4
    need = slots * per_slot * 4 + (1 << 28)
    _skip_unless_free_hbm(need, "the >2**31-element snapshot buffer")

    p = _make_problem(batch, seqlen, seed=43, num_v_heads=HV)
    small_out, small_pool, small_inter = _run_flydsl_sglang(p, save_inter=True)

    big_inter = torch.zeros(slots, seqlen, HV, V, K, device=DEVICE, dtype=STATE_DTYPE)
    far = torch.tensor(
        [slots - 1 - n for n in range(batch)], device=DEVICE, dtype=torch.int32
    )
    assert (
        far.long() * per_slot
    ).max() > _I32_ELEMS, "the probe did not actually reach past a 32-bit offset"
    big_out, big_pool, _ = _run_flydsl_sglang(
        p, save_inter=True, inter_indices=far, inter=big_inter
    )

    _assert_bit_exact("snapshots past 2**31 elements", "the output", big_out, small_out)
    _assert_bit_exact("snapshots past 2**31 elements", "the pool", big_pool, small_pool)
    for n in range(batch):
        _assert_bit_exact(
            "snapshots past 2**31 elements",
            f"sequence {n}'s record",
            big_inter[far[n]],
            small_inter[n],
        )
    del big_inter
    torch.cuda.empty_cache()


# -- the dispatch seam ----------------------------------------------------


def _triton_caller(p: _Problem, *, flydsl, inplace=False):
    """aiter's Triton entry, packed once and callable repeatedly.

    It takes the fused ``qkv`` projection rather than three tensors, so the
    problem's q/k/v are packed back into the layout the call site produces --
    once here, since a perf row calls the returned closure in a loop and the
    packing is not part of what is being measured.

    Returns the closure and the pool it writes through.
    """
    B, T = p.batch, p.seqlen
    H, HV, K, V = p.num_k_heads, p.num_v_heads, p.head_k_dim, p.head_v_dim
    tokens = B * T
    qkv = torch.cat(
        [
            p.q.reshape(tokens, H * K),
            p.k.reshape(tokens, H * K),
            p.v.reshape(tokens, HV * V),
        ],
        dim=-1,
    ).contiguous()
    pool = p.pool if inplace else p.pool.clone()
    a = p.a.reshape(tokens, HV)
    b = p.b.reshape(tokens, HV)

    def run():
        prev = os.environ.get("AITER_GDR_FLYDSL")
        os.environ["AITER_GDR_FLYDSL"] = "1" if flydsl else "0"
        try:
            out, _ = fused_rearrange_sigmoid_gated_delta_rule(
                A_log=p.A_log,
                a=a,
                b=b,
                dt_bias=p.dt_bias,
                qkv=qkv,
                key_dim=H * K,
                value_dim=HV * V,
                head_k_dim=K,
                head_v_dim=V,
                initial_state=pool,
                inplace_final_state=True,
                cu_seqlens=p.cu_seqlens,
                ssm_state_indices=p.chain_indices,
                num_accepted_tokens=p.num_accepted,
                use_qk_l2norm_in_kernel=True,
            )
        finally:
            if prev is None:
                os.environ.pop("AITER_GDR_FLYDSL", None)
            else:
                os.environ["AITER_GDR_FLYDSL"] = prev
        return out

    return run, pool


def _triton_call(p: _Problem, *, flydsl, inplace=False):
    run, pool = _triton_caller(p, flydsl=flydsl, inplace=inplace)
    return run(), pool


_SEAM_CASES = [(1, 2), (4, 4), (8, 4)]


@pytest.mark.parametrize("batch,seqlen", _SEAM_CASES)
def test_dispatch_seam_routes_to_flydsl(batch, seqlen):
    """With the seam on, the Triton entry must answer as the port does."""
    p = _make_problem(batch, seqlen, seed=batch * 3 + seqlen, accepted="mixed")
    ref = _oracle_vllm(p)
    routed_out, routed_pool = _triton_call(p, flydsl=True)
    direct_out, direct_pool = _run_flydsl_chain(p)

    # The entry's own contract: routing is an implementation detail, so the
    # shape it hands back must not depend on which path served the call.
    unrouted_out, _ = _triton_call(p, flydsl=False)
    assert routed_out.shape == unrouted_out.shape, (
        f"seam b{batch} s{seqlen}: routing changed the entry's output shape "
        f"({tuple(routed_out.shape)} routed, {tuple(unrouted_out.shape)} on Triton)"
    )

    _assert_bit_exact(
        f"seam b{batch} s{seqlen}",
        "the routed output against the direct call",
        routed_out.reshape(direct_out.shape),
        direct_out,
    )
    _assert_bit_exact(
        f"seam b{batch} s{seqlen}", "the routed pool", routed_pool, direct_pool
    )
    _assert_tracks_spec(
        f"seam b{batch} s{seqlen}: out",
        routed_out.reshape(direct_out.shape),
        ref.spec,
        ref.scale,
    )


def test_dispatch_seam_is_off_by_default():
    """The port is opt-in, and the default path must not reach it."""
    prev = os.environ.pop("AITER_GDR_FLYDSL", None)
    try:
        assert (
            not _flydsl_gdr_enabled()
        ), "AITER_GDR_FLYDSL is unset and the seam still reports enabled"
    finally:
        if prev is not None:
            os.environ["AITER_GDR_FLYDSL"] = prev


def test_dispatch_seam_declines_a_ragged_batch():
    """A ragged batch has no single draft window, so the port must decline.

    ``seq_length`` is a compile-time constant in the kernel, so a batch whose
    sequences differ in length cannot be expressed at all. The seam has to
    notice and fall through to Triton rather than compile the wrong window.
    """
    ragged = torch.tensor([0, 2, 5, 9], device=DEVICE, dtype=torch.int32)
    assert (
        _uniform_draft_window(ragged, 9) == -1
    ), "a ragged cu_seqlens was accepted as a uniform draft window"
    uniform = torch.tensor([0, 4, 8, 12], device=DEVICE, dtype=torch.int32)
    assert _uniform_draft_window(uniform, 12) == 4
    # And the routed path still answers, by falling through.
    p = _make_problem(3, 4, seed=53)
    out, _ = _triton_call(p, flydsl=True)
    assert torch.isfinite(out).all()


def test_dispatch_seam_agrees_with_triton():
    """Routed or not, the entry point has to give the same answer.

    The seam's whole promise is that turning it on changes the kernel and not
    the result, so the two paths are held against the same spec rather than
    against each other -- they round differently and pinning one to the other
    would pin the worse one's error.
    """
    p = _make_problem(4, 4, seed=59, accepted="mixed")
    ref = _oracle_vllm(p)
    fly_out, _ = _triton_call(p, flydsl=True)
    tri_out, _ = _triton_call(p, flydsl=False)
    shape = (p.batch, p.seqlen, p.num_v_heads, p.head_v_dim)
    _assert_tracks_spec("seam on: out", fly_out.reshape(shape), ref.spec, ref.scale)
    _assert_tracks_spec("seam off: out", tri_out.reshape(shape), ref.spec, ref.scale)
    _assert_no_worse_than(
        "seam on vs aiter Triton",
        fly_out.reshape(shape),
        tri_out.reshape(shape),
        ref.spec,
        ref.scale,
        "aiter Triton",
    )


# -- perf -----------------------------------------------------------------

#: ``vllm_chain`` is the linear-chain contract, measured against vLLM's own
#: kernels and aiter's Triton. ``sglang_chain`` and ``sglang_tree`` are the
#: snapshot contract with and without branching parents, measured against
#: SGLang's.
BENCH_MODES = ("vllm_chain", "sglang_chain", "sglang_tree")


def _bench_bytes(p: _Problem):
    """Bytes the contract forces the kernel to move, at best.

    The state dominates everything else and is why the large-batch rows
    converge: the chain writes one full state per draft token, since which token
    is accepted is not known until after the pass, and reads one per sequence
    for the rollback point. The snapshot mode writes the same volume into its
    own buffer instead.
    """
    slot = p.num_v_heads * p.head_v_dim * p.head_k_dim * p.pool.element_size()
    tokens = p.batch * p.seqlen
    state = p.batch * slot + tokens * slot
    qkv = (
        p.q.numel() + p.k.numel() + p.v.numel() + p.a.numel() + p.b.numel()
    ) * p.q.element_size()
    out = p.v.numel() * p.v.element_size()
    return state + qkv + out


# GPU time one sample aims to cover, and the bounds on the iteration count it
# buys. The floor is the harness default; the ceiling keeps a microsecond-scale
# row off the clock for more than a few tens of milliseconds.
_SAMPLE_BUDGET_US = 20_000
_MIN_ITERS = 101
_MAX_ITERS = 2001

# A cell whose repeats disagree by more than this was not measured, whatever the
# median says.
_SPREAD_WARN_PCT = 5.0


@benchmark()
def test_gdr_mtp_perf(
    batch: int = 4,
    seqlen: int = 2,
    mode: str = "vllm_chain",
    num_v_heads: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = STATE_DTYPE,
    repeats: int = 1,
) -> dict:
    """One row of the perf table: every candidate timed and checked on one shape.

    The defaults are a small row, so importing this under pytest costs a few
    cheap launches; ``main()`` sweeps the real shapes.

    ``repeats`` above one times every candidate that many times and reports the
    median, warning on any cell whose samples disagree by more than
    ``_SPREAD_WARN_PCT``. One sample is enough to see a candidate that has
    fallen off a cliff, which is all the pytest entry needs, but not to support
    a claim of a few percent.

    Each mode is measured against the upstream that defines its interface:

                    vllm_chain  sglang_chain  sglang_tree
        flydsl          y            y             y
        vllm            y            -             -
        sglang          -            y             y
        triton          y            -             -

    A blank is a call the kernel cannot express, not a gap: aiter's Triton
    implements the chain contract only, and neither upstream implements the
    other's. The candidate set is therefore a property of the mode, which is
    why the sweep emits one table per mode rather than their NaN-scattered
    union.
    """
    p = _make_problem(
        batch,
        seqlen,
        seed=batch * 31 + seqlen,
        accepted="mixed",
        dtype=dtype,
        state_dtype=state_dtype,
        num_v_heads=num_v_heads,
        num_k_heads=max(1, num_v_heads // 2),
    )
    label = f"{mode} b{batch} s{seqlen} hv{num_v_heads}"
    tokens = batch * seqlen
    HV = num_v_heads
    parents = _eagle_tree(batch, seqlen) if mode == "sglang_tree" else None

    # Each candidate gets its own pool once, before timing, and updates it in
    # place from then on, keeping a copy of the whole pool out of every
    # measurement. Re-running on an already-updated pool is safe to time: the
    # recurrence decays what it reads, so the state stays bounded across
    # iterations.
    if mode == "vllm_chain":
        ref = _oracle_vllm(p)
        pool_fly = p.pool.clone()
        pool_vllm = p.pool.clone()
        p_fly = p._replace(pool=pool_fly)

        def call_flydsl():
            return _run_flydsl_chain(p_fly, inplace=True)[0]

        def call_vllm():
            g, beta = fused_gdn_gating_vllm(
                p.A_log,
                p.a.reshape(tokens, HV),
                p.b.reshape(tokens, HV),
                p.dt_bias,
                SOFTPLUS_BETA,
                SOFTPLUS_THRESHOLD,
            )
            out, _ = fused_recurrent_gated_delta_rule_vllm(
                q=p.q.reshape(1, tokens, p.num_k_heads, p.head_k_dim),
                k=p.k.reshape(1, tokens, p.num_k_heads, p.head_k_dim),
                v=p.v.reshape(1, tokens, HV, p.head_v_dim),
                g=g.view(1, tokens, HV),
                beta=beta.view(1, tokens, HV),
                scale=p.head_k_dim**-0.5,
                initial_state=pool_vllm,
                inplace_final_state=True,
                cu_seqlens=p.cu_seqlens,
                ssm_state_indices=p.chain_indices,
                num_accepted_tokens=p.num_accepted,
                use_qk_l2norm_in_kernel=True,
            )
            return out.reshape(p.v.shape)

        run_triton, _ = _triton_caller(
            p._replace(pool=p.pool.clone()), flydsl=False, inplace=True
        )
        candidates = {
            "flydsl": call_flydsl,
            "vllm": call_vllm,
            "triton": lambda: run_triton().reshape(p.v.shape),
        }
    else:
        ref = _oracle_sglang(p, save_inter=True, parents=parents)
        pool_sglang = p.pool.clone()
        p_fly = p._replace(pool=p.pool.clone())
        inter_fly = torch.zeros(
            batch,
            seqlen,
            HV,
            p.head_v_dim,
            p.head_k_dim,
            device=DEVICE,
            dtype=p.pool.dtype,
        )
        inter = torch.zeros(
            batch,
            seqlen,
            HV,
            p.head_k_dim,
            p.head_v_dim,
            device=DEVICE,
            dtype=p.pool.dtype,
        )
        inter_idx = torch.arange(batch, device=DEVICE, dtype=torch.int32)

        def call_flydsl():
            return _run_flydsl_sglang(
                p_fly,
                save_inter=True,
                parents=parents,
                inter=inter_fly,
                inter_indices=inter_idx,
                inplace=True,
            )[0]

        def call_sglang():
            return fused_sigmoid_gating_delta_rule_update_sglang(
                A_log=p.A_log,
                a=p.a.reshape(tokens, HV),
                dt_bias=p.dt_bias,
                softplus_beta=SOFTPLUS_BETA,
                softplus_threshold=SOFTPLUS_THRESHOLD,
                q=p.q,
                k=p.k,
                v=p.v,
                b=p.b,
                initial_state_source=pool_sglang,
                initial_state_indices=p.seq_indices,
                scale=p.head_k_dim**-0.5,
                use_qk_l2norm_in_kernel=True,
                intermediate_states_buffer=inter,
                intermediate_state_indices=inter_idx,
                retrieve_parent_token=parents,
            )

        candidates = {"flydsl": call_flydsl, "sglang": call_sglang}

    # 4 multiply-adds per (state element, token): the decay, the h.k reduction,
    # the rank-1 update and the h.q readout.
    flops = 4 * 2 * tokens * HV * p.head_v_dim * p.head_k_dim
    nbytes = _bench_bytes(p)

    errs = {}
    for name, fn in candidates.items():
        errs[name] = checkAllclose(
            fn().reshape(p.v.shape).float(),
            ref.spec.reshape(p.v.shape).float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: {label}",
        )

    # The slowest candidate sets the iteration count for all of them, so the row
    # stays like-for-like. A fixed count would be a fraction of a millisecond of
    # work on the cheap shapes, where cells then wander between processes.
    probe = max(
        run_perftest(fn, num_iters=11, num_rotate_args=1)[1]
        for fn in candidates.values()
    )
    num_iters = min(_MAX_ITERS, max(_MIN_ITERS, int(_SAMPLE_BUDGET_US / probe)))

    # Round-robin, so the clock's drift over a row falls on every candidate
    # alike instead of on whichever one held the window it drifted in.
    # ``num_rotate_args`` is pinned for the same reason: at zero it is derived
    # from free GPU memory at the moment of the call, and every candidate is a
    # closure over its own tensors with no argument list to rotate.
    samples = {name: [] for name in candidates}
    for _ in range(repeats):
        for name, fn in candidates.items():
            samples[name].append(
                run_perftest(fn, num_iters=num_iters, num_rotate_args=1)[1]
            )

    ret = {
        "gfx": get_gfx(),
        "dtype": _DTYPE_NAME[dtype],
        "state_dtype": _STATE_DTYPE_NAME[state_dtype],
    }
    for name in candidates:
        seen = samples[name]
        us = statistics.median(seen)
        # The median keeps one bad sample out of the number but cannot say the
        # number was worth taking, so a wide spread is reported with it.
        spread = 100 * (max(seen) - min(seen)) / us
        if spread > _SPREAD_WARN_PCT:
            aiter.logger.warning(
                "gdr_mtp: %s: %s spread %.1f%% over %d repeats; "
                "treat that cell as unmeasured",
                label,
                name,
                spread,
                repeats,
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = errs[name]
    return ret


@pytest.mark.parametrize("mode", BENCH_MODES)
def test_perf_row_agrees_with_the_spec(mode):
    """Keep the timing path inside the suite, one small row per mode.

    Nothing else here calls ``run_perftest``, so the harness is the easiest
    thing in the file to break without noticing -- and ``checkAllclose`` only
    logs, so the row's error column has to be asserted on to mean anything.
    """
    row = test_gdr_mtp_perf(mode=mode, batch=2, seqlen=2, num_v_heads=4)
    for key, err in row.items():
        if key.endswith(" err"):
            assert err < 1e-2, f"{mode}: {key} = {err}"


# -- CI entry -------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Correctness and perf for the FlyDSL GDR MTP kernels.

        Every sweep axis defaults to its endpoints rather than its full range:
        CI runs this file directly, so the perf grid is what sets the shard's
        runtime, and each cell is a distinct build to compile. Widen any axis
        on the command line to get the full grid back.""",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 128])
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[4],
        help="""Draft window. A one-token window is decode, which the decode
        kernel covers and this one is not built for.""",
    )
    parser.add_argument(
        "--num-v-heads",
        type=int,
        nargs="*",
        default=[4, 32],
        help="""Value heads; the key heads follow at half that. This is the
        axis that moves the kernel's occupancy most, so the default keeps the
        ends of a Qwen3-Next-shaped model's range: 32 at tp1 down to 4 at
        tp8.""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["bf16"],
        choices=sorted(_DTYPE_BY_NAME),
    )
    parser.add_argument(
        "--state-dtype",
        type=str,
        nargs="*",
        default=["fp32"],
        choices=sorted(_STATE_DTYPE_BY_NAME),
        help="""Recurrent state dtype, swept separately from the activation
        dtype because it is the one that sets the traffic: the pool is most of
        what the large-batch rows move. The snapshot follows the state, since a
        snapshot wider than the state is refused. Defaults to the wider of the
        two, which bounds that traffic; bf16 state is covered for correctness
        either way.""",
    )
    parser.add_argument(
        "--mode", type=str, nargs="*", default=list(BENCH_MODES), choices=BENCH_MODES
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="""Times each candidate is measured; the table reports the median
        and a cell whose samples disagree is warned about. One sample per cell
        is not enough to compare kernels within a few percent of each other.""",
    )
    return parser.parse_args()


def _run_perf_sweep(args):
    # One table per mode; the candidate set is a property of the mode.
    for mode in args.mode:
        rows = [
            test_gdr_mtp_perf(
                batch=batch,
                seqlen=seqlen,
                mode=mode,
                num_v_heads=num_v_heads,
                dtype=_DTYPE_BY_NAME[dtype],
                state_dtype=_STATE_DTYPE_BY_NAME[state_dtype],
                repeats=args.repeats,
            )
            for dtype, state_dtype, num_v_heads, seqlen, batch in itertools.product(
                args.dtype, args.state_dtype, args.num_v_heads, args.seqlen, args.batch
            )
        ]
        if not rows:
            continue
        aiter.logger.info(
            "flydsl gdr mtp %s perf (markdown):\n%s",
            mode,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def main():
    """Run every check without pytest, then the perf sweep.

    Exits non-zero if any check fails. CI shards ``op_tests/test_*.py`` by
    invoking ``python3 <file>``, which would otherwise merely define the test
    functions above and report success.
    """
    args = _parse_args()
    if _SKIP_REASON is not None:
        aiter.logger.warning("%s; skipping the FlyDSL GDR MTP tests", _SKIP_REASON)
        return 0
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl gdr mtp is unsupported on %s; skipping", get_gfx())
        return 0

    aiter.logger.info("gdr_mtp: running correctness checks...")
    cases_run, failures = _run_correctness()
    if failures:
        for name, exc, tb in failures:
            aiter.logger.error("FAILED %s: %r\n%s", name, exc, tb)
        aiter.logger.error(
            "gdr_mtp: %d of %d check(s) failed; skipping perf sweep",
            len(failures),
            cases_run,
        )
        return 1
    aiter.logger.info("gdr_mtp: all %d checks passed", cases_run)

    _run_perf_sweep(args)
    return 0


def _run_correctness():
    torch.manual_seed(0)
    dtypes = [(d,) for d in sorted(_SUPPORTED_DTYPES, key=str)]
    state_dtypes = [(d,) for d in sorted(_SUPPORTED_STATE_DTYPES, key=str)]
    cases = [
        (test_chain_matches_upstream_vllm, _CHAIN_CASES),
        (
            test_chain_matches_upstream_across_dtypes,
            [(d, n) for (d,) in dtypes for n in (False, True)],
        ),
        (test_chain_matches_upstream_across_state_dtypes, state_dtypes),
        (test_chain_rolls_back_to_the_accepted_token, [()]),
        (test_chain_skips_the_null_block, [()]),
        (test_sglang_matches_upstream_sglang, _SGLANG_CASES),
        (test_sglang_matches_upstream_across_dtypes, dtypes),
        (test_sglang_disable_state_update_leaves_the_pool_alone, [()]),
        (test_sglang_tree_is_not_running_a_chain, [()]),
        (test_sglang_tree_without_a_snapshot_slot_runs_the_chain, [()]),
        (test_a_snapshot_wider_than_the_state_is_refused_at_every_layer, [()]),
        (test_chain_checkpoints_and_snapshots_are_the_same_state, [()]),
        (test_chain_addresses_a_state_pool_past_2gib_elements, [()]),
        (test_sglang_addresses_a_snapshot_buffer_past_2gib_elements, [()]),
        (test_dispatch_seam_routes_to_flydsl, _SEAM_CASES),
        (test_dispatch_seam_is_off_by_default, [()]),
        (test_dispatch_seam_declines_a_ragged_batch, [()]),
        (test_dispatch_seam_agrees_with_triton, [()]),
        (test_perf_row_agrees_with_the_spec, [(m,) for m in BENCH_MODES]),
    ]

    # CI runs this file, not pytest, so a test that never makes it into `cases`
    # is a test CI does not have. Compare by identity: `@benchmark` returns a
    # bare closure, so the wrapped perf fn's ``__name__`` is not its own.
    reached = {id(fn) for fn, _ in cases} | {id(test_gdr_mtp_perf)}
    unreached = sorted(
        name
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj) and id(obj) not in reached
    )
    assert not unreached, f"never run by the CI entry point: {unreached}"

    cases_run = 0
    skipped = 0
    failures = []
    for fn, arg_sets in cases:
        for case_args in arg_sets:
            name = f"{fn.__name__}{case_args if case_args else ''}"
            try:
                fn(*case_args)
            except pytest.skip.Exception as exc:
                # Skipped derives from BaseException, so it would sail past the
                # collector below and abort the shard. A case that skips itself
                # (too little free HBM for the >2**31-element pool) neither ran
                # nor failed.
                aiter.logger.info("gdr_mtp: skipped %s -- %s", name, exc)
                skipped += 1
            except Exception as exc:  # noqa: BLE001 - collect and keep going
                failures.append((name, exc, traceback.format_exc()))
                cases_run += 1
            else:
                cases_run += 1
    if skipped:
        aiter.logger.info("gdr_mtp: %d case(s) skipped", skipped)
    return cases_run, failures


if __name__ == "__main__":
    sys.exit(main())
