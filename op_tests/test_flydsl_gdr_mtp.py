# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for vLLM/SGLang FlyDSL GDR MTP."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import timeit
from typing import NamedTuple

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_kernels import (
    _SUPPORTED_DTYPES,
    _flydsl_gdr_mtp_supported,
    flydsl_gdr_mtp,
    flydsl_gdr_mtp_sglang,
)
from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
    _flydsl_gdr_enabled,
    fused_rearrange_sigmoid_gated_delta_rule,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.triton_tests.utils.gdr_mtp_refs import (
    fused_gdn_gating_vllm,
    fused_recurrent_gated_delta_rule_vllm,
    fused_sigmoid_gating_delta_rule_update_sglang,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32

SUPPORTED_GFX = ("gfx942", "gfx950")
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="FlyDSL GDR MTP requires gfx942 or gfx950",
)

HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 2
NUM_V_HEADS = 4

SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

_DTYPE_EPS = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_STATE_DTYPE_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16"}

_ERR_FACTOR = 64.0

NULL_BLOCK_ID = 0


# -- inputs ---------------------------------------------------------------


class _Problem(NamedTuple):

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
    parents = [0] * seqlen
    for t in range(1, seqlen):
        parents[t] = 0 if t <= 2 else t - 2
    return torch.tensor([parents] * batch, device=DEVICE, dtype=torch.int32)


# -- conditioning ---------------------------------------------------------


def _magnitudes(p: _Problem, *, init_slots, use_qk_l2norm=True, parents=None):
    B, T = p.batch, p.seqlen
    HV, K = p.num_v_heads, p.head_k_dim
    rep = HV // p.num_k_heads

    q, k = p.q.double(), p.k.double()
    if use_qk_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
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


def _torch_ref(p: _Problem, mode, parents=None, use_qk_l2norm=True):
    q, k = p.q.float(), p.k.float()
    if use_qk_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    repeat = p.num_v_heads // p.num_k_heads
    q = (q * p.head_k_dim**-0.5).repeat_interleave(repeat, dim=2)
    k = k.repeat_interleave(repeat, dim=2)
    decay = torch.exp(
        -torch.exp(p.A_log.float())
        * torch.nn.functional.softplus(p.a.float() + p.dt_bias.float())
    )
    beta = torch.sigmoid(p.b.float())
    if mode == "vllm_chain":
        rows = torch.arange(p.batch, device=DEVICE)
        slots = p.chain_indices[rows, p.num_accepted.long() - 1]
    else:
        slots = p.seq_indices
    state = p.pool.float()[slots.long()]
    steps = []
    out = torch.empty_like(p.v, dtype=torch.float32)
    for token in range(p.seqlen):
        if parents is not None and token:
            prior = torch.stack(steps, dim=1)
            rows = torch.arange(p.batch, device=DEVICE)
            state = prior[rows, parents[:, token].long()]
        state = state * decay[:, token, :, None, None]
        value = p.v[:, token].float() - torch.einsum(
            "bhvk,bhk->bhv", state, k[:, token]
        )
        state = (
            state
            + (value * beta[:, token, :, None])[..., None] * k[:, token, :, None, :]
        )
        out[:, token] = torch.einsum("bhvk,bhk->bhv", state, q[:, token])
        steps.append(state)
    return out


# -- oracles --------------------------------------------------------------


class _Oracle(NamedTuple):

    spec: torch.Tensor
    scale: torch.Tensor
    spec_pool: torch.Tensor
    scale_pool: torch.Tensor
    spec_inter: torch.Tensor | None
    scale_inter: torch.Tensor | None
    pool: torch.Tensor


def _run_vllm_upstream(p: _Problem, pool, *, use_qk_l2norm=True):
    tokens = p.batch * p.seqlen
    g, beta = fused_gdn_gating_vllm(
        p.A_log,
        p.a.reshape(tokens, p.num_v_heads),
        p.b.reshape(tokens, p.num_v_heads),
        p.dt_bias,
        SOFTPLUS_BETA,
        SOFTPLUS_THRESHOLD,
    )
    out, _ = fused_recurrent_gated_delta_rule_vllm(
        q=p.q.reshape(1, tokens, p.num_k_heads, p.head_k_dim),
        k=p.k.reshape(1, tokens, p.num_k_heads, p.head_k_dim),
        v=p.v.reshape(1, tokens, p.num_v_heads, p.head_v_dim),
        g=g.view(1, tokens, p.num_v_heads),
        beta=beta.view(1, tokens, p.num_v_heads),
        scale=p.head_k_dim**-0.5,
        initial_state=pool,
        inplace_final_state=True,
        cu_seqlens=p.cu_seqlens,
        ssm_state_indices=p.chain_indices,
        num_accepted_tokens=p.num_accepted,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    return out.reshape(p.v.shape)


def _run_sglang_upstream(
    p: _Problem,
    pool,
    *,
    use_qk_l2norm=True,
    disable_state_update=False,
    inter=None,
    inter_indices=None,
    parents=None,
):
    return fused_sigmoid_gating_delta_rule_update_sglang(
        A_log=p.A_log,
        a=p.a.reshape(p.batch * p.seqlen, p.num_v_heads),
        dt_bias=p.dt_bias,
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
        q=p.q,
        k=p.k,
        v=p.v,
        b=p.b,
        initial_state_source=pool,
        initial_state_indices=p.seq_indices,
        scale=p.head_k_dim**-0.5,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=inter,
        intermediate_state_indices=inter_indices,
        retrieve_parent_token=parents,
    )


def _oracle_vllm(p: _Problem, *, use_qk_l2norm=True):

    def run(problem):
        state = problem.pool.clone()
        return _run_vllm_upstream(problem, state, use_qk_l2norm=use_qk_l2norm), state

    _, pool = run(p)
    fp32 = p._replace(
        q=p.q.float(),
        k=p.k.float(),
        v=p.v.float(),
        a=p.a.float(),
        b=p.b.float(),
        dt_bias=p.dt_bias.float(),
        A_log=p.A_log.float(),
        pool=p.pool.float(),
    )
    _, spec_pool = run(fp32)
    spec = _torch_ref(p, "vllm_chain", use_qk_l2norm=use_qk_l2norm)
    rows = torch.arange(p.batch, device=p.pool.device)
    scale, steps = _magnitudes(
        p,
        init_slots=p.chain_indices[rows, p.num_accepted.long() - 1],
        use_qk_l2norm=use_qk_l2norm,
    )
    scale_pool = p.pool.double().abs()
    scale_pool[p.chain_indices.long().flatten()] = steps.flatten(0, 1)
    return _Oracle(
        spec.double(),
        scale,
        spec_pool.double(),
        scale_pool,
        None,
        None,
        pool,
    )


def _oracle_sglang(
    p: _Problem,
    *,
    use_qk_l2norm=True,
    save_inter=False,
    parents=None,
    disable_state_update=False,
):

    def run(problem):
        state = problem.pool.clone()
        inter = (
            torch.zeros(
                problem.batch,
                problem.seqlen,
                problem.num_v_heads,
                problem.head_k_dim,
                problem.head_v_dim,
                device=DEVICE,
                dtype=state.dtype,
            )
            if save_inter
            else None
        )
        out = _run_sglang_upstream(
            problem,
            state,
            use_qk_l2norm=use_qk_l2norm,
            disable_state_update=disable_state_update,
            inter=inter,
            inter_indices=(
                torch.arange(problem.batch, device=DEVICE, dtype=torch.int32)
                if save_inter
                else None
            ),
            parents=parents,
        )
        if inter is not None:
            inter = inter.view(
                problem.batch,
                problem.seqlen,
                problem.num_v_heads,
                problem.head_v_dim,
                problem.head_k_dim,
            )
        return out, state, inter

    _, pool, _ = run(p)
    fp32 = p._replace(
        q=p.q.float(),
        k=p.k.float(),
        v=p.v.float(),
        a=p.a.float(),
        b=p.b.float(),
        dt_bias=p.dt_bias.float(),
        A_log=p.A_log.float(),
        pool=p.pool.float(),
    )
    _, spec_pool, spec_inter = run(fp32)
    spec = _torch_ref(
        p,
        "sglang_tree" if parents is not None else "sglang_chain",
        parents,
        use_qk_l2norm,
    )
    scale, steps = _magnitudes(
        p,
        init_slots=p.seq_indices,
        use_qk_l2norm=use_qk_l2norm,
        parents=parents,
    )
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
        pool,
    )


def _run_flydsl_chain(
    p: _Problem, *, use_qk_l2norm=True, inplace=False, out=None, stream=None
):
    pool = p.pool if inplace else p.pool.clone()
    out = torch.empty_like(p.v) if out is None else out
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
        stream=stream,
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


def _assert_matches_oracle(label, actual, spec, scale):
    _assert_tracks_spec(label, actual, spec, scale)


def _assert_bit_exact(label, what, actual, expected):
    assert torch.equal(actual, expected), (
        f"{label}: {what} is not bit-exact "
        f"(max|delta|={(actual.float() - expected.float()).abs().max():.3e}); "
        "both are the same state written by two address computations, so this "
        "is placement, not rounding"
    )


def _touched_slots(p: _Problem):
    return p.chain_indices.reshape(-1).long()


# -- vLLM's chain contract ------------------------------------------------

_CHAIN_CASES = [
    (1, 2, "full"),
    (3, 2, "mixed"),
    (4, 4, "first"),
    (3, 8, "mixed"),
]


def _check_chain_case(
    batch,
    seqlen,
    accepted,
    *,
    dtype=DTYPE,
    state_dtype=STATE_DTYPE,
    use_qk_l2norm=True,
):
    p = _make_problem(
        batch,
        seqlen,
        seed=batch * 131 + seqlen * 7 + len(accepted),
        dtype=dtype,
        state_dtype=state_dtype,
        accepted=accepted,
    )
    label = f"chain b{batch} s{seqlen} {accepted} {_DTYPE_NAME[dtype]}"
    ref = _oracle_vllm(p, use_qk_l2norm=use_qk_l2norm)
    out, pool = _run_flydsl_chain(p, use_qk_l2norm=use_qk_l2norm)

    _assert_matches_oracle(f"{label}: out", out, ref.spec, ref.scale)

    slots = _touched_slots(p)
    got = pool[slots]
    _assert_matches_oracle(
        f"{label}: checkpoints",
        got,
        ref.spec_pool[slots],
        ref.scale_pool[slots],
    )
    assert torch.isfinite(got).all(), f"{label}: checkpoints have non-finite values"
    _assert_untouched_outside(label, pool, ref.pool, slots)


def _assert_untouched_outside(label, pool, ref_pool, slots):
    mask = torch.ones(pool.shape[0], dtype=torch.bool, device=pool.device)
    mask[slots] = False
    if mask.any():
        _assert_bit_exact(
            label, "pool outside the written slots", pool[mask], ref_pool[mask]
        )


@pytest.mark.parametrize("batch,seqlen,accepted", _CHAIN_CASES)
def test_chain_matches_upstream_vllm(batch, seqlen, accepted):
    _check_chain_case(batch, seqlen, accepted)


@pytest.mark.parametrize("dtype", sorted(_SUPPORTED_DTYPES, key=str))
@pytest.mark.parametrize("use_qk_l2norm", [False, True])
def test_chain_matches_upstream_across_dtypes(dtype, use_qk_l2norm):
    _check_chain_case(4, 4, "mixed", dtype=dtype, use_qk_l2norm=use_qk_l2norm)


def test_chain_matches_upstream_with_bf16_state():
    _check_chain_case(4, 4, "mixed", state_dtype=torch.bfloat16)


def test_chain_skips_the_null_block():
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

_SGLANG_CASES = [
    (4, 4, True, False),
    (4, 4, False, False),
    (4, 4, True, True),
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

    _assert_matches_oracle(f"{label}: out", out, ref.spec, ref.scale)

    slots = p.seq_indices.long()
    _assert_matches_oracle(
        f"{label}: state",
        pool[slots],
        ref.spec_pool[slots],
        ref.scale_pool[slots],
    )
    _assert_untouched_outside(label, pool, ref.pool, slots)
    if save_inter:
        _assert_matches_oracle(
            f"{label}: snapshots",
            inter,
            ref.spec_inter,
            ref.scale_inter,
        )


@pytest.mark.parametrize("batch,seqlen,save_inter,tree", _SGLANG_CASES)
def test_sglang_matches_upstream_sglang(batch, seqlen, save_inter, tree):
    _check_sglang_case(batch, seqlen, save_inter, tree)


def test_sglang_matches_upstream_with_fp16():
    _check_sglang_case(4, 4, True, True, dtype=torch.float16)


def test_sglang_disable_state_update_leaves_the_pool_alone():
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


# -- the dispatch seam ----------------------------------------------------


def _triton_caller(p: _Problem, *, flydsl, inplace=False, set_env=True):
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
    core = torch.empty(tokens, HV, V, device=DEVICE, dtype=p.q.dtype)

    def run():
        prev = os.environ.get("AITER_GDR_FLYDSL") if set_env else None
        if set_env:
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
                core_attn_out=core,
                draft_window=p.seqlen if flydsl else None,
            )
        finally:
            if set_env and prev is None:
                os.environ.pop("AITER_GDR_FLYDSL", None)
            elif set_env:
                os.environ["AITER_GDR_FLYDSL"] = prev
        return out

    return run, pool


def _triton_call(p: _Problem, *, flydsl, inplace=False):
    run, pool = _triton_caller(p, flydsl=flydsl, inplace=inplace)
    return run(), pool


def test_dispatch_seam_routes_to_flydsl():
    prev = os.environ.pop("AITER_GDR_FLYDSL", None)
    try:
        assert not _flydsl_gdr_enabled()
    finally:
        if prev is not None:
            os.environ["AITER_GDR_FLYDSL"] = prev

    batch, seqlen = 4, 4
    p = _make_problem(batch, seqlen, seed=batch * 3 + seqlen, accepted="mixed")
    ref = _oracle_vllm(p)
    routed_out, routed_pool = _triton_call(p, flydsl=True)
    direct_out, direct_pool = _run_flydsl_chain(p)

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

    slot_zero = _make_problem(1, 2, seed=61, accepted="first")
    slot_zero = slot_zero._replace(
        chain_indices=torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)
    )
    _, pool = _triton_call(slot_zero, flydsl=True)
    assert not torch.equal(pool[0], slot_zero.pool[0])
    assert not torch.equal(pool[1], slot_zero.pool[1])


def test_mtp_rejects_noncontiguous_output():
    p = _make_problem(2, 2, seed=67)
    backing = torch.full(
        (*p.v.shape[:-1], p.head_v_dim * 2), 123, device=DEVICE, dtype=p.v.dtype
    )
    out = backing[..., ::2]
    before = backing.clone()
    with pytest.raises(ValueError, match="`out` must be contiguous"):
        _run_flydsl_chain(p, out=out)
    assert torch.equal(backing, before)


def test_mtp_rejects_invalid_snapshot_buffer():
    p = _make_problem(2, 2, seed=69)
    valid = torch.empty(
        p.batch,
        p.seqlen,
        p.num_v_heads,
        p.head_v_dim,
        p.head_k_dim,
        device=DEVICE,
        dtype=p.pool.dtype,
    )
    invalid = [(valid.cpu(), "same device"), (valid.half(), "must have one of")]
    if torch.cuda.device_count() > 1:
        invalid.append((valid.to("cuda:1"), "same device"))

    for inter, match in invalid:
        with pytest.raises(ValueError, match=match):
            _run_flydsl_sglang(p, save_inter=True, inter=inter)


def test_mtp_dispatch_rejects_undersized_index_batches():
    p = _make_problem(2, 2, seed=70)
    args = (p.q, p.k, p.v, p.pool)
    assert not _flydsl_gdr_mtp_supported(*args, p.chain_indices[:1], p.num_accepted)
    assert not _flydsl_gdr_mtp_supported(*args, p.chain_indices, p.num_accepted[:1])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_mtp_stream_uses_the_input_device(monkeypatch):
    p = _make_problem(2, 2, seed=71)
    p = _Problem(*(x.to("cuda:1") if isinstance(x, torch.Tensor) else x for x in p))
    captured = {}
    monkeypatch.setattr(
        "aiter.ops.flydsl.linear_attention_kernels._mtp_launch",
        lambda **kwargs: captured.update(kwargs),
    )
    with torch.cuda.device(0):
        _run_flydsl_chain(p)
        assert captured["stream"].device == p.q.device
        wrong = torch.cuda.Stream(device=0)
        with pytest.raises(ValueError, match="`stream` must be on"):
            _run_flydsl_chain(p, stream=wrong)


BENCH_MODES = ("vllm_chain", "sglang_chain", "sglang_tree")


def _bench_bytes(p: _Problem):
    slot = p.num_v_heads * p.head_v_dim * p.head_k_dim * p.pool.element_size()
    tokens = p.batch * p.seqlen
    state = p.batch * slot + tokens * slot
    qkv = (
        p.q.numel() + p.k.numel() + p.v.numel() + p.a.numel() + p.b.numel()
    ) * p.q.element_size()
    out = p.v.numel() * p.v.element_size()
    return state + qkv + out


def _wall_us(fn, warmup=20, iterations=101):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = timeit.repeat(
        lambda: (fn(), torch.cuda.synchronize()), repeat=iterations, number=1
    )
    return float(pd.Series(samples).median() * 1e6)


@benchmark()
def test_gdr_mtp_perf(
    batch: int = 4,
    seqlen: int = 2,
    mode: str = "vllm_chain",
    num_v_heads: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = STATE_DTYPE,
) -> dict:
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

    if mode == "vllm_chain":
        ref = _torch_ref(p, mode)
        pool_fly = p.pool.clone()
        pool_vllm = p.pool.clone()
        pool_public = p.pool.clone()
        pool_public_off = p.pool.clone()
        p_fly = p._replace(pool=pool_fly)
        call_public, _ = _triton_caller(
            p._replace(pool=pool_public), flydsl=True, inplace=True, set_env=False
        )
        call_public_off, _ = _triton_caller(
            p._replace(pool=pool_public_off),
            flydsl=False,
            inplace=True,
            set_env=False,
        )

        def call_flydsl():
            return _run_flydsl_chain(p_fly, inplace=True)[0]

        def call_vllm():
            return _run_vllm_upstream(p, pool_vllm)

        # Triton is excluded: its state-index contract differs from vLLM MTP.
        candidates = {
            "flydsl_direct": call_flydsl,
            "public_entry_flydsl": call_public,
            "public_entry_triton": call_public_off,
            "vllm_reference": call_vllm,
        }
    else:
        ref = _torch_ref(p, mode, parents)
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
            return _run_sglang_upstream(
                p,
                pool_sglang,
                inter=inter,
                inter_indices=inter_idx,
                parents=parents,
            )

        candidates = {"flydsl": call_flydsl, "sglang": call_sglang}

    flops = 4 * 2 * tokens * HV * p.head_v_dim * p.head_k_dim
    nbytes = _bench_bytes(p)

    ret = {
        "gfx": get_gfx(),
        "dtype": _DTYPE_NAME[dtype],
        "state_dtype": _STATE_DTYPE_NAME[state_dtype],
    }
    for name, fn in candidates.items():
        previous = os.environ.get("AITER_GDR_FLYDSL")
        public_env = {
            "public_entry_flydsl": "1",
            "public_entry_triton": "0",
        }.get(name)
        if public_env is not None:
            os.environ["AITER_GDR_FLYDSL"] = public_env
        try:
            out = fn().clone()
            torch.cuda.synchronize()
            _, us = run_perftest(fn, num_rotate_args=1)
            wall_us = _wall_us(fn)
        finally:
            if public_env is not None and previous is None:
                os.environ.pop("AITER_GDR_FLYDSL", None)
            elif public_env is not None:
                os.environ["AITER_GDR_FLYDSL"] = previous
        ret[f"{name} us"] = us
        ret[f"{name} wall us"] = wall_us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.reshape(p.v.shape).to(dtypes.fp32),
            out.reshape(p.v.shape).to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: {label}",
        )
    return ret


test_gdr_mtp_perf.__test__ = False


# -- CI entry -------------------------------------------------------------


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl gdr mtp is unsupported on %s; skipping", gfx)
        return 0

    parser = argparse.ArgumentParser(description="FlyDSL GDR MTP correctness and perf")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 128])
    parser.add_argument("--seqlen", type=int, nargs="*", default=[4])
    # Qwen3-Next value heads at TP8 and TP1; key heads are half as many.
    parser.add_argument("--num-v-heads", type=int, nargs="*", default=[4, 32])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,"
    )
    parser.add_argument(
        "--state-dtype", type=dtypes.str2Dtype, nargs="*", default="fp32,"
    )
    parser.add_argument(
        "--mode", type=str, nargs="*", default=list(BENCH_MODES), choices=BENCH_MODES
    )
    args = parser.parse_args()

    aiter.logger.info("gdr_mtp: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "gdr_mtp: checks failed (pytest exit %s); skipping perf sweep", int(status)
        )
        return 1

    rows = [
        test_gdr_mtp_perf(batch, seqlen, mode, heads, dtype, state_dtype)
        for dtype, state_dtype, heads, seqlen, batch, mode in itertools.product(
            args.dtype,
            args.state_dtype,
            args.num_v_heads,
            args.seqlen,
            args.batch,
            args.mode,
        )
    ]
    aiter.logger.info(
        "flydsl gdr mtp summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
