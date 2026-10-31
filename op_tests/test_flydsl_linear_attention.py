# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention regressions.

Usage:
    python op_tests/test_flydsl_linear_attention.py
    pytest -sv op_tests/test_flydsl_linear_attention.py
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.gdn_decode_verify import (
    create_gdn_decode_verify_kernel,
)
from aiter.ops.flydsl.linear_attention_kernels import (
    flydsl_gdn_decode_varlen,
    flydsl_gdr_decode,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="FlyDSL GDR decode requires gfx942 or gfx950",
)
# The packed decode/verify kernel is tuned and validated on gfx950 only.
GDN_VARLEN_GFX = ["gfx950"]
_gdn_varlen_only = pytest.mark.skipif(
    get_gfx() not in GDN_VARLEN_GFX,
    reason="FlyDSL GDN packed decode/verify is validated on gfx950 only",
)
GDN_HEAD_K_DIM = 128
_PERF_ROTATION_BUDGET = 1024**3
_MAX_PERF_ROTATIONS = 101
# ``checkAllclose`` returns the fraction of mismatching elements. The kernel
# stores bf16 while the reference accumulates in fp32, so a value sitting on a
# rounding boundary can land one ULP away -- a handful of such elements per
# tensor is expected. A real correctness break moves far more than this.
_TOL_ERR_RATIO = 1e-3


@dataclass
class Args:
    dtype: torch.dtype
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    use_qk_l2norm: bool = True


def create_inputs(args):
    query = torch.randn(
        (args.b, args.sq, args.num_k_heads, args.head_k_dim),
        dtype=args.dtype,
        device="cuda",
    )
    key = torch.randn(
        (args.b, args.sq, args.num_k_heads, args.head_k_dim),
        dtype=args.dtype,
        device="cuda",
    )
    value = torch.randn(
        (args.b, args.sq, args.num_v_heads, args.head_v_dim),
        dtype=args.dtype,
        device="cuda",
    )
    a = torch.randn(
        (args.b, args.sq, args.num_v_heads), dtype=args.dtype, device="cuda"
    )
    b = torch.randn(
        (args.b, args.sq, args.num_v_heads), dtype=args.dtype, device="cuda"
    )
    dt_bias = torch.randn((args.num_v_heads), dtype=args.dtype, device="cuda")
    dt_bias.uniform_(1, 2)
    A_log = torch.randn((args.num_v_heads), dtype=torch.float32, device="cuda")
    A_log.uniform_(0, 16)
    indices = torch.arange(args.b - 1, -1, -1, dtype=torch.int32, device="cuda")
    state = torch.randn(
        (args.b, args.num_v_heads, args.head_k_dim, args.head_v_dim),
        dtype=torch.float32,
        device="cuda",
    )
    return (args, query, key, value, a, b, dt_bias, A_log, indices, state)


def create_outputs(args):
    out = torch.zeros(
        (args.b, args.sq, args.num_v_heads, args.head_v_dim),
        dtype=args.dtype,
        device="cuda",
    )
    return (out,)


def func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    flydsl_gdr_decode(
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        state,
        out,
        use_qk_l2norm=args.use_qk_l2norm,
        need_shuffle_state=True,
    )


def ref_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    """Pure PyTorch fp32 reference for the indexed GDR decode recurrence."""
    q = query.float()
    k = key.float()
    v = value.float()
    if args.use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (args.head_k_dim**-0.5)

    # GQA maps each contiguous group of value heads to one query/key head.
    heads_per_k_head = args.num_v_heads // args.num_k_heads
    q = q.repeat_interleave(heads_per_k_head, dim=2)
    k = k.repeat_interleave(heads_per_k_head, dim=2)

    gate_input = a.float() + dt_bias.float()
    decay = torch.exp(
        -torch.exp(A_log.float()) * F.softplus(gate_input, beta=1.0, threshold=20.0)
    )
    beta = torch.sigmoid(b.float())

    # A negative slot is padding: the kernel leaves its state and output untouched.
    for batch_idx, state_idx in enumerate(indices.tolist()):
        if state_idx < 0:
            continue
        h = state[state_idx].float()
        for token_idx in range(args.sq):
            h = h * decay[batch_idx, token_idx, :, None, None]
            residual = v[batch_idx, token_idx] - torch.einsum(
                "hkv,hk->hv", h, k[batch_idx, token_idx]
            )
            residual = residual * beta[batch_idx, token_idx, :, None]
            h = h + k[batch_idx, token_idx, :, :, None] * residual[:, None, :]
            out[batch_idx, token_idx].copy_(
                torch.einsum("hkv,hk->hv", h, q[batch_idx, token_idx])
            )
        state[state_idx].copy_(h)


def _recurrent_decode_work(args, query, state, A_log, indices):
    tokens = args.b * args.sq
    k_dim = args.head_k_dim
    v_dim = args.head_v_dim

    # Per token/value head, the dominant recurrent work is approximately:
    # state decay (K*V), state@key (2*K*V), outer-product update (2*K*V),
    # and query@state (2*K*V), plus residual/beta vector ops (2*V).
    # Q/K L2 normalization adds about 6*K FLOPs per distinct key head. Exp,
    # sigmoid, softplus, and rsqrt are deliberately omitted from this roofline
    # approximation because they do not have a useful conventional FLOP count.
    flops = tokens * args.num_v_heads * (7 * k_dim * v_dim + 2 * v_dim)
    if args.use_qk_l2norm:
        flops += tokens * args.num_k_heads * 6 * k_dim

    # Useful-byte lower bound: read Q/K/V/gates, read+write recurrent state,
    # and write output. It excludes cache effects and the wrapper's temporary
    # state reshuffle, so TB/s remains an algorithmic, implementation-neutral
    # bandwidth metric rather than an estimate of physical DRAM transactions.
    data_elements = (
        2 * tokens * args.num_k_heads * k_dim
        + 2 * tokens * args.num_v_heads * v_dim
        + 2 * tokens * args.num_v_heads
        + args.num_v_heads
    )
    state_elements = args.b * args.num_v_heads * k_dim * v_dim
    nbytes = (
        data_elements * query.element_size()
        + 2 * state_elements * state.element_size()
        + args.num_v_heads * A_log.element_size()
        + args.b * indices.element_size()
    )
    return flops, nbytes


def _validate_head_config(num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    if min(num_k_heads, num_v_heads, head_k_dim, head_v_dim) <= 0:
        raise ValueError("head counts and dimensions must be positive")
    if num_v_heads < num_k_heads or num_v_heads % num_k_heads:
        raise ValueError(
            "num_v_heads must be a positive multiple of num_k_heads, got "
            f"{num_k_heads=} {num_v_heads=}"
        )
    if head_k_dim % 32 or head_v_dim % 32:
        raise ValueError(
            "head_k_dim and head_v_dim must be multiples of 32, got "
            f"{head_k_dim=} {head_v_dim=}"
        )


def _perf_rotation_count(state, out):
    bytes_per_call = max(1, state.nbytes + out.nbytes)
    return max(
        1,
        min(_MAX_PERF_ROTATIONS, _PERF_ROTATION_BUDGET // bytes_per_call),
    )


@benchmark()
def test_flydsl_gdr_decode(
    b,
    sq,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    dtype,
    use_qk_l2norm,
):
    if b <= 0 or sq <= 0:
        raise ValueError(f"batch and sequence length must be positive, got {b=} {sq=}")
    _validate_head_config(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    args = Args(
        dtype=dtype,
        b=b,
        sq=sq,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        use_qk_l2norm=use_qk_l2norm,
    )
    (
        _,
        query,
        key,
        value,
        a,
        beta,
        dt_bias,
        A_log,
        indices,
        initial_state,
    ) = create_inputs(args)

    reference_state = initial_state.clone()
    reference_out = create_outputs(args)[0]
    ref_func(
        args,
        query,
        key,
        value,
        a,
        beta,
        dt_bias,
        A_log,
        indices,
        reference_state,
        reference_out,
    )

    def run_flydsl(candidate_state, candidate_out):
        return func(
            args,
            query,
            key,
            value,
            a,
            beta,
            dt_bias,
            A_log,
            indices,
            candidate_state,
            candidate_out,
        )

    candidates = {"flydsl": run_flydsl}
    flops, nbytes = _recurrent_decode_work(args, query, initial_state, A_log, indices)
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        # GDR updates state in place. Pass state/output as timing arguments so
        # run_perftest rotates a bounded pool of recurrent states while keeping
        # clone/reset costs outside the measured kernel/wrapper latency.
        perf_state = initial_state.clone()
        perf_out = create_outputs(args)[0]
        _, us = run_perftest(
            candidate,
            perf_state,
            perf_out,
            num_rotate_args=_perf_rotation_count(perf_state, perf_out),
        )

        # Correctness gets a pristine state/output, independent of the repeatedly
        # updated state used above.
        candidate_state = initial_state.clone()
        candidate_out = create_outputs(args)[0]
        candidate(candidate_state, candidate_out)
        err_out = checkAllclose(
            reference_out.to(dtypes.fp32),
            candidate_out.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: GDR decode output ",
        )
        err_state = checkAllclose(
            reference_state.to(dtypes.fp32),
            candidate_state.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: GDR decode state ",
        )
        assert err_out <= _TOL_ERR_RATIO and err_state <= _TOL_ERR_RATIO, (
            f"{name}: mismatch ratio exceeds {_TOL_ERR_RATIO:g} "
            f"(output {err_out:.3e}, state {err_state:.3e})"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err_out, err_state)
    return ret


# The argument-driven benchmark is run by main(); pytest collects the two
# contract regressions below.
test_flydsl_gdr_decode.__test__ = False


def test_flydsl_gdr_decode_default():
    result = test_flydsl_gdr_decode(
        1,
        1,
        2,
        8,
        128,
        128,
        dtypes.bf16,
        True,
    )
    assert result["flydsl err"] == 0


@pytest.mark.parametrize("input_index,input_name", [(1, "query"), (2, "key")])
def test_flydsl_gdr_decode_rejects_noncontiguous_vector_dimension(
    input_index, input_name
):
    args = Args(
        dtype=torch.bfloat16,
        b=2,
        sq=1,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
    )
    inouts = list(create_inputs(args) + create_outputs(args))
    tensor = inouts[input_index]
    strided_storage = torch.randn(
        *tensor.shape[:-1],
        tensor.shape[-1] * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    inouts[input_index] = strided_storage[..., ::2]
    assert inouts[input_index].shape == tensor.shape
    assert inouts[input_index].stride(-1) == 2

    with pytest.raises(
        ValueError,
        match=rf"`{input_name}` must have a contiguous last dimension",
    ):
        func(*inouts)


@pytest.mark.parametrize(
    "num_k_heads,num_v_heads",
    [(2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 32), (16, 64)],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_flydsl_gdr_decode_strided_inputs_and_split_state_indices(
    num_k_heads, num_v_heads, state_dtype
):
    batch, seq_length, dim = 2, 1, 128
    mixed_qkv = torch.randn(
        batch,
        seq_length,
        num_k_heads * 2 + num_v_heads,
        dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query, key, value = torch.split(
        mixed_qkv, [num_k_heads, num_k_heads, num_v_heads], dim=2
    )
    mixed_ba = torch.randn(batch, num_v_heads * 2, dtype=torch.bfloat16, device="cuda")
    a, b = (
        tensor.reshape(batch, seq_length, num_v_heads)
        for tensor in torch.split(mixed_ba, num_v_heads, dim=-1)
    )
    dt_bias = torch.randn(num_v_heads, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    read_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    write_indices = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    state = torch.randn(5, num_v_heads, dim, dim, dtype=state_dtype, device="cuda")
    reference_state = state.clone()
    reference_state[write_indices.long()] = reference_state[read_indices.long()]
    output = torch.empty_like(value)
    reference_output = torch.empty_like(value)

    common_kwargs = {
        "dt_bias": dt_bias,
        "A_log": A_log,
        "indices": write_indices,
        "use_qk_l2norm": True,
        "need_shuffle_state": False,
    }
    flydsl_gdr_decode(
        query,
        key,
        value,
        a,
        b,
        state=state,
        out=output,
        read_indices=read_indices,
        write_indices=write_indices,
        **common_kwargs,
    )
    flydsl_gdr_decode(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        a.contiguous(),
        b.contiguous(),
        state=reference_state,
        out=reference_output,
        **common_kwargs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)


def test_flydsl_gdr_decode_invalid_indices_zero_output_without_state_write():
    """Graph padding rows produce +0 without a separate output memset kernel."""
    batch, seq_length, num_k_heads, num_v_heads, dim = 4, 1, 16, 32, 128
    query = torch.randn(
        batch,
        seq_length,
        num_k_heads,
        dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn_like(query)
    value = torch.randn(
        batch,
        seq_length,
        num_v_heads,
        dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    a = torch.randn(batch, seq_length, num_v_heads, dtype=query.dtype, device="cuda")
    b = torch.randn_like(a)
    dt_bias = torch.randn(num_v_heads, dtype=query.dtype, device="cuda")
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    read_indices = torch.tensor([1, -1, 3, -1], dtype=torch.int32, device="cuda")
    write_indices = torch.tensor([2, -1, 4, -1], dtype=torch.int32, device="cuda")
    state = torch.randn(5, num_v_heads, dim, dim, dtype=torch.float32, device="cuda")
    state_before = state.clone()
    output = torch.full_like(value, float("nan"))

    flydsl_gdr_decode(
        query,
        key,
        value,
        a,
        b,
        dt_bias=dt_bias,
        A_log=A_log,
        indices=write_indices,
        state=state,
        out=output,
        use_qk_l2norm=True,
        need_shuffle_state=False,
        read_indices=read_indices,
        write_indices=write_indices,
    )
    torch.cuda.synchronize()

    padding_rows = torch.tensor([1, 3], device="cuda")
    assert torch.count_nonzero(output[padding_rows].view(torch.int16)).item() == 0
    untouched_slots = torch.tensor([0, 1, 3], device="cuda")
    assert torch.equal(
        state[untouched_slots].view(torch.int32),
        state_before[untouched_slots].view(torch.int32),
    )


def make_gdn_varlen_inputs(
    batch, tps, num_k_heads, num_v_heads, head_v_dim, gate_dtype, seed=0
):
    """Packed decode/verify inputs laid out as the SGLang GDN layer passes them.

    q/k/v are strided views of one fused ``[1, T, ...]`` projection, the state
    is a slot pool, and slots are interleaved and reversed so a slot index
    never equals its sequence index.
    """
    torch.manual_seed(seed)
    tokens = batch * tps
    k_width = num_k_heads * GDN_HEAD_K_DIM
    mixed = torch.randn(
        1, tokens, 2 * k_width + num_v_heads * head_v_dim, dtype=torch.bfloat16
    )
    q = mixed[..., :k_width].view(1, tokens, num_k_heads, GDN_HEAD_K_DIM)
    k = mixed[..., k_width : 2 * k_width].view(1, tokens, num_k_heads, GDN_HEAD_K_DIM)
    v = mixed[..., 2 * k_width :].view(1, tokens, num_v_heads, head_v_dim)
    state = torch.randn(2 * batch + 1, num_v_heads, head_v_dim, GDN_HEAD_K_DIM)
    return {
        "q": q,
        "k": k,
        "v": v,
        "a": torch.randn(tokens, num_v_heads, dtype=gate_dtype),
        "b": torch.randn(tokens, num_v_heads, dtype=gate_dtype),
        # Qwen3-Next initializes A_log as log(uniform(1, 16)).
        "A_log": torch.empty(num_v_heads).uniform_(1, 16).log(),
        "dt_bias": torch.rand(num_v_heads, dtype=gate_dtype),
        "state": (state * 0.1).to(torch.bfloat16),
        "state_indices": torch.arange(2 * batch, 0, -2, dtype=torch.int32),
        "cu_seqlens": torch.arange(0, tokens + 1, tps, dtype=torch.int32),
    }


def ref_gdn_decode_varlen(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    state_indices,
    cu_seqlens,
    disable_state_update=False,
    intermediate_states=None,
    intermediate_state_indices=None,
):
    """fp32 reference for ``flydsl_gdn_decode_varlen``.

    Returns the fp32 output and updates ``state`` / ``intermediate_states`` in
    place. A negative slot starts from a zero state and is never written back.
    """
    _, tokens, num_k_heads, head_k_dim = k.shape
    num_v_heads, head_v_dim = v.shape[-2:]
    group = num_v_heads // num_k_heads
    qf = q[0].float()
    kf = k[0].float()
    qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * head_k_dim**-0.5
    kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf = qf.repeat_interleave(group, dim=1)
    kf = kf.repeat_interleave(group, dim=1)
    vf = v[0].float()
    gate = a.reshape(tokens, num_v_heads).float() + dt_bias.float()
    decay = torch.exp(
        -torch.exp(A_log.float()) * F.softplus(gate, beta=1.0, threshold=20.0)
    )
    beta = torch.sigmoid(b.reshape(tokens, num_v_heads).float())

    out = torch.empty(tokens, num_v_heads, head_v_dim)
    offsets = cu_seqlens.tolist()
    for seq in range(len(offsets) - 1):
        slot = int(state_indices[seq])
        if slot >= 0:
            h = state[slot].float()
        else:
            h = torch.zeros(num_v_heads, head_v_dim, head_k_dim)
        for step, tok in enumerate(range(offsets[seq], offsets[seq + 1])):
            h = h * decay[tok, :, None, None]
            residual = vf[tok] - (h * kf[tok, :, None, :]).sum(-1)
            h = h + (residual * beta[tok, :, None])[:, :, None] * kf[tok, :, None, :]
            out[tok] = (h * qf[tok, :, None, :]).sum(-1)
            if intermediate_states is not None:
                cslot = int(intermediate_state_indices[seq])
                if cslot >= 0:
                    intermediate_states[cslot, step] = h.to(intermediate_states.dtype)
        if slot >= 0 and not disable_state_update:
            state[slot] = h.to(state.dtype)
    return out[None]


@benchmark()
def test_flydsl_gdn_decode_varlen(
    batch,
    tps,
    num_k_heads,
    num_v_heads,
    head_v_dim,
    gate_dtype,
    cache_states,
    update_state,
):
    """tps=1 is decode with write-back; tps>1 is target verify with snapshots."""
    inputs = make_gdn_varlen_inputs(
        batch, tps, num_k_heads, num_v_heads, head_v_dim, gate_dtype
    )
    initial_state = inputs.pop("state")
    initial_inter = None
    inter_indices = None
    if cache_states:
        # One spare snapshot slot that no sequence maps to; it must stay zero.
        initial_inter = torch.zeros(
            batch + 1,
            tps,
            num_v_heads,
            head_v_dim,
            GDN_HEAD_K_DIM,
            dtype=torch.bfloat16,
        )
        inter_indices = torch.arange(batch, 0, -1, dtype=torch.int32)

    reference_state = initial_state.clone()
    reference_inter = initial_inter.clone() if cache_states else None
    reference_out = ref_gdn_decode_varlen(
        **inputs,
        state=reference_state,
        disable_state_update=not update_state,
        intermediate_states=reference_inter,
        intermediate_state_indices=inter_indices,
    )

    def run_flydsl(state, inter):
        return flydsl_gdn_decode_varlen(
            **inputs,
            state=state,
            disable_state_update=not update_state,
            intermediate_states=inter,
            intermediate_state_indices=inter_indices,
        )

    candidates = {"flydsl": run_flydsl}

    tokens = batch * tps
    k_dim = GDN_HEAD_K_DIM
    flops = tokens * num_v_heads * (7 * k_dim * head_v_dim + 2 * head_v_dim)
    flops += tokens * num_k_heads * 6 * k_dim
    # Q/K/V and gates in, output out, one state read per sequence, plus the
    # final write-back and per-token snapshots when enabled.
    state_bytes = batch * num_v_heads * head_v_dim * k_dim * 2
    nbytes = (
        tokens * (2 * num_k_heads * k_dim + 2 * num_v_heads * head_v_dim) * 2
        + 2 * tokens * num_v_heads * inputs["a"].element_size()
        + state_bytes * (1 + int(update_state) + (tps if cache_states else 0))
    )

    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        perf_state = initial_state.clone()
        perf_inter = initial_inter.clone() if cache_states else None
        _, us = run_perftest(
            lambda state, candidate=candidate, inter=perf_inter: candidate(
                state, inter
            ),
            perf_state,
            num_rotate_args=_perf_rotation_count(perf_state, reference_out),
        )

        candidate_state = initial_state.clone()
        candidate_inter = initial_inter.clone() if cache_states else None
        candidate_out = candidate(candidate_state, candidate_inter)
        errs = [
            checkAllclose(
                reference_out,
                candidate_out.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                msg=f"{name}: GDN varlen output ",
            ),
            checkAllclose(
                reference_state.to(dtypes.fp32),
                candidate_state.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                msg=f"{name}: GDN varlen state pool ",
            ),
        ]
        if cache_states:
            errs.append(
                checkAllclose(
                    reference_inter.to(dtypes.fp32),
                    candidate_inter.to(dtypes.fp32),
                    rtol=1e-2,
                    atol=1e-2,
                    msg=f"{name}: GDN varlen snapshots ",
                )
            )
        assert (
            max(errs) <= _TOL_ERR_RATIO
        ), f"{name}: mismatch ratio exceeds {_TOL_ERR_RATIO:g}: {errs}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(errs)
    return ret


# Driven by main(); pytest collects the fixed cases below.
test_flydsl_gdn_decode_varlen.__test__ = False


@_gdn_varlen_only
@pytest.mark.parametrize("batch,tps", [(1, 1), (5, 1), (3, 2), (2, 4), (4, 5)])
@pytest.mark.parametrize(
    "cache_states,update_state", [(False, True), (True, False), (True, True)]
)
def test_flydsl_gdn_decode_varlen_matches_reference(
    batch, tps, cache_states, update_state
):
    result = test_flydsl_gdn_decode_varlen(
        batch, tps, 16, 48, 128, torch.bfloat16, cache_states, update_state
    )
    assert result["flydsl err"] <= _TOL_ERR_RATIO


@_gdn_varlen_only
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(4, 8), (16, 32)])
def test_flydsl_gdn_decode_varlen_fp32_gates(num_k_heads, num_v_heads):
    result = test_flydsl_gdn_decode_varlen(
        3, 2, num_k_heads, num_v_heads, 128, torch.float32, True, True
    )
    assert result["flydsl err"] <= _TOL_ERR_RATIO


@_gdn_varlen_only
def test_flydsl_gdn_decode_varlen_padding_slot():
    """A negative slot computes from a zero state and leaves the pool alone."""
    inputs = make_gdn_varlen_inputs(3, 2, 16, 48, 128, torch.bfloat16)
    inputs["state_indices"][1] = -1
    reference_state = inputs["state"].clone()
    reference_out = ref_gdn_decode_varlen(**{**inputs, "state": reference_state})
    out = flydsl_gdn_decode_varlen(**inputs)
    for label, ref, got in (
        ("output", reference_out, out),
        ("state pool", reference_state, inputs["state"]),
    ):
        err = checkAllclose(
            ref.float(), got.float(), rtol=1e-2, atol=1e-2, msg=f"padding {label} "
        )
        assert err <= _TOL_ERR_RATIO


@_gdn_varlen_only
def test_flydsl_gdn_decode_varlen_reuses_kernel_across_batch_sizes():
    """The output size is a launch argument, so a new batch size must neither
    build a new kernel nor reuse a stale bound.

    Starts at two tokens: torch reports a size-1 token axis with its own
    stride, so a single-token batch builds its own kernel once.
    """
    misses = None
    for batch in (2, 3, 7):
        inputs = make_gdn_varlen_inputs(batch, 1, 16, 48, 128, torch.bfloat16)
        reference_state = inputs["state"].clone()
        reference_out = ref_gdn_decode_varlen(**{**inputs, "state": reference_state})
        out = flydsl_gdn_decode_varlen(**inputs, viter=1)
        err = checkAllclose(
            reference_out, out.float(), rtol=1e-2, atol=1e-2, msg=f"{batch=} "
        )
        assert err <= _TOL_ERR_RATIO
        info = create_gdn_decode_verify_kernel.cache_info()
        if misses is None:
            misses = info.misses
        assert info.misses == misses, f"batch {batch} built a new kernel"


@_gdn_varlen_only
def test_flydsl_gdn_decode_varlen_rejects_short_snapshot_buffer():
    inputs = make_gdn_varlen_inputs(2, 4, 16, 48, 128, torch.bfloat16)
    inter = torch.zeros(4, 3, 48, 128, GDN_HEAD_K_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="steps per slot"):
        flydsl_gdn_decode_varlen(
            **inputs,
            intermediate_states=inter,
            intermediate_state_indices=torch.tensor([0, 2], dtype=torch.int32),
        )


@_gdn_varlen_only
def test_flydsl_gdn_decode_varlen_validate_data_rejects_ragged_offsets():
    inputs = make_gdn_varlen_inputs(2, 2, 16, 48, 128, torch.bfloat16)
    inputs["cu_seqlens"] = torch.tensor([0, 1, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="every sequence"):
        flydsl_gdn_decode_varlen(**inputs, validate_data=True)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL GDR decode unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL GDR decode correctness + performance sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16, dtypes.fp16],
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
        help="Input dtype. Example: -d bf16 fp16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 128],
        help="Batch sizes.",
    )
    parser.add_argument(
        "-s",
        "--sq",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Decode sequence lengths.",
    )
    parser.add_argument(
        "--head-configs",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(2, 8, 128, 128), (16, 32, 128, 128)],
        metavar="NKH,NVH,K,V",
        help="Head configs as num_k_heads,num_v_heads,head_k_dim,head_v_dim.",
    )
    parser.add_argument(
        "--l2norm",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="Whether to normalize Q/K in-kernel (0 or 1).",
    )
    parser.add_argument(
        "--tps",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="Tokens per sequence for the packed GDN sweep: 1 is decode with "
        "state write-back, >1 is target verify with per-token snapshots.",
    )
    parser.add_argument(
        "--gdn-heads",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(16, 48, 128)],
        metavar="NKH,NVH,V",
        help="Packed GDN head configs as num_k_heads,num_v_heads,head_v_dim "
        "(head_k_dim is fixed at 128). Default: Qwen3.8 at TP1.",
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch, sq, head_config, l2norm in itertools.product(
        args.dtype,
        args.batch,
        args.sq,
        args.head_configs,
        args.l2norm,
    ):
        if len(head_config) != 4:
            parser.error(
                "--head-configs entries must contain "
                "num_k_heads,num_v_heads,head_k_dim,head_v_dim"
            )
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = head_config
        try:
            if batch <= 0 or sq <= 0:
                raise ValueError(
                    f"batch and sequence length must be positive, got {batch=} {sq=}"
                )
            _validate_head_config(
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
        except ValueError as exc:
            parser.error(str(exc))
        rows.append(
            test_flydsl_gdr_decode(
                batch,
                sq,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                dtype,
                bool(l2norm),
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "FlyDSL GDR decode summary (markdown):\n%s", df.to_markdown(index=False)
    )

    if get_gfx() not in GDN_VARLEN_GFX:
        aiter.logger.warning(
            "%s: skipping the packed GDN decode/verify sweep (gfx950 only)", get_gfx()
        )
        return
    # The packed kernel takes bf16 q/k/v and state only; other -d entries are
    # covered by the GDR sweep above.
    gdn_dtypes = [dtype for dtype in args.dtype if dtype == dtypes.bf16]
    if len(gdn_dtypes) != len(args.dtype):
        aiter.logger.warning(
            "packed GDN decode/verify is bf16-only; skipping %s",
            [d for d in args.dtype if d != dtypes.bf16],
        )
    rows = []
    for dtype, batch, tps, heads in itertools.product(
        gdn_dtypes, args.batch, args.tps, args.gdn_heads
    ):
        if len(heads) != 3:
            parser.error("--gdn-heads entries must be num_k_heads,num_v_heads,V")
        # a/b/dt_bias arrive in the model dtype, as in SGLang's Qwen3.5 layer.
        # Decode writes the state back; target verify snapshots every token and
        # leaves the commit to the caller, as the SGLang GDN backend does.
        rows.append(
            test_flydsl_gdn_decode_varlen(batch, tps, *heads, dtype, tps > 1, tps == 1)
        )
    aiter.logger.info(
        "FlyDSL GDN packed decode/verify summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
