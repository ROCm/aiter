# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL Linear Attention (GDR and KDA decode) regressions.

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
from torch._subclasses.fake_tensor import FakeTensorMode

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_kernels import (
    flydsl_gdr_decode,
    flydsl_kda_decode,
    flydsl_kda_decode_with_f_b,
    is_flydsl_kda_decode_supported,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="FlyDSL GDR decode requires gfx942 or gfx950",
)
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


# Fused KDA decode is fixed to the gfx950 Kimi-K3 TP8 layout.
_requires_kda = pytest.mark.skipif(
    not is_flydsl_kda_decode_supported(),
    reason="FlyDSL KDA decode requires gfx950",
)
_KDA_DEVICE = torch.device("cuda")
_KDA_HEADS = 12
_KDA_DIM = 128
_KDA_CHANNELS = 3 * _KDA_HEADS * _KDA_DIM
_KDA_CONV_WIDTH = 4
_KDA_LOWER_BOUND = -5.0
_KDA_NORM_EPS = 1e-5


@dataclass
class KdaInputs:
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor


def _make_kda_inputs(
    batch: int, seed: int = 20260728, conv_width_major: bool = False
) -> KdaInputs:
    generator = torch.Generator(device=_KDA_DEVICE).manual_seed(seed + batch)
    slots = batch + 2

    x_storage = torch.randn(
        (batch, _KDA_CHANNELS + 17),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
        generator=generator,
    )
    x = x_storage[:, :_KDA_CHANNELS]
    conv_weight = 0.1 * torch.randn(
        (_KDA_CHANNELS, _KDA_CONV_WIDTH),
        dtype=torch.float32,
        device=_KDA_DEVICE,
        generator=generator,
    )

    # Kimi's hybrid cache can pad the slot stride. Exercise that layout
    # explicitly while keeping each slot's inner dimensions contiguous.
    conv_storage = torch.randn(
        (slots, _KDA_CHANNELS * (_KDA_CONV_WIDTH - 1) + 19),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
        generator=generator,
    )
    conv_state = conv_storage[:, : _KDA_CHANNELS * (_KDA_CONV_WIDTH - 1)]
    if conv_width_major:
        # SGLang keeps the cache as [slots, width - 1, channels] and passes
        # conv_states.transpose(-1, -2), so the inner strides are (1, channels).
        conv_state = conv_state.view(
            slots, _KDA_CONV_WIDTH - 1, _KDA_CHANNELS
        ).transpose(1, 2)
    else:
        conv_state = conv_state.view(slots, _KDA_CHANNELS, _KDA_CONV_WIDTH - 1)
    state_storage = 0.01 * torch.randn(
        (slots, _KDA_HEADS * _KDA_DIM * _KDA_DIM + 23),
        dtype=torch.float32,
        device=_KDA_DEVICE,
        generator=generator,
    )
    state = state_storage[:, : _KDA_HEADS * _KDA_DIM * _KDA_DIM].view(
        slots, _KDA_HEADS, _KDA_DIM, _KDA_DIM
    )

    raw_beta_storage = torch.randn(
        (1, batch, _KDA_HEADS + 1),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
        generator=generator,
    )
    output_gate_storage = torch.randn(
        (batch, _KDA_HEADS * _KDA_DIM + 7),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
        generator=generator,
    )
    return KdaInputs(
        x=x,
        conv_weight=conv_weight,
        conv_state=conv_state,
        raw_g=torch.randn(
            (1, batch, _KDA_HEADS, _KDA_DIM),
            dtype=torch.bfloat16,
            device=_KDA_DEVICE,
            generator=generator,
        ),
        raw_beta=raw_beta_storage[:, :, :_KDA_HEADS],
        A_log=0.5
        * torch.randn(
            (_KDA_HEADS,),
            dtype=torch.float32,
            device=_KDA_DEVICE,
            generator=generator,
        ),
        dt_bias=0.1
        * torch.randn(
            (_KDA_HEADS * _KDA_DIM,),
            dtype=torch.float32,
            device=_KDA_DEVICE,
            generator=generator,
        ),
        state=state,
        state_indices=torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device=_KDA_DEVICE,
        ),
        output_gate=output_gate_storage[:, : _KDA_HEADS * _KDA_DIM].view(
            batch, _KDA_HEADS, _KDA_DIM
        ),
        norm_weight=torch.randn(
            (_KDA_DIM,),
            dtype=torch.bfloat16,
            device=_KDA_DEVICE,
            generator=generator,
        ),
    )


def _copy_kda_inputs(inputs: KdaInputs) -> KdaInputs:
    def clone_preserving_strides(tensor: torch.Tensor) -> torch.Tensor:
        clone = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        clone.copy_(tensor)
        return clone

    return KdaInputs(
        **{
            name: clone_preserving_strides(getattr(inputs, name))
            for name in KdaInputs.__dataclass_fields__
        }
    )


def _kda_reference(inputs: KdaInputs) -> torch.Tensor:
    batch = inputs.x.shape[0]
    output = torch.zeros(
        (1, batch, _KDA_HEADS, _KDA_DIM),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
    )
    dt_bias = inputs.dt_bias.view(_KDA_HEADS, _KDA_DIM)

    for batch_idx in range(batch):
        state_idx = int(inputs.state_indices[batch_idx])
        if state_idx <= 0:
            continue

        history = inputs.conv_state[state_idx]
        conv_values = torch.cat(
            (
                history.float(),
                inputs.x[batch_idx, :, None].float(),
            ),
            dim=-1,
        )
        packed_qkv = F.silu((conv_values * inputs.conv_weight).sum(dim=-1)).to(
            torch.bfloat16
        )
        history[:, 0].copy_(history[:, 1])
        history[:, 1].copy_(history[:, 2])
        history[:, 2].copy_(inputs.x[batch_idx])

        q, k, v = packed_qkv.view(
            3,
            _KDA_HEADS,
            _KDA_DIM,
        ).unbind(0)
        for head_idx in range(_KDA_HEADS):
            q_head = q[head_idx].float()
            k_head = k[head_idx].float()
            q_head = q_head * torch.rsqrt(q_head.square().sum() + 1e-6)
            q_head = q_head * (_KDA_DIM**-0.5)
            k_head = k_head * torch.rsqrt(k_head.square().sum() + 1e-6)

            a = inputs.A_log[head_idx].exp()
            decay = (
                _KDA_LOWER_BOUND
                * torch.sigmoid(
                    (
                        inputs.raw_g[
                            0,
                            batch_idx,
                            head_idx,
                        ].float()
                        + dt_bias[head_idx]
                    )
                    * a
                )
            ).exp()
            decayed_state = inputs.state[state_idx, head_idx] * decay[None, :]
            state_dot_k = decayed_state @ k_head
            state_dot_q = decayed_state @ q_head
            k_dot_q = torch.dot(k_head, q_head)
            beta = torch.sigmoid(
                inputs.raw_beta[
                    0,
                    batch_idx,
                    head_idx,
                ].float()
            )
            v_new = (v[head_idx].float() - state_dot_k) * beta
            inputs.state[state_idx, head_idx].copy_(
                decayed_state + v_new[:, None] * k_head[None, :]
            )

            # The model materializes recurrent output in BF16 before the
            # normalization/gating operation.
            recurrent = (state_dot_q + v_new * k_dot_q).to(torch.bfloat16)
            recurrent_f32 = recurrent.float()
            inv_rms = torch.rsqrt(recurrent_f32.square().mean() + _KDA_NORM_EPS)
            output[0, batch_idx, head_idx] = (
                recurrent_f32
                * inv_rms
                * inputs.norm_weight.float()
                * torch.sigmoid(
                    inputs.output_gate[
                        batch_idx,
                        head_idx,
                    ].float()
                )
            ).to(torch.bfloat16)
    return output


# Worst element error vs the fp32 reference over this file on gfx950: output
# 1.2e-4 (|ref| up to 3.9, needs atol 9.7e-5 at rtol 1e-3), state 8.8e-8.
_KDA_OUTPUT_TOL = {"atol": 1e-3, "rtol": 1e-3}
_KDA_STATE_TOL = {"atol": 1e-6, "rtol": 1e-3}
# (output tol, state tol, allowed mismatch ratio) per benchmark candidate.
# At larger batches a convolved V value can round one BF16 ULP away from the
# reference, which moves one state row past the tight state tolerance.
_KDA_FLYDSL_TOLS = (_KDA_OUTPUT_TOL, _KDA_STATE_TOL, _TOL_ERR_RATIO)
# The Triton baseline keeps the convolved Q/K/V in FP32 and computes SiLU as
# x * sigmoid(x), so it only matches the BF16-rounded reference to the GDR
# decode tolerance, with a few outputs one BF16 ULP away.
_KDA_TRITON_TOLS = ({"atol": 1e-3, "rtol": 1e-3}, {"atol": 1e-3, "rtol": 1e-3}, 1e-2)


def _assert_kda_close(
    reference: torch.Tensor,
    actual: torch.Tensor,
    tol: dict[str, float],
    msg: str,
    tol_err_ratio: float = 0.0,
) -> float:
    err = checkAllclose(
        reference.float(),
        actual.float(),
        tol_err_ratio=tol_err_ratio,
        msg=msg,
        **tol,
    )
    assert err <= tol_err_ratio, f"{msg}: {err:.4%} of elements outside {tol}"
    return err


def _run_kda(inputs: KdaInputs, out: torch.Tensor | None = None) -> torch.Tensor:
    return flydsl_kda_decode(
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_g=inputs.raw_g,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_KDA_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_KDA_NORM_EPS,
        out=out,
    )


def _make_kda_fb_inputs(
    batch: int,
    seed: int = 20260728,
    conv_width_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, KdaInputs]:
    generator = torch.Generator(device=_KDA_DEVICE).manual_seed(seed + 10_000 + batch)
    f_a_storage = torch.randn(
        (batch, _KDA_DIM + 5),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
        generator=generator,
    )
    f_a = f_a_storage[:, :_KDA_DIM]
    f_b_weight = (
        0.05
        * torch.randn(
            (_KDA_HEADS, _KDA_DIM, _KDA_DIM),
            dtype=torch.bfloat16,
            device=_KDA_DEVICE,
            generator=generator,
        )
    ).to(torch.bfloat16)
    inputs = _make_kda_inputs(batch, seed, conv_width_major)
    projected = F.linear(
        f_a.float(),
        f_b_weight.view(_KDA_HEADS * _KDA_DIM, _KDA_DIM).float(),
    ).to(torch.bfloat16)
    inputs.raw_g.copy_(projected.view(1, batch, _KDA_HEADS, _KDA_DIM))
    return f_a, f_b_weight, inputs


def _run_kda_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    inputs: KdaInputs,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return flydsl_kda_decode_with_f_b(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_KDA_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_KDA_NORM_EPS,
        out=out,
    )


def _run_kda_triton(inputs: KdaInputs, cu_seqlens: torch.Tensor) -> torch.Tensor:
    batch = inputs.x.shape[0]
    return fused_kda_decode(
        mixed_qkv=inputs.x,
        conv_state=inputs.conv_state,
        conv_weight=inputs.conv_weight,
        gate=inputs.raw_g,
        beta=inputs.raw_beta,
        out_gate=inputs.output_gate.flatten(1),
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        ssm_state=inputs.state,
        ssm_state_indices=inputs.state_indices,
        cu_seqlens=cu_seqlens,
        norm_weight=inputs.norm_weight,
        norm_eps=_KDA_NORM_EPS,
        head_dim=_KDA_DIM,
        num_local_heads=_KDA_HEADS,
        lower_bound=_KDA_LOWER_BOUND,
    ).view(1, batch, _KDA_HEADS, _KDA_DIM)


def _kda_decode_work(batch: int, fuse_f_b: bool) -> tuple[int, int]:
    hidden = _KDA_HEADS * _KDA_DIM
    # Per token and head, the recurrence costs about 7*K*V + 2*V FLOPs as in
    # _recurrent_decode_work; the width-4 Q/K/V convolution adds 2*W per
    # channel and the fused f_b projection 2*K per output.
    flops = batch * _KDA_HEADS * (7 * _KDA_DIM * _KDA_DIM + 2 * _KDA_DIM)
    flops += batch * _KDA_CHANNELS * 2 * _KDA_CONV_WIDTH
    # Useful bytes: read+write both caches, read x/raw_g/beta/output gate and
    # the shared weights once, write the BF16 output.
    nbytes = batch * (
        2 * hidden * _KDA_DIM * 4
        + 2 * _KDA_CHANNELS * (_KDA_CONV_WIDTH - 1) * 2
        + (_KDA_CHANNELS + 3 * hidden + _KDA_HEADS) * 2
        + 4
    )
    nbytes += _KDA_CHANNELS * _KDA_CONV_WIDTH * 4 + hidden * 4 + _KDA_HEADS * 4
    nbytes += _KDA_DIM * 2
    if fuse_f_b:
        flops += batch * hidden * 2 * _KDA_DIM
        # f_a and f_b_weight replace the raw_g read.
        nbytes += batch * (_KDA_DIM - hidden) * 2 + hidden * _KDA_DIM * 2
    return flops, nbytes


@benchmark()
def test_flydsl_kda_decode(batch, conv_layout):
    f_a, f_b_weight, seed = _make_kda_fb_inputs(
        batch, conv_width_major=conv_layout == "width_major"
    )
    reference_inputs = _copy_kda_inputs(seed)
    reference = _kda_reference(reference_inputs)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int64, device=_KDA_DEVICE)

    # raw_g holds the f_b projection, so every candidate shares one reference.
    candidates = {
        "flydsl": (_run_kda, False, _KDA_FLYDSL_TOLS),
        "flydsl_f_b": (
            lambda inputs: _run_kda_with_f_b(f_a, f_b_weight, inputs),
            True,
            _KDA_FLYDSL_TOLS,
        ),
    }
    # The Triton kernel takes the conv cache as SGLang's transposed view only.
    if conv_layout == "width_major":
        candidates["triton"] = (
            lambda inputs: _run_kda_triton(inputs, cu_seqlens),
            False,
            _KDA_TRITON_TOLS,
        )

    ret = {"gfx": get_gfx()}
    for name, (candidate, fuse_f_b, tols) in candidates.items():
        output_tol, state_tol, tol_err_ratio = tols
        # Both caches update in place; run_perftest rotates deep copies, which
        # keep the padded and transposed cache strides.
        perf_inputs = _copy_kda_inputs(seed)
        _, us = run_perftest(
            candidate,
            perf_inputs,
            num_rotate_args=_perf_rotation_count(
                perf_inputs.state, perf_inputs.conv_state
            ),
        )

        actual_inputs = _copy_kda_inputs(seed)
        actual = candidate(actual_inputs)
        err_out = _assert_kda_close(
            reference,
            actual,
            output_tol,
            f"{name}: KDA decode output ",
            tol_err_ratio,
        )
        err_state = _assert_kda_close(
            reference_inputs.state,
            actual_inputs.state,
            state_tol,
            f"{name}: KDA decode state ",
            tol_err_ratio,
        )
        assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)
        flops, nbytes = _kda_decode_work(batch, fuse_f_b)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err_out, err_state)
    return ret


test_flydsl_kda_decode.__test__ = False


@_requires_kda
def test_flydsl_kda_decode_exports() -> None:
    import aiter.ops.flydsl as flydsl_ops
    from aiter.ops.flydsl import linear_attention_kernels

    expected = {
        "flydsl_kda_decode",
        "flydsl_kda_decode_with_f_b",
        "is_flydsl_kda_decode_supported",
    }
    assert expected <= set(linear_attention_kernels.__all__)
    assert expected <= set(flydsl_ops.__all__)
    assert flydsl_ops.flydsl_kda_decode is flydsl_kda_decode
    assert flydsl_ops.flydsl_kda_decode_with_f_b is flydsl_kda_decode_with_f_b
    assert flydsl_ops.is_flydsl_kda_decode_supported is is_flydsl_kda_decode_supported
    assert is_flydsl_kda_decode_supported(0)
    assert not is_flydsl_kda_decode_supported("cpu")


@_requires_kda
@pytest.mark.parametrize("conv_layout", ["channel_major", "width_major"])
@pytest.mark.parametrize("batch", [1, 8, 16])
def test_flydsl_kda_decode_matches_reference(batch: int, conv_layout: str) -> None:
    result = test_flydsl_kda_decode(batch, conv_layout)
    assert result["flydsl err"] == 0
    assert result["flydsl_f_b err"] == 0


@_requires_kda
@pytest.mark.parametrize("fused_f_b", [False, True])
def test_flydsl_kda_decode_non_positive_slots_do_not_modify_caches(
    fused_f_b: bool,
) -> None:
    f_a, f_b_weight, inputs = _make_kda_fb_inputs(batch=2)
    inputs.state_indices.copy_(
        torch.tensor([0, -1], dtype=torch.int32, device=_KDA_DEVICE)
    )
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    if fused_f_b:
        actual = _run_kda_with_f_b(f_a, f_b_weight, inputs)
    else:
        actual = _run_kda(inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


@_requires_kda
@pytest.mark.parametrize("fused_f_b", [False, True])
def test_flydsl_kda_decode_out_of_range_slots_do_not_modify_caches(
    fused_f_b: bool,
) -> None:
    batch = 4
    if fused_f_b:
        f_a, f_b_weight, inputs = _make_kda_fb_inputs(batch)
    else:
        inputs = _make_kda_inputs(batch)
    # Give the recurrent pool fewer slots than the convolution pool so the
    # bound must come from the smaller of the two caches.
    state_slots = inputs.state.shape[0] - 2
    assert state_slots < inputs.conv_state.shape[0]
    inputs.state = inputs.state[:state_slots]
    inputs.state_indices.copy_(
        torch.tensor(
            [state_slots, state_slots + 1, 2**31 - 1, 0],
            dtype=torch.int32,
            device=_KDA_DEVICE,
        )
    )
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    if fused_f_b:
        actual = _run_kda_with_f_b(f_a, f_b_weight, inputs)
    else:
        actual = _run_kda(inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


def _move_kda_to_large_caches(
    inputs: KdaInputs,
    slot_indices: tuple[int, ...],
    slots: int,
    conv_slot_stride: int,
) -> KdaInputs:
    """Copy each batch slot into zeroed caches large enough to overflow 32-bit
    byte offsets, at the high ``slot_indices``."""
    conv_storage = torch.zeros(
        (slots, conv_slot_stride),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
    )
    conv_state = conv_storage[:, : _KDA_CHANNELS * (_KDA_CONV_WIDTH - 1)].view(
        slots, _KDA_CHANNELS, _KDA_CONV_WIDTH - 1
    )
    state = torch.zeros(
        (slots, _KDA_HEADS, _KDA_DIM, _KDA_DIM),
        dtype=torch.float32,
        device=_KDA_DEVICE,
    )
    for batch_idx, slot in enumerate(slot_indices):
        small_slot = int(inputs.state_indices[batch_idx])
        conv_state[slot].copy_(inputs.conv_state[small_slot])
        state[slot].copy_(inputs.state[small_slot])
    large = _copy_kda_inputs(inputs)
    large.conv_state = conv_state
    large.state = state
    large.state_indices = torch.tensor(
        slot_indices,
        dtype=torch.int32,
        device=_KDA_DEVICE,
    )
    return large


@_requires_kda
@pytest.mark.parametrize("fused_f_b", [False, True])
def test_flydsl_kda_decode_large_cache_slot_offsets(fused_f_b: bool) -> None:
    # 11,000 [12, 128, 128] FP32 slots cross 2**31 elements of state, and a
    # 2**18-element BF16 conv slot stride crosses 2**32 bytes past slot 8,192.
    slots = 11_000
    slot_indices = (5_500, 10_999)
    batch = len(slot_indices)
    if fused_f_b:
        f_a, f_b_weight, seed = _make_kda_fb_inputs(batch)
    else:
        seed = _make_kda_inputs(batch)
    reference_inputs = _copy_kda_inputs(seed)
    actual_inputs = _move_kda_to_large_caches(
        seed,
        slot_indices,
        slots,
        conv_slot_stride=1 << 18,
    )

    reference = _kda_reference(reference_inputs)
    if fused_f_b:
        actual = _run_kda_with_f_b(f_a, f_b_weight, actual_inputs)
    else:
        actual = _run_kda(actual_inputs)
    torch.cuda.synchronize()

    _assert_kda_close(reference, actual, _KDA_OUTPUT_TOL, "output")
    for batch_idx, slot in enumerate(slot_indices):
        small_slot = int(reference_inputs.state_indices[batch_idx])
        _assert_kda_close(
            reference_inputs.state[small_slot],
            actual_inputs.state[slot],
            _KDA_STATE_TOL,
            "state",
        )
        assert torch.equal(
            reference_inputs.conv_state[small_slot],
            actual_inputs.conv_state[slot],
        )
        actual_inputs.state[slot].zero_()
        actual_inputs.conv_state[slot].zero_()
    # Nothing outside the addressed slots may be touched by wrapped offsets.
    assert torch.count_nonzero(actual_inputs.state) == 0
    assert torch.count_nonzero(actual_inputs.conv_state) == 0


@_requires_kda
def test_flydsl_kda_decode_rejects_invalid_projection_inputs() -> None:
    f_a, f_b_weight, inputs = _make_kda_fb_inputs(batch=1)

    with pytest.raises(ValueError, match="`f_a` must have rank 2"):
        _run_kda_with_f_b(f_a.unsqueeze(0), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_a` must have dtype"):
        _run_kda_with_f_b(f_a.float(), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have shape"):
        _run_kda_with_f_b(f_a, f_b_weight[:, :, :-1], inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have inner strides"):
        _run_kda_with_f_b(f_a, f_b_weight.transpose(1, 2), inputs)


@_requires_kda
def test_flydsl_kda_decode_rejects_invalid_input_rank() -> None:
    inputs = _make_kda_inputs(batch=1)
    inputs.x = inputs.x.unsqueeze(0)

    with pytest.raises(ValueError, match="`x` must have rank 2"):
        _run_kda(inputs)


@_requires_kda
@pytest.mark.parametrize("fused_f_b", [False, True])
@pytest.mark.parametrize(
    "name, strides",
    [
        ("conv_state", (0, 3, 1)),
        ("conv_state", (_KDA_CHANNELS * 3 - 1, 3, 1)),
        ("conv_state", (_KDA_CHANNELS * 3, 0, 1)),
        ("conv_state", (_KDA_CHANNELS * 3, 3, 0)),
        ("conv_state", (_KDA_CHANNELS * 6, 6, 2)),
        ("state", (0, _KDA_DIM * _KDA_DIM, _KDA_DIM, 1)),
        (
            "state",
            (_KDA_HEADS * _KDA_DIM * _KDA_DIM - 1, _KDA_DIM * _KDA_DIM, _KDA_DIM, 1),
        ),
        ("out", (0, _KDA_HEADS * _KDA_DIM, 0, 1)),
        ("out", (0, _KDA_HEADS * _KDA_DIM, _KDA_DIM - 1, 1)),
        ("out", (0, 0, _KDA_DIM, 1)),
        ("out", (0, (_KDA_HEADS - 1) * (_KDA_DIM + 7) + _KDA_DIM - 1, _KDA_DIM + 7, 1)),
    ],
)
def test_flydsl_kda_decode_rejects_unsafe_write_layouts(
    monkeypatch: pytest.MonkeyPatch,
    fused_f_b: bool,
    name: str,
    strides: tuple[int, ...],
) -> None:
    from aiter.ops.flydsl import linear_attention_kernels

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Unsafe write layouts must be rejected before kernel compilation.")

    monkeypatch.setattr(
        linear_attention_kernels, "create_kda_decode_kernel", unexpected_compile
    )
    with FakeTensorMode():
        f_a, f_b_weight, inputs = _make_kda_fb_inputs(batch=2)
        out = torch.empty(
            (1, 2, _KDA_HEADS, _KDA_DIM), dtype=torch.bfloat16, device=_KDA_DEVICE
        )
        tensor = out if name == "out" else getattr(inputs, name)
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_KDA_DEVICE
        )
        if name == "out":
            out = view
        else:
            setattr(inputs, name, view)
        with pytest.raises(ValueError, match=f"`{name}`.*non-overlapping"):
            if fused_f_b:
                _run_kda_with_f_b(f_a, f_b_weight, inputs, out=out)
            else:
                _run_kda(inputs, out=out)


@_requires_kda
@pytest.mark.parametrize("fused_f_b", [False, True])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("trailing_padding", [0, 17])
def test_flydsl_kda_decode_accepts_non_overlapping_write_layouts(
    fused_f_b: bool,
    padding: int,
    trailing_padding: int,
) -> None:
    f_a, f_b_weight, inputs = _make_kda_fb_inputs(batch=2)
    if padding == 0:
        inputs.conv_state = inputs.conv_state.contiguous()
        inputs.state = inputs.state.contiguous()
    reference_inputs = _copy_kda_inputs(inputs)
    reference = _kda_reference(reference_inputs)
    head_stride = _KDA_DIM + padding
    batch_stride = (_KDA_HEADS - 1) * head_stride + _KDA_DIM + trailing_padding
    out = torch.empty_strided(
        (1, 2, _KDA_HEADS, _KDA_DIM),
        (0, batch_stride, head_stride, 1),
        dtype=torch.bfloat16,
        device=_KDA_DEVICE,
    )
    if fused_f_b:
        actual = _run_kda_with_f_b(f_a, f_b_weight, inputs, out=out)
    else:
        actual = _run_kda(inputs, out=out)
    torch.cuda.synchronize()

    assert actual is out
    _assert_kda_close(reference, actual, _KDA_OUTPUT_TOL, "output")
    _assert_kda_close(reference_inputs.state, inputs.state, _KDA_STATE_TOL, "state")
    assert torch.equal(reference_inputs.conv_state, inputs.conv_state)


@_requires_kda
@pytest.mark.parametrize(
    "fused_f_b, name, dim",
    # Convolution inner strides are fixed by dense-cache validation, so their
    # byte span cannot overflow. Invalid layouts are covered above instead.
    [
        (fused_f_b, name, dim)
        for fused_f_b in (False, True)
        for name, dim in (
            ("x", 0),
            ("conv_weight", 0),
            ("conv_weight", 1),
            ("raw_beta", 1),
            ("output_gate", 0),
            ("output_gate", 1),
            ("out", 1),
            ("out", 2),
        )
    ]
    + [(False, "raw_g", 1), (True, "f_a", 0), (True, "f_b_weight", 0)],
)
@pytest.mark.parametrize("stride", [2**30, 2**31])
def test_flydsl_kda_decode_rejects_i32_address_overflow(
    monkeypatch: pytest.MonkeyPatch,
    fused_f_b: bool,
    name: str,
    dim: int,
    stride: int,
) -> None:
    from aiter.ops.flydsl import linear_attention_kernels

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Unsafe addressing must be rejected before kernel compilation.")

    monkeypatch.setattr(
        linear_attention_kernels, "create_kda_decode_kernel", unexpected_compile
    )
    # Fake views exercise multi-GiB spans without allocating or accessing them.
    # 2**30 fits the launcher parameter but overflows the resulting byte span;
    # 2**31 also overflows the stride parameter itself.
    with FakeTensorMode():
        inputs = _make_kda_inputs(batch=2)
        f_a = torch.empty((2, _KDA_DIM), dtype=torch.bfloat16, device=_KDA_DEVICE)
        f_b_weight = torch.empty(
            (_KDA_HEADS, _KDA_DIM, _KDA_DIM), dtype=torch.bfloat16, device=_KDA_DEVICE
        )
        out = torch.empty(
            (1, 2, _KDA_HEADS, _KDA_DIM), dtype=torch.bfloat16, device=_KDA_DEVICE
        )
        tensor = {"f_a": f_a, "f_b_weight": f_b_weight, "out": out}.get(name)
        if tensor is None:
            tensor = getattr(inputs, name)
        strides = list(tensor.stride())
        strides[dim] = stride
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_KDA_DEVICE
        )
        if name == "f_a":
            f_a = view
        elif name == "f_b_weight":
            f_b_weight = view
        elif name == "out":
            out = view
        else:
            setattr(inputs, name, view)
        with pytest.raises(ValueError, match=f"`{name}`.*int32"):
            if fused_f_b:
                _run_kda_with_f_b(f_a, f_b_weight, inputs, out=out)
            else:
                _run_kda(inputs, out=out)


@_requires_kda
def test_flydsl_kda_decode_unused_singleton_stride_must_fit_int32() -> None:
    from aiter.ops.flydsl.linear_attention_kernels import _check_i32_addressing

    tensor = torch.empty_strided(
        (1, _KDA_CHANNELS), (2**31, 1), dtype=torch.bfloat16, device="meta"
    )
    with pytest.raises(ValueError, match="`x` strides.*int32"):
        _check_i32_addressing("x", tensor)


@_requires_kda
def test_flydsl_kda_decode_i32_byte_span_boundary() -> None:
    from aiter.ops.flydsl.linear_attention_kernels import _check_i32_addressing

    # The starting offset fits for both tensors; the entire final BF16 element
    # must also fit. Use meta tensors to avoid allocating nearly two GiB.
    safe = torch.empty_strided((2,), (2**30 - 2,), dtype=torch.bfloat16, device="meta")
    unsafe = torch.empty_strided(
        (2,), (2**30 - 1,), dtype=torch.bfloat16, device="meta"
    )
    _check_i32_addressing("x", safe)
    with pytest.raises(ValueError, match="`x` byte span.*int32"):
        _check_i32_addressing("x", unsafe)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL GDR decode unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL GDR and KDA decode correctness + performance sweep",
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
        "--conv-layout",
        nargs="*",
        choices=["channel_major", "width_major"],
        default=["channel_major", "width_major"],
        help="KDA conv cache layout; width_major is SGLang's transposed view.",
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

    if not is_flydsl_kda_decode_supported():
        aiter.logger.warning("FlyDSL KDA decode unsupported on %s; skipping", get_gfx())
        return
    rows = [
        test_flydsl_kda_decode(batch, conv_layout)
        for conv_layout, batch in itertools.product(args.conv_layout, args.batch)
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "FlyDSL KDA decode summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
