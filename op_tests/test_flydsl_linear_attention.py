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
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
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


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    o: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_triton_kernel(
    out,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state,
    indices,
    scale,
    use_qk_l2norm_in_kernel,
):
    fused_sigmoid_gating_delta_rule_update(
        out,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None,
    )


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
    run_triton_kernel(
        out,
        A_log,
        dt_bias,
        query,
        key,
        value,
        a,
        b,
        state,
        indices,
        float(1.0 / (args.head_k_dim**0.5)),
        args.use_qk_l2norm,
    )


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


if __name__ == "__main__":
    main()
