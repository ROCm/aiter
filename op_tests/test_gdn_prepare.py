# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for the fused FlyDSL GDN prepare path.

The direct sweep compares the fused FlyDSL prepare kernel and the Triton
three-dispatch prepare path against an fp32 torch specification. The pipeline
sweep drives the model-facing ``chunk_gated_delta_rule_opt_vk`` wrapper with a
preallocated output buffer and compares both prepare backends against an fp32
recurrent torch reference.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import os
import warnings
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
BT = 64
ATOL_G = 1e-3
ATOL_WU = 5e-2
PERF_ITERS = 21
PERF_WARMUP = 3


@dataclass(frozen=True)
class GDNShape:
    t: int
    hg: int
    h: int
    k: int = 128
    v: int = 128
    seq_lens: tuple[int, ...] | None = None
    num_decodes: int = 0

    @property
    def mode(self) -> str:
        if self.seq_lens is None:
            return "dense"
        if self.num_decodes:
            return "decode_prefix"
        return "varlen"

    @property
    def num_decode_tokens(self) -> int:
        if self.seq_lens is None:
            return 0
        return sum(self.seq_lens[: self.num_decodes])

    @property
    def active_seq_lens(self) -> tuple[int, ...]:
        if self.seq_lens is None:
            return ()
        return self.seq_lens[self.num_decodes :]


# Qwen3.5-35B/397B model configs use Hk=16, Hv={32,64}, and K=V=128;
# tensor parallelism maps them to Hg=Hk/TP and H=Hv/TP.
PREPARE_SHAPES = {
    "qwen35_35b_tp2": GDNShape(t=2500, hg=8, h=16),
    "qwen35_397b_tp8": GDNShape(t=1024, hg=2, h=8),
    # Non-aligned dense and packed lengths exercise tail guards.
    "dense_tail": GDNShape(t=300, hg=4, h=8),
    "varlen_ragged": GDNShape(t=600, hg=4, h=16, seq_lens=(128, 172, 300)),
    # Skip two decode tokens; tensors contain the remaining prefill tokens.
    "decode_prefix": GDNShape(
        t=428, hg=4, h=8, seq_lens=(1, 1, 128, 300), num_decodes=2
    ),
}

# Compact shapes keep the recurrent reference fast while preserving each path.
PIPELINE_SHAPES = {
    "dense_tail": GDNShape(t=129, hg=4, h=4),
    "varlen_gqa": GDNShape(t=308, hg=4, h=8, seq_lens=(65, 15, 100, 128)),
    "decode_prefix": GDNShape(t=129, hg=4, h=8, seq_lens=(1, 1, 64, 65), num_decodes=2),
}


@functools.cache
def _load_kernels():
    """Import kernels lazily."""
    os.environ.setdefault("AITER_TRITON_ONLY", "1")
    os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        gdn_prepare_flydsl_supported,
        gdn_prepare_fwd_flydsl,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill import (
        fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
        fused_solve_tril_recompute_w_u,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
        build_gated_delta_rule_prefill_metadata,
    )
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

    return {
        "gdn_prepare_flydsl_supported": gdn_prepare_flydsl_supported,
        "gdn_prepare_fwd_flydsl": gdn_prepare_fwd_flydsl,
        "triton_cumsum_kkt": fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
        "triton_solve_wu": fused_solve_tril_recompute_w_u,
        "build_metadata": build_gated_delta_rule_prefill_metadata,
        "pipeline": chunk_gated_delta_rule_opt_vk,
    }


def _make_cu_seqlens(shape: GDNShape):
    if shape.seq_lens is None:
        return None
    offsets = [0]
    for length in shape.seq_lens:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int64)


def _make_metadata(shape: GDNShape, cu_seqlens):
    if cu_seqlens is None:
        return None
    return _load_kernels()["build_metadata"](
        list(shape.seq_lens),
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        num_decodes=shape.num_decodes,
        num_decode_tokens=shape.num_decode_tokens,
    )


def _make_prepare_inputs(b: int, shape: GDNShape, dtype: torch.dtype, seed: int = 2026):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(b, shape.t, shape.hg, shape.k, dtype=dtype, generator=gen) * 0.2
    v = torch.randn(b, shape.t, shape.h, shape.v, dtype=dtype, generator=gen) * 0.2
    beta = torch.rand(b, shape.t, shape.h, dtype=torch.float32, generator=gen).sigmoid()
    # GDN decay increments are negative.
    g = -(
        torch.rand(b, shape.t, shape.h, dtype=torch.float32, generator=gen) * 0.5 + 0.2
    )
    return k, v, g, beta


@torch.no_grad()
def run_torch_gdn_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    seq_lens: tuple[int, ...] | None = None,
    num_decodes: int = 0,
    use_exp2: bool = True,
):
    """Fp32 algebraic specification of cumsum, KKT, solve and WY GEMMs."""
    b, t, hg, k_dim = k.shape
    h, v_dim = v.shape[2], v.shape[3]
    if h % hg:
        raise ValueError(f"H={h} must be divisible by Hg={hg}")
    rep = h // hg

    kf, vf, gf, bf = k.float(), v.float(), g.float(), beta.float()
    w_bar = torch.zeros(b, t, h, k_dim, dtype=torch.float32)
    u_bar = torch.zeros(b, t, h, v_dim, dtype=torch.float32)
    g_cumsum = torch.zeros(b, t, h, dtype=torch.float32)
    eye = torch.eye(BT, dtype=torch.float32)

    if seq_lens is None:
        bounds = [(batch, 0, t) for batch in range(b)]
    else:
        active_lens = seq_lens[num_decodes:]
        bounds = []
        start = 0
        for length in active_lens:
            bounds.append((0, start, length))
            start += length
        if start != t:
            raise ValueError(f"active sequence lengths sum to {start}, expected T={t}")

    for batch, bos, seqlen in bounds:
        for chunk in range((seqlen + BT - 1) // BT):
            t0 = bos + chunk * BT
            length = min(BT, seqlen - chunk * BT)
            token_slice = slice(t0, t0 + length)

            gc = torch.cumsum(gf[batch, token_slice], dim=0)
            g_cumsum[batch, token_slice] = gc

            kh = kf[batch, token_slice].repeat_interleave(rep, dim=1).permute(1, 0, 2)
            kkt = torch.bmm(kh, kh.transpose(1, 2))
            gc_h = gc.permute(1, 0)
            beta_h = bf[batch, token_slice].permute(1, 0)
            decay = torch.exp(gc_h[:, :, None] - gc_h[:, None, :])
            strict_lower = torch.tril(
                torch.ones(length, length, dtype=torch.bool), diagonal=-1
            )
            a_mat = kkt * beta_h[:, :, None] * decay * strict_lower[None]

            ip_a = eye[:length, :length][None] + a_mat
            inverse = torch.linalg.solve(
                ip_a, eye[:length, :length][None].expand(h, length, length)
            )
            # Match the bf16 operands used by the candidates.
            inverse = inverse.to(torch.bfloat16).float()
            k_beta = (
                (kh * beta_h[:, :, None] * torch.exp(gc_h)[:, :, None])
                .to(torch.bfloat16)
                .float()
            )
            v_beta = (
                (vf[batch, token_slice].permute(1, 0, 2) * beta_h[:, :, None])
                .to(torch.bfloat16)
                .float()
            )

            w_bar[batch, token_slice] = torch.bmm(inverse, k_beta).permute(1, 0, 2)
            u_bar[batch, token_slice] = torch.bmm(inverse, v_beta).permute(1, 0, 2)

    if use_exp2:
        g_cumsum *= math.log2(math.e)
    return (
        w_bar.transpose(1, 2).contiguous().to(torch.bfloat16),
        u_bar.transpose(1, 2).contiguous().to(torch.bfloat16),
        g_cumsum.transpose(1, 2).contiguous(),
    )


def _run_triton_prepare(
    k,
    v,
    g,
    beta,
    *,
    cu_seqlens,
    use_exp2,
    shape,
    prefill_metadata,
):
    kernels = _load_kernels()
    g_cumsum, a_raw = kernels["triton_cumsum_kkt"](
        k,
        beta,
        g,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        use_exp2=use_exp2,
        num_decodes=shape.num_decodes,
        num_decode_tokens=shape.num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )
    w_bar, u_bar = kernels["triton_solve_wu"](
        a_raw,
        k,
        v,
        beta,
        g_cumsum,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=shape.num_decodes,
        num_decode_tokens=shape.num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )
    return w_bar, u_bar, g_cumsum


def _run_flydsl_prepare(
    k,
    v,
    g,
    beta,
    *,
    cu_seqlens,
    use_exp2,
    shape,
    prefill_metadata,
):
    return _load_kernels()["gdn_prepare_fwd_flydsl"](
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        BT=BT,
        Hg=shape.hg,
        use_exp2=use_exp2,
        num_decodes=shape.num_decodes,
        num_decode_tokens=shape.num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )


def _chunk_lengths(shape: GDNShape, b: int):
    if shape.seq_lens is None:
        return [shape.t] * b
    return list(shape.active_seq_lens)


def _prepare_work(shape: GDNShape, b: int, element_size: int, use_exp2: bool):
    """Return semantic FLOPs and public-tensor bytes.

    FMA counts as two FLOPs; private intermediates are excluded.
    """
    flops = 0.0
    for seq_len in _chunk_lengths(shape, b):
        for start in range(0, seq_len, BT):
            length = min(BT, seq_len - start)
            kkt = 2 * length * length * shape.k
            # Strict-lower inverse recurrence: (L^3-L)/3.
            triangular_inverse = (length**3 - length) / 3
            wy_gemms = 2 * length * length * (shape.k + shape.v)
            cumsum = length - 1
            gated_kkt = 4 * length * length
            k_beta = 2 * length * shape.k + length
            v_beta = length * shape.v
            output_rescale = length if use_exp2 else 0
            flops += shape.h * (
                kkt
                + triangular_inverse
                + wy_gemms
                + cumsum
                + gated_kkt
                + k_beta
                + v_beta
                + output_rescale
            )
    tokens = b * shape.t
    nbytes = element_size * (
        tokens * shape.hg * shape.k
        + tokens * shape.h * shape.v
        + tokens * shape.h * shape.k
        + tokens * shape.h * shape.v
    )
    nbytes += tokens * shape.h * (4 + 4 + 4)
    return flops, nbytes


def _quiet_check_allclose(*args, **kwargs):
    """Suppress successful checks, but rerun mismatches with diagnostics."""
    err = checkAllclose(*args, **kwargs, printLog=False)
    if err:
        checkAllclose(*args, **kwargs, printLog=True)
    return err


def _check_prepare_outputs(name, ref, out, b, shape):
    w_bar, u_bar, g_cumsum = out
    if w_bar.shape != (b, shape.h, shape.t, shape.k):
        raise AssertionError(f"{name}: unexpected w shape {tuple(w_bar.shape)}")
    if u_bar.shape != (b, shape.h, shape.t, shape.v):
        raise AssertionError(f"{name}: unexpected u shape {tuple(u_bar.shape)}")
    if g_cumsum.shape != (b, shape.h, shape.t):
        raise AssertionError(f"{name}: unexpected g shape {tuple(g_cumsum.shape)}")
    if w_bar.dtype is not dtypes.bf16 or u_bar.dtype is not dtypes.bf16:
        raise AssertionError(
            f"{name}: w/u must be bf16, got {w_bar.dtype}/{u_bar.dtype}"
        )
    if g_cumsum.dtype is not dtypes.fp32:
        raise AssertionError(f"{name}: g_cumsum must be fp32, got {g_cumsum.dtype}")
    if not all(x.is_contiguous() for x in out):
        raise AssertionError(f"{name}: prepare outputs must be contiguous")

    errors = [
        _quiet_check_allclose(
            ref[0].to(dtypes.fp32),
            w_bar.to(dtypes.fp32),
            rtol=0,
            atol=ATOL_WU,
            tol_err_ratio=0,
            max_abs_delta=ATOL_WU,
            msg=f"{name}: w_bar ",
        ),
        _quiet_check_allclose(
            ref[1].to(dtypes.fp32),
            u_bar.to(dtypes.fp32),
            rtol=0,
            atol=ATOL_WU,
            tol_err_ratio=0,
            max_abs_delta=ATOL_WU,
            msg=f"{name}: u_bar ",
        ),
        _quiet_check_allclose(
            ref[2].to(dtypes.fp32),
            g_cumsum.to(dtypes.fp32),
            rtol=0,
            atol=ATOL_G,
            tol_err_ratio=0,
            max_abs_delta=ATOL_G,
            msg=f"{name}: g_cumsum ",
        ),
    ]
    return max(errors)


@benchmark()
def test_gdn_prepare(
    shape_name,
    b,
    t,
    hg,
    h,
    k_dim,
    v_dim,
    mode,
    use_exp2,
    dtype,
):
    """Run one direct prepare shape."""
    shape = PREPARE_SHAPES[shape_name]
    if (t, hg, h, k_dim, v_dim, mode) != (
        shape.t,
        shape.hg,
        shape.h,
        shape.k,
        shape.v,
        shape.mode,
    ):
        raise ValueError(f"shape columns do not match {shape_name}")
    if dtype is not dtypes.bf16:
        raise ValueError("the fused FlyDSL prepare candidate only supports bf16")

    k, v, g, beta = _make_prepare_inputs(b, shape, dtype)
    cu_seqlens = _make_cu_seqlens(shape)
    prefill_metadata = _make_metadata(shape, cu_seqlens)
    ref = run_torch_gdn_prepare(
        k,
        v,
        g,
        beta,
        seq_lens=shape.seq_lens,
        num_decodes=shape.num_decodes,
        use_exp2=use_exp2,
    )

    common = {
        "cu_seqlens": cu_seqlens,
        "use_exp2": use_exp2,
        "shape": shape,
        "prefill_metadata": prefill_metadata,
    }
    candidates = {
        "prepare_flydsl": lambda: _run_flydsl_prepare(k, v, g, beta, **common),
        "prepare_triton": lambda: _run_triton_prepare(k, v, g, beta, **common),
    }
    flops, nbytes = _prepare_work(shape, b, k.element_size(), use_exp2)
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        out, us = run_perftest(candidate, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP)
        err = _check_prepare_outputs(name, ref, out, b, shape)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _make_pipeline_inputs(
    b: int, shape: GDNShape, dtype: torch.dtype, seed: int = 3026
):
    k, v, g, beta = _make_prepare_inputs(b, shape, dtype, seed=seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    q = torch.randn(b, shape.t, shape.hg, shape.k, dtype=dtype, generator=gen) * 0.2
    n_state = b if shape.seq_lens is None else len(shape.active_seq_lens)
    initial_state = (
        torch.randn(
            n_state,
            shape.h,
            shape.v,
            shape.k,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.1
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
    }


@torch.no_grad()
def _run_torch_recurrent(q, k, v, g, beta, initial_state, scale):
    h = v.shape[2]
    hg = k.shape[2]
    if h % hg:
        raise ValueError(f"H={h} must be divisible by Hg={hg}")
    repeat = h // hg
    qf = q.float().repeat_interleave(repeat, dim=2).transpose(1, 2)
    kf = k.float().repeat_interleave(repeat, dim=2).transpose(1, 2)
    vf = v.float().transpose(1, 2)
    gf = g.float().transpose(1, 2)
    bf = beta.float().transpose(1, 2)
    state = initial_state.float().transpose(-1, -2).contiguous()
    out = torch.zeros_like(vf, dtype=torch.float32)

    for token in range(q.shape[1]):
        state = state * gf[:, :, token].exp()[..., None, None]
        value = vf[:, :, token] - (state * kf[:, :, token, :, None]).sum(-2)
        value = value * bf[:, :, token, None]
        state = state + kf[:, :, token, :, None] * value[:, :, None, :]
        out[:, :, token] = torch.einsum("bhd,bhdv->bhv", qf[:, :, token] * scale, state)
    return out.transpose(1, 2).contiguous(), state


def run_torch_gdn_pipeline(inputs, shape: GDNShape):
    """Fp32 recurrent reference for dense and packed model-facing calls."""
    scale = shape.k**-0.5
    if shape.seq_lens is None:
        return _run_torch_recurrent(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["initial_state"],
            scale,
        )

    outputs = []
    states = []
    start = 0
    for state_idx, length in enumerate(shape.active_seq_lens):
        token_slice = slice(start, start + length)
        out, state = _run_torch_recurrent(
            inputs["q"][:, token_slice],
            inputs["k"][:, token_slice],
            inputs["v"][:, token_slice],
            inputs["g"][:, token_slice],
            inputs["beta"][:, token_slice],
            inputs["initial_state"][state_idx : state_idx + 1],
            scale,
        )
        outputs.append(out)
        states.append(state)
        start += length
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def _hidden_backend_kwargs(hidden_backend):
    if hidden_backend == "triton":
        return {}
    if hidden_backend == "flydsl":
        return {"use_chunk_flydsl": True}
    if hidden_backend == "hip":
        return {"use_chunk_hip": True}
    raise ValueError(f"unknown hidden backend {hidden_backend!r}")


def _run_pipeline(
    inputs,
    *,
    output,
    shape,
    cu_seqlens,
    prefill_metadata,
    hidden_backend,
    use_prepare_flydsl,
    seq_lens_cpu=None,
):
    return _load_kernels()["pipeline"](
        **inputs,
        o=output,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_prepare_flydsl=use_prepare_flydsl,
        use_exp2=True,
        num_decodes=shape.num_decodes,
        num_decode_tokens=shape.num_decode_tokens,
        seq_lens_cpu=seq_lens_cpu,
        prefill_metadata=prefill_metadata,
        **_hidden_backend_kwargs(hidden_backend),
    )


def _pipeline_work(shape, b, element_size):
    """Return recurrent-GDN FLOPs and public bytes.

    The reference performs ``7*K*V + K + 1`` FLOPs per token and head.
    """
    tokens = b * shape.t
    flops = tokens * shape.h * (7 * shape.k * shape.v + shape.k + 1)
    n_state = b if shape.seq_lens is None else len(shape.active_seq_lens)
    nbytes = element_size * (
        2 * tokens * shape.hg * shape.k + 2 * tokens * shape.h * shape.v
    )
    nbytes += 2 * tokens * shape.h * 4
    nbytes += 2 * n_state * shape.h * shape.k * shape.v * 4
    return flops, nbytes


def _check_pipeline_output_contract(name, out, final_state, output, inputs, shape):
    if out.data_ptr() != output.data_ptr():
        raise AssertionError(
            f"{name}: pipeline did not return the provided output buffer"
        )
    if out.shape != inputs["v"].shape or out.dtype is not inputs["v"].dtype:
        raise AssertionError(
            f"{name}: unexpected output contract {tuple(out.shape)} {out.dtype}"
        )
    if not out.is_contiguous():
        raise AssertionError(f"{name}: pipeline output must be contiguous")

    n_state = (
        inputs["q"].shape[0] if shape.seq_lens is None else len(shape.active_seq_lens)
    )
    expected_state_shape = (n_state, shape.h, shape.v, shape.k)
    if (
        final_state.shape != expected_state_shape
        or final_state.dtype is not dtypes.fp32
        or not final_state.is_contiguous()
    ):
        raise AssertionError(
            f"{name}: unexpected final-state contract "
            f"{tuple(final_state.shape)} {final_state.dtype}"
        )


@benchmark()
def test_gdn_prepare_pipeline(
    shape_name,
    b,
    t,
    hg,
    h,
    k_dim,
    v_dim,
    mode,
    hidden_backend,
    dtype,
):
    """Run one pipeline shape with both prepare candidates."""
    shape = PIPELINE_SHAPES[shape_name]
    if (t, hg, h, k_dim, v_dim, mode) != (
        shape.t,
        shape.hg,
        shape.h,
        shape.k,
        shape.v,
        shape.mode,
    ):
        raise ValueError(f"shape columns do not match {shape_name}")
    if dtype is not dtypes.bf16:
        raise ValueError("the fused FlyDSL prepare candidate only supports bf16")

    inputs = _make_pipeline_inputs(b, shape, dtype)
    cu_seqlens = _make_cu_seqlens(shape)
    prefill_metadata = _make_metadata(shape, cu_seqlens)
    ref_out, ref_state = run_torch_gdn_pipeline(inputs, shape)

    outputs = {
        "prepare_flydsl": torch.empty_like(inputs["v"]),
        "prepare_triton": torch.empty_like(inputs["v"]),
    }
    candidates = {
        "prepare_flydsl": lambda: _run_pipeline(
            inputs,
            output=outputs["prepare_flydsl"],
            shape=shape,
            cu_seqlens=cu_seqlens,
            prefill_metadata=prefill_metadata,
            hidden_backend=hidden_backend,
            use_prepare_flydsl=True,
        ),
        "prepare_triton": lambda: _run_pipeline(
            inputs,
            output=outputs["prepare_triton"],
            shape=shape,
            cu_seqlens=cu_seqlens,
            prefill_metadata=prefill_metadata,
            hidden_backend=hidden_backend,
            use_prepare_flydsl=False,
        ),
    }
    flops, nbytes = _pipeline_work(shape, b, inputs["q"].element_size())
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        (out, final_state), us = run_perftest(
            candidate, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP
        )
        _check_pipeline_output_contract(
            name, out, final_state, outputs[name], inputs, shape
        )
        err_out = _quiet_check_allclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=5e-2,
            atol=5e-2,
            tol_err_ratio=0.05,
            max_abs_delta=0.3,
            msg=f"{name}/{hidden_backend}: pipeline output ",
        )
        err_state = _quiet_check_allclose(
            ref_state.transpose(-1, -2).to(dtypes.fp32),
            final_state.to(dtypes.fp32),
            rtol=5e-2,
            atol=5e-2,
            tol_err_ratio=0.05,
            max_abs_delta=0.3,
            msg=f"{name}/{hidden_backend}: final state ",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err_out, err_state)
    return ret


def _expect_error(exc_type, fn, message):
    try:
        fn()
    except exc_type as exc:
        if message not in str(exc):
            raise AssertionError(
                f"expected {message!r} in {exc_type.__name__}: {exc}"
            ) from exc
    else:
        raise AssertionError(f"expected {exc_type.__name__}: {message}")


def _run_contract_checks():
    """Run wrapper and fallback regression checks."""
    kernels = _load_kernels()
    dense = PIPELINE_SHAPES["dense_tail"]
    k, v, g, beta = _make_prepare_inputs(1, dense, dtypes.bf16, seed=4026)
    supported = kernels["gdn_prepare_flydsl_supported"]
    flydsl = kernels["gdn_prepare_fwd_flydsl"]
    if not supported(k, v):
        raise AssertionError("supported bf16 K=V=128 CDNA case was rejected")

    # Repeated, nearly unit-normalized keys and slow decay occur for padded
    # Qwen3.5 prompts.  They are a numerical stress case for the triangular
    # inverse: bf16 intermediates can grow by orders of magnitude even though
    # the fp32 reference remains finite and well bounded.
    stress = GDNShape(t=64, hg=1, h=2)
    stress_k = torch.zeros(1, stress.t, stress.hg, stress.k, dtype=dtypes.bf16)
    stress_k[..., 0] = 1
    stress_v = (
        torch.linspace(
            -1,
            1,
            1 * stress.t * stress.h * stress.v,
            dtype=dtypes.fp32,
        )
        .reshape(1, stress.t, stress.h, stress.v)
        .to(dtypes.bf16)
    )
    stress_g = torch.full((1, stress.t, stress.h), -1e-3, dtype=dtypes.fp32)
    stress_beta = torch.full((1, stress.t, stress.h), 0.835, dtype=dtypes.fp32)
    stress_common = {
        "cu_seqlens": None,
        "use_exp2": True,
        "shape": stress,
        "prefill_metadata": None,
    }
    stress_ref = _run_triton_prepare(
        stress_k, stress_v, stress_g, stress_beta, **stress_common
    )
    stress_out = _run_flydsl_prepare(
        stress_k, stress_v, stress_g, stress_beta, **stress_common
    )
    _check_prepare_outputs(
        "prepare_flydsl_repeated_key", stress_ref, stress_out, 1, stress
    )

    bad_v = v.cpu()
    if supported(k, bad_v):
        raise AssertionError("cross-device operands must be rejected")
    _expect_error(
        ValueError,
        lambda: flydsl(k, bad_v, g, beta),
        "co-resident on one CDNA device",
    )
    _expect_error(
        TypeError,
        lambda: flydsl(k.half(), v.half(), g, beta),
        "requires bf16",
    )
    dim64 = GDNShape(t=64, hg=4, h=4, k=64, v=64)
    k64, v64, g64, beta64 = _make_prepare_inputs(1, dim64, dtypes.bf16, seed=4027)
    if supported(k64, v64):
        raise AssertionError("K=V=64 must be outside the fused prepare slice")
    _expect_error(
        ValueError,
        lambda: flydsl(k64, v64, g64, beta64),
        "K=V=128",
    )

    varlen = PIPELINE_SHAPES["varlen_gqa"]
    vk, vv, vg, vb = _make_prepare_inputs(1, varlen, dtypes.bf16, seed=4028)
    cu = _make_cu_seqlens(varlen)
    _expect_error(
        ValueError,
        lambda: flydsl(vk, vv, vg, vb, cu_seqlens=cu, Hg=varlen.hg),
        "prefill_metadata",
    )

    # Missing schedules warn and use the Triton fallback.
    inputs = _make_pipeline_inputs(1, varlen, dtypes.bf16, seed=4029)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_flag, state_flag = _run_pipeline(
            inputs,
            output=torch.empty_like(inputs["v"]),
            shape=varlen,
            cu_seqlens=cu,
            prefill_metadata=None,
            hidden_backend="triton",
            use_prepare_flydsl=True,
        )
    if not any("prefill schedule" in str(item.message) for item in caught):
        raise AssertionError("missing schedule did not emit the fallback warning")
    out_ref, state_ref = _run_pipeline(
        inputs,
        output=torch.empty_like(inputs["v"]),
        shape=varlen,
        cu_seqlens=cu,
        prefill_metadata=None,
        hidden_backend="triton",
        use_prepare_flydsl=False,
    )
    if not torch.equal(out_flag, out_ref) or not torch.equal(state_flag, state_ref):
        raise AssertionError("missing-schedule fallback diverged from Triton")

    # Host sequence lengths can build the schedule.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="use_prepare_flydsl needs a prefill schedule",
            category=UserWarning,
        )
        _run_pipeline(
            inputs,
            output=torch.empty_like(inputs["v"]),
            shape=varlen,
            cu_seqlens=cu,
            prefill_metadata=None,
            hidden_backend="triton",
            use_prepare_flydsl=True,
            seq_lens_cpu=list(varlen.seq_lens),
        )

    # Unsupported fp16 inputs use the Triton fallback.
    fp16_inputs = _make_pipeline_inputs(1, dense, dtypes.fp16, seed=4030)
    fp16_flag = _run_pipeline(
        fp16_inputs,
        output=torch.empty_like(fp16_inputs["v"]),
        shape=dense,
        cu_seqlens=None,
        prefill_metadata=None,
        hidden_backend="triton",
        use_prepare_flydsl=True,
    )
    fp16_ref = _run_pipeline(
        fp16_inputs,
        output=torch.empty_like(fp16_inputs["v"]),
        shape=dense,
        cu_seqlens=None,
        prefill_metadata=None,
        hidden_backend="triton",
        use_prepare_flydsl=False,
    )
    if not torch.equal(fp16_flag[0], fp16_ref[0]) or not torch.equal(
        fp16_flag[1], fp16_ref[1]
    ):
        raise AssertionError("fp16 prepare fallback diverged from Triton")

    # Warm scheduled calls must not synchronize with the host.
    metadata = _make_metadata(varlen, cu)
    output = torch.empty_like(inputs["v"])
    for _ in range(2):
        _run_pipeline(
            inputs,
            output=output,
            shape=varlen,
            cu_seqlens=cu,
            prefill_metadata=metadata,
            hidden_backend="triton",
            use_prepare_flydsl=True,
        )
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _run_pipeline(
            inputs,
            output=output,
            shape=varlen,
            cu_seqlens=cu,
            prefill_metadata=metadata,
            hidden_backend="triton",
            use_prepare_flydsl=True,
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)


def _str2bool(value):
    normalized = value.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _summarize(name, rows):
    if not rows:
        raise ValueError(f"{name}: no valid sweep cases after filtering")
    aiter.logger.info(
        "%s summary (markdown):\n%s",
        name,
        pd.DataFrame(rows).to_markdown(index=False),
    )


def _configure_noise_filters():
    """Hide known dependency chatter while preserving kernel/test warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r"tl\.make_block_ptr is deprecated\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: Profiler clears events at the end of each cycle\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Synchronization debug mode is a prototype feature.*",
        category=UserWarning,
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL GDN prepare unsupported on %s; skipping", get_gfx()
        )
        return
    if not is_flydsl_available():
        aiter.logger.warning("FlyDSL is unavailable; skipping GDN prepare")
        return

    _configure_noise_filters()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="GDN prepare correctness and performance sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="Input dtypes. The FlyDSL candidate currently supports bf16.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Dense batch sizes; packed varlen cases always use batch 1.",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        nargs="*",
        choices=list(PREPARE_SHAPES),
        default=list(PREPARE_SHAPES),
        help="Direct prepare shape labels.",
    )
    parser.add_argument(
        "--pipeline-shapes",
        type=str,
        nargs="*",
        choices=list(PIPELINE_SHAPES),
        default=list(PIPELINE_SHAPES),
        help="Model-facing pipeline shape labels.",
    )
    parser.add_argument(
        "--hidden-backends",
        type=str,
        nargs="*",
        choices=["triton", "flydsl", "hip"],
        default=["triton", "flydsl", "hip"],
        help="Hidden-state backends swept independently of prepare candidates.",
    )
    parser.add_argument(
        "--use-exp2",
        type=_str2bool,
        nargs="*",
        default=[True, False],
        help="Exponent modes for the direct prepare sweep.",
    )
    args = parser.parse_args()

    supported_dtypes = []
    for dtype in args.dtype:
        if dtype is dtypes.bf16:
            supported_dtypes.append(dtype)
        else:
            aiter.logger.warning(
                "Skipping unsupported GDN prepare dtype %s; FlyDSL requires bf16",
                dtype,
            )
    if not supported_dtypes:
        aiter.logger.warning("No supported GDN prepare dtypes requested; skipping")
        return

    _load_kernels()
    _run_contract_checks()

    direct_rows = []
    for dtype, b, shape_name, use_exp2 in itertools.product(
        supported_dtypes, args.batch, args.shapes, args.use_exp2
    ):
        shape = PREPARE_SHAPES[shape_name]
        if shape.seq_lens is not None and b != 1:
            # Packed varlen tensors use B=1 by API contract.
            continue
        direct_rows.append(
            test_gdn_prepare(
                shape_name,
                b,
                shape.t,
                shape.hg,
                shape.h,
                shape.k,
                shape.v,
                shape.mode,
                use_exp2,
                dtype,
            )
        )
    _summarize("gdn_prepare", direct_rows)

    pipeline_rows = []
    for dtype, b, shape_name, hidden_backend in itertools.product(
        supported_dtypes,
        args.batch,
        args.pipeline_shapes,
        args.hidden_backends,
    ):
        shape = PIPELINE_SHAPES[shape_name]
        if shape.seq_lens is not None and b != 1:
            continue
        pipeline_rows.append(
            test_gdn_prepare_pipeline(
                shape_name,
                b,
                shape.t,
                shape.hg,
                shape.h,
                shape.k,
                shape.v,
                shape.mode,
                hidden_backend,
                dtype,
            )
        )
    _summarize("gdn_prepare_pipeline", pipeline_rows)


if __name__ == "__main__":
    main()
