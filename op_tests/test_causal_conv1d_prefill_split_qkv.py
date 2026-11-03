# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the migrated prefill causal-conv1d split-qkv backends.

Compares the HIP and FlyDSL kernels against an
explicit PyTorch reference (and the in-tree Triton split-qkv kernel) on varlen /
continuous-batching shapes, including the conv_state (initial-state) path.
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

CONV_DIM = 8192
K_DIM = 2048
V_DIM = 4096
WIDTH = 4
STATE_LEN = WIDTH - 1
SUPPORTED_GFX = ("gfx942", "gfx950")
_MAX_PERF_ROTATIONS = 32
_PERF_ROTATION_BUDGET = 256 * 1024 * 1024


def _perf_rotation_count(state: torch.Tensor) -> int:
    return max(
        1,
        min(_MAX_PERF_ROTATIONS, _PERF_ROTATION_BUDGET // max(1, state.nbytes)),
    )


def _silu(t):
    return t * torch.sigmoid(t)


def torch_reference(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    k_dim,
    v_dim,
    activation="silu",
    pad_slot_id=-1,
):
    """Unambiguous fp32 reference. Returns (q, k, v, new_conv_states)."""
    dim, T = x.shape
    xf = x.float()
    wf = weight.float()
    bf = bias.float() if bias is not None else torch.zeros(dim, device=x.device)
    csf = conv_states.float()
    out = torch.zeros(dim, T, device=x.device, dtype=torch.float32)
    new_cs = csf.clone()
    qsl = query_start_loc.tolist()
    n = len(qsl) - 1
    for s in range(n):
        start, end = qsl[s], qsl[s + 1]
        L = end - start
        ci = int(cache_indices[s])
        if ci == pad_slot_id:
            continue
        hi = bool(has_initial_state[s])
        for p in range(L):
            acc = bf.clone()
            for k in range(WIDTH):
                j = p - (WIDTH - 1 - k)  # within-seq input position for this tap
                if j >= 0:
                    xv = xf[:, start + j]
                elif hi:
                    xv = csf[ci, :, STATE_LEN + j]  # j in {-1,-2,-3} -> slot {2,1,0}
                else:
                    xv = torch.zeros(dim, device=x.device)
                acc = acc + wf[:, k] * xv
            out[:, start + p] = acc
        # conv_state writeback: last STATE_LEN input tokens
        for slot in range(STATE_LEN):
            pos = L - STATE_LEN + slot
            if pos >= 0:
                new_cs[ci, :, slot] = xf[:, start + pos]
            elif hi:
                new_cs[ci, :, slot] = csf[ci, :, slot + L]
            else:
                new_cs[ci, :, slot] = 0.0
    if activation in ("silu", "swish"):
        out = _silu(out)
    q = out[:k_dim, :].t().contiguous()
    k = out[k_dim : 2 * k_dim, :].t().contiguous()
    v = out[2 * k_dim : 2 * k_dim + v_dim, :].t().contiguous()
    return q, k, v, new_cs


def make_inputs(
    cu_seqlens,
    with_initial_state=False,
    with_bias=True,
    channel_last=False,
    device="cuda",
    dtype=torch.bfloat16,
    seed=0,
):
    torch.manual_seed(seed)
    T = cu_seqlens[-1]
    n = len(cu_seqlens) - 1
    if channel_last:
        x = (torch.randn(T, CONV_DIM, dtype=dtype, device=device) * 0.1).t()
    else:
        x = torch.randn(CONV_DIM, T, dtype=dtype, device=device) * 0.1
    weight = torch.randn(CONV_DIM, WIDTH, dtype=dtype, device=device) * 0.1
    bias = (
        torch.randn(CONV_DIM, dtype=dtype, device=device) * 0.1 if with_bias else None
    )
    conv_states = (
        torch.randn(n, STATE_LEN, CONV_DIM, dtype=dtype, device=device).transpose(1, 2)
        * 0.1
    )
    cache_indices = torch.arange(n, dtype=torch.int32, device=device)
    has_initial_state = torch.full(
        (n,), with_initial_state, dtype=torch.bool, device=device
    )
    query_start_loc = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    return (
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
    )


SHAPES = [
    [0, 1000],  # single long seq
    [0, 8],  # single very short seq (< width)
    [0, 5, 6, 7, 8],  # mixed short
    [0, 1000, 1063, 2063],  # mixed long+short
    [0, 64, 128, 200, 5063],  # boundary-heavy + long tail
]


def _max_abs_rel(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().clamp_min(1e-3)
    return ((a - b).abs() / denom).max().item(), (a - b).abs().max().item()


def _call_backend(
    backend,
    *,
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    k_dim,
    v_dim,
    seq_lens_cpu,
    activation="silu",
    metadata=None,
):
    """Dispatch to the requested backend namespace for comparison tests."""
    if backend == "hip":
        from aiter.ops.causal_conv1d_fwd_split_qkv import (
            causal_conv1d_split_qkv_hip_fn,
        )

        return causal_conv1d_split_qkv_hip_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            k_dim=k_dim,
            v_dim=v_dim,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            metadata=metadata,
        )
    if backend == "flydsl":
        from aiter.ops.flydsl.causal_conv1d_flydsl import (
            causal_conv1d_split_qkv_flydsl_fn,
        )

        return causal_conv1d_split_qkv_flydsl_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            k_dim_size=k_dim,
            v_dim_size=v_dim,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            metadata=metadata,
        )
    if backend == "triton2d":
        from aiter.ops.triton.gated_delta_net.causal_conv1d_prefill import (
            causal_conv1d_split_qkv_triton_tile_fn,
        )

        return causal_conv1d_split_qkv_triton_tile_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            k_dim=k_dim,
            v_dim=v_dim,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            metadata=metadata,
        )
    if backend == "triton":
        from aiter.ops.triton.gated_delta_net.causal_conv1d_prefill import (
            causal_conv1d_split_qkv_triton_fn,
        )

        return causal_conv1d_split_qkv_triton_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            k_dim=k_dim,
            v_dim=v_dim,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
        )
    raise ValueError(f"unknown backend {backend!r}")


@pytest.mark.parametrize("cu", SHAPES)
@pytest.mark.parametrize("with_is", [False, True])
@pytest.mark.parametrize("backend", ["hip", "flydsl", "triton2d", "triton"])
def test_backend_matches_reference(cu, with_is, backend):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    x, w, b, cs, ci, hi, qsl = make_inputs(cu, with_initial_state=with_is)
    ref_q, ref_k, ref_v, ref_cs = torch_reference(
        x, w, b, cs.clone(), qsl, ci, hi, K_DIM, V_DIM
    )

    cs_work = cs.clone()
    q, k, v = _call_backend(
        backend,
        x=x,
        weight=w,
        bias=b,
        conv_states=cs_work,
        query_start_loc=qsl_to(qsl),
        cache_indices=ci,
        has_initial_state=hi,
        k_dim=K_DIM,
        v_dim=V_DIM,
        seq_lens_cpu=qsl_to(qsl).diff().tolist(),
        activation="silu",
    )

    for name, got, ref in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        rel, absd = _max_abs_rel(got, ref)
        assert (
            absd < 5e-2
        ), f"{backend} {name} mismatch: max_abs={absd:.4f} rel={rel:.4f}"

    # conv_state writeback
    rel, absd = _max_abs_rel(cs_work, ref_cs)
    assert absd < 5e-2, f"{backend} conv_state mismatch: max_abs={absd:.4f}"


@pytest.mark.parametrize("backend", ["hip", "triton2d", "flydsl"])
def test_backend_accepts_causal_conv_prefill_metadata(backend):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    from aiter.ops.prefill_batch_metadata import (
        build_causal_conv_prefill_metadata,
    )

    cu = [0, 8, 25]
    seq_lens_cpu = [8, 17]
    x, w, b, cs, ci, hi, qsl = make_inputs(cu)
    metadata = build_causal_conv_prefill_metadata(
        seq_lens_cpu,
        query_start_loc=qsl,
        block_sizes=(16, 64),
    )
    ref_q, ref_k, ref_v, ref_cs = torch_reference(
        x, w, b, cs.clone(), qsl, ci, hi, K_DIM, V_DIM
    )

    cs_work = cs.clone()
    q, k, v = _call_backend(
        backend,
        x=x,
        weight=w,
        bias=b,
        conv_states=cs_work,
        query_start_loc=qsl,
        cache_indices=ci,
        has_initial_state=hi,
        k_dim=K_DIM,
        v_dim=V_DIM,
        seq_lens_cpu=seq_lens_cpu,
        metadata=metadata,
    )

    for name, got, ref in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        rel, absd = _max_abs_rel(got, ref)
        assert (
            absd < 5e-2
        ), f"{backend} {name} mismatch: max_abs={absd:.4f} rel={rel:.4f}"

    rel, absd = _max_abs_rel(cs_work, ref_cs)
    assert (
        absd < 5e-2
    ), f"{backend} conv_state mismatch: max_abs={absd:.4f} rel={rel:.4f}"


@pytest.mark.parametrize("cu", [[0, 1], [0, 2], [0, 1, 3, 6], [0, 512]])
@pytest.mark.parametrize("with_is", [False, True])
def test_hip_channel_last_matches_reference(cu, with_is):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    x, w, b, cs, ci, hi, qsl = make_inputs(
        cu, with_initial_state=with_is, channel_last=True
    )
    assert x.stride() == (1, CONV_DIM)
    ref_q, ref_k, ref_v, ref_cs = torch_reference(
        x, w, b, cs.clone(), qsl, ci, hi, K_DIM, V_DIM
    )
    cs_work = cs.clone()
    q, k, v = _call_backend(
        "hip",
        x=x,
        weight=w,
        bias=b,
        conv_states=cs_work,
        query_start_loc=qsl,
        cache_indices=ci,
        has_initial_state=hi,
        k_dim=K_DIM,
        v_dim=V_DIM,
        seq_lens_cpu=qsl.diff().tolist(),
        activation="silu",
    )

    for name, got, ref in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        rel, absd = _max_abs_rel(got, ref)
        assert absd < 5e-2, f"hip channel-last {name}: max_abs={absd:.4f} rel={rel:.4f}"
    rel, absd = _max_abs_rel(cs_work, ref_cs)
    assert absd < 5e-2, f"hip channel-last state: max_abs={absd:.4f} rel={rel:.4f}"


@pytest.mark.parametrize(
    ("activation", "with_bias"),
    [
        (None, True),
        ("silu", False),
        (None, False),
    ],
)
def test_hip_channel_last_optional_activation_and_bias(activation, with_bias):
    """Cover channel-last dispatch with SiLU and/or bias disabled."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    cu = [0, 7, 31, 90]
    x, w, b, cs, ci, hi, qsl = make_inputs(
        cu,
        with_initial_state=True,
        with_bias=with_bias,
        channel_last=True,
    )
    assert x.stride() == (1, CONV_DIM)
    assert (b is not None) == with_bias

    ref_q, ref_k, ref_v, ref_cs = torch_reference(
        x,
        w,
        b,
        cs.clone(),
        qsl,
        ci,
        hi,
        K_DIM,
        V_DIM,
        activation=activation,
    )
    cs_work = cs.clone()
    q, k, v = _call_backend(
        "hip",
        x=x,
        weight=w,
        bias=b,
        conv_states=cs_work,
        query_start_loc=qsl,
        cache_indices=ci,
        has_initial_state=hi,
        k_dim=K_DIM,
        v_dim=V_DIM,
        seq_lens_cpu=qsl.diff().tolist(),
        activation=activation,
    )

    case = f"activation={activation!r} with_bias={with_bias}"
    for name, got, ref in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        rel, absd = _max_abs_rel(got, ref)
        assert (
            absd < 5e-2
        ), f"hip channel-last {name} {case}: max_abs={absd:.4f} rel={rel:.4f}"
    rel, absd = _max_abs_rel(cs_work, ref_cs)
    assert (
        absd < 5e-2
    ), f"hip channel-last state {case}: max_abs={absd:.4f} rel={rel:.4f}"


@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_is", [False, True])
def test_hip_cache_mapping_and_padding(channel_last, with_is):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    cu = [0, 5, 12, 21]
    x, w, b, cs, _, hi, qsl = make_inputs(
        cu, with_initial_state=with_is, channel_last=channel_last
    )
    # Sequence 0 writes cache line 2, sequence 1 is padding, and sequence 2
    # writes cache line 0. Cache line 1 must remain untouched.
    ci = torch.tensor([2, -1, 0], dtype=torch.int32, device=x.device)
    ref_q, ref_k, ref_v, ref_cs = torch_reference(
        x, w, b, cs.clone(), qsl, ci, hi, K_DIM, V_DIM, pad_slot_id=-1
    )

    cs_work = cs.clone()
    q, k, v = _call_backend(
        "hip",
        x=x,
        weight=w,
        bias=b,
        conv_states=cs_work,
        query_start_loc=qsl,
        cache_indices=ci,
        has_initial_state=hi,
        k_dim=K_DIM,
        v_dim=V_DIM,
        seq_lens_cpu=qsl.diff().tolist(),
        activation="silu",
    )

    # Padded output storage is intentionally unspecified; compare live ranges.
    live = torch.cat(
        (torch.arange(0, 5, device=x.device), torch.arange(12, 21, device=x.device))
    )
    for name, got, ref in (("q", q, ref_q), ("k", k, ref_k), ("v", v, ref_v)):
        rel, absd = _max_abs_rel(got[live], ref[live])
        assert absd < 5e-2, f"hip mapped {name}: max_abs={absd:.4f} rel={rel:.4f}"
    rel, absd = _max_abs_rel(cs_work, ref_cs)
    assert absd < 5e-2, f"hip mapped state: max_abs={absd:.4f} rel={rel:.4f}"


@benchmark()
def test_prefill_split_qkv_benchmark(
    batch: int = 1,
    seqlen: int = 8,
    k_dim: int = K_DIM,
    v_dim: int = V_DIM,
    dtype: torch.dtype = torch.bfloat16,
    layout: str = "channel_first",
    with_initial_state: bool = True,
):
    """Benchmark public prefill backends with model layouts and metadata."""
    conv_dim = 2 * k_dim + v_dim
    if conv_dim != CONV_DIM:
        raise ValueError(
            f"benchmark requires 2*k_dim + v_dim == {CONV_DIM}, got {conv_dim}"
        )
    if layout not in {"channel_first", "channel_last"}:
        raise ValueError(f"unsupported layout {layout!r}")

    cu = [i * seqlen for i in range(batch + 1)]
    x, weight, bias, conv_states, cache_indices, has_initial_state, qsl = make_inputs(
        cu,
        with_initial_state=with_initial_state,
        channel_last=layout == "channel_last",
        dtype=dtype,
        seed=batch * 31 + seqlen,
    )
    seq_lens_cpu = [seqlen] * batch
    from aiter.ops.prefill_batch_metadata import build_causal_conv_prefill_metadata

    metadata = build_causal_conv_prefill_metadata(
        seq_lens_cpu,
        query_start_loc=qsl,
        block_sizes=(8, 16, 32, 64),
    )
    ref_q, ref_k, ref_v, ref_state = torch_reference(
        x,
        weight,
        bias,
        conv_states.clone(),
        qsl,
        cache_indices,
        has_initial_state,
        k_dim,
        v_dim,
    )

    def run_backend(backend, state):
        return _call_backend(
            backend,
            x=x,
            weight=weight,
            bias=bias,
            conv_states=state,
            query_start_loc=qsl,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            k_dim=k_dim,
            v_dim=v_dim,
            seq_lens_cpu=seq_lens_cpu,
            activation="silu",
            metadata=metadata,
        )

    candidates = {
        "hip": lambda state: run_backend("hip", state),
    }
    if layout == "channel_first":
        # Only HIP's public contract is validated for the transposed
        # channel-last view; the other public backends consume channel-first x.
        candidates.update(
            {
                "flydsl": lambda state: run_backend("flydsl", state),
                "triton2d": lambda state: run_backend("triton2d", state),
                "triton": lambda state: run_backend("triton", state),
            }
        )

    tokens = batch * seqlen
    flops = 2 * tokens * conv_dim * WIDTH
    nbytes = (
        x.numel()
        + weight.numel()
        + bias.numel()
        + ref_q.numel()
        + ref_k.numel()
        + ref_v.numel()
        + 2 * conv_states.numel()
    ) * x.element_size()

    ret = {"gfx": get_gfx()}
    references = (ref_q, ref_k, ref_v, ref_state)
    for name, fn in candidates.items():
        checked_state = conv_states.clone()

        def checked_call(state, fn=fn):
            return fn(state)

        checked, _ = run_perftest(
            checked_call,
            checked_state,
            num_iters=1,
            num_warmup=0,
            num_rotate_args=1,
            use_cuda_event=True,
        )
        checked = (*checked, checked_state)
        errs = [
            checkAllclose(
                ref.to(dtypes.fp32),
                got.to(dtypes.fp32),
                rtol=1e-2,
                atol=5e-2,
                msg=f"{name}: {label}",
            )
            for label, got, ref in zip(
                ("q", "k", "v", "in-place conv_state"), checked, references
            )
        ]

        timing_state = conv_states.clone()
        _, us = run_perftest(
            fn,
            timing_state,
            num_rotate_args=_perf_rotation_count(timing_state),
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(errs)
    return ret


# The benchmark is driven by main(); the pytest correctness cases remain
# independently collectible without duplicating the perf sweep.
test_prefill_split_qkv_benchmark.__test__ = False


def qsl_to(qsl):
    if torch.is_tensor(qsl):
        return qsl
    return torch.tensor(qsl, dtype=torch.int32, device="cuda")


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning(
            "causal conv1d prefill split-qkv unsupported on %s; skipping", gfx
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark causal conv1d prefill split-qkv public backends",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[torch.bfloat16],
        choices=[torch.bfloat16],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8, K_DIM, V_DIM), (64, K_DIM, V_DIM)],
        help="seqlen,k_dim,v_dim",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        choices=["channel_first", "channel_last"],
        default=["channel_first", "channel_last"],
    )
    parser.add_argument(
        "--initial-state",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch, shape, layout, with_initial_state in itertools.product(
        args.dtype,
        args.batch,
        args.mnk,
        args.layout,
        args.initial_state,
    ):
        seqlen, k_dim, v_dim = shape
        rows.append(
            test_prefill_split_qkv_benchmark(
                batch,
                seqlen,
                k_dim,
                v_dim,
                dtype,
                layout,
                bool(with_initial_state),
            )
        )
    aiter.logger.info(
        "causal conv1d prefill split-qkv summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    # Preserve the mixed-varlen script coverage that existed before the
    # benchmark entrypoint was added. These direct calls retain the assertions
    # in the pytest correctness test and cover lengths up to 5063 tokens.
    for backend, cu, with_initial_state in itertools.product(
        ("hip", "flydsl", "triton2d", "triton"),
        SHAPES,
        (False, True),
    ):
        test_backend_matches_reference(cu, with_initial_state, backend)


if __name__ == "__main__":
    main()
