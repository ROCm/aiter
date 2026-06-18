# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the migrated prefill causal-conv1d split-qkv backends.

Compares the HIP and FlyDSL kernels against an
explicit PyTorch reference (and the in-tree Triton split-qkv kernel) on varlen /
continuous-batching shapes, including the conv_state (initial-state) path.
"""

import pytest
import torch

CONV_DIM = 8192
K_DIM = 2048
V_DIM = 4096
WIDTH = 4
STATE_LEN = WIDTH - 1


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
    cu_seqlens, with_initial_state=False, device="cuda", dtype=torch.bfloat16, seed=0
):
    torch.manual_seed(seed)
    T = cu_seqlens[-1]
    n = len(cu_seqlens) - 1
    x = torch.randn(CONV_DIM, T, dtype=dtype, device=device) * 0.1
    weight = torch.randn(CONV_DIM, WIDTH, dtype=dtype, device=device) * 0.1
    bias = torch.randn(CONV_DIM, dtype=dtype, device=device) * 0.1
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
):
    """Pattern-3 dispatch: each backend lives in its own namespace and the
    caller (here, the test) picks one explicitly. There is no in-library
    auto-dispatcher anymore."""
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

    if backend == "flydsl":
        from aiter.ops.flydsl.causal_conv1d_flydsl import is_flydsl_available

        if not is_flydsl_available():
            pytest.skip("flydsl not available")

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


def qsl_to(qsl):
    if torch.is_tensor(qsl):
        return qsl
    return torch.tensor(qsl, dtype=torch.int32, device="cuda")


if __name__ == "__main__":
    torch.manual_seed(0)
    from aiter.ops.flydsl.causal_conv1d_flydsl import is_flydsl_available

    backends = ["hip", "triton2d", "triton"] + (
        ["flydsl"] if is_flydsl_available() else []
    )
    for backend in backends:
        print(f"\n=== backend={backend} ===")
        for cu in SHAPES:
            for with_is in (False, True):
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
                    query_start_loc=qsl,
                    cache_indices=ci,
                    has_initial_state=hi,
                    k_dim=K_DIM,
                    v_dim=V_DIM,
                    seq_lens_cpu=qsl.diff().tolist(),
                    activation="silu",
                )
                rq = _max_abs_rel(q, ref_q)[1]
                rk = _max_abs_rel(k, ref_k)[1]
                rv = _max_abs_rel(v, ref_v)[1]
                rc = _max_abs_rel(cs_work, ref_cs)[1]
                ok = max(rq, rk, rv, rc) < 5e-2
                print(
                    f"  cu={cu} is={int(with_is)}  "
                    f"q={rq:.4f} k={rk:.4f} v={rv:.4f} cs={rc:.4f}  "
                    f"{'OK' if ok else 'FAIL'}"
                )
    print("\ndone")
