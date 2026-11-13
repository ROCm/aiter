# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and timing of aiter.ops.kda_replayssm.kda_replayssm_decode (HIP ReplaySSM KDA decode step).

The reference is a plain fp32 PyTorch implementation of the ReplaySSM step: replay the committed records onto the
checkpoint, flush when h + 2 * max_query_len > cache_len, run the gated delta rule over the step's tokens and append
their (k, u, g) records.  pytest runs the correctness grid; ``python test_kda_replayssm_decode.py`` prints timings
at Kimi-K3 shapes (12 KDA heads per rank, a 5-token DSpark verify window).
"""

import argparse

import pytest
import torch
from aiter.test_common import run_perftest

from aiter.ops.kda_replayssm import (
    kda_replayssm_decode,
    kda_replayssm_decode_supported,
)

DEV = "cuda"
K = V = 128


def make_inputs(N, T, H, HV, cap, gdtype, seed=0, ragged=False, pad_slot=False):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, s=1.0):
        return (torch.randn(*shape, device=DEV, generator=g) * s).to(dtype)

    lens = [T] * N
    if ragged and N > 2:
        lens[1] = max(1, T - 1)
        lens[2] = 0  # an empty sequence in the batch
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), device=DEV, dtype=torch.int32
    )
    T_tot = int(cu[-1])
    S = 2 * N + 1
    slot = torch.randperm(S, device=DEV, generator=g)[:N].to(torch.int32)
    if pad_slot and N > 1:
        slot[0] = -1  # padded entry: must be skipped
    wp = torch.randint(0, cap - T + 1, (S,), device=DEV, generator=g, dtype=torch.int32)
    wp[slot[-1].clamp(min=0)] = -1  # prefill sentinel: no records yet
    return {
        "q": rnd(1, T_tot, H, K),
        "k": rnd(1, T_tot, H, K),
        "v": rnd(1, T_tot, HV, V),
        "a": rnd(1, T_tot, HV, K),
        "b": rnd(1, T_tot, HV),
        "A_log": (torch.rand(HV, device=DEV, generator=g) * 2 - 1).float(),
        "dt_bias": (torch.randn(HV * K, device=DEV, generator=g) * 0.5).float(),
        "ckpt": torch.randn(S, HV, V, K, device=DEV, generator=g) * 0.1,
        "buf_k": rnd(S, HV, cap, K, s=0.1),
        "buf_u": rnd(S, HV, cap, V, s=0.1),
        "buf_g": (-torch.rand(S, HV, cap, K, device=DEV, generator=g) * 0.2).to(gdtype),
        "write_pos": wp,
        "slot_idx": slot,
        "cu_seqlens": cu,
        "max_query_len": T,
    }


def reference(d, H, HV, use_l2norm, lower_bound):
    """fp32 ReplaySSM KDA step; returns (o, ckpt, buf_k, buf_u, buf_g) after the call."""
    o = torch.zeros(d["v"].shape[1], HV, V, device=DEV)
    ckpt, bk, bu, bg = (d[x].clone() for x in ("ckpt", "buf_k", "buf_u", "buf_g"))
    cap, T_max, rep = bk.shape[2], d["max_query_len"], HV // H
    A, dtb = torch.exp(d["A_log"])[:, None], d["dt_bias"].view(HV, K)
    cu = d["cu_seqlens"].tolist()
    for n in range(len(cu) - 1):
        s, bos, T = int(d["slot_idx"][n]), cu[n], cu[n + 1] - cu[n]
        if T == 0 or s < 0:
            continue
        h = max(0, int(d["write_pos"][s]))
        flush = h + 2 * T_max > cap
        base = 0 if flush else h
        S = ckpt[s].float().clone()  # [HV, V, K]
        for j in range(h):
            S = S * torch.exp(bg[s, :, j].float())[:, None, :] + (
                bu[s, :, j].float()[:, :, None] * bk[s, :, j].float()[:, None, :]
            )
        if flush:
            ckpt[s] = S
        for t in range(T):
            i = bos + t
            qt = d["q"][0, i].float().repeat_interleave(rep, 0)
            kt = d["k"][0, i].float().repeat_interleave(rep, 0)
            if use_l2norm:
                qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * K**-0.5
            x = d["a"][0, i].float() + dtb
            if lower_bound is not None:
                gt = lower_bound * torch.sigmoid(A * x)
            else:
                gt = -A * torch.nn.functional.softplus(x)
            beta = torch.sigmoid(d["b"][0, i].float())
            S = S * torch.exp(gt)[:, None, :]
            ut = (d["v"][0, i].float() - (S * kt[:, None, :]).sum(-1)) * beta[:, None]
            S = S + ut[:, :, None] * kt[:, None, :]
            o[i] = (S * qt[:, None, :]).sum(-1)
            bk[s, :, base + t] = kt.to(bk.dtype)
            bu[s, :, base + t] = ut.to(bu.dtype)
            bg[s, :, base + t] = gt.to(bg.dtype)
    return o, ckpt, bk, bu, bg


def run(d, use_l2norm, lower_bound):
    # clone only the state the kernel updates in place: inputs keep their layout (clone() would make views dense)
    state = ("ckpt", "buf_k", "buf_u", "buf_g")
    args = {x: (t.clone() if x in state else t) for x, t in d.items()}
    o = kda_replayssm_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["A_log"],
        args["dt_bias"],
        args["ckpt"],
        args["buf_k"],
        args["buf_u"],
        args["buf_g"],
        args["write_pos"],
        args["slot_idx"],
        args["cu_seqlens"],
        args["max_query_len"],
        use_qk_l2norm_in_kernel=use_l2norm,
        lower_bound=lower_bound,
    )
    return o[0], args


def close(name, got, ref, atol, rtol):
    err = (got.float() - ref.float()).abs()
    bad = err > atol + rtol * ref.float().abs()
    assert (
        int(bad.sum()) == 0
    ), f"{name}: {int(bad.sum())} of {got.numel()} outside tol, max err {err.max():.3g}"


@pytest.mark.parametrize("N", [1, 3, 8, 32])
@pytest.mark.parametrize("T", [1, 2, 5, 8])
@pytest.mark.parametrize("H,HV", [(12, 12), (4, 8)])
@pytest.mark.parametrize("gdtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate", ["lower_bound", "softplus"])
def test_kda_replayssm_decode(N, T, H, HV, gdtype, gate):
    cap = 16
    lower_bound = -5.0 if gate == "lower_bound" else None
    d = make_inputs(
        N, T, H, HV, cap, gdtype, seed=N * 131 + T, ragged=N >= 8, pad_slot=N == 8
    )
    assert kda_replayssm_decode_supported(
        d["q"],
        d["v"],
        d["ckpt"],
        d["buf_k"],
        d["buf_u"],
        d["buf_g"],
        d["A_log"],
        d["dt_bias"],
        T,
    )
    o_ref, ck_ref, bk_ref, bu_ref, bg_ref = reference(d, H, HV, True, lower_bound)
    o, got = run(d, True, lower_bound)
    live = torch.zeros(o.shape[0], dtype=torch.bool, device=DEV)
    cu = d["cu_seqlens"].tolist()
    for n in range(len(cu) - 1):
        if int(d["slot_idx"][n]) >= 0:
            live[cu[n] : cu[n + 1]] = True
    scale = float(o_ref.abs().max())
    close("o", o[live], o_ref[live], 2e-2 * scale, 3e-2)
    close("ckpt", got["ckpt"], ck_ref, 1e-4, 1e-4)
    close("buf_k", got["buf_k"], bk_ref, 1e-2, 1e-2)
    close("buf_u", got["buf_u"], bu_ref, 1e-2 * float(bu_ref.float().abs().max()), 2e-2)
    close(
        "buf_g", got["buf_g"], bg_ref, 1e-5 if gdtype == torch.float32 else 1e-2, 1e-2
    )


@pytest.mark.parametrize("head_step", [1, 2])
def test_kda_replayssm_decode_strided_gate_beta(head_step):
    """gate and beta as column views of a wider fused buffer (Kimi-K3 passes beta that way): read by token
    stride, results bitwise equal to contiguous inputs; head_step 2 is a layout that must be copied.
    """
    HV = 12
    d = make_inputs(8, 5, HV, HV, 16, torch.float32, seed=11)
    T_tot = d["q"].shape[1]
    width = 4 * HV * K + 2 * HV * head_step + 64
    fused = torch.randn(T_tot, width, device=DEV, dtype=torch.bfloat16)
    gcol = 3 * HV * K
    gate = fused[:, gcol : gcol + HV * K].view(1, T_tot, HV, K)
    bcol = 4 * HV * K
    beta = fused[:, bcol : bcol + head_step * HV : head_step].unsqueeze(0)
    assert not gate.is_contiguous() and not beta.is_contiguous()
    d_view = dict(d, a=gate, b=beta)
    d_cont = dict(d, a=gate.contiguous(), b=beta.contiguous())
    o_v, got_v = run(d_view, True, -5.0)
    o_c, got_c = run(d_cont, True, -5.0)
    assert torch.equal(o_v, o_c)
    for x in ("ckpt", "buf_k", "buf_u", "buf_g"):
        assert torch.equal(got_v[x], got_c[x]), x


@pytest.mark.parametrize("l2norm", [True, False])
def test_kda_replayssm_decode_no_l2norm(l2norm):
    d = make_inputs(8, 5, 12, 12, 16, torch.float32, seed=7)
    o_ref, ck_ref, *_ = reference(d, 12, 12, l2norm, -5.0)
    o, got = run(d, l2norm, -5.0)
    close("o", o, o_ref, 2e-2 * float(o_ref.abs().max()), 3e-2)
    close("ckpt", got["ckpt"], ck_ref, 1e-4, 1e-4)


def bench(N_list=(8, 32, 110), T=5, HV=12, cap=16):
    rows = []
    for N in N_list:
        d = make_inputs(N, T, HV, HV, cap, torch.float32, seed=N)
        _, us = run_perftest(
            kda_replayssm_decode,
            d["q"],
            d["k"],
            d["v"],
            d["a"],
            d["b"],
            d["A_log"],
            d["dt_bias"],
            d["ckpt"],
            d["buf_k"],
            d["buf_u"],
            d["buf_g"],
            d["write_pos"],
            d["slot_idx"],
            d["cu_seqlens"],
            T,
            use_qk_l2norm_in_kernel=True,
            lower_bound=-5.0,
        )
        nbytes = (
            N * HV * V * K * 4
            + N * HV * cap * (2 * K + V + 4 * K)
            + N * T * HV * (2 * K + V + K) * 2
        )
        rows.append((N, T, HV, us, nbytes / us / 1e3))
    print(f"{'N':>4} {'T':>2} {'HV':>3} {'us':>9} {'GB/s':>8}")
    for n_req, n_tok, n_heads, us, gbs in rows:
        print(f"{n_req:>4} {n_tok:>2} {n_heads:>3} {us:9.2f} {gbs:8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-N", type=int, nargs="*", default=[8, 32, 110], help="requests per call"
    )
    ap.add_argument(
        "-T", type=int, default=5, help="verify window (tokens per request)"
    )
    ap.add_argument("--heads", type=int, default=12, help="value heads per rank")
    args = ap.parse_args()
    bench(args.N, args.T, args.heads)
