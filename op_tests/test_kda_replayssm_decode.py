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
    kda_replayssm_decode_conv,
    kda_replayssm_decode_conv_supported,
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


def conv_reference(x, conv_state, w, idx, idx_in, n_acc, cu):
    """Width-4 speculative causal conv + SiLU (causal_conv1d_update with num_accepted_tokens): y [T, dim] rounded to
    bf16 as the unfused path's q/k/v tensors are, and the conv state after the call."""
    st = conv_state.clone()
    y = torch.zeros(x.shape, dtype=torch.float32, device=DEV)
    cu = cu.tolist()
    for n in range(len(cu) - 1):
        bos, T = cu[n], cu[n + 1] - cu[n]
        si, so = int(idx_in[n]), int(idx[n])
        if T == 0 or si < 0:
            continue
        off = int(n_acc[n]) - 1
        hist = conv_state[si, :, off : off + 3].float()  # [dim, 3]
        xs = x[bos : bos + T].float().t()  # [dim, T]
        win = torch.cat([hist, xs], dim=1)
        for t in range(T):
            acc = (win[:, t : t + 4] * w).sum(-1)
            y[bos + t] = (acc / (1 + torch.exp(-acc))).to(torch.bfloat16).float()
        st[so, :, : 2 + T] = torch.cat([hist[:, 1:3], xs], dim=1).to(st.dtype)
    return y, st


@pytest.mark.parametrize("N", [1, 3, 8, 32])
@pytest.mark.parametrize("T", [1, 2, 5, 8])
@pytest.mark.parametrize("gdtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_layout", ["dim_major", "token_major"])
def test_kda_replayssm_decode_conv(N, T, gdtype, state_layout):
    """conv + ReplaySSM fused vs the torch conv reference feeding the ReplaySSM reference: output, checkpoint,
    records and the rewritten conv state (dim-major [lines, dim, len] or a token-major buffer seen transposed).
    """
    H = HV = 12
    dim = 3 * H * K
    d = make_inputs(
        N, T, H, HV, 16, gdtype, seed=N * 17 + T, ragged=N >= 8, pad_slot=N == 8
    )
    g = torch.Generator(device=DEV).manual_seed(N + 100 * T)
    T_tot, S = d["q"].shape[1], d["ckpt"].shape[0]
    x = torch.randn(T_tot, dim, device=DEV, generator=g).to(torch.bfloat16)
    w = torch.randn(dim, 4, device=DEV, generator=g) * 0.3
    slen = 2 + T
    if state_layout == "dim_major":
        cs = torch.randn(S, dim, slen, device=DEV, generator=g).to(torch.bfloat16)
    else:
        cs = torch.randn(S, slen, dim, device=DEV, generator=g).to(torch.bfloat16)
        cs = cs.transpose(1, 2)
    idx = d["slot_idx"]
    n_acc = torch.randint(1, T + 1, (N,), device=DEV, generator=g, dtype=torch.int32)
    assert kda_replayssm_decode_conv_supported(
        x,
        cs,
        w,
        H,
        d["ckpt"],
        d["buf_k"],
        d["buf_u"],
        d["buf_g"],
        d["A_log"],
        d["dt_bias"],
        T,
    )
    y, cs_ref = conv_reference(x, cs, w, idx, idx, n_acc, d["cu_seqlens"])
    qkv = {
        "q": y[:, : H * K].reshape(1, T_tot, H, K).to(torch.bfloat16),
        "k": y[:, H * K : 2 * H * K].reshape(1, T_tot, H, K).to(torch.bfloat16),
        "v": y[:, 2 * H * K :].reshape(1, T_tot, HV, V).to(torch.bfloat16),
    }
    o_ref, ck_ref, bk_ref, bu_ref, bg_ref = reference(dict(d, **qkv), H, HV, True, -5.0)
    state = {x_: d[x_].clone() for x_ in ("ckpt", "buf_k", "buf_u", "buf_g")}
    cs_got = (
        cs.clone()
        if state_layout == "dim_major"
        else cs.transpose(1, 2).clone().transpose(1, 2)
    )
    o = kda_replayssm_decode_conv(
        x,
        cs_got,
        w,
        idx,
        n_acc,
        d["a"],
        d["b"],
        d["A_log"],
        d["dt_bias"],
        state["ckpt"],
        state["buf_k"],
        state["buf_u"],
        state["buf_g"],
        d["write_pos"],
        d["slot_idx"],
        d["cu_seqlens"],
        T,
        H,
        use_qk_l2norm_in_kernel=True,
        lower_bound=-5.0,
    )
    live = torch.zeros(T_tot, dtype=torch.bool, device=DEV)
    cu = d["cu_seqlens"].tolist()
    for n in range(len(cu) - 1):
        if int(idx[n]) >= 0:
            live[cu[n] : cu[n + 1]] = True
    scale = float(o_ref.abs().max())
    close("o", o[live], o_ref[live], 2e-2 * scale, 3e-2)
    close("ckpt", state["ckpt"], ck_ref, 1e-4, 1e-4)
    close("buf_k", state["buf_k"], bk_ref, 2e-2, 2e-2)
    close(
        "buf_u", state["buf_u"], bu_ref, 2e-2 * float(bu_ref.float().abs().max()), 3e-2
    )
    close(
        "buf_g", state["buf_g"], bg_ref, 1e-5 if gdtype == torch.float32 else 1e-2, 1e-2
    )
    assert torch.equal(cs_got, cs_ref), "conv state"


def test_kda_replayssm_decode_conv_strided_indices():
    """Kimi-K3 hands the conv slots over as column 0 of its [batch, window] slot table: strided index views give
    bitwise the same result as contiguous ones (and a forked read slot, conv_state_indices_in, is honoured).
    """
    H, N, T = 12, 8, 5
    dim = 3 * H * K
    d = make_inputs(N, T, H, H, 16, torch.float32, seed=5)
    g = torch.Generator(device=DEV).manual_seed(3)
    T_tot, S = d["q"].shape[1], d["ckpt"].shape[0]
    x = torch.randn(T_tot, 4 * dim, device=DEV, generator=g).to(torch.bfloat16)[:, :dim]
    w = torch.randn(dim, 4, device=DEV, generator=g) * 0.3
    cs = (
        torch.randn(S, 2 + T, dim, device=DEV, generator=g)
        .to(torch.bfloat16)
        .transpose(1, 2)
    )
    idx = d["slot_idx"]
    idx_in = idx.clone()
    idx_in[3] = int(
        torch.nonzero(torch.isin(torch.arange(S, device=DEV), idx, invert=True))[0]
    )
    n_acc = torch.randint(1, T + 1, (N,), device=DEV, generator=g, dtype=torch.int32)
    table = torch.full((N, T), -7, dtype=torch.int32, device=DEV)
    table_in = table.clone()
    table_acc = table.clone()
    table[:, 0], table_in[:, 0], table_acc[:, 0] = idx, idx_in, n_acc

    def call(i, i_in, acc):
        st = {x_: d[x_].clone() for x_ in ("ckpt", "buf_k", "buf_u", "buf_g")}
        c = cs.transpose(1, 2).clone().transpose(1, 2)
        o = kda_replayssm_decode_conv(
            x,
            c,
            w,
            i,
            acc,
            d["a"],
            d["b"],
            d["A_log"],
            d["dt_bias"],
            st["ckpt"],
            st["buf_k"],
            st["buf_u"],
            st["buf_g"],
            d["write_pos"],
            d["slot_idx"],
            d["cu_seqlens"],
            T,
            H,
            conv_state_indices_in=i_in,
            use_qk_l2norm_in_kernel=True,
            lower_bound=-5.0,
        )
        return o, c, st

    o_s, c_s, st_s = call(table[:, 0], table_in[:, 0], table_acc[:, 0])
    o_c, c_c, st_c = call(idx, idx_in, n_acc)
    assert not table[:, 0].is_contiguous()
    assert torch.equal(o_s, o_c) and torch.equal(c_s, c_c)
    for x_ in st_s:
        assert torch.equal(st_s[x_], st_c[x_]), x_
    _, cs_ref = conv_reference(x, cs, w, idx, idx_in, n_acc, d["cu_seqlens"])
    assert torch.equal(c_c, cs_ref), "conv state with a forked read slot"


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
        # the conv-fused entry point on the same state: x is the mixed q|k|v input, every request resumes at the end
        # of a fully accepted window
        dim = 3 * HV * K
        x = torch.randn(d["q"].shape[1], dim, device=DEV).to(torch.bfloat16)
        w = torch.randn(dim, 4, device=DEV) * 0.3
        cs = torch.randn(d["ckpt"].shape[0], dim, 2 + T, device=DEV).to(torch.bfloat16)
        n_acc = torch.full((N,), T, dtype=torch.int32, device=DEV)
        _, us_conv = run_perftest(
            kda_replayssm_decode_conv,
            x,
            cs,
            w,
            d["slot_idx"],
            n_acc,
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
            HV,
            use_qk_l2norm_in_kernel=True,
            lower_bound=-5.0,
        )
        rows.append((N, T, HV, us, nbytes / us / 1e3, us_conv))
    print(f"{'N':>4} {'T':>2} {'HV':>3} {'us':>9} {'GB/s':>8} {'conv+ us':>9}")
    for n_req, n_tok, n_heads, us, gbs, us_conv in rows:
        print(f"{n_req:>4} {n_tok:>2} {n_heads:>3} {us:9.2f} {gbs:8.1f} {us_conv:9.2f}")


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
