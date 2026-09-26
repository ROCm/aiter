# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math

import pytest
import torch

from aiter.ops.triton.attention.chunk_kda import (
    CHUNK_SIZE,
    chunk_kda,
    chunk_kda_prepare,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref, kda_gate_ref, l2norm_ref

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx1250", reason=f"chunk KDA gluon needs gfx1250, got {get_arch()}"
)

DEVICE = "cuda"
D = 128
LOWER_BOUND = -5.0
# bf16 chunk operands and a bf16 state in the walk MMAs, as in the HIP / FLA chunk paths: a
# CPU mirror of this exact algorithm lands at 0.003-0.0045 against the fp32 token loop
RATIO = 0.008
POISON = 1e30

WALK_CONFIGS = [
    None,
    {"BV": 32, "num_warps": 2},
    {"BV": 64, "num_warps": 4},
    {"BV": 128, "num_warps": 4},
]


def err_ratio(ref, tri):
    """fla.utils.get_err_ratio: RMS(ref - tri) / RMS(ref)."""
    ref, tri = ref.detach().double(), tri.detach().double()
    return (
        (ref - tri).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-12)
    ).item()


def assert_close(name, ref, tri, ratio=RATIO):
    assert torch.isfinite(tri).all(), f"{name}: non-finite kernel output"
    r = err_ratio(ref, tri)
    assert r < ratio, f"{name}: err ratio {r:.6f} >= {ratio}"


def make_inputs(seqlens, H, seed=0, dup_keys=False, gate_shift=0.0):
    """K3 prefill inputs: raw projections, q/k/v as bands of one fused projection."""
    torch.manual_seed(seed)
    T = sum(seqlens)
    mixed = torch.randn(1, T, 3 * H * D, dtype=torch.bfloat16, device=DEVICE)
    q, k, v = (
        mixed[..., i * H * D : (i + 1) * H * D].unflatten(-1, (H, D)) for i in range(3)
    )
    if dup_keys:  # near-duplicate neighbouring keys: the case that breaks a bf16 solve
        k.copy_(k[:, :1] + 0.02 * k)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device=DEVICE) + gate_shift
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=DEVICE)
    A_log = torch.log(torch.empty(H, device=DEVICE).uniform_(1, 16))
    dt_bias = torch.randn(H * D, device=DEVICE)
    cu = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=DEVICE
    )
    return dict(
        q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu
    )


def run_ref(inp, initial_state=None):
    """Token-by-token fp32 reference; state is V-first [N, H, V, K]."""
    return chunk_kda_ref(
        **{n: inp[n] for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")},
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=inp["cu_seqlens"].long(),
    )


def run_kernel(inp, **kw):
    return chunk_kda(**inp, lower_bound=LOWER_BOUND, **kw)


def workspace_ref(inp, scale):
    """fp64 chunk operands straight from the chunk equations (log2 units), for the prepare test."""
    q, k = l2norm_ref(inp["q"])[0].double(), l2norm_ref(inp["k"])[0].double()
    v = inp["v"][0].double()
    G2 = kda_gate_ref(inp["g"], inp["A_log"], inp["dt_bias"], LOWER_BOUND)[
        0
    ].double() / math.log(2)
    beta = torch.sigmoid(inp["beta"][0].double())
    T, H, _ = q.shape
    out = dict(
        qg=torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        w=torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        u=torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        aqk=torch.zeros(T, H, CHUNK_SIZE, dtype=torch.float64, device=DEVICE),
        kg_t=[],
        decay=[],
    )
    for bos, eos in itertools.pairwise(inp["cu_seqlens"].tolist()):
        for t0 in range(bos, eos, CHUNK_SIZE):
            t1 = min(t0 + CHUNK_SIZE, eos)
            m = t1 - t0
            G = G2[t0:t1].cumsum(0).transpose(0, 1)  # [H, m, D]
            kc, qc, vc = (x[t0:t1].transpose(0, 1) for x in (k, q, v))
            bc = beta[t0:t1].transpose(0, 1)[..., None]
            kp, kn = kc * torch.exp2(G), kc * torch.exp2(-G)
            L = torch.tril(kp @ kn.transpose(1, 2), -1) * bc
            A_inv = torch.linalg.inv(
                torch.eye(m, dtype=torch.float64, device=DEVICE) + L
            )
            out["qg"][t0:t1] = (qc * torch.exp2(G)).transpose(0, 1)
            out["aqk"][t0:t1, :, :m] = (
                scale
                * torch.tril(out["qg"][t0:t1].transpose(0, 1) @ kn.transpose(1, 2))
            ).transpose(0, 1)
            out["w"][t0:t1] = (A_inv @ (bc * kp)).transpose(0, 1)
            out["u"][t0:t1] = (A_inv @ (bc * vc)).transpose(0, 1)
            kg = torch.zeros(H, D, CHUNK_SIZE, dtype=torch.float64, device=DEVICE)
            kg[..., :m] = (kc * torch.exp2(G[:, -1:] - G)).transpose(1, 2)
            out["kg_t"].append(kg)
            out["decay"].append(torch.exp2(G[:, -1]))
    out["kg_t"] = torch.stack(out["kg_t"])
    out["decay"] = torch.stack(out["decay"])
    return out


@pytest.mark.parametrize(
    "seqlens",
    [[64], [1], [63], [65], [300], [1, 64, 130, 7], [1000]],
)
@pytest.mark.parametrize("config", WALK_CONFIGS)
def test_chunk_kda(seqlens, config):
    H = 4
    inp = make_inputs(seqlens, H)
    h0 = torch.randn(len(seqlens), H, D, D, device=DEVICE)
    o_ref, s_ref = run_ref(inp, h0)
    o, s = run_kernel(inp, initial_state=h0, output_final_state=True, config=config)
    assert_close("o", o_ref, o)
    assert_close("final_state", s_ref, s)


@pytest.mark.parametrize("seqlens", [[64], [65], [1, 64, 130, 7]])
def test_chunk_kda_workspace(seqlens):
    H = 4
    inp = make_inputs(seqlens, H)
    ws = chunk_kda_prepare(**inp, lower_bound=LOWER_BOUND)
    ref = workspace_ref(inp, D**-0.5)
    for name in ("qg", "aqk", "w", "u"):
        assert_close(name, ref[name], ws[name][0])
    assert_close("kg_t", ref["kg_t"], ws["kg_t"])
    assert_close("decay", ref["decay"], ws["decay"])


@pytest.mark.parametrize("padded", [False, True])
def test_chunk_kda_paged(padded):
    """vLLM path: state cache read and written in place, out aliasing the dead v. ``padded`` lays the
    cache out like vLLM's hybrid pages: each slot's page holds the conv state first, then the
    recurrent state, then padding, so the slot stride is not H * D * D."""
    seqlens, H = [130, 1, 64, 257], 24
    inp = make_inputs(seqlens, H, seed=1)
    N = len(seqlens)
    lead, tail = (3 * 2 * H * D, 1000) if padded else (0, 0)
    page = lead + H * D * D + tail
    raw = torch.full((3 * N, page), POISON, device=DEVICE)
    cache = raw.as_strided((3 * N, H, D, D), (page, D * D, D, 1), storage_offset=lead)
    slots = torch.randperm(3 * N, device=DEVICE)[:N].int()
    has_init = torch.tensor([True, False, True, False], device=DEVICE)
    cache[slots.long()] = torch.randn(N, H, D, D, device=DEVICE)
    h0 = torch.where(has_init[:, None, None, None], cache[slots.long()], 0.0)
    o_ref, s_ref = run_ref(inp, h0)
    written = torch.zeros_like(raw, dtype=torch.bool)
    written.as_strided(cache.shape, cache.stride(), lead)[slots.long()] = True

    o, s = run_kernel(
        inp,
        out=inp["v"],
        state_cache=cache,
        state_indices=slots,
        has_initial_state=has_init,
    )
    assert s is None and o.data_ptr() == inp["v"].data_ptr()
    assert_close("o", o_ref, o)
    assert_close("final_state", s_ref, cache[slots.long()])
    assert (raw[~written] == POISON).all(), "cache memory outside the used slots was written"


def test_chunk_kda_fused_norm():
    seqlens, H = [200, 70], 4
    inp = make_inputs(seqlens, H, seed=2)
    og = torch.randn_like(inp["v"])
    nw = torch.rand(D, device=DEVICE) + 0.5
    eps = 1e-5
    o_ref, _ = run_ref(inp)
    of = o_ref.float()
    o_ref = (
        of
        * torch.rsqrt(of.square().mean(-1, keepdim=True) + eps)
        * nw
        * torch.sigmoid(og.float())
    )
    o, _ = run_kernel(
        inp,
        out_gate=og,
        norm_weight=nw,
        norm_eps=eps,
        config={"BV": 128, "num_warps": 4},
    )
    assert_close("o", o_ref, o)


@pytest.mark.parametrize(
    "dup_keys, gate_shift", [(True, 0.0), (False, 6.0), (True, 6.0)]
)
def test_chunk_kda_stress(dup_keys, gate_shift):
    """Near-duplicate keys and gates pinned at the lower bound: conditioning and pivot range."""
    seqlens, H = [512, 77], 4
    inp = make_inputs(seqlens, H, seed=3, dup_keys=dup_keys, gate_shift=gate_shift)
    h0 = torch.randn(len(seqlens), H, D, D, device=DEVICE)
    o_ref, s_ref = run_ref(inp, h0)
    o, s = run_kernel(inp, initial_state=h0, output_final_state=True)
    assert_close("o", o_ref, o, 1.25 * RATIO)
    assert_close("final_state", s_ref, s, 1.25 * RATIO)
