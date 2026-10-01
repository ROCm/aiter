# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math

import pytest
import torch

import aiter.ops.triton.attention.chunk_kda as chunk_kda_module
from aiter.ops.triton._gluon_kernels.gfx1250.attention.chunk_kda.walk import _get_config
from aiter.ops.triton.attention.chunk_kda import (
    CHUNK_SIZE,
    chunk_kda,
    chunk_kda_prepare,
    chunk_kda_walk,
    prepare_chunk_kda_metadata,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref, kda_gate_ref, l2norm_ref

ARCH = get_arch()
pytestmark = pytest.mark.skipif(
    ARCH not in ("gfx1250", "gfx950"),
    reason=f"chunk KDA gluon needs gfx1250 or gfx950, got {ARCH}",
)

DEVICE = "cuda"
D = 128
LOWER_BOUND = -5.0
RATIO = 0.008
POISON = 1e30
K3_H = 24
K3_PAGE = 442368
K3_STATE_OFF = 13824
K3_SLOTS = 64
if ARCH == "gfx950":
    WALK_TIERS = [
        {"BV": 16, "num_warps": 2, "KS": 2},
        {"BV": 32, "num_warps": 4, "KS": 2},
        {"BV": 64, "num_warps": 8, "KS": 2},
        {"BV": 128, "num_warps": 4, "KS": 1},
    ]
    WALK_EXTRA = [
        {"G": 4},
        {"G": 4, "BV": 128, "num_warps": 4, "KS": 1},
        {
            "G": 3,
            "num_stages": 1,
            "num_stages_scan": 1,
            "KS_pass1": 2,
            "BV_pass1": 64,
            "num_warps_pass1": 4,
        },
        {"BV": 64, "num_warps": 4, "KS": 1, "num_stages": 1},
    ]
else:
    # the tuned gfx1250 tiers
    WALK_TIERS = [
        {"BV": 16, "num_warps": 1},
        {"BV": 32, "num_warps": 2},
        {"BV": 64, "num_warps": 4, "waves_per_eu": 2},
        {"BV": 128, "num_warps": 4},
    ]
    WALK_EXTRA = [
        {"BV": 64, "num_warps": 4, "waves_per_eu": 3},
        {"BV": 128, "num_warps": 2},
    ]
WALK_WIDE = WALK_TIERS[-1]
WALK_CONFIGS = [None, *WALK_TIERS, *WALK_EXTRA]


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
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu,
    }


def cdiv(a, b):
    return (a + b - 1) // b


def make_vllm_inputs(seqlens, H=K3_H, nd_tok=0, spec=False, pad_tok=3, seed=0):
    """vLLM K3 operands: views past nd_tok decode rows, beta an in_proj column, out a
    core_attn_out slice (spec: index_select copies, out None). Returns (inp, kw, core).
    """
    torch.manual_seed(seed)
    T = sum(seqlens)
    width = 4 * H * D + D + H  # q | k | v | g2 | f_a | beta, padded to 16 columns
    width += -width % 16
    num_actual = 2 * T if spec else nd_tok + T  # spec: a draft token after each one
    num_tokens = num_actual + pad_tok
    proj = torch.randn(num_tokens, width, dtype=torch.bfloat16, device=DEVICE)
    beta = proj[:, 4 * H * D + D : 4 * H * D + D + H][None, :num_actual]
    g = torch.randn(1, num_tokens, H, D, dtype=torch.bfloat16, device=DEVICE)
    g = g[:, :num_actual]
    q, k, v = (
        torch.randn(1, num_actual, H, D, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3)
    )
    core = torch.randn(1, num_tokens, H, D, dtype=torch.bfloat16, device=DEVICE)
    if spec:
        idx = torch.arange(0, num_actual, 2, device=DEVICE)
        q, k, v, g, beta = (x.index_select(1, idx) for x in (q, k, v, g, beta))
        out = None
    else:
        q, k, v, g, beta = (x[:, nd_tok:num_actual] for x in (q, k, v, g, beta))
        out = core[:, nd_tok:num_actual]
        assert beta.stride(1) == width and beta.storage_offset() == (
            4 * H * D + D + nd_tok * width
        )
        assert q.stride(1) == H * D and q.storage_offset() == nd_tok * H * D
    chunk_indices = torch.tensor(
        [[n, c] for n, s in enumerate(seqlens) for c in range(cdiv(s, CHUNK_SIZE))],
        dtype=torch.int32,
        device=DEVICE,
    ).view(-1, 2)
    chunk_offsets = torch.tensor(
        [0, *itertools.accumulate(cdiv(s, CHUNK_SIZE) for s in seqlens)],
        dtype=torch.int64,
        device=DEVICE,
    )
    inp = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": torch.log(torch.empty(H, device=DEVICE).uniform_(1, 16)),
        "dt_bias": torch.randn(H * D, device=DEVICE),
        "cu_seqlens": torch.tensor(
            [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=DEVICE
        ),
    }
    return (
        inp,
        {"chunk_indices": chunk_indices, "chunk_offsets": chunk_offsets, "out": out},
        core,
    )


def make_page_cache(num_slots, H=K3_H, guard=0):
    """vLLM hybrid pages; ``guard`` POISON pages on each side catch -1 / num_slots rows."""
    raw = torch.full(((num_slots + 2 * guard) * K3_PAGE,), POISON, device=DEVICE)
    cache = raw.as_strided(
        (num_slots, H, D, D), (K3_PAGE, D * D, D, 1), guard * K3_PAGE + K3_STATE_OFF
    )
    return raw, cache


def seed_slots(raw, cache, slots, has_init):
    """Random valid slots; returns (expected start state, writable mask of raw, valid)."""
    valid = (slots >= 0) & (slots < cache.shape[0])
    rows = slots[valid].long()
    cache[rows] = torch.randn(len(rows), *cache.shape[1:], device=DEVICE)
    h0 = torch.zeros(len(slots), *cache.shape[1:], device=DEVICE)
    h0[valid & has_init] = cache[slots[valid & has_init].long()]
    written = torch.zeros_like(raw, dtype=torch.bool)
    written.as_strided(cache.shape, cache.stride(), cache.storage_offset())[rows] = True
    return h0, written, valid


def check_paged(o_ref, s_ref, o, raw, cache, slots, written, valid):
    assert_close("o", o_ref, o)
    assert_close("final_state", s_ref[valid], cache[slots[valid].long()])
    assert (
        raw[~written] == POISON
    ).all(), "cache memory outside the used slots was written"


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
    out = {
        "qg": torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        "w": torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        "u": torch.zeros(T, H, D, dtype=torch.float64, device=DEVICE),
        "aqk": torch.zeros(T, H, CHUNK_SIZE, dtype=torch.float64, device=DEVICE),
        "kg_t": [],
        "decay": [],
    }
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
    [[64], [1], [63], [65], [300], [1, 64, 130, 7], [1000], [5, 0, 70]],
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


@pytest.mark.parametrize("config", [None, WALK_WIDE])
@pytest.mark.parametrize("padded", [False, True])
def test_chunk_kda_paged(padded, config):
    """State cache in place, out aliasing v; ``padded`` gives a hybrid-page slot stride."""
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
        config=config,
    )
    assert s is None and o.data_ptr() == inp["v"].data_ptr()
    assert_close("o", o_ref, o)
    assert_close("final_state", s_ref, cache[slots.long()])
    assert (
        raw[~written] == POISON
    ).all(), "cache memory outside the used slots was written"


# None: the tuned default must take BV = V by itself when the norm is fused
@pytest.mark.parametrize("config", [{"BV": 128, "num_warps": 4}, None])
def test_chunk_kda_fused_norm(config):
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
        config=config,
    )
    assert_close("o", o_ref, o)


@pytest.mark.parametrize(
    "dup_keys, gate_shift", [(True, 0.0), (False, 6.0), (True, 6.0)]
)
def test_chunk_kda_stress(dup_keys, gate_shift):
    """Near-duplicate keys and gates pinned at the lower bound."""
    seqlens, H = [512, 77], 4
    inp = make_inputs(seqlens, H, seed=3, dup_keys=dup_keys, gate_shift=gate_shift)
    h0 = torch.randn(len(seqlens), H, D, D, device=DEVICE)
    o_ref, s_ref = run_ref(inp, h0)
    o, s = run_kernel(inp, initial_state=h0, output_final_state=True)
    assert_close("o", o_ref, o, 1.25 * RATIO)
    assert_close("final_state", s_ref, s, 1.25 * RATIO)


VLLM_CASES = {
    "split_4x1k": {
        "seqlens": [1030, 990, 956, 1005],
        "nd_tok": 114,
        "has_init": [False, True, False, False],
    },
    "single_4096": {"seqlens": [4096], "has_init": [True]},
    "long_default": {
        "seqlens": [2048, 2048],
        "nd_tok": 7,
        "has_init": [False, True],
        "config": WALK_WIDE,
    },
    "split_6": {
        "seqlens": [2, 130, 1, 64, 257, 3],
        "nd_tok": 5,
        "has_init": [True, False, True, True, False, False],
    },
    "spec": {
        "seqlens": [1] * 10 + [300, 0, 65],
        "spec": True,
        "has_init": [True] * 10 + [False, True, False],
    },
}


@pytest.mark.parametrize("case", list(VLLM_CASES))
def test_chunk_kda_vllm_layout(case):
    """vLLM's exact operand forms; no write outside the used slots or the out slice."""
    c = VLLM_CASES[case]
    seqlens, nd_tok = c["seqlens"], c.get("nd_tok", 0)
    N, T = len(seqlens), sum(seqlens)
    inp, kw, core = make_vllm_inputs(seqlens, nd_tok=nd_tok, spec=c.get("spec", False))
    raw, cache = make_page_cache(K3_SLOTS)
    assert cache.stride(0) == K3_PAGE and cache.storage_offset() == K3_STATE_OFF
    slots = (torch.randperm(K3_SLOTS - 1, device=DEVICE)[:N] + 1).int()
    has_init = torch.tensor(c["has_init"], device=DEVICE)
    h0, written, valid = seed_slots(raw, cache, slots, has_init)
    o_ref, s_ref = run_ref(inp, h0)
    core_before = core.clone()

    o, s = run_kernel(
        inp,
        **kw,
        state_cache=cache,
        state_indices=slots,
        has_initial_state=has_init,
        config=c.get("config"),
    )
    assert s is None
    check_paged(o_ref, s_ref, o, raw, cache, slots, written, valid)
    if kw["out"] is not None:
        assert o.data_ptr() == kw["out"].data_ptr()
        rest = torch.ones(core.shape[1], dtype=torch.bool, device=DEVICE)
        rest[nd_tok : nd_tok + T] = False
        assert torch.equal(core[:, rest], core_before[:, rest]), "wrote outside out"


# The Triton path expects every sequence to have a chunk: no empty-sequence cases.
@pytest.mark.parametrize(
    "case", [c for c, v in VLLM_CASES.items() if 0 not in v["seqlens"]]
)
def test_chunk_kda_triton_fallback(case, monkeypatch):
    """Archs without a Gluon build (gfx942) take the Triton path; forced here."""
    monkeypatch.setattr(chunk_kda_module, "_ARCH", "gfx942")
    test_chunk_kda_vllm_layout(case)


@pytest.mark.parametrize("nd_tok", [0, 3])
def test_chunk_kda_strided_state_indices(nd_tok):
    """block_table[:, 0] of an [R, 1 + num_spec] table: stride 3."""
    seqlens = [130, 2, 64, 257]
    N = len(seqlens)
    inp, kw, _ = make_vllm_inputs(seqlens, nd_tok=nd_tok, seed=4)
    raw, cache = make_page_cache(K3_SLOTS)
    ids = torch.randperm(K3_SLOTS - 1, device=DEVICE)[: 3 * (nd_tok + N)] + 1
    bt = ids.int().view(nd_tok + N, 3)
    slots = bt[nd_tok:, 0]
    init = torch.tensor([True, False, True, True], device=DEVICE)
    has_init = torch.stack([init, ~init], 1)[:, 0]
    assert slots.stride() == (3,) and has_init.stride() == (2,)
    h0, written, valid = seed_slots(raw, cache, slots, has_init)
    o_ref, s_ref = run_ref(inp, h0)
    bt_before = bt.clone()

    o, _ = run_kernel(
        inp, **kw, state_cache=cache, state_indices=slots, has_initial_state=has_init
    )
    check_paged(o_ref, s_ref, o, raw, cache, slots, written, valid)
    assert torch.equal(bt, bt_before)


@pytest.mark.parametrize(
    "seqlens",
    [
        [300, 1, 64, 130, 7, 1000, 65, 63],
        # spec decode reclassifying 1-token decodes as prefills, around a few prompts
        [1] * 36 + [700, 1, 129, 64],
    ],
    ids=["mixed_8", "spec_40"],
)
@pytest.mark.parametrize("config", [None, WALK_WIDE])
def test_chunk_kda_paged_many_seqs(seqlens, config):
    """Enough prefills at H = 24 to overfill the GPU at BV 64."""
    N = len(seqlens)
    if ARCH == "gfx1250":
        assert N * K3_H * (D // 64) > get_num_sms(), "too few pairs for the widest tier"
        assert _get_config(N, K3_H)["BV"] == 128
    inp, kw, _ = make_vllm_inputs(seqlens, nd_tok=2, seed=6)
    raw, cache = make_page_cache(K3_SLOTS)
    slots = (torch.randperm(K3_SLOTS - 1, device=DEVICE)[:N] + 1).int()
    has_init = torch.arange(N, device=DEVICE) % 3 != 1
    h0, written, valid = seed_slots(raw, cache, slots, has_init)
    o_ref, s_ref = run_ref(inp, h0)

    o, _ = run_kernel(
        inp,
        **kw,
        state_cache=cache,
        state_indices=slots,
        has_initial_state=has_init,
        config=config,
    )
    check_paged(o_ref, s_ref, o, raw, cache, slots, written, valid)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_chunk_kda_invalid_slots(dtype):
    """Slots -1 and num_slots start from zeros, still write o, and touch no cache page."""
    seqlens = [130, 64, 1, 257, 65]
    N = len(seqlens)
    inp, kw, _ = make_vllm_inputs(seqlens, seed=5)
    raw, cache = make_page_cache(K3_SLOTS, guard=1)
    ok = torch.randperm(K3_SLOTS, device=DEVICE)[:3].tolist()
    slots = torch.tensor(
        [-1, ok[0], K3_SLOTS, ok[1], ok[2]], dtype=dtype, device=DEVICE
    )
    has_init = torch.ones(N, dtype=torch.bool, device=DEVICE)
    h0, written, valid = seed_slots(raw, cache, slots, has_init)
    assert valid.tolist() == [False, True, False, True, True]
    o_ref, s_ref = run_ref(inp, h0)

    o, _ = run_kernel(
        inp, **kw, state_cache=cache, state_indices=slots, has_initial_state=has_init
    )
    check_paged(o_ref, s_ref, o, raw, cache, slots, written, valid)


def test_chunk_kda_chunked_prefill():
    """Prompts split at a 1536-token block: the second call continues from the cached states."""
    first, rest = [1536, 300], [500, 1]
    inp = make_inputs([a + b for a, b in zip(first, rest)], K3_H, seed=7)
    o_ref, s_ref = run_ref(inp)
    raw, cache = make_page_cache(8)
    slots = torch.tensor([5, 2], dtype=torch.int32, device=DEVICE)
    written = torch.zeros_like(raw, dtype=torch.bool)
    written.as_strided(cache.shape, cache.stride(), cache.storage_offset())[
        slots.long()
    ] = True
    cu = inp["cu_seqlens"].tolist()
    for has_init, spans in (
        (False, [(cu[n], cu[n] + first[n]) for n in range(2)]),
        (True, [(cu[n] + first[n], cu[n + 1]) for n in range(2)]),
    ):
        idx = torch.cat([torch.arange(a, b, device=DEVICE) for a, b in spans])
        part = {n: inp[n].index_select(1, idx) for n in ("q", "k", "v", "g", "beta")}
        part.update(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            cu_seqlens=torch.tensor(
                [0, *itertools.accumulate(b - a for a, b in spans)],
                dtype=torch.int32,
                device=DEVICE,
            ),
        )
        o, _ = run_kernel(
            part,
            state_cache=cache,
            state_indices=slots,
            has_initial_state=torch.full((2,), has_init, device=DEVICE),
        )
        assert_close("o", o_ref.index_select(1, idx), o)
    assert_close("final_state", s_ref, cache[slots.long()])
    assert (
        raw[~written] == POISON
    ).all(), "cache memory outside the used slots was written"


@pytest.mark.parametrize(
    "seqlens", [[5, 0, 70], [0, 64, 0, 0, 129, 1], [0], [64, 0], [1000]]
)
def test_chunk_kda_metadata(seqlens):
    """The chunk_indices=None fallback keeps sequence ids across zero-length sequences."""
    cu = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=DEVICE
    )
    chunk_indices, chunk_offsets = prepare_chunk_kda_metadata(cu)
    assert chunk_indices.dtype == torch.int32 and chunk_offsets.dtype == torch.int64
    assert chunk_indices.shape[1] == 2 and chunk_indices.is_contiguous()
    assert chunk_indices.tolist() == [
        [n, c] for n, s in enumerate(seqlens) for c in range(cdiv(s, CHUNK_SIZE))
    ]
    assert chunk_offsets.tolist() == [
        0,
        *itertools.accumulate(cdiv(s, CHUNK_SIZE) for s in seqlens),
    ]


HOST_CHECK_CASES = [
    "offsets_short",
    "offsets_float",
    "offsets_2d",
    "offsets_strided",
    "indices_3_cols",
    "indices_float",
    "indices_strided",
    "out_short",
    "out_heads",
    "out_dtype",
    "state_indices_short",
    "state_indices_float",
    "has_init_short",
    "initial_state_rows",
]


@pytest.mark.parametrize("bad", HOST_CHECK_CASES)
def test_chunk_kda_host_checks(bad):
    """Malformed metadata, out or state operands fail on the host."""
    seqlens, H = [65, 1, 130], 4
    N, T = len(seqlens), sum(seqlens)
    inp = make_inputs(seqlens, H)
    ci, co = prepare_chunk_kda_metadata(inp["cu_seqlens"])
    slots = torch.arange(N, dtype=torch.int32, device=DEVICE)
    paged = {
        "state_cache": torch.zeros(N, H, D, D, device=DEVICE),
        "state_indices": slots,
        "has_initial_state": torch.ones(N, dtype=torch.bool, device=DEVICE),
    }
    kw = {"chunk_indices": ci, "chunk_offsets": co}
    kw.update(
        {
            "offsets_short": {"chunk_offsets": co[:-1]},
            "offsets_float": {"chunk_offsets": co.float()},
            "offsets_2d": {"chunk_offsets": co[None]},
            "offsets_strided": {"chunk_offsets": torch.stack([co, co], 1)[:, 0]},
            "indices_3_cols": {"chunk_indices": torch.cat([ci, ci[:, :1]], 1)},
            "indices_float": {"chunk_indices": ci.float()},
            "indices_strided": {"chunk_indices": ci.t().contiguous().t()},
            "out_short": {"out": inp["v"][:, 1:]},
            "out_heads": {"out": inp["v"][:, :, 1:]},
            "out_dtype": {"out": torch.empty(1, T, H, D, device=DEVICE)},
            "state_indices_short": dict(paged, state_indices=slots[1:]),
            "state_indices_float": dict(paged, state_indices=slots.float()),
            "has_init_short": dict(
                paged, has_initial_state=paged["has_initial_state"][1:]
            ),
            "initial_state_rows": {
                "initial_state": torch.zeros(N - 1, H, D, D, device=DEVICE)
            },
        }[bad]
    )
    with pytest.raises(AssertionError):
        run_kernel(inp, **kw)


@pytest.mark.parametrize("bad", ["kg_t_rows", "decay_rows", "w_strided"])
def test_chunk_kda_walk_host_checks(bad):
    """The walk rejects a workspace that does not match its shapes or is not dense."""
    inp = make_inputs([65, 1, 130], 4)
    ws = chunk_kda_prepare(**inp, lower_bound=LOWER_BOUND)
    ws.update(
        {
            "kg_t_rows": {"kg_t": ws["kg_t"][1:]},
            "decay_rows": {"decay": ws["decay"][:-1]},
            "w_strided": {"w": ws["w"].transpose(1, 2).contiguous().transpose(1, 2)},
        }[bad]
    )
    with pytest.raises(AssertionError):
        chunk_kda_walk(**ws, cu_seqlens=inp["cu_seqlens"])


@pytest.mark.skipif(ARCH != "gfx1250", reason="gfx1250 walk tiers")
@pytest.mark.parametrize("N", [1, 2, 3, 5, 6, 11, 40])
def test_chunk_kda_walk_config_tiers(N):
    """config None takes the narrowest tier that fits the CUs; a BV override starts from it."""
    cus = get_num_sms()
    fits = [t["BV"] for t in WALK_TIERS if N * K3_H * (D // t["BV"]) <= cus]
    expect = fits[0] if fits else WALK_TIERS[-1]["BV"]
    config = _get_config(N, K3_H)
    tier = next(t for t in WALK_TIERS if t["BV"] == config["BV"])
    assert config["BV"] == expect and config["num_warps"] == tier["num_warps"]
    assert _get_config(N, K3_H, {"BV": 16})["num_warps"] == 1
    assert _get_config(N, K3_H, {"BV": 64, "num_warps": 2})["num_warps"] == 2
