# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import os

import pytest
import torch

from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi import kda_bi_fwd
from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi_decode import (
    kda_bi_cache_fill,
    kda_bi_cache_reset,
    kda_bi_decode,
    kda_bi_decode_cache,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="kda_bi_decode has pinned configs for gfx950 only"
)

D = 128
BT = 64
LB = -5.0
GOLDEN = os.path.join(os.path.dirname(__file__), "kda_bi_decode_golden.json")


def _params(H, seed, rng="cuda"):
    """(A_log [H], dt_bias [H * D]) fp32 on the GPU."""
    g = torch.Generator(device=rng).manual_seed(seed + 7)
    # Drawn in log space: torch.log on the CPU rounds differently across torch builds.
    A_log = torch.empty(H, dtype=torch.float32, device=rng).uniform_(
        0.0, 2.75, generator=g
    )
    dt_bias = 0.1 * torch.randn(H * D, dtype=torch.float32, device=rng, generator=g)
    return A_log.to("cuda"), dt_bias.to("cuda")


def _seqs(T, H, seed, rng="cuda"):
    """(q, k, v, g [1, T, H, D] bf16, beta [1, T, H] bf16) on the GPU. rng="cpu" gives the
    same values on every torch build (the golden test); the CUDA bf16 randn does not."""
    gen = torch.Generator(device=rng).manual_seed(seed)

    def bf16(*shape):
        x = torch.randn(shape, dtype=torch.float32, device=rng, generator=gen)
        return x.to(torch.bfloat16).to("cuda")

    q, k, v, g = (bf16(1, T, H, D) for _ in range(4))
    beta = torch.sigmoid(bf16(1, T, H).float()).to(torch.bfloat16)
    return q, k, v, g, beta


def _h0(N, H, seed, rng="cuda"):
    gen = torch.Generator(device=rng).manual_seed(seed + 3)
    h = 0.05 * torch.randn((N, H, D, D), dtype=torch.float32, device=rng, generator=gen)
    return h.to("cuda")


def _cu(lens):
    c = [0]
    for n in lens:
        c.append(c[-1] + n)
    return torch.tensor(c, dtype=torch.int32, device="cuda"), torch.tensor(c)


def _prefill(x, lens, A_log, dt_bias, h0, boundary=False):
    """kda_bi_fwd over packed sequences; returns (o [T, H, D], final, boundary or None)."""
    q, k, v, g, beta = x
    cu, cu_cpu = _cu(lens)
    out = kda_bi_fwd(
        q, k, v, g, beta, A_log, dt_bias, LB, cu,
        cu_seqlens_cpu=cu_cpu, initial_state=h0, output_final_state=True,
        output_boundary_state=boundary,
    )  # fmt: skip
    return out[0][0], out[1], out[2] if boundary else None


def _cut(x, starts, lens):
    """Packed rows [s, s + n) of each sequence of x (tuple of [1, T, ...])."""
    idx = torch.cat(
        [torch.arange(s, s + n, device="cuda") for s, n in zip(starts, lens)]
    ).long()
    return tuple(t[:, idx].contiguous() for t in x)


def _strided_state(n_slots, H, fill):
    """A vLLM-like state view: padded slot stride, contiguous [K, V] head blocks."""
    pad = 4096
    buf = torch.full(
        (n_slots * (H * D * D + pad),), fill, dtype=torch.float32, device="cuda"
    )
    return buf.as_strided((n_slots, H, D, D), (H * D * D + pad, D * D, D, 1))


class _Run:
    """N requests with their own slots / state rows; decodes the sequences after a
    prompt of prompt_len tokens, one token per active request per step."""

    def __init__(self, x, lens, H, A_log, dt_bias, h0, prompt_len, n_slots, seed):
        self.x, self.lens, self.H = x, lens, H
        self.A_log, self.dt_bias = A_log, dt_bias
        N = len(lens)
        g = torch.Generator().manual_seed(seed + 11)
        self.slot = torch.randperm(n_slots, generator=g)[:N].tolist()
        self.srow = torch.randperm(n_slots, generator=g)[:N].tolist()
        self.state = _strided_state(n_slots, H, float("nan"))
        self.cache = kda_bi_decode_cache(n_slots, H, "cuda")
        self.starts = [sum(lens[:i]) for i in range(N)]
        self.plen = [min(prompt_len, n) for n in lens]
        slots = torch.tensor(self.slot, dtype=torch.int32, device="cuda")
        srows = torch.tensor(self.srow, dtype=torch.int32, device="cuda")
        for i in range(N):
            self.state[self.srow[i]] = h0[i]
        kda_bi_cache_reset(self.cache, slots)
        if prompt_len > 0:
            px = _cut(x, self.starts, self.plen)
            _, _, b = _prefill(px, self.plen, A_log, dt_bias, h0, boundary=True)
            cu, _ = _cu(self.plen)
            kda_bi_cache_fill(
                self.cache, slots, *px, cu, A_log, dt_bias, LB, boundary_state=b,
                state=self.state, state_slots=srows,
            )  # fmt: skip
        self.t = list(self.plen)
        self.snap = []  # (request, chunk-aligned length, state after it)

    def step(self, order=None):
        """Decode one token of every unfinished request; returns {request: o row}."""
        act = [i for i in range(len(self.lens)) if self.t[i] < self.lens[i]]
        if order is not None:
            act = [i for i in order if i in act]
        if not act:
            return {}
        rows = torch.tensor(
            [self.starts[i] + self.t[i] for i in act], device="cuda"
        ).long()
        q, k, v, g, beta = (t[0, rows] for t in self.x)
        sl = torch.tensor([self.slot[i] for i in act], dtype=torch.int32, device="cuda")
        sr = torch.tensor([self.srow[i] for i in act], dtype=torch.int32, device="cuda")
        o = kda_bi_decode(
            q, k, v, g, beta, self.A_log, self.dt_bias, LB, self.state, self.cache,
            sl, state_slots=sr,
        )  # fmt: skip
        out = {}
        for j, i in enumerate(act):
            out[i] = o[j]
            self.t[i] += 1
            if self.t[i] % BT == 0:
                self.snap.append((i, self.t[i], self.state[self.srow[i]].clone()))
        return out

    def run(self):
        outs = {i: [] for i in range(len(self.lens))}
        while True:
            o = self.step()
            if not o:
                return outs
            for i, r in o.items():
                outs[i].append(r)


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16 if a.element_size() == 2 else torch.int32),
                       b.contiguous().view(torch.int16 if b.element_size() == 2 else torch.int32))  # fmt: skip


def generate_kda_bi_decode_inputs(R, H, seed=0):
    """kda_bi_decode arguments (in order) for R requests at random open-chunk positions (bench and
    tests): one new row each, a [R, H, 128, 128] state and a cache with R slots."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, g = (
        torch.randn((R, H, D), dtype=torch.bfloat16, device="cuda", generator=gen)
        for _ in range(4)
    )
    beta = torch.rand((R, H), dtype=torch.float32, device="cuda", generator=gen)
    A_log, dt_bias = _params(H, seed)
    cache = kda_bi_decode_cache(R, H, "cuda")
    for t in (cache.q, cache.k, cache.v, cache.g):
        t.normal_(generator=gen)
    cache.g.uniform_(LB / 4, 0.0, generator=gen)  # gate values
    cache.beta.uniform_(generator=gen)
    pos = torch.randint(0, BT, (R, 1), device="cuda", generator=gen)
    cache.pos.copy_(pos.expand(R, H))
    state = 0.05 * torch.randn((R, H, D, D), dtype=torch.float32, device="cuda", generator=gen)  # fmt: skip
    slots = torch.randperm(R, device="cuda", generator=gen).to(torch.int32)
    return q, k, v, g, beta, A_log, dt_bias, LB, state, cache, slots


LENS = [1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 200]


@pytest.mark.parametrize("prompt_len", [0, 1, 64, 70])
@pytest.mark.parametrize("H", [2, 12])
@pytest.mark.parametrize("random_h0", [False, True])
def test_decode_equals_prefill(prompt_len, H, random_h0):
    """Every decoded row and every chunk-end state equal kda_bi_fwd, bitwise."""
    seed = 1000 * prompt_len + H + int(random_h0)
    x = _seqs(sum(LENS), H, seed)
    A_log, dt_bias = _params(H, seed)
    N = len(LENS)
    h0 = _h0(N, H, seed) if random_h0 else torch.zeros((N, H, D, D), dtype=torch.float32, device="cuda")  # fmt: skip
    o_ref, _, b_ref = _prefill(x, LENS, A_log, dt_bias, h0, boundary=True)
    run = _Run(x, LENS, H, A_log, dt_bias, h0, prompt_len, n_slots=N + 5, seed=seed)
    outs = run.run()
    for i, n in enumerate(LENS):
        p = run.plen[i]
        if n == p:
            continue
        got = torch.stack(outs[i])
        want = o_ref[run.starts[i] + p : run.starts[i] + n]
        assert _bits_equal(got, want), f"o differs: request {i} (T={n}, prompt {p})"
        assert _bits_equal(
            run.state[run.srow[i]], b_ref[i]
        ), f"final boundary state differs: request {i}"
    # chunk-end states against the prefill of each chunk-aligned prefix
    if run.snap:
        who = [i for i, _, _ in run.snap]
        plens = [n for _, n, _ in run.snap]
        starts = [run.starts[i] for i in who]
        _, final, _ = _prefill(_cut(x, starts, plens), plens, A_log, dt_bias, h0[who])
        for j, (i, n, h) in enumerate(run.snap):
            assert _bits_equal(
                h, final[j]
            ), f"state differs: request {i} after {n} tokens"
    # untouched state rows keep their content
    used = set(run.srow)
    for r in range(run.state.shape[0]):
        if r not in used:
            assert torch.isnan(run.state[r]).all()


def test_batch_order_invariance():
    """The same requests give the same bits alone, together, and in another order."""
    H = 12
    lens = [70, 64, 130, 5, 63, 100]
    x = _seqs(sum(lens), H, 5)
    A_log, dt_bias = _params(H, 5)
    h0 = _h0(len(lens), H, 5)
    together = _Run(x, lens, H, A_log, dt_bias, h0, 30, 16, 1).run()
    perm = [3, 0, 5, 1, 4, 2]
    shuf = _Run(x, lens, H, A_log, dt_bias, h0, 30, 16, 2)
    shuffled = {i: [] for i in range(len(lens))}
    while True:
        o = shuf.step(order=perm)
        if not o:
            break
        for i, r in o.items():
            shuffled[i].append(r)
    for i in range(len(lens)):
        if not together[i]:
            continue
        one = _Run(*_one(x, lens, i), H, A_log, dt_bias, h0[i : i + 1], 30, 3, 3).run()[
            0
        ]
        a = torch.stack(together[i])
        assert _bits_equal(a, torch.stack(shuffled[i])), f"request {i}: order"
        assert _bits_equal(a, torch.stack(one)), f"request {i}: batch size"


def _one(x, lens, i):
    s = sum(lens[:i])
    return _cut(x, [s], [lens[i]]), [lens[i]]


def _ref_fp64(x, lens, A_log, dt_bias, h0, scale=D**-0.5):
    """Naive KDA recurrence in float64: (o [T, H, D], final state [N, H, K, V])."""
    q, k, v, g, beta = (t[0].double() for t in x)
    H = q.shape[1]
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gate = LB * torch.sigmoid(
        torch.exp(A_log.double())[None, :, None] * (g + dt_bias.double().view(H, D))
    )
    o = torch.empty_like(v)
    finals = []
    t0 = 0
    for n, (L, h) in enumerate(zip(lens, h0)):
        S = h.double().clone()
        for t in range(t0, t0 + L):
            S = S * torch.exp(gate[t])[:, :, None]
            u = beta[t][:, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], S))
            S = S + k[t][:, :, None] * u[:, None, :]
            o[t] = scale * torch.einsum("hk,hkv->hv", q[t], S)
        finals.append(S)
        t0 += L
    return o, torch.stack(finals)


def test_accuracy_vs_fp64():
    H = 2
    lens = [130, 40]
    x = _seqs(sum(lens), H, 9)
    A_log, dt_bias = _params(H, 9)
    h0 = _h0(len(lens), H, 9)
    run = _Run(x, lens, H, A_log, dt_bias, h0, 0, 4, 9)
    outs = run.run()
    o_ref, _ = _ref_fp64(x, lens, A_log, dt_bias, h0)
    got = torch.cat([torch.stack(outs[i]) for i in range(len(lens))]).double()
    torch.testing.assert_close(got, o_ref, rtol=2e-2, atol=2e-2)


def test_cuda_graph():
    """One captured step replayed over 130 tokens (two chunk ends) equals eager."""
    H = 12
    lens = [200, 200, 200]
    prompt = [0, 37, 63]
    x = _seqs(sum(lens), H, 21)
    A_log, dt_bias = _params(H, 21)
    h0 = _h0(len(lens), H, 21)
    R = len(lens)
    starts = [sum(lens[:i]) for i in range(R)]
    runs = []
    for _ in range(2):
        st = _strided_state(8, H, 0.0)
        c = kda_bi_decode_cache(8, H, "cuda")
        sl = torch.tensor([5, 1, 6], dtype=torch.int32, device="cuda")
        sr = torch.tensor([2, 7, 0], dtype=torch.int32, device="cuda")
        kda_bi_cache_reset(c, sl)
        for i in range(R):
            st[sr[i].item()] = h0[i]
        # warm the open chunks token by token (prompt rows), eagerly
        for t in range(max(prompt)):
            act = [i for i in range(R) if t < prompt[i]]
            rows = torch.tensor([starts[i] + t for i in act], device="cuda").long()
            kda_bi_decode(
                *(y[0, rows] for y in x), A_log, dt_bias, LB, st, c, sl[act], state_slots=sr[act]
            )  # fmt: skip
        runs.append((st, c, sl, sr))
    steps = 130
    rows = [[starts[i] + prompt[i] + j for i in range(R)] for j in range(steps)]
    eager = []
    st, c, sl, sr = runs[0]
    for j in range(steps):
        r = torch.tensor(rows[j], device="cuda").long()
        eager.append(kda_bi_decode(*(y[0, r] for y in x), A_log, dt_bias, LB, st, c, sl, state_slots=sr))  # fmt: skip
    st2, c2, sl2, sr2 = runs[1]
    inp = [torch.empty_like(y[0, :R]) for y in x]
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph_args = (*inp, A_log, dt_bias, LB, st2, c2, sl2)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            out = kda_bi_decode(*graph_args, state_slots=sr2)
    torch.cuda.current_stream().wait_stream(s)
    for j in range(steps):
        r = torch.tensor(rows[j], device="cuda").long()
        for b, y in zip(inp, x):
            b.copy_(y[0, r])
        graph.replay()
        assert _bits_equal(out, eager[j]), f"graph replay differs at step {j}"
    assert _bits_equal(st2, st)


def _digest(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


GOLDEN_CASES = {
    "h12_t200_prompt0": {"lens": [200], "H": 12, "prompt": 0, "seed": 0},
    "h12_mixed_prompt70": {"lens": [150, 90, 1, 64], "H": 12, "prompt": 70, "seed": 1},
    "h2_h0_prompt1": {"lens": [129, 17], "H": 2, "prompt": 1, "seed": 2},
}


def _golden_case(lens, H, prompt, seed):
    x = _seqs(sum(lens), H, seed, rng="cpu")
    A_log, dt_bias = _params(H, seed, rng="cpu")
    h0 = _h0(len(lens), H, seed, rng="cpu")
    run = _Run(x, lens, H, A_log, dt_bias, h0, prompt, len(lens) + 2, seed)
    outs = run.run()
    o = torch.cat([torch.stack(outs[i]) for i in range(len(lens)) if outs[i]])
    st = torch.stack([run.state[r] for r in run.srow])
    return _digest(o, st)


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_bits(name):
    got = _golden_case(**GOLDEN_CASES[name])
    if os.environ.get("KDA_BI_DECODE_WRITE_GOLDEN") == "1":
        d = {}
        if os.path.exists(GOLDEN):
            with open(GOLDEN) as f:
                d = json.load(f)
        d[name] = got
        with open(GOLDEN, "w") as f:
            json.dump(d, f, indent=2, sort_keys=True)
            f.write("\n")
    with open(GOLDEN) as f:
        want = json.load(f)[name]
    assert got == want, f"{name}: decode bits changed (contract version bump needed)"


def test_strided_inputs():
    """q, k, v as column slices of one fused [R, 3, H, D] tensor give the same bits."""
    H = 12
    lens = [10, 70]
    x = _seqs(sum(lens), H, 4)
    A_log, dt_bias = _params(H, 4)
    h0 = _h0(2, H, 4)
    ref = _Run(x, lens, H, A_log, dt_bias, h0, 0, 4, 4).run()
    fused = torch.stack([x[0][0], x[1][0], x[2][0]], 1)  # [T, 3, H, D]
    x2 = (fused[None, :, 0], fused[None, :, 1], fused[None, :, 2], x[3], x[4])
    got = _Run(x2, lens, H, A_log, dt_bias, h0, 0, 4, 4).run()
    for i in range(2):
        assert _bits_equal(torch.stack(got[i]), torch.stack(ref[i]))


def test_slot_reuse():
    """A reset slot starts a new request with no trace of the old one."""
    H = 2
    x = _seqs(90, H, 6)
    A_log, dt_bias = _params(H, 6)
    h0 = _h0(1, H, 6)
    fresh = _Run(x, [90], H, A_log, dt_bias, h0, 0, 2, 6).run()[0]
    y = _seqs(50, H, 7)
    used = _Run(y, [50], H, A_log, dt_bias, h0, 0, 2, 6)
    used.run()
    again = _Run(x, [90], H, A_log, dt_bias, h0, 0, 2, 6)
    again.cache = used.cache  # same slot ids (same seed), stale rows inside
    kda_bi_cache_reset(
        again.cache, torch.tensor(again.slot, dtype=torch.int32, device="cuda")
    )
    got = again.run()[0]
    assert _bits_equal(torch.stack(got), torch.stack(fresh))


def _call(**over):
    H, R = 2, 3
    a = {
        "q": torch.zeros((R, H, D), dtype=torch.bfloat16, device="cuda"),
        "k": torch.zeros((R, H, D), dtype=torch.bfloat16, device="cuda"),
        "v": torch.zeros((R, H, D), dtype=torch.bfloat16, device="cuda"),
        "g": torch.zeros((R, H, D), dtype=torch.bfloat16, device="cuda"),
        "beta": torch.zeros((R, H), dtype=torch.float32, device="cuda"),
        "A_log": torch.zeros(H, dtype=torch.float32, device="cuda"),
        "dt_bias": torch.zeros(H * D, dtype=torch.float32, device="cuda"),
        "lower_bound": LB,
        "state": torch.zeros((4, H, D, D), dtype=torch.float32, device="cuda"),
        "cache": kda_bi_decode_cache(4, H, "cuda"),
        "slots": torch.arange(R, dtype=torch.int32, device="cuda"),
    }
    a.update(over)
    return kda_bi_decode(**a)


_BAD = {
    "q_fp16": lambda: {"q": torch.zeros((3, 2, D), dtype=torch.float16, device="cuda")},
    "beta_fp16": lambda: {"beta": torch.zeros((3, 2), dtype=torch.float16, device="cuda")},
    "state_bf16": lambda: {"state": torch.zeros((4, 2, D, D), dtype=torch.bfloat16, device="cuda")},
    "state_vk": lambda: {"state": torch.zeros((4, 2, D, D), dtype=torch.float32, device="cuda").transpose(2, 3)},
    "head_dim_64": lambda: {"q": torch.zeros((3, 2, 64), dtype=torch.bfloat16, device="cuda")},
    "lb_zero": lambda: {"lower_bound": 0.0},
    "lb_low": lambda: {"lower_bound": -6.0},
    "slots_int64": lambda: {"slots": torch.arange(3, device="cuda")},
    "slots_short": lambda: {"slots": torch.arange(2, dtype=torch.int32, device="cuda")},
    "cache_heads": lambda: {"cache": kda_bi_decode_cache(4, 3, "cuda")},
}  # fmt: skip


@pytest.mark.parametrize("case", sorted(_BAD))
def test_refusals(case):
    with pytest.raises(ValueError):
        _call(**_BAD[case]())


def test_state_past_32bit_offsets():
    """A state row more than 2^31 elements into its buffer gives the same bits."""
    H, n = 2, 70
    big = 2**31 + 64
    if torch.cuda.mem_get_info()[0] < 4 * (big + H * D * D) + 2**30:
        pytest.skip("needs about 9 GB of free GPU memory")
    x = _seqs(n, H, 8)
    A_log, dt_bias = _params(H, 8)
    h0 = _h0(1, H, 8)
    ref = _Run(x, [n], H, A_log, dt_bias, h0, 0, 1, 8)
    want = ref.run()[0]
    buf = torch.zeros(big + H * D * D, dtype=torch.float32, device="cuda")
    far = _Run(x, [n], H, A_log, dt_bias, h0, 0, 1, 8)
    far.state = buf.as_strided((2, H, D, D), (big, D * D, D, 1))
    far.srow = [1]
    far.state[1] = h0[0]
    got = far.run()[0]
    assert _bits_equal(torch.stack(got), torch.stack(want))
    assert _bits_equal(far.state[1], ref.state[ref.srow[0]])


def test_both_launch_paths():
    """70 requests start on the multi-launch path and end on the one-launch path as they
    finish; every row and state equals kda_bi_fwd."""
    H = 2
    lens = [(7 * i) % 130 + 1 for i in range(70)]
    x = _seqs(sum(lens), H, 31)
    A_log, dt_bias = _params(H, 31)
    h0 = _h0(len(lens), H, 31)
    o_ref, _, b_ref = _prefill(x, lens, A_log, dt_bias, h0, boundary=True)
    run = _Run(x, lens, H, A_log, dt_bias, h0, 0, len(lens) + 3, 31)
    outs = run.run()
    for i, n in enumerate(lens):
        got = torch.stack(outs[i])
        assert _bits_equal(
            got, o_ref[run.starts[i] : run.starts[i] + n]
        ), f"request {i}"
        assert _bits_equal(run.state[run.srow[i]], b_ref[i]), f"state {i}"
