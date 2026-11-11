# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and performance of aiter.ops.muse_attnblk_mega (Muse Spark one-launch attention block, TP4).

Synthetic weights and a paged FP8 KV cache with scattered pages. The output is checked against an fp32 reference of
the same block (QKV projection, per-head RMSNorm, q scale, GPT-J RoPE, causal / sliding-window attention over the
cached history plus the 8 verify tokens, o RMSNorm, sigmoid gate, o_proj summed over the 4 ranks, gated-residual
boundary). The K/V rows the kernel writes for the verify tokens are checked, and a CUDA graph is replayed with
alternating inputs to catch stale data across calls. Latency comes from run_perftest with 8 rotating copies of the
weights, the KV cache and the inputs (more than the L2 and MALL), so every call reads from HBM.

KV length profiles: "long" = quantiles of the InferenceX AgentX c40 verify trace (p50 118k tokens); "short" = short
contexts and lengths around the 2047-token window and the 16-token page edges.

    python op_tests/multigpu_tests/test_muse_attnblk_mega.py
    python op_tests/multigpu_tests/test_muse_attnblk_mega.py -l sliding -b 1 4 8 -k short
"""

import argparse
import itertools
import math
import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

NR, HID, NQ, NKV, HD, DRAFT, PAGE = 4, 8192, 32, 4, 64, 8, 16
NQKV = NQ * HD + 2 * NKV * HD + NQ * HD
# A query at position q sees keys k with q - window < k <= q (the window counts the query token); serving passes
# sliding_window - 1 = 2047, as the production SWA verify kernel (q - k < 2047) does.
WINDOW = 2047
NORM_EPS, POST_EPS, PRE_EPS = 1e-5, 1e-8, 1e-5
Q_SCALE, SM_SCALE = 5.4730064863875, HD**-0.5
ACC_THRESHOLD = 0.06  # FP8 QKV / FP8 K,V storage / bf16 attention partials
KV_THRESHOLD = 0.08  # FP8 e4m3 storage of the new K / V
REPLAY_TOL = 2e-2
FP8_THRESHOLD = 0.05  # FP8 e4m3 copy of the FFN input vs its BF16 value
SCALE_TOL = 1e-2  # per-row FP8 scale vs amax / 448 of the BF16 output
# Cached tokens per request (before the 8 verify tokens). "long": quantiles of the AgentX c40 verify trace; a batch
# of b requests takes b evenly spaced entries. "short": short contexts, the 2047-token window edge, page edges.
KV_PROFILES = {
    "long": [700, 5000, 17000, 33000, 59000, 78000, 101000, 149000],
    "short": [1, 15, 16, 130, 2039, 2046, 2047, 2049],
}
SUPPORTED_GFX = ["gfx950"]
ROTATE = 8  # rotating input copies for run_perftest; the same count on every rank


def kv_lens(profile, bs):
    lens = KV_PROFILES[profile]
    return [lens[(2 * i + 1) * len(lens) // (2 * bs)] for i in range(bs)]


def _rel(a, b):
    """Relative L2 error; inf when either side is not finite, so a NaN can never pass a threshold."""
    a, b = a.float(), b.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("inf")
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _quant_rows(x):
    s = x.float().abs().amax(-1).clamp_min(1e-12) / 448.0
    return (x.float() / s[:, None]).to(torch.float8_e4m3fn), s.float().contiguous()


def _rope_cache(max_pos, dev):
    inv = 1.0 / (
        500000.0 ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD)
    )
    f = torch.arange(max_pos, device=dev, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([f.cos(), f.sin()], -1).contiguous()  # [max_pos, 64]: cos | sin


def _rope(cs, positions, t):
    c, s = cs[positions].chunk(2, -1)
    t = t.reshape(t.shape[0], -1, HD)
    x1, x2 = t[..., 0::2], t[..., 1::2]
    c, s = c[:, None, :], s[:, None, :]
    return (
        torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1)
        .flatten(-2)
        .reshape(t.shape[0], -1)
    )


class Layer:
    """Per-rank synthetic weights of one layer (replicated tensors seeded identically on all ranks)."""

    def __init__(self, rank, sliding, dev, window=WINDOW):
        g = torch.Generator(device=dev).manual_seed(1000 + rank)
        w = torch.randn(NQKV, HID, device=dev, generator=g) / math.sqrt(HID)
        self.w_qkv_bf16 = w.to(torch.bfloat16)
        self.w_qkv, self.w_qkv_scale = _quant_rows(self.w_qkv_bf16)
        self.w_o = (
            torch.randn(HID, NQ * HD, device=dev, generator=g) / math.sqrt(NR * NQ * HD)
        ).to(torch.bfloat16)
        gr = torch.Generator(device=dev).manual_seed(77)
        self.alpha = (0.5 + torch.rand(HID, device=dev, generator=gr)).contiguous()
        self.beta = (0.5 + torch.rand(HID, device=dev, generator=gr)).contiguous()
        self.pre_w = (0.5 + torch.rand(HID, device=dev, generator=gr)).contiguous()
        self.sliding = sliding
        self.window = window if sliding else 0
        self.cos_sin = _rope_cache(200000, dev) if sliding else None


class KV:
    """Paged FP8 KV of one verify batch; pages randomly scattered (sliding: per-request SWA ring + slot mapping)."""

    def __init__(self, rank, seq_lens, sliding, seed, dev, window=WINDOW):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.window = window
        bs = len(seq_lens)
        need = [(s + DRAFT + PAGE - 1) // PAGE for s in seq_lens]
        full_pages = sum(need) + 8
        perm = torch.randperm(full_pages, generator=g)
        table = torch.zeros(bs, max(need), dtype=torch.int64)
        o = 0
        for b, n in enumerate(need):
            table[b, :n] = perm[o : o + n]
            o += n
        self.out_cache_loc = torch.tensor(
            [
                int(table[b, (s + j) // PAGE]) * PAGE + (s + j) % PAGE
                for b, s in enumerate(seq_lens)
                for j in range(DRAFT)
            ],
            dtype=torch.int64,
        ).to(dev)
        self.positions = torch.tensor(
            [s + j for s in seq_lens for j in range(DRAFT)], dtype=torch.int64
        ).to(dev)
        self.swa_mapping = None
        if sliding:
            wpages = (window + DRAFT) // PAGE + 3
            swa_pages = bs * wpages + 8
            sperm = torch.randperm(swa_pages, generator=g)
            stable = torch.zeros(bs, max(need), dtype=torch.int64)
            mapping = torch.zeros(full_pages * PAGE, dtype=torch.int64)
            for b, n in enumerate(need):
                ring = sperm[b * wpages : (b + 1) * wpages]
                stable[b, :n] = ring[torch.arange(n) % wpages]
                for off in range(PAGE):
                    mapping[table[b, :n] * PAGE + off] = stable[b, :n] * PAGE + off
            self.swa_mapping = mapping.to(dev)
            slots, page_table = swa_pages * PAGE, stable
        else:
            slots, page_table = full_pages * PAGE, table
        gk = torch.Generator(device=dev).manual_seed(seed * 7 + rank)
        self.k_cache = torch.randn(slots, NKV, HD, device=dev, generator=gk).to(
            torch.float8_e4m3fn
        )
        self.v_cache = torch.randn(slots, NKV, HD, device=dev, generator=gk).to(
            torch.float8_e4m3fn
        )
        self.block_table = page_table.to(dev, torch.int32).contiguous()
        self.seq_lens_list = list(seq_lens)
        self.seqused = (
            torch.tensor(seq_lens, dtype=torch.int32, device=dev) + DRAFT
        ).contiguous()
        self.sliding = sliding

    def _slots(self, b, pos):
        return self.block_table[b, pos // PAGE].long() * PAGE + pos % PAGE

    def history(self):
        hk, hv = [], []
        for b, s in enumerate(self.seq_lens_list):
            lo = max(0, s - (self.window - 1)) if self.sliding else 0
            sl = self._slots(b, torch.arange(lo, s, device=self.k_cache.device))
            hk.append(self.k_cache[sl].float())
            hv.append(self.v_cache[sl].float())
        return hk, hv

    def written(self):
        sl = (
            self.out_cache_loc
            if self.swa_mapping is None
            else self.swa_mapping[self.out_cache_loc]
        )
        return self.k_cache[sl].float(), self.v_cache[sl].float()


def run_torch(L, kv, x, stream, hist_k, hist_v):
    """fp32 reference of the block on this rank (o_proj reduced over the TP group). Not timed."""
    qkv = x.float() @ L.w_qkv_bf16.float().t()
    q, k, v, gate = qkv.split([NQ * HD, NKV * HD, NKV * HD, NQ * HD], -1)

    def hn(t):
        r = t.reshape(t.shape[0], -1, HD)
        return (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + NORM_EPS)).reshape(
            t.shape
        )

    q, k = hn(q) * Q_SCALE, hn(k)
    if L.cos_sin is not None:
        q, k = _rope(L.cos_sin, kv.positions, q), _rope(L.cos_sin, kv.positions, k)
    q, k, v = q.view(-1, NQ, HD), k.view(-1, NKV, HD), v.reshape(-1, NKV, HD)
    outs = []
    for b, s in enumerate(kv.seq_lens_list):
        rows = slice(b * DRAFT, (b + 1) * DRAFT)
        kk, vv = torch.cat([hist_k[b], k[rows]], 0), torch.cat([hist_v[b], v[rows]], 0)
        pos_k = torch.arange(s + DRAFT - kk.shape[0], s + DRAFT, device=x.device)
        pos_q = kv.positions[rows]
        mask = pos_k[None, :] <= pos_q[:, None]
        if L.sliding:
            mask &= pos_k[None, :] > pos_q[:, None] - L.window
        sc = (
            torch.einsum("thgd,lhd->thgl", q[rows].view(DRAFT, NKV, NQ // NKV, HD), kk)
            * SM_SCALE
        )
        p = sc.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        outs.append(torch.einsum("thgl,lhd->thgd", p, vv).reshape(DRAFT, NQ, HD))
    o = torch.cat(outs, 0)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    o = (o.reshape(-1, NQ * HD) * torch.sigmoid(gate)).contiguous()
    branch = o @ L.w_o.float().t()
    dist.all_reduce(branch)
    normed = (
        (branch * torch.rsqrt(branch.pow(2).mean(-1, keepdim=True) + POST_EPS))
        .to(torch.bfloat16)
        .float()
    )
    new_stream = L.alpha * stream + L.beta * normed
    out = (
        new_stream
        * torch.rsqrt(new_stream.pow(2).mean(-1, keepdim=True) + PRE_EPS)
        * L.pre_w
    )
    return out.to(torch.bfloat16), new_stream, k, v


def make_inputs(rows, salt, dev):
    # replicated across ranks
    g = torch.Generator(device=dev).manual_seed(4242 + 7919 * salt + rows)
    x = torch.randn(rows, HID, device=dev, generator=g)
    x = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    xq, xs = _quant_rows(x)
    stream = (4.0 * torch.randn(rows, HID, device=dev, generator=g)).contiguous()
    return x, xq, xs, stream


def _flops_bytes(L, kv, m):
    vis = [min(s, L.window - 1) if L.sliding else s for s in kv.seq_lens_list]
    attn_keys = sum(DRAFT * (v + DRAFT) for v in vis)  # verify rows x keys
    flops = 2 * m * HID * NQKV + 4 * attn_keys * NQ * HD + 2 * m * NQ * HD * HID
    kv_bytes = sum(v + DRAFT for v in vis) * NKV * HD * 2  # FP8 K and V
    w_bytes = NQKV * HID + NQKV * 4 + HID * NQ * HD * 2
    io_bytes = (
        m * HID * (1 + 4 + 2 + 4 + 1)
    )  # x fp8, stream in / out, out bf16, out fp8
    return flops, kv_bytes + w_bytes + io_bytes


@benchmark()
def test_muse_attnblk_mega(
    layer, window, bs, kv_profile, op=None, L=None, rank=0, dev=None
):
    sliding = layer == "sliding"
    kv = KV(
        rank, kv_lens(kv_profile, bs), sliding, seed=100 + bs, dev=dev, window=window
    )
    m = DRAFT * bs
    hist = kv.history()
    ins = [make_inputs(m, salt, dev) for salt in (0, 1)]

    def run(xq, xs, stream, w_qkv, w_qkv_scale, w_o, k_cache, v_cache, check=False):
        return op(
            xq, xs, stream, w_qkv=w_qkv, w_qkv_scale=w_qkv_scale, w_o=w_o, k_cache=k_cache, v_cache=v_cache,
            block_table=kv.block_table, seq_lens=kv.seqused, out_cache_loc=kv.out_cache_loc, positions=kv.positions,
            swa_mapping=kv.swa_mapping, cos_sin_cache=L.cos_sin, alpha=L.alpha, beta=L.beta,
            pre_norm_weight=L.pre_w, norm_eps=NORM_EPS, q_scale=Q_SCALE, sm_scale=SM_SCALE, post_eps=POST_EPS,
            pre_eps=PRE_EPS, window=L.window, validate_indices=check,
        )  # fmt: skip

    def args(i):
        return (*ins[i][1:], L.w_qkv, L.w_qkv_scale, L.w_o, kv.k_cache, kv.v_cache)

    # correctness: output and new stream vs the fp32 reference, written K / V vs the reference K / V
    # Serving runs the ranks in step; start every direct call together (the kernel traps after a 600 s wait).
    outs, rel, rel_kv, err, rel_fp8, scale_err = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for i, (x, _, _, stream) in enumerate(ins):
        torch.cuda.synchronize()
        dist.barrier()
        out = run(*args(i), check=True)
        torch.cuda.synchronize()
        ref = run_torch(L, kv, x, stream, *hist)
        wk, wv = kv.written()
        outs.append(out)
        if not all(torch.isfinite(t.float()).all() for t in out):
            rel = float("inf")
        rel = max(rel, _rel(out[0], ref[0]), _rel(out[1], ref[1]))
        rel_kv = max(rel_kv, _rel(wk, ref[2]), _rel(wv, ref[3]))
        # FP8 copy of the FFN input: the per-row scale must be amax / 448 of the BF16 output (error relative
        # to that scale), and the dequantized copy must match the BF16 output
        want = out[0].float().abs().amax(-1) / 448.0
        scale_err = max(
            scale_err, ((out[3] - want).abs() / want.clamp_min(1e-30)).max().item()
        )
        rel_fp8 = max(rel_fp8, _rel(out[2].float() * out[3][:, None], out[0]))
        err = max(
            err,
            checkAllclose(
                ref[0].float(), out[0].float(), rtol=0.1, atol=0.1, printLog=False
            ),
            checkAllclose(
                ref[1].float(), out[1].float(), rtol=0.05, atol=0.2, printLog=False
            ),
        )

    # graph replay with alternating inputs: stale data or a broken call epoch shows up here
    static = [t.clone() for t in ins[0][1:]]
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = run(*static, L.w_qkv, L.w_qkv_scale, L.w_o, kv.k_cache, kv.v_cache)
    rel_replay = 0.0
    for i in range(32):
        for d, s_ in zip(static, ins[i % 2][1:]):
            d.copy_(s_)
        graph.replay()
        torch.cuda.synchronize()
        ref_i = outs[i % 2]
        rel_replay = max(rel_replay, *(_rel(gout[j], ref_i[j]) for j in (0, 1, 3)))
        if not torch.equal(gout[2].view(torch.uint8), ref_i[2].view(torch.uint8)):
            rel_replay = max(rel_replay, 1.0)
    del graph

    torch.cuda.synchronize()
    dist.barrier()
    _, us = run_perftest(run, *args(0), num_rotate_args=ROTATE)
    flops, nbytes = _flops_bytes(L, kv, m)
    stat = torch.tensor(
        [us, rel, rel_kv, rel_replay, err, rel_fp8, scale_err], device=dev
    )
    dist.all_reduce(stat, op=dist.ReduceOp.MAX)
    us, rel, rel_kv, rel_replay, err, rel_fp8, scale_err = stat.tolist()
    ok = (
        rel <= ACC_THRESHOLD
        and rel_kv <= KV_THRESHOLD
        and rel_replay <= REPLAY_TOL
        and err <= 0.05
        and rel_fp8 <= FP8_THRESHOLD
        and scale_err <= SCALE_TOL
    )
    return {
        "gfx": get_gfx_runtime(),
        "kv_lens": "/".join(str(v) for v in kv.seq_lens_list),
        "mega us": us,
        "mega TFLOPS": flops / us / 1e6,
        "mega TB/s": nbytes / us / 1e6,
        "mega relL2": rel,
        "mega err": err,
        "kv relL2": rel_kv,
        "replay relL2": rel_replay,
        "fp8 relL2": rel_fp8,
        "scale relerr": scale_err,
        "pass": int(ok),
    }


def _check_input_guards(op, L, rank, dev):
    """Bad inputs must be rejected on the host, before the launch (no rank reaches the kernel)."""
    kv = KV(rank, [100], L.sliding, seed=7, dev=dev)
    _, xq, xs, stream = make_inputs(DRAFT, 0, dev)
    other = torch.device("cuda", (dev.index + 1) % torch.cuda.device_count())
    base = {
        "w_qkv": L.w_qkv, "w_qkv_scale": L.w_qkv_scale, "w_o": L.w_o, "k_cache": kv.k_cache, "v_cache": kv.v_cache,
        "block_table": kv.block_table, "seq_lens": kv.seqused, "out_cache_loc": kv.out_cache_loc, "positions": kv.positions,
        "swa_mapping": kv.swa_mapping, "cos_sin_cache": L.cos_sin, "alpha": L.alpha, "beta": L.beta,
        "pre_norm_weight": L.pre_w, "norm_eps": NORM_EPS, "q_scale": Q_SCALE, "sm_scale": SM_SCALE, "post_eps": POST_EPS,
        "pre_eps": PRE_EPS, "window": L.window,
    }  # fmt: skip
    bad_table = kv.block_table.clone()
    bad_table[0, 0] = kv.k_cache.shape[0] // PAGE  # one page past the pool
    cases = {
        "stream shape": ((xq, xs, stream[:, :4096].contiguous()), {}),
        "stream dtype": ((xq, xs, stream.to(torch.bfloat16)), {}),
        "x_fp8 device": ((xq.to(other), xs, stream), {}),
        "w_o device": ((xq, xs, stream), {"w_o": L.w_o.to(other)}),
        "k_cache layout": (
            (xq, xs, stream),
            {"k_cache": kv.k_cache.view(-1, 2 * NKV, HD // 2)},
        ),
        "positions dtype": ((xq, xs, stream), {"positions": kv.positions.int()}),
        "block_table page": (
            (xq, xs, stream),
            {"block_table": bad_table, "validate_indices": True},
        ),
        # only rank 0 is bad: validate_indices is collective, so every rank must refuse the call
        "block_table page on rank 0": (
            (xq, xs, stream),
            {
                "block_table": bad_table if rank == 0 else kv.block_table,
                "validate_indices": True,
            },
        ),
        "q_scale NaN": ((xq, xs, stream), {"q_scale": float("nan")}),
        "q_scale too large": ((xq, xs, stream), {"q_scale": 100.0}),
        "w_qkv_scale strided": (
            (xq, xs, stream),
            {"w_qkv_scale": torch.stack([L.w_qkv_scale] * 2, 1)[:, 0]},
        ),
    }
    rejected = []
    for name, (a, kw) in cases.items():
        try:
            op(*a, **{**base, **kw})
        except ValueError:
            rejected.append(name)
    missed = sorted(set(cases) - set(rejected))
    assert not missed, f"muse_attnblk_mega accepted bad input: {missed}"
    # A call made while another device is current must still run on the instance's device, with the same result.
    torch.cuda.synchronize()
    dist.barrier()
    ref = op(xq, xs, stream, **base)
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.device(other):
        got = op(xq, xs, stream, **base)
    torch.cuda.synchronize(dev)
    same = all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(ref, got)
    )
    assert same, "muse_attnblk_mega: result differs when another device is current"


def _worker(rank, port, cases, results):
    from aiter.ops.muse_attnblk_mega import MuseAttnBlockMega

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=NR,
        device_id=dev,
    )
    cpu = dist.new_group(backend="gloo")
    built = {}
    rows = []
    layers = {}
    for layer, window, bs, profile in cases:
        sliding = layer == "sliding"
        if layer not in built:
            for bad in (
                0,
                513,
            ):  # identical on every rank: all ranks refuse it before any collective
                try:
                    MuseAttnBlockMega(cpu, dev, sliding, qs_min=bad)
                    raise AssertionError(f"qs_min={bad} accepted")
                except ValueError:
                    pass
            built[layer] = MuseAttnBlockMega(cpu, dev, sliding)
            _check_input_guards(built[layer], Layer(rank, sliding, dev), rank, dev)
        if (layer, window) not in layers:
            layers[(layer, window)] = Layer(rank, sliding, dev, window=window)
        op, L = built[layer], layers[(layer, window)]
        row = test_muse_attnblk_mega(
            layer, window, bs, profile, op=op, L=L, rank=rank, dev=dev
        )
        for k in ("op", "L", "rank", "dev"):
            row.pop(k, None)
        rows.append(row)
        torch.cuda.empty_cache()
    for op in built.values():
        op.close()
    if rank == 0:
        results.extend(rows)
    dist.barrier()
    dist.destroy_process_group()


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX or torch.cuda.device_count() < NR:
        aiter.logger.warning(
            "muse_attnblk_mega needs %d %s GPUs; skipping on %s",
            NR,
            SUPPORTED_GFX,
            get_gfx_runtime(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-l",
        "--layer",
        nargs="*",
        default=["full", "sliding"],
        choices=["full", "sliding"],
        help="attention layer kind (full: causal over the whole context; sliding: 2047-token window)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=list(range(1, 9)),
        help="verify requests per call (rows = 8 x batch), 1..8",
    )
    parser.add_argument("-w", "--window", type=int, nargs="*", default=[WINDOW, 8],
                        help="sliding-layer window (2047 = serving; at 8 one missing key moves the output ~10%%)")  # fmt: skip
    parser.add_argument("-k", "--kv", nargs="*", default=["long", "short"], choices=sorted(KV_PROFILES),
                        help="cached-token profile per request")  # fmt: skip
    args = parser.parse_args()
    assert all(1 <= b <= 8 for b in args.batch)
    windows = {"full": [0], "sliding": args.window}
    cases = [
        (layer, w, b, k)
        for layer in args.layer
        for w, b, k in itertools.product(windows[layer], args.batch, args.kv)
    ]
    port = 29500 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.list()
        mp.spawn(_worker, args=(port, cases, results), nprocs=NR, join=True)
        rows = list(results)
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "muse_attnblk_mega summary (markdown):\n%s", df.to_markdown(index=False)
    )
    failed = df[df["pass"] == 0]
    assert (
        len(rows) == len(cases) and failed.empty
    ), f"muse_attnblk_mega failed:\n{failed}"


if __name__ == "__main__":
    main()
