# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph replay time of the FlyDSL MXFP4 indexer top-512 against the
decode chain it replaces: aiter's ``paged_mxfp4_mqa_logits`` (the Gluon
kernel, with its step schedule) followed by a per-row top-k over the
materialised logits, as vLLM's ROCm MXFP4 indexer runs them.

Modes (one decode step of ``--reqs`` requests of ``--next-n`` rows each):

- ``dense``: ``flydsl_mxfp4_indexer_topk`` against logits + top-512.
- ``source``: the candidate source layer, ``flydsl_mxfp4_indexer_topk`` with
  ``candidates_out`` against logits + block maxima (one launch), top-2048 of
  the block maxima, top-512 of the logits.
- ``consumer``: ``flydsl_mxfp4_indexer_candidate_topk`` against
  ``build_candidate_gather`` (once a step for the consumer layers: counted
  ``1 / --consumers`` a call), the gathered logits, top-512 and the remap of
  candidate slots to positions.

Top-k of the baseline: vLLM's rule (``top_k_per_row_decode`` of vLLM below
128 rows and k < 2048, aiter's ``top_k_per_row_decode`` otherwise) when vLLM's
custom ops import, aiter's everywhere otherwise. The step schedule of the
baseline is built outside the timed graph (it is shared by a step's layers).

Request i of N has ``ctx * (spread + (1 - spread) * (i + 0.5) / N)`` entries.
``--inputs`` distinct steps (q, scales, weights, page tables, lengths) over a
cache pool larger than the last-level cache are captured in one graph; the
time per call is the median over rounds of replays, rounds ordered A B B A.

    python3 op_tests/flydsl_tests/bench_flydsl_mxfp4_indexer_topk.py \\
        --mode dense --ctx 65536 131072 --reqs 3 11 32 64 128
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import mxfp4_indexer_topk as ix
from aiter.ops.topk import top_k_per_row_decode as aiter_topk
from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa

H, ENTRY, TOPK, CANDIDATES, BLOCK = 32, 68, 512, 2048, 8

try:
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    _VLLM_TOPK = torch.ops._C.top_k_per_row_decode
except (ImportError, AttributeError, RuntimeError):  # vLLM absent: aiter's top-k
    _VLLM_TOPK = None


def baseline_topk(logits, lengths, out, k):
    rows, width = logits.shape
    if _VLLM_TOPK is not None and not (width >= k and not (rows < 128 and k < 2048)):
        _VLLM_TOPK(logits, 1, lengths, out, rows, logits.stride(0), 1, k)
    else:
        aiter_topk(logits, 1, lengths.reshape(-1), out, rows, logits.stride(0), 1, k=k)


@triton.jit
def _remap_kernel(idx_ptr, pos_ptr, pos_stride, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, K)
    slot = tl.load(idx_ptr + row * K + cols)
    hit = slot >= 0
    slot = tl.where(hit, slot, 0)
    start = tl.load(pos_ptr + row * pos_stride + slot // B, mask=hit, other=0)
    tl.store(idx_ptr + row * K + cols, tl.where(hit, start + slot % B, -1))


def remap(topk, positions):
    """Candidate slots to positions in place (slot j lies in pool block
    j // 8, which starts at positions[j // 8])."""
    _remap_kernel[(topk.shape[0],)](
        topk, positions, positions.stride(0), K=TOPK, B=BLOCK, num_warps=4
    )


def timed(g, nin, reps=5):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (reps * nin)


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()  # compiles and allocates the workspaces outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def abba(graphs, nin, rounds):
    t = {k: [] for k in graphs}
    order = list(graphs)
    for r in range(rounds):
        for k in order if r % 2 == 0 else order[::-1]:
            t[k].append(min(timed(graphs[k], nin) for _ in range(3)))
    return {k: statistics.median(v) for k, v in t.items()}


def contexts(ctx, reqs, spread):
    return [int(ctx * (spread + (1 - spread) * (i + 0.5) / reqs)) for i in range(reqs)]


def random_pool(pages, P, gen):
    pool = torch.randint(
        0, 256, (pages, P * ENTRY), dtype=torch.uint8, device="cuda", generator=gen
    )
    pool[:, P * 64 :] = torch.randint(
        120, 127, (pages, P * 4), dtype=torch.uint8, device="cuda", generator=gen
    )
    return pool.view(pages, P, ENTRY)


def make_step(lengths, nn, P, pages, gen):
    B = len(lengths)
    npg = [-(-c // P) for c in lengths]
    ids = torch.randperm(pages, device="cuda", generator=gen)[: sum(npg)].int()
    bt = torch.zeros(B, max(npg), dtype=torch.int32, device="cuda")
    o = 0
    for i, n in enumerate(npg):
        bt[i, :n] = ids[o : o + n]
        o += n
    T = B * nn
    cl = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    ends = cl[:, None] - nn + torch.arange(nn, device="cuda")[None] + 1
    return {
        "q": torch.randint(
            0, 256, (T, H, 64), dtype=torch.uint8, device="cuda", generator=gen
        ),
        "qs": torch.randint(
            120, 127, (T, H, 4), dtype=torch.uint8, device="cuda", generator=gen
        ),
        "w": torch.randn(T, H, device="cuda", generator=gen) * 0.1,
        "bt": bt,
        "cl": cl,
        "ends": ends.clamp(min=0).reshape(-1).int().contiguous(),
    }


def bench(mode, ctx, reqs, args, pool, gen):
    nn, P = args.next_n, pool.shape[1]
    kv4 = pool.view(pool.shape[0], P, 1, ENTRY)
    bound = max(args.max_ctx or ctx, ctx)  # the static bound of a captured graph
    width = -(-bound // 256) * 256  # the baseline's logits width
    lengths = contexts(ctx, reqs, args.spread)
    if sum(-(-c // P) for c in lengths) > pool.shape[0]:
        return None
    T = reqs * nn
    xs = [make_step(lengths, nn, P, pool.shape[0], gen) for _ in range(args.inputs)]
    logits = torch.empty(T, width, dtype=torch.float32, device="cuda")
    bscores = torch.empty(T, -(-width // BLOCK), dtype=torch.float32, device="cuda")
    topk = torch.empty(T, TOPK, dtype=torch.int32, device="cuda")
    cand = torch.empty(T, CANDIDATES, dtype=torch.int32, device="cuda")
    out = torch.empty(T, TOPK, dtype=torch.int32, device="cuda")
    sbuf = torch.empty(4 * pa.SCHED_SLOT_CAP, dtype=torch.int32, device="cuda")
    for x in xs:
        s = pa.build_schedule(
            x["cl"], nn, H, 128, P, out=sbuf, row_ends=x["ends"], max_model_len=width
        )
        x["sched"] = None if s is None else s.clone()
        x["q4"] = x["q"].view(reqs, nn, H, 64)
        x["qs4"] = x["qs"].view(reqs, nn, H, 4)
        x["bends"] = (x["ends"] + BLOCK - 1) // BLOCK

    def walk(x, block_scores):
        kw = {}
        if block_scores:
            kw = {"calc_block_scores": True, "block_scores": bscores}
        pa.paged_mxfp4_mqa_logits(
            x["q4"],
            x["qs4"],
            kv4,
            x["w"],
            x["cl"],
            x["bt"],
            width,
            out_logits=logits,
            clean_logits=False,
            row_ends=x["ends"],
            schedule=x["sched"],
            **kw,
        )

    graphs, parts = {}, {}
    if mode in ("dense", "source"):
        src = mode == "source"

        def base_walk():
            for x in xs:
                walk(x, src)

        def base_select():
            for x in xs:
                if src:
                    baseline_topk(bscores, x["bends"], cand, CANDIDATES)
                baseline_topk(logits, x["ends"], topk, TOPK)

        graphs["walk"] = graph(base_walk)
        graphs["select"] = graph(base_select)

        def mine():
            for x in xs:
                ix.flydsl_mxfp4_indexer_topk(
                    x["q"],
                    x["qs"],
                    x["w"],
                    pool,
                    x["bt"],
                    x["ends"],
                    out,
                    next_n=nn,
                    max_ctx=bound,
                    candidates_out=cand if src else None,
                )

    else:
        for x in xs:  # each step's candidate lists, from the source layer
            ix.flydsl_mxfp4_indexer_topk(
                x["q"],
                x["qs"],
                x["w"],
                pool,
                x["bt"],
                x["ends"],
                out,
                next_n=nn,
                max_ctx=bound,
                candidates_out=cand,
            )
            x["cand"] = cand.clone()
            x["btr"] = x["bt"].repeat_interleave(nn, 0).contiguous()
            x["gather"] = pa.build_candidate_gather(
                x["cand"], x["ends"], x["btr"], kv4, H, 128, BLOCK
            )
        compact = torch.empty(T, CANDIDATES * BLOCK, dtype=torch.float32, device="cuda")

        def build():
            for x in xs:
                pa.build_candidate_gather(
                    x["cand"], x["ends"], x["btr"], kv4, H, 128, BLOCK
                )

        def gathered():
            for x in xs:
                g, slot_ends = x["gather"]
                pa.paged_mxfp4_mqa_logits(
                    x["q"].view(T, 1, H, 64),
                    x["qs"].view(T, 1, H, 4),
                    kv4,
                    x["w"],
                    x["ends"],
                    x["btr"],
                    CANDIDATES * BLOCK,
                    out_logits=compact,
                    clean_logits=False,
                    row_ends=slot_ends,
                    use_gather=True,
                    candidates=g,
                )
                baseline_topk(compact, slot_ends, topk, TOPK)
                remap(topk, g["positions"])

        graphs["build"] = graph(build)
        graphs["gather+topk+remap"] = graph(gathered)

        def mine():
            for x in xs:
                ix.flydsl_mxfp4_indexer_candidate_topk(
                    x["q"],
                    x["qs"],
                    x["w"],
                    pool,
                    x["bt"],
                    x["ends"],
                    x["cand"],
                    out,
                    next_n=nn,
                    max_ctx=bound,
                )

    graphs["flydsl"] = graph(mine)
    t = abba(graphs, args.inputs, args.rounds)
    parts = {k: v for k, v in t.items() if k != "flydsl"}
    if "build" in parts:
        parts["build"] /= args.consumers
    base = sum(parts.values())
    kv_mb = sum(lengths) * ENTRY / 1e6
    return {
        "rows": T,
        "kv_mb": kv_mb,
        "base": base,
        "mine": t["flydsl"],
        "parts": ", ".join(f"{k} {v:.1f}" for k, v in parts.items()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", nargs="+", default=["dense", "source", "consumer"])
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384, 65536, 131072])
    ap.add_argument("--reqs", type=int, nargs="+", default=[3, 11, 32, 64, 128])
    ap.add_argument("--next-n", type=int, default=6)
    ap.add_argument("--spread", type=float, default=0.75)
    ap.add_argument("--inputs", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--consumers", type=int, default=4)
    ap.add_argument("--pool-gb", type=float, default=1.5)
    ap.add_argument(
        "--max-ctx",
        type=int,
        help="static bound of a row's entries (FlyDSL max_ctx, baseline logits "
        "width); default: --ctx",
    )
    args = ap.parse_args()
    if get_gfx_runtime() != "gfx950":
        print(f"needs gfx950, got {get_gfx_runtime()}")
        return
    gen = torch.Generator(device="cuda").manual_seed(1)
    print(f"top-k of the baseline: {'vLLM rule' if _VLLM_TOPK else 'aiter only'}")
    print("| mode | ctx | rows | KV MB | baseline us | flydsl us | speedup | parts |")
    print("|---|---|---|---|---|---|---|---|")
    for ctx in args.ctx:
        P = 128 if max(ctx, args.max_ctx or 0) > 65536 else 64
        pages = int(args.pool_gb * 2**30 / (P * ENTRY))
        pool = random_pool(pages, P, gen)
        for mode in args.mode:
            for reqs in args.reqs:
                r = bench(mode, ctx, reqs, args, pool, gen)
                if r is None:
                    print(f"| {mode} | {ctx} | {reqs * args.next_n} | pool too small |")
                    continue
                print(
                    f"| {mode} | {ctx} | {r['rows']} | {r['kv_mb']:.0f} | "
                    f"{r['base']:.1f} | {r['mine']:.1f} | "
                    f"{r['base'] / r['mine']:.2f}x | {r['parts']} |",
                    flush=True,
                )
                torch.cuda.empty_cache()
        del pool
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
