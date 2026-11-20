# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-replay timing of flydsl_sparse_mla_decode_fp8_h64 against
sparse_mla_fwd (aiter Triton, gfx950 fp8 kernel with the inverse-RoPE MXFP8
store), the same call: 64 heads x 512 dims, e4m3 q and caches with per-tensor
scales, a 128-key sliding window + top-512 list per row, an attention sink.

One CUDA graph holds ``--copies`` (100) launches, each with its own queries
and key lists over two 1 GiB pools (the 256 MB MALL cannot hold a launch's
rows twice), so K / V are cold. A step has requests of 6 verify rows (the
last request takes the rest of a row count that is no multiple of 6): the
window is the 128 slots before the row's position; the top-512 list shares
80% of its rows with the request's other rows and the rest is fresh. Per row
count the implementations are timed A B B A twice (triton, flydsl, flydsl,
triton, ...); a graph is replayed 10 times per round, a pass keeps the minimum
over 15 rounds; the table reports the mean of the passes (us per launch) and
the requested KV gather rate (640 keys x 512 B per row). The first and the
last launch are checked against the Triton kernel's dequantized output.

``--core`` times the attention core alone against aiter's HIP
``sparse_mla_fp8_fwd`` (64 heads x 512, per-tensor e4m3 KV, bf16 output),
which has no sink, no second list and no MXFP8 epilogue: the same 640 key ids
of a row as one list over one 1 GiB pool, bf16 rows out, a zero sink for
FlyDSL (which requires one).

    python3 op_tests/op_benchmarks/flydsl/bench_flydsl_sparse_mla_decode_fp8_h64.py \
        [--rows 16 192 768] [--p-terms 2] [--core]
"""

import argparse
import functools

import pandas as pd
import torch

from aiter import sparse_mla_fp8_fwd
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import flydsl_sparse_mla_decode_fp8_h64
from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

H, D = 64, 512
POOL_ROWS = 1 << 21  # 1 GiB of 512-byte rows
SWA_BLOCK, COMP_BLOCK = 32, 64
WINDOW, TOPK, SHARE = 128, 512, 0.8
SOFTMAX_SCALE = D**-0.5
F8 = torch.float8_e4m3fn


def e4m3(shape, std, gen):
    out = torch.empty(shape, dtype=F8, device="cuda")
    flat = out.view(torch.uint8).view(-1)
    for i in range(0, flat.numel(), 1 << 26):
        n = min(1 << 26, flat.numel() - i)
        x = torch.randn(n, generator=gen, device="cuda") * std
        flat[i : i + n] = x.to(F8).view(torch.uint8)
    return out


def make_launch(rows, gen, ctx=(8192, 131072)):
    """The arguments of one launch: ``rows`` query rows in requests of 6."""
    lens = [6] * (rows // 6) + ([rows % 6] if rows % 6 else [])
    B = len(lens)
    seed = int(torch.randint(0, 1 << 30, (1,), generator=gen, device="cuda"))
    cpu = torch.Generator().manual_seed(seed)
    ctxs = torch.randint(ctx[0], ctx[1], (B,), generator=cpu)
    req = torch.repeat_interleave(torch.arange(B), torch.tensor(lens))
    j = torch.cat([torch.arange(n) for n in lens])
    pos = (ctxs[req] + j).cuda()
    # window: the slots of the 128 positions up to the row's, per-request region
    region = torch.randint(0, POOL_ROWS // 256, (B,), generator=cpu)[req].cuda() * 256
    k = torch.arange(WINDOW, device="cuda")
    p_k = pos[:, None] - (WINDOW - 1) + k[None, :]
    keep = p_k >= 0
    main = (region[:, None] + p_k % 256)[keep]
    main_n = keep.sum(1)
    # top-k: the request's base set, 20% of every row replaced
    ncomp = (ctxs // 2).clamp_min(TOPK)
    base = torch.stack(
        [torch.randint(0, int(n), (TOPK,), generator=cpu) for n in ncomp]
    )
    base = base.cuda()[req.cuda()]
    fresh = (
        torch.rand((rows, TOPK), generator=gen, device="cuda") * (pos[:, None] // 2)
    ).long()
    shared = torch.rand((rows, TOPK), generator=gen, device="cuda") < SHARE
    sel = torch.where(shared, base, fresh)
    cbase = torch.randint(0, POOL_ROWS, (B,), generator=cpu)[req].cuda()
    extra = ((cbase[:, None] + sel) % POOL_ROWS).flatten()

    def indptr(n):
        zero = torch.zeros(1, dtype=torch.int64, device="cuda")
        return torch.cat([zero, n.cumsum(0)]).int()

    ang = torch.rand((int(pos.max()) + 8, 32), generator=gen, device="cuda") * 6.2832
    return {
        "q": e4m3((rows, H, D), 1.0, gen),
        "kv_indptr": indptr(main_n),
        "kv_indices": main.int(),
        "extra_indptr": (torch.arange(rows + 1, device="cuda") * TOPK).int(),
        "extra_indices": extra.int(),
        "inv_rope_positions": pos.int(),
        "inv_rope_cos_sin_cache": torch.cat([ang.cos(), ang.sin()], dim=1).float(),
    }


def run_flydsl(L, pools, out, p_terms=2):
    swa, comp, sink, qs, ks = pools
    flydsl_sparse_mla_decode_fp8_h64(
        L["q"],
        swa,
        L["kv_indptr"],
        L["kv_indices"],
        SOFTMAX_SCALE,
        kv_scale=ks,
        q_scale=qs,
        attn_sink=sink,
        extra_kv=comp,
        extra_indptr=L["extra_indptr"],
        extra_indices=L["extra_indices"],
        inv_rope_positions=L["inv_rope_positions"],
        inv_rope_cos_sin_cache=L["inv_rope_cos_sin_cache"],
        out_mxfp8=out,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        max_keys_per_row=WINDOW + TOPK,
        p_terms=p_terms,
    )


def run_triton(L, pools, out):
    swa, comp, sink, qs, ks = pools
    sparse_mla_fwd(
        L["q"],
        swa,
        L["kv_indptr"],
        L["kv_indices"],
        SOFTMAX_SCALE,
        kv_scale=ks,
        q_scale=qs,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        attn_sink=sink,
        extra_kv=comp,
        extra_indptr=L["extra_indptr"],
        extra_indices=L["extra_indices"],
        has_invalid=False,
        inv_rope_positions=L["inv_rope_positions"],
        inv_rope_cos_sin_cache=L["inv_rope_cos_sin_cache"],
        out_mxfp8=out,
    )


def core_launch(L):
    """The launch's 640 key ids of a row (window, then top-k) as one list."""
    rows = L["q"].shape[0]
    main = L["kv_indices"].view(rows, WINDOW).long()
    # both lists' slot ids are rows of one flat 1 GiB pool here
    ids = torch.cat([main, L["extra_indices"].view(rows, TOPK).long()], dim=1)
    return {
        "q": L["q"],
        "ids": ids.int().contiguous(),
        "indptr": (torch.arange(rows + 1, device="cuda") * (WINDOW + TOPK)).int(),
    }


def run_flydsl_core(C, pools, out, p_terms=2):
    flat, sink, qs, ks = pools
    flydsl_sparse_mla_decode_fp8_h64(
        C["q"],
        flat,
        C["indptr"],
        C["ids"].view(-1),
        SOFTMAX_SCALE,
        kv_scale=ks,
        q_scale=qs,
        attn_sink=sink,
        out=out,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        max_keys_per_row=WINDOW + TOPK,
        p_terms=p_terms,
    )


def run_hip_core(C, pools, out):
    flat, _, _, _ = pools
    sparse_mla_fp8_fwd(
        C["q"],
        flat,
        C["ids"],
        SOFTMAX_SCALE,
        q_scale=1.0,
        kv_scale=1.0,
        out=out,
        workspace=C["workspace"],
    )


def core_table(args, gen):
    """Attention core alone: FlyDSL (bf16 out) against HIP sparse_mla_fp8_fwd."""
    from aiter import allocate_sparse_mla_fp8_workspace

    flat = e4m3((POOL_ROWS, D), 1.0, gen)
    one = torch.ones(1, device="cuda")
    pools = (flat, torch.zeros(H, device="cuda"), one, one)
    names = ["hip"] + [f"flydsl_p{p}" for p in args.p_terms]
    table = []
    for rows in args.rows:
        launches = []
        for _ in range(args.copies):
            C = core_launch(make_launch(rows, gen))
            C["workspace"] = allocate_sparse_mla_fp8_workspace(C["q"])
            launches.append(C)
        outs = {
            n: torch.empty(rows, H, D, dtype=torch.bfloat16, device="cuda")
            for n in names
        }
        fns = {"hip": run_hip_core}
        for p in args.p_terms:
            fns[f"flydsl_p{p}"] = functools.partial(run_flydsl_core, p_terms=p)
        for i in (0, len(launches) - 1):
            for n in names:
                fns[n](launches[i], pools, outs[n])
            torch.cuda.synchronize()
            for n in names[1:]:
                # the FlyDSL rows carry the zero sink (exp(0) = 1 in the sum)
                cos = torch.nn.functional.cosine_similarity(
                    outs[n].double().flatten(1), outs["hip"].double().flatten(1)
                )
                assert cos.min().item() >= 0.99, ("vs hip", n, rows, cos.min().item())
        graphs = {n: graph_of(fns[n], launches, pools, outs[n]) for n in names}
        times = {n: [] for n in names}
        order = names + names[::-1]
        for name in order + order:
            times[name].append(replay_us(graphs[name], args.copies))
        row = {"gfx": get_gfx_runtime(), "rows": rows}
        for n in names:
            us = sum(times[n]) / len(times[n])
            row[f"{n} us"] = round(us, 1)
            row[f"{n} spread"] = f"{min(times[n]):.1f}-{max(times[n]):.1f}"
        table.append(row)
        print(row, flush=True)
        del launches, graphs
    return table


def graph_of(fn, launches, pools, out):
    for L in launches[:2]:
        fn(L, pools, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for L in launches:
            fn(L, pools, out)
    torch.cuda.synchronize()
    return graph


def replay_us(graph, copies, replays=10, rounds=15):
    for _ in range(3):
        graph.replay()
    best = float("inf")
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / (replays * copies))
    return best


def dequant(codes, scales):
    v = codes.view(F8).double().view(codes.shape[0], -1, 32)
    return (v * torch.exp2(scales.double() - 127.0)[..., None]).flatten(1)


def check(rows, launches, pools, out_a, out_b, flydsl):
    for i in (0, len(launches) - 1):
        flydsl(launches[i], pools, out_a)
        run_triton(launches[i], pools, out_b)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            dequant(*out_a), dequant(*out_b), dim=1
        )
        assert cos.min().item() >= 0.998, ("vs triton", rows, i, cos.min().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rows",
        nargs="*",
        type=int,
        default=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768],
    )
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument(
        "--core",
        action="store_true",
        help="time the attention core against HIP sparse_mla_fp8_fwd instead",
    )
    parser.add_argument(
        "--p-terms",
        nargs="*",
        type=int,
        default=[2, 1],
        help="2: P as hi + residual e4m3 (default), 1: one e4m3 P",
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        print(f"gfx950 only, got {get_gfx_runtime()}")
        return
    gen = torch.Generator(device="cuda").manual_seed(0)
    if args.core:
        with torch.inference_mode():
            print(pd.DataFrame(core_table(args, gen)).to_markdown(index=False))
        return
    swa = e4m3((POOL_ROWS // SWA_BLOCK, SWA_BLOCK, D), 1.0, gen)
    comp = e4m3((POOL_ROWS // COMP_BLOCK, COMP_BLOCK, D), 1.0, gen)
    sink = torch.randn(H, generator=gen, device="cuda")
    one = torch.ones(1, device="cuda")
    pools = (swa, comp, sink, one, one)
    names = ["triton"] + [f"flydsl_p{p}" for p in args.p_terms]
    table = []
    with torch.inference_mode():
        for rows in args.rows:
            launches = [make_launch(rows, gen) for _ in range(args.copies)]
            outs = {
                n: (
                    torch.empty(rows, H * D, dtype=F8, device="cuda"),
                    torch.empty(rows, H * D // 32, dtype=torch.uint8, device="cuda"),
                )
                for n in names
            }
            fns = {"triton": run_triton}
            for p in args.p_terms:
                fns[f"flydsl_p{p}"] = functools.partial(run_flydsl, p_terms=p)
                check(
                    rows,
                    launches,
                    pools,
                    outs[f"flydsl_p{p}"],
                    outs["triton"],
                    fns[f"flydsl_p{p}"],
                )
            graphs = {n: graph_of(fns[n], launches, pools, outs[n]) for n in names}
            times = {n: [] for n in names}
            order = names + names[::-1]
            for name in order + order:
                times[name].append(replay_us(graphs[name], args.copies))
            row = {"gfx": get_gfx_runtime(), "rows": rows}
            gather = rows * (WINDOW + TOPK) * D
            for n in names:
                us = sum(times[n]) / len(times[n])
                row[f"{n} us"] = round(us, 1)
                row[f"{n} spread"] = f"{min(times[n]):.1f}-{max(times[n]):.1f}"
                row[f"{n} gather TB/s"] = round(gather / us / 1e6, 2)
            table.append(row)
            print(row, flush=True)
            del launches, graphs
    print(pd.DataFrame(table).to_markdown(index=False))


if __name__ == "__main__":
    main()
