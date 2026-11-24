# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mega paged attention (PA + per-token FP8 + o_proj + TP all-reduce + Gemma
RMSNorm): correctness against a torch reference and per-launch timing.

Default shape: MiniMax-M3 dense attention at TP4 (64 q / 4 kv heads, head_dim
128 -> 16 query heads and one KV head a rank; hidden 6144), FP8 KV with
per-token scales, FP8 o_proj with per-channel scales.

    python3 op_tests/multigpu_tests/test_mega_paged_attention.py            # check + bench
    python3 op_tests/multigpu_tests/test_mega_paged_attention.py --rows 1 32 128 --trace
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

FP8 = (
    torch.float8_e4m3fn
    if get_gfx_runtime() in ("gfx950", "gfx1250")
    else torch.float8_e4m3fnuz
)
FP8_MAX = float(torch.finfo(FP8).max)


def _pertoken_fp8(x):
    """Per-token (last axis) FP8 quant: (fp8, fp32 scale with a kept last axis)."""
    amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax / FP8_MAX
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def make_inputs(args, rows, lengths, rank, device):
    """One rank's inputs. Residual, norm weight and lengths are rank-shared."""
    g = torch.Generator(device=device)
    dim, group, page, hidden = args.head_dim, args.group, args.block_size, args.hidden
    max_len = max(max(lengths), 1)
    pages_per_row = -(-max_len // page)
    num_pages = rows * pages_per_row + 7
    g.manual_seed(1000 + rank)
    table = (
        torch.randperm(num_pages, device=device, dtype=torch.int32, generator=g)[
            : rows * pages_per_row
        ]
        .reshape(rows, pages_per_row)
        .contiguous()
    )
    query = torch.empty((rows, group, dim), device=device, dtype=torch.bfloat16)
    query.uniform_(-1.0, 1.0, generator=g)
    key = torch.empty((num_pages, 1, page, dim), device=device).uniform_(
        -1, 1, generator=g
    )
    value = torch.empty_like(key).uniform_(-1, 1, generator=g)
    # Token magnitudes spanning a decade: a wrong per-token scale index fails.
    key.mul_(
        torch.randn(key[..., :1].shape, device=device, generator=g).mul_(0.4).exp_()
    )
    value.mul_(
        torch.randn(value[..., :1].shape, device=device, generator=g).mul_(0.4).exp_()
    )
    key_q, key_s = _pertoken_fp8(key)
    value_q, value_s = _pertoken_fp8(value)
    key_cache = (
        key_q.reshape(num_pages, 1, page, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value_q.reshape(num_pages, 1, page // 16, 16, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    k = group * dim
    w_full = torch.randn((hidden, k), device=device, generator=g) / k**0.5
    w_q, w_s = _pertoken_fp8(w_full)
    g.manual_seed(77)  # rank-shared
    residual = torch.randn((rows, hidden), device=device, generator=g).to(
        torch.bfloat16
    )
    norm_weight = (0.1 * torch.randn(hidden, device=device, generator=g)).to(
        torch.bfloat16
    )
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "key_scale": key_s.reshape(num_pages, 1, page).contiguous(),
        "value_scale": value_s.reshape(num_pages, 1, page).contiguous(),
        "block_tables": table,
        "context_lengths": context,
        "w": shuffle_weight(w_q, layout=(16, 16)).contiguous(),
        "w_scale": w_s.reshape(hidden).contiguous(),
        "residual": residual,
        "norm_weight": norm_weight,
        # reference-only
        "_key": key_q.float() * key_s,
        "_value": value_q.float() * value_s,
        "_w": w_q,
    }


def reference_attention(inp, args):
    """fp32 attention over the dequantized KV -> bf16 [rows, group * dim]."""
    q = inp["query"].float()
    rows, group, dim = q.shape
    page = args.block_size
    out = torch.zeros((rows, group, dim), device=q.device)
    for r, length in enumerate(inp["context_lengths"].tolist()):
        if length == 0:
            continue
        tok = torch.arange(length, device=q.device)
        pages = inp["block_tables"][r, tok // page].long()
        keys = inp["_key"][pages, 0, tok % page]
        values = inp["_value"][pages, 0, tok % page]
        p = torch.softmax(q[r] @ keys.T * args.softmax_scale, dim=-1)
        out[r] = p @ values
    return out.reshape(rows, group * dim).to(torch.bfloat16)


def reference_tail(x, x_scale, inp, world, eps):
    """o_proj + rank-order all-reduce + residual + Gemma RMSNorm from (x, x_scale)."""
    w = inp["_w"].float() * inp["w_scale"][:, None]
    y = ((x.float() * x_scale[:, None]) @ w.T).to(torch.bfloat16)
    parts = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(parts, y)
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    acc = acc.to(torch.bfloat16).float() + inp["residual"].float()
    residual_out = acc.to(torch.bfloat16)
    rstd = torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
    out = (acc * rstd * (1.0 + inp["norm_weight"].float())).to(torch.bfloat16)
    return out, residual_out


def _err(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), (d > 0.02 + 0.02 * b.float().abs()).float().mean().item()


def _call(mega, inp, out, res_out):
    return mega(
        inp["query"],
        inp["key_cache"],
        inp["value_cache"],
        inp["key_scale"],
        inp["value_scale"],
        inp["block_tables"],
        inp["context_lengths"],
        inp["w"],
        inp["w_scale"],
        inp["residual"],
        inp["norm_weight"],
        out=out,
        residual_out=res_out,
    )


def check(mega, args, rows, lengths, rank, world, device):
    inp = make_inputs(args, rows, lengths, rank, device)
    out = torch.full_like(inp["residual"], float("nan"))
    res_out = torch.full_like(inp["residual"], float("nan"))
    # Twice: the second launch runs the other parity of every counter.
    for _ in range(2):
        out.fill_(float("nan"))
        _call(mega, inp, out, res_out)
    torch.cuda.synchronize()
    x = mega.x[:rows].contiguous()
    x_scale = mega.x_scale[:rows].contiguous()
    attn = reference_attention(inp, args)
    x_deq = (x.float() * x_scale[:, None]).to(torch.bfloat16)
    tail_out, tail_res = reference_tail(x, x_scale, inp, world, args.eps)
    xr, xsr = _pertoken_fp8(attn)
    e2e_out, _ = reference_tail(xr, xsr.reshape(-1), inp, world, args.eps)
    stats = {
        "attn(x deq)": _err(x_deq, attn),
        "tail out": _err(out, tail_out),
        "tail res": _err(res_out, tail_res),
        "e2e out": _err(out, e2e_out),
    }
    finite = bool(torch.isfinite(out).all() and torch.isfinite(res_out).all())
    ok = (
        finite
        and stats["attn(x deq)"][1] < 1e-3
        and stats["tail res"][0] <= 0.0625
        and stats["tail out"][1] < 1e-3
        and stats["e2e out"][1] < 5e-3
    )
    return ok, finite, stats


def bench(mega, args, rows, lengths, rank, device, trace_mega=None):
    inp = make_inputs(args, rows, lengths, rank, device)
    out = torch.empty_like(inp["residual"])
    res_out = torch.empty_like(inp["residual"])
    for _ in range(3):
        _call(mega, inp, out, res_out)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _call(mega, inp, out, res_out)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.iters):
                _call(mega, inp, out, res_out)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / args.iters)
    del graph
    phases = None
    if trace_mega is not None:
        # Graph-replayed like the timing, so the ranks run in lockstep: the
        # stamps of the last launch, accumulators averaged over the launches.
        n = args.trace_iters
        for _ in range(3):
            _call(trace_mega, inp, out, res_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            for _ in range(n):
                _call(trace_mega, inp, out, res_out)
        torch.cuda.synchronize()
        trace_mega.trace.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        phases = summarize_trace(trace_mega.trace, n)
        if args.dump:
            torch.save(
                {"trace": trace_mega.trace.cpu(), "launches": n},
                f"{args.dump}/trace_r{rows}_rank{rank}.pt",
            )
        del graph
    return statistics.median(times), phases


def summarize_trace(trace, launches):
    """us from the last launch's start (min stamp 0 over CTAs) to each phase
    boundary (min/median/max over CTAs); per-launch averages of the
    accumulated o_proj ticks (max over CTAs)."""
    t = trace.double().cpu()
    t0 = t[:, 0].min()
    tick_us = 0.01  # s_memrealtime: 100 MHz

    def spread(i):
        v = (t[:, i] - t0) * tick_us
        return (v.min().item(), v.median().item(), v.max().item())

    def acc(i):
        return (t[:, i] / launches).max().item() * tick_us

    return {
        "start": spread(0),
        "plan": spread(14),
        "attn done": spread(1),
        "rows ready": spread(2),
        "o_proj done": spread(3),
        "AR+norm done": spread(4),
        "last pa task": (t[:, 5].max().item() * tick_us,),
        "last land": (t[:, 6].max().item() * tick_us,),
        "tasks/cta": (t[:, 7].max().item() / launches,),
        "tiles/cta": (t[:, 13].max().item() / launches,),
        "W+wait": (acc(8),),
        "gemm": (acc(9),),
        "reduce": (acc(10),),
        "push": (acc(11),),
    }


def worker(rank, world, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", str(args.port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    from aiter.ops.flydsl.mega_paged_attention import MegaPagedAttention

    def build(max_rows, trace=False):
        return MegaPagedAttention(
            rank=rank,
            world_size=world,
            device=device,
            head_dim=args.head_dim,
            query_group_size=args.group,
            block_size=args.block_size,
            max_rows=max_rows,
            max_context_length=args.max_context,
            hidden=args.hidden,
            kv_dtype=FP8,
            softmax_scale=args.softmax_scale,
            eps=args.eps,
            group=cpu_group,
            trace=trace,
            **args.tunables,
        )

    all_ok = True
    for rows in args.rows:
        max_rows = (
            max(16, -(-rows // 16) * 16) if args.max_rows is None else args.max_rows
        )
        mega = build(max_rows)
        g = torch.Generator().manual_seed(rows)
        if not args.no_check:
            # Ragged lengths with a graph-padding (empty) row and a 1-token row.
            lengths = torch.randint(
                1, args.max_context + 1, (rows,), generator=g
            ).tolist()
            if rows >= 3:
                lengths[1], lengths[2] = 0, 1
            ok, finite, stats = check(mega, args, rows, lengths, rank, world, device)
            oks = [None] * world
            dist.all_gather_object(oks, ok)
            all_ok &= all(oks)
            if rank == 0:
                print(
                    f"[check] rows={rows:4d} {'PASS' if all(oks) else 'FAIL'} "
                    f"finite={finite} "
                    + " ".join(
                        f"{k}: max {v[0]:.3g} bad {v[1]:.2g}" for k, v in stats.items()
                    ),
                    flush=True,
                )
        if not args.no_bench:
            lengths = [args.context] * rows
            trace_mega = build(max_rows, trace=True) if args.trace else None
            us, phases = bench(mega, args, rows, lengths, rank, device, trace_mega)
            all_us = [None] * world
            dist.all_gather_object(all_us, us)
            if rank == 0:
                print(
                    f"[bench] rows={rows:4d} ctx={args.context} max_rows={max_rows} "
                    f"us/launch: {max(all_us):8.2f}  "
                    f"(ranks {' '.join(f'{u:.2f}' for u in all_us)})",
                    flush=True,
                )
                if phases:
                    print(
                        "        "
                        + "; ".join(
                            f"{k}=" + "/".join(f"{x:.2f}" for x in v)
                            for k, v in phases.items()
                        ),
                        flush=True,
                    )
            if trace_mega is not None:
                trace_mega.close()
        mega.close()
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    if not all_ok:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--rows", type=int, nargs="*", default=[1, 16, 32, 64, 96, 128])
    p.add_argument(
        "--max-rows", type=int, default=None, help="default: rows rounded up to 16"
    )
    p.add_argument("--context", type=int, default=2048, help="bench context length")
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-iters", type=int, default=20)
    p.add_argument("--dump", type=str, default=None, help="dir for raw trace dumps")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--port", type=int, default=29761)
    p.add_argument(
        "--tune",
        type=str,
        nargs="*",
        default=[],
        help="MegaPagedAttention kwargs, e.g. block_n=96 max_tiles_per_task=8",
    )
    args = p.parse_args()
    args.softmax_scale = args.head_dim**-0.5
    args.tunables = {}
    for kv in args.tune:
        k, v = kv.split("=")
        args.tunables[k] = (
            None if v == "None" else (v == "True" if v in ("True", "False") else int(v))
        )
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        print(f"mega paged attention unsupported on {get_gfx_runtime()}; skipping")
        return
    if torch.cuda.device_count() < args.world:
        print(f"needs {args.world} GPUs; skipping")
        return
    mp.spawn(worker, args=(args.world, args), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
