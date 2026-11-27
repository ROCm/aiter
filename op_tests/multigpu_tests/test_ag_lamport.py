# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of the gfx1250 Lamport one-shot all-gather on the production path.

custom_all_gather, which tensor_model_parallel_all_gather(use_custom=True) and
CudaCommunicator.all_gather (ATOM's dcp_all_gather) both reach, takes the
Lamport kernel for a gather along dim 0 or the last dim, of whole 16B packs a
row, up to the Lamport size cap (AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE); see
CustomAllreduce.should_lamport_ag. Timing, and the kernel each case runs:
op_tests/multigpu_tests/gfx1250_poc/test_gfx1250_allgather.py.

Every gather is compared bit for bit with the concatenated inputs, except that
on the Lamport kernel a 0xffff half-word of an fp16/bf16 input comes out 0x7fff
(both NaN). The inputs hold the bits that collide with the inbox sentinels: an
all-ones pack, a 0xfffffffe word (a 0xffff half-word, not all ones), an
all-ones element; ints are token-id like with -1 padding.

  cases  --shape x --dtype x --dim, then edge cases: the Lamport cap and one row
         past it, a partial last wave, one pack, rows that are not whole 16B
         packs, a middle dim, 1- and 8-byte dtypes, model shapes, and the odd
         ranks' input one element into its storage (every rank must still take
         the Lamport kernel). Each through both entries, eager and in a graph
         replayed with fresh inputs.
  mixed  gathers of both codecs interleaved with the Lamport AR + add and
         AR + RMSNorm on the one shared inbox: all -1 packs followed rounds
         later by none on the same packs (a stale escape would show), an input
         aliasing its own output slice or another one, graph replays.
"""

import argparse
import itertools
from multiprocessing import Pool, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce_add,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime

SUPPORTED_GFX = ["gfx1250"]
REPLAYS = 3
# dtypes of the 0xffff half-word sentinel codec; the others go bit for bit
HALF_SENTINEL_DTYPES = (torch.float16, torch.bfloat16)
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# (shape, dtype, dim, misaligned, takes Lamport), run after --shape x --dtype x
# --dim; the cap cases are added per rank from the communicator's cap
EDGE_CASES = [
    ((1, 8), dtypes.bf16, 0, False, True),  # one pack
    ((3, 1000), dtypes.fp16, 0, False, True),  # the last wave partial
    ((64, 8), dtypes.bf16, -1, False, True),  # one pack a row
    ((12, 4), dtypes.bf16, 0, False, True),
    ((12, 4), dtypes.bf16, -1, False, False),  # 8B rows: RCCL
    ((1345,), dtypes.bf16, 0, False, False),  # not whole packs: older kernels
    ((4, 8, 32), dtypes.bf16, 1, False, False),  # a middle dim: RCCL
    ((4, 64, 32), dtypes.u8, 0, False, True),
    ((64, 7168), dtypes.fp8, -1, False, True),
    ((4, 8, 32), dtypes.i64, 0, False, True),
    # DeepSeek-V4 LM-head shard and MiniMax-M3 logits at TP4, Kimi-K3 DCP8 fp8
    # KV rows paired into fp16 as ATOM's dcp_all_gather sends them
    ((65, 32320), dtypes.bf16, -1, False, True),
    ((12, 50016), dtypes.bf16, -1, False, True),
    ((2048, 288), dtypes.fp16, -1, False, True),
    # the odd ranks' input one element into its storage
    ((16,), dtypes.bf16, 0, True, True),
    ((12, 50016), dtypes.bf16, -1, True, True),
]
# (shape, dtype, dim, dense -1) of the mixed sequence: (3, 1000) and (300, 1000)
# leave the last wave partial, (1, 8) is one pack; int16 is 2 bytes but must
# take the escape codec (its -1 is 0xffff)
MIXED_AG = [
    ((4, 64, 32), torch.int32, 0, True),
    ((409, 7168), torch.bfloat16, 0, False),
    ((4, 8, 32), torch.int32, -1, False),
    ((3, 1000), torch.float16, 0, False),
    ((48, 7168), torch.bfloat16, -1, False),
    ((4, 64, 32), torch.int16, 0, True),
    ((1, 8), torch.bfloat16, 0, False),
    ((300, 1000), torch.float32, -1, False),
]
MIXED_ADD = [(8, 5120), (3, 1000)]
MIXED_RMS = [(4, 7168), (300, 4096)]
GRAPH_AG = [
    ((4, 128, 32), torch.int32, 0),
    ((7, 1000), torch.bfloat16, -1),
    ((288, 7168), torch.bfloat16, -1),
    ((4, 8, 32), torch.int16, 0),
]


def make_inputs(tp, rank, seed, shape, dtype, dim, dense=False):
    """This rank's input, the gathered inputs and what the Lamport kernel
    returns (the gathered inputs with 0xffff -> 0x7fff for fp16/bf16)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if dtype.is_floating_point:
        xs = torch.randn((tp, *shape), device="cuda", generator=g).to(dtype)
    else:
        # token ids with -1 padding; dense: mostly -1
        lo = -300 if dense else -1
        xs = torch.randint(lo, 384, (tp, *shape), device="cuda", generator=g)
        xs = xs.clamp_(min=-1).to(dtype)
    raw = xs.view(tp, -1).view(torch.uint8)
    raw[:, :16] = 0xFF  # an all-ones pack
    tail = raw[:, 16:20]  # 0xfffffffe: a 0xffff half-word, not all ones
    pattern = torch.tensor([0xFE, 0xFF, 0xFF, 0xFF], dtype=torch.uint8, device="cuda")
    tail.copy_(pattern[: tail.shape[1]].expand_as(tail))
    # an all-ones element mid-tensor on the last rank
    ones = 0xFF if xs.element_size() == 1 else -1
    xs[tp - 1].view(BITS[xs.element_size()]).view(-1)[xs[0].numel() // 3] = ones
    ref = torch.cat(xs.unbind(0), dim=dim)
    want = ref.clone()
    if dtype in HALF_SENTINEL_DTYPES:
        bits = want.view(torch.int16)
        bits[bits == -1] = 0x7FFF
    return xs[rank].contiguous(), ref, want


def bit_mismatch(a, b):
    """Elements whose bits differ."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return a.numel()
    t = BITS[a.element_size()]
    return int((a.view(t) != b.view(t)).sum().item())


def gather_tp(x, dim):
    """GroupCoordinator.all_gather, custom kernels allowed."""
    return tensor_model_parallel_all_gather(x, use_custom=True, dim=dim)


def gather_dc(x, dim):
    """CudaCommunicator.all_gather, as ATOM's dcp_all_gather calls it."""
    return get_tp_group().device_communicator.all_gather(x, dim)


ENTRIES = {"tp": gather_tp, "dc": gather_dc}


def cap_cases(cap):
    """The Lamport cap, along both dims, and one row past it, for each codec."""
    cases = []
    for dtype in (dtypes.bf16, dtypes.i32):
        rows = cap // (8192 * dtype.itemsize)
        cases += [
            ((rows, 8192), dtype, 0, False, True),
            ((rows, 8192), dtype, -1, False, True),
            ((rows + 1, 8192), dtype, 0, False, False),
        ]
    return cases


def check_case(ca, tp, rank, shape, dtype, dim, misaligned, lamport_expected):
    """One case through both entries, eager and replayed in a graph with fresh
    inputs, against the reference bit for bit."""
    seed = sum(shape) * 1000 + len(shape) * 10 + dim + 2
    x, ref, want = make_inputs(tp, rank, seed, shape, dtype, dim)
    if misaligned and rank % 2:
        storage = torch.empty(x.numel() + 1, dtype=dtype, device="cuda")
        x = storage[1:].view(x.shape).copy_(x)
    lamport = ca.should_lamport_ag(x, dim)
    worst = 0
    for fn in ENTRIES.values():
        worst = max(worst, bit_mismatch(fn(x, dim), want if lamport else ref))

    # CudaCommunicator falls back to pynccl, which graph_capture() does not set
    # up for capture: graph it only where it takes the Lamport kernel
    graphed = ENTRIES if lamport else {"tp": gather_tp}
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        for fn in graphed.values():
            fn(x, dim)  # warmup, as before a model's capture
        with torch.cuda.graph(graph, stream=gc.stream):
            outs = [fn(x, dim) for fn in graphed.values()]
    for rep in range(REPLAYS):
        nx, nref, nwant = make_inputs(
            tp, rank, seed + 1 + rep, shape, dtype, dim, dense=rep % 2 == 0
        )
        x.copy_(nx)
        graph.replay()
        torch.cuda.synchronize()
        for out in outs:
            worst = max(worst, bit_mismatch(out, nwant if lamport else nref))
    graph.reset()
    return {
        "tp": tp,
        "shape": str(tuple(shape)),
        "dtype": str(dtype)[6:],
        "dim": dim,
        "KB/rank": x.numel() * x.element_size() / 1024,
        "misaligned": misaligned,
        "lamport": lamport,
        "route ok": lamport == lamport_expected,
        "bitdiff": worst,
    }


def check_mixed_sequence(ca, tp, rank):
    """Gathers of both codecs, the AR + add and the AR + RMSNorm interleaved
    on the one inbox, eager and in a graph, against the reference bit for bit."""
    worst, calls = 0, 0
    g = torch.Generator(device="cuda").manual_seed(1234)
    bf16 = torch.bfloat16

    def rms(m, n):
        x = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        res = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        w = torch.randn((n,), device="cuda", generator=g).to(bf16)
        assert ca.should_lamport_ar_rms(x, res, w, n), (m, n)
        tensor_model_parallel_fused_allreduce_rmsnorm(x, res, w, 1e-6)

    def add(m, n):
        x = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        assert ca.should_lamport_ar_add(x, x), (m, n)
        tensor_model_parallel_all_reduce_add(x, x)

    def ag(seed, shape, dtype, dim, dense=False):
        nonlocal worst, calls
        x, _, want = make_inputs(tp, rank, seed, shape, dtype, dim, dense)
        assert ca.should_lamport_ag(x, dim), (shape, dim)
        worst = max(worst, bit_mismatch(gather_tp(x, dim), want))
        calls += 1

    for i, case in enumerate(MIXED_AG * 3):
        ag(i, *case)
        if i % 2:
            rms(*MIXED_RMS[i % len(MIXED_RMS)])
        if i % 3 == 0:
            add(*MIXED_ADD[i % len(MIXED_ADD)])

    # all -1 on some packs, then rounds later no -1 on the same packs
    for rep in range(4):
        ag(100 + rep, (4, 128, 32), torch.int32, 0, dense=True)
        for _ in range(rep):
            add(*MIXED_ADD[0])
        sevens = torch.full((4, 128, 32), 7, dtype=torch.int32, device="cuda")
        out = gather_tp(sevens, 0)
        worst = max(worst, int((out != 7).sum().item()))
        calls += 1

    # a dim-0 gather whose input is a slice of its output: its own slice, read
    # in place, or the next rank's, copied into the registered pool first
    rows = 64
    for k, slot in enumerate((rank, (rank + 1) % tp)):
        x, ref, _ = make_inputs(tp, rank, 77 + k, (rows, 32), torch.int32, 0)
        out = torch.full_like(ref, 0x5A5A5A5A)
        own = out[slot * rows : (slot + 1) * rows]
        own.copy_(x)
        ca.lamport_all_gather(own, 0, out=out)
        worst = max(worst, bit_mismatch(out, ref))
        calls += 1

    ins = [make_inputs(tp, rank, 0, s, dt, d)[0] for s, dt, d in GRAPH_AG]
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        outs = [gather_tp(x, d) for x, (_, _, d) in zip(ins, GRAPH_AG)]
    for rep in range(6):
        refs = []
        for j, ((s, dt, d), x) in enumerate(zip(GRAPH_AG, ins)):
            nx, _, want = make_inputs(
                tp, rank, 500 + 10 * rep + j, s, dt, d, dense=rep % 2 == 0
            )
            x.copy_(nx)
            refs.append(want)
        graph.replay()
        torch.cuda.synchronize()
        for o, ref in zip(outs, refs):
            worst = max(worst, bit_mismatch(o, ref))
            calls += 1
        ag(900 + rep, *MIXED_AG[rep % len(MIXED_AG)])
        rms(*MIXED_RMS[rep % len(MIXED_RMS)])
        add(*MIXED_ADD[rep % len(MIXED_ADD)])
    graph.reset()
    return {"tp": tp, "calls": calls, "max bitdiff": worst}


def worker(tp, rank, cases, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    assert ca._lamport_max_bytes > 0, "the Lamport inbox is off"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)

    cases = cases + cap_cases(ca._lamport_max_bytes)
    rows = [check_case(ca, tp, rank, *c) for c in cases]
    mixed = check_mixed_sequence(ca, tp, rank)

    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows, mixed


def merge_ranks(per_rank):
    """The worst mismatch of any rank, case by case; the route must agree."""
    merged = []
    for cells in zip(*per_rank):
        row = dict(cells[0])
        for k in ("bitdiff", "max bitdiff"):
            if k in row:
                row[k] = max(c[k] for c in cells)
        if "route ok" in row:
            row["route ok"] = all(
                c["route ok"] and c["lamport"] == row["lamport"] for c in cells
            )
        merged.append(row)
    return merged


def run_tp(tp, cases):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, cases, init_method)) for r in range(tp)
        ]
        # a protocol bug hangs instead of failing
        results = [j.get(timeout=1800) for j in jobs]
    rows = merge_ranks([r[0] for r in results])
    mixed = merge_ranks([[r[1]] for r in results])[0]
    return rows, mixed


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lamport all-gather unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Correctness of the Lamport all-gather on the production path",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # [4, BS, 32] a rank at BS (concurrency) 8-128, then hidden rows
        default=[(4, bs, 32) for bs in (8, 16, 32, 64, 128)]
        + [(1, 7168), (64, 7168), (256, 7168)],
        help="per-rank input shape, e.g. -s 4,8,32 64,7168",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.i32, dtypes.bf16, dtypes.fp16, dtypes.i16, dtypes.fp32],
    )
    parser.add_argument("--dim", type=int, nargs="*", default=[0, -1])
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()

    cases = [
        (shape, dtype, dim, False, True)
        for dtype, dim, shape in itertools.product(args.dtype, args.dim, args.shape)
    ] + EDGE_CASES
    set_start_method("spawn", force=True)
    rows, mixed = [], []
    for tp in args.tp:
        if tp > ngpus:
            aiter.logger.warning("tp%d needs %d GPUs, have %d", tp, tp, ngpus)
            continue
        tp_rows, tp_mixed = run_tp(tp, cases)
        rows += tp_rows
        mixed.append(tp_mixed)
    aiter.logger.info(
        "lamport all-gather cases (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"),
    )
    aiter.logger.info(
        "lamport all-gather mixed sequence (markdown):\n%s",
        pd.DataFrame(mixed).to_markdown(index=False),
    )
    bad = [r for r in rows if r["bitdiff"] or not r["route ok"]]
    bad += [m for m in mixed if m["max bitdiff"]]
    assert not bad, f"{len(bad)} failing cases:\n{pd.DataFrame(bad)}"


if __name__ == "__main__":
    main()
