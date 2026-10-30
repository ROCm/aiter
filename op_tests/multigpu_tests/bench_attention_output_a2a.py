# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Latency benchmark: quantized attention-output A2A vs RCCL + canonical quantizer.

Run from the repo root on an idle node (one process per GPU, spawned here):

    python3 op_tests/multigpu_tests/bench_attention_output_a2a.py -w 8 --shapes wan2.2

The baseline is the work the output hop replaces: the inverse Ulysses all-to-all of
BF16 MHA output over RCCL, followed by the canonical activation quantizer of the
consuming GEMM on the destination. The candidate is AttentionOutputA2AIntraNodeOp.
For e4m3_external_scale the candidate also includes the caller's scale policy
(per-destination amax and a world-size MAX all-reduce), since a caller must pay it.
Before timing, the candidate output is compared bit-exactly with the baseline.

The script has its own timer instead of run_perftest because a row must rotate
through several distinct inputs and report the slowest rank.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

CODECS = ("e4m3", "e4m3_external_scale", "mxfp4", "mxfp6")
GFX950_CODECS = {"e4m3_external_scale", "mxfp4", "mxfp6"}
# (global sequence, total heads, head dim) of the attention output.
_SHAPES = {
    "wan2.2": (75776, 40, 128),
    "small": (8192, 16, 128),
}
_ROTATIONS = 4


def _quantizer(codec, fp8_dtype, fp8_max):
    """Canonical destination quantizer: matrix -> (payload, scale)."""
    import torch

    import aiter

    if codec == "e4m3":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        return lambda matrix, scale: quantize(matrix, quant_dtype=aiter.dtypes.fp8)
    if codec == "e4m3_external_scale":

        def tensorwise(matrix, scale):
            if scale is None:
                scale = matrix.abs().amax().float().clamp(min=1e-12) / fp8_max
            payload = (matrix.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
            return payload, scale.view(1, 1)

        return tensorwise
    if codec == "mxfp4":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        return lambda matrix, scale: quantize(
            matrix, quant_dtype=aiter.dtypes.fp4x2, shuffle=True
        )
    from aiter.ops.gemm_op_a6w6 import mxfp6_gemm_pack_size, quant_mxfp6_gemm_out

    # The packer leaves layout padding untouched, so its persistent buffers start
    # zeroed like the op's symmetric storage.
    buffers = {}

    def mxfp6(matrix, scale):
        if matrix.shape not in buffers:
            payload_size, scale_size = mxfp6_gemm_pack_size(*matrix.shape)
            buffers[matrix.shape] = (
                torch.zeros(payload_size, dtype=torch.uint8, device=matrix.device),
                torch.zeros(scale_size, dtype=torch.uint8, device=matrix.device),
            )
        return quant_mxfp6_gemm_out(matrix, *buffers[matrix.shape])

    return mxfp6


def _inverse_a2a(value, world, group):
    import torch
    import torch.distributed as dist

    _, sequence, heads_local, head_dim = value.shape
    send = value.view(world, sequence // world, heads_local, head_dim)
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return received.permute(1, 0, 2, 3).reshape(
        sequence // world, world * heads_local * head_dim
    )


def _time(run, warmup, iterations, group):
    """Median microseconds of run(index), max across ranks."""
    import torch
    import torch.distributed as dist

    for index in range(warmup):
        run(index)
    torch.cuda.synchronize()
    dist.barrier(group)
    events = []
    for index in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(index)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    median = torch.tensor(
        [statistics.median(s.elapsed_time(e) * 1000.0 for s, e in events)],
        dtype=torch.float64,
    )
    dist.all_reduce(median, op=dist.ReduceOp.MAX, group=group)
    return median.item()


def _run_codec(codec, shape_name, inputs, args, ctx):
    import torch
    import torch.distributed as dist

    world, rank = ctx["world"], ctx["rank"]
    fp8_dtype, fp8_max = ctx["fp8_dtype"], ctx["fp8_max"]
    quantize = _quantizer(codec, fp8_dtype, fp8_max)
    op = ctx["op_cls"](
        rank=rank,
        world_size=world,
        shape=tuple(inputs[0].shape),
        codec=codec,
        block_num=args.blocks,
    )

    def destination_scales(value):
        amax = value.view(world, -1).abs().amax(dim=1).float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)
        return amax.clamp(min=1e-12) / fp8_max

    def baseline(index):
        return quantize(_inverse_a2a(inputs[index % _ROTATIONS], world, None), None)

    def candidate(index):
        value = inputs[index % _ROTATIONS]
        if codec == "e4m3_external_scale":
            return op(value, destination_scales=destination_scales(value))
        return op(value)

    # The baseline's tensorwise scale is local to the gathered matrix; it equals the
    # candidate's per-destination scale, so payloads must match byte for byte.
    result = candidate(0)
    expected = baseline(0)
    torch.cuda.synchronize()
    exact = torch.equal(
        result.payload.view(torch.uint8).flatten(),
        expected[0].view(torch.uint8).flatten(),
    )
    verdict = torch.tensor([int(exact)], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=ctx["cpu_group"])
    record = {"shape": shape_name, "codec": codec, "exact": bool(verdict.item())}
    if not record["exact"]:
        return record
    record["rccl+quant us"] = _time(baseline, args.warmup, args.iterations, None)
    record["output a2a us"] = _time(candidate, args.warmup, args.iterations, None)
    record["speedup"] = record["rccl+quant us"] / record["output a2a us"]
    return record


def _render(records, world, arch, elapsed_s):
    lines = [
        f"# Attention-output A2A: world={world}, arch={arch}, {elapsed_s:.0f} s",
        "",
        "| shape | codec | exact | RCCL + quant (us) | output A2A (us) | speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        if not r["exact"]:
            lines.append(f"| {r['shape']} | {r['codec']} | no | - | - | - |")
            continue
        lines.append(
            f"| {r['shape']} | {r['codec']} | yes | {r['rccl+quant us']:.1f} "
            f"| {r['output a2a us']:.1f} | {r['speedup']:.2f}x |"
        )
    return "\n".join(lines)


def _worker(rank, world, args):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")

    import functools

    import mori.shmem as ms
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl.attention_output_a2a_intranode import (
        AttentionOutputA2AIntraNodeOp,
    )

    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{args.master_port}",
        rank=rank,
        world_size=world,
        device_id=device,
    )
    start = time.perf_counter()
    try:
        cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        ms.shmem_torch_process_group_init("mori")
        gfx = get_gfx()
        ctx = {
            "world": world,
            "rank": rank,
            "device": device,
            "cpu_group": cpu_group,
            "op_cls": functools.partial(AttentionOutputA2AIntraNodeOp, group=cpu_group),
            "fp8_dtype": (
                torch.float8_e4m3fnuz if gfx == "gfx942" else torch.float8_e4m3fn
            ),
            "fp8_max": 240.0 if gfx == "gfx942" else 448.0,
        }
        records = []
        for shape_name in args.shapes:
            sequence, heads, head_dim = _SHAPES[shape_name]
            if heads % world or sequence % world:
                raise ValueError(f"{shape_name} does not divide world={world}")
            inputs = []
            for rotation in range(_ROTATIONS):
                torch.manual_seed(1000 * rotation + rank)
                inputs.append(
                    torch.randn(
                        1,
                        sequence,
                        heads // world,
                        head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
            for codec in args.codecs:
                if codec in GFX950_CODECS and gfx != "gfx950":
                    continue
                records.append(_run_codec(codec, shape_name, inputs, args, ctx))
            del inputs
        if rank == 0:
            report = _render(records, world, gfx, time.perf_counter() - start)
            print(report, flush=True)
            output = f"/tmp/bench_attention_output_a2a_w{world}.md"
            with open(output, "w", encoding="utf-8") as handle:
                handle.write(report + "\n")
            print(f"Wrote {output}", flush=True)
    finally:
        # Skip finalize while an exception unwinds: a peer may be mid-collective.
        if sys.exc_info()[0] is None:
            try:
                ms.shmem_finalize()
            finally:
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-w", "--world-size", type=int, required=True, choices=(2, 4, 8)
    )
    parser.add_argument(
        "--shapes", choices=tuple(_SHAPES), nargs="+", default=["wan2.2"]
    )
    parser.add_argument("--codecs", choices=CODECS, nargs="+", default=list(CODECS))
    parser.add_argument("--blocks", type=int, default=128, help="workgroups per launch")
    parser.add_argument("--master-port", type=int, default=29519)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations positive")

    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        print("SKIP: attention output A2A requires ROCm GPUs")
        return 0
    if torch.cuda.device_count() < args.world_size:
        print(
            f"SKIP: world={args.world_size}, only {torch.cuda.device_count()} HIP devices visible"
        )
        return 0
    mp.spawn(_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
