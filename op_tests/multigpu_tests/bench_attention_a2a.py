# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Latency benchmark: fused attention A2A vs RCCL all_to_all_4d (+ MHA V4 packers).

Run from the repo root on an idle node (one process per GPU, spawned here):

    python3 op_tests/multigpu_tests/bench_attention_a2a.py -w 8 \
        --shapes wan2.2 ticket --recipes all

Each recipe times the fused AttentionA2AIntraNodeOp trio (Q, K, V submit_role)
against the yunchang-style all_to_all_4D in-hop reproduction over RCCL; packed
recipes add the exact MHA V4 packers after the exchange. Medians are taken per
role and per trio, and the slowest rank is reported. Before any timing, the
fused output is compared bit-exactly with the baseline; a mismatch skips timing.

This script has its own timer instead of run_perftest because a row must (a)
rotate through several distinct inputs so no iteration re-reads a warm cache,
and (b) report the max across ranks, since a collective finishes at the speed
of its slowest participant. run_perftest does neither.

Baseline provenance: every packer is a public aiter.ops.mha_v4_quant op except
MXFP4 V. The public DEFAULT-layout MXFP4 V packer was removed, so that baseline
uses the private HIP producer (_mxfp4_v_buffers + _quantize_v_mxfp4_hip), the
same path test_attention_a2a.py references.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass

# Requested global shapes (B, S, H, D); S is padded up to a multiple of world * 32.
_SHAPES = {
    "wan2.2": (1, 9419, 40, 128),
    "ticket": (1, 14867, 24, 128),
}
_ROTATIONS = 4  # distinct input sets cycled per iteration to defeat cache reuse
_BF16_BYTES = 2


@dataclass(frozen=True)
class Recipe:
    name: str
    quant: tuple[str, str] | None
    v_fp6_p: bool
    hadamard: bool
    # Packer names per role (Q, K, V); None for the unpacked bf16 recipe.
    packers: tuple[str, str, str] | None


# Hadamard follows the fused op: on for every packed recipe except INT8, which
# the op documents as unrotated. Q/K rotation lives inside the rotated packers.
_RECIPES = (
    Recipe("bf16", None, False, False, None),
    Recipe(
        "mxfp4_qk_mxfp4_v",
        ("mxfp4", "mxfp4"),
        False,
        True,
        ("mxfp4_q", "mxfp4_k", "mxfp4_v"),
    ),
    Recipe(
        "mxfp4_qk_mxfp6p_v",
        ("mxfp4", "mxfp6_p"),
        True,
        True,
        ("mxfp4_q", "mxfp4_k", "mxfp6p_v"),
    ),
    Recipe(
        "e4m3_qkv", ("e4m3", "e4m3"), False, True, ("fp8_rotated", "fp8_rotated", "fp8")
    ),
    Recipe("int8_qk_e4m3_v", ("int8", "e4m3"), False, False, ("int8", "int8", "fp8")),
)


def all_to_all_4d(input_tensor, group):
    """Yunchang all_to_all_4D in-hop body, scatter_idx=2 / gather_idx=1."""
    import torch
    import torch.distributed as dist

    batch, shard_sequence, heads, head_dim = input_tensor.shape
    world_size = dist.get_world_size(group)
    heads_local = heads // world_size
    input_t = (
        input_tensor.reshape(batch, shard_sequence, world_size, heads_local, head_dim)
        .transpose(0, 2)
        .contiguous()
    )
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    return (
        output.reshape(shard_sequence * world_size, batch, heads_local, head_dim)
        .transpose(0, 1)
        .contiguous()
    )


def _local_shape(shape, world):
    batch, sequence, heads, head_dim = shape
    alignment = world * 32
    sequence = (sequence + alignment - 1) // alignment * alignment
    return batch, sequence // world, heads, head_dim


def _make_input(rank, shape, salt, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(952_000 + rank * 101 + salt)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        .contiguous()
    )


def _build_packers(softmax_scale):
    from aiter.ops.mha_v4_quant import (
        _mxfp4_v_buffers,
        _quantize_v_mxfp4_hip,
        mha_v4_q_multiplier,
        quantize_fp8,
        quantize_fp8_rotated,
        quantize_int8,
        quantize_mxfp4_k,
        quantize_mxfp4_q,
        quantize_v_mxfp6_fp6_p,
    )

    def mxfp4_v(value):
        # The one baseline with no public packer; see the module docstring.
        batch, sequence, heads, _ = value.shape
        raw, scale = _mxfp4_v_buffers(value, batch, sequence, heads)
        _quantize_v_mxfp4_hip(raw, scale, value)
        return raw, scale

    multiplier = mha_v4_q_multiplier(softmax_scale)
    return {
        "mxfp4_q": lambda value: quantize_mxfp4_q(value, multiplier),
        "mxfp4_k": quantize_mxfp4_k,
        "mxfp4_v": mxfp4_v,
        "mxfp6p_v": quantize_v_mxfp6_fp6_p,
        "fp8_rotated": quantize_fp8_rotated,
        "fp8": quantize_fp8,
        "int8": quantize_int8,
    }


def _time_roles(run_trio, run_role, warmup, iterations, group):
    """Median microseconds per role (0, 1, 2) and for the whole trio."""
    import torch
    import torch.distributed as dist

    for index in range(warmup):
        run_trio(index)
    torch.cuda.synchronize()
    dist.barrier(group)

    medians = {}
    for role in (0, 1, 2, "trio"):
        events = []
        for index in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if role == "trio":
                start.record()
                run_trio(index)
                end.record()
            else:
                # run_role records start/end around one role of a full trio so
                # the surrounding roles keep the stream state realistic.
                run_role(role, index, start, end)
            events.append((start, end))
        torch.cuda.synchronize()
        medians[role] = statistics.median(s.elapsed_time(e) * 1000.0 for s, e in events)
    return medians


def _assert_equal(actual, expected, label):
    import torch

    if not torch.equal(actual, expected):
        mismatch = (
            torch.nonzero(actual != expected, as_tuple=False)[0].flatten().tolist()
        )
        raise AssertionError(f"{label}: mismatch at {mismatch}")


def _mxfp4_k_valid_mask(raw, scales):
    """Bytes of the tiled MXFP4 K payload that hold real tokens (not tile padding)."""
    import torch

    batch, sequence, heads, _ = scales.shape
    tiles = (sequence + 127) // 128
    mask = torch.zeros_like(raw, dtype=torch.bool)
    tiled = mask[: batch * heads * tiles * 8192].view(batch, heads, tiles, 4, 128, 16)
    tokens = torch.arange(tiles * 128, device=raw.device).view(1, 1, tiles, 1, 128, 1)
    tiled.copy_((tokens < sequence).expand_as(tiled))
    return mask


def _check_packed(recipe, actual, expected, sequence, heads_local):
    import torch

    payloads, scales = actual
    ref_payloads, ref_scales = expected

    def scales_equal(role, label):
        ref = ref_scales[role]
        _assert_equal(scales[role][: ref.numel()].view_as(ref), ref, f"{label} scales")

    for role, label in enumerate("QKV"):
        ref = ref_payloads[role]
        if role == 2 and recipe.quant[1] == "mxfp4":
            # V tokens are permuted within a tile; padding and slack must be zero.
            padded = (sequence + 127) // 128 * 128
            valid_bytes = heads_local * padded * 64
            _assert_equal(
                payloads[2][:valid_bytes], ref[:valid_bytes], "MXFP4 V payload"
            )
            _assert_equal(
                payloads[2][valid_bytes:],
                torch.zeros_like(payloads[2][valid_bytes:]),
                "MXFP4 V slack",
            )
        elif role == 1 and recipe.quant[0] == "mxfp4":
            valid = _mxfp4_k_valid_mask(payloads[1], ref_scales[1])
            _assert_equal(payloads[1][valid], ref[valid], "MXFP4 K payload")
            _assert_equal(
                payloads[1][~valid],
                torch.zeros_like(payloads[1][~valid]),
                "MXFP4 K padding",
            )
        elif role == 2 and recipe.v_fp6_p:
            # The FP6-P packer returns a strided view into a raw buffer that
            # also holds the transport slack; compare the flat storage.
            _assert_equal(
                payloads[2], ref.as_strided(payloads[2].shape, (1,)), "FP6-P V payload"
            )
        else:
            _assert_equal(
                payloads[role], ref.view(torch.uint8).flatten(), f"{label} payload"
            )
        scales_equal(role, f"{recipe.name} {label}")


def _run_recipe(
    rank, world, shape_name, shape, inputs, recipe, args, group, op_class, packers
):
    import torch
    import torch.distributed as dist

    from aiter.ops.mha_v4 import AttentionPack

    softmax_scale = shape[-1] ** -0.5
    packed = recipe.packers is not None
    logical_per_role = math.prod(shape) * _BF16_BYTES * (world - 1) / world

    def pack_role(role, value):
        if packed:
            return packers[recipe.packers[role]](value)
        return value

    def baseline_trio(index):
        return tuple(
            pack_role(role, all_to_all_4d(value, group))
            for role, value in enumerate(inputs[index % _ROTATIONS])
        )

    def baseline_role(role, index, start, end):
        values = inputs[index % _ROTATIONS]
        for current in range(3):
            if current == role:
                start.record()
            pack_role(current, all_to_all_4d(values[current], group))
            if current == role:
                end.record()

    op = op_class(
        rank=rank,
        world_size=world,
        shape=shape,
        quant=recipe.quant,
        return_packed=packed,
        v_pack=AttentionPack.V_FOR_FP6_P if recipe.v_fp6_p else AttentionPack.DEFAULT,
        softmax_scale=softmax_scale if packed else None,
        hadamard=recipe.hadamard,
    )

    def fused_role_trio(index, hook=None):
        values = inputs[index % _ROTATIONS]
        results = []
        for role in range(3):
            if hook:
                hook(role, "before")
            results.append(op.submit_role(role, values[role]))
            if hook:
                hook(role, "after")
        return results

    def fused_trio(index):
        return fused_role_trio(index)

    def fused_role(role, index, start, end):
        def hook(current, when):
            if current == role:
                (start if when == "before" else end).record()

        fused_role_trio(index, hook)

    # Wire bytes per role: payload plus scale backing, as allocated for receive.
    wire_per_role = tuple(
        payload.numel() * payload.element_size()
        + (scale.untyped_storage().nbytes() if recipe.quant else 0)
        for payload, scale in zip(
            op.outputs_sets[0], op.scale_storage_sets[0], strict=True
        )
    )

    # Untimed trios also compile every specialization before the timed region.
    reference = baseline_trio(0)
    results = fused_trio(0)
    torch.cuda.synchronize()
    ok = True
    try:
        if packed:
            actual = (
                tuple(r.payload for r in results),
                tuple(r.scale for r in results),
            )
            expected = (tuple(r[0] for r in reference), tuple(r[1] for r in reference))
            _check_packed(recipe, actual, expected, shape[1] * world, shape[2] // world)
        else:
            for role, (got, ref) in enumerate(zip(results[2], reference, strict=True)):
                ref = ref.permute(0, 2, 1, 3).contiguous()
                _assert_equal(got.view_as(ref), ref, f"bf16 {'QKV'[role]}")
    except AssertionError as error:
        print(f"rank {rank}: MISMATCH {recipe.name}: {error}", flush=True)
        ok = False
    flag = torch.tensor(int(ok), device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if not flag.item():
        if rank == 0:
            print(f"MISMATCH {recipe.name}: timing skipped", flush=True)
        return []
    if rank == 0:
        print(f"PASS {shape_name} {recipe.name}: all ranks match baseline", flush=True)

    records = []
    modes = (
        ("rccl+pack" if packed else "rccl_bf16", baseline_trio, baseline_role, None),
        ("fused", fused_trio, fused_role, wire_per_role),
    )
    for mode, trio, role_fn, wire in modes:
        medians = _time_roles(trio, role_fn, args.warmup, args.iterations, group)
        for role in (0, 1, 2, "trio"):
            scale = 3 if role == "trio" else 1
            wire_bytes = None
            if wire is not None:
                wire_bytes = (
                    (sum(wire) if role == "trio" else wire[role]) * (world - 1) / world
                )
            records.append(
                {
                    "shape": shape_name,
                    "recipe": recipe.name,
                    "mode": mode,
                    "role": "trio" if role == "trio" else "QKV"[role],
                    "us": medians[role],
                    "logical_bytes": logical_per_role * scale,
                    "wire_bytes": wire_bytes,
                }
            )
    return records


def _render(records, shapes, world, arch, args, elapsed_s):
    lines = [
        (
            f"world={world}; GPU arch={arch}; rotations={_ROTATIONS}; "
            f"warmup={args.warmup}; iters={args.iterations}"
        )
    ]
    for shape_name in shapes:
        rows = [r for r in records if r["shape"] == shape_name]
        if not rows:
            continue
        local = _local_shape(_SHAPES[shape_name], world)
        lines += [
            "",
            (
                f"## {shape_name} {list(_SHAPES[shape_name])} "
                f"(local input {list(local)}, padded S={local[1] * world})"
            ),
            "| recipe | mode | role | median us (max rank) | logical GB/s | wire GB/s |",
            "|---|---|---|---:|---:|---:|",
        ]
        grouped = {}
        for row in rows:
            grouped.setdefault((row["recipe"], row["mode"], row["role"]), []).append(
                row
            )
        for (recipe, mode, role), per_rank in grouped.items():
            max_us = max(r["us"] for r in per_rank)
            if not math.isfinite(max_us) or max_us <= 0:
                raise ValueError(
                    f"Invalid timing for {shape_name} {recipe} {mode} {role}"
                )
            wire_bytes = per_rank[0]["wire_bytes"]
            wire = "n/a" if wire_bytes is None else f"{wire_bytes / max_us / 1e3:.2f}"
            logical = per_rank[0]["logical_bytes"] / max_us / 1e3
            lines.append(
                f"| {recipe} | {mode} | {role} | {max_us:.2f} | {logical:.2f} | {wire} |"
            )
    lines += ["", f"Elapsed wall time: {elapsed_s:.1f} s"]
    return "\n".join(lines)


def _worker(rank, world, args):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")

    import mori.shmem as ms
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    from aiter.ops.flydsl.attention_a2a_intranode import AttentionA2AIntraNodeOp

    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{args.master_port}",
        rank=rank,
        world_size=world,
        device_id=device,
    )
    records = []
    start = time.perf_counter()
    try:
        cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        ms.shmem_torch_process_group_init("mori")
        for shape_name in args.shapes:
            requested = _SHAPES[shape_name]
            shape = _local_shape(requested, world)
            if requested[2] % world:
                raise ValueError(
                    f"{shape_name}: heads={requested[2]} must divide world={world}"
                )
            inputs = [
                tuple(
                    _make_input(rank, shape, rotation * 3 + role, device)
                    for role in range(3)
                )
                for rotation in range(_ROTATIONS)
            ]
            # Padded tokens carry zeros, as a real caller would send.
            valid = max(0, min(shape[1], requested[1] - rank * shape[1]))
            for values in inputs:
                for value in values:
                    value[:, valid:].zero_()
            packers = _build_packers(shape[-1] ** -0.5)
            for recipe in args.recipes:
                print(
                    f"rank {rank}: {shape_name} {recipe.name} local={list(shape)}",
                    flush=True,
                )
                records += _run_recipe(
                    rank,
                    world,
                    shape_name,
                    shape,
                    inputs,
                    recipe,
                    args,
                    dist.group.WORLD,
                    functools.partial(AttentionA2AIntraNodeOp, group=cpu_group),
                    packers,
                )
            del inputs
        gathered = [None] * world
        dist.all_gather_object(gathered, records, group=cpu_group)
        if rank == 0:
            all_records = [row for per_rank in gathered for row in per_rank]
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            report = _render(
                all_records, args.shapes, world, arch, args, time.perf_counter() - start
            )
            print(report, flush=True)
            output = f"/tmp/bench_attention_a2a_w{world}.md"
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
    names = [r.name for r in _RECIPES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-w", "--world-size", type=int, required=True, choices=(2, 4, 8)
    )
    parser.add_argument(
        "--shapes", choices=tuple(_SHAPES), nargs="+", default=list(_SHAPES)
    )
    parser.add_argument(
        "--recipes", choices=["all", *names], nargs="+", default=["all"]
    )
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations positive")
    selected = names if "all" in args.recipes else args.recipes
    args.recipes = [r for r in _RECIPES if r.name in selected]

    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        print("SKIP: attention A2A requires ROCm GPUs")
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
