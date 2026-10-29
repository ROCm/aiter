# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native BF16/FP8 eager QR correctness and lifecycle checks on gfx942.

Run with torchrun --standalone --nproc-per-node=2 (or 4). The Gloo group
exchanges handles; a separate RCCL group supplies FP32 reference sums only.
No AITER model-parallel state or size-based communicator routing is used.
"""

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import aiter as ops
from aiter.jit import core

MB = 1024 * 1024
FP8 = 1
DTYPE = torch.bfloat16


def emit(kind, **values):
    if dist.get_rank() == 0:
        print(json.dumps({"kind": kind, **values}), flush=True)


def check_every_rank(ok, description):
    statuses = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, bool(ok))
    assert all(statuses), f"{description}: per-rank status {statuses}"


@contextmanager
def communicator(capacity, *, legacy_init=False):
    rank, size = dist.get_rank(), dist.get_world_size()
    ptr = 0
    try:
        # Keep the historical three-argument API exercised as well.
        ptr = (
            ops.init_custom_qr(rank, size, capacity)
            if legacy_init
            else ops.init_custom_qr(rank, size, capacity, FP8)
        )
        handle = torch.empty(64, dtype=torch.uint8, device="cpu")
        ops.qr_get_handle(ptr, handle.data_ptr())
        handles = [None] * size
        dist.all_gather_object(handles, handle)
        ops.qr_open_handles(ptr, [h.data_ptr() for h in handles])
        assert not ops.qr_uses_pull_q4(ptr)
        yield ptr
    finally:
        # All peers finish using the allocation before any rank frees it.
        torch.cuda.synchronize()
        dist.barrier()
        if ptr:
            ops.qr_destroy(ptr)
        dist.barrier()


def make_input(shape, pattern, rank, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed + rank * 113)
    if pattern == "zero":
        return torch.zeros(shape, dtype=DTYPE, device="cuda")
    if pattern == "rank_constant":
        return torch.full(shape, rank + 1, dtype=DTYPE, device="cuda")
    if pattern == "cancellation":
        return torch.full(shape, 1 if rank % 2 == 0 else -1, dtype=DTYPE, device="cuda")
    scales = {"random": 1.0, "near_zero": 1e-4, "tiny": 1e-8, "large": 2e5}
    inp = torch.randn(shape, device="cuda", generator=generator)
    inp.mul_(scales[pattern])
    return inp.to(DTYPE)


def check_case(ptr, shape, pattern, seed, reference_group, label):
    rank, size = dist.get_rank(), dist.get_world_size()
    inp = make_input(shape, pattern, rank, seed)
    original = inp.clone()
    # Aligned output view, with guards against vector stores past either end.
    storage = torch.full((inp.numel() + 16,), -123, dtype=DTYPE, device="cuda")
    out = storage[8:-8].view(shape)
    out.fill_(float("nan"))
    reference = inp.float()
    dist.all_reduce(reference, group=reference_group)
    ops.qr_all_reduce(ptr, inp, out, FP8, False)
    torch.cuda.synchronize()
    check_every_rank(torch.isfinite(out).all().item(), f"{label}: finite")
    peer = out.clone()
    dist.broadcast(peer, src=0, group=reference_group)
    unchanged = torch.equal(inp, original)
    guards = bool((storage[:8] == -123).all() and (storage[-8:] == -123).all())
    check_every_rank(
        torch.equal(out, peer) and unchanged and guards, f"{label}: ranks/input/guards"
    )
    delta = out.float() - reference
    rms = reference.square().mean().sqrt().item()
    rmse = delta.square().mean().sqrt().item()
    max_abs = delta.abs().max().item()
    # Keep the relative bound at small magnitudes too: an absolute tolerance
    # would silently accept the additive-scale-epsilon bug's near-zero output.
    ok = rmse <= 0.06 * rms and max_abs <= 0.5 * rms
    if pattern in ("zero", "cancellation"):
        ok = max_abs == 0
    elif pattern == "rank_constant":
        ok = max_abs <= 0.05 * size * (size + 1) / 2
    elif rms > 0:
        ok = ok and out.count_nonzero().item() > 0
    metric = {
        "label": label,
        "shape": list(shape),
        "pattern": pattern,
        "bytes": inp.nbytes,
        "reference_rms": rms,
        "rmse": rmse,
        "max_abs": max_abs,
        "normalized_l2": rmse / rms if rms else 0.0,
        "nonzero": out.count_nonzero().item(),
        "normalized_l2_limit": 0.06,
        "same_on_all_ranks": True,
        "passed": bool(ok),
    }
    emit("case", **metric)
    check_every_rank(ok, label)
    return metric


def expect_error(call, message):
    error = ""
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
    check_every_rank(message in error, f"expected error {message!r}, got {error!r}")
    emit("rejection", expected=message, error=error)


def check_rejections(ptr, capacity):
    inp = torch.ones(8, dtype=DTYPE, device="cuda")
    out = torch.empty_like(inp)
    expect_error(lambda: ops.qr_all_reduce(ptr, inp, out, -1, False), "quant")
    expect_error(lambda: ops.qr_all_reduce(ptr, inp, out, 5, False), "quant")
    expect_error(
        lambda: ops.qr_all_reduce(ptr, inp, out.float(), FP8, False), "dtype mismatch"
    )
    expect_error(
        lambda: ops.qr_all_reduce(ptr, inp, out[:4], FP8, False), "numel mismatch"
    )
    empty = inp[:0]
    expect_error(lambda: ops.qr_all_reduce(ptr, empty, empty, FP8, False), "empty")
    expect_error(
        lambda: ops.qr_all_reduce(ptr, inp[:7], out[:7], FP8, False), "divisible by 16"
    )
    fp32 = inp.float()
    expect_error(
        lambda: ops.qr_all_reduce(ptr, fp32, fp32.clone(), FP8, False), "only supports"
    )
    oversized = torch.empty(capacity // 2 + 8, dtype=DTYPE, device="cuda")
    expect_error(
        lambda: ops.qr_all_reduce(ptr, oversized, oversized.clone(), FP8, False),
        "exceed kMaxProblemSize",
    )
    expect_error(
        lambda: ops.init_custom_qr(
            dist.get_rank(), dist.get_world_size(), 2**31 + 16, FP8
        ),
        "exceeds 2 GiB",
    )
    expect_error(
        lambda: ops.init_custom_qr(-1, dist.get_world_size(), capacity, FP8),
        "invalid rank",
    )


def stress(ptr, iterations):
    rank, size = dist.get_rank(), dist.get_world_size()
    inputs = [
        torch.empty(n, dtype=DTYPE, device="cuda") for n in (8, 2560, 32776, MB + 8)
    ]
    outputs = [torch.empty_like(x) for x in inputs]
    samples = torch.empty((iterations, 3), dtype=DTYPE, device="cuda")
    expected = []
    for i in range(iterations):
        index = i % len(inputs)
        value = (i % 5) * 4
        inputs[index].fill_(value + rank + 1)
        ops.qr_all_reduce(ptr, inputs[index], outputs[index], FP8, False)
        out = outputs[index]
        samples[i, 0] = out[0]
        samples[i, 1] = out[out.numel() // 2]
        samples[i, 2] = out[-1]
        expected.append(size * value + size * (size + 1) / 2)
    torch.cuda.synchronize()
    reference = torch.tensor(expected, device="cuda").unsqueeze(1)
    relative = ((samples.float() - reference).abs() / reference).max().item()
    check_every_rank(
        torch.isfinite(samples).all().item() and relative <= 0.05,
        "stress: fresh values across shapes",
    )
    emit("stress", iterations=iterations, max_relative_error=relative, passed=True)


def verify_build(manifest_path):
    assert os.environ.get("AITER_REBUILD", "0") == "0", "Build once before torchrun"
    manifest = json.loads(manifest_path.read_text())
    assert ops.qr_max_size() == 2**31
    module = core.get_module("module_quick_all_reduce")
    binary = Path(module.__file__).resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert str(binary) == manifest["binary"] and digest == manifest["binary_sha256"]
    root = Path(ops.__file__).resolve().parents[1]
    for name, expected in manifest["sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    mapped = [
        line.strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "module_quick_all_reduce" in line
    ]
    assert mapped and all(str(binary) in line for line in mapped)
    return {
        "rank": dist.get_rank(),
        "binary": str(binary),
        "sha256": digest,
        "mappings": mapped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stress-iterations", type=int, default=1000)
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert props.gcnArchName.split(":")[0] == "gfx942", props
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    reference_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=90))
    records = []
    build = verify_build(args.manifest)
    builds = [None] * dist.get_world_size()
    dist.all_gather_object(builds, build)
    emit("loaded_build", processes=builds)
    if args.numerics_only:
        with communicator(65536) as ptr:
            for seed, pattern in enumerate(
                ("zero", "random", "near_zero", "tiny", "large")
            ):
                check_case(ptr, (1, 2560), pattern, seed, reference_group, pattern)
        dist.destroy_process_group(reference_group)
        dist.destroy_process_group()
        return
    # Small odd capacity first, then a larger simultaneous communicator: this
    # exercises tile rounding and the former function-static allocation size.
    with communicator(4112, legacy_init=True) as small:
        records.append(
            check_case(
                small, (2056,), "random", 1, reference_group, "small_capacity_tail"
            )
        )
        with communicator(128 * MB) as large:
            for j, (shape, pattern) in enumerate(
                [
                    ((8,), "zero"),
                    ((8,), "rank_constant"),
                    ((1, 2560), "random"),
                    ((1, 2560), "cancellation"),
                    ((1, 2560), "near_zero"),
                    ((1, 2560), "tiny"),
                    ((1, 2560), "large"),
                    ((32760,), "random"),
                    ((32768,), "random"),
                    ((32776,), "random"),
                    ((1024, 2560), "random"),
                    ((16384, 2560), "random"),
                    ((64 * MB,), "random"),
                ]
            ):
                records.append(
                    check_case(
                        large, shape, pattern, 10 + j, reference_group, f"main_{j}"
                    )
                )
            records.append(
                check_case(
                    small,
                    (8,),
                    "rank_constant",
                    99,
                    reference_group,
                    "small_after_large",
                )
            )
            check_rejections(large, 128 * MB)
            stress(large, args.stress_iterations)
    for cycle, capacity in enumerate((4112, 2 * MB + 16, 4112)):
        with communicator(capacity) as ptr:
            records.append(
                check_case(
                    ptr,
                    (capacity // 2,),
                    "random",
                    100 + cycle,
                    reference_group,
                    f"recreate_{cycle}",
                )
            )
    report = {
        "passed": True,
        "world_size": dist.get_world_size(),
        "dtype": "bfloat16",
        "codec": "FP8",
        "cast_bf2half": False,
        "mode": "eager",
        "builds": builds,
        "cases": records,
        "stress_iterations": args.stress_iterations,
    }
    if dist.get_rank() == 0:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    emit("result", passed=True, cases=len(records), world_size=dist.get_world_size())
    dist.destroy_process_group(reference_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
