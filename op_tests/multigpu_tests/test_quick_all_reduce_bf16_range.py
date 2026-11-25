# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 range regression for QuickReduce.

With cast_bf2half the block-scaled codecs reduce in fp16, which saturates at
65504 while bf16 activations (e.g. gpt-oss) go well past it. An over-range
element must not turn into inf/NaN or poison its 32-value block. The FP codec
ignores the cast: it sends bf16 and accumulates in f32, so it must be exact over
the whole range and round only once. Covers the HIP plain two-shot kernel and
add+RMSNorm epilogue, and, through the QuickAllReduce dispatcher with
AITER_FLY_AR=1, the FlyDSL INT4 plain and fused add+RMSNorm paths.
"""

import argparse
import multiprocessing
import os
import socket
import sys

import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles

# Ids match QuickReduceRegime.
CODECS = {"FP": 0, "FP8": 1, "INT6": 2, "INT4": 3, "INT3": 4}
NUMEL = 1 << 20
HIDDEN = 4096


def _open_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _cases(world_size):
    # (name, value on every rank, every 32nd element set to 80000)
    sum_above = 40_000.0 if world_size == 2 else 20_000.0
    return [
        ("low", 2**-8, False),
        ("ordinary", 100.0, False),
        ("sum_above_fp16", sum_above, False),
        ("negative_sum_above_fp16", -sum_above, False),
        ("input_above_fp16", 80_000.0, False),
        ("mixed_input_above_fp16", 1.0, True),
    ]


def _input(value, mixed, device):
    inp = torch.full((NUMEL,), value, dtype=torch.bfloat16, device=device)
    if mixed:
        inp[::32] = 80_000.0
    return inp


def _check(out, expected, rtol, context):
    assert torch.isfinite(out).all().item(), f"{context}: non-finite output"
    assert torch.count_nonzero(out).item() > 0, f"{context}: all-zero output"
    if rtol is not None:
        torch.testing.assert_close(
            out, expected, rtol=rtol, atol=0, msg=lambda m: f"{context}\n{m}"
        )


def _check_mixed_blocks(out, expected, codec, context):
    # A quantization block holds the even (or odd) elements of 64 contiguous
    # ones, so every outlier sits in an even block and the odd blocks hold only
    # 1.0. Those must be untouched, and the outliers clamp to about 65504.
    rtol = 2e-2 if codec == "FP8" else 0.0
    _check(out[1::2], expected[1::2], rtol, f"{context} outlier-free blocks")
    outliers = out[::32].float()
    assert (outliers >= 0.9 * 65504).all().item(), f"{context}: outlier lost"


def _check_fp_rounding(out, inp, group, context):
    # One rounding to bf16 gives rel L2 of about 1.7e-3 on randn; rounding after
    # every rank gives 2.7e-3 at TP4 and 3.7e-3 at TP8.
    ref = inp.float().cpu()
    dist.all_reduce(ref, group=group)
    rel = ((out.float().cpu() - ref).norm() / ref.norm()).item()
    assert rel < 2.2e-3, f"{context}: rel L2 {rel:.5f} on randn"


def _arch():
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


def _worker(rank, world_size, port, casts, codec_name):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    codecs = {codec_name: CODECS[codec_name]}
    # gfx1250 only implements BF16 INT4 with the fp16 cast (PullQ4); the other
    # two-shot paths are stubs there.
    is_gfx1250 = "gfx1250" in _arch()
    if is_gfx1250:
        casts = tuple(c for c in casts if c)
    try:
        for codec, level in codecs.items():
            ptr = ops.init_custom_qr(rank, world_size, None, level)
            try:
                qr_exchange_handles(ptr, world_size, group)
                for cast in casts:
                    for case, value, mixed in _cases(world_size):
                        context = f"TP{world_size} {codec} cast={cast} {case}"
                        inp = _input(value, mixed, device)
                        expected = (inp.float() * world_size).to(torch.bfloat16)
                        out = torch.empty_like(inp)
                        dist.barrier(group)
                        ops.qr_all_reduce(ptr, inp, out, level, cast)
                        torch.cuda.synchronize()
                        # FP must stay exact across the whole range. Uniform in-range
                        # blocks quantize exactly with the integer codecs; FP8's 1/240
                        # block scale is inexact, so it gets a tolerance.
                        if codec == "FP":
                            rtol = 0.0
                        elif case in ("low", "ordinary"):
                            rtol = 2e-2 if codec == "FP8" else 0.0
                        else:
                            rtol = None
                        _check(out, expected, rtol, context)
                        check_blocks = mixed and codec != "FP" and not is_gfx1250
                        if check_blocks:
                            _check_mixed_blocks(out, expected, codec, context)

                        # The add+RMSNorm epilogue has no INT3 or gfx1250 path.
                        if codec == "INT3" or is_gfx1250:
                            continue
                        residual = torch.zeros_like(inp).view(-1, HIDDEN)
                        weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
                        norm_out = torch.empty_like(residual)
                        residual_out = torch.empty_like(residual)
                        dist.barrier(group)
                        ops.qr_all_reduce_rmsnorm(
                            ptr,
                            inp.view(-1, HIDDEN),
                            residual,
                            residual_out,
                            norm_out,
                            weight,
                            1e-6,
                            HIDDEN,
                            level,
                            cast,
                        )
                        torch.cuda.synchronize()
                        _check(
                            residual_out.view(-1), expected, rtol, f"{context} rmsnorm"
                        )
                        if check_blocks:
                            _check_mixed_blocks(
                                residual_out.view(-1),
                                expected,
                                codec,
                                f"{context} rmsnorm",
                            )
                        assert (
                            torch.isfinite(norm_out).all().item()
                        ), f"{context} rmsnorm: non-finite norm output"

                    if codec != "FP" or is_gfx1250:
                        continue
                    context = f"TP{world_size} FP cast={cast} randn"
                    torch.manual_seed(1234 + rank)
                    inp = torch.randn(NUMEL, device=device).to(torch.bfloat16)
                    out = torch.empty_like(inp)
                    dist.barrier(group)
                    ops.qr_all_reduce(ptr, inp, out, level, cast)
                    torch.cuda.synchronize()
                    _check_fp_rounding(out, inp, group, context)

                    residual = torch.zeros_like(inp).view(-1, HIDDEN)
                    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
                    norm_out = torch.empty_like(residual)
                    residual_out = torch.empty_like(residual)
                    dist.barrier(group)
                    ops.qr_all_reduce_rmsnorm(
                        ptr,
                        inp.view(-1, HIDDEN),
                        residual,
                        residual_out,
                        norm_out,
                        weight,
                        1e-6,
                        HIDDEN,
                        level,
                        cast,
                    )
                    torch.cuda.synchronize()
                    _check_fp_rounding(
                        residual_out.view(-1), inp, group, f"{context} rmsnorm"
                    )
            finally:
                ops.qr_destroy(ptr)
    finally:
        dist.destroy_process_group()


def _fly_worker(rank, world_size, port):
    from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    comm = QuickAllReduce(group=group, device=device)
    try:
        assert not comm.disabled, "QuickAllReduce disabled"
        if comm._fly_ar is None:
            # Every rank sees the same FlyDSL availability, so all skip together.
            print(f"skip: FlyDSL QuickReduce unavailable on rank {rank}")
            return
        # 16 MiB rows of HIDDEN, inside the FlyDSL policy window.
        rows = 2048
        for case, value, mixed in _cases(world_size):
            context = f"TP{world_size} FlyDSL INT4 {case}"
            inp = torch.full(
                (rows * HIDDEN,), value, dtype=torch.bfloat16, device=device
            )
            if mixed:
                inp[::32] = 80_000.0
            inp = inp.view(rows, HIDDEN)
            expected = (inp.float() * world_size).to(torch.bfloat16)
            assert comm._should_fly(inp), f"{context}: not served by FlyDSL"
            dist.barrier(group)
            out = comm.quick_all_reduce(inp)
            torch.cuda.synchronize()
            # FlyDSL INT4 quantizes even uniform blocks inexactly (E5M2 group scale).
            rtol = 0.15 if case in ("low", "ordinary") else None
            _check(out, expected, rtol, context)

            residual = torch.zeros_like(inp)
            weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
            if comm.should_fly_allreduce_rmsnorm(inp, residual, weight):
                dist.barrier(group)
                norm_out, residual_out = comm.fly_allreduce_rmsnorm(
                    inp, residual, weight, 1e-6
                )
                torch.cuda.synchronize()
                _check(residual_out, expected, rtol, f"{context} rmsnorm")
                assert (
                    torch.isfinite(norm_out).all().item()
                ), f"{context} rmsnorm: non-finite norm output"
    finally:
        comm.close()
        dist.destroy_process_group()


def _spawn(target, world_size, args=(), env=None):
    multiprocessing.set_start_method("spawn", force=True)
    port = _open_port()
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        processes = [
            multiprocessing.Process(target=target, args=(rank, world_size, port, *args))
            for rank in range(world_size)
        ]
        for p in processes:
            p.start()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for p in processes:
        p.join(timeout=300)
    for rank, p in enumerate(processes):
        if p.is_alive():
            p.terminate()
            p.join()
            raise RuntimeError(f"TP{world_size} rank {rank} timed out")
        assert p.exitcode == 0, f"TP{world_size} rank {rank} failed"


def test_quick_all_reduce_bf16_range(world_size, casts=(1, 0)):
    if "gfx1250" in _arch():
        codecs = ["INT4"]
    else:
        codecs = [c for c in CODECS if c != "INT3" or world_size == 2]
    # One process group per codec: creating a communicator again after qr_destroy
    # in the same process can fail in hipIpcGetMemHandle.
    for codec in codecs:
        _spawn(_worker, world_size, (casts, codec))


def test_flydsl_quick_all_reduce_bf16_range(world_size):
    # The dispatcher reads these when its module is imported and at init, so
    # they must be in the environment the workers start with.
    _spawn(
        _fly_worker,
        world_size,
        env={
            "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
            "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
            "AITER_FLY_AR": "1",
            "AITER_FLY_AR_FUSED_HIDDENS": str(HIDDEN),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--include-gfx1250",
        action="store_true",
        help="run on gfx1250, whose PullQ4 path does not guard the fp16 range yet",
    )
    args = parser.parse_args()
    if "gfx1250" in _arch() and not args.include_gfx1250:
        print("skip: gfx1250 PullQ4 has no fp16 range guard yet")
        sys.exit(0)
    for tp in args.tp:
        if tp <= torch.cuda.device_count():
            test_quick_all_reduce_bf16_range(tp)
            if any(a in _arch() for a in ("gfx942", "gfx950")):
                test_flydsl_quick_all_reduce_bf16_range(tp)
