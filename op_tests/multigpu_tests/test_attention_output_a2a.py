# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness and timing for the quantized attention-output A2A.

One row per (world_size, codec, rows, heads). Each row checks payloads and scales
against the canonical quantizer applied after an RCCL inverse all-to-all, checks
the consuming GEMM output bit for bit, exercises double-buffer lifetime and a
delayed rank, and then times the exchange with run_perftest.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import socket
from multiprocessing import freeze_support, get_context
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

SUPPORTED_GFX = ["gfx942", "gfx950"]
CODECS = ("e4m3", "e4m3_external_scale", "mxfp4", "mxfp6")
GFX950_CODECS = {"e4m3_external_scale", "mxfp4", "mxfp6"}
HEAD_DIM = 128
OUT_FEATURES = 1024

_CTX = SimpleNamespace()
_STATS = SimpleNamespace(checks=0)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _agree(label, body):
    """Run body on every rank and raise on all of them if any rank failed.

    The exchange waits on every peer, so a rank that stops after a local failure
    would leave the others spinning in the next launch.
    """
    failure = result = None
    try:
        result = body()
    except Exception as exc:  # noqa: BLE001
        failure = exc
    messages = [None] * _CTX.world_size
    dist.all_gather_object(
        messages, None if failure is None else repr(failure), group=_CTX.cpu_group
    )
    failed = {rank: message for rank, message in enumerate(messages) if message}
    if failed:
        raise RuntimeError(f"{label}: failed on rank(s) {failed}") from failure
    _STATS.checks += 1
    return result


def _expect_all_ranks_raise(label, body, match):
    def check():
        try:
            body()
        except ValueError as exc:
            if match not in str(exc):
                raise AssertionError(f"{label}: wrong ValueError: {exc}") from exc
            return
        raise AssertionError(f"{label}: did not raise")

    _agree(label, check)


def _exact(actual, expected, label):
    actual = actual.view(torch.uint8)
    expected = expected.view(torch.uint8)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        mismatches = (actual != expected).sum().item()
        raise AssertionError(f"{label}: {mismatches} differing bytes")


def _delay_current_stream():
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(100_000_000)
        return
    work = torch.randn(4096, 4096, device=_CTX.device)
    for _ in range(20):
        work = work @ work / 64


def _random_input(shape, salt):
    torch.manual_seed(1700 + 97 * salt + _CTX.rank)
    return torch.randn(shape, dtype=torch.bfloat16, device=_CTX.device)


def _corner_input(shape):
    value = torch.full(shape, -0.0, dtype=torch.bfloat16, device=_CTX.device)
    value[:, 0, 0] = torch.linspace(-1.0, 1.0, HEAD_DIM, device=_CTX.device)
    tiny = torch.finfo(torch.bfloat16).tiny
    value[:, 1, 0, ::2] = tiny
    value[:, 1, 0, 1::2] = -tiny
    value[:, 2, 0, 0] = torch.finfo(torch.bfloat16).max
    return value


def _inverse_a2a(value):
    """Source BSHD [1, S_global, H_local, D] -> destination matrix [S_local, H_total * D]."""
    world_size = _CTX.world_size
    _, sequence, heads_local, head_dim = value.shape
    send = value.view(world_size, sequence // world_size, heads_local, head_dim)
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send.contiguous())
    return received.permute(1, 0, 2, 3).reshape(
        sequence // world_size, world_size * heads_local * head_dim
    )


def _destination_scales(value):
    """One E4M3 descale per destination rank: global amax over its rows / FP8_MAX."""
    amax = value.view(_CTX.world_size, -1).abs().amax(dim=1).float()
    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
    return amax.clamp(min=1e-12) / _CTX.fp8_max


def _reference(codec, matrix, scale):
    if codec == "e4m3":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        return quantize(matrix, quant_dtype=aiter.dtypes.fp8)
    if codec == "e4m3_external_scale":
        payload = (matrix.float() / scale).clamp(-_CTX.fp8_max, _CTX.fp8_max)
        return payload.to(_CTX.fp8_dtype), scale.view(1, 1)
    if codec == "mxfp4":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        return quantize(matrix, quant_dtype=aiter.dtypes.fp4x2, shuffle=True)
    from aiter.ops.gemm_op_a6w6 import mxfp6_gemm_pack_size, quant_mxfp6_gemm_out

    payload_size, scale_size = mxfp6_gemm_pack_size(*matrix.shape)
    payload = torch.zeros(payload_size, dtype=torch.uint8, device=matrix.device)
    scales = torch.zeros(scale_size, dtype=torch.uint8, device=matrix.device)
    return quant_mxfp6_gemm_out(matrix, payload, scales)


def _operands(result, codec, rows, features):
    if codec == "e4m3":
        return (
            result.payload.view(aiter.dtypes.fp8).view(rows, features),
            result.scale.view(rows, features // HEAD_DIM),
        )
    if codec == "e4m3_external_scale":
        return result.payload.view(_CTX.fp8_dtype).view(rows, features), result.scale
    if codec == "mxfp4":
        return result.payload.view(aiter.dtypes.fp4x2), result.scale.view(
            aiter.dtypes.fp8_e8m0
        )
    return result.payload, result.scale


def _consumers(codec, rows, features):
    """GEMMs that read the transported activation, keyed by name."""
    torch.manual_seed(47)
    device = _CTX.device
    weight = torch.randn(OUT_FEATURES, features, dtype=torch.bfloat16, device=device)
    if codec == "e4m3":
        weight_q = weight.to(aiter.dtypes.fp8)
        weight_scale = (
            torch.rand(
                OUT_FEATURES // 128, features // 128, dtype=torch.float32, device=device
            )
            + 0.5
        )
        return {
            "a8w8_blockscale": lambda a: aiter.gemm_a8w8_blockscale(
                a[0], weight_q, a[1], weight_scale, torch.bfloat16
            )
        }
    if codec == "e4m3_external_scale":
        weight_scale = weight.abs().amax().float() / _CTX.fp8_max
        weight_q = (weight.float() / weight_scale).to(_CTX.fp8_dtype)
        return {
            "scaled_mm": lambda a: torch._scaled_mm(
                a[0],
                weight_q.t(),
                scale_a=a[1].reshape(()),
                scale_b=weight_scale,
                out_dtype=torch.bfloat16,
            )
        }
    if codec == "mxfp4":
        from aiter.ops.shuffle import shuffle_weight

        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_q, weight_scale = quantize(
            weight, quant_dtype=aiter.dtypes.fp4x2, shuffle=True
        )
        weight_q = shuffle_weight(weight_q, layout=(16, 16))
        return {
            "a4w4": lambda a: aiter.gemm_a4w4(
                a[0], weight_q, a[1], weight_scale, bpreshuffle=True
            )
        }
    from aiter.ops.gemm_op_a6w4 import (
        gemm_a6w4,
        mxfp4_gemm_pack_size,
        quant_mxfp4_gemm_hip_out,
    )
    from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

    weight6, weight6_scale = quant_mxfp6_gemm(weight)
    weight4_size, weight4_scale_size = mxfp4_gemm_pack_size(OUT_FEATURES, features)
    weight4 = torch.zeros(weight4_size, dtype=torch.uint8, device=device)
    weight4_scale = torch.zeros(weight4_scale_size, dtype=torch.uint8, device=device)
    quant_mxfp4_gemm_hip_out(weight, weight4, weight4_scale)
    return {
        "a6w6": lambda a: gemm_a6w6(
            a[0], weight6, a[1], weight6_scale, rows, OUT_FEATURES, features
        ),
        "a6w4": lambda a: gemm_a6w4(
            a[0], weight4, a[1], weight4_scale, rows, OUT_FEATURES, features
        ),
    }


class _Case:
    """One codec/shape: the op, its reference, and the per-call check."""

    def __init__(self, codec, rows, heads, blocks):
        from aiter.ops.flydsl.attention_output_a2a_intranode import (
            AttentionOutputA2AIntraNodeOp,
        )

        world_size = _CTX.world_size
        self.codec = codec
        self.rows = rows
        self.features = heads * HEAD_DIM
        self.shape = (1, rows * world_size, heads // world_size, HEAD_DIM)
        # Unused MXFP4 scale padding beyond the last valid row tile is not part of
        # the A4W4 contract, so tails compare payload and GEMM output only.
        self.compare_scales = codec != "mxfp4" or rows % 256 == 0
        self.op = AttentionOutputA2AIntraNodeOp(
            rank=_CTX.rank,
            world_size=world_size,
            shape=self.shape,
            group=_CTX.cpu_group,
            codec=codec,
            block_num=blocks,
        )
        self.consumers = _consumers(codec, rows, self.features)

    def scales_for(self, value):
        if self.codec != "e4m3_external_scale":
            return None
        return _destination_scales(value)

    def run(self, value, scales):
        if scales is None:
            return self.op(value)
        return self.op(value, destination_scales=scales)

    def expected(self, value, scales):
        scale = None if scales is None else scales[_CTX.rank]
        return _reference(self.codec, _inverse_a2a(value), scale)

    def check(self, result, expected, label):
        actual = _operands(result, self.codec, self.rows, self.features)
        _exact(actual[0], expected[0], f"{label} payload")
        if self.compare_scales:
            _exact(actual[1], expected[1], f"{label} scale")
        for name, gemm in self.consumers.items():
            # Bitwise: the BF16-maximum corner overflows to inf/NaN, and NaN != NaN.
            if not torch.equal(
                gemm(actual).view(torch.int16), gemm(expected).view(torch.int16)
            ):
                raise AssertionError(f"{label} {name} GEMM output differs")


def _check_input(case, label, value):
    scales = case.scales_for(value)
    expected = case.expected(value, scales)
    _agree(label, lambda: case.check(case.run(value, scales), expected, label))


def _check_case(case):
    _check_input(case, "random", _random_input(case.shape, salt=0))
    _check_input(case, "corner", _corner_input(case.shape))

    # Results alias double-buffered storage: a result stays valid across the next
    # call and is overwritten by the call after that.
    values = [_random_input(case.shape, salt=salt) for salt in (1, 2, 3)]
    scales = [case.scales_for(value) for value in values]
    expected = [case.expected(v, s) for v, s in zip(values, scales)]
    first = case.run(values[0], scales[0])
    case.run(values[1], scales[1])
    torch.cuda.synchronize()
    _agree(
        "result survives the next call",
        lambda: case.check(first, expected[0], "lifetime"),
    )
    third = case.run(values[2], scales[2])
    torch.cuda.synchronize()

    def reused():
        if third.payload.data_ptr() != first.payload.data_ptr():
            raise AssertionError("third call did not reuse the first call's parity")
        case.check(third, expected[2], "reuse")

    _agree("parity reuse", reused)

    # A late rank must not let peers read stale buffers or overwrite live ones.
    value = _random_input(case.shape, salt=4)
    delayed_scales = case.scales_for(value)
    delayed_expected = case.expected(value, delayed_scales)
    torch.cuda.synchronize()
    if _CTX.rank == _CTX.world_size - 1:
        _delay_current_stream()
    result = case.run(value, delayed_scales)
    torch.cuda.synchronize()
    _agree(
        "delayed rank",
        lambda: case.check(result, delayed_expected, "delayed"),
    )


def _check_negative_construction():
    from aiter.ops.flydsl.attention_output_a2a_intranode import (
        ATTENTION_OUTPUT_A2A_CODECS,
        AttentionOutputA2AIntraNodeOp,
    )

    def advertised():
        if tuple(ATTENTION_OUTPUT_A2A_CODECS) != CODECS:
            raise AssertionError(
                f"test covers {CODECS}, op advertises {ATTENTION_OUTPUT_A2A_CODECS}"
            )

    _agree("advertised codecs", advertised)
    rank, world_size = _CTX.rank, _CTX.world_size
    shape = (1, 64 * world_size, 1, HEAD_DIM)

    def op(**kwargs):
        args = {
            "rank": rank,
            "world_size": world_size,
            "shape": shape,
            "group": _CTX.cpu_group,
        }
        return AttentionOutputA2AIntraNodeOp(**(args | kwargs))

    _expect_all_ranks_raise("group=None", lambda: op(group=None), "group is required")
    _expect_all_ranks_raise(
        "swapped rank", lambda: op(rank=(rank + 1) % world_size), "rank"
    )
    _expect_all_ranks_raise(
        "unknown codec", lambda: op(codec="int8"), "unsupported attention output codec"
    )
    _expect_all_ranks_raise(
        "head dim 64",
        lambda: op(shape=(1, 64 * world_size, 1, 64)),
        "expected input shape",
    )
    _expect_all_ranks_raise(
        "block_num disagreement",
        lambda: op(block_num=64 if rank == 0 else 128),
        "disagree",
    )

    value = torch.zeros(shape, dtype=torch.bfloat16, device=_CTX.device)
    e4m3 = op(codec="e4m3")
    _expect_all_ranks_raise(
        "scales for e4m3",
        lambda: e4m3(
            value, destination_scales=torch.ones(world_size, device=_CTX.device)
        ),
        "does not accept destination_scales",
    )
    _expect_all_ranks_raise(
        "wrong input shape",
        lambda: e4m3(value[:, : 32 * world_size]),
        "expected contiguous",
    )
    if _CTX.gfx == "gfx950":
        external = op(codec="e4m3_external_scale")
        _expect_all_ranks_raise(
            "missing destination scales", lambda: external(value), "destination_scales"
        )
        _expect_all_ranks_raise(
            "per-rank instead of per-destination scales",
            lambda: external(
                value, destination_scales=torch.ones(1, device=_CTX.device)
            ),
            "destination_scales",
        )
    torch.cuda.synchronize()
    dist.barrier(group=_CTX.cpu_group)


@benchmark()
def test_output_a2a(world_size, codec, rows, heads, blocks):
    """Check one codec/shape against its reference, then time the exchange."""
    case = _agree(codec, lambda: _Case(codec, rows, heads, blocks))
    _check_case(case)
    value = _random_input(case.shape, salt=5)
    scales = case.scales_for(value)
    _, us = run_perftest(case.run, value, scales, num_iters=20, use_cuda_event=True)
    del case
    gc.collect()
    return {"gfx": _CTX.gfx, "a2a us": us}


def _run_rank(rank, world_size, port, codecs, rows_list, heads_list, blocks):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    import mori.shmem as ms

    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        ms.shmem_torch_process_group_init("mori")
        gfx = get_gfx()
        _CTX.rank, _CTX.world_size, _CTX.device, _CTX.gfx = (
            rank,
            world_size,
            device,
            gfx,
        )
        _CTX.cpu_group = cpu_group
        _CTX.fp8_dtype, _CTX.fp8_max = (
            (torch.float8_e4m3fnuz, 240.0)
            if gfx == "gfx942"
            else (torch.float8_e4m3fn, 448.0)
        )
        _check_negative_construction()
        rows = []
        for codec, sequence, heads in itertools.product(codecs, rows_list, heads_list):
            if heads % world_size or (codec in GFX950_CODECS and gfx != "gfx950"):
                continue
            rows.append(test_output_a2a(world_size, codec, sequence, heads, blocks))
        return rows, _STATS.checks
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def _run_world(world_size, codecs, rows_list, heads_list, blocks):
    visible = torch.cuda.device_count()
    if visible < world_size:
        aiter.logger.warning(
            "requires %d visible GPUs, found %d; skipping", world_size, visible
        )
        return [], 0
    port = _free_port()
    with get_context("spawn").Pool(processes=world_size) as pool:
        results = pool.starmap(
            _run_rank,
            [
                (rank, world_size, port, codecs, rows_list, heads_list, blocks)
                for rank in range(world_size)
            ],
        )
    return results[0][0], sum(checks for _, checks in results)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Attention-output intranode A2A correctness and timing",
    )
    parser.add_argument(
        "-w",
        "--world-sizes",
        type=int,
        nargs="*",
        choices=(2, 4, 8),
        default=[2],
        help="ranks per run",
    )
    parser.add_argument(
        "-c",
        "--codecs",
        type=str,
        nargs="*",
        choices=CODECS,
        default=list(CODECS),
        help="output codecs to run",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        nargs="*",
        default=[256, 1184],
        help="destination rows per rank (1184 exercises non-256-row tails)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=[8],
        help="total head counts (Wan2.2 deploys 40)",
    )
    parser.add_argument("--blocks", type=int, default=128, help="workgroups per launch")
    args = parser.parse_args()
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("attention output A2A needs gfx942 or gfx950; skipping")
        return 0
    rows, checks = [], 0
    for world_size in args.world_sizes:
        world_rows, world_checks = _run_world(
            world_size, args.codecs, args.rows, args.heads, args.blocks
        )
        rows += world_rows
        checks += world_checks
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "attention output A2A summary (markdown):\n%s", df.to_markdown(index=False)
    )
    print(f"completed rows={len(rows)} checks={checks} failures=0", flush=True)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
