# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness and timing for the pre-attention monokernel.

One row per (world_size, profile, rows, heads, K, prep, gemm). Every row checks, on every
rank and for two consecutive calls (both receive-buffer parities):
  - the fused projections against the host GEMM's standalone prep and maxima GEMMs, bit for
    bit;
  - Q's and K's row factors (the prep's row-factor op) against PyTorch;
  - every role's received payload and scale bytes against AttentionA2AIntraNodeOp's packed
    push of the same projections, bit for bit;
then times one call.

A rank that dies by a signal without reporting an error (MORI's bootstrap port race) reruns
the world, up to three times.
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import socket
import time
import traceback
from multiprocessing import freeze_support, get_context
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

HEAD_DIM = 128
EPS = 1e-6
# Global tokens past the valid sequence for profiles with direct K/V output, as Wan2.2
# 720p (75600 valid of 75776 on 8 ranks).
KV_TRIM = 176
ATTEMPTS = 3

_CTX = SimpleNamespace()


def _prep(name, profile):
    from aiter.ops.flydsl.pre_attention_monokernel import PrepSpec

    if name == "wan":
        return PrepSpec("rms_across_heads", "interleaved", True, profile.hadamard, EPS)
    if name == "per_head":
        return PrepSpec("rms_per_head", "rotate_half", True, profile.hadamard, EPS)
    if name == "flux3":
        return PrepSpec("rms_per_head", "interleaved", False, profile.hadamard, EPS)
    raise ValueError(f"unknown prep {name!r}")


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _quantize_rows(x):
    """Symmetric INT8 per row, as the INT8 linear does."""
    scale = x.float().abs().amax(1).clamp(min=1e-8) / 127.0
    q = torch.round(x.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _exact(actual, expected, label):
    """Zero tolerance: any differing element fails."""
    a = actual.reshape(-1)
    b = expected.reshape(-1)
    assert a.numel() == b.numel(), f"{label}: {a.numel()} vs {b.numel()} elements"
    if a.element_size() == 1:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    if not torch.equal(a, b):
        checkAllclose(a.float(), b.float(), rtol=0, atol=0, msg=f"{label}: ")
        raise AssertionError(f"{label}: mismatch")


def _build(profile_name, rows, heads, K, prep_name, gemm):
    """The monokernel, its reference and a timed call for one case on this rank."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.attention_a2a_intranode import AttentionA2AIntraNodeOp
    from aiter.ops.flydsl.attention_a2a_profiles import attention_a2a_profile
    from aiter.ops.flydsl.kernels.gemm_a6w6_4wave_host import compile_a6w6_host_4w
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.flydsl.pre_attention_monokernel import (
        PreAttentionMonokernel,
        host_weight_order,
    )
    from aiter.ops.shuffle import shuffle_weight

    rank, world, device = _CTX.rank, _CTX.world_size, _CTX.device
    profile = attention_a2a_profile(profile_name)
    prep = _prep(prep_name, profile)
    R = heads * HEAD_DIM
    seq_full = rows * world
    valid_kv_len = seq_full - KV_TRIM if profile.direct_kv_output else None
    row_factor = prep.row_factor

    # Weights are the same on every rank, tokens and positions are not.
    gen = torch.Generator(device=device).manual_seed(0)
    raw, weights, scales, biases, params = [], [], [], [], []
    for role in range(3):
        w = torch.randn(R, K, device=device, generator=gen, dtype=torch.bfloat16) * 0.02
        raw.append(w)
        w_q, w_s = _quantize_rows(w)
        weights.append(w_q)
        scales.append(w_s)
        biases.append(torch.randn(R, device=device, generator=gen).bfloat16().float())
        # Q's and K's norm weights; V's draw keeps the later draws in place.
        weight = 1 + 0.1 * torch.randn(R, device=device, generator=gen)
        params.append({"weight": weight.bfloat16().float()})
    params = params[:2]
    gen = torch.Generator(device=device).manual_seed(100 + rank)
    x = torch.randn(rows, K, device=device, generator=gen, dtype=torch.bfloat16)
    angles = torch.rand(rows, HEAD_DIM // 2, device=device, generator=gen) * 6.28
    tables = prep.pack_tables({"cos": angles.cos(), "sin": angles.sin()})
    cu = get_cu_num()
    if gemm == "a6w6":
        # MXFP6 operands; the weights' rows in the host layout. The A6W6 host applies
        # no column scales: they stay as ones in their place.
        scales = [torch.ones(R, device=device) for _ in range(3)]
        x_op, x_scale = aiter.quant_mxfp6_gemm(x)
        order = host_weight_order(3 * R).to(device)
        fused, fused_scale = aiter.quant_mxfp6_gemm(torch.cat(raw)[order].contiguous())
        order = host_weight_order(R).to(device)
        role_ops = [aiter.quant_mxfp6_gemm(w[order].contiguous()) for w in raw]
        K_PAD = -(-K // 128) * 128
        prep_gemm = compile_a6w6_host_4w(K=K_PAD, xcd_swizzle=8, num_cu=cu, prep=prep)
        amax_gemm = compile_a6w6_host_4w(
            K=K_PAD, xcd_swizzle=8, num_cu=cu, bias=prep.bias, amax=True
        )
    else:
        x_op, x_scale = _quantize_rows(x)
        fused = shuffle_weight(torch.cat(weights), layout=(16, 16)).contiguous()
        fused_scale = None
        role_ops = [
            (shuffle_weight(w, layout=(16, 16)).contiguous(), None) for w in weights
        ]
        prep_gemm = compile_int8_gemm_4w(K=K, xcd_swizzle=8, num_cu=cu, prep=prep)
        amax_gemm = compile_int8_gemm_4w(
            K=K, xcd_swizzle=8, num_cu=cu, bias=prep.bias, amax=True
        )
    # B_scale per column: the scales, the biases with a bias, then the prep's columns.
    columns = prep.pack_columns(torch.cat(scales), torch.cat(biases), params, heads)

    mono = PreAttentionMonokernel(
        rank=rank,
        world_size=world,
        rows=rows,
        K=K,
        heads=heads,
        profile=profile,
        group=_CTX.cpu_group,
        prep=prep,
        valid_kv_len=valid_kv_len,
        gemm=gemm,
    )

    # The standalone GEMMs of the same host.
    outs = [
        torch.full((rows * R,), float("nan"), device=device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    parts = [
        torch.full((2, rows, heads), float("nan"), device=device) for _ in range(3)
    ]
    role_columns = [
        prep.pack_columns(scales[r], biases[r], [params[r]], heads) for r in range(2)
    ]
    role_columns.append(torch.cat([scales[2], biases[2]] if prep.bias else [scales[2]]))
    stream = fx.Stream(torch.cuda.current_stream())
    for role in range(3):
        w_op, w_scale = role_ops[role]
        # The INT8 maxima GEMM takes no tables; the others read them only with a prep.
        table_args = (
            ((tables if tables is not None else role_columns[role]).view(-1),)
            if role < 2 or gemm == "a6w6"
            else ()
        )
        _run_compiled(
            prep_gemm if role < 2 else amax_gemm,
            x_op.view(-1),
            w_op.view(-1),
            outs[role],
            *((x_scale, w_scale) if gemm == "a6w6" else (x_scale,)),
            role_columns[role],
            *table_args,
            parts[role].view(-1),
            rows,
            R,
            stream,
        )

    def call():
        return mono(x_op, x_scale, fused, columns, tables, weight_scale=fused_scale)

    # The A2A op's packed push of the standalone projections, as a framework builds it.
    per_tensor_qk = profile.codecs[0] in ("int8", "e4m3")
    op = AttentionA2AIntraNodeOp(
        rank=rank,
        world_size=world,
        shape=(1, rows, heads, HEAD_DIM),
        block_num=profile.block_num,
        quant=profile.quant,
        return_packed=True,
        v_pack=profile.v_pack,
        softmax_scale=HEAD_DIM**-0.5,
        hadamard=False,
        valid_kv_len=valid_kv_len,
        row_scaled_qk=per_tensor_qk and row_factor is not None,
        group=_CTX.cpu_group,
    )
    side = torch.cuda.Stream(device=device)

    def check(label):
        results = call()
        torch.cuda.synchronize()
        for r in range(3):
            _exact(mono.projection(r), outs[r], f"{label} {'QKV'[r]} projection")
        factors = [None, None]
        if row_factor is not None:
            for r in range(2):
                factors[r] = mono.factors(r).clone()
                # The kernel adds the heads' partials in order.
                total = torch.zeros(rows, device=device)
                for h in range(heads):
                    total = total + parts[r][0, :, h]
                expected = row_factor.reference_finish(total, R)
                err = checkAllclose(
                    expected,
                    factors[r],
                    rtol=1e-6,
                    atol=0,
                    msg=f"{label} {'QK'[r]} factors: ",
                )
                assert err == 0, f"{label} {'QK'[r]} row factors differ from PyTorch"
        inputs = []
        for r in range(3):
            value = outs[r].view(rows, R)
            if r < 2 and row_factor is not None and not per_tensor_qk:
                # MX Q/K take BF16 values that include the row factor.
                value = (value.float() * factors[r][:, None]).to(torch.bfloat16)
            inputs.append(value.view(1, rows, heads, HEAD_DIM))
        # The push reads inputs made on this stream.
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ref = [
                op.submit_role(
                    r,
                    inputs[r],
                    stream=side,
                    row_scale=(
                        factors[r]
                        if r < 2 and per_tensor_qk and row_factor is not None
                        else None
                    ),
                )
                for r in range(3)
            ]
        torch.cuda.synchronize()
        for r in range(3):
            _exact(results[r].payload, ref[r].payload, f"{label} {'QKV'[r]} payload")
            _exact(results[r].scale, ref[r].scale, f"{label} {'QKV'[r]} scale")

    return mono, call, check


def _agree(world_size, failure, label):
    """Every rank raises if any failed: a call waits on every peer, so no rank may go on
    alone."""
    flags = torch.zeros(world_size, dtype=torch.int32)
    flags[_CTX.rank] = failure is not None
    dist.all_reduce(flags, group=_CTX.cpu_group)
    failed = flags.nonzero().flatten().tolist()
    if failed:
        raise RuntimeError(f"{label}: failed on rank(s) {failed}")


@benchmark()
def test_monokernel(world_size, profile, rows, heads, K, prep, gemm):
    label = f"{profile} rows={rows} heads={heads} K={K} {prep} {gemm}"
    mono = None
    for step in ("call 1", "call 2"):
        failure = None
        try:
            if mono is None:
                mono, call, check = _build(profile, rows, heads, K, prep, gemm)
            check(step)
        except Exception as exc:  # noqa: BLE001
            failure = exc
            traceback.print_exc()
        _agree(world_size, failure, f"{label} {step}")
    _, us = run_perftest(call, num_iters=20, use_cuda_event=True)
    mono.close()
    flops = 2 * rows * 3 * heads * HEAD_DIM * K
    return {"gfx": get_gfx_runtime(), "us": us, "TFLOPS": flops / us / 1e6, "err": 0.0}


def _run_rank(rank, world_size, port, cases, results):
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
    _CTX.rank, _CTX.world_size, _CTX.device = rank, world_size, device
    try:
        _CTX.cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", _CTX.cpu_group)
        ms.shmem_torch_process_group_init("mori")
        rows = []
        with torch.inference_mode():
            for case in cases:
                rows.append(test_monokernel(world_size, *case))
        results.put((rank, rows, None))
    except Exception:
        results.put((rank, None, traceback.format_exc()))
        raise
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def _wait(procs, results, reports, grace=120):
    """Exit codes when the first rank failed (or all finished); the rest are killed after
    ``grace`` seconds, since they wait on the failed rank."""
    first = None
    failed_at = None
    while True:
        try:
            rank, rows, error = results.get(timeout=1)
            reports[rank] = (rows, error)
            continue
        except queue.Empty:
            pass
        codes = [proc.exitcode for proc in procs]
        if all(code is not None for code in codes):
            return first or codes
        if failed_at is None and any(code not in (None, 0) for code in codes):
            first, failed_at = codes, time.monotonic()
        if failed_at is not None and time.monotonic() - failed_at > grace:
            for proc in procs:
                if proc.exitcode is None:
                    proc.kill()
            for proc in procs:
                proc.join()
            return first


def _run_world(world_size, cases):
    if torch.cuda.device_count() < world_size:
        aiter.logger.warning(
            "requires %d GPUs, found %d; skipping",
            world_size,
            torch.cuda.device_count(),
        )
        return []
    context = get_context("spawn")
    for attempt in range(1, ATTEMPTS + 1):
        results = context.Queue()
        port = _free_port()
        procs = [
            context.Process(
                target=_run_rank, args=(rank, world_size, port, cases, results)
            )
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()
        reports = {}
        codes = _wait(procs, results, reports)
        if all(code == 0 for code in codes):
            return reports[0][0]
        errors = [error for _, error in reports.values() if error]
        if errors or not any(code is not None and code < 0 for code in codes):
            raise RuntimeError(
                f"world {world_size} failed, exit codes {codes}:\n" + "\n".join(errors)
            )
        aiter.logger.warning(
            "world %d attempt %d: a rank died by a signal (exit codes %s), likely MORI's "
            "bootstrap port race; rerunning",
            world_size,
            attempt,
            codes,
        )
    raise RuntimeError(f"world {world_size}: ranks died in all {ATTEMPTS} attempts")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-attention monokernel correctness and timing",
    )
    parser.add_argument(
        "-w",
        "--world-sizes",
        type=int,
        nargs="*",
        choices=(2, 4, 8),
        default=[8],
        help="ranks per run",
    )
    parser.add_argument(
        "-p",
        "--profiles",
        type=str,
        nargs="*",
        default=[
            "e4m3-e4m3",
            "int8-e4m3",
            "mxfp8-e4m3",
            "e4m3-mxfp6",
            "mxfp6-e4m3",
            "mxfp6-mxfp6",
            "mxfp4-mxfp4",
            "mxfp6-mxfp4",
        ],
        help="Attention A2A profiles",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        nargs="*",
        default=["9472,40,5120"],
        help="rows per rank, heads and K (Wan2.2 A14B 720p on 8 ranks: 9472,40,5120)",
    )
    parser.add_argument(
        "--preps",
        type=str,
        nargs="*",
        choices=("wan", "per_head", "flux3"),
        default=["wan"],
        help="Q/K prep: wan (RMSNorm across heads, interleaved RoPE, bias), per_head "
        "(RMSNorm per head, rotate-half RoPE, bias) or flux3 (RMSNorm per head, "
        "interleaved RoPE, no bias)",
    )
    parser.add_argument(
        "--gemms",
        type=str,
        nargs="*",
        choices=("int8", "a6w6"),
        default=["int8"],
        help="host GEMM: int8 (INT8 4-wave) or a6w6 (MXFP6 A6W6 4-wave)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        aiter.logger.warning("the pre-attention monokernel needs gfx950; skipping")
        return 0
    shapes = [tuple(int(v) for v in shape.split(",")) for shape in args.shapes]
    rows = []
    for world_size in args.world_sizes:
        cases = [
            (profile, m, heads, k, prep, gemm)
            for (m, heads, k), profile, prep, gemm in itertools.product(
                shapes, args.profiles, args.preps, args.gemms
            )
        ]
        rows += _run_world(world_size, cases)
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "pre-attention monokernel summary (markdown):\n%s", df.to_markdown(index=False)
    )
    print(f"completed rows={len(rows)} failures=0", flush=True)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
