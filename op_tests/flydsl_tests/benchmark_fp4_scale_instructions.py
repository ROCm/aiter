#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Isolate gfx1250 FP4 WMMA and LDS scale-read costs, without changing tiles."""

import argparse
import json
import statistics
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from benchmark_grouped_gemm_fp4_scales import graph_time_us
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Constexpr, T

from aiter.ops.flydsl.kernels.fp4_scale import wmma_fp4
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import workgroup_barrier
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg


@flyc.jit
def launch_wmma(
    out: fx.Pointer,
    stream: fx.Stream,
    block_size: Constexpr[int],
    scale_format: Constexpr[int],
    accumulators: Constexpr[int],
):
    @flyc.kernel(
        name=f"fp4_scale_wmma_s{block_size}f{scale_format}a{accumulators}",
        known_block_size=[32, 1, 1],
    )
    def kernel(out: fx.Pointer):
        # E2M1 code 2 is 1.0; every scale byte below represents 1.0.
        a = fx.constant_vector(0x22222222, T.vec(16, T.i32))
        b = fx.constant_vector(0x22222222, T.vec(8, T.i32))
        byte = (127, 120, 56)[scale_format]
        scale_bits = sum(byte << (8 * i) for i in range(128 // block_size))
        scale = fx.Int64(scale_bits) if block_size == 16 else fx.Int32(scale_bits)
        c = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(accumulators)]
        # Distinct initial values keep all accumulator chains observable.
        for j in range_constexpr(accumulators):
            c[j].store(fx.constant_vector(float(j), T.vec(16, T.f32)))
        for _ in range(fx.Int32(0), fx.Int32(1024), fx.Int32(1)):
            for j in range_constexpr(accumulators):
                c[j].store(
                    wmma_fp4(
                        a.ir_value(),
                        b.ir_value(),
                        c[j].load().ir_value(),
                        scale,
                        scale,
                        block_size=block_size,
                        format_a=scale_format,
                        format_b=scale_format,
                        row_b=j % 2,
                    )
                )
        output = fx.recast_iter(
            fx.PointerType.get(
                elem_ty=fx.Float32.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=4,
            ),
            out,
        )
        for j in range_constexpr(accumulators):
            output[j * 32 + fx.thread_idx.x] = c[j].load()[0]

    kernel(out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@flyc.jit
def launch_lds(
    out: fx.Pointer,
    stream: fx.Stream,
    bits: Constexpr[int],
    stride: Constexpr[int],
):
    @flyc.kernel(name=f"fp4_scale_lds_b{bits}s{stride}", known_block_size=[32, 1, 1])
    def kernel(out: fx.Pointer):
        tid = fx.thread_idx.x
        shared = fx.SharedAllocator().allocate(4096)._ptr
        p32 = fx.recast_iter(
            fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=4,
            ),
            shared,
        )
        for i in range_constexpr(32):
            p32[tid + i * 32] = fx.Int32(1)
        workgroup_barrier()
        address = fx.Int32(tid * stride)
        for _, state in range(
            fx.Int32(0), fx.Int32(1024), fx.Int32(1), init=[fx.Int32(0)]
        ):
            loaded = llvm.inline_asm(
                T.i32 if bits == 32 else T.vec(bits // 32, T.i32),
                [address.ir_value()],
                f"ds_load_b{bits} $0, $1\n s_wait_dscnt 0",
                "=&v,v",
                has_side_effects=True,
            )
            first = (
                loaded
                if bits == 32
                else llvm.extractelement(loaded, fx.Int32(0).ir_value())
            )
            total = state[0] + fx.Int32(first)
            result = yield [total]
        output = fx.recast_iter(
            fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=4,
            ),
            out,
        )
        output[tid] = result

    kernel(out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("fp4_instructions.json"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        parser.error("these instruction probes require gfx1250")

    f32_out = torch.empty(512, dtype=torch.float32, device="cuda")
    i32_out = torch.empty(32, dtype=torch.int32, device="cuda")
    prepared = {}
    for accumulators in (1, 16):
        for block_size, scale_format in ((32, 0), (16, 0), (16, 1), (16, 2)):

            def run_wmma(b=block_size, f=scale_format, a=accumulators):
                launch_wmma(ptr_arg(f32_out), torch.cuda.current_stream(), b, f, a)

            run_wmma()
            expected = 131072.0 + torch.arange(
                accumulators, device="cuda"
            ).repeat_interleave(32)
            torch.testing.assert_close(
                f32_out[: 32 * accumulators], expected, rtol=0, atol=0
            )
            prepared[f"wmma_s{block_size}f{scale_format}a{accumulators}"] = run_wmma

    for bits, stride in ((32, 4), (64, 8), (64, 12), (64, 16), (32, 8), (128, 16)):

        def run_lds(b=bits, s=stride):
            launch_lds(ptr_arg(i32_out), torch.cuda.current_stream(), b, s)

        run_lds()
        torch.testing.assert_close(i32_out, torch.full_like(i32_out, 1024))
        prepared[f"lds_b{bits}s{stride}"] = run_lds

    samples = {name: [] for name in prepared}
    for _ in range(args.rounds):
        for name, fn in prepared.items():
            samples[name].append(graph_time_us(fn))
    results = {
        name: {"median_us": statistics.median(values), "samples_us": values}
        for name, values in samples.items()
    }
    serialized = json.dumps(results, indent=2) + "\n"
    args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
