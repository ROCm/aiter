# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Microbenchmark: MoE TDM tensor-load → LDS bandwidth.

Measures TDM (tensor data mover) global → LDS bandwidth for MoE weight
access pattern: weight (experts, inter_dim, model_dim).

Supported dtypes: f16, bf16, fp8, mxfp4.

Grid: gx = experts, gy = ceil(inter_dim / tile_n)

Pipeline follows the production pattern in moe_gemm_2stage_mxscale:
  num_stages=1: issue → tensor_wait(0) → barrier → read → barrier
  num_stages≥2: prologue fills pipeline, steady-state uses
                pipeline_fence_signal/wait to overlap TDM loads
                with LDS reads across rotating buffers.

All WMMA compute is stripped.  LDS readback is one scalar load per
thread (minimal anti-DCE).

Usage:
    python -m kernels.bench_moe_lds_bw [options]
"""

from __future__ import annotations

import argparse
import functools

import torch

torch.set_default_device("cuda")

# dtype → (elem_bytes for TDM descriptor, pack_factor on K-dim, torch_dtype)
_DTYPE_CFG = {
    "f16":   (2, 1, torch.float16),
    "bf16":  (2, 1, torch.bfloat16),
    "fp8":   (1, 1, torch.float8_e4m3fnuz),
    "mxfp4": (1, 2, torch.uint8),
}


@functools.lru_cache(maxsize=64)
def _compile_lds_bw_kernel(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    dtype: str,
    waves_per_eu: int | None,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
    from flydsl.expr.typing import T
    from flydsl.runtime.device import get_rocm_arch as get_hip_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

    from kernels.gemm_common_gfx1250 import pipeline_fence, pipeline_fence_signal, pipeline_fence_wait
    from kernels.moe_gemm_2stage_common_gfx1250 import (
        _finalize_alloc_and_launch_2d,
        _require_gfx1250,
    )

    _require_gfx1250()

    elem_bytes, pack_factor, _ = _DTYPE_CFG[dtype]

    WAVE_SIZE = 32
    LDS_PAD = 8
    num_warps = 4
    block_threads = num_warps * WAVE_SIZE

    packed_tile_k = tile_k // pack_factor
    packed_model_dim = model_dim // pack_factor
    num_k_tiles = model_dim // tile_k
    _nb = num_stages

    lds_stride = packed_tile_k + LDS_PAD
    lds_elems = tile_n * lds_stride + LDS_PAD

    # Pipeline params (matches moe_gemm_2stage_common_gfx1250 logic)
    _use_pipeline = _nb >= 2
    if _use_pipeline:
        pre_loaded = _nb - 1
        loop_iters = (num_k_tiles - pre_loaded) // _nb
        tail_start = loop_iters * _nb
        tail_tiles = num_k_tiles - tail_start - pre_loaded
        # 1 TDM load per step (just B weight)
        TDM_PER_STEP = 1
        fence_outstanding = TDM_PER_STEP * (_nb - 2)

    alloc = SmemAllocator(None, arch=str(get_hip_arch()), global_sym_name="bench_lds_bw")
    off_b = []
    for _ in range(_nb):
        o = alloc._align(alloc.ptr, 16)
        alloc.ptr = o + lds_elems * elem_bytes
        off_b.append(o)

    sink_off = alloc._align(alloc.ptr, 16)
    alloc.ptr = sink_off + block_threads * 2

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def bench_lds_bw_kernel(
        arg_w: fx.Tensor,
        arg_sink: fx.Tensor,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")  # expert
        by = gpu.block_id("y")  # inter-dim tile

        sink_rsrc = buffer_ops.create_buffer_resource(arg_sink, max_size=True)

        base_ptr = alloc.get_base()
        smem_b = []
        for s in range_constexpr(_nb):
            sb = SmemPtr(base_ptr, off_b[s], T.f16, shape=(lds_elems,))
            smem_b.append(get_op_result_or_value(sb.get()))

        n_base = bx * arith.index(inter_dim) + by * arith.index(tile_n)

        def issue_tdm_load(k_tile_idx, buf_idx):
            k_packed_off = arith.index(k_tile_idx * packed_tile_k)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_w,
                lds_memref=smem_b[buf_idx],
                global_offset=(n_base, k_packed_off),
                tensor_shape=(inter_dim * experts, packed_model_dim),
                strides=(packed_model_dim, 1),
                tile_shape=(tile_n, packed_tile_k),
                elem_bytes=elem_bytes,
                pad_interval=packed_tile_k,
                pad_amount=LDS_PAD,
                num_warps=num_warps,
            )
            tdm_ops.tensor_load_2d(desc)

        def lds_read_one(buf_idx):
            v = vector.load_op(
                ir.VectorType.get([1], T.f16),
                smem_b[buf_idx],
                [tx],
            )
            return vector.extract(v, static_position=[0], dynamic_position=[])

        sink_val = arith.constant(0.0, type=T.f16)

        if not _use_pipeline:
            # ── non-pipelined: single buffer ──
            for kt in range_constexpr(num_k_tiles):
                issue_tdm_load(kt, 0)
                tdm_ops.tensor_wait(0)
                gpu.barrier()
                sink_val = arith.addf(sink_val, lds_read_one(0))
                gpu.barrier()
        else:
            # ── prologue: fill num_stages-1 buffers ──
            for pi in range_constexpr(pre_loaded):
                issue_tdm_load(pi, pi)
            pipeline_fence(outstanding=0)

            # ── steady-state loop ──
            for li in range_constexpr(loop_iters):
                for bi in range_constexpr(_nb):
                    lb = (bi + _nb - 1) % _nb
                    kt = li * _nb + pre_loaded + bi

                    pipeline_fence_signal(outstanding=fence_outstanding)
                    pipeline_fence_wait()

                    issue_tdm_load(kt, lb)
                    sink_val = arith.addf(sink_val, lds_read_one(bi))

            # ── post-loop fence ──
            pipeline_fence(outstanding=0)

            # ── tail: drain remaining tiles ──
            for ti in range_constexpr(tail_tiles + pre_loaded):
                tail_kt = tail_start + pre_loaded + ti
                tail_buf = ti % _nb
                read_buf = (tail_start + ti) % _nb

                if tail_kt < num_k_tiles:
                    pipeline_fence_signal(outstanding=0)
                    pipeline_fence_wait()
                    issue_tdm_load(tail_kt, (tail_kt - 1) % _nb)
                    sink_val = arith.addf(sink_val, lds_read_one(read_buf))
                else:
                    pipeline_fence(outstanding=0)
                    sink_val = arith.addf(sink_val, lds_read_one(read_buf))

        buffer_ops.buffer_store(
            sink_val, sink_rsrc,
            arith.index_cast(T.i32, tx),
        )

    @flyc.jit
    def launch_lds_bw(
        arg_w: fx.Tensor,
        arg_sink: fx.Tensor,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        gx = fx.Index(experts)
        gy = fx.Index((inter_dim + tile_n - 1) // tile_n)
        launcher = bench_lds_bw_kernel(arg_w, arg_sink)
        _finalize_alloc_and_launch_2d(
            ctx=ctx, alloc=alloc, launcher=launcher,
            gx=gx, gy=gy, block_threads=block_threads,
            stream=stream, waves_per_eu=waves_per_eu, ir=ir,
        )

    launch_lds_bw.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
    }

    return launch_lds_bw


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def _is_ffm():
    import os
    return bool(os.environ.get("HSA_MODEL_LIB"))


def run_bench(
    experts: int,
    model_dim: int,
    inter_dim: int,
    tile_n: int,
    tile_k: int,
    num_stages: int,
    dtype: str,
    waves_per_eu: int | None,
    warmup: int,
    iters: int,
):
    elem_bytes, pack_factor, torch_dtype = _DTYPE_CFG[dtype]
    packed_model_dim = model_dim // pack_factor

    w = torch.randint(0, 127, (experts, inter_dim, packed_model_dim), dtype=torch.uint8)
    w = w.view(torch_dtype) if torch_dtype != torch.uint8 else w
    sink = torch.zeros(4 * 32, dtype=torch.float16)

    exe = _compile_lds_bw_kernel(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts,
        tile_n=tile_n, tile_k=tile_k,
        num_stages=num_stages, dtype=dtype, waves_per_eu=waves_per_eu,
    )

    stream = torch.cuda.current_stream()
    num_k_tiles = model_dim // tile_k
    packed_tile_k = tile_k // pack_factor
    gx = experts
    gy = (inter_dim + tile_n - 1) // tile_n
    total_bytes = packed_tile_k * tile_n * elem_bytes * num_k_tiles * gx * gy

    print(f"\n{'='*60}")
    print(f"MoE TDM → LDS BW Microbenchmark  [{dtype}]")
    print(f"{'='*60}")
    print(f"  Shape:  experts={experts}  inter_dim={inter_dim}  model_dim={model_dim}")
    print(f"  Tile:   tile_n={tile_n}  tile_k={tile_k}  packed_tile_k={packed_tile_k}")
    print(f"  Stages: num_stages={num_stages}  num_k_tiles={num_k_tiles}")
    print(f"  Grid:   gx={gx}(experts)  gy={gy}(inter_tiles)")
    print(f"  dtype={dtype}  elem_bytes={elem_bytes}  pack_factor={pack_factor}")
    print(f"  block_threads={4 * 32}  waves_per_eu={waves_per_eu}")

    if _is_ffm():
        # FFM: just run once for ISA dump / perf counter analysis
        exe(w, sink, stream)
        torch.cuda.synchronize()
        print(f"  [FFM mode] single launch, {total_bytes / 1e6:.2f} MB transferred")
        print(f"{'='*60}\n")
        return 0.0, 0.0

    # Real HW: measure BW
    for _ in range(warmup):
        exe(w, sink, stream)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        exe(w, sink, stream)
    end_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(end_ev)
    avg_us = elapsed_ms * 1000.0 / iters
    bw_gb_s = total_bytes / (avg_us * 1e-6) / 1e9

    print(f"  iters={iters}  avg_time={avg_us:.2f} us")
    print(f"  global→LDS bytes/launch = {total_bytes / 1e6:.2f} MB")
    print(f"  effective BW = {bw_gb_s:.2f} GB/s")
    print(f"{'='*60}\n")

    return avg_us, bw_gb_s


def main():
    p = argparse.ArgumentParser(description="MoE TDM → LDS BW microbenchmark")

    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--model_dim", type=int, default=7168)
    p.add_argument("--inter_dim", type=int, default=2048)

    p.add_argument("--tile_n", type=int, default=128)
    p.add_argument("--tile_k", type=int, default=64)

    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--dtype", type=str, default="f16",
                   choices=list(_DTYPE_CFG.keys()))
    p.add_argument("--waves_per_eu", type=int, default=None)

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)

    args = p.parse_args()

    run_bench(
        experts=args.experts,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        num_stages=args.num_stages,
        dtype=args.dtype,
        waves_per_eu=args.waves_per_eu,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
