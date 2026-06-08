#!/usr/bin/env python3
"""Verify: is store outer OOB failure caused by num_warps > 1?

Hypothesis: valid_outer is a global row count, but the descriptor is
per-warp with base already offset. So warp 2/3 see tdim1=32 which is
larger than their tile (16), meaning OOB check never fires.

Test: run store outer OOB with num_warps=1 (no warp distribution).
If result is DROP, the hypothesis is confirmed.
"""

import os, sys
import torch

_FLYDSL_SRC = os.environ.get("FLYDSL_ROOT", "/home/zhimding/FlyDSL")
if _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import (
    SmemAllocator, SmemPtr, check_smem_capacity, get_op_result_or_value,
)

ARCH = get_rocm_arch()
assert ARCH.startswith("gfx1250"), f"Requires gfx1250, got {ARCH}"
WAVE_SIZE = 32


def get_lds_memref(lds_ptr):
    if isinstance(lds_ptr, SmemPtr):
        return get_op_result_or_value(lds_ptr.get())
    return get_op_result_or_value(lds_ptr)


def compile_store_oob(*, tile_m, tile_k, valid_m, num_warps):
    elem_bytes = 2
    block_threads = num_warps * WAVE_SIZE
    lds_elems = tile_m * tile_k
    lds_bytes = lds_elems * elem_bytes
    f16_per_vec = 8
    vecs_per_thread = lds_elems // (block_threads * f16_per_vec)

    alloc = SmemAllocator(None, arch=ARCH,
                          global_sym_name=f"verify_store_oob_w{num_warps}_v{valid_m}")
    alloc.ptr = lds_bytes
    check_smem_capacity(alloc.ptr, ARCH)

    @flyc.kernel
    def kernel(arg_dst: fx.Tensor):
        tx = gpu.thread_id("x")
        lds_base = alloc.get_base()
        lds_smem = SmemPtr(lds_base, 0, T.f16, shape=(lds_elems,))
        lds_memref = get_lds_memref(lds_smem)

        # Fill LDS with 1.0
        one_word = arith.constant(0x3C003C00, type=T.i32)
        one_vec = vector.broadcast(ir.VectorType.get([4], T.i32), one_word)
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            typed_vec = vector.bitcast(ir.VectorType.get([f16_per_vec], T.f16), one_vec)
            vector.store(typed_vec, lds_memref, [base_elem])
        gpu.barrier()

        desc = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_dst,
            lds_memref=lds_memref,
            global_offset=(arith.index(0), arith.index(0)),
            tensor_shape=(tile_m, tile_k),
            strides=(tile_k, 1),
            tile_shape=(tile_m, tile_k),
            elem_bytes=elem_bytes,
            num_warps=num_warps,
            for_store=True,
            valid_outer=valid_m,
        )
        tdm_ops.tensor_store_2d(desc)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(arg_dst: fx.Tensor, stream: fx.Stream):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()
        kernel(arg_dst).launch(
            grid=(1, 1, 1), block=(block_threads, 1, 1), stream=stream,
        )

    return launch


def test_store_oob(num_warps, valid_m=32, tile_m=64, tile_k=64):
    print(f"\n--- Store outer OOB: num_warps={num_warps}, valid_outer={valid_m} ---")

    dst = torch.full((tile_m, tile_k), -1.0, dtype=torch.float16, device="cuda")
    launch = compile_store_oob(
        tile_m=tile_m, tile_k=tile_k, valid_m=valid_m, num_warps=num_warps,
    )
    launch(dst.view(-1), torch.cuda.current_stream())
    torch.cuda.synchronize()

    dst_np = dst.cpu().numpy()
    valid_region = dst_np[:valid_m, :]
    oob_region = dst_np[valid_m:, :]

    valid_ok = (valid_region == 1.0).all()
    oob_untouched = (oob_region == -1.0).all()

    print(f"  Valid rows [0:{valid_m}] all 1.0: {valid_ok}")
    print(f"  OOB rows [{valid_m}:{tile_m}] all -1.0 (untouched): {oob_untouched}")

    if oob_untouched:
        print(f"  => Store outer OOB = DROP")
    else:
        oob_ones = (oob_region == 1.0).sum()
        total_oob = oob_region.size
        print(f"  => Store outer OOB = WRITES THROUGH ({oob_ones}/{total_oob} are 1.0)")

    return "drop" if oob_untouched else "write-through"


if __name__ == "__main__":
    print(f"GPU: {ARCH}")
    print("=" * 60)
    print("Hypothesis: store outer OOB failure is caused by warp distribution")
    print("valid_outer is global, but descriptor base is per-warp offset")
    print("=" * 60)

    r1 = test_store_oob(num_warps=1)
    r4 = test_store_oob(num_warps=4)

    print(f"\n{'=' * 60}")
    print(f"RESULT:")
    print(f"  num_warps=1: {r1}")
    print(f"  num_warps=4: {r4}")
    if r1 == "drop" and r4 == "write-through":
        print(f"  => HYPOTHESIS CONFIRMED: warp distribution causes the bug")
        print(f"     valid_outer needs to be adjusted per-warp as:")
        print(f"     tdim1 = max(0, valid_outer - warp_off_outer)")
    elif r1 == "drop" and r4 == "drop":
        print(f"  => Both work - hypothesis wrong, maybe other issue")
    else:
        print(f"  => Unexpected result")
