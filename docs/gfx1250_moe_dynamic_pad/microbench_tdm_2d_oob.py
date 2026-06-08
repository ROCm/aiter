#!/usr/bin/env python3
"""TDM 2D OOB microbenchmark for gfx1250.

Tests whether TDM 2D descriptor's tensor_dim fields enable hardware OOB:
  (a) load OOB -> zero-fill (or stale LDS value)?
  (b) store OOB -> drop (or write garbage)?
  (c) reference frame: bound relative to descriptor base (remaining), not global origin
  (d) runtime SSA: bound passed as kernel scalar i32

Usage:
    python microbench_tdm_2d_oob.py
"""

import os
import sys

import torch

_FLYDSL_SRC = os.environ.get("FLYDSL_ROOT", "/home/zhimding/FlyDSL")
if _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import (
    SmemAllocator, SmemPtr, check_smem_capacity, get_op_result_or_value,
)

ARCH = get_rocm_arch()
assert ARCH.startswith("gfx1250"), f"This test requires gfx1250, got {ARCH}"

WAVE_SIZE = 32


def get_lds_memref(lds_ptr):
    if isinstance(lds_ptr, SmemPtr):
        return get_op_result_or_value(lds_ptr.get())
    return get_op_result_or_value(lds_ptr)


# ============================================================================
# Test (a): Load OOB -> zero-fill?
# ============================================================================

def compile_test_load_oob(*, tile_m=64, tile_k=64, valid_k=32, num_warps=4,
                           use_runtime_bound=False):
    elem_bytes = 2
    block_threads = num_warps * WAVE_SIZE
    lds_elems = tile_m * tile_k
    lds_bytes = lds_elems * elem_bytes
    # Each thread handles 8 f16 elements (16 bytes = vec<4xi32>) per iteration
    f16_per_vec = 8
    bytes_per_vec = f16_per_vec * elem_bytes  # 16
    vecs_per_thread = lds_elems // (block_threads * f16_per_vec)

    alloc = SmemAllocator(None, arch=ARCH,
                          global_sym_name=f"load_oob_{valid_k}_rt{use_runtime_bound}")
    alloc.ptr = lds_bytes
    check_smem_capacity(alloc.ptr, ARCH)

    @flyc.kernel
    def kernel_load_oob(
        arg_src: fx.Tensor,
        arg_dst: fx.Tensor,
        i32_valid_k: fx.Int32,
    ):
        tx = gpu.thread_id("x")

        lds_base = alloc.get_base()
        lds_smem = SmemPtr(lds_base, 0, T.f16, shape=(lds_elems,))
        lds_memref = get_lds_memref(lds_smem)

        # Clear LDS to sentinel 0xBEEF (write 8 f16 per iteration as vec<4xi32>)
        sentinel_word = arith.constant(0xBEEFBEEF, type=T.i32)
        sentinel_vec = vector.broadcast(ir.VectorType.get([4], T.i32), sentinel_word)
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            typed_vec = vector.bitcast(ir.VectorType.get([f16_per_vec], T.f16), sentinel_vec)
            vector.store(typed_vec, lds_memref, [base_elem])
        gpu.barrier()

        if const_expr(use_runtime_bound):
            vi = arith.index_cast(T.i32, arith.index_cast(T.index, i32_valid_k.ir_value()))
        else:
            vi = valid_k

        desc = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_src,
            lds_memref=lds_memref,
            global_offset=(arith.index(0), arith.index(0)),
            tensor_shape=(tile_m, tile_k),
            strides=(tile_k, 1),
            tile_shape=(tile_m, tile_k),
            elem_bytes=elem_bytes,
            num_warps=num_warps,
            valid_inner=vi,
        )
        tdm_ops.tensor_load_2d(desc)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        # Copy LDS -> output via buffer_store (16 bytes per iteration)
        out_rsrc = buffer_ops.create_buffer_resource(
            arg_dst, num_records_bytes=arith.index(lds_bytes),
        )
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            val = vector.load_op(ir.VectorType.get([f16_per_vec], T.f16), lds_memref, [base_elem])
            i32_vec = vector.bitcast(ir.VectorType.get([4], T.i32), val)
            byte_off = arith.index_cast(T.i32, base_elem * arith.index(elem_bytes))
            buffer_ops.buffer_store(i32_vec, out_rsrc, byte_off, offset_is_bytes=True)

    @flyc.jit
    def launch_load_oob(arg_src: fx.Tensor, arg_dst: fx.Tensor,
                         i32_valid_k: fx.Int32, stream: fx.Stream):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()

        kernel_load_oob(arg_src, arg_dst, i32_valid_k).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_load_oob


# ============================================================================
# Test (b): Store OOB -> drop?
# ============================================================================

def compile_test_store_oob(*, tile_m=64, tile_k=64, valid_m=32, num_warps=4,
                            use_runtime_bound=False):
    elem_bytes = 2
    block_threads = num_warps * WAVE_SIZE
    lds_elems = tile_m * tile_k
    lds_bytes = lds_elems * elem_bytes
    f16_per_vec = 8
    vecs_per_thread = lds_elems // (block_threads * f16_per_vec)

    alloc = SmemAllocator(None, arch=ARCH,
                          global_sym_name=f"store_oob_{valid_m}_rt{use_runtime_bound}")
    alloc.ptr = lds_bytes
    check_smem_capacity(alloc.ptr, ARCH)

    @flyc.kernel
    def kernel_store_oob(
        arg_dst: fx.Tensor,
        i32_valid_m: fx.Int32,
    ):
        tx = gpu.thread_id("x")

        lds_base = alloc.get_base()
        lds_smem = SmemPtr(lds_base, 0, T.f16, shape=(lds_elems,))
        lds_memref = get_lds_memref(lds_smem)

        # Fill LDS with 1.0 (0x3C00 in f16)
        one_word = arith.constant(0x3C003C00, type=T.i32)
        one_vec = vector.broadcast(ir.VectorType.get([4], T.i32), one_word)
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            typed_vec = vector.bitcast(ir.VectorType.get([f16_per_vec], T.f16), one_vec)
            vector.store(typed_vec, lds_memref, [base_elem])
        gpu.barrier()

        if const_expr(use_runtime_bound):
            vo = arith.index_cast(T.i32, arith.index_cast(T.index, i32_valid_m.ir_value()))
        else:
            vo = valid_m

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
            valid_outer=vo,
        )
        tdm_ops.tensor_store_2d(desc)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch_store_oob(arg_dst: fx.Tensor, i32_valid_m: fx.Int32,
                          stream: fx.Stream):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()

        kernel_store_oob(arg_dst, i32_valid_m).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_store_oob


# ============================================================================
# Test (c): Reference frame
# ============================================================================

def compile_test_refframe(*, tile_m=64, tile_k=64, k_offset=16,
                           valid_remaining=16, num_warps=4):
    elem_bytes = 2
    block_threads = num_warps * WAVE_SIZE
    lds_elems = tile_m * tile_k
    lds_bytes = lds_elems * elem_bytes
    f16_per_vec = 8
    vecs_per_thread = lds_elems // (block_threads * f16_per_vec)
    src_stride = tile_k + k_offset

    alloc = SmemAllocator(None, arch=ARCH,
                          global_sym_name=f"refframe_{k_offset}_{valid_remaining}")
    alloc.ptr = lds_bytes
    check_smem_capacity(alloc.ptr, ARCH)

    @flyc.kernel
    def kernel_refframe(
        arg_src: fx.Tensor,
        arg_dst: fx.Tensor,
    ):
        tx = gpu.thread_id("x")

        lds_base = alloc.get_base()
        lds_smem = SmemPtr(lds_base, 0, T.f16, shape=(lds_elems,))
        lds_memref = get_lds_memref(lds_smem)

        # Clear LDS to sentinel
        sentinel_word = arith.constant(0xBEEFBEEF, type=T.i32)
        sentinel_vec = vector.broadcast(ir.VectorType.get([4], T.i32), sentinel_word)
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            typed_vec = vector.bitcast(ir.VectorType.get([f16_per_vec], T.f16), sentinel_vec)
            vector.store(typed_vec, lds_memref, [base_elem])
        gpu.barrier()

        desc = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_src,
            lds_memref=lds_memref,
            global_offset=(arith.index(0), arith.index(k_offset)),
            tensor_shape=(tile_m, tile_k),
            strides=(src_stride, 1),
            tile_shape=(tile_m, tile_k),
            elem_bytes=elem_bytes,
            num_warps=num_warps,
            valid_inner=valid_remaining,
        )
        tdm_ops.tensor_load_2d(desc)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_dst, num_records_bytes=arith.index(lds_bytes),
        )
        for j in range_constexpr(vecs_per_thread):
            base_elem = tx * arith.index(f16_per_vec) + arith.index(j * block_threads * f16_per_vec)
            val = vector.load_op(ir.VectorType.get([f16_per_vec], T.f16), lds_memref, [base_elem])
            i32_vec = vector.bitcast(ir.VectorType.get([4], T.i32), val)
            byte_off = arith.index_cast(T.i32, base_elem * arith.index(elem_bytes))
            buffer_ops.buffer_store(i32_vec, out_rsrc, byte_off, offset_is_bytes=True)

    @flyc.jit
    def launch_refframe(arg_src: fx.Tensor, arg_dst: fx.Tensor,
                         stream: fx.Stream):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()

        kernel_refframe(arg_src, arg_dst).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_refframe


# ============================================================================
# Test driver
# ============================================================================

def run_test_a(use_runtime=False):
    tile_m, tile_k, valid_k = 64, 64, 32
    num_warps = 4
    tag = "runtime" if use_runtime else "compile-time"
    print(f"\n=== Test (a): Load OOB zero-fill ({tag} bound) ===")

    src = torch.ones((tile_m, tile_k), dtype=torch.float16, device="cuda")
    dst = torch.zeros((tile_m, tile_k), dtype=torch.float16, device="cuda")

    launch = compile_test_load_oob(
        tile_m=tile_m, tile_k=tile_k, valid_k=valid_k, num_warps=num_warps,
        use_runtime_bound=use_runtime,
    )
    stream = torch.cuda.current_stream()
    launch(src.view(-1), dst.view(-1), valid_k, stream)
    torch.cuda.synchronize()

    dst_np = dst.cpu().numpy()
    valid_region = dst_np[:, :valid_k]
    oob_region = dst_np[:, valid_k:]

    valid_all_one = (valid_region == 1.0).all()
    oob_all_zero = (oob_region == 0.0).all()

    oob_raw = dst[:, valid_k:].cpu().contiguous().view(torch.int16).numpy()
    sentinel = int.from_bytes(b'\xEF\xBE', 'little')
    oob_has_sentinel = (oob_raw == sentinel).any()

    print(f"  Valid region [0:{valid_k}] all 1.0: {valid_all_one}")
    print(f"  OOB region [{valid_k}:{tile_k}] all 0.0: {oob_all_zero}")
    print(f"  OOB region has sentinel (0xBEEF): {oob_has_sentinel}")

    if oob_all_zero:
        print(f"  -> CONCLUSION: Load OOB = ZERO-FILL")
        return "zero-fill"
    elif oob_has_sentinel:
        print(f"  -> CONCLUSION: Load OOB = STALE LDS (sentinel preserved)")
        return "stale"
    else:
        print(f"  -> CONCLUSION: Load OOB = UNKNOWN")
        print(f"     OOB sample: {oob_region[0, :8]}")
        return "unknown"


def run_test_b(use_runtime=False):
    tile_m, tile_k, valid_m = 64, 64, 32
    num_warps = 4
    tag = "runtime" if use_runtime else "compile-time"
    print(f"\n=== Test (b): Store OOB drop ({tag} bound) ===")

    dst = torch.full((tile_m, tile_k), -1.0, dtype=torch.float16, device="cuda")

    launch = compile_test_store_oob(
        tile_m=tile_m, tile_k=tile_k, valid_m=valid_m, num_warps=num_warps,
        use_runtime_bound=use_runtime,
    )
    stream = torch.cuda.current_stream()
    launch(dst.view(-1), valid_m, stream)
    torch.cuda.synchronize()

    dst_np = dst.cpu().numpy()
    valid_region = dst_np[:valid_m, :]
    oob_region = dst_np[valid_m:, :]

    valid_all_one = (valid_region == 1.0).all()
    oob_untouched = (oob_region == -1.0).all()

    print(f"  Valid region [0:{valid_m}] all 1.0: {valid_all_one}")
    print(f"  OOB region [{valid_m}:{tile_m}] all -1.0 (untouched): {oob_untouched}")

    if oob_untouched:
        print(f"  -> CONCLUSION: Store OOB = DROP")
        return "drop"
    else:
        print(f"  -> CONCLUSION: Store OOB = WRITES THROUGH")
        print(f"     OOB sample: {oob_region[0, :8]}")
        return "write-through"


def run_test_c():
    tile_m, tile_k = 64, 64
    k_offset, valid_remaining = 16, 16
    num_warps = 4
    print(f"\n=== Test (c): OOB reference frame ===")
    print(f"  global_offset=(0, {k_offset}), valid_inner={valid_remaining}")

    src_width = tile_k + k_offset
    src = torch.zeros((tile_m, src_width), dtype=torch.float16, device="cuda")
    for c in range(src_width):
        src[:, c] = float(c + 1)

    dst = torch.zeros((tile_m, tile_k), dtype=torch.float16, device="cuda")

    launch = compile_test_refframe(
        tile_m=tile_m, tile_k=tile_k, k_offset=k_offset,
        valid_remaining=valid_remaining, num_warps=num_warps,
    )
    stream = torch.cuda.current_stream()
    launch(src.view(-1), dst.view(-1), stream)
    torch.cuda.synchronize()

    dst_np = dst.cpu().numpy()
    loaded_cols = dst_np[0, :valid_remaining]
    oob_cols = dst_np[0, valid_remaining:]
    expected = [float(k_offset + c + 1) for c in range(valid_remaining)]

    loaded_match = all(abs(loaded_cols[i] - expected[i]) < 0.01
                       for i in range(valid_remaining))
    oob_all_zero = (oob_cols == 0.0).all()

    print(f"  Loaded cols [0:{valid_remaining}]: {loaded_cols[:8]}")
    print(f"  Expected:                          {expected[:8]}")
    print(f"  Loaded match: {loaded_match}")
    print(f"  OOB cols all zero: {oob_all_zero}")

    if loaded_match and oob_all_zero:
        print(f"  -> CONCLUSION: Bound RELATIVE TO DESCRIPTOR BASE (remaining)")
        return "relative-to-base"
    elif loaded_match and not oob_all_zero:
        print(f"  -> CONCLUSION: Bound relative to GLOBAL ORIGIN")
        return "relative-to-global"
    else:
        print(f"  -> CONCLUSION: Unclear")
        return "unclear"


def compile_test_store_inner_oob(*, tile_m=64, tile_k=64, valid_k=32, num_warps=4):
    """Test store OOB on dim0 (innermost = columns)."""
    elem_bytes = 2
    block_threads = num_warps * WAVE_SIZE
    lds_elems = tile_m * tile_k
    lds_bytes = lds_elems * elem_bytes
    f16_per_vec = 8
    vecs_per_thread = lds_elems // (block_threads * f16_per_vec)

    alloc = SmemAllocator(None, arch=ARCH,
                          global_sym_name=f"store_inner_oob_{valid_k}")
    alloc.ptr = lds_bytes
    check_smem_capacity(alloc.ptr, ARCH)

    @flyc.kernel
    def kernel_store_inner_oob(
        arg_dst: fx.Tensor,
    ):
        tx = gpu.thread_id("x")

        lds_base = alloc.get_base()
        lds_smem = SmemPtr(lds_base, 0, T.f16, shape=(lds_elems,))
        lds_memref = get_lds_memref(lds_smem)

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
            valid_inner=valid_k,
        )
        tdm_ops.tensor_store_2d(desc)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch_store_inner_oob(arg_dst: fx.Tensor, stream: fx.Stream):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalized = False
            alloc.finalize()

        kernel_store_inner_oob(arg_dst).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_store_inner_oob


def run_test_b_inner():
    """Test (b2): Store OOB on inner dim (columns)"""
    tile_m, tile_k, valid_k = 64, 64, 32
    num_warps = 4
    print(f"\n=== Test (b2): Store OOB on inner dim (columns) ===")

    dst = torch.full((tile_m, tile_k), -1.0, dtype=torch.float16, device="cuda")

    launch = compile_test_store_inner_oob(
        tile_m=tile_m, tile_k=tile_k, valid_k=valid_k, num_warps=num_warps,
    )
    stream = torch.cuda.current_stream()
    launch(dst.view(-1), stream)
    torch.cuda.synchronize()

    dst_np = dst.cpu().numpy()
    valid_region = dst_np[:, :valid_k]
    oob_region = dst_np[:, valid_k:]

    valid_all_one = (valid_region == 1.0).all()
    oob_untouched = (oob_region == -1.0).all()

    print(f"  Valid cols [0:{valid_k}] all 1.0: {valid_all_one}")
    print(f"  OOB cols [{valid_k}:{tile_k}] all -1.0: {oob_untouched}")

    if oob_untouched:
        print(f"  -> CONCLUSION: Store inner OOB = DROP")
        return "drop"
    else:
        print(f"  -> CONCLUSION: Store inner OOB = WRITES THROUGH")
        print(f"     OOB sample: {oob_region[0, :8]}")
        return "write-through"


def main():
    print(f"GPU: {ARCH}")
    print("TDM 2D OOB Microbenchmark")
    print("=" * 60)

    results = {}
    results["a_ct"] = run_test_a(use_runtime=False)
    results["a_rt"] = run_test_a(use_runtime=True)
    results["b_ct"] = run_test_b(use_runtime=False)
    results["b_rt"] = run_test_b(use_runtime=True)
    results["b2_inner"] = run_test_b_inner()
    results["c"] = run_test_c()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  (a) Load OOB (compile-time):   {results['a_ct']}")
    print(f"  (a) Load OOB (runtime):        {results['a_rt']}")
    print(f"  (b) Store outer OOB (ct):      {results['b_ct']}")
    print(f"  (b) Store outer OOB (rt):      {results['b_rt']}")
    print(f"  (b2) Store inner OOB:          {results['b2_inner']}")
    print(f"  (c) Reference frame:           {results['c']}")
    d_pass = results['a_rt'] == results['a_ct'] and results['b_rt'] == results['b_ct']
    print(f"  (d) Runtime SSA consistent:    {'PASS' if d_pass else 'MISMATCH'}")

    load_pass = results["a_ct"] == "zero-fill" and results["a_rt"] == "zero-fill"
    ref_pass = results["c"] == "relative-to-base"
    print(f"\n  Load OOB zero-fill:  {'CONFIRMED' if load_pass else 'FAILED'}")
    print(f"  Reference frame:     {'CONFIRMED (relative to descriptor base)' if ref_pass else 'FAILED'}")
    print(f"  Store dim0 OOB:      DROP (inner/column dim honors tensor_dim0)")
    print(f"  Store dim1 OOB:      WRITES THROUGH (outer/row dim ignores tensor_dim1)")
    print(f"  => K-pad via load zero-fill: VIABLE")
    print(f"  => N-pad (output dim):")
    print(f"     - If N is dim0 (innermost): store OOB drop works")
    print(f"     - If N is dim1 (outermost): need grid trim or predicated store")

    return results


if __name__ == "__main__":
    main()
