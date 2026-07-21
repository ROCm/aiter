# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""EP global->local expert LUT build (FlyDSL), single-block parallel scan.

Collapses the host ``ne + cumsum + sub + where`` chain (6 elementwise/scan
launches) into one kernel: read the (E_global,) 0/1 ``expert_mask``, do an
inclusive Hillis-Steele prefix sum in LDS, and write
``g2l_lut[i] = mask[i] ? prefix_incl[i]-1 : E`` (sentinel ``E`` = dropped route).

Mirrors ``moe_contiguous_psum`` (same single-block scan idiom) so the whole
gfx1250 grouped path stays on one compiler/runtime instead of pulling Triton
into the decode hot path. E_global fits in a single workgroup for supported
models; larger masks fall back to torch (see grouped_moe_gfx1250).

Also zero-inits the ``(E,)`` per-bucket route counter as a side output, folding
the separate host ``torch.zeros(E)`` launch (that ``moe_route_g2l`` atomically
increments) into this same pre-route kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate, _to_raw as _raw
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels.tensor_shim import (
    STensor,
    ptr_rsrc,
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

MAX_G2L_EXPERTS = 512


def build_moe_g2l_lut_module():
    """JIT launcher: single-block build of the EP global->local expert LUT."""

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="moe_g2l_lut_smem")
    lds0_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds0_off + MAX_G2L_EXPERTS * 4
    lds1_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds1_off + MAX_G2L_EXPERTS * 4

    @flyc.kernel(name="moe_g2l_lut", known_block_size=[MAX_G2L_EXPERTS, 1, 1])
    def g2l_kernel(
        mask: fx.Pointer,  # (n,) int32 0/1 expert mask
        lut: fx.Pointer,  # (n,) int32 out: global->local, sentinel E
        counter: fx.Pointer,  # (E,) int32 out: per-bucket route counter, zeroed
        nvt: fx.Pointer,  # (1,) int32 in: num_local_tokens (= total_recv)
        nvr_out: fx.Pointer,  # (1,) int32 out: num_valid_routes = nvt * topk
        n: Int32,
        E: Int32,
        topk: Int32,
    ):
        i32 = T.i32
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        tid = ArithValue(fx.thread_idx.x)

        # Fold num_valid_routes = num_local_tokens * topk (the EP dead-tail route
        # bound) into this pre-route single-block kernel: thread 0 reads the (1,)
        # int32 total_recv scalar and writes nvt*topk to nvr_out. This removes the
        # standalone torch ``_ep_nvt * topk`` elementwise launch on the decode hot
        # path; the route / psum-remap / quant kernels consume nvr_out as-is.
        is_t0 = arith.cmpi(CmpIPredicate.eq, tid, c0)
        _if_nvr = scf.IfOp(is_t0)
        with ir.InsertionPoint(_if_nvr.then_block):
            nvt_val = buffer_ops.buffer_load(
                ptr_rsrc(nvt), c0, vec_width=1, dtype=i32
            )
            nvr_val = arith.muli(nvt_val, _raw(ArithValue(topk)))
            buffer_ops.buffer_store(nvr_val, ptr_rsrc(nvr_out), c0)
            scf.YieldOp([])

        # Fold the route atomic-counter zero-init into this pre-route kernel:
        # bucket count E <= n <= block size, so thread tid<E clears counter[tid],
        # removing the separate host torch.zeros(E) launch before moe_route_g2l.
        in_bucket = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(E))
        _if_ctr = scf.IfOp(in_bucket)
        with ir.InsertionPoint(_if_ctr.then_block):
            buffer_ops.buffer_store(c0, ptr_rsrc(counter), tid)
            scf.YieldOp([])

        lds_base = allocator.get_base()
        lds0 = STensor(
            SmemPtr(lds_base, lds0_off, T.i32, shape=(MAX_G2L_EXPERTS,)),
            dtype=T.i32,
            shape=(MAX_G2L_EXPERTS,),
        )
        lds1 = STensor(
            SmemPtr(lds_base, lds1_off, T.i32, shape=(MAX_G2L_EXPERTS,)),
            dtype=T.i32,
            shape=(MAX_G2L_EXPERTS,),
        )

        m_rsrc = ptr_rsrc(mask)
        l_rsrc = ptr_rsrc(lut)

        in_range = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(n))

        # Load 0/1 into LDS.
        _if_load = scf.IfOp(in_range)
        with ir.InsertionPoint(_if_load.then_block):
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            nz = arith.cmpi(CmpIPredicate.ne, m, c0)
            lds0[fx.Index(tid)] = ArithValue(arith.select(nz, c1, c0))
            scf.YieldOp([])

        gpu.barrier()

        # Inclusive Hillis-Steele scan (identical to moe_contiguous_psum).
        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_G2L_EXPERTS):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_range)
            with ir.InsertionPoint(_if_scan.then_block):
                val = src[fx.Index(tid)]
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = src[fx.Index(tid - arith.constant(offset, type=i32))]
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([c0])
                dst[fx.Index(tid)] = ArithValue(val) + ArithValue(prev_if.results[0])
                scf.YieldOp([])
            gpu.barrier()
            src, dst = dst, src

        # lut[i] = enabled ? incl_prefix[i]-1 : E
        _if_store = scf.IfOp(in_range)
        with ir.InsertionPoint(_if_store.then_block):
            incl = src[fx.Index(tid)]
            m2 = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            nz2 = arith.cmpi(CmpIPredicate.ne, m2, c0)
            local = ArithValue(incl) - c1
            le = arith.select(nz2, _raw(local), _raw(ArithValue(E)))
            buffer_ops.buffer_store(ArithValue(le), l_rsrc, tid)
            scf.YieldOp([])

    @flyc.jit
    def launch_g2l(
        mask: fx.Pointer,
        lut: fx.Pointer,
        counter: fx.Pointer,
        nvt: fx.Pointer,
        nvr_out: fx.Pointer,
        n: fx.Int32,
        E: fx.Int32,
        topk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        g2l_kernel(mask, lut, counter, nvt, nvr_out, n, E, topk).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_G2L_EXPERTS, 1, 1),
            stream=stream,
        )

    launch_g2l.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_g2l
