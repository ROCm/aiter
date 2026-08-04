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
from flydsl.expr import const_expr, gpu, range_constexpr
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

MAX_G2L_EXPERTS = 512


def _lds_load(mr, idx):
    """Load i32 from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(mr + fx.Int64(idx))


def _lds_store(mr, val, idx):
    """Store i32 to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, mr + fx.Int64(idx))


def build_moe_g2l_lut_module():
    """JIT launcher: single-block build of the EP global->local expert LUT."""

    # Double-buffered LDS for the Hillis-Steele scan (ping-pong between passes).
    @fx.struct
    class SharedStorage:
        buf0: fx.Array[fx.Int32, MAX_G2L_EXPERTS, 16]
        buf1: fx.Array[fx.Int32, MAX_G2L_EXPERTS, 16]

    @flyc.kernel(name="moe_g2l_lut", known_block_size=[MAX_G2L_EXPERTS, 1, 1])
    def g2l_kernel(
        mask: fx.Pointer,  # (n,) int32 0/1 expert mask
        lut: fx.Pointer,  # (n,) int32 out: global->local, sentinel E
        counter: fx.Pointer,  # (E,) int32 out: per-bucket route counter, zeroed
        nvt: fx.Pointer,  # (1,) int32 in: num_local_tokens (= total_recv)
        nvr_out: fx.Pointer,  # (1,) int32 out: num_valid_routes = nvt * topk
        n: fx.Int32,
        E: fx.Int32,
        topk: fx.Int32,
    ):
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        tid = gpu.thread_idx.x

        m_rsrc = ptr_rsrc(mask)
        l_rsrc = ptr_rsrc(lut)

        # Fold num_valid_routes = num_local_tokens * topk (the EP dead-tail route
        # bound) into this pre-route single-block kernel: thread 0 reads the (1,)
        # int32 total_recv scalar and writes nvt*topk to nvr_out. This removes the
        # standalone torch ``_ep_nvt * topk`` elementwise launch on the decode hot
        # path; the route / psum-remap / quant kernels consume nvr_out as-is.
        if tid == c0:
            nvt_val = buffer_ops.buffer_load(
                ptr_rsrc(nvt), c0, vec_width=1, dtype=T.i32
            )
            buffer_ops.buffer_store(nvt_val * topk, ptr_rsrc(nvr_out), c0)

        # Fold the route atomic-counter zero-init into this pre-route kernel:
        # bucket count E <= n <= block size, so thread tid<E clears counter[tid],
        # removing the separate host torch.zeros(E) launch before moe_route_g2l.
        if tid < E:
            buffer_ops.buffer_store(c0, ptr_rsrc(counter), tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mr0 = lds.buf0.ptr
        mr1 = lds.buf1.ptr

        in_range = tid < n

        # Load 0/1 into LDS.
        if in_range:
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=T.i32)
            _lds_store(mr0, (m != c0).select(c1, c0), tid)

        gpu.barrier()

        # Inclusive Hillis-Steele scan (identical to moe_contiguous_psum).
        src, dst = mr0, mr1
        for offset in range_constexpr(1, MAX_G2L_EXPERTS):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_range:
                val = _lds_load(src, tid)
                has_prev = tid >= fx.Int32(offset)
                rd_idx = has_prev.select(tid - fx.Int32(offset), tid)
                prev = has_prev.select(_lds_load(src, rd_idx), c0)
                _lds_store(dst, val + prev, tid)
            gpu.barrier()
            src, dst = dst, src

        # lut[i] = enabled ? incl_prefix[i]-1 : E
        if in_range:
            incl = _lds_load(src, tid)
            m2 = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=T.i32)
            local = incl - c1
            buffer_ops.buffer_store((m2 != c0).select(local, E), l_rsrc, tid)

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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        g2l_kernel(mask, lut, counter, nvt, nvr_out, n, E, topk).launch(
            grid=(1, 1, 1),
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
