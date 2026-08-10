# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort.

Computes topids_to_rows (route -> grouped row) and rows_to_tokens (inverse)
via per-expert atomicAdd. One thread per route, no host-side argsort.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

from .buffer_view import (  # noqa: F401
    make_buffer,
    buffer_store,
    buffer_load,
)

_make_buffer = make_buffer
_buffer_store = buffer_store
_buffer_load = buffer_load

BLOCK_THREADS = 256
# Single-workgroup scan ceiling for the fused g2l+route kernel (matches
# moe_g2l_lut.MAX_G2L_EXPERTS): E_global must fit one block.
MAX_G2L_EXPERTS = 512
# Compile-time LDS bucket capacity for the two-level (LDS-reduce) route kernel.
# The per-block LDS counter array is sized to this; the dispatcher falls back to
# the plain device-atomic kernel when the local bucket count (E) exceeds it.
MAX_ROUTE_BUCKETS = 512


@fx.struct
class _RouteCntStorage:
    """LDS for the two-level route kernel.

    ``cnt`` is the per-bucket counter that routes bump with LDS atomics before
    a single device atomic publishes each bucket's base. The trailing 16 is the
    array's byte alignment.
    """

    cnt: fx.Array[fx.Int32, MAX_ROUTE_BUCKETS, 16]


@fx.struct
class _RouteG2LStorage:
    """LDS for the fused global->local LUT + route kernel.

    ``lds0``/``lds1`` are ping-pong buffers for the Hillis-Steele scan over the
    expert mask; ``lut`` holds the resulting global->local map, kept in LDS so
    the route phase needs no global LUT buffer.
    """

    lds0: fx.Array[fx.Int32, MAX_G2L_EXPERTS, 16]
    lds1: fx.Array[fx.Int32, MAX_G2L_EXPERTS, 16]
    lut: fx.Array[fx.Int32, MAX_G2L_EXPERTS, 16]


def _slot_ptr(base_i64, elem_idx, address_space=1):
    """Raw LLVM pointer to i32 element ``elem_idx`` of the buffer at ``base_i64``.

    The atomicrmw builder is a raw op and needs an ``!llvm.ptr<n>``; everything
    up to that boundary stays on the fx pointer surface.
    """
    space = (
        fx.AddressSpace.Shared if address_space == 3 else fx.AddressSpace.Global
    )
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, space, 4)
    return fx.to_llvm_ptr(fx.add_offset(fx.inttoptr(ptr_ty, base_i64), elem_idx))


def _lds_load(ptr, idx):
    """Scalar i32 load from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    """Scalar i32 store to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, ptr + fx.Int64(idx))


def build_moe_route_maps_module():
    """JIT launcher: builds topids_to_rows and rows_to_tokens in one pass."""

    @flyc.kernel(name="moe_route_maps")
    def route_maps_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32
        atomic_buffer: fx.Pointer,  # (E,) int32, init 0
        topids_to_rows: fx.Pointer,  # (numel,) int32 out: route -> grouped row
        rows_to_tokens: fx.Pointer,  # (E*max_m,) int32 out: grouped row -> token
        numel: Int32,
        topk: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        # Raw i32 constant: llvm.atomicrmw takes ir.Value operands, not fx types.
        c1_i32 = arith.constant(1, type=i32)
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + fx.Uint32(fx.thread_idx.x)
        in_range = route < fx.Uint32(numel)
        if in_range:
            topk_buf = _make_buffer(topk_ids, fx.Int32)
            c_buf = _make_buffer(topids_to_rows, fx.Int32)
            a_buf = _make_buffer(rows_to_tokens, fx.Int32)

            e = _buffer_load(topk_buf, route, fx.Int32)

            ptr = _slot_ptr(fx.Int64(ptrtoint(atomic_buffer)), e)
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                c1_i32,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

            row = fx.Uint32(slot) + fx.Uint32(e) * fx.Uint32(max_m)
            _buffer_store(c_buf, route, row, fx.Int32)
            token = route // fx.Uint32(topk)
            _buffer_store(a_buf, row, token, fx.Int32)

    @flyc.jit
    def launch_route_maps(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        rows_to_tokens: fx.Pointer,
        numel: fx.Int32,
        topk: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        launch = route_maps_kernel(
            topk_ids, atomic_buffer, topids_to_rows, rows_to_tokens, numel, topk, max_m
        )
        launch.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_route_maps.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_maps


def build_moe_topids_to_rows_module():
    """JIT launcher: builds topids_to_rows only (no rows_to_tokens inverse)."""

    @flyc.kernel(name="moe_route")
    def route_kernel(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        # Raw i32 constant: llvm.atomicrmw takes ir.Value operands, not fx types.
        c1_i32 = arith.constant(1, type=i32)
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + fx.Uint32(fx.thread_idx.x)
        in_range = route < fx.Uint32(numel)
        if in_range:
            topk_buf = _make_buffer(topk_ids, fx.Int32)
            out_buf = _make_buffer(topids_to_rows, fx.Int32)

            e = _buffer_load(topk_buf, route, fx.Int32)
            ptr = _slot_ptr(fx.Int64(ptrtoint(atomic_buffer)), e)
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                c1_i32,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + fx.Uint32(e) * fx.Uint32(max_m)
            _buffer_store(out_buf, route, row, fx.Int32)

    @flyc.jit
    def launch_topids_to_rows(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        launch = route_kernel(topk_ids, atomic_buffer, topids_to_rows, numel, max_m)
        launch.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_topids_to_rows.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_topids_to_rows


def build_moe_topids_to_rows_g2l_module(weight_dtype="bf16"):
    """topids_to_rows with a fused EP global->local expert remap.

    ``topk_ids`` holds GLOBAL expert ids; ``g2l_lut[global_id]`` gives the local
    bucket in [0, n_route_buckets) for enabled experts, or the sentinel value
    ``n_route_buckets`` for dropped (non-local) routes. Dropped routes are folded
    into bucket 0 (matching the previous host behaviour: they still take a unique
    atomic slot so they never collide with a real row).

    The route weights are cast from f32 ``weight_in`` to ``gather_w`` in
    ``weight_dtype`` in the same pass (kept -> cast, dropped -> 0), folding the
    host ``topk_weight.to(bf16)`` copy and the dropped-weight ``masked_fill``.

    This replaces the host-side cumsum/index/eq/where/masked_fill chain with a
    single on-device pass, which is the dominant per-route launch cost at decode.
    """

    @flyc.kernel(name="moe_route_g2l")
    def route_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32 GLOBAL expert ids
        g2l_lut: fx.Pointer,  # (E_global,) int32 global->local, sentinel=n_buckets
        atomic_buffer: fx.Pointer,  # (n_buckets,) int32, init 0
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        weight_in: fx.Pointer,  # (numel,) f32 route weights in
        gather_w: fx.Pointer,  # (numel,) weight_dtype out; kept->cast, drops->0
        num_valid_routes: fx.Pointer,  # (1,) int32; routes >= this are the EP dead-tail (skipped)
        numel: Int32,
        max_m: Int32,
        n_buckets: Int32,  # sentinel value == dropped
    ):
        i32 = T.i32
        c0 = arith.constant(0, type=i32)
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + fx.Uint32(fx.thread_idx.x)
        # Dynamic EP token count: the dispatch buffer is padded to a static numel
        # but only the first ``num_valid_routes`` (= total_recv*topk) routes are
        # valid. Routes >= nvr are the dead-tail padding rows (rows >= total_recv)
        # and must not be written or contribute to the counter; leaving their
        # topids_to_rows/gather_w slots unwritten matches the fused single-block
        # kernel (every downstream consumer is bounded by the same nvr/nvt). When
        # truncation is disabled the caller passes numel here, so nothing is oob.
        nvr_buf = _make_buffer(num_valid_routes, fx.Int32)
        nvr = _buffer_load(nvr_buf, c0, fx.Int32)
        in_range = route < fx.Uint32(nvr)
        if in_range:
            topk_buf = _make_buffer(topk_ids, fx.Int32)
            g2l_buf = _make_buffer(g2l_lut, fx.Int32)
            out_buf = _make_buffer(topids_to_rows, fx.Int32)
            wi_buf = _make_buffer(weight_in, fx.Float32)
            w_buf = _make_buffer(gather_w, fx.Numeric.from_ir_type(wdt))

            ge = fx.Uint32(_buffer_load(topk_buf, route, fx.Int32))
            le = fx.Uint32(_buffer_load(g2l_buf, ge, fx.Int32))
            is_drop = le == fx.Uint32(n_buckets)
            # Dropped routes fold to bucket 0 but still take a unique slot.
            eff_e = is_drop.select(fx.Uint32(0), le)

            # Fused weight cast+mask: read f32 route weight, write weight_dtype
            # (kept -> cast, dropped -> 0). Folds the host topk_weight.to(bf16)
            # copy and the dropped-weight masked_fill into this route pass.
            w_f32 = _buffer_load(wi_buf, route, fx.Float32)
            w_cast = arith.trunc_f(wdt, _raw(w_f32))
            w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)
            _buffer_store(w_buf, route, fx.Numeric.from_ir_type(wdt)(w_out), fx.Numeric.from_ir_type(wdt))

            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                _slot_ptr(fx.Int64(ptrtoint(atomic_buffer)), eff_e),
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + eff_e * fx.Uint32(max_m)
            _buffer_store(out_buf, route, row, fx.Int32)

    @flyc.jit
    def launch_topids_to_rows_g2l(
        topk_ids: fx.Pointer,
        g2l_lut: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        weight_in: fx.Pointer,
        gather_w: fx.Pointer,
        num_valid_routes: fx.Pointer,
        numel: fx.Int32,
        max_m: fx.Int32,
        n_buckets: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        launch = route_kernel(
            topk_ids,
            g2l_lut,
            atomic_buffer,
            topids_to_rows,
            weight_in,
            gather_w,
            num_valid_routes,
            numel,
            max_m,
            n_buckets,
        )
        launch.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_topids_to_rows_g2l.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_topids_to_rows_g2l


def build_moe_route_g2l_lds_module(weight_dtype="bf16"):
    """Multi-block EP route with a two-level (LDS -> global) atomic reduction.

    The plain ``moe_route_g2l`` kernel does one device-scope ``atomicAdd`` per
    route on the ``(E,)`` counter. Under EP the dropped (non-local) routes all
    fold into bucket 0, so bucket 0 sees ~O(numel) serialized device atomics on a
    single address -- the route-phase bottleneck. This kernel instead:

      1. each block privately counts its routes per bucket via *workgroup-scope*
         LDS atomics (``lds_cnt[eff_e] += 1``), which are ~an order of magnitude
         cheaper and contend only within the block;
      2. one thread per non-empty bucket issues a *single* device-scope
         ``atomicAdd(counter[b], block_count[b])`` to claim the block's base
         offset (device atomics on bucket 0 drop from ~numel to ~grid_blocks);
      3. each route computes its final row = ``base[eff_e] + intra_block_rank +
         eff_e*max_m`` from the LDS base + the rank it got in step 1.

    Rows stay a per-bucket bijection (disjoint block bases, unique intra-block
    ranks), so ``topids_to_rows``/``counter`` match the plain kernel's contract
    (order within a bucket is unspecified, exactly as the plain multi-block
    device-atomic kernel). Requires ``E <= MAX_ROUTE_BUCKETS`` (LDS capacity);
    the caller falls back to the plain kernel otherwise.
    """

    @flyc.kernel(
        name="moe_route_g2l_lds",
        known_block_size=[BLOCK_THREADS, 1, 1],
    )
    def route_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32 GLOBAL expert ids
        g2l_lut: fx.Pointer,  # (E_global,) int32 global->local, sentinel=n_buckets
        atomic_buffer: fx.Pointer,  # (n_buckets,) int32, init 0 (== masked_m out)
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        weight_in: fx.Pointer,  # (numel,) f32 route weights in
        gather_w: fx.Pointer,  # (numel,) weight_dtype out; kept->cast, drops->0
        num_valid_routes: fx.Pointer,  # (1,) int32; routes >= this are the EP dead-tail
        numel: Int32,
        max_m: Int32,
        n_buckets: Int32,  # local bucket count / sentinel value; <= MAX_ROUTE_BUCKETS
    ):
        i32 = T.i32
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        tid = fx.Uint32(fx.thread_idx.x)
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + tid

        lds_cnt = fx.SharedAllocator().allocate(_RouteCntStorage).peek().cnt.ptr
        # The LDS atomic below needs a raw addrspace(3) pointer; SharedAllocator
        # has already folded the array's offset into this base.
        cnt_base_i64 = fx.Int64(fx.ptrtoint(lds_cnt))

        tk_buf = _make_buffer(topk_ids, fx.Int32)
        g2l_buf = _make_buffer(g2l_lut, fx.Int32)
        wi_buf = _make_buffer(weight_in, fx.Float32)
        w_buf = _make_buffer(gather_w, fx.Numeric.from_ir_type(wdt))
        out_buf = _make_buffer(topids_to_rows, fx.Int32)

        nvr_buf = _make_buffer(num_valid_routes, fx.Int32)
        nvr = _buffer_load(nvr_buf, c0, fx.Int32)

        n_buckets_i32 = fx.Uint32(n_buckets)
        nvr_i32 = fx.Uint32(nvr)

        # Phase 0: zero the per-block LDS bucket counter ([0, n_buckets)).
        for b in range(tid, n_buckets_i32, BLOCK_THREADS):
            _lds_store(lds_cnt, fx.Int32(0), fx.Uint32(b))
        gpu.barrier()

        # Phase 1: classify each route, cast/mask its weight, and take an
        # intra-block per-bucket rank via a workgroup-scope LDS atomic. Routes
        # >= nvr (EP dead-tail padding) are skipped: no LDS increment, and their
        # topids_to_rows/gather_w slots are left unwritten (every downstream
        # consumer is bounded by the same nvr/nvt), matching the fused kernel.
        in_range = route < nvr_i32
        oob = route >= nvr_i32

        # Load the global expert id only for valid routes (dead-tail rows may
        # carry -1/stale ids that would OOB-read g2l_lut); oob folds to 0.
        ge = fx.Uint32(0)
        if in_range:
            ge = fx.Uint32(_buffer_load(tk_buf, route, fx.Int32))

        le = fx.Uint32(_buffer_load(g2l_buf, ge, fx.Int32))
        is_drop = (le == n_buckets_i32) | oob
        eff_e = is_drop.select(fx.Uint32(0), le)

        # Fused weight cast+mask (kept -> cast(f32->wdt), dropped -> 0).
        w_f32 = fx.Float32(0.0)
        if in_range:
            w_f32 = fx.Float32(_buffer_load(wi_buf, route, fx.Float32))
        w_cast = arith.trunc_f(wdt, _raw(w_f32))
        w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)

        my_rank = fx.Uint32(0)
        if in_range:
            _buffer_store(
                w_buf, route, fx.Numeric.from_ir_type(wdt)(w_out),
                fx.Numeric.from_ir_type(wdt),
            )
            my_rank = fx.Uint32(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    _slot_ptr(cnt_base_i64, eff_e, address_space=3),
                    c1,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="workgroup",
                    alignment=4,
                ).result
            )

        gpu.barrier()

        # Phase 2: one device-scope atomic per non-empty bucket to claim this
        # block's base offset; overwrite the LDS count in place with the base.
        for b in range(tid, n_buckets_i32, BLOCK_THREADS):
            cnt = _lds_load(lds_cnt, fx.Uint32(b))
            nz = cnt != 0
            base_v = fx.Int32(0)
            if nz:
                base_v = fx.Int32(
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        _slot_ptr(fx.Int64(ptrtoint(atomic_buffer)), b),
                        _raw(cnt),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                )
            _lds_store(lds_cnt, base_v, fx.Uint32(b))
        gpu.barrier()

        # Phase 3: final row = base[eff_e] + intra-block rank + eff_e*max_m.
        if in_range:
            base = fx.Uint32(_lds_load(lds_cnt, eff_e))
            row = base + my_rank + eff_e * fx.Uint32(max_m)
            _buffer_store(out_buf, route, row, fx.Int32)

    @flyc.jit
    def launch_route_g2l_lds(
        topk_ids: fx.Pointer,
        g2l_lut: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        weight_in: fx.Pointer,
        gather_w: fx.Pointer,
        num_valid_routes: fx.Pointer,
        numel: fx.Int32,
        max_m: fx.Int32,
        n_buckets: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        route_kernel(
            topk_ids,
            g2l_lut,
            atomic_buffer,
            topids_to_rows,
            weight_in,
            gather_w,
            num_valid_routes,
            numel,
            max_m,
            n_buckets,
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_route_g2l_lds.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_g2l_lds


def build_moe_route_g2l_fused_module(weight_dtype="bf16"):
    """Single-block fused g2l-LUT build + EP route (contiguous path).

    Folds ``moe_g2l_lut`` and ``moe_route_g2l`` into one launch. A single block:
      1. builds the global->local LUT in LDS via a Hillis-Steele prefix scan over
         ``expert_mask`` (E_global 0/1) and zeros the ``(E,)`` route counter,
      2. barriers,
      3. grid-strides over routes: LDS LUT lookup -> local bucket (dropped folds
         to bucket 0), global atomicAdd slot, writes ``topids_to_rows`` and the
         cast/masked ``gather_w`` (f32 ``weight_in`` -> weight_dtype).

    The LUT is consumed only inside this kernel, so it never touches global
    memory (no g2l_lut buffer) and the separate moe_g2l_lut launch is removed.
    Requires E_global <= MAX_G2L_EXPERTS (single-workgroup scan); the caller
    falls back to the two-kernel path otherwise.
    """

    @flyc.kernel(
        name="moe_route_g2l_fused",
        known_block_size=[MAX_G2L_EXPERTS, 1, 1],
    )
    def route_fused_kernel(
        expert_mask: fx.Pointer,  # (n,) int32 0/1 global expert mask
        topk_ids: fx.Pointer,  # (numel,) int32 GLOBAL expert ids
        weight_in: fx.Pointer,  # (numel,) f32 route weights in
        counter: fx.Pointer,  # (E,) int32 out (== masked_m), zeroed then atomic'd
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        gather_w: fx.Pointer,  # (numel,) weight_dtype out; kept->cast, drop->0
        num_valid_routes: fx.Pointer,  # (1,) int32; routes >= this are treated as dropped (EP dynamic token count)
        n: Int32,  # E_global (mask length)
        numel: Int32,
        max_m: Int32,
        E: Int32,  # local bucket count / sentinel value
    ):
        i32 = T.i32
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        tid = fx.Uint32(fx.thread_idx.x)
        e_count = fx.Uint32(E)

        lds = fx.SharedAllocator().allocate(_RouteG2LStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        lds_lut = lds.lut.ptr

        m_buf = _make_buffer(expert_mask, fx.Int32)
        ctr_buf = _make_buffer(counter, fx.Int32)

        # Zero the (E,) route counter (global); barrier below orders it before the
        # phase-B atomics (single block, so no cross-block hazard).
        in_bucket = tid < e_count
        if in_bucket:
            _buffer_store(ctr_buf, tid, fx.Int32(0), fx.Int32)

        # Phase A: load mask -> 0/1 into LDS.
        in_range = tid < fx.Uint32(n)
        if in_range:
            m = _buffer_load(m_buf, tid, fx.Int32)
            nz = m != c0
            _lds_store(lds0, fx.Int32(nz.select(c1, c0)), tid)

        gpu.barrier()

        # Inclusive Hillis-Steele scan (identical to moe_g2l_lut).
        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_G2L_EXPERTS):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_range:
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
            gpu.barrier()
            src, dst = dst, src

        # lut[i] = enabled ? incl_prefix[i]-1 : E ; keep in LDS for phase B.
        if in_range:
            incl = _lds_load(src, tid)
            m2 = _buffer_load(m_buf, tid, fx.Int32)
            nz2 = m2 != c0
            _lds_store(lds_lut, nz2.select(fx.Uint32(incl) - 1, e_count), tid)

        gpu.barrier()

        # Phase B: grid-stride over routes.
        tk_buf = _make_buffer(topk_ids, fx.Int32)
        wi_buf = _make_buffer(weight_in, fx.Float32)
        out_buf = _make_buffer(topids_to_rows, fx.Int32)
        w_buf = _make_buffer(gather_w, fx.Numeric.from_ir_type(wdt))

        # Dynamic EP token count: routes >= num_valid_routes belong to dead-tail
        # padding rows of the dispatch buffer (rows >= total_recv) and must not
        # contribute. Load once and fold into the per-route "dropped" predicate so
        # they reuse the existing drop path (gather_w=0, folded to bucket 0). When
        # truncation is disabled the caller passes numel here, so nothing is oob.
        nvr_buf = _make_buffer(num_valid_routes, fx.Int32)
        nvr = _buffer_load(nvr_buf, c0, fx.Int32)

        # Iterate only the valid routes ([0, nvr)); the dead-tail padding routes
        # (>= num_valid_routes) are skipped entirely, so topids_to_rows/gather_w
        # for those slots are left unwritten. Every downstream consumer of the
        # route buffers (contiguous_psum_remap, preshuffle route-ksplit,
        # gather-reduce) is bounded by the same nvr/nvt, so the dead tail is never
        # read. When truncation is disabled the caller passes numel == nvr.
        nvr_i32 = fx.Uint32(nvr)
        for route in range(tid, nvr_i32, MAX_G2L_EXPERTS):
            is_oob = fx.Uint32(route) >= nvr_i32
            ge_raw = fx.Uint32(_buffer_load(tk_buf, route, fx.Int32))
            # Clamp oob routes' global id to 0 BEFORE the LDS LUT lookup: dead-tail
            # dispatch rows (route >= num_valid_routes) may carry -1 / stale garbage
            # expert ids, which would otherwise OOB-read lds_lut. oob is forced to
            # the drop path below regardless of the clamped lookup result.
            ge = is_oob.select(fx.Uint32(0), ge_raw)
            le = fx.Uint32(_lds_load(lds_lut, ge))
            is_drop = (le == e_count) | is_oob
            eff_e = is_drop.select(fx.Uint32(0), le)

            # Fused weight cast+mask: kept -> cast(f32->wdt), dropped -> 0.
            w_f32 = _buffer_load(wi_buf, route, fx.Float32)
            w_cast = arith.trunc_f(wdt, _raw(w_f32))
            w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)
            _buffer_store(
                w_buf, route, fx.Numeric.from_ir_type(wdt)(w_out),
                fx.Numeric.from_ir_type(wdt),
            )

            # oob (dead-tail) routes must NOT claim a real slot: incrementing the
            # counter would inflate masked_m[0] by the whole padding tail,
            # reshuffling the contiguous GEMM layout so valid rows land in cells
            # the masked GEMM never writes (grouped_out is uninitialised). Add 0
            # for oob so masked_m matches the trimmed (total_recv) case exactly;
            # the row then points at an already-written bucket-0 cell and folds
            # away via gather_w=0. Normal expert-mask drops still take a slot.
            incr = _raw(is_oob.select(c0, c1))
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                _slot_ptr(fx.Int64(ptrtoint(counter)), eff_e),
                incr,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + eff_e * fx.Uint32(max_m)
            _buffer_store(out_buf, route, row, fx.Int32)

    @flyc.jit
    def launch_route_g2l_fused(
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        weight_in: fx.Pointer,
        counter: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        num_valid_routes: fx.Pointer,
        n: fx.Int32,
        numel: fx.Int32,
        max_m: fx.Int32,
        E: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        route_fused_kernel(
            expert_mask,
            topk_ids,
            weight_in,
            counter,
            topids_to_rows,
            gather_w,
            num_valid_routes,
            n,
            numel,
            max_m,
            E,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_G2L_EXPERTS, 1, 1),
            stream=stream,
        )

    launch_route_g2l_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_g2l_fused
