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

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

BLOCK_THREADS = 256
# Single-workgroup scan ceiling for the fused g2l+route kernel (matches
# moe_g2l_lut.MAX_G2L_EXPERTS): E_global must fit one block.
MAX_G2L_EXPERTS = 512
# Compile-time LDS bucket capacity for the two-level (LDS-reduce) route kernel.
# The per-block LDS counter array is sized to this; the dispatcher falls back to
# the plain device-atomic kernel when the local bucket count (E) exceeds it.
MAX_ROUTE_BUCKETS = 512

# ``topids_to_rows`` value for an EP route that owns no grouped row: a non-local
# (expert-mask dropped) route, or a dead-tail padding route. Such routes claim no
# per-expert slot, so they contribute nothing to masked_m / psum / the grouped
# GEMM -- under EP that is the difference between computing every received
# route and only the routes this rank actually owns. Consumers of the route map
# (contiguous psum remap, the stage1 route-indexed quant scatter, gather-reduce)
# must skip this value instead of using it as a row index; ``gather_w`` is 0 for
# the same routes.
DROPPED_ROUTE_ROW = -1


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

    The atomicrmw builder needs a raw ``!llvm.ptr<n>``, which the layout/buffer
    ops do not produce, so the byte address is formed by hand here.
    """
    ptr = buffer_ops.create_llvm_ptr(
        base_i64 + fx.Int64(elem_idx) * 4, address_space=address_space
    )
    return ptr._value if hasattr(ptr, "_value") else ptr


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
            topk_rsrc = ptr_rsrc(topk_ids)
            c_rsrc = ptr_rsrc(topids_to_rows)
            a_rsrc = ptr_rsrc(rows_to_tokens)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)

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
            buffer_ops.buffer_store(row, c_rsrc, route)
            token = route // fx.Uint32(topk)
            buffer_ops.buffer_store(token, a_rsrc, row)

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
            topk_rsrc = ptr_rsrc(topk_ids)
            out_rsrc = ptr_rsrc(topids_to_rows)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
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
            buffer_ops.buffer_store(row, out_rsrc, route)

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
    ``n_route_buckets`` for dropped (non-local) routes. Dropped routes claim no
    atomic slot and are tagged with ``DROPPED_ROUTE_ROW``, so they never occupy a
    grouped row: ``atomic_buffer`` (== masked_m) counts local routes only.

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
        f32 = T.f32
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        dropped_row = arith.constant(DROPPED_ROUTE_ROW, type=i32)
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + fx.Uint32(fx.thread_idx.x)
        # Dynamic EP token count: the dispatch buffer is padded to a static numel
        # but only the first ``num_valid_routes`` (= total_recv*topk) routes are
        # valid. Routes >= nvr are the dead-tail padding rows (rows >= total_recv)
        # and must not be written or contribute to the counter; leaving their
        # topids_to_rows/gather_w slots unwritten matches the fused single-block
        # kernel (every downstream consumer is bounded by the same nvr/nvt). When
        # truncation is disabled the caller passes numel here, so nothing is oob.
        nvr_rsrc = ptr_rsrc(num_valid_routes)
        nvr = buffer_ops.buffer_load(nvr_rsrc, c0, vec_width=1, dtype=i32)
        in_range = route < fx.Uint32(nvr)
        if in_range:
            topk_rsrc = ptr_rsrc(topk_ids)
            g2l_rsrc = ptr_rsrc(g2l_lut)
            out_rsrc = ptr_rsrc(topids_to_rows)
            wi_rsrc = ptr_rsrc(weight_in)
            w_rsrc = ptr_rsrc(gather_w)

            ge = fx.Uint32(
                buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
            )
            le = fx.Uint32(buffer_ops.buffer_load(g2l_rsrc, ge, vec_width=1, dtype=i32))
            is_drop = le == fx.Uint32(n_buckets)
            # Dropped routes address bucket 0 so the atomic stays in bounds, but
            # they add 0 to it (see incr below) and never keep the row.
            eff_e = is_drop.select(fx.Uint32(0), le)

            # Fused weight cast+mask: read f32 route weight, write weight_dtype
            # (kept -> cast, dropped -> 0). Folds the host topk_weight.to(bf16)
            # copy and the dropped-weight masked_fill into this route pass.
            w_f32 = buffer_ops.buffer_load(wi_rsrc, route, vec_width=1, dtype=f32)
            w_cast = arith.trunc_f(wdt, w_f32)
            w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)
            buffer_ops.buffer_store(w_out, w_rsrc, route)

            # Non-local routes take no slot (incr 0), so masked_m -- and with it
            # psum, the contiguous row count and the grouped GEMM's M -- covers
            # only the routes whose expert lives on this rank. They get the
            # DROPPED_ROUTE_ROW sentinel instead of a row that would alias the
            # bucket-0 route holding the same slot.
            incr = _raw(is_drop.select(c0, c1))
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                _slot_ptr(fx.Int64(ptrtoint(atomic_buffer)), eff_e),
                incr,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + eff_e * fx.Uint32(max_m)
            row_out = arith.select(_raw(is_drop), dropped_row, _raw(row))
            buffer_ops.buffer_store(row_out, out_rsrc, route)

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
    route on the ``(E,)`` counter, so a bucket that many routes land in serializes
    those atomics on a single address -- the route-phase bottleneck. This kernel
    instead:

      1. each block privately counts its *kept* routes per bucket via
         *workgroup-scope* LDS atomics (``lds_cnt[eff_e] += 1``), which are ~an
         order of magnitude cheaper and contend only within the block;
      2. one thread per non-empty bucket issues a *single* device-scope
         ``atomicAdd(counter[b], block_count[b])`` to claim the block's base
         offset (device atomics drop from ~numel to ~grid_blocks per bucket);
      3. each kept route computes its final row = ``base[eff_e] +
         intra_block_rank + eff_e*max_m`` from the LDS base + the rank it got in
         step 1; dropped routes get the ``DROPPED_ROUTE_ROW`` sentinel.

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
        f32 = T.f32
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        dropped_row = arith.constant(DROPPED_ROUTE_ROW, type=i32)
        tid = fx.Uint32(fx.thread_idx.x)
        route = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + tid

        lds_cnt = fx.SharedAllocator().allocate(_RouteCntStorage).peek().cnt.ptr
        # The LDS atomic below needs a raw addrspace(3) pointer; SharedAllocator
        # has already folded the array's offset into this base.
        cnt_base_i64 = fx.Int64(fx.ptrtoint(lds_cnt))

        tk_rsrc = ptr_rsrc(topk_ids)
        g2l_rsrc = ptr_rsrc(g2l_lut)
        wi_rsrc = ptr_rsrc(weight_in)
        w_rsrc = ptr_rsrc(gather_w)
        out_rsrc = ptr_rsrc(topids_to_rows)

        nvr_rsrc = ptr_rsrc(num_valid_routes)
        nvr = buffer_ops.buffer_load(nvr_rsrc, c0, vec_width=1, dtype=i32)

        n_buckets_i32 = fx.Uint32(n_buckets)
        nvr_i32 = fx.Uint32(nvr)

        # Phase 0: zero the per-block LDS bucket counter ([0, n_buckets)).
        for b in range(tid, n_buckets_i32, BLOCK_THREADS):
            _lds_store(lds_cnt, fx.Int32(0), fx.Uint32(b))
        gpu.barrier()

        # Phase 1: classify each route, cast/mask its weight, and take an
        # intra-block per-bucket rank via a workgroup-scope LDS atomic. Only KEPT
        # (local, in-range) routes take a rank, so the buckets -- and therefore
        # masked_m, psum and the grouped GEMM's row count -- hold this rank's own
        # routes only. Routes >= nvr (EP dead-tail padding) are skipped entirely:
        # no LDS increment, and their topids_to_rows/gather_w slots are left
        # unwritten (every downstream consumer is bounded by the same nvr/nvt),
        # matching the fused kernel.
        in_range = route < nvr_i32
        oob = route >= nvr_i32

        # Load the global expert id only for valid routes (dead-tail rows may
        # carry -1/stale ids that would OOB-read g2l_lut); oob folds to 0.
        ge = fx.Uint32(0)
        if in_range:
            ge = fx.Uint32(
                buffer_ops.buffer_load(tk_rsrc, route, vec_width=1, dtype=i32)
            )

        le = fx.Uint32(buffer_ops.buffer_load(g2l_rsrc, ge, vec_width=1, dtype=i32))
        is_drop = (le == n_buckets_i32) | oob
        # Kept == in-range AND local. ``oob`` routes read g2l_lut[0], which can be
        # a valid bucket, so in_range has to gate the rank as well as the lookup.
        is_kept = (le != n_buckets_i32) & in_range
        eff_e = is_drop.select(fx.Uint32(0), le)

        # Fused weight cast+mask (kept -> cast(f32->wdt), dropped -> 0).
        w_f32 = fx.Float32(0.0)
        if in_range:
            w_f32 = fx.Float32(
                buffer_ops.buffer_load(wi_rsrc, route, vec_width=1, dtype=f32)
            )
        w_cast = arith.trunc_f(wdt, _raw(w_f32))
        w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)

        if in_range:
            buffer_ops.buffer_store(w_out, w_rsrc, route)

        my_rank = fx.Uint32(0)
        if is_kept:
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

        # Phase 3: final row = base[eff_e] + intra-block rank + eff_e*max_m for a
        # kept route; the sentinel for a dropped one (it never claimed a rank, so
        # the computed row would alias the bucket's rank-0 route).
        if in_range:
            base = fx.Uint32(_lds_load(lds_cnt, eff_e))
            row = base + my_rank + eff_e * fx.Uint32(max_m)
            row_out = arith.select(_raw(is_drop), dropped_row, _raw(row))
            buffer_ops.buffer_store(row_out, out_rsrc, route)

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
      3. grid-strides over routes: LDS LUT lookup -> local bucket, global
         atomicAdd slot for kept routes (dropped ones get ``DROPPED_ROUTE_ROW``),
         writes ``topids_to_rows`` and the cast/masked ``gather_w`` (f32
         ``weight_in`` -> weight_dtype).

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
        f32 = T.f32
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        dropped_row = arith.constant(DROPPED_ROUTE_ROW, type=i32)
        tid = fx.Uint32(fx.thread_idx.x)
        e_count = fx.Uint32(E)

        lds = fx.SharedAllocator().allocate(_RouteG2LStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        lds_lut = lds.lut.ptr

        m_rsrc = ptr_rsrc(expert_mask)
        ctr_rsrc = ptr_rsrc(counter)

        # Zero the (E,) route counter (global); barrier below orders it before the
        # phase-B atomics (single block, so no cross-block hazard).
        in_bucket = tid < e_count
        if in_bucket:
            buffer_ops.buffer_store(c0, ctr_rsrc, tid)

        # Phase A: load mask -> 0/1 into LDS.
        in_range = tid < fx.Uint32(n)
        if in_range:
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
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
            m2 = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            nz2 = m2 != c0
            _lds_store(lds_lut, nz2.select(fx.Uint32(incl) - 1, e_count), tid)

        gpu.barrier()

        # Phase B: grid-stride over routes.
        tk_rsrc = ptr_rsrc(topk_ids)
        wi_rsrc = ptr_rsrc(weight_in)
        out_rsrc = ptr_rsrc(topids_to_rows)
        w_rsrc = ptr_rsrc(gather_w)

        # Dynamic EP token count: routes >= num_valid_routes belong to dead-tail
        # padding rows of the dispatch buffer (rows >= total_recv) and must not
        # contribute. Load once and fold into the per-route "dropped" predicate so
        # they reuse the existing drop path (gather_w=0, folded to bucket 0). When
        # truncation is disabled the caller passes numel here, so nothing is oob.
        nvr_rsrc = ptr_rsrc(num_valid_routes)
        nvr = buffer_ops.buffer_load(nvr_rsrc, c0, vec_width=1, dtype=i32)

        # Iterate only the valid routes ([0, nvr)); the dead-tail padding routes
        # (>= num_valid_routes) are skipped entirely, so topids_to_rows/gather_w
        # for those slots are left unwritten. Every downstream consumer of the
        # route buffers (contiguous_psum_remap, preshuffle route-ksplit,
        # gather-reduce) is bounded by the same nvr/nvt, so the dead tail is never
        # read. When truncation is disabled the caller passes numel == nvr.
        nvr_i32 = fx.Uint32(nvr)
        for route in range(tid, nvr_i32, MAX_G2L_EXPERTS):
            is_oob = fx.Uint32(route) >= nvr_i32
            ge_raw = fx.Uint32(
                buffer_ops.buffer_load(tk_rsrc, route, vec_width=1, dtype=i32)
            )
            # Clamp oob routes' global id to 0 BEFORE the LDS LUT lookup: dead-tail
            # dispatch rows (route >= num_valid_routes) may carry -1 / stale garbage
            # expert ids, which would otherwise OOB-read lds_lut. oob is forced to
            # the drop path below regardless of the clamped lookup result.
            ge = is_oob.select(fx.Uint32(0), ge_raw)
            le = fx.Uint32(_lds_load(lds_lut, ge))
            is_drop = (le == e_count) | is_oob
            eff_e = is_drop.select(fx.Uint32(0), le)

            # Fused weight cast+mask: kept -> cast(f32->wdt), dropped -> 0.
            w_f32 = buffer_ops.buffer_load(wi_rsrc, route, vec_width=1, dtype=f32)
            w_cast = arith.trunc_f(wdt, w_f32)
            w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)
            buffer_ops.buffer_store(w_out, w_rsrc, route)

            # Dropped routes (non-local expert, or the dead-tail padding) must NOT
            # claim a slot: counting them inflates masked_m, which grows psum and
            # makes the grouped GEMM compute rows that only ever fold away via
            # gather_w=0. Add 0 for them and tag the route with the sentinel, so
            # masked_m holds exactly this rank's own routes and every consumer can
            # tell "no grouped row" from a real row index.
            incr = _raw(is_drop.select(c0, c1))
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                _slot_ptr(fx.Int64(ptrtoint(counter)), eff_e),
                incr,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + eff_e * fx.Uint32(max_m)
            row_out = arith.select(_raw(is_drop), dropped_row, _raw(row))
            buffer_ops.buffer_store(row_out, out_rsrc, route)

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
