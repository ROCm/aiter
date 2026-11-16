# SPDX-License-Identifier: MIT
"""Single-workgroup fused EP route + psum + remap (item D A1, 2026-11-16).

EP counterpart of ``moe_contiguous_psum.build_moe_route_psum_fused_module`` (TP only). One workgroup of 512 threads:
  A. global expert mask (n <= 512, 0/1) -> inclusive scan in LDS -> LUT global id -> local bucket (or E = dropped),
     and the per-expert LDS counters zeroed (as moe_route_g2l_fused);
  B. per route: LUT lookup, gather_w = kept ? cast(weight) : 0, kept routes claim a slot with a workgroup-scope LDS
     atomic (row = slot + e * max_m), dropped routes get DROPPED_ROUTE_ROW;
  C. tile-aligned inclusive scan of the per-expert counts -> masked_m, starts, psum (as the TP fused kernel);
  D. in-place masked -> contiguous remap of every kept route's row.
Replaces moe_route_g2l_fused + moe_contiguous_psum_remap (two launches) on the static-EP decode path. Every route is
valid (no num_valid_routes truncation: static EP, token count = numel / topk). Slot order inside an expert follows the
LDS atomics instead of the global ones, which is as nondeterministic as before and harmless: GEMM rows are independent
and the gather-reduce sums each token's routes in fixed k order.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)

BLOCK = 512  # threads; also the max global expert count (mask length) and local expert count


@fx.struct
class _RoutePsumEpStorage:
    cnt: fx.Array[fx.Int32, BLOCK, 16]
    lds0: fx.Array[fx.Int32, BLOCK, 16]
    lds1: fx.Array[fx.Int32, BLOCK, 16]
    lut: fx.Array[fx.Int32, BLOCK, 16]


def build_moe_route_psum_fused_ep_module(weight_dtype: str = "bf16"):
    assert weight_dtype in ("bf16", "f16")

    @flyc.kernel(name=f"moe_route_psum_fused_ep_{weight_dtype}", known_block_size=[BLOCK, 1, 1])
    def route_psum_fused_ep_kernel(
        expert_mask: fx.Pointer,  # (n,) i32 0/1 global expert mask
        topk_ids: fx.Pointer,  # (numel,) i32 GLOBAL expert ids
        weight_in: fx.Pointer,  # (numel,) f32 route weights
        topids_to_rows: fx.Pointer,  # (numel,) i32 out: contiguous rows / DROPPED_ROUTE_ROW
        gather_w: fx.Pointer,  # (numel,) weight dtype out: kept -> cast, dropped -> 0
        masked_m: fx.Pointer,  # (E,) i32 out: kept routes per local expert
        starts: fx.Pointer,  # (E,) i32 out
        psum: fx.Pointer,  # (E,) i32 out (= m_tile_map)
        n: Int32,
        numel: Int32,
        experts: Int32,
        max_m: Int32,
        tile_m: Int32,
    ):
        w_fx = fx.BFloat16 if weight_dtype == "bf16" else fx.Float16
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1
        e_count = fx.Uint32(experts)

        lds = fx.SharedAllocator().allocate(_RoutePsumEpStorage).peek()
        lds_cnt = lds.cnt.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        lds_lut = lds.lut.ptr

        mk_p = ptr_buf_tensor(expert_mask)
        tk_p = ptr_buf_tensor(topk_ids)
        wi_p = ptr_buf_tensor(weight_in, fx.Float32)
        rows_p = ptr_buf_tensor(topids_to_rows)
        w_p = ptr_buf_tensor(gather_w, w_fx)
        m_p = ptr_buf_tensor(masked_m)
        s_p = ptr_buf_tensor(starts)
        p_p = ptr_buf_tensor(psum)

        in_expert = tid < e_count
        in_range = tid < fx.Uint32(n)

        # Phase A: zero the LDS counters; mask -> 0/1 into LDS, inclusive scan, LUT.
        if in_expert:
            lds_cnt[tid] = c0
        if in_range:
            mv = mk_p[tid]
            lds0[tid] = fx.Int32((mv != c0).select(c1, c0))
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_range:
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
            gpu.barrier()
            src, dst = dst, src

        if in_range:
            incl = src[tid]
            mv2 = mk_p[tid]
            lds_lut[tid] = fx.Int32((mv2 != c0).select(fx.Uint32(incl) - 1, e_count))
        gpu.barrier()

        # Phase B: route.
        numel_u = fx.Uint32(numel)
        for route in range(tid, numel_u, BLOCK):
            ge = fx.Uint32(tk_p[route])
            le = fx.Uint32(lds_lut[ge])
            is_drop = le == e_count
            w_f32 = wi_p[route]
            w_p[route] = is_drop.select(w_fx(0.0), w_f32.to(w_fx))
            if is_drop:
                rows_p[route] = fx.Int32(DROPPED_ROUTE_ROW)
            else:
                slot = fx.atomic_add(lds_cnt + fx.Int32(le), fx.Int32(1), syncscope="workgroup")
                rows_p[route] = fx.Uint32(slot) + le * fx.Uint32(max_m)
        gpu.barrier()

        # Phase C: tile-aligned inclusive scan of the per-expert counts.
        if in_expert:
            m = fx.Uint32(lds_cnt[tid])
            lds0[tid] = (m + tile_minus_1) // tile_v * tile_v
            m_p[tid] = m
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
            gpu.barrier()
            src, dst = dst, src

        if in_expert:
            is_not_first = tid != 0
            start = fx.Int32(0)
            if is_not_first:
                start = src[tid - 1]
            m_tid = lds_cnt[tid]
            s_p[tid] = start
            p_p[tid] = start + m_tid
        gpu.barrier()

        # Phase D: in-place masked -> contiguous remap (same block: the starts stores above are visible after the barrier).
        for route in range(tid, numel_u, BLOCK):
            row_raw = rows_p[route]
            if fx.Int32(row_raw) >= 0:
                row = fx.Uint32(row_raw)
                mm = fx.Uint32(max_m)
                expert = row // mm
                slot2 = row - expert * mm
                start2 = fx.Uint32(s_p[expert])
                rows_p[route] = start2 + slot2

    @flyc.jit
    def launch_route_psum_fused_ep(
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        weight_in: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        n: fx.Int32,
        numel: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        route_psum_fused_ep_kernel(
            expert_mask, topk_ids, weight_in, topids_to_rows, gather_w, masked_m, starts, psum,
            n, numel, experts, max_m, tile_m,
        ).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_route_psum_fused_ep.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_psum_fused_ep
