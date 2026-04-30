# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


"""Shared utilities for gfx1250 MoE 2-stage kernels.

Common helpers used by both the fp16 WMMA kernels and the mxscale
(fp4/fp8/a8w4) kernels.
"""

from __future__ import annotations

import inspect
from typing import Any

from flydsl.runtime.device import get_rocm_arch as get_hip_arch


def _require_gfx1250() -> None:
    arch = str(get_hip_arch())
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"Expected gfx1250 architecture, got {arch!r}")


def _align_up(v: int, a: int) -> int:
    return ((int(v) + int(a) - 1) // int(a)) * int(a)


def _pick_fp4_warp_shape(tile_m: int, tile_n: int) -> tuple[int, int]:
    """Pick a legal (m_warp, n_warp) for compile_mxfp4_gemm constraints."""
    for m_warp in (4, 2, 1):
        if tile_m % m_warp != 0:
            continue
        warp_tile_m = tile_m // m_warp
        if (warp_tile_m % 16) != 0:
            continue
        for n_warp in (4, 2, 1):
            if tile_n % n_warp != 0:
                continue
            warp_tile_n = tile_n // n_warp
            if (warp_tile_n % 32) == 0:
                return m_warp, n_warp
    raise ValueError(
        f"Cannot find legal (m_warp,n_warp) for FP4 GEMM with tile_m={tile_m}, tile_n={tile_n}. "
        "Need warp_tile_m multiple of 16 and warp_tile_n multiple of 32."
    )


def _pick_fp16_single_launch_shape(route_tile_m: int, route_tile_n: int,
                                    max_total_warps: int = 0) -> tuple[int, int, int, int]:
    """Pick launch shape for fp16 stage1 single-kernel path.

    Single-kernel path should follow route tile size (not backend-expanded 128x*)
    while keeping legal WMMA tile decomposition.
    """
    tile_m = _align_up(int(route_tile_m), 16)
    tile_n = _align_up(int(route_tile_n), 16)
    for mw in (4, 2, 1):
        if tile_m % mw != 0:
            continue
        if (tile_m // mw) % 16 != 0:
            continue
        for nw in (8, 4, 2, 1):
            if max_total_warps > 0 and mw * nw > max_total_warps:
                continue
            if tile_n % nw != 0:
                continue
            if (tile_n // nw) % 16 != 0:
                continue
            return tile_m, tile_n, mw, nw
    raise ValueError(
        f"Cannot find legal single-kernel fp16 shape for tile_m={route_tile_m}, tile_n={route_tile_n}"
    )


def _compile_with_optional_wpe(fn, kwargs: dict[str, Any]):
    sig = inspect.signature(fn)
    if "waves_per_eu" not in sig.parameters:
        kwargs = {k: v for k, v in kwargs.items() if k != "waves_per_eu"}
    return fn(**kwargs)


def _bf16_to_f16_wrapper(fp16_exe, x_arg: int, w_arg: int):
    """Wrap a compiled fp16 kernel to accept bf16 inputs by converting them to fp16 on the host."""
    import torch

    def wrapper(*args, **kwargs):
        args = list(args)
        for idx in (x_arg, w_arg):
            if idx < len(args) and hasattr(args[idx], 'dtype') and args[idx].dtype == torch.bfloat16:
                args[idx] = args[idx].to(torch.float16)
        return fp16_exe(*args, **kwargs)

    for attr in ('mode',):
        if hasattr(fp16_exe, attr):
            setattr(wrapper, attr, getattr(fp16_exe, attr))
    return wrapper


def _pick_mxscale_launch_shape(data_format: str, route_tile_m: int, tile_n: int) -> tuple[int, int, int, int]:
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")
    if data_format == "fp4":
        single_tile_m = _align_up(int(route_tile_m), 16)
        single_tile_n = _align_up(int(tile_n), 32)
        single_m_warp, single_n_warp = _pick_fp4_warp_shape(single_tile_m, single_tile_n)
        return single_tile_m, single_tile_n, single_m_warp, single_n_warp
    return _pick_fp16_single_launch_shape(int(route_tile_m), int(tile_n), max_total_warps=8)


def _make_moe_wave_layout(*, m_warp: int, n_warp: int, WAVE_SIZE: int, fx):
    return fx.make_layout(
        (int(m_warp), int(n_warp), 2, 16),
        (int(n_warp) * WAVE_SIZE, WAVE_SIZE, 16, 1),
    )


def _make_wmma_sub_tiles(
    *, wmma_m_rep: int, wmma_n_rep: int, WMMA_M: int, is_fp4: bool
) -> list[tuple[int, int, int, int]]:
    sub_tiles = []
    for wm in range(wmma_m_rep):
        for wn in range(wmma_n_rep):
            if is_fp4:
                for half in range(2):
                    sub_tiles.append((wm * wmma_n_rep + wn, half * 8, wm * WMMA_M, wn * 2 + half))
            else:
                sub_tiles.append((wm * wmma_n_rep + wn, 0, wm * WMMA_M, wn))
    return sub_tiles


def _moe_out_elem_ty(out_dtype: str, T):
    return T.f16 if out_dtype == "f16" else T.bf16


def _extract_sub8(acc, vec_base: int, *, vector, range_constexpr, ACC_VEC_SIZE: int):
    if ACC_VEC_SIZE == 8:
        return acc
    return vector.shuffle(acc, acc, [vec_base + i for i in range_constexpr(8)])


def _finalize_alloc_and_launch_2d(*, ctx, alloc, launcher, gx, gy, block_threads: int, stream, waves_per_eu, ir,
                                  cluster=None, gz=1):
    with ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalized = False
        alloc.finalize()
    for op in ctx.gpu_module_body.operations:
        if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
            if waves_per_eu is not None and int(waves_per_eu) >= 1:
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), int(waves_per_eu)
                )
            if cluster is not None:
                op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                    f"{cluster[0]},{cluster[1]},{cluster[2]}")
    launcher.launch(
        grid=(gx, gy, gz),
        block=(block_threads, 1, 1),
        stream=stream,
        cluster=cluster,
    )


def _emit_stage1_gate_up_epilogue(
    *,
    sub_tiles,
    by,
    tile_m: int,
    route_tile_m: int,
    warp_m_base,
    warp_n_base,
    blk_n,
    lane16,
    lane_kgrp,
    WMMA_N: int,
    i32_tokens_in,
    i32_inter_in,
    topk: int,
    num_valid_i32=None,
    block_row_start=None,
    sorted_rsrc,
    tw_rsrc,
    out_rsrc,
    doweight_stage1: bool,
    out_elem_ty,
    load_gate_up_sub8,
    silu_fn,
    ir,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    T,
):
    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    default_block_row_start = arith.index_cast(T.i32, by * arith.index(int(route_tile_m)))
    row_base_i32 = block_row_start if block_row_start is not None else default_block_row_start
    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
        sorted_row = by * arith.index(int(tile_m)) + row_local
        row_i32 = arith.index_cast(T.i32, row_local)
        sorted_i32 = arith.index_cast(T.i32, sorted_row)
        row_in_route = arith.cmpi(
            arith.CmpIPredicate.ult,
            row_i32,
            arith.constant(int(route_tile_m), type=T.i32),
        )
        if num_valid_i32 is None:
            row_ok_meta = row_in_route
        else:
            row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
            row_ok_meta = arith.andi(row_in_route, row_in_valid)
        sorted_safe = arith.select(
            row_ok_meta,
            sorted_i32,
            row_base_i32,
        )
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, arith.constant(int(topk), type=T.i32))
        row_ok = arith.andi(row_ok_meta, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))
        sub8g, sub8u = load_gate_up_sub8(acc_idx, vec_base)
        tw = buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32) if bool(doweight_stage1) else arith.constant(1.0, type=T.f32)
        col_base = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane_kgrp * fx.Index(8)
        for vi in range_constexpr(8):
            col = col_base + fx.Index(vi)
            col_ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, col), i32_inter_in)
            out_ok = arith.andi(row_ok, col_ok)
            _if_out = scf.IfOp(out_ok)
            with ir.InsertionPoint(_if_out.then_block):
                vg = vector.extract(sub8g, static_position=[vi], dynamic_position=[])
                vu = vector.extract(sub8u, static_position=[vi], dynamic_position=[])
                y = silu_fn(vg) * vu
                if bool(doweight_stage1):
                    y = y * tw
                out_v = arith.trunc_f(out_elem_ty, y)
                out_idx = ((tok * c_topk_i32 + slot) * i32_inter_in
                           + arith.index_cast(T.i32, col))
                buffer_ops.buffer_store(out_v, out_rsrc, out_idx)
                scf.YieldOp([])


def _emit_stage1_gate_up_splitk_epilogue(
    *,
    sub_tiles,
    by,
    tile_m: int,
    route_tile_m: int,
    warp_m_base,
    warp_n_base,
    blk_n,
    lane16,
    lane_kgrp,
    WMMA_N: int,
    i32_tokens_in,
    i32_inter_in,
    topk: int,
    num_valid_i32,
    block_row_start,
    sorted_rsrc,
    out_rsrc,
    out_elem_ty,
    load_gate_up_sub8,
    ir,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    rocdl,
    T,
):
    """Stage1 split-K epilogue.

    Writes per-K-slice gate/up partial sums to a ``[tokens*topk, 2*inter_dim]``
    output tensor with atomic fadd. The silu/mul fusion is skipped and must
    be applied by a separate reduction kernel (which also folds in the
    per-slot routing weight).

    Layout:
      out[row, col]                   += gate_partial[row, col]
      out[row, col + inter_dim]       += up_partial[row, col]
    where ``row = tok * topk + slot`` and ``col < inter_dim``.
    """
    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    c2_i32 = arith.constant(2, type=T.i32)
    zero_i32 = arith.constant(0, type=T.i32)
    mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)

    def atomic_add_x2(val_x2, byte_off_i32):
        rocdl.raw_ptr_buffer_atomic_fadd(val_x2, out_rsrc, byte_off_i32, zero_i32, zero_i32)

    inter_stride_i32 = i32_inter_in * c2_i32  # row stride for [tokens*topk, 2*inter_dim]

    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
        sorted_row = by * arith.index(int(tile_m)) + row_local
        row_i32 = arith.index_cast(T.i32, row_local)
        sorted_i32 = arith.index_cast(T.i32, sorted_row)
        row_in_route = arith.cmpi(
            arith.CmpIPredicate.ult,
            row_i32,
            arith.constant(int(route_tile_m), type=T.i32),
        )
        row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
        row_ok_meta = arith.andi(row_in_route, row_in_valid)
        sorted_safe = arith.select(row_ok_meta, sorted_i32, block_row_start)
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
        row_ok = arith.andi(row_ok_meta, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))

        sub8g, sub8u = load_gate_up_sub8(acc_idx, vec_base)
        col_base = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane_kgrp * fx.Index(8)
        row_elem_base = (tok * c_topk_i32 + slot) * inter_stride_i32

        for vpair in range_constexpr(4):
            vi0 = vpair * 2
            vi1 = vi0 + 1
            col0 = col_base + fx.Index(vi0)
            col1 = col_base + fx.Index(vi1)
            col0_i32 = arith.index_cast(T.i32, col0)
            col1_i32 = arith.index_cast(T.i32, col1)
            col0_ok = arith.cmpi(arith.CmpIPredicate.ult, col0_i32, i32_inter_in)
            col1_ok = arith.cmpi(arith.CmpIPredicate.ult, col1_i32, i32_inter_in)
            out_ok = arith.andi(row_ok, col0_ok)
            _if_out = scf.IfOp(out_ok)
            with ir.InsertionPoint(_if_out.then_block):
                # ---- gate partial ----
                vg0 = vector.extract(sub8g, static_position=[vi0], dynamic_position=[])
                vg1 = vector.extract(sub8g, static_position=[vi1], dynamic_position=[])
                vg1 = arith.select(col1_ok, vg1, arith.constant(0.0, type=T.f32))
                g0 = arith.trunc_f(out_elem_ty, vg0)
                g1 = arith.trunc_f(out_elem_ty, vg1)
                frag_g = vector.from_elements(T.vec(2, out_elem_ty), [g0, g1])
                idx_g0 = row_elem_base + col0_i32
                idx_g_even = idx_g0 & mask_even_i32
                byte_off_g = idx_g_even * c2_i32
                atomic_add_x2(frag_g, byte_off_g)

                # ---- up partial (offset by inter_dim) ----
                vu0 = vector.extract(sub8u, static_position=[vi0], dynamic_position=[])
                vu1 = vector.extract(sub8u, static_position=[vi1], dynamic_position=[])
                vu1 = arith.select(col1_ok, vu1, arith.constant(0.0, type=T.f32))
                u0 = arith.trunc_f(out_elem_ty, vu0)
                u1 = arith.trunc_f(out_elem_ty, vu1)
                frag_u = vector.from_elements(T.vec(2, out_elem_ty), [u0, u1])
                idx_u0 = row_elem_base + i32_inter_in + col0_i32
                idx_u_even = idx_u0 & mask_even_i32
                byte_off_u = idx_u_even * c2_i32
                atomic_add_x2(frag_u, byte_off_u)

                scf.YieldOp([])


def _emit_stage2_store_epilogue(
    *,
    sub_tiles,
    by,
    tile_m: int,
    route_tile_m: int,
    warp_m_base,
    warp_n_base,
    blk_n,
    lane16,
    lane_kgrp,
    WMMA_N: int,
    i32_tokens_in,
    i32_n_in,
    topk: int,
    num_valid_i32,
    block_row_start,
    sorted_rsrc,
    tw_rsrc,
    out_rsrc,
    doweight_stage2: bool,
    accumulate: bool,
    out_elem_ty,
    load_sub8,
    ir,
    fx,
    arith,
    buffer_ops,
    scf,
    vector,
    range_constexpr,
    rocdl,
    T,
):
    c_topk_i32 = arith.constant(int(topk), type=T.i32)
    c2_i32 = arith.constant(2, type=T.i32)
    zero_i32 = arith.constant(0, type=T.i32)
    mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)

    def atomic_add_x2(val_x2, byte_off_i32):
        rocdl.raw_ptr_buffer_atomic_fadd(val_x2, out_rsrc, byte_off_i32, zero_i32, zero_i32)

    for acc_idx, vec_base, m_off, wn in sub_tiles:
        row_local = warp_m_base + fx.Index(m_off) + lane16
        sorted_row = by * arith.index(int(tile_m)) + row_local
        row_i32 = arith.index_cast(T.i32, row_local)
        sorted_i32 = arith.index_cast(T.i32, sorted_row)
        row_in_route = arith.cmpi(arith.CmpIPredicate.ult, row_i32, arith.constant(int(route_tile_m), type=T.i32))
        row_in_valid = arith.cmpi(arith.CmpIPredicate.slt, sorted_i32, num_valid_i32)
        row_ok = arith.andi(row_in_route, row_in_valid)
        sorted_safe = arith.select(row_ok, sorted_i32, block_row_start)
        fused = buffer_ops.buffer_load(sorted_rsrc, sorted_safe, vec_width=1, dtype=T.i32)
        tok = fused & arith.constant((1 << 24) - 1, type=T.i32)
        slot = fused >> arith.constant(24, type=T.i32)
        tok_ok = arith.cmpi(arith.CmpIPredicate.ult, tok, i32_tokens_in)
        slot_ok0 = arith.cmpi(arith.CmpIPredicate.sge, slot, arith.constant(0, type=T.i32))
        slot_ok1 = arith.cmpi(arith.CmpIPredicate.slt, slot, c_topk_i32)
        row_store_ok = arith.andi(row_ok, arith.andi(tok_ok, arith.andi(slot_ok0, slot_ok1)))
        ts = tok * c_topk_i32 + slot
        sub8 = load_sub8(acc_idx, vec_base)
        tw = buffer_ops.buffer_load(tw_rsrc, sorted_safe, vec_width=1, dtype=T.f32) if bool(doweight_stage2) else arith.constant(1.0, type=T.f32)
        col_base = blk_n + warp_n_base + fx.Index(wn * WMMA_N) + lane_kgrp * fx.Index(8)
        if bool(accumulate):
            for vpair in range_constexpr(4):
                vi0 = vpair * 2
                vi1 = vi0 + 1
                col0 = col_base + fx.Index(vi0)
                col1 = col_base + fx.Index(vi1)
                col0_i32 = arith.index_cast(T.i32, col0)
                col1_i32 = arith.index_cast(T.i32, col1)
                col0_ok = arith.cmpi(arith.CmpIPredicate.ult, col0_i32, i32_n_in)
                col1_ok = arith.cmpi(arith.CmpIPredicate.ult, col1_i32, i32_n_in)
                out_ok = arith.andi(row_store_ok, col0_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v0 = vector.extract(sub8, static_position=[vi0], dynamic_position=[])
                    v1 = vector.extract(sub8, static_position=[vi1], dynamic_position=[])
                    if bool(doweight_stage2):
                        v0 = v0 * tw
                        v1 = v1 * tw
                    v1 = arith.select(col1_ok, v1, arith.constant(0.0, type=T.f32))
                    out0 = arith.trunc_f(out_elem_ty, v0)
                    out1 = arith.trunc_f(out_elem_ty, v1)
                    frag = vector.from_elements(T.vec(2, out_elem_ty), [out0, out1])
                    idx0 = tok * i32_n_in + col0_i32
                    idx_even = idx0 & mask_even_i32
                    byte_off = idx_even * c2_i32
                    atomic_add_x2(frag, byte_off)
                    scf.YieldOp([])
        else:
            for vi in range_constexpr(8):
                col = col_base + fx.Index(vi)
                col_ok = arith.cmpi(arith.CmpIPredicate.ult, arith.index_cast(T.i32, col), i32_n_in)
                out_ok = arith.andi(row_store_ok, col_ok)
                _if_out = scf.IfOp(out_ok)
                with ir.InsertionPoint(_if_out.then_block):
                    v = vector.extract(sub8, static_position=[vi], dynamic_position=[])
                    if bool(doweight_stage2):
                        v = v * tw
                    col_i32 = arith.index_cast(T.i32, col)
                    out_idx = ts * i32_n_in + col_i32
                    out_v = arith.trunc_f(out_elem_ty, v)
                    buffer_ops.buffer_store(out_v, out_rsrc, out_idx)
                    scf.YieldOp([])


def _pack_stage1_gate_up_tiles(tensor, *, experts: int, inter_dim: int, tile_n: int, cols: int):
    """Pack stage1 gate/up rows into [gate_tile0, up_tile0, gate_tile1, up_tile1, ...]."""
    import torch

    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for stage1 gate/up packing, got {type(tensor)!r}")
    if tensor.numel() == 0:
        return tensor
    elems_per_expert = int(2 * inter_dim) * int(cols)
    if tensor.numel() != int(experts) * elems_per_expert:
        if tensor.numel() % elems_per_expert != 0:
            raise ValueError(
                "Unexpected stage1 tensor size for gate/up packing: "
                f"numel={tensor.numel()} expected={int(experts) * elems_per_expert} "
                f"(experts={experts}, inter_dim={inter_dim}, cols={cols})"
            )
        experts = tensor.numel() // elems_per_expert
    expected_rows = int(experts) * int(2 * inter_dim)
    if int(inter_dim) % int(tile_n) != 0:
        raise ValueError(
            f"Stage1 gate/up packed layout requires inter_dim divisible by tile_n, got {inter_dim} and {tile_n}"
        )

    tensor_3d = tensor.contiguous().view(int(experts), int(2 * inter_dim), int(cols))
    gate = tensor_3d[:, :int(inter_dim), :]
    up = tensor_3d[:, int(inter_dim):, :]
    gate_tiles = gate.view(int(experts), int(inter_dim // tile_n), int(tile_n), int(cols))
    up_tiles = up.view(int(experts), int(inter_dim // tile_n), int(tile_n), int(cols))
    packed = torch.cat((gate_tiles, up_tiles), dim=2)
    return packed.view(expected_rows, int(cols))


class _Stage1GateUpPackedWrapper:
    """Host-side wrapper that repacks stage1 W1 rows to match the merged gate/up TDM layout."""

    def __init__(
        self,
        stage1_exe,
        *,
        experts: int,
        inter_dim: int,
        tile_n: int,
        packed_cols_w: int,
        packed_cols_scale: int,
    ):
        self._stage1_exe = stage1_exe
        self._experts = int(experts)
        self._inter_dim = int(inter_dim)
        self._tile_n = int(tile_n)
        self._packed_cols_w = int(packed_cols_w)
        self._packed_cols_scale = int(packed_cols_scale)
        self._cache = {}

        for attr in ("mode", "compile_hints"):
            if hasattr(stage1_exe, attr):
                setattr(self, attr, getattr(stage1_exe, attr))

    def _get_packed_operands(self, arg_w, arg_scale_w):
        key = (id(arg_w), id(arg_scale_w))
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]

        packed_w = _pack_stage1_gate_up_tiles(
            arg_w,
            experts=self._experts,
            inter_dim=self._inter_dim,
            tile_n=self._tile_n,
            cols=self._packed_cols_w,
        )
        if hasattr(arg_scale_w, "numel") and int(arg_scale_w.numel()) > 0:
            packed_scale_w = _pack_stage1_gate_up_tiles(
                arg_scale_w,
                experts=self._experts,
                inter_dim=self._inter_dim,
                tile_n=self._tile_n,
                cols=self._packed_cols_scale,
            )
        else:
            packed_scale_w = arg_scale_w

        # Store (result, original_refs) — the strong refs to originals
        # prevent id() reuse while the entry is alive.
        self._cache[key] = ((packed_w, packed_scale_w), (arg_w, arg_scale_w))
        return packed_w, packed_scale_w

    def __call__(self, *args, **kwargs):
        args = list(args)
        if len(args) > 4:
            args[2], args[4] = self._get_packed_operands(args[2], args[4])
        return self._stage1_exe(*args, **kwargs)


# ---------------------------------------------------------------------------
# MXScale format infrastructure helpers
# ---------------------------------------------------------------------------


def _mxscale_format_config(data_format: str) -> dict[str, int | bool]:
    if data_format not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"data_format must be 'fp4', 'fp8', or 'a8w4', got {data_format!r}")
    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    pack_factor_a = 1 if not is_fp4 else 2
    pack_factor_b = 2 if (is_fp4 or is_a8w4) else 1
    wmma_n_eff = 32 if is_fp4 else 16
    acc_vec_size = 16 if is_fp4 else 8
    ds_loads_per_a_frag = 2 if is_fp4 else 4
    return {
        "is_fp4": is_fp4,
        "is_a8w4": is_a8w4,
        "PACK_FACTOR_A": pack_factor_a,
        "PACK_FACTOR_B": pack_factor_b,
        "WMMA_N_EFF": wmma_n_eff,
        "ACC_VEC_SIZE": acc_vec_size,
        "DS_LOADS_PER_A_FRAG": ds_loads_per_a_frag,
    }


def _mxscale_precompute_preshuffled_b_data_bases(
    *,
    packed_tile_k_b: int,
    warp_tile_n,
    wave_n_idx,
    lane16,
    lane_kgrp,
    wmma_n_rep: int,
    arith,
    range_constexpr,
):
    ngroup_stride = packed_tile_k_b * 16
    n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
    row_off = lane16 * arith.index(16)
    k_tile_off = lane_kgrp * arith.index(256)
    bases = []
    for wn in range_constexpr(wmma_n_rep):
        ngroup_off = n_group_base * arith.index(ngroup_stride) + arith.index(wn * ngroup_stride)
        bases.append(ngroup_off + row_off + k_tile_off)
    return bases


def _mxscale_precompute_a_scale_lane_bases(
    *,
    warp_m_base,
    lane16,
    wmma_m_rep: int,
    interleaved_scale_cols_a: int,
    arith,
):
    warp_lds_row = warp_m_base / arith.index(wmma_m_rep) + lane16
    base = warp_lds_row * arith.index(interleaved_scale_cols_a)
    return [base]


def _mxscale_load_scale_b128(
    *,
    lds_buffer,
    scale_base,
    reps: int,
    ks,
    SCALES_PER_WMMA: int,
    _lds_load_b128,
    arith,
    vector,
    range_constexpr,
):
    ks_byte_off = ks * reps * SCALES_PER_WMMA
    eff_base = scale_base if ks_byte_off == 0 else scale_base + arith.index(ks_byte_off)
    num_loads = (reps + 3) // 4
    vecs = []
    for ld in range_constexpr(num_loads):
        off = eff_base if ld == 0 else eff_base + arith.index(ld * 16)
        vecs.append(_lds_load_b128(lds_buffer, off))
    results = []
    for i in range_constexpr(reps):
        vi = vector.extract(vecs[i // 4], static_position=[i % 4], dynamic_position=[])
        results.append(vi)
    return results


def _mxscale_load_preshuffled_b_frag(
    *,
    lds_buffer,
    b_lane_bases,
    wn: int,
    ks,
    is_fp4: bool,
    is_a8w4: bool,
    PACK_FACTOR_B: int,
    WMMA_K: int,
    _lds_load_b128,
    arith,
    vector,
):
    num_tiles = WMMA_K // PACK_FACTOR_B // 16
    k_subtile_off = arith.index(ks * num_tiles * 256)
    if is_fp4:
        base0 = b_lane_bases[wn * 2] + k_subtile_off
        base1 = b_lane_bases[wn * 2 + 1] + k_subtile_off
        v0 = _lds_load_b128(lds_buffer, base0)
        v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
        v2 = _lds_load_b128(lds_buffer, base1)
        v3 = _lds_load_b128(lds_buffer, base1 + arith.index(512))
        v01 = vector.shuffle(v0, v1, list(range(8)))
        v23 = vector.shuffle(v2, v3, list(range(8)))
        return vector.shuffle(v01, v23, list(range(16)))
    base0 = b_lane_bases[wn] + k_subtile_off
    v0 = _lds_load_b128(lds_buffer, base0)
    v1 = _lds_load_b128(lds_buffer, base0 + arith.index(512))
    if is_a8w4:
        return vector.shuffle(v0, v1, list(range(8)))
    v2 = _lds_load_b128(lds_buffer, base0 + arith.index(1024))
    v3 = _lds_load_b128(lds_buffer, base0 + arith.index(1536))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_load_scale_i32(
    *,
    lds_buffer,
    scale_base,
    ks,
    SCALES_PER_WMMA: int,
    llvm_dialect,
    ir,
    arith,
    T,
):
    byte_off = scale_base + arith.index(ks * SCALES_PER_WMMA)
    ptr_val = _mxscale_lds_ptr(lds_buffer, byte_off, ir=ir, arith=arith, T=T)
    return llvm_dialect.load(ir.IntegerType.get_signless(32), ptr_val)


def _mxscale_precompute_a_data_bases(
    *,
    warp_m_base,
    lane16,
    lane_kgrp,
    lds_a_stride_bytes: int,
    wmma_m_rep: int,
    WMMA_M: int,
    is_fp4: bool,
    arith,
    range_constexpr,
):
    row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
    k_half_off = lane_kgrp * arith.index(32 if is_fp4 else 16)
    return [
        row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
        for wm in range_constexpr(wmma_m_rep)
    ]


def _mxscale_precompute_rowmajor_b_data_bases(
    *,
    warp_n_base,
    lane16,
    lane_kgrp,
    lds_b_stride_bytes: int,
    wmma_n_rep: int,
    WMMA_N: int,
    arith,
    range_constexpr,
):
    return [
        (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
        + lane_kgrp * arith.index(32)
        + arith.index(wnh * WMMA_N * lds_b_stride_bytes)
        for wnh in range_constexpr(wmma_n_rep * 2)
    ]


def _mxscale_precompute_rowmajor_scale_lane_bases(
    *,
    warp_base,
    lane16,
    scale_k_per_tile: int,
    reps: int,
    WMMA_DIM: int,
    arith,
    range_constexpr,
):
    return [
        (warp_base + lane16) * arith.index(int(scale_k_per_tile))
        + arith.index(r * WMMA_DIM * int(scale_k_per_tile))
        for r in range_constexpr(reps)
    ]


def _mxscale_lds_ptr(lds_buffer, byte_offset, *, ir, arith, T):
    """Compute an ``!llvm.ptr<3>`` into LDS at *byte_offset*."""
    from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
    from flydsl.expr.arith import _to_raw as _raw
    from flydsl.expr.arith import ArithValue as _AV
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    raw_memref = arith.unwrap(lds_buffer)
    lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)
    total_byte = _AV(lds_base) + byte_offset
    addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
    return _llvm.inttoptr(lds_ptr_ty, addr_i32)


def _mxscale_lds_load_b128(lds_buffer, byte_offset, *, ir, arith, T, llvm_dialect):
    """Load a vec4<i32> (16 bytes) from LDS at the given byte offset."""
    ptr_val = _mxscale_lds_ptr(lds_buffer, byte_offset, ir=ir, arith=arith, T=T)
    return llvm_dialect.load(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), ptr_val,
    )


def _mxscale_load_data_frag(
    *,
    lds_buffer,
    lane_base,
    ks,
    PACK_FACTOR_A: int,
    WMMA_K: int,
    is_fp4: bool,
    _lds_load_b128,
    arith,
    vector,
):
    byte_off = lane_base + arith.index(ks * WMMA_K // PACK_FACTOR_A)
    v0 = _lds_load_b128(lds_buffer, byte_off)
    if is_fp4:
        v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(16))
        return vector.shuffle(v0, v1, list(range(8)))
    v1 = _lds_load_b128(lds_buffer, byte_off + arith.index(32))
    v2 = _lds_load_b128(lds_buffer, byte_off + arith.index(64))
    v3 = _lds_load_b128(lds_buffer, byte_off + arith.index(96))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_load_rowmajor_b_frag(
    *,
    lds_buffer,
    b_lane_bases,
    wn: int,
    ks,
    PACK_FACTOR_B: int,
    WMMA_K: int,
    _lds_load_b128,
    arith,
    vector,
):
    k_byte_off = arith.index(ks * WMMA_K // PACK_FACTOR_B)
    base0 = b_lane_bases[wn * 2] + k_byte_off
    base1 = b_lane_bases[wn * 2 + 1] + k_byte_off
    v0 = _lds_load_b128(lds_buffer, base0)
    v1 = _lds_load_b128(lds_buffer, base0 + arith.index(16))
    v2 = _lds_load_b128(lds_buffer, base1)
    v3 = _lds_load_b128(lds_buffer, base1 + arith.index(16))
    v01 = vector.shuffle(v0, v1, list(range(8)))
    v23 = vector.shuffle(v2, v3, list(range(8)))
    return vector.shuffle(v01, v23, list(range(16)))


def _mxscale_emit_wmma(
    *,
    accs,
    wm: int,
    wn: int,
    a_frag,
    b_frags,
    a_scales,
    b_scales,
    is_fp4: bool,
    is_a8w4: bool,
    use_scale_opsel: bool,
    rocdl,
    T,
):
    idx = wm * len(b_frags) + wn
    if use_scale_opsel:
        a_scale_idx = wm // 2
        a_opsel = wm % 2
    else:
        a_scale_idx = wm
        a_opsel = 0

    if is_fp4:
        accs[idx] = rocdl.wmma_scale_f32_32x16x128_f4(
            T.vec(16, T.f32),
            b_frags[wn], a_frag, accs[idx],
            b_scales[wn * 2], a_scales[a_scale_idx],
            scaleAType=0,
            scaleBType=a_opsel,
        )
        return

    if use_scale_opsel:
        b_scale_idx = wn // 2
        b_opsel = wn % 2
    else:
        b_scale_idx = wn
        b_opsel = 0
    accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        T.vec(8, T.f32),
        b_frags[wn], a_frag, accs[idx],
        b_scales[b_scale_idx], a_scales[a_scale_idx],
        fmtA=4 if is_a8w4 else 0,
        fmtB=0,
        scaleAType=b_opsel,
        scaleBType=a_opsel,
    )


# ---------------------------------------------------------------------------
# Shared tiling / pipeline / loader helpers for mxscale stage1 & stage2
# ---------------------------------------------------------------------------


def _compute_mxscale_tiling(
    *,
    data_format: str,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    out_dtype: str,
    num_buffers: int,
    cluster_m: int = 1,
    cluster_n: int = 1,
    stage_name: str = "",
) -> dict:
    """Derive all shared tiling / format constants for an mxscale stage kernel."""
    fmt_cfg = _mxscale_format_config(data_format)
    is_fp4 = bool(fmt_cfg["is_fp4"])
    is_a8w4 = bool(fmt_cfg["is_a8w4"])
    PACK_FACTOR_A = int(fmt_cfg["PACK_FACTOR_A"])
    PACK_FACTOR_B = int(fmt_cfg["PACK_FACTOR_B"])
    ACC_VEC_SIZE = int(fmt_cfg["ACC_VEC_SIZE"])
    WMMA_N_EFF = int(fmt_cfg["WMMA_N_EFF"])
    DS_LOADS_PER_A_FRAG = int(fmt_cfg["DS_LOADS_PER_A_FRAG"])

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
    SCALE_BLOCK = 32
    SCALES_PER_WMMA = WMMA_K // SCALE_BLOCK
    WAVE_SIZE = 32
    LDS_PAD_A_BYTES = 16
    LDS_PAD_B_BYTES = 16 if is_fp4 else 0

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(
            f"mxscale {stage_name} single kernel supports out_dtype "
            f"in ('f16','bf16'), got {out_dtype!r}"
        )
    if (K % int(tile_k)) != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if (int(tile_k) % WMMA_K) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {WMMA_K}")
    if (int(tile_k) % SCALE_BLOCK) != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by {SCALE_BLOCK}")
    if int(num_buffers) not in (1, 2, 3, 4):
        raise ValueError(f"num_buffers must be 1, 2, 3, or 4, got {num_buffers}")
    use_cluster = int(cluster_m) > 1 or int(cluster_n) > 1
    if use_cluster and int(cluster_m) * int(cluster_n) > 16:
        raise ValueError(
            f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}"
        )

    K_packed_a = K // PACK_FACTOR_A
    K_packed_b = K // PACK_FACTOR_B
    packed_tile_k_a = int(tile_k) // PACK_FACTOR_A
    packed_tile_k_b = int(tile_k) // PACK_FACTOR_B
    K_scale = K // SCALE_BLOCK
    scale_k_per_tile = int(tile_k) // SCALE_BLOCK
    block_threads = int(m_warp) * int(n_warp) * WAVE_SIZE
    warp_tile_m = int(tile_m) // int(m_warp)
    warp_tile_n = int(tile_n) // int(n_warp)
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    k_wmma_steps = int(tile_k) // WMMA_K
    n_accs = wmma_m_rep * wmma_n_rep
    num_k_tiles = K // int(tile_k)
    b_scale_load_rep = (wmma_n_rep * 2) if is_fp4 else wmma_n_rep
    interleaved_scale_cols_b = b_scale_load_rep * scale_k_per_tile

    if wmma_m_rep <= 0 or wmma_n_rep <= 0:
        raise ValueError(
            f"Invalid warp tiling for mxscale {stage_name} single kernel: "
            f"wmma_m_rep={wmma_m_rep}, wmma_n_rep={wmma_n_rep}"
        )

    lds_a_stride_bytes = packed_tile_k_a + LDS_PAD_A_BYTES
    lds_b_stride_bytes = packed_tile_k_b + LDS_PAD_B_BYTES
    lds_a_data_bytes = int(tile_m) * lds_a_stride_bytes
    lds_b_data_bytes = int(tile_n) * lds_b_stride_bytes
    lds_a_scale_bytes = int(tile_m) * scale_k_per_tile
    lds_b_scale_bytes = int(tile_n) * scale_k_per_tile
    interleaved_scale_cols_a = wmma_m_rep * scale_k_per_tile

    return dict(
        is_fp4=is_fp4, is_a8w4=is_a8w4,
        PACK_FACTOR_A=PACK_FACTOR_A, PACK_FACTOR_B=PACK_FACTOR_B,
        ACC_VEC_SIZE=ACC_VEC_SIZE, WMMA_N_EFF=WMMA_N_EFF,
        DS_LOADS_PER_A_FRAG=DS_LOADS_PER_A_FRAG,
        WMMA_M=WMMA_M, WMMA_N=WMMA_N, WMMA_K=WMMA_K,
        SCALE_BLOCK=SCALE_BLOCK, SCALES_PER_WMMA=SCALES_PER_WMMA,
        WAVE_SIZE=WAVE_SIZE,
        LDS_PAD_A_BYTES=LDS_PAD_A_BYTES, LDS_PAD_B_BYTES=LDS_PAD_B_BYTES,
        use_cluster=use_cluster,
        K=K, K_packed_a=K_packed_a, K_packed_b=K_packed_b,
        packed_tile_k_a=packed_tile_k_a, packed_tile_k_b=packed_tile_k_b,
        K_scale=K_scale, scale_k_per_tile=scale_k_per_tile,
        block_threads=block_threads,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n,
        wmma_m_rep=wmma_m_rep, wmma_n_rep=wmma_n_rep,
        k_wmma_steps=k_wmma_steps, n_accs=n_accs,
        num_k_tiles=num_k_tiles,
        b_scale_load_rep=b_scale_load_rep,
        interleaved_scale_cols_b=interleaved_scale_cols_b,
        lds_a_stride_bytes=lds_a_stride_bytes,
        lds_b_stride_bytes=lds_b_stride_bytes,
        lds_a_data_bytes=lds_a_data_bytes,
        lds_b_data_bytes=lds_b_data_bytes,
        lds_a_scale_bytes=lds_a_scale_bytes,
        lds_b_scale_bytes=lds_b_scale_bytes,
        interleaved_scale_cols_a=interleaved_scale_cols_a,
    )


def _compute_pipeline_plan(
    *,
    num_k_tiles: int,
    num_buffers: int,
    B_TDM_PER_STEP: int,
    tile_m: int,
    use_tdm_gather: bool,
    wave_specialized_tdm: bool,
    tdm_loader_waves: int,
) -> dict:
    """Compute pipeline pre-load / tail plan shared by mxscale stages."""
    from kernels.pipeline_utils import make_tail_plan

    pre_loaded = int(num_buffers) - 1
    loop_iters = (num_k_tiles - pre_loaded) // int(num_buffers)
    tail_start = loop_iters * int(num_buffers)
    extra = num_k_tiles - tail_start - pre_loaded
    A_GATHER_GROUPS = (int(tile_m) + 7) // 8 if bool(use_tdm_gather) else 0
    if bool(use_tdm_gather) and bool(wave_specialized_tdm):
        A_GATHER_TDM_PER_STEP = (
            (A_GATHER_GROUPS + tdm_loader_waves - 1) // tdm_loader_waves
        )
    else:
        A_GATHER_TDM_PER_STEP = A_GATHER_GROUPS
    TDM_PER_STEP = B_TDM_PER_STEP + A_GATHER_TDM_PER_STEP
    fence_outstanding = TDM_PER_STEP * (int(num_buffers) - 2)
    base_tail_plan = make_tail_plan(int(num_buffers), pre_loaded, extra)
    tail_plan = [
        (ls, cs, o * TDM_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in base_tail_plan
    ]
    if num_k_tiles < int(num_buffers):
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles}"
        )
    return dict(
        pre_loaded=pre_loaded,
        loop_iters=loop_iters,
        tail_start=tail_start,
        extra=extra,
        A_GATHER_GROUPS=A_GATHER_GROUPS,
        TDM_PER_STEP=TDM_PER_STEP,
        fence_outstanding=fence_outstanding,
        tail_plan=tail_plan,
    )


def _compute_tdm_store_layout(
    *,
    warp_tile_m: int,
    warp_tile_n: int,
    num_warps: int,
    WMMA_N: int,
    use_pipeline: bool,
) -> dict:
    """Compute TDM-store D output LDS layout, shared by mxscale stages."""
    LDS_PAD_D_BYTES = 16
    elem_bytes_d = 2  # f16/bf16
    lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
    warp_d_bytes = warp_tile_m * lds_d_row_stride
    total_d_bytes = num_warps * warp_d_bytes
    return dict(
        lds_d_row_stride=lds_d_row_stride,
        warp_d_bytes=warp_d_bytes,
        total_d_bytes=total_d_bytes,
        d_output_off=0,
        lds_d_stride_elems=lds_d_row_stride // 2,
        warp_d_elems=warp_d_bytes // 2,
        n_col_d_elems=WMMA_N * elem_bytes_d // 2,
        d_need_epilogue_fence=use_pipeline,
    )


def _make_mxscale_data_loaders(
    *,
    tiling: dict,
    warp_m_base,
    warp_n_base,
    wave_n_idx,
    lane16,
    lane_kgrp,
    ir,
    arith,
    vector,
    llvm_dialect,
    T,
    range_constexpr,
) -> dict:
    """Create the 9 LDS data-loading adapter closures shared by mxscale stages.

    Returns a dict whose keys match the local names used inside the
    ``moe_mxscale_stage*_single`` kernel functions.
    """
    is_fp4 = tiling["is_fp4"]
    is_a8w4 = tiling["is_a8w4"]
    PACK_FACTOR_A = tiling["PACK_FACTOR_A"]
    PACK_FACTOR_B = tiling["PACK_FACTOR_B"]
    WMMA_K = tiling["WMMA_K"]
    WMMA_M = tiling["WMMA_M"]
    WMMA_N = tiling["WMMA_N"]
    SCALES_PER_WMMA = tiling["SCALES_PER_WMMA"]
    lds_a_stride_bytes = tiling["lds_a_stride_bytes"]
    lds_b_stride_bytes = tiling["lds_b_stride_bytes"]
    packed_tile_k_b = tiling["packed_tile_k_b"]
    warp_tile_n = tiling["warp_tile_n"]
    wmma_m_rep = tiling["wmma_m_rep"]
    wmma_n_rep = tiling["wmma_n_rep"]
    scale_k_per_tile = tiling["scale_k_per_tile"]
    interleaved_scale_cols_a = tiling["interleaved_scale_cols_a"]

    def _lds_load_b128(lds_buffer, byte_offset):
        return _mxscale_lds_load_b128(
            lds_buffer, byte_offset,
            ir=ir, arith=arith, T=T, llvm_dialect=llvm_dialect,
        )

    def load_data_frag(lds_buffer, lane_base, ks):
        return _mxscale_load_data_frag(
            lds_buffer=lds_buffer, lane_base=lane_base, ks=ks,
            PACK_FACTOR_A=PACK_FACTOR_A, WMMA_K=WMMA_K, is_fp4=is_fp4,
            _lds_load_b128=_lds_load_b128, arith=arith, vector=vector,
        )

    def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
        if is_fp4:
            return _mxscale_load_rowmajor_b_frag(
                lds_buffer=lds_buffer, b_lane_bases=b_lane_bases,
                wn=wn, ks=ks,
                PACK_FACTOR_B=PACK_FACTOR_B, WMMA_K=WMMA_K,
                _lds_load_b128=_lds_load_b128, arith=arith, vector=vector,
            )
        return _mxscale_load_preshuffled_b_frag(
            lds_buffer=lds_buffer, b_lane_bases=b_lane_bases,
            wn=wn, ks=ks,
            is_fp4=is_fp4, is_a8w4=is_a8w4,
            PACK_FACTOR_B=PACK_FACTOR_B, WMMA_K=WMMA_K,
            _lds_load_b128=_lds_load_b128, arith=arith, vector=vector,
        )

    def load_scale_i32(lds_buffer, scale_base, ks):
        return _mxscale_load_scale_i32(
            lds_buffer=lds_buffer, scale_base=scale_base, ks=ks,
            SCALES_PER_WMMA=SCALES_PER_WMMA,
            llvm_dialect=llvm_dialect, ir=ir, arith=arith, T=T,
        )

    def _precompute_a_data_bases():
        return _mxscale_precompute_a_data_bases(
            warp_m_base=warp_m_base, lane16=lane16, lane_kgrp=lane_kgrp,
            lds_a_stride_bytes=lds_a_stride_bytes, wmma_m_rep=wmma_m_rep,
            WMMA_M=WMMA_M, is_fp4=is_fp4,
            arith=arith, range_constexpr=range_constexpr,
        )

    def _precompute_b_data_bases():
        if is_fp4:
            return _mxscale_precompute_rowmajor_b_data_bases(
                warp_n_base=warp_n_base, lane16=lane16, lane_kgrp=lane_kgrp,
                lds_b_stride_bytes=lds_b_stride_bytes, wmma_n_rep=wmma_n_rep,
                WMMA_N=WMMA_N, arith=arith, range_constexpr=range_constexpr,
            )
        return _mxscale_precompute_preshuffled_b_data_bases(
            packed_tile_k_b=packed_tile_k_b, warp_tile_n=warp_tile_n,
            wave_n_idx=wave_n_idx, lane16=lane16, lane_kgrp=lane_kgrp,
            wmma_n_rep=wmma_n_rep, arith=arith, range_constexpr=range_constexpr,
        )

    def _precompute_a_scale_lane_bases():
        if is_fp4:
            return _mxscale_precompute_rowmajor_scale_lane_bases(
                warp_base=warp_m_base, lane16=lane16,
                scale_k_per_tile=scale_k_per_tile, reps=wmma_m_rep,
                WMMA_DIM=WMMA_M, arith=arith, range_constexpr=range_constexpr,
            )
        return _mxscale_precompute_a_scale_lane_bases(
            warp_m_base=warp_m_base, lane16=lane16,
            wmma_m_rep=wmma_m_rep,
            interleaved_scale_cols_a=interleaved_scale_cols_a, arith=arith,
        )

    def _precompute_b_scale_lane_bases():
        return _mxscale_precompute_rowmajor_scale_lane_bases(
            warp_base=warp_n_base, lane16=lane16,
            scale_k_per_tile=scale_k_per_tile, reps=wmma_n_rep * 2,
            WMMA_DIM=WMMA_N, arith=arith, range_constexpr=range_constexpr,
        )

    def load_scale_b128(lds_buffer, scale_base, reps, ks=0):
        return _mxscale_load_scale_b128(
            lds_buffer=lds_buffer, scale_base=scale_base,
            reps=reps, ks=ks, SCALES_PER_WMMA=SCALES_PER_WMMA,
            _lds_load_b128=_lds_load_b128,
            arith=arith, vector=vector, range_constexpr=range_constexpr,
        )

    return dict(
        _lds_load_b128=_lds_load_b128,
        load_data_frag=load_data_frag,
        load_b_frag=load_b_frag,
        load_scale_i32=load_scale_i32,
        _precompute_a_data_bases=_precompute_a_data_bases,
        _precompute_b_data_bases=_precompute_b_data_bases,
        _precompute_a_scale_lane_bases=_precompute_a_scale_lane_bases,
        _precompute_b_scale_lane_bases=_precompute_b_scale_lane_bases,
        load_scale_b128=load_scale_b128,
    )
