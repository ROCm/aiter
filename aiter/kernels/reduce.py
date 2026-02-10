"""Shared reduction helpers for FLIR example kernels.

These helpers build MLIR ops (flir/gpu/scf/vector/etc). They are extracted from
softmax/layernorm/rmsnorm kernels to de-duplicate code without changing codegen.
"""
from __future__ import annotations

from flydsl.dialects.ext.python_control_flow import lower_range_for_loops


def reduce_vec_max(vec_val, *, VEC_WIDTH, compute_type, vector):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    # Avoid fastmath on bf16 max reduction; some backends can fail to select.
    # The vector dialect expects a raw MLIR Value, not wrapper objects.
    try:
        from flydsl.dialects.ext import arith as _arith
        vec_val = _arith.as_value(vec_val)
    except Exception:
        pass
    return vector.reduction(compute_type, "maxnumf", vec_val)


def reduce_vec_sum(vec_val, *, VEC_WIDTH, compute_type, vector, fm_fast):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    try:
        from flydsl.dialects.ext import arith as _arith
        vec_val = _arith.as_value(vec_val)
    except Exception:
        pass
    return vector.reduction(compute_type, "add", vec_val, fastmath=fm_fast)


def make_block_reduce(*, tid, BLOCK_SIZE, compute_type, arith, gpu, flir, s_red_tv, T, ir, c_zero, c_neg_inf, c_zero_idx, fm_fast):
    """Return a `block_reduce(val, reduce_op_name)` function (softmax-style)."""

    def block_reduce(val, reduce_op_name):
        # AMD wavefront size is 64 on gfx9+/gfx10+/gfx11.
        WARP_SIZE = 64
        NUM_WAVES = (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE  # python int
        # Use Flir layout algebra to compute LDS indices for the reduction scratch.
        c_num_waves = flir.const_index(NUM_WAVES)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(c_num_waves)
        stride_red = flir.make_stride(c1)
        layout_red = flir.make_layout(shape_red, stride_red)

        # Some call sites pass ArithValue wrappers; normalize all operands to raw MLIR Values.
        tid_v = arith.as_value(tid)
        tid_i32 = arith.as_value(flir.arith.IndexCastOp(T.i32(), tid_v).result)
        c_warp_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        lane_i32 = arith.as_value(flir.arith.RemUIOp(tid_i32, c_warp_i32).result)
        wave_i32 = arith.as_value(flir.arith.DivUIOp(tid_i32, c_warp_i32).result)

        width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        w = arith.as_value(val)

        # Intra-wave reduction via xor shuffle
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.as_value(arith.constant(sh, type=T.i32()))
            peer = arith.as_value(gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult)
            if reduce_op_name == "max":
                w = flir.arith.MaximumFOp(arith.as_value(w), peer).result
            else:
                w = flir.arith.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result

        # lane0 writes per-wave partial into LDS s_red[wave_id]
        is_lane0 = arith.as_value(flir.arith.CmpIOp(
            flir.arith.CmpIPredicate.eq,
            lane_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        if is_lane0:
            wave_idx = flir.arith.IndexCastOp(T.index(), wave_i32).result
            red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)
            s_red_tv[red_idx] = w
        gpu.barrier()

        # wave0 reduces NUM_WAVES partials (still using shuffle)
        is_wave0 = arith.as_value(flir.arith.CmpIOp(
            flir.arith.CmpIPredicate.eq,
            wave_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        if is_wave0:
            in_range = arith.as_value(flir.arith.CmpIOp(
                flir.arith.CmpIPredicate.ult,
                lane_i32,
                arith.as_value(arith.constant(NUM_WAVES, type=T.i32())),
            ).result)

            # Predicated load: clamp lane index to 0 when out-of-range, then select.
            c0_i32 = arith.as_value(arith.constant(0, type=T.i32()))
            lane_safe_i32 = arith.as_value(flir.arith.SelectOp(in_range, lane_i32, c0_i32).result)
            lane_safe_idx = arith.as_value(flir.arith.IndexCastOp(T.index(), lane_safe_i32).result)
            red_idx = flir.crd2idx(flir.make_coord(lane_safe_idx), layout_red)
            v = arith.as_value(s_red_tv[red_idx])
            neutral = arith.as_value(c_neg_inf if reduce_op_name == "max" else c_zero)
            ww = arith.as_value(flir.arith.SelectOp(in_range, v, neutral).result)

            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.as_value(arith.constant(sh, type=T.i32()))
                peer = arith.as_value(gpu.ShuffleOp(arith.as_value(ww), off, width_i32, mode="xor").shuffleResult)
                if reduce_op_name == "max":
                    ww = flir.arith.MaximumFOp(arith.as_value(ww), peer).result
                else:
                    ww = flir.arith.AddFOp(arith.as_value(ww), peer, fastmath=fm_fast).result

            # lane0 writes final to s_red[0]
            is_lane0_2 = arith.as_value(flir.arith.CmpIOp(
                flir.arith.CmpIPredicate.eq,
                lane_i32,
                arith.as_value(arith.constant(0, type=T.i32())),
            ).result)
            if is_lane0_2:
                red_idx0 = flir.crd2idx(flir.make_coord(c_zero_idx), layout_red)
                s_red_tv[red_idx0] = ww
        gpu.barrier()

        red_idx0 = flir.crd2idx(flir.make_coord(c_zero_idx), layout_red)
        return s_red_tv[red_idx0]

    try:
        return lower_range_for_loops(block_reduce)
    except Exception:
        return block_reduce


def make_block_reduce_add(*, tid, fm_fast, WARP_SIZE, RED_SLOTS, gpu, arith, arith_ops, flir, T, ir, zero_idx, scratch_tv_shape_stride=(None, None)):
    """Return a `block_reduce_add(val_f32, scratch_memref)` function (norm-style)."""
    shape_unused, stride_unused = scratch_tv_shape_stride
    _ = shape_unused
    _ = stride_unused

    def block_reduce_add(val_f32, scratch_memref):
        # Fast path: single-wave block (RED_SLOTS==1) needs no LDS and no barrier.
        # After xor-shuffle reduction, all lanes hold the same reduced value.
        if RED_SLOTS == 1:
            width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
            w = arith.as_value(val_f32)
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.as_value(arith.constant(sh, type=T.i32()))
                peer = arith.as_value(gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult)
                w = arith.as_value(arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result)
            return w

        scratch_tv = flir.make_tensor(scratch_memref, shape=(RED_SLOTS,), strides=(1,))
        tid_v = tid.value if hasattr(tid, "value") else tid
        tid_v = arith.as_value(tid_v)
        tid_i32 = arith.as_value(arith_ops.IndexCastOp(T.i32(), tid_v).result)
        c_warp_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        lane_i32 = arith.as_value(arith_ops.RemUIOp(tid_i32, c_warp_i32).result)
        wave_i32 = arith.as_value(arith_ops.DivUIOp(tid_i32, c_warp_i32).result)
        width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        # Use Flir layout algebra to compute LDS indices for the reduction scratch.
        c_num_waves = flir.const_index(RED_SLOTS)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(c_num_waves)
        stride_red = flir.make_stride(c1)
        layout_red = flir.make_layout(shape_red, stride_red)

        w = arith.as_value(val_f32)
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.as_value(arith.constant(sh, type=T.i32()))
            peer = arith.as_value(gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult)
            w = arith.as_value(arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result)

        is_lane0 = arith.as_value(arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            lane_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        if is_lane0:
            wave_idx = arith_ops.IndexCastOp(T.index(), wave_i32).result
            red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)
            scratch_tv[red_idx] = w
        gpu.barrier()

        NUM_WAVES = RED_SLOTS
        is_wave0 = arith.as_value(arith_ops.CmpIOp(
            arith_ops.CmpIPredicate.eq,
            wave_i32,
            arith.as_value(arith.constant(0, type=T.i32())),
        ).result)
        # Only wave0 does final reduction and writes scratch[0].
        if is_wave0:
            in_range = arith.as_value(arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.ult,
                lane_i32,
                arith.as_value(arith.constant(NUM_WAVES, type=T.i32())),
            ).result)

            c0_i32 = arith.as_value(arith.constant(0, type=T.i32()))
            lane_safe_i32 = arith.as_value(flir.arith.SelectOp(in_range, lane_i32, c0_i32).result)
            lane_safe_idx = arith.as_value(arith_ops.IndexCastOp(T.index(), lane_safe_i32).result)
            red_idx = flir.crd2idx(flir.make_coord(lane_safe_idx), layout_red)
            v = scratch_tv[red_idx]
            z = arith.as_value(arith.constant(0.0, type=T.f32()))
            ww = arith.as_value(flir.arith.SelectOp(in_range, v, z).result)

            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.as_value(arith.constant(sh, type=T.i32()))
                peer = arith.as_value(gpu.ShuffleOp(arith.as_value(ww), off, width_i32, mode="xor").shuffleResult)
                ww = arith.as_value(arith_ops.AddFOp(arith.as_value(ww), peer, fastmath=fm_fast).result)

            is_lane0_2 = arith.as_value(arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                lane_i32,
                arith.as_value(arith.constant(0, type=T.i32())),
            ).result)
            if is_lane0_2:
                red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
                scratch_tv[red_idx0] = ww

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
        return scratch_tv[red_idx0]

    try:
        return lower_range_for_loops(block_reduce_add)
    except Exception:
        return block_reduce_add


def make_block_reduce_add2(*, tid, fm_fast, WARP_SIZE, RED_SLOTS, gpu, arith, arith_ops, flir, T, ir, zero_idx):
    """Return a `block_reduce_add2(a_f32, b_f32, scratch_a, scratch_b)` function.

    This is NOT pair-reduce: it reduces two independent scalars but shares the same
    cross-wave synchronization so we only pay the barriers once.
    """

    def _wave_reduce_add(x):
        # Normalize operands to raw MLIR Values: Shuffle/AddFOp expect `Value`, not wrappers.
        width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        w = arith.as_value(x)
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.as_value(arith.constant(sh, type=T.i32()))
            peer = arith.as_value(
                gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult
            )
            w = arith.as_value(arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result)
        return w

    def block_reduce_add2(val0_f32, val1_f32, scratch0_memref, scratch1_memref):
        # Single-wave block: no LDS/no barrier, just two wave reductions.
        if RED_SLOTS == 1:
            return _wave_reduce_add(val0_f32), _wave_reduce_add(val1_f32)

        scratch0_tv = flir.make_tensor(scratch0_memref, shape=(RED_SLOTS,), strides=(1,))
        scratch1_tv = flir.make_tensor(scratch1_memref, shape=(RED_SLOTS,), strides=(1,))

        tid_v = tid.value if hasattr(tid, "value") else tid
        tid_v = arith.as_value(tid_v)
        tid_i32 = arith.as_value(arith_ops.IndexCastOp(T.i32(), tid_v).result)
        c_warp_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
        lane_i32 = arith.as_value(arith_ops.RemUIOp(tid_i32, c_warp_i32).result)
        wave_i32 = arith.as_value(arith_ops.DivUIOp(tid_i32, c_warp_i32).result)

        # Layout for LDS scratch.
        c_num_waves = flir.const_index(RED_SLOTS)
        c1 = flir.const_index(1)
        shape_red = flir.make_shape(c_num_waves)
        stride_red = flir.make_stride(c1)
        layout_red = flir.make_layout(shape_red, stride_red)

        # Intra-wave reduce both values independently.
        w0 = _wave_reduce_add(val0_f32)
        w1 = _wave_reduce_add(val1_f32)

        # lane0 writes per-wave partials into LDS for both sums.
        is_lane0 = arith.as_value(
            arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                lane_i32,
                arith.as_value(arith.constant(0, type=T.i32())),
            ).result
        )
        if is_lane0:
            wave_idx = arith_ops.IndexCastOp(T.index(), wave_i32).result
            red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)
            scratch0_tv[red_idx] = w0
            scratch1_tv[red_idx] = w1
        gpu.barrier()

        # wave0 loads NUM_WAVES partials for both, reduces each with shuffle, writes scratch[0].
        is_wave0 = arith.as_value(
            arith_ops.CmpIOp(
                arith_ops.CmpIPredicate.eq,
                wave_i32,
                arith.as_value(arith.constant(0, type=T.i32())),
            ).result
        )
        if is_wave0:
            in_range = arith.as_value(
                arith_ops.CmpIOp(
                    arith_ops.CmpIPredicate.ult,
                    lane_i32,
                    arith.as_value(arith.constant(RED_SLOTS, type=T.i32())),
                ).result
            )

            c0_i32 = arith.as_value(arith.constant(0, type=T.i32()))
            lane_safe_i32 = arith.as_value(flir.arith.SelectOp(in_range, lane_i32, c0_i32).result)
            lane_safe_idx = arith.as_value(arith_ops.IndexCastOp(T.index(), lane_safe_i32).result)
            red_idx = flir.crd2idx(flir.make_coord(lane_safe_idx), layout_red)
            v0 = scratch0_tv[red_idx]
            v1 = scratch1_tv[red_idx]
            z = arith.as_value(arith.constant(0.0, type=T.f32()))
            ww0 = arith.as_value(flir.arith.SelectOp(in_range, v0, z).result)
            ww1 = arith.as_value(flir.arith.SelectOp(in_range, v1, z).result)

            ww0 = _wave_reduce_add(ww0)
            ww1 = _wave_reduce_add(ww1)

            is_lane0_2 = arith.as_value(
                arith_ops.CmpIOp(
                    arith_ops.CmpIPredicate.eq,
                    lane_i32,
                    arith.as_value(arith.constant(0, type=T.i32())),
                ).result
            )
            if is_lane0_2:
                red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
                scratch0_tv[red_idx0] = ww0
                scratch1_tv[red_idx0] = ww1

        gpu.barrier()
        red_idx0 = flir.crd2idx(flir.make_coord(zero_idx), layout_red)
        return scratch0_tv[red_idx0], scratch1_tv[red_idx0]

    try:
        return lower_range_for_loops(block_reduce_add2)
    except Exception:
        return block_reduce_add2
