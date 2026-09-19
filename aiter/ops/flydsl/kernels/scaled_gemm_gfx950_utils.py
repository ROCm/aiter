# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 LDS layouts and asynchronous loads from flydsl-examples.

Kept separate from A16W16: FP8 uses a different swizzle and copy layout.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

_LDS_BANK_PERIOD_LOG2 = 6
_LDS_READ_B128_BASE = 3
_LDS_READ_TR16_BASE = 4
_FP8_KCONTIG_SWIZZLE_MASK = 3
_FP8_KCONTIG_SWIZZLE_BASE = 4
_FP8_KCONTIG_SWIZZLE_SHIFT = 4


def _make_xor_swizzle(contiguous_extent, base):
    mask = _LDS_BANK_PERIOD_LOG2 - base
    shift = contiguous_extent.bit_length() - 1 - base
    return fx.static(fx.SwizzleType.get(mask, base, shift))


def make_lds_layout(rows, block_k, is_transposed):
    if const_expr(is_transposed):
        contiguous_extent = rows
        base = _LDS_READ_TR16_BASE
        order = (0, 1)
    else:
        contiguous_extent = block_k
        base = _LDS_READ_B128_BASE
        order = (1, 0)

    base_layout = fx.make_ordered_layout((rows, block_k), order)
    extent_log2 = contiguous_extent.bit_length() - 1
    mask = _LDS_BANK_PERIOD_LOG2 - base
    shift = extent_log2 - base
    is_power_of_two = contiguous_extent == 1 << extent_log2
    if const_expr(not is_power_of_two or shift < mask):
        return base_layout
    return fx.make_composed_layout(
        _make_xor_swizzle(contiguous_extent, base),
        base_layout,
    )


def _fp8_kcontig_swizzle(block_k):
    """K-contiguous ds_read_b128 XOR. Dest bits must stay inside K so the
    swizzle permutes only columns (r == row); otherwise G2S ``% block_k``
    and S2R ``crd2idx`` disagree.

    ``Swizzle<3,4,4>`` is FlyDSL ``swizzle_128``: dest [4, 7), which fits
    ``block_k >= 128``. ``block_k == 64`` only has two K-bits above the
    16-element vector, so dest is [4, 6) and the source is row[1:3].
    """
    if const_expr(block_k >= 128):
        return fx.SwizzleType.get(
            _FP8_KCONTIG_SWIZZLE_MASK,
            _FP8_KCONTIG_SWIZZLE_BASE,
            _FP8_KCONTIG_SWIZZLE_SHIFT,
        )
    if const_expr(block_k == 64):
        return fx.SwizzleType.get(2, 4, 3)
    return None


def make_fp8_lds_layout(rows, block_k, is_k_major):
    """LDS layout for gfx950 FP8 PTPC.

    K-contiguous (NT A/B) uses a block_k-dependent XOR swizzle; see
    ``_fp8_kcontig_swizzle``. K-major uses the same 16-element XOR groups
    as ``ds_read_tr16``. ``ds_read_tr8`` is still a 16-lane cooperative
    op; scrambling bits below 16-element alignment breaks the address
    pattern the instruction expects, even though each load is only 8 bytes.
    """
    if const_expr(is_k_major):
        return make_lds_layout(rows, block_k, is_transposed=True)
    base_layout = fx.make_ordered_layout((rows, block_k), (1, 0))
    swizzle = _fp8_kcontig_swizzle(block_k)
    if swizzle is None:
        return base_layout
    return fx.make_composed_layout(fx.static(swizzle), base_layout)


def swizzled_contiguous_idx(idx0, idx1, layout, extent):
    # The XOR swizzle is self-inverse. Map each physical contiguous position
    # written by direct-to-LDS DMA back to its logical global vector.
    elem_offset = fx.get_scalar(fx.crd2idx((idx0, idx1), layout))
    return elem_offset % extent


def async_load_operand(
    operand,
    lds_base,
    global_outer_offset,
    k_tile,
):
    context = operand.context
    param = context.param
    tid = context.tid
    block_threads = param.block_threads
    async_load_vec_size = param.async_load_bytes // param.in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_k = param.block_k
    elem_bytes = operand.src_base.dtype.width // 8
    lds_ptr = lds_base + fx.Int32(context.wave_offset) // elem_bytes
    g2s_copy_layout = fx.make_layout(async_load_vec_size, 1)
    for i in range_constexpr(operand.load_iters):
        global_tid = block_threads * i + tid
        if const_expr(operand.is_k_major):
            outer_x_threads = operand.outer_tile_size // async_load_vec_size
            outer_lds_idx = global_tid % outer_x_threads * async_load_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = swizzled_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                operand.lds_layout,
                operand.outer_tile_size,
            )
            global_k_idx = context.ks_begin + k_tile * block_k + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = (
                context.ks_begin
                + k_tile * block_k
                + swizzled_contiguous_idx(
                    outer_local_idx,
                    k_local_idx,
                    operand.lds_layout,
                    block_k,
                )
            )
        global_outer_idx = global_outer_offset + outer_local_idx
        safe_global_outer_idx = (global_outer_idx < operand.outer_bound).select(
            global_outer_idx, 0
        )
        if const_expr(operand.is_preshuffled):
            # shuffle_weight(..., layout=(16, 16)): [N/16,K/16,16,16].
            # Every DMA is a contiguous 16-byte strip; permute only its base.
            global_offset = (
                (safe_global_outer_idx // 16) * operand.leading_stride * 16
                + (global_k_idx // 16) * 256
                + (safe_global_outer_idx % 16) * 16
                + global_k_idx % 16
            )
        elif const_expr(operand.is_k_major):
            global_offset = (
                global_k_idx * operand.leading_stride + safe_global_outer_idx
            )
        else:
            global_offset = (
                safe_global_outer_idx * operand.leading_stride + global_k_idx
            )
        src = fx.make_view(operand.src_base + global_offset, g2s_copy_layout)
        dst = fx.make_view(lds_ptr, g2s_copy_layout)
        rocdl.sched_barrier(0)
        fx.copy_atom_call(context.async_g2s_copy_atom, src, dst)
        rocdl.sched_barrier(0)
        if i < operand.load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size
