# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A-preshuffled MXFP4 quant and scatter kernels for the (4, 16, 16) layout."""

from collections.abc import Callable
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import gemm_common_gfx1250 as gemm_common
from aiter.ops.flydsl.kernels import moe_fused_route_quant_scatter as route_quant
from aiter.ops.flydsl.kernels import tensor_shim
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _ROUND_MODE as _lane16_round_mode,
)
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _SCALE_RSRC_MAX_BYTES,
    _cvt_scalef32_pk8_fp4_bf16,
    _emit_quant_one_k_group,
    _quant_layout,
    emit_mx_e8m0_scale,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)


@fx.struct
class _ApreFusedTileStorageK1024:
    """M16 x K1024 tile with one padding dword per M row and K256 slice."""

    payload: fx.Array[fx.Int32, 2112, 16]


@fx.struct
class _ApreFusedScaleStorage:
    """Padded M16 x 8-dword scale transpose tile."""

    scale: fx.Array[fx.Int32, 144, 16]


def build_moe_apre_quant_module(
    feat_dim: int,
    wmma_rep: int,
):
    """Source-centric quantization for the A-preshuffle quant/scatter path."""
    if feat_dim > 0 and feat_dim % 512 == 0 and int(_lane16_round_mode) == 1:
        return _build_apre_quant_k512(feat_dim, wmma_rep)
    L = _quant_layout(feat_dim, "fp4", wmma_rep)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle quant requires gfx1250 pk8")
    if feat_dim % 256:
        raise ValueError("A-preshuffle quant requires K divisible by 256")
    if L.wave_size != 32:
        raise NotImplementedError("A-preshuffle quant requires wave32")

    block_threads = 512
    waves_per_block = 16
    k256_tiles = feat_dim // 256
    payload_bytes_per_row = L.payload_bytes_per_row
    scale_dwords_per_row = L.mx_blocks_per_row // 4
    module_name = f"moe_apre_quant_fd{feat_dim}_r{wmma_rep}_fp4_pk8_w16_qks"

    @flyc.kernel(name=module_name, known_block_size=[block_threads, 1, 1])
    def quant_kernel(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        token_num: Int32,
        grid_blocks: Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_wave = arith.constant(L.wave_size, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        wave = tid // fx.Uint32(L.wave_size)
        lane = tid - wave * fx.Uint32(L.wave_size)
        q_wave = bid * fx.Uint32(waves_per_block) + wave
        q_stride = fx.Uint32(grid_blocks) * fx.Uint32(waves_per_block)
        # Split the source row into independent K256 jobs; quant math is unchanged.
        for job in range(q_wave, fx.Uint32(token_num) * k256_tiles, q_stride):
            job_u32 = fx.Uint32(job)
            src_token = job_u32 // fx.Uint32(k256_tiles)
            k256 = job_u32 - src_token * fx.Uint32(k256_tiles)
            for _ in range_constexpr(1):
                block_in_wave = lane // fx.Uint32(L.lanes_per_mx_block)
                lane_in_block = lane - block_in_wave * fx.Uint32(L.lanes_per_mx_block)
                qc = SimpleNamespace(
                    i32=i32,
                    f32=f32,
                    block_iters=1,
                    payload_base=fx.Int64(ptrtoint(compact_payload)),
                    payload_bytes_per_row=payload_bytes_per_row,
                    hidden_base=fx.Int64(ptrtoint(hidden)),
                    feat_bytes_per_row=feat_dim * 2,
                    feat_row_i32=src_token,
                    payload_num_records=payload_bytes_per_row,
                    prequantized=False,
                    mx_blocks_per_wave_iter=L.mx_blocks_per_wave_iter,
                    mx_blocks_per_row=L.mx_blocks_per_row,
                    amax_shuffle_dists=L.amax_shuffle_dists,
                    is_fp8=False,
                    use_native=L.use_native,
                    use_pk8=True,
                    mx_dtype=L.mx_dtype,
                    c0_i32=c0_i32,
                    c1_i32=c1_i32,
                    c4_i32=c4_i32,
                    c23_i32=c23_i32,
                    c254_i32=c254_i32,
                    c0_f32=c0_f32,
                    c_wave=c_wave,
                    c_elems_per_lane=arith.constant(L.elems_per_lane, type=i32),
                    c_payload_bytes_per_block=arith.constant(
                        L.payload_bytes_per_block, type=i32
                    ),
                    c_payload_bytes_per_lane=arith.constant(
                        L.payload_bytes_per_lane, type=i32
                    ),
                    c_wmma_rep=arith.constant(wmma_rep, type=i32),
                    block_in_wave=block_in_wave,
                    lane_in_block=lane_in_block,
                    is_block_lead=lane_in_block == c0_i32,
                    payload_dests=[SimpleNamespace(payload_row_i32=src_token)],
                    dests=[
                        SimpleNamespace(
                            payload_row_i32=src_token,
                            scale_row_dword_base=c0_i32,
                        )
                    ],
                    scale_t=ptr_buf_tensor(compact_scale, fx.Int8),
                    scale_rsrc=buffer_ops.create_buffer_resource_from_addr(
                        fx.Int64(ptrtoint(compact_scale)),
                        num_records_bytes=_SCALE_RSRC_MAX_BYTES,
                    ),
                    scale_pack_dwords=True,
                    compact_scale=True,
                    c_scale_dwords_per_row=arith.constant(
                        scale_dwords_per_row, type=i32
                    ),
                    mx_group_base=k256,
                )
                _emit_quant_one_k_group(qc, k256)

    @flyc.jit
    def launch_quant(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        token_num: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quant_kernel(
            hidden,
            compact_payload,
            compact_scale,
            token_num,
            grid_blocks,
        ).launch(
            grid=(arith.index_cast(T.index, grid_blocks), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_quant.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_quant


def _build_apre_quant_k512(feat_dim: int, wmma_rep: int):
    """One wave owns K512; each two-lane pair retains one original K32 scale."""
    L = _quant_layout(feat_dim, "fp4", wmma_rep)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle quant requires gfx1250 pk8")
    if L.wave_size != 32:
        raise NotImplementedError("A-preshuffle quant requires wave32")
    block_threads = 512
    name = f"moe_apre_quant_fd{feat_dim}_r{wmma_rep}_fp4_pk8_w16_qks_k512_pkamax"

    from aiter.ops.flydsl.kernels import moe_fused_route_quant_scatter as common

    def _packed16_roundup_amax(bf16_lo, bf16_hi, c):
        # The fast factory is RoundUp-only. Finite positive BF16 magnitudes have
        # the same unsigned ordering; every NaN still yields scale code 255.
        lo_words = fx.Vector(bf16_lo).bitcast(fx.Uint32)
        hi_words = fx.Vector(bf16_hi).bitcast(fx.Uint32)
        masked = [(lo_words[j] & fx.Uint32(0x7FFF7FFF)) for j in range_constexpr(4)]
        masked += [(hi_words[j] & fx.Uint32(0x7FFF7FFF)) for j in range_constexpr(4)]
        first = fx.Uint32(
            llvm.inline_asm(
                T.i32,
                [common._raw(x) for x in masked[:3]],
                "v_pk_max3_u16 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )
        )
        second = fx.Uint32(
            llvm.inline_asm(
                T.i32,
                [common._raw(x) for x in masked[3:6]],
                "v_pk_max3_u16 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )
        )
        tail = fx.Uint32(
            llvm.inline_asm(
                T.i32,
                [common._raw(masked[6]), common._raw(masked[7]), common._raw(first)],
                "v_pk_max3_u16 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )
        )
        both = fx.Uint32(
            llvm.inline_asm(
                T.i32,
                [common._raw(tail), common._raw(second)],
                "v_pk_max_u16 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            )
        )
        magnitude = fx.max(both & fx.Uint32(0xFFFF), both >> fx.Uint32(16))
        positive = (magnitude << fx.Uint32(16)).bitcast(fx.Float32)
        # Retain the original FP32 input-denorm handling in the final maximum.
        return fx.max(fx.Float32(c.c0_f32), positive)

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def quant_kernel(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        token_num: Int32,
        grid_blocks: Int32,
    ):
        i32, f32 = T.i32, T.f32
        c0_f32 = arith.constant(0.0, type=f32)
        c_wave = arith.constant(32, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        wave = tid // fx.Uint32(32)
        lane = tid - wave * fx.Uint32(32)
        q_wave = fx.Uint32(
            rocdl.readfirstlane(T.i32, fx.Int32(bid * fx.Uint32(16) + wave).ir_value())
        )
        q_stride = fx.Uint32(grid_blocks) * fx.Uint32(16)
        # Preserve the parent's Uint32 job-count boundary before pairing jobs.
        limit = (fx.Uint32(token_num) * fx.Uint32(feat_dim // 256)) // fx.Uint32(2)
        scales = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(compact_scale)), num_records_bytes=_SCALE_RSRC_MAX_BYTES
        )
        for job in range(q_wave, limit, q_stride):
            cell = fx.Uint32(job)
            # Cell c covers original K256 jobs 2*c and 2*c+1.
            source = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(hidden)) + fx.Uint64(cell) * 1024,
                num_records_bytes=1024,
            )
            payload = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(compact_payload)) + fx.Uint64(cell) * 256,
                num_records_bytes=256,
            )
            input_dword = lane * fx.Uint32(8)
            lo = fx.Vector(
                buffer_ops.buffer_load(source, input_dword, vec_width=4, dtype=i32)
            )
            hi = fx.Vector(
                buffer_ops.buffer_load(
                    source, input_dword + fx.Uint32(4), vec_width=4, dtype=i32
                )
            )
            bf16_type = fx.Numeric.from_ir_type(T.bf16)
            bf16_lo, bf16_hi = lo.bitcast(bf16_type), hi.bitcast(bf16_type)
            context = SimpleNamespace(c0_f32=c0_f32)
            local_amax = _packed16_roundup_amax(bf16_lo, bf16_hi, context)
            peer_amax = local_amax.shuffle_xor(arith.constant(1, type=i32), c_wave)
            block_amax = fx.max(local_amax, peer_amax)
            scale = emit_mx_e8m0_scale(
                block_amax, mode=_lane16_round_mode, dtype=L.mx_dtype
            )
            scale_f32 = (ArithValue(scale) << c23_i32).bitcast(f32)
            packed_lo = _cvt_scalef32_pk8_fp4_bf16(bf16_lo, scale_f32, i32_ty=i32)
            packed_hi = _cvt_scalef32_pk8_fp4_bf16(bf16_hi, scale_f32, i32_ty=i32)
            pair = fx.Vector.from_elements(
                [fx.Int32(packed_lo), fx.Int32(packed_hi)], fx.Numeric.from_ir_type(i32)
            )
            buffer_ops.buffer_store(pair, payload, lane * fx.Uint32(2))
            byte = fx.Uint32(scale) & fx.Uint32(255)
            half = byte | (
                byte.shuffle_xor(arith.constant(2, type=i32), c_wave) << fx.Uint32(8)
            )
            word = half | (
                half.shuffle_xor(arith.constant(4, type=i32), c_wave) << fx.Uint32(16)
            )
            if lane % fx.Uint32(8) == fx.Uint32(0):
                buffer_ops.buffer_store(
                    word, scales, cell * fx.Uint32(4) + lane // fx.Uint32(8)
                )

    @flyc.jit
    def launch_quant(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        token_num: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        quant_kernel(
            hidden, compact_payload, compact_scale, token_num, grid_blocks
        ).launch(
            grid=(arith.index_cast(T.index, grid_blocks), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_quant.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_quant


def build_moe_apre_fused_quant_scatter_module(
    feat_dim: int,
    wmma_rep: int,
    scatter_only: bool = False,
):
    """Persistent two-phase source-quant + M16 A-preshuffle scatter.

    Phase Q is source-centric: each wave owns one ``(source token, K256)`` job
    and writes row-major compact FP4 payload/e8m0 scale exactly once.  After an
    agent-scope grid barrier, Phase S is destination-centric: one 16-wave
    workgroup owns one complete M16 and processes a configurable K tile, loads
    the 16 compact source rows selected by ``row_to_token``, and transposes them
    through conflict-free LDS into the ``(4,16,16-byte)`` GEMM layout.

    With ``scatter_only=True``, use an independent direct-gather launch after
    quant completes on the same stream, without a grid barrier or LDS transpose.

    The fused launch grid must be persistent (all workgroups resident together), since
    the phase boundary is a hand-rolled global barrier.  On gfx1250, four
    512-thread workgroups per CU fill the 64 available wave slots.
    """
    L = _quant_layout(feat_dim, "fp4", wmma_rep)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle fused quant requires gfx1250 pk8")
    tile_k = 1024
    if feat_dim % tile_k:
        raise ValueError("A-preshuffle fused quant requires K divisible by 1024")

    wave_size = L.wave_size
    if wave_size != 32:
        raise NotImplementedError("A-preshuffle fused quant requires wave32")
    if scatter_only:
        return _build_apre_gather_scatter(feat_dim, wmma_rep)
    block_threads = 512
    waves_per_block = 16
    k256_tiles = feat_dim // 256
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_dwords_per_row = payload_bytes_per_row // 4
    scale_dwords_per_row = L.mx_blocks_per_row // 4
    module_kind = "scatter" if scatter_only else "fused_quant_scatter"
    module_name = (
        f"moe_apre_{module_kind}_fd{feat_dim}_r{wmma_rep}_" f"fp4_pk8_w16_tk{tile_k}"
    )
    _is_gfx12 = str(L.arch).startswith("gfx12")

    @flyc.kernel(name=module_name, known_block_size=[block_threads, 1, 1])
    def fused_kernel(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        barrier: fx.Pointer,
        token_num: Int32,
        n_rows: Int32,
        grid_blocks: Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_wave = arith.constant(wave_size, type=i32)
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        def _wait_mem():
            if const_expr(_is_gfx12):
                rocdl.s_wait_loadcnt(0)
                rocdl.s_wait_storecnt(0)
            else:
                rocdl.s_waitcnt(0)

        def _elem_ptr(tensor, elem_idx_i32):
            addr = fx.Int64(ptrtoint(tensor)) + fx.Int64(elem_idx_i32) * 4
            p = create_llvm_ptr(addr)
            return p._value if hasattr(p, "_value") else p

        def _atomic_add(tensor, elem_idx_i32, addend):
            return fx.Uint32(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    _elem_ptr(tensor, elem_idx_i32),
                    addend,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
            )

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        wave = tid // fx.Uint32(wave_size)
        lane = tid - wave * fx.Uint32(wave_size)

        # ================= Phase Q: source token -> compact MX row =================
        q_wave = bid * fx.Uint32(waves_per_block) + wave
        q_stride = fx.Uint32(grid_blocks) * fx.Uint32(waves_per_block)
        q_token_num = fx.Uint32(0) if const_expr(scatter_only) else fx.Uint32(token_num)
        for src_token in range(q_wave, q_token_num, q_stride):
            for k256 in range(fx.Uint32(0), fx.Uint32(k256_tiles), 1):
                block_in_wave = lane // fx.Uint32(L.lanes_per_mx_block)
                lane_in_block = lane - block_in_wave * fx.Uint32(L.lanes_per_mx_block)
                qc = SimpleNamespace(
                    i32=i32,
                    f32=f32,
                    block_iters=1,
                    payload_base=fx.Int64(ptrtoint(compact_payload)),
                    payload_bytes_per_row=payload_bytes_per_row,
                    hidden_base=fx.Int64(ptrtoint(hidden)),
                    feat_bytes_per_row=feat_dim * 2,
                    feat_row_i32=src_token,
                    payload_num_records=payload_bytes_per_row,
                    prequantized=False,
                    mx_blocks_per_wave_iter=L.mx_blocks_per_wave_iter,
                    mx_blocks_per_row=L.mx_blocks_per_row,
                    amax_shuffle_dists=L.amax_shuffle_dists,
                    is_fp8=False,
                    use_native=L.use_native,
                    use_pk8=True,
                    mx_dtype=L.mx_dtype,
                    c0_i32=c0_i32,
                    c1_i32=c1_i32,
                    c4_i32=c4_i32,
                    c23_i32=c23_i32,
                    c254_i32=c254_i32,
                    c0_f32=c0_f32,
                    c_wave=c_wave,
                    c_elems_per_lane=arith.constant(L.elems_per_lane, type=i32),
                    c_payload_bytes_per_block=arith.constant(
                        L.payload_bytes_per_block, type=i32
                    ),
                    c_payload_bytes_per_lane=arith.constant(
                        L.payload_bytes_per_lane, type=i32
                    ),
                    c_wmma_rep=arith.constant(wmma_rep, type=i32),
                    block_in_wave=block_in_wave,
                    lane_in_block=lane_in_block,
                    is_block_lead=lane_in_block == c0_i32,
                    payload_dests=[SimpleNamespace(payload_row_i32=src_token)],
                    dests=[
                        SimpleNamespace(
                            payload_row_i32=src_token, scale_row_dword_base=c0_i32
                        )
                    ],
                    scale_t=ptr_buf_tensor(compact_scale, fx.Int8),
                    scale_rsrc=buffer_ops.create_buffer_resource_from_addr(
                        fx.Int64(ptrtoint(compact_scale)),
                        num_records_bytes=_SCALE_RSRC_MAX_BYTES,
                    ),
                    scale_pack_dwords=True,
                    compact_scale=True,
                    c_scale_dwords_per_row=arith.constant(
                        scale_dwords_per_row, type=i32
                    ),
                    mx_group_base=k256,
                )
                _emit_quant_one_k_group(qc, k256)

        # Every compact row must be globally visible before any destination reads
        # it. All launched workgroups are resident, so this atomic spin barrier
        # cannot wait on an unscheduled producer.
        if const_expr(not scatter_only):
            # Publish every wave's compact stores at device scope before the WG
            # leader announces arrival; waitcnt alone does not promote scope.
            if q_wave < q_token_num:
                llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="agent")
                _wait_mem()
                rocdl.sched_barrier(0)
            gpu.barrier()
            if tid == fx.Uint32(0):
                arrival = _atomic_add(barrier, c0_i32, c1_i32)
                last = arrival == fx.Uint32(grid_blocks) - fx.Uint32(1)
                if last:
                    _wait_mem()
                    _atomic_add(barrier, c1_i32, c1_i32)
                else:
                    released = fx.Uint32(0)
                    while released == fx.Uint32(0):
                        released = _atomic_add(barrier, c1_i32, c0_i32)
            gpu.barrier()
            # All consumer waves must invalidate stale compact rows after the
            # leader observes the release, including rows produced by other WGs.
            llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="agent")
            _wait_mem()
            rocdl.sched_barrier(0)

        # ============== Phase S: compact rows -> one complete M16 tile ==============
        shared_allocator = fx.SharedAllocator()
        tile_storage_obj = shared_allocator.allocate(_ApreFusedTileStorageK1024).peek()
        tile_lds = tile_storage_obj.payload.ptr
        tile_lds_idx = fx.Index(fx.ptrtoint(tile_lds))
        scale_lds = shared_allocator.allocate(_ApreFusedScaleStorage).peek().scale.ptr
        compact_payload_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(compact_payload)),
            num_records_bytes=_SCALE_RSRC_MAX_BYTES,
        )
        grouped_payload_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_payload)),
            num_records_bytes=_SCALE_RSRC_MAX_BYTES,
        )
        compact_scale_dwords = fx.Uint32(token_num) * fx.Uint32(scale_dwords_per_row)
        grouped_scale_dwords = fx.Uint32(n_rows) * fx.Uint32(scale_dwords_per_row)
        compact_scale_p = ptr_buf_tensor(
            compact_scale,
            fx.Int32,
            num_records_bytes=compact_scale_dwords * fx.Uint32(4),
        )
        scale_p = ptr_buf_tensor(
            grouped_scale,
            fx.Int32,
            num_records_bytes=grouped_scale_dwords * fx.Uint32(4),
        )
        r2t_p = ptr_buf_tensor(row_to_token)

        m16_tiles = fx.Uint32(n_rows) // fx.Uint32(16)
        for tile in range(bid, m16_tiles, fx.Uint32(grid_blocks)):
            row16 = wave
            row = tile * fx.Uint32(16) + row16
            # Fetch each row mapping once per wave, then use LDS both to
            # broadcast it across the wave and to let thread 0 detect a fully
            # empty M16 tile.  The resulting branch is uniform for the TG, so
            # it is safe to skip all K-slice barriers together.
            if lane == fx.Uint32(0):
                tile_lds[row16] = r2t_p[row]
            gpu.barrier()
            src_raw = fx.Int32(tile_lds[row16])
            valid = src_raw >= fx.Int32(0)
            src_row = valid.select(fx.Uint32(src_raw), fx.Uint32(0))

            if tid == fx.Uint32(0):
                tile_has_rows = c0_i32
                for tile_row in range(16):
                    tile_row_valid = fx.Int32(tile_lds[fx.Uint32(tile_row)]) >= c0_i32
                    tile_has_rows = fx.Int32(
                        arith.ori(
                            arith.unwrap(tile_has_rows),
                            arith.unwrap(tile_row_valid.select(c1_i32, c0_i32)),
                        )
                    )
                tile_lds[fx.Uint32(16)] = tile_has_rows
            gpu.barrier()
            tile_has_rows = fx.Int32(tile_lds[fx.Uint32(16)]) != c0_i32

            if tile_has_rows:
                for k_tile in range(fx.Uint32(0), fx.Uint32(k256_tiles), 4):
                    # K1024 contains eight packed scale dwords per row. Map them
                    # onto lanes 0..7 so the global scale accesses coalesce.
                    scale_lane = lane < fx.Uint32(8)
                    scale_k_inner = lane // fx.Uint32(2)
                    scale_half = lane - scale_k_inner * fx.Uint32(2)
                    scale_k256 = k_tile + scale_k_inner
                    src_scale_idx = (valid & scale_lane).select(
                        src_row * fx.Uint32(scale_dwords_per_row)
                        + scale_k256 * fx.Uint32(2)
                        + scale_half,
                        compact_scale_dwords,
                    )
                    scale_dword = compact_scale_p[src_scale_idx]
                    rocdl.sched_barrier(0)

                    # One wave owns one compact source row. Across K1024,
                    # 32 lanes x 4 dwords exactly cover its 128 payload dwords.
                    src_vec_dword = (
                        src_row * fx.Uint32(payload_dwords_per_row)
                        + k_tile * fx.Uint32(32)
                        + lane * fx.Uint32(4)
                    )
                    packed_raw = fx.Vector(
                        buffer_ops.buffer_load(
                            compact_payload_rsrc,
                            src_vec_dword,
                            vec_width=4,
                            dtype=i32,
                        )
                    )
                    packed_vec = fx.Vector.from_elements(
                        [
                            valid.select(ArithValue(packed_raw[j]), c0_i32)
                            for j in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    rocdl.sched_barrier(0)

                    # The ninth dword in each scale row breaks stride-8 LDS bank
                    # aliasing. Inactive lanes write only to the padding column.
                    scale_scratch_row = (lane + fx.Uint32(8)) % fx.Uint32(16)
                    scale_lds_idx = scale_lane.select(
                        row16 * fx.Uint32(9) + lane,
                        scale_scratch_row * fx.Uint32(9) + fx.Uint32(8),
                    )
                    scale_lds[scale_lds_idx] = scale_dword
                    rocdl.sched_barrier(0)

                    lds_k_inner = lane // fx.Uint32(8)
                    lds_k32 = lane - lds_k_inner * fx.Uint32(8)
                    lds_group = (lds_k32 + row16 // fx.Uint32(4)) % fx.Uint32(8)
                    lds_idx = (
                        lds_k_inner * fx.Uint32(528)
                        + row16 * fx.Uint32(33)
                        + lds_group * fx.Uint32(4)
                    )
                    lds_store_b128(
                        tile_lds_idx, fx.Index(lds_idx * fx.Uint32(4)), packed_vec
                    )
                    gpu.barrier()
                    tile_base = tile * fx.Uint32(16 * payload_dwords_per_row)
                    vec_dword = tid * fx.Uint32(4)
                    vec_k_inner = vec_dword // fx.Uint32(512)
                    vec_in_k256 = vec_dword - vec_k_inner * fx.Uint32(512)
                    vec_k32 = vec_in_k256 // fx.Uint32(64)
                    vec_rem = vec_in_k256 - vec_k32 * fx.Uint32(64)
                    vec_row = vec_rem // fx.Uint32(4)
                    vec_group = (vec_k32 + vec_row // fx.Uint32(4)) % fx.Uint32(8)
                    vec_lds_base = (
                        vec_k_inner * fx.Uint32(528)
                        + vec_row * fx.Uint32(33)
                        + vec_group * fx.Uint32(4)
                    )
                    out_vec = lds_load_b128(
                        tile_lds_idx,
                        fx.Index(vec_lds_base * fx.Uint32(4)),
                    )

                    scale_out_slot = row16 // fx.Uint32(2)
                    scale_out_row = (row16 - scale_out_slot * fx.Uint32(2)) * fx.Uint32(
                        8
                    ) + lane
                    scale_out_active = lane < fx.Uint32(8)
                    scale_scratch_row = (lane + fx.Uint32(8)) % fx.Uint32(16)
                    scale_read_idx = scale_out_active.select(
                        scale_out_row * fx.Uint32(9) + scale_out_slot,
                        scale_scratch_row * fx.Uint32(9) + fx.Uint32(8),
                    )
                    scale_out = scale_lds[scale_read_idx]

                    scale_out_k_inner = scale_out_slot // fx.Uint32(2)
                    scale_out_half = scale_out_slot - scale_out_k_inner * fx.Uint32(2)
                    scale_out_k256 = k_tile + scale_out_k_inner
                    scale_tile = tile // fx.Uint32(wmma_rep)
                    wmma_row = tile - scale_tile * fx.Uint32(wmma_rep)
                    scale_base = scale_tile * fx.Uint32(
                        scale_dwords_per_row * wmma_rep * 16
                    ) + wmma_row * fx.Uint32(16)
                    dst_scale_idx = scale_out_active.select(
                        scale_base
                        + scale_out_k256 * fx.Uint32(2 * wmma_rep * 16)
                        + scale_out_half * fx.Uint32(wmma_rep * 16)
                        + scale_out_row,
                        grouped_scale_dwords,
                    )
                    gpu.barrier()
                    scale_p[dst_scale_idx] = scale_out
                    buffer_ops.buffer_store(
                        out_vec,
                        grouped_payload_rsrc,
                        tile_base + k_tile * fx.Uint32(512) + vec_dword,
                    )

    @flyc.jit
    def launch_fused(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        barrier: fx.Pointer,
        token_num: fx.Int32,
        n_rows: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        fused_kernel(
            hidden,
            compact_payload,
            compact_scale,
            grouped_payload,
            grouped_scale,
            row_to_token,
            barrier,
            token_num,
            n_rows,
            grid_blocks,
        ).launch(
            grid=(arith.index_cast(T.index, grid_blocks), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_fused


def _build_apre_gather_scatter(feat_dim: int, wmma_rep: int):
    """Gather four M16/K1024 tiles per WG after quant completes on the same stream."""
    block_threads = 512
    k_tiles = feat_dim // 1024
    payload_dwords = feat_dim // 8
    scale_dwords = feat_dim // 128
    name = f"moe_apre_gather_fd{feat_dim}_r{wmma_rep}_w16_m64_tk1024_oob_smem512"

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def scatter_kernel(
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        token_num: Int32,
        n_rows: Int32,
    ):
        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        wg_tile = bid // fx.Uint32(k_tiles)
        kt = bid - wg_tile * fx.Uint32(k_tiles)
        local_tid = tid % fx.Uint32(128)
        tile = wg_tile * fx.Uint32(4) + tid // fx.Uint32(128)
        row16 = tid % fx.Uint32(16)
        if tile < fx.Uint32(n_rows) // fx.Uint32(16):
            mapping_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(row_to_token)) + fx.Int64(tile) * 64,
                num_records_bytes=64,
            )
            raw_rows = fx.Vector(
                llvm.call_intrinsic(
                    ir.VectorType.get([16], T.i32),
                    "llvm.amdgcn.s.buffer.load.v16i32",
                    [
                        buffer_ops._ptr8_to_v4i32(mapping_rsrc),
                        fx.Int32(0).ir_value(),
                        fx.Int32(16).ir_value(),
                    ],
                    [],
                    [],
                )
            )
            source_rows = [fx.Int32(raw_rows[r]) for r in range_constexpr(16)]
            selected_rows = source_rows
            for bit in range_constexpr(4):
                select_hi = (row16 & fx.Uint32(1 << bit)) != fx.Uint32(0)
                selected_rows = [
                    select_hi.select(selected_rows[r + 1], selected_rows[r])
                    for r in range_constexpr(0, 16 >> bit, 2)
                ]
            src_raw = selected_rows[0]
            valid = src_raw >= fx.Int32(0)
            src_row = valid.select(fx.Uint32(src_raw), fx.Uint32(0))
            # The scalar load contains every signed mapping row of this M16.
            max_row = source_rows[0]
            for r in range_constexpr(1, 16):
                max_row = fx.max(max_row, source_rows[r])
            if max_row >= fx.Int32(0):
                payload_in = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(compact_payload)),
                    num_records_bytes=_SCALE_RSRC_MAX_BYTES,
                )
                payload_out = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(grouped_payload)),
                    num_records_bytes=_SCALE_RSRC_MAX_BYTES,
                )
                scales_in = ptr_buf_tensor(
                    compact_scale,
                    fx.Int32,
                    num_records_bytes=fx.Uint32(token_num) * (scale_dwords * 4),
                )
                scales_out = ptr_buf_tensor(
                    grouped_scale,
                    fx.Int32,
                    num_records_bytes=fx.Uint32(n_rows) * (scale_dwords * 4),
                )
                scale_col = local_tid // fx.Uint32(16)
                scale = fx.Int32(scales_in[src_row * scale_dwords + kt * 8 + scale_col])
                # All four B128 loads of an invalid row are beyond num_records;
                # hardware returns zero without a select for each payload dword.
                payload_dword = valid.select(
                    src_row * payload_dwords + kt * 128 + scale_col * 4,
                    fx.Uint32(0x20000000),
                )
                chunks = []
                for it in range_constexpr(4):
                    chunks.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                payload_in,
                                payload_dword,
                                vec_width=4,
                                dtype=T.i32,
                                soffset_bytes=it * 128,
                            )
                        )
                    )
                for it in range_constexpr(4):
                    buffer_ops.buffer_store(
                        chunks[it],
                        payload_out,
                        tile * (16 * payload_dwords)
                        + kt * 2048
                        + it * 512
                        + local_tid * 4,
                    )
                dst_scale = (
                    (tile // wmma_rep) * (16 * wmma_rep * scale_dwords)
                    + (tile % wmma_rep) * 16
                    + (kt * 8 + scale_col) * (16 * wmma_rep)
                    + row16
                )
                scales_out[dst_scale] = valid.select(scale, fx.Int32(0))

    @flyc.jit
    def launch_scatter(
        hidden: fx.Pointer,
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        barrier: fx.Pointer,
        token_num: fx.Int32,
        n_rows: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        scatter_kernel(
            compact_payload,
            compact_scale,
            grouped_payload,
            grouped_scale,
            row_to_token,
            token_num,
            n_rows,
        ).launch(
            grid=(
                arith.index_cast(
                    T.index,
                    fx.max(((n_rows // fx.Int32(16) + 3) // 4) * k_tiles, fx.Int32(1)),
                ),
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_scatter.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_scatter


# Each K1024 slice contains four K256 blocks. Padding and the K32 swizzle
# distribute vector payload accesses and scale transposes across LDS banks.
_M_TILE_ROWS = 16
_MAX_K_TILE_SIZE = 16384
_K_SLICE_SIZE = 1024
_K256_SIZE = 256
_MX_BLOCK_SIZE = 32
_DWORD_BYTES = 4
_FP4_VALUES_PER_BYTE = 2
_PAYLOAD_VECTOR_BYTES = 16
_PAYLOAD_VECTOR_DWORDS = _PAYLOAD_VECTOR_BYTES // _DWORD_BYTES
_K256_PER_SLICE = _K_SLICE_SIZE // _K256_SIZE
_K32_PER_K256 = _K256_SIZE // _MX_BLOCK_SIZE
_ROWS_PER_SWIZZLE_GROUP = 4
# E8M0 has one byte per MX block; four scales fit in one dword.
_SCALE_DWORDS_PER_SLICE_ROW = _K_SLICE_SIZE // _MX_BLOCK_SIZE // _DWORD_BYTES
_SCALE_DWORDS_PER_K256 = _K256_SIZE // _MX_BLOCK_SIZE // _DWORD_BYTES
_PAYLOAD_K256_ROW_DWORDS = _K256_SIZE // _FP4_VALUES_PER_BYTE // _DWORD_BYTES
_PAYLOAD_LDS_ROW_DWORDS = _PAYLOAD_K256_ROW_DWORDS + 1
_PAYLOAD_LDS_K256_DWORDS = _M_TILE_ROWS * _PAYLOAD_LDS_ROW_DWORDS
_PAYLOAD_LDS_SLICE_DWORDS = _K256_PER_SLICE * _PAYLOAD_LDS_K256_DWORDS
_SCALE_LDS_ROW_DWORDS = _SCALE_DWORDS_PER_SLICE_ROW + 1
_SCALE_LDS_SLICE_DWORDS = _M_TILE_ROWS * _SCALE_LDS_ROW_DWORDS


def validate_apre_k(feat_dim: int) -> None:
    """Require complete K1024 slices that fit the full-row LDS specialization."""
    if not 0 < feat_dim <= _MAX_K_TILE_SIZE or feat_dim % _K_SLICE_SIZE:
        raise ValueError(
            "A-preshuffle requires K divisible by "
            f"{_K_SLICE_SIZE} with 0 < K <= {_MAX_K_TILE_SIZE}, got K={feat_dim}"
        )


def build_moe_apre_scatter_module(
    feat_dim: int,
    wmma_rep: int,
    src_scale_stride_bytes: int = 0,
) -> Callable:
    """Scatter compact MXFP4 rows into M16 A-preshuffled payload and scales.

    Each 16-wave workgroup gathers one M16 x K tile through padded LDS.
    All K1024 slices are loaded asynchronously before one data-ready barrier.
    Ordinary LDS reads and buffer stores write the A-preshuffled output.
    LDS is never reused; there is no persistent loop or cross-tile prefetch.
    Each wave ballots the same M16 row map, so early quit needs no TG barrier.
    Eight waves write scales; all 16 waves transfer payloads.

    Args:
        feat_dim: Logical feature count K, a multiple of 1024 up to 16384.
        wmma_rep: Number of M16 tiles grouped in the output scale layout.
        src_scale_stride_bytes: Input scale row stride, including transport
            padding. Zero selects the compact stride K / 32.

    Returns:
        A JIT launcher accepting compact and output payload/scale buffers,
        a row-to-token map, token/row/workgroup counts, and an optional stream.
        Negative map entries are padding; fully empty M16 tiles are untouched.

    Raises:
        ValueError: K is outside the supported range, not divisible by 1024,
            or the input scale stride is invalid.
        NotImplementedError: The target lacks gfx1250 pk8 or wave32 support.
    """
    validate_apre_k(feat_dim)
    quant_layout = route_quant._quant_layout(feat_dim, "fp4", wmma_rep)
    if not quant_layout.use_pk8 or quant_layout.lanes_per_mx_block != 4:
        raise NotImplementedError("A-preshuffle scatter requires gfx1250 pk8")

    wave_size = quant_layout.wave_size
    if wave_size != 32:
        raise NotImplementedError("A-preshuffle scatter requires wave32")
    threads_per_block = _M_TILE_ROWS * wave_size
    num_k_slices = feat_dim // _K_SLICE_SIZE
    payload_bytes_per_row = quant_layout.payload_bytes_per_row
    payload_dwords_per_row = payload_bytes_per_row // _DWORD_BYTES
    scale_dwords_per_row = quant_layout.mx_blocks_per_row // _DWORD_BYTES
    src_scale_stride_bytes = src_scale_stride_bytes or quant_layout.mx_blocks_per_row
    if (
        src_scale_stride_bytes < quant_layout.mx_blocks_per_row
        or src_scale_stride_bytes % _DWORD_BYTES
    ):
        raise ValueError("input scale stride must cover the row and be dword aligned")
    source_scale_stride_dwords = src_scale_stride_bytes // _DWORD_BYTES
    scale_load_rounds = (scale_dwords_per_row + wave_size - 1) // wave_size
    payload_slice_row_dwords = payload_dwords_per_row // num_k_slices
    output_k256_dwords = _M_TILE_ROWS * _PAYLOAD_K256_ROW_DWORDS
    output_k32_dwords = _M_TILE_ROWS * _PAYLOAD_VECTOR_DWORDS

    @fx.struct
    class ScatterLds:
        payload: fx.Array[
            fx.Int32, num_k_slices * _PAYLOAD_LDS_SLICE_DWORDS, _PAYLOAD_VECTOR_BYTES
        ]
        scale: fx.Array[
            fx.Int32, num_k_slices * _SCALE_LDS_SLICE_DWORDS, _PAYLOAD_VECTOR_BYTES
        ]

    module_name = (
        f"moe_apre_scatter_fd{feat_dim}_r{wmma_rep}_fp4_pk8"
        f"_w{_M_TILE_ROWS}_tk{feat_dim}"
    )
    if src_scale_stride_bytes != quant_layout.mx_blocks_per_row:
        module_name += f"_ss{src_scale_stride_bytes}"
    module_name += "_grid_sw8_async_ballot"

    @flyc.kernel(name=module_name, known_block_size=[threads_per_block, 1, 1])
    def scatter_kernel(
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        num_tokens: Int32,
        num_output_rows: Int32,
        num_workgroups: Int32,
    ):
        lds_load_b128, _ = gemm_common.make_lds_copy_ops(128)

        thread_id = fx.Uint32(fx.thread_idx.x)
        block_id = fx.Uint32(fx.block_idx.x)
        wave_id = thread_id // fx.Uint32(wave_size)
        lane_id = thread_id - wave_id * fx.Uint32(wave_size)

        shared_allocator = fx.SharedAllocator()
        storage = shared_allocator.allocate(ScatterLds).peek()
        payload_lds = storage.payload.ptr
        payload_lds_base_bytes = fx.Index(fx.ptrtoint(payload_lds))
        scale_lds = storage.scale.ptr
        output_payload_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_payload)),
            num_records_bytes=route_quant._SCALE_RSRC_MAX_BYTES,
        )
        output_scale_size_dwords = fx.Uint32(num_output_rows) * fx.Uint32(
            scale_dwords_per_row
        )
        output_scale_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_scale)),
            num_records_bytes=output_scale_size_dwords * fx.Uint32(_DWORD_BYTES),
        )
        row_to_token_buffer = tensor_shim.ptr_buf_tensor(row_to_token)

        row_in_m16 = wave_id

        def load_payload_from_lds(slice_id):
            """Read a 16-byte payload vector in destination order."""
            output_slice_offset_dwords = thread_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            output_k256_in_slice = output_slice_offset_dwords // fx.Uint32(
                output_k256_dwords
            )
            output_k256_offset_dwords = (
                output_slice_offset_dwords
                - output_k256_in_slice * fx.Uint32(output_k256_dwords)
            )
            output_k32_in_k256 = output_k256_offset_dwords // fx.Uint32(
                output_k32_dwords
            )
            output_k32_offset_dwords = (
                output_k256_offset_dwords
                - output_k32_in_k256 * fx.Uint32(output_k32_dwords)
            )
            output_row_in_m16 = output_k32_offset_dwords // fx.Uint32(
                _PAYLOAD_VECTOR_DWORDS
            )
            output_swizzled_k32 = (
                output_k32_in_k256
                + output_row_in_m16 // fx.Uint32(_ROWS_PER_SWIZZLE_GROUP)
            ) % fx.Uint32(_K32_PER_K256)
            read_payload_lds_offset_dwords = (
                output_k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + output_row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + output_swizzled_k32 * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            output_payload_vec = lds_load_b128(
                payload_lds_base_bytes
                + fx.Index(slice_id * _PAYLOAD_LDS_SLICE_DWORDS * _DWORD_BYTES),
                fx.Index(read_payload_lds_offset_dwords * fx.Uint32(_DWORD_BYTES)),
            )

            return output_payload_vec

        def load_scale_from_lds(slice_id):
            """Read a scale dword in destination order."""
            output_scale_dword = fx.Int32(0)
            # Eight waves each write 64 B for a K1024 scale slice.
            output_scale_column = row_in_m16
            output_scale_row = lane_id
            is_scale_store_lane = (
                row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
            ) & (lane_id < fx.Uint32(_M_TILE_ROWS))
            read_scale_lds_offset_dwords = is_scale_store_lane.select(
                output_scale_row * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                + output_scale_column,
                fx.Uint32(0),
            )
            if row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW):
                output_scale_dword = scale_lds[
                    fx.Uint32(slice_id * _SCALE_LDS_SLICE_DWORDS)
                    + read_scale_lds_offset_dwords
                ]

            return output_scale_dword

        def store_scale_to_gmem(k256_offset, output_scale_dword, m16_tile_id):
            """Store scales contiguously across the M16 output rows."""
            output_scale_column = row_in_m16
            output_scale_row = lane_id
            is_scale_store_lane = (
                row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
            ) & (lane_id < fx.Uint32(_M_TILE_ROWS))
            scale_k256_in_slice = output_scale_column // fx.Uint32(
                _SCALE_DWORDS_PER_K256
            )
            scale_k128_in_k256 = output_scale_column - scale_k256_in_slice * fx.Uint32(
                _SCALE_DWORDS_PER_K256
            )
            output_scale_k256 = k256_offset + scale_k256_in_slice
            scale_row_group = m16_tile_id // fx.Uint32(wmma_rep)
            m16_in_scale_row_group = m16_tile_id - scale_row_group * fx.Uint32(wmma_rep)
            scale_group_offset_dwords = scale_row_group * fx.Uint32(
                scale_dwords_per_row * wmma_rep * _M_TILE_ROWS
            ) + m16_in_scale_row_group * fx.Uint32(_M_TILE_ROWS)
            output_scale_offset_dwords = is_scale_store_lane.select(
                scale_group_offset_dwords
                + output_scale_k256
                * fx.Uint32(_SCALE_DWORDS_PER_K256 * wmma_rep * _M_TILE_ROWS)
                + scale_k128_in_k256 * fx.Uint32(wmma_rep * _M_TILE_ROWS)
                + output_scale_row,
                output_scale_size_dwords,
            )
            if row_in_m16 < fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW):
                buffer_ops.buffer_store(
                    output_scale_dword,
                    output_scale_resource,
                    output_scale_offset_dwords * fx.Uint32(_DWORD_BYTES),
                    offset_is_bytes=True,
                )

        def store_payload_to_gmem(k256_offset, output_payload_vec, m16_tile_id):
            """Store a 16-byte vector at its contiguous A-preshuffled address."""
            buffer_ops.buffer_store(
                output_payload_vec,
                output_payload_resource,
                m16_tile_id * fx.Uint32(_M_TILE_ROWS * payload_dwords_per_row)
                + k256_offset * fx.Uint32(output_k256_dwords)
                + thread_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS),
            )

        def copy_payload_from_gmem_to_lds(slice_id, source_row):
            """Direct asynchronous b128 copy; no payload VGPR staging."""
            k256_in_slice = lane_id // fx.Uint32(_K32_PER_K256)
            k32_in_k256 = lane_id % fx.Uint32(_K32_PER_K256)
            swizzled_k32 = (
                k32_in_k256 + row_in_m16 // fx.Uint32(_ROWS_PER_SWIZZLE_GROUP)
            ) % fx.Uint32(_K32_PER_K256)
            dst_dword = (
                fx.Uint32(slice_id * _PAYLOAD_LDS_SLICE_DWORDS)
                + k256_in_slice * fx.Uint32(_PAYLOAD_LDS_K256_DWORDS)
                + row_in_m16 * fx.Uint32(_PAYLOAD_LDS_ROW_DWORDS)
                + swizzled_k32 * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            src_dword = (
                source_row * fx.Uint32(payload_dwords_per_row)
                + fx.Uint32(slice_id * payload_slice_row_dwords)
                + lane_id * fx.Uint32(_PAYLOAD_VECTOR_DWORDS)
            )
            gptr = create_llvm_ptr(
                fx.Int64(ptrtoint(compact_payload))
                + fx.Int64(src_dword) * _DWORD_BYTES,
                address_space=1,
            )
            lptr = create_llvm_ptr(
                fx.Int32(payload_lds_base_bytes) + fx.Int32(dst_dword) * _DWORD_BYTES,
                address_space=3,
            )
            rocdl.global_load_async_to_lds_b128(gptr, lptr, 0)

        def copy_scale_from_gmem_to_lds(source_row, dst_scale_lds):
            """Copy the row scales in wave-sized rounds, masking the final tail."""
            for scale_round in range_constexpr(scale_load_rounds):
                column = fx.Uint32(scale_round * wave_size) + lane_id
                if column < fx.Uint32(scale_dwords_per_row):
                    dst_dword = (
                        (column // fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW))
                        * fx.Uint32(_SCALE_LDS_SLICE_DWORDS)
                        + row_in_m16 * fx.Uint32(_SCALE_LDS_ROW_DWORDS)
                        + column % fx.Uint32(_SCALE_DWORDS_PER_SLICE_ROW)
                    )
                    src_dword = (
                        source_row * fx.Uint32(source_scale_stride_dwords) + column
                    )
                    gptr = create_llvm_ptr(
                        fx.Int64(ptrtoint(compact_scale))
                        + fx.Int64(src_dword) * _DWORD_BYTES,
                        address_space=1,
                    )
                    lptr = create_llvm_ptr(
                        fx.Int32(ptrtoint(dst_scale_lds))
                        + fx.Int32(dst_dword) * _DWORD_BYTES,
                        address_space=3,
                    )
                    rocdl.global_load_async_to_lds_b32(gptr, lptr, 0)

        # Each TG owns one M16 x K tile. LDS is written once and never reused.
        m16_tile_id = block_id
        # Every wave reads the same M16 map. The second lane half repeats the
        # first half's addresses, so all 32 lanes can load without an OOB mask.
        map_row = lane_id % fx.Uint32(_M_TILE_ROWS)
        token_on_lane = fx.Int32(
            row_to_token_buffer[m16_tile_id * fx.Uint32(_M_TILE_ROWS) + map_row]
        )
        valid_rows = fx.Uint32(rocdl.ballot(T.i32, token_on_lane >= fx.Int32(0)))
        tile_has_valid_rows = valid_rows != fx.Uint32(0)
        # Wave w owns row w; broadcast that lane's mapping within the wave.
        mapped_token = fx.Int32(rocdl.readlane(T.i32, token_on_lane, wave_id))

        # Every wave observes the same flag before entering the payload barrier.
        if tile_has_valid_rows:
            # Padding rows are not consumed by the MoE output. Redirect them
            # to token 0 so every wave can issue valid, unconditional loads.
            # A nonempty M16 tile implies that at least one source token exists.
            source_row = fx.Uint32(fx.max(mapped_token, fx.Int32(0)))

            copy_scale_from_gmem_to_lds(source_row, scale_lds)
            for slice_id in range_constexpr(num_k_slices):
                copy_payload_from_gmem_to_lds(slice_id, source_row)
            rocdl.s_wait_asynccnt(0)
            gpu.barrier()

            for slice_id in range_constexpr(num_k_slices):
                k256_offset = fx.Uint32(slice_id * _K256_PER_SLICE)
                output_scale_dword = load_scale_from_lds(slice_id)
                output_payload_vec = load_payload_from_lds(slice_id)
                store_scale_to_gmem(k256_offset, output_scale_dword, m16_tile_id)
                store_payload_to_gmem(k256_offset, output_payload_vec, m16_tile_id)

    @flyc.jit
    def launch_scatter(
        compact_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        num_tokens: fx.Int32,
        num_output_rows: fx.Int32,
        num_workgroups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        scatter_kernel(
            compact_payload,
            compact_scale,
            grouped_payload,
            grouped_scale,
            row_to_token,
            num_tokens,
            num_output_rows,
            num_workgroups,
        ).launch(
            grid=(
                arith.index_cast(T.index, num_output_rows) // _M_TILE_ROWS,
                1,
                1,
            ),
            block=(threads_per_block, 1, 1),
            stream=stream,
        )

    launch_scatter.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": (
                tensor_shim.AITER_FLYDSL_KERNARG_PRELOAD_COUNT
            ),
        },
    }
    return launch_scatter
