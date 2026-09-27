# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Concrete dispatch + GEMM1 emitter for the gfx1250 stage-1 mega-kernel."""

from dataclasses import dataclass

import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.gemm1_consumer_gfx1250 import (
    GEMM_TILE_CORE_VERSION,
    emit_gemm_a8w4_tile,
    gemm_a8w4_tile_config,
)

from .compact_plan import COMPACT_TILE_GEN_DW, compact_tile_layout
from .dispatch_tdm import CompactPayloadSpec, emit_compact_payload_rows
from .stage1_mega_kernel import EMITTER_ABI_VERSION, publish_tile_arrivals

# Opaque mega-kernel argument ABI. Keep this mapping next to the only concrete
# emitter so the host launcher and both sides of the kernel cannot drift.
ARG_ARENA = 0
ARG_SOURCE_PAYLOAD = 1
ARG_SOURCE_SCALE = 2
ARG_TOKEN_MAP = 3
ARG_TOPK_WEIGHTS = 4
ARG_OUTPUT_PAYLOAD = 5
ARG_WEIGHT = 6
ARG_WEIGHT_SCALE = 7
ARG_PSUM = 8
ARG_OUTPUT_SCALE = 9
ARG_BIAS = 10
ARG_INPUT_PAYLOAD = 11
ARG_INPUT_SCALE = 12
ARG_TOKEN_COUNT = 13
ARG_CONTIGUOUS_M = 14


def _global_i8_ptr(address):
    ptr_ty = fx.PointerType.get(
        elem_ty=T.i8, address_space=fx.AddressSpace.Global, alignment=16
    )
    return fx.inttoptr(ptr_ty, fx.Int64(address))


@comm_ops.traced
def _wait_remote_generations(
    *, tid, world_size, window, tile_state_offset, generation
):
    if tid == fx.Int32(0):
        for peer in range(world_size):
            remote = fx.Int64(window.lsa_ptr(peer, tile_state_offset))
            comm_ops.wait_i32_until_equals(
                remote + fx.Int64(COMPACT_TILE_GEN_DW * 4), generation
            )
            comm_ops.fence_system_acquire()
    fx.barrier()


@comm_ops.traced
def _zero_i32_lds(*, tid, n, stride, base):
    for i in range(tid, n, stride):
        comm_ops.store_i32_lds(base + fx.Int64(i) * fx.Int64(4), fx.Int32(0))


@comm_ops.traced
def _accumulate_tile_hit(
    *, lane, route, dest_tok, dest_pe, tile_m, tile_count, hist_base
):
    real_tile = dest_tok // fx.Int32(tile_m)
    if (
        (lane == fx.Int32(route))
        & (real_tile >= fx.Int32(0))
        & (real_tile < fx.Int32(tile_count))
    ):
        slot = dest_pe * fx.Int32(tile_count) + real_tile
        comm_ops.atomic_add_lds(hist_base + fx.Int64(slot) * fx.Int64(4), fx.Int32(1))


@comm_ops.traced
def _flush_tile_hist(
    *,
    tid,
    block,
    world_size,
    tile_count,
    hist_base,
    window,
    tile_state_offset,
    ready_off,
    epoch_off,
    generation,
):
    fx.barrier()
    slots = world_size * tile_count
    for i in range(tid, slots, block):
        count = fx.Int32(comm_ops.load_i32_lds(hist_base + fx.Int64(i) * fx.Int64(4)))
        if count > fx.Int32(0):
            dest_pe = i // fx.Int32(tile_count)
            tile_id = i - dest_pe * fx.Int32(tile_count)
            remote = fx.Int64(window.lsa_ptr(dest_pe, tile_state_offset))
            publish_tile_arrivals(
                tile_id=tile_id,
                count=count,
                generation=generation,
                tile_count=tile_count,
                addr_tile_ready=remote + fx.Int64(ready_off),
                addr_ready_epoch=remote + fx.Int64(epoch_off),
            )


@dataclass(frozen=True, slots=True)
class Gfx1250Stage1MegaEmitter:
    """Inline compact dispatch producers and one scheduled GEMM tile."""

    ABI_VERSION = EMITTER_ABI_VERSION
    GEMM_CORE_VERSION = GEMM_TILE_CORE_VERSION

    rank: int
    world_size: int
    topk: int
    max_tokens_per_rank: int
    output_offset: int
    rowmap_offset: int
    tile_state_offset: int
    compact_cap: int
    wire_row_stride: int
    dispatch_payload_elems: int
    dispatch_elem_size: int
    dispatch_scale_bytes: int
    K: int
    N: int
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    out_is_f16: int
    num_buffers: int
    a_is_fp4: int
    n_experts: int
    stage1_act: int
    quant_wmma_rep: int
    next_stage_prefetch: int
    num_waves_per_tensor_tdm: int
    tdm_as_in_prologue: int
    tdm_b_th: int
    swiglu_limit: float
    situ_beta: float
    situ_linear_beta: float

    @property
    def WAVES_PER_CTA(self) -> int:
        return int(self.m_warp) * int(self.n_warp)

    @property
    def _gemm_arena_bytes(self) -> int:
        a_pack = 2 if self.a_is_fp4 else 1
        stage_a = self.tile_m * (self.tile_k // a_pack + 16)
        stage_b = (self.tile_n // 16) * (self.tile_k // 2) * 16
        sa_kdw = self.tile_k // 128
        sa_pad = (
            1 if sa_kdw > 0 and (sa_kdw & (sa_kdw - 1)) == 0 and sa_kdw % 2 == 0 else 0
        )
        stage_sa = self.tile_m * (sa_kdw + sa_pad) * 4
        stage_sb = (self.tile_n // 32) * (self.tile_k // 4) * 4
        pitch = ((stage_a + stage_b + stage_sa + stage_sb + 511) // 512) * 512

        as_full_kdw = self.K // 128
        as_full_b = ((self.tile_m * as_full_kdw * 4 + 127) // 128) * 128
        gemm_arena = self.num_buffers * pitch
        if self.tdm_as_in_prologue:
            gemm_arena += as_full_b
        c_store = ((self.tile_m * (self.tile_n + 16) * 2 + 127) // 128) * 128
        return max(gemm_arena, c_store)

    @property
    def LDS_BYTES(self) -> int:
        """Shared arena: dispatch finishes before GEMM reuses the same LDS."""

        producer_arena = self.WAVES_PER_CTA * self.wire_row_stride
        hist_tiles = (self.compact_cap + self.tile_m - 1) // self.tile_m
        producer_arena += self.world_size * hist_tiles * 4
        producer_arena = ((producer_arena + 127) // 128) * 128
        return max(self._gemm_arena_bytes, producer_arena)

    def __post_init__(self):
        if self.world_size != 4:
            raise ValueError("stage1 mega emitter supports EP4 only")
        if self.dispatch_elem_size not in (1, 2):
            raise ValueError("unsupported compact payload element size")
        if self.wire_row_stride % 128:
            raise ValueError("compact wire rows must be 128-byte aligned")
        if self.K != 7168:
            raise ValueError("stage1 mega emitter supports K=7168 only")
        if self.N <= 0 or self.tile_n <= 0:
            raise ValueError("invalid GEMM N geometry")
        if self.WAVES_PER_CTA not in (4, 8):
            raise ValueError("GEMM wave grid must contain 4 or 8 waves")

    def emit_producer(
        self,
        *,
        config,
        tid,
        wave_id,
        generation,
        producer_id,
        producer_ctas,
        publish_tile_arrival,
        lds_base_ptr,
        user_args,
        **_,
    ) -> None:
        spec = CompactPayloadSpec(
            rank=self.rank,
            npes=self.world_size,
            topk=self.topk,
            hidden_dim=self.dispatch_payload_elems,
            hidden_elem_size=self.dispatch_elem_size,
            compact_row_stride=self.wire_row_stride,
            tile_bytes=self.wire_row_stride,
            max_tok_slot_stride=self.max_tokens_per_rank * self.topk,
            off_out_tok=self.output_offset,
            off_ep_rowmap=self.rowmap_offset,
            scale_bytes=self.dispatch_scale_bytes,
        )
        producer_lds = self.WAVES_PER_CTA * spec.tile_bytes
        hist_off = producer_lds
        hist_slots = self.world_size * int(config.tile_count)
        if hist_off + hist_slots * 4 > self.LDS_BYTES:
            raise ValueError("dispatch producer LDS exceeds shared mega arena")

        lane = fx.thread_idx.x & fx.Int32(31)
        lds_base_i32 = arith.index_cast(
            T.i32, fx.index_cast(T.index, fx.ptrtoint(lds_base_ptr))
        )
        tile_addr = lds_base_i32 + fx.Int32(wave_id) * fx.Int32(spec.tile_bytes)
        hist_base = fx.Int64(lds_base_i32) + fx.Int64(hist_off)
        block = fx.Int32(self.WAVES_PER_CTA * 32)
        window = cco.Window(user_args[ARG_ARENA])
        rsrc_tok_map = create_buffer_resource_from_addr(user_args[ARG_TOKEN_MAP])
        rsrc_weights = create_buffer_resource_from_addr(user_args[ARG_TOPK_WEIGHTS])
        rsrc_scale = create_buffer_resource_from_addr(user_args[ARG_SOURCE_SCALE])
        layout = compact_tile_layout(compact_cap=self.compact_cap)
        ready_off = layout.ready_dw * 4
        epoch_off = layout.queue_dw * 4

        _zero_i32_lds(tid=tid, n=hist_slots, stride=block, base=hist_base)
        _wait_remote_generations(
            tid=tid,
            world_size=self.world_size,
            window=window,
            tile_state_offset=self.tile_state_offset,
            generation=generation,
        )

        def on_route_complete(*, dest_pe, dest_tok, route, lane, **_ignored):
            _accumulate_tile_hit(
                lane=lane,
                route=route,
                dest_tok=dest_tok,
                dest_pe=dest_pe,
                tile_m=self.tile_m,
                tile_count=config.tile_count,
                hist_base=hist_base,
            )

        emit_compact_payload_rows(
            spec,
            work_id=producer_id * fx.Int32(self.WAVES_PER_CTA) + wave_id,
            work_stride=producer_ctas * fx.Int32(self.WAVES_PER_CTA),
            token_limit=fx.Int32(user_args[ARG_TOKEN_COUNT]),
            lane=lane,
            tile_addr=tile_addr,
            window=window,
            addr_inp_tok=user_args[ARG_SOURCE_PAYLOAD],
            rsrc_tok_map=rsrc_tok_map,
            rsrc_inp_wts=rsrc_weights,
            rsrc_inp_scale=rsrc_scale,
            on_route_complete=on_route_complete,
        )
        _flush_tile_hist(
            tid=tid,
            block=block,
            world_size=self.world_size,
            tile_count=config.tile_count,
            hist_base=hist_base,
            window=window,
            tile_state_offset=self.tile_state_offset,
            ready_off=ready_off,
            epoch_off=epoch_off,
            generation=generation,
        )
        _ = publish_tile_arrival

    def emit_consumer(
        self,
        *,
        work_id,
        num_work_tiles,
        lds_base_ptr,
        user_args,
        **_,
    ) -> None:
        tile_cfg = gemm_a8w4_tile_config(
            K=self.K,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            m_warp=self.m_warp,
            n_warp=self.n_warp,
            out_is_f16=self.out_is_f16,
            num_buffers=self.num_buffers,
            a_is_fp4=self.a_is_fp4,
            n_experts=self.n_experts,
            stage1_act=self.stage1_act,
            has_bias=0,
            stage1_quant_out=1,
            quant_wmma_rep=self.quant_wmma_rep,
            cluster_n=1,
            next_stage_prefetch=self.next_stage_prefetch,
            num_waves_per_tensor_tdm=self.num_waves_per_tensor_tdm,
            tdm_as_in_prologue=self.tdm_as_in_prologue,
            tdm_b_th=self.tdm_b_th,
            enable_ep_scatter=0,
            ep_arena_handle=0,
            ep_combine_input_offset=0,
            ep_slot_stride_bytes=0,
            ep_destination_stride=0,
            ep_world_size=0,
            ep_quant_bits=0,
            row_major_ascale=1,
            a_row_stride_bytes=self.wire_row_stride,
            a_scale_row_stride_bytes=self.wire_row_stride,
        )
        if tile_cfg.block != self.WAVES_PER_CTA * 32:
            raise ValueError("GEMM tile block size does not match mega workgroup")
        if tile_cfg.consts["ARENA_B"] > self.LDS_BYTES:
            raise ValueError(
                f"GEMM tile LDS {tile_cfg.consts['ARENA_B']} exceeds "
                f"shared mega arena {self.LDS_BYTES}"
            )

        arg_c = _global_i8_ptr(user_args[ARG_OUTPUT_PAYLOAD])
        emit_gemm_a8w4_tile(
            arg_c=arg_c,
            arg_a=_global_i8_ptr(user_args[ARG_INPUT_PAYLOAD]),
            arg_b=_global_i8_ptr(user_args[ARG_WEIGHT]),
            arg_scale_a=_global_i8_ptr(user_args[ARG_INPUT_SCALE]),
            arg_scale_b=_global_i8_ptr(user_args[ARG_WEIGHT_SCALE]),
            arg_m_tile_map=_global_i8_ptr(user_args[ARG_PSUM]),
            arg_bias=_global_i8_ptr(user_args[ARG_BIAS]),
            arg_quant_scale=_global_i8_ptr(user_args[ARG_OUTPUT_SCALE]),
            arg_ep_row_map=arg_c,
            i32_m=num_work_tiles * fx.Int32(self.tile_m),
            i32_n=fx.Int32(self.N),
            f32_swiglu_limit=fx.Float32(self.swiglu_limit),
            f32_situ_beta=fx.Float32(self.situ_beta),
            f32_situ_linear_beta=fx.Float32(self.situ_linear_beta),
            bid_x=work_id,
            lds_base_ptr=lds_base_ptr,
            **tile_cfg.consts,
        )


__all__ = [
    "ARG_ARENA",
    "ARG_BIAS",
    "ARG_CONTIGUOUS_M",
    "ARG_INPUT_PAYLOAD",
    "ARG_INPUT_SCALE",
    "ARG_OUTPUT_PAYLOAD",
    "ARG_OUTPUT_SCALE",
    "ARG_PSUM",
    "ARG_SOURCE_PAYLOAD",
    "ARG_SOURCE_SCALE",
    "ARG_TOKEN_COUNT",
    "ARG_TOKEN_MAP",
    "ARG_TOPK_WEIGHTS",
    "ARG_WEIGHT",
    "ARG_WEIGHT_SCALE",
    "Gfx1250Stage1MegaEmitter",
]
