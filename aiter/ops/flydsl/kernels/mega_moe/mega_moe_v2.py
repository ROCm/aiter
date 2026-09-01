# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

import ctypes
import os
from dataclasses import replace

import flydsl.expr as fx
import mori.shmem as ms
import torch

from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from .mega_moe_config import (
    FIXED_SLOT_MAX_MTPR,
    MegaMoEConfig,
    Stage1Config,
    build_mega_moe_bundle_plan,
)
from .quant import per_1x32_mx_quant

__all__ = ["MegaMoEV2"]


def _create_masked_stream(device: torch.device, cu_num: int, cus_per_word: int):
    """Create an experimental HIP stream using the high CUs of each 32-CU group."""
    if not 0 < cus_per_word <= 32:
        raise ValueError("cus_per_word must be in [1, 32]")
    word_count = (cu_num + 31) // 32
    word = ((1 << cus_per_word) - 1) << (32 - cus_per_word)
    words = (ctypes.c_uint32 * word_count)(*([word] * word_count))
    hip = ctypes.CDLL("libamdhip64.so")
    create = hip.hipExtStreamCreateWithCUMask
    create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    create.restype = ctypes.c_int
    raw_stream = ctypes.c_void_p()
    result = create(ctypes.byref(raw_stream), word_count, words)
    if result != 0 or not raw_stream.value:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed with {result}")
    stream = torch.cuda.ExternalStream(raw_stream.value, device=device)
    return stream, hip, raw_stream


class MegaMoEV2:
    """Fused dispatch, GEMM1, GEMM2, and combine with one in-flight launch per instance."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0,
        fanout_masks: tuple[int, ...] = ()):
    # fmt: on
        if quant != "a8w4":
            raise ValueError("MegaMoEV2 currently supports quant='a8w4' only")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        self._bundle_plan = build_mega_moe_bundle_plan(
            self.mtpr,
            experts_per_rank=self.epr,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
        )
        self._active_bundle_entry = None
        debug_fanout = os.environ.get("MEGA_DEBUG_FANOUT_MASKS", "")
        if not fanout_masks and debug_fanout:
            fanout_masks = tuple(
                int(item, 0) for item in debug_fanout.split(",") if item
            )
        self._s1_fanout_masks = tuple(int(mask) for mask in fanout_masks)
        if self._s1_fanout_masks and len(self._s1_fanout_masks) != self.world_size:
            raise ValueError("fanout_masks must contain one mask per destination")
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")
        self.dev = torch.device("cuda", rank)
        self.max_recv = self.world_size * self.mtpr
        compact = self.mtpr > FIXED_SLOT_MAX_MTPR
        # Compact Stage1 always uses the deterministic runtime fanout protocol.
        # Keeping this wire format identical for every MAX-MTPR bucket is what
        # makes uneven-rank dynamic prefill safe.
        self._s1_runtime_fanout = compact
        capacity_tile_m = 128 if compact else 32
        self._s1_fixed_slot = not compact
        self._s1_scale_dim = self.model_dim // 32
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=capacity_tile_m,
            gm_scheme=mega_scheme, gm_compact=compact, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self._build_fused_stage1(w1, w1_scale)
        self._build_fused_stage2()
        if os.environ.get("AITER_MEGA_MOE_PRELOAD", "0") == "1":
            self.preload_aot_bundles()

    def preload_aot_bundles(self):
        """Load the paired Stage1 and Stage2 production bundles."""
        self.preload_stage1_bundle()
        self.preload_stage2_bundle()

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_prepare import (
            preload_mega_moe_prepare,
            run_mega_moe_prepare,
        )
        from .mega_moe_stage1 import (
            preload_mega_moe_stage1_bundle,
            run_mega_moe_stage1,
            run_mega_moe_stage1_bundle,
        )

        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_tile_input_base = torch.zeros(
            metadata_blocks, dtype=torch.int32, device=self.dev
        )
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_tile_state_stride = metadata_blocks
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._s1_prepare_cu_capacity = self._s1_num_cu
        # Quant is shorter than compact prepare.  Limiting its resident CTA
        # population keeps it under prepare's critical path without letting
        # quant saturate all CUs and stretch the histogram/P2P allgather.
        # Keep one artifact/workspace capacity for every token bucket while the
        # config selects the useful quant CTA count.  Large prefill batches need
        # more than 64 CTAs; small batches continue to launch only their useful
        # subset and do not pay for the capacity.
        self._s1_quant_cu_capacity = self._s1_num_cu
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._s1_mega = run_mega_moe_stage1
        self._s1_mega_bundle = run_mega_moe_stage1_bundle
        self._s1_preload_bundle = preload_mega_moe_stage1_bundle
        self._s1_prepare = run_mega_moe_prepare
        self._s1_preload_prepare = preload_mega_moe_prepare

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._s1_quant_x = torch.empty(
            self.mtpr,
            self.model_dim,
            dtype=torch.float8_e4m3fn,
            device=self.dev,
        )
        self._s1_quant_scale = torch.empty(
            self.mtpr,
            self._s1_scale_dim,
            dtype=torch.uint8,
            device=self.dev,
        )
        self._build_v2_disp_table()

    def _allocate_dispatch_workspace(self, op, metadata_blocks):
        total_experts = self.world_size * self.epr
        total_segments = total_experts + self.world_size
        tile_state_blocks = 2 * metadata_blocks
        workspace = {
            "local_hist": torch.zeros(total_segments, dtype=torch.int32, device=self.dev),
            "block_hist": torch.empty(
                self._s1_num_cu * total_segments,
                dtype=torch.int32,
                device=self.dev,
            ),
            "local_cursor": torch.zeros(total_segments, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "route_segment": torch.empty(
                self.mtpr * self.topk, dtype=torch.int32, device=self.dev
            ),
            "pair_base": torch.empty(total_segments, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=self.dev),
            "epoch_gate": torch.zeros(10, dtype=torch.int32, device=self.dev),
            # Fused prepare is keyed by (route workers, quant workers).  Each
            # key has a fixed launch grid, so its generation counter remains
            # valid when dynamic token shapes alternate under CUDA Graphs.
            "prep_entry_count": torch.zeros(
                (self._s1_prepare_cu_capacity + 1)
                * (self._s1_quant_cu_capacity + 1),
                dtype=torch.int64,
                device=self.dev,
            ),
            "prep_epoch_gate": torch.zeros(
                (self._s1_prepare_cu_capacity + 1)
                * (self._s1_quant_cu_capacity + 1),
                dtype=torch.int32,
                device=self.dev,
            ),
            "pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "work_head": torch.zeros(8 * 16, dtype=torch.int32, device=self.dev),
            "work_tail": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "expert_tile_end": torch.empty(self.epr, dtype=torch.int32, device=self.dev),
            "max_expert_tiles": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_chunk_done": torch.zeros(total_segments, dtype=torch.int32, device=self.dev),
            "payload_blocks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "payload_chunks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            # Count/fill phase counters are keyed by the same (prepare, quant)
            # shape as PREP_ENTRY_COUNT.  Different dynamic quant grids may
            # share route-worker count P, but must never share a phase counter.
            "group_done": torch.zeros(
                (self._s1_prepare_cu_capacity + 1)
                * (self._s1_quant_cu_capacity + 1),
                dtype=torch.int32,
                device=self.dev,
            ),
        }
        initial_pairs = []
        for destination in range(self.world_size):
            mask = (
                int(self._s1_fanout_masks[destination])
                if destination < len(self._s1_fanout_masks)
                else 0
            )
            if mask:
                if mask.bit_count() != 2:
                    raise ValueError("runtime fanout currently requires expert pairs")
                pair_a = (mask & -mask).bit_length() - 1
                pair_b = (mask ^ (1 << pair_a)).bit_length() - 1
                initial_pairs.append(pair_a | (pair_b << 8) | (1 << 16))
            else:
                initial_pairs.append(0)
        workspace["fanout_pair_config"] = torch.tensor(
            initial_pairs * 2, dtype=torch.int32, device=self.dev
        )
        # The full source histogram is small (392 int32 values for V4-Pro) and
        # lets every rank derive every destination offset without a base
        # push-back round trip.  The legacy compact matrix uses only the prefix.
        workspace["bigcnt"] = op._sym(
            (self.world_size * total_segments,), torch.int32
        )
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["group_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        workspace["launch_ready"] = op._sym((self.world_size,), torch.int32)
        workspace["tile_ready"] = op._sym((tile_state_blocks,), torch.int32)
        workspace["tile_expected"] = op._sym((tile_state_blocks,), torch.int32)
        workspace["ready_tile_queue"] = op._sym((tile_state_blocks,), torch.int32)
        workspace["ready_tile_epoch"] = op._sym((tile_state_blocks,), torch.int32)
        workspace["ready_tile_tail"] = op._sym((2,), torch.int32)
        workspace["payload_ready_rows"] = op._sym((1,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_group_base"] = op._p2p_table(workspace["group_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        workspace["p2p_launch_ready"] = op._p2p_table(workspace["launch_ready"])
        workspace["p2p_tile_ready"] = op._p2p_table(workspace["tile_ready"])
        workspace["p2p_tile_expected"] = op._p2p_table(workspace["tile_expected"])
        workspace["p2p_ready_tile_queue"] = op._p2p_table(
            workspace["ready_tile_queue"]
        )
        workspace["p2p_ready_tile_epoch"] = op._p2p_table(
            workspace["ready_tile_epoch"]
        )
        workspace["p2p_ready_tile_tail"] = op._p2p_table(
            workspace["ready_tile_tail"]
        )
        workspace["p2p_payload_ready_rows"] = op._p2p_table(workspace["payload_ready_rows"])
        self._s1_dispatch_workspace = workspace

    def _build_v2_disp_table(self):
        op = self._s1_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        table[DispatchSlot.PAIR_BASE] = workspace["pair_base"].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.TILE_INPUT_BASE] = self._s1_tile_input_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.SRCMAP] = op.srcmap_em.data_ptr()
        table[DispatchSlot.LOCAL_HIST] = workspace["local_hist"].data_ptr()
        table[DispatchSlot.BLOCK_HIST] = workspace["block_hist"].data_ptr()
        table[DispatchSlot.COUNT_MATRIX] = workspace["bigcnt"].data_ptr()
        table[DispatchSlot.P2P_COUNT_MATRIX] = workspace["p2p_bigcnt"].data_ptr()
        table[DispatchSlot.COUNT_DONE] = workspace["count_done"].data_ptr()
        table[DispatchSlot.P2P_COUNT_DONE] = workspace["p2p_count_done"].data_ptr()
        table[DispatchSlot.TASK_ROW_BASE] = workspace["my_base"].data_ptr()
        table[DispatchSlot.GROUP_TASK_BASE] = workspace["group_base"].data_ptr()
        table[DispatchSlot.LOCAL_CURSOR] = workspace["local_cursor"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY] = workspace["p2p_payload_ready"].data_ptr()
        table[DispatchSlot.PAIR_ORDER] = workspace["pair_order"].data_ptr()
        table[DispatchSlot.ROUTE_SEGMENT] = workspace["route_segment"].data_ptr()
        table[DispatchSlot.P2P_TASK_ROW_BASE] = workspace["p2p_my_base"].data_ptr()
        table[DispatchSlot.P2P_GROUP_TASK_BASE] = workspace[
            "p2p_group_base"
        ].data_ptr()
        table[DispatchSlot.P2P_PLAN_READY] = workspace["p2p_plan_ready"].data_ptr()
        table[DispatchSlot.PLAN_READY] = workspace["plan_ready"].data_ptr()
        table[DispatchSlot.PAIR_READY] = workspace["pair_ready"].data_ptr()
        table[DispatchSlot.ENTRY_COUNT] = workspace["entry_count"].data_ptr()
        table[DispatchSlot.EPOCH_GATE] = workspace["epoch_gate"].data_ptr()
        table[DispatchSlot.PREP_ENTRY_COUNT] = workspace["prep_entry_count"].data_ptr()
        table[DispatchSlot.PREP_EPOCH_GATE] = workspace["prep_epoch_gate"].data_ptr()
        table[DispatchSlot.PAIR_ORDER_READY] = workspace["pair_order_ready"].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace["expert_tile_end"].data_ptr()
        table[DispatchSlot.GROUP_DONE] = workspace["group_done"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[DispatchSlot.LAUNCH_READY] = workspace["launch_ready"].data_ptr()
        table[DispatchSlot.P2P_LAUNCH_READY] = workspace["p2p_launch_ready"].data_ptr()
        table[DispatchSlot.MAX_EXPERT_TILES] = workspace["max_expert_tiles"].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNK_DONE] = workspace["payload_chunk_done"].data_ptr()
        table[DispatchSlot.TILE_READY] = workspace["tile_ready"].data_ptr()
        table[DispatchSlot.P2P_TILE_READY] = workspace["p2p_tile_ready"].data_ptr()
        table[DispatchSlot.TILE_EXPECTED] = workspace["tile_expected"].data_ptr()
        table[DispatchSlot.P2P_TILE_EXPECTED] = workspace[
            "p2p_tile_expected"
        ].data_ptr()
        table[DispatchSlot.FANOUT_PAIR_CONFIG] = workspace[
            "fanout_pair_config"
        ].data_ptr()
        table[DispatchSlot.READY_TILE_QUEUE] = workspace[
            "ready_tile_queue"
        ].data_ptr()
        table[DispatchSlot.P2P_READY_TILE_QUEUE] = workspace[
            "p2p_ready_tile_queue"
        ].data_ptr()
        table[DispatchSlot.READY_TILE_EPOCH] = workspace[
            "ready_tile_epoch"
        ].data_ptr()
        table[DispatchSlot.P2P_READY_TILE_EPOCH] = workspace[
            "p2p_ready_tile_epoch"
        ].data_ptr()
        table[DispatchSlot.READY_TILE_TAIL] = workspace[
            "ready_tile_tail"
        ].data_ptr()
        table[DispatchSlot.P2P_READY_TILE_TAIL] = workspace[
            "p2p_ready_tile_tail"
        ].data_ptr()
        table[DispatchSlot.PAYLOAD_READY_ROWS] = workspace["payload_ready_rows"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY_ROWS] = workspace["p2p_payload_ready_rows"].data_ptr()
        table[DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION] = workspace[
            "payload_blocks_per_destination"
        ].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION] = workspace[
            "payload_chunks_per_destination"
        ].data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def preload_stage1_bundle(self):
        """Load every production Stage1/prepare variant without GPU dispatch."""
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        for entry in self._bundle_plan.entries:
            config = entry.config.stage1
            bucket = entry.token_bucket
            if self._s1_fixed_slot:
                continue
            prepare_blocks = max(1, min(config.prepare_cu, (bucket + 63) // 64))
            quant_groups = bucket * self._s1_scale_dim
            quant_blocks = min(
                self._s1_quant_cu_capacity,
                config.prepare_quant_cu,
                (quant_groups + 511) // 512,
            )
            # Production executes the quant+prepare artifact, while standalone
            # Stage1 attribution reuses the same prepare kernel with quant
            # disabled.  Preload both finite identities so run-only mode also
            # covers diagnostics without compiling after warmup.
            for preload_quant_blocks in sorted({0, quant_blocks}):
                self._s1_preload_prepare(
                    fx.Int64(self._s1_disp.data_ptr()),
                    fx.Int32(bucket),
                    fx.Int64(0),
                    fx.Int64(self._s1_epoch_parity.data_ptr()),
                    fx.Int64(self._s1_epoch_expected.data_ptr()),
                    fx.Int64(0),
                    fx.Int64(0),
                    fx.Int64(0),
                    stream,
                    rank=self.rank,
                    experts_per_rank=self.epr,
                    fuse_npes=self.world_size,
                    fuse_topk=self.topk,
                    fuse_mtpr=self.mtpr,
                    sort_block_m=config.sort_block_m,
                    num_dispatch_cu=config.num_dispatch_cu,
                    num_prepare_cu=prepare_blocks,
                    num_quant_cu=preload_quant_blocks,
                    quant_cu_capacity=self._s1_quant_cu_capacity,
                    model_dim=self.model_dim,
                    payload_chunk_rows=config.payload_chunk_rows,
                    payload_tile_ready=config.payload_tile_ready,
                    tile_state_stride=self._s1_tile_state_stride,
                    fanout_masks=(),
                    runtime_fanout=self._s1_runtime_fanout,
                    dynamic_fanout=self._s1_runtime_fanout,
                )

        op = self._s1_op
        self._s1_preload_bundle(
            self._s1_out,
            self._s1_rx,
            self._s1_w1,
            self._s1_scale_i32,
            self._s1_w1_scale,
            op.tile_row_base,
            op.sorted_expert_ids,
            op.num_valid,
            self._s1_osd,
            fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()),
            fx.Int32(1),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            fx.Int64(self._s1_epoch_expected.data_ptr()),
            0,
            stream,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            rank=self.rank,
            experts_per_rank=self.epr,
            fuse_npes=self.world_size,
            fuse_topk=self.topk,
            fuse_cap=self._s1_cap,
            fuse_mtpr=self.mtpr,
            fuse_scale_dim=self._s1_scale_dim,
            fixed_slot_dispatch=self._s1_fixed_slot,
            num_cu=self._s1_num_cu,
            tile_state_stride=self._s1_tile_state_stride,
            variants=self._bundle_plan.stage1_variants,
            swiglu_limit=self.swiglu_limit,
        )

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        entry = self._bundle_plan.entry_for_tokens(tokens)
        config = entry.config
        self._active_bundle_entry = entry
        self._active_config = config
        return config

    def _run_compact_prepare(
        self, topk_ids, cur_tok, config, stream, *, quant_input=None
    ):
        if self._s1_fixed_slot:
            return
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        compile_fanout_masks = (
            () if self._s1_runtime_fanout else self._s1_fanout_masks
        )
        entry = self._active_bundle_entry
        compile_tokens = (
            entry.token_bucket
            if entry is not None and entry.config.stage1 == config
            else cur_tok
        )
        if compile_fanout_masks or self._s1_runtime_fanout:
            prepare_blocks = (compile_tokens + 63) // 64
        else:
            prepare_blocks = (compile_tokens * self.topk + 511) // 512
        prepare_blocks = max(1, min(config.prepare_cu, prepare_blocks))
        quant_blocks = 0
        quant_input_ptr = 0
        quant_output_ptr = 0
        quant_scale_ptr = 0
        if quant_input is not None:
            if quant_input.dtype != torch.bfloat16 or not quant_input.is_contiguous():
                raise ValueError("quant_input must be contiguous bfloat16")
            if tuple(quant_input.shape) != (cur_tok, self.model_dim):
                raise ValueError(
                    f"quant_input must have shape ({cur_tok}, {self.model_dim})"
                )
            quant_groups = compile_tokens * self._s1_scale_dim
            quant_blocks = min(
                self._s1_quant_cu_capacity,
                config.prepare_quant_cu,
                (quant_groups + 511) // 512,
            )
            quant_input_ptr = quant_input.data_ptr()
            quant_output_ptr = self._s1_quant_x.data_ptr()
            quant_scale_ptr = self._s1_quant_scale.data_ptr()
        self._s1_prepare(
            fx.Int64(self._s1_disp.data_ptr()),
            fx.Int32(cur_tok),
            fx.Int64(topk_ids.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            fx.Int64(self._s1_epoch_expected.data_ptr()),
            fx.Int64(quant_input_ptr),
            fx.Int64(quant_output_ptr),
            fx.Int64(quant_scale_ptr),
            stream,
            rank=self.rank,
            experts_per_rank=self.epr,
            fuse_npes=self.world_size,
            fuse_topk=self.topk,
            fuse_mtpr=self.mtpr,
            sort_block_m=config.sort_block_m,
            num_dispatch_cu=config.num_dispatch_cu,
            num_prepare_cu=prepare_blocks,
            num_quant_cu=quant_blocks,
            quant_cu_capacity=self._s1_quant_cu_capacity,
            model_dim=self.model_dim,
            payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready,
            tile_state_stride=self._s1_tile_state_stride,
            fanout_masks=compile_fanout_masks,
            runtime_fanout=self._s1_runtime_fanout,
            dynamic_fanout=self._s1_runtime_fanout,
        )
        if os.environ.get("MEGA_DEBUG_PREPARE_ONLY") == "1":
            torch.cuda.synchronize()
            pair_rows = self._s1_dispatch_workspace["fanout_pair_config"].view(
                2, self.world_size
            ).cpu().tolist()
            print(
                f"[MEGA_DEBUG] rank={self.rank} prepare-kernel-complete "
                f"group_done={int(self._s1_dispatch_workspace['group_done'][0].item())} "
                f"parity={int(self._s1_epoch_parity.item())} pairs={pair_rows}",
                flush=True,
            )
            raise RuntimeError("MEGA_DEBUG_PREPARE_ONLY complete")

    def _run_fused_stage1(
        self,
        x,
        wts,
        scales,
        topk_ids,
        stream=None,
        config: Stage1Config | None = None,
        *,
        prepared: bool = False,
    ):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(stream, "cuda_stream"):
            stream = fx.Stream(stream.cuda_stream)
        cur_tok = int(x.shape[0])
        if cur_tok > self.mtpr:
            raise ValueError(f"run_tokens={cur_tok} > max_tok_per_rank={self.mtpr}")
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if tuple(x.shape) != (cur_tok, self.model_dim):
            raise ValueError(f"x must have shape ({cur_tok}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (cur_tok, self.topk):
            raise ValueError(f"wts must have shape ({cur_tok}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        if config is None:
            config = self._select_config(cur_tok).stage1
        if prepared and self._s1_fixed_slot:
            raise ValueError("fixed-slot Stage1 cannot consume compact prepare")
        if not prepared:
            self._run_compact_prepare(topk_ids, cur_tok, config, stream)
        op = self._s1_op
        common_args = (
            self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
        )
        common_compile = {
            "model_dim": self.model_dim,
            "inter_dim": self.inter_dim,
            "rank": self.rank,
            "experts_per_rank": self.epr,
            "fuse_npes": self.world_size,
            "fuse_topk": self.topk,
            "fuse_cap": self._s1_cap,
            "fuse_mtpr": self.mtpr,
            "fuse_scale_dim": self._s1_scale_dim,
            "fixed_slot_dispatch": self._s1_fixed_slot,
            "num_cu": self._s1_num_cu,
            "swiglu_limit": self.swiglu_limit,
        }
        entry = self._active_bundle_entry
        debug_role = int(os.environ.get("MEGA_DEBUG_STAGE1_ROLE", "0"))
        use_bundle = (
            entry is not None
            and entry.config.stage1 == config
            and not self._s1_fanout_masks
            and debug_role == 0
        )
        if use_bundle:
            self._s1_mega_bundle(
                *common_args,
                entry.stage1_variant_id,
                stream,
                **common_compile,
                tile_state_stride=self._s1_tile_state_stride,
                variants=self._bundle_plan.stage1_variants,
            )
        else:
            self._s1_mega(
                *common_args,
                stream,
                **common_compile,
                sort_block_m=config.sort_block_m,
                tile_n=config.tile_n,
                tile_k=config.tile_k,
                num_waves=config.num_waves,
                grid_mult=config.grid_mult,
                pipe_weights=config.pipe_weights,
                mfma_amajor=config.mfma_amajor,
                swizzle_a=config.swizzle_a,
                async_a_copy=config.async_a_copy,
                num_dispatch_cu=config.num_dispatch_cu,
                use_tile_resource=config.use_tile_resource,
                waves_per_eu_hint=config.waves_per_eu_hint,
                b_nt=config.b_nt,
                work_shards=config.work_shards,
                payload_chunk_rows=config.payload_chunk_rows,
                payload_tile_ready=config.payload_tile_ready,
                tile_state_stride=self._s1_tile_state_stride,
                fanout_masks=(
                    () if self._s1_runtime_fanout else self._s1_fanout_masks
                ),
                runtime_fanout=self._s1_runtime_fanout,
                debug_role_mode=debug_role,
            )
        if os.environ.get("MEGA_DEBUG_STAGE1_ONLY") == "1":
            torch.cuda.synchronize()
            torch.distributed.barrier()
            tile_ready = self._s1_dispatch_workspace["tile_ready"]
            tile_expected = self._s1_dispatch_workspace["tile_expected"]
            num_valid = int(op.num_valid[0].item())
            num_tiles = (num_valid + config.sort_block_m - 1) // config.sort_block_m
            parity = int(self._s1_epoch_parity.item())
            state_base = parity * self._s1_tile_state_stride
            ready = tile_ready[state_base : state_base + num_tiles].cpu()
            expected = tile_expected[state_base : state_base + num_tiles].cpu()
            expert = op.sorted_expert_ids[:num_tiles].cpu()
            tile_row = op.tile_row_base[:num_tiles].cpu()
            tile_input = self._s1_tile_input_base[:num_tiles].cpu()
            mismatch = ready != expected
            expert_low = self.rank * self.epr
            expert_high = expert_low + self.epr
            bad_expert = (expert < expert_low) | (expert >= expert_high)
            bad_row = (tile_row < 0) | (tile_row + config.sort_block_m > self._s1_nvm)
            bad_input = (tile_input < 0) | (
                tile_input + config.sort_block_m > self._s1_nvm
            )
            print(
                f"[MEGA_DEBUG] rank={self.rank} stage1-kernel-complete "
                f"tiles={num_tiles} mismatch={int(mismatch.sum())} "
                f"ready_sum={int(ready.sum())} expected_sum={int(expected.sum())} "
                f"bad_expert={int(bad_expert.sum())} bad_row={int(bad_row.sum())} "
                f"bad_input={int(bad_input.sum())} "
                f"trb=[{int(tile_row.min())},{int(tile_row.max())}] "
                f"tib=[{int(tile_input.min())},{int(tile_input.max())}]",
                flush=True,
            )
            dump_prefix = os.environ.get("MEGA_DEBUG_DUMP_META", "")
            if dump_prefix:
                workspace = self._s1_dispatch_workspace
                torch.save(
                    {
                        "expert": expert,
                        "tile_row": tile_row,
                        "tile_input": tile_input,
                        "expected": expected,
                        "pair_order": workspace["pair_order"][: x.shape[0] * self.topk]
                        .cpu(),
                        "route_segment": workspace["route_segment"][: x.shape[0] * self.topk]
                        .cpu(),
                        "pair_base": workspace["pair_base"].cpu(),
                        "local_cursor": workspace["local_cursor"].cpu(),
                        "task_base": workspace["my_base"].cpu(),
                        "group_base": workspace["group_base"].cpu(),
                        "count_matrix": workspace["bigcnt"].cpu(),
                    },
                    f"{dump_prefix}.rank{self.rank}.pt",
                )
            raise RuntimeError("MEGA_DEBUG_STAGE1_ONLY complete")
        self._s1_active_tile_m = config.sort_block_m
        return self._s1_active_tile_m

    def quantize(self, x_bf16):
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def _run_joint(
        self,
        x,
        scales,
        wts,
        topk_ids,
        run_tokens,
        stream,
        slice_output,
        *,
        config=None,
        prepared=False,
    ):
        if config is None:
            config = self._select_config(run_tokens)
        self._run_fused_stage1(
            x,
            wts,
            scales,
            topk_ids,
            stream=stream,
            config=config.stage1,
            prepared=prepared,
        )
        return self._run_stage2(run_tokens, stream, slice_output, config)

    def _run_stage2(self, run_tokens, stream, slice_output, config: MegaMoEConfig):
        if config.stage2.aligned_pair:
            return self._run_aligned_pair_stage2_candidate(
                run_tokens,
                config,
                stream,
                pair_cu=config.stage2.pair_cu,
                pair_bm=config.stage2.pair_block_m,
                pair_bn=config.stage2.pair_block_n,
                parallel=True,
                pair_work_weight=config.stage2.pair_work_weight,
                pair_dual_accumulator=True,
                residual_block_m=config.stage2.block_m,
                residual_persist_cu=config.stage2.persist_cu,
                scatter_vec=config.stage2.pair_scatter_vec,
                pair_main_first=True,
                pair_m_swizzle=True,
            )
        ret = self._run_fused_stage2(run_tokens, config, stream)
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if self._s1_fixed_slot:
            x_q, scales = self.quantize(x_bf16)
            return self._run_joint(
                x_q, scales, wts, topk_ids, run_tokens, stream, slice_output
            )

        config = self._select_config(run_tokens)
        prepare_stream = stream
        if prepare_stream is None:
            prepare_stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(prepare_stream, "cuda_stream"):
            prepare_stream = fx.Stream(prepare_stream.cuda_stream)
        self._run_compact_prepare(
            topk_ids,
            run_tokens,
            config.stage1,
            prepare_stream,
            quant_input=x_bf16,
        )
        x_q = self._s1_quant_x[:run_tokens]
        scales = self._s1_quant_scale[:run_tokens]
        return self._run_joint(
            x_q,
            scales,
            wts,
            topk_ids,
            run_tokens,
            stream,
            slice_output,
            config=config,
            prepared=True,
        )

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self):
        from .mega_moe_stage2 import (
            preload_mega_moe_stage2,
            run_mega_moe_stage2,
        )
        from .mega_moe_stage2_aligned_pair import (
            preload_mega_moe_stage2_aligned_pair,
            run_mega_moe_stage2_aligned_pair,
        )

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        self._g2_run = run_mega_moe_stage2
        self._g2_preload = preload_mega_moe_stage2
        self._g2_pair_run = run_mega_moe_stage2_aligned_pair
        self._g2_pair_preload = preload_mega_moe_stage2_aligned_pair
        self._g2_pair_stream = torch.cuda.Stream(device=dev, priority=-1)
        self._g2_pair_start = torch.cuda.Event()
        self._g2_pair_done = torch.cuda.Event()
        self._g2_residual_stream = None
        self._g2_residual_stream_owner = None
        self._g2_residual_stream_raw = None
        self._g2_residual_aux_stream = torch.cuda.Stream(device=dev, priority=0)
        self._g2_residual_start = torch.cuda.Event()
        self._g2_residual_done = torch.cuda.Event()
        masked_ranks = {
            int(value)
            for value in os.environ.get(
                "MEGA_DEBUG_STAGE2_MASKED_RESIDUAL_RANKS", ""
            ).split(",")
            if value
        }
        if self.rank in masked_ranks:
            cus_per_word = int(
                os.environ.get("MEGA_DEBUG_STAGE2_RESIDUAL_CUS_PER_XCD", "4")
            )
            (
                self._g2_residual_stream,
                self._g2_residual_stream_owner,
                self._g2_residual_stream_raw,
            ) = _create_masked_stream(dev, int(cu_num), cus_per_word)
        self._g2_invariants_by_quant = {}
        for p2p_quant in ("none", "fp8_blockwise_1x32"):
            p2p_row_nbytes = (
                int(comb_cfg.hidden_dim) + int(comb_cfg.hidden_dim) // 32
                if p2p_quant == "fp8_blockwise_1x32"
                else int(comb_cfg.hidden_dim) * 2
            )
            self._g2_invariants_by_quant[p2p_quant] = {
                "model_dim": int(comb_cfg.hidden_dim), "inter_dim": int(self.inter_dim),
                "experts": int(comb_cfg.num_experts_per_rank), "topk": int(k), "rank": int(comb_cfg.rank),
                "npes": int(comb_cfg.world_size), "max_tok": int(comb_cfg.max_num_inp_token_per_rank),
                "recv_cap": int(self.max_recv),
                "comb_inp_nbytes": int(comb_cfg.max_num_inp_token_per_rank) * int(k) * p2p_row_nbytes,
                "HIDDEN_MAX": int(comb_cfg.hidden_dim), "INTER_MAX": int(self.inter_dim), "cu_num": int(cu_num),
                "p2p_quant_type": p2p_quant, "fixed_slot_dispatch": bool(self._s1_fixed_slot),
            }
        self._g2_combine_placeholder = torch.empty(
            1, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev
        )

    def _preload_fused_stage2(
        self,
        config: MegaMoEConfig,
        stream,
        *,
        runtime_pair_skip: bool = False,
        scatter_vec: int = 8,
    ):
        comb_op = self.comb_op
        op = self._s1_op
        stage2 = config.stage2
        invariants = self._g2_invariants_by_quant[config.p2p_quant]
        self._g2_preload(
            fx.Int64(self._s1_out.view(-1).data_ptr()),
            fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()),
            fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["expert_tile_end"].data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["bigcnt"].data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["fanout_pair_config"].data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            comb_op._fx_p2p_comb_inp,
            self._s1_nvm,
            self._g2v2_inter,
            self._g2v2_hidden,
            stream,
            BM=stage2.block_m,
            SBM=config.stage1.sort_block_m,
            BN=stage2.block_n,
            BK=stage2.block_k,
            use_nt=stage2.use_nt,
            g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch,
            g2_spart=stage2.spatial_partition,
            persist=stage2.persist,
            persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided,
            skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds,
            runtime_pair_skip=runtime_pair_skip,
            scatter_vec=scatter_vec,
            **invariants,
        )

    def _preload_aligned_pair_stage2(self, config: MegaMoEConfig, stream):
        stage2 = config.stage2
        invariants = self._g2_invariants_by_quant[config.p2p_quant]
        workspace = self._s1_dispatch_workspace
        op = self._s1_op
        self._g2_pair_preload(
            fx.Int64(self._s1_out.view(-1).data_ptr()),
            fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()),
            fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(workspace["expert_tile_end"].data_ptr()),
            fx.Int64(workspace["bigcnt"].data_ptr()),
            fx.Int64(workspace["fanout_pair_config"].data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            self.comb_op._fx_p2p_comb_inp,
            self._s1_nvm,
            self._g2v2_inter,
            self._g2v2_hidden,
            stream,
            model_dim=int(invariants["model_dim"]),
            inter_dim=int(invariants["inter_dim"]),
            experts=int(invariants["experts"]),
            topk=int(invariants["topk"]),
            rank=int(invariants["rank"]),
            npes=int(invariants["npes"]),
            max_tok=int(invariants["max_tok"]),
            recv_cap=int(invariants["recv_cap"]),
            comb_inp_nbytes=int(invariants["comb_inp_nbytes"]),
            pair_mask=0,
            runtime_pair=self._s1_runtime_fanout,
            BM=stage2.pair_block_m,
            SBM=config.stage1.sort_block_m,
            BN=stage2.pair_block_n,
            BK=stage2.block_k,
            INTER_MAX=int(invariants["INTER_MAX"]),
            use_nt=stage2.use_nt,
            cu_num=stage2.pair_cu,
            g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch,
            pair_work_weight=stage2.pair_work_weight,
            dual_accumulator=True,
            scatter_vec=stage2.pair_scatter_vec,
            m_swizzle=True,
        )

    def preload_stage2_bundle(self):
        """Load every production Stage2 variant without GPU dispatch."""
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        seen = set()
        for entry in self._bundle_plan.entries:
            config = entry.config
            if entry.stage2_variant_id in seen:
                continue
            seen.add(entry.stage2_variant_id)
            if not config.stage2.aligned_pair:
                self._preload_fused_stage2(config, stream)
                continue
            residual_stage2 = replace(
                config.stage2,
                skew_cu=config.stage2.persist_cu,
            )
            residual_config = replace(config, stage2=residual_stage2)
            self._preload_fused_stage2(
                residual_config,
                stream,
                runtime_pair_skip=self._s1_runtime_fanout,
                scatter_vec=config.stage2.pair_scatter_vec,
            )
            self._preload_aligned_pair_stage2(config, stream)
        for entry in self._bundle_plan.entries:
            self.comb_op.preload_combine_no_stage1(
                self._g2_combine_placeholder,
                cur_tok=entry.token_bucket,
                enable_weights=False,
                stage2_p2p_quant=entry.config.p2p_quant,
            )

    def _run_fused_stage2(
        self,
        run_tokens,
        config: MegaMoEConfig,
        stream=None,
        *,
        skip_pair_mask: int = 0,
        runtime_pair_skip: bool = False,
        combine: bool = True,
        lds_reserve_bytes: int = 0,
        skip_pair_compact_work: bool = False,
        skip_pair_tiles_per_cu: int = 0,
        scatter_vec: int = 8,
    ):
        comb_op = self.comb_op
        op = self._s1_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        stage2 = config.stage2
        p2p_quant = config.p2p_quant
        invariants = self._g2_invariants_by_quant[p2p_quant]
        # fmt: off
        self._g2_run(
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["expert_tile_end"].data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["bigcnt"].data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["fanout_pair_config"].data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            comb_op._fx_p2p_comb_inp, self._s1_nvm,
            self._g2v2_inter, self._g2v2_hidden, s_fx, BM=stage2.block_m,
            SBM=config.stage1.sort_block_m, BN=stage2.block_n, BK=stage2.block_k,
            use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_spart=stage2.spatial_partition,
            persist=stage2.persist, persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided, skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds, skip_pair_mask=int(skip_pair_mask),
            runtime_pair_skip=bool(runtime_pair_skip),
            lds_reserve_bytes=int(lds_reserve_bytes),
            skip_pair_compact_work=bool(skip_pair_compact_work),
            skip_pair_tiles_per_cu=int(skip_pair_tiles_per_cu),
            scatter_vec=int(scatter_vec),
            **invariants)
        # fmt: on
        self._g2_active_block_m = stage2.block_m
        if not combine:
            return None
        return comb_op.combine_no_stage1(
            self._g2_combine_placeholder, None, None, cur_tok=run_tokens, enable_weights=False,
            stage2_p2p_quant=p2p_quant,
        )

    def _run_aligned_pair_stage2(
        self,
        config: MegaMoEConfig,
        stream=None,
        *,
        pair_cu: int = 112,
        pair_bm: int = 64,
        pair_bn: int = 0,
        pair_first_bf16: bool = False,
        pair_no_scatter: bool = False,
        pair_include_residual: bool = False,
        pair_work_weight: int = 2,
        pair_lds_reserve: int = 0,
        pair_dual_accumulator: bool = False,
        pair_parallel_experts: bool = False,
        pair_scatter_vec: int = 8,
        pair_m_swizzle: bool = False,
    ):
        """Run the isolated aligned-common-row Stage2 prototype."""
        if config.p2p_quant != "fp8_blockwise_1x32":
            raise ValueError("aligned-pair Stage2 currently requires FP8 P2P output")
        if not self._s1_fanout_masks and not self._s1_runtime_fanout:
            raise ValueError("aligned-pair Stage2 requires Stage1 fanout masks")
        pair_mask = (
            0
            if self._s1_runtime_fanout
            else int(self._s1_fanout_masks[self.rank])
        )
        if not self._s1_runtime_fanout and pair_mask.bit_count() != 2:
            raise ValueError("aligned-pair Stage2 currently requires one expert pair per rank")
        if stream is None:
            stream = torch.cuda.current_stream()
        stage2 = config.stage2
        invariants = self._g2_invariants_by_quant[config.p2p_quant]
        workspace = self._s1_dispatch_workspace
        op = self._s1_op
        pair_use_nt = bool(
            int(
                os.environ.get(
                    "MEGA_DEBUG_STAGE2_PAIR_USE_NT", int(stage2.use_nt)
                )
            )
        )
        pair_bhoist = bool(
            int(
                os.environ.get(
                    "MEGA_DEBUG_STAGE2_PAIR_BHOIST", int(stage2.b_hoist)
                )
            )
        )
        pair_ascale_pf = bool(
            int(
                os.environ.get(
                    "MEGA_DEBUG_STAGE2_PAIR_ASCALE_PF",
                    int(stage2.ascale_prefetch),
                )
            )
        )
        self._g2_pair_run(
            fx.Int64(self._s1_out.view(-1).data_ptr()),
            fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()),
            fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(workspace["expert_tile_end"].data_ptr()),
            fx.Int64(workspace["bigcnt"].data_ptr()),
            fx.Int64(workspace["fanout_pair_config"].data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            self.comb_op._fx_p2p_comb_inp,
            self._s1_nvm,
            self._g2v2_inter,
            self._g2v2_hidden,
            fx.Stream(stream.cuda_stream),
            model_dim=int(invariants["model_dim"]),
            inter_dim=int(invariants["inter_dim"]),
            experts=int(invariants["experts"]),
            topk=int(invariants["topk"]),
            rank=int(invariants["rank"]),
            npes=int(invariants["npes"]),
            max_tok=int(invariants["max_tok"]),
            recv_cap=int(invariants["recv_cap"]),
            comb_inp_nbytes=int(invariants["comb_inp_nbytes"]),
            pair_mask=pair_mask,
            runtime_pair=self._s1_runtime_fanout,
            BM=int(pair_bm),
            SBM=config.stage1.sort_block_m,
            BN=int(pair_bn or stage2.block_n),
            BK=stage2.block_k,
            INTER_MAX=int(invariants["INTER_MAX"]),
            use_nt=pair_use_nt,
            cu_num=int(pair_cu),
            g2_bhoist=pair_bhoist,
            g2_ascale_pf=pair_ascale_pf,
            first_bf16=bool(pair_first_bf16),
            diagnostic_no_scatter=bool(pair_no_scatter),
            include_residual=bool(pair_include_residual),
            pair_work_weight=int(pair_work_weight),
            lds_reserve_bytes=int(pair_lds_reserve),
            dual_accumulator=bool(pair_dual_accumulator),
            parallel_experts=bool(pair_parallel_experts),
            scatter_vec=int(pair_scatter_vec),
            m_swizzle=bool(pair_m_swizzle),
        )

    def _run_aligned_pair_stage2_candidate(
        self,
        run_tokens: int,
        config: MegaMoEConfig,
        stream=None,
        *,
        pair_cu: int = 256,
        pair_bm: int = 32,
        pair_bn: int = 0,
        pair_first_bf16: bool = False,
        pair_no_scatter: bool = False,
        parallel: bool = False,
        unified: bool = False,
        pair_work_weight: int = 2,
        co_resident: bool = False,
        pair_dual_accumulator: bool = False,
        pair_parallel_experts: bool = False,
        residual_compact_work: bool = False,
        residual_tiles_per_cu: int = 0,
        residual_block_m: int = 0,
        residual_persist_cu: int = 0,
        scatter_vec: int = 8,
        pair_first_submit: bool = False,
        pair_main_first: bool = False,
        pair_m_swizzle: bool = False,
    ):
        """Run direct-skip plus aligned pair fusion, then the normal combine."""
        pair_mask = (
            0
            if self._s1_runtime_fanout
            else int(self._s1_fanout_masks[self.rank])
        )
        main_stream = torch.cuda.current_stream() if stream is None else stream
        if unified:
            self._run_aligned_pair_stage2(
                config,
                main_stream,
                pair_cu=pair_cu,
                pair_bm=pair_bm,
                pair_bn=pair_bn,
                pair_first_bf16=pair_first_bf16,
                pair_no_scatter=pair_no_scatter,
                pair_include_residual=True,
                pair_work_weight=pair_work_weight,
                pair_dual_accumulator=pair_dual_accumulator,
                pair_parallel_experts=pair_parallel_experts,
                pair_scatter_vec=scatter_vec,
            )
        else:
            residual_config = config
            if residual_block_m or residual_persist_cu:
                residual_stage2 = config.stage2
                if residual_block_m:
                    residual_stage2 = replace(
                        residual_stage2, block_m=int(residual_block_m)
                    )
                if residual_persist_cu:
                    residual_stage2 = replace(
                        residual_stage2,
                        persist_cu=int(residual_persist_cu),
                        skew_cu=int(residual_persist_cu),
                    )
                residual_config = replace(
                    config,
                    stage2=residual_stage2,
                )
            pair_main_first = parallel and pair_main_first
            masked_residual = (
                parallel
                and not pair_main_first
                and self._g2_residual_stream is not None
            )
            if pair_main_first:
                self._g2_residual_start.record(main_stream)
                self._g2_residual_aux_stream.wait_event(self._g2_residual_start)
                self._run_aligned_pair_stage2(
                    config,
                    main_stream,
                    pair_cu=pair_cu,
                    pair_bm=pair_bm,
                    pair_bn=pair_bn,
                    pair_first_bf16=pair_first_bf16,
                    pair_no_scatter=pair_no_scatter,
                    pair_dual_accumulator=pair_dual_accumulator,
                    pair_parallel_experts=pair_parallel_experts,
                    pair_scatter_vec=scatter_vec,
                    pair_m_swizzle=pair_m_swizzle,
                )
                self._run_fused_stage2(
                    run_tokens,
                    residual_config,
                    self._g2_residual_aux_stream,
                    skip_pair_mask=pair_mask,
                    runtime_pair_skip=self._s1_runtime_fanout,
                    combine=False,
                    lds_reserve_bytes=94 * 1024 if co_resident else 0,
                    skip_pair_compact_work=residual_compact_work,
                    skip_pair_tiles_per_cu=residual_tiles_per_cu,
                    scatter_vec=scatter_vec,
                )
                self._g2_residual_done.record(self._g2_residual_aux_stream)
                main_stream.wait_event(self._g2_residual_done)
            elif masked_residual:
                self._g2_residual_start.record(main_stream)
                self._g2_residual_stream.wait_event(self._g2_residual_start)
                self._run_fused_stage2(
                    run_tokens,
                    residual_config,
                    self._g2_residual_stream,
                    skip_pair_mask=pair_mask,
                    runtime_pair_skip=self._s1_runtime_fanout,
                    combine=False,
                    lds_reserve_bytes=94 * 1024 if co_resident else 0,
                    skip_pair_compact_work=residual_compact_work,
                    skip_pair_tiles_per_cu=residual_tiles_per_cu,
                    scatter_vec=scatter_vec,
                )
                self._g2_residual_done.record(self._g2_residual_stream)
                self._run_aligned_pair_stage2(
                    config,
                    main_stream,
                    pair_cu=pair_cu,
                    pair_bm=pair_bm,
                    pair_bn=pair_bn,
                    pair_first_bf16=pair_first_bf16,
                    pair_no_scatter=pair_no_scatter,
                    pair_dual_accumulator=pair_dual_accumulator,
                    pair_parallel_experts=pair_parallel_experts,
                    pair_scatter_vec=scatter_vec,
                    pair_m_swizzle=pair_m_swizzle,
                )
                main_stream.wait_event(self._g2_residual_done)
            elif parallel:
                self._g2_pair_start.record(main_stream)
            if (
                parallel
                and not pair_main_first
                and not masked_residual
                and pair_first_submit
            ):
                self._g2_pair_stream.wait_event(self._g2_pair_start)
                self._run_aligned_pair_stage2(
                    config,
                    self._g2_pair_stream,
                    pair_cu=pair_cu,
                    pair_bm=pair_bm,
                    pair_bn=pair_bn,
                    pair_first_bf16=pair_first_bf16,
                    pair_no_scatter=pair_no_scatter,
                    pair_dual_accumulator=pair_dual_accumulator,
                    pair_parallel_experts=pair_parallel_experts,
                    pair_scatter_vec=scatter_vec,
                    pair_m_swizzle=pair_m_swizzle,
                )
                self._g2_pair_done.record(self._g2_pair_stream)
            if not pair_main_first and not masked_residual:
                self._run_fused_stage2(
                    run_tokens,
                    residual_config,
                    main_stream,
                    skip_pair_mask=pair_mask,
                    runtime_pair_skip=self._s1_runtime_fanout,
                    combine=False,
                    lds_reserve_bytes=94 * 1024 if co_resident else 0,
                    skip_pair_compact_work=residual_compact_work,
                    skip_pair_tiles_per_cu=residual_tiles_per_cu,
                    scatter_vec=scatter_vec,
                )
            if (
                parallel
                and not pair_main_first
                and not masked_residual
                and not pair_first_submit
            ):
                self._g2_pair_stream.wait_event(self._g2_pair_start)
                self._run_aligned_pair_stage2(
                    config,
                    self._g2_pair_stream,
                    pair_cu=pair_cu,
                    pair_bm=pair_bm,
                    pair_bn=pair_bn,
                    pair_first_bf16=pair_first_bf16,
                    pair_no_scatter=pair_no_scatter,
                    pair_lds_reserve=(
                        0 if pair_dual_accumulator else 95 * 1024
                    )
                    if co_resident
                    else 0,
                    pair_dual_accumulator=pair_dual_accumulator,
                    pair_parallel_experts=pair_parallel_experts,
                    pair_scatter_vec=scatter_vec,
                    pair_m_swizzle=pair_m_swizzle,
                )
                self._g2_pair_done.record(self._g2_pair_stream)
            if parallel and not pair_main_first and not masked_residual:
                main_stream.wait_event(self._g2_pair_done)
            elif not parallel:
                self._run_aligned_pair_stage2(
                    config,
                    main_stream,
                    pair_cu=pair_cu,
                    pair_bm=pair_bm,
                    pair_bn=pair_bn,
                    pair_first_bf16=pair_first_bf16,
                    pair_no_scatter=pair_no_scatter,
                    pair_dual_accumulator=pair_dual_accumulator,
                    pair_parallel_experts=pair_parallel_experts,
                    pair_scatter_vec=scatter_vec,
                )
        ret = self.comb_op.combine_no_stage1(
            self._g2_combine_placeholder,
            None,
            None,
            cur_tok=run_tokens,
            enable_weights=False,
            stage2_p2p_quant=config.p2p_quant,
        )
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[
                    : self.mtpr * cfg.combine_token_bytes
                ]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens]
