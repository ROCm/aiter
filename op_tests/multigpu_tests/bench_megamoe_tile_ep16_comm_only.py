# SPDX-License-Identifier: MIT
"""EP16 Stage-1 A/B harness for communication and full-GMM1 variants.

Non-rejoin probe modes remove the GEMM contraction loops.  Rejoin and tilepipe
modes retain the production GMM1, SiLU and A4 requant in the same Stage-1
launch.  HIP events split Stage-1 and Stage-2.  The MORI path is split more
finely so two receiver-readiness boundaries are visible:

* ``bf16_to_a4 + dispatch``: MORI's public communication call boundary;
* the above plus ``local_prepare``: expert selection/sort/scale placement,
  which the fused receiver scoreboard performs inside Stage-1.

Stage-2 keeps the full production direct-LSA FP32 atomic epilogue, scoreboard,
RAIL return and final combine.  Its zero accumulator changes values, not the
number or destination addresses of atomic operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import validate_public_stage1_contract
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    SharedInputs,
    _prepare_local_a4w4,
    _sample_stats,
    _setup_dist,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    MoriBf16A4W4BaselinePath,
    _path_summary,
    _run_path,
    _validate_direct_tile_debug_snapshot,
    _validate_rank_balanced_routes,
)
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
    STAGE1_FULL_GMM1_SUFFIX,
    STAGE1_PROBE_SUFFIX,
    STAGE2_PROBE_SUFFIX,
)


class SplitMoriCommunicationPath(MoriBf16A4W4BaselinePath):
    """MORI reference with expert preparation separated from both GEMMs."""

    name = "mori_inter_node_v1_split_comm"
    stage_names = (
        "bf16_to_a4",
        "dispatch",
        "local_prepare",
        "partial_prepare",
        "combine",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = None
        self._prepare_workspace = None

    def _prepare_stage(self):
        if self._dispatch is None:
            raise RuntimeError("dispatch stage has not run")
        dispatched, weights, scales, ids, _ = self._dispatch
        self._context = _prepare_local_a4w4(
            dispatched[: self.valid_recv],
            scales[: self.valid_recv],
            ids[: self.valid_recv],
            weights[: self.valid_recv],
            self.shape,
            self.rank,
            validate_routes=self.validate_routes,
            output_workspace=self._prepare_workspace,
        )
        if self._prepare_workspace is None:
            self._prepare_workspace = {
                name: self._context[name]
                for name in ("inter_q", "inter_s", "hidden_dummy")
            }
        if self._local_full is None:
            # Constructor/prime-only allocation.  Timed iterations reuse it;
            # zero/copy remain in the partial-store stage, as required by
            # MORI combine's full receive-capacity input contract.
            self._local_full = torch.empty(
                (dispatched.shape[0], self.shape.hidden),
                dtype=torch.bfloat16,
                device=dispatched.device,
            )
        return self._context

    def _partial_prepare_stage(self):
        if self._context is None or self._local_full is None:
            raise RuntimeError("partial preparation requires receiver metadata")
        # The communication comparison does not execute either GEMM.  A full
        # receive-capacity zero partial preserves MORI combine's production
        # message count, size, synchronization and destination addresses.
        self._local_full.zero_()
        return self._local_full

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        timer.stage("bf16_to_a4", self._input_quant_stage)
        timer.stage("dispatch", self._dispatch_stage)
        timer.stage("local_prepare", self._prepare_stage)
        timer.stage("partial_prepare", self._partial_prepare_stage)
        return timer.stage("combine", self._combine_stage)

    def prime_and_check(self) -> torch.Tensor:
        self._input_quant_stage()
        dispatched = self._dispatch_stage()
        recv_count = int(dispatched[4].item())
        if recv_count != self.valid_recv:
            raise AssertionError(
                f"rank-balanced recv_count={recv_count}, expected {self.valid_recv}"
            )
        self._prepare_stage()
        self._partial_prepare_stage()
        output = self._combine_stage()
        torch.cuda.synchronize(output.device)
        if not torch.isfinite(output[: self.shape.tokens].float()).all().item():
            raise AssertionError("MORI communication reference produced non-finite output")
        self.validate_routes = False
        return output


def _lightweight_shared_inputs(
    shape: BenchmarkShape,
    rank: int,
    world: int,
    device: torch.device,
    *,
    quantize_for_mori: bool,
    prepare_stage1_weights: bool = False,
    route_pattern: str = "rank-balanced-hot",
    remote_token_count: int = 64,
    hot_rank: int = 0,
    local_route_count: int = 8,
) -> SharedInputs:
    """Build route/activation inputs without materializing unread GEMM weights."""

    if quantize_for_mori:
        from op_tests.multigpu_tests.bench_mega_moe_v2 import make_inputs

        x, route_weights, topk_ids = make_inputs(
            shape.tokens,
            rank,
            world,
            shape.hidden,
            shape.experts,
            shape.topk,
            "rank-balanced-hot",
            0.6,
            device,
        )
        # Only MORI consumes these buffers.  Keep its measured public BF16
        # boundary unchanged when that path is selected.
        from aiter.ops.quant import per_1x32_f4_quant

        a_quant, a_scale = per_1x32_f4_quant(x, shuffle=False)
    else:
        # Build the balanced route entirely on CPU.  Even tiny first-use GPU
        # rand/topk/softmax operations contend badly when eight rank processes
        # initialize together on this bring-up image.  This deterministic hash
        # preserves one route per EP rank and an approximately 60% hot expert.
        if shape.topk != world:
            raise ValueError("probe-only CPU route requires topk == EP size")
        local_experts = shape.experts // world
        token = torch.arange(shape.tokens, dtype=torch.int64).view(-1, 1)
        remote_token_count = int(remote_token_count)
        if not 0 <= remote_token_count <= shape.tokens:
            raise ValueError("remote_token_count must be in [0, tokens]")
        hot_rank = int(hot_rank)
        if not 0 <= hot_rank < world:
            raise ValueError("hot_rank must be in [0, world)")
        local_route_count = int(local_route_count)
        if not 0 <= local_route_count <= shape.topk:
            raise ValueError("local_route_count must be in [0, topk]")
        if route_pattern == "rank-balanced-hot":
            owner = torch.arange(world, dtype=torch.int64).view(1, -1)
        elif route_pattern == "paired-rank-half-remote":
            # Every token has two distinct expert routes on each of eight
            # destination ranks.  Even tokens stay on the source node; odd
            # tokens target the remote node.  Across all sources this keeps
            # exactly 2048 routes per destination rank while exercising both
            # duplicate-rank expansion and 50% cross-node payload elision.
            slot = torch.arange(world, dtype=torch.int64).view(1, -1)
            source_node = rank // shape.gpus_per_node
            target_node = torch.where(
                (token & 1) == 0,
                torch.full_like(token, source_node),
                torch.full_like(token, 1 - source_node),
            )
            owner = target_node * shape.gpus_per_node + slot // 2
        elif route_pattern == "paired-rank-local-only":
            # Same 16 valid routes and two routes per destination rank as the
            # half-remote fixture, but every token remains on its source node.
            # This gives sparse and bulk transport identical GMM/fanout work
            # while making all cross-node payload provably redundant.
            slot = torch.arange(world, dtype=torch.int64).view(1, -1)
            source_node = rank // shape.gpus_per_node
            owner = source_node * shape.gpus_per_node + slot // 2
        elif route_pattern == "paired-rank-remote-prefix":
            # Tokens [0,N) target the remote node and [N,128) stay local.
            # Every token still has 16 valid routes (two per destination rank),
            # so destination GMM work remains exactly 2048 routes for all N.
            slot = torch.arange(world, dtype=torch.int64).view(1, -1)
            source_node = rank // shape.gpus_per_node
            target_node = torch.where(
                token < remote_token_count,
                torch.full_like(token, 1 - source_node),
                torch.full_like(token, source_node),
            )
            owner = target_node * shape.gpus_per_node + slot // 2
        elif route_pattern == "node-route-count":
            slot = torch.arange(world, dtype=torch.int64).view(1, -1)
            source_node = rank // shape.gpus_per_node
            local_owner = source_node * shape.gpus_per_node + slot % 8
            remote_owner = (
                (1 - source_node) * shape.gpus_per_node + slot % 8
            )
            owner = torch.where(
                slot < local_route_count, local_owner, remote_owner
            )
        elif route_pattern in ("single-rank-max", "single-expert-max"):
            owner = torch.full((1, world), hot_rank, dtype=torch.int64)
        elif route_pattern == "local-only-padded":
            node_base = (rank // shape.gpus_per_node) * shape.gpus_per_node
            valid_owner = torch.arange(
                node_base,
                node_base + shape.gpus_per_node,
                dtype=torch.int64,
            ).view(1, -1)
            owner = torch.full(
                (1, world), -1, dtype=torch.int64
            )
            owner[:, : shape.gpus_per_node] = valid_owner
        else:
            raise ValueError(f"unsupported probe route_pattern={route_pattern!r}")
        if route_pattern in (
            "paired-rank-half-remote",
            "paired-rank-local-only",
            "paired-rank-remote-prefix",
        ):
            pair_member = torch.arange(world, dtype=torch.int64).view(1, -1) & 1
            # Keep per-expert load invariant across source rank, target node,
            # and remote_token_count: each token deterministically selects one
            # adjacent expert pair on every destination rank.
            local_expert = (token % (local_experts // 2)) * 2 + pair_member
        elif route_pattern == "single-rank-max":
            local_expert = torch.arange(
                world, dtype=torch.int64
            ).view(1, -1) + token * 0
        elif route_pattern == "single-expert-max":
            local_expert = torch.zeros(
                (shape.tokens, world), dtype=torch.int64
            )
        elif route_pattern == "node-route-count":
            local_expert = torch.arange(
                world, dtype=torch.int64
            ).view(1, -1) + token * 0
        else:
            hot_hash = (token * 131 + owner * 17 + rank * 29) % 100
            cold = 1 + (token * 19 + owner * 23 + rank * 31) % (
                local_experts - 1
            )
            local_expert = torch.where(hot_hash < 60, 0, cold)
        topk_ids_cpu = torch.where(
            owner >= 0,
            owner * local_experts + local_expert,
            torch.full_like(owner, -1),
        ).to(torch.int32)
        if route_pattern == "rank-balanced-hot":
            expected_owner = torch.arange(world, dtype=torch.int32).view(1, world)
            actual_owner = torch.div(
                topk_ids_cpu, local_experts, rounding_mode="floor"
            )
            if torch.count_nonzero(actual_owner != expected_owner).item():
                raise AssertionError("CPU probe route is not rank balanced")
        if prepare_stage1_weights:
            cpu_index = torch.arange(
                shape.tokens * shape.hidden, dtype=torch.int32
            ).view(shape.tokens, shape.hidden)
            x_cpu = (
                ((cpu_index % 257) - 128).to(torch.float32) / 128.0
            ).to(torch.bfloat16)
            x = x_cpu.to(device)
        else:
            x = torch.zeros(
                (shape.tokens, shape.hidden), dtype=torch.bfloat16
            ).to(device)
        if route_pattern.startswith("paired-rank-") or route_pattern in (
            "single-rank-max",
            "single-expert-max",
            "node-route-count",
        ):
            slot_weight = (
                (
                    torch.arange(world, dtype=torch.int64).view(1, -1)
                    + token
                )
                % world
                + 1
            ).to(torch.float32)
            route_weights = torch.where(
                topk_ids_cpu >= 0,
                slot_weight,
                torch.zeros_like(slot_weight),
            )
        else:
            route_weights = (topk_ids_cpu >= 0).to(torch.float32)
        route_weights /= route_weights.sum(dim=1, keepdim=True)
        route_weights = route_weights.to(device)
        topk_ids = topk_ids_cpu.to(device)
        # The fused probe consumes x_bf16 and performs the real quantization
        # inside Stage1.  These MORI-only fields are never read.
        from aiter.utility import dtypes

        a_quant = torch.empty(
            (shape.tokens, shape.hidden // 2),
            dtype=dtypes.fp4x2,
            device=device,
        )
        a_scale = torch.empty(
            (shape.tokens, shape.hidden // 32),
            dtype=torch.uint8,
            device=device,
        ).view(dtypes.fp8_e8m0)
    from aiter.ops.flydsl.kernels.megamoe_tile.compute_v2 import PreparedA4W4Weights

    sentinels = [torch.empty(1, dtype=torch.uint8, device=device) for _ in range(2)]
    if prepare_stage1_weights:
        from aiter.ops.quant import per_1x32_f4_quant
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        generator = torch.Generator(device=device).manual_seed(90_000 + rank)
        w1 = torch.randn(
            (shape.local_experts, 2 * shape.inter, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w1.mul_(shape.hidden**-0.25)
        w1q, w1s = per_1x32_f4_quant(w1, shuffle=False)
        del w1
        w1q = shuffle_weight(w1q, layout=(16, 16))
        w1s = e8m0_shuffle(w1s)
        torch.cuda.empty_cache()
    else:
        w1q = sentinels[0]
        w1s = sentinels[1]
    w2q = torch.empty(1, dtype=torch.uint8, device=device)
    w2s = torch.empty(1, dtype=torch.uint8, device=device)
    prepared = PreparedA4W4Weights(
        w1q,
        w1s,
        w2q,
        w2s,
        shape.local_experts,
        shape.hidden,
        shape.inter,
        None,
        None,
    )
    if quantize_for_mori:
        local_mask = torch.zeros(shape.experts, dtype=torch.int32, device=device)
        first = rank * shape.local_experts
        local_mask[first : first + shape.local_experts] = 1
    else:
        local_mask = torch.empty(1, dtype=torch.int32, device=device)
    return SharedInputs(
        x,
        a_quant,
        a_scale,
        route_weights,
        topk_ids,
        prepared,
        local_mask,
    )


class FusedCommunicationProbePath:
    """Two production-protocol launches, split by independent HIP events."""

    name = "fused_cco_comm_probe"
    stage_names = (
        "stage1_quant_hier_dispatch_scoreboard",
        "stage2_direct_atomic_return_combine",
    )

    def __init__(
        self,
        operator,
        shared,
        shape,
        device,
        *,
        probe_stage: str,
        stage1_mode: str,
        stage1_phase: str,
        cco_chunks_per_flush: int,
        cco_geometry: str,
        quant_two_cta_per_token: bool,
        prequant_input: bool,
        stage2_mode: str,
        canonical_stage1: bool,
        route_pattern: str = "rank-balanced-hot",
    ):
        if probe_stage not in ("stage1", "stage2", "both"):
            raise ValueError("probe_stage must be stage1, stage2, or both")
        self.operator = operator
        self.shared = shared
        self.shape = shape
        self.device = device
        self.probe_stage = probe_stage
        self.stage1_mode = stage1_mode
        self.stage1_phase = stage1_phase
        self.cco_chunks_per_flush = int(cco_chunks_per_flush)
        self.cco_geometry = cco_geometry
        self.route_pattern = route_pattern
        self.quant_two_cta_per_token = bool(quant_two_cta_per_token)
        self.prequant_input = bool(prequant_input)
        self._prequant_q = None
        self._prequant_scale = None
        self._prequant_op = None
        if self.prequant_input and (
            probe_stage != "stage1" or stage1_phase != "full"
        ):
            raise ValueError(
                "prequant input requires a full Stage1-only probe"
            )
        if self.prequant_input:
            from aiter.ops.quant import dynamic_per_group_scaled_quant
            from aiter.utility import dtypes

            self._prequant_q = torch.empty(
                (shape.tokens, shape.hidden // 2),
                dtype=dtypes.fp4x2,
                device=device,
            )
            self._prequant_scale = torch.empty(
                (shape.tokens, shape.hidden // 32),
                dtype=torch.uint8,
                device=device,
            ).view(dtypes.fp8_e8m0)
            self._prequant_op = dynamic_per_group_scaled_quant
            self._validate_prequant_buffers()
        self.stage2_mode = stage2_mode
        self.canonical_stage1 = bool(canonical_stage1)
        local_topk = self.shared.topk_ids.detach().cpu().tolist()
        all_topk: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(all_topk, local_topk)
        local_weight_bits = (
            self.shared.route_weights.detach()
            .contiguous()
            .view(torch.int32)
            .cpu()
            .tolist()
        )
        all_weight_bits: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(all_weight_bits, local_weight_bits)
        self.expected_local_routes = sum(
            int(0 <= int(expert) < self.shape.experts)
            and int(expert) // (self.shape.experts // dist.get_world_size())
            == dist.get_rank()
            for rank_rows in all_topk
            for row in rank_rows
            for expert in row
        )
        self.expected_local_source_rows = sum(
            any(
                0 <= int(expert) < self.shape.experts
                and int(expert) // (self.shape.experts // dist.get_world_size())
                == dist.get_rank()
                for expert in row
            )
            for rank_rows in all_topk
            for row in rank_rows
        )
        expected_metadata = []
        destination_rank = dist.get_rank()
        for source_rank, (rank_rows, rank_weights) in enumerate(
            zip(all_topk, all_weight_bits)
        ):
            for source_token, (row, weight_row) in enumerate(
                zip(rank_rows, rank_weights)
            ):
                for topk_slot, (expert, weight_bits) in enumerate(
                    zip(row, weight_row)
                ):
                    expert = int(expert)
                    if (
                        0 <= expert < self.shape.experts
                        and expert // self.shape.local_experts
                        == destination_rank
                    ):
                        packed = (
                            source_rank * self.shape.tokens + source_token
                        ) | (topk_slot << 24)
                        expected_metadata.append(
                            (
                                packed,
                                expert % self.shape.local_experts,
                                int(weight_bits) & 0xFFFFFFFF,
                            )
                        )
        expected_metadata.sort(key=lambda item: item[0])
        expected_metadata_sha = hashlib.sha256()
        for packed, local_expert, weight_bits in expected_metadata:
            expected_metadata_sha.update(
                struct.pack("<III", packed, local_expert, weight_bits)
            )
        self.expected_metadata_sha256 = expected_metadata_sha.hexdigest()
        local_experts = self.shape.experts // dist.get_world_size()
        local_node = dist.get_rank() // self.shape.gpus_per_node
        remote_node = 1 - local_node
        self.expected_remote_send_tokens = sum(
            any(
                0 <= int(expert) < self.shape.experts
                and (int(expert) // local_experts)
                // self.shape.gpus_per_node
                == remote_node
                for expert in row
            )
            for row in local_topk
        )
        remote_source_rank = (
            remote_node * self.shape.gpus_per_node
            + dist.get_rank() % self.shape.gpus_per_node
        )
        self.expected_remote_receive_tokens = sum(
            any(
                0 <= int(expert) < self.shape.experts
                and (int(expert) // local_experts)
                // self.shape.gpus_per_node
                == local_node
                for expert in row
            )
            for row in all_topk[remote_source_rank]
        )

        def remote_token_masks(rows, target_node):
            masks = [0] * 4
            for source_token, row in enumerate(rows):
                if any(
                    0 <= int(expert) < self.shape.experts
                    and (int(expert) // local_experts)
                    // self.shape.gpus_per_node
                    == target_node
                    for expert in row
                ):
                    masks[source_token % 4] |= 1 << (source_token // 4)
            return masks

        self.expected_remote_send_masks = remote_token_masks(
            local_topk, remote_node
        )
        self.expected_remote_receive_masks = remote_token_masks(
            all_topk[remote_source_rank], local_node
        )
        self.real_gmm1 = stage1_mode in (
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64_rejoin",
        )
        self.tile_pipeline_fanout_shards = int(
            getattr(operator._stage1, "tile_pipeline_fanout_shards", 16)
        )
        if probe_stage == "stage1":
            self.name = (
                (
                    "fused_cco_stage1_wave64_rejoin256_probe"
                    if stage1_mode == "internodev1_wave64_rejoin"
                    else "fused_cco_stage1_wave64_probe"
                )
                if stage1_mode.startswith("internodev1_wave64")
                else (
                    (
                        "fused_stage1_split"
                        f"{8 * self.tile_pipeline_fanout_shards}x2_tilepipe"
                        "256r"
                    )
                    if stage1_mode == "internodev1_tilepipe"
                    else (
                        "fused_stage1_split128x2_rejoin256_posteos"
                        if stage1_mode == "internodev1_split128x2_rejoin"
                        else (
                            "fused_cco_stage1_internodev1_split128x2_probe"
                            if stage1_mode.startswith("internodev1_split128x2")
                            else "fused_cco_stage1_comm_probe"
                        )
                    )
                )
            )
            if (
                self.cco_chunks_per_flush != 1
                and self.cco_geometry == "chunked"
            ):
                self.name += f"_flushb{self.cco_chunks_per_flush}"
            if self.cco_geometry != "chunked":
                self.name += f"_{self.cco_geometry}"
            if self.quant_two_cta_per_token:
                self.name += "_quant2cta"
            self.stage_names = (
                (
                    "stage1_full_quant_transport_fanout_gmm1_silu_requant"
                    if self.real_gmm1
                    else "stage1_quant_hier_dispatch_scoreboard"
                ),
            )
            if self.prequant_input:
                self.name += "_prequant_hip"
                self.stage_names = (
                    "prequant_bf16_to_a4_hip",
                    (
                        "stage1_prequant_transport_fanout_gmm1_silu_requant"
                        if self.real_gmm1
                        else "stage1_prequant_hier_dispatch_scoreboard"
                    ),
                )
            if self.stage1_phase != "full":
                self.name += f"_{self.stage1_phase}"
                self.stage_names = (f"stage1_{self.stage1_phase}",)
        elif probe_stage == "stage2":
            self.name = f"fused_cco_stage2_{stage2_mode}_probe"
            self.stage_names = ("stage2_direct_atomic_return_combine",)
        else:
            self.name = "fused_cco_stage1_stage2_probe"
            self.stage_names = (
                (
                    "stage1_full_quant_transport_fanout_gmm1_silu_requant"
                    if self.real_gmm1
                    else "stage1_quant_hier_dispatch_scoreboard"
                ),
                "stage2_direct_atomic_return_combine",
            )

    def _validate_prequant_buffers(self) -> None:
        if not self.prequant_input:
            return
        from aiter.utility import dtypes

        q = self._prequant_q
        scale = self._prequant_scale
        if q is None or scale is None or self._prequant_op is None:
            raise RuntimeError("prequant buffers and direct binding are required")
        expected_q_shape = (self.shape.tokens, self.shape.hidden // 2)
        expected_scale_shape = (self.shape.tokens, self.shape.hidden // 32)
        if tuple(q.shape) != expected_q_shape or q.dtype != dtypes.fp4x2:
            raise ValueError("prequant q buffer has an incompatible shape or dtype")
        if (
            tuple(scale.shape) != expected_scale_shape
            or scale.dtype != dtypes.fp8_e8m0
        ):
            raise ValueError(
                "prequant scale buffer has an incompatible shape or dtype"
            )
        for name, tensor in (("q", q), ("scale", scale)):
            if tensor.device != self.device:
                raise ValueError(f"prequant {name} buffer is on the wrong device")
            if not tensor.is_contiguous() or tensor.storage_offset() != 0:
                raise ValueError(
                    f"prequant {name} buffer must be base-aligned and contiguous"
                )
            if tensor.element_size() != 1:
                raise ValueError(f"prequant {name} storage must be byte-sized")
            required = tensor.numel() * tensor.element_size()
            if tensor.untyped_storage().nbytes() < required:
                raise ValueError(f"prequant {name} storage is undersized")
        if q.stride() != (self.shape.hidden // 2, 1):
            raise ValueError("prequant q buffer must use row-major byte stride")
        if scale.stride() != (self.shape.hidden // 32, 1):
            raise ValueError("prequant scale buffer must use row-major byte stride")
        if q.data_ptr() % 16:
            raise ValueError("prequant q buffer must be 16-byte aligned")

    def _run_prequant(self) -> torch.Tensor:
        if (
            self._prequant_q is None
            or self._prequant_scale is None
            or self._prequant_op is None
        ):
            raise RuntimeError("prequant path is not initialized")
        self._prequant_op(
            self._prequant_q,
            self.shared.x,
            self._prequant_scale,
            32,
            shuffle_scale=False,
        )
        return self._prequant_q

    def _begin_generation(self):
        op = self.operator
        run_tokens = validate_public_stage1_contract(
            self.shared.x,
            self.shared.route_weights,
            self.shared.topk_ids,
            hidden=op.model_dim,
            topk=op.topk,
            max_tokens=op.mtpr,
        )
        if run_tokens != op.mtpr:
            raise ValueError("comm probe requires exactly 128 tokens/rank")
        op._generation += 1
        return run_tokens, op._generation, op._flydsl_stream(None)

    def _launch_pair(self, timer: HipStageTimer | None):
        op = self.operator
        run_tokens, generation, stream = self._begin_generation()
        if self.stage1_phase != "full":
            setup_kernels = []
            if self.stage1_phase in (
                "transport_only",
                "fanout_only",
                "dispatch_only",
            ):
                setup_kernels.append(
                    op._stage1_quant_setup.kernel_name
                )
            if self.stage1_phase == "fanout_only":
                setup_kernels.append(
                    op._stage1_transport_setup.kernel_name
                )
            print(
                "MEGAMOE_PHASE_BOUNDARY "
                + json.dumps(
                    {
                        "rank": dist.get_rank(),
                        "phase": self.stage1_phase,
                        "setup_kernels_excluded_from_stage_event": setup_kernels,
                        "primary_kernel": op.stage1_kernel_name,
                        "timed_stage": self.stage_names[0],
                        "payload_bytes_per_rank": (
                            self.expected_remote_send_tokens * 4096
                            if self.cco_geometry == "sparse_wqe"
                            else 128 * 4096
                        ),
                        "cco_chunks_per_flush": self.cco_chunks_per_flush,
                        "cco_geometry": self.cco_geometry,
                        "route_pattern": self.route_pattern,
                        "quant_two_cta_per_token": self.quant_two_cta_per_token,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def stage1():
            stage1_input = (
                self._prequant_q if self.prequant_input else self.shared.x
            )
            if stage1_input is None:
                raise RuntimeError("prequant Stage1 input is unavailable")
            op._launch_stage1(
                stage1_input,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
                input_scale=(
                    self._prequant_scale if self.prequant_input else None
                ),
            )

        def prequant():
            return self._run_prequant()

        def stage2():
            op._launch_stage2(run_tokens, generation, stream)

        def phase_setup():
            if self.stage1_phase in (
                "transport_only",
                "fanout_only",
                "dispatch_only",
            ):
                op.launch_stage1_quant_setup(
                    self.shared.x,
                    self.shared.route_weights,
                    self.shared.topk_ids,
                    run_tokens,
                    generation,
                    stream,
                )
            if self.stage1_phase == "fanout_only":
                op.launch_stage1_transport_setup(
                    self.shared.x,
                    self.shared.route_weights,
                    self.shared.topk_ids,
                    run_tokens,
                    generation,
                    stream,
                )

        phase_setup()
        if self.stage1_phase in (
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ):
            # Setup is excluded from the phase event only after every rank has
            # completed it; otherwise a fast primary waits on a slow peer's
            # ready words inside the measured interval.
            torch.cuda.synchronize(self.device)
            dist.barrier()
        if timer is None:
            if self.prequant_input:
                prequant()
            stage1()
            if self.probe_stage in ("stage2", "both"):
                stage2()
        else:
            if self.probe_stage in ("stage1", "both"):
                if self.prequant_input:
                    timer.stage("prequant_bf16_to_a4_hip", prequant)
                    timer.stage(
                        "stage1_prequant_hier_dispatch_scoreboard", stage1
                    )
                else:
                    timer.stage(self.stage_names[0], stage1)
            else:
                # Metadata preparation is ordered on the same stream but lies
                # before the Stage2 start event.
                stage1()
            if self.probe_stage in ("stage2", "both"):
                timer.stage("stage2_direct_atomic_return_combine", stage2)
        if self.probe_stage == "stage1" or self.stage2_mode == "atomic_only":
            return self.shared.x
        return op._output[:run_tokens]

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        return self._launch_pair(timer)

    def prime_and_check(self) -> torch.Tensor:
        op = self.operator
        run_tokens, generation, stream = self._begin_generation()
        if self.stage1_phase in (
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ):
            op.launch_stage1_quant_setup(
                self.shared.x,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
            )
        if self.stage1_phase == "fanout_only":
            op.launch_stage1_transport_setup(
                self.shared.x,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
            )
        if self.stage1_phase in (
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ):
            torch.cuda.synchronize(self.device)
            dist.barrier()
        if self.prequant_input:
            self._validate_prequant_buffers()
            self._run_prequant()
        print(
            f"MEGAMOE_COMM_PROBE_PRIME rank={dist.get_rank()} "
            "phase=stage1_launch",
            flush=True,
        )
        op._launch_stage1(
            self._prequant_q if self.prequant_input else self.shared.x,
            self.shared.route_weights,
            self.shared.topk_ids,
            run_tokens,
            generation,
            stream,
            input_scale=(
                self._prequant_scale if self.prequant_input else None
            ),
        )
        torch.cuda.synchronize(self.device)
        stage1_snapshot = op.debug_stage1_comm_snapshot()
        prequant_record_snapshot = (
            op.debug_quant_pack_snapshot()
            if self.prequant_input and self.canonical_stage1
            else None
        )
        print(
            "MEGAMOE_COMM_PROBE_PRIME "
            + json.dumps(
                {
                    "rank": dist.get_rank(),
                    "phase": "stage1_done",
                    "state": stage1_snapshot,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.probe_stage == "stage1":
            if self.stage1_phase == "quant_core_only":
                if stage1_snapshot["dispatch_staging_ready_count"] != 0:
                    raise AssertionError(
                        "quant-core unexpectedly published staging records"
                    )
                if stage1_snapshot["stage1_error_count"] != 0:
                    raise AssertionError("quant-core reported a protocol error")
                if self.canonical_stage1:
                    import hashlib

                    from aiter.ops.quant import per_1x32_f4_quant_hip

                    reference_q, reference_scale = per_1x32_f4_quant_hip(
                        self.shared.x, shuffle=False
                    )
                    torch.cuda.synchronize(self.device)
                    q_bytes = (
                        reference_q.contiguous()
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
                    scale_bytes = (
                        reference_scale.contiguous()
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
                    fused = op.debug_quant_core_snapshot()
                    reference = {
                        "q_bytes": len(q_bytes),
                        "scale_bytes": len(scale_bytes),
                        "q_sha256": hashlib.sha256(q_bytes).hexdigest(),
                        "scale_sha256": hashlib.sha256(scale_bytes).hexdigest(),
                    }
                    if any(fused[key] != reference[key] for key in reference):
                        raise AssertionError(
                            f"quant-core differs from HIP fast path: "
                            f"fused={fused}, reference={reference}"
                        )
                    print(
                        "MEGAMOE_QUANT_CORE_CANONICAL "
                        + json.dumps(
                            {
                                "rank": dist.get_rank(),
                                "fused": fused,
                                "hip_reference": reference,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                return self.shared.x
            if self.stage1_phase == "quant_pack_only":
                if stage1_snapshot["dispatch_staging_ready_count"] != 128:
                    raise AssertionError("quant-pack did not publish 128 records")
                if stage1_snapshot["stage1_error_count"] != 0:
                    raise AssertionError("quant-pack reported a protocol error")
                expected_half_done = (
                    128 if self.quant_two_cta_per_token else 0
                )
                if (
                    stage1_snapshot["quant_half_done_count"]
                    != expected_half_done
                ):
                    raise AssertionError("quant half completion count mismatch")
                if self.canonical_stage1:
                    quant_canonical = op.debug_quant_pack_snapshot()
                    print(
                        "MEGAMOE_QUANT_PACK_CANONICAL "
                        + json.dumps(
                            {
                                "rank": dist.get_rank(),
                                "quant_two_cta_per_token": (
                                    self.quant_two_cta_per_token
                                ),
                                **quant_canonical,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                return self.shared.x
            if self.stage1_phase == "transport_only":
                expected_words = 2 if self.cco_geometry == "mori64x2" else 32
                expected_requests = (
                    2
                    if self.cco_geometry == "mori64x2"
                    else 4 * (8 // self.cco_chunks_per_flush)
                )
                if stage1_snapshot["dispatch_staging_ready_count"] != 128:
                    raise AssertionError("transport setup staging is incomplete")
                if (
                    stage1_snapshot["remote_chunk_ready_count"]
                    != expected_words
                ):
                    raise AssertionError("transport receive ready coverage is incomplete")
                if (
                    stage1_snapshot["remote_chunk_credit_count"]
                    != expected_words
                ):
                    raise AssertionError("transport credit coverage is incomplete")
                if (
                    stage1_snapshot["remote_request_nonzero_count"]
                    != expected_requests
                ):
                    raise AssertionError(
                        "transport retained request count mismatch"
                    )
                if stage1_snapshot["stage1_error_count"] != 0:
                    raise AssertionError("transport reported a protocol error")
                return self.shared.x
            if stage1_snapshot["comm_role_eos"] != [generation] * 8:
                raise AssertionError("Stage1-only probe is missing communication EOS")
            if stage1_snapshot["stage1_done"] != generation:
                raise AssertionError("Stage1-only probe did not publish stage1_done")
            if (
                stage1_snapshot["expert_count_sum"]
                != self.expected_local_routes
                or stage1_snapshot["tile_arrived_sum"]
                != self.expected_local_routes
            ):
                raise AssertionError("Stage1-only probe lost dispatched routes")
            if stage1_snapshot["stage1_error_count"] != 0:
                raise AssertionError("Stage1-only probe reported a protocol error")
            if self.stage1_mode == "internodev1_tilepipe":
                expected_jobs = (
                    stage1_snapshot["tile_alloc"]
                    * self.operator.stage1_layout.h1_n_blocks
                )
                if (
                    stage1_snapshot["h1_queue_tail"] != expected_jobs
                    or stage1_snapshot["compute_done"] != expected_jobs
                    or stage1_snapshot["queue_permutation_mismatch"] != 0
                ):
                    raise AssertionError(
                        "tile pipeline queue lost, duplicated, or skipped GMM1 jobs"
                    )
                if self.operator.stage1_tile_pipeline_instrument:
                    early_tiles = stage1_snapshot["early_full_tiles"]
                    started = stage1_snapshot[
                        "gmm_jobs_started_before_all_comm_eos"
                    ]
                    completed = stage1_snapshot[
                        "gmm_jobs_completed_before_all_comm_eos"
                    ]
                    valid_overlap = (
                        0 < early_tiles <= stage1_snapshot["tile_alloc"]
                        and 0 < completed <= started <= expected_jobs
                        if expected_jobs > 0
                        else early_tiles == started == completed == 0
                    )
                    if not valid_overlap:
                        raise AssertionError(
                            "tile pipeline did not demonstrate valid pre-EOS "
                            "GMM1 overlap"
                        )
            if self.stage1_mode != "legacy":
                expected_flags = (
                    2 * self.tile_pipeline_fanout_shards
                    if self.stage1_mode
                    == "internodev1_tilepipe"
                    else 32
                )
                if stage1_snapshot["split_flags_ready_per_dest"] != [
                    expected_flags
                ] * 8:
                    raise AssertionError("split fanout producer flags are incomplete")
                if self.cco_geometry == "sparse_wqe":
                    expected_sparse = {
                        "sparse_remote_token_ready_count": (
                            self.expected_remote_receive_tokens
                        ),
                        "sparse_remote_qp_ready_count": 4,
                        "sparse_remote_request_nonzero_count": 4,
                        "sparse_remote_batch_ready": generation,
                        "sparse_remote_credit": generation,
                        "sparse_remote_consumed": 8,
                        "sparse_remote_send_count": (
                            self.expected_remote_send_tokens
                        ),
                    }
                    for field, expected_value in expected_sparse.items():
                        if stage1_snapshot[field] != expected_value:
                            raise AssertionError(
                                f"sparse transport {field}="
                                f"{stage1_snapshot[field]}, expected={expected_value}"
                            )
                    if (
                        stage1_snapshot["sparse_remote_qp_token_masks"]
                        != self.expected_remote_receive_masks
                    ):
                        raise AssertionError(
                            "sparse transport QP token bitmap mismatch"
                        )
                else:
                    expected_consumed = (
                        [8, 8] + [0] * 30
                        if self.cco_geometry == "mori64x2"
                        else [8] * 32
                    )
                    if (
                        stage1_snapshot["remote_chunk_consumed"]
                        != expected_consumed
                    ):
                        raise AssertionError(
                            "split fanout consumed counts are not 8"
                        )
            if self.canonical_stage1:
                from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import (
                    MegaMoETileA4W4,
                )

                heavy = MegaMoETileA4W4.debug_direct_tile_snapshot(op)
                canonical = heavy["canonical_h1"]
                if (
                    canonical["duplicate_packed_keys"] != 0
                    or (
                        self.route_pattern == "rank-balanced-hot"
                        and canonical["missing_low_sources"] != 0
                    )
                    or canonical["valid_rows"]
                    != self.expected_local_routes
                    or canonical["unique_input_rows"]
                    != self.expected_local_source_rows
                    or canonical["shared_input_route_rows"]
                    != self.expected_local_routes
                    - self.expected_local_source_rows
                    or canonical["invalid_input_rows"] != 0
                    or canonical["metadata_sha256"]
                    != self.expected_metadata_sha256
                ):
                    raise AssertionError("Stage1 canonical source mapping is invalid")
                canonical_fields = [
                    "metadata_sha256",
                    "grouped_input_q_sha256",
                    "grouped_input_scale_sha256",
                    "tile_row_input_sha256",
                    "invalid_input_rows",
                    "unique_input_rows",
                    "shared_input_route_rows",
                ]
                if self.stage1_mode in (
                    "internodev1_split128x2_rejoin",
                    "internodev1_tilepipe",
                ):
                    canonical_fields.extend(
                        ["h1_output_q_sha256", "h1_output_scale_sha256"]
                    )
                    if (
                        canonical.get("fused_vs_standalone_changed_rows", 0)
                        != 0
                    ):
                        raise AssertionError(
                            "rejoin H1 differs from standalone replay"
                        )
                canonical_record = {
                    "rank": dist.get_rank(),
                    "generation": generation,
                    "stage1_mode": self.stage1_mode,
                    **{field: canonical[field] for field in canonical_fields},
                }
                print(
                    "MEGAMOE_STAGE1_CANONICAL "
                    + json.dumps(canonical_record, sort_keys=True),
                    flush=True,
                )
                if self.prequant_input:
                    dist.barrier()
                    (
                        reference_tokens,
                        reference_generation,
                        reference_stream,
                    ) = self._begin_generation()
                    op.launch_stage1_internal_quant_reference(
                        self.shared.x,
                        self.shared.route_weights,
                        self.shared.topk_ids,
                        reference_tokens,
                        reference_generation,
                        reference_stream,
                    )
                    torch.cuda.synchronize(self.device)
                    reference_state = op.debug_stage1_comm_snapshot()
                    if (
                        reference_state["stage1_error_count"] != 0
                        or reference_state["expert_count_sum"] != 2048
                        or reference_state["tile_arrived_sum"] != 2048
                    ):
                        raise AssertionError(
                            "internal-quant reference Stage1 protocol is invalid"
                        )
                    internal_record_snapshot = op.debug_quant_pack_snapshot()
                    assert prequant_record_snapshot is not None
                    for field in ("bytes", "sha256"):
                        if (
                            prequant_record_snapshot[field]
                            != internal_record_snapshot[field]
                        ):
                            raise AssertionError(
                                "prequant Stage1 record differs from internal "
                                f"quant at {field}: prequant="
                                f"{prequant_record_snapshot[field]}, internal="
                                f"{internal_record_snapshot[field]}"
                            )
                    print(
                        "MEGAMOE_PREQUANT_RECORD_CANONICAL "
                        + json.dumps(
                            {
                                "rank": dist.get_rank(),
                                "prequant": prequant_record_snapshot,
                                "internal_quant": internal_record_snapshot,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if self.stage1_mode in (
                    "internodev1_split128x2_rejoin",
                    "internodev1_tilepipe",
                ):
                    reference = {
                        field: canonical[field] for field in canonical_fields
                    }
                    for repeat in range(3):
                        dist.barrier()
                        (
                            repeat_tokens,
                            repeat_generation,
                            repeat_stream,
                        ) = self._begin_generation()
                        op._launch_stage1(
                            (
                                self._prequant_q
                                if self.prequant_input
                                else self.shared.x
                            ),
                            self.shared.route_weights,
                            self.shared.topk_ids,
                            repeat_tokens,
                            repeat_generation,
                            repeat_stream,
                            input_scale=(
                                self._prequant_scale
                                if self.prequant_input
                                else None
                            ),
                        )
                        torch.cuda.synchronize(self.device)
                        repeat_state = op.debug_stage1_comm_snapshot()
                        if (
                            repeat_state["stage1_error_count"] != 0
                            or repeat_state["expert_count_sum"] != 2048
                            or repeat_state["tile_arrived_sum"] != 2048
                        ):
                            raise AssertionError(
                                "rejoin repeat Stage1 protocol is invalid"
                            )
                        repeat_heavy = (
                            MegaMoETileA4W4.debug_direct_tile_snapshot(op)
                        )
                        repeat_canonical = repeat_heavy["canonical_h1"]
                        current = {
                            field: repeat_canonical[field]
                            for field in canonical_fields
                        }
                        if current != reference:
                            raise AssertionError(
                                f"rejoin canonical drift at repeat {repeat}: "
                                f"{current} != {reference}"
                            )
                        if (
                            repeat_canonical.get(
                                "fused_vs_standalone_changed_rows", 0
                            )
                            != 0
                        ):
                            raise AssertionError(
                                "rejoin repeat differs from standalone replay"
                            )
                        print(
                            "MEGAMOE_STAGE1_CANONICAL "
                            + json.dumps(
                                {
                                    "rank": dist.get_rank(),
                                    "generation": repeat_generation,
                                    "stage1_mode": self.stage1_mode,
                                    **current,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            return self.shared.x
        # Debug-only stage boundary: every rank must have completed and
        # snapshotted Stage1 before any rank can enter Stage2.
        dist.barrier()
        print(
            f"MEGAMOE_COMM_PROBE_PRIME rank={dist.get_rank()} "
            "phase=stage2_launch",
            flush=True,
        )
        op._launch_stage2(run_tokens, generation, stream)
        torch.cuda.synchronize(self.device)
        print(
            f"MEGAMOE_COMM_PROBE_PRIME rank={dist.get_rank()} "
            "phase=stage2_done",
            flush=True,
        )
        # All peer LSA writers must have completed before host-side counters
        # are inspected, particularly in atomic-only mode.
        dist.barrier()
        output = (
            self.shared.x
            if self.stage2_mode == "atomic_only"
            else op._output[:run_tokens]
        )
        if tuple(output.shape) != (self.shape.tokens, self.shape.hidden):
            raise AssertionError("comm probe output shape mismatch")
        if self.stage2_mode != "atomic_only" and not torch.isfinite(
            output.float()
        ).all().item():
            raise AssertionError("comm probe produced a non-finite output")
        if self.stage2_mode == "return_only":
            snapshot = self.operator.debug_stage2_return_snapshot()
            if snapshot["final_done"] != snapshot["final_expected"]:
                raise AssertionError("return-only final combine did not complete")
            if (
                snapshot["return_groups_ready"]
                != snapshot["return_groups_expected"]
            ):
                raise AssertionError("return-only receive groups are incomplete")
            if snapshot["return_consumed"] != generation:
                raise AssertionError("return-only remote payload was not credited")
            if snapshot["node_ready_count"] != snapshot["node_ready_expected"]:
                raise AssertionError("return-only synthetic readiness is incomplete")
            if (
                snapshot["node_done_expected_count"]
                != snapshot["node_ready_expected"]
            ):
                raise AssertionError("return-only synthetic node_done is incomplete")
            if snapshot["stage2_error_count"] != 0:
                raise AssertionError("return-only Stage2 reported a protocol error")
        else:
            snapshot = self.operator.debug_direct_tile_snapshot()
            _validate_direct_tile_debug_snapshot(snapshot)
            if self.stage2_mode == "atomic_only":
                if (
                    snapshot["node_expected_done_mismatch"] != 0
                    or snapshot["node_not_ready"] != 0
                ):
                    raise AssertionError("atomic-only node scoreboard is incomplete")
                if snapshot["final_done"] != 0:
                    raise AssertionError("atomic-only unexpectedly ran final combine")
                if snapshot["return_groups_ready"] != 0:
                    raise AssertionError("atomic-only unexpectedly ran CCO return")
                if snapshot["return_consumed"] == generation:
                    raise AssertionError("atomic-only unexpectedly returned credit")
        print(
            "MEGAMOE_COMM_PROBE_SNAPSHOT "
            + json.dumps(
                {"rank": dist.get_rank(), "generation": self.operator._generation,
                 "state": snapshot},
                sort_keys=True,
            ),
            flush=True,
        )
        return output

    def close(self) -> None:
        self.operator.close()


def _boundary_stats(result, tail_iterations: int, fields: tuple[str, ...]):
    if len(fields) == 1:
        tail = result.rank_max_samples[-tail_iterations:]
        return _sample_stats([sample.stage_us[fields[0]] for sample in tail])

    # A composite boundary must be reduced after summing stages on each rank.
    # Summing independently reduced stage maxima can combine the slowest stage
    # from different ranks and systematically overstate the critical path.
    local = torch.tensor(
        [
            sum(sample.stage_us[field] for field in fields)
            for sample in result.local_samples
        ],
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return _sample_stats(local.tolist()[-tail_iterations:])


def _ratio(numerator: dict[str, float], denominator: dict[str, float]):
    return {
        key: numerator[key] / denominator[key]
        for key in ("mean", "p50", "p95")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", choices=("mori", "probe", "both"), default="both"
    )
    parser.add_argument(
        "--probe-stage", choices=("stage1", "stage2", "both"), default="both"
    )
    parser.add_argument(
        "--stage1-mode",
        choices=(
            "legacy",
            "internodev1_split128x2",
            "internodev1_split128x2_no_arrival_rmw",
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--stage1-phase",
        choices=(
            "full",
            "quant_core_only",
            "quant_pack_only",
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ),
        default="full",
    )
    parser.add_argument(
        "--stage2-mode",
        choices=("full", "atomic_only", "return_only"),
        default="full",
    )
    parser.add_argument("--stage2-workers", type=int, default=0)
    parser.add_argument(
        "--cco-chunks-per-flush",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
    )
    parser.add_argument(
        "--cco-geometry",
        choices=("chunked", "mori64x2", "sparse_wqe"),
        default="chunked",
    )
    parser.add_argument("--quant-two-cta-per-token", action="store_true")
    parser.add_argument("--prequant-input", action="store_true")
    parser.add_argument("--tile-pipeline-instrument", action="store_true")
    parser.add_argument(
        "--tile-pipeline-fanout-shards",
        type=int,
        choices=(8, 12, 16),
        default=16,
    )
    parser.add_argument("--canonical-stage1", action="store_true")
    parser.add_argument(
        "--route-pattern",
        choices=(
            "rank-balanced-hot",
            "paired-rank-half-remote",
            "paired-rank-local-only",
            "paired-rank-remote-prefix",
            "single-rank-max",
            "single-expert-max",
            "node-route-count",
            "local-only-padded",
        ),
        default="rank-balanced-hot",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--tail-iters", type=int, default=3)
    parser.add_argument("--remote-token-count", type=int, default=64)
    parser.add_argument("--hot-rank", type=int, default=0)
    parser.add_argument("--local-route-count", type=int, default=8)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters >= 1")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1, iters]")
    if not 0 <= args.remote_token_count <= 128:
        raise ValueError("remote-token-count must be in [0, 128]")
    if not 0 <= args.hot_rank < 16:
        raise ValueError("hot-rank must be in [0, 16)")
    if not 0 <= args.local_route_count <= 16:
        raise ValueError("local-route-count must be in [0, 16]")
    if args.prequant_input:
        if args.paths != "probe" or args.probe_stage != "stage1":
            raise ValueError("prequant input requires --paths probe --probe-stage stage1")
        if args.stage1_phase != "full":
            raise ValueError("prequant input currently requires --stage1-phase full")
    if (
        args.tile_pipeline_instrument
        and args.stage1_mode != "internodev1_tilepipe"
    ):
        raise ValueError(
            "tile pipeline instrumentation requires internodev1_tilepipe mode"
        )
    if (
        args.stage1_mode == "internodev1_tilepipe"
        and args.cco_geometry != "sparse_wqe"
    ):
        raise ValueError("internodev1_tilepipe requires --cco-geometry sparse_wqe")
    if args.route_pattern != "rank-balanced-hot" and args.paths != "probe":
        raise ValueError("non-balanced routes currently require --paths probe")
    if (
        args.route_pattern == "local-only-padded"
        and args.cco_geometry != "sparse_wqe"
    ):
        raise ValueError(
            "local-only-padded currently requires --cco-geometry sparse_wqe"
        )
    if (
        args.route_pattern.startswith("paired-rank-")
        or args.route_pattern in ("single-rank-max", "single-expert-max")
        or args.route_pattern == "node-route-count"
    ) and not (
        args.stage1_mode.startswith("internodev1_split")
        or args.stage1_mode == "internodev1_tilepipe"
    ):
        raise ValueError(
            "paired-rank routes currently require split fanout"
        )

    shape = BenchmarkShape(
        tokens=128,
        hidden=7168,
        inter=3072,
        experts=896,
        topk=16,
        ep_size=16,
        gpus_per_node=8,
        activation="silu",
    )
    shape.validate()
    needs_mori = args.paths in ("mori", "both")
    needs_stage1_weights = (
        args.stage1_mode
        in (
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64_rejoin",
        )
    )
    rank, world, _local_rank, device = _setup_dist(needs_mori=needs_mori)
    if world != 16:
        raise ValueError(f"comm probe requires EP16, got world={world}")
    print(f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=inputs_begin", flush=True)
    shared = _lightweight_shared_inputs(
        shape,
        rank,
        world,
        device,
        quantize_for_mori=needs_mori,
        prepare_stage1_weights=needs_stage1_weights,
        route_pattern=args.route_pattern,
        remote_token_count=args.remote_token_count,
        hot_rank=args.hot_rank,
        local_route_count=args.local_route_count,
    )
    if needs_mori:
        _validate_rank_balanced_routes(shared, shape)
    print(f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=inputs_ready", flush=True)

    mori_path = None
    if needs_mori:
        mori_path = SplitMoriCommunicationPath(shape, shared, rank, world)
        print(f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=mori_ready", flush=True)
    weights = shared.prepared_weights
    probe_op = None
    probe_path = None
    stage2_workers = args.stage2_workers or (
        32 if args.stage2_mode == "atomic_only" else 160
    )
    if args.paths in ("probe", "both"):
        probe_op = MegaMoETileA4W4CommProbe(
            rank=rank,
            world_size=world,
            model_dim=shape.hidden,
            inter_dim=shape.inter,
            experts=shape.experts,
            topk=shape.topk,
            quant="a4w4",
            w1=weights.w1,
            w1_scale=weights.w1_scale,
            w2=weights.w2,
            w2_scale=weights.w2_scale,
            max_tok_per_rank=shape.tokens,
            mega_scheme="hierarchical",
            swiglu_limit=0.0,
            probe_stage=args.probe_stage,
            stage1_mode=args.stage1_mode,
            stage1_cco_chunks_per_flush=args.cco_chunks_per_flush,
            stage1_cco_geometry=args.cco_geometry,
            stage1_quant_two_cta_per_token=args.quant_two_cta_per_token,
            stage1_prequant_input=args.prequant_input,
            stage1_tile_pipeline_instrument=args.tile_pipeline_instrument,
            stage1_tile_pipeline_fanout_shards=(
                args.tile_pipeline_fanout_shards
            ),
            stage1_phase=args.stage1_phase,
            stage2_mode=args.stage2_mode,
            stage2_worker_blocks=stage2_workers,
        )
        print(f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=cco_probe_ready", flush=True)
        expected_symbol_suffix = (
            STAGE1_FULL_GMM1_SUFFIX
            if getattr(probe_op._stage1, "gemm1_contraction", False)
            else STAGE1_PROBE_SUFFIX
        )
        if expected_symbol_suffix not in probe_op.stage1_kernel_name:
            raise AssertionError(
                "Stage1 probe did not receive its compute-mode-specific symbol"
            )
        if STAGE2_PROBE_SUFFIX not in probe_op.stage2_kernel_name:
            raise AssertionError("Stage2 comm probe did not receive an independent symbol")
        expected_gemm1 = (
            args.stage1_mode
            in (
                "internodev1_split128x2_rejoin",
                "internodev1_tilepipe",
                "internodev1_wave64_rejoin",
            )
        )
        if (
            bool(getattr(probe_op._stage1, "gemm1_contraction", False))
            != expected_gemm1
        ):
            raise AssertionError("Stage1 GMM1 contraction mode mismatch")
        expected_tile_pipeline = (
            args.stage1_mode == "internodev1_tilepipe"
        )
        if (
            bool(getattr(probe_op._stage1, "tile_pipeline", False))
            != expected_tile_pipeline
        ):
            raise AssertionError("Stage1 tile pipeline mode mismatch")
        if expected_tile_pipeline:
            architecture = probe_op._stage1.architecture_contract
            expected_architecture = {
                "early_full_tile_enqueue": True,
                "queue_publication": (
                    "full_tile_last_arrival_plus_partial_tile_post_8_role_eos"
                ),
                "gmm_scheduler": (
                    "concurrent_ready_queue_8_shards_256_all_roles_rejoin"
                ),
                "tile_pipeline_fanout_shards": (
                    args.tile_pipeline_fanout_shards
                ),
            }
            mismatch = {
                name: (architecture.get(name), value)
                for name, value in expected_architecture.items()
                if architecture.get(name) != value
            }
            if mismatch:
                raise AssertionError(
                    f"Stage1 tile pipeline architecture mismatch: {mismatch}"
                )
        if (
            bool(getattr(probe_op._stage1, "full_stage1_fusion", False))
            != expected_gemm1
        ):
            raise AssertionError("Stage1 full-fusion contract mismatch")
        if getattr(probe_op._stage2, "gemm2_contraction", True):
            raise AssertionError("Stage2 communication probe still contains GMM2")
        probe_path = FusedCommunicationProbePath(
            probe_op,
            shared,
            shape,
            device,
            probe_stage=args.probe_stage,
            stage1_mode=args.stage1_mode,
            stage1_phase=args.stage1_phase,
            cco_chunks_per_flush=args.cco_chunks_per_flush,
            cco_geometry=args.cco_geometry,
            quant_two_cta_per_token=args.quant_two_cta_per_token,
            prequant_input=args.prequant_input,
            stage2_mode=args.stage2_mode,
            canonical_stage1=args.canonical_stage1,
            route_pattern=args.route_pattern,
        )

    paths = tuple(
        path for path in (mori_path, probe_path) if path is not None
    )
    results = {}
    try:
        for path in paths:
            print(
                f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=run_begin path={path.name}",
                flush=True,
            )
            results[path.name] = _run_path(
                path,
                device,
                warmup=args.warmup,
                iterations=args.iters,
            )
            if rank == 0:
                print(
                    "MEGAMOE_EP16_COMM_PATH_RESULT "
                    + json.dumps(
                        _path_summary(path, results[path.name], args.tail_iters),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            print(
                f"MEGAMOE_COMM_PROBE_PHASE rank={rank} phase=run_done path={path.name}",
                flush=True,
            )

        comparison = {}
        if mori_path is not None:
            mori = results[mori_path.name]
            comparison.update(
                {
                    "mori_dispatch_us": _boundary_stats(
                        mori, args.tail_iters, ("dispatch",)
                    ),
                    "mori_dispatch_plus_local_prepare_us": _boundary_stats(
                        mori,
                        args.tail_iters,
                        ("dispatch", "local_prepare"),
                    ),
                    "mori_quant_plus_dispatch_us": _boundary_stats(
                        mori, args.tail_iters, ("bf16_to_a4", "dispatch")
                    ),
                    "mori_quant_dispatch_plus_local_prepare_us": _boundary_stats(
                        mori,
                        args.tail_iters,
                        ("bf16_to_a4", "dispatch", "local_prepare"),
                    ),
                    "mori_combine_us": _boundary_stats(
                        mori, args.tail_iters, ("combine",)
                    ),
                }
            )
        if probe_path is not None:
            probe = results[probe_path.name]
            if args.probe_stage in ("stage1", "both"):
                if args.prequant_input:
                    comparison["prequant_hip_us"] = _boundary_stats(
                        probe,
                        args.tail_iters,
                        ("prequant_bf16_to_a4_hip",),
                    )
                    prequant_stage1_key = (
                        "prequant_stage1_full_gmm1_silu_requant_us"
                        if probe_path.real_gmm1
                        else "prequant_stage1_only_us"
                    )
                    comparison[prequant_stage1_key] = _boundary_stats(
                        probe,
                        args.tail_iters,
                        (probe_path.stage_names[1],),
                    )
                    comparison["prequant_stage_sum_us"] = _boundary_stats(
                        probe,
                        args.tail_iters,
                        (
                            "prequant_bf16_to_a4_hip",
                            "stage1_prequant_hier_dispatch_scoreboard",
                        ),
                    )
                    output_ready = _sample_stats(
                        [
                            sample.gpu_e2e_us
                            for sample in probe.rank_max_samples[
                                -args.tail_iters :
                            ]
                        ]
                    )
                    comparison[
                        "prequant_output_ready_gpu_e2e_us"
                    ] = output_ready
                    # Compatibility alias for logs produced before the timing
                    # boundary was named explicitly.
                    comparison["prequant_total_gpu_e2e_us"] = output_ready
                else:
                    stage1_key = (
                        "fused_stage1_full_gmm1_silu_requant_us"
                        if (
                            args.stage1_phase == "full"
                            and probe_path.real_gmm1
                        )
                        else (
                            "fused_stage1_quant_dispatch_scoreboard_us"
                            if args.stage1_phase == "full"
                            else f"fused_stage1_{args.stage1_phase}_us"
                        )
                    )
                    comparison[stage1_key] = _boundary_stats(
                        probe,
                        args.tail_iters,
                        (probe_path.stage_names[0],),
                    )
            if args.probe_stage in ("stage2", "both"):
                comparison[
                    "fused_stage2_direct_atomic_return_combine_us"
                ] = _boundary_stats(
                    probe,
                    args.tail_iters,
                    ("stage2_direct_atomic_return_combine",),
                )
        if mori_path is not None and probe_path is not None:
            if (
                args.probe_stage in ("stage1", "both")
                and args.stage1_phase == "full"
                and not probe_path.real_gmm1
            ):
                comparison.update(
                    {
                        "fused_stage1_over_mori_quant_dispatch": _ratio(
                            comparison[
                                "fused_stage1_quant_dispatch_scoreboard_us"
                            ],
                            comparison["mori_quant_plus_dispatch_us"],
                        ),
                        "fused_stage1_over_mori_receiver_ready": _ratio(
                            comparison[
                                "fused_stage1_quant_dispatch_scoreboard_us"
                            ],
                            comparison[
                                "mori_quant_dispatch_plus_local_prepare_us"
                            ],
                        ),
                    }
                )
            elif (
                args.probe_stage in ("stage1", "both")
                and args.stage1_phase == "dispatch_only"
            ):
                comparison[
                    "diagnostic_fused_post_record_over_"
                    "mori_public_dispatch_plus_local_prepare"
                ] = _ratio(
                    comparison["fused_stage1_dispatch_only_us"],
                    comparison["mori_dispatch_plus_local_prepare_us"],
                )
            if args.probe_stage in ("stage2", "both"):
                comparison["fused_stage2_over_mori_combine"] = _ratio(
                    comparison[
                        "fused_stage2_direct_atomic_return_combine_us"
                    ],
                    comparison["mori_combine_us"],
                )

        summaries = [
            _path_summary(path, results[path.name], args.tail_iters)
            for path in paths
        ]
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(
            {"rank": rank, "summaries": summaries, "comparison": comparison},
            gathered,
            dst=0,
        )
        if rank == 0:
            all_rank_tail_mean_us = {}
            for path_index, path in enumerate(paths):
                local_stats = [
                    row["summaries"][path_index]["tail_local_rank_stats_us"]
                    for row in gathered
                ]
                all_rank_tail_mean_us[path.name] = {
                    field: sum(
                        float(stats[field]["mean"])
                        for stats in local_stats
                    )
                    / world
                    for field in local_stats[0]
                }
            print(
                "MEGAMOE_EP16_COMM_ONLY_BENCH "
                + json.dumps(
                    {
                        "shape": shape.__dict__,
                        "paths": args.paths,
                        "probe_stage": args.probe_stage,
                        "stage1_mode": args.stage1_mode,
                        "stage1_phase": args.stage1_phase,
                        "cco_chunks_per_flush": args.cco_chunks_per_flush,
                        "cco_geometry": args.cco_geometry,
                        "quant_two_cta_per_token": args.quant_two_cta_per_token,
                        "prequant_input": args.prequant_input,
                        "tile_pipeline_instrument": (
                            args.tile_pipeline_instrument
                        ),
                        "tile_pipeline_fanout_shards": (
                            args.tile_pipeline_fanout_shards
                        ),
                        "stage2_mode": args.stage2_mode,
                        "stage2_worker_blocks": stage2_workers,
                        "canonical_stage1": args.canonical_stage1,
                        "route_pattern": args.route_pattern,
                        "remote_token_count": args.remote_token_count,
                        "hot_rank": args.hot_rank,
                        "local_route_count": args.local_route_count,
                        "dispatch_only_comparison_contract": (
                            {
                                "candidate_start": "packed_4096B_record_ready",
                                "candidate_end": "gmm1_consumable_grouped_layout_ready",
                                "candidate_wire_bytes_per_token": 4096,
                                "mori_start": "public_dispatch_including_copy_to_staging",
                                "mori_end": "minimal_local_prepare_complete",
                                "mori_wire_bytes_per_token": (
                                    shape.hidden // 2
                                    + shape.hidden // 32
                                    + shape.topk * 8
                                    + 4
                                ),
                                "strictly_equivalent_start_boundary": False,
                            }
                            if args.stage1_phase == "dispatch_only"
                            else None
                        ),
                        "warmup": args.warmup,
                        "iterations": args.iters,
                        "tail_iterations": args.tail_iters,
                        "stage1_probe_contract": (
                            probe_op.stage1_contains if probe_op is not None else None
                        ),
                        "stage1_has_real_gmm1": (
                            bool(
                                getattr(
                                    probe_op._stage1,
                                    "gemm1_contraction",
                                    False,
                                )
                            )
                            if probe_op is not None
                            else None
                        ),
                        "stage1_fusion_contract": (
                            probe_op._stage1.architecture_contract
                            if probe_op is not None
                            else None
                        ),
                        "stage2_probe_contract": (
                            probe_op.stage2_contains if probe_op is not None else None
                        ),
                        "stage1_symbol": (
                            probe_op.stage1_kernel_name if probe_op is not None else None
                        ),
                        "stage2_symbol": (
                            probe_op.stage2_kernel_name if probe_op is not None else None
                        ),
                        "rank0_rank_max_comparison": comparison,
                        "by_rank": gathered,
                        "tail_all_rank_sample_mean_us": (
                            all_rank_tail_mean_us
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if probe_path is not None:
            probe_path.close()
        if needs_mori:
            import mori.shmem as ms

            ms.shmem_finalize()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
