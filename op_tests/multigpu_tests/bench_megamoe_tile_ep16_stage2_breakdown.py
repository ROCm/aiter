# SPDX-License-Identifier: MIT
"""Isolate the EP16 MegaMoE Stage-2 cost at the production K3 shape.

Candidate modes always run the production fused Stage-1 first, synchronize all
16 ranks outside timing, and then time one Stage-2 launch.  This gives every
mode the same real H1 A4/scale/route metadata while allowing the Stage-2 body
to be compiled with selected roles disabled::

    init_only    common waits, clears, validation and grid barriers
    atomic_only  init + zero-valued direct-LSA atomic/scoreboard epilogue
    gmm2_only    init + real GMM2 + a per-work-item checksum sink
    gmm2_atomic_only
                 init + real GMM2 + direct-LSA atomic/scoreboard epilogue
    return_only  init + synthetic-ready RAIL return + final combine
    full         production GMM2/atomic/return/final-combine launch

The MORI reference path uses ``fused_moe.kernel_bench_callable`` to capture
the exact selected GMM2 launcher.  Dispatch plus fused_moe preparation runs
outside timing on every iteration; the captured output is cleared, then the
benchmark measures either GMM2 alone or GMM2 immediately followed by the
matching MORI combine.  A combine is still issued outside timing for the
GMM2-only case so MORI's dispatch buffers complete their normal lifecycle.

With ``--device-timeline``, the candidate runs a real Stage1 on every
iteration without a host barrier between Stage1 and Stage2. Compile-time
``s_memrealtime`` markers then report the device-entry-to-first-Stage2-return
doorbell interval and its internal gates. Marker reads happen after the timed
GPU work and are not included in the HIP-event interval.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.megamoe_tile import MegaMoETileA4W4
from aiter.ops.flydsl.kernels.megamoe_tile.markers import (
    profiler_pause,
    profiler_resume,
    roctx_range,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import (
    validate_public_stage1_contract,
)
from aiter.ops.flydsl.kernels.megamoe_tile.stage2 import (
    compile_megamoe_tile_ep16_stage2_a4w4,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    IterationTiming,
    _global_max_timing,
    _sample_stats,
    _setup_dist,
    _shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    COMPARISON_CASE_LABEL,
    MoriFusedMoeBaselinePath,
    _validate_direct_tile_debug_snapshot,
)


CANDIDATE_MODES = (
    "full",
    "init_only",
    "atomic_only",
    "gmm2_only",
    "gmm2_atomic_only",
    "route_store_only",
    "return_only",
)
MORI_MODES = ("gmm2_only", "gmm2_combine")
TIMELINE_INTERVALS = {
    "s1_entry_to_s1_dispatch_flush_pre": (
        "stage1_entry",
        "stage1_dispatch_flush_pre",
    ),
    "s1_dispatch_flush_call": (
        "stage1_dispatch_flush_pre",
        "stage1_dispatch_flush_post",
    ),
    "s1_entry_to_s1_dispatch_flush_post": (
        "stage1_entry",
        "stage1_dispatch_flush_post",
    ),
    "s1_entry_to_s1_done": ("stage1_entry", "stage1_done_publish"),
    "s1_done_to_s2_entry": ("stage1_done_publish", "stage2_entry"),
    "s2_entry_to_stage1_gate": ("stage2_entry", "stage2_stage1_gate_done"),
    "s2_stage1_gate_to_init_gate": (
        "stage2_stage1_gate_done",
        "stage2_init_gate_done",
    ),
    "s2_init_gate_to_qp0_tokens_ready": (
        "stage2_init_gate_done",
        "stage2_qp0_tokens_ready",
    ),
    "s2_init_gate_to_qp1_tokens_ready": (
        "stage2_init_gate_done",
        "stage2_qp1_tokens_ready",
    ),
    "s2_init_gate_to_qp2_tokens_ready": (
        "stage2_init_gate_done",
        "stage2_qp2_tokens_ready",
    ),
    "s2_init_gate_to_qp3_tokens_ready": (
        "stage2_init_gate_done",
        "stage2_qp3_tokens_ready",
    ),
    "s2_qp0_tokens_to_first_batch_ready": (
        "stage2_qp0_tokens_ready",
        "stage2_first_batch_ready",
    ),
    "s2_init_gate_to_first_batch_ready": (
        "stage2_init_gate_done",
        "stage2_first_batch_ready",
    ),
    "s2_first_batch_ready_to_qp0_payload_posted": (
        "stage2_first_batch_ready",
        "stage2_qp0_payload_posted",
    ),
    "s2_qp0_tokens_ready_to_payload_posted": (
        "stage2_qp0_tokens_ready",
        "stage2_qp0_payload_posted",
    ),
    "s2_first_batch_ready_to_qp1_payload_posted": (
        "stage2_first_batch_ready",
        "stage2_qp1_payload_posted",
    ),
    "s2_first_batch_ready_to_qp2_payload_posted": (
        "stage2_first_batch_ready",
        "stage2_qp2_payload_posted",
    ),
    "s2_first_batch_ready_to_qp3_payload_posted": (
        "stage2_first_batch_ready",
        "stage2_qp3_payload_posted",
    ),
    "s2_qp0_payload_to_all_payloads_posted": (
        "stage2_qp0_payload_posted",
        "stage2_first_batch_payloads_posted",
    ),
    "s2_all_payloads_to_return_flush_pre": (
        "stage2_first_batch_payloads_posted",
        "stage2_return_flush_pre",
    ),
    "s2_qp0_payload_to_return_flush_pre": (
        "stage2_qp0_payload_posted",
        "stage2_return_flush_pre",
    ),
    "s2_first_batch_ready_to_return_flush_pre": (
        "stage2_first_batch_ready",
        "stage2_return_flush_pre",
    ),
    "s2_return_flush_call": (
        "stage2_return_flush_pre",
        "stage2_return_flush_post",
    ),
    "s2_return_doorbell_to_request_done": (
        "stage2_return_flush_post",
        "stage2_return_request_done",
    ),
    "s1_done_to_s2_return_doorbell_lower": (
        "stage1_done_publish",
        "stage2_return_flush_pre",
    ),
    "s1_done_to_s2_return_doorbell_upper": (
        "stage1_done_publish",
        "stage2_return_flush_post",
    ),
    "s2_entry_to_s2_return_doorbell_lower": (
        "stage2_entry",
        "stage2_return_flush_pre",
    ),
    "s2_entry_to_s2_return_doorbell_upper": (
        "stage2_entry",
        "stage2_return_flush_post",
    ),
    "s2_init_gate_to_s2_return_doorbell_lower": (
        "stage2_init_gate_done",
        "stage2_return_flush_pre",
    ),
    "s2_entry_to_first_gmm_worker_done": (
        "stage2_entry",
        "stage2_first_gmm_worker_done",
    ),
    "s2_entry_to_all_gmm_done": (
        "stage2_entry",
        "stage2_all_gmm_done",
    ),
    "s2_init_gate_to_all_gmm_done": (
        "stage2_init_gate_done",
        "stage2_all_gmm_done",
    ),
    "s1_entry_to_all_gmm_done": (
        "stage1_entry",
        "stage2_all_gmm_done",
    ),
    "s2_return_doorbell_to_all_gmm_done": (
        "stage2_return_flush_post",
        "stage2_all_gmm_done",
    ),
    "s1_entry_to_s2_return_doorbell_lower": (
        "stage1_entry",
        "stage2_return_flush_pre",
    ),
    "s1_entry_to_s2_return_doorbell_upper": (
        "stage1_entry",
        "stage2_return_flush_post",
    ),
}
SIGNED_TIMELINE_INTERVALS = {
    "s2_return_doorbell_to_all_gmm_done",
    # qp_prepost intentionally builds payload WQEs before the token-ready
    # acquire and rings their doorbells only afterwards.
    "s2_qp0_tokens_ready_to_payload_posted",
    "s2_first_batch_ready_to_qp0_payload_posted",
    "s2_first_batch_ready_to_qp1_payload_posted",
    "s2_first_batch_ready_to_qp2_payload_posted",
    "s2_first_batch_ready_to_qp3_payload_posted",
}
TIMELINE_PER_RANK_FIELDS = (
    "s1_entry_to_s2_return_doorbell_lower_us",
    "s1_entry_to_s2_return_doorbell_upper_us",
    "s1_entry_to_all_gmm_done_us",
    "s2_entry_to_all_gmm_done_us",
    "s2_return_doorbell_to_all_gmm_done_us",
)


class _Stage2Captured(RuntimeError):
    """Internal sentinel used to stop fused_moe immediately before GMM2."""


class _StopBeforeStage2(list):
    def append(self, item) -> None:
        super().append(item)
        if item[0] == "stage2":
            # fused_moe appends the fully bound callable immediately before it
            # invokes it.  Stop there so the isolated event is GMM2's first
            # execution for this freshly prepared intermediate.
            raise _Stage2Captured


def _shape(tokens: int) -> BenchmarkShape:
    shape = BenchmarkShape(
        tokens=int(tokens),
        hidden=7168,
        inter=3072,
        experts=896,
        topk=16,
        ep_size=16,
        gpus_per_node=8,
        activation="silu",
    )
    shape.validate()
    return shape


def _validate_node_reduce_rejoin_config(
    *,
    diagnostic_mode: str,
    worker_blocks: int,
    final_combine_blocks: int,
    gmm_schedule: str,
    node_accumulation_mode: str,
    node_reduce_blocks: int,
    node_reduce_work_schedule: str,
    node_reduce_rejoin_blocks: int,
    rail_return_schedule: str,
) -> None:
    if node_reduce_work_schedule not in ("static_strided", "dynamic_head"):
        raise ValueError(
            "stage2_node_reduce_work_schedule must be static_strided or "
            "dynamic_head"
        )
    rejoin_blocks = int(node_reduce_rejoin_blocks)
    if rejoin_blocks not in (0, 8, 16, 32):
        raise ValueError(
            "stage2_node_reduce_rejoin_blocks must be one of 0,8,16,32"
        )
    if (
        node_reduce_work_schedule == "dynamic_head"
        and node_accumulation_mode != "rank_local"
    ):
        raise ValueError("dynamic_head node reduction requires rank_local")
    if rejoin_blocks == 0:
        return
    if diagnostic_mode not in (
        "full",
        "atomic_only",
        "gmm2_atomic_only",
    ):
        raise ValueError(
            "node-reduce GMM rejoin requires a queue-producing, "
            "reducer-enabled diagnostic mode (full, atomic_only, or "
            "gmm2_atomic_only)"
        )
    if (
        node_accumulation_mode != "rank_local"
        or rail_return_schedule != "compact"
        or gmm_schedule != "persistent_queue"
        or node_reduce_work_schedule != "dynamic_head"
    ):
        raise ValueError(
            "node-reduce GMM rejoin requires rank_local accumulation, compact "
            "return, persistent_queue GMM, and dynamic_head reduction"
        )
    gmm_cta_count = int(worker_blocks) - (
        1 + int(node_reduce_blocks) + int(final_combine_blocks)
    )
    if rejoin_blocks > gmm_cta_count:
        raise ValueError(
            "stage2_node_reduce_rejoin_blocks exceeds the available GMM2 CTA "
            f"count ({gmm_cta_count})"
        )


class Stage2ModeOperator(MegaMoETileA4W4):
    """Production operator with only the Stage-2 compile-time mode changed."""

    diagnostic_only = True

    def __init__(
        self,
        *args,
        stage2_mode: str,
        stage2_worker_blocks: int = 160,
        stage2_accumulator_dtype: str = "bf16",
        stage2_final_combine_blocks: int = 14,
        stage2_gmm_schedule: str = "persistent_queue",
        stage2_return_chunk_tokens: int = 8,
        stage2_bf16_atomic_kind: str = "buffer",
        stage2_node_accumulation_mode: str = "direct_atomic",
        stage2_node_reduce_blocks: int = 32,
        stage2_node_reduce_vec_bytes: int = 4,
        stage2_node_reduce_schedule: str = "token",
        stage2_node_reduce_load_schedule: str = "interleaved",
        stage2_node_reduce_work_schedule: str = "static_strided",
        stage2_node_reduce_rejoin_blocks: int = 0,
        stage2_rank_epilogue_lds_addressing: str = "expanded",
        stage2_rank_accumulation_mode: str = "atomic",
        stage2_rail_return_schedule: str = "lockstep",
        stage2_rail_quant_type: str = "none",
        stage2_gmm_work_swizzle: str = "token_major",
        stage2_window_n_groups: int = 2,
        stage2_ready_granularity: str = "token",
        stage2_epilogue_schedule: str = "lane32_meta",
        stage2_n_tile_group: int = 2,
        stage2_group_pipeline_schedule: str = "a_double_buffer",
        stage2_scoreboard_schedule: str = "wave0",
        stage2_atomic_issue_schedule: str = "interleaved",
        stage2_waves_per_eu_hint: int = 2,
        stage2_timeline_instrument: bool = False,
        stage1_diagnostic_phase: str = "full",
        **kwargs,
    ):
        if stage2_mode not in CANDIDATE_MODES:
            raise ValueError(f"unsupported Stage2 mode {stage2_mode!r}")
        if not 8 <= int(stage2_worker_blocks) <= 256:
            raise ValueError("stage2_worker_blocks must be in [8, 256]")
        if stage2_accumulator_dtype not in ("fp32", "bf16"):
            raise ValueError("stage2_accumulator_dtype must be fp32 or bf16")
        if not 1 <= int(stage2_final_combine_blocks) <= 56:
            raise ValueError("stage2_final_combine_blocks must be in [1,56]")
        if stage2_gmm_schedule not in ("persistent_queue", "static_strided"):
            raise ValueError("unsupported Stage2 GMM schedule")
        if int(stage2_return_chunk_tokens) not in (4, 8, 16):
            raise ValueError("stage2_return_chunk_tokens must be 4, 8, or 16")
        if stage2_bf16_atomic_kind not in ("buffer", "global_system"):
            raise ValueError("unsupported BF16 atomic kind")
        if stage2_node_accumulation_mode not in (
            "direct_atomic",
            "route_store",
            "rank_local",
        ):
            raise ValueError("unsupported Stage2 node accumulation mode")
        if int(stage2_node_reduce_blocks) not in (8, 16, 32, 56):
            raise ValueError("stage2_node_reduce_blocks must be one of 8,16,32,56")
        if int(stage2_node_reduce_vec_bytes) not in (4, 8, 16):
            raise ValueError(
                "stage2_node_reduce_vec_bytes must be 4, 8, or 16"
            )
        if stage2_rank_epilogue_lds_addressing not in ("expanded", "dynamic_base"):
            raise ValueError(
                "stage2_rank_epilogue_lds_addressing must be expanded or dynamic_base"
            )
        if stage2_rank_epilogue_lds_addressing == "dynamic_base" and not (
            stage2_node_accumulation_mode == "rank_local"
            and int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and int(stage2_node_reduce_rejoin_blocks) == 0
        ):
            raise ValueError(
                "dynamic_base LDS addressing requires rank_local, vec8, "
                "load_first, static_strided reduction, and rejoin_blocks=0"
            )
        if stage2_rank_accumulation_mode not in ("atomic", "staged_reduce", "staged_ring"):
            raise ValueError(
                "stage2_rank_accumulation_mode must be atomic, staged_reduce, or staged_ring"
            )
        if stage2_rank_accumulation_mode == "staged_reduce" and not (
            stage2_node_accumulation_mode == "rank_local"
            and int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and int(stage2_node_reduce_rejoin_blocks) == 0
            and stage2_rank_epilogue_lds_addressing == "expanded"
            and int(stage2_n_tile_group) == 2
        ):
            raise ValueError(
                "staged_reduce requires rank_local vec8/load_first/static_strided "
                "reduction, expanded LDS addressing, n_tile_group=2, rejoin0"
            )
        if stage2_rank_accumulation_mode == "staged_ring" and not (
            stage2_node_accumulation_mode == "rank_local"
            and int(stage2_node_reduce_vec_bytes) == 8
            and stage2_node_reduce_load_schedule == "load_first"
            and stage2_node_reduce_work_schedule == "static_strided"
            and int(stage2_node_reduce_rejoin_blocks) == 0
            and stage2_n_tile_group == 2
        ):
            raise ValueError(
                "staged_ring requires rank_local vec8/load_first/static_strided "
                "reduction, n_tile_group=2, rejoin0"
            )
        if (
            int(stage2_node_reduce_vec_bytes) == 16
            and stage2_node_accumulation_mode != "rank_local"
        ):
            raise ValueError(
                "16-byte node reduction requires rank_local accumulation"
            )
        if stage2_node_reduce_schedule not in ("token", "group", "tile"):
            raise ValueError(
                "stage2_node_reduce_schedule must be token, group, or tile"
            )
        if stage2_node_reduce_load_schedule not in (
            "interleaved",
            "load_first",
        ):
            raise ValueError("unsupported Stage2 node-reduce load schedule")
        if stage2_node_accumulation_mode == "rank_local" and (
            stage2_node_reduce_schedule != "token"
        ):
            raise ValueError(
                "rank_local requires token node reduction"
            )
        if stage2_rail_return_schedule not in (
            "lockstep",
            "qp_independent",
            "qp_prepost",
            "compact",
        ):
            raise ValueError("unsupported Stage2 RAIL return schedule")
        _validate_node_reduce_rejoin_config(
            diagnostic_mode=stage2_mode,
            worker_blocks=stage2_worker_blocks,
            final_combine_blocks=stage2_final_combine_blocks,
            gmm_schedule=stage2_gmm_schedule,
            node_accumulation_mode=stage2_node_accumulation_mode,
            node_reduce_blocks=stage2_node_reduce_blocks,
            node_reduce_work_schedule=stage2_node_reduce_work_schedule,
            node_reduce_rejoin_blocks=stage2_node_reduce_rejoin_blocks,
            rail_return_schedule=stage2_rail_return_schedule,
        )
        if stage2_epilogue_schedule not in (
            "lane32",
            "lane32_meta",
            "lane32_meta_expected",
            "wave64_meta",
        ):
            raise ValueError("unsupported Stage2 epilogue schedule")
        if int(stage2_n_tile_group) not in (1, 2):
            raise ValueError("stage2_n_tile_group must be 1 or 2")
        if stage2_group_pipeline_schedule not in (
            "baseline",
            "expert_meta_hoist",
            "runtime_loop",
            "a_double_buffer",
        ):
            raise ValueError("unsupported Stage2 group pipeline schedule")
        if stage2_scoreboard_schedule not in ("wave0", "four_wave"):
            raise ValueError("unsupported Stage2 scoreboard schedule")
        if stage2_atomic_issue_schedule not in ("interleaved", "preload_pairs"):
            raise ValueError("unsupported Stage2 atomic issue schedule")
        if stage2_rail_quant_type not in ("none", "fp8_blockwise"):
            raise ValueError("unsupported Stage2 rail quant type")
        if stage2_gmm_work_swizzle not in ("token_major", "n_major_window"):
            raise ValueError("unsupported Stage2 GMM work swizzle")
        if int(stage2_window_n_groups) not in (1, 2, 4, 7, 14):
            raise ValueError("stage2_window_n_groups must be one of 1,2,4,7,14")
        if stage2_ready_granularity not in ("token", "tile"):
            raise ValueError("unsupported Stage2 ready granularity")
        if int(stage2_waves_per_eu_hint) not in (1, 2, 3, 4):
            raise ValueError("stage2_waves_per_eu_hint must be in [1,4]")
        self.stage2_mode = str(stage2_mode)
        self.stage2_worker_blocks = int(stage2_worker_blocks)
        self.stage2_accumulator_dtype = str(stage2_accumulator_dtype)
        self.stage2_final_combine_blocks = int(stage2_final_combine_blocks)
        self.stage2_gmm_schedule = str(stage2_gmm_schedule)
        self.stage2_return_chunk_tokens = int(stage2_return_chunk_tokens)
        self.stage2_bf16_atomic_kind = str(stage2_bf16_atomic_kind)
        self.stage2_node_accumulation_mode = str(stage2_node_accumulation_mode)
        self.stage2_node_reduce_blocks = int(stage2_node_reduce_blocks)
        self.stage2_node_reduce_vec_bytes = int(stage2_node_reduce_vec_bytes)
        self.stage2_node_reduce_schedule = str(stage2_node_reduce_schedule)
        self.stage2_node_reduce_load_schedule = str(
            stage2_node_reduce_load_schedule
        )
        self.stage2_node_reduce_work_schedule = str(
            stage2_node_reduce_work_schedule
        )
        self.stage2_node_reduce_rejoin_blocks = int(
            stage2_node_reduce_rejoin_blocks
        )
        self.stage2_rank_epilogue_lds_addressing = str(
            stage2_rank_epilogue_lds_addressing
        )
        self.stage2_rank_accumulation_mode = str(stage2_rank_accumulation_mode)
        self.stage2_rail_return_schedule = str(stage2_rail_return_schedule)
        self.stage2_rail_quant_type = str(stage2_rail_quant_type)
        self.stage2_gmm_work_swizzle = str(stage2_gmm_work_swizzle)
        self.stage2_window_n_groups = int(stage2_window_n_groups)
        self.stage2_ready_granularity = str(stage2_ready_granularity)
        self.stage2_epilogue_schedule = str(stage2_epilogue_schedule)
        self.stage2_n_tile_group = int(stage2_n_tile_group)
        self.stage2_group_pipeline_schedule = str(
            stage2_group_pipeline_schedule
        )
        self.stage2_scoreboard_schedule = str(stage2_scoreboard_schedule)
        self.stage2_atomic_issue_schedule = str(stage2_atomic_issue_schedule)
        self.stage2_waves_per_eu_hint = int(stage2_waves_per_eu_hint)
        self.stage1_diagnostic_phase = str(stage1_diagnostic_phase)
        self.timeline_instrument = bool(stage2_timeline_instrument)
        if self.timeline_instrument and self.stage2_mode != "full":
            raise ValueError("device timeline requires full Stage2 mode")
        if self.stage2_mode in (
            "full",
            "atomic_only",
            "gmm2_only",
            "gmm2_atomic_only",
            "route_store_only",
        ) and (
            self.stage2_worker_blocks <= 1 + self.stage2_final_combine_blocks
        ):
            raise ValueError("Stage2 mode requires at least one GMM2 worker CTA")
        if (
            self.stage2_node_accumulation_mode
            in ("route_store", "rank_local")
            and self.stage2_mode
            in (
                "full",
                "atomic_only",
                "gmm2_only",
                "gmm2_atomic_only",
                "route_store_only",
            )
            and self.stage2_worker_blocks
            <= 1
            + self.stage2_final_combine_blocks
            + self.stage2_node_reduce_blocks
        ):
            raise ValueError(
                "deferred node reduction requires at least one GMM2 worker CTA after "
                "RAIL, final-combine, and node-reduce roles"
            )
        reduce_role_blocks = (
            self.stage2_node_reduce_blocks
            if self.stage2_node_accumulation_mode
            in ("route_store", "rank_local")
            else 0
        )
        if self.stage2_mode == "return_only" and (
            self.stage2_worker_blocks
            < 1 + reduce_role_blocks + self.stage2_final_combine_blocks
        ):
            raise ValueError("return_only requires every communication-role CTA")
        kwargs["stage2_rail_quant_type"] = self.stage2_rail_quant_type
        kwargs["stage2_gmm_work_swizzle"] = self.stage2_gmm_work_swizzle
        kwargs["stage2_window_n_groups"] = self.stage2_window_n_groups
        kwargs["stage2_ready_granularity"] = self.stage2_ready_granularity
        super().__init__(*args, **kwargs)
        # The base class intentionally fixes production Stage2 at 160.  The
        # diagnostic launcher accepts the grid size as a runtime argument.
        self.worker_blocks = self.stage2_worker_blocks

    def _compile_stage2(self):
        launcher = compile_megamoe_tile_ep16_stage2_a4w4(
            self.layout,
            rank=self.rank,
            BM=32,
            BN=256,
            BK=256,
            WORK_SHARDS=8,
            waves_per_eu_hint=self.stage2_waves_per_eu_hint,
            team="rail",
            diagnostic_mode=self.stage2_mode,
            accumulator_dtype=self.stage2_accumulator_dtype,
            final_combine_blocks=self.stage2_final_combine_blocks,
            gmm_schedule=self.stage2_gmm_schedule,
            return_chunk_tokens=self.stage2_return_chunk_tokens,
            bf16_atomic_kind=self.stage2_bf16_atomic_kind,
            node_accumulation_mode=self.stage2_node_accumulation_mode,
            node_reduce_blocks=self.stage2_node_reduce_blocks,
            node_reduce_vec_bytes=self.stage2_node_reduce_vec_bytes,
            node_reduce_schedule=self.stage2_node_reduce_schedule,
            node_reduce_load_schedule=self.stage2_node_reduce_load_schedule,
            node_reduce_work_schedule=self.stage2_node_reduce_work_schedule,
            node_reduce_rejoin_blocks=self.stage2_node_reduce_rejoin_blocks,
            rank_epilogue_lds_addressing=self.stage2_rank_epilogue_lds_addressing,
            rank_accumulation_mode=self.stage2_rank_accumulation_mode,
            rail_return_schedule=self.stage2_rail_return_schedule,
            rail_quant_type=self.stage2_rail_quant_type,
            gmm_work_swizzle=self.stage2_gmm_work_swizzle,
            window_n_groups=self.stage2_window_n_groups,
            ready_granularity=self.stage2_ready_granularity,
            epilogue_schedule=self.stage2_epilogue_schedule,
            n_tile_group=self.stage2_n_tile_group,
            group_pipeline_schedule=self.stage2_group_pipeline_schedule,
            scoreboard_schedule=self.stage2_scoreboard_schedule,
            atomic_issue_schedule=self.stage2_atomic_issue_schedule,
            timeline_instrument=self.timeline_instrument,
        )
        if launcher.diagnostic_mode != self.stage2_mode:
            raise AssertionError("Stage2 diagnostic mode did not reach the compiler")
        expected_gmm2 = self.stage2_mode in (
            "full",
            "gmm2_only",
            "gmm2_atomic_only",
            "route_store_only",
        )
        if bool(launcher.gemm2_contraction) != expected_gmm2:
            raise AssertionError("Stage2 GMM2 contraction manifest mismatch")
        if launcher.node_accumulation_mode != self.stage2_node_accumulation_mode:
            raise AssertionError("Stage2 node accumulation mode manifest mismatch")
        if int(launcher.node_reduce_blocks) != self.stage2_node_reduce_blocks:
            raise AssertionError("Stage2 node-reduce CTA manifest mismatch")
        if int(launcher.node_reduce_vec_bytes) != self.stage2_node_reduce_vec_bytes:
            raise AssertionError("Stage2 node-reduce vector width mismatch")
        if launcher.node_reduce_schedule != self.stage2_node_reduce_schedule:
            raise AssertionError("Stage2 node-reduce schedule mismatch")
        if (
            launcher.node_reduce_load_schedule
            != self.stage2_node_reduce_load_schedule
        ):
            raise AssertionError("Stage2 node-reduce load schedule mismatch")
        if (
            launcher.node_reduce_work_schedule
            != self.stage2_node_reduce_work_schedule
        ):
            raise AssertionError("Stage2 node-reduce work schedule mismatch")
        if (
            int(launcher.node_reduce_rejoin_blocks)
            != self.stage2_node_reduce_rejoin_blocks
        ):
            raise AssertionError("Stage2 node-reduce rejoin count mismatch")
        if (
            launcher.rank_epilogue_lds_addressing
            != self.stage2_rank_epilogue_lds_addressing
        ):
            raise AssertionError("Stage2 rank epilogue LDS addressing mismatch")
        if launcher.rank_accumulation_mode != self.stage2_rank_accumulation_mode:
            raise AssertionError("Stage2 rank accumulation mode mismatch")
        architecture = getattr(launcher, "architecture_contract", {})
        expected_architecture = {
            "node_reduce_work_schedule": self.stage2_node_reduce_work_schedule,
            "node_reduce_rejoin_blocks": (
                self.stage2_node_reduce_rejoin_blocks
            ),
            "rank_epilogue_lds_addressing": self.stage2_rank_epilogue_lds_addressing,
            "rank_accumulation_mode": self.stage2_rank_accumulation_mode,
        }
        architecture_mismatch = {
            name: (architecture.get(name, "<missing>"), value)
            for name, value in expected_architecture.items()
            if architecture.get(name, "<missing>") != value
        }
        if architecture_mismatch:
            raise AssertionError(
                "Stage2 node-reduce architecture manifest mismatch: "
                f"{architecture_mismatch}"
            )
        return launcher


class CandidateStage2Path:
    def __init__(self, operator, shared, shape, device, mode: str):
        self.operator = operator
        self.shared = shared
        self.shape = shape
        self.device = device
        self.mode = mode
        self.name = (
            f"candidate_stage2_isolated_{mode}_"
            f"acc{operator.stage2_accumulator_dtype}_"
            f"fc{operator.stage2_final_combine_blocks}_"
            f"gs{operator.stage2_gmm_schedule}_"
            f"rt{operator.stage2_return_chunk_tokens}_"
            "nrtoken_"
            f"ba{operator.stage2_bf16_atomic_kind}"
            f"_na{operator.stage2_node_accumulation_mode}"
            f"_nr{operator.stage2_node_reduce_blocks}"
            f"v{operator.stage2_node_reduce_vec_bytes}"
            f"{operator.stage2_node_reduce_schedule}"
            f"{operator.stage2_node_reduce_load_schedule}"
            f"_nrw{operator.stage2_node_reduce_work_schedule}"
            f"_nrr{operator.stage2_node_reduce_rejoin_blocks}"
            f"_rla{operator.stage2_rank_epilogue_lds_addressing}"
            f"_rr{operator.stage2_rail_return_schedule}"
            f"_ws{operator.stage2_gmm_work_swizzle}"
            f"w{operator.stage2_window_n_groups}"
            f"_rg{operator.stage2_ready_granularity}"
            f"_epi{operator.stage2_epilogue_schedule}"
            f"_ng{operator.stage2_n_tile_group}"
            f"_gp{operator.stage2_group_pipeline_schedule}"
            f"_sb{operator.stage2_scoreboard_schedule}"
            f"_ai{operator.stage2_atomic_issue_schedule}"
            f"_weu{operator.stage2_waves_per_eu_hint}"
            + ("_timeline" if operator.timeline_instrument else "")
        )
        self.stage_names = (self.name,)
        self.total_field = self.name
        self._run_tokens = 0
        self._generation = 0
        self._stream = None
        self._stage1_frozen = False
        timeline_names = tuple(TIMELINE_INTERVALS)
        if operator.stage2_rail_return_schedule in (
            "qp_independent",
            "qp_prepost",
        ):
            timeline_names = tuple(
                name
                for name in timeline_names
                if "qp1_" not in name
                and "qp2_" not in name
                and "qp3_" not in name
                and "first_batch" not in name
                and "all_payloads" not in name
            )
        self.timeline_interval_names = timeline_names
        self.timeline_fields = (
            tuple(f"{name}_us" for name in timeline_names)
            if operator.timeline_instrument
            else ()
        )
        self._wall_clock_rate_khz = None
        if operator.timeline_instrument:
            import mori

            self._wall_clock_rate_khz = float(
                mori.cpp.get_cur_device_wall_clock_freq_mhz()
            )

    def prepare_iteration(self) -> None:
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
            raise ValueError(
                "Stage2 breakdown requires exactly the configured tokens/rank: "
                f"expected {op.mtpr}, got {run_tokens}"
            )
        stream = op._flydsl_stream(None)
        if op.timeline_instrument:
            dist.barrier()
            op._generation += 1
            generation = op._generation
            op._launch_stage1(
                self.shared.x,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
            )
        elif not self._stage1_frozen:
            op._generation += 1
            generation = op._generation
            op._launch_stage1(
                self.shared.x,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
            )
            self._stage1_frozen = True
        else:
            # Keep the same parity so every Stage2 replay consumes the exact
            # same physical H1 tiles, scales, route rows and expected counts.
            # Only the absolute generation word changes.
            from aiter.ops.flydsl.kernels.megamoe_tile.cco import write_window_u64

            generation = op._generation + 2
            op._generation = generation
            runtime = op._runtime
            if runtime is None:
                raise RuntimeError("CCO runtime is closed")
            stage1_done = (
                int(runtime.window.local_ptr)
                + int(op.layout.stage2_offset)
                + int(
                    op.stage2_layout.offset(
                        "stage1_done", parity=generation & 1
                    )
                )
            )
            write_window_u64(stage1_done, (generation,))
        # This barrier is deliberately outside the Stage2 event.  It removes
        # rank-to-rank Stage1 completion skew from the isolated Stage2 number.
        if not op.timeline_instrument:
            torch.cuda.synchronize(self.device)
            dist.barrier()
        self._run_tokens = run_tokens
        self._generation = generation
        self._stream = stream

    def timed_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        def launch_stage2() -> None:
            self.operator._launch_stage2(
                self._run_tokens, self._generation, self._stream
            )

        timer.stage(self.stage_names[0], launch_stage2)
        if self.mode in ("full", "return_only"):
            return self.operator._output[: self._run_tokens]
        return self.shared.x

    def finish_iteration(self) -> dict[str, float]:
        if not self.operator.timeline_instrument:
            return {}
        snapshot = self.operator.debug_device_timeline()
        ticks = snapshot["ticks"]
        if self._wall_clock_rate_khz is None or self._wall_clock_rate_khz <= 0:
            raise AssertionError("invalid device wall-clock frequency")
        scale = 1000.0 / self._wall_clock_rate_khz
        durations = {}
        for name in self.timeline_interval_names:
            begin, end = TIMELINE_INTERVALS[name]
            begin_tick = int(ticks[begin])
            end_tick = int(ticks[end])
            if begin_tick <= 0 or (
                end_tick < begin_tick
                and name not in SIGNED_TIMELINE_INTERVALS
            ):
                raise AssertionError(
                    f"invalid device timeline {name}: {begin_tick}->{end_tick}"
                )
            durations[f"{name}_us"] = (end_tick - begin_tick) * scale
        return durations

    def validate_prime(self, output: torch.Tensor) -> None:
        torch.cuda.synchronize(self.device)
        if self.mode in ("full", "return_only"):
            expected = (self.shape.tokens, self.shape.hidden)
            if tuple(output.shape) != expected or output.dtype is not torch.bfloat16:
                raise AssertionError("Stage2 output contract mismatch")
            if not torch.isfinite(output.float()).all().item():
                raise AssertionError("Stage2 produced non-finite output")
        snapshot = self.operator.debug_direct_tile_snapshot()
        if snapshot["stage2_error_count"] != 0:
            raise AssertionError(
                "Stage2 diagnostic reported a protocol error: "
                f"{snapshot}"
            )
        if self.mode == "full":
            _validate_direct_tile_debug_snapshot(
                snapshot,
                expected_routes=self.shape.tokens * self.shape.topk,
                expected_tokens=self.shape.tokens,
                expect_rank_token_completion=(
                    self.operator.stage2_ready_granularity == "token"
                ),
            )
        elif self.mode in (
            "atomic_only",
            "gmm2_atomic_only",
            "route_store_only",
        ):
            if (
                self.operator.stage2_node_accumulation_mode == "rank_local"
                and self.mode == "route_store_only"
            ):
                if self.operator.stage2_ready_granularity == "tile":
                    print(
                        "MEGAMOE_TILE_READY_DEBUG "
                        + json.dumps(
                            {
                                key: snapshot[key]
                                for key in (
                                    "rank_local_active_tokens",
                                    "tile_pending_nonzero",
                                    "tile_group_arrival_mismatch",
                                    "tile_rank_ready_missing",
                                    "tile_reduce_queue_tail",
                                    "tile_node_arrived_nonzero",
                                    "node_ready_mask_full_count",
                                    "tile_partial_ready_count",
                                    "stage2_error_count",
                                )
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if snapshot["tile_group_arrival_mismatch"] != 0:
                        raise AssertionError(
                            "Stage2 tile group arrival counters did not match route counts"
                        )
                    if snapshot["tile_rank_ready_missing"] != 0:
                        raise AssertionError(
                            "Stage2 tile rank readiness did not complete"
                        )
                    return
                if snapshot["rank_local_pending_nonzero"] != 0:
                    raise AssertionError(
                        "Stage2 rank-local pending counters did not drain: "
                        f"{snapshot}"
                    )
                if snapshot["rank_local_ready_missing"] != 0:
                    raise AssertionError(
                        "Stage2 rank-local readiness did not complete"
                    )
                return
            if (
                self.operator.stage2_ready_granularity == "tile"
                and self.mode == "gmm2_atomic_only"
            ):
                print(
                    "MEGAMOE_TILE_REDUCER_DEBUG "
                    + json.dumps(
                        {
                            key: snapshot[key]
                            for key in (
                                "tile_reduce_queue_tail",
                                "node_ready_mask_full_count",
                                "tile_partial_ready_count",
                                "tile_partial_ready_planes",
                                "rank_return_counts",
                                "stage2_error_count",
                            )
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if snapshot["node_expected_done_mismatch"] != 0:
                raise AssertionError("Stage2 producer scoreboard did not complete")
            readiness_field = (
                "node_route_store_not_ready"
                if self.mode == "route_store_only"
                else "node_not_ready"
            )
            if snapshot[readiness_field] != 0:
                raise AssertionError("Stage2 producer readiness did not complete")

    def close(self) -> None:
        self.operator.close()


class MoriStage2Path:
    """Exact selected fused_moe GMM2, optionally followed by MORI combine."""

    def __init__(self, baseline: MoriFusedMoeBaselinePath, mode: str):
        if mode not in MORI_MODES:
            raise ValueError(f"unsupported MORI mode {mode!r}")
        self.baseline = baseline
        self.mode = mode
        self.name = f"mori_{mode}"
        self.stage_names = (
            ("standalone_fused_moe_gmm2",)
            if mode == "gmm2_only"
            else ("standalone_fused_moe_gmm2", "mori_combine")
        )
        self.total_field = (
            "standalone_fused_moe_gmm2"
            if mode == "gmm2_only"
            else "gmm2_plus_mori_combine"
        )
        self.device = baseline.shared.x.device
        self._stage2_call = None
        self._local_out = None
        self._combined = False

    def _dispatch_and_capture(self, *, validate_recv: bool) -> None:
        base = self.baseline
        base._quant_op(
            base._quant_q,
            base.shared.x,
            base._quant_scale,
            32,
            shuffle_scale=False,
        )
        dispatched = base.op.dispatch(
            base._quant_q,
            base.shared.route_weights,
            base._quant_scale,
            base.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=8,
        )
        recv_q, recv_weights, recv_scales, recv_ids, recv_tokens = dispatched
        if validate_recv and int(recv_tokens.item()) != base.valid_recv:
            raise AssertionError("MORI Stage2 probe received an unexpected row count")

        fused_moe_module = importlib.import_module("aiter.fused_moe")
        captured: list[tuple[str, object]] = _StopBeforeStage2()
        fused_moe_module.kernel_bench_callable = captured
        try:
            weights = base.shared.prepared_weights
            try:
                base._fused_moe(
                    recv_q,
                    weights.w1,
                    weights.w2,
                    recv_weights,
                    recv_ids,
                    base.shared.local_expert_mask,
                    activation=base._activation,
                    quant_type=base._quant_type,
                    doweight_stage1=False,
                    w1_scale=weights.w1_scale,
                    w2_scale=weights.w2_scale,
                    a1_scale=recv_scales,
                    num_local_tokens=recv_tokens,
                    dtype=torch.bfloat16,
                    swiglu_limit=0.0,
                    gate_mode=base._gate_mode,
                )
            except _Stage2Captured:
                pass
            else:
                raise AssertionError("fused_moe did not expose a Stage2 callable")
        finally:
            fused_moe_module.kernel_bench_callable = None

        stage2_calls = [call for name, call in captured if name == "stage2"]
        if len(stage2_calls) != 1:
            raise AssertionError(
                f"expected one captured fused_moe Stage2 call, got {len(stage2_calls)}"
            )
        stage2_call = stage2_calls[0]
        if len(stage2_call.args) < 7:
            raise AssertionError("captured fused_moe Stage2 callable is malformed")
        local_out = stage2_call.args[6]
        if not isinstance(local_out, torch.Tensor):
            raise AssertionError("captured Stage2 output is not a tensor")

        # moe_sorting normally owns this clear. Keep an explicit untimed clear
        # so the isolated replay does not rely on a backend-specific sort path.
        local_out.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier()
        self._stage2_call = stage2_call
        self._local_out = local_out
        self._combined = False

    def prepare_iteration(self) -> None:
        self._dispatch_and_capture(validate_recv=False)

    def _combine(self) -> torch.Tensor:
        if self._local_out is None:
            raise RuntimeError("MORI Stage2 output is unavailable")
        result = self.baseline.op.combine(
            self._local_out,
            None,
            self.baseline.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=4,
        )
        self._combined = True
        return result[0] if isinstance(result, tuple) else result

    def timed_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        if self._stage2_call is None:
            raise RuntimeError("captured fused_moe Stage2 call is unavailable")
        timer.stage("standalone_fused_moe_gmm2", self._stage2_call)
        if self.mode == "gmm2_combine":
            return timer.stage("mori_combine", self._combine)
        return self._local_out

    def finish_iteration(self) -> None:
        # MORI dispatch/combine uses a normal producer/consumer lifecycle.
        # Close it even when combine is intentionally excluded from timing.
        if not self._combined:
            self._combine()
            torch.cuda.synchronize(self.device)

    def validate_prime(self, output: torch.Tensor) -> None:
        torch.cuda.synchronize(self.device)
        if self.mode == "gmm2_combine":
            output = output[: self.baseline.shape.tokens]
            if not torch.isfinite(output.float()).all().item():
                raise AssertionError("MORI GMM2+combine produced non-finite output")

    def close(self) -> None:
        return None


@dataclass
class BreakdownResult:
    local_samples: list[IterationTiming]
    rank_max_samples: list[IterationTiming]


def _add_total(sample: IterationTiming, path) -> None:
    if path.total_field not in sample.stage_us:
        sample.stage_us[path.total_field] = sum(
            sample.stage_us[name] for name in path.stage_names
        )


def _run(path, device, *, warmup: int, iterations: int) -> BreakdownResult:
    path.prepare_iteration()
    prime_timer = HipStageTimer(device, path.stage_names)
    prime_timer.begin_iteration()
    prime = path.timed_iteration(prime_timer)
    prime_sample = prime_timer.finish_iteration()
    _add_total(prime_sample, path)
    prime_sample.stage_us.update(path.finish_iteration() or {})
    path.validate_prime(prime)

    timer = HipStageTimer(device, path.stage_names)
    for _ in range(warmup):
        path.prepare_iteration()
        timer.begin_iteration()
        path.timed_iteration(timer)
        sample = timer.finish_iteration()
        _add_total(sample, path)
        sample.stage_us.update(path.finish_iteration() or {})

    local_samples: list[IterationTiming] = []
    rank_max_samples: list[IterationTiming] = []
    for iteration in range(iterations):
        profiler_pause()
        path.prepare_iteration()
        timer.begin_iteration()
        profiler_resume()
        with roctx_range(f"MEGAMOE_EP16_STAGE2_BREAKDOWN_{path.name}_{iteration}"):
            path.timed_iteration(timer)
            sample = timer.finish_iteration()
        profiler_pause()
        _add_total(sample, path)
        sample.stage_us.update(path.finish_iteration() or {})
        local_samples.append(sample)
        rank_max_samples.append(
            _global_max_timing(
                sample,
                tuple(
                    dict.fromkeys(
                        (
                            *path.stage_names,
                            path.total_field,
                            *getattr(path, "timeline_fields", ()),
                        )
                    )
                ),
            )
        )
    return BreakdownResult(local_samples, rank_max_samples)


def _summary(path, result: BreakdownResult, tail_iterations: int) -> dict[str, object]:
    fields = tuple(
        dict.fromkeys(
            (
                *path.stage_names,
                path.total_field,
                *getattr(path, "timeline_fields", ()),
            )
        )
    )
    rank_tail = result.rank_max_samples[-tail_iterations:]
    local_tail = result.local_samples[-tail_iterations:]
    rank_stats = {
        field: _sample_stats([sample.stage_us[field] for sample in rank_tail])
        for field in fields
    }
    all_rank_mean = {}
    for field in fields:
        local_sum = sum(sample.stage_us[field] for sample in local_tail)
        reduced = torch.tensor(local_sum, dtype=torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        all_rank_mean[field] = float(
            reduced.item() / (dist.get_world_size() * tail_iterations)
        )
    local_tail_matrix = torch.tensor(
        [[sample.stage_us[field] for field in fields] for sample in local_tail],
        dtype=torch.float64,
    )
    gathered_tail = [
        torch.empty_like(local_tail_matrix) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_tail, local_tail_matrix)
    per_rank_stats = {}
    for rank, values in enumerate(gathered_tail):
        per_rank_stats[str(rank)] = {
            field: _sample_stats(values[:, field_index].tolist())
            for field_index, field in enumerate(fields)
        }
    rank_min_stats = {
        field: _sample_stats(
            [
                min(
                    float(gathered_tail[rank][iteration, field_index].item())
                    for rank in range(dist.get_world_size())
                )
                for iteration in range(tail_iterations)
            ]
        )
        for field_index, field in enumerate(fields)
    }
    summary = {
        "path": path.name,
        "tail_iterations": tail_iterations,
        "tail_rank_max_stats_us": rank_stats,
        "tail_all_rank_sample_mean_us": all_rank_mean,
        "tail_rank_min_stats_us": rank_min_stats,
        "tail_per_rank_stats_us": per_rank_stats,
        "rank_max_samples_us": [
            {field: sample.stage_us[field] for field in fields}
            for sample in result.rank_max_samples
        ],
    }
    wall_clock_rate_khz = getattr(path, "_wall_clock_rate_khz", None)
    if wall_clock_rate_khz is not None:
        summary["device_wall_clock_rate_khz"] = wall_clock_rate_khz
        summary["first_stage2_rail_send_definition"] = (
            "doorbell is bracketed by s2_return_flush_pre/post"
        )
        local_timeline = torch.tensor(
            [
                [sample.stage_us[field] for field in TIMELINE_PER_RANK_FIELDS]
                for sample in local_tail
            ],
            dtype=torch.float64,
        )
        gathered_timeline = [
            torch.empty_like(local_timeline) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_timeline, local_timeline)
        per_rank = {}
        for rank, values in enumerate(gathered_timeline):
            rank_fields = {}
            for field_index, field in enumerate(TIMELINE_PER_RANK_FIELDS):
                samples = values[:, field_index].tolist()
                stats = _sample_stats(samples)
                stats["min"] = min(samples)
                stats["max"] = max(samples)
                if field == "s2_return_doorbell_to_all_gmm_done_us":
                    stats["doorbell_before_gmm_done_fraction"] = sum(
                        int(value > 0.0) for value in samples
                    ) / len(samples)
                rank_fields[field] = stats
            per_rank[str(rank)] = rank_fields
        summary["timeline_per_rank_stats_us"] = per_rank
    return summary


def _build_candidate(
    shape,
    shared,
    rank,
    device,
    mode,
    workers,
    accumulator_dtype,
    final_combine_blocks,
    gmm_schedule,
    return_chunk_tokens,
    bf16_atomic_kind,
    node_accumulation_mode,
    node_reduce_blocks,
    node_reduce_vec_bytes,
    node_reduce_schedule,
    node_reduce_load_schedule,
    node_reduce_work_schedule,
    node_reduce_rejoin_blocks,
    rank_epilogue_lds_addressing,
    rank_accumulation_mode,
    rail_return_schedule,
    rail_quant_type,
    gmm_work_swizzle,
    window_n_groups,
    ready_granularity,
    epilogue_schedule,
    n_tile_group,
    group_pipeline_schedule,
    scoreboard_schedule,
    atomic_issue_schedule,
    waves_per_eu_hint,
    timeline_instrument,
    max_routes_per_token_per_rank,
    stage1_diagnostic_phase,
):
    weights = shared.prepared_weights
    if max_routes_per_token_per_rank is not None:
        cap = int(max_routes_per_token_per_rank)
        owners = torch.div(
            shared.topk_ids,
            shape.experts // shape.ep_size,
            rounding_mode="floor",
        )
        max_multiplicity = max(
            int((owners == owner).sum(dim=1).max().item())
            for owner in range(shape.ep_size)
        )
        if max_multiplicity > cap:
            raise ValueError(
                "route fixture exceeds max_routes_per_token_per_rank: "
                f"observed {max_multiplicity}, capacity {cap}"
            )
    operator = Stage2ModeOperator(
        rank=rank,
        world_size=shape.ep_size,
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
        max_routes_per_token_per_rank=max_routes_per_token_per_rank,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
        # agent: sparse_wqe当前只有每QP 32-bit token bitmap；大capacity
        # 使用chunked transport，避免把128-token协议误用于512+。
        stage1_transport="chunked",
        stage2_mode=mode,
        stage2_worker_blocks=workers,
        stage2_accumulator_dtype=accumulator_dtype,
        stage2_final_combine_blocks=final_combine_blocks,
        stage2_gmm_schedule=gmm_schedule,
        stage2_return_chunk_tokens=return_chunk_tokens,
        stage2_bf16_atomic_kind=bf16_atomic_kind,
        stage2_node_accumulation_mode=node_accumulation_mode,
        stage2_node_reduce_blocks=node_reduce_blocks,
        stage2_node_reduce_vec_bytes=node_reduce_vec_bytes,
        stage2_node_reduce_schedule=node_reduce_schedule,
        stage2_node_reduce_load_schedule=node_reduce_load_schedule,
        stage2_node_reduce_work_schedule=node_reduce_work_schedule,
        stage2_node_reduce_rejoin_blocks=node_reduce_rejoin_blocks,
        stage2_rank_epilogue_lds_addressing=rank_epilogue_lds_addressing,
        stage2_rank_accumulation_mode=rank_accumulation_mode,
        stage2_rail_return_schedule=rail_return_schedule,
        stage2_rail_quant_type=rail_quant_type,
        stage2_gmm_work_swizzle=gmm_work_swizzle,
        stage2_window_n_groups=window_n_groups,
        stage2_ready_granularity=ready_granularity,
        stage2_epilogue_schedule=epilogue_schedule,
        stage2_n_tile_group=n_tile_group,
        stage2_group_pipeline_schedule=group_pipeline_schedule,
        stage2_scoreboard_schedule=scoreboard_schedule,
        stage2_atomic_issue_schedule=atomic_issue_schedule,
        stage2_waves_per_eu_hint=waves_per_eu_hint,
        stage2_timeline_instrument=timeline_instrument,
        stage1_diagnostic_phase=stage1_diagnostic_phase,
    )
    return CandidateStage2Path(operator, shared, shape, device, mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("candidate", "mori"), required=True)
    parser.add_argument("--candidate-mode", choices=CANDIDATE_MODES, default="full")
    parser.add_argument("--mori-mode", choices=MORI_MODES, default="gmm2_only")
    parser.add_argument(
        "--mori-combine-quant-type",
        choices=("none", "fp8_direct_cast", "fp8_blockwise", "fp4_blockwise"),
        default="none",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument(
        "--max-routes-per-token-per-rank", type=int, default=None
    )
    parser.add_argument(
        "--stage1-diagnostic-phase",
        choices=(
            "full",
            "quant_pack_only",
            "transport_only",
            "fanout_only",
            "quant_core_only",
            "dispatch_only",
        ),
        default="full",
    )
    parser.add_argument("--stage2-workers", type=int, default=160)
    parser.add_argument(
        "--candidate-accumulator", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument("--candidate-final-combine-blocks", type=int, default=14)
    parser.add_argument(
        "--candidate-gmm-schedule",
        choices=("persistent_queue", "static_strided"),
        default="persistent_queue",
    )
    parser.add_argument(
        "--candidate-return-chunk-tokens", type=int, choices=(4, 8, 16), default=8
    )
    parser.add_argument(
        "--candidate-bf16-atomic-kind",
        choices=("buffer", "global_system"),
        default="buffer",
    )
    parser.add_argument(
        "--candidate-node-accumulation-mode",
        choices=("direct_atomic", "route_store", "rank_local"),
        default="direct_atomic",
    )
    parser.add_argument(
        "--candidate-node-reduce-blocks",
        type=int,
        choices=(8, 16, 32, 56),
        default=32,
    )
    parser.add_argument(
        "--candidate-node-reduce-vec-bytes",
        type=int,
        choices=(4, 8, 16),
        default=4,
    )
    parser.add_argument(
        "--candidate-node-reduce-schedule",
        choices=("token", "group", "tile"),
        default="token",
    )
    parser.add_argument(
        "--candidate-node-reduce-load-schedule",
        choices=("interleaved", "load_first"),
        default="interleaved",
    )
    parser.add_argument(
        "--candidate-node-reduce-work-schedule",
        choices=("static_strided", "dynamic_head"),
        default="static_strided",
    )
    parser.add_argument(
        "--candidate-node-reduce-rejoin-blocks",
        type=int,
        choices=(0, 8, 16, 32),
        default=0,
    )
    parser.add_argument(
        "--candidate-rank-epilogue-lds-addressing",
        choices=("expanded", "dynamic_base"),
        default="expanded",
    )
    parser.add_argument(
        "--candidate-rank-accumulation-mode",
        choices=("atomic", "staged_reduce", "staged_ring"),
        default="atomic",
    )
    parser.add_argument(
        "--candidate-rail-return-schedule",
        choices=("lockstep", "qp_independent", "qp_prepost", "compact"),
        default="lockstep",
    )
    parser.add_argument(
        "--candidate-rail-quant-type",
        choices=("none", "fp8_blockwise"),
        default="none",
    )
    parser.add_argument(
        "--candidate-gmm-work-swizzle",
        choices=("token_major", "n_major_window"),
        default="token_major",
    )
    parser.add_argument(
        "--candidate-window-n-groups",
        type=int,
        choices=(1, 2, 4, 7, 14),
        default=2,
    )
    parser.add_argument(
        "--candidate-ready-granularity",
        choices=("token", "tile"),
        default="token",
    )
    parser.add_argument(
        "--candidate-epilogue-schedule",
        choices=(
            "lane32",
            "lane32_meta",
            "lane32_meta_expected",
            "wave64_meta",
        ),
        default="lane32_meta",
    )
    parser.add_argument(
        "--candidate-n-tile-group", type=int, choices=(1, 2), default=2
    )
    parser.add_argument(
        "--candidate-group-pipeline-schedule",
        choices=(
            "baseline",
            "expert_meta_hoist",
            "runtime_loop",
            "a_double_buffer",
        ),
        default="a_double_buffer",
    )
    parser.add_argument(
        "--candidate-scoreboard-schedule",
        choices=("wave0", "four_wave"),
        default="wave0",
    )
    parser.add_argument(
        "--candidate-atomic-issue-schedule",
        choices=("interleaved", "preload_pairs"),
        default="interleaved",
    )
    parser.add_argument(
        "--candidate-waves-per-eu-hint", type=int, choices=(1, 2, 3, 4), default=2
    )
    parser.add_argument("--device-timeline", action="store_true")
    parser.add_argument("--direct-packed-weights", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--tail-iters", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1, iters]")
    if (
        args.candidate_node_reduce_vec_bytes == 16
        and args.candidate_node_accumulation_mode != "rank_local"
    ):
        raise ValueError(
            "16-byte node reduction requires rank_local accumulation"
        )
    if args.candidate_rank_epilogue_lds_addressing == "dynamic_base" and not (
        args.candidate_node_accumulation_mode == "rank_local"
        and args.candidate_node_reduce_vec_bytes == 8
        and args.candidate_node_reduce_load_schedule == "load_first"
        and args.candidate_node_reduce_work_schedule == "static_strided"
        and args.candidate_node_reduce_rejoin_blocks == 0
    ):
        raise ValueError(
            "dynamic_base LDS addressing requires rank_local, vec8, "
            "load_first, static_strided reduction, and rejoin_blocks=0"
        )
    _validate_node_reduce_rejoin_config(
        diagnostic_mode=args.candidate_mode,
        worker_blocks=args.stage2_workers,
        final_combine_blocks=args.candidate_final_combine_blocks,
        gmm_schedule=args.candidate_gmm_schedule,
        node_accumulation_mode=args.candidate_node_accumulation_mode,
        node_reduce_blocks=args.candidate_node_reduce_blocks,
        node_reduce_work_schedule=args.candidate_node_reduce_work_schedule,
        node_reduce_rejoin_blocks=args.candidate_node_reduce_rejoin_blocks,
        rail_return_schedule=args.candidate_rail_return_schedule,
    )

    if not 1 <= args.tokens <= 4096:
        raise ValueError("tokens must be in [1, 4096]")
    shape = _shape(args.tokens)
    mode = args.candidate_mode if args.path == "candidate" else args.mori_mode
    contract = {
        "case_label": (
            f"TPR{shape.tokens}_TopK{shape.topk}_E{shape.experts}_"
            f"H{shape.hidden}_I{shape.inter}_EP{shape.ep_size}_A4W4"
        ),
        "shape": shape.__dict__,
        "path": args.path,
        "mode": mode,
        "mori_combine_quant_type": args.mori_combine_quant_type,
        "route_pattern": "paired-rank-half-remote",
        "candidate_stage1_input": "one frozen physical fused Stage1 H1/scale/route arena",
        "candidate_stage1_timing": "excluded_by_device_sync_and_ep16_barrier",
        "statistic": "per-iteration EP16 rank MAX; tail mean/P50/P95; all-rank mean",
        "stage2_workers": args.stage2_workers,
        "candidate_accumulator": args.candidate_accumulator,
        "candidate_final_combine_blocks": args.candidate_final_combine_blocks,
        "candidate_gmm_schedule": args.candidate_gmm_schedule,
        "candidate_return_chunk_tokens": args.candidate_return_chunk_tokens,
        "candidate_rail_quant_type": args.candidate_rail_quant_type,
        "candidate_gmm_work_swizzle": args.candidate_gmm_work_swizzle,
        "candidate_window_n_groups": args.candidate_window_n_groups,
        "candidate_ready_granularity": args.candidate_ready_granularity,
        "candidate_ready_group_tiles": args.candidate_n_tile_group,
        "candidate_ready_group_count": (
            (shape.hidden // 256 + args.candidate_n_tile_group - 1)
            // args.candidate_n_tile_group
        ),
        "max_routes_per_token_per_rank": args.max_routes_per_token_per_rank,
        "candidate_node_ready_granularity": args.candidate_ready_granularity,
        "candidate_bf16_atomic_kind": args.candidate_bf16_atomic_kind,
        "candidate_node_accumulation_mode": (
            args.candidate_node_accumulation_mode
        ),
        "candidate_node_reduce_blocks": args.candidate_node_reduce_blocks,
        "candidate_node_reduce_vec_bytes": args.candidate_node_reduce_vec_bytes,
        "candidate_node_reduce_schedule": args.candidate_node_reduce_schedule,
        "candidate_node_reduce_load_schedule": (
            args.candidate_node_reduce_load_schedule
        ),
        "candidate_node_reduce_work_schedule": (
            args.candidate_node_reduce_work_schedule
        ),
        "candidate_node_reduce_rejoin_blocks": (
            args.candidate_node_reduce_rejoin_blocks
        ),
        "candidate_rank_epilogue_lds_addressing": (
            args.candidate_rank_epilogue_lds_addressing
        ),
        "candidate_rank_accumulation_mode": args.candidate_rank_accumulation_mode,
        "candidate_rail_return_schedule": args.candidate_rail_return_schedule,
        "candidate_epilogue_schedule": args.candidate_epilogue_schedule,
        "candidate_n_tile_group": args.candidate_n_tile_group,
        "candidate_group_pipeline_schedule": (
            args.candidate_group_pipeline_schedule
        ),
        "candidate_scoreboard_schedule": args.candidate_scoreboard_schedule,
        "candidate_atomic_issue_schedule": args.candidate_atomic_issue_schedule,
        "candidate_waves_per_eu_hint": args.candidate_waves_per_eu_hint,
        "device_timeline": args.device_timeline,
        "direct_packed_weights": (
            args.device_timeline or args.direct_packed_weights
        ),
        "warmup": args.warmup,
        "iterations": args.iters,
        "tail_iterations": args.tail_iters,
    }
    if args.plan_only:
        print("MEGAMOE_EP16_STAGE2_BREAKDOWN_PLAN " + json.dumps(contract, sort_keys=True))
        return 0
    if args.device_timeline and (
        args.path != "candidate" or args.candidate_mode != "full"
    ):
        raise ValueError("--device-timeline requires --path candidate --candidate-mode full")

    rank, world, _local_rank, device = _setup_dist(needs_mori=args.path == "mori")
    if world != shape.ep_size:
        raise ValueError(f"Stage2 breakdown requires world=16, got {world}")
    profiler_pause()
    shared = _shared_inputs(
        shape,
        rank,
        world,
        device,
        route_pattern="paired-rank-half-remote",
        direct_packed_weights=(
            args.device_timeline or args.direct_packed_weights
        ),
    )
    if args.path == "candidate":
        path = _build_candidate(
            shape,
            shared,
            rank,
            device,
            args.candidate_mode,
            args.stage2_workers,
            args.candidate_accumulator,
            args.candidate_final_combine_blocks,
            args.candidate_gmm_schedule,
            args.candidate_return_chunk_tokens,
            args.candidate_bf16_atomic_kind,
            args.candidate_node_accumulation_mode,
            args.candidate_node_reduce_blocks,
            args.candidate_node_reduce_vec_bytes,
            args.candidate_node_reduce_schedule,
            args.candidate_node_reduce_load_schedule,
            args.candidate_node_reduce_work_schedule,
            args.candidate_node_reduce_rejoin_blocks,
            args.candidate_rank_epilogue_lds_addressing,
            args.candidate_rank_accumulation_mode,
            args.candidate_rail_return_schedule,
            args.candidate_rail_quant_type,
            args.candidate_gmm_work_swizzle,
            args.candidate_window_n_groups,
            args.candidate_ready_granularity,
            args.candidate_epilogue_schedule,
            args.candidate_n_tile_group,
            args.candidate_group_pipeline_schedule,
            args.candidate_scoreboard_schedule,
            args.candidate_atomic_issue_schedule,
            args.candidate_waves_per_eu_hint,
            args.device_timeline,
            args.max_routes_per_token_per_rank,
            args.stage1_diagnostic_phase,
        )
    else:
        baseline = MoriFusedMoeBaselinePath(
            shape,
            shared,
            rank,
            world,
            valid_recv=shape.tokens * world // 2,
            combine_quant_type=args.mori_combine_quant_type,
        )
        path = MoriStage2Path(baseline, args.mori_mode)

    try:
        result = _run(
            path,
            device,
            warmup=args.warmup,
            iterations=args.iters,
        )
        summary = _summary(path, result, args.tail_iters)
        if rank == 0:
            print(
                "MEGAMOE_EP16_STAGE2_BREAKDOWN_RESULT "
                + json.dumps({**contract, "timing": summary}, sort_keys=True),
                flush=True,
            )
    finally:
        dist.barrier()
        path.close()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
