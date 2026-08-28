# SPDX-License-Identifier: MIT
"""Diagnostic and full-GMM1 factories for the EP16 two-kernel MegaMoE path.

This module deliberately lives under ``op_tests``: it is a measurement aid,
not an alternate production operator.  It reuses the production Stage-1 and
Stage-2 kernel bodies and changes only their imported GEMM helpers:

* Stage-1 communication modes keep BF16->A4, hierarchical dispatch, receiver
  placement, scoreboard/EOS and ready-queue publication while removing GMM1.
* Stage-1 rejoin/tilepipe modes retain the real GMM1, SiLU and A4 requant; the
  latter consumes full tiles before communication EOS in the same kernel.
* Stage-2 replaces the GMM2 K-loop with correctly shaped zero accumulators.
  The production direct-node epilogue therefore still performs the exact
  metadata loads, weighted FP32 LSA atomics, done/ready publication, CCO RAIL
  return, credit and final BF16 combine for every work item.

The decorators are intercepted while each production factory is built so the
probe code objects have independent symbols/JIT keys.  Production sources and
MORI are not modified.
"""

from __future__ import annotations

from contextlib import contextmanager
import time

import flydsl.compiler as flyc
import flydsl.compiler.jit_function as jit_function_impl
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr.typing import Float32
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.megamoe_tile.mega_moe_tile_a4w4 import MegaMoETileA4W4
from aiter.ops.flydsl.kernels.megamoe_tile import stage1 as stage1_impl
from aiter.ops.flydsl.kernels.megamoe_tile import stage2 as stage2_impl
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import SPARSE_QP_GENERATION_SHIFT


STAGE1_PROBE_SUFFIX = "commprobe_quant_dispatch_nogemm1_v1"
STAGE1_FULL_GMM1_SUFFIX = "fullstage1_quant_transport_fanout_gmm1_silu_requant_v1"
STAGE2_PROBE_SUFFIX = "commprobe_zero_gemm2_direct_combine_v1"


def split128x2_assignment(ticket: int) -> tuple[str, int, int, tuple[int, ...]]:
    """CPU contract for the diagnostic 128 inter + 128 intra ticket map."""

    ticket = int(ticket)
    if not 0 <= ticket < 256:
        raise ValueError("split fanout ticket must be in [0, 256)")
    domain = "remote" if ticket < 128 else "local"
    worker = ticket if ticket < 128 else ticket - 128
    dest = worker % 8
    shard = worker // 8
    return domain, dest, shard, tuple(range(shard, 128, 16))


def tilepipe_assignment(
    ticket: int, fanout_shards: int = 16
) -> tuple[str | None, int | None, int | None, tuple[int, ...]]:
    """CPU contract for tunable remote/local fanout plus queue consumers."""

    ticket = int(ticket)
    if not 0 <= ticket < 256:
        raise ValueError("tile-pipeline ticket must be in [0, 256)")
    fanout_shards = int(fanout_shards)
    if fanout_shards not in (8, 12, 16):
        raise ValueError("fanout_shards must be 8, 12, or 16")
    fanout_ctas = 8 * fanout_shards
    if ticket < fanout_ctas:
        domain = "remote"
        worker = ticket
    elif 128 <= ticket < 128 + fanout_ctas:
        domain = "local"
        worker = ticket - 128
    else:
        return None, None, None, ()
    dest = worker % 8
    shard = worker // 8
    return domain, dest, shard, tuple(range(shard, 128, fanout_shards))


def split64x2_tilepipe_assignment(
    ticket: int,
) -> tuple[str | None, int | None, int | None, tuple[int, ...]]:
    """Backward-compatible helper for the 8-shard/64-CTA tuning point."""

    return tilepipe_assignment(ticket, 8)


def wave64_assignment(
    ticket: int, wave: int
) -> tuple[str, int, int, tuple[int, ...]]:
    """CPU contract for full-grid fanout with one active wave per CTA."""

    ticket = int(ticket)
    wave = int(wave)
    if not 0 <= ticket < 256:
        raise ValueError("wave fanout ticket must be in [0, 256)")
    if not 0 <= wave < 4:
        raise ValueError("wave fanout wave must be in [0, 4)")
    if ticket < 128:
        domain = "remote"
        worker = ticket
    else:
        domain = "local"
        worker = ticket - 128
    dest = worker % 8
    shard = worker // 8
    tokens = tuple(range(shard, 128, 16)) if wave == 0 else ()
    return domain, dest, shard, tokens


def cco_flush_batch_contract(chunks_per_flush: int) -> dict[str, object]:
    """CPU contract for 8 chunks x 4 QPs data/credit batching."""

    chunks_per_flush = int(chunks_per_flush)
    if chunks_per_flush not in (1, 2, 4, 8):
        raise ValueError("chunks_per_flush must be one of 1,2,4,8")
    batches = tuple(
        tuple(range(first, first + chunks_per_flush))
        for first in range(0, 8, chunks_per_flush)
    )
    request_slots = tuple(
        (batch[0], qp) for batch in batches for qp in range(4)
    )
    covered = tuple(
        (chunk, qp) for batch in batches for chunk in batch for qp in range(4)
    )
    return {
        "batches": batches,
        "data_ready_coverage": covered,
        "credit_coverage": covered,
        "request_slots": request_slots,
        "data_doorbells": 4 * len(batches),
        "credit_doorbells": 4 * len(batches),
        "logical_doorbells": 2 * 4 * len(batches),
        "data_wqes": 2 * 8 * 4,
        "credit_wqes": 8 * 4,
    }


def cco_mori64x2_contract() -> dict[str, object]:
    """CPU contract for two 64-token contiguous CCO transfers."""

    halves = (
        {"half": 0, "qp": 0, "token_begin": 0, "token_end": 64},
        {"half": 1, "qp": 1, "token_begin": 64, "token_end": 128},
    )
    return {
        "halves": halves,
        "payload_bytes_per_half": 64 * 4096,
        "payload_bytes_per_rank": 128 * 4096,
        "ready_indices": (0, 1),
        "consumed_indices": (0, 1),
        "credit_indices": (0, 1),
        "request_indices": (0, 1),
        "data_wqes": 4,
        "credit_wqes": 2,
        "logical_doorbells": 4,
    }


def quant_two_cta_assignment(
    ticket: int,
) -> tuple[int, int, tuple[int, ...], bool]:
    """Return token, half, 1x32 groups, and metadata ownership."""

    ticket = int(ticket)
    if not 0 <= ticket < 256:
        raise ValueError("quant ticket must be in [0,256)")
    token = ticket & 127
    half = ticket // 128
    groups = tuple(range(half * 112, (half + 1) * 112))
    return token, half, groups, half == 0


@flyc.jit
def _noop_issue_a_load_lds_dt(
    arg_aq,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=32,
):
    # The production Stage-2 prologue issues kStages A loads before entering
    # gemm2_compute_v2.  Suppress those loads together with the K-loop; the
    # direct-node communication/combine epilogue remains untouched.
    return None


@flyc.jit
def _zero_gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    expert_offset=0,
    preloaded_expert=None,
):
    """Return the production accumulator geometry without a GEMM2 K-loop."""

    num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
    m_block_idx = fx.Int32(bx_i32) // num_n_blocks
    n_block_idx = fx.Int32(bx_i32) - m_block_idx * num_n_blocks
    m_row = m_block_idx * fx.Int32(BM)
    # gemm2_compute_v2 returns BM/16 row groups, each containing
    # (BN/4)/16 four-lane FP32 vectors.  direct_node_epilog consumes exactly
    # this shape and consequently emits the production number of LSA atomics.
    accm = [
        [Vec.filled(4, 0.0, Float32) for _ in range((BN // 4) // 16)]
        for _ in range(BM // 16)
    ]
    return accm, m_row, n_block_idx, fx.Int32(i32_hidden)


@contextmanager
def _unique_kernel_symbol(suffix: str):
    """Append ``suffix`` to every kernel decorator used in this scope."""

    original = flyc.kernel

    def decorated(*args, **kwargs):
        if "name" in kwargs:
            kwargs = dict(kwargs)
            kwargs["name"] = f"{kwargs['name']}_{suffix}"
        elif args and isinstance(args[0], str):
            args = (f"{args[0]}_{suffix}", *args[1:])
        else:
            raise RuntimeError("comm probe requires an explicitly named FlyDSL kernel")
        return original(*args, **kwargs)

    flyc.kernel = decorated
    try:
        yield
    finally:
        flyc.kernel = original


class _NoLaunch:
    def __call__(self, _args):
        return None


def _precompile_without_launch(launcher, args) -> None:
    """Populate one launcher's in-process artifact cache without GPU work."""

    original = jit_function_impl._build_call_state
    try:
        # ``flyc.compile`` normally executes the just-built launcher once.
        # Replace only that final CallState; MLIR/LLVM/HSACO generation and
        # module loading still run in full.
        jit_function_impl._build_call_state = (
            lambda *unused_args, **unused_kwargs: _NoLaunch()
        )
        flyc.compile(launcher, *args)
    finally:
        jit_function_impl._build_call_state = original
        # The fake CallState must never become the hot-path fast hit.  The
        # CompiledArtifact remains in ``_mem_cache``; the first real call only
        # builds a normal CallState around the already-loaded code object.
        launcher._call_state_cache.clear()


class MegaMoETileA4W4CommProbe(MegaMoETileA4W4):
    """EP16 protocol probe; rejoin modes retain the real fused GMM1."""

    diagnostic_only = True
    stage1_contains = "bf16_to_a4+hier_dispatch+scoreboard+eos+queue_publish"
    stage2_contains = "direct_lsa_atomic+done_ready+rail_return+final_combine"

    def __init__(
        self,
        *args,
        probe_stage: str = "both",
        stage1_mode: str = "legacy",
        stage1_cco_chunks_per_flush: int = 1,
        stage1_cco_geometry: str = "chunked",
        stage1_quant_two_cta_per_token: bool = False,
        stage1_prequant_input: bool = False,
        stage1_tile_pipeline_instrument: bool = False,
        stage1_tile_pipeline_fanout_shards: int = 16,
        stage1_phase: str = "full",
        stage2_mode: str = "full",
        stage2_worker_blocks: int = 160,
        stage2_node_accumulation_mode: str = "direct_atomic",
        stage2_node_reduce_blocks: int = 16,
        stage2_node_reduce_vec_bytes: int = 8,
        stage2_node_reduce_schedule: str = "token",
        stage2_node_reduce_load_schedule: str = "interleaved",
        stage2_node_reduce_work_schedule: str = "static_strided",
        stage2_node_reduce_rejoin_blocks: int = 0,
        stage2_rank_epilogue_lds_addressing: str = "expanded",
        stage2_return_chunk_tokens: int = 8,
        stage2_rail_return_schedule: str = "lockstep",
        **kwargs,
    ):
        if probe_stage not in ("stage1", "stage2", "both"):
            raise ValueError("probe_stage must be stage1, stage2, or both")
        if stage1_mode not in (
            "legacy",
            "internodev1_split128x2",
            "internodev1_split128x2_no_arrival_rmw",
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        ):
            raise ValueError(
                "stage1_mode must be legacy, internodev1_split128x2, "
                "internodev1_split128x2_no_arrival_rmw, "
                "internodev1_split128x2_rejoin, "
                "internodev1_tilepipe, internodev1_wave64, "
                "or internodev1_wave64_rejoin"
            )
        # This diagnostic class replaces GMM2 with zero accumulators, so its
        # Stage2 protocol can safely follow a comm-only split Stage1.  The
        # real-GMM2 subclass below forces a GMM1 rejoin mode.
        if stage2_mode not in ("full", "atomic_only", "return_only"):
            raise ValueError(
                "stage2_mode must be full, atomic_only, or return_only"
            )
        self.probe_stage = probe_stage
        self.stage1_mode = stage1_mode
        self.stage1_cco_chunks_per_flush = int(
            stage1_cco_chunks_per_flush
        )
        if self.stage1_cco_chunks_per_flush not in (1, 2, 4, 8):
            raise ValueError(
                "stage1_cco_chunks_per_flush must be one of 1,2,4,8"
            )
        if stage1_cco_geometry not in (
            "chunked",
            "mori64x2",
            "sparse_wqe",
        ):
            raise ValueError("invalid stage1_cco_geometry")
        self.stage1_cco_geometry = stage1_cco_geometry
        self.stage1_quant_two_cta_per_token = bool(
            stage1_quant_two_cta_per_token
        )
        self.stage1_prequant_input = bool(stage1_prequant_input)
        self.stage1_tile_pipeline_instrument = bool(
            stage1_tile_pipeline_instrument
        )
        self.stage1_tile_pipeline_fanout_shards = int(
            stage1_tile_pipeline_fanout_shards
        )
        if self.stage1_tile_pipeline_fanout_shards not in (8, 12, 16):
            raise ValueError(
                "stage1_tile_pipeline_fanout_shards must be 8, 12, or 16"
            )
        real_gmm1 = stage1_mode in (
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64_rejoin",
        )
        if real_gmm1:
            self.stage1_contains = (
                "bf16_to_a4+hier_dispatch+scoreboard+eos+queue_publish+"
                "gmm1+silu+a4_requant"
            )
        if self.stage1_prequant_input:
            self.stage1_contains = (
                "prequant_a4_scale_copy+hier_dispatch+scoreboard+eos+"
                "queue_publish"
                + ("+gmm1+silu+a4_requant" if real_gmm1 else "")
            )
        if stage1_phase not in (
            "full",
            "quant_core_only",
            "quant_pack_only",
            "transport_only",
            "fanout_only",
            "dispatch_only",
        ):
            raise ValueError("invalid stage1_phase")
        if stage1_phase != "full" and not (
            stage1_mode.startswith("internodev1_split128x2")
            or stage1_mode.startswith("internodev1_wave64")
        ):
            raise ValueError("non-full Stage1 phase requires split128x2 mode")
        if stage1_phase != "full" and probe_stage != "stage1":
            raise ValueError("Stage1 phase probes are stage1-only")
        if self.stage1_prequant_input and stage1_phase != "full":
            raise ValueError("prequant Stage1 currently requires full phase")
        if self.stage1_tile_pipeline_instrument and (
            stage1_mode != "internodev1_tilepipe"
        ):
            raise ValueError(
                "tile pipeline instrumentation requires tilepipe mode"
            )
        self.stage1_phase = stage1_phase
        self.stage2_mode = stage2_mode
        self.stage2_worker_blocks = int(stage2_worker_blocks)
        self.stage2_node_accumulation_mode = str(
            stage2_node_accumulation_mode
        )
        if self.stage2_node_accumulation_mode == "route_store":
            self.stage2_contains = (
                "route_slot_store+register_reduce+done_ready+"
                "rail_return+final_combine"
            )
        elif self.stage2_node_accumulation_mode == "rank_local":
            self.stage2_contains = (
                "rank_local_accumulate+node_forward+register_reduce+"
                "done_ready+rail_return+final_combine"
            )
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
        self.stage2_return_chunk_tokens = int(stage2_return_chunk_tokens)
        self.stage2_rail_return_schedule = str(stage2_rail_return_schedule)
        if self.stage2_node_accumulation_mode not in (
            "direct_atomic",
            "route_store",
            "rank_local",
        ):
            raise ValueError("invalid Stage2 node accumulation mode")
        if self.stage2_node_reduce_blocks not in (8, 16, 32, 56):
            raise ValueError("stage2_node_reduce_blocks must be one of 8,16,32,56")
        if self.stage2_node_reduce_vec_bytes not in (4, 8, 16):
            raise ValueError(
                "stage2_node_reduce_vec_bytes must be 4, 8, or 16"
            )
        if (
            self.stage2_node_reduce_vec_bytes == 16
            and self.stage2_node_accumulation_mode != "rank_local"
        ):
            raise ValueError(
                "16-byte node reduction requires rank_local accumulation"
            )
        if self.stage2_node_reduce_load_schedule not in (
            "interleaved",
            "load_first",
        ):
            raise ValueError("invalid Stage2 node-reduce load schedule")
        if self.stage2_node_reduce_work_schedule not in (
            "static_strided",
            "dynamic_head",
        ):
            raise ValueError("invalid Stage2 node-reduce work schedule")
        if self.stage2_node_reduce_rejoin_blocks not in (0, 8, 16, 32):
            raise ValueError(
                "stage2_node_reduce_rejoin_blocks must be one of 0,8,16,32"
            )
        if self.stage2_rank_epilogue_lds_addressing not in (
            "expanded",
            "dynamic_base",
        ):
            raise ValueError(
                "stage2_rank_epilogue_lds_addressing must be expanded or dynamic_base"
            )
        if self.stage2_rank_epilogue_lds_addressing == "dynamic_base" and not (
            self.stage2_node_accumulation_mode == "rank_local"
            and self.stage2_node_reduce_vec_bytes == 8
            and self.stage2_node_reduce_load_schedule == "load_first"
            and self.stage2_node_reduce_work_schedule == "static_strided"
            and self.stage2_node_reduce_rejoin_blocks == 0
        ):
            raise ValueError(
                "dynamic_base LDS addressing requires rank_local, vec8, "
                "load_first, static_strided reduction, and rejoin_blocks=0"
            )
        if self.stage2_return_chunk_tokens not in (4, 8, 16):
            raise ValueError("stage2_return_chunk_tokens must be 4, 8, or 16")
        if self.stage2_rail_return_schedule not in (
            "lockstep",
            "qp_independent",
            "qp_prepost",
            "compact",
        ):
            raise ValueError("invalid Stage2 RAIL return schedule")
        if (
            self.stage2_rail_return_schedule != "lockstep"
            and self.stage2_return_chunk_tokens <= 4
        ):
            raise ValueError(
                "independent Stage2 RAIL schedules require chunks larger than 4"
            )
        if self.stage2_rail_return_schedule == "compact" and (
            self.stage2_node_accumulation_mode != "rank_local"
        ):
            raise ValueError("compact Stage2 RAIL return requires rank_local")
        if self.stage2_node_reduce_rejoin_blocks > 0 and not (
            self.stage2_node_accumulation_mode == "rank_local"
            and self.stage2_rail_return_schedule == "compact"
            and self.stage2_node_reduce_work_schedule == "dynamic_head"
            and self.stage2_mode in ("full", "atomic_only")
        ):
            raise ValueError(
                "Stage2 reducer rejoin requires a queue-producing rank_local "
                "compact dynamic_head mode"
            )
        gmm_cta_count = self.stage2_worker_blocks - (
            1 + self.stage2_node_reduce_blocks + 14
        )
        if (
            self.stage2_node_reduce_rejoin_blocks > 0
            and self.stage2_node_reduce_rejoin_blocks > gmm_cta_count
        ):
            raise ValueError(
                "stage2_node_reduce_rejoin_blocks exceeds the available "
                f"GMM2 CTA count ({gmm_cta_count})"
            )
        reduce_role_blocks = (
            self.stage2_node_reduce_blocks
            if self.stage2_node_accumulation_mode
            in ("route_store", "rank_local")
            else 0
        )
        min_stage2_worker_blocks = (
            1
            + reduce_role_blocks
            + 14
            + (0 if stage2_mode == "return_only" else 1)
        )
        max_stage2_worker_blocks = (
            160 + self.stage2_node_reduce_blocks
            if self.stage2_node_accumulation_mode
            in ("route_store", "rank_local")
            else 160
        )
        if not (
            min_stage2_worker_blocks
            <= self.stage2_worker_blocks
            <= max_stage2_worker_blocks
        ):
            raise ValueError(
                "stage2_worker_blocks must be in "
                f"[{min_stage2_worker_blocks}, "
                f"{max_stage2_worker_blocks}] for "
                f"{self.stage2_node_accumulation_mode}/{stage2_mode}"
            )
        super().__init__(*args, **kwargs)

    def _validate_weight_capacity(self, w1, w1_scale, w2, w2_scale) -> None:
        # Pure communication modes omit both contraction loops and accept tiny
        # sentinel weights.  Rejoin/tilepipe modes retain real GMM1 and must
        # therefore validate the complete native W1/W1-scale capacity.
        if self.stage1_mode in (
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64_rejoin",
        ):
            expected_w1 = self.epr * (2 * self.inter_dim) * self.model_dim // 2
            expected_scale = (
                self.epr * (2 * self.inter_dim) * (self.model_dim // 32)
            )
            actual = (
                w1.numel() * w1.element_size(),
                w1_scale.numel() * w1_scale.element_size(),
            )
            if actual != (expected_w1, expected_scale):
                raise ValueError(
                    "split rejoin requires native packed W1/W1-scale capacity; "
                    f"got={actual}, expected={(expected_w1, expected_scale)}"
                )
        for name, tensor in (
            ("w1", w1),
            ("w1_scale", w1_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ):
            if tensor.numel() < 1:
                raise ValueError(f"comm probe {name} sentinel must be non-empty")

    def _build_stage1_probe_launcher(
        self,
        phase: str | None = None,
        *,
        prequant_input: bool | None = None,
    ):
        phase = self.stage1_phase if phase is None else str(phase)
        prequant_input = (
            self.stage1_prequant_input
            if prequant_input is None
            else bool(prequant_input)
        )
        split = self.stage1_mode in (
            "internodev1_split128x2",
            "internodev1_split128x2_no_arrival_rmw",
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        )
        rejoin = self.stage1_mode in (
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64_rejoin",
        )
        tile_pipeline = (
            self.stage1_mode == "internodev1_tilepipe"
        )
        wave_fanout = self.stage1_mode in (
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        )
        no_arrival_rmw = (
            self.stage1_mode
            == "internodev1_split128x2_no_arrival_rmw"
        )
        if split:
            device_cus = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            if device_cus < 256:
                raise RuntimeError(
                    f"split fanout requires at least 256 CUs, got {device_cus}"
                )
        if tile_pipeline:
            fanout_ctas = 8 * self.stage1_tile_pipeline_fanout_shards
            suffix = (
                f"{STAGE1_FULL_GMM1_SUFFIX}_split{fanout_ctas}x2_"
                "tilepipe256r"
            )
        elif wave_fanout:
            suffix = (
                f"{STAGE1_FULL_GMM1_SUFFIX}_wave64x1_fullgrid256_rejoin"
                if rejoin
                else f"{STAGE1_PROBE_SUFFIX}_wave64x1_fullgrid256"
            )
        elif split:
            suffix = (
                f"{STAGE1_FULL_GMM1_SUFFIX}_split128x2_rejoin256_posteos"
                if rejoin
                else (
                    f"{STAGE1_PROBE_SUFFIX}_"
                    "internodev1_split128x2_grid256_no_arrival_rmw"
                    if no_arrival_rmw
                    else f"{STAGE1_PROBE_SUFFIX}_internodev1_split128x2_grid256"
                )
            )
        else:
            suffix = STAGE1_PROBE_SUFFIX
        if phase != "full":
            suffix += f"_phase_{phase}"
        if prequant_input:
            suffix += "_prequant_input"
        elif self.stage1_prequant_input:
            suffix += "_internal_quant_reference"
        with _unique_kernel_symbol(suffix):
            launcher = stage1_impl.compile_megamoe_tile_ep16_stage1(
                self.stage1_layout,
                self.stage2_layout,
                rank=self.rank,
                stage2_window_offset=self.layout.stage2_offset,
                worker_blocks=256 if split else self.worker_blocks,
                waves_per_eu_hint=2,
                diagnostic_comm_only=not rejoin,
                diagnostic_split_fanout=split,
                diagnostic_wave_fanout=wave_fanout,
                diagnostic_no_arrival_rmw=no_arrival_rmw,
                cco_chunks_per_flush=self.stage1_cco_chunks_per_flush,
                cco_geometry=self.stage1_cco_geometry,
                diagnostic_phase=phase,
                quant_two_cta_per_token=self.stage1_quant_two_cta_per_token,
                prequant_input=prequant_input,
                tile_pipeline=tile_pipeline,
                tile_pipeline_instrument=(
                    self.stage1_tile_pipeline_instrument
                ),
                tile_pipeline_fanout_shards=(
                    self.stage1_tile_pipeline_fanout_shards
                ),
            )
        launcher.kernel_name = f"{launcher.kernel_name}_{suffix}"
        launcher.comm_probe = self.stage1_contains
        launcher.gemm1_contraction = rejoin and phase == "full"
        launcher.full_stage1_fusion = rejoin and phase == "full"
        launcher.comm_probe_phase = phase
        if split:
            flag_region = self.stage1_layout.region("fanout_done")
            flag_plane_bytes = flag_region.nbytes // self.stage1_layout.parity_depth
            if launcher.split_flag_bytes != flag_plane_bytes:
                raise AssertionError("split flag ABI capacity mismatch")
            launcher.split_flag_alignment = flag_region.alignment
        if (
            launcher.cco_chunks_per_flush
            != self.stage1_cco_chunks_per_flush
        ):
            raise AssertionError("Stage1 CCO batch contract mismatch")
        if launcher.cco_geometry != self.stage1_cco_geometry:
            raise AssertionError("Stage1 CCO geometry contract mismatch")
        if (
            launcher.quant_two_cta_per_token
            != self.stage1_quant_two_cta_per_token
        ):
            raise AssertionError("Stage1 quant CTA contract mismatch")
        if launcher.diagnostic_wave_fanout != wave_fanout:
            raise AssertionError("Stage1 wave fanout contract mismatch")
        if launcher.prequant_input != prequant_input:
            raise AssertionError("Stage1 prequant input contract mismatch")
        if launcher.tile_pipeline != tile_pipeline:
            raise AssertionError("Stage1 tile pipeline contract mismatch")
        if (
            launcher.tile_pipeline_instrument
            != self.stage1_tile_pipeline_instrument
        ):
            raise AssertionError(
                "Stage1 tile pipeline instrumentation contract mismatch"
            )
        expected_fanout_shards = (
            self.stage1_tile_pipeline_fanout_shards
            if tile_pipeline
            else 16
        )
        if launcher.tile_pipeline_fanout_shards != expected_fanout_shards:
            raise AssertionError("Stage1 tile pipeline fanout geometry mismatch")
        return launcher

    def _build_stage2_probe_launcher(self):
        # The fake accumulator feeds the unmodified direct-node epilogue, so
        # this measures combine traffic rather than merely publishing flags.
        stage2_impl.issue_a_load_lds_dt = _noop_issue_a_load_lds_dt
        stage2_impl.gemm2_compute_v2 = _zero_gemm2_compute_v2
        suffix = f"{STAGE2_PROBE_SUFFIX}_{self.stage2_mode}"
        with _unique_kernel_symbol(suffix):
            launcher = stage2_impl.compile_megamoe_tile_ep16_stage2_a4w4(
                self.layout,
                rank=self.rank,
                BM=32,
                BN=256,
                BK=256,
                WORK_SHARDS=8,
                waves_per_eu_hint=2,
                team="rail",
                diagnostic_mode=self.stage2_mode,
                accumulator_dtype="bf16",
                final_combine_blocks=14,
                gmm_schedule="persistent_queue",
                return_chunk_tokens=self.stage2_return_chunk_tokens,
                bf16_atomic_kind="buffer",
                rail_return_schedule=self.stage2_rail_return_schedule,
                epilogue_schedule="lane32_meta",
                n_tile_group=2,
                group_pipeline_schedule="a_double_buffer",
                node_accumulation_mode=self.stage2_node_accumulation_mode,
                node_reduce_blocks=self.stage2_node_reduce_blocks,
                node_reduce_vec_bytes=self.stage2_node_reduce_vec_bytes,
                node_reduce_schedule=self.stage2_node_reduce_schedule,
                node_reduce_load_schedule=(
                    self.stage2_node_reduce_load_schedule
                ),
                node_reduce_work_schedule=(
                    self.stage2_node_reduce_work_schedule
                ),
                node_reduce_rejoin_blocks=(
                    self.stage2_node_reduce_rejoin_blocks
                ),
                rank_epilogue_lds_addressing=(
                    self.stage2_rank_epilogue_lds_addressing
                ),
            )
        launcher.kernel_name = f"{launcher.kernel_name}_{suffix}"
        launcher.comm_probe = self.stage2_contains
        launcher.gemm2_contraction = False
        launcher.preserves_direct_lsa_atomic_epilogue = (
            self.stage2_node_accumulation_mode == "direct_atomic"
        )
        launcher.preserves_route_store_register_reduce = (
            self.stage2_node_accumulation_mode == "route_store"
        )
        launcher.preserves_rank_local_register_reduce = (
            self.stage2_node_accumulation_mode == "rank_local"
        )
        return launcher

    def _compile_stage1(self):
        stage1 = self._build_stage1_probe_launcher()
        internal_quant_reference = (
            self._build_stage1_probe_launcher(
                "full", prequant_input=False
            )
            if self.stage1_prequant_input
            else None
        )
        quant_setup = (
            self._build_stage1_probe_launcher("quant_pack_only")
            if self.stage1_phase
            in ("transport_only", "fanout_only", "dispatch_only")
            else None
        )
        transport_setup = (
            self._build_stage1_probe_launcher("transport_only")
            if self.stage1_phase == "fanout_only"
            else None
        )
        stage2 = self._build_stage2_probe_launcher()
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime must exist before probe precompile")
        stream = self._flydsl_stream(None)
        stage1_args = (
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            self._output.data_ptr(),  # valid BF16-shaped pointer; never loaded
            self._output.data_ptr(),  # valid prequant scale pointer; never loaded
            self._output.data_ptr(),  # valid runtime pointer; never loaded
            self._output.data_ptr(),  # valid runtime pointer; never loaded
            self._w1.data_ptr(),
            self._w1_scale.data_ptr(),
            self.mtpr,
            1,
            stream,
        )
        stage2_args = (
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            self._w2.data_ptr(),
            self._w2_scale.data_ptr(),
            1,
            self.mtpr,
            self.stage2_worker_blocks,
            self._output.data_ptr(),
            stream,
        )

        # Two compilers (same local rank on the two nodes) run concurrently;
        # the other fourteen ranks wait in Gloo.  This avoids the severe CPU
        # contention observed when eight large FlyDSL modules compile per node.
        for turn in range(self.gpus_per_node):
            dist.barrier()
            if self.local_rank == turn:
                started = time.perf_counter()
                print(
                    f"MEGAMOE_COMM_PROBE_PRECOMPILE rank={self.rank} "
                    f"turn={turn} phase=begin",
                    flush=True,
                )
                _precompile_without_launch(stage1, stage1_args)
                if internal_quant_reference is not None:
                    _precompile_without_launch(
                        internal_quant_reference, stage1_args
                    )
                if quant_setup is not None:
                    _precompile_without_launch(quant_setup, stage1_args)
                if transport_setup is not None:
                    _precompile_without_launch(transport_setup, stage1_args)
                if self.probe_stage in ("stage2", "both"):
                    _precompile_without_launch(stage2, stage2_args)
                elapsed = time.perf_counter() - started
                print(
                    f"MEGAMOE_COMM_PROBE_PRECOMPILE rank={self.rank} "
                    f"turn={turn} phase=done elapsed_s={elapsed:.6f}",
                    flush=True,
                )
            dist.barrier()

        stage1.precompiled_no_launch = True
        if internal_quant_reference is not None:
            internal_quant_reference.precompiled_no_launch = True
        if quant_setup is not None:
            quant_setup.precompiled_no_launch = True
        if transport_setup is not None:
            transport_setup.precompiled_no_launch = True
        self._stage1_quant_setup = quant_setup
        self._stage1_transport_setup = transport_setup
        self._stage1_internal_quant_reference = internal_quant_reference
        stage2.precompiled_no_launch = self.probe_stage in ("stage2", "both")
        self._probe_stage2_launcher = stage2
        return stage1

    def _compile_stage2(self):
        launcher = getattr(self, "_probe_stage2_launcher", None)
        if launcher is None:
            raise RuntimeError("Stage2 probe was not built/precompiled with Stage1")
        return launcher

    def _launch_stage1_with(
        self,
        launcher,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        run_tokens: int,
        generation: int,
        stream,
        *,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime is closed")
        launcher(
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            x_bf16.data_ptr(),
            0 if input_scale is None else input_scale.data_ptr(),
            wts.data_ptr(),
            topk_ids.data_ptr(),
            self._w1.data_ptr(),
            self._w1_scale.data_ptr(),
            run_tokens,
            generation,
            stream=stream,
        )

    def launch_stage1_quant_setup(
        self, x_bf16, wts, topk_ids, run_tokens, generation, stream
    ) -> None:
        if self._stage1_quant_setup is None:
            raise RuntimeError("quant setup launcher is unavailable")
        self._launch_stage1_with(
            self._stage1_quant_setup,
            x_bf16,
            wts,
            topk_ids,
            run_tokens,
            generation,
            stream,
        )

    def launch_stage1_internal_quant_reference(
        self, x_bf16, wts, topk_ids, run_tokens, generation, stream
    ) -> None:
        launcher = self._stage1_internal_quant_reference
        if launcher is None:
            raise RuntimeError("internal quant reference launcher is unavailable")
        self._launch_stage1_with(
            launcher,
            x_bf16,
            wts,
            topk_ids,
            run_tokens,
            generation,
            stream,
        )

    def launch_stage1_transport_setup(
        self, x_bf16, wts, topk_ids, run_tokens, generation, stream
    ) -> None:
        if self._stage1_transport_setup is None:
            raise RuntimeError("transport setup launcher is unavailable")
        self._launch_stage1_with(
            self._stage1_transport_setup,
            x_bf16,
            wts,
            topk_ids,
            run_tokens,
            generation,
            stream,
        )

    def _launch_stage2(
        self,
        run_tokens: int,
        generation: int,
        stream,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime is closed")
        self._stage2(
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            self._w2.data_ptr(),
            self._w2_scale.data_ptr(),
            generation,
            run_tokens,
            self.stage2_worker_blocks,
            self._output.data_ptr(),
            stream=stream,
        )

    def _debug_rank_local_completion(
        self,
        *,
        generation: int,
        s1_ptr,
        s2_ptr,
    ) -> dict[str, int]:
        """Summarize the logical rank-local completion words for one epoch."""

        if not self.stage2_layout.include_rank_partials:
            return {
                "rank_local_active_tokens": 0,
                "rank_local_pending_nonzero": 0,
                "rank_local_pending_nonzero_all": 0,
                "rank_local_ready_missing": 0,
                "rank_local_ready_unexpected": 0,
                "rank_reduce_queue_expected": 0,
                "rank_reduce_queue_count": 0,
                "rank_reduce_queue_tail": 0,
                "rank_reduce_queue_head": 0,
                "rank_reduce_queue_permutation_mismatch": 0,
            }

        from aiter.ops.flydsl.kernels.megamoe_tile.cco import (
            read_window_u32,
            read_window_u64,
        )

        source_capacity = self.world_size * self.mtpr
        num_valid = int(read_window_u32(s1_ptr("num_valid"), 1)[0])
        packed_sources = read_window_u32(
            s1_ptr("tile_row_source"), num_valid
        )
        active_sources = {
            int(packed) & 0x00FFFFFF
            for packed in packed_sources
            if (int(packed) & 0x00FFFFFF) < source_capacity
        }
        pending_raw = read_window_u32(
            s2_ptr("rank_token_pending"), source_capacity * 16
        )
        pending = [
            int(pending_raw[index * 16]) for index in range(source_capacity)
        ]
        ready = [
            int(value)
            for value in read_window_u64(
                s2_ptr("rank_token_ready"), source_capacity
            )
        ]
        token_items = 2 * self.mtpr
        dest_masks = [
            int(value) & 0xFF
            for value in read_window_u32(
                s2_ptr("node_dest_rank_mask"), token_items
            )
        ]
        compact = self.stage2_rail_return_schedule == "compact"
        expected_queue = (
            [index for index, mask in enumerate(dest_masks) if mask != 0]
            if compact
            else list(range(token_items))
        )
        queue_tail = int(
            read_window_u32(s2_ptr("rank_reduce_queue_tail"), 1)[0]
        )
        queue_head = (
            int(read_window_u32(s2_ptr("rank_reduce_queue_head"), 1)[0])
            if getattr(
                self,
                "stage2_node_reduce_work_schedule",
                "static_strided",
            )
            == "dynamic_head"
            else 0
        )
        queue_read_count = min(max(queue_tail, 0), token_items)
        queue_jobs = [
            int(value)
            for value in read_window_u32(
                s2_ptr("rank_reduce_queue"), queue_read_count
            )
        ]
        queue_count = (
            int(read_window_u32(s2_ptr("rank_return_count"), 3)[2])
            if compact
            else token_items
        )
        queue_mismatch = (
            int(queue_tail != len(expected_queue))
            + int(queue_count != len(expected_queue))
            + abs(len(queue_jobs) - len(expected_queue))
            + sum(
                int(actual != expected)
                for actual, expected in zip(
                    sorted(queue_jobs), expected_queue
                )
            )
        )
        return {
            "rank_local_active_tokens": len(active_sources),
            "rank_local_pending_nonzero": sum(
                int(pending[source] != 0) for source in active_sources
            ),
            "rank_local_pending_nonzero_all": sum(
                int(value != 0) for value in pending
            ),
            "rank_local_ready_missing": sum(
                int(ready[source] < generation) for source in active_sources
            ),
            "rank_local_ready_unexpected": sum(
                int(value >= generation)
                for source, value in enumerate(ready)
                if source not in active_sources
            ),
            "rank_reduce_queue_expected": len(expected_queue),
            "rank_reduce_queue_count": queue_count,
            "rank_reduce_queue_tail": queue_tail,
            "rank_reduce_queue_head": queue_head,
            "rank_reduce_queue_permutation_mismatch": queue_mismatch,
        }

    def debug_direct_tile_snapshot(self) -> dict[str, object]:
        """Read only protocol counters; never copy/hash H1 payload buffers."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_u32, read_window_u64

        torch.cuda.synchronize(self.device)
        generation = int(self._generation)
        parity = generation & 1
        base = int(self._runtime.window.local_ptr)
        s2_base = base + int(self.layout.stage2_offset)

        def s1_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return base + int(
                self.stage1_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        def s2_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return s2_base + int(
                self.stage2_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        tile_alloc = int(read_window_u32(s1_ptr("tile_alloc"), 1)[0])
        queue_tail = int(read_window_u32(s1_ptr("h1_queue_tail"), 1)[0])
        queue_jobs = [
            int(value)
            for value in read_window_u32(
                s1_ptr("h1_ready_queue"), queue_tail
            )
        ]
        queue_expected = list(range(queue_tail))
        queue_for_validation = (
            sorted(queue_jobs)
            if getattr(self._stage1, "tile_pipeline", False)
            else queue_jobs
        )
        raw_arrived = list(
            read_window_u32(
                s1_ptr("tile_row_done"), self.stage1_layout.max_route_tiles
            )
        )
        arrived = [
            int(raw_arrived[index]) if index < tile_alloc else 0
            for index in range(self.stage1_layout.max_route_tiles)
        ]
        expert_count = list(
            read_window_u32(
                s1_ptr("expert_count"), self.stage1_layout.local_experts
            )
        )
        comm_eos = list(
            read_window_u64(s1_ptr("comm_eos"), self.gpus_per_node)
        )
        stage1_errors = int(
            read_window_u32(s1_ptr("error_count", parity_indexed=False), 1)[0]
        )
        stage2_errors = int(
            read_window_u32(
                s2_ptr("stage2_error_count", parity_indexed=False), 1
            )[0]
        )
        final_done = int(
            read_window_u32(
                s2_ptr("final_done", parity_indexed=False), 1
            )[0]
        )
        return_ready = list(
            read_window_u64(
                s2_ptr("return_group_ready"), self.stage2_layout.return_groups
            )
        )
        return_consumed = int(
            read_window_u64(s2_ptr("return_consumed"), 1)[0]
        )

        ntiles = self.model_dim // 256
        scoreboard_size = 2 * self.mtpr * ntiles
        expected_all = list(
            read_window_u32(s2_ptr("node_expected"), scoreboard_size)
        )
        done_all = list(read_window_u32(s2_ptr("node_done"), scoreboard_size))
        token_items = 2 * self.mtpr
        token_done_raw = list(
            read_window_u32(s2_ptr("node_token_done"), token_items * 16)
        )
        token_done_all = [
            int(token_done_raw[index * 16]) for index in range(token_items)
        ]
        partial_done_all = (
            [
                int(value)
                for index, value in enumerate(
                    read_window_u32(
                        s2_ptr("node_partial_done"), token_items * 16
                    )
                )
                if index % 16 == 0
            ]
            if (
                self.stage2_layout.include_route_slots
                or self.stage2_layout.include_rank_partials
            )
            and self.stage2_node_reduce_schedule == "tile"
            else token_done_all
        )
        token_ready_all = list(
            read_window_u64(s2_ptr("node_token_ready"), token_items)
        )
        node_ready_all = (
            list(read_window_u64(s2_ptr("node_partial_ready"), token_items))
            if self.stage2_node_accumulation_mode
            in ("route_store", "rank_local")
            else token_ready_all
        )
        rank_local = self.stage2_node_accumulation_mode == "rank_local"
        rank_local_state = self._debug_rank_local_completion(
            generation=generation,
            s1_ptr=s1_ptr,
            s2_ptr=s2_ptr,
        )
        local_base = self.node * self.mtpr * ntiles
        expected = []
        done = []
        ready = []
        for token in range(self.mtpr):
            start = local_base + token * ntiles
            end = start + ntiles
            expected.append(min(expected_all[start:end]))
            done.append(
                min(expected_all[start:end])
                if rank_local
                else min(done_all[start:end])
            )
            token_index = self.node * self.mtpr + token
            ready.append(int(node_ready_all[token_index] >= generation))

        return {
            "comm_role_eos": [int(value) for value in comm_eos],
            "alloc_count": arrived,
            "tile_arrived": arrived,
            "tile_ready": [int(value > 0) for value in arrived],
            "tail_tile": [int(0 < value < 32) for value in arrived],
            "tail_sealed": [int(0 < value < 32) for value in arrived],
            "node_atomic_expected": expected,
            "node_atomic_done": done,
            "node_atomic_ready": ready,
            "node_ready_granularity": "token",
            "node_token_done_mismatch": (
                sum(int(value != ntiles) for value in token_done_all)
                if not rank_local
                else 0
            ),
            "node_partial_done_mismatch": (
                sum(int(value != ntiles) for value in partial_done_all)
                if not rank_local
                else 0
            ),
            "node_expected_uniform_mismatch": sum(
                int(len(set(expected_all[start : start + ntiles])) != 1)
                for start in range(0, scoreboard_size, ntiles)
            ),
            "protocol_error_count": [stage1_errors + stage2_errors],
            "generation": generation,
            "stage1_done": int(read_window_u64(s2_ptr("stage1_done"), 1)[0]),
            "compute_done": int(
                read_window_u32(s1_ptr("h1_compute_done"), 1)[0]
            ),
            "tile_alloc": tile_alloc,
            "queue_tail": queue_tail,
            "queue_permutation_mismatch": sum(
                int(actual != expected)
                for actual, expected in zip(
                    queue_for_validation, queue_expected
                )
            ),
            "queue_order_identity_mismatch": sum(
                int(actual != expected)
                for actual, expected in zip(queue_jobs, queue_expected)
            ),
            "early_full_tiles": int(
                read_window_u32(s1_ptr("h1_early_full_tiles"), 1)[0]
            ),
            "gmm_jobs_started_before_all_comm_eos": int(
                read_window_u32(
                    s1_ptr("h1_gmm_started_before_all_comm_eos"), 1
                )[0]
            ),
            "gmm_jobs_completed_before_all_comm_eos": int(
                read_window_u32(
                    s1_ptr("h1_gmm_completed_before_all_comm_eos"), 1
                )[0]
            ),
            "expert_count_sum": sum(int(value) for value in expert_count),
            "node_expected_done_mismatch": (
                sum(
                    int(int(lhs) != int(rhs))
                    for lhs, rhs in zip(expected_all, done_all)
                )
                if not rank_local
                else 0
            ),
            "node_not_ready": sum(
                int(int(value) < generation) for value in node_ready_all
            ),
            "stage1_error_count": stage1_errors,
            "stage2_error_count": stage2_errors,
            "node_accumulation_mode": self.stage2_node_accumulation_mode,
            **rank_local_state,
            "final_done": final_done,
            "final_expected": self.mtpr if rank_local else self.mtpr * ntiles,
            "return_groups_ready": sum(
                int(int(value) >= generation) for value in return_ready
            ),
            "return_consumed": return_consumed,
            "snapshot_kind": "protocol_counters_only",
            "stage1_full_fusion": bool(
                getattr(self._stage1, "full_stage1_fusion", False)
            ),
            "tile_pipeline": bool(
                getattr(self._stage1, "tile_pipeline", False)
            ),
            "tile_pipeline_instrument": bool(
                getattr(self._stage1, "tile_pipeline_instrument", False)
            ),
        }

    def debug_stage1_comm_snapshot(self) -> dict[str, object]:
        """Minimal post-Stage1 snapshot used before Stage2 is launched."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_u32, read_window_u64

        generation = int(self._generation)
        parity = generation & 1
        base = int(self._runtime.window.local_ptr)
        s2_base = base + int(self.layout.stage2_offset)

        def s1_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return base + int(
                self.stage1_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        def s2_ptr(name: str) -> int:
            return s2_base + int(self.stage2_layout.offset(name, parity=parity))

        tile_alloc = int(read_window_u32(s1_ptr("tile_alloc"), 1)[0])
        queue_tail = int(read_window_u32(s1_ptr("h1_queue_tail"), 1)[0])
        arrived = list(read_window_u32(s1_ptr("tile_row_done"), tile_alloc))
        expert_count = list(
            read_window_u32(
                s1_ptr("expert_count"), self.stage1_layout.local_experts
            )
        )
        state = {
            "generation": generation,
            "comm_role_eos": [
                int(value)
                for value in read_window_u64(
                    s1_ptr("comm_eos"), self.gpus_per_node
                )
            ],
            "stage1_done": int(read_window_u64(s2_ptr("stage1_done"), 1)[0]),
            "tile_alloc": tile_alloc,
            "tile_arrived_sum": sum(int(value) for value in arrived),
            "expert_count": [int(value) for value in expert_count],
            "expert_count_sum": sum(int(value) for value in expert_count),
            "num_valid": int(
                read_window_u32(s1_ptr("num_valid"), 1)[0]
            ),
            "h1_queue_tail": queue_tail,
            "compute_done": int(
                read_window_u32(s1_ptr("h1_compute_done"), 1)[0]
            ),
            "stage1_error_count": int(
                read_window_u32(
                    s1_ptr("error_count", parity_indexed=False), 1
                )[0]
            ),
        }
        if self.stage1_mode == "internodev1_tilepipe":
            queue_jobs = [
                int(value)
                for value in read_window_u32(
                    s1_ptr("h1_ready_queue"), queue_tail
                )
            ]
            queue_expected = list(range(queue_tail))
            state.update(
                {
                    "queue_permutation_mismatch": sum(
                        int(actual != expected)
                        for actual, expected in zip(
                            sorted(queue_jobs), queue_expected
                        )
                    ),
                    "queue_order_identity_mismatch": sum(
                        int(actual != expected)
                        for actual, expected in zip(
                            queue_jobs, queue_expected
                        )
                    ),
                    "early_full_tiles": int(
                        read_window_u32(
                            s1_ptr("h1_early_full_tiles"), 1
                        )[0]
                    ),
                    "gmm_jobs_started_before_all_comm_eos": int(
                        read_window_u32(
                            s1_ptr("h1_gmm_started_before_all_comm_eos"), 1
                        )[0]
                    ),
                    "gmm_jobs_completed_before_all_comm_eos": int(
                        read_window_u32(
                            s1_ptr("h1_gmm_completed_before_all_comm_eos"), 1
                        )[0]
                    ),
                }
            )
        staging_ready = list(
            read_window_u64(
                s1_ptr("dispatch_staging_ready"), self.mtpr
            )
        )
        quant_half_done = list(
            read_window_u64(s1_ptr("quant_half_done"), self.mtpr)
        )
        chunk_words = (
            self.stage1_layout.dispatch_chunks * self.stage1_layout.num_qp
        )
        remote_ready = list(
            read_window_u64(s1_ptr("remote_chunk_ready"), chunk_words)
        )
        remote_credit = list(
            read_window_u64(s1_ptr("remote_chunk_credit"), chunk_words)
        )
        requests = list(
            read_window_u64(s1_ptr("remote_chunk_request"), chunk_words)
        )
        state["dispatch_staging_ready_count"] = sum(
            int(int(value) >= generation) for value in staging_ready
        )
        state["quant_half_done_count"] = sum(
            int(int(value) >= generation) for value in quant_half_done
        )
        state["remote_chunk_ready_count"] = sum(
            int(int(value) >= generation) for value in remote_ready
        )
        state["remote_chunk_credit_count"] = sum(
            int(int(value) >= generation) for value in remote_credit
        )
        state["remote_request_nonzero_count"] = sum(
            int(int(value) != 0) for value in requests
        )
        if self.stage1_cco_geometry == "sparse_wqe":
            sparse_send_flags = list(
                read_window_u64(
                    s1_ptr("sparse_remote_token_ready"), self.mtpr
                )
            )
            sparse_qp_ready = list(
                read_window_u64(
                    s1_ptr("sparse_remote_qp_ready"),
                    self.stage1_layout.num_qp,
                )
            )
            sparse_requests = list(
                read_window_u64(
                    s1_ptr("sparse_remote_request"),
                    self.stage1_layout.num_qp,
                )
            )
            sparse_token_masks = [
                int(value) & 0xFFFFFFFF
                if (
                    (int(value) >> SPARSE_QP_GENERATION_SHIFT) >= generation
                )
                else 0
                for value in sparse_qp_ready
            ]
            state.update(
                {
                    "sparse_remote_token_ready_count": sum(
                        value.bit_count() for value in sparse_token_masks
                    ),
                    "sparse_remote_qp_ready_count": sum(
                        int(
                            (int(value) >> SPARSE_QP_GENERATION_SHIFT)
                            >= generation
                        )
                        for value in sparse_qp_ready
                    ),
                    "sparse_remote_qp_token_masks": sparse_token_masks,
                    "sparse_remote_request_nonzero_count": sum(
                        int(int(value) != 0) for value in sparse_requests
                    ),
                    "sparse_remote_batch_ready": int(
                        read_window_u64(
                            s1_ptr("sparse_remote_batch_ready"), 1
                        )[0]
                    ),
                    "sparse_remote_credit": int(
                        read_window_u64(
                            s1_ptr("sparse_remote_credit"), 1
                        )[0]
                    ),
                    "sparse_remote_consumed": int(
                        read_window_u32(
                            s1_ptr("sparse_remote_consumed"), 1
                        )[0]
                    ),
                    "sparse_remote_send_count": int(
                        sum(int(int(value) != 0) for value in sparse_send_flags)
                    ),
                }
            )
        if self.stage1_mode in (
            "internodev1_split128x2",
            "internodev1_split128x2_no_arrival_rmw",
            "internodev1_split128x2_rejoin",
            "internodev1_tilepipe",
            "internodev1_wave64",
            "internodev1_wave64_rejoin",
        ):
            raw_flags = list(
                read_window_u64(s1_ptr("fanout_done"), 8 * 32)
            )
            state["split_flags_ready_per_dest"] = [
                sum(
                    int(int(value) >= generation)
                    for value in raw_flags[dest * 32 : (dest + 1) * 32]
                )
                for dest in range(8)
            ]
            consumed_items = (
                self.stage1_layout.dispatch_chunks * self.stage1_layout.num_qp
            )
            state["remote_chunk_consumed"] = [
                int(value)
                for value in read_window_u32(
                    s1_ptr("remote_chunk_consumed"), consumed_items
                )
            ]
        return state

    def debug_quant_pack_snapshot(self) -> dict[str, object]:
        """Hash the complete 128x4096-byte staging plane outside timing."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        import hashlib

        from aiter.ops.flydsl.kernels.megamoe_tile.cco import (
            read_window_bytes,
            read_window_u32,
        )

        generation = int(self._generation)
        parity = generation & 1
        base = int(self._runtime.window.local_ptr)
        ptr = base + int(
            self.stage1_layout.offset("dispatch_staging", parity=parity)
        )
        nbytes = self.mtpr * self.stage1_layout.wire.record_bytes
        payload = read_window_bytes(ptr, nbytes)
        return {
            "generation": generation,
            "bytes": nbytes,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def debug_quant_core_snapshot(self) -> dict[str, object]:
        """Hash only packed FP4 payload/scales from strided staging records."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        import hashlib

        from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_bytes

        generation = int(self._generation)
        parity = generation & 1
        base = int(self._runtime.window.local_ptr)
        ptr = base + int(
            self.stage1_layout.offset("dispatch_staging", parity=parity)
        )
        wire = self.stage1_layout.wire
        raw = read_window_bytes(ptr, self.mtpr * wire.record_bytes)
        q = b"".join(
            raw[token * wire.record_bytes : token * wire.record_bytes + wire.payload_bytes]
            for token in range(self.mtpr)
        )
        scales = b"".join(
            raw[
                token * wire.record_bytes
                + wire.payload_bytes : token * wire.record_bytes
                + wire.payload_bytes
                + wire.scale_bytes
            ]
            for token in range(self.mtpr)
        )
        return {
            "generation": generation,
            "q_bytes": len(q),
            "scale_bytes": len(scales),
            "q_sha256": hashlib.sha256(q).hexdigest(),
            "scale_sha256": hashlib.sha256(scales).hexdigest(),
        }

    def debug_stage2_return_snapshot(self) -> dict[str, object]:
        """Protocol counters for the synthetic-ready return-only mode."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_u32, read_window_u64

        generation = int(self._generation)
        parity = generation & 1
        base = (
            int(self._runtime.window.local_ptr) + int(self.layout.stage2_offset)
        )

        def ptr(name: str, *, parity_indexed: bool = True) -> int:
            return base + int(
                self.stage2_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        return_ready = list(
            read_window_u64(
                ptr("return_group_ready"), self.stage2_layout.return_groups
            )
        )
        token_items = 2 * self.mtpr
        node_ready = list(
            read_window_u64(ptr("node_token_ready"), token_items)
        )
        token_done_raw = list(
            read_window_u32(ptr("node_token_done"), token_items * 16)
        )
        node_done_expected_count = sum(
            int(int(token_done_raw[index * 16]) == self.model_dim // 256)
            for index in range(token_items)
        )
        return {
            "generation": generation,
            "final_done": int(
                read_window_u32(ptr("final_done", parity_indexed=False), 1)[0]
            ),
            "final_expected": (
                self.mtpr
                if self.stage2_node_accumulation_mode == "rank_local"
                else self.mtpr * (self.model_dim // 256)
            ),
            "return_groups_ready": sum(
                int(int(value) >= generation) for value in return_ready
            ),
            "return_groups_expected": (
                self.mtpr // int(self._stage2.return_chunk_tokens)
            ),
            "return_consumed": int(
                read_window_u64(ptr("return_consumed"), 1)[0]
            ),
            "node_ready_count": sum(
                int(int(value) >= generation) for value in node_ready
            ),
            "node_ready_expected": token_items,
            "node_done_expected_count": node_done_expected_count,
            "stage2_error_count": int(
                read_window_u32(
                    ptr("stage2_error_count", parity_indexed=False), 1
                )[0]
            ),
        }

    def debug_stage2_scoreboard_snapshot(self) -> dict[str, object]:
        """Small Stage2 snapshot without copying Stage1 payload arenas."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import read_window_u32, read_window_u64

        generation = int(self._generation)
        parity = generation & 1
        arena_base = int(self._runtime.window.local_ptr)
        base = arena_base + int(self.layout.stage2_offset)

        def s1_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return arena_base + int(
                self.stage1_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        def ptr(name: str, *, parity_indexed: bool = True) -> int:
            return base + int(
                self.stage2_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        hidden_tiles = self.model_dim // 256
        scoreboard_items = 2 * self.mtpr * hidden_tiles
        expected_all = list(
            read_window_u32(ptr("node_expected"), scoreboard_items)
        )
        done_all = list(read_window_u32(ptr("node_done"), scoreboard_items))
        token_items = 2 * self.mtpr
        token_done_raw = list(
            read_window_u32(ptr("node_token_done"), token_items * 16)
        )
        token_done_all = [
            int(token_done_raw[index * 16]) for index in range(token_items)
        ]
        token_ready_all = list(
            read_window_u64(ptr("node_token_ready"), token_items)
        )
        partial_done_all = (
            [
                int(value)
                for index, value in enumerate(
                    read_window_u32(
                        ptr("node_partial_done"), token_items * 16
                    )
                )
                if index % 16 == 0
            ]
            if (
                self.stage2_layout.include_route_slots
                or self.stage2_layout.include_rank_partials
            )
            and self.stage2_node_reduce_schedule == "tile"
            else token_done_all
        )
        partial_ready_all = (
            list(read_window_u64(ptr("node_partial_ready"), token_items))
            if (
                self.stage2_layout.include_route_slots
                or self.stage2_layout.include_rank_partials
            )
            else token_ready_all
        )
        rank_local = self.stage2_node_accumulation_mode == "rank_local"
        dest_masks = (
            [
                int(value) & 0xFF
                for value in read_window_u32(
                    ptr("node_dest_rank_mask"), token_items
                )
            ]
            if rank_local
            else [1] * token_items
        )
        rank_local_state = self._debug_rank_local_completion(
            generation=generation,
            s1_ptr=s1_ptr,
            s2_ptr=ptr,
        )
        local_base = self.node * self.mtpr * hidden_tiles
        expected = []
        done = []
        ready = []
        for token in range(self.mtpr):
            start = local_base + token * hidden_tiles
            end = start + hidden_tiles
            expected_slice = expected_all[start:end]
            done_slice = done_all[start:end]
            expected.append(min(int(value) for value in expected_slice))
            done.append(
                min(int(value) for value in expected_slice)
                if rank_local
                else min(int(value) for value in done_slice)
            )
            token_index = self.node * self.mtpr + token
            ready.append(
                int(
                    dest_masks[token_index] == 0
                    or partial_ready_all[token_index] >= generation
                )
            )
        return_ready = list(
            read_window_u64(
                ptr("return_group_ready"), self.stage2_layout.return_groups
            )
        )
        return {
            "generation": generation,
            "node_expected": expected,
            "node_done": done,
            "node_ready": ready,
            "node_ready_granularity": "token",
            "node_token_done_mismatch": (
                sum(int(value != hidden_tiles) for value in token_done_all)
                if not rank_local
                else 0
            ),
            "node_partial_done_mismatch": (
                sum(int(value != hidden_tiles) for value in partial_done_all)
                if not rank_local
                else 0
            ),
            "node_expected_done_mismatch": (
                sum(
                    int(int(want) != int(got))
                    for want, got in zip(expected_all, done_all)
                )
                if not rank_local
                else 0
            ),
            "node_not_ready": sum(
                int(mask != 0 and int(value) < generation)
                for mask, value in zip(dest_masks, partial_ready_all)
            ),
            "node_route_store_not_ready": (
                sum(int(int(value) < generation) for value in token_ready_all)
                if not rank_local
                else 0
            ),
            "node_accumulation_mode": self.stage2_node_accumulation_mode,
            **rank_local_state,
            "final_done": int(
                read_window_u32(ptr("final_done", parity_indexed=False), 1)[0]
            ),
            "final_expected": (
                self.mtpr
                if self.stage2_node_accumulation_mode == "rank_local"
                else self.mtpr * hidden_tiles
            ),
            "return_groups_ready": sum(
                int(int(value) >= generation) for value in return_ready
            ),
            "return_groups_expected": (
                (
                    int(read_window_u32(ptr("rank_return_count"), 2)[1])
                    + int(self._stage2.return_chunk_tokens)
                    - 1
                )
                // int(self._stage2.return_chunk_tokens)
                if rank_local
                and self.stage2_rail_return_schedule == "compact"
                else self.mtpr // int(self._stage2.return_chunk_tokens)
            ),
            "return_consumed": int(
                read_window_u64(ptr("return_consumed"), 1)[0]
            ),
            "stage2_error_count": int(
                read_window_u32(
                    ptr("stage2_error_count", parity_indexed=False), 1
                )[0]
            ),
        }

    def poison_stage2_buffers(self) -> None:
        """Fill current-parity Stage2 payload/scoreboard storage with sentinels."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import (
            fill_window_bytes,
            write_window_u64,
        )

        generation = int(self._generation)
        parity = generation & 1
        base = (
            int(self._runtime.window.local_ptr) + int(self.layout.stage2_offset)
        )

        def region_ptr(name: str) -> tuple[int, int]:
            region = self.stage2_layout.region(name)
            return (
                base + int(self.stage2_layout.offset(name, parity=parity)),
                region.nbytes // self.stage2_layout.parity_depth,
            )

        for name, byte in (
            ("node_accumulator", 0x3F),
            ("node_done", 0x7F),
            ("node_token_done", 0x7F),
            ("remote_node_tx", 0x3F),
            ("remote_partial_rx", 0x3F),
        ):
            address, nbytes = region_ptr(name)
            fill_window_bytes(address, nbytes, byte)
        old_generation = max(0, generation - 2)
        token_ready_address, token_ready_nbytes = region_ptr(
            "node_token_ready"
        )
        write_window_u64(
            token_ready_address,
            [old_generation] * (token_ready_nbytes // 8),
        )
        if (
            self.stage2_layout.include_route_slots
            or self.stage2_layout.include_rank_partials
        ):
            partial_done_address, partial_done_nbytes = region_ptr(
                "node_partial_done"
            )
            fill_window_bytes(
                partial_done_address, partial_done_nbytes, 0x7F
            )
            partial_ready_address, partial_ready_nbytes = region_ptr(
                "node_partial_ready"
            )
            write_window_u64(
                partial_ready_address,
                [old_generation] * (partial_ready_nbytes // 8),
            )
        if self.stage2_layout.include_route_slots:
            route_slots_address, route_slots_nbytes = region_ptr("route_slots")
            fill_window_bytes(route_slots_address, route_slots_nbytes, 0x3F)
        if self.stage2_layout.include_rank_partials:
            for name, byte in (
                ("rank_accumulator", 0x3F),
                ("rank_token_pending", 0x7F),
                ("rank_return_tx_slot", 0x7F),
                ("rank_return_rx_slot", 0x7F),
                ("rank_return_count", 0x7F),
                ("rank_reduce_queue", 0x7F),
                ("rank_reduce_queue_tail", 0x7F),
                ("rank_reduce_queue_head", 0x7F),
            ):
                address, nbytes = region_ptr(name)
                fill_window_bytes(address, nbytes, byte)
            rank_ready_address, rank_ready_nbytes = region_ptr(
                "rank_token_ready"
            )
            write_window_u64(
                rank_ready_address,
                [old_generation] * (rank_ready_nbytes // 8),
            )
            queue_ready_address, queue_ready_nbytes = region_ptr(
                "rank_reduce_queue_ready"
            )
            write_window_u64(
                queue_ready_address,
                [old_generation] * (queue_ready_nbytes // 8),
            )
        self._output.fill_(1)
        torch.cuda.synchronize(self.device)

    def debug_stage2_zero_payload_snapshot(self) -> dict[str, int]:
        """Count nonzero bytes after a zero-GMM2 Stage2 launch."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        from aiter.ops.flydsl.kernels.megamoe_tile.cco import (
            read_window_bytes,
            read_window_u32,
        )

        generation = int(self._generation)
        parity = generation & 1
        arena_base = int(self._runtime.window.local_ptr)
        base = arena_base + int(self.layout.stage2_offset)

        def s1_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return arena_base + int(
                self.stage1_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        def s2_ptr(name: str, *, parity_indexed: bool = True) -> int:
            return base + int(
                self.stage2_layout.offset(
                    name, parity=parity if parity_indexed else None
                )
            )

        def nonzero(name: str, logical_bytes: int | None = None) -> int:
            region = self.stage2_layout.region(name)
            address = s2_ptr(name)
            payload = read_window_bytes(
                address,
                (
                    region.nbytes // self.stage2_layout.parity_depth
                    if logical_bytes is None
                    else int(logical_bytes)
                ),
            )
            return sum(int(value != 0) for value in payload)

        accumulator_dtype = str(
            getattr(self._stage2, "accumulator_dtype", "fp32")
        )
        accumulator_logical_bytes = (
            2 * self.mtpr * self.model_dim * 2
            if accumulator_dtype == "bf16"
            else None
        )
        rank_local_state = self._debug_rank_local_completion(
            generation=generation,
            s1_ptr=s1_ptr,
            s2_ptr=s2_ptr,
        )
        compact_rank_local = (
            self.stage2_layout.include_rank_partials
            and self.stage2_rail_return_schedule == "compact"
        )
        if compact_rank_local:
            row_bytes = self.model_dim * 2
            token_items = 2 * self.mtpr
            dest_masks = [
                int(value) & 0xFF
                for value in read_window_u32(
                    s2_ptr("node_dest_rank_mask"), token_items
                )
            ]
            accumulator_payload = read_window_bytes(
                s2_ptr("node_accumulator"), token_items * row_bytes
            )
            local_row_base = self.node * self.mtpr
            node_accumulator_nonzero = sum(
                int(value != 0)
                for token in range(self.mtpr)
                if dest_masks[local_row_base + token] != 0
                for value in accumulator_payload[
                    (local_row_base + token) * row_bytes :
                    (local_row_base + token + 1) * row_bytes
                ]
            )
            return_counts = read_window_u32(
                s2_ptr("rank_return_count"), 2
            )
            tx_rows = int(return_counts[0])
            rx_rows = int(return_counts[1])
            remote_node_tx_nonzero = nonzero(
                "remote_node_tx", tx_rows * row_bytes
            )
            remote_partial_rx_nonzero = nonzero(
                "remote_partial_rx", rx_rows * row_bytes
            )
        else:
            node_accumulator_nonzero = nonzero(
                "node_accumulator", accumulator_logical_bytes
            )
            # Direct-BF16 return reads node_accumulator as the CCO source and
            # intentionally leaves the legacy FP32->BF16 staging region alone.
            remote_node_tx_nonzero = (
                0
                if accumulator_dtype == "bf16"
                else nonzero("remote_node_tx")
            )
            remote_partial_rx_nonzero = nonzero("remote_partial_rx")

        return {
            "generation": generation,
            "node_accumulator_nonzero_bytes": node_accumulator_nonzero,
            "remote_node_tx_nonzero_bytes": remote_node_tx_nonzero,
            "remote_partial_rx_nonzero_bytes": remote_partial_rx_nonzero,
            "rank_accumulator_nonzero_bytes": (
                nonzero("rank_accumulator")
                if self.stage2_layout.include_rank_partials
                else 0
            ),
            "rank_token_pending_nonzero": rank_local_state[
                "rank_local_pending_nonzero_all"
            ],
            "rank_token_ready_missing": rank_local_state[
                "rank_local_ready_missing"
            ],
            "rank_token_ready_unexpected": rank_local_state[
                "rank_local_ready_unexpected"
            ],
            "output_nonzero_bytes": int(
                torch.count_nonzero(self._output.view(torch.uint8)).item()
            ),
        }


class MegaMoETileA4W4SparseRealGmm2(MegaMoETileA4W4CommProbe):
    """Sparse split Stage1 with the unmodified production GMM2 Stage2.

    This test-only factory retains the probe's serialized per-local-rank JIT
    compilation, but it does not install the zero-GMM2 diagnostic hooks.
    """

    diagnostic_only = False
    stage2_contains = (
        "production_weighted_gmm2+direct_lsa_atomic+done_ready+"
        "rail_return+final_combine"
    )

    def __init__(self, *args, **kwargs):
        required = {
            "probe_stage": "both",
            "stage1_mode": "internodev1_tilepipe",
            "stage1_cco_geometry": "sparse_wqe",
            "stage1_phase": "full",
            "stage2_mode": "full",
            "stage2_worker_blocks": 160,
            "stage1_transport": "sparse_wqe",
        }
        for name, value in required.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(
                    f"{name} must be {value!r} for sparse real-GMM2 validation"
                )
            kwargs[name] = value
        super().__init__(*args, **kwargs)

    def _validate_weight_capacity(self, w1, w1_scale, w2, w2_scale) -> None:
        MegaMoETileA4W4._validate_weight_capacity(
            self, w1, w1_scale, w2, w2_scale
        )

    def _build_stage2_probe_launcher(self):
        if (
            stage2_impl.gemm2_compute_v2 is _zero_gemm2_compute_v2
            or stage2_impl.issue_a_load_lds_dt
            is _noop_issue_a_load_lds_dt
        ):
            raise RuntimeError(
                "real-GMM2 factory cannot follow a zero-GMM2 probe in the "
                "same Python process"
            )
        launcher = MegaMoETileA4W4._compile_stage2(self)
        launcher.gemm2_contraction = True
        launcher.comm_probe = self.stage2_contains
        launcher.preserves_direct_lsa_atomic_epilogue = True
        if "zero_gemm2" in launcher.kernel_name:
            raise AssertionError("real-GMM2 factory selected a zero-GMM2 kernel")
        return launcher


__all__ = [
    "MegaMoETileA4W4CommProbe",
    "MegaMoETileA4W4SparseRealGmm2",
    "STAGE1_FULL_GMM1_SUFFIX",
    "STAGE1_PROBE_SUFFIX",
    "STAGE2_PROBE_SUFFIX",
    "split64x2_tilepipe_assignment",
    "split128x2_assignment",
    "tilepipe_assignment",
    "wave64_assignment",
    "cco_flush_batch_contract",
    "cco_mori64x2_contract",
    "quant_two_cta_assignment",
]
