# SPDX-License-Identifier: MIT
"""Strict EP16 two-kernel MegaMoE validation and benchmark harness.

The candidate public constructor and ``forward`` boundary match MegaMoEV2,
except that only ``quant='a4w4'`` is valid.  The hot call is always

    output_bf16 = op.forward(x_bf16, route_weights, topk_ids)

and must launch exactly two GPU kernels:

* Stage1: BF16->A4 quant + InterNodeV1 transport + receive-side scoreboard
  direct-to-expert-tile placement + GMM1 + SiLU + A4 requant
* Stage2: weighted GMM2 with a packed-BF16 direct LSA atomic epilogue into the
  source-aligned node accumulator + EP return/combine

JIT, packed-weight construction and workspace construction are primed before
timing.  For the candidate, input quantization is part of every Stage1 launch.

The production baseline follows ``bench_mega_moe_v2.py`` at the same K3 shape:
standalone BF16-to-A4 quantization, MORI A4 dispatch, ``fused_moe`` and MORI
combine.  No benchmark-side route expansion or intermediate requant is added.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.megamoe_tile.markers import (
    profiler_pause,
    profiler_resume,
    roctx_range,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    IterationTiming,
    MoriBaselinePath,
    SharedInputs,
    _comparison_metrics,
    _global_max_timing,
    _sample_stats,
    _setup_dist,
    _shared_inputs,
)


TARGET_STAGE1_SYMBOL = r".*megamoe_tile_ep16_stage1.*"
TARGET_STAGE2_SYMBOL = r".*megamoe_tile_ep16_stage2.*"
COMPARISON_CASE_LABEL = "TPR128_TopK16_E896_H7168_I3072_EP16_A4W4"
DIRECT_TILE_STAGE1_CONTRACT = {
    "dispatch": "scoreboard_direct_to_expert_tile",
    "receive_comm_roles": 8,
    "cross_node_comm_roles": 1,
    "intra_node_comm_roles": 7,
    "allocation_counter": "alloc_count",
    "arrival_counter": "tile_arrived",
    "eos_tail": True,
    "uses_rank_inbox": False,
    "uses_source_activation_inbox": True,
    "uses_group_sort": False,
}
DIRECT_TILE_STAGE2_CONTRACT = {
    "epilogue": "direct_lsa_atomic_source_aligned_node_accumulator",
    "node_accumulator_dtype": "bf16",
    "node_ready_granularity": "token",
    "node_accumulation_mode": "direct_atomic",
    "uses_rank_partial": False,
    "uses_node_scan": False,
    "uses_external_reduce_kernel": False,
}
ROUTE_STORE_STAGE2_CONTRACT = {
    **DIRECT_TILE_STAGE2_CONTRACT,
    "epilogue": "source_aligned_route_slot_store_then_register_reduce",
    "node_accumulation_mode": "route_store",
}
RANK_LOCAL_STAGE2_CONTRACT = {
    **DIRECT_TILE_STAGE2_CONTRACT,
    "epilogue": "rank_local_atomic_then_peer_pull_node_reduce",
    "node_accumulation_mode": "rank_local",
    "uses_rank_partial": True,
    "uses_node_scan": True,
}
PUBLIC_CONSTRUCTOR_ARGUMENTS = (
    "rank",
    "world_size",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "quant",
    "w1",
    "w1_scale",
    "w2",
    "w2_scale",
    "max_tok_per_rank",
    "mega_scheme",
    "swiglu_limit",
)


class MegaMoETwoKernelLike(Protocol):
    def forward(
        self,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
        slice_output: bool = True,
    ) -> torch.Tensor: ...


def _load_factory(spec: str) -> Callable[..., MegaMoETwoKernelLike]:
    if ":" not in spec:
        raise ValueError("operator factory must be module:callable")
    module_name, callable_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), callable_name)


def _validate_factory_signature(factory: Callable[..., object]) -> None:
    """Catch accidental API drift before allocating the EP16 operator."""

    target = factory.__init__ if inspect.isclass(factory) else factory
    signature = inspect.signature(target)
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return
    missing = [name for name in PUBLIC_CONSTRUCTOR_ARGUMENTS if name not in parameters]
    if missing:
        raise TypeError(
            "two-kernel constructor must be MegaMoEV2-compatible; missing "
            f"arguments {missing}; signature={signature}"
        )


def _validate_rank_balanced_routes(
    shared: SharedInputs, shape: BenchmarkShape
) -> None:
    owners = torch.div(
        shared.topk_ids, shape.local_experts, rounding_mode="floor"
    ).sort(dim=1).values
    expected = torch.arange(
        shape.ep_size, dtype=owners.dtype, device=owners.device
    ).view(1, shape.ep_size)
    if torch.count_nonzero(owners != expected).item():
        raise AssertionError(
            "rank-balanced-hot/topk16 must select exactly one route on every EP rank"
        )


def _validate_architecture_manifest(
    manifest: object, expected: dict[str, object], label: str
) -> None:
    if not isinstance(manifest, dict):
        raise TypeError(f"{label} launcher must expose architecture_contract dict")
    mismatch = {
        key: (manifest.get(key, "<missing>"), value)
        for key, value in expected.items()
        if manifest.get(key, "<missing>") != value
    }
    if mismatch:
        detail = ", ".join(
            f"{key}={got!r} (expected {want!r})"
            for key, (got, want) in mismatch.items()
        )
        raise AssertionError(f"{label} architecture contract mismatch: {detail}")


def _validate_direct_tile_operator(operator: object) -> None:
    stage1 = getattr(operator, "_stage1", None)
    stage2 = getattr(operator, "_stage2", None)
    if stage1 is None or stage2 is None:
        raise TypeError("candidate must expose compiled _stage1/_stage2 launchers")
    if getattr(stage1, "gemm1_contraction", True) is not True:
        raise AssertionError("candidate Stage1 does not contain real GMM1")
    if getattr(stage2, "gemm2_contraction", True) is not True:
        raise AssertionError("candidate Stage2 does not contain real GMM2")
    if "zero_gemm2" in getattr(stage2, "kernel_name", ""):
        raise AssertionError("candidate selected a zero-GMM2 diagnostic kernel")
    stage1_manifest = getattr(stage1, "architecture_contract", None)
    stage2_manifest = getattr(stage2, "architecture_contract", None)
    if isinstance(stage1_manifest, dict) and stage1_manifest.get(
        "allocation_counter"
    ) == stage1_manifest.get("arrival_counter"):
        raise AssertionError("alloc_count and tile_arrived must be distinct counters")
    _validate_architecture_manifest(
        stage1_manifest, DIRECT_TILE_STAGE1_CONTRACT, "Stage1"
    )
    node_accumulation_mode = getattr(
        stage2, "node_accumulation_mode", "direct_atomic"
    )
    if node_accumulation_mode == "route_store":
        expected_stage2 = dict(ROUTE_STORE_STAGE2_CONTRACT)
    elif node_accumulation_mode == "rank_local":
        expected_stage2 = dict(RANK_LOCAL_STAGE2_CONTRACT)
    else:
        expected_stage2 = dict(DIRECT_TILE_STAGE2_CONTRACT)
    if node_accumulation_mode in ("route_store", "rank_local"):
        for name in (
            "node_reduce_blocks",
            "node_reduce_vec_bytes",
            "node_reduce_schedule",
            "node_reduce_load_schedule",
        ):
            value = getattr(stage2, name, "<missing>")
            if value == "<missing>":
                raise AssertionError(
                    f"Stage2 {node_accumulation_mode} launcher is missing {name}"
                )
            expected_stage2[name] = value
    _validate_architecture_manifest(stage2_manifest, expected_stage2, "Stage2")


def _snapshot_values(snapshot: dict[str, object], name: str) -> list[int]:
    if name not in snapshot:
        raise KeyError(f"direct-tile debug snapshot is missing {name!r}")
    value = snapshot[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"snapshot {name!r} must be a tensor/list/tuple")
    return [int(item) for item in value]


def _validate_direct_tile_debug_snapshot(
    snapshot: dict[str, object],
    *,
    expected_routes: int = 2048,
    expected_tokens: int = 128,
    block_m: int = 32,
    expect_rank_token_completion: bool = True,
) -> None:
    """Validate a completed prime epoch; this always runs outside timing."""

    eos = _snapshot_values(snapshot, "comm_role_eos")
    if len(eos) != 8 or any(value == 0 for value in eos):
        raise AssertionError("prime epoch requires EOS from exactly 8 communication roles")
    allocated = _snapshot_values(snapshot, "alloc_count")
    arrived = _snapshot_values(snapshot, "tile_arrived")
    ready = _snapshot_values(snapshot, "tile_ready")
    tail = _snapshot_values(snapshot, "tail_tile")
    tail_sealed = _snapshot_values(snapshot, "tail_sealed")
    lengths = {len(allocated), len(arrived), len(ready), len(tail), len(tail_sealed)}
    if len(lengths) != 1 or not allocated:
        raise AssertionError("tile scoreboard arrays must be non-empty and equal length")
    if sum(allocated) != expected_routes:
        raise AssertionError(
            f"alloc_count sum={sum(allocated)}, expected {expected_routes} routes"
        )
    active_tiles = 0
    for tile, (alloc, seen, is_ready, is_tail, sealed) in enumerate(
        zip(allocated, arrived, ready, tail, tail_sealed)
    ):
        if alloc == 0:
            if seen or is_ready or is_tail or sealed:
                raise AssertionError(f"tile {tile}: inactive tile has published state")
            continue
        active_tiles += 1
        if not 0 < alloc <= block_m:
            raise AssertionError(
                f"tile {tile}: alloc_count={alloc} is outside [1,{block_m}]"
            )
        if seen != alloc:
            raise AssertionError(
                f"tile {tile}: tile_arrived={seen}, alloc_count={alloc}"
            )
        if not is_ready:
            raise AssertionError(f"tile {tile}: completed tile is not ready")
        if alloc < block_m and (not is_tail or not sealed):
            raise AssertionError(
                f"tile {tile}: partial tail was not EOS-sealed before readiness"
            )
    if active_tiles == 0:
        raise AssertionError("prime epoch did not allocate any expert tile")
    node_expected = _snapshot_values(snapshot, "node_atomic_expected")
    node_done = _snapshot_values(snapshot, "node_atomic_done")
    node_ready = _snapshot_values(snapshot, "node_atomic_ready")
    if not (
        len(node_expected)
        == len(node_done)
        == len(node_ready)
        == expected_tokens
    ):
        raise AssertionError("Stage2 node-atomic scoreboards must have 128 token entries")
    if node_done != node_expected or any(value == 0 for value in node_ready):
        raise AssertionError(
            "Stage2 node accumulator was published before all contributions"
        )
    if int(snapshot.get("node_expected_uniform_mismatch", 0)) != 0:
        raise AssertionError("Stage2 node expected count differs across hidden tiles")
    if int(snapshot.get("node_expected_done_mismatch", 0)) != 0:
        raise AssertionError("Stage2 node tile contributor count did not complete")
    if int(snapshot.get("node_not_ready", 0)) != 0:
        raise AssertionError("Stage2 node readiness did not complete")
    if snapshot.get("node_ready_granularity") != "token":
        raise AssertionError("Stage2 must publish whole-token readiness")
    if int(snapshot.get("node_token_done_mismatch", -1)) != 0:
        raise AssertionError("Stage2 token tile-count did not complete")
    if (
        snapshot.get("node_accumulation_mode") == "rank_local"
        and expect_rank_token_completion
    ):
        if int(snapshot.get("rank_local_active_tokens", 0)) == 0:
            raise AssertionError("rank-local Stage2 produced no active tokens")
        if int(snapshot.get("rank_local_pending_nonzero", -1)) != 0:
            raise AssertionError("rank-local token completion did not drain")
        if int(snapshot.get("rank_local_ready_missing", -1)) != 0:
            raise AssertionError("rank-local token readiness did not complete")
    errors = _snapshot_values(snapshot, "protocol_error_count")
    if len(errors) != 1 or errors[0] != 0:
        raise AssertionError(f"device protocol_error_count={errors}")
    if snapshot.get("tile_pipeline", False):
        if not snapshot.get("stage1_full_fusion", False):
            raise AssertionError("tile pipeline must retain the real fused GMM1")
        expected_jobs = active_tiles * 24
        if (
            int(snapshot.get("queue_tail", -1)) != expected_jobs
            or int(snapshot.get("compute_done", -1)) != expected_jobs
            or int(snapshot.get("queue_permutation_mismatch", -1)) != 0
        ):
            raise AssertionError(
                "tile pipeline queue is incomplete, duplicated, or unconsumed"
            )
        if snapshot.get("tile_pipeline_instrument", False):
            early_tiles = int(snapshot.get("early_full_tiles", 0))
            started = int(
                snapshot.get("gmm_jobs_started_before_all_comm_eos", 0)
            )
            completed = int(
                snapshot.get("gmm_jobs_completed_before_all_comm_eos", 0)
            )
            valid_overlap = (
                0 < early_tiles <= active_tiles
                and 0 < completed <= started <= expected_jobs
                if expected_jobs > 0
                else early_tiles == started == completed == 0
            )
            if not valid_overlap:
                raise AssertionError(
                    "tile pipeline did not demonstrate valid communication/GMM1 overlap"
                )


class MoriBf16A4W4BaselinePath(MoriBaselinePath):
    """MORI reference measured from the same BF16 public input boundary."""

    name = "mori_bf16_a4w4_baseline"
    stage_names = ("bf16_to_a4", "dispatch", "local_a4w4", "combine")

    def __init__(self, *args, expand_local_routes: bool = False, **kwargs):
        self.expand_local_routes = bool(expand_local_routes)
        super().__init__(*args, **kwargs)
        self._iteration_a_quant = None
        self._iteration_a_scale = None

    def _input_quant_stage(self):
        from aiter.ops.quant import per_1x32_f4_quant

        self._iteration_a_quant, self._iteration_a_scale = per_1x32_f4_quant(
            self.shared.x, shuffle=False
        )
        return self._iteration_a_quant

    def _dispatch_stage(self):
        if self._iteration_a_quant is None or self._iteration_a_scale is None:
            raise RuntimeError("BF16 input has not been quantized for this iteration")
        self._dispatch = self.op.dispatch(
            self._iteration_a_quant,
            self.shared.route_weights,
            self._iteration_a_scale,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=8,
        )
        return self._dispatch

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        timer.stage("bf16_to_a4", self._input_quant_stage)
        timer.stage("dispatch", self._dispatch_stage)
        timer.stage("local_a4w4", self._compute_stage)
        return timer.stage("combine", self._combine_stage)

    def prime_and_check(self) -> torch.Tensor:
        self._input_quant_stage()
        return super().prime_and_check()


class MoriFusedMoeBaselinePath:
    """Production A4 quant + MORI dispatch + fused_moe + combine baseline.

    This mirrors ``bench_mega_moe_v2.py::mori_body`` at the K3 EP16 shape.
    Quantization happens before communication so MORI transports packed FP4
    activations and E8M0 scales.  The dispatch tuple is passed directly to
    ``fused_moe``; route sorting, scale-layout conversion and the inter-stage
    representation are owned by ``fused_moe`` rather than benchmark glue.
    """

    name = "quant_mori_a4_dispatch_fused_moe_combine"
    stage_names = ("quant_dispatch_fused_moe_combine",)
    full_stage_field = stage_names[0]

    def __init__(
        self,
        shape: BenchmarkShape,
        shared: SharedInputs,
        rank: int,
        world: int,
        *,
        valid_recv: int,
        combine_quant_type: str = "none",
    ):
        import aiter
        import mori
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.quant import dynamic_per_group_scaled_quant

        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        config = mori.ops.EpDispatchCombineConfig(
            data_type=shared.a_quant.dtype,
            rank=rank,
            world_size=world,
            hidden_dim=shape.hidden,
            scale_dim=shape.hidden // 32,
            scale_type_size=1,
            max_token_type_size=torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=max(128, shape.tokens),
            max_total_recv_tokens=0,
            num_experts_per_rank=shape.local_experts,
            num_experts_per_token=shape.topk,
            warp_num_per_block=8,
            block_num=256,
            kernel_type=kernel_type,
            rdma_block_num=128,
            gpu_per_node=shape.gpus_per_node,
            quant_type=combine_quant_type,
        )
        self.shape = shape
        self.shared = shared
        self.rank = rank
        self.valid_recv = int(valid_recv)
        self.combine_quant_type = str(combine_quant_type)
        self.op = mori.ops.EpDispatchCombineOp(config)
        self._fused_moe = fused_moe
        self._activation = aiter.ActivationType.Silu
        self._quant_type = aiter.QuantType.per_1x32
        self._quant_op = dynamic_per_group_scaled_quant
        self._quant_q = shared.a_quant
        self._quant_scale = shared.a_scale
        self._gate_mode = GateMode.SEPARATED.value

    def _forward(self, *, validate_recv: bool = False) -> torch.Tensor:
        self._quant_op(
            self._quant_q,
            self.shared.x,
            self._quant_scale,
            32,
            shuffle_scale=False,
        )
        dispatched, recv_weights, recv_scales, recv_ids, recv_tokens = self.op.dispatch(
            self._quant_q,
            self.shared.route_weights,
            self._quant_scale,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=8,
        )
        if validate_recv:
            actual_recv = int(recv_tokens.item())
            if actual_recv != self.valid_recv:
                raise AssertionError(
                    f"fused_moe baseline recv_tokens={actual_recv}, "
                    f"expected {self.valid_recv}"
                )
        weights = self.shared.prepared_weights
        local_out = self._fused_moe(
            dispatched,
            weights.w1,
            weights.w2,
            recv_weights,
            recv_ids,
            self.shared.local_expert_mask,
            activation=self._activation,
            quant_type=self._quant_type,
            doweight_stage1=False,
            w1_scale=weights.w1_scale,
            w2_scale=weights.w2_scale,
            a1_scale=recv_scales,
            num_local_tokens=recv_tokens,
            dtype=torch.bfloat16,
            swiglu_limit=0.0,
            gate_mode=self._gate_mode,
        )
        result = self.op.combine(
            local_out,
            None,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=4,
        )
        return result[0] if isinstance(result, tuple) else result

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        return timer.stage(self.stage_names[0], self._forward)

    def prime_and_check(self) -> torch.Tensor:
        output = self._forward(validate_recv=True)
        torch.cuda.synchronize(output.device)
        if output.shape[0] < self.shape.tokens:
            raise AssertionError("fused_moe baseline output is shorter than local tokens")
        if not torch.isfinite(output[: self.shape.tokens].float()).all().item():
            raise AssertionError("fused_moe baseline produced non-finite output")
        return output


class TwoKernelCandidatePath:
    name = "ep16_two_kernel_candidate"
    stage_names = ("stage1_stage2_forward",)
    full_stage_field = stage_names[0]

    def __init__(
        self,
        operator: MegaMoETwoKernelLike,
        shared: SharedInputs,
        shape: BenchmarkShape,
        device: torch.device,
    ):
        self.operator = operator
        self.shared = shared
        self.shape = shape
        self.device = device

    def _forward(self) -> torch.Tensor:
        # This exact public call is the contract. In particular, do not use a
        # forward_prequant path or materialize A4 input in the harness.
        return self.operator.forward(
            self.shared.x,
            self.shared.route_weights,
            self.shared.topk_ids,
        )

    def _validate_output(self, output: torch.Tensor) -> None:
        expected = (self.shape.tokens, self.shape.hidden)
        if tuple(output.shape) != expected:
            raise AssertionError(
                f"candidate output shape={tuple(output.shape)}, expected {expected}"
            )
        if output.dtype is not torch.bfloat16:
            raise AssertionError(
                f"candidate output dtype={output.dtype}, expected torch.bfloat16"
            )
        if output.device != self.device:
            raise AssertionError(
                f"candidate output device={output.device}, expected {self.device}"
            )
        if not torch.isfinite(output.float()).all().item():
            raise AssertionError("candidate produced a non-finite output")

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        return timer.stage("stage1_stage2_forward", self._forward)

    def prime_and_check(self) -> torch.Tensor:
        output = self._forward()
        torch.cuda.synchronize(self.device)
        self._validate_output(output)
        snapshot = getattr(self.operator, "debug_direct_tile_snapshot", None)
        if snapshot is not None:
            state = snapshot()
            self.prime_debug_snapshot = state
            print(
                "MEGAMOE_DIRECT_TILE_SNAPSHOT "
                + json.dumps(
                    {"rank": dist.get_rank(), "phase": "prime", "state": state},
                    sort_keys=True,
                ),
                flush=True,
            )
            _validate_direct_tile_debug_snapshot(state)
        return output

    def close(self) -> None:
        close = getattr(self.operator, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class PathResult:
    prime: torch.Tensor
    timed: torch.Tensor
    local_samples: list[IterationTiming]
    rank_max_samples: list[IterationTiming]
    # Candidate-only bring-up snapshots.  These live on CPU so retaining an
    # iteration does not launch a device clone kernel between the two-kernel
    # forwards.  The generation comes from the operator rather than being
    # inferred, which keeps the diagnostic meaningful when warmup > 0.
    debug_generation_outputs: tuple[tuple[int, torch.Tensor], ...] = ()


def _run_path(
    path,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> PathResult:
    prime = path.prime_and_check().clone()
    torch.cuda.synchronize(device)
    debug_snapshot = getattr(
        getattr(path, "operator", None), "debug_direct_tile_snapshot", None
    )
    debug_generation_outputs: list[tuple[int, torch.Tensor]] = []
    generation = int(getattr(getattr(path, "operator", None), "_generation", 1))
    # D2H after synchronization is deliberately outside every HIP timing
    # event and emits no GPU kernel. Keep prime + the first three measured
    # outputs for same-path and MORI-vs-candidate generation comparisons.
    debug_generation_outputs.append((generation, prime.detach().cpu()))
    dist.barrier()
    timer = HipStageTimer(device, path.stage_names)

    # The barrier is outside both HIP-event and host timing. It is required by
    # the bring-up epoch protocol and makes every round start from aligned EP16
    # ranks. No numerical collective executes in the hot region.
    for _ in range(warmup):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        path.run_iteration(timer)
        timer.finish_iteration()

    local_samples: list[IterationTiming] = []
    output = None
    # With MEGAMOE_TILE_PROFILE_REGIONS=1, rocprof records only the following
    # timed forwards. Constructor/JIT/warmup and post-loop correctness kernels
    # cannot be mistaken for hot-path launches by the exact-count auditor.
    profiler_resume()
    for iteration in range(iterations):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        with roctx_range(f"MEGAMOE_EP16_TWO_KERNEL_TIMED_{iteration}"):
            output = path.run_iteration(timer)
            local = timer.finish_iteration()
        local_samples.append(local)
        if len(debug_generation_outputs) < 4:
            generation = int(
                getattr(getattr(path, "operator", None), "_generation", iteration + 2)
            )
            # finish_iteration synchronized the device. This is a host-side
            # D2H snapshot after the timed range, not another GPU operator.
            debug_generation_outputs.append((generation, output.detach().cpu()))
    profiler_pause()

    if output is None:
        raise AssertionError("timed loop produced no output")
    if debug_snapshot is not None:
        timed_state = debug_snapshot()
        path.timed_debug_snapshot = timed_state
        print(
            "MEGAMOE_DIRECT_TILE_SNAPSHOT "
            + json.dumps(
                {"rank": dist.get_rank(), "phase": "timed", "state": timed_state},
                sort_keys=True,
            ),
            flush=True,
        )
    # Keep rank-max collectives and the output copy after all timed rounds.
    rank_max = [
        _global_max_timing(sample, path.stage_names) for sample in local_samples
    ]
    return PathResult(
        prime,
        output.clone(),
        local_samples,
        rank_max,
        tuple(debug_generation_outputs),
    )


def _debug_iteration_output_matrix(
    result: PathResult, *, rank: int, path_name: str
) -> list[dict[str, float | int | str]]:
    """Compare the retained prime/timed outputs in every pairwise direction."""

    retained = result.debug_generation_outputs
    if len(retained) < 4:
        return []
    comparisons: list[dict[str, float | int | str]] = []
    for left in range(len(retained)):
        left_generation, left_output = retained[left]
        for right in range(left + 1, len(retained)):
            right_generation, right_output = retained[right]
            comparisons.append(
                _comparison_metrics(
                    left_output,
                    right_output,
                    rank=rank,
                    label=(
                        f"{path_name}_debug_gen{left_generation}"
                        f"_vs_gen{right_generation}"
                    ),
                )
            )
    print(
        "MEGAMOE_ITERATION_OUTPUT_MATRIX "
        + json.dumps(
            {
                "rank": rank,
                "path": path_name,
                "generations": [generation for generation, _ in retained],
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return comparisons


def _path_summary(path, result: PathResult, tail_iterations: int) -> dict[str, object]:
    tail = result.rank_max_samples[-tail_iterations:]
    local_tail = result.local_samples[-tail_iterations:]
    fields = (*path.stage_names, "gpu_e2e", "host_critical")

    def values(field: str) -> list[float]:
        if field == "gpu_e2e":
            return [sample.gpu_e2e_us for sample in tail]
        if field == "host_critical":
            return [sample.host_critical_us for sample in tail]
        return [sample.stage_us[field] for sample in tail]

    def local_values(field: str) -> list[float]:
        if field == "gpu_e2e":
            return [sample.gpu_e2e_us for sample in local_tail]
        if field == "host_critical":
            return [sample.host_critical_us for sample in local_tail]
        return [sample.stage_us[field] for sample in local_tail]

    return {
        "path": path.name,
        "full_stage_field": getattr(path, "full_stage_field", path.stage_names[0]),
        "tail_iterations": tail_iterations,
        "tail_rank_max_stats_us": {
            field: _sample_stats(values(field)) for field in fields
        },
        # Each rank emits this field in the final gathered payload.  Averaging
        # its per-rank means gives the true all-card/sample mean; it is distinct
        # from the critical-path rank-max headline above.
        "tail_local_rank_stats_us": {
            field: _sample_stats(local_values(field)) for field in fields
        },
        "rank_max_gpu_e2e_us": [
            sample.gpu_e2e_us for sample in result.rank_max_samples
        ],
    }


def _global_max(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _build_candidate(
    factory_spec: str,
    *,
    shape: BenchmarkShape,
    shared: SharedInputs,
    rank: int,
    device: torch.device,
    stage1_transport: str | None = None,
) -> TwoKernelCandidatePath:
    factory = _load_factory(factory_spec)
    _validate_factory_signature(factory)
    weights = shared.prepared_weights
    extra = (
        {"stage1_transport": stage1_transport}
        if stage1_transport is not None
        else {}
    )
    operator = factory(
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
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
        **extra,
    )
    try:
        _validate_direct_tile_operator(operator)
        if not callable(getattr(operator, "forward", None)):
            raise TypeError(
                "two-kernel operator must expose forward(x_bf16, wts, topk_ids)"
            )
        quant = getattr(operator, "quant_mode", "a4w4")
        if quant != "a4w4":
            raise ValueError(f"candidate quant mode must be a4w4, got {quant!r}")
    except Exception:
        close = getattr(operator, "close", None)
        if close is not None:
            close()
        raise
    return TwoKernelCandidatePath(operator, shared, shape, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator-factory",
        default="aiter.ops.flydsl.kernels.megamoe_tile:HierarchicalMegaMoEV2",
    )
    parser.add_argument("--paths", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument(
        "--order",
        choices=("baseline-first", "candidate-first"),
        default="baseline-first",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tail-iters", type=int, default=20)
    parser.add_argument("--rel-l2-threshold", type=float, default=5.0e-2)
    parser.add_argument(
        "--route-pattern",
        choices=(
            "rank-balanced-hot",
            "paired-rank-half-remote",
            "paired-rank-local-only",
        ),
        default="rank-balanced-hot",
    )
    parser.add_argument("--direct-packed-weights", action="store_true")
    parser.add_argument(
        "--candidate-stage1-transport",
        choices=("default", "sparse_wqe"),
        default="default",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be >=1")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1, iters]")

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
    contract = {
        "case_label": COMPARISON_CASE_LABEL,
        "shape": shape.__dict__,
        "route_pattern": args.route_pattern,
        "direct_packed_weights": args.direct_packed_weights,
        "candidate_stage1_transport": args.candidate_stage1_transport,
        "order": args.order,
        "quant": "a4w4",
        "public_forward": "forward(x_bf16, wts, topk_ids) -> local_bf16",
        "baseline": "bf16_to_a4+mori_a4_dispatch+fused_moe_a4w4_silu+mori_combine",
        "baseline_external_requant": False,
        "stage1": (
            "bf16_to_a4+internodev1_transport+scoreboard_direct_expert_tile+"
            "gmm1+silu+a4_requant"
        ),
        "stage2": (
            "weighted_gmm2+direct_lsa_atomic_bf16x2_source_aligned_node_accum+"
            "return+combine"
        ),
        "inter_node_payload": "once_per_source_token_per_destination_node",
        "inter_node_return": "once_per_source_token_per_remote_node_partial",
        "stage1_architecture_assert": DIRECT_TILE_STAGE1_CONTRACT,
        "stage2_architecture_assert": DIRECT_TILE_STAGE2_CONTRACT,
        "hot_kernel_launches_per_iteration": 2,
        "stage1_kernel_regex": TARGET_STAGE1_SYMBOL,
        "stage2_kernel_regex": TARGET_STAGE2_SYMBOL,
        "barrier": "gloo_before_each_iteration_outside_timing",
        "statistic": "per_iteration_ep16_rank_max_then_tail_mean_p50_p95",
    }
    if args.plan_only:
        print("MEGAMOE_EP16_TWO_KERNEL_PLAN " + json.dumps(contract, sort_keys=True))
        return 0
    # Both paths use GPU-side EP communication. Gloo is coordination-only; do
    # not create an RCCL group that can select an unrelated RoCE rail.
    needs_mori = args.paths in ("baseline", "both")
    rank, world, _local_rank, device = _setup_dist(needs_mori=needs_mori)
    if world != shape.ep_size:
        raise ValueError(f"strict benchmark requires world=16, got {world}")
    # Pause as early as possible when region profiling is requested. The
    # matching resume/pause pair lives immediately around each timed loop.
    profiler_pause()
    shared = _shared_inputs(
        shape,
        rank,
        world,
        device,
        route_pattern=args.route_pattern,
        direct_packed_weights=args.direct_packed_weights,
    )
    if args.route_pattern == "rank-balanced-hot":
        _validate_rank_balanced_routes(shared, shape)

    paths = []
    if args.paths in ("baseline", "both"):
        paths.append(
            MoriFusedMoeBaselinePath(
                shape,
                shared,
                rank,
                world,
                valid_recv=(
                    shape.tokens * world
                    if args.route_pattern == "rank-balanced-hot"
                    else shape.tokens * world // 2
                ),
            )
        )
    if args.paths in ("candidate", "both"):
        paths.append(
            _build_candidate(
                args.operator_factory,
                shape=shape,
                shared=shared,
                rank=rank,
                device=device,
                stage1_transport=(
                    None
                    if args.candidate_stage1_transport == "default"
                    else args.candidate_stage1_transport
                ),
            )
        )
    if args.paths == "both" and args.order == "candidate-first":
        paths.reverse()

    results: dict[str, PathResult] = {}
    local_summaries = []
    for path in paths:
        result = _run_path(
            path,
            device,
            warmup=args.warmup,
            iterations=args.iters,
        )
        results[path.name] = result
        local_summaries.append(_path_summary(path, result, args.tail_iters))

    diagnostics = []
    failed = False
    for path in paths:
        result = results[path.name]
        diagnostics.append(
            _comparison_metrics(
                result.prime,
                result.timed,
                rank=rank,
                label=f"{path.name}_prime_vs_timed",
            )
        )
        # Candidate debug mode retains prime + three measured generations on
        # CPU and reports the complete 4x4 upper triangle. With warmup=0 this
        # explicitly includes gen1-vs-gen3 and gen2-vs-gen4 (same parity), in
        # addition to adjacent cross-parity comparisons.
        diagnostics.extend(
            _debug_iteration_output_matrix(
                result,
                rank=rank,
                path_name=path.name,
            )
        )
    comparison_rel_l2 = None
    if args.paths == "both":
        baseline = results["quant_mori_a4_dispatch_fused_moe_combine"].timed
        candidate = results["ep16_two_kernel_candidate"].timed
        comparison = _comparison_metrics(
            baseline,
            candidate,
            rank=rank,
            label="two_kernel_candidate_vs_quant_mori_a4_fused_moe",
        )
        diagnostics.append(comparison)
        baseline_generations = results[
            "quant_mori_a4_dispatch_fused_moe_combine"
        ].debug_generation_outputs
        candidate_generations = results[
            "ep16_two_kernel_candidate"
        ].debug_generation_outputs
        for sample, ((_, baseline_output), (candidate_generation, candidate_output)) in enumerate(
            zip(baseline_generations, candidate_generations)
        ):
            diagnostics.append(
                _comparison_metrics(
                    baseline_output,
                    candidate_output,
                    rank=rank,
                    label=(
                        f"two_kernel_candidate_vs_mori_sample{sample}_"
                        f"candidate_gen{candidate_generation}"
                    ),
                )
            )
        comparison_rel_l2 = _global_max(float(comparison["rel_l2"]))
    local_worst_rel_l2 = max(
        (float(item["rel_l2"]) for item in diagnostics), default=0.0
    )
    all_checks_rel_l2 = _global_max(local_worst_rel_l2)
    failed = all_checks_rel_l2 >= args.rel_l2_threshold

    gathered_summaries = [None] * world if rank == 0 else None
    gathered_diagnostics = [None] * world if rank == 0 else None
    dist.gather_object(local_summaries, gathered_summaries, dst=0)
    dist.gather_object(diagnostics, gathered_diagnostics, dst=0)
    if rank == 0:
        timing_comparison = None
        tail_all_rank_sample_mean_us = None
        if args.paths == "both":
            summary_by_name = {row["path"]: row for row in local_summaries}
            baseline_name = "quant_mori_a4_dispatch_fused_moe_combine"
            candidate_name = "ep16_two_kernel_candidate"
            baseline_field = summary_by_name[baseline_name]["full_stage_field"]
            candidate_field = summary_by_name[candidate_name]["full_stage_field"]
            baseline_stats = summary_by_name[baseline_name][
                "tail_rank_max_stats_us"
            ][baseline_field]
            candidate_stats = summary_by_name[candidate_name][
                "tail_rank_max_stats_us"
            ][candidate_field]
            timing_comparison = {
                key: {
                    "baseline_us": baseline_stats[key],
                    "candidate_us": candidate_stats[key],
                    "candidate_change_pct": 100.0
                    * (candidate_stats[key] / baseline_stats[key] - 1.0),
                }
                for key in ("mean", "p50", "p95")
            }
            tail_all_rank_sample_mean_us = {}
            for path_name, field in (
                (baseline_name, baseline_field),
                (candidate_name, candidate_field),
            ):
                per_rank = [
                    next(
                        row for row in rank_rows if row["path"] == path_name
                    )["tail_local_rank_stats_us"][field]["mean"]
                    for rank_rows in gathered_summaries
                ]
                tail_all_rank_sample_mean_us[path_name] = sum(per_rank) / world
            timing_comparison["all_rank_mean"] = {
                "baseline_us": tail_all_rank_sample_mean_us[baseline_name],
                "candidate_us": tail_all_rank_sample_mean_us[candidate_name],
                "candidate_change_pct": 100.0
                * (
                    tail_all_rank_sample_mean_us[candidate_name]
                    / tail_all_rank_sample_mean_us[baseline_name]
                    - 1.0
                ),
            }
        print(
            "MEGAMOE_EP16_TWO_KERNEL_BENCH "
            + json.dumps(
                {
                    **contract,
                    "warmup": args.warmup,
                    "iterations": args.iters,
                    "tail_iterations": args.tail_iters,
                    "candidate_vs_mori_rank_max_rel_l2": comparison_rel_l2,
                    "all_checks_rank_max_rel_l2": all_checks_rel_l2,
                    "rel_l2_threshold": args.rel_l2_threshold,
                    "rank_summaries": gathered_summaries,
                    "correctness_by_rank": gathered_diagnostics,
                    "timing_comparison": timing_comparison,
                    "tail_all_rank_sample_mean_us": tail_all_rank_sample_mean_us,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier()
    for path in paths:
        close = getattr(path, "close", None)
        if close is not None:
            close()
    if needs_mori:
        import mori.shmem as ms

        ms.shmem_finalize()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
