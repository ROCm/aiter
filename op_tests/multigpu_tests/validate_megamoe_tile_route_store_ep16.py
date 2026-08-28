# SPDX-License-Identifier: MIT
"""Validate deferred-reduction Stage2 paths against direct atomic Stage2."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.megamoe_tile import HierarchicalMegaMoEV2
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    TWO_KERNEL_ROUTE_PATTERNS,
    _comparison_metrics,
    _setup_dist,
    _shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    _validate_direct_tile_debug_snapshot,
)
from op_tests.multigpu_tests.megamoe_tile_route_store_factory import (
    MegaMoETileA4W4RouteStore,
)
from op_tests.multigpu_tests.megamoe_tile_rank_local_factory import (
    MegaMoETileA4W4RankLocal,
)


CASE_LABEL = "TPR128_TopK16_E896_H7168_I3072_EP16_A4W4"
REFERENCE_SCHEMA_VERSION = 2
DIRECT_PACKED_FIXTURE_VERSION = "fp4x2_0x11_e8m0_scale120_v1"
VALIDATION_ROUTE_PATTERNS = (
    "paired-rank-half-remote",
    "permuted-arbitrary-topk",
)
ROUTE_STORE_LAUNCHER_CONTRACT = {
    "worker_blocks": 176,
    "node_accumulation_mode": "route_store",
    "node_reduce_blocks": 16,
    "node_reduce_vec_bytes": 8,
    "node_reduce_schedule": "token",
    "node_reduce_load_schedule": "interleaved",
}
RANK_LOCAL_LAUNCHER_CONTRACT = {
    **ROUTE_STORE_LAUNCHER_CONTRACT,
    "node_accumulation_mode": "rank_local",
    "node_reduce_work_schedule": "static_strided",
    "node_reduce_rejoin_blocks": 0,
    "rank_epilogue_lds_addressing": "expanded",
    "rank_accumulation_mode": "atomic",
    "return_chunk_tokens": 16,
    "rail_return_schedule": "compact",
}
_PROTOCOL_ZERO_FIELDS = (
    "stage1_error_count",
    "stage2_error_count",
    "node_expected_uniform_mismatch",
    "node_expected_done_mismatch",
    "node_not_ready",
    "node_route_store_not_ready",
    "node_token_done_mismatch",
    "node_partial_done_mismatch",
    "queue_permutation_mismatch",
)


def _build(factory, shape, shared, rank):
    weights = shared.prepared_weights
    return factory(
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
        stage1_transport="sparse_wqe",
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _fixture_metadata(
    shape: BenchmarkShape,
    shared,
    *,
    rank: int,
    route_pattern: str,
) -> dict[str, object]:
    weights = shared.prepared_weights
    return {
        "case_label": CASE_LABEL,
        "route_pattern": route_pattern,
        "rank": int(rank),
        "shape": dict(shape.__dict__),
        "direct_packed_weights": True,
        "packed_fixture_version": DIRECT_PACKED_FIXTURE_VERSION,
        "x_sha256": _tensor_sha256(shared.x),
        "topk_ids_sha256": _tensor_sha256(shared.topk_ids),
        "route_weights_sha256": _tensor_sha256(shared.route_weights),
        "packed_weight_shapes": {
            "w1": list(weights.w1.shape),
            "w1_scale": list(weights.w1_scale.shape),
            "w2": list(weights.w2.shape),
            "w2_scale": list(weights.w2_scale.shape),
        },
    }


def _full_packed_weight_errors(
    shape: BenchmarkShape, metadata: dict[str, object]
) -> list[str]:
    expected = {
        "w1": [shape.local_experts, 2 * shape.inter, shape.hidden // 2],
        "w1_scale": [
            shape.local_experts,
            2 * shape.inter,
            shape.hidden // 32,
        ],
        "w2": [shape.local_experts, shape.hidden, shape.inter // 2],
        "w2_scale": [
            shape.local_experts,
            shape.hidden,
            shape.inter // 32,
        ],
    }
    actual = metadata.get("packed_weight_shapes")
    return [] if actual == expected else [f"packed weight shapes={actual}, expected={expected}"]


def _route_store_launcher_contract(operator) -> dict[str, object]:
    launcher = operator._stage2
    expected = dict(ROUTE_STORE_LAUNCHER_CONTRACT)
    expected["node_reduce_vec_bytes"] = int(
        getattr(
            operator,
            "stage2_node_reduce_vec_bytes",
            expected["node_reduce_vec_bytes"],
        )
    )
    actual = {
        "worker_blocks": int(operator.stage2_worker_blocks),
        "node_accumulation_mode": getattr(
            launcher, "node_accumulation_mode", "<missing>"
        ),
        "node_reduce_blocks": getattr(launcher, "node_reduce_blocks", "<missing>"),
        "node_reduce_vec_bytes": getattr(
            launcher, "node_reduce_vec_bytes", "<missing>"
        ),
        "node_reduce_schedule": getattr(
            launcher, "node_reduce_schedule", "<missing>"
        ),
        "node_reduce_load_schedule": getattr(
            launcher, "node_reduce_load_schedule", "<missing>"
        ),
    }
    manifest = getattr(launcher, "architecture_contract", {})
    manifest_actual = {
        name: manifest.get(name, "<missing>")
        for name in expected
        if name != "worker_blocks"
    }
    expected_manifest = {
        name: value
        for name, value in expected.items()
        if name != "worker_blocks"
    }
    errors = []
    if actual != expected:
        errors.append(
            f"route-store launcher={actual}, expected={expected}"
        )
    if manifest_actual != expected_manifest:
        errors.append(
            f"route-store manifest={manifest_actual}, expected={expected_manifest}"
        )
    return {
        "expected": expected,
        "launcher": actual,
        "manifest": manifest_actual,
        "errors": errors,
    }


def _rank_local_launcher_contract(operator) -> dict[str, object]:
    launcher = operator._stage2
    expected = dict(RANK_LOCAL_LAUNCHER_CONTRACT)
    expected["node_reduce_vec_bytes"] = int(
        getattr(
            operator,
            "stage2_node_reduce_vec_bytes",
            expected["node_reduce_vec_bytes"],
        )
    )
    expected["node_reduce_load_schedule"] = str(
        getattr(
            operator,
            "stage2_node_reduce_load_schedule",
            expected["node_reduce_load_schedule"],
        )
    )
    expected["node_reduce_work_schedule"] = str(
        getattr(
            operator,
            "stage2_node_reduce_work_schedule",
            expected["node_reduce_work_schedule"],
        )
    )
    expected["node_reduce_rejoin_blocks"] = int(
        getattr(
            operator,
            "stage2_node_reduce_rejoin_blocks",
            expected["node_reduce_rejoin_blocks"],
        )
    )
    expected["rank_epilogue_lds_addressing"] = str(
        getattr(
            operator,
            "stage2_rank_epilogue_lds_addressing",
            expected["rank_epilogue_lds_addressing"],
        )
    )
    expected["rank_accumulation_mode"] = str(
        getattr(
            operator,
            "stage2_rank_accumulation_mode",
            expected["rank_accumulation_mode"],
        )
    )
    # staged_ring reserves one additional CTA for the dedicated local
    # stage-reducer.  The legacy atomic/staged_reduce contracts retain the
    # original worker count.
    if expected["rank_accumulation_mode"] == "staged_ring":
        expected["worker_blocks"] = int(RANK_LOCAL_LAUNCHER_CONTRACT["worker_blocks"]) + 1
    actual = {
        "worker_blocks": int(operator.stage2_worker_blocks),
        "node_accumulation_mode": getattr(
            launcher, "node_accumulation_mode", "<missing>"
        ),
        "node_reduce_blocks": getattr(
            launcher, "node_reduce_blocks", "<missing>"
        ),
        "node_reduce_vec_bytes": getattr(
            launcher, "node_reduce_vec_bytes", "<missing>"
        ),
        "node_reduce_schedule": getattr(
            launcher, "node_reduce_schedule", "<missing>"
        ),
        "node_reduce_load_schedule": getattr(
            launcher, "node_reduce_load_schedule", "<missing>"
        ),
        "node_reduce_work_schedule": getattr(
            launcher, "node_reduce_work_schedule", "<missing>"
        ),
        "node_reduce_rejoin_blocks": getattr(
            launcher, "node_reduce_rejoin_blocks", "<missing>"
        ),
        "rank_epilogue_lds_addressing": getattr(
            launcher,
            "rank_epilogue_lds_addressing",
            expected["rank_epilogue_lds_addressing"],
        ),
        "rank_accumulation_mode": getattr(
            launcher,
            "rank_accumulation_mode",
            expected["rank_accumulation_mode"],
        ),
        "return_chunk_tokens": getattr(
            launcher, "return_chunk_tokens", "<missing>"
        ),
        "rail_return_schedule": getattr(
            launcher, "rail_return_schedule", "<missing>"
        ),
    }
    manifest = getattr(launcher, "architecture_contract", {})
    manifest_actual = {
        name: manifest.get(name, expected[name])
        for name in expected
        if name != "worker_blocks"
    }
    expected_manifest = {
        name: value for name, value in expected.items() if name != "worker_blocks"
    }
    errors = []
    if actual != expected:
        errors.append(f"rank-local launcher={actual}, expected={expected}")
    if manifest_actual != expected_manifest:
        errors.append(
            f"rank-local manifest={manifest_actual}, expected={expected_manifest}"
        )
    return {
        "expected": expected,
        "launcher": actual,
        "manifest": manifest_actual,
        "errors": errors,
    }


def _compact_protocol_snapshot(
    snapshot: dict[str, object], *, rank: int, generation: int
) -> dict[str, object]:
    expected = [int(value) for value in snapshot.get("node_atomic_expected", [])]
    done = [int(value) for value in snapshot.get("node_atomic_done", [])]
    ready = [int(value) for value in snapshot.get("node_atomic_ready", [])]
    raw_protocol_errors = snapshot.get("protocol_error_count", [])
    protocol_errors = (
        [int(value) for value in raw_protocol_errors]
        if isinstance(raw_protocol_errors, (list, tuple))
        else [int(raw_protocol_errors)]
    )
    errors: list[str] = []
    actual_generation = int(snapshot.get("generation", -1))
    if actual_generation != int(generation):
        errors.append(
            f"snapshot generation={actual_generation}, expected={generation}"
        )
    stage1_done = int(snapshot.get("stage1_done", -1))
    if stage1_done != int(generation):
        errors.append(f"stage1_done={stage1_done}, expected={generation}")
    comm_role_eos = [
        int(value) for value in snapshot.get("comm_role_eos", [])
    ]
    if comm_role_eos != [int(generation)] * 8:
        errors.append(
            f"comm_role_eos={comm_role_eos}, expected={[int(generation)] * 8}"
        )
    if len(expected) != 128 or len(done) != 128 or len(ready) != 128:
        errors.append(
            "node scoreboard lengths="
            f"{(len(expected), len(done), len(ready))}, expected=(128,128,128)"
        )
    if expected != done:
        errors.append("node expected/done vectors differ")
    if any(value == 0 for value in ready):
        errors.append("node ready vector is incomplete")
    if not protocol_errors or any(protocol_errors):
        errors.append(f"protocol_error_count={protocol_errors}")

    zero_fields = {}
    for name in _PROTOCOL_ZERO_FIELDS:
        if (
            snapshot.get("node_accumulation_mode") == "rank_local"
            and name == "node_route_store_not_ready"
        ):
            continue
        if name not in snapshot:
            # The production debug snapshot does not expose tile-reducer-only
            # diagnostics unless that schedule is active.
            if name == "node_partial_done_mismatch":
                continue
            errors.append(f"missing protocol field {name}")
            continue
        value = int(snapshot[name])
        zero_fields[name] = value
        if value != 0:
            errors.append(f"{name}={value}, expected=0")

    return {
        "rank": int(rank),
        "generation": actual_generation,
        "stage1_done": stage1_done,
        "comm_role_eos_count": len(comm_role_eos),
        "comm_role_eos_generation_mismatch": sum(
            int(value != int(generation)) for value in comm_role_eos
        ),
        "node_expected_min": min(expected) if expected else None,
        "node_expected_max": max(expected) if expected else None,
        "node_done_min": min(done) if done else None,
        "node_done_max": max(done) if done else None,
        "node_ready_count": sum(int(value != 0) for value in ready),
        "node_ready_expected": 128,
        "protocol_error_count": max(protocol_errors) if protocol_errors else -1,
        **zero_fields,
        "errors": errors,
    }


def _safe_comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rank: int,
    label: str,
) -> dict[str, object]:
    try:
        return dict(
            _comparison_metrics(reference, actual, rank=rank, label=label)
        )
    except Exception as error:
        return {
            "rank": int(rank),
            "label": label,
            "error": f"{type(error).__name__}: {error}",
        }


def _aggregate_generation_records(
    records: list[object], *, rel_l2_threshold: float
) -> tuple[dict[str, object], list[str]]:
    """Aggregate already-gathered rank records without raising locally."""

    failures: list[str] = []
    normalized: list[dict[str, object]] = []
    rel_l2_values: list[float] = []
    max_abs_values: list[float] = []
    generations = set()
    device_generations = set()
    for expected_rank, raw in enumerate(records):
        if not isinstance(raw, dict):
            failures.append(f"rank {expected_rank}: missing generation record")
            continue
        record = dict(raw)
        normalized.append(record)
        rank = int(record.get("rank", -1))
        if rank != expected_rank:
            failures.append(
                f"gather slot {expected_rank}: record rank={rank}"
            )
        generations.add(int(record.get("generation_index", -1)))
        device_generations.add(int(record.get("device_generation", -1)))
        for error in record.get("errors", []):
            failures.append(f"rank {rank}: {error}")
        protocol = record.get("protocol")
        if not isinstance(protocol, dict):
            failures.append(f"rank {rank}: missing compact protocol snapshot")
        else:
            for error in protocol.get("errors", []):
                failures.append(f"rank {rank}: {error}")
        for comparison in record.get("comparisons", []):
            label = str(comparison.get("label", "unnamed"))
            if "error" in comparison:
                failures.append(
                    f"rank {rank} {label}: {comparison['error']}"
                )
                continue
            rel_l2 = float(comparison.get("rel_l2", math.inf))
            max_abs = float(comparison.get("max_abs", math.inf))
            rel_l2_values.append(rel_l2)
            max_abs_values.append(max_abs)
            if not math.isfinite(rel_l2) or rel_l2 >= rel_l2_threshold:
                failures.append(
                    f"rank {rank} {label}: rel_l2={rel_l2} >= "
                    f"{rel_l2_threshold}"
                )
    if len(generations) != 1:
        failures.append(f"gathered generation indices differ: {sorted(generations)}")
    if len(device_generations) != 1:
        failures.append(
            "gathered device generations differ: "
            f"{sorted(device_generations)}"
        )
    return (
        {
            "generation_index": (
                next(iter(generations)) if len(generations) == 1 else None
            ),
            "device_generation": (
                next(iter(device_generations))
                if len(device_generations) == 1
                else None
            ),
            "rank_max_rel_l2": max(rel_l2_values) if rel_l2_values else None,
            "rank_max_abs": max(max_abs_values) if max_abs_values else None,
            "protocol_error_count": max(
                (
                    int(record.get("protocol", {}).get("protocol_error_count", -1))
                    for record in normalized
                    if isinstance(record.get("protocol"), dict)
                ),
                default=-1,
            ),
            "ranks": normalized,
        },
        failures,
    )


def _gather_preflight(
    local: dict[str, object], *, world: int
) -> list[dict[str, object]]:
    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, local)
    failures = [
        f"rank {rank}: {error}"
        for rank, record in enumerate(gathered)
        for error in (
            record.get("errors", [])
            if isinstance(record, dict)
            else ["missing preflight record"]
        )
    ]
    if failures:
        raise AssertionError(
            "route-store validation preflight failed after all-rank gather: "
            + json.dumps(failures, sort_keys=True)
        )
    return [dict(record) for record in gathered]


def _reference_path(
    output_dir: Path, route_pattern: str, rank: int | str
) -> Path:
    return output_dir / f"direct_{CASE_LABEL}_{route_pattern}_rank{rank}.pt"


def _load_reference(
    path: Path,
    *,
    shape: BenchmarkShape,
    metadata: dict[str, object],
    generations: int,
) -> tuple[list[torch.Tensor] | None, list[int] | None, list[str]]:
    errors: list[str] = []
    document = None
    try:
        document = torch.load(path, map_location="cpu")
    except Exception as error:
        errors.append(f"cannot load reference {path}: {type(error).__name__}: {error}")
    if not isinstance(document, dict):
        if document is not None:
            errors.append("reference document must be a dictionary")
        return None, None, errors
    if document.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        errors.append(
            f"reference schema_version={document.get('schema_version')}, "
            f"expected={REFERENCE_SCHEMA_VERSION}"
        )
    if metadata.get("packed_fixture_version") != DIRECT_PACKED_FIXTURE_VERSION:
        errors.append(
            "current fixture metadata has an unknown packed_fixture_version"
        )
    if document.get("metadata") != metadata:
        errors.append("reference fixture metadata does not match this run")
    outputs = document.get("outputs")
    if not isinstance(outputs, list) or len(outputs) < generations:
        errors.append(
            f"reference has {len(outputs) if isinstance(outputs, list) else 0} "
            f"generations, requires {generations}"
        )
        return None, None, errors
    try:
        recorded_generations = int(document.get("generations", -1))
    except (TypeError, ValueError):
        recorded_generations = -1
    if recorded_generations != len(outputs):
        errors.append(
            f"reference generations={recorded_generations}, outputs={len(outputs)}"
        )
    if not all(isinstance(output, torch.Tensor) for output in outputs):
        errors.append("reference outputs must be tensors")
        return None, None, errors
    expected_shape = (int(shape.tokens), int(shape.hidden))
    for generation_index, output in enumerate(outputs[:generations]):
        if tuple(output.shape) != expected_shape:
            errors.append(
                f"reference output {generation_index} shape={tuple(output.shape)}, "
                f"expected={expected_shape}"
            )
        if output.dtype != torch.bfloat16:
            errors.append(
                f"reference output {generation_index} dtype={output.dtype}, "
                "expected=torch.bfloat16"
            )
    device_generations = document.get("device_generations")
    if not isinstance(device_generations, list) or len(device_generations) != len(
        outputs
    ):
        errors.append(
            "reference device_generations must align one-to-one with outputs"
        )
        return None, None, errors
    try:
        device_generations = [int(value) for value in device_generations]
    except (TypeError, ValueError) as error:
        errors.append(f"invalid reference device_generations: {error}")
        return None, None, errors
    if any(
        right != left + 1
        for left, right in zip(device_generations, device_generations[1:])
    ):
        errors.append(
            f"reference device_generations are not consecutive: {device_generations}"
        )
    return outputs[:generations], device_generations[:generations], errors


def _run(
    factory,
    shape,
    shared,
    rank,
    world,
    device,
    *,
    mode: str,
    generations: int,
    rel_l2_threshold: float,
    fixture_metadata: dict[str, object],
    reference_outputs: list[torch.Tensor] | None,
    reference_device_generations: list[int] | None,
    reference_errors: list[str],
):
    operator = _build(factory, shape, shared, rank)
    outputs: list[torch.Tensor] = []
    generation_summaries: list[dict[str, object]] = []
    try:
        launcher_contract = None
        if mode == "route":
            launcher_contract = _route_store_launcher_contract(operator)
        elif mode == "rank":
            launcher_contract = _rank_local_launcher_contract(operator)
        local_preflight_errors = list(reference_errors)
        local_preflight_errors.extend(
            _full_packed_weight_errors(shape, fixture_metadata)
        )
        if launcher_contract is not None:
            local_preflight_errors.extend(launcher_contract["errors"])
        preflight = _gather_preflight(
            {
                "rank": int(rank),
                "fixture_metadata": fixture_metadata,
                "launcher_contract": launcher_contract,
                "errors": local_preflight_errors,
            },
            world=world,
        )

        first_output = None
        for generation_index in range(generations):
            dist.barrier()
            output = operator.forward(
                shared.x, shared.route_weights, shared.topk_ids
            )
            torch.cuda.synchronize(device)

            local_errors: list[str] = []
            try:
                snapshot = operator.debug_direct_tile_snapshot()
                protocol = _compact_protocol_snapshot(
                    snapshot,
                    rank=rank,
                    generation=int(operator._generation),
                )
                try:
                    _validate_direct_tile_debug_snapshot(
                        snapshot,
                        expected_routes=2048,
                        expected_tokens=128,
                    )
                except Exception as error:
                    protocol["errors"].append(
                        "full protocol validation failed: "
                        f"{type(error).__name__}: {error}"
                    )
            except Exception as error:
                protocol = {
                    "rank": int(rank),
                    "generation": int(operator._generation),
                    "protocol_error_count": -1,
                    "errors": [
                        "snapshot failed: "
                        f"{type(error).__name__}: {error}"
                    ],
                }

            output_cpu = output.detach().cpu().clone()
            outputs.append(output_cpu)
            comparisons: list[dict[str, object]] = []
            if (
                reference_device_generations is not None
                and int(operator._generation)
                != reference_device_generations[generation_index]
            ):
                local_errors.append(
                    f"device generation={int(operator._generation)}, reference="
                    f"{reference_device_generations[generation_index]}"
                )
            if reference_outputs is not None:
                comparisons.append(
                    _safe_comparison(
                        reference_outputs[generation_index],
                        output_cpu,
                        rank=rank,
                        label=(
                            f"route_generation_{generation_index}_vs_"
                            f"direct_generation_{generation_index}"
                        ),
                    )
                )
            if first_output is None:
                first_output = output_cpu
            else:
                comparisons.append(
                    _safe_comparison(
                        first_output,
                        output_cpu,
                        rank=rank,
                        label=(
                            f"{mode}_generation_0_vs_"
                            f"generation_{generation_index}"
                        ),
                    )
                )

            local_record = {
                "rank": int(rank),
                "generation_index": generation_index,
                "device_generation": int(operator._generation),
                "protocol": protocol,
                "comparisons": comparisons,
                "errors": local_errors,
            }
            gathered: list[object] = [None] * world
            dist.all_gather_object(gathered, local_record)
            summary, failures = _aggregate_generation_records(
                gathered,
                rel_l2_threshold=rel_l2_threshold,
            )
            generation_summaries.append(summary)
            if failures:
                raise AssertionError(
                    "route-store validation failed after all-rank generation "
                    f"{generation_index} gather: "
                    + json.dumps(failures, sort_keys=True)
                )
    finally:
        operator.close()
    dist.barrier()
    torch.cuda.empty_cache()
    return outputs, generation_summaries, preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("direct", "route", "rank"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route-pattern",
        choices=VALIDATION_ROUTE_PATTERNS,
        default="paired-rank-half-remote",
    )
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--rel-l2-threshold", type=float, default=5.0e-2)
    parser.add_argument(
        "--node-reduce-vec-bytes", type=int, choices=(4, 8, 16), default=8
    )
    parser.add_argument(
        "--node-reduce-load-schedule",
        choices=("interleaved", "load_first"),
        default="interleaved",
    )
    parser.add_argument(
        "--node-reduce-work-schedule",
        choices=("static_strided", "dynamic_head"),
        default="static_strided",
    )
    parser.add_argument(
        "--node-reduce-rejoin-blocks", type=int, choices=(0, 8, 16, 32), default=0
    )
    parser.add_argument(
        "--rank-epilogue-lds-addressing",
        choices=("expanded", "dynamic_base"),
        default="expanded",
    )
    parser.add_argument(
        "--rank-accumulation-mode",
        choices=("atomic", "staged_reduce", "staged_ring"),
        default="atomic",
    )
    args = parser.parse_args()
    if args.generations < 4:
        parser.error("--generations must be at least 4")
    if not 0.0 < args.rel_l2_threshold <= 5.0e-2:
        parser.error("--rel-l2-threshold must be in (0,0.05]")
    if args.route_pattern not in TWO_KERNEL_ROUTE_PATTERNS:
        parser.error(f"unsupported shared route pattern {args.route_pattern!r}")
    if args.node_reduce_vec_bytes == 16 and args.mode != "rank":
        parser.error("16-byte node reduction requires --mode=rank")
    if args.rank_epilogue_lds_addressing == "dynamic_base" and not (
        args.mode == "rank"
        and args.node_reduce_vec_bytes == 8
        and args.node_reduce_load_schedule == "load_first"
        and args.node_reduce_work_schedule == "static_strided"
        and args.node_reduce_rejoin_blocks == 0
    ):
        parser.error(
            "dynamic_base LDS addressing requires rank mode, vec8, load_first, "
            "static_strided reduction, and rejoin_blocks=0"
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
    rank, world, _local_rank, device = _setup_dist(needs_mori=False)
    try:
        if world != shape.ep_size:
            raise ValueError(f"route-store validation requires EP16, got {world}")
        shared = _shared_inputs(
            shape,
            rank,
            world,
            device,
            route_pattern=args.route_pattern,
            direct_packed_weights=True,
        )
        metadata = _fixture_metadata(
            shape,
            shared,
            rank=rank,
            route_pattern=args.route_pattern,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        reference_path = _reference_path(
            args.output_dir, args.route_pattern, rank
        )
        reference_outputs = None
        reference_device_generations = None
        reference_errors: list[str] = []
        if args.mode != "direct":
            (
                reference_outputs,
                reference_device_generations,
                reference_errors,
            ) = _load_reference(
                reference_path,
                shape=shape,
                metadata=metadata,
                generations=args.generations,
            )

        factory = HierarchicalMegaMoEV2
        if args.mode == "route":
            factory = partial(
                MegaMoETileA4W4RouteStore,
                stage2_node_reduce_vec_bytes=args.node_reduce_vec_bytes,
            )
        elif args.mode == "rank":
            factory = partial(
                MegaMoETileA4W4RankLocal,
                stage2_node_reduce_vec_bytes=args.node_reduce_vec_bytes,
                stage2_node_reduce_load_schedule=(
                    args.node_reduce_load_schedule
                ),
                stage2_node_reduce_work_schedule=(
                    args.node_reduce_work_schedule
                ),
                stage2_node_reduce_rejoin_blocks=(
                    args.node_reduce_rejoin_blocks
                ),
                stage2_rank_epilogue_lds_addressing=(
                    args.rank_epilogue_lds_addressing
                ),
                stage2_rank_accumulation_mode=args.rank_accumulation_mode,
            )
        outputs, generation_summaries, preflight = _run(
            factory,
            shape,
            shared,
            rank,
            world,
            device,
            mode=args.mode,
            generations=args.generations,
            rel_l2_threshold=args.rel_l2_threshold,
            fixture_metadata=metadata,
            reference_outputs=reference_outputs,
            reference_device_generations=reference_device_generations,
            reference_errors=reference_errors,
        )
        if args.mode == "direct":
            torch.save(
                {
                    "schema_version": REFERENCE_SCHEMA_VERSION,
                    "metadata": metadata,
                    "generations": args.generations,
                    "device_generations": [
                        int(summary["device_generation"])
                        for summary in generation_summaries
                    ],
                    "outputs": outputs,
                    "local_protocol": [
                        summary["ranks"][rank]["protocol"]
                        for summary in generation_summaries
                    ],
                },
                reference_path,
            )

        if rank == 0:
            comparisons = [
                comparison
                for summary in generation_summaries
                for record in summary["ranks"]
                for comparison in record["comparisons"]
            ]
            rel_l2_values = [
                float(comparison["rel_l2"])
                for comparison in comparisons
                if "rel_l2" in comparison
            ]
            max_abs_values = [
                float(comparison["max_abs"])
                for comparison in comparisons
                if "max_abs" in comparison
            ]
            print(
                "MEGAMOE_ROUTE_STORE_VALIDATION "
                + json.dumps(
                    {
                        "case_label": CASE_LABEL,
                        "mode": args.mode,
                        "route_pattern": args.route_pattern,
                        "generations": args.generations,
                        "node_reduce_vec_bytes": (
                            args.node_reduce_vec_bytes
                            if args.mode != "direct"
                            else None
                        ),
                        "node_reduce_load_schedule": (
                            args.node_reduce_load_schedule
                            if args.mode == "rank"
                            else None
                        ),
                        "node_reduce_work_schedule": (
                            args.node_reduce_work_schedule
                            if args.mode == "rank"
                            else None
                        ),
                        "node_reduce_rejoin_blocks": (
                            args.node_reduce_rejoin_blocks
                            if args.mode == "rank"
                            else None
                        ),
                        "rank_accumulation_mode": (
                            args.rank_accumulation_mode
                            if args.mode == "rank"
                            else None
                        ),
                        "rel_l2_threshold": args.rel_l2_threshold,
                        "reference_path_pattern": str(
                            _reference_path(
                                args.output_dir, args.route_pattern, "{rank}"
                            )
                        ),
                        "rank_max_rel_l2": (
                            max(rel_l2_values) if rel_l2_values else None
                        ),
                        "rank_max_abs": (
                            max(max_abs_values) if max_abs_values else None
                        ),
                        "protocol_errors": max(
                            int(summary["protocol_error_count"])
                            for summary in generation_summaries
                        ),
                        "preflight_by_rank": preflight,
                        "generation_results": generation_summaries,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
