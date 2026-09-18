"""Actual input identity and EPLB workload checks, outside Graph timing.

The benchmark's two paths share one builder. Equal seeds alone do not prove
that separate processes received equal tensors: hash every byte, including
packed model weights and their scales. No tensor values are logged.
"""

from __future__ import annotations

import hashlib
import json


PUBLIC_INPUTS = ("x", "topk_ids", "route_weights")
WEIGHTS = ("w1", "w1_scale", "w2", "w2_scale")
IMMUTABLE_INPUTS = PUBLIC_INPUTS + WEIGHTS + ("local_expert_mask",)
INPUT_FIELDS = IMMUTABLE_INPUTS + ("a_quant", "a_scale")


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_fingerprint(tensor, *, chunk_bytes=16 << 20):
    """Hash full contiguous storage, preserving BF16/FP4/E8M0 bit patterns.

    16 MiB chunks bound host-copy memory even when all eight ranks hash GiB
    weights together. This is not sampling. All copies finish before warmup
    or after both timing batches, never inside capture or between replays.
    """
    import torch

    if chunk_bytes < 1 or not tensor.is_contiguous():
        raise ValueError("input fingerprint requires contiguous storage and positive chunk size")
    raw = tensor.detach().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for begin in range(0, raw.numel(), chunk_bytes):
        host = raw[begin:begin + chunk_bytes].cpu().numpy()
        digest.update(memoryview(host))
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "bytes": raw.numel(), "sha256": digest.hexdigest()}


def fingerprint_shared_inputs(shared, fields=INPUT_FIELDS):
    return {name: tensor_fingerprint(getattr(shared.prepared_weights if name in WEIGHTS else shared, name))
            for name in fields}


def changed_inputs(original, current):
    return [name for name, value in current.items() if original.get(name) != value]


def route_histogram(ids, *, rank, shape):
    """Count original slots, including repeated expert IDs, on the CPU."""
    tokens, topk = shape["tokens"], shape["topk"]
    experts, world = shape["experts"], shape["ep_size"]
    per_node = shape["gpus_per_node"]
    if len(ids) != tokens or experts % world or not 0 <= rank < world:
        raise ValueError("invalid routing shape/rank")
    counts = [0] * experts
    one_route_per_rank = True
    remote_slots = set()
    for row in ids:
        if len(row) != topk or any(type(e) is not int or not 0 <= e < experts for e in row):
            raise ValueError("invalid original TopK slots")
        owners = [0] * world
        remote = 0
        for expert in row:
            counts[expert] += 1
            owner = expert // (experts // world)
            owners[owner] += 1
            remote += owner // per_node != rank // per_node
        one_route_per_rank &= all(count == 1 for count in owners)
        remote_slots.add(remote)
    return {"expert_counts": counts, "one_route_per_rank_per_token": one_route_per_rank,
            "remote_slots_per_token": sorted(remote_slots)}


def aggregate_input_identity(rows, *, shape):
    """Check all ranks and retain actual per-expert GEMM load, including pad."""
    world, experts = shape["ep_size"], shape["experts"]
    if sorted(row["rank"] for row in rows) != list(range(world)):
        raise ValueError("input identity needs each rank exactly once")
    ordered = sorted(rows, key=lambda row: row["rank"])
    for row in ordered:
        if set(row["tensors"]) != set(INPUT_FIELDS):
            raise ValueError(f"rank {row['rank']}: missing input/weight fingerprints")
        if len(row["routes"]["expert_counts"]) != experts:
            raise ValueError("missing expert workload counts")
    counts = [sum(row["routes"]["expert_counts"][expert] for row in ordered)
              for expert in range(experts)]
    epr = experts // world
    per_rank = [counts[rank * epr:(rank + 1) * epr] for rank in range(world)]
    routes = [sum(values) for values in per_rank]
    # Both Stage1 producers use BM32 sorting. Padding is GEMM work even when
    # the valid route totals match; do not label a skewed expert layout EPLB.
    padded = [sum((count + 31) // 32 * 32 for count in values) for values in per_rank]
    eplb = (all(row["routes"]["one_route_per_rank_per_token"] for row in ordered)
            and len(set(routes)) == 1 and len(set(padded)) == 1
            and len({tuple(sorted(values)) for values in per_rank}) == 1)
    return {"schema": "stage2_actual_inputs_v1", "sha256": json_sha256(ordered),
            "ranks": ordered, "eplb": eplb, "routes_per_rank": routes,
            "padded_rows_bm32_per_rank": padded, "expert_counts": counts,
            "prequant_scope": "initial reference buffers; both full-forward paths quantize shared BF16 x",
            "hash_scope": "every byte of all listed tensors; no sampled checksums"}


def performance_exclusions(report, identity):
    reasons = []
    if report["route_pattern"] != "cross_node" or not identity["eplb"]:
        reasons.append("performance optimization is restricted to verified cross_node EPLB")
    if (report["warmup"], report["iterations"], report["tail_iterations"]) != (10, 40, 20):
        reasons.append("formal comparison requires warmup10/profile40/tail20")
    for field in ("device_timeline", "graph_debug_reference", "graph_coalesce_reference_duplicates", "direct_packed_weights"):
        if report.get(field):
            reasons.append(f"diagnostic option {field}")
    if report["mori_combine_quant_type"] != "none" or report["candidate_rail_quant_type"] != "none":
        reasons.append("this A/B contract fixes BF16 combine and return")
    return reasons
