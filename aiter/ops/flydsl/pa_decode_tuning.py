# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline PA-decode budget tuning and explicit, static-shape lookup.

CPU-only: --help, --dry-run, shape/key/selection helpers.
Torch and the native PA implementation are imported only by the GPU runner.
Candidates are checked in eager mode before graph-based timing.
Each CSV row contains device-local shape, workgroup budget, and performance.
Only successful measurements are saved; dry-run previews leave timings blank.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import itertools
import math
import random
import statistics
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ATOL = RTOL = 0.005
NEAR_BEST = 0.97
DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096)
STATIC_SHAPE_FIELDS = (
    "batch_size",
    "context_length",
    "query_length",
    "num_query_heads",
    "num_kv_heads",
    "query_group_size",
    "head_dim",
    "page_size",
    "query_dtype",
    "per_token_kv",
    "trans_v",
    "sliding_window",
    "softmax_scale",
)
BENCHMARK_FIELDS = (
    "length_mode",
    "seed",
    "lengths",
)
CSV_FIELDS = (
    "architecture",
    "num_cu",
    *STATIC_SHAPE_FIELDS,
    "workgroup_budget",
    "median_us",
    "min_us",
    "max_us",
    "unique_kv_tb_s",
)


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def make_shape(
    batch_size,
    context_length,
    query_length=1,
    *,
    num_query_heads,
    num_kv_heads,
    head_dim,
    page_size=128,
    dtype="bfloat16",
    per_token=True,
    trans_v=True,
    window=0,
    length_mode="uniform",
    seed=0,
):
    """Build a tuning shape from explicit device-local attention geometry."""
    num_query_heads = positive_int(num_query_heads, "num_query_heads")
    num_kv_heads = positive_int(num_kv_heads, "num_kv_heads")
    head_dim = positive_int(head_dim, "head_dim")
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if head_dim != 64 and (head_dim % 128 or not 128 <= head_dim <= 1024):
        raise ValueError(
            "Supported head dimensions: 64 or multiples of 128 through 1024"
        )
    batch_size = positive_int(batch_size, "batch_size")
    context_length = positive_int(context_length, "context_length")
    query_length = positive_int(query_length, "query_length")
    page_size = positive_int(page_size, "page_size")
    seed = positive_int(seed, "seed", 0)
    if batch_size > 4096 or context_length >= 2**31 or seed >= 2**63:
        raise ValueError("B<=4096, context<int32 max and seed<int64 max are required")
    if query_length > context_length:
        raise ValueError("Context lengths must include every query token")
    if page_size not in (16, 64, 128) or dtype not in ("bfloat16", "float16"):
        raise ValueError("Use page16/64/128 and bfloat16/float16 queries")
    if type(per_token) is not bool or type(trans_v) is not bool:
        raise TypeError("per_token and trans_v must be bool")
    positive_int(window, "window", 0)
    if length_mode not in ("uniform", "varlen"):
        raise ValueError("length_mode must be uniform or varlen")
    lengths = [context_length] * batch_size
    if length_mode == "varlen" and batch_size > 1:
        rng = random.Random(seed)
        lengths = [rng.randint(query_length, context_length) for _ in lengths]
        lengths[0], lengths[-1] = query_length, context_length
    shape = {
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "query_group_size": num_query_heads // num_kv_heads,
        "head_dim": head_dim,
    }
    shape.update(
        batch_size=batch_size,
        context_length=context_length,
        query_length=query_length,
        page_size=page_size,
        query_dtype=dtype,
        per_token_kv=per_token,
        trans_v=trans_v,
        sliding_window=window,
        length_mode=length_mode,
        seed=seed,
        lengths=lengths,
        softmax_scale=head_dim**-0.5,
    )
    return shape


def _validate_shape(shape, *, benchmark=False):
    """Validate static geometry, optionally including the benchmark sample."""
    expected = make_shape(
        shape["batch_size"],
        shape["context_length"],
        shape["query_length"],
        num_query_heads=shape["num_query_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_dim=shape["head_dim"],
        page_size=shape["page_size"],
        dtype=shape["query_dtype"],
        per_token=shape["per_token_kv"],
        trans_v=shape["trans_v"],
        window=shape["sliding_window"],
        length_mode=shape["length_mode"] if benchmark else "uniform",
        seed=shape["seed"] if benchmark else 0,
    )
    fields = (
        STATIC_SHAPE_FIELDS + BENCHMARK_FIELDS if benchmark else STATIC_SHAPE_FIELDS
    )
    if any(
        type(shape[k]) is not type(expected[k]) or shape[k] != expected[k]
        for k in fields
    ):
        raise ValueError("Shape fields must agree with make_shape")


def make_key(shape, architecture, num_cu):
    """Build a lookup key from the device and static attention geometry."""
    if architecture not in ("gfx942", "gfx950"):
        raise ValueError("Supported architectures are gfx942 and gfx950")
    positive_int(num_cu, "num_cu")
    _validate_shape(shape)
    return {
        "architecture": architecture,
        "num_cu": num_cu,
        "kv_dtype": "float8_e4m3fn" if architecture == "gfx950" else "float8_e4m3fnuz",
        "shape": {k: shape[k] for k in STATIC_SHAPE_FIELDS},
    }


def candidate_groups(budgets, batch_size, kv_heads, num_cu):
    """Deduplicate by final capacity, retaining every requested-budget alias."""
    for value, name in ((batch_size, "B"), (kv_heads, "Hkv"), (num_cu, "CU")):
        positive_int(value, name)
    budgets = list(budgets)
    if not budgets:
        raise ValueError("At least one candidate budget is required")
    for budget in budgets:
        positive_int(budget, "budget")
    baseline = 2 * num_cu
    grouped = {}
    for budget in sorted(set(budgets) | {baseline}):
        capacity = min(
            batch_size * num_cu, max(batch_size, (budget + kv_heads - 1) // kv_heads)
        )
        grouped.setdefault(capacity, []).append(budget)
    return [
        {
            "capacity": capacity,
            "budget_aliases": aliases,
            "workgroup_budget": min(aliases),
            "launch_budget": baseline if baseline in aliases else min(aliases),
            "is_baseline": baseline in aliases,
        }
        for capacity, aliases in sorted(grouped.items())
    ]


def rotated_order(count, round_index):
    positive_int(count, "candidate count")
    positive_int(round_index, "round index", 0)
    shift = round_index % count
    order = list(range(shift, count)) + list(range(shift))
    return order[::-1] if (round_index // count) % 2 else order


def unique_kv_bytes(shape):
    window, ql = shape["sliding_window"], shape["query_length"]
    tokens = sum(min(n, window + ql - 1) if window else n for n in shape["lengths"])
    return 2 * shape["num_kv_heads"] * shape["head_dim"] * tokens


def _finite(value):
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _finite_positive(value):
    return _finite(value) and value > 0


def candidate_valid(candidate):
    if not isinstance(candidate, dict):
        return False
    accuracy = candidate.get("accuracy")
    return (
        isinstance(accuracy, dict)
        and accuracy.get("passed") is True
        and accuracy.get("finite") is True
        and accuracy.get("atol") == ATOL
        and accuracy.get("rtol") == RTOL
        and isinstance(accuracy.get("elements"), int)
        and _finite_positive(accuracy.get("elements"))
        and _finite(accuracy.get("max_abs_error"))
        and accuracy.get("max_abs_error", -1) >= 0
        and _finite(accuracy.get("max_tolerance_ratio"))
        and 0 <= accuracy.get("max_tolerance_ratio", -1) <= 1
        and candidate.get("status") == "PASS"
        and candidate.get("plan_unchanged") is True
        and isinstance(candidate.get("samples_us"), list)
        and bool(candidate["samples_us"])
        and all(_finite_positive(value) for value in candidate["samples_us"])
    )


def summarize_candidates(candidates, kv_bytes):
    """Populate descriptive metrics; never recommend after baseline failure."""
    positive_int(kv_bytes, "kv_bytes")
    baseline = [c for c in candidates if c.get("is_baseline") is True]
    if len(baseline) != 1 or not candidate_valid(baseline[0]):
        return None
    baseline = baseline[0]
    base_median = statistics.median(baseline["samples_us"])
    if not _finite_positive(base_median):
        return None
    valid = []
    for c in candidates:
        if not candidate_valid(c) or len(c["samples_us"]) != len(
            baseline["samples_us"]
        ):
            continue
        median = statistics.median(c["samples_us"])
        if not _finite_positive(median):
            continue
        speedup = base_median / median
        bandwidth = kv_bytes / median / 1e6
        ratios = [b / t for b, t in zip(baseline["samples_us"], c["samples_us"])]
        if not all(_finite_positive(value) for value in (speedup, bandwidth, *ratios)):
            continue
        c.update(
            median_us=median,
            min_us=min(c["samples_us"]),
            max_us=max(c["samples_us"]),
            unique_kv_tb_s=bandwidth,
            baseline_speedup=speedup,
            round_speedups=ratios,
        )
        valid.append(c)
    if baseline not in valid:
        return None
    best = max(valid, key=lambda c: (c["baseline_speedup"], -c["workgroup_budget"]))
    near = [
        c
        for c in valid
        if c["baseline_speedup"] >= NEAR_BEST * best["baseline_speedup"] - 1e-12
    ]
    conservative = min(near, key=lambda c: c["workgroup_budget"])
    return {
        "best_budget": best["workgroup_budget"],
        "best_capacity": best["capacity"],
        "best_speedup": best["baseline_speedup"],
        "conservative_budget": conservative["workgroup_budget"],
        "conservative_capacity": conservative["capacity"],
        "near_optimal_budgets": sorted(b for c in near for b in c["budget_aliases"]),
        "near_optimal_capacities": sorted(c["capacity"] for c in near),
        "valid_capacities": len(valid),
    }


def _validate_results(rows):
    """Validate flat shape/performance rows and reject duplicate budgets."""
    if not isinstance(rows, list):
        raise TypeError("Tuning results must be a list of CSV rows")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(CSV_FIELDS):
            raise ValueError(
                "Tuning rows require exactly the shape/performance columns"
            )
        make_key(row, row["architecture"], row["num_cu"])
        positive_int(row["workgroup_budget"], "workgroup_budget")
        for field in ("median_us", "min_us", "max_us", "unique_kv_tb_s"):
            if not _finite_positive(row[field]):
                raise ValueError(f"{field} must be finite and positive")
        if not row["min_us"] <= row["median_us"] <= row["max_us"]:
            raise ValueError("Timings must satisfy min_us <= median_us <= max_us")
        identity = (
            row["architecture"],
            row["num_cu"],
            *(row[field] for field in STATIC_SHAPE_FIELDS),
            row["workgroup_budget"],
        )
        if identity in seen:
            raise ValueError("Duplicate tuning row for the same shape and budget")
        seen.add(identity)


def _csv_row(key, candidate):
    return {
        "architecture": key["architecture"],
        "num_cu": key["num_cu"],
        **key["shape"],
        "workgroup_budget": candidate["workgroup_budget"],
        "median_us": candidate.get("median_us", ""),
        "min_us": candidate.get("min_us", ""),
        "max_us": candidate.get("max_us", ""),
        "unique_kv_tb_s": candidate.get("unique_kv_tb_s", ""),
    }


def _write_results_csv(stream, rows):
    _validate_results(rows)
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _read_results_csv(stream):
    reader = csv.DictReader(stream, strict=True)
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(CSV_FIELDS)
        or set(reader.fieldnames) != set(CSV_FIELDS)
    ):
        raise ValueError("Unexpected tuning CSV columns")
    rows = []
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("Malformed tuning CSV row width")
        row = {}
        for field, value in raw.items():
            if field in ("architecture", "query_dtype"):
                row[field] = value
            elif field in ("per_token_kv", "trans_v"):
                row[field] = {"true": True, "false": False, "1": True, "0": False}[
                    value.strip().lower()
                ]
            elif field in (
                "softmax_scale",
                "median_us",
                "min_us",
                "max_us",
                "unique_kv_tb_s",
            ):
                row[field] = float(value)
            else:
                row[field] = int(value)
        rows.append(row)
    _validate_results(rows)
    return rows


def load_results(path):
    """Load a flat shape/performance CSV without GPU imports."""
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            return _read_results_csv(stream)
    except (csv.Error, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed tuning-result CSV") from exc


def lookup_budget(rows, key, *, best=False):
    """Select a measured budget for an exact device and shape."""
    try:
        _validate_results(rows)
        if key != make_key(key["shape"], key["architecture"], key["num_cu"]):
            return None
        matches = [
            row
            for row in rows
            if row["architecture"] == key["architecture"]
            and row["num_cu"] == key["num_cu"]
            and all(row[field] == key["shape"][field] for field in STATIC_SHAPE_FIELDS)
        ]
        if not matches:
            return None
        fastest = min(
            matches, key=lambda row: (row["median_us"], row["workgroup_budget"])
        )
        if best:
            return fastest["workgroup_budget"]
        return min(
            row["workgroup_budget"]
            for row in matches
            if fastest["median_us"] / row["median_us"] >= NEAR_BEST - 1e-12
        )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def save_results(path, rows, *, overwrite=False):
    """Save flat CSV rows; create exclusively or replace atomically."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Tuning results require a .csv output path")
    buffer = io.StringIO(newline="")
    _write_results_csv(buffer, rows)
    payload = buffer.getvalue()
    if not overwrite:
        with path.open("x", newline="", encoding="utf-8") as stream:
            stream.write(payload)
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
        newline="",
        encoding="utf-8",
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_csv(path, rows, *, overwrite=False):
    """Alias for saving the shape/performance CSV."""
    save_results(path, rows, overwrite=overwrite)


def _make_inputs(torch, shape, device, fp8):
    """Chunked random initialization avoids a second whole-cache FP32 allocation."""
    b, ql, hq, h, d, page = (
        shape[k]
        for k in (
            "batch_size",
            "query_length",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "page_size",
        )
    )
    counts = [(n + page - 1) // page for n in shape["lengths"]]
    pages = sum(counts)
    generator = torch.Generator(device=device).manual_seed(shape["seed"])
    query = torch.empty(
        (b * ql, hq, d), dtype=getattr(torch, shape["query_dtype"]), device=device
    )
    query.uniform_(-0.5, 0.5, generator=generator)
    keys = torch.empty((pages, h, d // 16, page, 16), dtype=fp8, device=device)
    vshape = (pages, h, page // 16, d, 16) if shape["trans_v"] else (pages, h, d, page)
    values = torch.empty(vshape, dtype=fp8, device=device)
    scale_shape = (pages, h, page, 1) if shape["per_token_kv"] else (1,)
    ks = torch.empty(scale_shape, dtype=torch.float32, device=device)
    vs = torch.empty_like(ks)
    limit = torch.finfo(fp8).max
    if not shape["per_token_kv"]:
        ks.fill_(0.5 / limit)
        vs.fill_(0.5 / limit)
    for begin in range(0, pages, 64):
        end = min(begin + 64, pages)
        for kind, cache, scales in (("key", keys, ks), ("value", values, vs)):
            raw = torch.empty(
                (end - begin, h, page, d), dtype=torch.float32, device=device
            )
            raw.uniform_(-0.5, 0.5, generator=generator)
            scale = (
                raw.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / limit
                if shape["per_token_kv"]
                else scales
            )
            quant = (raw / scale).clamp(-limit, limit).to(fp8)
            if shape["per_token_kv"]:
                scales[begin:end].copy_(scale)
            packed = (
                quant.reshape(end - begin, h, page, d // 16, 16).permute(0, 1, 3, 2, 4)
                if kind == "key"
                else (
                    quant.reshape(end - begin, h, page // 16, 16, d).permute(
                        0, 1, 2, 4, 3
                    )
                    if shape["trans_v"]
                    else quant.permute(0, 1, 3, 2)
                )
            )
            cache[begin:end].copy_(packed)
    table = torch.zeros((b, max(counts)), dtype=torch.int32, device=device)
    start = 0
    for seq, count in enumerate(counts):
        table[seq, :count] = torch.arange(
            start, start + count, dtype=torch.int32, device=device
        )
        start += count
    return {
        "query": query,
        "key": keys,
        "value": values,
        "key_scale": ks,
        "value_scale": vs,
        "lengths": torch.tensor(shape["lengths"], dtype=torch.int32, device=device),
        "table": table,
    }


def _reference(torch, inputs, shape, chunk_size=4096):
    """Full FP32 dequantized attention, streamed over all visible keys."""
    ql, h, g, d, page = (
        shape[k]
        for k in (
            "query_length",
            "num_kv_heads",
            "query_group_size",
            "head_dim",
            "page_size",
        )
    )
    query = inputs["query"].reshape(shape["batch_size"], ql, h, g, d).float()
    output = torch.empty_like(query)
    device, window = query.device, shape["sliding_window"]
    for seq, length in enumerate(shape["lengths"]):
        right = length - ql + 1 + torch.arange(ql, device=device)
        maximum = torch.full((ql, h, g), -math.inf, dtype=torch.float32, device=device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((ql, h, g, d), dtype=torch.float32, device=device)
        first = max(0, length - ql + 1 - window) if window else 0
        for begin in range(first, length, chunk_size):
            tokens = torch.arange(begin, min(length, begin + chunk_size), device=device)
            pages, offsets = inputs["table"][seq, tokens // page].long(), tokens % page
            # Byte gathers also work on Torch builds without FP8 index kernels.
            keys = inputs["key"].view(torch.uint8)[pages, :, :, offsets, :]
            keys = keys.contiguous().view(inputs["key"].dtype).float().reshape(-1, h, d)
            cache = inputs["value"].view(torch.uint8)
            values = (
                cache[pages, :, offsets // 16, :, offsets % 16]
                if shape["trans_v"]
                else cache[pages, :, :, offsets]
            )
            values = values.contiguous().view(inputs["value"].dtype).float()
            if shape["per_token_kv"]:
                keys *= inputs["key_scale"][pages, :, offsets, 0].unsqueeze(-1)
                values *= inputs["value_scale"][pages, :, offsets, 0].unsqueeze(-1)
            else:
                keys *= inputs["key_scale"]
                values *= inputs["value_scale"]
            scores = (
                torch.einsum("qhgd,thd->qhgt", query[seq], keys)
                * shape["softmax_scale"]
            )
            mask = tokens[None, :] >= right[:, None]
            if window:
                mask |= tokens[None, :] < (right - window)[:, None]
            scores.masked_fill_(mask[:, None, None, :], -math.inf)
            next_max = torch.maximum(maximum, scores.amax(-1))
            safe_max = torch.where(torch.isfinite(next_max), next_max, 0)
            correction = torch.exp(maximum - safe_max)
            weights = torch.exp(scores - safe_max[..., None])
            numerator = numerator * correction[..., None] + torch.einsum(
                "qhgt,thd->qhgd", weights, values
            )
            denominator = denominator * correction + weights.sum(-1)
            maximum = next_max
        output[seq] = numerator / denominator[..., None]
    return output.reshape_as(inputs["query"])


def _accuracy(torch, output, reference):
    finite = bool(
        torch.isfinite(output).all().item() and torch.isfinite(reference).all().item()
    )
    result = {
        "finite": finite,
        "passed": False,
        "atol": ATOL,
        "rtol": RTOL,
        "elements": output.numel(),
    }
    if finite:
        error = (output.float() - reference).abs()
        tolerance = ATOL + RTOL * reference.abs()
        result.update(
            passed=bool((error <= tolerance).all().item()),
            max_abs_error=float(error.max().item()),
            max_tolerance_ratio=float((error / tolerance).max().item()),
        )
    return result


def _prepare_candidate(
    torch, pa, inputs, shape, candidate, reference, num_cu, iterations, warmup, stream
):
    plan = pa.plan_pa_decode(
        inputs["lengths"],
        shape["num_kv_heads"],
        max_partitions=num_cu,
        workgroup_budget=candidate["launch_budget"],
        sliding_window=shape["sliding_window"],
        query_length=shape["query_length"],
    )
    if plan.capacity != candidate["capacity"] or plan.max_partitions != num_cu:
        raise RuntimeError("Native planner disagrees with the recorded capacity/CU cap")
    work_before, reduce_before = plan.work_info.clone(), plan.reduce_info.clone()
    scratch_shape = (
        shape["num_kv_heads"],
        plan.capacity,
        shape["query_length"] * shape["query_group_size"],
    )
    output = torch.full_like(inputs["query"], math.nan)
    sums = torch.full(
        scratch_shape, math.nan, dtype=torch.float32, device=output.device
    )
    maxima = torch.full_like(sums, math.nan)
    partials = torch.full(
        (*scratch_shape, shape["head_dim"]),
        math.nan,
        dtype=output.dtype,
        device=output.device,
    )

    def launch():
        pa.pa_decode(
            output,
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["lengths"],
            inputs["table"],
            shape["softmax_scale"],
            shape["query_length"],
            compute_type=inputs["key"].dtype,
            key_scale=inputs["key_scale"],
            value_scale=inputs["value_scale"],
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partials,
            sliding_window=shape["sliding_window"],
            work_plan=plan,
        )

    launch()
    torch.cuda.synchronize()
    candidate["accuracy"] = _accuracy(torch, output, reference)
    if not candidate["accuracy"]["passed"]:
        raise ArithmeticError("eager accuracy failed")
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    parts = plan.reduce_info[:, 1].cpu().tolist()
    candidate["plan"] = {
        "active_slots": sum(parts),
        "parts_per_sequence": parts,
        "max_parts": max(parts),
    }
    return {
        "graph": graph,
        "output": output,
        "plan": plan,
        "snapshots": (work_before, reduce_before),
        "launch": launch,
    }


def _error_status(torch, exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "OOM"
    return "FAIL" if isinstance(exc, ArithmeticError) else "ERROR"


@contextmanager
def _full_fp32(torch):
    """Protect direct API callers as well as CLI users; restore global flags."""
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(flags[2])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[
            :2
        ]


def tune_shape(
    torch,
    pa,
    shape,
    device_info,
    budgets,
    *,
    rounds=16,
    iterations=100,
    warmup=5,
):
    """GPU-only. Each graph contains only native decode+reducer; no refresh."""
    for value, name, minimum in (
        (rounds, "rounds", 1),
        (iterations, "iterations", 1),
        (warmup, "warmup", 0),
    ):
        positive_int(value, name, minimum)
    _validate_shape(shape, benchmark=True)
    key = make_key(shape, device_info["architecture"], device_info["num_cu"])
    candidates = candidate_groups(
        budgets, shape["batch_size"], shape["num_kv_heads"], device_info["num_cu"]
    )
    for c in candidates:
        c.update(
            status="NOT_RUN",
            accuracy={},
            samples_us=[],
            errors=[],
            plan_unchanged=False,
        )
    record = {
        "key": key,
        "status": "RUNNING",
        "candidates": candidates,
        "selection": None,
        "errors": [],
        "unique_kv_bytes": unique_kv_bytes(shape),
    }
    live = {}
    try:
        device = torch.device("cuda", device_info["device"])
        props = torch.cuda.get_device_properties(device)
        if (
            props.multi_processor_count != key["num_cu"]
            or props.gcnArchName.split(":")[0] != key["architecture"]
        ):
            raise ValueError("Architecture/CU count must match the actual device")
        stream = torch.cuda.Stream(device=device)
        with _full_fp32(torch), torch.no_grad(), torch.cuda.device(
            device
        ), torch.cuda.stream(stream):
            inputs = _make_inputs(torch, shape, device, getattr(torch, key["kv_dtype"]))
            reference = _reference(torch, inputs, shape)
            if not bool(torch.isfinite(reference).all().item()):
                raise RuntimeError("FP32 reference is nonfinite")
            for c in sorted(candidates, key=lambda c: not c["is_baseline"]):
                try:
                    live[c["capacity"]] = _prepare_candidate(
                        torch,
                        pa,
                        inputs,
                        shape,
                        c,
                        reference,
                        device_info["num_cu"],
                        iterations,
                        warmup,
                        stream,
                    )
                    c["status"] = "READY"
                except Exception as exc:  # noqa: BLE001 - Record candidate failures.
                    c["status"] = _error_status(torch, exc)
                    c["errors"].append(f"{type(exc).__name__}: {exc}")
                    if c["is_baseline"]:
                        record["status"] = "BASELINE_FAILED"
                        return record
            ready = [c for c in candidates if c["status"] == "READY"]
            for r in range(rounds):
                order = [ready[i] for i in rotated_order(len(ready), r)]
                for c in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record(stream)
                    live[c["capacity"]]["graph"].replay()
                    end.record(stream)
                    end.synchronize()
                    elapsed = start.elapsed_time(end) * 1000 / iterations
                    if not _finite_positive(elapsed):
                        raise RuntimeError("Nonfinite or nonpositive event timing")
                    c["samples_us"].append(elapsed)
            for c in ready:
                resource = live[c["capacity"]]
                plan = resource["plan"]
                c["plan_unchanged"] = bool(
                    torch.equal(plan.work_info, resource["snapshots"][0])
                    and torch.equal(plan.reduce_info, resource["snapshots"][1])
                )
                c["status"] = "PASS" if c["plan_unchanged"] else "FAIL"
        record["selection"] = summarize_candidates(
            candidates, record["unique_kv_bytes"]
        )
        record["status"] = "PASS" if record["selection"] else "BASELINE_FAILED"
    except Exception as exc:  # noqa: BLE001 - Preserve partial results on failure.
        record["status"] = _error_status(torch, exc)
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record["selection"] = None
    finally:
        live.clear()
    return record


def _list_arg(text, converter=int, *, unique=True):
    try:
        result = [converter(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or (unique and len(result) != len(set(result))):
        raise argparse.ArgumentTypeError("Provide a nonempty list without duplicates")
    return result


def _load_gpu(args):
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("GPU tuning requires a supported ROCm PyTorch device")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    architecture, cu = props.gcnArchName.split(":")[0], props.multi_processor_count
    if (args.architecture and args.architecture != architecture) or (
        args.num_cu and args.num_cu != cu
    ):
        raise ValueError(
            "Explicit architecture/CU count disagrees with the actual device"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    return (
        torch,
        pa,
        {
            "architecture": architecture,
            "num_cu": cu,
            "device": args.device,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=lambda s: _list_arg(s, unique=False),
        required=True,
        metavar="Hq,Hkv,D",
        help="Device-local query heads, KV heads, and head dimension",
    )
    for name, default in (
        ("batch-sizes", [1]),
        ("context-lengths", [4096]),
        ("query-lengths", [1]),
        ("windows", [0]),
        ("page-sizes", [128]),
    ):
        parser.add_argument("--" + name, type=_list_arg, default=default)
    parser.add_argument(
        "--dtypes", type=lambda s: _list_arg(s, str), default=["bfloat16"]
    )
    parser.add_argument(
        "--quant-modes", type=lambda s: _list_arg(s, str), default=["per_token"]
    )
    parser.add_argument("--trans-v", choices=("yes", "no", "both"), default="yes")
    parser.add_argument(
        "--length-mode", choices=("uniform", "varlen"), default="uniform"
    )
    parser.add_argument("--seed", type=int, default=0)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--budgets", type=_list_arg)
    budgets.add_argument("--budget-cu", type=lambda s: _list_arg(s, float))
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Decode+reducer calls in each captured graph",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--architecture", choices=("gfx942", "gfx950"))
    parser.add_argument("--num-cu", type=int)
    result_output = parser.add_mutually_exclusive_group()
    result_output.add_argument(
        "--output", type=Path, help="Shape, budget, and performance in a new .csv file"
    )
    result_output.add_argument(
        "--csv", dest="output", type=Path, help="Alias for --output"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print a CSV preview with blank timings"
    )
    args = parser.parse_args(argv)
    try:
        if len(args.shape) != 3:
            raise ValueError("--shape requires exactly Hq,Hkv,D")
        hq, hkv, dim = args.shape
        if any(mode not in ("per_token", "per_tensor") for mode in args.quant_modes):
            raise ValueError("quant-modes must be per_token and/or per_tensor")
        layouts = [True, False] if args.trans_v == "both" else [args.trans_v == "yes"]
        shapes = [
            make_shape(
                b,
                ctx,
                ql,
                num_query_heads=hq,
                num_kv_heads=hkv,
                head_dim=dim,
                page_size=page,
                dtype=dtype,
                per_token=quant == "per_token",
                trans_v=layout,
                window=window,
                length_mode=args.length_mode,
                seed=args.seed,
            )
            for b, ctx, ql, window, page, dtype, quant, layout in itertools.product(
                args.batch_sizes,
                args.context_lengths,
                args.query_lengths,
                args.windows,
                args.page_sizes,
                args.dtypes,
                args.quant_modes,
                layouts,
            )
        ]
        for value, name, minimum in (
            (args.rounds, "rounds", 1),
            (args.iterations, "iterations", 1),
            (args.warmup, "warmup", 0),
        ):
            positive_int(value, name, minimum)
        positive_int(args.device, "device", 0)
        if args.num_cu is not None:
            positive_int(args.num_cu, "num_cu")
        for budget in args.budgets or DEFAULT_BUDGETS:
            positive_int(budget, "budget")
        if args.dry_run:
            if args.num_cu is None or args.architecture is None:
                raise ValueError(
                    "--dry-run requires explicit --architecture and --num-cu; it never probes a GPU"
                )
            info = {"architecture": args.architecture, "num_cu": args.num_cu}
        else:
            if args.output is None:
                raise ValueError(
                    "GPU tuning requires --output; existing files are never overwritten"
                )
            if args.output.suffix.lower() != ".csv":
                raise ValueError("Tuning results require a .csv output path")
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("Output CSV must be a new path")
            if not args.output.parent.is_dir():
                raise FileNotFoundError(
                    "Output CSV parent directory must already exist"
                )
            args.output = args.output.resolve()
            torch, pa, info = _load_gpu(args)
        candidates = args.budgets or list(DEFAULT_BUDGETS)
        if args.budget_cu is not None:
            scaled = [x * info["num_cu"] for x in args.budget_cu]
            if any(not _finite_positive(x) or not x.is_integer() for x in scaled):
                raise ValueError(
                    "CU-multiple budgets must produce positive integer workgroup counts"
                )
            candidates = [int(x) for x in scaled]
        preview = []
        for shape in shapes:
            key = make_key(shape, info["architecture"], info["num_cu"])
            preview.extend(
                _csv_row(key, candidate)
                for candidate in candidate_groups(
                    candidates,
                    shape["batch_size"],
                    shape["num_kv_heads"],
                    info["num_cu"],
                )
            )
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(preview)
        return 0

    rows = []
    save_results(args.output, rows)
    all_passed = True
    try:
        for index, shape in enumerate(shapes, 1):
            print(
                f"TUNE {index}/{len(shapes)} B={shape['batch_size']} L={shape['context_length']} QL={shape['query_length']}",
                flush=True,
            )
            result = tune_shape(
                torch,
                pa,
                shape,
                info,
                candidates,
                rounds=args.rounds,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            if result["status"] == "PASS":
                rows.extend(
                    _csv_row(result["key"], candidate)
                    for candidate in result["candidates"]
                    if candidate_valid(candidate)
                )
            else:
                all_passed = False
            save_results(args.output, rows, overwrite=True)
            print(f"RESULT {result['status']} {result['selection']}", flush=True)
            for candidate in result["candidates"]:
                for error in candidate["errors"]:
                    print(
                        f"ERROR budget={candidate['workgroup_budget']}: {error}",
                        file=sys.stderr,
                    )
            for error in result["errors"]:
                print(f"ERROR {error}", file=sys.stderr)
            torch.cuda.empty_cache()
    finally:
        save_results(args.output, rows, overwrite=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
