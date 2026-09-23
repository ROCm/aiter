# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline PA-decode budget tuning and explicit, static-shape lookup.

CPU-only: --help, --list-models, --dry-run, model/shape/key/selection helpers.
Torch and the native PA implementation are imported only by the GPU runner.
This module is deliberately not imported by the runtime selector.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import itertools
import json
import math
import random
import statistics
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ATOL = RTOL = 0.005
NEAR_BEST = 0.97
DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096)
ACCURACY_STAGES = ("eager", "graph", "post_timing")
UPSTREAM = (
    "https://github.com/ROCm/FlyDSL/blob/"
    "76ca04c92a9b7459d63a9f8fded6a1f35257b723/kernels/attention/pa_metadata_tuning.py"
)
PRESETS = {
    "mqa-synthetic": (16, 1, 128, "Synthetic MQA-like shape; not an official M3 model"),
    "qwen3-235b-a22b": (64, 4, 128, "Qwen/Qwen3-235B-A22B attention geometry"),
    "qwen3-32b": (64, 8, 128, "Qwen/Qwen3-32B attention geometry"),
    "qwen3-8b": (32, 8, 128, "Qwen/Qwen3-8B attention geometry"),
    "gemma3-4b": (8, 4, 256, "Google Gemma 3 4B text attention geometry"),
}
SOURCE_FILES = (
    "pa_decode.py",
    "kernels/pa_decode_plan.py",
    "kernels/pa_decode_kernel.py",
    "kernels/pa_decode_reduce.py",
    "pa_decode_tuning.py",
)
STATIC_SHAPE_FIELDS = (
    "batch_size",
    "context_length",
    "query_length",
    "num_query_heads",
    "num_kv_heads",
    "query_group_size",
    "head_dim",
    "global_num_query_heads",
    "global_num_kv_heads",
    "tp_size",
    "kv_replication_factor",
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
    "lengths_sha256",
    "input_generator",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def resolve_model(
    name=None,
    config=None,
    *,
    num_query_heads=None,
    num_kv_heads=None,
    head_dim=None,
    tp_size=1,
    allow_kv_replication=False,
):
    """Resolve global head overrides, then strict even per-rank TP geometry."""
    if name is not None and config is not None:
        raise ValueError("Choose a preset or a local config, not both")
    if type(allow_kv_replication) is not bool:
        raise TypeError("allow_kv_replication must be bool")
    if config is not None:
        path = Path(config)
        content = path.read_bytes()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise TypeError("config must be a JSON object")
        data = data.get("text_config", data)
        if not isinstance(data, dict):
            raise TypeError("config/text_config must be a JSON object")
        if data.get("kv_lora_rank") is not None:
            raise ValueError(
                "MLA configs are unsupported; use explicit synthetic heads"
            )
        hq, hkv, dim = (
            data.get("num_attention_heads"),
            data.get("num_key_value_heads", data.get("num_attention_heads")),
            data.get("head_dim"),
        )
        origin = {
            "config": str(path.resolve()),
            "config_sha256": hashlib.sha256(content).hexdigest(),
        }
    else:
        name = name or "mqa-synthetic"
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        hq, hkv, dim, description = PRESETS[name]
        data = {}
        origin = {"preset": name, "description": description}
    original_hq = hq
    hq = positive_int(num_query_heads if num_query_heads is not None else hq, "Hq")
    hkv = positive_int(num_kv_heads if num_kv_heads is not None else hkv, "Hkv")
    dim = head_dim if head_dim is not None else dim
    if dim is None:
        hidden = positive_int(data.get("hidden_size"), "hidden_size")
        original_hq = positive_int(original_hq, "config Hq for head_dim inference")
        if hidden % original_hq:
            raise ValueError("config hidden_size/Hq is not integral; specify head_dim")
        dim = hidden // original_hq
    dim = positive_int(dim, "head_dim")
    if dim != 64 and (dim % 128 or not 128 <= dim <= 1024):
        raise ValueError(
            "Supported head dimensions: 64 or multiples of 128 through 1024"
        )
    tp_size = positive_int(tp_size, "tp_size")
    if hq % hkv or hq % tp_size:
        raise ValueError("Hq must be divisible by both Hkv and TP")
    replication = 1
    if hkv % tp_size == 0:
        local_kv = hkv // tp_size
    elif tp_size > hkv and tp_size % hkv == 0 and allow_kv_replication:
        local_kv, replication = 1, tp_size // hkv
    else:
        raise ValueError(
            "TP must divide Hkv; TP>Hkv requires explicit, even KV replication"
        )
    local_q = hq // tp_size
    if local_q % local_kv:
        raise ValueError("Local Hq must be divisible by local Hkv")
    return {
        "num_query_heads": local_q,
        "num_kv_heads": local_kv,
        "query_group_size": local_q // local_kv,
        "head_dim": dim,
        "global_num_query_heads": hq,
        "global_num_kv_heads": hkv,
        "tp_size": tp_size,
        "kv_replication_factor": replication,
        "origin": origin,
        "overrides": {"Hq": num_query_heads, "Hkv": num_kv_heads, "D": head_dim},
    }


def make_shape(
    model,
    batch_size,
    context_length,
    query_length=1,
    *,
    page_size=128,
    dtype="bfloat16",
    per_token=True,
    trans_v=True,
    window=0,
    length_mode="uniform",
    seed=0,
):
    batch_size = positive_int(batch_size, "batch_size")
    context_length = positive_int(context_length, "context_length")
    query_length = positive_int(query_length, "query_length")
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
    shape = {k: v for k, v in model.items() if k not in ("origin", "overrides")}
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
        lengths_sha256=fingerprint(lengths),
        input_generator="uniform_nonzero_fp8_v1",
        softmax_scale=model["head_dim"] ** -0.5,
    )
    return shape


def source_identity():
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _validate_shape(shape, *, benchmark=False):
    """Validate static geometry, optionally including the benchmark sample."""
    model = resolve_model(
        num_query_heads=shape["global_num_query_heads"],
        num_kv_heads=shape["global_num_kv_heads"],
        head_dim=shape["head_dim"],
        tp_size=shape["tp_size"],
        allow_kv_replication=True,
    )
    expected = make_shape(
        model,
        shape["batch_size"],
        shape["context_length"],
        shape["query_length"],
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
    if canonical({k: shape[k] for k in fields}) != canonical(
        {k: expected[k] for k in fields}
    ):
        raise ValueError("Shape fields must agree with make_shape")


def make_key(shape, architecture, num_cu, sources=None):
    """Static-shape key: no length distribution, seed or sample fingerprint."""
    if architecture not in ("gfx942", "gfx950"):
        raise ValueError("Supported architectures are gfx942 and gfx950")
    positive_int(num_cu, "num_cu")
    _validate_shape(shape)
    sources = source_identity() if sources is None else sources
    if (
        not isinstance(sources, dict)
        or set(sources) != set(SOURCE_FILES)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            for digest in sources.values()
        )
    ):
        raise ValueError("sources must contain every native source SHA256")
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "num_cu": num_cu,
        "kv_dtype": "float8_e4m3fn" if architecture == "gfx950" else "float8_e4m3fnuz",
        "shape": {k: copy.deepcopy(shape[k]) for k in STATIC_SHAPE_FIELDS},
        "sources": copy.deepcopy(sources),
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
    if not isinstance(accuracy, dict):
        return False
    for stage in ACCURACY_STAGES:
        result = accuracy.get(stage)
        if not isinstance(result, dict) or not (
            result.get("passed") is True
            and result.get("finite") is True
            and result.get("atol") == ATOL
            and result.get("rtol") == RTOL
            and isinstance(result.get("elements"), int)
            and _finite_positive(result.get("elements"))
            and _finite(result.get("max_abs_error"))
            and result.get("max_abs_error", -1) >= 0
            and _finite(result.get("max_tolerance_ratio"))
            and 0 <= result.get("max_tolerance_ratio", -1) <= 1
        ):
            return False
    return (
        candidate.get("status") == "PASS"
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


def _validate_results(data):
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != SCHEMA_VERSION
        or not isinstance(data.get("records"), list)
    ):
        raise ValueError("Unsupported tuning-result schema")
    canonical(data)  # Reject NaN/Infinity anywhere, not just in the winning row.
    seen = set()
    for record in data["records"]:
        key = record["key"]
        expected = make_key(
            key["shape"], key["architecture"], key["num_cu"], key["sources"]
        )
        if canonical(key) != canonical(expected):
            raise ValueError("Invalid static-shape key schema")
        benchmark = record["benchmark_input"]
        if not isinstance(benchmark, dict) or set(benchmark) != set(BENCHMARK_FIELDS):
            raise ValueError("benchmark_input may contain only benchmark sample fields")
        _validate_shape({**key["shape"], **benchmark}, benchmark=True)
        digest = fingerprint(key)
        if digest != record["key_sha256"] or digest in seen:
            raise ValueError(
                "Invalid or duplicate static-shape key; separate or explicitly aggregate samples"
            )
        seen.add(digest)


def load_results(path):
    """Load and validate an offline result; importing the GPU runtime is unnecessary."""
    data = json.loads(Path(path).read_text())
    try:
        _validate_results(data)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed tuning-result schema") from exc
    return data


def lookup_budget(data, key, *, best=False):
    """Explicit in-memory lookup; no files, GPU imports, approximate matches or tuning."""
    try:
        _validate_results(data)
        digest = fingerprint(key)
        matches = [
            r for r in data["records"] if r["key_sha256"] == digest and r["key"] == key
        ]
        if len(matches) != 1 or matches[0].get("status") != "PASS":
            return None
        record = matches[0]
        if record.get("sources_after") != key["sources"]:
            return None
        candidates = record["candidates"]
        groups = candidate_groups(
            [b for c in candidates for b in c["budget_aliases"]],
            key["shape"]["batch_size"],
            key["shape"]["num_kv_heads"],
            key["num_cu"],
        )
        if len(groups) != len(candidates) or any(
            any(c.get(k) != v for k, v in group.items())
            for c, group in zip(candidates, groups)
        ):
            return None
        elements = math.prod(
            key["shape"][k]
            for k in ("batch_size", "query_length", "num_query_heads", "head_dim")
        )
        if any(
            candidate_valid(c)
            and any(
                c["accuracy"][stage]["elements"] != elements
                for stage in ACCURACY_STAGES
            )
            for c in candidates
        ):
            return None
        benchmark_shape = {**key["shape"], **record["benchmark_input"]}
        _validate_shape(benchmark_shape, benchmark=True)
        kv_bytes = unique_kv_bytes(benchmark_shape)
        if kv_bytes != record["unique_kv_bytes"]:
            return None
        selection = summarize_candidates(copy.deepcopy(candidates), kv_bytes)
        if selection is None or selection != record.get("selection"):
            return None
        return selection["best_budget" if best else "conservative_budget"]
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def save_results(path, data, *, overwrite=False):
    _validate_results(data)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if not overwrite:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_csv(path, data, *, overwrite=False):
    _validate_results(data)
    fields = (
        "key_sha256",
        "shape_status",
        "architecture",
        "num_cu",
        "batch_size",
        "context_length",
        "benchmark_length_mode",
        "benchmark_lengths_sha256",
        "benchmark_seed",
        "num_query_heads",
        "num_kv_heads",
        "query_group_size",
        "head_dim",
        "global_num_query_heads",
        "global_num_kv_heads",
        "tp_size",
        "kv_replication_factor",
        "query_length",
        "page_size",
        "query_dtype",
        "kv_dtype",
        "per_token_kv",
        "trans_v",
        "sliding_window",
        "workgroup_budget",
        "budget_aliases",
        "capacity",
        "is_baseline",
        "status",
        "median_us",
        "min_us",
        "max_us",
        "unique_kv_tb_s",
        "baseline_speedup",
        "is_best",
        "is_conservative",
        "accuracy_json",
        "errors_json",
        "key_json",
        "benchmark_input_json",
        "source_paths_json",
    )
    with Path(path).open(
        "w" if overwrite else "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in data["records"]:
            key, selection = record["key"], record.get("selection") or {}
            for candidate in record["candidates"]:
                benchmark_shape = {**key["shape"], **record["benchmark_input"]}
                row = {k: benchmark_shape.get(k) for k in fields}
                row.update({k: candidate.get(k) for k in fields if k in candidate})
                row.update(
                    key_sha256=record["key_sha256"],
                    shape_status=record["status"],
                    architecture=key["architecture"],
                    num_cu=key["num_cu"],
                    kv_dtype=key["kv_dtype"],
                    budget_aliases=canonical(candidate["budget_aliases"]),
                    is_best=candidate["capacity"] == selection.get("best_capacity"),
                    is_conservative=candidate["capacity"]
                    == selection.get("conservative_capacity"),
                    accuracy_json=canonical(candidate.get("accuracy", {})),
                    errors_json=canonical(
                        candidate.get("errors", []) + record.get("errors", [])
                    ),
                    key_json=canonical(key),
                    benchmark_input_json=canonical(record["benchmark_input"]),
                    benchmark_length_mode=record["benchmark_input"]["length_mode"],
                    benchmark_lengths_sha256=record["benchmark_input"][
                        "lengths_sha256"
                    ],
                    benchmark_seed=record["benchmark_input"]["seed"],
                    source_paths_json=canonical(record.get("source_paths", {})),
                )
                writer.writerow(row)


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
            num_cu,
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
    candidate["accuracy"]["eager"] = _accuracy(torch, output, reference)
    if not candidate["accuracy"]["eager"]["passed"]:
        raise ArithmeticError("eager accuracy failed")
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            launch()
    output.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()
    candidate["accuracy"]["graph"] = _accuracy(torch, output, reference)
    if not candidate["accuracy"]["graph"]["passed"]:
        raise ArithmeticError("graph accuracy failed")
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


def _backend_paths(pa):
    root = Path(__file__).resolve().parent
    paths = {"pa_decode_tuning.py": str(Path(__file__).resolve())}
    for name in SOURCE_FILES[:-1]:
        module = (
            pa
            if name == "pa_decode.py"
            else importlib.import_module(
                "aiter.ops.flydsl." + name.removesuffix(".py").replace("/", ".")
            )
        )
        actual = Path(module.__file__).resolve()
        if actual != (root / name).resolve():
            raise RuntimeError(
                f"Imported {name} is not from this tool's source tree: {actual}"
            )
        paths[name] = str(actual)
    return paths


def tune_shape(
    torch,
    pa,
    shape,
    model,
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
        "key_sha256": fingerprint(key),
        "model": model,
        "status": "RUNNING",
        "candidates": candidates,
        "selection": None,
        "errors": [],
        "unique_kv_bytes": unique_kv_bytes(shape),
        "round_orders": [],
        "benchmark_input": {k: copy.deepcopy(shape[k]) for k in BENCHMARK_FIELDS},
        "fp32_reference": {
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    live = {}
    try:
        record["source_paths"] = _backend_paths(pa)
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
                except Exception as exc:  # noqa: BLE001 - Persist candidate failures.
                    c["status"] = _error_status(torch, exc)
                    c["errors"].append(f"{type(exc).__name__}: {exc}")
                    if c["is_baseline"]:
                        record["status"] = "BASELINE_FAILED"
                        return record
            ready = [c for c in candidates if c["status"] == "READY"]
            for r in range(rounds):
                order = [ready[i] for i in rotated_order(len(ready), r)]
                record["round_orders"].append([c["capacity"] for c in order])
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
                resource["output"].fill_(math.nan)
                resource["graph"].replay()
                torch.cuda.synchronize()
                c["accuracy"]["post_timing"] = _accuracy(
                    torch, resource["output"], reference
                )
                plan = resource["plan"]
                c["plan_unchanged"] = bool(
                    torch.equal(plan.work_info, resource["snapshots"][0])
                    and torch.equal(plan.reduce_info, resource["snapshots"][1])
                )
                c["status"] = (
                    "PASS"
                    if c["accuracy"]["post_timing"]["passed"] and c["plan_unchanged"]
                    else "FAIL"
                )
        record["selection"] = summarize_candidates(
            candidates, record["unique_kv_bytes"]
        )
        record["status"] = "PASS" if record["selection"] else "BASELINE_FAILED"
    except Exception as exc:  # noqa: BLE001 - Preserve partial results on failure.
        record["status"] = _error_status(torch, exc)
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record["selection"] = None
    finally:
        try:
            record["sources_after"] = source_identity()
        except OSError as exc:
            record["sources_after"] = None
            record["errors"].append(f"Source check failed: {exc}")
        if record["sources_after"] != key["sources"]:
            record["status"], record["selection"] = "SOURCE_CHANGED", None
            record["errors"].append("Source changed during tuning")
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
    paths = _backend_paths(pa)
    return (
        torch,
        pa,
        {
            "architecture": architecture,
            "num_cu": cu,
            "device": args.device,
            "name": props.name,
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "source_paths": paths,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", choices=tuple(PRESETS))
    source.add_argument(
        "--config",
        type=Path,
        help="Local HF config.json; nested text_config is supported",
    )
    parser.add_argument(
        "--shape",
        type=lambda s: _list_arg(s, unique=False),
        help="Global Hq,Hkv,D; individual head overrides take precedence",
    )
    parser.add_argument("--query-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--allow-kv-replication", action="store_true")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.list_models:
        print(
            json.dumps(
                {
                    n: {"Hq": v[0], "Hkv": v[1], "D": v[2], "description": v[3]}
                    for n, v in PRESETS.items()
                },
                indent=2,
            )
        )
        return 0
    try:
        if args.shape is not None and len(args.shape) != 3:
            raise ValueError("--shape requires exactly Hq,Hkv,D")
        triple = args.shape or (None, None, None)
        model = resolve_model(
            args.model,
            args.config,
            num_query_heads=(
                args.query_heads if args.query_heads is not None else triple[0]
            ),
            num_kv_heads=args.kv_heads if args.kv_heads is not None else triple[1],
            head_dim=args.head_dim if args.head_dim is not None else triple[2],
            tp_size=args.tp_size,
            allow_kv_replication=args.allow_kv_replication,
        )
        if any(mode not in ("per_token", "per_tensor") for mode in args.quant_modes):
            raise ValueError("quant-modes must be per_token and/or per_tensor")
        layouts = [True, False] if args.trans_v == "both" else [args.trans_v == "yes"]
        shapes = [
            make_shape(
                model,
                b,
                ctx,
                ql,
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
            args.output = args.output.resolve()
            args.csv = args.csv or args.output.with_suffix(".csv")
            args.csv = args.csv.resolve()
            if args.output == args.csv or args.output.exists() or args.csv.exists():
                raise FileExistsError("Output/CSV must be distinct new paths")
            if not args.output.parent.is_dir() or not args.csv.parent.is_dir():
                raise FileNotFoundError(
                    "Output/CSV parent directories must already exist"
                )
            torch, pa, info = _load_gpu(args)
        candidates = args.budgets or list(DEFAULT_BUDGETS)
        if args.budget_cu is not None:
            scaled = [x * info["num_cu"] for x in args.budget_cu]
            if any(not _finite_positive(x) or not x.is_integer() for x in scaled):
                raise ValueError(
                    "CU-multiple budgets must produce positive integer workgroup counts"
                )
            candidates = [int(x) for x in scaled]
        preview = [
            {
                "key": make_key(s, info["architecture"], info["num_cu"]),
                "model": model,
                "benchmark_input": {k: s[k] for k in BENCHMARK_FIELDS},
                "candidates": candidate_groups(
                    candidates, s["batch_size"], s["num_kv_heads"], info["num_cu"]
                ),
            }
            for s in shapes
        ]
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(
            json.dumps({"dry_run": True, "shapes": preview}, indent=2, allow_nan=False)
        )
        return 0
    data = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": info,
        "scope": "Synthetic attention shapes, not model inference; logical unique-KV bandwidth, not HBM traffic",
        "upstream_reference": UPSTREAM,
        "protocol": {
            "atol": ATOL,
            "rtol": RTOL,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "plan_refresh_in_timing": False,
            "native_query_splits": True,
            "baseline_budget": 2 * info["num_cu"],
            "near_best_speedup_fraction": NEAR_BEST,
        },
        "records": [],
    }
    save_results(args.output, data)
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
                model,
                info,
                candidates,
                rounds=args.rounds,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            data["records"].append(result)
            save_results(args.output, data, overwrite=True)
            print(f"RESULT {result['status']} {result['selection']}", flush=True)
            torch.cuda.empty_cache()
    finally:
        data["status"] = (
            "COMPLETE"
            if len(data["records"]) == len(shapes)
            and all(r["status"] == "PASS" for r in data["records"])
            else "COMPLETE_WITH_FAILURES"
        )
        save_results(args.output, data, overwrite=True)
        save_csv(args.csv, data)
    return 0 if data["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
