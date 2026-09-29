# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline PA-decode tuning backed by FlyDSL's Autotuner and config cache.

Run this file with --help for the shape sweep options. --help and --dry-run
need only Python. GPU runs search on a cache miss; FLYDSL_AUTOTUNE=1 forces
a fresh search. FLYDSL_AUTOTUNE_CACHE_DIR controls FlyDSL's cache directory.
An optional CSV is a measurement report, not a runtime configuration file.

Each candidate uses a fixed plan and a graph containing decode plus reduction.
FlyDSL validates candidates, measures them, and caches the smallest budget
within 97% of the fastest measured performance. Runtime lookup never benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import math
import os
import random
import statistics
import sys
from contextlib import contextmanager
from functools import lru_cache
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


def unique_kv_bytes(shape):
    window, ql = shape["sliding_window"], shape["query_length"]
    tokens = sum(min(n, window + ql - 1) if window else n for n in shape["lengths"])
    return 2 * shape["num_kv_heads"] * shape["head_dim"] * tokens


def _contiguous_strides(shape):
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def _synthetic_storage_key(shape):
    h, d, page = (shape[k] for k in ("num_kv_heads", "head_dim", "page_size"))
    pages = sum((n + page - 1) // page for n in shape["lengths"])
    extent = pages * h * d * page
    vshape = (pages, h, page // 16, d, 16) if shape["trans_v"] else (pages, h, d, page)
    scales = _contiguous_strides((pages, h, page)) if shape["per_token_kv"] else ()
    return (
        (shape["num_query_heads"] * d, d, 1),
        _contiguous_strides((pages, h, d // 16, page, 16)),
        _contiguous_strides(vshape),
        scales,
        scales,
        extent >= 2**31,
        extent < 2**32,
    )


def storage_key(query, key, value, key_scale, value_scale):
    """Host-only layout/address specialization shared by search and lookup."""

    def scale_strides(scale):
        if scale.numel() == 1:
            return ()
        strides = tuple(scale.stride())
        # pa_decode normalizes (..., page, 1) per-token scales with squeeze(-1).
        return strides[:-1] if scale.ndim == 4 and scale.shape[-1] == 1 else strides

    extent = max(key.numel(), value.numel())
    return (
        tuple(query.stride()),
        tuple(key.stride()),
        tuple(value.stride()),
        scale_strides(key_scale),
        scale_strides(value_scale),
        extent >= 2**31,
        extent < 2**32,
    )


@lru_cache(maxsize=1)
def _implementation_hash():
    # FlyDSL supplies compiler/environment fingerprints. Include the PA sources
    # as well: a budget measured before a kernel/policy change must not be reused.
    root = Path(__file__).resolve().parent
    files = sorted((root / "kernels" / "pa_decode").glob("*.py"))
    files += [root / "pa_decode.py", Path(__file__).resolve()]
    files += [
        root / "kernels" / name
        for name in (
            "pa_decode_kernel.py",
            "pa_decode_reduce.py",
            "pa_decode_plan.py",
            "buffer_ops.py",
            "tensor_shim.py",
            "kernels_common.py",
            "dpp_utils.py",
            "utils.py",
        )
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _OfflineTuningRequired(RuntimeError):
    pass


def _pa_decode_budget(
    shape_key,
    architecture,
    num_cu,
    implementation,
    storage_key,
    *,
    workgroup_budget,
    tuning_session=None,
):
    tuning_session.launch(workgroup_budget)


def _configs(*args, tuning_session=None, **kwargs):
    if tuning_session is None:
        # FLYDSL_AUTOTUNE bypasses even a runtime default. Keep budget lookup
        # a lookup-only operation when that flag is inherited.
        raise _OfflineTuningRequired("Run the offline PA tuner to measure budgets")
    return tuning_session.configs()


@contextmanager
def _validate_config(arguments):
    with arguments["tuning_session"].validate(arguments["workgroup_budget"]):
        yield


def _select_config(results):
    """Prefer the smallest budget within the existing 97% performance band."""
    best = min(elapsed for _, elapsed in results)
    return min(
        (
            (config, elapsed)
            for config, elapsed in results
            if best / elapsed >= NEAR_BEST - 1e-12
        ),
        key=lambda pair: pair[0].kwargs["workgroup_budget"],
    )


def _default_config(*args, num_cu, **kwargs):
    from flydsl.autotune import Config

    return Config(workgroup_budget=2 * num_cu)


def _make_autotuner(session=None):
    from flydsl.autotune import autotune

    return autotune(
        configs=_configs,
        key=["shape_key", "architecture", "num_cu", "implementation", "storage_key"],
        warmup=0,
        rep=session.rounds if session is not None else 1,
        validate_hook=_validate_config,
        do_bench=session.benchmark if session is not None else None,
        select_config=session.select if session is not None else _select_config,
        default=None if session is not None else _default_config,
    )(_pa_decode_budget)


@lru_cache(maxsize=8)
def _get_runtime_tuner(cache_dir):
    return _make_autotuner()


def _resolve_config(shape, architecture, num_cu, session=None, *, storage_key=None):
    arguments = {
        "shape_key": tuple(shape[field] for field in STATIC_SHAPE_FIELDS),
        "architecture": architecture,
        "num_cu": num_cu,
        "implementation": _implementation_hash(),
        "storage_key": (
            _synthetic_storage_key(shape) if storage_key is None else storage_key
        ),
    }
    cache_dir = os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR")
    tuner = (
        _get_runtime_tuner(cache_dir) if session is None else _make_autotuner(session)
    )
    config = tuner.resolve_config(**arguments, tuning_session=session)
    values = config.all_kwargs()
    if (
        set(values) != {"workgroup_budget"}
        or type(values["workgroup_budget"]) is not int
        or values["workgroup_budget"] <= 0
        or config.compiler_opts()
        or config.pre_hook is not None
    ):
        raise ValueError(
            "Invalid PA budget in FlyDSL cache; use FLYDSL_AUTOTUNE=1 to retune"
        )
    if session is not None:
        # Both objects use FlyDSL's cache/key format. Make a completed search
        # visible to a runtime resolver already constructed in this process.
        _get_runtime_tuner(cache_dir).cache.update(tuner.cache)
    return config


def get_cached_budget(shape, architecture, num_cu, *, storage_key=None):
    """Read FlyDSL's config cache, or use 2*CU; never compile or benchmark.

    Call before graph capture when preparing an explicit work plan. Pass the
    actual tensor storage key to distinguish packed or padded caches from the
    tuner's synthetic contiguous inputs. Use the current CUDA device's metadata.
    """
    try:
        return _resolve_config(
            shape, architecture, num_cu, storage_key=storage_key
        ).kwargs["workgroup_budget"]
    except (_OfflineTuningRequired, ValueError):
        return 2 * num_cu


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


class _BaselineFailed(ArithmeticError):
    pass


class _TuningSession:
    """PA resources and correctness hooks for one FlyDSL config search."""

    def __init__(
        self, torch, pa, shape, info, candidates, rounds, iterations, warmup, stream
    ):
        self.torch, self.pa, self.problem, self.info = torch, pa, shape, info
        self.candidates = {c["workgroup_budget"]: c for c in candidates}
        self.rounds, self.iterations, self.warmup = rounds, iterations, warmup
        self.stream = stream
        self.resources = {}
        self.current = None
        self.baseline = next(c for c in candidates if c["is_baseline"])

    def _prepare(self, candidate):
        budget = candidate["workgroup_budget"]
        try:
            self.resources[budget] = _prepare_candidate(
                self.torch,
                self.pa,
                self.inputs,
                self.problem,
                candidate,
                self.reference,
                self.info["num_cu"],
                self.iterations,
                self.warmup,
                self.stream,
            )
            candidate["status"] = "READY"
        except Exception as exc:
            self._failure(candidate, exc)
            raise

    def _failure(self, candidate, exc):
        candidate["status"] = _error_status(self.torch, exc)
        candidate["errors"].append(f"{type(exc).__name__}: {exc}")

    def configs(self):
        from flydsl.autotune import Config

        torch = self.torch
        device = torch.device("cuda", self.info["device"])
        fp8 = (
            torch.float8_e4m3fn
            if self.info["architecture"] == "gfx950"
            else torch.float8_e4m3fnuz
        )
        self.inputs = _make_inputs(torch, self.problem, device, fp8)
        actual = storage_key(
            *(
                self.inputs[name]
                for name in ("query", "key", "value", "key_scale", "value_scale")
            )
        )
        if actual != _synthetic_storage_key(self.problem):
            raise RuntimeError("Synthetic storage key disagrees with allocated inputs")
        self.reference = _reference(torch, self.inputs, self.problem)
        if not bool(torch.isfinite(self.reference).all().item()):
            raise RuntimeError("FP32 reference is nonfinite")
        # A failed baseline aborts before FlyDSL starts searching other configs.
        try:
            self._prepare(self.baseline)
        except Exception as exc:
            raise _BaselineFailed("Baseline preparation failed") from exc
        return [
            Config(workgroup_budget=c["workgroup_budget"])
            for c in sorted(
                self.candidates.values(), key=lambda c: not c["is_baseline"]
            )
        ]

    @contextmanager
    def validate(self, budget):
        candidate = self.candidates[budget]
        self.current = candidate
        if budget not in self.resources:
            self._prepare(candidate)
        try:
            yield
            self.torch.cuda.synchronize()
            candidate["graph_accuracy"] = _accuracy(
                self.torch, self.resources[budget]["output"], self.reference
            )
            if not candidate["graph_accuracy"]["passed"]:
                raise ArithmeticError("graph accuracy failed")
        except Exception as exc:
            self._failure(candidate, exc)
            raise

    def launch(self, budget):
        self.resources[budget]["graph"].replay()

    def benchmark(self, fn, *, warmup, rep):
        from flydsl.autotune import do_bench

        candidate = self.current
        try:
            # Every replay contains iterations native decode+reduce calls.
            # FlyDSL supplies the GPU backlog and event timing. Search order
            # is owned by Autotuner; these are per-config repeated samples.
            for _ in range(rep):
                elapsed = do_bench(fn, warmup=warmup, rep=1) * 1000 / self.iterations
                if not _finite_positive(elapsed):
                    raise RuntimeError("Nonfinite or nonpositive FlyDSL timing")
                candidate["samples_us"].append(elapsed)
            resource = self.resources[candidate["workgroup_budget"]]
            plan = resource["plan"]
            candidate["plan_unchanged"] = bool(
                self.torch.equal(plan.work_info, resource["snapshots"][0])
                and self.torch.equal(plan.reduce_info, resource["snapshots"][1])
            )
            if not candidate["plan_unchanged"]:
                raise ArithmeticError("Work plan changed during benchmarking")
            samples = candidate["samples_us"]
            candidate.update(
                status="PASS",
                median_us=statistics.median(samples),
                min_us=min(samples),
                max_us=max(samples),
                unique_kv_tb_s=unique_kv_bytes(self.problem)
                / statistics.median(samples)
                / 1e6,
            )
            return candidate["median_us"] / 1000
        except Exception as exc:
            self._failure(candidate, exc)
            raise

    def select(self, results):
        if not any(
            config.kwargs["workgroup_budget"] == self.baseline["workgroup_budget"]
            for config, _ in results
        ):
            raise _BaselineFailed("Baseline validation or measurement failed")
        return _select_config(results)


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
    """Search on a FlyDSL cache miss; cached results allocate no benchmark data."""
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
    for candidate in candidates:
        candidate.update(status="NOT_RUN", samples_us=[], errors=[])
    record = {
        "key": key,
        "status": "RUNNING",
        "candidates": candidates,
        "selection": None,
        "errors": [],
    }
    session = None
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
            session = _TuningSession(
                torch,
                pa,
                shape,
                device_info,
                candidates,
                rounds,
                iterations,
                warmup,
                stream,
            )
            config = _resolve_config(shape, key["architecture"], key["num_cu"], session)
            record.update(
                status="PASS",
                cache_hit=not bool(session.resources),
                selection={"workgroup_budget": config.kwargs["workgroup_budget"]},
            )
    except Exception as exc:  # noqa: BLE001 - Preserve the failed search report.
        record["status"] = (
            "BASELINE_FAILED"
            if isinstance(exc, _BaselineFailed)
            else _error_status(torch, exc)
        )
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if session is not None:
            session.resources.clear()
    return record


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
    parser.add_argument(
        "--rounds",
        type=int,
        default=16,
        help="Repeated graph measurements per candidate",
    )
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
        "--output",
        type=Path,
        help="Optional measurement report; runtime configurations use FlyDSL cache",
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
            if args.output is not None:
                if args.output.suffix.lower() != ".csv":
                    raise ValueError("Measurement reports require a .csv output path")
                if args.output.exists() or args.output.is_symlink():
                    raise FileExistsError("Output CSV must be a new path")
                if not args.output.parent.is_dir():
                    raise FileNotFoundError(
                        "Output CSV parent directory must already exist"
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

    report = (
        args.output.open("x", newline="", encoding="utf-8") if args.output else None
    )
    writer = (
        csv.DictWriter(report, fieldnames=CSV_FIELDS, lineterminator="\n")
        if report
        else None
    )
    if writer:
        writer.writeheader()
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
            all_passed &= result["status"] == "PASS"
            if writer and result["status"] == "PASS":
                writer.writerows(
                    _csv_row(result["key"], candidate)
                    for candidate in result["candidates"]
                    if candidate["status"] == "PASS"
                )
                report.flush()
            print(
                f"RESULT {result['status']} cache_hit={result.get('cache_hit', False)} {result['selection']}",
                flush=True,
            )
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
        if report:
            report.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
