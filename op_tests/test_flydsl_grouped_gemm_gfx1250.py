#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 grouped MoE GEMM tests through ``aiter.fused_moe``.

Two formats covered:

* **a4w4** -- MXFP4 activations × MXFP4 weights (``w1.dtype = fp4x2``).
* **a8w4** -- MXFP8 activations × MXFP4 weights (``w1.dtype = uint8``).

Both go through the public ``fused_moe`` API; we never call the underlying
grouped GEMM launcher directly. The grouped path is opted-in via the
``AITER_USE_GROUPED_GEMM=1`` env (set automatically by the runner below).

Pytest covers a small correctness case for each format. Direct execution
(``python op_tests/test_flydsl_grouped_gemm_gfx1250.py``) runs a
DeepSeek-style perf bench (``--scenario bench``, end-to-end fused_moe), a
per-kernel bench that times gemm1 and gemm2 in isolation
(``--scenario kernel``), a tiny correctness check
(``--scenario verify``), or a full sweep of every setting in a tuned-config
CSV (``--scenario csv``; defaults to ``aiter/configs/tuned_grouped_fmoe.csv``,
one benched case per row, override the file with ``--csv-path``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from contextlib import nullcontext

import pytest
import torch

from aiter import ActivationType, QuantType, logger
from aiter.aot.flydsl.common import run_only_env
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.flydsl.moe_common import GateMode, apply_gate_up
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.utility import dtypes, fp4_utils

# Build every tensor straight on the device (like op_tests/test_moe_2stage.py) so
# the test body has no `.cuda()` / `.float().cuda()` plumbing.
torch.set_default_device("cuda")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# Routing: normal (random) by default; round-robin balanced only when
# AITER_MOE_EXPERT_BALANCE=1 (mirrors op_tests/test_moe_2stage.py).
AITER_MOE_EXPERT_BALANCE = (
    os.environ.get("AITER_MOE_EXPERT_BALANCE", "False").lower() == "true"
)


# Force topk to activate only the first n experts (ids 0..n-1). 0 = unset.
# Takes precedence over AITER_MOE_EXPERT_BALANCE when set (> 0).
def parse_num_expert_activated():
    try:
        val = int(os.environ.get("AITER_MOE_NUM_EXPERT_ACTIVATED", "0"))
    except ValueError:
        raise ValueError("AITER_MOE_NUM_EXPERT_ACTIVATED must be an integer")
    if val < 0:
        raise ValueError(f"AITER_MOE_NUM_EXPERT_ACTIVATED must be >= 0, got {val}")
    return val


AITER_MOE_NUM_EXPERT_ACTIVATED = parse_num_expert_activated()

SCALE_BLOCK = 32
DEFAULT_SCALE_BYTE = 127  # e8m0 byte for 2^0 = 1.0
# User-facing TB/s applies the requested binary/decimal conversion:
# 1.024**4 = 2**40 / 10**12.
TBPS_DIVISOR = 1.024 ** 4
_ACT_BY_NAME = {
    "silu": ActivationType.Silu,
    "swiglu": ActivationType.Swiglu,
    "situv2": ActivationType.Situv2,
}
_STAGE1_ACT_ID = {
    ActivationType.Silu: 1,
    ActivationType.Swiglu: 2,
    ActivationType.Situv2: 3,
}


def _bytes_per_microsecond_to_tbps(
    bytes_count: int | float,
    time_us: int | float,
) -> float:
    return bytes_count / time_us / 1e6 / TBPS_DIVISOR


_GEMM_SUMMARY_COLUMNS = (
    "gemm1 executed",
    "gemm1 effective read",
    "gemm1 output write",
    "gemm1 effective R+W",
    "gemm2 executed",
    "gemm2 effective read",
    "gemm2 output write",
    "gemm2 effective R+W",
)
_TDM_KERNEL_RE = re.compile(
    r"a8w4_tdm_(?P<a_format>fp4|fp8)"
    r"_t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)"
    r"_w(?P<m_warp>\d+)x(?P<n_warp>\d+)_b(?P<num_buffers>\d+)"
    r"_K(?P<K>\d+)_e(?P<experts>\d+)"
    r"(?:_act(?P<stage1_act>\d+))?"
    r"(?P<has_bias>_bias)?"
    r"(?:_q(?P<quant_out>\d+)r(?P<quant_wmma_rep>\d+))?"
    r"(?:_cn(?P<cluster_n>\d+))?"
    r"(?:_prefetch)?"
    r"(?:_wpt\d+)?"
)
_MOE_GEMM1_ASM_SYMBOL = (
    "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1"
)

VERIFY_TOL_A4W4 = 0.02
VERIFY_TOL_A8W4 = 0.02
# Production MoE accuracy gate (matches op_tests/test_moe_2stage.py calc_diff):
# logits_diff = ||x-y||^2 / (||x||^2 + ||y||^2).  rel_l2 is kept as an
# informational print only; logits_diff < 0.01 is the actual pass/fail gate.
LOGITS_DIFF_TOL = 0.01


def _empty_gemm_summary() -> dict[str, str]:
    return {column: "N/A" for column in _GEMM_SUMMARY_COLUMNS}


def _calculate_effective_stage_metrics(
    *,
    route_counts: list[int],
    tile_m: int,
    tile_n: int,
    tile_k: int,
    n: int,
    k: int,
    output_n: int,
    latency_us: float,
    cluster_n: int,
    a_format: str,
    has_bias: bool,
    stage1_act: int,
    quant_out: int,
) -> dict[str, int | float | str]:
    """Pure-Python executed-work and unique-logical-byte accounting.

    Compute uses tile-aligned rows actually executed by WMMA. Effective traffic
    instead counts each valid A row and each active expert's B/bias surface once,
    independent of M/N tiling and cluster multicast. It is a useful-work metric,
    not requested bytes or measured HBM traffic; LDS and metadata are excluded.
    """
    if not route_counts or any(count < 0 for count in route_counts):
        raise ValueError("route_counts must contain non-negative expert counts")
    if min(tile_m, tile_n, tile_k, n, k, output_n) <= 0:
        raise ValueError("tile and problem dimensions must be positive")
    if cluster_n < 1:
        raise ValueError("cluster_n must be positive")
    if n % tile_n or k % tile_k:
        raise ValueError("N and K must be exact multiples of the observed tiles")
    if a_format not in ("fp4", "fp8"):
        raise ValueError(f"unsupported activation format: {a_format}")
    if k % 2 or k % SCALE_BLOCK:
        raise ValueError("K must be divisible by FP4 packing and the scale block")
    if latency_us <= 0:
        raise ValueError("kernel latency must be positive")
    if quant_out not in (0, 1):
        raise ValueError(f"unsupported quant_out={quant_out}")
    if quant_out and (a_format != "fp8" or not stage1_act or has_bias):
        raise ValueError("fused quant output requires bias-free A8W4 stage1")

    active_m_tiles = sum((count + tile_m - 1) // tile_m for count in route_counts)
    active_experts = sum(count > 0 for count in route_counts)
    valid_rows = sum(route_counts)
    aligned_rows = active_m_tiles * tile_m

    # Unique logical A/scale rows are counted once; B/scale surfaces once per
    # active expert. tile_n, M-tile replication, alignment padding, and cluster
    # multicast intentionally do not affect effective bytes.
    a_row_bytes = k // 2 if a_format == "fp4" else k
    a_payload_bytes = valid_rows * a_row_bytes
    a_scale_bytes = valid_rows * (k // SCALE_BLOCK)
    b_payload_bytes = active_experts * n * (k // 2)
    b_scale_bytes = active_experts * n * (k // SCALE_BLOCK)
    # The wrapper converts bias to the BF16 output dtype. Count one logical
    # bias vector per active expert, not its repeated per-row kernel loads.
    bias_read_bytes = active_experts * n * 2 if has_bias else 0
    effective_read_bytes = (
        a_payload_bytes
        + a_scale_bytes
        + b_payload_bytes
        + b_scale_bytes
        + bias_read_bytes
    )

    if quant_out:
        if output_n % SCALE_BLOCK:
            raise ValueError("quantized output width must be block-32 aligned")
        # q1 in the generated symbol means the GEMM epilogue itself stores both
        # MXFP8 payload and direct-global E8M0 scale bytes.
        output_payload_bytes = valid_rows * output_n  # MXFP8: 1 B/value
        output_scale_bytes = valid_rows * (output_n // SCALE_BLOCK)
        output_kind = "FP8+E8M0"
    else:
        # This test always requests BF16 fused_moe output. In the biased A8W4
        # path GEMM1 is qout0, so its later standalone quant kernel is outside
        # this GEMM kernel's logical output writes and timing.
        output_payload_bytes = valid_rows * output_n * 2
        output_scale_bytes = 0
        output_kind = "BF16"
    output_write_bytes = output_payload_bytes + output_scale_bytes
    executed_flops = 2 * aligned_rows * n * k
    effective_read_write_bytes = effective_read_bytes + output_write_bytes

    return {
        "valid_rows": valid_rows,
        "aligned_rows": aligned_rows,
        "active_m_tiles": active_m_tiles,
        "active_experts": active_experts,
        "a_payload_bytes": a_payload_bytes,
        "a_scale_bytes": a_scale_bytes,
        "b_payload_bytes": b_payload_bytes,
        "b_scale_bytes": b_scale_bytes,
        "bias_read_bytes": bias_read_bytes,
        "output_payload_bytes": output_payload_bytes,
        "output_scale_bytes": output_scale_bytes,
        "output_kind": output_kind,
        "executed_flops": executed_flops,
        "effective_read_bytes": effective_read_bytes,
        "output_write_bytes": output_write_bytes,
        "effective_read_write_bytes": effective_read_write_bytes,
        "executed_tflops": executed_flops / latency_us / 1e6,
        "effective_read_tbps": _bytes_per_microsecond_to_tbps(
            effective_read_bytes, latency_us
        ),
        "output_write_tbps": _bytes_per_microsecond_to_tbps(
            output_write_bytes, latency_us
        ),
        "effective_read_write_tbps": _bytes_per_microsecond_to_tbps(
            effective_read_write_bytes, latency_us
        ),
    }


def _format_effective_stage_metrics(
    stage: dict[str, int | float | str],
) -> tuple[str, ...]:
    return (
        f"{stage['executed_flops']:,} FLOP \u2192 "
        f"{stage['executed_tflops']:,.1f} TFLOP/s",
        f"{stage['effective_read_bytes']:,} B \u2192 "
        f"{stage['effective_read_tbps']:.3f} TB/s",
        f"{stage['output_write_bytes']:,} B ({stage['output_kind']}) \u2192 "
        f"{stage['output_write_tbps']:.3f} TB/s",
        f"{stage['effective_read_write_bytes']:,} B \u2192 "
        f"{stage['effective_read_write_tbps']:.3f} TB/s",
    )


def _parse_tdm_kernel_name(name: str) -> dict[str, int | str | bool] | None:
    """Parse the exact generated TDM symbol inside an optional trace prefix."""
    if _MOE_GEMM1_ASM_SYMBOL in name:
        return {
            "name": _MOE_GEMM1_ASM_SYMBOL,
            "a_format": "fp4",
            "tile_m": 256,
            "tile_n": 256,
            "tile_k": 256,
            "m_warp": 2,
            "n_warp": 2,
            "num_buffers": 4,
            "K": 7168,
            "experts": 96,
            "stage1_act": 1,
            "quant_out": 0,
            "quant_wmma_rep": 1,
            "cluster_n": 4,
            "has_bias": False,
        }
    match = _TDM_KERNEL_RE.search(name)
    if match is None:
        return None
    parsed: dict[str, int | str | bool] = {
        key: int(match.group(key))
        for key in (
            "tile_m",
            "tile_n",
            "tile_k",
            "m_warp",
            "n_warp",
            "num_buffers",
            "K",
            "experts",
        )
    }
    parsed.update(
        {
            "name": match.group(0),
            "a_format": match.group("a_format"),
            "stage1_act": int(match.group("stage1_act") or 0),
            "quant_out": int(match.group("quant_out") or 0),
            "quant_wmma_rep": int(match.group("quant_wmma_rep") or 1),
            "cluster_n": int(match.group("cluster_n") or 1),
            "has_bias": match.group("has_bias") is not None,
        }
    )
    return parsed


def _extract_profiled_gemm_timings(
    *,
    trace_df,
    data_format: str,
    activation: ActivationType,
    experts: int,
    model_dim: int,
    inter_dim: int,
) -> tuple[
    dict[str, float | None],
    dict[str, dict[str, int | float | str | bool]],
    str | None,
]:
    """Extract unique GEMM stage timings from the post-processed profiler rows.

    Kernel symbols are generated as ``a8w4_tdm_fp{4,8}_t..._K..._e...``.
    Searching permits profiler namespace/mangling text around that exact symbol;
    format, K, expert count, and stage activation must still all agree.
    """
    timings: dict[str, float | None] = {"gemm1_us": None, "gemm2_us": None}
    if trace_df is None:
        return timings, {}, "trace has no profiler table"
    expected_a_format = {"a4w4": "fp4", "a8w4": "fp8"}.get(data_format)
    stage1_act = _STAGE1_ACT_ID.get(activation)
    if expected_a_format is None or stage1_act is None:
        return timings, {}, f"unsupported format/activation: {data_format}"

    candidates = []
    for rec in trace_df.to_dict("records"):
        parsed = _parse_tdm_kernel_name(str(rec.get("name", "")))
        if (
            parsed is None
            or parsed["a_format"] != expected_a_format
            or parsed["experts"] != experts
        ):
            continue
        try:
            latency_us = float(rec.get("device_time_avg"))
        except (TypeError, ValueError):
            continue
        if not latency_us > 0:
            continue
        parsed["latency_us"] = latency_us
        try:
            parsed["count"] = int(rec.get("cnt"))
        except (TypeError, ValueError):
            parsed["count"] = None
        candidates.append(parsed)

    stage_matches = {
        "gemm1": [
            cfg
            for cfg in candidates
            if cfg["K"] == model_dim and cfg["stage1_act"] == stage1_act
        ],
        "gemm2": [
            cfg
            for cfg in candidates
            if cfg["K"] == inter_dim and cfg["stage1_act"] == 0
        ],
    }
    kernels = {}
    debug = []
    for stage, matches in stage_matches.items():
        if len(matches) == 1:
            kernels[stage] = matches[0]
            timings[f"{stage}_us"] = float(matches[0]["latency_us"])
        else:
            state = "missing" if not matches else f"ambiguous ({len(matches)} matches)"
            debug.append(f"{stage} {state}")
    return timings, kernels, "; ".join(debug) or None


def _build_effective_summary(
    *,
    profiled_kernels: dict[str, dict[str, int | float | str | bool]],
    route_counts: list[int],
    data_format: str,
    use_bias: bool,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
) -> dict[str, str]:
    """Build eight A4W4/A8W4 effective columns or explicit N/A values.

    Effective bytes are tile/cluster-independent unique logical traffic.
    Incomplete profiler matches, inconsistent bias/format flags, and unsupported
    fused-quant modes remain guarded rather than estimated.
    """
    empty = _empty_gemm_summary()
    expected_a_format = {"a4w4": "fp4", "a8w4": "fp8"}.get(data_format)
    if expected_a_format is None:
        return empty
    if len(route_counts) != experts or sum(route_counts) != tokens * topk:
        return empty
    gemm1 = profiled_kernels.get("gemm1")
    gemm2 = profiled_kernels.get("gemm2")
    if gemm1 is None or gemm2 is None:
        return empty
    if (
        gemm1["a_format"] != expected_a_format
        or gemm2["a_format"] != expected_a_format
        or bool(gemm1["has_bias"]) != use_bias
        or bool(gemm2["has_bias"]) != use_bias
    ):
        return empty

    try:
        stage1 = _calculate_effective_stage_metrics(
            route_counts=route_counts,
            tile_m=int(gemm1["tile_m"]),
            tile_n=int(gemm1["tile_n"]),
            tile_k=int(gemm1["tile_k"]),
            n=2 * inter_dim,
            k=model_dim,
            output_n=inter_dim,
            latency_us=float(gemm1["latency_us"]),
            cluster_n=int(gemm1["cluster_n"]),
            a_format=str(gemm1["a_format"]),
            has_bias=bool(gemm1["has_bias"]),
            stage1_act=int(gemm1["stage1_act"]),
            quant_out=int(gemm1["quant_out"]),
        )
        stage2 = _calculate_effective_stage_metrics(
            route_counts=route_counts,
            tile_m=int(gemm2["tile_m"]),
            tile_n=int(gemm2["tile_n"]),
            tile_k=int(gemm2["tile_k"]),
            n=model_dim,
            k=inter_dim,
            output_n=model_dim,
            latency_us=float(gemm2["latency_us"]),
            cluster_n=int(gemm2["cluster_n"]),
            a_format=str(gemm2["a_format"]),
            has_bias=bool(gemm2["has_bias"]),
            stage1_act=int(gemm2["stage1_act"]),
            quant_out=int(gemm2["quant_out"]),
        )
    except (TypeError, ValueError):
        return empty

    return dict(
        zip(
            _GEMM_SUMMARY_COLUMNS,
            _format_effective_stage_metrics(stage1)
            + _format_effective_stage_metrics(stage2),
        )
    )


def test_a4w4_effective_stage_metrics_fixture():
    """CPU-only arithmetic fixture for the 512-token balanced benchmark."""
    route_counts = [32] * 96
    gemm1 = _calculate_effective_stage_metrics(
        route_counts=route_counts,
        tile_m=64,
        tile_n=256,
        tile_k=256,
        n=2 * 3072,
        k=7168,
        output_n=3072,
        latency_us=144.7,
        cluster_n=1,
        a_format="fp4",
        has_bias=False,
        stage1_act=1,
        quant_out=0,
    )
    gemm2 = _calculate_effective_stage_metrics(
        route_counts=route_counts,
        tile_m=64,
        tile_n=256,
        tile_k=256,
        n=7168,
        k=3072,
        output_n=7168,
        latency_us=76.1,
        cluster_n=1,
        a_format="fp4",
        has_bias=False,
        stage1_act=0,
        quant_out=0,
    )

    assert gemm1["valid_rows"] == 3072
    assert gemm1["aligned_rows"] == 6144
    assert TBPS_DIVISOR == pytest.approx(1.099511627776)
    for key, bytes_count in (
        ("effective_read_tbps", gemm1["effective_read_bytes"]),
        ("output_write_tbps", gemm1["output_write_bytes"]),
        ("effective_read_write_tbps", gemm1["effective_read_write_bytes"]),
    ):
        decimal_tbps = bytes_count / 144.7 / 1e6
        assert gemm1[key] == pytest.approx(decimal_tbps / TBPS_DIVISOR)
        assert gemm1[key] * TBPS_DIVISOR == pytest.approx(decimal_tbps)
    assert _format_effective_stage_metrics(gemm1) == (
        "541,165,879,296 FLOP \u2192 3,739.9 TFLOP/s",
        "2,257,747,968 B \u2192 14.191 TB/s",
        "18,874,368 B (BF16) \u2192 0.119 TB/s",
        "2,276,622,336 B \u2192 14.309 TB/s",
    )
    assert gemm2["valid_rows"] == 3072
    assert gemm2["aligned_rows"] == 6144
    assert _format_effective_stage_metrics(gemm2) == (
        "270,582,939,648 FLOP \u2192 3,555.6 TFLOP/s",
        "1,128,038,400 B \u2192 13.482 TB/s",
        "44,040,192 B (BF16) \u2192 0.526 TB/s",
        "1,172,078,592 B \u2192 14.008 TB/s",
    )


def test_effective_bytes_are_tile_independent_fixture():
    """Empty experts, repeated M tiles, tile_n, and cluster do not change bytes."""
    common = {
        "route_counts": [0, 1, 65, 0, 128],
        "n": 256,
        "k": 256,
        "output_n": 128,
        "latency_us": 10.0,
        "a_format": "fp8",
        "has_bias": True,
        "stage1_act": 1,
        "quant_out": 0,
    }
    fine_tiles = _calculate_effective_stage_metrics(
        **common,
        tile_m=16,
        tile_n=64,
        tile_k=128,
        cluster_n=1,
    )
    coarse_tiles_multicast = _calculate_effective_stage_metrics(
        **common,
        tile_m=64,
        tile_n=256,
        tile_k=256,
        cluster_n=4,
    )
    effective_keys = (
        "valid_rows",
        "active_experts",
        "a_payload_bytes",
        "a_scale_bytes",
        "b_payload_bytes",
        "b_scale_bytes",
        "bias_read_bytes",
        "output_write_bytes",
        "effective_read_bytes",
        "effective_read_write_bytes",
        "effective_read_tbps",
        "output_write_tbps",
        "effective_read_write_tbps",
    )
    assert {key: fine_tiles[key] for key in effective_keys} == {
        key: coarse_tiles_multicast[key] for key in effective_keys
    }
    assert fine_tiles["valid_rows"] == 194
    assert fine_tiles["active_experts"] == 3
    assert fine_tiles["aligned_rows"] == 224
    assert coarse_tiles_multicast["aligned_rows"] == 320
    assert fine_tiles["executed_flops"] != coarse_tiles_multicast["executed_flops"]


def test_profiled_grouped_gemm_timing_extraction_fixture():
    """CPU-only trace fixtures for A8W4/A4W4 matching and ambiguity guards."""

    class TraceRecords:
        def __init__(self, records):
            self.records = records

        def to_dict(self, orient):
            assert orient == "records"
            return self.records

    a8_gemm1 = (
        "void aiter::a8w4_tdm_fp8_t16x256x256_w1x4_b2_"
        "K7168_e384_act1_bias(...)"
    )
    a8_gemm2 = (
        "__mangled_prefix_a8w4_tdm_fp8_t16x512x128_w1x4_b2_"
        "K768_e384_bias_wpt1.extra"
    )
    a8_records = [
        {"name": a8_gemm1, "device_time_avg": 496.191},
        {"name": a8_gemm2, "device_time_avg": 261.143},
        # Strict stage filters must ignore all of these plausible distractors.
        {
            "name": "a8w4_tdm_fp8_t16x256x256_w1x4_b2_K7168_e383_act1",
            "device_time_avg": 1.0,
        },
        {
            "name": "a8w4_tdm_fp8_t16x256x256_w1x4_b2_K7168_e384_act2",
            "device_time_avg": 2.0,
        },
        {
            "name": "a8w4_tdm_fp8_t16x512x128_w1x4_b2_K769_e384",
            "device_time_avg": 3.0,
        },
        {
            "name": "a8w4_tdm_fp8_t16x512x128_w1x4_b2_K768_e384",
            "device_time_avg": 0.0,
        },
    ]
    a8_timings, a8_kernels, a8_debug = _extract_profiled_gemm_timings(
        trace_df=TraceRecords(a8_records),
        data_format="a8w4",
        activation=ActivationType.Silu,
        experts=384,
        model_dim=7168,
        inter_dim=768,
    )
    assert a8_timings == {"gemm1_us": 496.191, "gemm2_us": 261.143}
    assert a8_kernels["gemm1"]["a_format"] == "fp8"
    assert a8_kernels["gemm1"]["has_bias"] is True
    assert a8_kernels["gemm2"]["stage1_act"] == 0
    assert a8_debug is None
    a8_summary = _build_effective_summary(
        profiled_kernels=a8_kernels,
        route_counts=[64] * 384,
        data_format="a8w4",
        use_bias=True,
        experts=384,
        tokens=4096,
        topk=6,
        model_dim=7168,
        inter_dim=768,
    )
    assert tuple(a8_summary.values()) == (
        "541,165,879,296 FLOP \u2192 1,090.6 TFLOP/s",
        "2,428,895,232 B \u2192 4.452 TB/s",
        "37,748,736 B (BF16) \u2192 0.069 TB/s",
        "2,466,643,968 B \u2192 4.521 TB/s",
        "270,582,939,648 FLOP \u2192 1,036.1 TFLOP/s",
        "1,147,994,112 B \u2192 3.998 TB/s",
        "352,321,536 B (BF16) \u2192 1.227 TB/s",
        "1,500,315,648 B \u2192 5.225 TB/s",
    )
    a8_gemm1_metrics = _calculate_effective_stage_metrics(
        route_counts=[64] * 384,
        tile_m=16,
        tile_n=256,
        tile_k=256,
        n=1536,
        k=7168,
        output_n=768,
        latency_us=496.191,
        cluster_n=1,
        a_format="fp8",
        has_bias=True,
        stage1_act=1,
        quant_out=0,
    )
    assert {
        key: a8_gemm1_metrics[key]
        for key in (
            "aligned_rows",
            "active_experts",
            "a_payload_bytes",
            "a_scale_bytes",
            "b_payload_bytes",
            "b_scale_bytes",
            "bias_read_bytes",
            "output_payload_bytes",
            "output_scale_bytes",
            "effective_read_bytes",
            "effective_read_write_bytes",
            "executed_flops",
        )
    } == {
        "aligned_rows": 24576,
        "active_experts": 384,
        "a_payload_bytes": 176_160_768,
        "a_scale_bytes": 5_505_024,
        "b_payload_bytes": 2_113_929_216,
        "b_scale_bytes": 132_120_576,
        "bias_read_bytes": 1_179_648,
        "output_payload_bytes": 37_748_736,
        "output_scale_bytes": 0,
        "effective_read_bytes": 2_428_895_232,
        "effective_read_write_bytes": 2_466_643_968,
        "executed_flops": 541_165_879_296,
    }
    a8_gemm2_metrics = _calculate_effective_stage_metrics(
        route_counts=[64] * 384,
        tile_m=16,
        tile_n=512,
        tile_k=128,
        n=7168,
        k=768,
        output_n=7168,
        latency_us=261.143,
        cluster_n=1,
        a_format="fp8",
        has_bias=True,
        stage1_act=0,
        quant_out=0,
    )
    assert {
        key: a8_gemm2_metrics[key]
        for key in (
            "aligned_rows",
            "active_experts",
            "a_payload_bytes",
            "a_scale_bytes",
            "b_payload_bytes",
            "b_scale_bytes",
            "bias_read_bytes",
            "output_payload_bytes",
            "output_scale_bytes",
            "effective_read_bytes",
            "effective_read_write_bytes",
            "executed_flops",
        )
    } == {
        "aligned_rows": 24576,
        "active_experts": 384,
        "a_payload_bytes": 18_874_368,
        "a_scale_bytes": 589_824,
        "b_payload_bytes": 1_056_964_608,
        "b_scale_bytes": 66_060_288,
        "bias_read_bytes": 5_505_024,
        "output_payload_bytes": 352_321_536,
        "output_scale_bytes": 0,
        "effective_read_bytes": 1_147_994_112,
        "effective_read_write_bytes": 1_500_315_648,
        "executed_flops": 270_582_939_648,
    }
    fused_quant = _calculate_effective_stage_metrics(
        route_counts=[16],
        tile_m=16,
        tile_n=64,
        tile_k=128,
        n=64,
        k=128,
        output_n=32,
        latency_us=1.0,
        cluster_n=1,
        a_format="fp8",
        has_bias=False,
        stage1_act=1,
        quant_out=1,
    )
    assert fused_quant["output_payload_bytes"] == 512
    assert fused_quant["output_scale_bytes"] == 16
    assert fused_quant["output_write_bytes"] == 528
    assert fused_quant["output_kind"] == "FP8+E8M0"

    a4_records = [
        {
            "name": "aiter::a8w4_tdm_fp4_t64x256x256_w1x4_b2_"
            "K7168_e96_act1",
            "device_time_avg": 144.7,
        },
        {
            "name": "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96",
            "device_time_avg": 76.1,
        },
    ]
    a4_timings, a4_kernels, a4_debug = _extract_profiled_gemm_timings(
        trace_df=TraceRecords(a4_records),
        data_format="a4w4",
        activation=ActivationType.Silu,
        experts=96,
        model_dim=7168,
        inter_dim=3072,
    )
    assert a4_timings == {"gemm1_us": 144.7, "gemm2_us": 76.1}
    assert a4_debug is None
    a4_summary = _build_effective_summary(
        profiled_kernels=a4_kernels,
        route_counts=[32] * 96,
        data_format="a4w4",
        use_bias=False,
        experts=96,
        tokens=512,
        topk=6,
        model_dim=7168,
        inter_dim=3072,
    )
    assert tuple(a4_summary) == _GEMM_SUMMARY_COLUMNS
    assert tuple(a4_summary.values()) == (
        "541,165,879,296 FLOP \u2192 3,739.9 TFLOP/s",
        "2,257,747,968 B \u2192 14.191 TB/s",
        "18,874,368 B (BF16) \u2192 0.119 TB/s",
        "2,276,622,336 B \u2192 14.309 TB/s",
        "270,582,939,648 FLOP \u2192 3,555.6 TFLOP/s",
        "1,128,038,400 B \u2192 13.482 TB/s",
        "44,040,192 B (BF16) \u2192 0.526 TB/s",
        "1,172,078,592 B \u2192 14.008 TB/s",
    )

    duplicate_records = a8_records + [
        {
            "name": "a8w4_tdm_fp8_t32x256x256_w1x4_b2_K7168_e384_act1_bias",
            "device_time_avg": 999.0,
        }
    ]
    duplicate_timings, duplicate_kernels, duplicate_debug = (
        _extract_profiled_gemm_timings(
            trace_df=TraceRecords(duplicate_records),
            data_format="a8w4",
            activation=ActivationType.Silu,
            experts=384,
            model_dim=7168,
            inter_dim=768,
        )
    )
    assert duplicate_timings == {"gemm1_us": None, "gemm2_us": 261.143}
    assert "gemm1" not in duplicate_kernels
    assert duplicate_debug == "gemm1 ambiguous (2 matches)"


# ---------------------------------------------------------------------------
# Environment / arch guards
# ---------------------------------------------------------------------------
def _require_gfx1250() -> None:
    # AITER_FORCE_GFX1250=1 forces the grouped path on other archs (e.g. gfx942)
    # to exercise the tiny operators with the GEMM mocked (default; pass
    # --real-gemm to call the real gfx1250 kernel instead).
    if os.environ.get("AITER_FORCE_GFX1250", "0") in ("1", "true", "True", "yes"):
        return
    try:
        from flydsl.runtime.device import get_rocm_arch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FlyDSL not importable: {exc}")
    arch = get_rocm_arch()
    if "gfx1250" not in arch.lower():
        pytest.skip(f"requires gfx1250, got {arch!r}")


def is_gfx1250() -> bool:
    """True only on actual gfx1250 hardware. AITER_FORCE_GFX1250 does NOT count:
    forcing the grouped path onto another arch (e.g. gfx942) still needs the GEMM
    mocked, so real-gemm defaults on only when the real WMMA kernel can run."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        return "gfx1250" in get_rocm_arch().lower()
    except Exception:  # noqa: BLE001
        return False


# Weights/scales use the public shuffle APIs directly:
#   shuffle_weight(b, layout=(16, 16))            -> FP4 TDM B layout (16-row x
#       16-byte chunks) the grouped FlyDSL kernels consume.
#   moe_shuffle_scale(s, experts_cnt=E) -> arch-aware MoE B-scale shuffle; on
#       gfx1250 it folds to the grouped-only n32k4 e8m0 layout (shuffle_scale_n32k4).
# ---------------------------------------------------------------------------
# Reference: aiter's own ``torch_moe_stage1`` + ``torch_moe_stage2``
# (high-precision fp32 baseline that decodes mxfp4/e8m0 internally and
# evaluates the same swiglu+bias formula the grouped path uses). It still
# diverges from the quantised grouped GEMM path by mxfp4/mxfp8 round noise
# (~0.2 rel_l2 on random uint8 weights, ~0.02 on real model weights). The
# point is to catch *catastrophic* regressions, not chase fp32 parity.
# ---------------------------------------------------------------------------
def _torch_moe_ref(
    hidden: torch.Tensor,  # (T, K) bf16
    w1_packed: torch.Tensor,  # (E, 2*I, K_pack) uint8 (GGUU)
    w1_scale_raw: torch.Tensor,  # (E, 2*I, K//32) uint8 (raw e8m0)
    w1_bias: torch.Tensor,  # (E, 2*I) fp32
    w2_packed: torch.Tensor,  # (E, K, I_pack) uint8
    w2_scale_raw: torch.Tensor,  # (E, K, I//32) uint8
    w2_bias: torch.Tensor,  # (E, K) fp32
    topk_w: torch.Tensor,  # (T, topk) bf16
    topk_id: torch.Tensor,  # (T, topk) int32
    *,
    data_format: str,
    activation: ActivationType,
    swiglu_limit: float,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """Two-stage MoE reference reusing ``aiter.fused_moe.torch_moe_stage{1,2}``."""
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")

    def _per_1x32_fp8_dequant(x: torch.Tensor) -> torch.Tensor:
        """Mirror grouped a8w4's per-block-32 MXFP8 input quant, then dequant."""
        block = 32
        dtype_max = 448.0
        x_shape = x.shape
        flat = x.contiguous().view(-1, x_shape[-1]).float()
        blk = flat.view(-1, block)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = blk.abs().amax(dim=1)
        scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(
            max_abs, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
        )
        scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
        scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
        scale_f32[scale_f32 == 0] = 1.0
        q_f32 = (blk / scale_f32.unsqueeze(1)).clamp(min=-dtype_max, max=dtype_max)
        q = q_f32.contiguous().to(dtypes.fp8).to(torch.float32).view_as(blk)
        return (q * scale_f32.unsqueeze(1)).view(x_shape).to(x.dtype)

    w1_scale = w1_scale_raw.view(dtypes.fp8_e8m0)
    w2_scale = w2_scale_raw.view(dtypes.fp8_e8m0)
    if data_format == "a4w4":
        # Match the grouped a4w4 path: stage1 input is MXFP4, not bf16.
        stage1_hidden, stage1_hidden_scale = per_1x32_f4_quant(
            hidden, quant_dtype=dtypes.fp4x2, shuffle=False
        )
    else:
        # Match grouped a8w4: stage1 input is MXFP8 with per-1x32 e8m0 scale.
        stage1_hidden, stage1_hidden_scale = _per_1x32_fp8_dequant(hidden), None
    a2 = torch_moe_stage1(
        stage1_hidden,
        w1_packed,
        w2_packed,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        a1_scale=stage1_hidden_scale,
        w1_scale=w1_scale,
        w1_bias=w1_bias,
        # swiglu_limit clamps gate/up for both SwiGLU and SiLU: the grouped
        # FlyDSL epilogue now applies it in either branch, so the reference
        # passes it through unconditionally to stay in sync.
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if data_format == "a4w4":
        # Match the grouped a4w4 path again: stage2 input is MXFP4.
        T, topk = topk_id.shape
        inter = w2_packed.shape[-1] * 2
        a2_q, a2_scale = per_1x32_f4_quant(
            a2.contiguous().view(T * topk, inter),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        a2 = a2_q.view(T, topk, inter // 2)
    else:
        # Match grouped a8w4 stage2: per-block-32 MXFP8 quant + dequant.
        # This matters for SiLU because the unclamped stage1 output can exceed
        # fp8's unit-scale range; grouped now uses a real e8m0 block scale.
        a2 = _per_1x32_fp8_dequant(a2)
        a2_scale = None
    out = torch_moe_stage2(
        a2,
        w1_packed,
        w2_packed,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=w2_bias,
        doweight=True,
    )
    return out


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------
def _pattern_packed(
    experts: int, rows: int, k_pack: int, *, const_init: float | None = None
) -> torch.Tensor:
    """mxfp4 packed bytes ``(E, rows, k_pack) uint8`` from the global RNG."""
    if const_init is not None:
        return torch.full((experts, rows, k_pack), int(const_init), dtype=torch.uint8)
    return torch.randint(0, 256, (experts, rows, k_pack), dtype=torch.uint8)


def init_weight_scales(
    experts: int, rows: int, n_blocks: int, *, const_init: float | None = None
) -> torch.Tensor:
    """Per-block e8m0 weight scale: random small scales (drawn from the global
    RNG) so the n32k4 B-scale preshuffle layout is actually exercised."""
    if const_init is not None:
        return torch.full((experts, rows, n_blocks), int(const_init), dtype=torch.uint8)
    r = torch.randint(0, 3, (experts, rows, n_blocks), dtype=torch.int16)
    return (r + (DEFAULT_SCALE_BYTE - 1)).to(torch.uint8)


def _make_routing_score(tokens: int, experts: int, topk: int) -> torch.Tensor:
    """Build the ``(tokens, experts)`` gating score honoring the routing env
    controls: ``AITER_MOE_NUM_EXPERT_ACTIVATED=n`` (highest priority) activates
    n randomly-chosen experts (round-robin balanced); ``AITER_MOE_EXPERT_BALANCE``
    round-robins over all experts; otherwise random gating. Shared by the FlyDSL
    (``_make_topk``) and gluon routing paths so both react to the same env."""
    if AITER_MOE_NUM_EXPERT_ACTIVATED > 0:
        n_act = AITER_MOE_NUM_EXPERT_ACTIVATED
        if n_act < topk or n_act > experts or n_act > tokens * topk:
            raise ValueError(
                f"AITER_MOE_NUM_EXPERT_ACTIVATED={n_act} is invalid: must be in "
                f"[topk={topk}, min(experts={experts}, tokens*topk={tokens * topk})]"
            )
        sel = torch.randperm(experts)[:n_act]  # random active expert ids
        score = torch.full((tokens, experts), float("-inf"), dtype=torch.float32)
        slot = torch.arange(tokens * topk) % n_act  # round-robin over active set
        rows = torch.arange(tokens).repeat_interleave(topk)
        score[rows, sel[slot]] = 1.0
    elif AITER_MOE_EXPERT_BALANCE:
        score = torch.zeros((tokens, experts), dtype=torch.float32)
        start_col, end_col = 0, topk
        for token_id in range(tokens):
            score[token_id, start_col:end_col] = 1.0
            start_col = end_col % experts
            end_col = start_col + topk
    else:
        score = torch.randn((tokens, experts), dtype=torch.float32)
    return score


def _make_topk(
    hidden_states: torch.Tensor, experts: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route via ``fused_topk``: normal (random gating) by default; round-robin
    balanced gating when ``AITER_MOE_EXPERT_BALANCE=1`` (mirrors
    op_tests/test_moe_2stage.py). ``AITER_MOE_NUM_EXPERT_ACTIVATED=n`` (highest
    priority) restricts topk to the first n experts. Returns
    ``(topk_ids, topk_weights)`` on the same device as ``hidden_states``."""
    tokens = hidden_states.shape[0]
    score = _make_routing_score(tokens, experts, topk)
    topk_w, topk_id = fused_topk(hidden_states, score, topk, True)
    return topk_id.to(torch.int32), topk_w


def _gguu_to_gugu_rows(t: torch.Tensor) -> torch.Tensor:
    """``(E, 2*I, ...)`` GGUU ``[g0..g_{I-1}, u0..u_{I-1}]`` -> GUGU ``[g0,u0,g1,u1,...]``."""
    _E, two_inter = t.shape[:2]
    inter = two_inter // 2
    g = t[:, :inter]
    u = t[:, inter:]
    return torch.stack([g, u], dim=2).flatten(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Core runner: build inputs, invoke fused_moe, optionally compare to ref
# ---------------------------------------------------------------------------
def _run_grouped_via_fused_moe(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,  # "a4w4" | "a8w4"
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    use_bias: bool = True,
    bench: bool = False,
    kernel_bench: bool = False,
    seed: int = 0,
    warmup: int = 5,
    iters: int = 101,
    const_init: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float | None,
    dict | None,
    object | None,
    list[int],
]:
    """Build mxfp4 weights + routing, dispatch through ``fused_moe``.

    Stage1 weights are always laid out GUGU (gate/up row-interleaved) paired
    with ``GateMode.INTERLEAVE``, which is the only layout the felix TDM grouped
    GEMM reads. The PyTorch reference evaluates the GGUU logical weights, so the
    numerical result is unchanged.

    Correctness is always checked against the reference. ``bench`` selects the
    path that is validated and timed: when set, the output comes from
    ``run_perftest(..., testGraph=False)`` and ``us`` is the profiler-derived
    eager timing; otherwise the output is a single eager call and ``us`` is
    None. ``kernel_bench`` instead times the gemm1/gemm2 kernels in isolation
    (looping each launch alone) and returns their per-kernel us in ``kernel_us``.
    Returns ``(out, ref, us_or_None, kernel_us_or_None, trace_df, route_counts)``.
    """
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")

    pipeline_injection = nullcontext(None)
    pipeline_backend_raw = os.environ.get(
        "AITER_MOE_GEMM1_LAUNCH_BACKEND",
        "",
    )
    if pipeline_backend_raw:
        from my_code.isa_runner import moe_cpp_backend

        pipeline_backend = moe_cpp_backend.selected_pipeline_launch_backend()
        if pipeline_backend == moe_cpp_backend.PIPELINE_LAUNCH_BACKEND_CPP:
            activation_name = {
                ActivationType.Silu: "silu",
                ActivationType.Swiglu: "swiglu",
                ActivationType.Situv2: "situv2",
            }[activation]
            moe_cpp_backend.validate_pipeline_gemm1_case(
                experts=experts,
                tokens=tokens,
                topk=topk,
                model_dim=model_dim,
                inter_dim=inter_dim,
                data_format=data_format,
                activation=activation_name,
                use_bias=use_bias,
                expert_balance=AITER_MOE_EXPERT_BALANCE,
                num_expert_activated=AITER_MOE_NUM_EXPERT_ACTIVATED,
                tile_m_override=os.environ.get("AITER_TDM_TILE_M"),
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            # Build/import happens here, before _call exists and before any
            # run_perftest warmup or profiler iteration.
            pipeline_injection = (
                moe_cpp_backend.prepare_pipeline_gemm1_injection(
                    torch_module=torch,
                )
            )

    K = model_dim
    inter = inter_dim
    K_pack = K // 2
    inter_pack = inter // 2

    # Logical weights/scale/bias: always GGUU (gate rows then up rows).
    # One global seed per case; every draw below uses the global RNG.
    torch.manual_seed(seed)
    w1_logical = _pattern_packed(experts, 2 * inter, K_pack, const_init=const_init)
    w2_logical = _pattern_packed(experts, K, inter_pack, const_init=const_init)
    w1_scale_raw = init_weight_scales(
        experts, 2 * inter, K // SCALE_BLOCK, const_init=const_init
    )
    w2_scale_raw = init_weight_scales(
        experts, K, inter // SCALE_BLOCK, const_init=const_init
    )
    if use_bias:
        if const_init is not None:
            bias1 = torch.full((experts, 2 * inter), float(const_init))
            bias2 = torch.full((experts, K), float(const_init))
        else:
            bias1 = (torch.randn((experts, 2 * inter)) * 1e-3).float()
            bias2 = (torch.randn((experts, K)) * 1e-3).float()
    else:
        bias1 = torch.zeros((experts, 2 * inter))
        bias2 = torch.zeros((experts, K))
    # Activations: bf16; fused_moe handles the dispatched quant internally.
    if const_init is not None:
        hidden = torch.full((tokens, K), float(const_init), dtype=torch.bfloat16)
    else:
        hidden = (torch.randn((tokens, K)) * 0.5).to(torch.bfloat16)

    # Routing: normal (random) by default; balanced if AITER_MOE_EXPERT_BALANCE.
    topk_id, topk_w = _make_topk(hidden, experts, topk)
    topk_w = topk_w.to(torch.bfloat16)

    # ---- prep grouped GEMM inputs ----
    # Stage1 weight/scale/bias are rearranged to the physical GUGU layout;
    # stage2 has no GUGU concept (single N=hidden GEMM).
    bias1_phys = _gguu_to_gugu_rows(bias1)
    gate_mode = GateMode.INTERLEAVE

    # moe_shuffle_weight interleaves gate/up rows internally, so it takes the
    # logical GGUU weight.
    w1_grouped = moe_shuffle_weight(
        w1_logical,
        experts_cnt=experts,
        is_guinterleave=True,
        gate_up=True,
    )
    w2_grouped = moe_shuffle_weight(w2_logical, experts_cnt=experts)
    # GUGU B-scale is built the production way: feed the RAW GGUU scale to
    # moe_shuffle_scale(is_guinterleave=True), which interleaves gate/up rows
    # then folds n32k4 (aiter.ops.shuffle.shuffle_scale_n32k4 end to end) --
    # the weights/bias are row-interleaved above.
    w1_scale = moe_shuffle_scale(
        w1_scale_raw.contiguous(),
        experts_cnt=experts,
        is_guinterleave=True,
        gate_up=True,
    )
    w2_scale = moe_shuffle_scale(w2_scale_raw.contiguous(), experts_cnt=experts)

    if data_format == "a4w4":
        w1_arg = w1_grouped.view(dtypes.fp4x2)
        w2_arg = w2_grouped.view(dtypes.fp4x2)
    else:  # a8w4
        w1_arg = w1_grouped  # uint8 -> grouped helper sets q_dtype_a=fp8
        w2_arg = w2_grouped

    def _call():  # the grouped path is auto-enabled on gfx1250
        return fused_moe(
            hidden,
            w1_arg,
            w2_arg,
            topk_w,
            topk_id,
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1_phys if use_bias else None,
            bias2=bias2 if use_bias else None,
            gate_mode=gate_mode.value,
            dtype=dtypes.bf16,
            swiglu_limit=swiglu_limit,
            beta=situ_beta,
            linear_beta=situ_linear_beta,
        )

    kernel_us = None
    trace_df = None
    cpp_adapter = None
    with pipeline_injection as cpp_adapter:
        torch.cuda.synchronize()
        if kernel_bench:
            # Kernel-bench: time gemm1 and gemm2 in isolation. One eager call
            # populates the per-stage launch callables (and yields a correct ``out`` to
            # verify); then loop each kernel alone. ``us`` (end-to-end) stays None.
            from aiter.ops.flydsl import grouped_moe_gfx1250 as _grouped
            from aiter.test_common import run_perftest

            kernel_bench_callable: list = []
            _grouped.kernel_bench_callable = kernel_bench_callable
            try:
                out = _call()
            finally:
                _grouped.kernel_bench_callable = None
            us = None
            kernel_us = {}
            for _name, callable in kernel_bench_callable:
                _, _us = run_perftest(
                    callable,
                    num_warmup=warmup,
                    num_iters=iters,
                    testGraph=False,
                )
                kernel_us[_name] = _us
        elif bench:
            # Bench: validate + time the eager production path. The returned
            # data is the last profiled output.
            from aiter.test_common import run_perftest

            out, us, trace_df = run_perftest(
                _call,
                num_warmup=warmup,
                num_iters=iters,
                testGraph=False,
                return_trace_df=True,
            )
        else:
            # Verify: validate the eager (graph-off) path; no timing.
            out = _call()
            us = None
    if cpp_adapter is not None:
        cpp_adapter.assert_complete()

    # Reference always uses GGUU logical inputs (layouts are numerically
    # equivalent; only physical packing differs).
    ref = _torch_moe_ref(
        hidden,
        w1_logical,
        w1_scale_raw,
        bias1,
        w2_logical,
        w2_scale_raw,
        bias2,
        topk_w,
        topk_id,
        data_format=data_format,
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    ).to(out.dtype)
    # Read the real routing histogram only after timing; this synchronization is
    # summary post-processing and cannot perturb any measured iteration.
    route_counts = (
        torch.bincount(topk_id.reshape(-1).long(), minlength=experts).cpu().tolist()
    )
    return out, ref, us, kernel_us, trace_df, route_counts


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    base = expected.float().norm().clamp(min=1e-12)
    return float(diff / base)


def _logits_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """MoE accuracy metric from op_tests/test_moe_2stage.py (calc_diff):

        1 - 2*<x,y>/(||x||^2 + ||y||^2)  ==  ||x-y||^2 / (||x||^2 + ||y||^2)

    A magnitude-weighted cosine-style diff. Relation to rel_l2: when the two
    norms match, logits_diff ~= rel_l2**2 / 2.  Production strict gate: < 0.01.
    """
    x = actual.double()
    y = expected.double()
    denom = (x * x + y * y).sum() + 1e-8
    return float(((x - y) ** 2).sum() / denom)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Pytest correctness suite
# ---------------------------------------------------------------------------
def run_moe(
    data_format: str,
    *,
    experts: int = 4,
    tokens: int = 8,
    topk: int = 2,
    model_dim: int = 512,
    inter_dim: int = 512,
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    use_bias: bool = True,
    tol: float = VERIFY_TOL_A4W4,
    raise_on_fail: bool = True,
    bench: bool = False,
    kernel_bench: bool = False,
    warmup: int = 5,
    iters: int = 101,
    const_init: float | None = None,
    check_aot_cache: bool = True,
) -> dict:
    """Compare grouped FlyDSL MoE vs a PyTorch fp32 ref. ``bench`` selects the
    validated path: bench checks and profiles the eager production path with
    ``testGraph=False``; verify checks one eager call.

    Correctness gate: production-consistent logits_diff < LOGITS_DIFF_TOL
    (op_tests/test_moe_2stage.py).  rel_l2 (~= sqrt(2*logits_diff)) is printed
    for reference only.  Returns a metrics dict (with ``us`` when benched).
    """
    _require_gfx1250()
    act = {
        ActivationType.Silu: "silu",
        ActivationType.Swiglu: "swiglu",
        ActivationType.Situv2: "situv2",
    }[activation]
    tag = f"{data_format} {act}"

    # --- grouped FlyDSL vs PyTorch fp32 ref (profiled eager if bench) ---
    run_only = run_only_env() if check_aot_cache else nullcontext()
    with run_only:
        out, ref, us, kernel_us, trace_df, route_counts = _run_grouped_via_fused_moe(
            experts=experts,
            tokens=tokens,
            topk=topk,
            model_dim=model_dim,
            inter_dim=inter_dim,
            data_format=data_format,
            activation=activation,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_bias=use_bias,
            bench=bench,
            kernel_bench=kernel_bench,
            warmup=warmup,
            iters=iters,
            const_init=const_init,
        )
    mode = "kernel" if kernel_bench else ("eager-profile" if bench else "eager")
    ld = _logits_diff(out, ref)
    rel = _rel_l2(out, ref)
    output_sha256 = _tensor_sha256(out)
    print(
        f"[sanity {tag}] {mode}: logits_diff={ld:.4e} rel_l2={rel:.4e} "
        f"(gate<{LOGITS_DIFF_TOL}, ref_norm={float(ref.float().norm()):.4e})",
        flush=True,
    )
    print(f"[sanity {tag}] output_sha256={output_sha256}", flush=True)
    passed = ld < LOGITS_DIFF_TOL
    if raise_on_fail:
        assert (
            passed
        ), f"grouped {tag} {mode} vs ref logits_diff={ld:.4e} > {LOGITS_DIFF_TOL}"
    metrics = {
        "logits_diff": ld,
        "rel_l2": rel,
        "passed": passed,
        "grouped_norm": float(out.float().norm()),
        "ref_norm": float(ref.float().norm()),
        "output_sha256": output_sha256,
    }
    metrics.update(_empty_gemm_summary())

    # --- perf (bench only): timed end-to-end inside _run_grouped_via_fused_moe ---
    if bench:
        print(
            f"[bench {tag}] fused_moe end-to-end us = {us:.2f} (testGraph=False)",
            flush=True,
        )
        metrics["us"] = us
        profiled_kernel_us, profiled_kernels, timing_debug = (
            _extract_profiled_gemm_timings(
                trace_df=trace_df,
                data_format=data_format,
                activation=activation,
                experts=experts,
                model_dim=model_dim,
                inter_dim=inter_dim,
            )
        )
        metrics.update(profiled_kernel_us)
        for stage in ("gemm1", "gemm2"):
            profiled = profiled_kernels.get(stage)
            count = None if profiled is None else profiled.get("count")
            metrics[f"{stage}_count"] = count
            if profiled is not None:
                print(
                    f"[bench kernel timing] {stage}: "
                    f"device_time_avg={profiled['latency_us']:.3f} us "
                    f"count={count} symbol={profiled['name']}",
                    flush=True,
                )
        if timing_debug:
            print(f"[bench kernel timing] {timing_debug}", flush=True)
        gemm_summary = _build_effective_summary(
            profiled_kernels=profiled_kernels,
            route_counts=route_counts,
            data_format=data_format,
            use_bias=use_bias,
            experts=experts,
            tokens=tokens,
            topk=topk,
            model_dim=model_dim,
            inter_dim=inter_dim,
        )
        metrics.update(gemm_summary)
    # --- perf (kernel-bench only): per-kernel gemm1/gemm2 timing (looped alone) ---
    if kernel_bench:
        kernel_us = kernel_us or {}
        g1 = kernel_us.get("gemm1")
        g2 = kernel_us.get("gemm2")
        if g1 is None and g2 is None:
            print(
                f"[kernel-bench {tag}] no grouped kernels captured "
                "(grouped path not taken?)",
                flush=True,
            )
        else:
            g1s = "n/a" if g1 is None else f"{g1:.2f}"
            g2s = "n/a" if g2 is None else f"{g2:.2f}"
            print(
                f"[kernel-bench {tag}] gemm1 us = {g1s} gemm2 us = {g2s}",
                flush=True,
            )
        # If both flags are ever passed programmatically, isolated kernel-bench
        # timings intentionally take precedence over end-to-end trace averages.
        metrics["gemm1_us"] = g1
        metrics["gemm2_us"] = g2
    return metrics


# model_dim=512 (not the 256 default): the grouped kernel needs
# num_k_tiles = (K // split_k) // tile_k >= 2, i.e. K >= 2*tile_k = 512.
def test_grouped_a4w4_silu_matches_torch_ref():
    run_moe(
        "a4w4",
        activation=ActivationType.Silu,
        model_dim=512,
        inter_dim=512,
    )


def test_grouped_a4w4_swiglu_matches_torch_ref():
    run_moe(
        "a4w4",
        activation=ActivationType.Swiglu,
        model_dim=512,
        inter_dim=512,
    )


def test_situv2_activation_matches_torch():
    torch.manual_seed(0)
    gate = torch.randn(4, 32)
    up = torch.randn(4, 32)
    beta, linear_beta = 4.0, 25.0
    expected = (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * linear_beta
        * torch.tanh(up / linear_beta)
    )
    actual = apply_gate_up(
        gate,
        up,
        "situv2",
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )
    torch.testing.assert_close(actual, expected)


def test_grouped_a4w4_situv2_matches_torch_ref():
    run_moe(
        "a4w4",
        activation=ActivationType.Situv2,
        model_dim=512,
        inter_dim=512,
    )


def test_grouped_a8w4_situv2_matches_torch_ref():
    # a8w4 takes the fused stage1 quant epilogue (batched activation), which is
    # a separate code path from a4w4's bf16 intermediate (element-wise).
    run_moe(
        "a8w4",
        activation=ActivationType.Situv2,
        model_dim=512,
        inter_dim=512,
        tol=VERIFY_TOL_A8W4,
    )


@pytest.mark.parametrize("activation", [ActivationType.Silu, ActivationType.Swiglu])
def test_grouped_a4w4_swiglu_limit_clamps(activation):
    run_moe("a4w4", activation=activation, swiglu_limit=1.0)


# ---------------------------------------------------------------------------
# Contiguous-M prefix scan
#
# The scan sits in front of every grouped MoE launch: it turns the per-expert
# row counts into the tile-aligned starts/psum the GEMM schedules on, and
# rewrites the route rows in place. It is also the one piece whose width is set
# by the expert count rather than the token count, so it gets its own coverage
# above and below the block size -- a wrong row here does not produce a bad
# number, it produces an out-of-bounds write in whichever kernel consumes the
# row next.
# ---------------------------------------------------------------------------
def _psum_ref(masked_m: torch.Tensor, tile_m: int):
    """starts / psum / contiguous_m from a tile-aligned cumulative sum."""
    aligned = ((masked_m + tile_m - 1) // tile_m) * tile_m
    inclusive = torch.cumsum(aligned.to(torch.int64), 0)
    starts = inclusive - aligned
    return (
        starts.to(torch.int32),
        (starts + masked_m).to(torch.int32),
        max(int(inclusive[-1]), tile_m),
    )


def _random_route_counts(experts: int, topk: int, tokens: int, seed: int = 0):
    """Per-expert counts from a real (unbalanced) random routing."""
    torch.manual_seed(seed)
    topk = min(topk, experts)
    topk_ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(tokens)]).to(
        torch.int32
    )
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=experts)
    return topk_ids, counts.to(torch.int32)


# 512 is MAX_EXPERTS_PER_BLOCK: one thread per expert covers E up to that in a
# single pass, and everything above it needs the chunked sweep. Kimi-K3 is 896.
@pytest.mark.parametrize("experts", [8, 256, 512, 513, 896, 1024])
def test_contiguous_psum_matches_cumsum(experts):
    _require_gfx1250()
    from aiter.ops.flydsl.grouped_moe_gfx1250 import contiguous_psum

    tile_m = 64
    _topk_ids, masked_m = _random_route_counts(experts, topk=16, tokens=128)
    ref_starts, ref_psum, ref_total = _psum_ref(masked_m, tile_m)

    starts, psum, contiguous_m = contiguous_psum(masked_m, experts, tile_m)
    torch.cuda.synchronize()

    bad = int((starts != ref_starts).sum())
    assert bad == 0, (
        f"E={experts}: {bad} experts have a wrong start, first at "
        f"{int((starts != ref_starts).nonzero()[0][0])}"
    )
    assert torch.equal(psum, ref_psum), f"E={experts}: psum mismatch"
    assert (
        int(contiguous_m[0]) == ref_total
    ), f"E={experts}: contiguous_m {int(contiguous_m[0])} != {ref_total}"


@pytest.mark.parametrize("experts", [8, 256, 512, 513, 896, 1024])
def test_contiguous_psum_remap_rows_stay_in_bounds(experts):
    """The remap is what the MoE actually calls; an unscanned expert lands here
    as a row index pointing outside the contiguous buffer."""
    _require_gfx1250()
    from aiter.ops.flydsl.grouped_moe_gfx1250 import contiguous_psum_remap

    tile_m, topk, tokens = 64, 16, 128
    topk_ids, masked_m = _random_route_counts(experts, topk, tokens)
    ref_starts, _ref_psum, ref_total = _psum_ref(masked_m, tile_m)

    # Masked layout: row = expert * max_m + slot, which is what
    # flydsl_moe_topids_to_rows produces and the remap folds down.
    flat = topk_ids.reshape(-1)
    max_m = max(tile_m, ((flat.numel() + tile_m - 1) // tile_m) * tile_m)
    slot = torch.zeros(experts, dtype=torch.int64)
    rows = torch.empty(flat.numel(), dtype=torch.int32)
    for i, e in enumerate(flat.tolist()):
        rows[i] = e * max_m + int(slot[e])
        slot[e] += 1

    remapped = rows.clone()
    contiguous_psum_remap(masked_m, remapped, experts, max_m, tile_m)
    torch.cuda.synchronize()

    expected = ref_starts[flat.long()].long() + (rows.long() - flat.long() * max_m)
    assert torch.equal(remapped.long(), expected), f"E={experts}: row remap mismatch"
    oob = int((remapped >= ref_total).sum())
    assert oob == 0, (
        f"E={experts}: {oob} remapped rows land outside the contiguous buffer "
        f"(bound {ref_total}, max row {int(remapped.max())})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _mock_grouped_gemm() -> None:
    """Run the grouped MoE path without the gfx1250-only kernels.

    Two patches let the tiny operators (route maps, scatter/gather, quant,
    scale preshuffle, m-tile map, gather-reduce) run on any arch (e.g. gfx942
    via AITER_FORCE_GFX1250=1):

    1. Replace the felix TDM grouped GEMM with a no-op -- the GEMM executes
       nothing; stage outputs are left as-is.
    2. Route the fp4 a1/a2 quant through the Triton implementation, since the
       HIP ``per_1x32_f4_quant_hip`` has no fp4x2 output support off gfx1250.

    The library imports all these names at call time, so patching the source
    modules is enough -- no library edits required.
    """
    import aiter.ops.flydsl.grouped_gemm_mxfp4 as grouped_gemm
    import aiter.ops.quant as q

    def _noop_gemm(*_a, **_k):
        return None

    grouped_gemm.flydsl_grouped_gemm_a8w4_masked = _noop_gemm

    q.per_1x32_f4_quant_hip = q.per_1x32_f4_quant_triton


def summarize(rows: list):
    """Build a precision summary table from per-case metrics and print it.

    Mirrors the pandas DataFrame reporting in op_tests/test_moe_2stage.py.
    Returns the DataFrame (or the raw rows if pandas is unavailable).
    """
    if not rows:
        return None
    try:
        import pandas as pd
    except ImportError:
        print("[precision summary] pandas not installed; raw rows:", flush=True)
        for r in rows:
            print(f"  {r}", flush=True)
        return rows
    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        # to_markdown needs the optional `tabulate` package; plain fallback.
        table = df.to_string(index=False)
    print("\n[precision summary]\n" + table, flush=True)
    return df


def set_data_format(data_format: str) -> None:
    """Select the grouped GEMM data format.

    a8w4 needs ``AITER_FORCE_A8W4=1`` so ``fused_moe`` routes the a8w4 path
    (see fused_moe.py); a4w4 needs it unset so the fp4x2 activation path is
    taken. Toggled per-row so a mixed-format CSV sweep routes each case
    correctly.
    """
    if data_format == "a8w4":
        os.environ["AITER_FORCE_A8W4"] = "1"
    else:
        os.environ.pop("AITER_FORCE_A8W4", None)
    logger.info("grouped GEMM data format: %s", data_format)


# Default tuned-config CSV: <repo>/aiter/configs/tuned_grouped_fmoe.csv. Every
# row is one grouped-MoE setting; the --scenario csv sweep runs them all.
DEFAULT_CSV_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "aiter",
        "configs",
        "tuned_grouped_fmoe.csv",
    )
)


def _csv_data_format(q_dtype_a: str) -> str:
    """Map the CSV ``q_dtype_a`` column to this test's data_format tag.

    fp8 activations -> a8w4 (MXFP8 x MXFP4); fp4 activations -> a4w4
    (MXFP4 x MXFP4). Weights are MXFP4 either way.
    """
    a = q_dtype_a.strip()
    if a in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz"):
        return "a8w4"
    if a in ("torch.float4_e2m1fn_x2",):
        return "a4w4"
    raise ValueError(f"unsupported q_dtype_a in CSV: {q_dtype_a!r}")


def _csv_activation(act_type: str) -> ActivationType:
    a = act_type.strip()
    if a == "ActivationType.Swiglu":
        return ActivationType.Swiglu
    if a == "ActivationType.Silu":
        return ActivationType.Silu
    raise ValueError(f"unsupported act_type in CSV: {act_type!r}")


def run_csv_scenario(args) -> None:
    """Sweep every setting in a tuned_grouped_fmoe-style CSV.

    Each CSV row (token, model_dim, inter_dim, expert, topk, act_type,
    q_dtype_a) becomes one ``run_moe`` case. The GEMM
    tuned config is looked up from the same CSV by the kernel via the problem
    shape, so simply running each shape exercises its tuned setting.

    Each row is benched with ``run_perftest(..., testGraph=False)`` and its
    correctness checked; one out-of-gate row is recorded rather than aborting
    the sweep.
    """
    csv_path = args.csv_path or DEFAULT_CSV_PATH
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    print(f"[csv] sweeping settings from {csv_path}", flush=True)

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        csv_rows = list(reader)
    if not csv_rows:
        raise SystemExit(f"CSV has no data rows: {csv_path}")

    activation_override = None
    if args.act is not None:
        activation_override = _ACT_BY_NAME[args.act]

    rows = []
    for idx, rec in enumerate(csv_rows):
        try:
            tokens = int(rec["token"])
            model_dim = int(rec["model_dim"])
            inter_dim = int(rec["inter_dim"])
            experts = int(rec["expert"])
            topk = int(rec["topk"])
            data_format = _csv_data_format(rec["q_dtype_a"])
            activation = activation_override or _csv_activation(rec["act_type"])
        except (KeyError, ValueError) as exc:
            print(f"[csv] row {idx}: skipped ({exc})", flush=True)
            continue

        # topk == -1 marks an EP row; run_moe is single-rank and has no EP setup.
        if topk == -1:
            print(f"[csv] row {idx}: skipped topk=-1 (EP row)", flush=True)
            continue

        # The grouped kernels need K/inter >= 512 (tile_k=256 -> two K tiles).
        if model_dim < 512 or inter_dim < 512:
            print(
                f"[csv] row {idx}: skipped model_dim={model_dim} "
                f"inter_dim={inter_dim} (< 512 grouped-kernel floor)",
                flush=True,
            )
            continue

        act = "swiglu" if activation == ActivationType.Swiglu else "silu"
        print(
            f"\n===== csv row {idx}: {data_format} {act} "
            f"tokens={tokens} model_dim={model_dim} inter_dim={inter_dim} "
            f"experts={experts} topk={topk} =====",
            flush=True,
        )
        # Route each row to its format (a8w4 needs AITER_FORCE_A8W4=1).
        set_data_format(data_format)
        tol = VERIFY_TOL_A8W4 if data_format == "a8w4" else VERIFY_TOL_A4W4
        try:
            metrics = run_moe(
                data_format,
                experts=experts,
                tokens=tokens,
                topk=topk,
                model_dim=model_dim,
                inter_dim=inter_dim,
                tol=tol,
                activation=activation,
                swiglu_limit=args.swiglu_limit,
                use_bias=not args.no_bias,
                check_aot_cache=not args.no_check_aot_cache,
                raise_on_fail=False,
                bench=True,
                kernel_bench=False,
                warmup=args.warmup,
                iters=args.iters,
                const_init=args.const_init,
            )
        except Exception as exc:  # noqa: BLE001 - record, keep sweeping
            print(f"[csv] row {idx}: ERROR {exc!r}", flush=True)
            rows.append(
                {
                    "row": idx,
                    "data_format": data_format,
                    "act": act,
                    "tokens": tokens,
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "experts": experts,
                    "topk": topk,
                    "logits_diff": float("nan"),
                    "rel_l2": float("nan"),
                    "pass": False,
                    "error": repr(exc),
                    "us": None,
                    "gemm1_us": None,
                    "gemm2_us": None,
                    **_empty_gemm_summary(),
                }
            )
            continue

        rows.append(
            {
                "row": idx,
                "data_format": data_format,
                "act": act,
                "tokens": tokens,
                "model_dim": model_dim,
                "inter_dim": inter_dim,
                "experts": experts,
                "topk": topk,
                "logits_diff": metrics["logits_diff"],
                "rel_l2": metrics["rel_l2"],
                "pass": metrics["passed"],
                "error": None,
                "us": metrics.get("us"),
                "gemm1_us": metrics.get("gemm1_us"),
                "gemm2_us": metrics.get("gemm2_us"),
                **{column: metrics[column] for column in _GEMM_SUMMARY_COLUMNS},
            }
        )

    summarize(rows)
    failed = [r for r in rows if not r["pass"]]
    if failed:
        details = "; ".join(
            f"row={r['row']} {r['data_format']} {r['act']} "
            f"tokens={r['tokens']} "
            + (
                f"error={r['error']}"
                if r.get("error")
                else f"logits_diff={r['logits_diff']:.4e} rel_l2={r['rel_l2']:.4e}"
            )
            for r in failed
        )
        assert not failed, (
            f"{len(failed)}/{len(rows)} CSV case(s) failed "
            f"(gate {LOGITS_DIFF_TOL}): {details}"
        )


def main() -> None:
    if not is_gfx1250():
        print("skipping: requires gfx1250")
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("bench", "verify", "kernel", "csv"),
        default="bench",
        help="bench: profile fused_moe end-to-end with testGraph=False. verify: eager "
        "correctness only. kernel: time the gemm1 and gemm2 kernels in "
        "isolation (loop each launch alone). csv: sweep every setting in "
        "--csv-path (one run_moe case per row).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV of grouped-MoE settings to sweep with --scenario csv "
        f"(default: {DEFAULT_CSV_PATH}). Each row is one case.",
    )
    parser.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[64],
        metavar="N",
        help="one or more space-separated token counts; the scenario runs "
        "once per value, e.g. --tokens 64 128 256",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument(
        "--act",
        choices=("silu", "swiglu", "situv2"),
        default=None,
        help="stage1 activation: silu => silu(gate)*up; "
        "swiglu => gpt-oss swiglu with clamp/alpha/residual; "
        "situv2 => Kimi-K3 SiTUv2 (see --situ-beta / --situ-linear-beta). "
        "Default: swiglu "
        "for bench/verify/kernel; for --scenario csv, unset means use each "
        "row's act_type (pass --act to force one activation for all rows).",
    )
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument(
        "--situ-beta",
        type=float,
        default=4.0,
        help="SiTUv2 gate beta (Kimi-K3 activation_situ_beta).",
    )
    parser.add_argument(
        "--situ-linear-beta",
        type=float,
        default=25.0,
        help="SiTUv2 up beta (Kimi-K3 activation_situ_linear_beta).",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="run with zero stage1/stage2 bias tensors",
    )
    parser.add_argument(
        "--const-init",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        metavar="VALUE",
        help="initialize activations (A), weights (B), weight scales (Bs), and "
        "bias to the constant VALUE instead of random values. Bare --const-init "
        "uses 0.0 (zero-init). uint8 tensors (weights, scales) are filled with "
        "int(VALUE).",
    )
    parser.add_argument(
        "--real-gemm",
        action="store_true",
        default=is_gfx1250(),
        help="call the real grouped WMMA GEMM kernel. Default: True on gfx1250, "
        "False elsewhere (mock the GEMM so the tiny operators run on any arch).",
    )
    parser.add_argument(
        "--no-check-aot-cache",
        action="store_true",
        help="disable the default AOT cache-miss check. By default the test "
        "runs in FlyDSL run-only mode (FLYDSL_RUNTIME_RUN_ONLY=1): kernels "
        "load the AOT-precompiled artifact and never JIT-compile, so a cache "
        "miss raises. Pass this flag to allow runtime JIT compilation.",
    )
    args = parser.parse_args()
    if not args.real_gemm:
        _mock_grouped_gemm()

    # CSV sweep: settings come from the CSV, not the shape flags. Each row sets
    # its own data format / shape, so skip the single-shape guards below.
    if args.scenario == "csv":
        run_csv_scenario(args)
        return

    # The >=512 floor is a FlyDSL grouped-kernel constraint (tile_k=256 needs two
    # K tiles).
    if args.model_dim < 512 or args.inter_dim < 512:
        raise SystemExit(
            f"model_dim ({args.model_dim}) and inter_dim ({args.inter_dim}) must be "
            "at least 512 for the grouped GEMM kernels (tile_k=256 requires at "
            "least two K tiles)."
        )

    set_data_format(args.data_format)

    # --tokens accepts one or more counts; run once per value. Each iteration
    # sets args.tokens to a single int so run_moe reads it unchanged.
    token_list = args.tokens if isinstance(args.tokens, list) else [args.tokens]
    # None (unset) defaults to swiglu for the single-shape scenarios.
    activation = _ACT_BY_NAME.get(args.act, ActivationType.Swiglu)
    rows = []
    for _tok in token_list:
        args.tokens = _tok
        if len(token_list) > 1:
            print(f"\n===== tokens={_tok} =====", flush=True)

        tol = VERIFY_TOL_A8W4 if args.data_format == "a8w4" else VERIFY_TOL_A4W4
        # raise_on_fail=False so one out-of-gate token does not abort the
        # sweep; the failure is recorded and reported after the table.
        metrics = run_moe(
            args.data_format,
            experts=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tol=tol,
            activation=activation,
            swiglu_limit=args.swiglu_limit,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            use_bias=not args.no_bias,
            check_aot_cache=not args.no_check_aot_cache,
            raise_on_fail=False,
            bench=args.scenario == "bench",
            kernel_bench=args.scenario == "kernel",
            warmup=args.warmup,
            iters=args.iters,
            const_init=args.const_init,
        )
        rows.append(
            {
                "data_format": args.data_format,
                "act": args.act,
                "init": "random" if args.const_init is None else "const",
                "experts": args.experts,
                "tokens": _tok,
                "topk": args.topk,
                "model_dim": args.model_dim,
                "inter_dim": args.inter_dim,
                "logits_diff": metrics["logits_diff"],
                "rel_l2": metrics["rel_l2"],
                "pass": metrics["passed"],
                "us": metrics.get("us"),
                "gemm1_us": metrics.get("gemm1_us"),
                "gemm2_us": metrics.get("gemm2_us"),
                **{column: metrics[column] for column in _GEMM_SUMMARY_COLUMNS},
            }
        )

    # Always print the summary table (verify and bench).
    summarize(rows)
    # Preserve CI semantics: non-zero exit if any case missed the accuracy gate.
    failed = [r for r in rows if not r["pass"]]
    if failed:
        details = "; ".join(
            f"tokens={r['tokens']} act={r['act']} "
            f"logits_diff={r['logits_diff']:.4e} rel_l2={r['rel_l2']:.4e}"
            for r in failed
        )
        assert not failed, (
            f"{len(failed)}/{len(rows)} case(s) exceeded logits_diff "
            f"gate {LOGITS_DIFF_TOL}: {details}"
        )


if __name__ == "__main__":
    main()
