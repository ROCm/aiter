#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unified benchmark: GDN K5 and K5+K6 forward kernels.

Two subcommands share a common set of preset shapes, input builders, and
timing infrastructure:

  k5    — A/B benchmark of the inter-chunk state-scan (K5) kernel in isolation.
           Compares: FlyDSL (variant-parametrized), Triton, HIP.

  k5k6  — A/B benchmark of the full K5+K6 pipeline (state scan + output).
           Compares: FlyDSL fused K5+K6, FlyDSL K5 + Triton K6, Triton K5 +
           Triton K6, HIP K5 + Triton K6, and the auto-dispatch combined
           wrapper (K5K6_combined) that applies the fused-vs-separate heuristic.

Usage
-----
    # List preset shapes:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5 --list
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5k6 --list

    # K5 bench: shape 1, graph mode, verify against reference:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5 \\
        --shape-index 1 --mode graph --verification reference

    # K5+K6 bench: all impls including the combined auto-dispatch wrapper:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5k6 \\
        --shape-index 1 --mode graph --impl all

    # Auto-dispatch combined wrapper only, shapes 1-5:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5k6 \\
        --impl K5K6_combined --shape-range 1-5 --mode graph

    # KDA shapes, FlyDSL all variants vs Triton:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py k5 \\
        --gate gk --flydsl-variants all --baseline triton
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import random
import sys
import warnings
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Repo root on sys.path for direct script execution
# --------------------------------------------------------------------------- #
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BENCH_DIR = str(Path(__file__).parent)
sys.path.insert(0, _BENCH_DIR)
from utils._bench_timing import EmptyGraphCaptureError, MeasureConfig
from utils._bench_timing import measure as _time_measure
from utils.bench_common import (
    add_output_args,
    add_timing_args,
    add_verification_args,
    collect_env_info,
    make_measure_config,
    print_result_table,
    write_bench_markdown,
)
from utils.plot_perf import (
    category_label,
    make_bar_chart,
    make_fill_scatter,
    make_summary_md,
    parse_bench_md,
)

# --------------------------------------------------------------------------- #
# Preset shapes  (shared by both subcommands)
# --------------------------------------------------------------------------- #
# Shape tuple: (model_tag, H, Hg, T_flat, N, K, V, BT, gate, seq_pattern)
#
# ``seq_pattern`` controls how the T_flat tokens are split across the N
# sequences. Every N>1 shape already runs the varlen code path (cu_seqlens is
# built for it), but with "equal" lengths -- which hides the load-balance
# question entirely, because each CTA then does an identical number of chunks.
# Under 1 CTA/CU the makespan is set by the longest sequence.
PRESET_SHAPES: list[tuple] = [
    # KDA Kimi-K3 TP8  (H=12 is the primary serving shape from the ticket)
    ("kda_tp8", 12, 12, 8192, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 32768, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 8192, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 32768, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 8192, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 32768, 8, 128, 128, 64, "gk", "equal"),
    # KDA Kimi-K3 TP4
    ("kda_tp4", 24, 24, 8192, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp4", 24, 24, 32768, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp4", 24, 24, 8192, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp4", 24, 24, 32768, 8, 128, 128, 64, "gk", "equal"),
    # GDN Qwen3-Next TP8
    ("gdn_q3n_tp8", 4, 2, 8192, 8, 128, 128, 64, "g", "equal"),
    ("gdn_q3n_tp8", 4, 2, 32768, 8, 128, 128, 64, "g", "equal"),
    # GDN Qwen3-Next TP4
    ("gdn_q3n_tp4", 8, 4, 8192, 4, 128, 128, 64, "g", "equal"),
    ("gdn_q3n_tp4", 8, 4, 32768, 4, 128, 128, 64, "g", "equal"),
    # GDN Qwen3.5-MoE TP1
    ("gdn_q35_tp1", 16, 16, 8192, 1, 128, 128, 64, "g", "equal"),
    ("gdn_q35_tp1", 32, 8, 8192, 1, 128, 128, 64, "g", "equal"),
    ("gdn_q35_tp1", 32, 8, 32768, 1, 128, 128, 64, "g", "equal"),
    # -- varlen / ragged batches -------------------------------------------
    # Same (H, Hg, T_flat, N) as shapes 10 and 12, so the only difference from
    # their "equal" counterparts is the sequence-length distribution; any delta
    # is attributable to load imbalance rather than to shape.
    ("kda_tp4", 24, 24, 32768, 8, 128, 128, 64, "gk", "ragged"),
    ("kda_tp4", 24, 24, 32768, 8, 128, 128, 64, "gk", "bimodal"),
    ("kda_tp4", 24, 24, 32768, 8, 128, 128, 64, "gk", "skew"),
    ("gdn_q3n_tp8", 4, 2, 32768, 8, 128, 128, 64, "g", "ragged"),
    ("gdn_q3n_tp8", 4, 2, 32768, 8, 128, 128, 64, "g", "skew"),
    # Control for dispatch order: identical length multiset to shape 20, but the
    # long sequence is last. Only affects shapes whose grid exceeds the CU count.
    ("kda_tp4", 24, 24, 32768, 8, 128, 128, 64, "gk", "skew_last"),
    # -- chiplet-remap partial-grid stress cases ---------------------------
    # grid_nh = N*H is NOT a multiple of nXCD=8, so the XCD remap co-locates
    # each head's V-tiles only for the full nXCD*GRID_V cycles; the tail passes
    # through as the round-robin identity. These verify the partial case still
    # benefits (majority of heads co-located) and never regresses (tail == base).
    # H=4 (real Qwen3-Next TP8 head count); N chosen to break divisibility.
    (
        "gdn_q3n_rmp",
        4,
        2,
        8192,
        1,
        128,
        128,
        64,
        "g",
        "equal",
    ),  # grid_nh=4  (<nXCD: identity)
    (
        "gdn_q3n_rmp",
        4,
        2,
        8192,
        3,
        128,
        128,
        64,
        "g",
        "equal",
    ),  # grid_nh=12 (67% clean)
    (
        "gdn_q3n_rmp",
        4,
        2,
        8192,
        5,
        128,
        128,
        64,
        "g",
        "equal",
    ),  # grid_nh=20 (80% clean)
    (
        "gdn_q3n_rmp",
        4,
        2,
        8192,
        7,
        128,
        128,
        64,
        "g",
        "equal",
    ),  # grid_nh=28 (86% clean)
    # -- variant-autoselect gap-fill sweep ---------------------------------
    # Signatures the tuned table (gate,H,N-bucket,is_varlen) did not yet cover.
    # skew / skew_last share a SELECTION signature with equal (seq_pattern is
    # invisible to the selector); they are here only to pick the robust variant
    # for the varlen KDA groups. Each new signature runs at T=8192 AND 32768 to
    # confirm the table's T-invariance holds for these (previously untested) groups.
    # Tier 1: interior N-bucket fills for gate+H already in the table.
    ("kda_tp4_g", 24, 24, 8192, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp4_g", 24, 24, 32768, 4, 128, 128, 64, "gk", "equal"),
    ("gdn_h8_g", 8, 4, 8192, 1, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 8192, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 8192, 8, 128, 128, 64, "g", "skew"),
    ("gdn_h8_g", 8, 4, 32768, 1, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 32768, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 32768, 8, 128, 128, 64, "g", "skew"),
    ("gdn_h16_g", 16, 16, 8192, 4, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 8192, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 8192, 8, 128, 128, 64, "g", "skew"),
    ("gdn_h16_g", 16, 16, 32768, 4, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 8, 128, 128, 64, "g", "skew"),
    ("gdn_h32_g", 32, 8, 8192, 4, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 8, 128, 128, 64, "g", "skew"),
    ("gdn_h32_g", 32, 8, 32768, 4, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 8, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 8, 128, 128, 64, "g", "skew"),
    # Tier 2: KDA TP2 (H48) and TP1 (H96), all N-buckets, both T.
    ("kda_tp2", 48, 48, 8192, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 8192, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 8192, 4, 128, 128, 64, "gk", "skew"),
    ("kda_tp2", 48, 48, 8192, 4, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp2", 48, 48, 8192, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 8192, 8, 128, 128, 64, "gk", "skew"),
    ("kda_tp2", 48, 48, 8192, 8, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp2", 48, 48, 32768, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 32768, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 32768, 4, 128, 128, 64, "gk", "skew"),
    ("kda_tp2", 48, 48, 32768, 4, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp2", 48, 48, 32768, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 32768, 8, 128, 128, 64, "gk", "skew"),
    ("kda_tp2", 48, 48, 32768, 8, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp1", 96, 96, 8192, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 8192, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 8192, 4, 128, 128, 64, "gk", "skew"),
    ("kda_tp1", 96, 96, 8192, 4, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp1", 96, 96, 8192, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 8192, 8, 128, 128, 64, "gk", "skew"),
    ("kda_tp1", 96, 96, 8192, 8, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp1", 96, 96, 32768, 1, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 32768, 4, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 32768, 4, 128, 128, 64, "gk", "skew"),
    ("kda_tp1", 96, 96, 32768, 4, 128, 128, 64, "gk", "skew_last"),
    ("kda_tp1", 96, 96, 32768, 8, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 32768, 8, 128, 128, 64, "gk", "skew"),
    ("kda_tp1", 96, 96, 32768, 8, 128, 128, 64, "gk", "skew_last"),
    # -- N-bucket disambiguation (equal only) ------------------------------
    # The auto-selector buckets N as {1, <=4, >=5}, and the >=5 bucket for these
    # (gate,H) groups is currently anchored only at N=8. The scalar-g equal optimum
    # tracks the H*N grid-size product (H*N~32->bv16, ~64->bv32, >=128->bv64w8), so
    # the >=5 bucket may not be flat. These probe N inside the bucket (6,12,16) and
    # the <=4 / >=5 boundary (N=2) to confirm the bucketing or justify a finer split.
    # Skew omitted -- undetectable at selection time, so it can't inform the key.
    ("gdn_h16_g", 16, 16, 8192, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 8192, 6, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 8192, 12, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 8192, 16, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 6, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 12, 128, 128, 64, "g", "equal"),
    ("gdn_h16_g", 16, 16, 32768, 16, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 6, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 12, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 8192, 16, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 6, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 12, 128, 128, 64, "g", "equal"),
    ("gdn_h32_g", 32, 8, 32768, 16, 128, 128, 64, "g", "equal"),
    ("kda_tp4_g", 24, 24, 8192, 6, 128, 128, 64, "gk", "equal"),
    ("kda_tp4_g", 24, 24, 8192, 16, 128, 128, 64, "gk", "equal"),
    ("kda_tp4_g", 24, 24, 32768, 6, 128, 128, 64, "gk", "equal"),
    ("kda_tp4_g", 24, 24, 32768, 16, 128, 128, 64, "gk", "equal"),
    # -- N=2 boundary coverage (equal only) --------------------------------
    # The <=4 N-bucket is not flat: N=2 halves the grid and drops the optimal tile
    # one regime below N=4 (measured: g H16 N2->bv16 vs N4->bv32; g H32 N2->bv32 vs
    # N4->bv64w8). N=2 was previously measured only for g H16/H32; these add it for
    # the remaining (gate,H) groups so a <=2 sub-bucket can be populated with data
    # rather than extrapolation.
    ("gdn_q3n_tp8", 4, 2, 8192, 2, 128, 128, 64, "g", "equal"),
    ("gdn_q3n_tp8", 4, 2, 32768, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 8192, 2, 128, 128, 64, "g", "equal"),
    ("gdn_h8_g", 8, 4, 32768, 2, 128, 128, 64, "g", "equal"),
    ("kda_tp8", 12, 12, 8192, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp8", 12, 12, 32768, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp4", 24, 24, 8192, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp4", 24, 24, 32768, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 8192, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp2", 48, 48, 32768, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 8192, 2, 128, 128, 64, "gk", "equal"),
    ("kda_tp1", 96, 96, 32768, 2, 128, 128, 64, "gk", "equal"),
    # -- ragged/bimodal/skew/skew_last sampling for the fusion-selection
    # heuristic (added for the final heuristics sweep). Existing non-equal
    # coverage was concentrated at high fill and mostly on KDA; these add the
    # four realistic length distributions across the fill64 = 2*N*H/CU axis,
    # weighted around the ~0.5 fusion decision boundary, with low- and high-fill
    # tails. Selection is blind to the pattern (only cu_seqlens is visible), so
    # these quantify how much the input distribution shifts fused-vs-unfused at a
    # given nominal fill -- the second-order term the fill axis cannot capture.
    # ---- near the ~0.5 boundary (fill64 0.39-0.63): the critical band ----
    ("gdn_h16_g", 16, 16, 32768, 4, 128, 128, 64, "g", "ragged"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 32768, 4, 128, 128, 64, "g", "bimodal"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 32768, 4, 128, 128, 64, "g", "skew"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 32768, 4, 128, 128, 64, "g", "skew_last"),  # fill64=0.42
    ("gdn_h32_g", 32, 8, 32768, 2, 128, 128, 64, "g", "ragged"),  # fill64=0.42
    ("gdn_h32_g", 32, 8, 32768, 2, 128, 128, 64, "g", "bimodal"),  # fill64=0.42
    ("gdn_h32_g", 32, 8, 32768, 2, 128, 128, 64, "g", "skew"),  # fill64=0.42
    ("gdn_h32_g", 32, 8, 32768, 2, 128, 128, 64, "g", "skew_last"),  # fill64=0.42
    ("kda_tp8", 12, 12, 32768, 5, 128, 128, 64, "gk", "ragged"),  # fill64=0.39
    ("kda_tp8", 12, 12, 32768, 5, 128, 128, 64, "gk", "bimodal"),  # fill64=0.39
    ("kda_tp8", 12, 12, 32768, 5, 128, 128, 64, "gk", "skew"),  # fill64=0.39
    ("kda_tp8", 12, 12, 32768, 5, 128, 128, 64, "gk", "skew_last"),  # fill64=0.39
    ("kda_tp8", 12, 12, 32768, 6, 128, 128, 64, "gk", "ragged"),  # fill64=0.47
    ("kda_tp8", 12, 12, 32768, 6, 128, 128, 64, "gk", "bimodal"),  # fill64=0.47
    ("kda_tp8", 12, 12, 32768, 6, 128, 128, 64, "gk", "skew"),  # fill64=0.47
    ("kda_tp8", 12, 12, 32768, 6, 128, 128, 64, "gk", "skew_last"),  # fill64=0.47
    ("kda_tp4", 24, 24, 32768, 4, 128, 128, 64, "gk", "ragged"),  # fill64=0.63
    ("kda_tp4", 24, 24, 32768, 4, 128, 128, 64, "gk", "bimodal"),  # fill64=0.63
    ("kda_tp4", 24, 24, 32768, 4, 128, 128, 64, "gk", "skew"),  # fill64=0.63
    ("kda_tp4", 24, 24, 32768, 4, 128, 128, 64, "gk", "skew_last"),  # fill64=0.63
    ("gdn_h16_g", 16, 16, 32768, 6, 128, 128, 64, "g", "ragged"),  # fill64=0.63
    ("gdn_h16_g", 16, 16, 32768, 6, 128, 128, 64, "g", "bimodal"),  # fill64=0.63
    ("gdn_h16_g", 16, 16, 32768, 6, 128, 128, 64, "g", "skew"),  # fill64=0.63
    ("gdn_h16_g", 16, 16, 32768, 6, 128, 128, 64, "g", "skew_last"),  # fill64=0.63
    ("kda_tp2", 48, 48, 32768, 2, 128, 128, 64, "gk", "ragged"),  # fill64=0.63
    ("kda_tp2", 48, 48, 32768, 2, 128, 128, 64, "gk", "bimodal"),  # fill64=0.63
    ("kda_tp2", 48, 48, 32768, 2, 128, 128, 64, "gk", "skew"),  # fill64=0.63
    ("kda_tp2", 48, 48, 32768, 2, 128, 128, 64, "gk", "skew_last"),  # fill64=0.63
    ("gdn_h16_g", 16, 16, 8192, 4, 128, 128, 64, "g", "ragged"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 8192, 4, 128, 128, 64, "g", "bimodal"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 8192, 4, 128, 128, 64, "g", "skew"),  # fill64=0.42
    ("gdn_h16_g", 16, 16, 8192, 4, 128, 128, 64, "g", "skew_last"),  # fill64=0.42
    ("kda_tp4", 24, 24, 8192, 4, 128, 128, 64, "gk", "ragged"),  # fill64=0.63
    ("kda_tp4", 24, 24, 8192, 4, 128, 128, 64, "gk", "bimodal"),  # fill64=0.63
    ("kda_tp4", 24, 24, 8192, 4, 128, 128, 64, "gk", "skew"),  # fill64=0.63
    ("kda_tp4", 24, 24, 8192, 4, 128, 128, 64, "gk", "skew_last"),  # fill64=0.63
    # ---- low-fill tail (fill64 <= 0.32) ----
    ("gdn_h8_g", 8, 4, 32768, 4, 128, 128, 64, "g", "ragged"),  # fill64=0.21
    ("gdn_h8_g", 8, 4, 32768, 4, 128, 128, 64, "g", "bimodal"),  # fill64=0.21
    ("gdn_h8_g", 8, 4, 32768, 4, 128, 128, 64, "g", "skew"),  # fill64=0.21
    ("gdn_h8_g", 8, 4, 32768, 4, 128, 128, 64, "g", "skew_last"),  # fill64=0.21
    ("kda_tp8", 12, 12, 32768, 4, 128, 128, 64, "gk", "ragged"),  # fill64=0.32
    ("kda_tp8", 12, 12, 32768, 4, 128, 128, 64, "gk", "bimodal"),  # fill64=0.32
    ("kda_tp8", 12, 12, 32768, 4, 128, 128, 64, "gk", "skew"),  # fill64=0.32
    ("kda_tp8", 12, 12, 32768, 4, 128, 128, 64, "gk", "skew_last"),  # fill64=0.32
    ("kda_tp4", 24, 24, 32768, 2, 128, 128, 64, "gk", "ragged"),  # fill64=0.32
    ("kda_tp4", 24, 24, 32768, 2, 128, 128, 64, "gk", "bimodal"),  # fill64=0.32
    ("kda_tp4", 24, 24, 32768, 2, 128, 128, 64, "gk", "skew"),  # fill64=0.32
    ("kda_tp4", 24, 24, 32768, 2, 128, 128, 64, "gk", "skew_last"),  # fill64=0.32
    # ---- high-fill tail (fill64 >= 1.26), esp. GDN scalar-g ----
    ("gdn_h16_g", 16, 16, 32768, 12, 128, 128, 64, "g", "ragged"),  # fill64=1.26
    ("gdn_h16_g", 16, 16, 32768, 12, 128, 128, 64, "g", "bimodal"),  # fill64=1.26
    ("gdn_h16_g", 16, 16, 32768, 12, 128, 128, 64, "g", "skew"),  # fill64=1.26
    ("gdn_h16_g", 16, 16, 32768, 12, 128, 128, 64, "g", "skew_last"),  # fill64=1.26
    ("gdn_h32_g", 32, 8, 32768, 6, 128, 128, 64, "g", "ragged"),  # fill64=1.26
    ("gdn_h32_g", 32, 8, 32768, 6, 128, 128, 64, "g", "bimodal"),  # fill64=1.26
    ("gdn_h32_g", 32, 8, 32768, 6, 128, 128, 64, "g", "skew"),  # fill64=1.26
    ("gdn_h32_g", 32, 8, 32768, 6, 128, 128, 64, "g", "skew_last"),  # fill64=1.26
    ("gdn_h32_g", 32, 8, 32768, 12, 128, 128, 64, "g", "ragged"),  # fill64=2.53
    ("gdn_h32_g", 32, 8, 32768, 12, 128, 128, 64, "g", "bimodal"),  # fill64=2.53
    ("gdn_h32_g", 32, 8, 32768, 12, 128, 128, 64, "g", "skew"),  # fill64=2.53
    ("gdn_h32_g", 32, 8, 32768, 12, 128, 128, 64, "g", "skew_last"),  # fill64=2.53
    ("kda_tp4", 24, 24, 32768, 16, 128, 128, 64, "gk", "ragged"),  # fill64=2.53
    ("kda_tp4", 24, 24, 32768, 16, 128, 128, 64, "gk", "bimodal"),  # fill64=2.53
    ("kda_tp4", 24, 24, 32768, 16, 128, 128, 64, "gk", "skew"),  # fill64=2.53
    ("kda_tp4", 24, 24, 32768, 16, 128, 128, 64, "gk", "skew_last"),  # fill64=2.53
    # -- fusion decision-boundary sweep (ACTUAL-fill axis) ------------------
    # The simplified fuse rule keys on the ACTUAL grid fill of the best fused
    # instance (⌈V/BV_best⌉·N·H / CU), not fill64. Near the boundary the best
    # instance is bv16, so fill ≈ 8·H·N/304; the empirical split sits around
    # 0.32. These densify actual-fill(bv16) in [0.21, 0.53] in ~0.05 steps to
    # localize the threshold: both gates, H decorrelated from N (incl. synthetic
    # H=6/10/14/18/20 with Hg=H/2 or H), and skew/ragged at the key fills (the
    # second-order term the fill axis can't capture -- the only high-fill losses
    # were H=4 skew/ragged). ``gdn_bnd_g`` / ``kda_bnd_gk`` are synthetic tags for
    # these probes (not real model configs).
    ("gdn_bnd_g", 10, 5, 8192, 1, 128, 128, 64, "g", "equal"),  # fill16=0.263
    ("gdn_bnd_g", 10, 5, 8192, 1, 128, 128, 64, "g", "skew"),  # fill16=0.263
    ("gdn_bnd_g", 10, 5, 32768, 1, 128, 128, 64, "g", "equal"),  # fill16=0.263
    ("gdn_bnd_g", 10, 5, 32768, 1, 128, 128, 64, "g", "skew"),  # fill16=0.263
    ("gdn_bnd_g", 14, 7, 8192, 1, 128, 128, 64, "g", "equal"),  # fill16=0.368
    ("gdn_bnd_g", 14, 7, 8192, 1, 128, 128, 64, "g", "skew"),  # fill16=0.368
    ("gdn_bnd_g", 14, 7, 32768, 1, 128, 128, 64, "g", "equal"),  # fill16=0.368
    ("gdn_bnd_g", 14, 7, 32768, 1, 128, 128, 64, "g", "skew"),  # fill16=0.368
    ("gdn_bnd_g", 18, 9, 8192, 1, 128, 128, 64, "g", "equal"),  # fill16=0.474
    ("gdn_bnd_g", 18, 9, 8192, 1, 128, 128, 64, "g", "skew"),  # fill16=0.474
    ("gdn_bnd_g", 18, 9, 32768, 1, 128, 128, 64, "g", "equal"),  # fill16=0.474
    ("gdn_bnd_g", 18, 9, 32768, 1, 128, 128, 64, "g", "skew"),  # fill16=0.474
    ("gdn_bnd_g", 20, 10, 8192, 1, 128, 128, 64, "g", "equal"),  # fill16=0.526
    ("gdn_bnd_g", 20, 10, 32768, 1, 128, 128, 64, "g", "equal"),  # fill16=0.526
    ("gdn_bnd_g", 6, 3, 8192, 2, 128, 128, 64, "g", "equal"),  # fill16=0.316
    ("gdn_bnd_g", 6, 3, 32768, 2, 128, 128, 64, "g", "equal"),  # fill16=0.316
    ("gdn_bnd_g", 10, 5, 8192, 2, 128, 128, 64, "g", "equal"),  # fill16=0.526
    ("gdn_bnd_g", 10, 5, 32768, 2, 128, 128, 64, "g", "equal"),  # fill16=0.526
    ("kda_bnd_gk", 10, 10, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.263
    ("kda_bnd_gk", 10, 10, 8192, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.263
    ("kda_bnd_gk", 10, 10, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.263
    ("kda_bnd_gk", 10, 10, 32768, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.263
    ("kda_bnd_gk", 14, 14, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.368
    ("kda_bnd_gk", 14, 14, 8192, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.368
    ("kda_bnd_gk", 14, 14, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.368
    ("kda_bnd_gk", 14, 14, 32768, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.368
    ("kda_bnd_gk", 18, 18, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.474
    ("kda_bnd_gk", 18, 18, 8192, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.474
    ("kda_bnd_gk", 18, 18, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.474
    ("kda_bnd_gk", 18, 18, 32768, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.474
    ("kda_bnd_gk", 20, 20, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.526
    ("kda_bnd_gk", 20, 20, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.526
    ("kda_bnd_gk", 8, 8, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.211
    ("kda_bnd_gk", 8, 8, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.211
    ("kda_bnd_gk", 16, 16, 8192, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.421
    ("kda_bnd_gk", 16, 16, 32768, 1, 128, 128, 64, "gk", "equal"),  # fill16=0.421
    ("kda_tp8", 12, 12, 8192, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.316
    ("kda_tp8", 12, 12, 8192, 1, 128, 128, 64, "gk", "ragged"),  # fill16=0.316
    ("kda_tp8", 12, 12, 32768, 1, 128, 128, 64, "gk", "skew"),  # fill16=0.316
    ("kda_tp8", 12, 12, 32768, 1, 128, 128, 64, "gk", "ragged"),  # fill16=0.316
    ("gdn_h8_g", 8, 4, 8192, 1, 128, 128, 64, "g", "skew"),  # fill16=0.211
    ("gdn_h8_g", 8, 4, 8192, 1, 128, 128, 64, "g", "ragged"),  # fill16=0.211
    ("gdn_h8_g", 8, 4, 32768, 1, 128, 128, 64, "g", "skew"),  # fill16=0.211
    ("gdn_h8_g", 8, 4, 32768, 1, 128, 128, 64, "g", "ragged"),  # fill16=0.211
    ("gdn_h16_g", 16, 16, 8192, 1, 128, 128, 64, "g", "skew"),  # fill16=0.421
    ("gdn_h16_g", 16, 16, 8192, 1, 128, 128, 64, "g", "ragged"),  # fill16=0.421
    ("gdn_h16_g", 16, 16, 32768, 1, 128, 128, 64, "g", "skew"),  # fill16=0.421
    ("gdn_h16_g", 16, 16, 32768, 1, 128, 128, 64, "g", "ragged"),  # fill16=0.421
    ("gdn_q3n_tp8", 4, 2, 8192, 3, 128, 128, 64, "g", "equal"),  # fill16=0.316
    ("gdn_q3n_tp8", 4, 2, 32768, 3, 128, 128, 64, "g", "equal"),  # fill16=0.316
    ("gdn_q3n_tp8", 4, 2, 8192, 4, 128, 128, 64, "g", "equal"),  # fill16=0.421
    ("gdn_q3n_tp8", 4, 2, 8192, 4, 128, 128, 64, "g", "skew"),  # fill16=0.421
    ("gdn_q3n_tp8", 4, 2, 32768, 4, 128, 128, 64, "g", "equal"),  # fill16=0.421
    ("gdn_q3n_tp8", 4, 2, 32768, 4, 128, 128, 64, "g", "skew"),  # fill16=0.421
]

# The bench builds gates (``g``/``gk``) in the natural-log domain and the fp32
# reference applies ``torch.exp``. The K5 wrapper's ``use_exp2=True`` path expects
# the scalar ``g`` already pre-scaled to log2 by an upstream K1+K2 producer (it does
# NOT rescale ``g`` itself), which would double-apply log2(e) here and corrupt the
# scalar-gate result. ``use_exp2=False`` selects the kernel's ``exp2(x*log2(e)) ==
# exp(x)`` lowering, matching the reference for both the ``g`` and ``gk`` paths.
_USE_EXP2 = False


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #
def _make_seqlens(pattern: str, T_flat: int, N: int, BT: int = 64) -> list[int]:
    """Per-sequence token counts for ``pattern``, summing exactly to ``T_flat``."""
    if N <= 1:
        return [T_flat]
    if pattern == "equal":
        per = T_flat // N
        return [per] * (N - 1) + [T_flat - per * (N - 1)]
    if pattern == "skew":
        weights = [float(N - 1)] + [1.0] * (N - 1)
    elif pattern == "skew_last":
        weights = [1.0] * (N - 1) + [float(N - 1)]
    elif pattern == "bimodal":
        weights = [0.35 if i % 2 == 0 else 1.65 for i in range(N)]
    elif pattern == "ragged":
        rng = random.Random(0x5EED + N)
        weights = [rng.uniform(0.3, 1.7) for _ in range(N)]
    else:
        raise ValueError(
            f"unknown seq_pattern {pattern!r}; expected one of "
            "equal / ragged / bimodal / skew"
        )
    total = sum(weights)
    lens = [max(BT, int(T_flat * w / total)) for w in weights[:-1]]
    last = T_flat - sum(lens)
    if last < BT:
        raise ValueError(
            f"seq_pattern {pattern!r} with T_flat={T_flat}, N={N} leaves a "
            f"final sequence of {last} tokens (< BT={BT}); use a larger T_flat"
        )
    lens.append(last)
    return lens


def _shape_label(idx: int, shape: tuple) -> str:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate, seq_pattern = shape
    varlen = "" if seq_pattern == "equal" else f" seqs={seq_pattern}"
    return (
        f"Shape {idx}: {model_tag} H={H} Hg={Hg} T={T_flat} N={N} "
        f"gate={gate}{varlen}"
    )


def _make_inputs(shape: tuple, device="cuda"):
    """Build K5 input tensors for the given shape tuple.

    Returns:
        k, w_hm, u_hm, w_tm, g, gk, initial_state, cu_seqlens
        where w_hm/u_hm are head-major (kernel input) and w_tm is token-major
        (reference input). ``cu_seqlens`` is ``None`` when ``N == 1``.
    """
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    B = 1
    dtype = torch.bfloat16

    k = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    w_tm = torch.randn(B, T_flat, H, K, dtype=dtype, device=device) * 0.1
    u_tm = torch.randn(B, T_flat, H, V, dtype=dtype, device=device) * 0.1
    w_hm = w_tm.permute(0, 2, 1, 3).contiguous()
    u_hm = u_tm.permute(0, 2, 1, 3).contiguous()

    g, gk = None, None
    if gate_mode == "g":
        g = (
            (torch.randn(H, T_flat, dtype=torch.float32, device=device).abs() * -0.5)
            .cumsum(dim=1)
            .contiguous()
        )
    elif gate_mode == "gk":
        gk = (
            (torch.randn(T_flat, H, K, dtype=torch.float32, device=device).abs() * -0.1)
            .cumsum(dim=0)
            .contiguous()
        )

    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    if N > 1:
        lens = _make_seqlens(seq_pattern, T_flat, N, BT)
        bounds = [0]
        for length in lens:
            bounds.append(bounds[-1] + length)
        assert bounds[-1] == T_flat, (bounds[-1], T_flat)
        cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    else:
        cu = None

    return k, w_hm, u_hm, w_tm, g, gk, h0, cu


# --------------------------------------------------------------------------- #
# HIP graph-capture machinery  (shared by both subcommands)
# --------------------------------------------------------------------------- #
# Maps id(cu_seqlens tensor) -> reusable GatedDeltaRulePrefillMetadata, built
# once per shape BEFORE warmup/capture so neither the HIP adapter nor the
# Triton K6 call does a device-to-host read inside the graph-captured closure.
_hip_meta_cache: dict = {}


class _CaptureSafeMeta:
    """Thin proxy over ``GatedDeltaRulePrefillMetadata`` that no-ops ``validate``.

    The production metadata's ``validate()`` raises unconditionally while a HIP/
    CUDA graph is capturing. Everything else the HIP wrapper and Triton K6 touch
    on the metadata (``get_chunk_schedule``) is capture-safe. The bench validates
    the metadata once before capture, so the in-capture no-op is sound.
    """

    def __init__(
        self, meta, cu_seqlens, *, chunk_size, total_prefill_tokens, num_sequences
    ):
        self._meta = meta
        meta.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            num_decodes=0,
            num_decode_tokens=0,
            total_prefill_tokens=total_prefill_tokens,
            num_sequences=num_sequences,
        )

    def validate(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._meta, name)


def _adapt_hip(hip_fn):
    """Wrap the HIP K5 wrapper so it accepts the same call shape as FlyDSL/Triton.

    Reshapes the 2-D ``g`` tensor from ``[H, T_flat]`` to ``[1, H, T_flat]``
    and sets ``g_head_major=True``. Looks up reusable prefill metadata from
    ``_hip_meta_cache`` so graph capture doesn't trigger a device-to-host read.
    """

    @functools.wraps(hip_fn)
    def _wrapped(*, k, w, u, g=None, gk=None, cu_seqlens=None, **kwargs):
        g_hip = g.unsqueeze(0) if (g is not None and g.dim() == 2) else g
        prefill_metadata = _hip_meta_cache.get(
            id(cu_seqlens) if cu_seqlens is not None else None
        )
        return hip_fn(
            k=k,
            w=w,
            u=u,
            g=g_hip,
            gk=gk,
            cu_seqlens=cu_seqlens,
            g_head_major=True,
            prefill_metadata=prefill_metadata,
            **kwargs,
        )

    return _wrapped


def _build_hip_meta(cu, T_flat, BT):
    """Build and cache a capture-safe prefill metadata object for the given cu."""
    try:
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )

        bounds = cu.detach().to("cpu", torch.int64)
        seq_lens = (bounds[1:] - bounds[:-1]).tolist()
        meta = build_gated_delta_rule_prefill_metadata(
            seq_lens,
            cu_seqlens=cu,
            chunk_size=BT,
        )
        _hip_meta_cache[id(cu)] = _CaptureSafeMeta(
            meta,
            cu,
            chunk_size=BT,
            total_prefill_tokens=int(T_flat),
            num_sequences=len(seq_lens),
        )
    except Exception as e:
        warnings.warn(f"prefill metadata build failed: {e}")


def _filter_shapes(shapes_all, gate_filter, shape_index, shape_range):
    """Return (idx, shape) pairs for the requested subset."""
    n = len(shapes_all)
    if shape_index is not None:
        if not (1 <= shape_index <= n):
            print(f"--shape-index must be 1–{n}", file=sys.stderr)
            sys.exit(1)
        return [(shape_index, shapes_all[shape_index - 1])]
    if shape_range is not None:
        lo, hi = shape_range
        if not (1 <= lo <= hi <= n):
            print(
                f"--shape-range must satisfy 1 <= START <= END <= {n}", file=sys.stderr
            )
            sys.exit(1)
        return [(i, shapes_all[i - 1]) for i in range(lo, hi + 1)]
    return [
        (i + 1, s)
        for i, s in enumerate(shapes_all)
        if gate_filter == "all" or s[8] == gate_filter
    ]


def _shape_range_type(s: str) -> tuple[int, int]:
    """Parse ``START-END`` (1-based, inclusive) or a single ``N`` into (lo, hi)."""
    parts = s.split("-")
    try:
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"--shape-range expects 'START-END' or 'N' (1-based); got {s!r}"
    )


def _add_shape_args(parser):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--shape-index",
        type=int,
        default=None,
        metavar="N",
        help="Run one preset shape by 1-based index (see --list).",
    )
    grp.add_argument(
        "--shape-range",
        type=_shape_range_type,
        default=None,
        metavar="START-END",
        help="Run a 1-based inclusive range of shapes, e.g. '5-8' or a single 'N'.",
    )
    grp.add_argument(
        "--list",
        action="store_true",
        help="List all preset shapes with indices and exit.",
    )


def _list_shapes():
    print(f"{'#':>3}  {'label':<60}  gate")
    print("-" * 75)
    for i, s in enumerate(PRESET_SHAPES, 1):
        model_tag, H, Hg, T_flat, N, K, V, BT, gate, seq_pattern = s
        sp = "" if seq_pattern == "equal" else f"  seqs={seq_pattern}"
        label = f"{model_tag}  H={H} Hg={Hg} T={T_flat} N={N} K={K} V={V}{sp}"
        print(f"{i:>3}  {label:<60}  {gate}")


# =========================================================================== #
# K5 subcommand
# =========================================================================== #
_K5_BENCH_TITLE = "GDN K5 inter-chunk state scan"

FLYDSL_PREFIX = "flydsl:"
AUTO_VARIANT = "auto"


def _calculate_tflops_k5(N, H, T_flat, K, V, time_us, BT=64, seq_lens=None):
    """Total FLOPs for K5 = GEMM1 (w@h^T) + GEMM2 (k^T@v_new) = 4·BT·K·V per chunk."""
    if time_us <= 0:
        return float("nan")
    lens = list(seq_lens) if seq_lens else [T_flat]
    n_chunks = sum(-(-length // BT) for length in lens)
    return 4 * H * n_chunks * BT * K * V / (time_us * 1e-6) / 1e12


def _available_k5_variants():
    """``(tuple_of_tags, default_tag)``, or ``(None, None)`` if FlyDSL is absent."""
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            K5_DEFAULT_VARIANT,
            K5_VARIANTS,
        )
    except ImportError:
        return None, None
    return tuple(K5_VARIANTS), K5_DEFAULT_VARIANT


def _auto_variant_for_shape(shape, cu) -> str | None:
    """The BV tag the K5 heuristic picks for this shape (for display)."""
    _tag, H, Hg, T_flat, N, _K, V, _BT, _gate, _pat = shape
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import _auto_variant

        return _auto_variant(
            H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=cu is not None
        )
    except Exception:
        return None


def _load_k5_impls(which: str, flydsl_variants: list[str] | None = None) -> dict:
    requested = {s.strip() for s in which.split(",")} if which != "all" else None
    if flydsl_variants is None:
        flydsl_variants = [AUTO_VARIANT]

    def _want(name):
        return requested is None or name in requested

    impls: dict = {}

    if _want("triton"):
        try:
            from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h_opt_vk,
            )

            impls["triton"] = chunk_gated_delta_rule_fwd_h_opt_vk
        except ImportError as e:
            warnings.warn(f"Triton K5 not available: {e}")

    if _want("flydsl"):
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl,
            )

            available, _default = _available_k5_variants()
            tags = list(flydsl_variants)
            if tags == ["all"]:
                tags = list(available or ())
            for tag in tags:
                if tag == AUTO_VARIANT:
                    impls[FLYDSL_PREFIX + AUTO_VARIANT] = (
                        chunk_gated_delta_rule_fwd_h_flydsl
                    )
                    continue
                if available is not None and tag not in available:
                    raise SystemExit(
                        f"[error] unknown FlyDSL K5 variant {tag!r}; available: "
                        f"{list(available)} (or {AUTO_VARIANT!r})."
                    )
                impls[FLYDSL_PREFIX + tag] = functools.partial(
                    chunk_gated_delta_rule_fwd_h_flydsl, variant=tag
                )
        except ImportError as e:
            warnings.warn(f"FlyDSL K5 not available: {e}")

    if _want("hip"):
        try:
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )

            impls["hip"] = _adapt_hip(chunk_gated_delta_rule_fwd_h_hip_fn)
        except ImportError as e:
            warnings.warn(f"HIP K5 not available: {e}")

    return impls


def _make_k5_closure(fn, k, w_hm, u_hm, g, gk, h0, cu):
    def _run():
        fn(
            k=k,
            w=w_hm,
            u=u_hm,
            g=g,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
        )

    return _run


_k5_ref_fn = None


def _load_k5_ref():
    global _k5_ref_fn
    if _k5_ref_fn is not None:
        return _k5_ref_fn
    try:
        mod = importlib.import_module("op_tests.test_flydsl_linear_attention_prefill")
    except Exception:
        test_path = (
            Path(_REPO_ROOT) / "op_tests/test_flydsl_linear_attention_prefill.py"
        )
        spec = importlib.util.spec_from_file_location("_test_prefill", test_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _k5_ref_fn = mod.ref_chunk_gated_delta_rule_fwd_h
    return _k5_ref_fn


# Verification thresholds.
_RMSE_TOL = 0.05
_MAXREL_TOL = 0.25


def _verdict(got, want) -> str:
    """PASS/FAIL string comparing ``got`` to ``want`` on both error measures."""
    a, b = got.float(), want.float()
    err = (a - b).abs()
    ratio = (err.pow(2).mean().sqrt() / (b.pow(2).mean().sqrt() + 1e-8)).item()
    max_rel = (err.max() / b.abs().max().clamp_min(1e-6)).item()
    ok = ratio < _RMSE_TOL and max_rel < _MAXREL_TOL
    return (
        f"{'PASS' if ok else 'FAIL'}" f"(rmse_ratio={ratio:.2e},max_rel={max_rel:.2e})"
    )


def _verify_k5_impl(
    fn, k, w_hm, u_hm, w_tm, g, gk, h0, cu, verification, baseline_fn=None
) -> str:
    if verification == "none":
        return "N/A"
    try:
        h, v_new, fs = fn(
            k=k,
            w=w_hm,
            u=u_hm,
            g=g,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
        )
    except Exception as e:
        return f"ERROR({type(e).__name__})"

    if verification == "reference":
        try:
            ref = _load_k5_ref()
            h_ref, _, _ = ref(
                k=k,
                w=w_tm,
                u=u_hm.permute(0, 2, 1, 3),
                g=g,
                gk=gk,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu,
            )
        except Exception as e:
            return f"REF-ERROR({e})"
        return _verdict(h, h_ref)

    if verification == "baseline" and baseline_fn is not None:
        try:
            h_base, _, _ = baseline_fn(
                k=k,
                w=w_hm,
                u=u_hm,
                g=g,
                gk=gk,
                initial_state=h0,
                output_final_state=True,
                save_new_value=False,
                cu_seqlens=cu,
                use_exp2=_USE_EXP2,
            )
        except Exception as e:
            return f"BASELINE-ERROR({e})"
        return _verdict(h, h_base)

    return "N/A"


def _run_one_k5(idx, impls, shape, args, cfg: MeasureConfig) -> dict:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    seq_lens = _make_seqlens(seq_pattern, T_flat, N, BT)
    label = _shape_label(idx, shape)

    try:
        k, w_hm, u_hm, w_tm, g, gk, h0, cu = _make_inputs(shape)
    except Exception as e:
        return {"label": label, "error": str(e)}

    if cu is not None and "hip" in impls:
        _build_hip_meta(cu, T_flat, BT)

    modes = ["eager", "graph"] if args.mode == "all" else [args.mode]
    baseline_name = args.baseline
    baseline_fn = impls.get(baseline_name)
    baseline_times: dict = {}
    results_by_impl: dict = {}

    # Relabel the auto row to show the concrete variant chosen for this shape.
    auto_key = FLYDSL_PREFIX + AUTO_VARIANT
    auto_label = None
    if auto_key in impls:
        resolved = _auto_variant_for_shape(shape, cu)
        if resolved:
            auto_label = FLYDSL_PREFIX + resolved
            if auto_label in impls:
                print(
                    f"  [skip] {auto_key} resolves to {auto_label}, already requested"
                )
                impls = {k2: v for k2, v in impls.items() if k2 != auto_key}

    for impl_name, fn in impls.items():
        if impl_name == auto_key and auto_label:
            impl_name = auto_label
        print(f"  {impl_name}...", end=" ", flush=True)
        closure = _make_k5_closure(fn, k, w_hm, u_hm, g, gk, h0, cu)

        try:
            closure()
            torch.cuda.synchronize()
        except NotImplementedError as e:
            print("NOT_IMPL")
            results_by_impl[impl_name] = {"error": f"NOT_IMPL: {e}"}
            continue
        except Exception as e:
            print(f"PROBE-FAIL: {e}")
            results_by_impl[impl_name] = {"error": str(e)}
            continue

        timing: dict = {}
        tflops_d: dict = {}
        for mode in modes:
            try:
                stats = _time_measure(closure, mode, cfg)
                timing[mode] = stats
                tflops_d[mode] = _calculate_tflops_k5(
                    N, H, T_flat, K, V, stats.median_us, BT, seq_lens=seq_lens
                )
            except EmptyGraphCaptureError as e:
                timing[mode] = f"GRAPH-FAIL: {e}"
                tflops_d[mode] = None
            except Exception as e:
                timing[mode] = f"ERROR: {e}"
                tflops_d[mode] = None

        verify_str = "N/A"
        if args.verification != "none":
            verify_str = _verify_k5_impl(
                fn,
                k,
                w_hm,
                u_hm,
                w_tm,
                g,
                gk,
                h0,
                cu,
                verification=args.verification,
                baseline_fn=baseline_fn if impl_name != baseline_name else None,
            )

        results_by_impl[impl_name] = {
            "timing": timing,
            "tflops": tflops_d,
            "verify": verify_str,
        }

        if impl_name == baseline_name:
            for mode in modes:
                t = timing.get(mode)
                if hasattr(t, "median_us"):
                    baseline_times[mode] = t.median_us

        for mode in modes:
            t = timing.get(mode)
            tf = tflops_d.get(mode)
            if hasattr(t, "median_us"):
                base = baseline_times.get(mode)
                sp = (
                    f"  ×{base / t.median_us:.2f}"
                    if (base and impl_name != baseline_name and t.median_us > 0)
                    else ""
                )
                tf_s = f"{tf:.3f}" if tf is not None else "—"
                print(f"[{mode}] {t.median_us:.1f} us  {tf_s} TFLOPs{sp}", end="  ")
            else:
                print(f"[{mode}] {t}", end="  ")
        print(f"  verify={verify_str}")

    return {
        "label": label,
        "shape": shape,
        "impls": results_by_impl,
        "baseline_times": baseline_times,
        "baseline_name": baseline_name,
        "modes": modes,
        "N": N,
        "H": H,
        "T_flat": T_flat,
        "K": K,
        "V": V,
    }


def run_k5(args):
    impls = _load_k5_impls(args.impl, getattr(args, "flydsl_variants", None))
    if not impls:
        print("No K5 implementations available.", file=sys.stderr)
        sys.exit(1)
    if args.baseline not in impls:
        warnings.warn(
            f"Baseline '{args.baseline}' not in loaded impls {list(impls)}; "
            "speedup columns will be empty."
        )
    cfg = make_measure_config(args)
    shapes_to_run = _filter_shapes(
        PRESET_SHAPES, args.gate, args.shape_index, getattr(args, "shape_range", None)
    )
    all_rows = []
    for idx, shape in shapes_to_run:
        print(f"\n{'='*60}\n{_shape_label(idx, shape)}\n{'='*60}")
        row = _run_one_k5(idx, impls, shape, args, cfg)
        print_result_table(row)
        all_rows.append(row)

    if args.output:
        env = collect_env_info()
        write_bench_markdown(
            args.output,
            _K5_BENCH_TITLE,
            all_rows,
            env,
            baseline_name=args.baseline,
        )
        print(f"\nMarkdown report written to {args.output}")
        try:
            results = parse_bench_md(args.output)
            stem = Path(args.output).stem
            out_dir = Path(args.output).parent
            png_path = str(out_dir / f"{stem}-plot.png")
            modes_plot = ["eager", "graph"] if args.mode == "all" else [args.mode]
            make_bar_chart(
                results,
                png_path,
                title=_K5_BENCH_TITLE,
                mode=modes_plot[0],
                baseline_label=category_label(args.baseline),
            )
            summary_md = str(out_dir / f"{stem}-summary.md")
            make_summary_md(
                results,
                summary_md,
                png_path,
                args.output,
                title=_K5_BENCH_TITLE,
                mode=modes_plot[0],
                baseline_label=category_label(args.baseline),
            )
        except Exception as e:
            warnings.warn(f"Plot/summary generation failed: {e}")


# =========================================================================== #
# K5+K6 subcommand
# =========================================================================== #
_K5K6_BENCH_TITLE = "GDN K5+K6 fused forward"

FUSED_PREFIX = "K5K6_flydsl_fused"
COMBINED_KEY = "K5K6_combined"
FUSED_VARIANTS = ("bv16", "bv32", "bv64", "bv64w8")

_K5K6_IMPL_KEYS = (
    "K5_triton+K6_triton",
    "K5_hip+K6_triton",
    "K5_flydsl+K6_triton",
    FUSED_PREFIX,
    COMBINED_KEY,
)


def _calculate_tflops_k5k6(N, H, T_flat, K, V, time_us, BT=64, seq_lens=None):
    """Total FLOPs = K5 + K6 per chunk per head."""
    if time_us <= 0:
        return float("nan")
    lens = list(seq_lens) if seq_lens else [T_flat]
    n_chunks = sum(-(-length // BT) for length in lens)
    per_chunk = (
        4 * K * V  # K5: GEMM1 (w@h) + GEMM2 (k^T@v_new)
        + 2 * K * V  # K6 GEMM3: q@h
        + 2 * BT * K  # K6 GEMM4a: q@k^T
        + 2 * BT * V  # K6 GEMM4b: A@v_new
    )
    return H * n_chunks * BT * per_chunk / (time_us * 1e-6) / 1e12


def _fused_auto_variant_for_shape(shape, cu) -> str | None:
    """The BV tag the fused kernel's auto selection picks for this shape."""
    _tag, H, _Hg, _T_flat, N, _K, V, _BT, _gate, _pat = shape
    del cu
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            _fused_bv_for_shape,
        )

        bv, num_waves = _fused_bv_for_shape(H=H, V=V, N=N, variant=None)
        return f"bv{bv}w{num_waves}" if num_waves > 4 else f"bv{bv}"
    except Exception:
        return None


def _combined_dispatch_label_for_shape(shape, cu) -> str | None:
    """Show whether the combined wrapper would pick fused or separate for this shape."""
    _tag, H, Hg, T_flat, N, _K, V, _BT, _gate, _pat = shape
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            _auto_variant,
            _fused_bv_for_shape,
            should_use_fused_gfx942,
        )

        n = (cu.shape[0] - 1) if cu is not None else 1
        if should_use_fused_gfx942(H=H, N=n, V=V):
            bv, num_waves = _fused_bv_for_shape(H=H, V=V, N=n, variant=None)
            tag = f"bv{bv}w{num_waves}" if num_waves > 4 else f"bv{bv}"
            return f"fused:{tag}"
        else:
            tag = _auto_variant(
                H=H, Hg=Hg, V=V, T_flat=T_flat, N=n, is_varlen=cu is not None
            )
            return f"separate:{tag}"
    except Exception:
        return None


def _make_q(shape, device="cuda"):
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    return torch.randn(1, T_flat, Hg, K, dtype=torch.bfloat16, device=device) * 0.1


def _separate_runner(k5_fn, k6_fn):
    """Closure running a separate K5 then Triton K6, returning o."""

    def _run(*, q, k, w, u, g, gk, h0, cu, scale, o):
        meta = _hip_meta_cache.get(id(cu)) if cu is not None else None
        h, v_new, _ = k5_fn(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
        )
        k6_fn(
            q=q,
            k=k,
            v=v_new,
            o=o,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
            prefill_metadata=meta,
        )
        return o

    return _run


def _make_fused_runner(fused_fn, variant):
    """Closure running the fused K5+K6 kernel with a fixed variant (or None for auto)."""

    def _run(*, q, k, w, u, g, gk, h0, cu, scale, o):
        meta = _hip_meta_cache.get(id(cu)) if cu is not None else None
        fused_fn(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
            o=o,
            prefill_metadata=meta,
            variant=variant,
        )
        return o

    return _run


def _make_combined_runner(combined_fn):
    """Closure running the auto-dispatch combined wrapper."""

    def _run(*, q, k, w, u, g, gk, h0, cu, scale, o):
        meta = _hip_meta_cache.get(id(cu)) if cu is not None else None
        combined_fn(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            use_exp2=_USE_EXP2,
            o=o,
            prefill_metadata=meta,
        )
        return o

    return _run


def _load_k5k6_impls(which: str, fused_variants: list[str] | None = None) -> dict:
    requested = {s.strip() for s in which.split(",")} if which != "all" else None
    if fused_variants is None:
        fused_variants = [AUTO_VARIANT]

    def _want(name):
        return requested is None or name in requested

    impls: dict = {}

    k6_triton = None
    try:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
            chunk_fwd_o_opt_vk,
        )

        k6_triton = chunk_fwd_o_opt_vk
    except ImportError as e:
        warnings.warn(f"Triton K6 not available: {e}")

    if _want("K5_triton+K6_triton") and k6_triton is not None:
        try:
            from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h_opt_vk,
            )

            impls["K5_triton+K6_triton"] = _separate_runner(
                chunk_gated_delta_rule_fwd_h_opt_vk, k6_triton
            )
        except ImportError as e:
            warnings.warn(f"Triton K5 not available: {e}")

    if _want("K5_hip+K6_triton") and k6_triton is not None:
        try:
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )

            impls["K5_hip+K6_triton"] = _separate_runner(
                _adapt_hip(chunk_gated_delta_rule_fwd_h_hip_fn), k6_triton
            )
        except ImportError as e:
            warnings.warn(f"HIP K5 not available: {e}")

    if _want("K5_flydsl+K6_triton") and k6_triton is not None:
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl,
            )

            impls["K5_flydsl+K6_triton"] = _separate_runner(
                chunk_gated_delta_rule_fwd_h_flydsl, k6_triton
            )
        except ImportError as e:
            warnings.warn(f"FlyDSL K5 not available: {e}")

    if _want(FUSED_PREFIX):
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_o_flydsl,
            )

            tags = list(fused_variants)
            if tags == ["all"]:
                tags = list(FUSED_VARIANTS)
            for tag in tags:
                if tag == AUTO_VARIANT:
                    impls[FUSED_PREFIX] = _make_fused_runner(
                        chunk_gated_delta_rule_fwd_h_o_flydsl, None
                    )
                    continue
                if tag not in FUSED_VARIANTS:
                    raise SystemExit(
                        f"[error] unknown fused variant {tag!r}; available: "
                        f"{list(FUSED_VARIANTS)} (or {AUTO_VARIANT!r})."
                    )
                impls[f"{FUSED_PREFIX}:{tag}"] = _make_fused_runner(
                    chunk_gated_delta_rule_fwd_h_o_flydsl, tag
                )
        except ImportError as e:
            warnings.warn(f"FlyDSL fused K5+K6 not available: {e}")

    if _want(COMBINED_KEY):
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_o_auto,
            )

            impls[COMBINED_KEY] = _make_combined_runner(
                chunk_gated_delta_rule_fwd_h_o_auto
            )
        except ImportError as e:
            warnings.warn(f"FlyDSL combined K5+K6 not available: {e}")

    return impls


def _make_k5k6_closure(fn, q, k, w_hm, u_hm, g, gk, h0, cu, scale, o):
    def _run():
        fn(q=q, k=k, w=w_hm, u=u_hm, g=g, gk=gk, h0=h0, cu=cu, scale=scale, o=o)

    return _run


def _run_one_k5k6(idx, impls, shape, args, cfg) -> dict:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    seq_lens = _make_seqlens(seq_pattern, T_flat, N, BT)
    label = _shape_label(idx, shape)

    try:
        k, w_hm, u_hm, w_tm, g, gk, h0, cu = _make_inputs(shape)
        q = _make_q(shape)
    except Exception as e:
        return {"label": label, "error": str(e)}

    scale = K**-0.5
    o = u_hm.new_empty(1, T_flat, H, V)

    if cu is not None:
        _build_hip_meta(cu, T_flat, BT)

    modes = ["eager", "graph"] if args.mode == "all" else [args.mode]
    baseline_name = args.baseline
    baseline_times: dict = {}
    results_by_impl: dict = {}
    o_ref = None

    # Per-shape display labels for auto-selected impls.
    fused_auto_display = None
    if FUSED_PREFIX in impls:
        resolved = _fused_auto_variant_for_shape(shape, cu)
        if resolved:
            fused_auto_display = f"{FUSED_PREFIX} ({resolved})"

    combined_display = None
    if COMBINED_KEY in impls:
        resolved = _combined_dispatch_label_for_shape(shape, cu)
        if resolved:
            combined_display = f"{COMBINED_KEY} ({resolved})"

    for impl_name, fn in impls.items():
        if impl_name == FUSED_PREFIX and fused_auto_display:
            display_name = fused_auto_display
        elif impl_name == COMBINED_KEY and combined_display:
            display_name = combined_display
        else:
            display_name = impl_name

        print(f"  {display_name}...", end=" ", flush=True)
        closure = _make_k5k6_closure(fn, q, k, w_hm, u_hm, g, gk, h0, cu, scale, o)

        try:
            closure()
            torch.cuda.synchronize()
        except NotImplementedError as e:
            print("NOT_IMPL")
            results_by_impl[display_name] = {"error": f"NOT_IMPL: {e}"}
            continue
        except Exception as e:
            print(f"PROBE-FAIL: {e}")
            results_by_impl[display_name] = {"error": str(e)}
            continue

        verify_str = "N/A"
        if args.verification != "none":
            o_now = o.detach().clone()
            if o_ref is None:
                o_ref = o_now
                verify_str = "REF"
            else:
                verify_str = _verdict(o_now, o_ref)

        timing: dict = {}
        tflops_d: dict = {}
        for mode in modes:
            try:
                stats = _time_measure(closure, mode, cfg)
                timing[mode] = stats
                tflops_d[mode] = _calculate_tflops_k5k6(
                    N, H, T_flat, K, V, stats.median_us, BT, seq_lens=seq_lens
                )
            except EmptyGraphCaptureError as e:
                timing[mode] = f"GRAPH-FAIL: {e}"
                tflops_d[mode] = None
            except Exception as e:
                timing[mode] = f"ERROR: {e}"
                tflops_d[mode] = None

        results_by_impl[display_name] = {
            "timing": timing,
            "tflops": tflops_d,
            "verify": verify_str,
        }
        if impl_name == baseline_name:
            for mode in modes:
                t = timing.get(mode)
                if hasattr(t, "median_us"):
                    baseline_times[mode] = t.median_us

        for mode in modes:
            t = timing.get(mode)
            tf = tflops_d.get(mode)
            if hasattr(t, "median_us"):
                base = baseline_times.get(mode)
                sp = (
                    f"  ×{base / t.median_us:.2f}"
                    if (base and impl_name != baseline_name and t.median_us > 0)
                    else ""
                )
                tf_s = f"{tf:.3f}" if tf is not None else "—"
                print(f"[{mode}] {t.median_us:.1f} us  {tf_s} TFLOPs{sp}", end="  ")
            else:
                print(f"[{mode}] {t}", end="  ")
        print(f"  verify={verify_str}")

    return {
        "label": label,
        "shape": shape,
        "impls": results_by_impl,
        "baseline_times": baseline_times,
        "baseline_name": baseline_name,
        "modes": modes,
        "N": N,
        "H": H,
        "T_flat": T_flat,
        "K": K,
        "V": V,
    }


def run_k5k6(args):
    impls = _load_k5k6_impls(args.impl, getattr(args, "fused_variants", None))
    if not impls:
        print("No K5+K6 implementations available.", file=sys.stderr)
        sys.exit(1)
    if args.baseline not in impls:
        warnings.warn(
            f"Baseline '{args.baseline}' not in loaded impls {list(impls)}; "
            "speedup columns will be empty."
        )
    cfg = make_measure_config(args)
    shapes_to_run = _filter_shapes(
        PRESET_SHAPES, args.gate, args.shape_index, getattr(args, "shape_range", None)
    )
    all_rows = []
    for idx, shape in shapes_to_run:
        print(f"\n{'='*60}\n{_shape_label(idx, shape)}\n{'='*60}")
        row = _run_one_k5k6(idx, impls, shape, args, cfg)
        print_result_table(row)
        all_rows.append(row)

    if args.output:
        env = collect_env_info()
        write_bench_markdown(
            args.output,
            _K5K6_BENCH_TITLE,
            all_rows,
            env,
            baseline_name=args.baseline,
        )
        print(f"\nMarkdown report written to {args.output}")
        modes_out = ["eager", "graph"] if args.mode == "all" else [args.mode]
        plot_mode = "graph" if "graph" in modes_out else modes_out[0]
        try:
            results = parse_bench_md(args.output)
            png = str(Path(args.output).with_suffix("")) + "-fill-scatter.png"
            make_fill_scatter(
                results,
                png,
                title=f"{_K5K6_BENCH_TITLE}: speedup vs grid fill ({plot_mode})",
                mode=plot_mode,
            )
        except Exception as e:
            warnings.warn(f"fill-scatter generation failed: {e}")


# =========================================================================== #
# Argument parsers
# =========================================================================== #
def _build_k5_parser(sub):
    p = sub.add_parser(
        "k5",
        help="Benchmark K5 (inter-chunk state scan) in isolation.",
        description="A/B benchmark: GDN K5 inter-chunk state scan.",
    )
    _add_shape_args(p)
    p.add_argument(
        "--impl",
        default="all",
        metavar="IMPLS",
        help="Comma-separated: triton, flydsl, hip. Default: all.",
    )
    p.add_argument(
        "--baseline",
        default="triton",
        choices=("triton", "hip"),
        help="Speedup baseline (default: triton).",
    )
    p.add_argument(
        "--flydsl-variants",
        type=str,
        default=AUTO_VARIANT,
        metavar="V1,V2,...",
        help=(
            "Comma-separated FlyDSL K5 variant tags: bv16, bv32, bv64, bv64w8, … "
            "'all' runs every registered variant; 'auto' (default) uses the "
            "shape-adaptive heuristic."
        ),
    )
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="List the registered FlyDSL K5 variants and exit.",
    )
    p.add_argument(
        "--gate",
        default="all",
        choices=("g", "gk", "all"),
        help="Filter shapes by gate type: g (GDN), gk (KDA), all (default).",
    )
    add_timing_args(p)
    add_output_args(p)
    add_verification_args(p)
    return p


def _build_k5k6_parser(sub):
    p = sub.add_parser(
        "k5k6",
        help="Benchmark K5+K6 (state scan + output), fused vs separate.",
        description="A/B benchmark: GDN K5+K6 fused forward.",
    )
    _add_shape_args(p)
    p.add_argument(
        "--impl",
        default="all",
        metavar="IMPLS",
        help=(
            f"Comma-separated impls: {', '.join(_K5K6_IMPL_KEYS)}. "
            "Default: all. "
            f"'{COMBINED_KEY}' is the heuristic-driven combined wrapper."
        ),
    )
    p.add_argument(
        "--baseline",
        default="K5_flydsl+K6_triton",
        choices=_K5K6_IMPL_KEYS,
        help="Speedup baseline (default: K5_flydsl+K6_triton).",
    )
    p.add_argument(
        "--fused-variants",
        type=str,
        default=AUTO_VARIANT,
        metavar="V1,V2,...",
        help=(
            "Comma-separated fused-kernel BV variant tags (bv16, bv32, bv64, bv64w8, …). "
            "'all' runs every variant; 'auto' (default) uses shape-adaptive selection. "
            f"Applies only to '{FUSED_PREFIX}'; the separate and combined paths are not "
            "variant-parametrized."
        ),
    )
    p.add_argument(
        "--list-fused-variants",
        action="store_true",
        help="List the fused-kernel BV variants and exit.",
    )
    p.add_argument(
        "--gate",
        default="all",
        choices=("g", "gk", "all"),
        help="Filter shapes by gate type.",
    )
    add_timing_args(p)
    add_output_args(p)
    add_verification_args(p)
    return p


def main():
    top = argparse.ArgumentParser(
        description="Unified GDN K5 / K5+K6 benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = top.add_subparsers(
        dest="bench_mode",
        required=True,
        title="subcommands",
        description="Choose which benchmark to run.",
    )
    _build_k5_parser(sub)
    _build_k5k6_parser(sub)

    args = top.parse_args()

    if args.bench_mode == "k5":
        if args.list_variants:
            avail, default = _available_k5_variants()
            if avail is None:
                print("FlyDSL is unavailable; no K5 variants to list.")
            else:
                print("FlyDSL GDN K5 kernel variants (BV = V-tile width):")
                for v in avail:
                    print(f"    {v}")
                print(
                    f"  {'*' if default == AUTO_VARIANT else ' '} {AUTO_VARIANT}"
                    "  (shape-adaptive)"
                )
            return

        args.flydsl_variants = [
            v.strip() for v in args.flydsl_variants.split(",") if v.strip()
        ] or [AUTO_VARIANT]

        if args.list:
            _list_shapes()
            return

        run_k5(args)

    else:  # k5k6
        if args.list_fused_variants:
            print("Fused GDN K5+K6 kernel BV variants:")
            for v in FUSED_VARIANTS:
                note = "  (aliases lds_A onto lds_h to fit LDS)" if v == "bv64" else ""
                print(f"    {v}{note}")
            print(f"  * {AUTO_VARIANT}  (shape-adaptive)")
            return

        args.fused_variants = [
            v.strip() for v in args.fused_variants.split(",") if v.strip()
        ] or [AUTO_VARIANT]

        if args.list:
            _list_shapes()
            return

        run_k5k6(args)


if __name__ == "__main__":
    main()
