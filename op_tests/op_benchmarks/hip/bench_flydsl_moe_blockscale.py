# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone benchmark for the FlyDSL FP8 blockscale MoE GEMM kernel.

Sweeps a list of model shapes (presets) × token counts (M) and compares
FlyDSL stage1+stage2 against CK 2-stage and aiter's ASM fused fallback.
Defaults focus on DeepSeek-R1 (M=8192, dim=7168, idim=512, E=257, k=9).

Usage::

    # Default: DSR1 preset, M sweep, all comparators, CSV out to logs/.
    python op_tests/op_benchmarks/hip/bench_flydsl_moe_blockscale.py

    # Single shape, FlyDSL only
    python op_tests/op_benchmarks/hip/bench_flydsl_moe_blockscale.py \\
        --M 8192 --compare ""

    # Sweep every preset at two M values
    python op_tests/op_benchmarks/hip/bench_flydsl_moe_blockscale.py \\
        --preset all --M 1024 4096 --compare ck

Caveats:
  * Only the ``dsr1`` preset has a tuned tile row in the dispatcher CSV.
    Other presets run with the default tile (64,128,128) and may be far
    from optimal — read the non-DSR1 numbers as exploratory, not tuned.
  * Tier-C kwarg combinations on non-DSR1 shapes are skipped (logged),
    not raised, so a multi-preset run completes end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch

# Make the test-file helpers importable. The flydsl_tests dir has no
# __init__.py, so load the module directly from its path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_FILE = _REPO_ROOT / "op_tests" / "flydsl_tests" / "test_flydsl_blockscale_moe.py"

_spec = importlib.util.spec_from_file_location(
    "_flydsl_blockscale_test_helpers", _TEST_FILE
)
_helpers = importlib.util.module_from_spec(_spec)
sys.modules["_flydsl_blockscale_test_helpers"] = _helpers
_spec.loader.exec_module(_helpers)

_prepare_data = _helpers._prepare_data
_launch_flydsl_stage1 = _helpers._launch_flydsl_stage1
_launch_flydsl_stage2 = _helpers._launch_flydsl_stage2

import aiter  # noqa: E402
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import moe_sorting as aiter_moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.blockscale_moe_gemm_2stage import (  # noqa: E402
    SCALE_BLOCK_K_DEFAULT,
    SCALE_BLOCK_N_DEFAULT,
)
from aiter.ops.moe_op import ck_moe_stage1_fwd, ck_moe_stage2_fwd  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Shape presets
# ---------------------------------------------------------------------------
# model_dim, inter_dim (post-TP, per-rank), experts, topk
PRESETS: dict[str, dict[str, int]] = {
    "dsr1": dict(model_dim=7168, inter_dim=512, experts=257, topk=9),
    "dsr1-tp4": dict(model_dim=7168, inter_dim=1024, experts=257, topk=9),
    "dsr1-tp1": dict(model_dim=7168, inter_dim=2048, experts=257, topk=9),
    "mixtral-8x7b": dict(model_dim=4096, inter_dim=1792, experts=8, topk=2),
    "mixtral-8x22b": dict(model_dim=6144, inter_dim=2048, experts=8, topk=2),
    "qwen3-235b": dict(model_dim=4096, inter_dim=1536, experts=128, topk=8),
    "gpt-oss-120b": dict(model_dim=2880, inter_dim=2880, experts=128, topk=4),
}

DEFAULT_M_SWEEP = [1, 8, 64, 256, 1024, 4096, 8192]


# ---------------------------------------------------------------------------
# FLOPS accounting (matches the gated DSR1 perf test)
# ---------------------------------------------------------------------------
def _flops(M: int, model_dim: int, inter_dim: int, topk: int):
    flops_s1 = 2 * M * topk * (2 * inter_dim) * model_dim
    flops_s2 = 2 * M * topk * model_dim * inter_dim
    return flops_s1, flops_s2, flops_s1 + flops_s2


def _tflops(flops: int, us: float) -> float:
    if us <= 0 or flops <= 0:
        return 0.0
    return flops / (us / 1e6) / 1e12


# ---------------------------------------------------------------------------
# Comparator launchers
# ---------------------------------------------------------------------------
def _bench_ck(
    data, fly_out1, a2_bq, a2_scale_2d, *, num_iters, num_warmup, ck_block_m=32
):
    tokens = data["tokens"]
    model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]
    experts = data["experts"]
    topk = data["topk"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT

    sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a, _ = aiter_moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        experts,
        model_dim,
        torch.bfloat16,
        ck_block_m,
    )
    out_ck_s1 = torch.zeros(tokens, topk, inter_dim, device=DEVICE, dtype=torch.float16)
    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_n_w2 = model_dim // blk_n
    nblk_k_w1 = model_dim // blk_k
    nblk_k_w2 = inter_dim // blk_k
    w1_scale_ck = (
        data["w1_bscale_flat"].view(experts, nblk_n_w1, nblk_k_w1).contiguous()
    )
    w2_scale_ck = (
        data["w2_bscale_flat"].view(experts, nblk_n_w2, nblk_k_w2).contiguous()
    )

    def _run_ck_s1():
        out_ck_s1.zero_()
        ck_moe_stage1_fwd(
            hidden_states=data["a1_bq"],
            w1=data["w1_bq_shuf"],
            w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids_a,
            sorted_expert_ids=sorted_e_a,
            num_valid_ids=num_valid_a,
            out=out_ck_s1,
            topk=topk,
            kernelName="",
            w1_scale=w1_scale_ck,
            a1_scale=data["a1_bscale"],
            block_m=ck_block_m,
            sorted_weights=None,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
            dst_type=torch.float16,
        )

    _, us_ck_s1 = run_perftest(_run_ck_s1, num_iters=num_iters, num_warmup=num_warmup)

    out_ck_s2 = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.float16)
    a2_for_ck = a2_bq.view(tokens, topk, inter_dim)

    def _run_ck_s2():
        out_ck_s2.zero_()
        ck_moe_stage2_fwd(
            inter_states=a2_for_ck,
            w1=data["w1_bq_shuf"],
            w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids_a,
            sorted_expert_ids=sorted_e_a,
            num_valid_ids=num_valid_a,
            out=out_ck_s2,
            topk=topk,
            kernelName="",
            w2_scale=w2_scale_ck,
            a2_scale=a2_scale_2d,
            block_m=ck_block_m,
            sorted_weights=sorted_w_a,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
        )

    _, us_ck_s2 = run_perftest(_run_ck_s2, num_iters=num_iters, num_warmup=num_warmup)
    return us_ck_s1, us_ck_s2, sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a


def _bench_asm(
    data, sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a, *, num_iters, num_warmup
):
    tokens = data["tokens"]
    model_dim = data["model_dim"]
    topk = data["topk"]
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT
    a1_scale_aiter = data["a1_bscale"].t().contiguous()
    out_asm = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.bfloat16)

    def _run_asm():
        out_asm.zero_()
        aiter.fmoe_fp8_blockscale_g1u1(
            out_asm,
            data["a1_bq"],
            data["w1_bq_shuf"],
            data["w2_bq_shuf"],
            sorted_ids_a,
            sorted_w_a,
            sorted_e_a,
            num_valid_a,
            topk,
            a1_scale_aiter,
            data["w1_bscale_flat"],
            data["w2_bscale_flat"],
            "",
            blk_n,
            blk_k,
            None,
        )

    _, us = run_perftest(_run_asm, num_iters=num_iters, num_warmup=num_warmup)
    return us


# ---------------------------------------------------------------------------
# Per (preset, M) measurement
# ---------------------------------------------------------------------------
def _bench_one(
    preset_name: str,
    shape: dict,
    M: int,
    *,
    tile: tuple[int, int, int],
    waves_per_eu: int,
    act: str,
    compare: set[str],
    num_iters: int,
    num_warmup: int,
):
    tile_m, tile_n, tile_k = tile
    model_dim = shape["model_dim"]
    inter_dim = shape["inter_dim"]
    experts = shape["experts"]
    topk = shape["topk"]

    try:
        data = _prepare_data(M, model_dim, inter_dim, experts, topk)
    except Exception as e:
        return [
            {
                "preset": preset_name,
                "M": M,
                "kernel": "data-prep",
                "stage": "-",
                "us": float("nan"),
                "tflops": 0.0,
                "vs_flydsl_total": float("nan"),
                "note": f"skipped ({type(e).__name__}: {str(e)[:80]})",
            }
        ]

    # FlyDSL
    try:
        out1, a1_bq, _ref, sorted_ids, sorted_w, sorted_e, num_valid, us_fly_s1 = (
            _launch_flydsl_stage1(
                data,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                waves_per_eu=waves_per_eu,
                act=act,
            )
        )
        out2, a2_bq, a2_scale_2d, us_fly_s2 = _launch_flydsl_stage2(
            data,
            out1,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            waves_per_eu=waves_per_eu,
            sorted_ids=sorted_ids,
            sorted_w=sorted_w,
            sorted_e=sorted_e,
            num_valid=num_valid,
        )
        us_fly_total = us_fly_s1 + us_fly_s2
        fly_note = ""
    except (NotImplementedError, RuntimeError, AssertionError) as e:
        return [
            {
                "preset": preset_name,
                "M": M,
                "kernel": "FlyDSL",
                "stage": "-",
                "us": float("nan"),
                "tflops": 0.0,
                "vs_flydsl_total": float("nan"),
                "note": f"skipped ({type(e).__name__}: {str(e)[:120]})",
            }
        ]

    flops_s1, flops_s2, flops_total = _flops(M, model_dim, inter_dim, topk)
    rows = [
        {
            "preset": preset_name,
            "M": M,
            "kernel": "FlyDSL",
            "stage": "s1",
            "us": us_fly_s1,
            "tflops": _tflops(flops_s1, us_fly_s1),
            "vs_flydsl_total": float("nan"),
            "note": fly_note,
        },
        {
            "preset": preset_name,
            "M": M,
            "kernel": "FlyDSL",
            "stage": "s2",
            "us": us_fly_s2,
            "tflops": _tflops(flops_s2, us_fly_s2),
            "vs_flydsl_total": float("nan"),
            "note": "",
        },
        {
            "preset": preset_name,
            "M": M,
            "kernel": "FlyDSL",
            "stage": "total",
            "us": us_fly_total,
            "tflops": _tflops(flops_total, us_fly_total),
            "vs_flydsl_total": 1.0,
            "note": "",
        },
    ]

    sorted_ids_a = sorted_w_a = sorted_e_a = num_valid_a = None

    if "ck" in compare:
        try:
            us_ck_s1, us_ck_s2, sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a = (
                _bench_ck(
                    data,
                    out1,
                    a2_bq,
                    a2_scale_2d,
                    num_iters=num_iters,
                    num_warmup=num_warmup,
                )
            )
            us_ck_total = us_ck_s1 + us_ck_s2
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "CK",
                    "stage": "s1",
                    "us": us_ck_s1,
                    "tflops": _tflops(flops_s1, us_ck_s1),
                    "vs_flydsl_total": us_ck_s1 / us_fly_s1,
                    "note": "",
                }
            )
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "CK",
                    "stage": "s2",
                    "us": us_ck_s2,
                    "tflops": _tflops(flops_s2, us_ck_s2),
                    "vs_flydsl_total": us_ck_s2 / us_fly_s2,
                    "note": "",
                }
            )
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "CK",
                    "stage": "total",
                    "us": us_ck_total,
                    "tflops": _tflops(flops_total, us_ck_total),
                    "vs_flydsl_total": us_ck_total / us_fly_total,
                    "note": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "CK",
                    "stage": "total",
                    "us": float("nan"),
                    "tflops": 0.0,
                    "vs_flydsl_total": float("nan"),
                    "note": f"skipped ({type(e).__name__}: {str(e)[:120]})",
                }
            )

    if "asm" in compare:
        try:
            if sorted_ids_a is None:
                # Need the ck_block_m=32 sorting buffers; rebuild.
                sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a, _ = (
                    aiter_moe_sorting(
                        data["topk_ids"],
                        data["topk_weights"],
                        experts,
                        model_dim,
                        torch.bfloat16,
                        32,
                    )
                )
            us_asm = _bench_asm(
                data,
                sorted_ids_a,
                sorted_w_a,
                sorted_e_a,
                num_valid_a,
                num_iters=num_iters,
                num_warmup=num_warmup,
            )
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "ASM",
                    "stage": "fused",
                    "us": us_asm,
                    "tflops": _tflops(flops_total, us_asm),
                    "vs_flydsl_total": us_asm / us_fly_total,
                    "note": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "preset": preset_name,
                    "M": M,
                    "kernel": "ASM",
                    "stage": "fused",
                    "us": float("nan"),
                    "tflops": 0.0,
                    "vs_flydsl_total": float("nan"),
                    "note": f"skipped ({type(e).__name__}: {str(e)[:120]})",
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Print + CSV
# ---------------------------------------------------------------------------
def _print_block(preset_name: str, shape: dict, M: int, rows: list[dict]):
    print(
        f"\n=== preset={preset_name}  M={M}  "
        f"dim={shape['model_dim']}  idim={shape['inter_dim']}  "
        f"E={shape['experts']}  k={shape['topk']} ==="
    )
    print(
        f"  {'kernel':<8s} {'stage':<6s} | {'us':>10s} | {'TFLOPS':>9s} | {'vs FlyDSL':>10s}  note"
    )
    print(f"  {'-'*8} {'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}--{'-'*40}")
    for r in rows:
        us = r["us"]
        vs = r["vs_flydsl_total"]
        us_s = f"{us:>10.1f}" if us == us else f"{'nan':>10s}"
        tf_s = f"{r['tflops']:>9.2f}" if r["tflops"] else f"{'-':>9s}"
        vs_s = f"{vs:>9.2f}x" if vs == vs else f"{'-':>10s}"
        print(
            f"  {r['kernel']:<8s} {r['stage']:<6s} | {us_s} | {tf_s} | {vs_s}  {r['note']}"
        )


def _write_csv(out_path: Path, all_rows: list[dict], tile, waves_per_eu, act):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "M",
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "tile_m",
        "tile_n",
        "tile_k",
        "waves_per_eu",
        "act",
        "kernel",
        "stage",
        "us",
        "tflops",
        "vs_flydsl_total",
        "note",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            shape = PRESETS[r["preset"]]
            w.writerow(
                {
                    "preset": r["preset"],
                    "M": r["M"],
                    "model_dim": shape["model_dim"],
                    "inter_dim": shape["inter_dim"],
                    "experts": shape["experts"],
                    "topk": shape["topk"],
                    "tile_m": tile[0],
                    "tile_n": tile[1],
                    "tile_k": tile[2],
                    "waves_per_eu": waves_per_eu,
                    "act": act,
                    "kernel": r["kernel"],
                    "stage": r["stage"],
                    "us": r["us"],
                    "tflops": r["tflops"],
                    "vs_flydsl_total": r["vs_flydsl_total"],
                    "note": r["note"],
                }
            )
    print(f"\nWrote CSV: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_shapes_csv(path: Path) -> dict[str, dict]:
    shapes: dict[str, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes[row["name"]] = dict(
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
                experts=int(row["experts"]),
                topk=int(row["topk"]),
            )
    return shapes


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="bench_flydsl_moe_blockscale")
    p.add_argument(
        "--preset",
        default="dsr1",
        help=f"Shape preset name, or 'all'. Options: " f"{', '.join(PRESETS)}, all",
    )
    p.add_argument(
        "--shapes-csv",
        type=Path,
        default=None,
        help="Optional CSV (header: name,model_dim,inter_dim,experts,topk) "
        "to use instead of built-in presets.",
    )
    p.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=None,
        help=f"Token counts to sweep. Default: {DEFAULT_M_SWEEP}",
    )
    p.add_argument(
        "--tile",
        type=int,
        nargs=3,
        default=[64, 128, 128],
        metavar=("TILE_M", "TILE_N", "TILE_K"),
    )
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument(
        "--out-dtype",
        default="bf16",
        help="Output dtype (informational; helpers currently emit f16).",
    )
    p.add_argument("--act", default="silu", choices=["silu", "gelu"])
    p.add_argument(
        "--compare",
        default="ck,asm",
        help="Comma list subset of {ck,asm}, or '' for FlyDSL only.",
    )
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV output path. Default: logs/flydsl_moe_blockscale_<act>.csv",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP device required", file=sys.stderr)
        return 2

    if args.shapes_csv:
        shapes_pool = _load_shapes_csv(args.shapes_csv)
        # Inject into PRESETS so _write_csv can look up the shape dict by name.
        PRESETS.update(shapes_pool)
        preset_names = list(shapes_pool)
    elif args.preset == "all":
        preset_names = list(PRESETS)
    else:
        if args.preset not in PRESETS:
            print(
                f"ERROR: unknown preset {args.preset!r}; "
                f"choose from {list(PRESETS)} or 'all'",
                file=sys.stderr,
            )
            return 2
        preset_names = [args.preset]

    M_list = args.M if args.M else DEFAULT_M_SWEEP

    compare = set(s.strip() for s in args.compare.split(",") if s.strip())
    bad = compare - {"ck", "asm"}
    if bad:
        print(f"ERROR: --compare contains unknown comparators: {bad}", file=sys.stderr)
        return 2

    out_csv = args.out_csv or Path("logs") / f"flydsl_moe_blockscale_{args.act}.csv"

    all_rows: list[dict] = []
    for preset_name in preset_names:
        shape = PRESETS[preset_name]
        for M in M_list:
            rows = _bench_one(
                preset_name,
                shape,
                M,
                tile=tuple(args.tile),
                waves_per_eu=args.waves_per_eu,
                act=args.act,
                compare=compare,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
            )
            _print_block(preset_name, shape, M, rows)
            all_rows.extend(rows)

    _write_csv(out_csv, all_rows, tuple(args.tile), args.waves_per_eu, args.act)
    return 0


if __name__ == "__main__":
    sys.exit(main())
