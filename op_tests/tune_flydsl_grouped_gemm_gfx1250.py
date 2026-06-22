#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tile-size tuner for the gfx1250 grouped MoE GEMM (``aiter.fused_moe``).

gemm1 (N=2*inter_dim, K=model_dim) and gemm2 (N=model_dim, K=inter_dim) have
different shapes, so this tuner tunes the two stages **independently** -- the
same separate-then-combine strategy ``FmoeTuner`` uses in
``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`` (each stage benchmarked
on its own, results merged on the shared M-tile ``block_m``).

In the grouped path the M-tiling (``tile_m`` / ``m_warp``) is shared between the
stages -- it drives ``max_m`` padding and the per-row scale layout that both
stages read from the same grouped buffers -- while ``tile_n`` / ``tile_k`` /
``n_warp`` / ``num_buffers`` / ``split_k`` are per-stage. So we sweep:

    for each shared (tile_m, m_warp):
        best stage1 N/K config  (stage2 held at a baseline)   # tune gemm1
        best stage2 N/K config  (stage1 held at best-stage1)  # tune gemm2
        combined run + accuracy check                         # verify precision
    pick the (tile_m, m_warp) with the lowest combined latency

Because ``fused_moe`` runs both stages in one call we cannot time a stage in
isolation; instead we hold the other stage fixed and minimise end-to-end us --
only the swept stage's time varies, so the arg-min is that stage's optimum
(coordinate descent). The winning config is then validated end-to-end against
the PyTorch reference (logits_diff gate) so a fast-but-wrong tile config is
never written.

Per-stage configs are emitted as ``tile_n1/tile_k1/n_warp1/num_buffers1`` (+
``split_k1``) and the ``*2`` set; ``grouped_moe_gfx1250.py`` reads them and
falls back to the legacy shared columns when absent (backward compatible).

Examples::

    # Dry-run: enumerate the per-stage search spaces (no GPU; any arch)
    python op_tests/tune_flydsl_grouped_gemm_gfx1250.py --dry-run \
        --data-format a8w4 --model-dim 7168 --inter-dim 2048 \
        --experts 256 --topk 8

    # Tune on gfx1250, write the best per-stage config per token count
    python op_tests/tune_flydsl_grouped_gemm_gfx1250.py \
        --data-format a8w4 --model-dim 7168 --inter-dim 2048 \
        --experts 256 --topk 8 --tokens 1 8 64 \
        --tune-output /tmp/tuned_grouped.csv
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import os
import sys
import tempfile

import torch

from aiter import ActivationType, QuantType, logger
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import moe_shuffle_scale, shuffle_weight
from aiter.utility import dtypes

# Reuse the test module's tensor builders / routing helpers + the precision
# harness verbatim so tuned inputs and the accuracy gate match the test suite.
from test_flydsl_grouped_gemm_gfx1250 import (
    LOGITS_DIFF_TOL,
    SCALE_BLOCK,
    _gguu_to_gugu_rows,
    _logits_diff,
    _make_topk,
    _pattern_packed,
    _run_grouped_via_fused_moe,
    init_weight_scales,
    is_gfx1250,
    set_data_format,
)

# Build every tensor straight on the device (mirrors the test module).
torch.set_default_device("cuda")


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
TUNE_TILE_M = [16, 32, 64, 128, 256]
TUNE_M_WARP = [1, 2, 4]
TUNE_TILE_N = [64, 128, 256, 512]
TUNE_TILE_K = [128, 256, 512]
TUNE_N_WARP = [1, 2, 4]
TUNE_NUM_BUFFERS = [2, 3, 4]
TUNE_SPLIT_K = [1, 2]

# WMMA / wave constants for gfx1250 (see gemm_mxscale_gfx1250.py).
WMMA_M = 16
WMMA_N = 16
WMMA_K = 128
WAVE_SIZE = 32


# ---------------------------------------------------------------------------
# Candidate generation + constraint filtering
# ---------------------------------------------------------------------------
def _valid_m_config(tile_m: int, m_warp: int) -> bool:
    """Shared M-tiling validity (independent of stage N/K)."""
    if tile_m % WMMA_M != 0 or tile_m % m_warp != 0:
        return False
    if (tile_m // m_warp) % WMMA_M != 0:
        return False
    return True


def _valid_stage_config(
    tile_n: int,
    tile_k: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    *,
    m_warp: int,
    K: int,
    wmma_n_eff: int,
    pack_a: int,
    pack_b: int,
) -> bool:
    """Per-stage N/K-tiling validity for one stage's K dim and the shared
    ``m_warp`` (the thread-count limit couples m_warp and n_warp)."""
    if tile_n % WMMA_N != 0 or tile_k % WMMA_K != 0:
        return False
    if tile_n % n_warp != 0 or (tile_n // n_warp) % wmma_n_eff != 0:
        return False
    if (m_warp * n_warp) * WAVE_SIZE > 1024:  # block_threads <= 1024
        return False
    if (tile_k // pack_a) % 4 != 0 or (tile_k // pack_b) % 4 != 0:
        return False
    if K % tile_k != 0 or K % split_k != 0:
        return False
    split_k_chunk = K // split_k
    if split_k_chunk % tile_k != 0:
        return False
    if (split_k_chunk // tile_k) < num_buffers:  # pipeline depth
        return False
    return True


def _generate_m_configs() -> list[dict]:
    return [
        {"tile_m": tm, "m_warp": mw}
        for tm in TUNE_TILE_M
        for mw in TUNE_M_WARP
        if _valid_m_config(tm, mw)
    ]


def _generate_stage_configs(K: int, m_warp: int, data_format: str) -> list[dict]:
    """Valid (tile_n, tile_k, n_warp, num_buffers, split_k) for one stage."""
    is_fp4 = data_format in ("a4w4", "fp4")
    wmma_n_eff = 32 if is_fp4 else 16
    pack_a = 2 if is_fp4 else 1  # fp4 activation packs 2/byte; a8w4 fp8 is 1/byte
    pack_b = 2  # fp4 weight in both formats

    out = []
    for tile_n in TUNE_TILE_N:
        for tile_k in TUNE_TILE_K:
            for n_warp in TUNE_N_WARP:
                for num_buffers in TUNE_NUM_BUFFERS:
                    for split_k in TUNE_SPLIT_K:
                        if _valid_stage_config(
                            tile_n,
                            tile_k,
                            n_warp,
                            num_buffers,
                            split_k,
                            m_warp=m_warp,
                            K=K,
                            wmma_n_eff=wmma_n_eff,
                            pack_a=pack_a,
                            pack_b=pack_b,
                        ):
                            out.append(
                                {
                                    "tile_n": tile_n,
                                    "tile_k": tile_k,
                                    "n_warp": n_warp,
                                    "num_buffers": num_buffers,
                                    "split_k": split_k,
                                }
                            )
    return out


def _baseline_stage(configs: list[dict]) -> dict:
    """Pick a sensible coordinate-descent starting point: the production default
    (tile_n=256, tile_k=256, n_warp=4, num_buffers=2, split_k=1) if present,
    else the first valid config."""
    for c in configs:
        if (
            c["tile_n"] == 256
            and c["tile_k"] == 256
            and c["n_warp"] == 4
            and c["num_buffers"] == 2
            and c["split_k"] == 1
        ):
            return c
    return configs[0]


# ---------------------------------------------------------------------------
# Config CSV row (tuned_grouped_fmoe.csv schema + per-stage columns)
# ---------------------------------------------------------------------------
def _build_csv_row(
    m_cfg: dict,
    s1: dict,
    s2: dict,
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    activation: ActivationType,
    data_format: str,
    layout: str,
) -> dict:
    """Build one config row with shared M-tiling + per-stage gemm1/gemm2 tiling.

    ``grouped_moe_gfx1250._find_grouped_config`` matches on the key columns and
    reads ``tile_m``/``m_warp`` (shared) plus ``tile_n1/tile_k1/n_warp1/
    num_buffers1/split_k1`` (stage1) and the ``*2`` set (stage2)."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    warp_tile_m = m_cfg["tile_m"] // m_cfg["m_warp"]
    max_m_raw = (token * topk + experts - 1) // experts
    max_m = max(
        warp_tile_m, ((max_m_raw + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
    )

    is_fp4 = data_format in ("a4w4", "fp4")
    q_a = "torch.float4_e2m1fn_x2" if is_fp4 else "torch.float8_e4m3fn"
    q_w = "torch.float4_e2m1fn_x2"
    act_str = (
        "ActivationType.Swiglu"
        if activation == ActivationType.Swiglu
        else "ActivationType.Silu"
    )
    gate_str = "GateMode.INTERLEAVE" if layout == "gugu" else "GateMode.SEPARATED"
    data_fmt = "fp4" if is_fp4 else "a8w4"

    gpu = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
    return {
        "gfx": get_gfx_runtime(),
        "cu_num": cu_num,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": experts,
        "topk": topk,
        "act_type": act_str,
        "dtype": "torch.bfloat16",
        "q_dtype_a": q_a,
        "q_dtype_w": q_w,
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "gate_mode": gate_str,
        "max_m": max_m,
        "tile_m": m_cfg["tile_m"],
        "m_warp": m_cfg["m_warp"],
        # stage1 (gemm1)
        "tile_n1": s1["tile_n"],
        "tile_k1": s1["tile_k"],
        "n_warp1": s1["n_warp"],
        "num_buffers1": s1["num_buffers"],
        "split_k1": s1["split_k"],
        # stage2 (gemm2)
        "tile_n2": s2["tile_n"],
        "tile_k2": s2["tile_k"],
        "n_warp2": s2["n_warp"],
        "num_buffers2": s2["num_buffers"],
        "split_k2": s2["split_k"],
        "grouped_persistent_m": 0,
        "persistent_workers": "",
        "stage1_weight_layout": layout,
        "kernelName1": f"grouped_gemm1_{data_fmt}_{layout}",
        "kernelName2": f"grouped_gemm2_{data_fmt}",
        "us": 0,
        "tflops": 0,
        "bw": 0,
    }


# ---------------------------------------------------------------------------
# Input data (shared across all candidates for a given token count)
# ---------------------------------------------------------------------------
def _prepare_data(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,
    layout: str,
):
    """Build mxfp4 weights + routing, dispatch-ready for ``fused_moe``.

    Mirrors ``_run_grouped_via_fused_moe`` in the test module (without the bias
    path / reference compare, which the sweep does not need)."""
    K = model_dim
    inter = inter_dim
    K_pack = K // 2
    inter_pack = inter // 2

    torch.manual_seed(0)
    w1_logical = _pattern_packed(experts, 2 * inter, K_pack)
    w2_logical = _pattern_packed(experts, K, inter_pack)
    w1_scale_raw = init_weight_scales(experts, 2 * inter, K // SCALE_BLOCK)
    w2_scale_raw = init_weight_scales(experts, K, inter // SCALE_BLOCK)
    hidden = (torch.randn((tokens, K)) * 0.5).to(torch.bfloat16)
    topk_id, topk_w = _make_topk(hidden, experts, topk)
    topk_w = topk_w.to(torch.bfloat16)

    if layout == "gugu":
        w1_phys = _gguu_to_gugu_rows(w1_logical)
        w1_scale_phys = _gguu_to_gugu_rows(w1_scale_raw)
    else:
        w1_phys = w1_logical
        w1_scale_phys = w1_scale_raw

    w1_grouped = shuffle_weight(w1_phys, layout=(16, 16))
    w2_grouped = shuffle_weight(w2_logical, layout=(16, 16))
    w1_scale = moe_shuffle_scale(w1_scale_phys.contiguous(), experts_cnt=experts)
    w2_scale = moe_shuffle_scale(w2_scale_raw.contiguous(), experts_cnt=experts)

    if data_format == "a4w4":
        w1_arg = w1_grouped.view(dtypes.fp4x2)
        w2_arg = w2_grouped.view(dtypes.fp4x2)
    else:  # a8w4
        w1_arg = w1_grouped
        w2_arg = w2_grouped

    return hidden, w1_arg, w2_arg, topk_w, topk_id, w1_scale, w2_scale


# ---------------------------------------------------------------------------
# Config injection + benchmark
# ---------------------------------------------------------------------------
def _clear_grouped_caches():
    import aiter.ops.flydsl.grouped_moe_gfx1250 as grouped_mod
    import aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 as gk

    cache = getattr(grouped_mod, "_GROUPED_CONFIG_CACHE", None)
    if cache is not None:
        cache.clear()
    find_fn = getattr(grouped_mod, "_find_grouped_config", None)
    if find_fn is not None and hasattr(find_fn, "cache_clear"):
        find_fn.cache_clear()
    for fn_name in (
        "compile_moe_grouped_gemm1_a8w4_masked",
        "compile_moe_grouped_gemm2_a8w4_masked",
    ):
        fn = getattr(gk, fn_name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def _write_config_csv(csv_row: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="tune_grouped_", delete=False, newline=""
    )
    writer = csv_mod.DictWriter(tmp, fieldnames=list(csv_row.keys()))
    writer.writeheader()
    writer.writerow(csv_row)
    tmp.close()
    return tmp.name


def _with_injected_config(csv_row: dict, fn):
    """Point ``AITER_CONFIG_GROUPED_FMOE`` at a one-row CSV, clear caches, run
    ``fn()``, then restore env + caches. Mirrors GroupedFmoeTuner._run_candidate.
    """
    path = _write_config_csv(csv_row)
    saved = {
        k: os.environ.get(k)
        for k in (
            "AITER_CONFIG_GROUPED_FMOE",
            "AITER_USE_GROUPED_GEMM",
            "AITER_FORCE_GFX1250",
        )
    }
    try:
        os.environ["AITER_CONFIG_GROUPED_FMOE"] = path
        os.environ["AITER_USE_GROUPED_GEMM"] = "1"
        os.environ["AITER_FORCE_GFX1250"] = "1"
        _clear_grouped_caches()
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _clear_grouped_caches()
        try:
            os.unlink(path)
        except OSError:
            pass


def _bench_config(
    csv_row: dict,
    *,
    data,
    activation: ActivationType,
    layout: str,
    warmup: int,
    iters: int,
) -> float:
    """Inject one config and time ``fused_moe`` end-to-end (mean us)."""
    from aiter.test_common import run_perftest

    hidden, w1_arg, w2_arg, topk_w, topk_id, w1_scale, w2_scale = data
    gate_mode = GateMode.INTERLEAVE if layout == "gugu" else GateMode.SEPARATED

    def _call():
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
            dtype=dtypes.bf16,
            gate_mode=gate_mode.value,
        )

    def _run():
        _call()
        torch.cuda.synchronize()
        _, us = run_perftest(_call, num_warmup=warmup, num_iters=iters)
        return round(float(us), 4)

    return _with_injected_config(csv_row, _run)


def _precision_check(
    csv_row: dict,
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,
    layout: str,
    activation: ActivationType,
) -> float:
    """Run the full grouped path with this config and return logits_diff vs the
    PyTorch reference (reuses the test module's end-to-end harness)."""

    def _run():
        out, ref, _ = _run_grouped_via_fused_moe(
            experts=experts,
            tokens=tokens,
            topk=topk,
            model_dim=model_dim,
            inter_dim=inter_dim,
            data_format=data_format,
            layout=layout,
            activation=activation,
            use_bias=True,
            bench=False,
        )
        return _logits_diff(out, ref.to(out.dtype))

    return _with_injected_config(csv_row, _run)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(rows: list):
    if not rows:
        return None
    try:
        import pandas as pd
    except ImportError:
        print("[tune summary] pandas not installed; raw rows:", flush=True)
        for r in rows:
            print(f"  {r}", flush=True)
        return rows
    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)
    print("\n[tune summary]\n" + table, flush=True)
    return df


def _fmt_stage(s: dict) -> str:
    return (
        f"tile_n={s['tile_n']:3d} tile_k={s['tile_k']:3d} "
        f"n_warp={s['n_warp']} nb={s['num_buffers']} sk={s['split_k']}"
    )


# ---------------------------------------------------------------------------
# Orchestration: separate stage1/stage2 tuning, combine on the shared M-tile
# ---------------------------------------------------------------------------
def _tune_one_token(args, tok: int, activation: ActivationType) -> dict | None:
    """Coordinate-descent over a shared (tile_m, m_warp): tune gemm1, then gemm2,
    then a combined run. Returns the best {m_cfg, s1, s2, us} or None."""
    m_configs = _generate_m_configs()
    data = _prepare_data(
        experts=args.experts,
        tokens=tok,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        data_format=args.data_format,
        layout=args.layout,
    )
    torch.cuda.synchronize()

    def _row(m_cfg, s1, s2):
        return _build_csv_row(
            m_cfg,
            s1,
            s2,
            token=tok,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            experts=args.experts,
            topk=args.topk,
            activation=activation,
            data_format=args.data_format,
            layout=args.layout,
        )

    def _bench(m_cfg, s1, s2):
        return _bench_config(
            _row(m_cfg, s1, s2),
            data=data,
            activation=activation,
            layout=args.layout,
            warmup=args.warmup,
            iters=args.iters,
        )

    best = None
    for m_cfg in m_configs:
        s1_cands = _generate_stage_configs(
            args.model_dim, m_cfg["m_warp"], args.data_format
        )
        s2_cands = _generate_stage_configs(
            args.inter_dim, m_cfg["m_warp"], args.data_format
        )
        if not s1_cands or not s2_cands:
            continue
        # Baseline (coordinate-descent hold-fixed point) comes from the full set;
        # an optional cap then bounds the swept set but always keeps the baseline.
        base_s1 = _baseline_stage(s1_cands)
        base_s2 = _baseline_stage(s2_cands)
        if args.max_stage_candidates:
            s1_cands = s1_cands[: args.max_stage_candidates]
            s2_cands = s2_cands[: args.max_stage_candidates]
            if base_s1 not in s1_cands:
                s1_cands = [base_s1] + s1_cands
            if base_s2 not in s2_cands:
                s2_cands = [base_s2] + s2_cands
        mtag = f"tile_m={m_cfg['tile_m']:3d} m_warp={m_cfg['m_warp']}"
        print(
            f"\n--- M-config {mtag}: "
            f"{len(s1_cands)} gemm1 + {len(s2_cands)} gemm2 candidates ---",
            flush=True,
        )

        # 1) tune gemm1 with gemm2 held at baseline
        best_s1, best_s1_us = base_s1, float("inf")
        for s1 in s1_cands:
            try:
                us = _bench(m_cfg, s1, base_s2)
            except Exception as exc:
                print(f"  gemm1 {_fmt_stage(s1)} => FAILED: {exc}", flush=True)
                continue
            mark = " *" if us < best_s1_us else ""
            print(f"  gemm1 {_fmt_stage(s1)} => {us:.2f} us{mark}", flush=True)
            if us < best_s1_us:
                best_s1, best_s1_us = s1, us
        if best_s1_us == float("inf"):
            print(f"  [skip] no working gemm1 for {mtag}", flush=True)
            continue

        # 2) tune gemm2 with gemm1 held at the stage1 winner
        best_s2, best_s2_us = base_s2, float("inf")
        for s2 in s2_cands:
            try:
                us = _bench(m_cfg, best_s1, s2)
            except Exception as exc:
                print(f"  gemm2 {_fmt_stage(s2)} => FAILED: {exc}", flush=True)
                continue
            mark = " *" if us < best_s2_us else ""
            print(f"  gemm2 {_fmt_stage(s2)} => {us:.2f} us{mark}", flush=True)
            if us < best_s2_us:
                best_s2, best_s2_us = s2, us
        if best_s2_us == float("inf"):
            print(f"  [skip] no working gemm2 for {mtag}", flush=True)
            continue

        combined_us = best_s2_us  # last run already used best_s1 + best_s2
        print(
            f"  [{mtag}] best combined => {combined_us:.2f} us "
            f"(gemm1 {_fmt_stage(best_s1)} | gemm2 {_fmt_stage(best_s2)})",
            flush=True,
        )
        if best is None or combined_us < best["us"]:
            best = {"m_cfg": m_cfg, "s1": best_s1, "s2": best_s2, "us": combined_us}

    if best is None:
        return None

    # 3) precision gate on the winner
    if not args.no_verify:
        try:
            ld = _precision_check(
                _row(best["m_cfg"], best["s1"], best["s2"]),
                experts=args.experts,
                tokens=tok,
                topk=args.topk,
                model_dim=args.model_dim,
                inter_dim=args.inter_dim,
                data_format=args.data_format,
                layout=args.layout,
                activation=activation,
            )
            best["logits_diff"] = ld
            passed = ld < LOGITS_DIFF_TOL
            best["passed"] = passed
            print(
                f"\n[tune] precision check tokens={tok}: logits_diff={ld:.4e} "
                f"(gate<{LOGITS_DIFF_TOL}) -> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            if not passed:
                print(
                    "[tune] WARNING: best config failed the accuracy gate; "
                    "not writing it. Re-run with a narrower search or inspect "
                    "the config.",
                    flush=True,
                )
        except Exception as exc:
            best["logits_diff"] = -1
            best["passed"] = False
            print(f"[tune] precision check raised: {exc}", flush=True)
    return best


def tune(args) -> None:
    activation = ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
    token_list = args.tokens if isinstance(args.tokens, list) else [args.tokens]

    m_configs = _generate_m_configs()
    print(
        f"[tune] {len(m_configs)} shared (tile_m, m_warp) M-configs for "
        f"model_dim={args.model_dim} inter_dim={args.inter_dim} "
        f"data_format={args.data_format}",
        flush=True,
    )

    if args.dry_run:
        total = 0
        for m_cfg in m_configs:
            s1 = _generate_stage_configs(
                args.model_dim, m_cfg["m_warp"], args.data_format
            )
            s2 = _generate_stage_configs(
                args.inter_dim, m_cfg["m_warp"], args.data_format
            )
            n1 = min(len(s1), args.max_stage_candidates) if args.max_stage_candidates else len(s1)
            n2 = min(len(s2), args.max_stage_candidates) if args.max_stage_candidates else len(s2)
            total += n1 + n2
            print(
                f"  tile_m={m_cfg['tile_m']:3d} m_warp={m_cfg['m_warp']}: "
                f"gemm1={len(s1)} gemm2={len(s2)} candidates "
                f"(coordinate-descent runs ~= {n1 + n2})",
                flush=True,
            )
        print(
            f"\n[tune] ~{total} benchmarked runs per token count "
            f"(x{len(token_list)} token counts). Trim the TUNE_* lists or pass "
            f"--max-stage-candidates to reduce.",
            flush=True,
        )
        return

    summary_rows = []
    best_per_token = {}
    for tok in token_list:
        print(f"\n===== tuning tokens={tok} =====", flush=True)
        best = _tune_one_token(args, tok, activation)
        if best is None:
            print(f"[tune] all candidates failed for tokens={tok}", flush=True)
            continue
        print(
            f"\n[tune] BEST tokens={tok}: "
            f"tile_m={best['m_cfg']['tile_m']} m_warp={best['m_cfg']['m_warp']} | "
            f"gemm1[{_fmt_stage(best['s1'])}] | gemm2[{_fmt_stage(best['s2'])}] "
            f"=> {best['us']:.2f} us",
            flush=True,
        )
        if args.no_verify or best.get("passed", True):
            best_per_token[tok] = best
        summary_rows.append(
            {
                "tokens": tok,
                "tile_m": best["m_cfg"]["tile_m"],
                "m_warp": best["m_cfg"]["m_warp"],
                "g1_tile_n": best["s1"]["tile_n"],
                "g1_tile_k": best["s1"]["tile_k"],
                "g1_n_warp": best["s1"]["n_warp"],
                "g1_nb": best["s1"]["num_buffers"],
                "g1_sk": best["s1"]["split_k"],
                "g2_tile_n": best["s2"]["tile_n"],
                "g2_tile_k": best["s2"]["tile_k"],
                "g2_n_warp": best["s2"]["n_warp"],
                "g2_nb": best["s2"]["num_buffers"],
                "g2_sk": best["s2"]["split_k"],
                "us": best["us"],
                "logits_diff": best.get("logits_diff"),
                "pass": best.get("passed"),
            }
        )

    summarize(summary_rows)

    if args.tune_output and best_per_token:
        csv_rows = []
        for tok, best in sorted(best_per_token.items()):
            row = _build_csv_row(
                best["m_cfg"],
                best["s1"],
                best["s2"],
                token=tok,
                model_dim=args.model_dim,
                inter_dim=args.inter_dim,
                experts=args.experts,
                topk=args.topk,
                activation=activation,
                data_format=args.data_format,
                layout=args.layout,
            )
            row["us"] = best["us"]
            csv_rows.append(row)
        with open(args.tune_output, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(
            f"[tune] wrote {len(csv_rows)} best config(s) to {args.tune_output}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument(
        "--layout",
        choices=("gguu", "gugu"),
        default="gguu",
        help="stage1 weight physical layout. gguu pairs with "
        "GateMode.SEPARATED (default), gugu with INTERLEAVE.",
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[64],
        metavar="N",
        help="one or more token counts; tuned independently, e.g. --tokens 1 8 64",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument(
        "--act",
        choices=("silu", "swiglu"),
        default="swiglu",
        help="stage1 activation",
    )
    parser.add_argument(
        "--tune-output",
        default="",
        metavar="CSV",
        help="write the best per-stage config per token count to this CSV "
        "(tuned_grouped_fmoe.csv schema + tile_*1/tile_*2 columns).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the end-to-end accuracy (logits_diff) gate on the winner.",
    )
    parser.add_argument(
        "--max-stage-candidates",
        type=int,
        default=0,
        metavar="N",
        help="cap the number of per-stage candidates benchmarked per M-config "
        "(0 = no cap). Useful to bound tuning time on the full search space.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list per-stage candidate counts and exit (no GPU; any arch).",
    )
    args = parser.parse_args()

    # --dry-run only enumerates candidates: no GPU, any arch. Everything else
    # needs real gfx1250 hardware.
    if not args.dry_run and not is_gfx1250():
        print("skipping: requires gfx1250")
        sys.exit(0)

    if args.model_dim < 512 or args.inter_dim < 512:
        raise SystemExit(
            f"model_dim ({args.model_dim}) and inter_dim ({args.inter_dim}) must be "
            "at least 512 for the grouped GEMM kernels (tile_k=256 requires at "
            "least two K tiles)."
        )

    set_data_format(args.data_format)
    logger.info("grouped GEMM tuner: data_format=%s", args.data_format)
    tune(args)


if __name__ == "__main__":
    main()
