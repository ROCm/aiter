#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""conv3d_implicit with and without its tuned tile table, same session.

``bench_three_way.py`` compares the conv against two GEMM arms; this answers the
narrower question of what the offline tuner
(``csrc/flydsl_conv3d/conv3d_tune.py``) changed, by measuring only the conv arm
with the config table enabled and disabled.

Both arms share ``bench_three_way``'s shape list and its ``bench()``: the
``conv3d_implicit_kernel`` on-device time, rotated over ``NROT`` input buffers
so the L2 is cold, best of three reps. That is the same figure of merit the
three-way plot reports, so the ratios here compose with the ones stored in
``three_way.json`` -- the hipBLASLt arm is untouched by this change.

The two arms are interleaved per shape rather than run as two passes, because a
clock or thermal drift between passes would otherwise land entirely on one of
them. The table is toggled in-process by repointing the config env var and
dropping the two caches that sit in front of it.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/bench_conv_tuned_ab.py
"""

import json
import os
import tempfile
from pathlib import Path

import torch
from bench_three_way import DEV, NROT, SHAPES, bench, gemm_shape

from aiter.jit import core as jit_core
from aiter.ops.flydsl.kernels import conv3d_implicit as ci

HERE = Path(__file__).resolve().parent
ENV = "AITER_CONFIG_CONV3D_BF16"
REAL_TABLE = (
    Path(jit_core.AITER_ROOT_DIR) / "aiter" / "configs" / "conv3d_bf16_tuned.csv"
)


def _select_table(path):
    """Point the runtime at ``path`` and drop both layers of caching."""
    os.environ[ENV] = str(path)
    type(jit_core.AITER_CONFIGS).get_config_file.cache_clear()
    ci._load_tuned_table.cache_clear()


def _empty_table(tmpdir):
    """A header-only CSV, so every lookup misses and the heuristic runs."""
    path = Path(tmpdir) / "conv3d_bf16_tuned.csv"
    path.write_text(
        ",".join(
            (
                *ci.TUNED_KEY_COLUMNS,
                "gfx",
                "cu_num",
                "tile_m",
                "tile_n",
                "wave_m",
                "wave_n",
                "wgm",
            )
        )
        + "\n"
    )
    return path


def _hit(cin, cout, hin, stride, pad):
    """Does the enabled table actually cover this shape? Reported, not assumed."""
    key = (
        1,
        cin,
        1,
        hin,
        hin,
        cout,
        1,
        3,
        3,
        1,
        stride,
        stride,
        0,
        pad,
        pad,
        1,
        1,
        1,
        1,
        False,
    )
    return ci._lookup_tuned_tile(key, DEV)


def main():
    stored = {}
    three_way = HERE / "three_way.json"
    if three_way.exists():
        stored = {r["sid"]: r for r in json.loads(three_way.read_text())}

    with tempfile.TemporaryDirectory() as tmp:
        off_table = _empty_table(tmp)
        rows = []
        print(
            f"{'case':>20s} {'M':>9s} {'N':>5s} {'K':>6s} "
            f"{'启发式':>9s} {'查表':>9s} {'加速':>8s}  tile"
        )
        for sid, cin, cout, hin, stride, pad, freq in SHAPES:
            m, n, k = gemm_shape(cin, cout, hin, stride, pad)
            torch.manual_seed(0)
            pairs = [
                (
                    torch.randn((1, cin, hin, hin), device=DEV, dtype=torch.bfloat16),
                    torch.randn((cout, cin, 3, 3), device=DEV, dtype=torch.bfloat16),
                )
                for _ in range(NROT)
            ]
            try:
                _select_table(REAL_TABLE)
                tile = _hit(cin, cout, hin, stride, pad)

                offs, ons = [], []
                for _ in range(2):
                    _select_table(off_table)
                    offs.append(bench(pairs, stride, pad))
                    _select_table(REAL_TABLE)
                    ons.append(bench(pairs, stride, pad))
                t_off, t_on = min(offs), min(ons)
            finally:
                del pairs
                torch.cuda.empty_cache()

            rec = {
                "sid": sid,
                "cin": cin,
                "cout": cout,
                "hin": hin,
                "stride": stride,
                "freq": freq,
                "M": m,
                "N": n,
                "K": k,
                "conv_heuristic": t_off,
                "conv_tuned": t_on,
                "speedup": (t_off / t_on) if t_on else None,
                "tile": str(tile) if tile else None,
                "hip": stored.get(sid, {}).get("hip"),
            }
            rows.append(rec)
            print(
                f"{sid:>20s} {m:9d} {n:5d} {k:6d} {t_off:9.2f} {t_on:9.2f} "
                f"{t_off / t_on:7.3f}x  {tile if tile else '未收录'}",
                flush=True,
            )

    tw = sum(r["conv_heuristic"] * r["freq"] for r in rows)
    tt = sum(r["conv_tuned"] * r["freq"] for r in rows)
    print(f"\n按调用次数加权：启发式 {tw:.1f} us -> 查表 {tt:.1f} us = {tw / tt:.4f}x")
    hits = sum(r["tile"] is not None for r in rows)
    print(f"命中查表的形状：{hits}/{len(rows)}")

    if all(r["hip"] for r in rows):
        rh = sum(r["hip"] * r["freq"] for r in rows) / tw
        rt = sum(r["hip"] * r["freq"] for r in rows) / tt
        print(
            f"对 hipBLASLt 加权比（沿用 three_way.json 的 hip 臂）："
            f"{rh:.3f}x -> {rt:.3f}x"
        )

    out = HERE / "conv_tuned_ab.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
