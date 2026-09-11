# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tile tuner for the FlyDSL implicit-GEMM conv3d.

Follows the csrc tuner pattern (like gemm_a16w16, ck_gemm_a8w8): read shapes
from an untuned CSV, sweep the launch configs ``conv3d_policy`` enumerates for
each shape, and write the winner to a checked-in tuned CSV that
``conv3d_implicit._lookup_tuned_tile`` reads at runtime.

Differences from the GEMM tuners, all forced by the operator rather than by
preference:

* **One backend.** There is no asm/CK/triton alternative for this kernel, so
  there is no ``libtype`` column and no per-backend task builder.
* **Explicit config columns instead of ``solidx``.** The whole launch config is
  five integers. GEMM stores an index because its asm/CK kernel instances are
  opaque; here an index would only add a way for a reordered candidate list to
  silently invalidate a checked-in CSV.
* **Subclasses ``TunerCommon``, not ``GemmCommonTuner``.** The latter hardcodes
  ``M``/``N``/``K`` in ``sort_keys`` and unpacks the first five key columns in
  ``calculate``; a 20-column conv key breaks both.

Usage::

    python3 csrc/flydsl_conv3d/conv3d_tune.py \\
        -i aiter/configs/conv3d_bf16_untuned.csv \\
        -o aiter/configs/conv3d_bf16_tuned.csv
"""

from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl import flydsl_conv_implicit
from aiter.ops.flydsl.conv3d_policy import (
    get_flydsl_conv3d_configs,
    tile_kernel_name,
)
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Matches conv3d_implicit.TUNED_KEY_COLUMNS, prefixed with the device keys that
# TunerCommon adds. The untuned CSV header must equal this list minus gfx/cu_num.
SHAPE_KEYS = [
    "N",
    "C",
    "D",
    "H",
    "W",
    "K",
    "kT",
    "kH",
    "kW",
    "stride_d",
    "stride_h",
    "stride_w",
    "pad_d",
    "pad_h",
    "pad_w",
    "dil_d",
    "dil_h",
    "dil_w",
    "groups",
    "bias",
]
KEYS = ["gfx", "cu_num", *SHAPE_KEYS]

RESULT_LIST = [
    "tile_m",
    "tile_n",
    "wave_m",
    "wave_n",
    "wgm",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]

# Same tolerance as tests/kernels/test_conv3d_implicit.py. The reference is bf16
# rather than fp32 on purpose: the tuner needs to catch a config that computes
# the wrong thing, not to measure bf16 rounding, and a matching rounding regime
# keeps err_ratio at ~0 for every correct candidate.
RTOL = ATOL = 2e-2


def _out_extent(size, pad, dil, kernel, stride):
    return (size + 2 * pad - (dil * (kernel - 1) + 1)) // stride + 1


def _row_params(row):
    """Normalize one CSV row into the kwargs conv3d_implicit takes."""
    return {
        "stride": (int(row["stride_d"]), int(row["stride_h"]), int(row["stride_w"])),
        "padding": (int(row["pad_d"]), int(row["pad_h"]), int(row["pad_w"])),
        "dilation": (int(row["dil_d"]), int(row["dil_h"]), int(row["dil_w"])),
        "groups": int(row["groups"]),
    }


def generate_data(
    n, c, d, h, w, k, kt, kh, kw, groups, has_bias, seed=0, device="cuda:0"
):
    torch.manual_seed(seed)
    x = torch.randn((n, c, d, h, w), device=device, dtype=dtypes.bf16)
    weight = torch.randn((k, c // groups, kt, kh, kw), device=device, dtype=dtypes.bf16)
    bias = torch.randn((k,), device=device, dtype=dtypes.fp32) if has_bias else None
    return {"x": x, "weight": weight, "bias": bias}


def run_flydsl_conv3d(x, weight, bias, params, tile, wgm):
    return flydsl_conv_implicit(x, weight, bias=bias, tile=tile, wgm=wgm, **params)


def conv3d_ref(x, weight, bias, params):
    ref_bias = bias.to(x.dtype) if bias is not None else None
    return F.conv3d(x, weight, bias=ref_bias, **params)


class Conv3dTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "sort": True,
        "untune_file": "aiter/configs/conv3d_bf16_untuned.csv",
        "tune_file": "aiter/configs/conv3d_bf16_tuned.csv",
        "config_env_name": "AITER_CONFIG_CONV3D_BF16",
    }

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--max_configs",
            type=int,
            default=96,
            help="cap on enumerated candidates per shape (baseline tiles are "
            "always added on top, so the real count is slightly higher)",
        )

    # -------------------------------------------------------------------
    # Shape bookkeeping
    # -------------------------------------------------------------------

    def _gemm_dims(self, keys):
        """(M, N, K) of the implicit GEMM this conv lowers to."""
        kv = dict(zip(self.keys, keys))
        do = _out_extent(
            int(kv["D"]),
            int(kv["pad_d"]),
            int(kv["dil_d"]),
            int(kv["kT"]),
            int(kv["stride_d"]),
        )
        ho = _out_extent(
            int(kv["H"]),
            int(kv["pad_h"]),
            int(kv["dil_h"]),
            int(kv["kH"]),
            int(kv["stride_h"]),
        )
        wo = _out_extent(
            int(kv["W"]),
            int(kv["pad_w"]),
            int(kv["dil_w"]),
            int(kv["kW"]),
            int(kv["stride_w"]),
        )
        groups = int(kv["groups"])
        m = int(kv["N"]) * do * ho * wo
        n = int(kv["K"]) // groups
        k = (int(kv["C"]) // groups) * int(kv["kT"]) * int(kv["kH"]) * int(kv["kW"])
        return m, n, k, (do, ho, wo)

    def pre_process(self, args):
        """Load untuned shapes, stamp the device keys, drop already-tuned rows."""
        if args.all:
            self.get_retune_gemm_list(args)
            return
        self.untunedf = self.get_untuned_gemm_list(args.untune_file)
        self.untunedf["gfx"] = self.get_gfx()
        self.untunedf["cu_num"] = self.get_cu_num()
        self.untunedf = self.untunedf[self.keys]
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if "gfx" not in self.tunedf.columns and "gfx" in self.untunedf.columns:
            self.tunedf.insert(0, "gfx", self.get_gfx())
        if len(self.tunedf) != 0:
            cols = self.untunedf.columns
            mask = self.untunedf.apply(tuple, axis=1).isin(
                self.tunedf[cols].apply(tuple, axis=1)
            )
            if args.verbose:
                logger.info("skipped tuned shapes:")
                print(self.untunedf[mask])
            self.untunedf = self.untunedf[~mask].reset_index(drop=True)

    # -------------------------------------------------------------------
    # Tuning
    # -------------------------------------------------------------------

    def _shape_tasks(self, keys, max_configs):
        from aiter.ops.flydsl.kernels.conv3d_implicit import _resolve_splitk

        kv = dict(zip(self.keys, keys))
        n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
        k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
        groups = int(kv["groups"])
        has_bias = str(kv["bias"]).lower() == "true"
        params = _row_params(kv)
        m_gemm, n_gemm, _k_gemm, _ = self._gemm_dims(keys)

        configs = get_flydsl_conv3d_configs(
            m_gemm, n_gemm, groups, get_cu_num(), max_configs=max_configs
        )

        # crs is built from the *padded* per-group channel count, matching what
        # the kernel computes; splitK divisibility depends on it.
        cgp = -(-(c // groups) // 8) * 8
        crs = cgp * kt * kh * kw

        tasks = []
        for tile_m, tile_n, wave_m, wave_n, wgm in configs:
            tile = (tile_m, tile_n, wave_m, wave_n)
            sk = _resolve_splitk(
                None, m_gemm, crs, k, torch.cuda.current_device(), tile, groups
            )
            info = (
                keys,
                tile_m,
                tile_n,
                wave_m,
                wave_n,
                wgm,
                sk,
                tile_kernel_name(tile_m, tile_n, wave_m, wave_n, wgm),
            )
            tasks.append(
                (
                    info,
                    generate_data,
                    (n, c, d, h, w, k, kt, kh, kw, groups, has_bias),
                    run_flydsl_conv3d,
                    (["x", "weight", "bias"], params, tile, wgm),
                    {},
                    conv3d_ref,
                    (["x", "weight", "bias"], params),
                    {},
                    None,
                    RTOL,
                    ATOL,
                    None,  # compare_fn
                    None,  # max_abs_delta
                    # No output_keys: conv3d_implicit allocates and returns its
                    # own output, so there is no caller-owned buffer to NaN-fill.
                    None,
                )
            )
        return tasks

    def tune(self, untunedf, tunedf, args):
        tasks = []
        in_datas = []
        for _, row in untunedf.iterrows():
            keys = tuple(row[k] for k in self.keys)
            shape_tasks = self._shape_tasks(keys, args.max_configs)
            if not shape_tasks:
                logger.warning(f"no legal candidate for {keys}")
                continue
            m, n, k, _ = self._gemm_dims(keys)
            logger.info(
                f"conv3d candidates for M={m}, N={n}, K={k}: {len(shape_tasks)}"
            )
            tasks.extend(shape_tasks)
            in_datas.append((len(shape_tasks), ()))
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            in_datas,
            args.mp,
            False,
            True,  # shape_grouped: keep one shape's candidates on one GPU
            args.errRatio,
            args.timeout,
            args.verbose,
        )

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------

    def getKernelName(self, kernel_id):
        return kernel_id if isinstance(kernel_id, str) else str(kernel_id)

    def calculate(self, results, bpes=(2, 2, 2)):
        """TFLOPS and the bytes the conv actually moves.

        FLOP is the GEMM formula unchanged, because implicit GEMM's dimensions
        are exactly (N*Do*Ho*Wo, K/groups, C/groups*kT*kH*kW). Bandwidth is not:
        the GEMM form assumes A is read once, while im2col gathers each input
        element up to kT*kH*kW times. Counting tensor bytes instead keeps this
        column comparable with the shape tables, not with the GEMM tuners'.
        """
        info, time, _err = results
        if time == self.INVALID_TIME or time in (0, self.INF_TIME):
            return 0, 0
        keys = info[0]
        m, n, k, (do, ho, wo) = self._gemm_dims(keys)
        kv = dict(zip(self.keys, keys))
        tflops = round(m * n * k * 2 / (time * 1e6), 2)

        in_bpe, w_bpe, out_bpe = bpes
        x_elems = (
            int(kv["N"]) * int(kv["C"]) * int(kv["D"]) * int(kv["H"]) * int(kv["W"])
        )
        w_elems = (
            int(kv["K"])
            * (int(kv["C"]) // int(kv["groups"]))
            * int(kv["kT"])
            * int(kv["kH"])
            * int(kv["kW"])
        )
        y_elems = int(kv["N"]) * int(kv["K"]) * do * ho * wo
        moved = x_elems * in_bpe + w_elems * w_bpe + y_elems * out_bpe
        bw = round(moved / (time * 1e-6) / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, tile_m, tile_n, wave_m, wave_n, wgm, splitk, kernel_name = info
            tflops, bw = self.calculate(el)
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "wave_m": wave_m,
                    "wave_n": wave_n,
                    "wgm": wgm,
                    "splitK": splitk,
                    "us": time,
                    "kernelName": kernel_name,
                    "err_ratio": err_ratio,
                    "tflops": tflops,
                    "bw": bw,
                }
            )
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(dict(zip(self.keys, keys))).strip('{}')} "
                    f"is tile=({tile_m},{tile_n},{wave_m},{wave_n}) wgm={wgm} "
                    f"splitK={splitk}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    def result_to_csv(self, resultdf, file, concat=False):
        old_df = self.get_tuned_gemm_list(file)
        bad = (resultdf["us"] == self.INVALID_TIME) | (resultdf["us"] == self.INF_TIME)
        self.failed = pd.concat([self.failed, resultdf[bad]], ignore_index=True)
        self.success = pd.concat([self.success, resultdf[~bad]], ignore_index=True)
        good = resultdf[~bad]
        if not concat:
            out = self.update_tunedf(old_df, good)
        else:
            out = pd.concat([old_df, good], ignore_index=True)
        out.to_csv(file, index=False, na_rep="Null")

    def _clear_op_caches(self):
        from aiter.ops.flydsl.kernels import conv3d_implicit

        conv3d_implicit._load_tuned_table.cache_clear()

    def run_config(self, args):
        """Benchmark the production entry point (no explicit tile) per shape."""
        from aiter.test_common import run_perftest

        self._clear_op_caches()
        results = []
        for _, row in self.untunedf.iterrows():
            keys = tuple(row[k] for k in self.keys)
            kv = dict(zip(self.keys, keys))
            n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
            k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
            groups = int(kv["groups"])
            has_bias = str(kv["bias"]).lower() == "true"
            params = _row_params(kv)
            shape = f"{n}x{c}x{d}x{h}x{w}->{k} {kt}x{kh}x{kw}"
            try:
                data = generate_data(
                    n, c, d, h, w, k, kt, kh, kw, groups, has_bias, device="cuda:0"
                )
                out, us = run_perftest(
                    flydsl_conv_implicit,
                    data["x"],
                    data["weight"],
                    bias=data["bias"],
                    **params,
                )
                ref = conv3d_ref(data["x"], data["weight"], data["bias"], params)
                ok = torch.allclose(out, ref, rtol=RTOL, atol=ATOL)
                results.append(
                    {
                        "shape": shape,
                        "e2e_us": round(us, 4),
                        "kernel_us": round(us, 4),
                        "status": "ok" if ok else "mismatch",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {"shape": shape, "e2e_us": -1, "status": f"error: {exc}"}
                )
        return results


if __name__ == "__main__":
    tuner = Conv3dTuner(
        "conv3d_bf16_tuned",
        KEYS,
        RESULT_LIST,
        description="FlyDSL implicit-GEMM conv3d bf16 tile tuner",
    )
    args = tuner.parse_args()
    # tune_summary() already exits non-zero when a shape failed or went untuned.
    tuner.run(args, False)
