# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tile tuner for FlyDSL conv3d.

Reads an untuned CSV, sweeps ``conv3d_policy`` configs, writes winners to a
per-model tuned CSV. One backend (``libtype=flydsl``), explicit tile columns
(not solidx). Pass ``-i``/``-o``; the canonical pair is header-only.

    python3 csrc/flydsl_conv3d/conv3d_tune.py \\
        -i aiter/configs/model_configs/qwenimage_vae_bf16_untuned_conv3d.csv \\
        -o aiter/configs/model_configs/qwenimage_vae_bf16_tuned_conv3d.csv
"""

import os
import time
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_CONV3D_BF16
from aiter.ops.flydsl import flydsl_conv_implicit
from aiter.ops.flydsl.conv3d_policy import (
    get_flydsl_conv3d_configs,
    tile_kernel_name,
)
from aiter.ops.flydsl.conv_kernels import (
    LIBTYPE_FLYDSL,
    TUNED_DEVICE_COLUMNS,
    TUNED_KEY_COLUMNS,
    TUNED_LIBTYPE_COLUMN,
    TUNED_RESULT_COLUMNS,
    _check_supported_arch,
    _is_matmul_fast_path,
    _pad_channels,
    _parse_tuned_bool,
)
from aiter.ops.flydsl.kernels.conv.conv3d_gfx950_utils import out_extent
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# The runtime lookup's key columns are the tuning key, and the untuned CSV
# header must equal them; TunerCommon prefixes the device columns on top.
SHAPE_KEYS = list(TUNED_KEY_COLUMNS)
KEYS = [*TUNED_DEVICE_COLUMNS, *SHAPE_KEYS]

RESULT_LIST = [
    # Ahead of the config columns it qualifies, where the GEMM tables put it.
    # This tuner only enumerates FlyDSL candidates, so it always writes that.
    TUNED_LIBTYPE_COLUMN,
    *TUNED_RESULT_COLUMNS,
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]

# Readings per shape in the compare benchmark; the gate keeps the fastest.
RUN_CONFIG_REPS = 3

# Same tolerance as op_tests/test_flydsl_conv_implicit.py. The reference is bf16
# rather than fp32 on purpose: the tuner needs to catch a config that computes
# the wrong thing, not to measure bf16 rounding, and a matching rounding regime
# keeps err_ratio at ~0 for every correct candidate.
RTOL = ATOL = 2e-2


# Taken from the kernel rather than restated: this is what decides npq, and a
# tuner that sized the GEMM by its own copy would report the wrong M and hand
# conv3d_policy the wrong shape to enumerate against.
_out_extent = out_extent


def _row_params(row):
    """Normalize one CSV row into the kwargs conv3d_implicit takes."""
    return {
        "stride": (int(row["stride_d"]), int(row["stride_h"]), int(row["stride_w"])),
        "padding": (int(row["pad_d"]), int(row["pad_h"]), int(row["pad_w"])),
        "dilation": (int(row["dil_d"]), int(row["dil_h"]), int(row["dil_w"])),
        "groups": int(row["groups"]),
    }


def generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias, seed=0, device=None):
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    x = torch.randn((n, c, d, h, w), device=device, dtype=dtypes.bf16)
    weight = torch.randn((k, c // groups, kt, kh, kw), device=device, dtype=dtypes.bf16)
    bias = torch.randn((k,), device=device, dtype=dtypes.fp32) if has_bias else None
    return {"x": x, "weight": weight, "bias": bias}


def run_flydsl_conv3d(x, weight, bias, params, tile, wgm, splitk):
    # splitk is pinned rather than re-derived by the dispatch, so the CSV column
    # is the value that was timed and AOT compiles the artifact the runtime asks
    # for. NCDHW in and out, so the timing includes the pre-transpose the model
    # also pays; see Conv3dTuner.calculate.
    return flydsl_conv_implicit(
        x, weight, bias=bias, tile=tile, wgm=wgm, splitk=splitk, **params
    )


def conv3d_ref(x, weight, bias, params):
    ref_bias = bias.to(x.dtype) if bias is not None else None
    return F.conv3d(x, weight, bias=ref_bias, **params)


def _shape_key(row):
    """One row's shape identity, normalized so both CSVs hash the same way."""
    return tuple(
        _parse_tuned_bool(v) if c == "bias" else int(v) for c, v in zip(SHAPE_KEYS, row)
    )


class Conv3dTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "sort": True,
        # The canonical pair, as every other tuner defaults to. Both ship
        # header-only, so a run without -i finds no shapes rather than tuning
        # someone else's; a model's own table is passed explicitly.
        "untune_file": "aiter/configs/bf16_untuned_conv3d.csv",
        "tune_file": f"{AITER_CONFIG_CONV3D_BF16}",
        "config_env_name": "AITER_CONFIG_CONV3D_BF16",
        # Zero: a nonzero err_ratio is a wrong config, and the op test bars at 0.
        "errRatio": 0.0,
    }

    def get_cu_num(self):
        """CU stamp for tuned rows: ``chip_info.get_cu_num()``, matching runtime lookup."""
        from aiter.jit.utils.chip_info import get_cu_num as _chip_get_cu_num

        return _chip_get_cu_num()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--max_configs",
            type=int,
            default=96,
            help="cap on enumerated candidates per shape (baseline tiles and the "
            "tile the shape would borrow are always added on top, so the real "
            "count is slightly higher)",
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

    def _drop_matmul_fast_path(self):
        if self.untunedf is None or self.untunedf.empty:
            return
        skip = self.untunedf.apply(_is_matmul_fast_path, axis=1)
        n_matmul = int(skip.sum())
        if n_matmul:
            logger.info(
                f"skipping {n_matmul} 1x1 stride-1 pad-0 shapes "
                "(torch.matmul fast path; no kernel to tune)"
            )
            self.untunedf = self.untunedf[~skip].reset_index(drop=True)

    def _check_shapes_expressible(self):
        """Reject a row no candidate could run, naming the row rather than the task."""
        if self.untunedf is None or self.untunedf.empty:
            return
        bad = []
        suspect = []
        for _, row in self.untunedf.iterrows():
            keys = tuple(row[k] for k in self.keys)
            shape = ", ".join(f"{c}={row[c]}" for c in SHAPE_KEYS)
            # Ahead of the extents: _gemm_dims divides by the stride, and a
            # negative pad or zero dilation reaches _conv3d_impl's asserts intact.
            p = _row_params(dict(zip(self.keys, keys)))
            if min(p["stride"]) < 1 or min(p["dilation"]) < 1 or min(p["padding"]) < 0:
                bad.append(
                    f"  {shape} -> stride {p['stride']} and dilation "
                    f"{p['dilation']} must be >= 1, padding {p['padding']} >= 0"
                )
                continue
            try:
                _parse_tuned_bool(row["bias"])
            except ValueError as exc:
                bad.append(f"  {shape} -> bias: {exc}")
                continue
            _, _, _, extents = self._gemm_dims(keys)
            if min(extents) < 1:
                bad.append(
                    f"  {shape} -> output {extents}: the dilated filter is larger "
                    f"than the padded input"
                )
            # Legal, so only warned: a 2D conv written as kT=1, D=1 with pad_d
            # left over tunes a depth-(1 + 2*pad_d) problem whose padded slices
            # are bias only. Not rejected, because the row has to match what the
            # model really calls with to be looked up at all.
            elif int(row["kT"]) == 1 and int(row["D"]) == 1 and int(row["pad_d"]) > 0:
                suspect.append(f"  {shape} -> output depth {extents[0]}")
        if suspect:
            logger.warning(
                "row(s) with kT=1, D=1 but pad_d>0 tune a depth > 1 problem; "
                "clear pad_d if a 2D conv was meant:\n" + "\n".join(suspect)
            )
        if bad:
            raise ValueError(
                "untuned CSV has row(s) this tuner cannot express:\n" + "\n".join(bad)
            )

    def pre_process(self, args):
        """Load untuned shapes, stamp the device keys, drop already-tuned rows."""
        _check_supported_arch()
        self._untune_file = args.untune_file
        if args.all:
            self.get_retune_gemm_list(args)
            self._drop_matmul_fast_path()
            self._check_shapes_expressible()
            return
        self.untunedf = self.get_untuned_gemm_list(args.untune_file)
        self.untunedf["gfx"] = self.get_gfx()
        self.untunedf["cu_num"] = self.get_cu_num()
        self.untunedf = self.untunedf[self.keys]
        self._drop_matmul_fast_path()
        self._check_shapes_expressible()
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
        from aiter.ops.flydsl.conv_kernels import _borrow_tuned_tile, _resolve_splitk

        kv = dict(zip(self.keys, keys))
        n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
        k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
        groups = int(kv["groups"])
        has_bias = _parse_tuned_bool(kv["bias"])
        params = _row_params(kv)
        m_gemm, n_gemm, _k_gemm, _ = self._gemm_dims(keys)

        configs = get_flydsl_conv3d_configs(
            m_gemm, n_gemm, groups, self.get_cu_num(), max_configs=max_configs
        )
        # BASELINE_TILES covers the heuristic default but not a borrowed one: a
        # tile tuned at another resolution of this layer can fall outside this
        # npq's max_configs cut, and the runtime serves it until this row exists.
        borrowed = _borrow_tuned_tile(
            (self.get_gfx(), self.get_cu_num()),
            _shape_key(keys[len(TUNED_DEVICE_COLUMNS) :]),
        )
        if borrowed is not None and (*borrowed[0], borrowed[1]) not in configs:
            configs.append((*borrowed[0], borrowed[1]))

        # crs is built from the *padded* per-group channel count, matching what
        # the kernel computes; splitK divisibility depends on it.
        cgp = _pad_channels(c // groups)
        crs = cgp * kt * kh * kw

        tasks = []
        for tile_m, tile_n, wave_m, wave_n, wgm in configs:
            tile = (tile_m, tile_n, wave_m, wave_n)
            # splitK is derived, not swept. A split only buys occupancy where
            # the M/N grid alone cannot fill the device, and measured against
            # the heuristic's own 3/4*CU bar that is 1 of the 76 shipped rows --
            # the rest would spend tuning time confirming splitK=1 while paying
            # an fp32 staging buffer and an atomic reduce to do it. It is still
            # pinned rather than re-derived at dispatch, so the column records
            # the value that ran and AOT compiles the artifact the runtime asks
            # for.
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
                    (["x", "weight", "bias"], params, tile, wgm, sk),
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
        """TFLOPS from implicit GEMM dims; bandwidth from tensor bytes (im2col reuse).

        Both are derived from an end-to-end time, so both are diluted by the
        NCDHW->NDHWC pre-transpose and the weight repack the entry point runs
        before the kernel -- a constant per shape, which leaves the ranking
        alone but makes these two columns a floor rather than a kernel figure.
        Do not compare them against another implementation's kernel-only
        numbers.
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
                    TUNED_LIBTYPE_COLUMN: LIBTYPE_FLYDSL,
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

    def sortResults(self, tune_file, issorted, values):
        """Keep tuned rows in the untuned CSV's order."""
        super().sortResults(tune_file, issorted, values)

        path = getattr(self, "_untune_file", None)
        if not path or not os.path.exists(path):
            return
        untunedf = pd.read_csv(path)
        untunedf.columns = untunedf.columns.str.strip()
        tunedf = pd.read_csv(tune_file)
        tunedf.columns = tunedf.columns.str.strip()
        if any(c not in df.columns for df in (untunedf, tunedf) for c in SHAPE_KEYS):
            return

        order = {
            _shape_key(row): i
            for i, row in enumerate(untunedf[SHAPE_KEYS].itertuples(index=False))
        }
        rank = [
            order.get(_shape_key(row), len(order))
            for row in tunedf[SHAPE_KEYS].itertuples(index=False)
        ]
        tunedf = (
            tunedf.assign(_rank=rank)
            .sort_values("_rank", kind="stable")
            .drop(columns="_rank")
        )
        tunedf.to_csv(tune_file, index=False)

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
        from aiter.ops.flydsl import conv_kernels

        conv_kernels._load_tuned_table.cache_clear()
        # Built from the table above, so it goes stale with it: a borrow after
        # the config swap would still pick among the old table's rows.
        conv_kernels._tuned_rows_by_layer.cache_clear()
        conv_kernels._TUNED_LOOKUP_LOGGED.clear()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        """Also drop what the swapped-in table left cached.

        The base restores the env var only, so with ``--batch`` below the shape
        count the next batch's pre-tune baseline would still read the previous
        batch's candidate table.
        """
        from aiter.jit.core import AITER_CONFIGS

        super()._restore_config_env(env_name, old_val, old_rebuild)
        AITER_CONFIGS.get_config_file.cache_clear()
        self._clear_op_caches()

    @staticmethod
    def _ramp_clocks(seconds=2.0):
        """Busy-wait so pre/post-tune clocks match."""
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        deadline = time.time() + seconds
        while time.time() < deadline:
            for _ in range(20):
                a = torch.mm(a, a).clamp_(-1.0, 1.0)
            torch.cuda.synchronize()
        del a
        torch.cuda.empty_cache()

    def run_config(self, args):
        """Benchmark the production entry point (no explicit tile) per shape."""
        from aiter.test_common import run_perftest

        self._clear_op_caches()
        self._ramp_clocks()
        results = []
        for _, row in self.untunedf.iterrows():
            keys = tuple(row[k] for k in self.keys)
            kv = dict(zip(self.keys, keys))
            n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
            k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
            groups = int(kv["groups"])
            has_bias = _parse_tuned_bool(kv["bias"])
            params = _row_params(kv)
            shape = f"{n}x{c}x{d}x{h}x{w}->{k} {kt}x{kh}x{kw}"
            try:
                data = generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias)
                # Best of a few, not a single reading: the gate's threshold is
                # 3%, so a one-shot measurement whose own spread exceeds that
                # decides by noise.
                out, us = None, float("inf")
                for _ in range(RUN_CONFIG_REPS):
                    out_i, us_i = run_perftest(
                        flydsl_conv_implicit,
                        data["x"],
                        data["weight"],
                        bias=data["bias"],
                        **params,
                    )
                    if us_i < us:
                        out, us = out_i, us_i
                ref = conv3d_ref(data["x"], data["weight"], data["bias"], params)
                ok = torch.allclose(out, ref, rtol=RTOL, atol=ATOL)
                # e2e only: run_perftest includes transpose and weight repack.
                results.append(
                    {
                        "shape": shape,
                        "e2e_us": round(us, 4),
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
        "bf16_tuned_conv3d",
        KEYS,
        RESULT_LIST,
        description="FlyDSL implicit-GEMM conv3d bf16 tile tuner",
    )
    args = tuner.parse_args()
    # tune_summary() already exits non-zero when a shape failed or went untuned.
    tuner.run(args, False)
