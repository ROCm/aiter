# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tile tuner for the FlyDSL implicit-GEMM conv3d.

Follows the csrc tuner pattern (like gemm_a16w16, ck_gemm_a8w8): read shapes
from an untuned CSV, sweep the launch configs ``conv3d_policy`` enumerates for
each shape, and write the winner to a checked-in tuned CSV that
``conv_kernels._lookup_tuned_tile`` reads at runtime.

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

Write winners into the per-model file, not the header-only canonical
``aiter/configs/bf16_tuned_conv3d.csv`` (runtime merges model_configs/).

Model tables are per model, so -i and -o name one::

    python3 csrc/flydsl_conv3d/conv3d_tune.py \\
        -i aiter/configs/model_configs/qwenimage_vae_bf16_untuned_conv3d.csv \\
        -o aiter/configs/model_configs/qwenimage_vae_bf16_tuned_conv3d.csv

    python3 csrc/flydsl_conv3d/conv3d_tune.py \\
        -i aiter/configs/model_configs/wan21_vae_bf16_untuned_conv3d.csv \\
        -o aiter/configs/model_configs/wan21_vae_bf16_tuned_conv3d.csv
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
    _is_matmul_fast_path,
    _pad_channels,
    _parse_tuned_bool,
)
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


def generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias, seed=0, device=None):
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    x = torch.randn((n, c, d, h, w), device=device, dtype=dtypes.bf16)
    weight = torch.randn((k, c // groups, kt, kh, kw), device=device, dtype=dtypes.bf16)
    bias = torch.randn((k,), device=device, dtype=dtypes.fp32) if has_bias else None
    return {"x": x, "weight": weight, "bias": bias}


def run_flydsl_conv3d(x, weight, bias, params, tile, wgm, splitk):
    # splitk is passed rather than left to the dispatch to re-derive: the value
    # the caller recorded is the value that runs, so the CSV's splitK column
    # describes a measurement instead of a second derivation that happens to
    # agree. The AOT pass compiles against that column, so a drift between the
    # two would ship an artifact for a split nobody timed.
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
        # Zero, not the common 0.05. The reference here is bf16 and so shares
        # the kernel's rounding regime (see RTOL/ATOL), which puts every correct
        # candidate at err_ratio 0 -- a nonzero one is a config that computes the
        # wrong thing, most often a boundary tile it does not write. At 0.05 such
        # a candidate is still eligible to win, and since the op test's own bar
        # (ERR_TOL in test_flydsl_conv_implicit.py) is zero mismatched elements,
        # the tuner would be writing rows that test then fails. --errRatio still
        # raises it for a deliberate investigation.
        "errRatio": 0.0,
    }

    def get_cu_num(self):
        """The CU count the tuned rows are stamped with.

        ``chip_info.get_cu_num()``, not ``TunerCommon``'s
        ``torch.cuda.get_device_properties().multi_processor_count``:
        ``_lookup_tuned_tile`` keys the runtime lookup on the former, so a row
        written under the latter is one the runtime cannot find. They differ only
        under ``CU_NUM`` or a CU partition -- and where they do, every shape in
        the table misses at once and silently falls back to the heuristic tile.

        Also what ``conv3d_policy`` enumerates against here, and what
        ``conv_kernels._num_cu`` sizes the split-K and tile heuristics by, so the
        candidate set and the shipped default agree with the runtime's own view.
        """
        from aiter.jit.utils.chip_info import get_cu_num as _chip_get_cu_num

        return _chip_get_cu_num()

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

    def pre_process(self, args):
        """Load untuned shapes, stamp the device keys, drop already-tuned rows."""
        # sortResults reorders against this file, and by then untunedf has had
        # the already-tuned rows dropped, so keep the path rather than the frame.
        self._untune_file = args.untune_file
        if args.all:
            self.get_retune_gemm_list(args)
            self._drop_matmul_fast_path()
            return
        self.untunedf = self.get_untuned_gemm_list(args.untune_file)
        self.untunedf["gfx"] = self.get_gfx()
        self.untunedf["cu_num"] = self.get_cu_num()
        self.untunedf = self.untunedf[self.keys]
        self._drop_matmul_fast_path()
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
        from aiter.ops.flydsl.conv_kernels import _resolve_splitk

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

        # crs is built from the *padded* per-group channel count, matching what
        # the kernel computes; splitK divisibility depends on it.
        cgp = _pad_channels(c // groups)
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
        """Leave the tuned table in the untuned table's row order.

        The untuned tables list a model's shapes in encode->decode call order;
        sorting the winners by key throws that away and makes every re-tune
        reshuffle the file. Rank by the untuned row instead. Anything with no
        untuned counterpart keeps the base key order, after the rows that have
        one, so this only ever reorders and never drops.
        """
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
        conv_kernels._TUNED_LOOKUP_LOGGED.clear()

    @staticmethod
    def _ramp_clocks(seconds=2.0):
        """Hold the GPU busy until the clocks settle, before anything is timed.

        The compare gate reads the pre-tune benchmark on a device that has been
        idle and the post-tune one right after a sweep has hammered it, so
        without this the two halves are measured at different clock states and
        every verdict carries that bias. Observed at up to 2.1x on the same
        config and the same shape, which is far larger than the 3% the gate
        decides on.
        """
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
        "bf16_tuned_conv3d",
        KEYS,
        RESULT_LIST,
        description="FlyDSL implicit-GEMM conv3d bf16 tile tuner",
    )
    args = tuner.parse_args()
    # tune_summary() already exits non-zero when a shape failed or went untuned.
    tuner.run(args, False)
