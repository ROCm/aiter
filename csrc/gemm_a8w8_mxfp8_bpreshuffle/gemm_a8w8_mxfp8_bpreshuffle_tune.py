# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a8w8_mxfp8_bpreshuffle (FP8 GEMM, e8m0 microscales) on gfx950.

gfx950 runs two FlyDSL pipelines on unshuffled row-major scales (x_scale
[M, K/32], w_scale [N, K/32]): the mxpsh kernel and the 8-wave 256x256 one
(ids from 1_000_000). Every candidate of both that fits the shape is timed and
the fastest lands in the same table the gfx1250 rows live in, keyed by gfx and
w_scale_block (1x32: a shape list without the column is one):

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        -i aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv \\
        -o aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv

--mxscale_table tunes gemm_a8w8_blockscale_bpreshuffle's e8m0 table instead,
the only one the 32x32 (w_scale [N/32, K/32], as bmm's) and 128x128 blocks
(w_scale [N/128, K/128], x_scale 1x128 as its column-major bytes) have, and
the one its 1x32 calls read: the batched GEMM at B = 1 (bmm) joins the mxfp8
pipelines on every block (its candidates: the table's tuned bmm kernels for
M/2..2M, its heuristic pick, and their split-K neighbours), and a row's
kernelId names the backend that runs it, "mxfp8" or "bmm":

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        --mxscale_table \\
        -i <shapes with a w_scale_block column> \\
        -o aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv
"""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    MXFP8_ROW_PIPELINES,
    MXFP8_ROW_W32X32_PIPELINES,
    MXFP8_ROW_W128X128_PIPELINES,
)
from aiter.ops.gemm_op_common import with_mxscale_w_scale_block
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv"
# The block of a shape list without a w_scale_block column, and of a tuned
# table from before the column.
W_SCALE_BLOCK = "1x32"
PIPELINES_BY_W_SCALE_BLOCK = {
    "1x32": MXFP8_ROW_PIPELINES,
    "32x32": MXFP8_ROW_W32X32_PIPELINES,
    "128x128": MXFP8_ROW_W128X128_PIPELINES,
}


def generate_data(m, n, k, seed, w_scale_block=W_SCALE_BLOCK, device="cuda"):
    """fp8 operands with e8m0 scales around 2^-3..2^3, unshuffled, per
    w_scale_block: 1x32 x_scale [m, k/32] with a 1x32 [n, k/32] or 32x32
    [n/32, k/32] w_scale; or the blockscale pair, a 1x128 x_scale (a contiguous
    (m, k/128) holding the column-major [k/128, m] bytes) and a 128x128
    [n/128, k/128] w_scale."""
    blocks = {"1x32": (1, 32), "32x32": (32, 32), "128x128": (128, 128)}
    assert w_scale_block in blocks, f"no {w_scale_block} w_scale data"
    w_rows, kb = blocks[w_scale_block]
    torch.manual_seed(seed)
    x = (torch.randn((m, k), device=device) * 2).to(dtypes.fp8)
    weight = (torch.randn((n, k), device=device) * 2).to(dtypes.fp8)
    x_scale = torch.randint(124, 131, (m, k // kb), dtype=torch.uint8, device=device)
    w_scale = torch.randint(
        124, 131, (n // w_rows, k // kb), dtype=torch.uint8, device=device
    )

    def dequant(q, s, rows=1):
        s = torch.exp2(s.float() - 127).repeat_interleave(rows, 0)
        return q.to(dtypes.fp32) * s.repeat_interleave(kb, 1)

    # A 1x128 x_scale is passed as its column-major bytes.
    x_scale_arg = x_scale if kb == 32 else x_scale.T.contiguous().view(m, k // kb)
    return {
        "x": x,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        "x_scale": x_scale_arg.view(dtypes.fp8_e8m0),
        "w_scale": w_scale.view(dtypes.fp8_e8m0),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": dequant(x, x_scale),
        "w_deq": dequant(weight, w_scale, w_rows),
    }


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


def opus_kernel_name(kid, *, cu_num):
    from csrc.opus_gemm.opus_gemm_common import get_registry

    instance = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)[int(kid)]
    return instance.name


def opus_candidates(m, n, k, seeds=(), *, cu_num, top_geometries=4):
    from csrc.opus_gemm.opus_gemm_common import (
        a8w8_mxscale_bpreshuffle_tune_candidates,
        candidate_kids_for_shape,
        get_registry,
    )

    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    candidates = [
        (kid, registry[kid])
        for kid in candidate_kids_for_shape("gfx950", m, n, k, cu_num=cu_num)
    ]
    if not candidates:
        return []
    seed_kids = set(seeds) & {kid for kid, _ in candidates}
    candidate_count = len(candidates)
    candidates = a8w8_mxscale_bpreshuffle_tune_candidates(candidates, seed_kids)
    groups = defaultdict(list)
    for kid, instance in candidates:
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        geometry = (
            entry["pipeline"],
            instance.B_M,
            instance.B_N,
            instance.B_K,
            instance.T_M,
            instance.T_N,
        )
        groups[geometry].append(kid)
    by_pipeline = defaultdict(list)
    for geometry in groups:
        pipeline, bm, bn, _, _, _ = geometry
        tiles = math.ceil(m / bm) * math.ceil(n / bn)
        padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
        occupancy = tiles / max(cu_num, 1)
        cost = (
            padding
            - 1
            + 2 * max(0, 1 - occupancy)
            + 0.1 * (math.ceil(occupancy) / occupancy - 1)
            + 8 / bm
            + 8 / bn
        )
        by_pipeline[pipeline].append((cost, geometry))
    retained = {
        geometry
        for geometries in by_pipeline.values()
        for _, geometry in sorted(geometries)[:top_geometries]
    }
    legal = {kid for kid, _ in candidates}
    selected = {kid for geometry in retained for kid in groups[geometry]}
    selected.update(seed_kids & legal)
    print(
        f"[OPUS search] shape={(m, n, k)} legal={candidate_count} "
        f"geometries={len(retained)} candidates={len(selected)} seeds={len(seed_kids)}",
        flush=True,
    )
    return [(kid, registry[kid]) for kid in sorted(selected)]


def ensure_opus_kids_compiled(kids, *, cu_num):
    if not kids:
        return False
    from csrc.opus_gemm.opus_gemm_common import get_registry

    registry = get_registry("a8w8_blockscale_bpreshuffle", "gfx950", cu_num)
    missing = set(kids) - registry.keys()
    if missing:
        raise FileNotFoundError(
            f"OPUS COs are unavailable for kernel IDs {sorted(missing)}; "
            "run csrc/opus_gemm/gen_co/build_co.py --arch gfx950 first."
        )
    opus_dir = str(Path(__file__).resolve().parents[1] / "opus_gemm")
    added_path = opus_dir not in sys.path
    if added_path:
        sys.path.insert(0, opus_dir)
    try:
        from opus_gemm_tune import _ensure_kids_compiled

        return _ensure_kids_compiled(
            kids, family="a8w8_blockscale_bpreshuffle", gfx="gfx950", cu_num=cu_num
        )
    finally:
        if added_path:
            sys.path.remove(opus_dir)


def run_gemm_opus(x, weight_shuffle, x_scale, w_scale, out, kid):
    from aiter.ops.opus import opus_gemm

    return opus_gemm(
        x,
        weight_shuffle,
        out,
        kid=kid,
        family="a8w8_blockscale_bpreshuffle",
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    )


# The batched GEMM at B = 1 in --mxscale_table mode: its ids follow mxfp8's
# (kernelId records the backend in that table).
KERNEL_ID_BASE_BMM = 3_000_000
MXSCALE_KERNEL_ID = "mxfp8"
MXSCALE_BMM_KERNEL_ID = "bmm"


def _bmm_scale_args(w_scale_block):
    from aiter.ops.gemm_op_common import mxscale_bmm_groups_of_block

    rows, cols = mxscale_bmm_groups_of_block(w_scale_block)
    # generate_data's x_scale is as wide as the w_scale block; a 1x128 one
    # (blockscale) is passed as its column-major bytes.
    return {
        "x_scale_k": cols,
        "w_scale_n": rows,
        "w_scale_k": cols,
        "x_scale_transposed": cols == 128,
    }


def bmm_candidates(m, n, k, w_scale_block, seeds=()):
    """bmm kernelNames to time for a shape: ``seeds`` (tuned bmm rows of nearby
    M), the heuristic pick, and their split-K neighbours that run it. A
    1x128 x_scale is read row-major at M = 1, which a row may serve through
    padded M, so 128x128 candidates must run both layouts."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        bmm_candidate_names,
        bmm_config_runs,
        parse_bmm_kernel_name,
    )

    scale_args = _bmm_scale_args(w_scale_block)
    names = bmm_candidate_names(1, m, n, k, seeds, **scale_args)
    row_major = {**scale_args, "x_scale_transposed": False}
    out = []
    for name in names:
        cfg = parse_bmm_kernel_name(name)
        tiles = math.ceil(m / cfg["tile_m"]) * (n // cfg["tile_n"])
        if cfg["splits"] > 1 and tiles > (1 << 16):
            continue
        if scale_args["x_scale_transposed"] and not bmm_config_runs(
            1, n, k, row_major, cfg
        ):
            continue
        out.append(name)
    return out


def run_bmm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

    m, k = x.shape
    n = weight_shuffle.shape[0]
    run_bmm_a8w8_mxfp8(
        x.view(m, 1, k),
        weight_shuffle.view(1, n, k),
        x_scale.view(m, 1, x_scale.shape[-1]),
        w_scale.view(1, *w_scale.shape),
        out.view(m, 1, n),
        kernel_name=kernel_name,
        x_scale_transposed=k // x_scale.shape[-1] == 128 and m > 1,
    )
    return out


def run_gemm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    return run_gemm_a8w8_mxfp8_gfx950(
        x, weight_shuffle, x_scale, w_scale, out, kernel_name
    )


class GemmA8W8MXFP8BpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": TUNED_FILE,
        "untune_file": "aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE",
    }

    def getKernelName(self, kernelId):
        # Pipeline id ranges are disjoint (8-wave ids start at 1_000_000).
        for pipe in MXFP8_ROW_PIPELINES:
            if kernelId in pipe.kernels_list:
                return pipe.kernels_list[kernelId].name
        return None

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op._get_mxfp8_bpreshuffle_config.cache_clear()
        _op._load_mxfp8_bpreshuffle_tuned.cache_clear()
        getattr(
            _op, "_get_mxscale_bpreshuffle_config", _op.get_mxscale_bpreshuffle_config
        ).cache_clear()
        _op._load_mxscale_bpreshuffle_tuned.cache_clear()
        _op._mxscale_bpreshuffle_tuned_nk.cache_clear()
        _op._MXSCALE_UNTUNED_REPORTED.clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--mxscale_table",
            action="store_true",
            help="Tune gemm_a8w8_blockscale_bpreshuffle's e8m0 table: bmm (the "
            "batched GEMM at B = 1) joins on 1x32 / 32x32 / 128x128; OPUS JIT/CO "
            "joins on 128x128. FlyDSL kernelId names mxfp8 or bmm; "
            "OPUS IDs are numeric.",
        )
        self.parser.add_argument(
            "--libtype",
            choices=("both", "opus", "flydsl"),
            default="both",
            help="Backends to search; OPUS filters candidates before a single measurement pass "
            "and requires --mxscale_table and 128x128. "
            "Prebuild COs with csrc/opus_gemm/gen_co/build_co.py --arch gfx950.",
        )

    def pre_process(self, args):
        self._mxscale_table = bool(args.mxscale_table)
        if self._mxscale_table:
            self.ARG_DEFAULTS = {
                **type(self).ARG_DEFAULTS,
                "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            }
        elif args.libtype == "opus":
            self.parser.error("OPUS tuning requires --mxscale_table")
        super().pre_process(args)

    def _bmm_seeds(self):
        """{(N, K, w_scale_block): [(M, bmm kernelName)]} of the tuned
        gemm_a8w8_blockscale_bpreshuffle table, this arch and CU count."""
        from aiter.jit.core import AITER_CONFIGS

        path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
        try:
            df = with_mxscale_w_scale_block(pd.read_csv(path), path)
        except (OSError, pd.errors.EmptyDataError):
            return {}
        df = df[
            (df["gfx"] == self.get_gfx())
            & (df["cu_num"] == self.get_cu_num())
            & (df["kernelId"].astype(str) == MXSCALE_BMM_KERNEL_ID)
        ]
        seeds = {}
        for r in df.itertuples():
            key = (int(r.N), int(r.K), str(r.w_scale_block))
            seeds.setdefault(key, []).append((int(r.M), str(r.kernelName)))
        return seeds

    def get_untuned_gemm_list(self, untuned_gemm_file):
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        if "w_scale_block" not in df.columns:
            df["w_scale_block"] = W_SCALE_BLOCK
        df["w_scale_block"] = df["w_scale_block"].fillna(W_SCALE_BLOCK)
        unknown = set(df["w_scale_block"]) - set(PIPELINES_BY_W_SCALE_BLOCK)
        if unknown:
            raise ValueError(
                f"{untuned_gemm_file}: no gfx950 mxfp8 kernels tuned for w_scale "
                f"block {sorted(unknown)}; tunable: {sorted(PIPELINES_BY_W_SCALE_BLOCK)}"
            )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        df = with_mxscale_w_scale_block(df, tuned_gemm_file, default=W_SCALE_BLOCK)
        rest = [c for c in df.columns if c not in self.keys]
        return df[[c for c in self.keys if c in df.columns] + rest]

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import (
            gemm_a8w8_blockscale_bpreshuffle,
            gemm_a8w8_mxfp8_bpreshuffle,
        )
        from aiter.test_common import checkAllclose, run_perftest

        if getattr(self, "_mxscale_table", False) and "libtype" in self.untunedf:
            kids = {
                int(row.kernelId)
                for row in self.untunedf.itertuples()
                if row.libtype == "opus"
            }
            ensure_opus_kids_compiled(kids, cu_num=self.get_cu_num())
        op = (
            gemm_a8w8_blockscale_bpreshuffle
            if getattr(self, "_mxscale_table", False)
            else gemm_a8w8_mxfp8_bpreshuffle
        )
        results = []
        for i in range(len(self.untunedf)):
            row = self.untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            block = row.get("w_scale_block", W_SCALE_BLOCK)
            shape_str = f"({M}, {N}, {K}, {block})"
            allowed, allowed_desc = self._get_run_config_err_ratio_limit(row, args)
            try:
                d = generate_data(M, N, K, 0, block)
                out, us = run_perftest(
                    op,
                    d["x"],
                    d["weight_shuffle"],
                    d["x_scale"],
                    d["w_scale"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                err_ratio = checkAllclose(
                    out,
                    run_torch(d["x_deq"], d["w_deq"]),
                    msg=f"run_config {shape_str}",
                )
                status = (
                    "ok"
                    if err_ratio <= allowed
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        if gfx != "gfx950":
            logger.warning(
                f"gemm_a8w8_mxfp8_bpreshuffle tuner covers gfx950 only, got {gfx}"
            )
            return []
        cu_num = self.get_cu_num()
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_args = (["x_deq", "w_deq"], dtypes.bf16)
        run_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        seed = 0

        task = []
        self._mxscale_table = bool(getattr(args, "mxscale_table", False))
        coarse = sorted(set(untunedf["w_scale_block"]) - {W_SCALE_BLOCK})
        if coarse and not self._mxscale_table:
            raise ValueError(
                f"w_scale block {coarse} is tuned for gemm_a8w8_blockscale_"
                f"bpreshuffle only: pass --mxscale_table"
            )
        seeds = self._bmm_seeds() if self._mxscale_table else {}
        opus_by_shape = {}
        if args.libtype != "flydsl" and self._mxscale_table:
            if args.libtype == "opus" and set(untunedf.w_scale_block) != {"128x128"}:
                raise ValueError("OPUS supports A 1x128 / B 128x128 scales only")
            opus_seeds = defaultdict(set)
            if "128x128" in set(untunedf.w_scale_block):
                from aiter.jit.core import AITER_CONFIGS

                paths = {
                    args.tune_file,
                    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE,
                }
                for path in paths:
                    try:
                        table = pd.read_csv(path)
                    except (OSError, pd.errors.EmptyDataError):
                        continue
                    for row in table.itertuples():
                        if (
                            getattr(row, "libtype", None) == "opus"
                            and str(getattr(row, "gfx", "gfx950")) == "gfx950"
                            and int(getattr(row, "cu_num", cu_num)) == cu_num
                            and str(getattr(row, "w_scale_block", "128x128"))
                            == "128x128"
                            and float(row.us) > 0
                            and math.isfinite(float(row.us))
                            and float(row.errRatio) <= args.errRatio
                        ):
                            opus_seeds[(int(row.M), int(row.N), int(row.K))].add(
                                int(row.kernelId)
                            )
            for row in untunedf.itertuples():
                if row.w_scale_block == "128x128":
                    shape = (int(row.M), int(row.N), int(row.K))
                    opus_by_shape[shape] = opus_candidates(
                        *shape, opus_seeds[shape], cu_num=cu_num
                    )
                    if args.libtype == "opus" and not opus_by_shape[shape]:
                        raise ValueError(f"No OPUS candidate supports {shape}")
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            block = str(untunedf.loc[i, "w_scale_block"])
            info_keys = (gfx, cu_num, M, N, K, block)
            opus = opus_by_shape.get((M, N, K), ()) if block == "128x128" else ()
            cands = [
                (kernel_id, ki)
                for pipe in PIPELINES_BY_W_SCALE_BLOCK[block]
                for kernel_id, ki in pipe.kernels_list.items()
                if args.libtype != "opus" and pipe.fits(ki, M, N, K)
            ]
            runs = [
                (kernel_id, ki.split_k, ki.name, run_gemm_flydsl)
                for kernel_id, ki in cands
            ]
            if self._mxscale_table and args.libtype != "opus":
                near = [
                    name
                    for mm, name in seeds.get((N, K, block), ())
                    if M / 2 <= mm <= M * 2
                ]
                for j, name in enumerate(bmm_candidates(M, N, K, block, near)):
                    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
                        parse_bmm_kernel_name,
                    )

                    splits = parse_bmm_kernel_name(name)["splits"]
                    runs.append((KERNEL_ID_BASE_BMM + j, splits, name, run_bmm_flydsl))
            runs.extend(
                (kid, 1, opus_kernel_name(kid, cu_num=cu_num), run_gemm_opus)
                for kid, _ in opus
            )
            for kernel_id, split_k, name, run in runs:
                is_opus = run is run_gemm_opus
                info = (
                    info_keys,
                    kernel_id,
                    split_k,
                    name,
                    "opus" if is_opus else "flydsl",
                )
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, block),
                        run,
                        (gemm_keys, kernel_id) if is_opus else (gemm_keys, name),
                        dict(run_kwargs),
                        run_torch,
                        ref_args,
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
            if not runs:
                logger.warning(f"no mxfp8 {block} kernel fits M={M} N={N} K={K}")

        if not task:
            return []
        return self._run_tasks(task, args)

    def _run_tasks(self, tasks, args):
        if not tasks:
            return []
        ensure_opus_kids_compiled(
            {task[0][1] for task in tasks if task[0][-1] == "opus"},
            cu_num=self.get_cu_num(),
        )
        counts = Counter(task[0][0] for task in tasks)
        return mp_tuner(
            tasks,
            [(count, ()) for count in counts.values()],
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            time = round(time, 4)
            keys, kernelId, splitK, kernelName, libtype = info
            if time in (self.INVALID_TIME, self.INF_TIME):
                kernelName = "None"
            elif libtype == "opus":
                kernelName = opus_kernel_name(kernelId, cu_num=int(keys[1]))
            elif libtype == "flydsl" and getattr(self, "_mxscale_table", False):
                kernelId = (
                    MXSCALE_BMM_KERNEL_ID
                    if kernelId >= KERNEL_ID_BASE_BMM
                    else MXSCALE_KERNEL_ID
                )
            tflops, bw = self.calculate((info, time, err_ratio))
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "tflops": [tflops],
                    "bw": [bw],
                    "errRatio": [err_ratio],
                }
            )
            rows.append(key_dict)
        if not rows:
            return pd.DataFrame(columns=self.columns)
        return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmA8W8MXFP8BpreshuffleTuner(
        "a8w8_mxfp8_bpreshuffle_tuned_gemm",
        key=key,
        resultList=resultList,
        description="Tune gfx950 MXFP8 bpreshuffle GEMM (FlyDSL and OPUS JIT/CO)",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
