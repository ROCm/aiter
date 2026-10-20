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
(w_scale [N/128, K/128], x_scale 1x128 as its column-major bytes) have: the
batched GEMM at B = 1 (bmm) joins as a third backend on them (its candidates:
the table's tuned bmm kernels for M/2..2M, its heuristic pick, and their
split-K neighbours). OPUS also joins on 128x128, using the same scale bytes:
ordinary candidates use JIT and pin-AGPR candidates use their pre-built CO.
FlyDSL rows identify the backend with kernelId "mxfp8" or "bmm"; OPUS rows
keep libtype "opus" and the exact numeric kernelId. All backends share the
BF16 torch reference and the same checkAllclose/--errRatio acceptance rules:

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        --mxscale_table \\
        -i <shapes with a w_scale_block column> \\
        -o aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
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
MXSCALE_TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
# The block of a shape list without a w_scale_block column, and of a tuned
# table from before the column.
W_SCALE_BLOCK = "1x32"
# w_scale block -> the pipelines tuned for it.
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
    data = {
        "x": x,
        "weight": weight,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        "x_scale": x_scale_arg.view(dtypes.fp8_e8m0),
        "w_scale": w_scale.view(dtypes.fp8_e8m0),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": dequant(x, x_scale),
        "w_deq": dequant(weight, w_scale, w_rows),
    }
    if w_scale_block == "128x128":
        # Same bytes, logical [M,K/128] with the OPUS stride (1,M).
        data["x_scale_opus"] = data["x_scale"].view(k // 128, m).T
    return data


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


def parse_opus_kids(value):
    try:
        kids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--opus-kids requires integer IDs") from exc
    if not kids or min(kids) < 0 or len(kids) != len(set(kids)):
        raise argparse.ArgumentTypeError("--opus-kids requires unique nonnegative IDs")
    return frozenset(kids)


def opus_candidates(m, n, k, kids=None):
    from csrc.opus_gemm.opus_gemm_common import (
        a8w8_mxscale_gemm_bpreshuffle_kernels_list,
        candidate_kids_for_shape,
    )

    selected = candidate_kids_for_shape(
        "gfx950", m, n, k, all_candidates=True, exhaustive=kids is None,
    )
    return [
        (kid, a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid])
        for kid in selected
        if kids is None or kid in kids
    ]


def _opus_configuration(instance):
    entry = instance.bpreshuffle_entry
    depth, output, ahead = map(
        int, re.search(r"_p(\d+)_o(\d+)_ba(\d+)_sk", entry["config_name"]).groups()
    )
    wave_k = instance.BLOCK_SIZE // (64 * instance.T_M * instance.T_N)
    geometry = (
        entry["pipeline"], instance.B_M, instance.B_N, instance.B_K,
        instance.T_M, instance.T_N, wave_k, instance.bpreshuffle_split_k,
    )
    return geometry, depth, ahead, output


def _opus_coarse_candidates(candidates):
    """Time existing COs and two JIT samples; defer new pin variants."""
    groups = defaultdict(list)
    coarse = set()
    for kid, instance in candidates:
        if instance.bpreshuffle_co is not None:
            if not instance.bpreshuffle_co.get("pin_variant"):
                coarse.add(kid)
            continue
        geometry, depth, ahead, output = _opus_configuration(instance)
        groups[geometry].append((kid, instance, depth, ahead, output))
    for variants in groups.values():
        # Sample a shallow/default-cache schedule and a deeper/streaming one.
        output_modes = sorted({item[4] for item in variants})
        for target_depth, cache, wg, output in (
            (2, (0, 0), 2, output_modes[0]),
            (4, (1, 17), 1, output_modes[-1]),
        ):
            selected = min(
                variants,
                key=lambda item: (
                    item[4] != output,
                    abs(item[2] - target_depth),
                    (item[1].cachectl_a, item[1].cachectl_b) != cache,
                    item[1].WG_PER_CU != wg,
                    item[3], item[0],
                ),
            )
            coarse.add(selected[0])
    return sorted(coarse), {
        geometry: [item[0] for item in variants]
        for geometry, variants in groups.items()
    }


def _opus_refinement_candidates(
    groups, measurements, already_measured, top_groups=3, *, err_ratio=0.05,
):
    """Expand the fastest valid geometries per pipeline under the shared error limit.

    If all samples fail, expand the first three geometries as a bounded fallback.
    """
    measured = {
        int(kid): float(us) for kid, us, error in measurements
        if math.isfinite(float(us)) and float(us) > 0 and float(error) <= err_ratio
    }
    by_pipeline = defaultdict(list)
    for geometry, variants in groups.items():
        best = min((measured[kid] for kid in variants if kid in measured), default=math.inf)
        by_pipeline[geometry[0]].append((best, geometry, variants))
    selected = set()
    for geometries in by_pipeline.values():
        ranked = sorted(geometries, key=lambda item: (item[0], item[1]))
        viable = [item for item in ranked if math.isfinite(item[0])]
        for _, _, variants in (viable or ranked)[:top_groups]:
            selected.update(variants)
    return sorted(selected - set(already_measured))


def _opus_pin_candidates(candidates, measurements, top_groups=3, *, err_ratio=0.05):
    """Expand pin masks of the fastest measured CO schedules per pipeline."""
    valid = {int(kid): float(us) for kid, us, error in measurements
             if math.isfinite(float(us)) and 0 < float(us) and float(error) <= err_ratio}
    groups = defaultdict(list)
    for kid, instance in candidates:
        entry = instance.bpreshuffle_co or {}
        if "pin_group" in entry:
            groups[(entry["pipeline"], entry["pin_group"])].append((kid, entry))
    ranked = defaultdict(list)
    for (pipeline, base), members in groups.items():
        best = min((valid[kid] for kid, _ in members if kid in valid), default=math.inf)
        if math.isfinite(best):
            variants = [kid for kid, entry in members if entry.get("pin_variant")]
            ranked[pipeline].append((best, base, variants))
    return sorted({kid for schedules in ranked.values()
                   for _, _, variants in sorted(schedules)[:top_groups] for kid in variants})


def prepare_opus_pin_kids(kids):
    """Build selected CO variants and keep only spill-free, effective pin layouts."""
    from csrc.opus_gemm.opus_gemm_common import gfx950_co_kernels_declared

    kids = sorted(kid for kid in kids if gfx950_co_kernels_declared.get(kid, {}).get("pin_variant"))
    if not kids:
        return set()
    opus_dir = Path(__file__).resolve().parents[1] / "opus_gemm"
    co_root = Path(os.environ.get("OPUS_GEN_CO_DIR") or opus_dir / "gen_co")
    command = [sys.executable, str(opus_dir / "gen_co/build_co.py"), "--arch", "gfx950",
               "--jobs", "4", "--skip-existing", "--out-dir", str(co_root / "gfx950"),
               "--kids", *map(str, kids)]
    compiler = (os.environ.get("OPUS_CO_LLVM_BIN") or os.environ.get("OPUS_HIP_CLANG_PATH")
                or os.environ.get("HIP_CLANG_PATH"))
    if compiler:
        command += ["--llvm-bin", compiler]
    with tempfile.TemporaryDirectory(prefix="opus_pin_build_") as directory:
        report = Path(directory) / "build_info.json"
        subprocess.run([*command, "--report", str(report)], check=True)
        with report.open() as stream:
            records = json.load(stream)["kernels"]
    ready = {row["kid"] for row in records if row["status"] == "ready"}
    for row in records:
        if row["status"] != "ready":
            print(f"[OPUS pin] rejected kid={row['kid']}: {row['reason']}", flush=True)
    print(f"[OPUS pin] selected={len(kids)} ready={len(ready)}", flush=True)
    return ready


def ensure_opus_kids_compiled(kids):
    """Prepare the existing exact-kid JIT/CO host launchers before spawning."""
    if not kids:
        return False
    from aiter.jit import core

    try:
        core.get_module("module_aiter_core")
    except ModuleNotFoundError:
        config = core.get_args_of_build("module_aiter_core")
        keys = (
            "srcs", "flags_extra_cc", "flags_extra_hip", "blob_gen_cmd",
            "extra_include", "extra_ldflags", "verbose", "is_python_module",
            "is_standalone", "torch_exclude", "third_party",
        )
        core.build_module(
            md_name="module_aiter_core",
            **{key: config[key] for key in keys},
            hipify=config.get("hipify", False),
            flags_extra_hip_per_source=config.get("flags_extra_hip_per_source", {}),
        )

    opus_dir = str(Path(__file__).resolve().parents[1] / "opus_gemm")
    added_path = opus_dir not in sys.path
    if added_path:
        sys.path.insert(0, opus_dir)
    previous_compiler = os.environ.get("HIP_CLANG_PATH")
    compiler = os.environ.get("OPUS_HIP_CLANG_PATH")
    try:
        from opus_gemm_tune import _ensure_kids_compiled

        if compiler:
            os.environ["HIP_CLANG_PATH"] = compiler
        return _ensure_kids_compiled(kids)
    finally:
        if compiler:
            if previous_compiler is None:
                os.environ.pop("HIP_CLANG_PATH", None)
            else:
                os.environ["HIP_CLANG_PATH"] = previous_compiler
        if added_path:
            sys.path.remove(opus_dir)


def run_gemm_opus(x, weight_shuffle, x_scale, w_scale, out, kid):
    from aiter.ops.opus import opus_gemm

    return opus_gemm(
        x, weight_shuffle, out, kid=kid, layout="bpreshuffle",
        x_scale=x_scale, w_scale=w_scale,
    )


# The batched GEMM at B = 1 in --mxscale_table mode: its ids follow mxfp8's
# (kernelId records the backend in that table), its split-K variants the
# batched tuner's.
KERNEL_ID_BASE_BMM = 3_000_000
_BMM_SPLITS = (1, 2, 4, 8, 16)
MXSCALE_KERNEL_ID = "mxfp8"
MXSCALE_BMM_KERNEL_ID = "bmm"


def _bmm_scale_args(w_scale_block, m=None):
    g = int(w_scale_block.split("x")[1])
    # The blockscale x_scale (1x128) is passed as its column-major bytes.
    return {
        "x_scale_k": g,
        "w_scale_n": g,
        "w_scale_k": g,
        "x_scale_transposed": g == 128 and (m is None or m > 1),
    }


def bmm_candidates(m, n, k, w_scale_block, seeds=()):
    """bmm kernelNames to time for a shape: ``seeds`` (tuned bmm rows of nearby
    M), the heuristic pick, and their split-K neighbours that run it."""
    if w_scale_block not in ("32x32", "128x128"):
        return []
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        bmm_kernel_name,
        check_bmm_config,
        parse_bmm_kernel_name,
        pick_bmm_kernel_name,
    )

    scale_args = _bmm_scale_args(w_scale_block, m)
    names = set(seeds)
    try:
        names.add(pick_bmm_kernel_name(1, m, n, k, **scale_args))
    except ValueError:
        pass
    out = set()
    for name in names:
        cfg = parse_bmm_kernel_name(name)
        if cfg is None:
            continue
        base = cfg["splits"]
        for sp in _BMM_SPLITS:
            if sp != base and not base / 2 <= sp <= base * 2:
                continue
            c = {**cfg, "splits": sp}
            try:
                check_bmm_config(n, k, 1, **c, **scale_args)
            except ValueError:
                continue
            tiles = math.ceil(m / c["tile_m"]) * (n // c["tile_n"])
            if c["splits"] > 1 and tiles > (1 << 16):
                continue
            out.add(bmm_kernel_name(**c))
    return sorted(out)


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
        _op.get_mxscale_bpreshuffle_config.cache_clear()
        _op._load_mxscale_bpreshuffle_tuned.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--mxscale_table",
            action="store_true",
            help="Tune gemm_a8w8_blockscale_bpreshuffle's e8m0 table: bmm (the "
            "batched GEMM at B = 1) joins on 32x32 / 128x128; OPUS JIT/CO "
            "joins on 128x128. A shape list without a block column uses 128x128.",
        )
        self.parser.add_argument(
            "--libtype", choices=("both", "opus", "flydsl"), default="both",
            help="Backends to search; OPUS requires --mxscale_table and 128x128.",
        )
        self.parser.add_argument(
            "--opus-kids", type=parse_opus_kids,
            help="Optional comma-separated exact OPUS IDs; explicit IDs bypass performance pruning.",
        )
        self.parser.add_argument(
            "--opus-search", choices=("filtered", "exhaustive"), default="filtered",
            help="OPUS search policy: shape pruning, coarse geometry search, schedule and pin refinement "
            "(default), or every legal configuration. Timing and correctness checks are identical.",
        )
        self.parser.add_argument(
            "--compare-upstream", action="store_true",
            help="Compare current OPUS against FlyDSL BMM and GEMM #6603 using all saved BMM seeds.",
        )
        self.parser.add_argument(
            "--bmm-seeds", type=Path,
            help="Additional saved BMM rows in the standard tuned CSV format.",
        )
        self.parser.add_argument(
            "--flydsl-candidates", type=Path,
            help="Optional per-shape FlyDSL shortlist CSV (M,N,K,w_scale_block,kernelName). "
            "Only listed legal BMM and PR6603 candidates are timed; OPUS search is unchanged.",
        )

    def pre_process(self, args):
        self._mxscale_table = bool(args.mxscale_table)
        if self._mxscale_table:
            self.ARG_DEFAULTS = {
                **type(self).ARG_DEFAULTS,
                "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
            }
            if args.tune_file == TUNED_FILE:
                args.tune_file = MXSCALE_TUNED_FILE
        elif args.libtype == "opus" or args.opus_kids is not None:
            self.parser.error("OPUS tuning requires --mxscale_table")
        if args.compare_upstream and not args.mxscale_table:
            self.parser.error("--compare-upstream requires --mxscale_table")
        if args.libtype == "flydsl" and args.opus_kids is not None:
            self.parser.error("--opus-kids requires --libtype opus or both")
        if args.opus_kids is not None:
            from csrc.opus_gemm.opus_gemm_common import (
                a8w8_mxscale_gemm_bpreshuffle_kernels_list,
            )

            unknown = args.opus_kids.difference(a8w8_mxscale_gemm_bpreshuffle_kernels_list)
            if unknown:
                self.parser.error(f"Unknown or unavailable OPUS IDs: {sorted(unknown)}")
        super().pre_process(args)

    def _bmm_seeds(self, extra_path=None):
        """{(N, K, w_scale_block): [(M, bmm kernelName)]} of the tuned
        gemm_a8w8_blockscale_bpreshuffle table, this arch and CU count."""
        from aiter.jit.core import AITER_CONFIGS

        path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
        try:
            df = with_mxscale_w_scale_block(pd.read_csv(path), path)
        except (OSError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        if extra_path is not None:
            extra = with_mxscale_w_scale_block(pd.read_csv(extra_path), extra_path)
            df = pd.concat((df, extra), ignore_index=True)
        if df.empty:
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
        # Default to 1x32 for the original table, 128x128 for --mxscale_table.
        # pre_process slices untunedf down to self.keys.
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        default_block = "128x128" if getattr(self, "_mxscale_table", False) else W_SCALE_BLOCK
        if "w_scale_block" not in df.columns:
            df["w_scale_block"] = default_block
        df["w_scale_block"] = df["w_scale_block"].fillna(default_block)
        unknown = set(df["w_scale_block"]) - set(PIPELINES_BY_W_SCALE_BLOCK)
        if unknown:
            raise ValueError(
                f"{untuned_gemm_file}: no gfx950 mxfp8 kernels tuned for w_scale "
                f"block {sorted(unknown)}; tunable: {sorted(PIPELINES_BY_W_SCALE_BLOCK)}"
            )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        default_block = "128x128" if getattr(self, "_mxscale_table", False) else W_SCALE_BLOCK
        df = with_mxscale_w_scale_block(df, tuned_gemm_file, default=default_block)
        # A table from before the column gets it where the tuner writes it.
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
            from csrc.opus_gemm.opus_gemm_common import gfx950_co_kernels_declared

            pins = {kid for kid in kids if gfx950_co_kernels_declared.get(kid, {}).get("pin_variant")}
            if pins - prepare_opus_pin_kids(pins):
                raise RuntimeError("Requested OPUS pin COs failed build/resource checks")
            ensure_opus_kids_compiled(kids)
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
                    out, run_torch(d["x_deq"], d["w_deq"]),
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
        tasks_data = []
        self._mxscale_table = bool(getattr(args, "mxscale_table", False))
        coarse = sorted(set(untunedf["w_scale_block"]) - {W_SCALE_BLOCK})
        if coarse and not self._mxscale_table:
            # gemm_a8w8_mxfp8_bpreshuffle serves 1x32 only; these blocks run
            # through gemm_a8w8_blockscale_bpreshuffle's table, tuned with bmm.
            raise ValueError(
                f"w_scale block {coarse} is tuned for gemm_a8w8_blockscale_"
                f"bpreshuffle only: pass --mxscale_table"
            )
        seeds = self._bmm_seeds(args.bmm_seeds) if self._mxscale_table else {}
        flydsl_shortlist = None
        if getattr(args, "flydsl_candidates", None) is not None:
            shortlist = pd.read_csv(args.flydsl_candidates)
            required = ["M", "N", "K", "w_scale_block", "kernelName"]
            if not set(required).issubset(shortlist.columns):
                raise ValueError(f"--flydsl-candidates requires columns {required}")
            flydsl_shortlist = {}
            for row in shortlist.itertuples():
                key = (int(row.M), int(row.N), int(row.K), str(row.w_scale_block))
                flydsl_shortlist.setdefault(key, set()).add(str(row.kernelName))
        comparison_seeds = {}
        if args.compare_upstream:
            for (n, k, block), entries in seeds.items():
                comparison_seeds.setdefault(block, set()).update(
                    name for _, name in entries
                )
        opus_by_shape = {}
        if args.libtype != "flydsl" and self._mxscale_table:
            if args.libtype == "opus" and set(untunedf.w_scale_block) != {"128x128"}:
                raise ValueError("OPUS supports A 1x128 / B 128x128 scales only")
            for row in untunedf.itertuples():
                if row.w_scale_block == "128x128":
                    shape = (int(row.M), int(row.N), int(row.K))
                    opus_by_shape[shape] = opus_candidates(*shape, args.opus_kids)
                    if args.libtype == "opus" and not opus_by_shape[shape]:
                        raise ValueError(f"No OPUS candidate supports {shape}")
            deferred = args.opus_search == "filtered" and args.opus_kids is None
            pin_kids = {kid for candidates in opus_by_shape.values() for kid, instance in candidates
                        if (instance.bpreshuffle_co or {}).get("pin_variant")}
            ready = set() if deferred else prepare_opus_pin_kids(pin_kids)
            if not deferred:
                rejected = pin_kids - ready
                if args.opus_kids is not None and rejected:
                    raise RuntimeError(f"Requested OPUS pin COs failed build/resource checks: {sorted(rejected)}")
                opus_by_shape = {shape: [(kid, inst) for kid, inst in candidates if kid not in rejected]
                                 for shape, candidates in opus_by_shape.items()}
            ensure_opus_kids_compiled({kid for candidates in opus_by_shape.values()
                                       for kid, _ in candidates if kid not in pin_kids or kid in ready})
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            block = str(untunedf.loc[i, "w_scale_block"])
            allowed_flydsl = None if flydsl_shortlist is None else flydsl_shortlist.get((M, N, K, block))
            if flydsl_shortlist is not None and not allowed_flydsl:
                raise ValueError(f"No FlyDSL shortlist for {(M, N, K, block)}")
            info_keys = (gfx, cu_num, M, N, K, block)
            # One pass per pipeline of the block, in their order.
            opus = opus_by_shape.get((M, N, K), ()) if block == "128x128" else ()
            cands = [
                (kernel_id, ki)
                for pipe in PIPELINES_BY_W_SCALE_BLOCK[block]
                for kernel_id, ki in pipe.kernels_list.items()
                if args.libtype != "opus" and pipe.fits(ki, M, N, K)
                and (allowed_flydsl is None or ki.name in allowed_flydsl)
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
                if args.compare_upstream:
                    near = sorted(set(near) | comparison_seeds.get(block, set()))
                if allowed_flydsl is not None:
                    near = sorted(allowed_flydsl)
                for j, name in enumerate(bmm_candidates(M, N, K, block, near)):
                    if allowed_flydsl is not None and name not in allowed_flydsl:
                        continue
                    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
                        parse_bmm_kernel_name,
                    )

                    splits = parse_bmm_kernel_name(name)["splits"]
                    runs.append((KERNEL_ID_BASE_BMM + j, splits, name, run_bmm_flydsl))
            for kernel_id, split_k, name, run in runs:
                # kernelName carries the full launch config (incl. split-K and
                # the pipeline), so dispatch never needs kernelId -- it is
                # recorded for reference (or, with --mxscale_table, replaced by
                # the backend's tag).
                info = (info_keys, kernel_id, split_k, name, "flydsl")
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, block),
                        run,
                        (gemm_keys, name),
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
            for kid, instance in opus:
                info = (info_keys, kid, instance.bpreshuffle_split_k, instance.name, "opus")
                opus_keys = [*gemm_keys]
                opus_keys[2] = "x_scale_opus"
                task.append(
                    (
                        info, generate_data,
                        (M, N, K, seed, block),
                        run_gemm_opus, (opus_keys, kid), dict(run_kwargs),
                        run_torch, ref_args, {}, None,
                        1e-2, 0.01, None, None, ("out",),
                    )
                )
            if not runs and not opus:
                logger.warning(f"no mxfp8 {block} kernel fits M={M} N={N} K={K}")
            tasks_data.append((len(runs) + len(opus), ()))

        if not task:
            return []
        if getattr(args, "opus_search", "filtered") == "filtered" and args.opus_kids is None:
            return self._run_filtered_opus(task, opus_by_shape, args)
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def _run_filtered_opus(self, tasks, candidates_by_shape, args):
        def run_pass(pass_tasks):
            counts = Counter(task[0][0] for task in pass_tasks)
            return list(mp_tuner(
                pass_tasks, [(count, ()) for count in counts.values()],
                args.mp, False, args.shape_grouped, args.errRatio,
                timeout=args.timeout, verbose=args.verbose,
            )) if pass_tasks else []

        def task_shape(info):
            return tuple(int(value) for value in info[0][2:5])

        plans = {}
        for shape, candidates in candidates_by_shape.items():
            coarse, groups = _opus_coarse_candidates(candidates)
            plans[shape] = (set(coarse), groups)
            print(f"[OPUS search] shape={shape} legal={len(candidates)} coarse={len(coarse)}", flush=True)
        first_tasks = [
            task for task in tasks
            if task[0][-1] != "opus" or task[0][1] in plans[task_shape(task[0])][0]
        ]
        results = run_pass(first_tasks)
        refinement = {}
        pin_kids = set()
        for shape, (coarse, groups) in plans.items():
            measurements = [
                (info[1], us, error) for info, us, error in results
                if info[-1] == "opus" and task_shape(info) == shape
            ]
            selected = _opus_refinement_candidates(
                groups, measurements, coarse, err_ratio=args.errRatio,
            )
            pins = _opus_pin_candidates(candidates_by_shape[shape], measurements, err_ratio=args.errRatio)
            pin_kids.update(pins)
            refinement[shape] = set(selected) | set(pins)
            print(f"[OPUS refine] shape={shape} schedule={len(selected)} pin={len(pins)}", flush=True)
        ready = prepare_opus_pin_kids(pin_kids)
        ensure_opus_kids_compiled(ready)
        for shape in refinement:
            refinement[shape].difference_update(pin_kids - ready)
        results.extend(run_pass([
            task for task in tasks
            if task[0][-1] == "opus" and task[0][1] in refinement[task_shape(task[0])]
        ]))
        return results

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time in (self.INVALID_TIME, self.INF_TIME):
                kernelName = "None"
            elif libtype == "flydsl" and getattr(self, "_mxscale_table", False):
                kernelId = (
                    MXSCALE_BMM_KERNEL_ID
                    if kernelId >= KERNEL_ID_BASE_BMM
                    else MXSCALE_KERNEL_ID
                )
            tflops, bw = self.calculate(el)
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
    # Same columns as the table the gfx1250 rows already live in.
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
