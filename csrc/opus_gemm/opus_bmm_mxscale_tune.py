# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune MXFP8 BMM with local kernel IDs for the current gfx/CU."""

import argparse
import math
import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw
from aiter.ops.opus.launch_plan import _get_cached_a8w8_mxscale_bmm_plan
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner, TunerCommon
from aiter.utility.mp_tuner import mp_tuner

# Make tuner helpers importable in spawned workers.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OPTESTS = os.path.join(_REPO, "op_tests")
for _p in (_HERE, _OPTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opus_gemm_common import (
    _opus_sf_shuf_sub,
    get_registry,
    parse_integer,
)
from test_opus_a8w8_bmm import (
    GROUP,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

from aiter.ops.opus._arch import _device_arch_and_cu

_tune_gfx, _tune_cu = _device_arch_and_cu(torch.cuda.current_device())
_CODEGEN_BMM = get_registry("a8w8_mxscale_bmm", _tune_gfx, _tune_cu)


def _kid_groups(kid):
    inst = _CODEGEN_BMM[kid]
    return inst.GROUP_N, inst.GROUP_K


def _kid_block(kid):
    return "{}x{}".format(*_kid_groups(kid))


# Local ids select split-K choices in the current CU table.
_SPLITK_SEEDS = {
    (1, 2, 4, 8): (
        0,
        1,
        2,
        9,
        10,
        *range(62, 76),
        77,
        78,
        140,
        141,
        143,
        144,
        145,
        146,
        147,
        149,
        *range(185, 198),
        199,
        200,
        226,
        227,
        229,
        230,
        *range(231, 243),
    ),
    (1, 4): range(107, 115),
    (1, 2, 4): (115, 116),
}
_SHUF_POLICY = {}
_splitk_choices = {
    kid: list(splits) for splits, kids in _SPLITK_SEEDS.items() for kid in kids
}
_DEFAULT_EXCLUDED_KIDS = frozenset(
    (0, 85, 86, 87, *range(117, 125), 145, 207, 208, 209, *range(220, 226))
)
_TUNE_POLICY = {
    kid: _splitk_choices.get(kid, [1])
    for kid in _CODEGEN_BMM
    if kid not in _DEFAULT_EXCLUDED_KIDS
}
_CANDIDATE_KIDS = tuple(_TUNE_POLICY) + tuple(_SHUF_POLICY)


def _shuf_arm(inst):
    """Return "reg" or "lds" according to the instance's shuffled-scale storage."""
    return "lds" if inst.sf_shuf_in_lds else "reg"


_SHUF_FAMILIES = frozenset(
    {
        "a8w8_mxscale_bmm_bpreshuffle_wave8n4",
        "a8w8_mxscale_bmm_bpreshuffle_wavetm1",
        "a8w8_mxscale_bmm_bpreshuffle_bdirect",
        "a8w8_mxscale_bmm_bpreshuffle_bdirect_tilen",
        "a8w8_mxscale_bmm_bpreshuffle_blds",
        "a8w8_mxscale_bmm_flatmm_splitk",
    }
)

SHUF_SUB = _opus_sf_shuf_sub()


def _shuf_sub(inst):
    """Return SHUF_SUB when the tile supports the shipped scale layout."""
    if inst.kernel_tag not in _SHUF_FAMILIES:
        return None
    com_rep_k = inst.B_K // inst.GROUP_K
    if com_rep_k not in (1, 2, 4):
        return None
    kd = com_rep_k // 2 if com_rep_k >= 2 else 1
    assert kd * 2 >= com_rep_k, "SFG::KD must cover every K block the tile owns"
    sf_mb = inst.T_M * inst.W_M
    paired = sf_mb <= SHUF_SUB and SHUF_SUB % sf_mb == 0
    wide = sf_mb == 2 * SHUF_SUB
    if not (paired or wide) or inst.B_M % sf_mb:
        return None
    if SHUF_SUB < inst.B_M < 2 * SHUF_SUB:
        return None
    return SHUF_SUB


# Candidate pools select compatible weight and scale layouts.
POOLS = ("all", "preb", "rowb", "subok", "shuf", "shuf_reg", "shuf_lds")


def _applicable(kid, g, m, n, k, pool="all", *, split_ks=None):
    """Return legal split-K candidates for the shape and layout pool."""
    k_inst = _CODEGEN_BMM[kid]
    shuf = kid in _SHUF_POLICY
    # The pool determines the scale layout shared by all candidates.
    if pool in ("shuf", "shuf_reg", "shuf_lds"):
        if not shuf:
            return []
        if pool != "shuf" and _shuf_arm(k_inst) != pool.split("_")[1]:
            return []
    elif shuf:
        return []
    if pool == "preb" and not k_inst.needs_preshuffled_b:
        return []
    if pool == "rowb" and k_inst.needs_preshuffled_b:
        return []
    if pool == "subok":
        if not k_inst.needs_preshuffled_b:
            return []
        if _shuf_sub(k_inst) is None:
            return []
    candidates = split_ks if split_ks is not None else _TUNE_POLICY[kid]
    legal = []
    for split in candidates:
        try:
            _get_cached_a8w8_mxscale_bmm_plan(
                _tune_gfx, kid, dtypes.bf16, m, g, n, k, split, _tune_cu
            )
        except ValueError:
            continue
        legal.append(split)
    return legal


SHIPPED_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_tuned.csv",
)
DEFAULT_OUT = os.path.join(_REPO, "dsv4_bmm_mxscale_retuned.csv")
# The w_scale block every OPUS MXFP8 BMM kernel reads (a tuned-CSV key column).
W_SCALE_BLOCK = "128x128"

# Preshuffled-weight model configurations use a separate CSV.
BPRESHUFFLE_CSV = os.path.join(
    _REPO,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv",
)


# ---------------------------------------------------------------------------
# mp_tuner hooks (module-level so the spawn workers can import them by name).
# ---------------------------------------------------------------------------
def _gen_varied(shape, k, device, group=GROUP):
    """Signed, per-128-K-block varied-magnitude bf16 (mirrors _block_varied)."""
    x = torch.randn(shape, dtype=dtypes.fp32, device=device)
    amp = torch.exp2(
        torch.randint(-4, 4, (*shape[:-1], k // group), device=device).float()
    )
    return (x * amp.repeat_interleave(group, dim=-1)).to(dtypes.bf16)


def _workspace_numel(kernel_id, split_k, batch, m, n, k):
    if split_k <= 1:
        return 0
    plan = _get_cached_a8w8_mxscale_bmm_plan(
        _tune_gfx, int(kernel_id), dtypes.fp32, m, batch, n, k, split_k, _tune_cu
    )
    return plan.workspace_spec.shape[0] if plan.workspace_spec is not None else 0


def gen_bmm_mxscale_data(
    batch, m, n, k, seed, out_dtype, kernel_id, split_k, device="cuda"
):
    """Prepare operands, workspace and reference outside the timed region."""
    torch.manual_seed(seed)
    group_n, group = _kid_groups(kernel_id)
    O_bf16 = _gen_varied((batch, m, k), k, device, group)
    W_bf16 = _gen_varied((batch, n, k), k, device, group)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16, group=group)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16, group=group, group_n=group_n)
    Y = torch.empty((m, batch, n), dtype=out_dtype, device=device)

    workspace_numel = _workspace_numel(kernel_id, split_k, batch, m, n, k)
    workspace = (
        torch.empty(workspace_numel, dtype=torch.float32, device=device)
        if workspace_numel
        else None
    )
    ref = (
        run_torch(O_mx, W_mx, xs_fp32, ws_fp32, group=group, group_n=group_n)
        .transpose(0, 1)
        .to(out_dtype)
    )
    # Prepare preshuffled weights before timing.
    W_sh = shuffle_weight(W_mx, layout=(16, 16))
    # Both backends receive contiguous token-major inputs and scales.
    A_mx = O_mx.transpose(0, 1).contiguous()
    A_scale = xs_mx.transpose(0, 1).contiguous()
    return (A_mx, W_mx, Y, A_scale, ws_mx, workspace, ref, W_sh)


def run_bmm_mxscale_bench(
    A_mx, W_mx, Y, A_scale, ws_mx, workspace, W_sh, kernelId, splitK
):
    """Select the kernel's weight/scale layouts, launch it, and return Y.

    All layouts and any split-K workspace are prepared by the data generator;
    this function performs no layout conversion or workspace allocation."""
    inst = _CODEGEN_BMM[kernelId]
    Wb = W_sh if inst.needs_preshuffled_b else W_mx
    # A, its plain scale and Y all arrive [M, batch, ...] contiguous, which is
    # both what the launcher takes and what a serving stack holds.
    sfa, sfb = A_scale, ws_mx
    _opus_gemm_a8w8_mxscale_bmm_launch_raw(
        A_mx,
        Wb,
        Y,
        sfa,
        sfb,
        workspace=workspace,
        kid=kernelId,
        split_k=splitK,
    )
    return Y


def _bmm_ref_passthrough(ref):
    """ref_func: the fp32 reference is precomputed in gen_data (slot 6)."""
    return ref


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------
BMM_BENCH_KEYS = (0, 1, 2, 3, 4, 5, 7)
# Data preparation only depends on the kernel's scale groups with splitK=1.
BMM_BLOCK_DATA_KIDS = {_kid_block(kid): kid for kid in _CODEGEN_BMM}
BMM_DATA_KIDS = {
    group: BMM_BLOCK_DATA_KIDS[f"{group}x{group}"]
    for group in (128, 32)
    if f"{group}x{group}" in BMM_BLOCK_DATA_KIDS
}


def make_bmm_tuning_task(
    info, generate, gen_args, run, bench_args, perf_kwargs, *, ref_index, output_index
):
    """Shared mp_tuner contract: rotate inputs and NaN-initialize output."""
    return (
        info,
        generate,
        gen_args,
        run,
        bench_args,
        perf_kwargs,
        _bmm_ref_passthrough,
        ([ref_index],),
        {},
        None,
        1e-2,
        1e-2,
        None,
        None,
        [output_index],
    )


class OpusBmmMxscaleTuner(GemmCommonTuner):
    def get_gfx(self):
        return _device_arch_and_cu(torch.cuda.current_device())[0]

    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "",
        "untune_file": "",
        # Accept the FP8 quantization mismatch floor.
        "errRatio": 0.02,
        "batch": 100,
        "config_env_name": "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE",
    }

    KEYS: ClassVar[list[str]] = ["gfx", "cu_num", "b", "m", "n", "k", "w_scale_block"]
    RESULTS: ClassVar[list[str]] = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]

    def __init__(self):
        # Initialize the lowercase batched schema directly.
        TunerCommon.__init__(
            self,
            type(self).__name__,
            self.KEYS,
            self.RESULTS,
            description="Tune opus fp8 e8m0 mxscale flatmm split-K BMM (DSV4 wo_a)",
        )
        # Sort by shape and scale block.
        self.sort_keys = ["gfx", "cu_num", "b", "n", "m", "k", "w_scale_block"]

    # --- schema helpers -----------------------------------------------------
    def getKernelName(self, kernelId):
        k_inst = _CODEGEN_BMM.get(int(kernelId))
        return k_inst.name if k_inst else None

    def calculate(self, results, bpes=None):
        info, time, _err = results
        if time == self.INVALID_TIME:
            return 0, 0
        shape = dict(zip(self.keys, info[0]))
        b, m, n, k = (int(shape[name]) for name in ("b", "m", "n", "k"))
        us_s = time * 1e-6
        tflops = round(2 * b * m * n * k / us_s / 1e12, 1)
        # fp8 A + fp8 W + bf16 out.
        bw = round((b * m * k + b * n * k + 2 * b * m * n) / us_s / 1e9, 2)
        return tflops, bw

    def result_to_df(self, results):
        rows = []
        for info, time, err in results:
            keys, kernelId, splitK, kernelName = info
            # Invalid timings must not win the per-shape ranking.
            if not (time > 0):
                logger.warning(
                    "%s: kid %s timed at %s, not ranking it", keys, kernelId, time
                )
                time = float("inf")
            resolved = kernelName or self.getKernelName(kernelId)
            tflops, bw = self.calculate((info, time, err))
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": "opus",
                    "kernelId": int(kernelId),
                    "splitK": int(splitK),
                    "us": time,
                    "kernelName": "None" if resolved is None else str(resolved),
                    "tflops": tflops,
                    "bw": bw,
                    "errRatio": err,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    # --- CLI ----------------------------------------------------------------
    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--input_file",
            dest="untune_file",
            default=argparse.SUPPRESS,
            help="Input CSV (alias for -i)",
        )
        self.parser.add_argument(
            "--tuned_file",
            dest="tune_file",
            default=argparse.SUPPRESS,
            help="Output CSV (alias for -o)",
        )
        # Free the base "-k/--splitK" store_true so we can reuse -k for the K dim.
        for action in list(self.parser._actions):
            if "-k" in action.option_strings or "--splitK" in action.option_strings:
                self.parser._actions.remove(action)
                for s in action.option_strings:
                    self.parser._option_string_actions.pop(s, None)
                for grp in self.parser._action_groups:
                    if action in grp._group_actions:
                        grp._group_actions.remove(action)
                break

        def _intlist(s):
            return [int(x) for x in str(s).split(",") if x != ""]

        def _blocklist(s):
            blocks = tuple(dict.fromkeys(part.strip().lower() for part in s.split(",")))
            allowed = ("1x32", "32x32", "128x128")
            if any(block not in allowed for block in blocks):
                raise argparse.ArgumentTypeError(
                    "expected comma-separated blocks from " + ", ".join(allowed)
                )
            return blocks

        self.parser.add_argument(
            "-g",
            "--batch_g",
            type=_intlist,
            default=None,
            help="comma list of batch g (e.g. 2,8,16)",
        )
        self.parser.add_argument(
            "-m",
            "--M",
            type=_intlist,
            default=None,
            help="comma list of M (e.g. 1,16,64)",
        )
        self.parser.add_argument(
            "-n",
            "--N",
            type=_intlist,
            default=[1024],
            help="comma list of N (default 1024)",
        )
        self.parser.add_argument(
            "-k",
            "--K",
            type=_intlist,
            default=[4096],
            help="comma list of K (default 4096)",
        )
        self.parser.add_argument(
            "--w_scale_block",
            type=_blocklist,
            default=None,
            help="comma-separated weight-scale blocks (1x32,32x32,128x128); "
            "filter CSV rows or expand shape-only inputs",
        )
        self.parser.add_argument(
            "--apply",
            action="store_true",
            default=False,
            help="retune the selected shipped table in place (alias for -o TABLE --all)",
        )
        self.parser.add_argument(
            "--bpreshuffle",
            action="store_true",
            default=False,
            help="use preshuffled-B candidates and the preshuffle table by default",
        )
        self.parser.add_argument(
            "--pool",
            choices=POOLS,
            default="all",
            help="restrict candidates by B layout (--bpreshuffle implies preb)",
        )
        self.parser.add_argument(
            "--graph_m_max",
            type=int,
            default=0,
            help="use graph timing for M <= this value (0 disables it)",
        )

    def get_arg_defaults(self):
        defaults = super().get_arg_defaults()
        if getattr(self, "_bpreshuffle", False):
            defaults["config_env_name"] = (
                "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE"
            )
        return defaults

    def _normalize_rows(self, df, *, default_blocks=("128x128",), blocks=None):
        """Preserve CSV shape/scale keys; expand scales only when unspecified."""
        aliases = {name: name for name in self.keys}
        aliases["g"] = "b"
        df = df.rename(
            columns={c: aliases.get(str(c).strip().lower(), c) for c in df.columns}
        ).copy()
        if df.columns.duplicated().any():
            raise ValueError("MXFP8 BMM CSV has duplicate shape columns")
        missing = {"b", "m", "n", "k"}.difference(df.columns)
        if missing:
            raise ValueError(f"MXFP8 BMM CSV is missing columns: {sorted(missing)}")
        for name in ("b", "m", "n", "k"):
            values = pd.to_numeric(df[name], errors="raise")
            if (values.isna() | (values <= 0) | (values % 1 != 0)).any():
                raise ValueError(f"{name} must contain positive integers")
            df[name] = values.astype("int64")
        if "gfx" not in df:
            df["gfx"] = self.get_gfx()
        if "cu_num" not in df:
            df["cu_num"] = self.get_cu_num()
        if "w_scale_block" not in df:
            df = pd.concat(
                [
                    df.assign(w_scale_block=block)
                    for block in (blocks or default_blocks)
                ],
                ignore_index=True,
            )
        df["w_scale_block"] = df["w_scale_block"].astype(str).str.strip().str.lower()
        if not df["w_scale_block"].isin(BMM_BLOCK_DATA_KIDS).all():
            raise ValueError("w_scale_block must be 1x32, 32x32 or 128x128")
        n_align = df["w_scale_block"].map({"1x32": 16, "32x32": 128, "128x128": 128})
        if ((df["n"] % n_align != 0) | (df["k"] % 128 != 0)).any():
            raise ValueError(
                "MXFP8 BMM requires aligned N and K: N16/K128 for 1x32, N128/K128 otherwise"
            )
        if blocks is not None:
            df = df[df["w_scale_block"].isin(blocks)]
        return df.reset_index(drop=True)

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        return self._normalize_rows(df) if not df.empty else df

    def pre_process(self, args):
        gfx = self.get_gfx()
        if gfx != "gfx950":
            raise RuntimeError(f"MXFP8 BMM tuning is gfx950-only; detected {gfx!r}")
        self._bpreshuffle = args.bpreshuffle or args.pool in (
            "preb",
            "subok",
            "shuf",
            "shuf_reg",
            "shuf_lds",
        )
        if args.bpreshuffle and args.pool == "all":
            args.pool = "preb"
        source = BPRESHUFFLE_CSV if self._bpreshuffle else SHIPPED_CSV
        if args.apply:
            args.tune_file, args.all = source, True
        elif not args.tune_file:
            args.tune_file = source if self._bpreshuffle else DEFAULT_OUT
        if (args.batch_g is None) != (args.M is None):
            raise ValueError("-g/--batch_g and -m/--M must be provided together")
        if args.batch_g is not None:
            df = pd.DataFrame(
                [
                    (b, m, n, k)
                    for b in args.batch_g
                    for m in args.M
                    for n in args.N
                    for k in args.K
                ],
                columns=["b", "m", "n", "k"],
            )
        else:
            path = (
                args.run_config
                if isinstance(args.run_config, str)
                else args.untune_file or source
            )
            df = (
                self.get_tuned_gemm_list(path)
                if isinstance(args.run_config, str)
                else self.get_untuned_gemm_list(path)
            )
        self.untunedf = self._normalize_rows(
            df, default_blocks=("32x32", "128x128"), blocks=args.w_scale_block
        )
        if not self._bpreshuffle and self.untunedf["w_scale_block"].eq("1x32").any():
            raise ValueError("1x32 tuning requires --bpreshuffle")
        self.untunedf = (
            self.untunedf[
                (self.untunedf["gfx"] == gfx)
                & (self.untunedf["cu_num"] == self.get_cu_num())
            ][self.keys]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not (args.all or args.run_config or args.compare) and not self.tunedf.empty:
            have = set(self.tunedf[self.keys].apply(tuple, axis=1))
            self.untunedf = self.untunedf[
                ~self.untunedf[self.keys].apply(tuple, axis=1).isin(have)
            ].reset_index(drop=True)
        self.opus_policy = {
            kid: _TUNE_POLICY.get(kid, _SHUF_POLICY.get(kid)) for kid in _CANDIDATE_KIDS
        }

    def _clear_op_caches(self):
        from aiter.jit.core import AITER_CONFIGS
        from aiter.ops import batched_gemm_op_a8w8
        from aiter.ops.opus import policy

        AITER_CONFIGS.get_config_file.cache_clear()
        policy._load_mxscale_bmm_tuned.cache_clear()
        policy.lookup_mxscale_bmm_config.cache_clear()
        batched_gemm_op_a8w8._get_mxscale_bmm_launch_plan.cache_clear()

    def _set_config_env_for_run_config(self, args, config_file=None):
        # OPUS dispatches exact compiled kids; changing the CSV needs no rebuild.
        from aiter.jit import core

        env_name = self.get_arg_defaults()["config_env_name"]
        previous = os.environ.get(env_name)
        os.environ[env_name] = str(config_file or self.get_out_file(args.tune_file))
        self._clear_op_caches()
        return previous, core.AITER_REBUILD

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()

    def _perf_kwargs(self, args, m):
        return {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
            "testGraph": bool(args.graph_m_max and m <= args.graph_m_max),
        }

    def _saved_benchmark(self, row, seed):
        if str(row["libtype"]).lower() != "opus":
            raise ValueError(
                "The OPUS tuner requires libtype=opus; use the joint preshuffle tuner for FlyDSL"
            )
        gfx = str(row["gfx"]).lower().split(":", 1)[0]
        cu_num = int(row["cu_num"])
        if (gfx, cu_num) != (self.get_gfx(), self.get_cu_num()):
            raise ValueError("saved kernel namespace does not match tuning device")
        kid = parse_integer(row["kernelId"], name="saved kernel id")
        instance = _CODEGEN_BMM.get(kid)
        name = str(row.get("kernelName", ""))
        if instance is None or (
            name not in ("", "nan", "None") and name != instance.name
        ):
            raise ValueError(f"Saved symbol does not match kid {kid}")
        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        split = int(row["splitK"])
        _get_cached_a8w8_mxscale_bmm_plan(
            gfx, kid, dtypes.bf16, m, b, n, k, split, cu_num
        )
        if row["w_scale_block"] != _kid_block(kid):
            raise ValueError(f"Scale block does not match kid {kid}")
        data = gen_bmm_mxscale_data(b, m, n, k, seed, dtypes.bf16, kid, split)
        data[2].fill_(float("nan"))
        return (
            run_bmm_mxscale_bench,
            tuple(data[i] for i in BMM_BENCH_KEYS) + (kid, split),
            data[6],
        )

    def _default_benchmark(self, row, seed):
        from aiter.ops.batched_gemm_op_a8w8 import (
            batched_gemm_a8w8_mxscale,
            batched_gemm_a8w8_mxscale_bpreshuffle,
        )

        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        data = gen_bmm_mxscale_data(
            b, m, n, k, seed, dtypes.bf16, BMM_BLOCK_DATA_KIDS[row["w_scale_block"]], 1
        )
        run = (
            batched_gemm_a8w8_mxscale_bpreshuffle
            if self._bpreshuffle
            else batched_gemm_a8w8_mxscale
        )
        weight = data[7] if self._bpreshuffle else data[1]
        return run, (data[0], weight, data[3], data[4]), data[6]

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        # The shared run() reloads saved configs after pre_process. Reapply
        # the requested scale filter before reporting standalone benchmarks.
        if isinstance(args.run_config, str) and args.w_scale_block is not None:
            self.untunedf = self._normalize_rows(
                self.untunedf, blocks=args.w_scale_block
            )
        required = {"libtype", "kernelId", "splitK"}
        present = required.intersection(self.untunedf.columns)
        if present and present != required:
            raise ValueError(
                f"Saved BMM config is missing {sorted(required - present)}"
            )
        results = []
        for seed, (_, row) in enumerate(self.untunedf.iterrows(), start=1):
            shape = ",".join(f"{name}={row[name]}" for name in self.keys)
            allowed, description = self._get_run_config_err_ratio_limit(row, args)
            try:
                prepare = self._saved_benchmark if present else self._default_benchmark
                run, operands, ref = prepare(row, seed)
                out, us = run_perftest(
                    run, *operands, **self._perf_kwargs(args, int(row["m"]))
                )
                err = checkAllclose(
                    out,
                    ref,
                    rtol=1e-2,
                    atol=1e-2,
                    tol_err_ratio=allowed,
                    msg=f"run_config {shape}",
                    printLog=args.verbose,
                )
                if (
                    not math.isfinite(us)
                    or us <= 0
                    or not math.isfinite(err)
                    or err > allowed
                ):
                    raise RuntimeError(
                        f"us={us}, errRatio={err}, allowed={description}"
                    )
                results.append(
                    {"shape": shape, "e2e_us": us, "errRatio": err, "status": "ok"}
                )
            except Exception as exc:  # noqa: BLE001
                results.append({"shape": shape, "e2e_us": -1, "status": f"error:{exc}"})
        return results

    def _iter_opus_tasks(self, row, seed, args):
        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        for kid, splits in self.opus_policy.items():
            if _kid_block(kid) != row["w_scale_block"]:
                continue
            for split in _applicable(kid, b, m, n, k, args.pool, split_ks=splits):
                yield make_bmm_tuning_task(
                    (tuple(row[name] for name in self.keys), kid, split, ""),
                    gen_bmm_mxscale_data,
                    (b, m, n, k, seed, dtypes.bf16, kid, split),
                    run_bmm_mxscale_bench,
                    (BMM_BENCH_KEYS, kid, split),
                    self._perf_kwargs(args, m),
                    ref_index=6,
                    output_index=2,
                )

    def _iter_tuning_tasks(self, row, seed, args):
        yield from self._iter_opus_tasks(row, seed, args)

    def tune(self, untunedf, tunedf, args):
        tasks, tasks_data = [], []
        for seed, (_, row) in enumerate(untunedf.iterrows(), start=1):
            candidates = list(self._iter_tuning_tasks(row, seed, args))
            if args.verbose:
                logger.info("%s: %d candidates", tuple(row[self.keys]), len(candidates))
            tasks.extend(candidates)
            tasks_data.append((len(candidates), ()))
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_data,
            mp_num=args.mp,
            shape_grouped=args.shape_grouped,
            err_ratio=args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = OpusBmmMxscaleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
