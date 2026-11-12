# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MXFP4 a4w4 port tuner (--mxfp4-flydsl)."""

import argparse
import functools
import math
import os
from argparse import ArgumentTypeError
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_connections
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import torch

import aiter
from aiter import (
    ActivationType,
    QuantType,
    dtypes,
)
from aiter.fused_moe import (
    AsmMoeMode,
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2_fw,
    moe_sorting,
)
from aiter.jit.core import (
    AITER_ROOT_DIR,
)
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    resolve_swiglu_limit,
)
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_stage2_v2_kernels,
    runtime_swiglu_limit,
)
from aiter.ops.flydsl.mxfp4_kname import (
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    parse_g2_kname_any,
)
from aiter.ops.quant import (
    per_1x32_mx_quant_hip,
)
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_weight,
)
from aiter.test_common import (
    cosine_diff_compare,
)
from aiter.tuning.moe.fused_moe import (
    FmoeTuner,
    _frame_keeping_watts,
    _mxfp4_search_mode,
)
from aiter.tuning.moe.mxfp4_staged_search import (
    DEFAULT_SCREEN_TOPK,
    AccumulationMode,
    GEMM2ScreeningKey,
    PipelinePairKey,
    build_staged_candidate_plan,
    reference_consumer_order,
    screening_shortlist,
    select_reference_producer,
)
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    populate_v2_intermediate_from_ref as _v2_populate_stage2,
)
from aiter.utility.dtypes import str2ActivationType, str2Dtype
from aiter.utility.mp_tuner import _is_accelerator_error

if TYPE_CHECKING:
    from types import SimpleNamespace

    from aiter.tuning.moe.mxfp4_staged_search import StagedSearchPlan


def _parse_tuning_type(value):
    if isinstance(value, (torch.dtype, ActivationType, QuantType)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"unsupported tuning type: {value!r}")  # noqa: TRY004

    namespace, separator, name = value.strip().rpartition(".")
    try:
        if namespace == "torch" and separator:
            parsed = getattr(torch, name)
        elif namespace == "ActivationType" and separator:
            parsed = str2ActivationType(name)
        elif (namespace == "QuantType" and separator) or not separator:
            parsed = str2Dtype(name)
        else:
            raise ValueError
    except (ArgumentTypeError, AttributeError, TypeError, ValueError):
        raise ValueError(f"unsupported tuning type: {value!r}") from None

    if not isinstance(parsed, (torch.dtype, ActivationType, QuantType)):
        raise ValueError(f"unsupported tuning type: {value!r}")  # noqa: TRY004
    return parsed


class Mxfp4AccuracyError(RuntimeError):
    """An MXFP4 candidate failed its numerical accuracy gate."""


MXFP4_FINAL_GLOBAL_TIMING_GATE = 0.01


def _parse_fast_scan_iters(value):
    """Require enough profiler samples for ``get_trace_perf``."""
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 2") from exc
    if value < 2:
        raise argparse.ArgumentTypeError("must be >= 2 for profiler timing")
    return value


def _normalized_latency_difference(latency, fastest_latency):
    if fastest_latency > 0:
        return (latency - fastest_latency) / fastest_latency
    return 0.0 if latency == fastest_latency else math.inf


def _parse_candidate_accuracy(value):
    """Numeric accuracy ratio from a raw value; None when unavailable."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        percent = text.endswith("%")
        try:
            value = float(text[:-1] if percent else text)
        except ValueError:
            return None
        if percent:
            value /= 100.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_accuracy_error(candidate):
    """Numeric accuracy error for funnel decisions.

    ``_run_candidate`` records the raw ratio in the private ``_err`` key, while
    the tuned CSV keeps upstream's percent string in ``err1``. Accept both so
    funnel decisions never depend on the display format.
    """
    for key in ("_err", "err1"):
        if key in candidate:
            value = _parse_candidate_accuracy(candidate[key])
            if value is not None:
                return value
    return math.inf


def _select_mxfp4_performance_finalists(candidates, finalist_count):
    """Select the lowest-latency coarse survivors, including the fastest one."""
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["us"]),
            candidate["kernelName1"],
            candidate["kernelName2"],
        ),
    )[:finalist_count]


def _select_mxfp4_finalist(candidates, timing_gate=MXFP4_FINAL_GLOBAL_TIMING_GATE):
    """Select the most accurate finalist inside a global fastest-latency band."""
    if not candidates:
        return None
    fastest_latency = min(float(candidate["us"]) for candidate in candidates)
    performance_equivalent = [
        candidate
        for candidate in candidates
        if _normalized_latency_difference(float(candidate["us"]), fastest_latency)
        <= timing_gate
    ]
    return min(
        performance_equivalent,
        key=lambda candidate: (
            _candidate_accuracy_error(candidate),
            float(candidate["us"]),
            candidate["kernelName1"],
            candidate["kernelName2"],
        ),
    )


def _select_mxfp4_accuracy_promotion(candidates, performance_finalists):
    """Promote the most accurate candidate not already selected for speed."""
    selected = {
        (candidate["kernelName1"], candidate["kernelName2"])
        for candidate in performance_finalists
    }
    remaining = [
        candidate
        for candidate in candidates
        if (candidate["kernelName1"], candidate["kernelName2"]) not in selected
    ]
    if not remaining:
        return None
    return min(
        remaining,
        key=lambda candidate: (
            _candidate_accuracy_error(candidate),
            float(candidate["us"]),
            candidate["kernelName1"],
            candidate["kernelName2"],
        ),
    )


class Mxfp4FlydslTuner(FmoeTuner):
    """Tune the FlyDSL mxfp4 a4w4 *port* (flydsl_mxmoe_g{1,2}_a4w4_*) as one coupled
    unit.

    By default, prune GEMM1 using M_est = ceil(token * topk / expert) and screen
    the pruned stages (staged search); with --fast-scan and no explicit mode,
    every pruned pair is scanned instead. Pruning reduces candidate evaluation
    work and is expected to shorten tuning wall time; the actual speedup has not
    been measured. Kernel performance and winner retention on unseen shapes
    require separate validation. Use --mxfp4-search-mode full to enumerate all
    statically supported GEMM1 variants.
    """

    SUPPORTS_FAST_SCAN = True
    SUPPORTS_MXFP4_SEARCH_MODE = True
    KERNEL_NAME_COLUMNS: ClassVar[tuple[str, ...]] = ("kernelName1", "kernelName2")
    TAG_ROWS_DROPPED: ClassVar[str] = "fallback"

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--fast-scan",
            action="store_true",
            required=False,
            help="Scan every candidate with lightweight timing. Candidates are "
            "accuracy-checked and scored by summed active GPU-kernel time; "
            "inaccurate candidates are skipped, and only the fastest valid "
            "candidates are retimed with --fast-scan-final-iters. "
            "--select power retimes every accurate candidate and ranks them "
            "by package energy instead of that latency shortlist.",
        )
        self.parser.add_argument(
            "--fast-scan-warmup-accuracy-checks",
            type=int,
            default=1,
            help="Untimed warm-up launches whose outputs are accuracy-checked "
            "before the coarse measurement (default: 1). Finalists reuse the "
            "coarse-stage warm state without another warm-up.",
        )
        self.parser.add_argument(
            "--fast-scan-iters",
            type=_parse_fast_scan_iters,
            default=10,
            help="Profiled launches per candidate after the accuracy warm-up in "
            "the coarse fast-scan pass (default: 10; must be >= 2).",
        )
        self.parser.add_argument(
            "--fast-scan-final-iters",
            type=_parse_fast_scan_iters,
            default=100,
            help="Profiled launches per finalist with no additional warm-up "
            "(default: 100; must be >= 2).",
        )
        self.parser.add_argument(
            "--fast-scan-finalists",
            type=int,
            default=5,
            help="Performance candidates to retime with "
            "--fast-scan-final-iters (default: 5). One additional non-duplicate "
            "accuracy candidate is promoted when available.",
        )
        for action in self.parser._actions:
            if action.dest == "timeout":
                action.help = (
                    "best-effort per-operation SIGALRM timeout in seconds (default 1800); "
                    "needs control to return to Python. No parent or shape hard deadline; "
                    "a still-live permanent hang may wait indefinitely."
                )

    input_metadata: ClassVar[tuple[str, ...]] = ("gate_mode",)
    PROFILE_HELP: ClassVar[str] = (
        "ordinary fMoE profiling output; the dedicated --mxfp4-flydsl tuner "
        "does not write stage-ranking or full-pair profile artifacts"
    )

    @staticmethod
    def _capture_configuration() -> "dict[str, float | None]":
        limit, _source = resolve_swiglu_limit()
        return {"swiglu_limit": limit}

    def _invocation_configuration(
        self, args: "argparse.Namespace | SimpleNamespace | None" = None
    ) -> "dict[str, float | None]":
        # Callers keep these variables unchanged for the whole run.
        if args is not None and hasattr(args, "mxfp4_configuration"):
            self._configuration = args.mxfp4_configuration
        elif args is not None or not hasattr(self, "_configuration"):
            self._configuration = self._capture_configuration()
        if args is not None:
            args.mxfp4_configuration = self._configuration
        return self._configuration

    def _require_distinct_input_output(
        self, args: "argparse.Namespace | SimpleNamespace"
    ) -> "None":
        input_file = args.untune_file
        output_file = self.get_out_file(args.tune_file)
        same_file = os.path.realpath(input_file) == os.path.realpath(output_file)
        if not same_file and os.path.exists(input_file) and os.path.exists(output_file):
            same_file = os.path.samefile(input_file, output_file)
        if same_file:
            self.parser.error(
                "--mxfp4-flydsl requires different input and output files "
                "(-i/--untune_file and -o/--tune_file)"
            )

    def parse_args(self) -> "argparse.Namespace":
        args = super().parse_args()
        self._require_distinct_input_output(args)
        self._configuration = self._capture_configuration()
        args.mxfp4_configuration = self._configuration
        return args

    def pre_process(self, args: "argparse.Namespace | SimpleNamespace") -> "None":
        self._require_distinct_input_output(args)
        self._invocation_configuration(args)
        return super().pre_process(args)

    def _unsupported_row_reason(
        self, row: "Mapping[str, object] | pd.Series"
    ) -> "str | None":
        for field, expected in (
            ("dtype", dtypes.bf16),
            ("q_dtype_a", dtypes.fp4x2),
            ("q_dtype_w", dtypes.fp4x2),
            ("q_type", QuantType.per_1x32),
        ):
            if str(row.get(field)) != str(expected):
                return f"{field} must be {expected}, got {row.get(field)!r}"
        for field, allowed in (
            ("use_g1u1", ("1", "1.0", "true")),
            ("doweight_stage1", ("0", "0.0", "false")),
        ):
            if str(row.get(field)).lower() not in allowed:
                return f"{field} must be {allowed[0]}, got {row.get(field)!r}"
        activation = str(row.get("act_type")).rsplit(".", 1)[-1].lower()
        if activation not in ("silu", "situv2", "swiglu"):
            return f"unsupported act_type: {row.get('act_type')!r}"
        if (
            activation == "situv2"
            and self._invocation_configuration()["swiglu_limit"] is not None
        ):
            return "MXMOE cannot apply a configured swiglu_limit to Situv2"
        gate = row.get("gate_mode", "separated")
        if pd.isna(gate) or str(gate).strip() == "":
            gate = "separated"
        if str(gate).rsplit(".", 1)[-1].lower() != "separated":
            return f"gate_mode must be separated, got {gate!r}"
        for field in ("token", "model_dim", "inter_dim", "expert", "topk"):
            try:
                value = int(row.get(field, 0))
                if value > 0 and value == float(row[field]):
                    continue
            except (TypeError, ValueError, OverflowError):
                pass
            return f"{field} must be a positive integer, got {row.get(field)!r}"
        if int(row["topk"]) > int(row["expert"]):
            return "topk must not exceed expert count"
        return None

    def _drop_superseded_rows(
        self, tunedf: "pd.DataFrame", dedup_keys: "list[str]"
    ) -> "pd.DataFrame":
        """Drop only rows the runtime can never load.

        get_cfg_2stages uses the first row per key, and a later one only as an
        activation fallback, which skips MXFP4 and CK 2-stage kernels. A row
        without a kernel is dropped when a configured row has its key, since the
        runtime would otherwise serve whichever of the two comes first.
        """
        del dedup_keys
        configured = self._names_kernel(tunedf)
        keys = tunedf.loc[configured, self.keys].astype(str)
        later = keys.duplicated(keep="first").reindex(tunedf.index, fill_value=False)
        configured_keys = set(keys.itertuples(index=False, name=None))
        placeholder = ~configured & pd.Series(
            [
                key in configured_keys
                for key in tunedf[self.keys]
                .astype(str)
                .itertuples(index=False, name=None)
            ],
            index=tunedf.index,
            dtype=bool,
        )
        kernel1 = tunedf.get("kernelName1", pd.Series("", index=tunedf.index))
        kernel2 = tunedf.get("kernelName2", pd.Series("", index=tunedf.index))
        no_fallback = (
            kernel1.fillna("").astype(str).str.contains("moe_ck2stages")
            | kernel1.map(_is_mxfp4_kname)
            | kernel2.map(_is_mxfp4_kname)
        )
        return tunedf[~(later & no_fallback) & ~placeholder]

    def _already_tuned_mask(self, dedup_cols: "list[str]") -> "pd.Series":
        mask = super()._already_tuned_mask(dedup_cols)
        unsupported = self.untunedf.apply(
            lambda row: self._unsupported_row_reason(row) is not None, axis=1
        )
        return mask & ~unsupported

    @staticmethod
    def _runtime_device_identity(gpu: "int") -> "tuple[str, int]":
        properties = torch.cuda.get_device_properties(gpu)
        arch = str(properties.gcnArchName).split(":", 1)[0].lower()
        cu = int(properties.multi_processor_count)
        if arch != "gfx950":
            raise ValueError(f"MXFP4 tuning GPU{gpu} must be gfx950, got {arch}")
        return arch, cu

    def get_gfx(self) -> "str":
        # Single-slot pruned and full runs use the current GPU.
        return (
            str(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).gcnArchName
            )
            .split(":", 1)[0]
            .lower()
        )

    def get_cu_num(self) -> "int":
        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )

    def _preflight_devices(
        self, mp_num: "int", ngpu: "int", *, staged: "bool" = False
    ) -> "list[tuple[str, int]]":
        if ngpu < 1:
            raise ValueError("MXFP4 tuning requires a visible runtime GPU")
        selected = (
            [torch.cuda.current_device()]
            if mp_num == 1 and not staged
            else range(mp_num)
        )
        return [self._runtime_device_identity(gpu) for gpu in selected]

    def _summary_shape_keys(self) -> "list[str]":
        # Rows may run on any scheduled GPU; output keeps that GPU's gfx/CU.
        return [key for key in self.keys if key not in ("gfx", "cu_num")]

    def _normalize_tuned_csv(self, tunedf: "pd.DataFrame") -> "pd.DataFrame":
        tunedf = self._apply_tag_rule(super()._normalize_tuned_csv(tunedf))
        for column in self.columns:
            if column not in tunedf:
                tunedf[column] = "" if column.startswith("kernelName") else 0
        for column in (
            "block_m",
            "ksplit",
            "us1",
            "us2",
            "run_1stage",
            "xbf16",
            "flat",
            "tflops",
            "bw",
            "nt",
        ):
            if column in tunedf:
                tunedf[column] = pd.to_numeric(tunedf[column], errors="coerce").fillna(
                    0
                )
        tunedf = tunedf.drop(columns=list(self.input_metadata), errors="ignore")
        # Keep columns owned by other rows of shared files (nt, _tag, ...).
        return tunedf[
            self.columns + [column for column in tunedf if column not in self.columns]
        ]

    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **FmoeTuner.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/model_configs/kimik2_fp4_untuned_fmoe.csv",
        "tune_file": f"{AITER_ROOT_DIR}/aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv",
        "config_env_name": "AITER_CONFIG_FMOE",
    }

    #: Key columns holding a torch dtype rather than a plain scalar.
    DTYPE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"dtype", "q_dtype_a", "q_dtype_w"}
    )

    @staticmethod
    def _g1_kname(
        bm,
        use_nt,
        inline_quant,
        act="silu",
        prefetch_hidden=False,
        bn=256,
        bk=256,
        k_wave=1,
        xcd_swizzle=0,
        num_waves=4,
        a_dtype="fp4",
        out_dtype="fp4",
    ):
        # flydsl_mxmoe_g1_a{4,8}w4_<BM>x<BN>x<BK>[_f16in][_hpf][_nt][_fp8out]
        #   [_situv2|_swiglu][_kw<n>][_xcd<n>][_w2];
        # token order must match _parse_mxfp4_g1_kname in mxfp4_kname.py.
        # The tuner only enumerates A4W4; A8W4 names serve the coupled tests.
        if prefetch_hidden and not inline_quant:
            raise ValueError("hidden prefetch requires inline quantization")
        a_bits = 8 if a_dtype == "fp8" else 4
        name = f"flydsl_mxmoe_g1_a{a_bits}w4_{bm}x{bn}x{bk}"
        if inline_quant:
            name += "_f16in"
        if prefetch_hidden:
            name += "_hpf"
        if use_nt:
            name += "_nt"
        if out_dtype == "fp8":
            name += "_fp8out"
        if act == "situv2":
            name += "_situv2"
        elif act == "swiglu":
            name += "_swiglu"
        if k_wave > 1:
            name += f"_kw{int(k_wave)}"
        if xcd_swizzle:
            name += f"_xcd{int(xcd_swizzle)}"
        if num_waves == 2:
            name += "_w2"
        return name

    # GEMM1 axes swept on top of (BM, use_nt, inline_quant). Their constraints
    # interact (BN64 implies BM32 non-inline; num_waves==2 implies
    # BN64; k_wave>1 implies BM32 non-inline and num_waves*k_wave<=8), so
    # _g1_variants enumerates the cross-product and lets the kernel's own
    # _assert_supported reject the rest instead of duplicating that logic.
    _G1_BN = (64, 128, 256)
    _G1_XCD_SWIZZLE = (0, 2, 4)
    _G1_K_WAVE = (1, 2, 4)
    _G1_NUM_WAVES = (4, 2)

    # Below this M_est, staged search validates every pruned pair, unscreened.
    _STAGED_SMALL_BATCH_M_EST = 4

    @staticmethod
    def _m_est(row: dict[str, Any]) -> int:
        """Estimated tokens per expert: ceil(token * topk / expert)."""
        ne = int(row["expert"])
        return (int(row["token"]) * int(row["topk"]) + ne - 1) // ne

    @staticmethod
    def _g1_matches_m_est(g1: dict[str, Any], m_est: int) -> bool:
        """Apply M_est allowlists; require an explicit rule for each BM family."""
        bm = g1["bm"]
        if bm == 16:
            return m_est < 16
        if bm == 32:
            return 4 <= m_est <= 128 and (
                m_est <= 32
                or (g1["num_waves"] == 4 and g1["k_wave"] == 1 and not g1["use_nt"])
            )
        if bm == 64:
            return m_est >= 16 and (m_est <= 64 or not g1["use_nt"])
        if bm == 128:
            return m_est >= 64
        raise ValueError(
            f"Missing GEMM1 pruning rule for BM{bm}; add an M_est pruning rule "
            "for this BM in Mxfp4FlydslTuner._g1_matches_m_est before pruning by M_est."
        )

    def _g1_variants(
        self, row: dict[str, Any], full_search: bool = False
    ) -> list[dict[str, Any]]:
        """Supported _g1_kname kwargs, pruned by M_est unless full_search is set."""
        from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported
        from aiter.ops.flydsl.mxfp4_kname import MXFP4_G1_VARIANTS

        ne, h, e = int(row["expert"]), int(row["model_dim"]), int(row["inter_dim"])
        topk = int(row["topk"])
        act = self._row_act(row)
        out = []
        for bm, use_nt, inline_quant in sorted(MXFP4_G1_VARIANTS["fp4"]):
            for bn in self._G1_BN:
                for num_waves in self._G1_NUM_WAVES:
                    for k_wave in self._G1_K_WAVE:
                        # Hidden prefetch hoists the next K-tile's hidden_states
                        # load, which only the inline-quant path performs.
                        for hpf in (False, True) if inline_quant else (False,):
                            for xcd in self._G1_XCD_SWIZZLE:
                                try:
                                    _assert_supported(
                                        D_HIDDEN=h,
                                        D_INTER=e,
                                        BM=bm,
                                        BN=bn,
                                        BK=256,
                                        use_nt=use_nt,
                                        inline_quant=inline_quant,
                                        prefetch_hidden=hpf,
                                        act=act,
                                        situ_beta=DEFAULT_SITUV2_BETA,
                                        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
                                        num_waves=num_waves,
                                        k_wave=k_wave,
                                        native_scale_layout=bm == 16,
                                    )
                                except NotImplementedError:
                                    continue
                                out.append(
                                    {
                                        "bm": bm,
                                        "use_nt": use_nt,
                                        "inline_quant": inline_quant,
                                        "act": act,
                                        "prefetch_hidden": hpf,
                                        "bn": bn,
                                        "bk": 256,
                                        "k_wave": k_wave,
                                        "xcd_swizzle": xcd,
                                        "num_waves": num_waves,
                                    }
                                )
        if full_search:
            return out
        m_est = self._m_est(row)
        kept = [g1 for g1 in out if self._g1_matches_m_est(g1, m_est)]
        if len(kept) < len(out):
            # Name the shape in full: under --mp the workers' stdout interleaves,
            # and M_est alone does not identify a row (different shapes share it).
            print(
                f"[mxfp4-port] pruned G1: pid={os.getpid()} "
                f"token={int(row['token'])} model_dim={h} inter_dim={e} "
                f"expert={ne} topk={topk} act={act} "
                f"M_est={m_est} valid={len(out)} kept={len(kept)}",
                flush=True,
            )
        return kept

    @staticmethod
    def _row_act(row):
        """The GEMM1 activation tag for this row.

        Folding Situv2 into Silu makes the tuner write a name without
        `_situv2`, which fused_moe then rejects on activation mismatch. Anything
        MXMOE has no kernel for (Gelu, GeluTanh, ...) must raise rather than fall
        through to Silu, or the row is tuned and validated against a SiLU
        reference and ships as a silently wrong activation.
        """
        act_type = str(row.get("act_type", ""))
        for suffix, tag in (
            ("Situv2", "situv2"),
            ("Swiglu", "swiglu"),
            ("Silu", "silu"),
        ):
            if act_type.endswith(suffix):
                return tag
        raise ValueError(f"no MXMOE GEMM1 kernel for activation {act_type!r}")

    @staticmethod
    def _g2_kname(bm, use_nt, epilog):
        # flydsl_mxmoe_g2_a4w4_<BM>x256x256[_atomic[_nt] | _f4out | _cshuffle].
        name = f"flydsl_mxmoe_g2_a4w4_{bm}x256x256"
        if epilog == "atomic":
            name += "_atomic" + ("_nt" if use_nt else "")
        elif epilog == "nonatomic_mxfp4":
            name += "_f4out"
        elif epilog == "nonatomic_cshuffle":
            name += "_cshuffle"
        return name

    def _candidate_row(self, row, bm, kn1, kn2):
        cand = {k: row[k] for k in self.keys}
        cand.update(
            {
                "block_m": bm,
                "ksplit": 0,
                "us1": 0,
                "kernelName1": kn1,
                "err1": 0,
                "us2": 0,
                "kernelName2": kn2,
                "err2": 0,
                "us": 0,
                "run_1stage": 0,
                "xbf16": 0,
                "flat": AsmMoeMode.SORTED.value,
                "tflops": 0,
                "bw": 0,
            }
        )
        return cand

    def _candidate_rows(
        self, row: dict[str, Any], full_search: bool = False
    ) -> list[dict[str, Any]]:
        cands = []
        for g1 in self._g1_variants(row, full_search=full_search):
            bm = g1["bm"]
            kn1 = self._g1_kname(**g1)
            # a4w4 pairs flydsl_mxmoe_g1_* with flydsl_moe2_layout_* only. The
            # native flydsl_mxmoe_g2_a4w4_* family is deliberately not proposed:
            # its BK=256 contraction requires D_INTER % 256 == 0, so it cannot
            # serve inter_dim like 384, and the layout family covers the same
            # tile space with the sort_block_m contract the port's intermediate
            # needs. Only native SBM==tile_m==bm variants are used; re-tiling
            # (tile_m < bm) is not supported by that layout.
            for kn2v, kp in get_flydsl_stage2_v2_kernels(
                "fp4",
                "fp4",
                "bf16",
                bm,
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
            ).items():
                if kp["tile_m"] != bm:
                    continue
                cands.append(self._candidate_row(row, bm, kn1, kn2v))
        return cands

    def _staged_candidate_plan(self, row: "dict[str, object]") -> "StagedSearchPlan":
        reason = self._unsupported_row_reason(row)
        if reason is not None:
            raise ValueError(reason)
        return build_staged_candidate_plan(
            row, self._candidate_rows(row, full_search=False)
        )

    @staticmethod
    def _prepare_case(token, model_dim, inter_dim, expert, topk, dtype, a_dtype="fp4"):
        data = FmoeTuner.generate_data(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            dtype,
            dtypes.fp8 if a_dtype == "fp8" else dtypes.fp4x2,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            16,
            device="cuda",
        )
        if a_dtype == "fp8":
            # generate_data keeps BF16 A for the production reference; the
            # coupled A8W4 pair is compared against actual MXFP8 A instead.
            data["a1_qt"], data["a1_scale"] = per_1x32_mx_quant_hip(
                data["input"],
                quant_dtype=dtypes.fp8,
                scale_type=dtypes.fp8_e8m0,
                shuffle=False,
            )
        # True a4w4 runs the SEPARATED gate/up layout (gemm1 interleave=False),
        # so prepare weights/scales with is_guinterleave=False; a8w4 runs the
        # interleaved layout (is_guinterleave=True). Feeding either layout to
        # the other port produces garbage. w2 (down-proj, no gate/up) is
        # layout-invariant.
        interleave = a_dtype == "fp8"
        data["w1_a16"] = shuffle_weight(
            data["w1_qt"], (16, 16), is_guinterleave=interleave, gate_up=True
        )
        data["w1s_a16"] = shuffle_scale(
            data["w1_scale"], expert, is_guinterleave=interleave, gate_up=True
        )
        data["w2_a16"] = shuffle_weight(
            data["w2_qt"], (16, 16), is_guinterleave=False, gate_up=False
        )
        data["w2s_a16"] = shuffle_scale(
            data["w2_scale"], expert, is_guinterleave=False, gate_up=False
        )
        return data

    # Every data entry _port_e2e reads; timing rotates exactly these tensors.
    _PORT_E2E_INPUTS: ClassVar[tuple[str, ...]] = (
        "input",
        "w1_a16",
        "w2_a16",
        "w1s_a16",
        "w2s_a16",
        "topk_ids",
        "topk_weights",
    )

    @staticmethod
    def _port_e2e(data, kn1, kn2, topk, ne, h, dtype, swiglu_limit=None):
        # kn2 may name either gemm2 family (path B or native mxmoe).
        _g2 = parse_g2_kname_any(kn2)
        atomic = _g2["atomic"]
        p1 = _parse_mxfp4_g1_kname(kn1)
        # One block_m for the whole pipeline, taken from kernelName1 -- mirrors
        # production, where metadata.block_m comes from kernelName1 and feeds the
        # sort, the prequant, stage1 and stage2 alike. _candidate_rows only pairs
        # a g1 with a g2 of the same tile_m, so the two always agree today;
        # assert it rather than reading both, so relaxing that pairing cannot
        # silently desync the prequant's block stride from the sorted_ids it walks.
        BM = p1["BM"]
        assert (
            _g2["BM"] == BM
        ), f"block_m mismatch between {kn1!r} (BM={BM}) and {kn2!r} (BM={_g2['BM']})"
        M = data["input"].shape[0]
        sti, sw, sei, nvi, moe_buf, m_indices, reverse_sorted = moe_sorting(
            data["topk_ids"],
            data["topk_weights"],
            ne,
            h,
            dtype,
            block_size=BM,
            accumulate=atomic,
            output_aux="opus",
        )
        moe_out = moe_buf if moe_buf.numel() else torch.empty((M, h), dtype=dtype)
        stage1_input = data["input"]
        stage1_scale = None
        # Keep the tuner aligned with the production fused_moe_2stages pipeline:
        # BM16 quantizes inline; the other variants consume the fused Opus
        # prequant output in their GEMM1 activation precision.
        if BM != 16:
            quant = (
                aiter.fused_dynamic_mxfp8_quant_moe_sort
                if p1["a_dtype"] == "fp8"
                else aiter.fused_dynamic_mxfp4_quant_moe_sort
            )
            stage1_input, stage1_scale = quant(
                input=data["input"],
                sorted_ids=sti,
                num_valid_ids=nvi,
                token_num=M,
                topk=topk,
                block_size=BM,
                sorted_weights=sw,
                num_experts_upper_bound=ne,
            )
        inter_q, inter_s = _mxfp4_a4w4_stage1_fw(
            stage1_input,
            data["w1_a16"],
            data["w2_a16"],
            sti,
            sei,
            nvi,
            None,
            topk,
            block_m=BM,
            a1_scale=stage1_scale,
            w1_scale=data["w1s_a16"],
            kernelName1=kn1,
            interleave=p1["a_dtype"] == "fp8",
            m_indices=m_indices,
            moe_buf=moe_buf,
            # _torch_ref runs SiTUv2 at run_torch_moe_stage1's default betas, so
            # the kernel has to use the same pair or the candidate fails the
            # accuracy gate for a reason that is not the kernel's fault.
            situ_beta=DEFAULT_SITUV2_BETA if p1["act"] == "situv2" else 1.0,
            situ_linear_beta=(
                DEFAULT_SITUV2_LINEAR_BETA if p1["act"] == "situv2" else 1.0
            ),
            swiglu_limit=swiglu_limit,
        )
        return _mxfp4_a4w4_stage2_fw(
            inter_q,
            data["w1_a16"],
            data["w2_a16"],
            sti,
            sei,
            nvi,
            moe_out,
            topk,
            w2_scale=data["w2s_a16"],
            a2_scale=inter_s,
            block_m=BM,
            sorted_weights=sw,
            topk_weights=data["topk_weights"],
            kernelName2=kn2,
            reverse_sorted=reverse_sorted,
        )

    @staticmethod
    def _torch_ref(data, topk, dtype, activation, swiglu_limit=None):
        ref1 = FmoeTuner.run_torch_moe_stage1(
            data["a1_qt"],
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            data["a1_scale"],
            data["w1_scale"],
            dtype=dtype,
            activation=activation,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            topk=topk,
            swiglu_limit=swiglu_limit,
        )
        return FmoeTuner.run_torch_moe_stage2(
            ref1,
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            a2_scale=None,
            w2_scale=data["w2_scale"],
            dtype=dtype,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
        )

    def _run_candidate(
        self,
        row,
        candidate,
        args,
        data=None,
        reference=None,
        num_warmup=None,
        num_iters=None,
        final_timing=True,
    ):
        """Time one pair. ``final_timing`` is False for a coarse screening pass,
        whose time does not pick the winner."""
        from aiter.test_common import run_perftest

        ne, h, e = int(row["expert"]), int(row["model_dim"]), int(row["inter_dim"])
        token, topk = int(row["token"]), int(row["topk"])
        dtype = dtypes.bf16
        kn1, kn2 = candidate["kernelName1"], candidate["kernelName2"]
        activation = {
            "situv2": ActivationType.Situv2,
            "swiglu": ActivationType.Swiglu,
            "silu": ActivationType.Silu,
        }[self._row_act(row)]
        swiglu_limit = self._invocation_configuration(args)["swiglu_limit"]
        fast_scan = bool(getattr(args, "fast_scan", False))
        # Never let a finalist retime erase a larger error already observed for
        # this candidate during the coarse pass. `_err` holds the exact ratio;
        # err1/err2 are the rounded display strings, so only fall back to them
        # when the raw value is unavailable.
        previously_observed_errors = []
        raw_error = _parse_candidate_accuracy(candidate.get("_err"))
        if raw_error is not None:
            previously_observed_errors.append(raw_error)
        else:
            for field in ("err1", "err2"):
                observed = _parse_candidate_accuracy(candidate.get(field))
                if observed is not None:
                    previously_observed_errors.append(observed)
        if data is None:
            data = self._prepare_case(token, h, e, ne, topk, dtype)
        requested_warmup = int(args.warmup if num_warmup is None else num_warmup)
        measured_iters = int(args.iters if num_iters is None else num_iters)

        def validate_output(out, phase):
            accuracy_error = cosine_diff_compare(
                reference,
                out,
                msg=f"port[{kn1}+{kn2}]",
                printLog=False,
            )
            # NaN must reject explicitly: `nan > errRatio` is False, so a
            # candidate producing garbage would otherwise pass the gate and,
            # being fast, win.
            if (
                accuracy_error is None
                or not math.isfinite(float(accuracy_error))
                or float(accuracy_error) > args.errRatio
            ):
                raise Mxfp4AccuracyError(
                    f"{phase} cosine err_ratio {accuracy_error} > {args.errRatio}"
                )
            return float(accuracy_error)

        if reference is not None:
            # Fast scan: every untimed warm-up launch is validated against the
            # shared reference before the coarse measurement starts.
            accuracy_errors = []
            for check_index in range(max(0, requested_warmup)):
                out = self._port_e2e(
                    data, kn1, kn2, topk, ne, h, dtype, swiglu_limit=swiglu_limit
                )
                accuracy_errors.append(
                    validate_output(
                        out,
                        f"warmup-and-accuracy-check {check_index + 1}/"
                        f"{requested_warmup}",
                    )
                )
            # Warm-up launches have already run and been validated. Passing zero
            # prevents run_perftest from launching a duplicate warm-up stage.
            timed_warmup = 0
        elif fast_scan:
            err = 0.0
            timed_warmup = requested_warmup
        else:
            out = self._port_e2e(
                data, kn1, kn2, topk, ne, h, dtype, swiglu_limit=swiglu_limit
            )
            ref = self._torch_ref(
                data, topk, dtype, activation, swiglu_limit=swiglu_limit
            )
            err = cosine_diff_compare(ref, out, msg=f"port[{kn1}+{kn2}]")
            if (
                err is None
                or not math.isfinite(float(err))
                or float(err) > args.errRatio
            ):
                raise RuntimeError(f"cosine err_ratio {err} > {args.errRatio}")
            timed_warmup = requested_warmup
        # run_perftest rotates deep copies of its arguments, sized from the
        # positional tensors; anything a closure captures is never copied.
        inputs = self._PORT_E2E_INPUTS

        def timed_pipeline(*tensors):
            return self._port_e2e(
                dict(zip(inputs, tensors)),
                kn1,
                kn2,
                topk,
                ne,
                h,
                dtype,
                swiglu_limit=swiglu_limit,
            )

        timed_out, us = run_perftest(
            timed_pipeline,
            *(data[key] for key in inputs),
            num_warmup=timed_warmup,
            num_iters=measured_iters,
            # The fast-scan funnel owns warm-up explicitly: count every requested
            # profiler sample, keep one input set and skip the memory-probe launch.
            # Other modes rotate the inputs so each iteration reads cold data.
            profile_warm_iters=0 if fast_scan else 1,
            num_rotate_args=1 if fast_scan else 0,
        )
        if reference is not None:
            # run_perftest synchronizes before returning. Validate its last
            # output now, outside the profiler interval, to catch result
            # instability without changing reported kernel-active latency.
            accuracy_errors.append(validate_output(timed_out, "measured output"))
            err = max(accuracy_errors)
        if previously_observed_errors:
            err = max(float(err), *previously_observed_errors)
        us = round(float(us), 4)
        if not math.isfinite(us) or us <= 0:
            raise RuntimeError(f"invalid MXFP4 validation time {us}")
        if final_timing:
            self.policy_for(args).record(
                candidate,
                "watts",
                lambda: self._port_e2e(
                    data, kn1, kn2, topk, ne, h, dtype, swiglu_limit=swiglu_limit
                ),
                kernel_us=us,
            )
        # The pair is timed as one unit, so the fused-MoE estimate in calculate()
        # is the right roofline for it. Untuned rows keep dtypes as strings, while
        # calculate() looks bpe up by torch dtype.
        key = tuple(
            _parse_tuning_type(row[col]) if col in self.DTYPE_KEYS else row[col]
            for col in self.keys
        )
        tflops, bw = self.calculate((key, "", kn1, candidate["block_m"], us, err))
        candidate.update(
            {
                "us1": us,
                "us": us,
                "err1": f"{float(err):.1%}",
                "err2": f"{float(err):.1%}",
                # Raw ratio for the fast-scan funnel; err1/err2 keep the
                # upstream percent-string display format.
                "_err": float(err),
                "tflops": tflops,
                "bw": bw,
            }
        )
        return us

    def _precompile_candidates(self, row, candidates):
        """Populate the shared FlyDSL cache before GPU validation and timing.

        Compile-only workers use CPU dummy tensors, so they neither duplicate
        the shape fixture on the GPU nor inherit the parent's HIP context.  A
        failed precompile is only an optimization miss: the parent scan keeps
        its normal JIT fallback for that candidate.
        """
        try:
            from aiter.aot.flydsl.common import get_max_workers, run_jobs_parallel
            from aiter.aot.flydsl.mxfp4_moe import (
                compile_job_batch,
                jobs_from_rows,
            )

            candidate_rows = [dict(row, **candidate) for candidate in candidates]
            jobs = jobs_from_rows(candidate_rows, include_bias_variants=False)
            if not jobs:
                return 0, 0
            # jobs_from_rows always compiles the unclamped defaults (+inf for
            # SiLU, 7.0 for SwiGLU). Runtime launches pass the tuned clamp
            # instead, so without this the cache misses and the scan JITs
            # during scoring.
            swiglu_limit = self._invocation_configuration()["swiglu_limit"]
            if swiglu_limit is not None:
                for job in jobs:
                    if job["stage"] == 1:
                        job["swiglu_limit"] = runtime_swiglu_limit(
                            swiglu_limit, job["act"]
                        )
            worker_count = get_max_workers(len(jobs))
            batches = [jobs[index::worker_count] for index in range(worker_count)]
            batch_results = run_jobs_parallel(
                compile_job_batch,
                [{"jobs": batch} for batch in batches],
                # Runtime tuning can begin after a framework initialized HIP.
                # Fork clean compile-only children from Python's dedicated
                # fork server instead of inheriting the parent's accelerator
                # state.  Do not hide devices while starting that persistent
                # server: it would retain the temporary environment after the
                # parent restored it.
                start_method="forkserver",
            )
            failed = 0
            for batch, batch_result in zip(batches, batch_results):
                results = batch_result.get("results")
                if not isinstance(results, list) or len(results) != len(batch):
                    failed += len(batch)
                    continue
                failed += sum(result.get("compile_time") is None for result in results)
            print(
                f"[mxfp4-port] precompiled {len(jobs) - failed}/{len(jobs)} "
                "deduplicated FlyDSL kernels",
                flush=True,
            )
            return len(jobs), failed
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mxfp4-port] parallel precompile unavailable; "
                f"falling back to runtime JIT: {exc}",
                flush=True,
            )
            return 0, 0

    def _screening_limit(self) -> "float | None":
        return self._invocation_configuration()["swiglu_limit"]

    def _prepare_screening_case(
        self, row: dict[str, Any], block_m: int, atomic: bool
    ) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
        ne, h, e = int(row["expert"]), int(row["model_dim"]), int(row["inter_dim"])
        token, topk = int(row["token"]), int(row["topk"])
        data = self._prepare_case(token, h, e, ne, topk, dtypes.bf16)
        sorting = moe_sorting(
            data["topk_ids"],
            data["topk_weights"],
            ne,
            h,
            dtypes.bf16,
            block_size=block_m,
            accumulate=atomic,
            output_aux="opus",
        )
        return data, sorting

    @staticmethod
    def _prepare_gemm1_screening(
        data: dict[str, Any],
        sorting: tuple[torch.Tensor, ...],
        kn1: str,
        topk: int,
        swiglu_limit: float | None = None,
    ) -> tuple[
        Callable[..., None], tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor
    ]:
        """Return (launch, tensors, inter_q, inter_s); time launch(*tensors)."""
        from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

        cfg = _parse_mxfp4_g1_kname(kn1)
        bm = cfg["BM"]
        sti, sw, sei, nvi, moe_buf, m_indices, _reverse_sorted = sorting
        stage1_input, stage1_scale = data["input"], None
        if bm == 16:
            if stage1_input.dtype != dtypes.bf16:
                raise ValueError("GEMM1 BM16 inline quantization requires BF16 input")
        else:
            stage1_input, stage1_scale = aiter.fused_dynamic_mxfp4_quant_moe_sort(
                input=stage1_input,
                sorted_ids=sti,
                num_valid_ids=nvi,
                token_num=stage1_input.shape[0],
                topk=topk,
                block_size=bm,
                sorted_weights=sw,
                num_experts_upper_bound=data["w1_a16"].shape[0],
            )
            if stage1_input.dtype != dtypes.fp4x2 or stage1_scale is None:
                raise ValueError(
                    f"GEMM1 BM{bm} requires fused-prequantized FP4 input and scales"
                )
        situ_beta = DEFAULT_SITUV2_BETA if cfg["act"] == "situv2" else 1.0
        situ_linear_beta = DEFAULT_SITUV2_LINEAR_BETA if cfg["act"] == "situv2" else 1.0
        # Run the wrapper once untimed to allocate buffers; time the kernel only.
        inter_q, inter_s = _mxfp4_a4w4_stage1_fw(
            stage1_input,
            data["w1_a16"],
            data["w2_a16"],
            sti,
            sei,
            nvi,
            None,
            topk,
            block_m=bm,
            a1_scale=stage1_scale,
            w1_scale=data["w1s_a16"],
            kernelName1=kn1,
            m_indices=m_indices,
            moe_buf=moe_buf,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=runtime_swiglu_limit(swiglu_limit, cfg["act"]),
        )
        placeholder = torch.empty(0, device=stage1_input.device, dtype=torch.uint8)
        # Every tensor the kernel touches, so run_perftest can rotate them.
        tensor_args = {
            "a_quant": placeholder if bm == 16 else stage1_input,
            "a_scale_sorted_shuffled": placeholder if bm == 16 else stage1_scale,
            "w1_u8": data["w1_a16"].view(torch.uint8),
            "w1_scale_u8": data["w1s_a16"].view(torch.uint8),
            "sorted_expert_ids": sei,
            "cumsum_tensor": nvi,
            "m_indices": m_indices,
            "inter_sorted_quant": inter_q,
            "inter_sorted_shuffled_scale": inter_s,
            "hidden_states": stage1_input,
        }
        names = tuple(tensor_args)
        kernel = functools.partial(
            flydsl_mxfp4_gemm1,
            n_tokens=stage1_input.shape[0],
            NE=data["w1_a16"].shape[0],
            D_HIDDEN=data["w2_a16"].shape[1],
            D_INTER=data["w1_a16"].shape[1] // 2,
            topk=topk,
            BM=bm,
            BN=cfg["BN"],
            BK=cfg["BK"],
            use_nt=cfg["use_nt"],
            inline_quant=cfg["inline_quant"],
            prefetch_hidden=cfg["prefetch_hidden"],
            xcd_swizzle=cfg["xcd_swizzle"],
            num_waves=cfg["num_waves"],
            k_wave=cfg["k_wave"],
            interleave=False,
            a_dtype=cfg["a_dtype"],
            out_dtype=cfg["out_dtype"],
            native_scale_layout=bm == 16,
            act=cfg["act"],
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=runtime_swiglu_limit(swiglu_limit, cfg["act"]),
        )

        def launch(*tensors: torch.Tensor) -> None:
            return kernel(**dict(zip(names, tensors)))

        return launch, tuple(tensor_args.values()), inter_q, inter_s

    def _screen_gemm1(self, row: dict[str, Any], kn1: str, args: Any) -> float:
        from aiter.test_common import run_perftest

        data, sorting = self._prepare_screening_case(
            row, _parse_mxfp4_g1_kname(kn1)["BM"], False
        )
        launch, tensors, _inter_q, _inter_s = self._prepare_gemm1_screening(
            data, sorting, kn1, int(row["topk"]), self._screening_limit()
        )
        _, us = run_perftest(
            launch, *tensors, num_warmup=int(args.warmup), num_iters=int(args.iters)
        )
        return float(us)

    @staticmethod
    def _gemm2_screening_call(
        data: dict[str, Any],
        sorting: tuple[torch.Tensor, ...],
        inter_q: torch.Tensor,
        inter_s: torch.Tensor,
        kn2: str,
        topk: int,
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        """Return (launch, tensors); time launch(*tensors) so inputs rotate.

        w1 stays bound: the stage2 wrapper reads only its shape.
        """
        cfg = parse_g2_kname_any(kn2)
        sti, sw, sei, nvi, moe_buf, _m_indices, reverse_sorted = sorting
        out = moe_buf if moe_buf.numel() else torch.empty_like(data["input"])
        w1 = data["w1_a16"]

        def launch(
            inter_q: torch.Tensor,
            inter_s: torch.Tensor,
            w2: torch.Tensor,
            w2_scale: torch.Tensor,
            sti: torch.Tensor,
            sw: torch.Tensor,
            sei: torch.Tensor,
            nvi: torch.Tensor,
            out: torch.Tensor,
            topk_weights: torch.Tensor,
            reverse_sorted: torch.Tensor,
        ) -> torch.Tensor:
            # Atomic output accumulates, so clear it every iteration.
            if cfg["atomic"]:
                out.zero_()
            return _mxfp4_a4w4_stage2_fw(
                inter_q,
                w1,
                w2,
                sti,
                sei,
                nvi,
                out,
                topk,
                w2_scale=w2_scale,
                a2_scale=inter_s,
                block_m=cfg["BM"],
                sorted_weights=sw,
                topk_weights=topk_weights,
                kernelName2=kn2,
                reverse_sorted=reverse_sorted,
            )

        tensors = (
            inter_q,
            inter_s,
            data["w2_a16"],
            data["w2s_a16"],
            sti,
            sw,
            sei,
            nvi,
            out,
            data["topk_weights"],
            reverse_sorted,
        )
        return launch, tensors

    def _screen_gemm2(
        self, row: dict[str, Any], kn1: str, kn2: str, args: Any
    ) -> float:
        from aiter.test_common import run_perftest

        cfg = parse_g2_kname_any(kn2)
        data, sorting = self._prepare_screening_case(row, cfg["BM"], cfg["atomic"])
        # Run the reference producer once, untimed, for this candidate only.
        _producer, _producer_tensors, inter_q, inter_s = self._prepare_gemm1_screening(
            data, sorting, kn1, int(row["topk"]), self._screening_limit()
        )
        launch, tensors = self._gemm2_screening_call(
            data, sorting, inter_q, inter_s, kn2, int(row["topk"])
        )
        _, us = run_perftest(
            launch, *tensors, num_warmup=int(args.warmup), num_iters=int(args.iters)
        )
        return float(us)

    @staticmethod
    def _check_reference_output(
        ref: torch.Tensor, out: torch.Tensor, err_ratio: float, label: str
    ) -> bool:
        err = cosine_diff_compare(ref, out, msg=label)
        if err is None or not math.isfinite(float(err)) or float(err) > err_ratio:
            raise RuntimeError(f"cosine err_ratio {err} > {err_ratio}")
        return True

    def _check_reference_consumer(
        self, row: dict[str, Any], kn2: str, args: Any
    ) -> bool:
        cfg = parse_g2_kname_any(kn2)
        data, sorting = self._prepare_screening_case(row, cfg["BM"], True)
        token, topk = int(row["token"]), int(row["topk"])
        activation = {
            "silu": ActivationType.Silu,
            "swiglu": ActivationType.Swiglu,
            "situv2": ActivationType.Situv2,
        }[self._row_act(row)]
        ref1 = FmoeTuner.run_torch_moe_stage1(
            data["a1_qt"],
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            data["a1_scale"],
            data["w1_scale"],
            dtype=dtypes.bf16,
            activation=activation,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            topk=topk,
            swiglu_limit=self._screening_limit(),
        )
        sti, _sw, _sei, nvi, *_rest = sorting
        intermediate = {
            "sti": sti,
            "cumsum": nvi,
            "max_sorted": sti.shape[0],
            "n": int(nvi[0].item()),
        }
        # Build the GEMM2 input from the Torch reference, not a GEMM1 kernel.
        _v2_populate_stage2(
            {"ref1": ref1, "adtype": "fp4"}, intermediate, token, topk, cfg["BM"]
        )
        launch, tensors = self._gemm2_screening_call(
            data, sorting, intermediate["isq"], intermediate["iss"], kn2, topk
        )
        out = launch(*tensors)
        ref = FmoeTuner.run_torch_moe_stage2(
            ref1,
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            a2_scale=None,
            w2_scale=data["w2_scale"],
            dtype=dtypes.bf16,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
        )
        return self._check_reference_output(
            ref, out, float(args.errRatio), f"reference consumer[{kn2}]"
        )

    def _check_reference_pair(
        self, row: dict[str, Any], kn1: str, kn2: str, args: Any
    ) -> bool:
        ne, h, e = int(row["expert"]), int(row["model_dim"]), int(row["inter_dim"])
        token, topk = int(row["token"]), int(row["topk"])
        activation = {
            "silu": ActivationType.Silu,
            "swiglu": ActivationType.Swiglu,
            "situv2": ActivationType.Situv2,
        }[self._row_act(row)]
        data = self._prepare_case(token, h, e, ne, topk, dtypes.bf16)
        out = self._port_e2e(
            data,
            kn1,
            kn2,
            topk,
            ne,
            h,
            dtypes.bf16,
            swiglu_limit=self._screening_limit(),
        )
        ref = self._torch_ref(
            data, topk, dtypes.bf16, activation, swiglu_limit=self._screening_limit()
        )
        return self._check_reference_output(
            ref, out, float(args.errRatio), f"reference pair[{kn1}+{kn2}]"
        )

    @contextmanager
    def _staged_timeout(self, timeout: int) -> Iterator[None]:
        """Apply one best-effort staged-operation timeout."""
        import signal

        if timeout <= 0:
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)
        if previous_handler is None:
            print(
                "[mxfp4-staged] --timeout disabled because the existing SIGALRM "
                "handler cannot be restored; a stalled operation may wait "
                "indefinitely",
                flush=True,
            )
            yield
            return

        class _StagedOperationTimeout(Exception):
            pass

        def _on_alarm(signum: int, frame: Any) -> None:
            raise _StagedOperationTimeout(f"exceeded {timeout}s timeout")

        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        try:
            signal.signal(signal.SIGALRM, _on_alarm)
        except ValueError:
            print(
                "[mxfp4-staged] --timeout disabled because SIGALRM cannot be "
                "installed outside the main thread; a stalled operation may "
                "wait indefinitely",
                flush=True,
            )
            yield
            return
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous_handler)
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def _screen_candidate(
        self, launch: Callable[[], float], label: str, timeout: int
    ) -> float:
        try:
            with self._staged_timeout(timeout):
                us = launch()
                if not math.isfinite(us) or us <= 0:
                    raise ValueError(f"non-positive or non-finite screening time: {us}")
                print(f"[mxfp4-staged] {label} screening us={us}", flush=True)
                return us
        except Exception as exc:
            if _is_accelerator_error(exc):
                raise
            print(f"[mxfp4-staged] {label} screening failed: {exc}", flush=True)
            return self.INVALID_TIME

    def _staged_shortlists(
        self, row: dict[str, Any], plan: "StagedSearchPlan", args: Any, timeout: int
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """Screen GEMM1/GEMM2 per group and return their top-K shortlists."""
        screen_topk = getattr(args, "mxfp4_screen_topk", DEFAULT_SCREEN_TOPK)
        gemm2_groups = {group.key: group for group in plan.gemm2_groups}
        producers = {}
        for group in plan.gemm1_groups:
            atomic_group = gemm2_groups.get(
                GEMM2ScreeningKey(group.key.block_m, AccumulationMode.ATOMIC)
            )
            if atomic_group is None:
                continue
            consumers = reference_consumer_order(atomic_group.candidates)
            if not consumers:
                continue
            consumer = None
            for reference in consumers:
                try:
                    with self._staged_timeout(timeout):
                        if self._check_reference_consumer(
                            row, reference.identity.kernel_name, args
                        ):
                            consumer = reference.identity.kernel_name
                            break
                        raise RuntimeError(
                            "reference consumer returned an incorrect output"
                        )
                except Exception as exc:
                    if _is_accelerator_error(exc):
                        raise
                    print(
                        f"[mxfp4-staged] reference consumer "
                        f"{reference.identity.kernel_name} failed: {exc}",
                        flush=True,
                    )
            if consumer is None:
                continue
            preferred = select_reference_producer(group.candidates)
            producer_order = (preferred,) + tuple(
                candidate for candidate in group.candidates if candidate != preferred
            )
            for reference in producer_order:
                producer = reference.identity.kernel_name
                try:
                    with self._staged_timeout(timeout):
                        if self._check_reference_pair(row, producer, consumer, args):
                            producers[group.key.block_m] = producer
                            break
                        raise RuntimeError(
                            "reference producer returned an incorrect output"
                        )
                except Exception as exc:
                    if _is_accelerator_error(exc):
                        raise
                    print(
                        f"[mxfp4-staged] reference producer {producer} failed: {exc}",
                        flush=True,
                    )

        gemm1_shortlists = {
            group.key: screening_shortlist(
                (
                    (
                        candidate,
                        self._screen_candidate(
                            functools.partial(
                                self._screen_gemm1,
                                row,
                                candidate.identity.kernel_name,
                                args,
                            ),
                            f"GEMM1 {candidate.identity.kernel_name}",
                            timeout,
                        ),
                    )
                    for candidate in group.candidates
                ),
                screen_topk,
            )
            for group in plan.gemm1_groups
            if group.key.block_m in producers
        }
        gemm2_shortlists = {
            group.key: screening_shortlist(
                (
                    (
                        candidate,
                        self._screen_candidate(
                            functools.partial(
                                self._screen_gemm2,
                                row,
                                producers[group.key.block_m],
                                candidate.identity.kernel_name,
                                args,
                            ),
                            f"GEMM2 {candidate.identity.kernel_name}",
                            timeout,
                        ),
                    )
                    for candidate in group.candidates
                ),
                screen_topk,
            )
            for group in plan.gemm2_groups
            if group.key.block_m in producers
        }
        return gemm1_shortlists, gemm2_shortlists

    def _tune_staged_shape(
        self, row: dict[str, Any], args: Any, timeout: int
    ) -> dict[str, Any]:
        plan = self._staged_candidate_plan(row)
        if not plan.pairs:
            return _mxfp4_failed_row(self.keys, row, "no pipeline-pair candidates")
        policy = self.policy_for(args)
        m_est = self._m_est(row)
        small_batch = m_est < self._STAGED_SMALL_BATCH_M_EST
        if small_batch:
            # Isolated stage time does not predict pipeline time at this scale,
            # so time every pruned pair end to end.
            print(
                f"[mxfp4-staged] small batch M_est={m_est}: validating all "
                f"{len(plan.pairs)} pruned pairs without screening",
                flush=True,
            )
            gemm1_shortlists, gemm2_shortlists = {}, {}
        else:
            gemm1_shortlists, gemm2_shortlists = self._staged_shortlists(
                row, plan, args, timeout
            )
        best = None
        evaluated_pairs: dict[PipelinePairKey, tuple[float, str | None]] = {}
        for group in plan.pipeline_groups:
            gemm1 = {
                candidate.identity
                for candidate in gemm1_shortlists.get(group.gemm1_screening_key, ())
            }
            gemm2 = {
                candidate.identity
                for candidate in gemm2_shortlists.get(group.gemm2_screening_key, ())
            }
            shortlisted = tuple(
                pair
                for pair in group.pairs
                if pair.key.gemm1 in gemm1 and pair.key.gemm2 in gemm2
            )
            if not shortlisted and not small_batch:
                print(
                    f"[mxfp4-staged] exhaustive fallback BM{group.key.block_m} "
                    f"{group.key.accumulation_mode.value}: no usable reference/shortlist",
                    flush=True,
                )
            group_best = None
            for pairs in (shortlisted, group.pairs):
                for pair in pairs:
                    if pair.key in evaluated_pairs:
                        continue
                    candidate = dict(pair.candidate_row)
                    try:
                        with self._staged_timeout(timeout):
                            us = float(self._run_candidate(row, candidate, args))
                            failure = (
                                None
                                if math.isfinite(us) and us > 0
                                else f"non-positive or non-finite validation time: {us}"
                            )
                    except Exception as exc:
                        if _is_accelerator_error(exc):
                            raise
                        us, failure = self.INVALID_TIME, str(exc)
                    # Cache results, not exceptions (tracebacks hold tensors).
                    evaluated_pairs[pair.key] = (us, failure)
                    print(
                        f"[mxfp4-staged] {candidate['kernelName1']} + "
                        f"{candidate['kernelName2']} "
                        + (
                            f"validation failed: {failure}"
                            if failure is not None
                            else f"us={us}"
                            + (
                                policy.watt_text(candidate["watts"])
                                if "watts" in candidate
                                else ""
                            )
                        ),
                        flush=True,
                    )
                    if failure is None and policy.beats(group_best, candidate):
                        group_best = candidate
                # Stop after a shortlist winner; fallback runs the whole group.
                if group_best is not None:
                    break
            if group_best is not None and policy.beats(best, group_best):
                best = group_best
        if best is None:
            return _mxfp4_failed_row(self.keys, row, "FAILED: no valid staged pair")
        return best

    def _tune_one_shape(self, row, args):
        """Sweep all (g1, g2) candidates for one shape; return the best row dict.

        --timeout (if > 0) bounds each candidate via SIGALRM. This runs on the
        main thread of whichever process owns the shape, so it works for both the
        sequential path and each per-shape worker in the --mp parallel path. A
        true GPU hang only aborts once control returns to Python; slow/looping
        candidates are caught reliably.
        """
        import signal

        self._invocation_configuration(args)
        reason = self._unsupported_row_reason(row)
        if reason is not None:
            print(f"[mxfp4-port] unsupported shape: {reason}", flush=True)
            return _mxfp4_failed_row(self.keys, row, f"UNSUPPORTED: {reason}")

        timeout = int(getattr(args, "timeout", 0) or 0)
        if _mxfp4_search_mode(args) == "staged":
            return self._tune_staged_shape(row, args, timeout)

        class _CandidateTimeout(Exception):
            pass

        def _on_alarm(signum, frame):
            raise _CandidateTimeout(f"exceeded {timeout}s timeout")

        if timeout > 0:
            try:
                signal.signal(signal.SIGALRM, _on_alarm)
            except ValueError:
                timeout = 0  # not on the main thread; cannot arm SIGALRM

        candidates = self._candidate_rows(
            row, full_search=_mxfp4_search_mode(args) == "full"
        )
        fast_scan = bool(getattr(args, "fast_scan", False))
        policy = self.policy_for(args)
        best, failures, successful = None, [], []

        # Candidate kernels for one shape consume the same immutable inputs and
        # shuffled weights. In scan mode, prepare that fixture once instead of
        # regenerating and reshuffling the full expert tensors for every pair.
        scan_data = None
        scan_reference = None
        if fast_scan:
            self._precompile_candidates(row, candidates)
            scan_data = self._prepare_case(
                int(row["token"]),
                int(row["model_dim"]),
                int(row["inter_dim"]),
                int(row["expert"]),
                int(row["topk"]),
                dtypes.bf16,
            )
            activation = {
                "situv2": ActivationType.Situv2,
                "swiglu": ActivationType.Swiglu,
                "silu": ActivationType.Silu,
            }[self._row_act(row)]
            swiglu_limit = self._invocation_configuration(args)["swiglu_limit"]
            # Build the expensive torch reference once per shape. Every coarse
            # warm-up and every final measured output is checked against it.
            # Apply the same swiglu clamp _run_candidate applies per candidate:
            # a reference built without the limit would reject every candidate
            # whenever AITER_MXFP4_TUNE_SWIGLU_LIMIT is set.
            scan_reference = self._torch_ref(
                scan_data,
                int(row["topk"]),
                dtypes.bf16,
                activation,
                swiglu_limit=swiglu_limit,
            )
        accuracy_checks = max(
            0, int(getattr(args, "fast_scan_warmup_accuracy_checks", 1))
        )
        scan_iters = _parse_fast_scan_iters(getattr(args, "fast_scan_iters", 10))
        for candidate in candidates:
            if timeout > 0:
                signal.alarm(timeout)
            try:
                us = self._run_candidate(
                    row,
                    candidate,
                    args,
                    data=scan_data,
                    reference=scan_reference,
                    num_warmup=accuracy_checks if fast_scan else None,
                    num_iters=scan_iters if fast_scan else None,
                    final_timing=not fast_scan,
                )
                watt_text = (
                    policy.watt_text(candidate["watts"]) if "watts" in candidate else ""
                )
                if not math.isfinite(float(us)) or us <= 0:
                    raise RuntimeError(f"invalid MXFP4 validation time {us}")
                print(
                    f"[mxfp4-port] token={row['token']} inter={row['inter_dim']} "
                    f"{candidate['kernelName1']} + {candidate['kernelName2']} "
                    f"us={us}{watt_text}",
                    flush=True,
                )
                if policy.beats(best, candidate):
                    best = candidate
                if fast_scan:
                    successful.append(candidate)
            except Mxfp4AccuracyError as exc:
                failures.append(
                    f"{candidate['kernelName1']}/{candidate['kernelName2']}: "
                    f"accuracy {exc}"
                )
                print(f"[mxfp4-port] candidate rejected: {failures[-1]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"{candidate['kernelName1']}/{candidate['kernelName2']}: {exc}"
                )
                print(f"[mxfp4-port] candidate failed: {failures[-1]}", flush=True)
            finally:
                if timeout > 0:
                    signal.alarm(0)
        if fast_scan and successful:
            shortlist = max(1, int(getattr(args, "fast_scan_finalists", 5)))
            # A power winner can sit outside the latency shortlist, same as
            # the hipBLASLt accurate pass timing every heuristic.
            finalist_count = policy.accurate_count(shortlist, len(successful))
            if finalist_count != min(shortlist, len(successful)):
                print(
                    f"[mxfp4-port] select={policy.name}: retiming all "
                    f"{finalist_count} accurate candidates",
                    flush=True,
                )
            finalist_iters = _parse_fast_scan_iters(
                getattr(args, "fast_scan_final_iters", 100)
            )
            # One stable latency ordering drives both the initial performance
            # finalists and all later backfill. This preserves the fastest
            # remaining coarse survivor when a finalist fails final validation;
            # the separate accuracy promotion below covers the accuracy side.
            performance_order = _select_mxfp4_performance_finalists(
                successful, len(successful)
            )
            performance_finalists = performance_order[:finalist_count]
            accuracy_promotion = _select_mxfp4_accuracy_promotion(
                successful, performance_finalists
            )
            finalists = list(performance_finalists)
            if accuracy_promotion is not None:
                finalists.append(accuracy_promotion)
            finalist_target = min(
                len(successful),
                finalist_count + int(accuracy_promotion is not None),
            )
            finalist_queue = []
            finalist_keys = set()
            # Accuracy promotion is tried once after the speed finalists. Any
            # further replacement is strictly next-lowest-latency order; the
            # accuracy-bucket rank is diagnostic only and must not reorder it.
            for candidate in finalists + performance_order:
                candidate_key = (
                    candidate["kernelName1"],
                    candidate["kernelName2"],
                )
                if candidate_key not in finalist_keys:
                    finalist_keys.add(candidate_key)
                    finalist_queue.append(candidate)
            print(
                f"[mxfp4-port] retiming {len(performance_finalists)} performance "
                f"finalists + {int(accuracy_promotion is not None)} "
                f"accuracy-promoted finalist with "
                f"no additional warm-up + {finalist_iters} profiled iterations; "
                f"backfilling to {finalist_target} successful retimes",
                flush=True,
            )
            best = None
            retimed = []
            for candidate in finalist_queue:
                if len(retimed) >= finalist_target:
                    break
                if timeout > 0:
                    signal.alarm(timeout)
                try:
                    us = self._run_candidate(
                        row,
                        candidate,
                        args,
                        data=scan_data,
                        reference=scan_reference,
                        num_warmup=0,
                        num_iters=finalist_iters,
                    )
                    if not math.isfinite(float(us)) or us <= 0:
                        raise RuntimeError(f"invalid MXFP4 validation time {us}")
                    watt_text = policy.watt_text(candidate.get("watts"))
                    print(
                        f"[mxfp4-port] finalist token={row['token']} "
                        f"{candidate['kernelName1']} + "
                        f"{candidate['kernelName2']} us={us}{watt_text}",
                        flush=True,
                    )
                    retimed.append(candidate)
                except Mxfp4AccuracyError as exc:
                    failures.append(
                        f"finalist {candidate['kernelName1']}/"
                        f"{candidate['kernelName2']}: accuracy {exc}"
                    )
                    print(
                        f"[mxfp4-port] finalist rejected: {failures[-1]}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"finalist {candidate['kernelName1']}/"
                        f"{candidate['kernelName2']}: {exc}"
                    )
                    print(f"[mxfp4-port] candidate failed: {failures[-1]}", flush=True)
                finally:
                    if timeout > 0:
                        signal.alarm(0)
            if retimed:
                fastest_us = min(float(candidate["us"]) for candidate in retimed)
                best = policy.best(retimed, latency_choice=_select_mxfp4_finalist)
                ranked_finalists = [best] + sorted(
                    (candidate for candidate in retimed if candidate is not best),
                    key=policy.rank_of,
                )
                for rank, candidate in enumerate(ranked_finalists, start=1):
                    latency_us = float(candidate["us"])
                    accuracy_error = _candidate_accuracy_error(candidate)
                    normalized_difference = _normalized_latency_difference(
                        latency_us, fastest_us
                    )
                    watt_text = policy.watt_text(candidate.get("watts"))
                    print(
                        f"[mxfp4-port] finalist-summary rank={rank} "
                        f"selected={int(candidate is best)} "
                        f"inside_global_gate="
                        f"{int(normalized_difference <= MXFP4_FINAL_GLOBAL_TIMING_GATE)} "
                        f"us={latency_us:.4f}{watt_text} "
                        f"normalized_diff={normalized_difference:.6f} "
                        f"cosine_error={accuracy_error:.6f} "
                        f"similarity_pct={(1.0 - accuracy_error) * 100:.4f} "
                        f"kernel1={candidate['kernelName1']} "
                        f"kernel2={candidate['kernelName2']}",
                        flush=True,
                    )
        if best is None:
            best = candidates[0] if candidates else self._candidate_row(row, 0, "", "")
            best["us"] = self.INVALID_TIME
            best["kernelName1"] = ("FAILED: " + "; ".join(failures))[:240]
            print(
                f"[mxfp4-port] all candidates failed for "
                f"{tuple(row[k] for k in self.keys)}",
                flush=True,
            )
        return best

    def tune(self, untunedf, tunedf, args):
        del tunedf
        if getattr(args, "fast_scan", False) and _mxfp4_search_mode(args) == "staged":
            raise ValueError(
                "--fast-scan applies to pruned or full search, not "
                "--mxfp4-search-mode staged"
            )
        self._invocation_configuration(args)
        rows = [row.to_dict() for _, row in untunedf.iterrows()]

        if not rows:
            return []

        mp_num = int(getattr(args, "mp", 1) or 1)
        ngpu = torch.cuda.device_count()
        mp_num = max(1, min(mp_num, ngpu, len(rows)))
        staged = _mxfp4_search_mode(args) == "staged"
        identities = (
            self._preflight_devices(mp_num, ngpu, staged=True)
            if staged
            else self._preflight_devices(mp_num, ngpu)
        )

        if mp_num <= 1 and not staged:
            for row in rows:
                row["gfx"], row["cu_num"] = identities[0]
            return [self._tune_one_shape(row, args) for row in rows]

        # One fresh process per shape (memory fully released between shapes),
        # spread across mp_num GPUs, each on a distinct device.
        import multiprocessing as _mp

        print(
            f"[mxfp4-port] tuning {len(rows)} shapes across {mp_num} GPUs", flush=True
        )
        ctx = _mp.get_context("spawn")
        payloads = [(self.keys, row, args, None) for row in rows]
        results = (
            _run_staged_shapes_isolated(payloads, mp_num, ctx)
            if staged
            else _run_shapes_isolated(payloads, mp_num, ctx)
        )
        # A None means the shape's process died mid-shape (e.g. a C++ abort()).
        # Report it as a failed shape rather than dropping it, so the run
        # finishes and the failure shows up in the summary.
        return [
            (
                res
                if res is not None
                else _mxfp4_failed_row(self.keys, row, "FAILED: worker died mid-shape")
            )
            for row, res in zip(rows, results)
        ]

    def post_process(self, results, args, topk=-1, fast_mode=False):
        del args, topk, fast_mode
        return _frame_keeping_watts(self.columns, results)

    def result_to_csv(self, results, file, concat=False):
        del concat
        old_tunedf = self.get_tuned_gemm_list(file, self.columns)
        times = pd.to_numeric(results["us"], errors="coerce")
        valid = results[(times > 0) & (times < self.INF_TIME)]
        invalid = results[~results.index.isin(valid.index)]
        self.failed = pd.concat([self.failed, invalid], ignore_index=True)
        resultdf = self._normalize_tuned_csv(self.update_tunedf(old_tunedf, valid))
        # The updated row is the first per key; drop rows it made unreachable.
        self._drop_superseded_rows(resultdf, self.keys).to_csv(file, index=False)
        # Count winners only once saved; unsaved shapes report as untuned.
        self.success = pd.concat([self.success, valid], ignore_index=True)


def _mxfp4_tune_shape_worker(payload):
    """Spawned worker: tune one shape on the parent-assigned GPU (for
    --mxfp4-flydsl --mp). Rebuilds a minimal tuner since the instance need only
    carry ``keys``."""
    keys, row, args, gpu = payload
    try:
        torch.cuda.set_device(gpu)
        torch.set_default_device("cuda")
        tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
        tuner.keys = keys
        row["gfx"], row["cu_num"] = tuner._runtime_device_identity(gpu)
        print(
            f"[mxfp4-port] shape token={row['token']} inter={row['inter_dim']} "
            f"expert={row['expert']} topk={row['topk']} -> GPU{gpu}",
            flush=True,
        )
        return tuner._tune_one_shape(row, args)
    except Exception as exc:  # noqa: BLE001
        # Catastrophic (non per-candidate) failure: record as a failed shape so
        # the run keeps going instead of aborting.
        print(f"[mxfp4-port] shape failed on GPU{gpu}: {exc}", flush=True)
        return _mxfp4_failed_row(keys, row, f"FAILED(GPU{gpu}): {exc}")


def _mxfp4_failed_row(keys, row, reason):
    """A tuned-CSV row standing in for a shape that produced no timing."""
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    tuner.keys = keys
    cand = tuner._candidate_row(row, 0, reason[:240], "")
    cand["us"] = Mxfp4FlydslTuner.INVALID_TIME
    return cand


def _mxfp4_shape_proc(payload, out_q, idx):
    """Child entry point: post the shape's result so the parent can tell a
    finished shape apart from a worker that died mid-shape."""
    out_q.put((idx, _mxfp4_tune_shape_worker(payload)))


def _mxfp4_staged_shape_proc(
    payload: tuple[Any, dict[str, Any], Any, int | None], result_tx: Any
) -> None:
    """Send one staged shape result through its worker-private Pipe."""
    try:
        result_tx.send(_mxfp4_tune_shape_worker(payload))
    finally:
        result_tx.close()


@dataclass
class _StagedShapeSlot:
    proc: Any
    gpu: int
    result_rx: Any
    result_received: bool = False
    result: dict[str, Any] | None = None


def _run_staged_shapes_isolated(
    payloads: list[tuple[Any, dict[str, Any], Any, int | None]],
    mp_num: int,
    ctx: Any,
    entry: Callable[..., None] = _mxfp4_staged_shape_proc,
) -> list[dict[str, Any] | None]:
    """Run each staged shape in a fresh worker with a private result Pipe.

    A GPU is reused only after its worker exits. There is no parent deadline.
    """
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    running: dict[int, _StagedShapeSlot] = {}
    cleanup_slots: list[_StagedShapeSlot] = []
    free_gpus = list(range(mp_num))
    done: set[int] = set()
    next_payload = 0

    def _receive_result(slot: _StagedShapeSlot) -> None:
        receive_failed = False
        try:
            try:
                payload = slot.result_rx.recv_bytes()
            except EOFError:
                return
            slot.result = ForkingPickler.loads(payload)
            slot.result_received = True
        except BaseException:
            receive_failed = True
            raise
        finally:
            try:
                slot.result_rx.close()
            except BaseException:
                if not receive_failed:
                    raise

    try:
        while len(done) < len(payloads):
            while next_payload < len(payloads) and free_gpus:
                gpu = free_gpus.pop(0)
                keys, row, args, _ = payloads[next_payload]
                result_rx, result_tx = ctx.Pipe(duplex=False)
                start_failed = False
                try:
                    proc = ctx.Process(
                        target=entry,
                        args=((keys, row, args, gpu), result_tx),
                        daemon=True,
                    )
                    slot = _StagedShapeSlot(proc, gpu, result_rx)
                    cleanup_slots.append(slot)
                    proc.start()
                except BaseException:
                    start_failed = True
                    try:
                        result_rx.close()
                    except BaseException:  # noqa: BLE001,S110
                        pass
                    raise
                finally:
                    try:
                        result_tx.close()
                    except BaseException:
                        if not start_failed:
                            raise
                running[next_payload] = slot
                next_payload += 1

            waitables = []
            for slot in running.values():
                if not slot.result_rx.closed:
                    waitables.append(slot.result_rx)
                waitables.append(slot.proc.sentinel)
            ready = wait_for_connections(waitables)

            for slot in running.values():
                if not slot.result_rx.closed and slot.result_rx in ready:
                    _receive_result(slot)

            for idx, slot in list(running.items()):
                if slot.proc.sentinel not in ready:
                    continue
                if not slot.result_rx.closed:
                    _receive_result(slot)
                slot.proc.join()
                exitcode = slot.proc.exitcode
                if slot.result_received and exitcode == 0:
                    results[idx] = slot.result
                else:
                    print(
                        f"[mxfp4-port] worker for shape {idx} failed "
                        f"(exitcode={exitcode}, result_received={slot.result_received})",
                        flush=True,
                    )
                slot.proc.close()
                cleanup_slots.remove(slot)
                running.pop(idx)
                free_gpus.append(slot.gpu)
                done.add(idx)
    finally:
        # Best-effort cleanup of unfinished workers; never mask a raised error.
        for slot in cleanup_slots:
            try:
                if slot.proc.is_alive():
                    slot.proc.terminate()
            except BaseException:  # noqa: BLE001,S110
                pass
        # Kill workers that survived terminate().
        for slot in cleanup_slots:
            try:
                if slot.proc.is_alive():
                    slot.proc.kill()
            except BaseException:  # noqa: BLE001,S110
                pass
        for slot in cleanup_slots:
            try:
                if slot.proc.pid is not None:
                    slot.proc.join()
            except BaseException:  # noqa: BLE001,S110
                pass
            try:
                if not slot.result_rx.closed:
                    slot.result_rx.close()
            except BaseException:  # noqa: BLE001,S110
                pass
            try:
                slot.proc.close()
            except BaseException:  # noqa: BLE001,S110
                pass

    return results


def _run_shapes_isolated(payloads, mp_num, ctx, entry=_mxfp4_shape_proc):
    """Run each payload in its own fresh process, at most ``mp_num`` at a time.

    Returns one entry per payload: the worker's result, or None if its process
    died without producing one.

    Why not Pool.map: a pool worker that dies mid-task (a C++ ``abort()`` is not
    catchable from Python, so an unsupported shape takes the process down with
    it) is never noticed -- the result simply never arrives and map() waits
    forever, while the pool quietly spawns idle replacements. Owning the
    processes lets us read ``exitcode`` and turn a dead worker into a reported
    failure. GPU ids come from a parent-held free list rather than a shared
    queue for the same reason: a killed child cannot leak the id it was holding
    and starve every later shape.
    """
    import queue as _queue

    out_q = ctx.Queue()
    results = [None] * len(payloads)
    running = {}  # idx -> (Process, gpu)
    free_gpus = list(range(mp_num))
    done, nxt = set(), 0

    while len(done) < len(payloads):
        while nxt < len(payloads) and free_gpus:
            gpu = free_gpus.pop(0)
            keys, row, args, _ = payloads[nxt]
            proc = ctx.Process(
                target=entry, args=((keys, row, args, gpu), out_q, nxt), daemon=True
            )
            proc.start()
            running[nxt] = (proc, gpu)
            nxt += 1

        def _reap(idx):
            proc, gpu = running.pop(idx)
            proc.join()
            free_gpus.append(gpu)
            done.add(idx)

        try:
            idx, res = out_q.get(timeout=1.0)
            results[idx] = res
            _reap(idx)
        except _queue.Empty:
            # Nothing arrived for a full second, so any result a just-exited
            # child had queued would already be here: an exited-but-unreported
            # index really did die mid-shape.
            for idx, (proc, _gpu) in list(running.items()):
                if not proc.is_alive():
                    print(
                        f"[mxfp4-port] worker for shape {idx} died "
                        f"(exitcode={proc.exitcode}) without a result",
                        flush=True,
                    )
                    _reap(idx)
    return results
