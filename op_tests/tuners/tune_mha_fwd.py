# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pick the fastest MHA forward backend per shape for ``flash_attn_varlen_func``.

Each backend runs as shipped: this tuner never enumerates tiles. A Triton or
Gluon row records the digest of the tile Triton resolved while it was
measured, and dispatch drops the row once Triton resolves another. A Triton
candidate's time includes the copy into ``out`` that dispatch does for it.
A plain run writes the fastest correct backend per shape. Update the tuned
CSV through ``--compare --update_improved``: it verifies each row's dispatch
in a fresh process and keeps a row only if it beats both the current row and
auto-select by ``--min_improvement_pct``.
"""

import json
import math
import os
import sys
import zlib
from itertools import accumulate
from typing import Any, ClassVar, NamedTuple

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import pandas as pd
import torch

from aiter.jit.core import AITER_CONFIG_MHA_FWD
from aiter.ops.mha import (
    flash_attn_varlen_func,
    fmha_fwd_bf16_opus_varlen_fwd,
    fmha_v3_varlen_fwd,
    mha_varlen_fwd,
)
from aiter.ops.mha_fwd_policy import (
    MHA_FWD_BACKENDS,
    MHA_FWD_CONFIG_ENV,
    MHA_FWD_FAMILY,
    MHA_FWD_PROBLEM_KEY_FIELDS,
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TRITON_BACKENDS,
    MHA_FWD_TUNING_KEY_FIELDS,
    MHA_FWD_UNTUNED_CSV,
    MhaFwdProblem,
    as_bool,
    clear_mha_fwd_tuning_cache,
    config_digest,
    triton_config,
    triton_unsupported_reason,
)
from aiter.test_common import run_perftest
from aiter.utility import dispatch_verification
from aiter.utility.base_tuner import TunerCommon, _write_csv_atomic
from aiter.utility.mp_tuner import mp_tuner


class _Task(NamedTuple):
    """One mp_tuner candidate, in the tuple order mp_tuner unpacks."""

    info: Any
    gen_data: Any
    gen_args: Any
    func: Any
    args: Any
    kwargs: dict
    ref_func: Any
    ref_args: Any
    ref_kwargs: dict
    ref: Any
    rtol: float
    atol: float


RESULT_FIELDS = (
    "backend",
    "config_digest",
    "resolved_config",
    "us",
    "errRatio",
    "status",
    "detail",
    "tflops",
)
BOOL_FIELDS = (
    "causal",
    "return_lse",
    "return_attn_probs",
    "has_bias",
    "has_alibi",
    "has_sink",
    "has_block_table",
    "has_q_descale",
    "has_physical_padding",
    "is_grad",
)
# Inputs this tuner cannot synthesize; a row that needs one is refused.
UNSYNTHESIZED_FIELDS = (
    "return_attn_probs",
    "has_bias",
    "has_alibi",
    "has_sink",
    "has_block_table",
    "has_q_descale",
    "has_physical_padding",
    "is_grad",
)
# No output element may fall outside these tolerances. On these random inputs
# the output spread falls as 1/sqrt(keys), so _error_atol shrinks ERROR_ATOL
# beyond ATOL_REFERENCE_KEYS keys per query, or a kernel that drops a long
# sequence's last keys would pass.
ERROR_RTOL = 2e-2
ERROR_ATOL = 2e-2
ATOL_REFERENCE_KEYS = 1024
MAX_ERROR_RATIO = 0.0
TASK_TIMEOUT_S = 7200
# Score storage the reference may hold at once.
REFERENCE_CHUNK_BYTES = 64 << 20
# Kernels that ignore logits_soft_cap; dispatch never routes a soft-capped
# call to them, so they cannot serve such a shape.
SOFT_CAP_BLIND_BACKENDS = ("asm_v3", "flydsl", "opus")
# Architectures where this tuner can time every kernel dispatch may pick;
# gfx1250 dispatch picks an asm sink-varlen kernel it does not.
TUNABLE_GFX = ("gfx942", "gfx950")


def candidate_backends(gfx: str) -> tuple[str, ...]:
    """Backends that exist on ``gfx``; one that declines a shape (FlyDSL
    screens its own inputs) comes back as ``unsupported``."""
    backends = ["asm_v3", "ck", "flydsl", "triton"]
    if gfx == "gfx950":
        backends.extend(("opus", "gluon"))
    return tuple(sorted(backends))


def _balanced_lengths(total: int, batch: int, maximum: int) -> list[int]:
    """The first sequence is ``maximum`` long, the rest share what is left.

    The key holds only totals and maxima, so this one distribution stands in
    for every batch with the same key.
    """
    if total < maximum or total > batch * maximum:
        raise ValueError(
            f"total={total} cannot have batch={batch} and maximum={maximum}"
        )
    lengths = [maximum]
    remaining = total - maximum
    for index in range(1, batch):
        slots = batch - index
        length = min(maximum, (remaining + slots - 1) // slots)
        lengths.append(length)
        remaining -= length
    if min(lengths) < 1:
        raise ValueError(
            f"total={total} over batch={batch} with maximum={maximum} "
            "leaves a sequence empty"
        )
    return lengths


def _prefix_sum(lengths: list[int], device) -> torch.Tensor:
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


def generate_data(
    batch,
    total_q,
    total_k,
    max_seqlen_q,
    max_seqlen_k,
    nhead_q,
    nhead_k,
    hdim_q,
    hdim_v,
    dtype,
    seed,
    device="cuda",
):
    dtype_obj = getattr(torch, dtype)
    q_lengths = _balanced_lengths(total_q, batch, max_seqlen_q)
    k_lengths = _balanced_lengths(total_k, batch, max_seqlen_k)
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, dtype=dtype_obj, device=device, generator=generator)

    return {
        "q": randn(total_q, nhead_q, hdim_q),
        "k": randn(total_k, nhead_k, hdim_q),
        "v": randn(total_k, nhead_k, hdim_v),
        "cu_q": _prefix_sum(q_lengths, device),
        "cu_k": _prefix_sum(k_lengths, device),
    }


def _chunked_reference(
    q,
    k,
    v,
    cu_q,
    cu_k,
    softmax_scale,
    causal,
    window_left,
    window_right,
    return_lse,
    logits_soft_cap=0.0,
):
    """Bounded-memory fp32 oracle; score storage stays below
    ``REFERENCE_CHUNK_BYTES``.

    Masks are bottom-right aligned, as the kernels align them, and LSE is
    (nhead_q, total_q), CK's layout. A soft-cap applies ``cap * tanh(s / cap)``
    to the scaled scores before masking, as ``attention_ref`` does.
    """
    cap = float(logits_soft_cap)
    output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = (
        torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    q_offsets = cu_q.cpu().tolist()
    k_offsets = cu_k.cpu().tolist()
    groups = q.shape[1] // k.shape[1]
    for batch_index in range(len(q_offsets) - 1):
        q_begin, q_end = q_offsets[batch_index : batch_index + 2]
        k_begin, k_end = k_offsets[batch_index : batch_index + 2]
        q_seq = q[q_begin:q_end]
        k_seq = k[k_begin:k_end].float()
        v_seq = v[k_begin:k_end].float()
        sq, sk, heads = q_seq.shape[0], k_seq.shape[0], q_seq.shape[1]
        rows = max(1, REFERENCE_CHUNK_BYTES // max(4, 4 * heads * sk))
        key_positions = torch.arange(sk, device=q.device)
        offset = sk - sq
        for row_begin in range(0, sq, rows):
            row_end = min(sq, row_begin + rows)
            q_chunk = (
                q_seq[row_begin:row_end]
                .float()
                .reshape(row_end - row_begin, k_seq.shape[1], groups, q_seq.shape[-1])
            )
            scores = torch.einsum("qhgd,khd->hgqk", q_chunk, k_seq).reshape(
                heads, row_end - row_begin, sk
            )
            scores.mul_(float(softmax_scale))
            if cap > 0.0:
                scores = torch.tanh(scores / cap).mul_(cap)
            query_positions = torch.arange(
                row_begin, row_end, device=q.device
            ).unsqueeze(1)
            if causal:
                scores.masked_fill_(
                    key_positions > query_positions + offset, float("-inf")
                )
            if int(window_left) >= 0:
                scores.masked_fill_(
                    key_positions < query_positions + offset - int(window_left),
                    float("-inf"),
                )
            if int(window_right) >= 0:
                scores.masked_fill_(
                    key_positions > query_positions + offset + int(window_right),
                    float("-inf"),
                )
            probabilities = torch.softmax(scores, dim=-1)
            probabilities.nan_to_num_(nan=0.0)
            out = torch.einsum(
                "hgqk,khd->qhgd",
                probabilities.reshape(k_seq.shape[1], groups, row_end - row_begin, sk),
                v_seq,
            ).reshape(row_end - row_begin, heads, v_seq.shape[-1])
            output[q_begin + row_begin : q_begin + row_end].copy_(out.to(q.dtype))
            if lse is not None:
                lse[:, q_begin + row_begin : q_begin + row_end] = torch.logsumexp(
                    scores, dim=-1
                )
    return (output, lse) if lse is not None else output


def _normalize_result(result, return_lse, total_q, nhead_q):
    if not return_lse:
        return result[0] if isinstance(result, tuple) else result
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("candidate did not return the requested LSE")
    out, lse = result[:2]
    if tuple(lse.shape) == (int(total_q), int(nhead_q)):
        lse = lse.transpose(0, 1).contiguous()
    if tuple(lse.shape) != (int(nhead_q), int(total_q)):
        raise RuntimeError(f"unexpected varlen LSE shape {tuple(lse.shape)}")
    return out, lse


def _run_candidate(
    q,
    k,
    v,
    cu_q,
    cu_k,
    backend,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q,
    dropout_p,
    softmax_scale,
    logits_soft_cap,
    how_v3_bf16_cvt,
    causal,
    window_left,
    window_right,
    return_lse,
):
    """One launch of ``backend`` as its owners ship it."""
    total_q, nhead_q = q.shape[0], q.shape[1]
    if backend == "asm_v3":
        out, lse, _, _ = fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            int(min_seqlen_q),
            float(dropout_p),
            float(softmax_scale),
            float(logits_soft_cap),
            False,
            bool(causal),
            int(window_left),
            int(window_right),
            bool(return_lse),
            False,
            int(how_v3_bf16_cvt),
        )
        return _normalize_result((out, lse), return_lse, total_q, nhead_q)
    if backend == "ck":
        out, lse, _, _ = mha_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            int(min_seqlen_q),
            float(dropout_p),
            float(softmax_scale),
            float(logits_soft_cap),
            False,
            bool(causal),
            int(window_left),
            int(window_right),
            0,
            bool(return_lse),
            False,
        )
        return _normalize_result((out, lse), return_lse, total_q, nhead_q)
    if backend in MHA_FWD_TRITON_BACKENDS:
        from aiter.ops.triton.attention.mha import (
            flash_attn_varlen_func as flash_attn_varlen_func_triton,
        )

        # No config: Triton resolves its own tile, as dispatch will.
        result = flash_attn_varlen_func_triton(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            dropout_p=float(dropout_p),
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            window_size=(int(window_left), int(window_right)),
            return_lse=bool(return_lse),
            backend=backend,
        )
        return _normalize_result(result, return_lse, total_q, nhead_q)
    if backend == "flydsl":
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_fwd

        out = flydsl_flash_attn_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            window_size=(int(window_left), int(window_right), 0),
            return_lse=bool(return_lse),
        )
        if out is None:
            raise RuntimeError("FlyDSL rejected this MHA problem")
        return _normalize_result(out, return_lse, total_q, nhead_q)
    if backend == "opus":
        result = fmha_fwd_bf16_opus_varlen_fwd(
            q,
            k,
            v,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            seqstart_q=cu_q,
            seqstart_k=cu_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            return_lse=bool(return_lse),
        )
        return _normalize_result(result, return_lse, total_q, nhead_q)
    raise ValueError(f"unknown backend {backend!r}")


def _names(value) -> list[str]:
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def _read_table(path) -> pd.DataFrame:
    """The tuned table as written, in the columns dispatch reads."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(MHA_FWD_RUNTIME_CSV_FIELDS))
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.reindex(columns=list(MHA_FWD_RUNTIME_CSV_FIELDS), fill_value="")


def _problem(row) -> MhaFwdProblem:
    return MhaFwdProblem.from_mapping(
        {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
    )


def _seed(problem: MhaFwdProblem) -> int:
    return zlib.crc32(",".join(problem.key()).encode("utf-8"))


def _error_atol(problem: MhaFwdProblem) -> float:
    keys = problem.max_seqlen_k
    if problem.window_left >= 0 and (problem.window_right >= 0 or problem.causal):
        keys = min(keys, problem.window_left + max(problem.window_right, 0) + 1)
    return ERROR_ATOL * min(1.0, math.sqrt(ATOL_REFERENCE_KEYS / max(keys, 1)))


def _data_args(problem: MhaFwdProblem) -> tuple:
    return (
        problem.batch,
        problem.total_q,
        problem.total_k,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        problem.nhead_q,
        problem.nhead_k,
        problem.hdim_q,
        problem.hdim_v,
        problem.dtype,
        _seed(problem),
    )


INPUTS = ["q", "k", "v", "cu_q", "cu_k"]


def _reference_args(problem: MhaFwdProblem) -> tuple:
    """``_chunked_reference`` arguments after the inputs."""
    return (
        problem.hdim_q**-0.5,
        problem.causal,
        problem.window_left,
        problem.window_right,
        problem.return_lse,
        problem.logits_soft_cap,
    )


class MhaFwdTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_MHA_FWD,
        "untune_file": f"aiter/configs/{MHA_FWD_UNTUNED_CSV}",
        "batch": 8,
        "errRatio": MAX_ERROR_RATIO,
        "timeout": TASK_TIMEOUT_S,
        "config_env_name": MHA_FWD_CONFIG_ENV,
    }
    SELECTION_PROBE_MODULE = "op_tests.tuners.tune_mha_fwd"
    SELECTION_FAMILY = MHA_FWD_FAMILY
    COMPARE_AUTOSELECT = True

    def __init__(self):
        super().__init__(
            "MhaFwdTuner",
            list(MHA_FWD_TUNING_KEY_FIELDS),
            list(RESULT_FIELDS),
            "Pick the fastest packed-varlen MHA forward backend per shape. "
            "Shapes that already have a current row are skipped unless --all "
            "is given.",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--backends",
            default="",
            help=(
                "comma-separated backends to measure (default: all that can "
                "serve the shape here); --compare times what serves it today"
            ),
        )

    def _setup_common_arguments(self):
        super()._setup_common_arguments()
        self.parser.set_defaults(shape_grouped=True)

    def getKernelName(self, kernel_id):
        return str(kernel_id)

    def calculate(self, result, inbpe=2, outbpe=2):
        info, us, _ = result
        problem = MhaFwdProblem.from_mapping(
            dict(zip(MHA_FWD_TUNING_KEY_FIELDS, info[0]))
        )
        q_lengths = _balanced_lengths(
            problem.total_q, problem.batch, problem.max_seqlen_q
        )
        k_lengths = _balanced_lengths(
            problem.total_k, problem.batch, problem.max_seqlen_k
        )
        # Score entries of the generated batch; causal masks about half.
        scores = sum(sq * sk for sq, sk in zip(q_lengths, k_lengths))
        if problem.causal:
            scores /= 2
        flop = 2 * problem.nhead_q * scores * (problem.hdim_q + problem.hdim_v)
        return 0.0 if us <= 0 or not math.isfinite(us) else flop / us / 1e6

    def pre_process(self, args):
        frame = pd.read_csv(args.untune_file)
        missing = [f for f in MHA_FWD_PROBLEM_KEY_FIELDS if f not in frame.columns]
        if missing:
            raise ValueError(f"MHA problem CSV is missing columns: {missing}")
        if (frame["mode"].astype(str) != "varlen").any():
            raise ValueError("the forward tuner accepts mode=varlen rows")
        frame = frame.assign(
            **{field: frame[field].map(as_bool).astype(int) for field in BOOL_FIELDS}
        )
        if frame[list(UNSYNTHESIZED_FIELDS)].astype(bool).any(axis=None):
            raise ValueError(
                "this tuner cannot synthesize bias/alibi/sink/paging/"
                "quantization/physical-padding/training payloads"
            )
        if (frame["dropout_p"].astype(float) != 0.0).any():
            raise ValueError("dropout tuning requires a reproducible dropout mask")
        if (frame["sink_size"].astype(int) != 0).any():
            raise ValueError("sink-token tuning requires explicit sink-token payloads")
        if (frame["min_seqlen_q"].astype(int) != 0).any():
            # CK skips sequences no longer than min_seqlen_q and other backends
            # do not, so their outputs cannot be judged against one reference.
            raise ValueError("min_seqlen_q tuning is not supported; use 0")
        for _, row in frame.iterrows():
            for side in "qk":
                total, longest = row[f"total_{side}"], row[f"max_seqlen_{side}"]
                _balanced_lengths(int(total), int(row["batch"]), int(longest))
        unknown = sorted(
            set(_names(getattr(args, "backends", ""))) - set(MHA_FWD_BACKENDS)
        )
        if unknown:
            raise ValueError(
                f"unknown --backends {unknown}; choose from {sorted(MHA_FWD_BACKENDS)}"
            )
        frame = self.stamp_hardware(frame)
        if self.get_gfx() not in TUNABLE_GFX:
            raise ValueError(
                f"MHA forward tuning supports {', '.join(TUNABLE_GFX)}, "
                f"not {self.get_gfx()}"
            )
        # Canonical spelling, so a row written as bf16 is the shape dispatch
        # looks up and the summary matches results against it.
        keys = [self.lookup_key(row) for _, row in frame.iterrows()]
        self.untunedf = pd.DataFrame(
            keys, columns=list(MHA_FWD_TUNING_KEY_FIELDS), dtype=str
        ).drop_duplicates(ignore_index=True)
        self._args = args
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not args.all:
            current = self.current_rows(self.tunedf)
            tuned = {self.lookup_key(row) for _, row in current.iterrows()}
            self.untunedf = self.untunedf[
                [tuple(key) not in tuned for key in self.untunedf.itertuples(False)]
            ].reset_index(drop=True)

    def stamp_hardware(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with the hardware columns set to this GPU's; a shape list
        naming other hardware is refused rather than tuned under it."""
        frame = frame.copy()
        for field, resolve in self._live_hardware().items():
            value = resolve()
            if field in frame.columns:
                given = frame[field].fillna("").astype(str).str.strip().str.lower()
                if ((given != "") & (given != str(value).lower())).any():
                    raise ValueError(
                        f"shape list {field} does not match this GPU ({value})"
                    )
            frame[field] = value
        return frame

    def lookup_key(self, row) -> tuple[str, ...]:
        return _problem(row).key()

    def stale_reason(self, row) -> str:
        """A Triton or Gluon row dispatch skips because Triton now resolves
        another tile for it."""
        backend = str(row["backend"]).strip()
        problem = _problem(row)
        if backend not in MHA_FWD_TRITON_BACKENDS or problem.gfx != self.get_gfx():
            return ""
        try:
            current = config_digest(triton_config(problem, backend))
        # Dispatch skips a row whose digest cannot be computed, so it is stale.
        except Exception as error:  # noqa: BLE001
            return f"the {backend} config does not resolve: {error}"
        if str(row.get("config_digest", "")).strip() != current:
            return f"Triton now resolves another {backend} config (digest {current})"
        return ""

    def expected_selection(self, row) -> dict[str, Any] | None:
        backend = row.get("backend")
        if backend is None or pd.isna(backend) or str(backend) == "":
            return None
        return {"backend": str(backend)}

    def candidate_field(self, row) -> list[str]:
        """The backends measured for one shape: every backend that could
        serve it here, restricted by ``--backends``."""
        problem = _problem(row)
        allowed = self._restricted_backends()
        field = []
        for backend in candidate_backends(problem.gfx):
            if allowed is not None and backend not in allowed:
                continue
            if backend in MHA_FWD_TRITON_BACKENDS:
                reason = triton_unsupported_reason(problem, backend)
                if reason is not None:
                    print(f"not measuring {backend}: {reason}", flush=True)
                    continue
            if backend in SOFT_CAP_BLIND_BACKENDS and problem.logits_soft_cap:
                print(f"not measuring {backend}: no logits soft-cap", flush=True)
                continue
            field.append(backend)
        return field

    def tune(self, untunedf, tunedf, args):
        tasks, group_sizes = [], []
        for _, row in untunedf.iterrows():
            problem = _problem(row)
            key = problem.key()
            field = self.candidate_field(row)
            if not field:
                print(f"not tuning MHA {key}: no backend to measure", flush=True)
                continue
            print(f"tuning MHA {key}: {field}", flush=True)
            softmax_scale = problem.hdim_q**-0.5
            launch = (
                problem.max_seqlen_q,
                problem.max_seqlen_k,
                problem.min_seqlen_q,
                problem.dropout_p,
                softmax_scale,
                problem.logits_soft_cap,
                problem.how_v3_bf16_cvt,
                problem.causal,
                problem.window_left,
                problem.window_right,
                problem.return_lse,
            )
            for backend in field:
                tasks.append(
                    _Task(
                        info=(key, backend, self._resolved_config(problem, backend)),
                        gen_data=generate_data,
                        gen_args=_data_args(problem),
                        func=_run_candidate,
                        args=(INPUTS, backend, *launch),
                        kwargs={"num_warmup": args.warmup, "num_iters": args.iters},
                        ref_func=_chunked_reference,
                        ref_args=(INPUTS, *_reference_args(problem)),
                        ref_kwargs={},
                        ref=None,
                        rtol=ERROR_RTOL,
                        atol=_error_atol(problem),
                    )
                )
            group_sizes.append((len(field), ()))
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            group_sizes,
            args.mp,
            False,
            True,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
            return_status=True,
        )

    @staticmethod
    def _resolved_config(problem: MhaFwdProblem, backend: str) -> str:
        """The tile Triton resolves for this shape, as JSON; "" off Triton."""
        if backend not in MHA_FWD_TRITON_BACKENDS:
            return ""
        config = triton_config(problem, backend)
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def result_to_df(self, results):
        rows = []
        for info, us, err_ratio, status, detail in results:
            key, backend, resolved = info
            row = dict(zip(MHA_FWD_TUNING_KEY_FIELDS, key))
            row.update(
                {
                    "backend": backend,
                    "config_digest": (
                        config_digest(json.loads(resolved)) if resolved else ""
                    ),
                    "resolved_config": resolved,
                    "us": us,
                    "errRatio": err_ratio,
                    "status": status,
                    "detail": (
                        ""
                        if status == "ok"
                        else detail or f"candidate {status} with no diagnostic"
                    ),
                    "tflops": self.calculate((info, us, err_ratio)),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        """The fastest correct candidate per shape."""
        rets = super().post_process(rets, args)
        resultdf = self.result_to_df(rets)
        us = pd.to_numeric(resultdf["us"], errors="coerce")
        valid = (
            (resultdf["status"] == "ok")
            & (us > 0)
            & (us < float("inf"))
            & (resultdf["errRatio"] <= args.errRatio)
        )
        winners, failed = [], []
        for _, group in resultdf.groupby(self.keys, sort=False):
            if valid[group.index].any():
                winners.append(group[valid[group.index]].sort_values("us").iloc[0])
                continue
            row = group.iloc[0].copy()
            row["status"] = "failed"
            row["detail"] = "no correct candidate: " + "; ".join(
                f"{b} {s}" for b, s in zip(group["backend"], group["status"])
            )
            failed.append(row)
        winners = pd.DataFrame(winners, columns=self.columns).reset_index(drop=True)
        failed = pd.DataFrame(failed, columns=self.columns)
        for name, rows in (("success", winners), ("failed", failed)):
            old = getattr(self, name)
            if not rows.empty:
                setattr(self, name, rows if old.empty else pd.concat([old, rows]))
        return winners

    def result_to_csv(self, resultdf, file, concat=False):
        """Replace the rows for ``resultdf``'s shapes and keep every other row,
        with or without ``--all``, as ``GemmCommonTuner`` does."""
        new = self.stamp_hardware(resultdf)[list(MHA_FWD_RUNTIME_CSV_FIELDS)]
        new = new.astype(str)
        table = _read_table(file)
        replaced = {self.lookup_key(row) for _, row in new.iterrows()}
        kept = table[
            [self.lookup_key(row) not in replaced for _, row in table.iterrows()]
        ]
        _write_csv_atomic(pd.concat([kept, new], ignore_index=True), file, index=False)

    def sortResults(self, tune_file, issorted, values):
        """The table in the columns dispatch reads, one row per key; the
        compare merge brings in the result columns."""
        if not os.path.exists(tune_file):
            return
        table = _read_table(tune_file)
        if issorted:
            table = table.sort_values(list(values))
        keys = pd.Series([self.lookup_key(row) for _, row in table.iterrows()])
        _write_csv_atomic(
            table[~keys.duplicated(keep="last").to_numpy()], tune_file, index=False
        )

    def _restricted_backends(self) -> list[str] | None:
        return _names(getattr(self._args, "backends", "")) or None

    def _clear_op_caches(self):
        clear_mha_fwd_tuning_cache()


def selection_probe(row, warmup: int, iters: int) -> dict[str, Any]:
    """The child side of verify_dispatch: run one problem through the public
    operator, check it against the reference and time it."""
    problem = MhaFwdProblem.from_mapping(row)
    data = generate_data(*_data_args(problem))
    inputs = [data[name] for name in INPUTS]
    reference = _chunked_reference(*inputs, *_reference_args(problem))

    def invoke(*inputs):
        return _normalize_result(
            flash_attn_varlen_func(
                *inputs,
                problem.max_seqlen_q,
                problem.max_seqlen_k,
                min_seqlen_q=problem.min_seqlen_q,
                dropout_p=problem.dropout_p,
                softmax_scale=problem.hdim_q**-0.5,
                logits_soft_cap=problem.logits_soft_cap,
                causal=problem.causal,
                window_size=(
                    problem.window_left,
                    problem.window_right,
                    problem.sink_size,
                ),
                return_lse=problem.return_lse,
                return_attn_probs=problem.return_attn_probs,
                how_v3_bf16_cvt=problem.how_v3_bf16_cvt,
            ),
            problem.return_lse,
            problem.total_q,
            problem.nhead_q,
        )

    torch.testing.assert_close(
        invoke(*inputs), reference, rtol=ERROR_RTOL, atol=_error_atol(problem)
    )

    # Kernel time, as the candidates are timed: graph-captured serving pays
    # no host-side dispatch, so it must not decide a row. The inputs go in as
    # arguments so run_perftest rotates copies of them, as for the candidates;
    # a closure over them would be timed warm. The profiler needs at least two
    # iterations.
    iterations = max(2, int(iters))
    _, latency_us = run_perftest(
        invoke, *inputs, num_warmup=int(warmup), num_iters=iterations
    )
    return {"latency_us": float(latency_us), "iterations": iterations}


if __name__ == "__main__":
    if sys.argv[1:] == [dispatch_verification.SELECTION_PROBE_FLAG]:
        raise SystemExit(dispatch_verification.run_selection_probe(selection_probe))
    tuner = MhaFwdTuner()
    tuner.run(tuner.parse_args(), False)
