# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pick the fastest MHA forward backend per shape, across every backend.

The input CSV is a workload catalogue, the optional profile CSV is the
measurement record, and the output CSV is the exact-key runtime table that
``flash_attn_varlen_func`` reads: which backend serves each shape.

Each backend is measured as its owners ship it. CK, asm_v3, opus and FlyDSL
pick their own kernels; Triton and Gluon resolve their own tiles from
Triton's JSON configs, so this tuner never enumerates or writes a tile. A
Triton or Gluon row records the digest of the tile Triton resolved while it
was measured, and dispatch drops the row once Triton resolves another.

Candidates are timed on device kernel time, as the other aiter tuners time
theirs, so host launch cost does not decide between backends. A Triton
candidate's time includes copying its result into the caller's ``out``
buffer, which dispatch does for a Triton row.

``--triton-tiles`` (proposal) first runs the Triton harness for each shape,
unchanged and exhaustive. The harness installs into Triton's config tree, by
its own rules; this tuner then measures every backend, Triton as installed,
and puts the shape file back when that tile fails the reference, since the
harness checks no numerics.

What is MHA-specific lives here: the data generator, the chunked reference,
the candidate field and the probe. Deciding what reaches the table (the
incumbent, the promotion bar, the fresh-process proof) is ``GatedTuner``'s.
"""

import json
import math
import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, ClassVar

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import pandas as pd
import torch

from aiter.jit.core import AITER_CONFIG_MHA_FWD
from aiter.ops.mha import (
    clear_mha_fwd_tuning_cache,
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
    config_digest,
    triton_config,
    triton_unsupported_reason,
)
from aiter.utility import selection_proof
from aiter.utility.gated_tuner import GatedTuner
from aiter.utility.mp_tuner import MpTunerTask, mp_tuner

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
# Inputs this tuner cannot synthesize, so a catalogue row that needs one is
# refused rather than measured on something else.
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
# errRatio is the fraction of output elements outside these tolerances, and
# none may be. Stricter than op_tests/test_mha_varlen.py (rtol 0.02, atol 0.5).
ERROR_RTOL = 2e-2
ERROR_ATOL = 2e-2
MAX_ERROR_RATIO = 0.0
TASK_TIMEOUT_S = 7200
# Score storage the reference may hold at once.
REFERENCE_CHUNK_BYTES = 64 << 20
# Kernels that ignore logits_soft_cap; dispatch never routes a soft-capped
# call to them, so they cannot serve such a shape.
SOFT_CAP_BLIND_BACKENDS = ("asm_v3", "opus")
# Architectures whose incumbent kernels this tuner can time as dispatch runs
# them; gfx1250 dispatch picks an asm sink-varlen kernel it does not.
TUNABLE_GFX = ("gfx942", "gfx950")


def candidate_backends(gfx: str) -> tuple[str, ...]:
    """The backends worth measuring on one architecture.

    A backend that exists here but declines a shape (FlyDSL screens its own
    inputs) comes back from the worker as ``unsupported``.
    """
    backends = ["ck", "flydsl", "triton"]
    if gfx in ("gfx942", "gfx950"):
        backends.append("asm_v3")
    if gfx == "gfx950":
        backends.extend(("opus", "gluon"))
    return tuple(sorted(backends))


def _balanced_lengths(total: int, batch: int, maximum: int) -> list[int]:
    """Sequence lengths for a key: the first is ``maximum``, the rest share
    what is left as evenly as possible.

    The key holds only totals and maxima, so this one distribution stands in
    for every batch with the same key.
    """
    if batch < 1 or maximum < 1:
        raise ValueError("batch and max sequence lengths must be positive")
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
    if remaining != 0 or max(lengths) != maximum:
        raise ValueError("failed to construct the requested packed-varlen shape")
    if min(lengths) < 1:
        raise ValueError(
            f"total={total} over batch={batch} with maximum={maximum} "
            "leaves a sequence empty"
        )
    return lengths


def _prefix_sum(lengths: list[int], device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


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
    dtype_obj = getattr(torch, str(dtype))
    q_lengths = _balanced_lengths(int(total_q), int(batch), int(max_seqlen_q))
    k_lengths = _balanced_lengths(int(total_k), int(batch), int(max_seqlen_k))
    element_size = torch.empty((), dtype=dtype_obj).element_size()
    input_bytes = element_size * (
        int(total_q) * int(nhead_q) * int(hdim_q)
        + int(total_k) * int(nhead_k) * (int(hdim_q) + int(hdim_v))
    )
    output_bytes = element_size * int(total_q) * int(nhead_q) * int(hdim_v)
    split_scratch = 8 * output_bytes + 8 * int(total_q) * int(nhead_q) * 4
    required = input_bytes + 2 * output_bytes + split_scratch
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if required > int(free_bytes * 0.85):
        raise torch.cuda.OutOfMemoryError(
            f"MHA shape needs about {required / 2**30:.2f} GiB before backend workspace; "
            f"only {free_bytes / 2**30:.2f} GiB is free"
        )

    generator = torch.Generator(device=device).manual_seed(int(seed))

    def randn(*shape):
        return torch.randn(shape, dtype=dtype_obj, device=device, generator=generator)

    return {
        "q": randn(int(total_q), int(nhead_q), int(hdim_q)),
        "k": randn(int(total_k), int(nhead_k), int(hdim_q)),
        "v": randn(int(total_k), int(nhead_k), int(hdim_v)),
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
        if not return_lse:
            # Dispatch copies a Triton result into the caller's out buffer.
            out = torch.empty_like(result)
            out.copy_(result)
            result = out
        return _normalize_result(result, return_lse, total_q, nhead_q)
    if backend == "flydsl":
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_func

        out = flydsl_flash_attn_varlen_func(
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


def _problem(row) -> MhaFwdProblem:
    return MhaFwdProblem.from_mapping(
        {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
    )


def _seed(problem: MhaFwdProblem) -> int:
    return zlib.crc32(",".join(problem.key()).encode("utf-8"))


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


class MhaFwdTuner(GatedTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GatedTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_MHA_FWD,
        "untune_file": f"aiter/configs/{MHA_FWD_UNTUNED_CSV}",
        "batch": 8,
        "errRatio": MAX_ERROR_RATIO,
        "timeout": TASK_TIMEOUT_S,
        "config_env_name": MHA_FWD_CONFIG_ENV,
    }
    SELECTION_PROBE_MODULE = "op_tests.tuners.tune_mha_fwd"
    SELECTION_FAMILY = MHA_FWD_FAMILY
    RUNTIME_FIELDS = MHA_FWD_RUNTIME_CSV_FIELDS
    CANDIDATE_FIELDS = ("backend",)

    def __init__(self):
        super().__init__(
            "MhaFwdTuner",
            list(MHA_FWD_TUNING_KEY_FIELDS),
            list(RESULT_FIELDS),
            "Pick the fastest packed-varlen MHA forward backend per shape. "
            "Shapes that already have a row are skipped unless --all is given.",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--backends",
            default="",
            help=(
                "comma-separated backends to measure besides what serves the "
                "shape today (default: all). The incumbent is always measured"
            ),
        )
        self.parser.add_argument(
            "--triton-tiles",
            action="store_true",
            help=(
                "before measuring, tune the Triton tile of each shape with the "
                "Triton harness (sweep_configs.py mha_varlen_fwd), which installs "
                "into Triton's config tree"
            ),
        )
        self.parser.add_argument(
            "--triton-runs-dir",
            default="triton_tile_runs",
            help="the harness's --runs-dir for --triton-tiles (resumable)",
        )

    def _setup_common_arguments(self):
        super()._setup_common_arguments()
        defaults = self.get_arg_defaults()
        self.parser.set_defaults(
            verbose=False,
            splitK=False,
            shape_grouped=True,
            sort=False,
            errRatio=defaults["errRatio"],
            batch=defaults["batch"],
            all=False,
        )

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
        if not args.untune_file or not os.path.isfile(args.untune_file):
            raise FileNotFoundError(f"MHA problem CSV not found: {args.untune_file}")
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
            _balanced_lengths(
                int(row["total_q"]), int(row["batch"]), int(row["max_seqlen_q"])
            )
            _balanced_lengths(
                int(row["total_k"]), int(row["batch"]), int(row["max_seqlen_k"])
            )
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
        self.load_published(args)

    def lookup_key(self, row) -> tuple[str, ...]:
        return _problem(row).key()

    def candidate_from_selection(self, selection) -> dict[str, str] | None:
        backend = selection.get("backend")
        return {"backend": backend} if backend in MHA_FWD_BACKENDS else None

    def expected_selection(self, row) -> dict[str, Any] | None:
        backend = row.get("backend")
        if backend is None or pd.isna(backend) or str(backend) == "":
            return None
        return {"backend": str(backend)}

    def candidate_field(self, row) -> list[str]:
        """The backends measured for one shape: every backend that could
        serve it here, plus whatever serves it today."""
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
        for incumbent in self.incumbents(row):
            if incumbent["backend"] not in field:
                field.append(incumbent["backend"])
        return field

    def tune(self, untunedf, tunedf, args):
        installed = {}
        if args.triton_tiles:
            installed = self._tune_triton_tiles(untunedf, args)
        results = self._measure(untunedf, args)
        self._roll_back_wrong_tiles(results, installed, args)
        return results

    def _tune_triton_tiles(self, untunedf, args) -> dict[tuple, tuple[str, str | None]]:
        """Have the Triton harness install a tile per shape. Returns
        key -> (shape file, previous text)."""
        installed = {}
        for _, row in untunedf.iterrows():
            problem = _problem(row)
            reason = triton_unsupported_reason(problem, "triton") or harness_reason(
                problem
            )
            if reason is not None:
                print(f"not tuning a Triton tile for {problem.key()}: {reason}")
                continue
            path = triton_shape_file(problem)
            previous = Path(path).read_text() if os.path.exists(path) else None
            command = harness_command(problem, args)
            print("triton tiles:", " ".join(command[1:]), flush=True)
            returncode = subprocess.run(
                command,
                cwd=TRITON_HARNESS_DIR,
                env=dict(os.environ, PYTHONPATH=REPO_ROOT),
                check=False,
            ).returncode
            if returncode != 0:
                print(f"the Triton harness failed ({returncode}) for {problem.key()}")
            installed[problem.key()] = (path, previous)
        clear_triton_config_caches()
        return installed

    def _roll_back_wrong_tiles(self, results, installed, args) -> None:
        """The harness checks no numerics: put back a shape file whose tile
        fails the reference here."""
        for (key, backend, _), _, err_ratio, status, _ in results:
            if backend != "triton" or key not in installed:
                continue
            if status == "ok" and err_ratio <= args.errRatio:
                continue
            path, previous = installed[key]
            if previous is None:
                if os.path.exists(path):
                    os.remove(path)
            else:
                Path(path).write_text(previous)
            print(
                f"{key}: the installed Triton tile failed ({status}); {path} restored"
            )
        clear_triton_config_caches()

    def _measure(self, untunedf, args):
        tasks, group_sizes = [], []
        for _, row in untunedf.iterrows():
            problem = _problem(row)
            key = problem.key()
            field = self.candidate_field(row)
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
                    MpTunerTask(
                        info=(key, backend, self._resolved_config(problem, backend)),
                        gen_data=generate_data,
                        gen_args=_data_args(problem),
                        func=_run_candidate,
                        args=(["q", "k", "v", "cu_q", "cu_k"], backend, *launch),
                        kwargs=self.measurement_kwargs(args),
                        ref_func=_chunked_reference,
                        ref_args=(
                            ["q", "k", "v", "cu_q", "cu_k"],
                            softmax_scale,
                            problem.causal,
                            problem.window_left,
                            problem.window_right,
                            problem.return_lse,
                            problem.logits_soft_cap,
                        ),
                        ref_kwargs={},
                        ref=None,
                        rtol=ERROR_RTOL,
                        atol=ERROR_ATOL,
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
        """The tile Triton resolves for this shape, as JSON, or "" for a
        backend that keeps its configs elsewhere."""
        if backend not in MHA_FWD_TRITON_BACKENDS:
            return ""
        config = triton_config(problem, backend)
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def result_to_df(self, results):
        """One evidence row per candidate, from mp_tuner's typed results."""
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

    def _restricted_backends(self) -> list[str] | None:
        return _names(getattr(self._args, "backends", "")) or None

    def _clear_op_caches(self):
        clear_mha_fwd_tuning_cache()


REPO_ROOT = str(Path(__file__).resolve().parents[2])
TRITON_HARNESS_DIR = os.path.join(
    REPO_ROOT, "aiter", "ops", "triton", "utils", "_triton", "tuning"
)


def harness_reason(problem: MhaFwdProblem) -> str | None:
    """Why the Triton harness's mha_varlen_fwd cannot stand for this shape."""
    if problem.dtype != "bfloat16":
        return "the harness times bf16"
    if (problem.window_left, problem.window_right) != (-1, -1):
        return "the harness times no sliding window"
    if (
        problem.total_q != problem.batch * problem.max_seqlen_q
        or problem.total_k != problem.batch * problem.max_seqlen_k
    ):
        return "the harness times equal-length sequences"
    return None


def triton_shape_file(problem: MhaFwdProblem) -> str:
    from aiter.ops.triton._triton_kernels.attention.mha import _shape_config_name
    from aiter.ops.triton.utils.config_utils import resolve_config_dir

    return os.path.join(
        resolve_config_dir("attention", "MHA", backend="triton"),
        _shape_config_name(problem.hdim_q, problem.hdim_v, problem.causal),
    )


def harness_command(problem: MhaFwdProblem, args) -> list[str]:
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]
    return [
        sys.executable,
        os.path.join(TRITON_HARNESS_DIR, "sweep_configs.py"),
        "mha_varlen_fwd",
        *("--Z", str(problem.batch), "--HQ", str(problem.nhead_q)),
        *("--HK", str(problem.nhead_k), "--SQ", str(problem.max_seqlen_q)),
        *("--SK", str(problem.max_seqlen_k), "--DQ", str(problem.hdim_q)),
        *("--DV", str(problem.hdim_v), "--CAUSAL", str(int(problem.causal))),
        *("--gpu", gpu, "--runs-dir", os.path.abspath(args.triton_runs_dir)),
    ]


def clear_triton_config_caches() -> None:
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config
    from aiter.ops.triton.utils.config_utils import load_config_json

    _get_config.cache_clear()
    load_config_json.cache_clear()


def selection_probe(row, warmup: int, iters: int) -> dict[str, Any]:
    """Run one problem through the public operator, the way serving calls it.

    The child side of TunerCommon.prove_selection: dispatch records what it
    selected, this checks the output against the reference and times it.
    """

    problem = MhaFwdProblem.from_mapping(row)
    data = generate_data(*_data_args(problem))
    softmax_scale = problem.hdim_q**-0.5
    reference = _chunked_reference(
        data["q"],
        data["k"],
        data["v"],
        data["cu_q"],
        data["cu_k"],
        softmax_scale,
        problem.causal,
        problem.window_left,
        problem.window_right,
        problem.return_lse,
        problem.logits_soft_cap,
    )

    def invoke():
        return _normalize_result(
            flash_attn_varlen_func(
                data["q"],
                data["k"],
                data["v"],
                data["cu_q"],
                data["cu_k"],
                problem.max_seqlen_q,
                problem.max_seqlen_k,
                min_seqlen_q=problem.min_seqlen_q,
                dropout_p=problem.dropout_p,
                softmax_scale=softmax_scale,
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

    observed = invoke()
    reference_items = reference if isinstance(reference, tuple) else (reference,)
    observed_items = observed if isinstance(observed, tuple) else (observed,)
    if len(reference_items) != len(observed_items):
        raise AssertionError("public MHA result structure differs from reference")
    for expected, actual in zip(reference_items, observed_items):
        torch.testing.assert_close(actual, expected, rtol=ERROR_RTOL, atol=ERROR_ATOL)

    iterations = max(1, int(iters))
    for _ in range(max(0, int(warmup))):
        invoke()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        invoke()
    end.record()
    end.synchronize()
    latency_us = float(start.elapsed_time(end) * 1000.0 / iterations)
    return {"latency_us": latency_us, "iterations": iterations}


if __name__ == "__main__":
    if sys.argv[1:] == [selection_proof.SELECTION_PROBE_FLAG]:
        raise SystemExit(selection_proof.run_selection_probe(selection_probe))
    tuner = MhaFwdTuner()
    tuner.run(tuner.parse_args(), False)
