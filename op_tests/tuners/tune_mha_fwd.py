#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exhaustive, correctness-gated tuner for packed-varlen MHA forward.

The input CSV is a workload catalogue, the optional profile CSV plus journal
are measurement evidence, and the output CSV is the exact-key runtime dispatch
artifact: backend, num_splits, and backend_config for the winning candidate.
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import triton  # noqa: F401  # ROCm environments may require Triton before torch.
import torch

from aiter.jit.core import AITER_CONFIG_MHA_FWD
from aiter.jit.utils.chip_info import get_gpu_model
from aiter.ops.mha import (
    _fmha_v3_varlen_splitkv_fwd,
    _load_mha_fwd_tuning_table,
    flash_attn_varlen_func,
    fmha_fwd_bf16_opus_varlen_fwd,
    mha_varlen_fwd,
)
from aiter.ops.mha_fwd_policy import (
    _as_bool,
    MHA_FWD_CANDIDATE_FIELDS,
    MHA_FWD_HARDWARE_KEY_FIELDS,
    MHA_FWD_INDIFFERENCE_DELTA,
    MHA_FWD_METRIC_FIELDS,
    MHA_FWD_PROBLEM_KEY_FIELDS,
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_SEARCH_STRATEGIES,
    MHA_FWD_SIGNIFICANCE_SIGMA,
    MHA_FWD_TILE_CONFIG_BACKENDS,
    MHA_FWD_TUNING_KEY_FIELDS,
    MhaFwdCandidate,
    MhaFwdProblem,
    canonical_backend_config,
    enumerate_mha_fwd_candidates,
    mha_fwd_candidate_id,
)
from aiter.test_common import checkAllclose
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.block_race import (
    JsonlBlockJournal,
    RaceEntrant,
    cuda_event_timer,
    race,
)
from aiter.utility.mp_tuner import mp_tuner


# Fixed so that --race-candidates draws the same subset on every run; a
# sampled field that changed between runs would make two runs incomparable.
MHA_FWD_RACE_SAMPLE_SEED = 20240917

UNTUNED_FIELDS = MHA_FWD_PROBLEM_KEY_FIELDS
RESULT_FIELDS = (*MHA_FWD_CANDIDATE_FIELDS, *MHA_FWD_METRIC_FIELDS)
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


def _balanced_lengths(total: int, batch: int, maximum: int) -> list[int]:
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
    q = torch.randn(
        (int(total_q), int(nhead_q), int(hdim_q)),
        dtype=dtype_obj,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (int(total_k), int(nhead_k), int(hdim_q)),
        dtype=dtype_obj,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (int(total_k), int(nhead_k), int(hdim_v)),
        dtype=dtype_obj,
        device=device,
        generator=generator,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
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
):
    """Bounded-memory fp32 oracle; score storage stays below roughly 64 MiB."""
    output = torch.empty(
        (*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device
    )
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
        rows = max(1, (64 << 20) // max(4, 4 * heads * sk))
        key_positions = torch.arange(sk, device=q.device)
        offset = sk - sq
        for row_begin in range(0, sq, rows):
            row_end = min(sq, row_begin + rows)
            q_chunk = q_seq[row_begin:row_end].float().reshape(
                row_end - row_begin, k_seq.shape[1], groups, q_seq.shape[-1]
            )
            scores = torch.einsum("qhgd,khd->hgqk", q_chunk, k_seq).reshape(
                heads, row_end - row_begin, sk
            )
            scores.mul_(float(softmax_scale))
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
                probabilities.reshape(
                    k_seq.shape[1], groups, row_end - row_begin, sk
                ),
                v_seq,
            ).reshape(row_end - row_begin, heads, v_seq.shape[-1])
            output[q_begin + row_begin : q_begin + row_end].copy_(out.to(q.dtype))
            if lse is not None:
                lse[
                    :, q_begin + row_begin : q_begin + row_end
                ] = torch.logsumexp(scores, dim=-1)
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
    num_splits,
    config,
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
    if backend == "asm_v3":
        out, lse, _, _ = _fmha_v3_varlen_splitkv_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            float(softmax_scale),
            bool(return_lse),
            int(num_splits),
        )
        return _normalize_result(
            (out, lse), return_lse, q.shape[0], q.shape[1]
        )
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
        return _normalize_result(
            (out, lse), return_lse, q.shape[0], q.shape[1]
        )
    if backend in ("triton", "gluon"):
        from aiter.ops.triton.attention.mha import flash_attn_varlen_func

        result = flash_attn_varlen_func(
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
            config=config,
            backend=backend,
        )
        return _normalize_result(result, return_lse, q.shape[0], q.shape[1])
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
        return _normalize_result(out, return_lse, q.shape[0], q.shape[1])
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
        return _normalize_result(result, return_lse, q.shape[0], q.shape[1])
    raise ValueError(f"unknown backend {backend!r}")


class MhaFwdTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_MHA_FWD,
        "untune_file": "aiter/configs/untuned_mha_fwd.csv",
        "batch": 8,
        "errRatio": 0.0,
        "timeout": 7200,
        "config_env_name": "AITER_CONFIG_MHA_FWD",
        "finalist_rounds": 3,
    }

    def __init__(self):
        super().__init__(
            "MhaFwdTuner",
            list(MHA_FWD_TUNING_KEY_FIELDS),
            list(RESULT_FIELDS),
            "Exhaustively tune packed-varlen MHA forward",
        )
        self._samples_by_info: dict[tuple, tuple[float, ...]] = {}
        self._journal_path = ""
        self._evidence_path = ""
        self._args = None
        self._run_started_at = 0.0
        self._last_results = pd.DataFrame(columns=self.columns)
        self._all_results = pd.DataFrame(columns=self.columns)
        self._selection_proofs: list[dict[str, Any]] = []
        self._incumbents_by_key: dict[tuple, set[tuple[str, str]]] = {}
        self._promotions: list[dict[str, Any]] = []
        self._race_reports: list[dict[str, Any]] = []
        self._race_winner_by_key: dict[tuple, tuple] = {}

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--journal-file",
            default="",
            help="append-only candidate checkpoint (default: <profile-or-output>.journal.jsonl)",
        )
        self.parser.add_argument(
            "--evidence-file",
            default="",
            help="atomic run manifest (default: <output>.evidence.json)",
        )
        self.parser.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="resume completed candidate phases from the checkpoint journal",
        )
        self.parser.add_argument(
            "--finalist-rounds",
            type=int,
            default=self.get_arg_defaults()["finalist_rounds"],
            help="fresh-worker measurement rounds for each top-eight finalist",
        )
        self.parser.add_argument(
            "--strategy",
            choices=(*MHA_FWD_SEARCH_STRATEGIES, "race"),
            default="exhaustive",
            help=(
                "candidate search strategy; smoke samples each tile grid so the "
                "measure-publish-replay path can be exercised without the full "
                "catalogue, and records itself in the evidence. race measures "
                "the exhaustive catalogue as an interleaved elimination race "
                "instead of screening every candidate once in its own worker"
            ),
        )
        self.parser.add_argument(
            "--delta",
            type=float,
            default=MHA_FWD_INDIFFERENCE_DELTA,
            help=(
                "race only: the indifference zone. Candidates within this of "
                "the leader are treated as settled rather than as a harder "
                "question, and the same threshold decides whether a challenger "
                "displaces the incumbent, so one definition of "
                "indistinguishable holds end to end"
            ),
        )
        self.parser.add_argument(
            "--race-alpha",
            type=float,
            default=0.05,
            help="race only: error budget, spread over every candidate and look",
        )
        self.parser.add_argument(
            "--race-block-calls",
            type=int,
            default=10,
            help="race only: timed calls per candidate per block",
        )
        self.parser.add_argument(
            "--race-min-blocks",
            type=int,
            default=3,
            help="race only: blocks before any candidate may be eliminated",
        )
        self.parser.add_argument(
            "--race-max-blocks",
            type=int,
            default=30,
            help=(
                "race only: ceiling on blocks. Reaching it without certifying "
                "returns a ranking rather than a guarantee, and the evidence "
                "records that it was not certified"
            ),
        )
        self.parser.add_argument(
            "--candidate-sample",
            type=int,
            default=None,
            help=(
                "draw a seeded sample of this many catalogue entries instead of "
                "the whole catalogue. Applies to every strategy, because the "
                "point is to give two strategies the identical field when "
                "comparing them; the sample is fixed by seed so two runs are "
                "comparable, and the evidence records that it was sampled"
            ),
        )
        self.parser.add_argument(
            "--backends",
            default="",
            help=(
                "comma-separated backends to measure (default: all). A control "
                "for comparing contracts, not a tuning mode: the winner is the "
                "fastest of what was allowed to run, and the evidence says so"
            ),
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
        key = dict(zip(MHA_FWD_TUNING_KEY_FIELDS, info[0]))
        flop = (
            2
            * int(key["nhead_q"])
            * int(key["total_q"])
            * int(key["total_k"])
            * (int(key["hdim_q"]) + int(key["hdim_v"]))
            / max(1, int(key["batch"]))
        )
        return 0.0 if us <= 0 or not math.isfinite(us) else flop / us / 1e6

    def pre_process(self, args):
        if args.finalist_rounds < 1:
            raise ValueError("--finalist-rounds must be positive")
        if not args.untune_file or not os.path.isfile(args.untune_file):
            raise FileNotFoundError(f"MHA problem CSV not found: {args.untune_file}")
        frame = pd.read_csv(args.untune_file)
        missing = [field for field in UNTUNED_FIELDS if field not in frame.columns]
        if missing:
            raise ValueError(f"MHA problem CSV is missing columns: {missing}")
        if (frame["mode"].astype(str) != "varlen").any():
            raise ValueError("the forward tuner currently accepts mode=varlen rows")
        unsupported = [
            "return_attn_probs",
            "has_bias",
            "has_alibi",
            "has_sink",
            "has_block_table",
            "has_q_descale",
            "has_physical_padding",
            "is_grad",
        ]
        frame = frame.assign(
            **{
                field: frame[field].map(_as_bool).astype(int)
                for field in BOOL_FIELDS
            }
        )
        if frame[unsupported].astype(bool).any(axis=None):
            raise ValueError(
                "this tuner input schema cannot synthesize bias/alibi/sink/paging/"
                "quantization/physical-padding/training payloads"
            )
        if (frame["dropout_p"].astype(float) != 0.0).any():
            raise ValueError("dropout tuning requires a reproducible dropout mask")
        if (frame["logits_soft_cap"].astype(float) != 0.0).any():
            raise ValueError("the enumerated Triton/FlyDSL paths require logits_soft_cap=0")
        if (frame["sink_size"].astype(int) != 0).any():
            raise ValueError("sink-token tuning requires explicit sink-token payloads")
        if (frame["how_v3_bf16_cvt"].astype(int) != 1).any():
            raise ValueError("only how_v3_bf16_cvt=1 is reproducibly enumerable")
        frame = frame.copy()
        hardware = {
            "gfx": self.get_gfx(),
            "gpu_model": get_gpu_model(torch.cuda.current_device()),
            "cu_num": self.get_cu_num(),
        }
        for field, live_value in hardware.items():
            if field in frame.columns:
                mismatched = frame[field].astype(str).str.lower() != str(
                    live_value
                ).lower()
                if mismatched.any():
                    raise ValueError(
                        f"workload catalogue {field} does not match the tuning GPU "
                        f"({live_value})"
                    )
            frame[field] = pd.Series(live_value, index=frame.index)
        self.untunedf = frame[list(MHA_FWD_TUNING_KEY_FIELDS)].drop_duplicates()
        if os.path.exists(args.tune_file):
            self.tunedf = pd.read_csv(args.tune_file)
        else:
            self.tunedf = pd.DataFrame(columns=MHA_FWD_RUNTIME_CSV_FIELDS)
        self._args = args
        self._run_started_at = time.time()
        evidence_base = args.profile_file or args.tune_file
        self._journal_path = args.journal_file or f"{evidence_base}.journal.jsonl"
        self._evidence_path = args.evidence_file or f"{args.tune_file}.evidence.json"
        if not args.resume and os.path.exists(self._journal_path):
            os.remove(self._journal_path)

    @staticmethod
    def _problem_and_candidate(info):
        key, backend, num_splits, backend_config = info
        problem = MhaFwdProblem.from_mapping(
            dict(zip(MHA_FWD_TUNING_KEY_FIELDS, key))
        )
        candidate = MhaFwdCandidate(
            backend,
            int(num_splits),
            json.loads(backend_config) if backend_config else None,
        )
        return problem, candidate

    @staticmethod
    def _atomic_write_csv(frame: pd.DataFrame, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        try:
            frame.to_csv(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_write_json(payload: dict[str, Any], path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _append_journal_result(self, phase: str, result) -> None:
        info, us, err_ratio, status = result[:4]
        detail = result[4] if len(result) > 4 else ""
        problem, candidate = self._problem_and_candidate(info)
        record = {
            "schema_version": 2,
            "candidate_id": mha_fwd_candidate_id(problem, candidate),
            "phase": phase,
            "problem": problem.as_row(),
            "candidate": {
                "backend": candidate.backend,
                "num_splits": candidate.num_splits,
                "backend_config": candidate.config_json,
            },
            "status": status,
            "detail": str(detail),
            "us": float(us) if math.isfinite(float(us)) else None,
            "errRatio": float(err_ratio),
            "recorded_at_unix_s": time.time(),
        }
        destination = Path(self._journal_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        with os.fdopen(descriptor, "ab") as file:
            file.write(
                (
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            file.flush()
            os.fsync(file.fileno())

    def _load_journal(self, force: bool = False) -> dict[tuple[str, str], tuple]:
        records: dict[tuple[str, str], tuple] = {}
        if (not self._args.resume and not force) or not os.path.isfile(
            self._journal_path
        ):
            return records
        # A process kill can leave one partial final write. Remove only that
        # unterminated suffix before any resumed append so future records do
        # not become concatenated onto invalid JSON.
        with open(self._journal_path, "rb+") as file:
            payload = file.read()
            if payload and not payload.endswith(b"\n"):
                last_newline = payload.rfind(b"\n")
                file.truncate(last_newline + 1)
        with open(self._journal_path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    problem = MhaFwdProblem.from_mapping(record["problem"])
                    candidate_row = record["candidate"]
                    candidate = MhaFwdCandidate(
                        candidate_row["backend"],
                        int(candidate_row["num_splits"]),
                        (
                            json.loads(candidate_row["backend_config"])
                            if candidate_row.get("backend_config")
                            else None
                        ),
                    )
                    candidate_id = mha_fwd_candidate_id(problem, candidate)
                    if candidate_id != record["candidate_id"]:
                        raise ValueError("candidate ID does not match record payload")
                    info = (
                        problem.key(),
                        candidate.backend,
                        candidate.num_splits,
                        candidate.config_json,
                    )
                    result = (
                        info,
                        (
                            float(record["us"])
                            if record.get("us") is not None
                            else float("inf")
                        ),
                        float(record["errRatio"]),
                        str(record["status"]),
                        str(record.get("detail", "")),
                    )
                    records[(candidate_id, str(record["phase"]))] = result
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(
                        f"ignoring incomplete journal record {line_number}: {exc}",
                        file=sys.stderr,
                    )
        return records

    def _catalogue_strategy(self) -> str:
        """Which catalogue the candidates come from.

        ``race`` names how the field is measured, not which field it is, so it
        draws from the full catalogue. Keeping the two apart means the race is
        never quietly handed a sampled grid and reported as a complete search.
        """
        strategy = getattr(self._args, "strategy", "exhaustive")
        return "exhaustive" if strategy == "race" else strategy

    def tune(self, untunedf, tunedf, args):
        all_infos, task_by_info, plans = self._plan_candidates(untunedf, args)
        if getattr(args, "strategy", "exhaustive") == "race":
            return self._tune_by_race(args, untunedf, plans, all_infos)
        return self._tune_by_screening(args, untunedf, all_infos, task_by_info)

    def _plan_candidates(self, untunedf, args):
        """Enumerate every candidate once, for whichever measurement follows.

        Screening and racing need the same three things -- the candidate list,
        the arguments that build the data, and the arguments that launch a
        candidate on it -- so they are assembled here rather than twice.
        """
        all_infos = []
        task_by_info = {}
        plans = []
        for row_index, row in untunedf.iterrows():
            problem = MhaFwdProblem.from_mapping(
                {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
            )
            key = problem.key()
            softmax_scale = int(row.hdim_q) ** -0.5
            candidates = list(
                enumerate_mha_fwd_candidates(
                    str(row.gfx),
                    self._catalogue_strategy(),
                    self._restricted_backends(),
                )
            )
            sample = getattr(args, "candidate_sample", None)
            if sample is not None and sample < len(candidates):
                candidates = random.Random(MHA_FWD_RACE_SAMPLE_SEED).sample(
                    candidates, sample
                )
            # Measuring what the kernel resolves today, in the same sweep and
            # on the same GPU, is what lets the run tell an improvement from a
            # reordering of noise. Without it the comparison is against a
            # number from another session.
            incumbents = self._incumbent_candidates(row)
            known = {candidate.identity for candidate in candidates}
            for incumbent in incumbents:
                if incumbent.identity not in known:
                    candidates.append(incumbent)
            self._incumbents_by_key[key] = {
                (candidate.backend, canonical_backend_config(candidate.backend_config))
                for candidate in incumbents
            }
            candidates = tuple(candidates)
            print(
                f"tuning MHA row {row_index}: {len(candidates)} candidates for {key}",
                flush=True,
            )
            gen_args = (
                int(row.batch),
                int(row.total_q),
                int(row.total_k),
                int(row.max_seqlen_q),
                int(row.max_seqlen_k),
                int(row.nhead_q),
                int(row.nhead_k),
                int(row.hdim_q),
                int(row.hdim_v),
                problem.dtype,
                zlib.crc32(",".join(map(str, key)).encode("utf-8")),
            )
            for candidate in candidates:
                info = (
                    key,
                    candidate.backend,
                    candidate.num_splits,
                    candidate.config_json,
                )
                all_infos.append(info)
                config = (
                    dict(candidate.backend_config)
                    if candidate.backend_config is not None
                    else None
                )
                task = (
                    info,
                    generate_data,
                    gen_args,
                    _run_candidate,
                    (
                        ["q", "k", "v", "cu_q", "cu_k"],
                        candidate.backend,
                        candidate.num_splits,
                        config,
                        int(row.max_seqlen_q),
                        int(row.max_seqlen_k),
                        int(row.min_seqlen_q),
                        float(row.dropout_p),
                        softmax_scale,
                        float(row.logits_soft_cap),
                        int(row.how_v3_bf16_cvt),
                        bool(row.causal),
                        int(row.window_left),
                        int(row.window_right),
                        bool(row.return_lse),
                    ),
                    {
                        "num_warmup": args.warmup,
                        "num_iters": args.iters,
                        "use_cuda_event": True,
                    },
                    _chunked_reference,
                    (
                        ["q", "k", "v", "cu_q", "cu_k"],
                        softmax_scale,
                        bool(row.causal),
                        int(row.window_left),
                        int(row.window_right),
                        bool(row.return_lse),
                    ),
                    {},
                    None,
                    2e-2,
                    2e-2,
                )
                task_by_info[info] = task

            plans.append(
                {
                    "key": key,
                    "row": row,
                    "gen_args": gen_args,
                    "candidates": candidates,
                    "softmax_scale": softmax_scale,
                    # Everything _run_candidate needs after the five tensors.
                    "launch_tail": (
                        int(row.max_seqlen_q),
                        int(row.max_seqlen_k),
                        int(row.min_seqlen_q),
                        float(row.dropout_p),
                        softmax_scale,
                        float(row.logits_soft_cap),
                        int(row.how_v3_bf16_cvt),
                        bool(row.causal),
                        int(row.window_left),
                        int(row.window_right),
                        bool(row.return_lse),
                    ),
                }
            )

        return all_infos, task_by_info, plans

    def _tune_by_screening(self, args, untunedf, all_infos, task_by_info):
        journal = self._load_journal()
        problem_keys = [
            MhaFwdProblem.from_mapping(
                {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
            ).key()
            for _, row in untunedf.iterrows()
        ]

        def pending_for_phase(phase: str, infos):
            pending_tasks = []
            pending_data = []
            for key in problem_keys:
                tasks_for_shape = []
                for info in infos:
                    if info[0] != key:
                        continue
                    problem, candidate = self._problem_and_candidate(info)
                    if (
                        mha_fwd_candidate_id(problem, candidate),
                        phase,
                    ) not in journal:
                        tasks_for_shape.append(task_by_info[info])
                if tasks_for_shape:
                    pending_tasks.extend(tasks_for_shape)
                    pending_data.append((len(tasks_for_shape), ()))
            return pending_tasks, pending_data

        tasks, tasks_data = pending_for_phase("first", all_infos)
        while tasks:
            before = len(journal)
            mp_tuner(
                tasks,
                tasks_data,
                args.mp,
                False,
                True,
                args.errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
                return_status=True,
                result_callback=lambda result: self._append_journal_result(
                    "first", result
                ),
            )
            journal = self._load_journal(force=True)
            if len(journal) <= before:
                raise RuntimeError("MHA checkpoint made no progress")
            tasks, tasks_data = pending_for_phase("first", all_infos)

        first_by_info = {
            result[0]: result
            for (_, phase), result in journal.items()
            if phase == "first"
        }
        first_pass = [first_by_info[info] for info in all_infos if info in first_by_info]

        finalists_by_key: dict[tuple, list[tuple]] = {}
        for result in first_pass:
            info, us, err_ratio, status = result[:4]
            if (
                status == "ok"
                and us > 0
                and math.isfinite(us)
                and err_ratio <= args.errRatio
            ):
                finalists_by_key.setdefault(info[0], []).append(result)

        finalist_infos = []
        for _, row in untunedf.iterrows():
            key = MhaFwdProblem.from_mapping(
                {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
            ).key()
            finalist_infos.extend(
                result[0]
                for result in sorted(
                    finalists_by_key.get(key, ()), key=lambda item: item[1]
                )[:8]
            )
        if not finalist_infos:
            return first_pass

        print(
            f"re-measuring {len(finalist_infos)} finalists for "
            f"{args.finalist_rounds} fresh-worker rounds",
            flush=True,
        )
        round_results: dict[tuple, list[tuple]] = {
            info: [] for info in finalist_infos
        }
        for round_index in range(args.finalist_rounds):
            phase = f"finalist:{round_index}"
            finalist_tasks, finalist_data = pending_for_phase(
                phase, finalist_infos
            )
            while finalist_tasks:
                before = len(journal)
                mp_tuner(
                    finalist_tasks,
                    finalist_data,
                    args.mp,
                    False,
                    True,
                    args.errRatio,
                    timeout=args.timeout,
                    verbose=args.verbose,
                    return_status=True,
                    result_callback=lambda result, phase=phase: (
                        self._append_journal_result(phase, result)
                    ),
                )
                journal = self._load_journal(force=True)
                if len(journal) <= before:
                    raise RuntimeError(
                        f"MHA checkpoint made no progress during {phase}"
                    )
                finalist_tasks, finalist_data = pending_for_phase(
                    phase, finalist_infos
                )
            for info in finalist_infos:
                problem, candidate = self._problem_and_candidate(info)
                round_results[info].append(
                    journal[(mha_fwd_candidate_id(problem, candidate), phase)]
                )

        final_by_info = {}
        for info, results in round_results.items():
            statuses = [result[3] for result in results]
            failed_status = next((status for status in statuses if status != "ok"), None)
            samples = tuple(
                float(result[1])
                for result in results
                if result[3] == "ok" and math.isfinite(result[1]) and result[1] > 0
            )
            self._samples_by_info[info] = samples
            if failed_status is not None or len(samples) != args.finalist_rounds:
                status = failed_status or "crash"
                us = float("inf")
                detail = next(
                    (
                        str(result[4])
                        for result in results
                        if len(result) > 4 and result[3] != "ok" and result[4]
                    ),
                    f"only {len(samples)} of {args.finalist_rounds} "
                    "finalist rounds produced a measurement",
                )
            else:
                status = "ok"
                us = float(statistics.median(samples))
                detail = ""
            err_ratio = max((float(result[2]) for result in results), default=1.0)
            final_by_info[info] = (info, us, err_ratio, status, detail)

        return [final_by_info.get(result[0], result) for result in first_pass]

    def _tune_by_race(self, args, untunedf, plans, all_infos):
        """Measure each shape as one interleaved elimination race.

        Screening measures every candidate once, alone, in its own worker, so
        a single contended measurement drops a candidate permanently and the
        top eight have to be re-measured to undo that. A race does not have
        that weakness: every survivor is measured in every block, paired
        against the others, and leaving the field requires statistical proof
        rather than one unlucky sample. The finalist rounds it replaces exist
        only to patch the weakness, so they go with it.

        Nothing re-measures the winner afterwards. The one thing a
        single-process race holds fixed is the tensor allocation, but a
        process-wide effect moves every candidate together and so cannot
        change their order; only an allocation-by-candidate interaction could,
        and separating that from noise needs many fresh processes rather than
        one. A single extra round is weaker evidence than the blocks it would
        overrule.
        """
        results = []
        for plan in plans:
            results.extend(self._race_one_shape(args, plan))
        return results

    def _race_block_journal(self, key):
        """Per-block checkpoint for one shape, beside the candidate journal.

        The candidate journal records finished candidate-phases; a race's unit
        of durable progress is the finished block. The records cannot share a
        file, but they share the ``--resume`` flag and the directory, so there
        is still only one thing for an operator to know about.
        """
        if not self._journal_path:
            return None
        digest = sha256(",".join(map(str, key)).encode("utf-8")).hexdigest()[:12]
        return f"{self._journal_path}.race-{digest}.jsonl"

    def _race_one_shape(self, args, plan):
        key = plan["key"]
        row = plan["row"]
        candidates = plan["candidates"]
        data = generate_data(*plan["gen_args"])
        tensors = (data["q"], data["k"], data["v"], data["cu_q"], data["cu_k"])

        expected = _chunked_reference(
            *tensors,
            plan["softmax_scale"],
            bool(row.causal),
            int(row.window_left),
            int(row.window_right),
            bool(row.return_lse),
        )

        def launch(candidate):
            config = (
                dict(candidate.backend_config)
                if candidate.backend_config is not None
                else None
            )
            return _run_candidate(
                *tensors,
                candidate.backend,
                candidate.num_splits,
                config,
                *plan["launch_tail"],
            )

        # Correctness and warm-up in one pass, before any timed call, so a
        # compile is never charged to a measurement and a wrong candidate never
        # reaches the race.
        entrants = []
        info_by_label = {}
        rejected = []
        for candidate in candidates:
            info = (key, candidate.backend, candidate.num_splits, candidate.config_json)
            label = f"{candidate.backend}|{candidate.num_splits}|{candidate.config_json}"
            try:
                produced = launch(candidate)
                torch.cuda.synchronize()
                err_ratio = self._error_ratio(produced, expected, row, args)
            except Exception as error:  # noqa: BLE001 - unsupported is an outcome
                rejected.append((info, 1.0, "crash", f"{type(error).__name__}: {error}"))
                continue
            if err_ratio > args.errRatio:
                rejected.append(
                    (info, err_ratio, "failed", f"error ratio {err_ratio:.4f}")
                )
                continue
            protected = (
                candidate.backend,
                canonical_backend_config(candidate.backend_config),
            ) in self._incumbents_by_key.get(key, set())
            entrants.append(
                RaceEntrant(label=label, payload=candidate, protected=protected)
            )
            info_by_label[label] = (info, err_ratio)

        if not entrants:
            print(f"no candidate survived correctness for {key}", flush=True)
            return [
                (info, float("inf"), err, status, detail)
                for info, err, status, detail in rejected
            ]

        print(
            f"racing {len(entrants)} candidates for {key} "
            f"(delta={args.delta:.1%}, at most {args.race_max_blocks} blocks)",
            flush=True,
        )
        journal_path = self._race_block_journal(key)
        journal = (
            JsonlBlockJournal(journal_path, resume=args.resume) if journal_path else None
        )
        outcome = race(
            entrants,
            cuda_event_timer(lambda entrant: launch(entrant.payload)),
            delta=args.delta,
            alpha=args.race_alpha,
            block_calls=args.race_block_calls,
            min_blocks=args.race_min_blocks,
            max_blocks=args.race_max_blocks,
            seed=MHA_FWD_RACE_SAMPLE_SEED,
            journal=journal,
            resume=args.resume,
            verbose=args.verbose,
        )

        return self._race_results(args, key, outcome, info_by_label, rejected)

    @staticmethod
    def _error_ratio(produced, expected, row, args):
        """Same correctness bar the worker path applies, applied in process."""
        actual = _normalize_result(
            produced, bool(row.return_lse), int(row.total_q), int(row.nhead_q)
        )
        pairs = (
            zip(actual, expected)
            if isinstance(expected, tuple)
            else ((actual, expected),)
        )
        worst = 0.0
        for got, want in pairs:
            worst = max(
                worst,
                float(
                    checkAllclose(
                        got,
                        want,
                        rtol=2e-2,
                        atol=2e-2,
                        tol_err_ratio=args.errRatio,
                        printLog=False,
                    )
                ),
            )
        return worst

    def _race_results(self, args, key, outcome, info_by_label, rejected):
        """Turn one race into the result rows the rest of the tuner expects.

        Every raced candidate is reported with its race estimate, so the
        numbers in the profile are all on the same footing.
        """
        results = [
            (info, float("inf"), err, status, detail)
            for info, err, status, detail in rejected
        ]

        winner_label = outcome.winner
        tie_break = outcome.tie_break

        self._race_reports.append(
            {
                "key": list(key),
                "delta": args.delta,
                "alpha": args.race_alpha,
                "block_calls": args.race_block_calls,
                "candidates": len(info_by_label),
                "rejected": len(rejected),
                "blocks_run": outcome.blocks_run,
                "blocks_replayed": outcome.blocks_replayed,
                "calls_spent": outcome.calls_spent,
                "certified": outcome.certified,
                "eliminated": sum(
                    1 for v in outcome.verdicts if v.state == "eliminated"
                ),
                "survivors": [
                    {
                        "label": label,
                        "estimate_us": outcome.samples[label].estimate,
                        "relative_spread": outcome.samples[label].relative_spread,
                    }
                    for label in outcome.survivors
                ],
                "winner": winner_label,
                "tie_break": tie_break,
                "history": [
                    {
                        "block": record.block,
                        "active": record.active,
                        "eliminated": record.eliminated,
                        "wall_seconds": record.wall_seconds,
                    }
                    for record in outcome.history
                ],
            }
        )
        self._race_winner_by_key[key] = winner_label and info_by_label.get(
            winner_label, (None,)
        )[0]

        for verdict in outcome.verdicts:
            info, err_ratio = info_by_label[verdict.label]
            samples = tuple(outcome.samples[verdict.label].block_medians)
            self._samples_by_info[info] = samples
            results.append(
                (
                    info,
                    float(outcome.samples[verdict.label].estimate),
                    float(err_ratio),
                    "ok",
                    verdict.note,
                )
            )
        return results

    def result_to_df(self, results):
        rows = []
        for result in results:
            info, us, err_ratio = result[:3]
            status = (
                result[3]
                if len(result) > 3
                else (
                    "ok"
                    if us > 0
                    and math.isfinite(us)
                    and err_ratio <= self.ARG_DEFAULTS["errRatio"]
                    else "crash"
                )
            )
            key, backend, num_splits, backend_config = info
            row = dict(zip(MHA_FWD_TUNING_KEY_FIELDS, key))
            samples = self._samples_by_info.get(
                info,
                (float(us),)
                if status == "ok" and us > 0 and math.isfinite(us)
                else (),
            )
            row.update(
                {
                    "backend": backend,
                    "num_splits": num_splits,
                    "backend_config": backend_config,
                    "us": us,
                    "errRatio": err_ratio,
                    "status": status,
                    "detail": (
                        ""
                        if status == "ok"
                        else (
                            str(result[4])
                            if len(result) > 4 and result[4]
                            else f"candidate {status} with no diagnostic recorded"
                        )
                    ),
                    "samples_us": json.dumps(samples, separators=(",", ":")),
                    "tflops": self.calculate((info, us, err_ratio)),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        resultdf = self.result_to_df(rets)
        self._last_results = resultdf.copy()
        self._all_results = (
            resultdf.copy()
            if self._all_results.empty
            else pd.concat([self._all_results, resultdf], ignore_index=True)
        ).drop_duplicates(
            subset=[*MHA_FWD_TUNING_KEY_FIELDS, *MHA_FWD_CANDIDATE_FIELDS],
            keep="last",
        )
        if args.profile_file:
            if os.path.exists(args.profile_file):
                old = pd.read_csv(args.profile_file)
                resultdf_for_profile = pd.concat([old, resultdf], ignore_index=True)
            else:
                resultdf_for_profile = resultdf
            dedup = [
                *MHA_FWD_TUNING_KEY_FIELDS,
                *MHA_FWD_CANDIDATE_FIELDS,
            ]
            resultdf_for_profile = resultdf_for_profile.drop_duplicates(
                subset=dedup, keep="last"
            )
            self._atomic_write_csv(resultdf_for_profile, args.profile_file)

        winners = []
        failures = []
        for key, group in resultdf.groupby(list(MHA_FWD_TUNING_KEY_FIELDS), dropna=False):
            valid = group[
                (group["status"] == "ok")
                & (group["us"] > 0)
                & (group["errRatio"] <= args.errRatio)
            ].sort_values("us")
            if valid.empty:
                failed = group.iloc[0].copy()
                failed["status"] = "crash"
                failed["detail"] = "no correctness-gated candidate completed"
                failures.append(failed)
                continue
            winner = self._gate_against_incumbent(key, valid)
            winners.append(winner)

        winnerdf = pd.DataFrame(winners, columns=self.columns)
        failuredf = pd.DataFrame(failures, columns=self.columns)
        if not winnerdf.empty:
            self.success = (
                winnerdf.copy()
                if self.success.empty
                else pd.concat([self.success, winnerdf], ignore_index=True)
            )
        if not failuredf.empty:
            self.failed = (
                failuredf.copy()
                if self.failed.empty
                else pd.concat([self.failed, failuredf], ignore_index=True)
            )
        return winnerdf

    def result_to_csv(self, resultdf, file, concat=False):
        if resultdf is None or resultdf.empty:
            runtime = pd.DataFrame(columns=MHA_FWD_RUNTIME_CSV_FIELDS)
        else:
            runtime = resultdf.loc[:, list(MHA_FWD_RUNTIME_CSV_FIELDS)].copy()
            runtime.loc[:, "backend_config"] = runtime["backend_config"].fillna("")
        if os.path.exists(file):
            old = pd.read_csv(file)
            if old.empty:
                old = pd.DataFrame(columns=MHA_FWD_RUNTIME_CSV_FIELDS)
            else:
                missing = [
                    column
                    for column in MHA_FWD_RUNTIME_CSV_FIELDS
                    if column not in old.columns
                ]
                if missing:
                    raise ValueError(
                        f"{file} is missing MHA runtime columns: {missing}"
                    )
                old = old[list(MHA_FWD_RUNTIME_CSV_FIELDS)]
        else:
            old = pd.DataFrame(columns=MHA_FWD_RUNTIME_CSV_FIELDS)
        combined = (
            runtime.copy()
            if old.empty
            else old.copy()
            if runtime.empty
            else pd.concat([old, runtime], ignore_index=True)
        )
        combined = combined.drop_duplicates(
            subset=list(MHA_FWD_TUNING_KEY_FIELDS), keep="last"
        )
        combined = combined.sort_values(list(MHA_FWD_TUNING_KEY_FIELDS))
        self._atomic_write_csv(combined, file)

        self._selection_proofs.extend(
            self._run_fresh_probe(row, file) for _, row in runtime.iterrows()
        )
        failed_proofs = [
            proof for proof in self._selection_proofs if proof["status"] != "verified"
        ]
        if failed_proofs:
            proof_failures = resultdf.copy()
            proof_failures["status"] = "crash"
            proof_failures["detail"] = "fresh-process selection proof failed"
            self.failed = (
                proof_failures.copy()
                if self.failed.empty
                else pd.concat([self.failed, proof_failures], ignore_index=True)
            )
        covered = (
            self.failed.copy()
            if self.success.empty
            else self.success.copy()
            if self.failed.empty
            else pd.concat([self.success, self.failed], ignore_index=True)
        )
        covered_count = (
            covered.drop_duplicates(subset=list(MHA_FWD_TUNING_KEY_FIELDS)).shape[0]
            if not covered.empty
            else 0
        )
        run_state = (
            "partial"
            if not self.failed.empty or covered_count < len(self.untunedf)
            else "verified"
            if self._selection_proofs
            else "measured"
        )
        self._write_evidence(run_state, file)

    def sortResults(self, tune_file, issorted, values):
        if not os.path.exists(tune_file):
            return
        frame = pd.read_csv(tune_file)
        frame = frame.drop_duplicates(
            subset=list(MHA_FWD_TUNING_KEY_FIELDS), keep="last"
        )
        if issorted:
            frame = frame.sort_values(list(MHA_FWD_TUNING_KEY_FIELDS))
        self._atomic_write_csv(frame[list(MHA_FWD_RUNTIME_CSV_FIELDS)], tune_file)

    @staticmethod
    def _standard_error_us(row) -> float:
        """Standard error of a candidate's finalist-round measurements.

        The finalist rounds are the only repeated observation the run has, so
        their scatter is what it knows about its own reproducibility. The
        standard error rather than the range: the range of a sample grows as
        samples are added, so a range-based threshold would make the run
        harder to satisfy the more evidence it gathered. The standard error
        shrinks as 1/sqrt(n), which makes --finalist-rounds the lever for
        resolving smaller improvements.
        """
        try:
            samples = json.loads(row.get("samples_us") or "[]")
        except (TypeError, ValueError):
            return 0.0
        samples = [float(s) for s in samples if math.isfinite(float(s)) and s > 0]
        if len(samples) < 2:
            return 0.0
        return statistics.stdev(samples) / math.sqrt(len(samples))

    def _gate_against_incumbent(self, key, valid):
        """Return the row to publish: the fastest candidate, unless it cannot
        be told apart from the configuration already in use.

        Publishing a winner that is inside measurement noise of the incumbent
        buys nothing and risks shipping a regression that a contended sweep
        happened to rank first. When the two cannot be separated the incumbent
        is kept, which is the outcome that changes nothing.
        """
        fastest = self._race_pick(key, valid)
        incumbents = self._incumbents_by_key.get(key, set())
        if not incumbents:
            return fastest
        if (fastest["backend"], fastest["backend_config"]) in incumbents:
            fastest["detail"] = "incumbent retained: nothing measured beat it"
            self._record_promotion(key, fastest, fastest, 0.0, 0.0, "incumbent_fastest")
            return fastest

        measured = valid[
            valid.apply(
                lambda r: (r["backend"], r["backend_config"]) in incumbents, axis=1
            )
        ]
        if measured.empty:
            fastest["detail"] = "incumbent not measured; improvement unverified"
            self._record_promotion(key, fastest, None, None, None, "incumbent_absent")
            return fastest

        incumbent = measured.iloc[0]
        gain_us = float(incumbent["us"]) - float(fastest["us"])
        margin = gain_us / float(incumbent["us"])
        noise = self._indifference_threshold(fastest, incumbent)
        if margin <= noise:
            kept = incumbent.copy()
            kept["detail"] = (
                f"incumbent retained: winner was {margin:+.2%} against "
                f"{noise:.2%} measurement spread"
            )
            self._record_promotion(
                key, fastest, incumbent, margin, noise, "within_noise"
            )
            return kept
        fastest["detail"] = f"beat incumbent by {margin:.2%} against {noise:.2%} spread"
        self._record_promotion(key, fastest, incumbent, margin, noise, "promoted")
        return fastest

    def _indifference_threshold(self, challenger, incumbent) -> float:
        """How much better a challenger has to be before it displaces the
        configuration already in use.

        Under the race, delta. The race declares anything inside delta a
        settled question and stops gathering evidence there, so applying a
        standard-error test on top would let the gate promote a challenger the
        measurement itself called a tie -- and because the standard error
        shrinks as blocks accumulate, it would do so more eagerly the longer
        the race ran. One definition of indistinguishable, used end to end.

        Otherwise the screening path's two-sample separation at roughly 95%:
        the gain has to clear twice the combined standard error of the two
        candidates' round means.
        """
        args = getattr(self, "_args", None)
        if getattr(args, "strategy", "exhaustive") == "race":
            return float(getattr(args, "delta", MHA_FWD_INDIFFERENCE_DELTA))
        combined_se = math.hypot(
            self._standard_error_us(challenger), self._standard_error_us(incumbent)
        )
        return (
            (MHA_FWD_SIGNIFICANCE_SIGMA * combined_se) / float(incumbent["us"])
            if float(incumbent["us"]) > 0
            else 0.0
        )

    def _race_pick(self, key, valid):
        """The candidate the race selected, rather than whichever sorted first.

        Sorting by latency would throw away the tie-break: inside the
        indifference zone the fastest point estimate is the noisiest thing to
        choose on, which is the whole reason the race picks by steadiness and
        prefers the incumbent.
        """
        picked = getattr(self, "_race_winner_by_key", {}).get(key)
        if picked is not None:
            match = valid[
                (valid["backend"] == picked[1])
                & (valid["num_splits"] == picked[2])
                & (valid["backend_config"] == picked[3])
            ]
            if not match.empty:
                return match.iloc[0].copy()
        return valid.iloc[0].copy()

    def _record_promotion(self, key, challenger, incumbent, margin, noise, decision):
        """Keep why each shape was or was not retuned, for the evidence file.

        Without this a reader of the published table cannot tell a measured
        improvement from a tie that happened to sort first, which is the
        distinction the incumbent comparison exists to make.
        """
        self._promotions.append(
            {
                "key": list(key) if isinstance(key, tuple) else key,
                "decision": decision,
                "challenger": {
                    "backend": challenger["backend"],
                    "backend_config": challenger["backend_config"],
                    "us": float(challenger["us"]),
                },
                "incumbent": (
                    None
                    if incumbent is None
                    else {
                        "backend": incumbent["backend"],
                        "backend_config": incumbent["backend_config"],
                        "us": float(incumbent["us"]),
                    }
                ),
                "margin": None if margin is None else round(float(margin), 6),
                "measurement_spread": None if noise is None else round(float(noise), 6),
            }
        )

    def _incumbent_candidates(self, row) -> list[MhaFwdCandidate]:
        """The tile dicts the dict-config kernels resolve for this shape today.

        A tuning run that never measures the configuration already in use
        cannot tell an improvement from a regression. Any gap in the candidate
        catalogue, and any single contended measurement, then publishes a
        result that is worse than shipping nothing. Measuring the incumbent in
        the same sweep, on the same GPU, under the same conditions, turns that
        guarantee from a policy into a comparison.

        This asks each kernel's own resolver what it would launch, so the
        incumbent is whatever the shipped default actually is rather than a
        value restated here.
        """
        import torch

        dtype = getattr(torch, str(row.dtype), torch.bfloat16)

        incumbents = []
        for backend in sorted(MHA_FWD_TILE_CONFIG_BACKENDS):
            try:
                if backend == "gluon":
                    from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
                        _get_config as resolve,
                    )

                    config = resolve(is_fp8=False, has_pe=False)
                else:
                    from aiter.ops.triton._triton_kernels.attention.mha import (
                        _get_config as resolve,
                    )

                    config = resolve(
                        float(row.dropout_p) > 0,
                        dtype,
                        has_pe=False,
                        head_dim_v=int(row.hdim_v),
                    )
            except Exception:
                # A backend with no resolvable default has no incumbent to
                # beat, which is a weaker claim than one we can measure but
                # not a reason to abandon the sweep.
                continue
            if not isinstance(config, dict):
                continue
            incumbents.append(
                MhaFwdCandidate(
                    backend=backend, num_splits=0, backend_config=dict(config)
                )
            )
        return incumbents

    def _restricted_backends(self) -> list[str] | None:
        """Backends this run is allowed to measure, or None for all of them."""
        value = (getattr(self._args, "backends", "") or "").strip()
        return [name.strip() for name in value.split(",") if name.strip()] or None

    def _run_fresh_probe(self, row, config_file: str) -> dict[str, Any]:
        descriptor, proof_path = tempfile.mkstemp(prefix="mha-selection-", suffix=".jsonl")
        os.close(descriptor)
        os.unlink(proof_path)
        problem = {
            field: row[field]
            for field in MHA_FWD_TUNING_KEY_FIELDS
        }
        problem["_proof_warmup"] = int(self._args.warmup)
        problem["_proof_iters"] = int(self._args.iters)
        # Run the child as a module from the repository root, not as a file
        # path. Executing a path puts op_tests/tuners/ on sys.path instead of
        # the root, and "import aiter" then resolves to whatever copy is
        # installed in site-packages -- so the probe either dies on aiter.jit
        # or, worse, silently proves a claim about a different checkout than
        # the one being tuned.
        repository_root = Path(__file__).parents[2]
        environment = os.environ.copy()
        environment.update(
            {
                "AITER_CONFIG_MHA_FWD": os.path.abspath(config_file),
                "AITER_GPU_MODEL": str(row["gpu_model"]),
                "AITER_MHA_FWD_SELECTION_PROOF_FILE": proof_path,
                "AITER_MHA_FWD_PROBE_PROBLEM": json.dumps(problem),
                "PYTHONPATH": os.pathsep.join(
                    [str(repository_root), os.environ.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            }
        )
        started = time.time()
        expected = (
            {
                "backend": str(row["backend"]),
                "num_splits": int(row["num_splits"]),
                "backend_config": (
                    str(row["backend_config"])
                    if "backend_config" in row and pd.notna(row["backend_config"])
                    else ""
                ),
            }
            if "backend" in row and pd.notna(row["backend"])
            else None
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "op_tests.tuners.tune_mha_fwd",
                    "--_selection_probe",
                ],
                cwd=str(repository_root),
                env=environment,
                capture_output=True,
                text=True,
                timeout=int(self._args.timeout),
                check=False,
            )
            records = []
            if os.path.exists(proof_path):
                with open(proof_path, encoding="utf-8") as file:
                    records = [json.loads(line) for line in file if line.strip()]
            selected = records[-1] if records else {}
            observed = {
                "backend": selected.get("backend"),
                "num_splits": selected.get("num_splits"),
                "backend_config": selected.get("backend_config", ""),
            }
            probe_line = next(
                (
                    line.removeprefix("MHA_PROBE_RESULT=")
                    for line in completed.stdout.splitlines()
                    if line.startswith("MHA_PROBE_RESULT=")
                ),
                "",
            )
            probe_result = json.loads(probe_line) if probe_line else {}
            verified = (
                completed.returncode == 0
                and (expected is None or observed == expected)
                and probe_result.get("correctness") == "ok"
            )
            return {
                "status": "verified" if verified else "failed",
                "expected": expected,
                "observed": observed,
                "latency_us": probe_result.get("latency_us"),
                "correctness": probe_result.get("correctness", "failed"),
                "returncode": completed.returncode,
                "elapsed_s": time.time() - started,
                "stderr": completed.stderr[-4000:],
            }
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
            return {
                "status": "failed",
                "expected": expected,
                "error": str(exc),
                "elapsed_s": time.time() - started,
            }
        finally:
            if os.path.exists(proof_path):
                os.unlink(proof_path)

    def _write_evidence(self, run_state: str, runtime_file: str) -> None:
        status_counts = Counter(self._all_results.get("status", pd.Series(dtype=str)))
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[2],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        runtime_path = Path(runtime_file)
        runtime_bytes = runtime_path.read_bytes() if runtime_path.is_file() else b""
        payload = {
            "schema_version": 1,
            "family": "mha_fwd",
            "run_state": run_state,
            "strategy": self._args.strategy,
            "hardware": [
                {
                    field: row[field]
                    for field in MHA_FWD_HARDWARE_KEY_FIELDS
                }
                for _, row in self.untunedf[
                    list(MHA_FWD_HARDWARE_KEY_FIELDS)
                ].drop_duplicates().iterrows()
            ],
            "software": {
                "aiter_revision": revision,
                "rocm": torch.version.hip,
                "torch": torch.__version__,
            },
            "measurement": {
                "warmup": int(self._args.warmup),
                "iterations": int(self._args.iters),
                "finalist_rounds": int(self._args.finalist_rounds),
                "statistic": "median of finalist round means",
                "rtol": 2e-2,
                "atol": 2e-2,
                "error_metric": "fraction failing elementwise allclose",
                "maximum_error_ratio": float(self._args.errRatio),
                "candidate_count": len(self._all_results),
                "status_counts": dict(sorted(status_counts.items())),
            },
            "artifacts": {
                "workload_catalogue": os.path.abspath(self._args.untune_file),
                "candidate_journal": os.path.abspath(self._journal_path),
                "measurement_record": (
                    os.path.abspath(self._args.profile_file)
                    if self._args.profile_file
                    else None
                ),
                "runtime_config": os.path.abspath(runtime_file),
                "runtime_config_sha256": (
                    sha256(runtime_bytes).hexdigest() if runtime_bytes else None
                ),
            },
            "selection_proofs": self._selection_proofs,
            "search_strategy": getattr(self._args, "strategy", "exhaustive"),
            "candidate_sample": getattr(self._args, "candidate_sample", None),
            "restricted_backends": self._restricted_backends(),
            "promotions": self._promotions,
            # What the gate actually compared against, rather than what a
            # reader would have to infer from the strategy name.
            "indifference_threshold": (
                float(getattr(self._args, "delta", MHA_FWD_INDIFFERENCE_DELTA))
                if getattr(self._args, "strategy", "") == "race"
                else f"{MHA_FWD_SIGNIFICANCE_SIGMA} x combined standard error"
            ),
            "races": self._race_reports,
            "coverage_limits": [
                "CK tile recipes remain the default CK launch; this tuner does not dump PR #5024 JSON",
            ],
            "command": [sys.executable, *sys.argv],
            "started_at_unix_s": self._run_started_at,
            "finished_at_unix_s": time.time(),
        }
        self._atomic_write_json(payload, self._evidence_path)

    def tune_summary(self, status):
        if status != "Finished":
            self._write_evidence(
                "partial" if not self._all_results.empty else "failed",
                self._args.tune_file,
            )
        return super().tune_summary(status)

    def _clear_op_caches(self):
        _load_mha_fwd_tuning_table.cache_clear()

    def run_config(self, args):
        config_file = os.environ.get("AITER_CONFIG_MHA_FWD", args.tune_file)
        results = []
        for _, row in self.untunedf.iterrows():
            proof = self._run_fresh_probe(row, config_file)
            results.append(
                {
                    "shape": str(
                        tuple(row[field] for field in MHA_FWD_PROBLEM_KEY_FIELDS)
                    ),
                    "e2e_us": proof.get("latency_us", float("inf")),
                    "status": "ok" if proof["status"] == "verified" else "error",
                }
            )
        return results


def _selection_probe() -> int:
    """Fresh-process public-operator correctness, timing, and dispatch probe."""

    raw_problem = os.environ.get("AITER_MHA_FWD_PROBE_PROBLEM", "")
    if not raw_problem:
        raise RuntimeError("AITER_MHA_FWD_PROBE_PROBLEM is required")
    row = json.loads(raw_problem)
    problem = MhaFwdProblem.from_mapping(row)
    data = generate_data(
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
        zlib.crc32(",".join(problem.key()).encode("utf-8")),
    )
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
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    warmup = max(0, int(row.get("_proof_warmup", 5)))
    iterations = max(1, int(row.get("_proof_iters", 101)))
    for _ in range(warmup):
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
    print(
        "MHA_PROBE_RESULT="
        + json.dumps(
            {
                "correctness": "ok",
                "latency_us": latency_us,
                "iterations": iterations,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--_selection_probe"]:
        raise SystemExit(_selection_probe())
    tuner = MhaFwdTuner()
    tuner.run(tuner.parse_args(), False)
