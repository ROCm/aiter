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
    MHA_FWD_METRIC_FIELDS,
    MHA_FWD_PROBLEM_KEY_FIELDS,
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MHA_FWD_TUNING_KEY_FIELDS,
    MhaFwdCandidate,
    MhaFwdProblem,
    enumerate_mha_fwd_candidates,
    mha_fwd_candidate_id,
)
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.mp_tuner import mp_tuner


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
            choices=("exhaustive",),
            default="exhaustive",
            help="candidate search strategy (only exhaustive is currently supported)",
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

    def tune(self, untunedf, tunedf, args):
        journal = self._load_journal()
        all_infos = []
        task_by_info = {}
        for row_index, row in untunedf.iterrows():
            problem = MhaFwdProblem.from_mapping(
                {field: row[field] for field in MHA_FWD_TUNING_KEY_FIELDS}
            )
            key = problem.key()
            softmax_scale = int(row.hdim_q) ** -0.5
            candidates = enumerate_mha_fwd_candidates(str(row.gfx))
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
            winner = valid.iloc[0].copy()
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
