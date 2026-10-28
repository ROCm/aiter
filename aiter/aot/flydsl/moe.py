#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL kernels of tuned fused-MoE configs.

Each tuned row is replayed through the runtime's own implementation --
``_fused_moe_impl``, or ``_fhmoe_impl`` for heterogeneous shared-expert rows --
on FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run`. On gfx1250,
``_fused_moe_impl`` serves the grouped table's rows through FlyDSL's grouped
GEMM. The runtime
looks the row up, selects and parameterizes its kernels, sizes intermediate
buffers and runs its auxiliary kernels exactly as it does at inference, and
every FlyDSL launch on that path is compiled into the cache instead. This
module only builds what a caller passes in: activations, routing, and weights
quantized and preshuffled the way ``op_tests/test_moe_2stage.py`` prepares
them, plus the caller-side choices a row does not pin down (bias, gate/up
layout, SiTUv2 betas, ``num_local_tokens``, FHMoE's ``clamp_shared``).

A row tuned at token count T also serves smaller M that ``get_padded_M`` maps
to T, and some compile-time parameters follow the actual M, so each row is
replayed at the smallest such M as well as at T. Where those two request
different kernels, the M between them are bisected down to every M at which
the runtime switches kernels: a row can switch more than once in its bucket
(the gfx1250 stage-1 quant's K split steps through several values).

A caller behind an expert-parallel dispatch also passes an expert mask, which
brings in launches no tuned row reaches: the gfx950 stage-2 reduction, compiled
for the mask's length, and the gfx1250 route. The mask's length follows from
the deployment (global experts, EPLB replicas, fused shared experts), which no
tuned row records, so these callers come from the ``fused_moe`` deployments of
``moe_deployments.json`` (see :mod:`aiter.aot.flydsl.deployments`): what one
rank passes ``fused_moe`` and the largest M it passes. The file also lists
tensor-parallel deployments, whose rank passes no mask but the padded, split
intermediate dim and the shared-expert handling its framework uses (FHMoE
among them). Each deployment is replayed at every bucket's smallest and largest
M up to that M, bisected as above, and reads the tuned tables serving reads; it
needs no tuned row.

A row is compiled for the arch its ``gfx`` column names. With ``GPU_ARCHS``
naming targets, as the wheel build sets it, only rows of those archs are
replayed; unset or ``native``, every arch the tables have. aiter fixes some
arch-dependent values at import (e.g. ``dtypes.fp8``), so the replay runs in
one process per arch, started with ``GPU_ARCHS`` naming that arch and
``AITER_AOT_REPLAY=1``, under which aiter's runtime arch queries answer it
without probing a device. Those processes need no GPU; they run side by side
and split the worker budget by job count. Within one, a worker replays several
jobs of one row in turn, so the kernels they share compile once.

Usage:
    # Compile every FlyDSL kernel the default tuned CSVs reach
    python -m aiter.aot.flydsl.moe

    # Only the gfx1250 rows, which a gfx942;gfx950 wheel build leaves out
    GPU_ARCHS=gfx1250 python -m aiter.aot.flydsl.moe

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv

    # Custom deployment file(s) only
    python -m aiter.aot.flydsl.moe --deployments /path/to/deployments.json

    # Verify a cache instead: fail on any kernel the replay would compile
    python -m aiter.aot.flydsl.moe --check

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    GPU_ARCHS                 Archs to replay (default: every arch the tables have)
"""

import argparse
import csv
import functools
import os
import subprocess
import sys
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from unittest import mock

from aiter.aot.flydsl.common import (
    aot_target_archs,
    check_replay_process,
    collect_aot_jobs,
    cu_num_to_arch,
    dedupe_jobs,
    get_max_workers,
    replay_env,
    run_jobs_parallel,
)
from aiter.aot.flydsl.deployments import DEFAULT_DEPLOYMENTS, FusedMoECall
from aiter.aot.flydsl.deployments import load as load_deployments
from aiter.jit.core import AITER_CONFIGS
from aiter_worker_limits import adopt_legacy_max_jobs

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
    AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
    AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE,
]
MOE_AOT_ARCH_DEFAULT = "gfx950"

_ROW_FIELDS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "shared_expert_id",
    "ep_fused",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
    "kernelName1",
    "kernelName2",
    "expert_mask_size",
)
# gate_mode is overwritten by each job's variant.
_ROW_KEY = ("csv_path", *(f for f in _ROW_FIELDS if f != "gate_mode"))
# (q_dtype_a, q_dtype_w) of the deployments' quant recipes; gfx950 and gfx1250,
# the archs fused_moe deployments target, take OCP FP8.
_DEPLOYMENT_DTYPES = {
    "a8w4": ("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2"),
    "a4w4": ("torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2"),
}
# get_padded_M: M below the first tier takes the next power of two, larger M
# the highest tier not above it.
_PADDED_M_TIERS = (32768, 131072)
# Jobs of one row share most of their kernels, so one worker replays them in
# turn; capped so no single worker trails the rest of the build.
_JOBS_PER_WORKER = 12
_FP8 = ("float8_e4m3fn", "float8_e4m3fnuz")
_FP4 = "float4_e2m1fn_x2"


def _int(value, default=0) -> int:
    value = str(value or "").strip()
    return int(float(value)) if value else default


def _flag(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes")


def _name(value) -> str:
    """``QuantType.per_1x32`` -> ``per_1x32``, ``torch.bfloat16`` -> ``bfloat16``."""
    return str(value or "").strip().split(".")[-1]


def job_arch(job: dict) -> str:
    return job["gfx"] or cu_num_to_arch(_int(job["cu_num"]), MOE_AOT_ARCH_DEFAULT)


def _is_grouped(row: dict) -> bool:
    """Served by FlyDSL's grouped GEMM: a row of the gfx1250 grouped table, or
    a gfx1250 deployment."""
    if row.get("deployment"):
        return job_arch(row) == "gfx1250"
    return (row.get("kernelName1") or "").strip().startswith("grouped_gemm")


def _reaches_flydsl(row: dict) -> bool:
    kernel1 = (row.get("kernelName1") or "").strip()
    kernel2 = (row.get("kernelName2") or "").strip()
    # A cktile stage1 finishes with a FlyDSL activation epilogue.
    return (
        kernel1.startswith(("flydsl_", "cktile_"))
        or kernel2.startswith("flydsl_")
        or _is_grouped(row)
    )


def _token_points(token: int, edges: bool) -> list[int]:
    if not edges or token < 2 or token & (token - 1) or token > 32768:
        return [token]
    return [token // 2 + 1, token]


def _caller_gate_mode(row: dict):
    """Gate/up weight layout serving uses for a quant family (gu-interleaved
    for fp8 activations, as with ATOM_MOE_GU_ITLV=1). gfx1250 takes only the
    gu-interleaved layout."""
    from aiter.ops.flydsl.moe_common import GateMode

    if job_arch(row) == "gfx1250":
        return GateMode.INTERLEAVE
    q_dtype_a, q_dtype_w = _name(row["q_dtype_a"]), _name(row["q_dtype_w"])
    mxfp8 = q_dtype_w in _FP8 and _name(row["q_type"]) == "per_1x32"
    if q_dtype_a in _FP8 and (q_dtype_w == _FP4 or mxfp8):
        return GateMode.INTERLEAVE
    return GateMode.SEPARATED


def _variants(row: dict) -> list[dict]:
    """Caller-side choices a tuned row leaves open; the first is the one
    serving uses by default, which the runtime must serve (see
    ``_judge_default_caller``)."""
    from aiter.ops.flydsl.moe_common import (
        DEFAULT_SITUV2_BETA,
        DEFAULT_SITUV2_LINEAR_BETA,
        GateMode,
    )
    from aiter.ops.flydsl.mxfp4_kname import _is_mxfp4_kname

    clamps = [True]
    if _int(row["shared_expert_id"], -1) >= 0:
        gate_modes = [GateMode[_name(row["gate_mode"]) or "SEPARATED"]]
        biases = ["none"]
        # Only MXFP8 routed weights let a caller leave the shared expert unclamped.
        if _name(row["q_dtype_w"]) in _FP8:
            clamps.append(False)
    else:
        gate_modes = [_caller_gate_mode(row)]
        if _is_mxfp4_kname(row["kernelName1"]):
            layouts = (GateMode.SEPARATED, GateMode.INTERLEAVE)
            gate_modes += [mode for mode in layouts if mode not in gate_modes]
        biases = ["none"]
        # Mirror fused_moe.enable_bias: only Swiglu (GPT-OSS) takes expert
        # bias; Silu / SiTUv2 callers never pass one, so do not AOT those.
        if (
            _name(row["act_type"]) == "Swiglu"
            and _name(row["q_type"]) == "per_1x32"
            and _name(row["dtype"]) in ("bfloat16", "float16")
            and _name(row["q_dtype_w"]) == _FP4
        ):
            # A stage-2 bias rules out some stage-2 backends (Opus), and the
            # runtime then re-selects the whole config.
            biases += ["both", "stage1"]
    betas = [(None, None)]
    if _name(row["act_type"]) == "Situv2":
        # Unset betas launch with the model defaults; plain SiTU (1.0) must be
        # passed explicitly.
        betas += [(1.0, 1.0), (DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA)]
    # Callers behind an EP dispatch pass the received token count.
    return [
        {
            "gate_mode": gate_mode.value,
            "bias": bias,
            "beta": beta,
            "linear_beta": linear_beta,
            "local_tokens": local_tokens,
            "clamp_shared": clamp_shared,
        }
        for local_tokens in (False, True)
        for gate_mode in gate_modes
        for bias in biases
        for beta, linear_beta in betas
        for clamp_shared in clamps
    ]


def parse_csv(csv_path: str, edges: bool = True) -> list[dict]:
    """One job per (row, replayed token count, caller-side variant)."""
    csv_path = os.path.abspath(csv_path)
    jobs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            # The runtime drops these rows when it loads the table.
            if (row.get("_tag") or "").strip() == "flydsl_fallback":
                continue
            if not _reaches_flydsl(row):
                continue
            # The runtime picks these grouped rows only under a caller's expert
            # mask (topk -1) or MegaMoE's stage-2 scatter (ep_fused); the table
            # records neither the global expert count nor the real topk.
            if _int(row.get("topk")) < 0 or _flag(row.get("ep_fused")):
                continue
            shape = {field: (row.get(field) or "").strip() for field in _ROW_FIELDS}
            label = f"{shape['kernelName1'] or '-'} | {shape['kernelName2'] or '-'}"
            for index, variant in enumerate(_variants(shape)):
                for m in _token_points(_int(shape["token"]), edges):
                    jobs.append(
                        {
                            **shape,
                            **variant,
                            "csv_path": csv_path,
                            "m": m,
                            "primary": index == 0,
                            "kernel_name": label,
                        }
                    )
    return dedupe_jobs(jobs)


def _buckets(max_token: int) -> list[tuple[int, int, int]]:
    """``(bucket, smallest M, largest M)`` of each tuning bucket that
    ``get_padded_M`` maps some M in 1..max_token to."""
    first, *rest = _PADDED_M_TIERS
    result, low = [], 1
    while low <= max_token:
        if low < first:
            bucket = 1 << (low - 1).bit_length()
        else:
            bucket = max(tier for tier in _PADDED_M_TIERS if tier <= low)
        above = [tier for tier in rest if tier > bucket]
        high = bucket if bucket < first else (above[0] - 1 if above else max_token)
        result.append((bucket, low, min(high, max_token)))
        low = high + 1
    return result


def deployment_row(call: FusedMoECall) -> dict:
    """What one rank of a deployment passes ``fused_moe``, in the columns of a
    tuned row."""
    q_dtype_a, q_dtype_w = _DEPLOYMENT_DTYPES[call.quant]
    row = dict.fromkeys(_ROW_FIELDS, "")
    row.update(
        deployment=call.id,
        gfx=call.gfx,
        cu_num=str(call.cu_num),
        model_dim=str(call.hidden_dim),
        inter_dim=str(call.inter_dim),
        expert=str(call.local_experts),
        topk=str(call.topk_cols),
        expert_mask_size=str(call.mask_size or ""),
        shared_expert_id=(
            str(call.shared_expert_id) if call.shared_expert_id >= 0 else ""
        ),
        hidden_pad=str(call.hidden_pad or ""),
        intermediate_pad=str(call.intermediate_pad or ""),
        act_type=f"ActivationType.{call.model.activation}",
        dtype="torch.bfloat16",
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type="QuantType.per_1x32",
        use_g1u1="1",
        doweight_stage1="0",
    )
    if call.shared_expert_id >= 0:
        # FHMoE reads the gate/up layout from the row, not from the caller.
        row["gate_mode"] = f"GateMode.{_caller_gate_mode(row).name}"
    return row


def parse_deployments(paths: Sequence[str], edges: bool = True) -> list[dict]:
    """One job per (``fused_moe`` deployment, tuning bucket, replayed M,
    caller-side variant) of the deployment files ``paths``."""
    jobs = []
    for call in load_deployments(paths).fused_moe:
        shape = deployment_row(call)
        variants = _variants(shape)
        if not call.is_ep:
            # num_local_tokens is the count an EP dispatch received.
            variants = [v for v in variants if not v["local_tokens"]]
        for bucket, low, high in _buckets(call.max_tokens):
            row = {**shape, "token": str(bucket)}
            points = [low, high] if edges and low < high else [high]
            for index, variant in enumerate(variants):
                for m in points:
                    jobs.append(
                        {
                            **row,
                            **variant,
                            "csv_path": call.path,
                            "m": m,
                            "primary": index == 0,
                            "kernel_name": call.id,
                        }
                    )
    return dedupe_jobs(jobs)


def _collect_jobs(csv_paths, deployments, edges: bool) -> list[dict]:
    jobs = collect_aot_jobs(csv_paths, functools.partial(parse_csv, edges=edges))
    if deployments:
        jobs += parse_deployments(deployments, edges)
    return dedupe_jobs(jobs)


def _dtype(name: str):
    import torch

    return getattr(torch, _name(name))


def _caller_inputs(job: dict) -> dict:
    """What a caller hands fused_moe for this row: bf16 activations, routing,
    and weights/scales quantized and preshuffled per quant family."""
    import torch

    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import (
        moe_shuffle_scale,
        moe_shuffle_weight,
        pack_int8_to_packed_int4,
        shuffle_scale_a16w4,
        shuffle_scale_for_int4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda")

    m, experts, topk = job["m"], _int(job["expert"]), _int(job["topk"])
    model_dim, inter_dim = _int(job["model_dim"]), _int(job["inter_dim"])
    w1_rows = 2 * inter_dim if _flag(job["use_g1u1"]) else inter_dim
    q_type = getattr(QuantType, _name(job["q_type"]))
    q_dtype_a = _dtype(job["q_dtype_a"])
    q_dtype_w = _dtype(job["q_dtype_w"])

    if (
        job_arch(job) == "gfx1250"
        and q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
    ):
        # gu-interleaved (GUGU) weights with n32k4 scales, from raw GGUU rows.
        w1 = moe_shuffle_weight(
            empty(experts, w1_rows, model_dim // 2, dtype=dtypes.u8),
            experts,
            is_guinterleave=True,
            gate_up=True,
        ).view(q_dtype_w)
        w2 = moe_shuffle_weight(
            empty(experts, model_dim, inter_dim // 2, dtype=dtypes.u8), experts
        ).view(q_dtype_w)
        w1_scale = moe_shuffle_scale(
            empty(experts, w1_rows, model_dim // 32, dtype=dtypes.u8),
            experts,
            is_guinterleave=True,
            gate_up=True,
        )
        w2_scale = moe_shuffle_scale(
            empty(experts, model_dim, inter_dim // 32, dtype=dtypes.u8), experts
        )
    elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2:
        w1 = empty(experts, w1_rows, model_dim // 2, dtype=q_dtype_w)
        w2 = empty(experts, model_dim, inter_dim // 2, dtype=q_dtype_w)
        w1_scale = empty(experts * w1_rows, model_dim // 32, dtype=dtypes.fp8_e8m0)
        w2_scale = empty(experts * model_dim, inter_dim // 32, dtype=dtypes.fp8_e8m0)
        if q_dtype_a in (dtypes.bf16, dtypes.fp16, dtypes.fp8):
            gate_up = q_dtype_a == dtypes.fp8
            w1 = shuffle_weight_a16w4(w1, 16, gate_up)
            w1_scale = shuffle_scale_a16w4(w1_scale, experts, gate_up)
            w2 = shuffle_weight_a16w4(w2, 16, False)
            w2_scale = shuffle_scale_a16w4(w2_scale, experts, False)
        else:
            w1 = shuffle_weight(w1, layout=(16, 16))
            w2 = shuffle_weight(w2, layout=(16, 16))
            w1_scale = e8m0_shuffle(w1_scale)
            w2_scale = e8m0_shuffle(w2_scale)
    elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp8:
        w1 = empty(experts, w1_rows, model_dim, dtype=q_dtype_w)
        w2 = empty(experts, model_dim, inter_dim, dtype=q_dtype_w)
        w1_scale = empty(experts * w1_rows, model_dim // 32, dtype=dtypes.fp8_e8m0)
        w2_scale = empty(experts * model_dim, inter_dim // 32, dtype=dtypes.fp8_e8m0)
        w1 = shuffle_weight_a16w4(w1, 16, True)
        w1_scale = shuffle_scale_a16w4(w1_scale, experts, True)
        w2 = shuffle_weight_a16w4(w2, 16, False)
        w2_scale = e8m0_shuffle(w2_scale)
    elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:

        def packed_int4(rows, cols):
            shuffled = shuffle_weight(empty(experts, rows, cols, dtype=dtypes.i8))
            packed = pack_int8_to_packed_int4(shuffled)
            return packed.view(experts, rows, cols // 2).view(dtypes.i4x2)

        def group_scale(cols, rows):
            scale = empty(experts, cols // 32, rows, dtype=dtypes.bf16)
            return shuffle_scale_for_int4(scale, group_size=32).view(-1).contiguous()

        w1 = packed_int4(w1_rows, model_dim)
        w1_scale = group_scale(model_dim, w1_rows)
        w2 = packed_int4(model_dim, inter_dim)
        w2_scale = group_scale(inter_dim, model_dim)
    elif q_type in (QuantType.per_Token, QuantType.per_Tensor):
        w1 = shuffle_weight(empty(experts, w1_rows, model_dim, dtype=q_dtype_w))
        w2 = shuffle_weight(empty(experts, model_dim, inter_dim, dtype=q_dtype_w))
        if q_type == QuantType.per_Token:
            w1_scale = empty(experts, w1_rows, 1, dtype=dtypes.fp32)
            w2_scale = empty(experts, model_dim, 1, dtype=dtypes.fp32)
        else:
            w1_scale = empty(experts, 1, dtype=dtypes.fp32)
            w2_scale = empty(experts, 1, dtype=dtypes.fp32)
    else:
        raise NotImplementedError(
            f"no AOT input recipe for {q_type!s} with {q_dtype_a} x {q_dtype_w}"
        )

    inputs = {
        "hidden_states": empty(m, model_dim, dtype=_dtype(job["dtype"])),
        "w1": w1,
        "w2": w2,
        "topk_weight": empty(m, topk, dtype=dtypes.fp32),
        "topk_ids": empty(m, topk, dtype=dtypes.i32),
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }
    if job["local_tokens"]:
        inputs["num_local_tokens"] = empty(1, dtype=dtypes.i32)
    if job["expert_mask_size"]:
        inputs["expert_mask"] = empty(_int(job["expert_mask_size"]), dtype=dtypes.i32)
    if job["bias"] in ("both", "stage1"):
        inputs["bias1"] = empty(experts, w1_rows, dtype=dtypes.fp32)
    if job["bias"] == "both":
        inputs["bias2"] = empty(experts, model_dim, dtype=dtypes.fp32)
    if _int(job["shared_expert_id"], -1) >= 0:
        inputs["shared_w1"] = empty(1, w1_rows, model_dim, dtype=dtypes.fp8)
        inputs["shared_w2"] = empty(1, model_dim, inter_dim, dtype=dtypes.fp8)
        inputs["shared_w1_scale"] = e8m0_shuffle(
            empty(w1_rows, model_dim // 32, dtype=dtypes.fp8_e8m0)
        )
        inputs["shared_w2_scale"] = e8m0_shuffle(
            empty(model_dim, inter_dim // 32, dtype=dtypes.fp8_e8m0)
        )
    return inputs


def _replay(job: dict) -> None:
    from aiter import ActivationType, QuantType

    common = {
        **_caller_inputs(job),
        "activation": getattr(ActivationType, _name(job["act_type"])).value,
        "quant_type": getattr(QuantType, _name(job["q_type"])).value,
        "doweight_stage1": _flag(job["doweight_stage1"]),
        "hidden_pad": _int(job["hidden_pad"]),
        "intermediate_pad": _int(job["intermediate_pad"]),
        "gate_mode": job["gate_mode"],
    }
    shared_expert_id = _int(job["shared_expert_id"], -1)
    if shared_expert_id >= 0:
        from aiter.fhmoe import _fhmoe_impl

        _fhmoe_impl(
            **common,
            shared_expert_id=shared_expert_id,
            clamp_shared=job["clamp_shared"],
        )
        return

    from aiter.fused_moe import _fused_moe_impl

    # Pin the row's activation dtype: the runtime derives it from M and from
    # opt-in env modes, and every tuned row must stay reachable.
    _fused_moe_impl(
        **common,
        beta=job["beta"],
        linear_beta=job["linear_beta"],
        quant_dtype_a=_dtype(job["q_dtype_a"]),
    )


@contextmanager
def _without_device_queries():
    """Host queries whose C++ side reads the live device; their answers only
    size buffers of HIP kernels the replay never launches."""
    import aiter

    with mock.patch.object(
        aiter, "moe_sorting_opus_get_workspace_size", lambda *_a, **_k: 0
    ):
        yield


@contextmanager
def _runtime_reads(job: dict):
    """Point the runtime's tuned-config lookup at the row's own table. A
    deployment reads the tables serving reads."""
    if job.get("deployment"):
        env = {}
    elif _is_grouped(job):
        env = {"AITER_CONFIG_GROUPED_FMOE": job["csv_path"]}
    elif _int(job["shared_expert_id"], -1) >= 0:
        env = {"AITER_CONFIG_FHMOE": job["csv_path"]}
    else:
        env = {"AITER_CONFIG_FMOE": job["csv_path"]}
    get_config_file = type(AITER_CONFIGS).get_config_file
    with mock.patch.dict(os.environ, env):
        get_config_file.cache_clear()
        try:
            yield
        finally:
            get_config_file.cache_clear()


def _describe(job: dict) -> str:
    variant = [f"gate={job['gate_mode']}"]
    if job["expert_mask_size"]:
        variant.insert(0, f"expert_mask={job['expert_mask_size']}")
    if job["bias"] != "none":
        variant.append(f"bias={job['bias']}")
    if job["beta"] is not None:
        variant.append(f"beta={job['beta']:g}/{job['linear_beta']:g}")
    if job["local_tokens"]:
        variant.append("num_local_tokens")
    if not job["clamp_shared"]:
        variant.append("clamp_shared=False")
    return (
        f"{job['kernel_name']}  token={job['token']} m={job['m']} "
        f"dims=({job['model_dim']},{job['inter_dim']}) E={job['expert']} "
        f"topk={job['topk']} {' '.join(variant)}"
    )


def compile_one_config(**job) -> dict:
    """Replay one job and leave its kernels in the cache."""
    from aiter.aot.flydsl.dry_run import dry_run

    arch = job_arch(job)
    description = _describe(job)
    result = {
        "kernel_name": job["kernel_name"],
        "shape": description,
        "compile_time": None,
        "compile_arch": arch,
    }
    started = time.time()
    try:
        with (
            _runtime_reads(job),
            dry_run(arch, _int(job["cu_num"])),
            _without_device_queries(),
        ):
            _replay(job)
        result["compile_time"] = time.time() - started
    except NotImplementedError as error:
        # The runtime refuses the call; _replay_batch judges the refusal.
        result["compile_time"] = 0.0
        result["unsupported"] = str(error)
    except Exception as error:  # noqa: BLE001
        print(
            f"  [FAIL] {description}  arch={arch}: {type(error).__name__}: {error}",
            flush=True,
        )
    return result


def _load_host_helpers(arch: str, cu_num: int) -> None:
    """Load (building if needed) the C++ module of the host-side query the
    replay calls for real, once here instead of in every worker."""
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.ops.moe_mxfp4_aux import _mxfp4_moe_sort_internal_is_supported

    with dry_run(arch, cu_num):
        _mxfp4_moe_sort_internal_is_supported(256, 8, 7168, 32, True)


@contextmanager
def _requested_kernels():
    """Collect the cache identity of every FlyDSL launch a replay reaches."""
    from flydsl.compiler.jit_function import JitFunction

    kernels = set()
    build_key = JitFunction._build_full_cache_key

    def record(self, *args, **kwargs):
        key = build_key(self, *args, **kwargs)
        kernels.add((self.manager_key, key))
        return key

    with mock.patch.object(JitFunction, "_build_full_cache_key", record):
        yield kernels


def _replay_recorded(job: dict) -> tuple[dict, frozenset | None]:
    with _requested_kernels() as kernels:
        result = compile_one_config(**job)
    return result, (None if result["compile_time"] is None else frozenset(kernels))


def _bisect(job: dict, kernels: dict, low: int, high: int) -> list[dict]:
    """Replay the M between ``low`` and ``high`` at which the requested kernels
    change. Each choice the runtime makes from M is monotonic, so two M that
    request the same kernels bound a run that requests nothing else."""
    if high - low < 2 or kernels[low] == kernels[high]:
        return []
    middle = (low + high) // 2
    result, kernels[middle] = _replay_recorded({**job, "m": middle})
    if kernels[middle] is None:
        return [result]
    return [
        result,
        *_bisect(job, kernels, low, middle),
        *_bisect(job, kernels, middle, high),
    ]


def _judge_default_caller(outcomes: list[dict]) -> bool:
    """Classify the runtime's refusals of a row's default caller. Refused at
    only some of the row's token counts, the row meets an M-dependent runtime
    limit (e.g. the 32-bit route-reduce workspace) and those token counts are
    reported; refused at every one, the row is never served, which fails.
    Returns whether the default caller replayed at any token count."""
    served = any(
        r["compile_time"] is not None and "unsupported" not in r for r in outcomes
    )
    for result in outcomes:
        if "unsupported" not in result:
            continue
        reason = result.pop("unsupported")
        if served:
            result["refused"] = reason
        else:
            result["compile_time"] = None
            print(
                f"  [FAIL] {result['shape']}  arch={result['compile_arch']}: {reason}",
                flush=True,
            )
    return served


def _replay_batch(jobs: list[dict], kernel_name: str) -> dict:
    """Replay the jobs, then bisect each variant's bucket where its two token
    counts request different kernels: a row can switch kernels more than once
    inside its bucket."""
    results, probes, variants = [], [], {}
    for job in jobs:
        result, kernels = _replay_recorded(job)
        results.append(result)
        variant = tuple(sorted((k, v) for k, v in job.items() if k != "m"))
        _, by_m, outcomes = variants.setdefault(variant, (job, {}, []))
        by_m[job["m"]] = kernels
        outcomes.append(result)
    for job, kernels, outcomes in variants.values():
        if job["primary"] and not _judge_default_caller(outcomes):
            continue
        if len(kernels) == 2 and None not in kernels.values():
            probes += _bisect(job, kernels, *sorted(kernels))
    return {
        "kernel_name": kernel_name,
        "compile_time": 0.0,
        "results": results,
        "probes": probes,
    }


def _batches(jobs: list[dict]) -> list[list[dict]]:
    """Each row's jobs in runs of at most ``_JOBS_PER_WORKER``, largest first.
    A run keeps both token counts of a variant together. Equal-sized runs are
    shuffled (stably) so that neighbouring rows of one model, which share
    kernels and would wait on each other's compiles, do not start together."""
    rows: dict[tuple, list[dict]] = {}
    for job in jobs:
        rows.setdefault(tuple(job[f] for f in _ROW_KEY), []).append(job)
    batches = [
        (row, row_jobs[start : start + _JOBS_PER_WORKER])
        for row, row_jobs in rows.items()
        for start in range(0, len(row_jobs), _JOBS_PER_WORKER)
    ]
    batches.sort(key=lambda b: (-len(b[1]), zlib.crc32(repr(b[0]).encode())))
    return [batch for _, batch in batches]


def _replay_arch(jobs: list[dict], arch: str) -> int:
    """Replay ``jobs`` in this process, which ``run`` started for ``arch``."""
    check_replay_process(arch)
    # Only the two-stage path queries it; the gfx1250 grouped path never does.
    if not all(_is_grouped(job) for job in jobs):
        _load_host_helpers(arch, _int(jobs[0]["cu_num"]))
    batches = _batches(jobs)
    outcomes = run_jobs_parallel(
        _replay_batch,
        [{"jobs": batch, "kernel_name": batch[0]["kernel_name"]} for batch in batches],
    )
    results, probes = [], []
    for batch, outcome in zip(batches, outcomes):
        # A worker that died takes its whole batch with it.
        results += outcome.get("results") or [{"compile_time": None}] * len(batch)
        probes += outcome.get("probes") or []
    failed = sum(1 for r in results if r["compile_time"] is None)
    rejected = Counter(r["unsupported"] for r in results if r.get("unsupported"))
    unsupported = sum(rejected.values())
    refused = [r for r in results if r.get("refused")]
    summary = (
        f"[aiter] FlyDSL MoE replay {arch}: "
        f"{len(results) - failed - unsupported - len(refused)} ok, {failed} failed, "
        f"{unsupported} unsupported caller-side variants"
    )
    if refused:
        summary += f", {len(refused)} token counts the runtime refuses"
    print(summary, flush=True)
    for reason, count in rejected.most_common():
        print(f"  [SKIP] {count} x {reason}", flush=True)
    for result in refused:
        print(f"  [REFUSED] {result['shape']}: {result['refused']}", flush=True)
    probes_failed = sum(1 for r in probes if r["compile_time"] is None)
    if probes:
        print(
            f"  [BISECT] {len(probes)} token counts inside buckets, "
            f"{probes_failed} failed",
            flush=True,
        )
    return failed + probes_failed


def run(
    csv_paths: Sequence[str],
    *,
    deployments: Sequence[str] = (),
    edges: bool = True,
    check: bool = False,
) -> int:
    """Replay every FlyDSL row of ``csv_paths`` and every ``fused_moe``
    deployment of the files ``deployments`` whose arch ``GPU_ARCHS`` names (all
    of them when it names none), one process per arch. Returns the number of
    archs whose replay failed."""
    jobs = _collect_jobs(csv_paths, deployments, edges)
    targets = aot_target_archs()
    skipped = Counter(
        job_arch(job) for job in jobs if targets and job_arch(job) not in targets
    )
    jobs = [job for job in jobs if job_arch(job) not in skipped]
    archs = sorted({job_arch(job) for job in jobs})
    rows = {
        tuple(job[f] for f in _ROW_KEY) for job in jobs if not job.get("deployment")
    }
    calls = {job["deployment"] for job in jobs if job.get("deployment")}
    print(
        f"[aiter] FlyDSL MoE replay: {len(rows)} rows, {len(calls)} deployments, "
        f"{len(jobs)} jobs, archs {', '.join(archs) or '-'}",
        flush=True,
    )
    for arch, count in sorted(skipped.items()):
        print(f"  [SKIP] {count} jobs for {arch}, which GPU_ARCHS omits", flush=True)
    # The archs replay side by side and split one worker budget by job count.
    workers = get_max_workers(len(jobs))
    jobs_per_arch = Counter(job_arch(job) for job in jobs)
    if jobs_per_arch:
        # Importing aiter builds its C++ core module, and the two-stage replays
        # load another for a host-side query. Built by a replay, each would take
        # that replay's arch flags, which not every toolchain builds (ROCm 7.2's
        # HIP lacks the cluster launch gfx1250 builds turn on), and its share of
        # the workers. The replays run only their host code, so build both once,
        # first, with the whole budget.
        two_stage = Counter(job_arch(job) for job in jobs if not _is_grouped(job))
        arch = (two_stage or jobs_per_arch).most_common(1)[0][0]
        prepare = "import aiter"
        if two_stage:
            cu_num = next(_int(job["cu_num"]) for job in jobs if job_arch(job) == arch)
            prepare = (
                "from aiter.aot.flydsl.moe import _load_host_helpers; "
                f"_load_host_helpers({arch!r}, {cu_num})"
            )
        command = [sys.executable, "-c", prepare]
        subprocess.run(command, env=replay_env(arch), check=True)
    replays = []
    for arch in archs:
        command = [sys.executable, "-m", "aiter.aot.flydsl.moe", "--arch", arch]
        if csv_paths:
            command += ["--csv", *csv_paths]
        if deployments:
            command += ["--deployments", *deployments]
        if not edges:
            command.append("--row-tokens-only")
        if check:
            command.append("--check")
        share = max(1, round(workers * jobs_per_arch[arch] / len(jobs)))
        replays.append(subprocess.Popen(command, env=replay_env(arch, share)))
    return sum(replay.wait() != 0 for replay in replays)


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile the FlyDSL kernels of tuned fused-MoE configs "
        "and MoE deployments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    parser.add_argument(
        "--deployments",
        type=str,
        nargs="+",
        help="Path(s) to MoE deployment file(s), whose fused_moe deployments are\n"
        "replayed; default moe_deployments.json. Without --csv and\n"
        "--deployments both defaults are replayed, with one of them only\n"
        "what it names.",
    )
    parser.add_argument(
        "--row-tokens-only",
        action="store_true",
        help="replay each row only at its tuned token count, and each EP\n"
        "deployment at each bucket's largest M",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile nothing; fail on any kernel missing from the cache",
    )
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.csv is None and args.deployments is None:
        args.csv, args.deployments = DEFAULT_CSVS, DEFAULT_DEPLOYMENTS
    csv_paths = [os.path.abspath(p) for p in args.csv or ()]
    deployments = [os.path.abspath(p) for p in args.deployments or ()]
    for path in csv_paths + deployments:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)
    if args.check:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    edges = not args.row_tokens_only

    if args.arch:
        jobs = _collect_jobs(csv_paths, deployments, edges)
        jobs = [job for job in jobs if job_arch(job) == args.arch]
        sys.exit(1 if jobs and _replay_arch(jobs, args.arch) else 0)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    print("=" * 72)
    print(f"FlyDSL MoE AOT {'cache check' if args.check else 'pre-compilation'}")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    for path in deployments:
        print(f"  Deployments:  {path}")
    print(f"  Cache dir:    {cache_dir}")
    print("=" * 72, flush=True)

    total_t0 = time.time()
    failed_archs = run(
        csv_paths, deployments=deployments, edges=edges, check=args.check
    )
    print(f"\n  Total time:   {time.time() - total_t0:.1f}s")
    if failed_archs:
        print("Some replays failed. Check output above for details.")
        sys.exit(1)
    print("All replays succeeded. Cache is ready.")
    sys.exit(0)


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
