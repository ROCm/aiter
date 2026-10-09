#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL kernels of tuned fused-MoE configs.

Each tuned row is replayed through the runtime's own implementation --
``_fused_moe_impl``, or ``_fhmoe_impl`` for heterogeneous shared-expert rows --
on FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run`. The runtime
looks the row up, selects and parameterizes its kernels, sizes intermediate
buffers and runs its auxiliary kernels exactly as it does at inference, and
every FlyDSL launch on that path is compiled into the cache instead. This
module only builds what a caller passes in: activations, routing, and weights
quantized and preshuffled the way ``op_tests/test_moe_2stage.py`` prepares
them, plus the caller-side choices a row does not pin down (bias, gate/up
layout, SiTUv2 betas).

A row tuned at token count T also serves smaller M that ``get_padded_M`` maps
to T, and some compile-time parameters follow the actual M, so each row is
replayed at the smallest such M as well as at T.

aiter fixes some arch-dependent values at import (e.g. ``dtypes.fp8``), so the
replay runs in one process per target arch, with the GPUs hidden and
``GPU_ARCHS`` set before aiter is imported. Those processes need no GPU.

Usage:
    # Compile every FlyDSL kernel the default tuned CSVs reach
    python -m aiter.aot.flydsl.moe

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv

    # Verify a cache instead: fail on any kernel the replay would compile
    python -m aiter.aot.flydsl.moe --check

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
"""

import argparse
import csv
import functools
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from unittest import mock

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    cu_num_to_arch,
    dedupe_jobs,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_FMOE_FILE,
    AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE,
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
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
    "kernelName1",
    "kernelName2",
)
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


def _reaches_flydsl(row: dict) -> bool:
    kernel1 = (row.get("kernelName1") or "").strip()
    kernel2 = (row.get("kernelName2") or "").strip()
    # A cktile stage1 finishes with a FlyDSL activation epilogue.
    return kernel1.startswith(("flydsl_", "cktile_")) or kernel2.startswith("flydsl_")


def _token_points(token: int, edges: bool) -> list[int]:
    if not edges or token < 2 or token & (token - 1) or token > 32768:
        return [token]
    return [token // 2 + 1, token]


def _caller_gate_mode(row: dict):
    """Gate/up weight layout serving uses for a quant family (gu-interleaved
    for fp8 activations, as with ATOM_MOE_GU_ITLV=1)."""
    from aiter.ops.flydsl.moe_common import GateMode

    q_dtype_a, q_dtype_w = _name(row["q_dtype_a"]), _name(row["q_dtype_w"])
    mxfp8 = q_dtype_w in _FP8 and _name(row["q_type"]) == "per_1x32"
    if q_dtype_a in _FP8 and (q_dtype_w == _FP4 or mxfp8):
        return GateMode.INTERLEAVE
    return GateMode.SEPARATED


def _variants(row: dict) -> list[dict]:
    """Caller-side choices a tuned row leaves open; the first is the one
    serving uses by default and must replay cleanly."""
    from aiter.ops.flydsl.moe_common import (
        DEFAULT_SITUV2_BETA,
        DEFAULT_SITUV2_LINEAR_BETA,
        GateMode,
    )
    from aiter.ops.flydsl.mxfp4_kname import _is_mxfp4_kname

    if _int(row["shared_expert_id"], -1) >= 0:
        gate_modes = [GateMode[_name(row["gate_mode"]) or "SEPARATED"]]
        biases = ["none"]
    else:
        gate_modes = [_caller_gate_mode(row)]
        if _is_mxfp4_kname(row["kernelName1"]):
            layouts = (GateMode.SEPARATED, GateMode.INTERLEAVE)
            gate_modes += [mode for mode in layouts if mode not in gate_modes]
        biases = ["none"]
        if (
            _name(row["q_type"]) == "per_1x32"
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
    return [
        {
            "gate_mode": gate_mode.value,
            "bias": bias,
            "beta": beta,
            "linear_beta": linear_beta,
        }
        for gate_mode in gate_modes
        for bias in biases
        for beta, linear_beta in betas
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


def _dtype(name: str):
    import torch

    return getattr(torch, _name(name))


def _caller_inputs(job: dict) -> dict:
    """What a caller hands fused_moe for this row: bf16 activations, routing,
    and weights/scales quantized and preshuffled per quant family."""
    import torch

    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import (
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

    if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2:
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

        _fhmoe_impl(**common, shared_expert_id=shared_expert_id)
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
def _runtime_reads(csv_path: str, shared_expert: bool):
    """Point the runtime's tuned-config lookup at the row's own table."""
    env_name = "AITER_CONFIG_FHMOE" if shared_expert else "AITER_CONFIG_FMOE"
    get_config_file = type(AITER_CONFIGS).get_config_file
    with mock.patch.dict(os.environ, {env_name: csv_path}):
        get_config_file.cache_clear()
        try:
            yield
        finally:
            get_config_file.cache_clear()


def _describe(job: dict) -> str:
    variant = [f"gate={job['gate_mode']}"]
    if job["bias"] != "none":
        variant.append(f"bias={job['bias']}")
    if job["beta"] is not None:
        variant.append(f"beta={job['beta']:g}/{job['linear_beta']:g}")
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
            _runtime_reads(job["csv_path"], _int(job["shared_expert_id"], -1) >= 0),
            dry_run(arch, _int(job["cu_num"])),
            _without_device_queries(),
        ):
            _replay(job)
        result["compile_time"] = time.time() - started
    except NotImplementedError as error:
        if job["primary"]:
            print(f"  [FAIL] {description}  arch={arch}: {error}", flush=True)
        else:
            # The runtime rejects this caller-side combination for the row.
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


def _replay_arch(jobs: list[dict], arch: str) -> int:
    """Replay ``jobs`` in this process; GPU_ARCHS must already name ``arch``."""
    if os.environ.get("GPU_ARCHS") != arch:
        raise RuntimeError(
            f"replaying {arch} jobs needs GPU_ARCHS={arch} set before aiter is "
            "imported; run through `python -m aiter.aot.flydsl.moe`"
        )
    _load_host_helpers(arch, _int(jobs[0]["cu_num"]))
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(1 for r in results if r["compile_time"] is None)
    rejected = Counter(r["unsupported"] for r in results if r.get("unsupported"))
    unsupported = sum(rejected.values())
    print(
        f"[aiter] FlyDSL MoE replay {arch}: {len(results) - failed - unsupported} ok, "
        f"{failed} failed, {unsupported} unsupported caller-side variants",
        flush=True,
    )
    for reason, count in rejected.most_common():
        print(f"  [SKIP] {count} x {reason}", flush=True)
    return failed


def _arch_env(arch: str) -> dict:
    env = dict(os.environ)
    env.pop("AITER_AOT_IMPORT", None)
    env.update(
        GPU_ARCHS=arch,
        HIP_VISIBLE_DEVICES="-1",
        ROCR_VISIBLE_DEVICES="-1",
        CUDA_VISIBLE_DEVICES="-1",
    )
    return env


def run(csv_paths: list[str], *, edges: bool = True, check: bool = False) -> int:
    """Replay every FlyDSL row of ``csv_paths``, one process per target arch.
    Returns the number of archs whose replay failed."""
    jobs = collect_aot_jobs(csv_paths, functools.partial(parse_csv, edges=edges))
    archs = sorted({job_arch(job) for job in jobs})
    # gate_mode is overwritten by each job's variant.
    row_fields = ("csv_path", *(f for f in _ROW_FIELDS if f != "gate_mode"))
    rows = {tuple(job[f] for f in row_fields) for job in jobs}
    print(
        f"[aiter] FlyDSL MoE replay: {len(rows)} rows, {len(jobs)} jobs, "
        f"archs {', '.join(archs) or '-'}",
        flush=True,
    )
    failed = 0
    for arch in archs:
        command = [sys.executable, "-m", "aiter.aot.flydsl.moe", "--arch", arch]
        command += ["--csv"]
        command += csv_paths
        if not edges:
            command.append("--row-tokens-only")
        if check:
            command.append("--check")
        replay = subprocess.run(command, env=_arch_env(arch), check=False)
        failed += replay.returncode != 0
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile the FlyDSL kernels of tuned fused-MoE configs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    parser.add_argument(
        "--row-tokens-only",
        action="store_true",
        help="replay each row only at its tuned token count",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile nothing; fail on any kernel missing from the cache",
    )
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    if args.check:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    edges = not args.row_tokens_only

    if args.arch:
        jobs = collect_aot_jobs(csv_paths, functools.partial(parse_csv, edges=edges))
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
    print(f"  Cache dir:    {cache_dir}")
    print("=" * 72, flush=True)

    total_t0 = time.time()
    failed_archs = run(csv_paths, edges=edges, check=args.check)
    print(f"\n  Total time:   {time.time() - total_t0:.1f}s")
    if failed_archs:
        print("Some replays failed. Check output above for details.")
        sys.exit(1)
    print("All replays succeeded. Cache is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
