#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Targeted alternating pair replay for a concrete calibration ranking concern.

This is not a search: only the two recorded CSV winners execute. Fixed generated
input/routes/weights and normal kernel-duration-sum profiling are retained.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def alternating_samples(measure, repeats):
    samples = []
    for cycle in range(repeats):
        order = ("left", "right") if cycle % 2 == 0 else ("right", "left")
        for label in order:
            samples.append({"cycle": cycle, "label": label, **measure(label)})
    report = {"samples": samples}
    for label in ("left", "right"):
        values = [
            sample["kernel_sum_us"] for sample in samples if sample["label"] == label
        ]
        report[label] = {
            "kernel_sum_median_us": statistics.median(values),
            "kernel_sum_min_us": min(values),
            "kernel_sum_max_us": max(values),
            "kernel_sum_population_stddev_us": statistics.pstdev(values),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--reason",
        required=True,
        help="Concrete observed discrepancy motivating this replay",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two cycles are needed to reverse the pair order")
    started = time.perf_counter()
    import torch

    from aiter import ActivationType, dtypes
    from aiter.test_common import run_perftest
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
        Mxfp4FlydslTuner,
        cosine_diff_compare,
    )

    rows = {}
    for label, path in (("left", args.left), ("right", args.right)):
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == 1, records
        rows[label] = records[0]
    fixed_keys = (
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
    )
    assert all(rows["left"][key] == rows["right"][key] for key in fixed_keys)
    shape = {
        key: int(rows["left"][key])
        for key in ("token", "model_dim", "inter_dim", "expert", "topk")
    }
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    configuration = tuner._invocation_configuration(SimpleNamespace())
    data = tuner._prepare_case(**shape, dtype=dtypes.bf16)
    activation = getattr(ActivationType, rows["left"]["act_type"].rsplit(".", 1)[-1])
    reference = tuner._torch_ref(
        data,
        shape["topk"],
        dtypes.bf16,
        activation,
        swiglu_limit=configuration["swiglu_limit"],
    )

    def launch(label):
        row = rows[label]
        return tuner._port_e2e(
            data,
            row["kernelName1"],
            row["kernelName2"],
            shape["topk"],
            shape["expert"],
            shape["model_dim"],
            dtypes.bf16,
            swiglu_limit=configuration["swiglu_limit"],
        )

    def measure(label):
        output = launch(label)
        torch.cuda.synchronize()
        error = float(cosine_diff_compare(reference, output, printLog=False))
        assert (
            math.isfinite(error)
            and error <= 0.1
            and torch.isfinite(output).all().item()
        ), error
        begin = time.perf_counter()
        _, us = run_perftest(lambda: launch(label), num_warmup=5, num_iters=101)
        assert math.isfinite(us) and us > 0, us
        return {
            "kernel_sum_us": float(us),
            "cosine_diff": error,
            "profiling_attempt_wall_seconds": time.perf_counter() - begin,
        }

    result = alternating_samples(measure, args.repeats)

    def fingerprint(tensor):
        return hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    result.update(
        reason=args.reason,
        shape=shape,
        winners=rows,
        implementation_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        swiglu_limit=configuration["swiglu_limit"],
        configuration=configuration,
        environment={
            name: os.environ.get(name)
            for name in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "TUNE_MOE_ROUTING_SEED",
                "TUNE_MOE_EXPERT_BALANCE",
                "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
                "AITER_FLYDSL_STAGE2_FP8",
                "AITER_MXFP4_INTERMEDIATE",
                "FLYDSL_RUNTIME_CACHE_DIR",
                "CK_DIR",
            )
        },
        fingerprints={
            key: fingerprint(data[key]) for key in ("input", "topk_ids", "topk_weights")
        },
        measurement={
            "warmup": 5,
            "iters": 101,
            "metric": "normal continuous main run_perftest device kernel-duration sum; profiling wall is search diagnostic, not serving latency",
            "repeats_per_pair": args.repeats,
            "one_prepared_fixture": True,
        },
        wall_seconds=time.perf_counter() - started,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
