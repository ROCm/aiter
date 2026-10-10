#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prepare all 27 searches and count the real shared enumeration without a GPU.

The imports need the installed ROCm/FlyDSL toolchain. GPU discovery is synthetic
and all visibility is empty for the whole process; no GPU work or JIT is allowed.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name) != "":
            parser.error(f"{name} must be explicitly empty for this whole process")
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

    moe = import_moe_tuner()
    import flydsl.compiler
    import torch

    from csrc.ck_gemm_moe_2stages_codegen.mxfp4_staged_search import (
        build_staged_candidate_plan,
    )
    from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import KEYS, shape

    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    tuner.keys = KEYS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.output_dir / "inputs"
    inputs.mkdir(exist_ok=True)
    jobs = []
    for token in (64, 128, 512, 1024, 2048, 4096, 8192):
        row = shape(
            token,
            inter_dim=384,
            expert=256,
            topk=8,
            act_type="ActivationType.Silu",
            gate_mode="separated",
        )
        modes = (
            ("full", "prune", "staged")
            if token in (64, 2048, 8192)
            else ("prune", "staged")
        )
        for mode in modes:
            jobs.append(
                dict(
                    id=f"minimax-{token}-{mode}",
                    model="minimax",
                    token=token,
                    config="default",
                    mode=mode,
                    fp8=False,
                    fp4=False,
                    limit=None,
                    row=row,
                )
            )
    for config, fp8, fp4, limit in (
        ("default", False, False, None),
        ("fp8", True, False, None),
        ("fp4", False, True, None),
        ("limit5", False, False, 5),
        ("combined5", True, True, 5),
    ):
        for mode in ("prune", "staged"):
            jobs.append(
                dict(
                    id=f"gpt-2048-{config}-{mode}",
                    model="gpt",
                    token=2048,
                    config=config,
                    mode=mode,
                    fp8=fp8,
                    fp4=fp4,
                    limit=limit,
                    row=shape(2048, gate_mode="separated"),
                )
            )

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError("offline counting reached compilation or GPU work")

    counts = []
    with mock.patch.object(flydsl.compiler, "compile", side_effect=forbidden):
        with mock.patch.multiple(
            torch.cuda,
            **{
                name: forbidden
                for name in ("_lazy_init", "synchronize", "Event", "CUDAGraph")
            },
        ):
            for job in jobs:
                row = job["row"]
                pairs = tuner._candidate_rows(row, full_search=job["mode"] == "full")
                plan = build_staged_candidate_plan(row, pairs)
                groups = []
                for group in plan.pipeline_groups:
                    n1, n2 = len(group.gemm1_candidates), len(group.gemm2_candidates)
                    assert len(group.pairs) == n1 * n2
                    groups.append(
                        dict(
                            block_m=group.key.block_m,
                            accumulation_mode=group.key.accumulation_mode.value,
                            gemm1_candidates=n1,
                            gemm2_candidates=n2,
                            pairs=len(group.pairs),
                            k2_shortlist_pairs_upper=min(2, n1) * min(2, n2),
                        )
                    )
                counts.append(
                    dict(
                        id=job["id"],
                        m_est=(row["token"] * row["topk"] + row["expert"] - 1)
                        // row["expert"],
                        total_pairs=len(pairs),
                        unique_gemm1=sum(len(g.candidates) for g in plan.gemm1_groups),
                        unique_gemm2=sum(len(g.candidates) for g in plan.gemm2_groups),
                        k2_shortlist_pairs_upper=sum(
                            g["k2_shortlist_pairs_upper"] for g in groups
                        ),
                        staged_exhaustive_fallback_pairs_upper=len(pairs),
                        groups=groups,
                    )
                )
                path = inputs / f"{job['id']}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                    writer.writerow(row)
                job["input"] = os.path.relpath(path, root)
                del job["row"]
    assert len(jobs) == 27 and Counter(job["model"] for job in jobs) == {
        "minimax": 17,
        "gpt": 10,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(jobs, indent=2) + "\n")
    output = dict(
        implementation_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        dependency="ROCm/FlyDSL imports with synthetic discovery; whole-process empty visibility; guarded GPU/JIT boundaries",
        legality="Static admitted pairs, not runtime compilation or numerical validity",
        records=counts,
    )
    (args.output_dir / "candidate-counts.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    lines = [
        "# Current shared-enumeration counts",
        "",
        output["dependency"],
        "",
        output["legality"],
        "",
        "G1 ranks are reused across each BM's two accumulation groups. Reduce/scatter share the non-atomic group. K2 counts are initial shortlist upper counts; reference/score failures can exhaust all admitted pairs.",
        "",
        "| Search | M_est | G1 | G2 | Pairs | K2 shortlist upper |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in counts:
        lines.append(
            f"| {row['id']} | {row['m_est']} | {row['unique_gemm1']} | {row['unique_gemm2']} | {row['total_pairs']} | {row['k2_shortlist_pairs_upper']} |"
        )
    lines += [
        "",
        "Per-BM/accumulation counts and every anchor's unpruned universe are in candidate-counts.json.",
    ]
    (args.output_dir / "candidate-counts.md").write_text("\n".join(lines) + "\n")
    print(f"Prepared {len(jobs)} separate search inputs and shared-enumeration counts")


if __name__ == "__main__":
    main()
