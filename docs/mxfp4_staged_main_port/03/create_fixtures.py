"""Prepare admitted specified-pair CSV requests without compiling or GPU work."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

# (id, model, token, BM, BN, BK, epilog, FP8, FP4, limit, clamp fixture)
REQUESTS = (
    (
        "minimax128-bm16-reduce",
        "minimax",
        128,
        16,
        128,
        128,
        "reduce",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax128-bm32-atomic",
        "minimax",
        128,
        32,
        128,
        128,
        "atomic",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax2048-bm64-reduce",
        "minimax",
        2048,
        64,
        128,
        128,
        "reduce",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax2048-bm128-atomic",
        "minimax",
        2048,
        128,
        128,
        128,
        "atomic",
        False,
        False,
        None,
        False,
    ),
    (
        "gpt2048-combined5-atomic",
        "gpt",
        2048,
        128,
        128,
        128,
        "atomic",
        True,
        True,
        5,
        False,
    ),
    (
        "minimax128-bm16-atomic",
        "minimax",
        128,
        16,
        128,
        128,
        "atomic",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax128-bm32-reduce",
        "minimax",
        128,
        32,
        128,
        128,
        "reduce",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax2048-bm64-atomic",
        "minimax",
        2048,
        64,
        128,
        128,
        "atomic",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax2048-bm128-reduce",
        "minimax",
        2048,
        128,
        128,
        128,
        "reduce",
        False,
        False,
        None,
        False,
    ),
    (
        "minimax2048-bm128-scatter",
        "minimax",
        2048,
        128,
        128,
        128,
        "scatter",
        False,
        False,
        None,
        False,
    ),
    (
        "gpt128-fp8-bm16-k128",
        "gpt",
        128,
        16,
        128,
        128,
        "reduce",
        True,
        False,
        None,
        False,
    ),
    (
        "gpt128-fp8-bm32-k256",
        "gpt",
        128,
        32,
        128,
        256,
        "reduce",
        True,
        False,
        None,
        False,
    ),
    (
        "gpt2048-fp8-bm64-k128",
        "gpt",
        2048,
        64,
        128,
        128,
        "reduce",
        True,
        False,
        None,
        False,
    ),
    (
        "gpt2048-fp8-bm128-k256",
        "gpt",
        2048,
        128,
        128,
        256,
        "reduce",
        True,
        False,
        None,
        False,
    ),
    (
        "gpt2048-fp4-n128",
        "gpt",
        2048,
        128,
        128,
        128,
        "scatter",
        False,
        True,
        None,
        False,
    ),
    (
        "gpt2048-fp4-n256",
        "gpt",
        2048,
        128,
        256,
        128,
        "scatter",
        False,
        True,
        None,
        False,
    ),
    (
        "gpt2048-limit5-reduce-clamp",
        "gpt",
        2048,
        128,
        128,
        128,
        "reduce",
        False,
        False,
        5,
        True,
    ),
    (
        "gpt2048-combined7-reduce",
        "gpt",
        2048,
        128,
        128,
        128,
        "reduce",
        True,
        True,
        None,
        False,
    ),
    (
        "gpt2048-combined7-scatter",
        "gpt",
        2048,
        128,
        128,
        128,
        "scatter",
        True,
        True,
        None,
        False,
    ),
    (
        "gpt2048-combined5-reduce-clamp",
        "gpt",
        2048,
        128,
        128,
        128,
        "reduce",
        True,
        True,
        5,
        True,
    ),
    (
        "gpt2048-combined5-scatter-clamp",
        "gpt",
        2048,
        128,
        256,
        128,
        "scatter",
        True,
        True,
        5,
        True,
    ),
    (
        "gpt2048-situv2-reduce",
        "situv2",
        2048,
        128,
        128,
        128,
        "reduce",
        False,
        False,
        None,
        False,
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name) != "":
            parser.error(
                f"{name} must be explicitly empty for offline fixture preparation"
            )

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

    moe = import_moe_tuner()
    import flydsl.compiler
    import torch

    from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel
    from op_tests.tuning_tests.test_mxfp4_tuner_lifecycle import KEYS, shape

    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    tuner.keys = KEYS
    candidates_by_shape = {}
    fixtures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "fixtures"
    csv_dir.mkdir(exist_ok=True)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "offline fixture enumeration reached compilation or GPU work"
        )

    with (
        mock.patch.object(flydsl.compiler, "compile", side_effect=forbidden),
        mock.patch.object(torch.cuda, "_lazy_init", side_effect=forbidden),
        mock.patch.object(torch.cuda, "synchronize", side_effect=forbidden),
        mock.patch.object(torch.cuda, "Event", side_effect=forbidden),
        mock.patch.object(torch.cuda, "CUDAGraph", side_effect=forbidden),
    ):
        for request in REQUESTS:
            fixture_id, model, token, bm, bn, bk, epilog, fp8, fp4, limit, clamp = (
                request
            )
            shape_key = (model, token)
            if shape_key not in candidates_by_shape:
                row = shape(token)
                if model == "minimax":
                    row.update(
                        inter_dim=384,
                        expert=256,
                        topk=8,
                        act_type="ActivationType.Silu",
                    )
                elif model == "situv2":
                    row["act_type"] = "ActivationType.Situv2"
                candidates_by_shape[shape_key] = (
                    row,
                    tuner._candidate_rows(row, full_search=False),
                )
            row, candidates = candidates_by_shape[shape_key]
            selected = None
            for candidate in candidates:
                config = parse_flydsl_v2_gemm2_kernel(candidate["kernelName2"])
                if (
                    config["tile_m"] == bm
                    and config["tile_n"] == bn
                    and config["tile_k"] == bk
                    and config["epilog"] == epilog
                ):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError(
                    f"No admitted pruned pipeline pair matches {fixture_id}"
                )
            path = csv_dir / f"{fixture_id}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=selected)
                writer.writeheader()
                writer.writerow(selected)
            fixtures.append(
                {
                    "id": fixture_id,
                    "csv": os.path.relpath(path, repo_root),
                    "shape": {
                        key: row[key]
                        for key in (
                            "token",
                            "model_dim",
                            "inter_dim",
                            "expert",
                            "topk",
                            "act_type",
                        )
                    },
                    "fp8": fp8,
                    "fp4": fp4,
                    "limit": limit,
                    "clamp_fixture": clamp,
                    "kn1": selected["kernelName1"],
                    "kn2": selected["kernelName2"],
                    "bm": bm,
                    "bn": bn,
                    "bk": bk,
                }
            )
    (args.output_dir / "fixtures.json").write_text(
        json.dumps(fixtures, indent=2) + "\n"
    )
    origin = {
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "kind": "specified-pair requests from current admitted M_est-pruned enumeration",
        "numerical_acceptance": False,
        "timing_acceptance": False,
        "gpu_work": False,
        "flydsl_compilation": False,
        "fixture_count": len(fixtures),
        "shape_candidate_counts": {
            f"{model}-{token}": len(candidates)
            for (model, token), (_, candidates) in candidates_by_shape.items()
        },
    }
    (args.output_dir / "fixtures-origin.json").write_text(
        json.dumps(origin, indent=2) + "\n"
    )
    print(
        f"Prepared {len(fixtures)} specified-pair requests; no GPU or FlyDSL compilation"
    )


if __name__ == "__main__":
    main()
