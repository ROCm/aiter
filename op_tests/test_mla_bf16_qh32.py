# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 BF16 QH32 MLA regressions and optional GPU graph benchmarks.

The default run is discovered by the op-test CI and checks causal tails,
nonuniform lengths, empty split outputs, and physical KV offsets at 2/4 GiB.
Numerical failures produce a nonzero exit status after all cases are recorded.

    python op_tests/test_mla_bf16_qh32.py
    python op_tests/test_mla_bf16_qh32.py --suite performance --output perf.jsonl

Performance measures full MLA GPU graph latency with reused inputs. Failed
correctness cases are still timed and marked, so base/PR comparisons can record
known failures without presenting their timings as correct implementations.
"""

import argparse
import hashlib
import itertools
import json
import statistics
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch

import aiter
from aiter.jit.core import AITER_ASM_DIR
from aiter.mla import get_meta_param, mla_decode_fwd

if __package__:
    from .test_mla import torch_mla_extend
else:
    from test_mla import torch_mla_extend

CO = "gfx950/mla/MLA_A16W16_1TG_4W_16mx4_32nx1_Coex0_Msk1_QH16.co"


def cases(suite):
    if suite == "correctness":
        for b, q, kv, splits in itertools.product(
            (1, 3), (1, 2, 4), (21, 63, 64, 65, 127, 128, 129, 256), (1, 2, 4)
        ):
            yield {
                "group": "tail_batch_split",
                "b": b,
                "q": q,
                "kv": [kv] * b,
                "splits": splits,
                "placement": "random",
            }
        for placement, splits in itertools.product(
            ("low", "below_2g", "cross_2g", "below_4g", "cross_4g", "above_4g"),
            (1, 2, 4),
        ):
            yield {
                "group": "kv_boundary",
                "b": 2,
                "q": 4,
                "kv": [65, 129],
                "splits": splits,
                "placement": placement,
            }
        for q, splits, lengths in itertools.product(
            (1, 2, 4), (2, 4, 8), ([21, 65, 129], [63, 64, 65])
        ):
            yield {
                "group": "variable_lengths",
                "b": 3,
                "q": q,
                "kv": lengths,
                "splits": splits,
                "placement": "random",
            }
    elif suite == "performance":
        for b, q, kv in itertools.product((1, 16, 64), (1, 4), (256, 2048, 8192)):
            yield {
                "group": "auto_splits",
                "b": b,
                "q": q,
                "kv": [kv] * b,
                "splits": None,
                "placement": "random",
            }
        for b, q, kv in itertools.product((1, 16), (1, 4), (256, 8192)):
            yield {
                "group": "single_split",
                "b": b,
                "q": q,
                "kv": [kv] * b,
                "splits": 1,
                "placement": "random",
            }
        for b, q in itertools.product((1, 16), (1, 4)):
            yield {
                "group": "short_tail",
                "b": b,
                "q": q,
                "kv": [65] * b,
                "splits": 4,
                "placement": "random",
            }


def inputs(case, seed, pool):
    torch.manual_seed(seed)
    b, qlen = case["b"], case["q"]
    lengths = case["kv"]
    count = sum(lengths)
    placement = case["placement"]
    if placement == "random":
        kv = pool
        pages = torch.randint(0, pool.shape[0] - 256, (count,), dtype=torch.int32)
    else:
        boundary_2g = (2**31 + 1151) // 1152
        boundary_4g = (2**32 + 1151) // 1152
        start = {
            "low": 256,
            "below_2g": boundary_2g - count - 256,
            "cross_2g": boundary_2g - count // 2,
            "below_4g": boundary_4g - count - 256,
            "cross_4g": boundary_4g - count // 2,
            "above_4g": boundary_4g + 256,
        }[placement]
        # Zero all other pages, so an incorrect wrapped address has deterministic data.
        kv = torch.zeros((start + count + 512, 1, 1, 576), dtype=torch.bfloat16)
        pages = torch.arange(start, start + count, dtype=torch.int32)
        kv[pages.long()] = torch.randn((count, 1, 1, 576), dtype=torch.bfloat16)
    query = torch.randn((b * qlen, 32, 576), dtype=torch.bfloat16)
    out = torch.full((b * qlen, 32, 512), float("nan"), dtype=torch.bfloat16)
    qptr = torch.arange(0, (b + 1) * qlen, qlen, dtype=torch.int32)
    kptr = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    last = torch.ones((b,), dtype=torch.int32)
    return query, kv, out, qptr, kptr, pages, last


def diff_metrics(reference, output):
    ref, out = reference.float(), output.float()
    finite = bool(torch.isfinite(out).all().item())
    mismatch = float((~torch.isclose(ref, out, rtol=1e-2, atol=1e-2)).float().mean())
    delta = out - ref
    cos_diff = float(
        1
        - 2
        * (ref.double() * out.double()).sum()
        / (ref.double().square() + out.double().square()).sum().clamp_min(1e-12)
    )
    return {
        "passed": finite and mismatch == 0.0,
        "finite": finite,
        "mismatch_fraction": mismatch,
        "max_abs_error": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_difference": cos_diff,
    }


def graph_time(call, graph_calls=64, samples=21):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            call()
    # Warm GPU and graph; allocation, compilation and capture are excluded from timing.
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / graph_calls)
    del graph
    return {
        "median_us": statistics.median(timings),
        "samples_us": timings,
        "graph_calls": graph_calls,
        "samples": samples,
    }


def poisoned_stage1(original_stage1, *stage_args, **stage_kwargs):
    stage_args[14].fill_(float("nan"))
    stage_args[15].fill_(float("nan"))
    return original_stage1(*stage_args, **stage_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("correctness", "performance"), default="correctness"
    )
    parser.add_argument("--label", default="current")
    parser.add_argument(
        "--output", type=Path, help="Also write JSONL results to this file"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--only", type=int, nargs="*")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "").split(":")[0]
    if arch != "gfx950":
        print(
            f"Skipping BF16 QH32 ASM regressions: requires gfx950, got {arch or properties.name}"
        )
        return
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(aiter.__file__).resolve().parent.parent
    metadata = {
        "type": "metadata",
        "label": args.label,
        "suite": args.suite,
        "aiter_source": str(root),
        "co_sha256": hashlib.sha256(
            (Path(AITER_ASM_DIR) / CO).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": properties.name,
        "arch": properties.gcnArchName,
        "time": time.time(),
        "rtol": 1e-2,
        "atol": 1e-2,
        "max_mismatch_fraction": 0.0,
        "timing": "GPU event around graph of 64 mla_decode_fwd calls; reused inputs",
    }
    failures = []
    tested = 0
    output_context = (
        args.output.open("w", buffering=1) if args.output else nullcontext()
    )
    with output_context as output_file:

        def emit(item):
            if output_file is not None:
                output_file.write(json.dumps(item) + "\n")
            print(json.dumps(item), flush=True)

        emit(metadata)
        torch.manual_seed(6578)
        pool_pages = 2097152 if args.suite == "performance" else 65536
        pool = torch.randn((pool_pages, 1, 1, 576), dtype=torch.bfloat16)
        for index, case in enumerate(cases(args.suite)):
            if args.only is not None and index not in args.only:
                continue
            tensors = inputs(case, 20261021 + index, pool)
            query, kv, out, qptr, kptr, pages, last = tensors
            actual_splits, _ = get_meta_param(
                case["splits"],
                case["b"],
                sum(case["kv"]),
                32,
                case["q"],
                torch.bfloat16,
            )

            call = partial(
                mla_decode_fwd,
                *tensors,
                case["q"],
                sm_scale=576**-0.5,
                num_kv_splits=case["splits"],
                causal=True,
            )

            ref, _ = torch_mla_extend(
                query,
                kv.view(-1, 1, 576),
                qptr,
                kptr,
                pages,
                576**-0.5,
                512,
                64,
                torch.bfloat16,
                is_causal=True,
            )
            call()
            torch.cuda.synchronize()
            record = dict(
                type="case",
                index=index,
                **case,
                actual_splits=actual_splits,
                correctness=diff_metrics(ref, out),
                page_byte_min=int(pages.min()) * 1152,
                page_byte_max=int(pages.max()) * 1152,
            )
            if args.suite == "correctness" and actual_splits > 1:
                # Exercise the empty-split contract deterministically: poison scratch
                # allocated by the unchanged wrapper, immediately before stage 1.
                original_stage1 = aiter.mla_decode_stage1_asm_fwd

                aiter.mla_decode_stage1_asm_fwd = partial(
                    poisoned_stage1, original_stage1
                )
                try:
                    out.fill_(float("nan"))
                    call()
                    torch.cuda.synchronize()
                    record["poisoned_scratch_correctness"] = diff_metrics(ref, out)
                finally:
                    aiter.mla_decode_stage1_asm_fwd = original_stage1
            if args.profile and (args.only is not None or index == 0):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    call()
                    torch.cuda.synchronize()
                names = sorted(
                    {
                        event.name
                        for event in prof.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    }
                )
                record["profile_kernels"] = names
                record["target_kernel_seen"] = any(
                    "mla_a16w16_qh16_m16x4_n32x1_coex0_mask1" in n for n in names
                )
                assert record["target_kernel_seen"], names
            if args.suite == "performance":
                record["latency"] = graph_time(call)
            tested += 1
            if (
                not record["correctness"]["passed"]
                or not record.get("poisoned_scratch_correctness", {"passed": True})[
                    "passed"
                ]
            ):
                failures.append(index)
            emit(record)
            del call, query, kv, out, qptr, kptr, pages, last, tensors, ref
        emit(
            {
                "type": "complete",
                "label": args.label,
                "tested": tested,
                "failed_indices": failures,
            }
        )
    assert tested > 0, "No cases selected"
    assert (
        not failures
    ), f"BF16 QH32 MLA failed {len(failures)}/{tested} cases: {failures}"


if __name__ == "__main__":
    main()
