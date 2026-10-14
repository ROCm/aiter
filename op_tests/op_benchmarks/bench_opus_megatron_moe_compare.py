# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Same-shape Opus and TE/CK local-expert comparison.

The measured segment starts after Megatron has grouped local routes by expert.
It excludes the token dispatcher, EP communication, optimizer, and forward
cache production. Opus is reported with its allocating public wrapper and with
preallocated full-chain outputs. TransformerEngine is reported through the
same operation-fuser API used by Megatron, with per-expert parameters and
``single_grouped_weight``.
"""

import argparse
import json
from dataclasses import dataclass

import pandas as pd
import torch

from aiter.ops.opus import moe_backward as opus
from aiter.ops.opus.megatron_moe import (
    _opus_megatron_moe_poison_invalid_backward_raw,
    build_megatron_local_metadata,
    pad_megatron_local_rows,
)
from op_tests.op_benchmarks.opus_megatron_cases import CASES


@dataclass
class OpusPreallocated:
    output: opus.OpusMoeBackwardOutput
    d_scores_workspace: torch.Tensor
    d_x_route: torch.Tensor


def event_samples(fn, *, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) * 1000 for start, end in zip(starts, ends))
    return values[len(values) // 2], values[int(0.9 * (len(values) - 1))]


def compare(name, actual, expected):
    if actual is None:
        raise AssertionError(f"{name} gradient was not produced")
    af = actual.float()
    ef = expected.float()
    relative_l2 = ((af - ef).norm() / ef.norm().clamp_min(1e-12)).item()
    max_abs = (af - ef).abs().max().item()
    if not torch.isfinite(af).all() or relative_l2 >= 0.025:
        raise AssertionError((name, relative_l2, max_abs))
    return {"relative_l2": relative_l2, "max_abs": max_abs}


def make_counts(experts, routes, distribution):
    if experts == 1:
        return torch.tensor([routes], device="cuda", dtype=torch.int32)
    counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
    if distribution == "balanced":
        counts.fill_(routes // experts)
        counts[: routes % experts] += 1
    elif distribution == "skewed":
        hot = routes * 3 // 4
        counts[0] = hot
        tail = routes - hot
        counts[1:] = tail // (experts - 1)
        counts[1 : 1 + tail % (experts - 1)] += 1
    elif distribution == "empty":
        active = max(1, experts // 2)
        counts[:active] = routes // active
        counts[: routes % active] += 1
    elif distribution == "hot":
        counts[0] = routes - min(routes, experts - 1)
        counts[1 : 1 + min(routes, experts - 1)] = 1
    else:
        raise ValueError(f"unknown route distribution: {distribution}")
    return counts


def allocate_opus_preallocated(x, z, w1, w2, scores, metadata):
    inter_dim = w2.shape[2]
    down_block_n = opus._select_fixed_down_block_n(
        inter_dim,
        metadata.sorted_expert_ids,
        metadata.expert_padded_offsets,
        -1,
    )
    d_scores_parts = (inter_dim + down_block_n - 1) // down_block_n
    d_scores_workspace = torch.empty(
        (scores.numel(), d_scores_parts) if d_scores_parts > 1 else (0, 0),
        device=x.device,
        dtype=torch.float32,
    )
    output = opus.OpusMoeBackwardOutput(
        d_x=torch.empty_like(x),
        d_w1=torch.empty_like(w1),
        d_w2=torch.empty_like(w2),
        d_scores=torch.empty_like(scores),
        d_z_sorted=torch.empty_like(z),
        a_scaled=torch.empty(
            (z.shape[0], inter_dim), device=x.device, dtype=torch.bfloat16
        ),
    )
    return OpusPreallocated(
        output=output,
        d_scores_workspace=d_scores_workspace,
        d_x_route=torch.empty(
            (z.shape[0], x.shape[1]), device=x.device, dtype=torch.bfloat16
        ),
    )


def launch_opus_preallocated(
    dout, x, z, w1, w2, scores, metadata, workspace, *, consume_status
):
    output = workspace.output
    opus._opus_moe_full_bwd_raw(
        dout,
        x,
        x,
        z,
        w1,
        w2,
        scores,
        metadata.sorted_token_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        metadata.reverse_sorted,
        metadata.expert_padded_offsets,
        workspace.d_scores_workspace,
        output.d_z_sorted,
        output.a_scaled,
        output.d_scores,
        workspace.d_x_route,
        output.d_x,
        output.d_w1,
        output.d_w2,
        int(metadata.block_m),
        -1,
        -1,
        -1,
        -1,
        -1,
        False,
    )
    if consume_status:
        _opus_megatron_moe_poison_invalid_backward_raw(
            metadata.num_valid_ids,
            output.d_x,
            output.d_w1,
            output.d_w2,
            output.d_scores,
            output.d_z_sorted,
            output.a_scaled,
        )
    return output


def make_te_proxy_backend(e, d, i, counts, w1, w2, *, grouped_weight):
    import transformer_engine
    import transformer_engine.pytorch as te

    linear_kwargs = {
        "bias": False,
        "dtype": torch.bfloat16,
        "device": "cuda",
        "accumulate_into_main_grad": False,
        "single_grouped_weight": grouped_weight,
    }
    fc1 = te.ops.GroupedLinear(e, d, 2 * i, **linear_kwargs)
    activation = te.ops.ScaledSwiGLU(glu_interleave_size=None)
    fc2 = te.ops.GroupedLinear(e, i, d, **linear_kwargs)
    ops = te.ops.Sequential(fc1, activation, fc2)

    with torch.no_grad():
        if grouped_weight:
            fc1.weight.copy_(w1)
            fc2.weight.copy_(w2)
        else:
            for expert in range(e):
                getattr(fc1, f"weight{expert}").copy_(w1[expert])
                getattr(fc2, f"weight{expert}").copy_(w2[expert])

    def weight_parameters(layer):
        if grouped_weight:
            return [layer.weight]
        return [getattr(layer, f"weight{expert}") for expert in range(e)]

    fc1_weights = weight_parameters(fc1)
    fc2_weights = weight_parameters(fc2)

    def clear_grads():
        for weight in (*fc1_weights, *fc2_weights):
            weight.grad = None

    def graph(x, scores):
        clear_grads()
        x_leaf = x.detach().requires_grad_(True)
        score_leaf = scores.detach().reshape(-1).requires_grad_(True)
        output = ops(x_leaf, counts, score_leaf, counts)
        return output, x_leaf, score_leaf, fc1_weights, fc2_weights

    def weight_grads(weights):
        if grouped_weight:
            return weights[0].grad
        return torch.stack([weight.grad for weight in weights])

    return transformer_engine.__version__, graph, weight_grads


def measure_te_proxy(graph, x, scores, dout, *, warmup, iterations):
    def one_timed_backward():
        output, *_ = graph(x, scores)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output.backward(dout)
        end.record()
        return start, end

    for _ in range(warmup):
        one_timed_backward()
    torch.cuda.synchronize()
    pairs = [one_timed_backward() for _ in range(iterations)]
    torch.cuda.synchronize()
    backward = sorted(start.elapsed_time(end) * 1000 for start, end in pairs)

    def forward_backward():
        output, *_ = graph(x, scores)
        output.backward(dout)

    total_median, total_p90 = event_samples(
        forward_backward, warmup=warmup, iterations=iterations
    )
    return {
        "backward_median_us": backward[len(backward) // 2],
        "backward_p90_us": backward[int(0.9 * (len(backward) - 1))],
        "forward_backward_median_us": total_median,
        "forward_backward_p90_us": total_p90,
    }


def add_roofline_metrics(timing, *, flops, nbytes):
    timing["equivalent_tflops"] = flops / timing["median_us"] / 1e6
    timing["traffic_tb_s"] = nbytes / timing["median_us"] / 1e6
    return timing


def run(case_name, distribution, seed, *, warmup, iterations):
    shape = CASES[case_name]
    routes = shape.routes
    e, d, i = shape.local_experts, shape.model_dim, shape.expert_inter_dim
    if e == 1 and distribution != "balanced":
        return None
    torch.manual_seed(seed)

    def rand(*dims):
        return torch.randn(*dims, device="cuda", dtype=torch.bfloat16)

    counts = make_counts(e, routes, distribution)
    counts_host = counts.cpu().tolist()
    metadata = build_megatron_local_metadata(counts, routes)
    x, dout = rand(routes, d), rand(routes, d)
    w1, w2 = rand(e, 2 * i, d), rand(e, d, i)
    w1.mul_(d**-0.5)
    w2.mul_(i**-0.5)
    scores = torch.rand(routes, 1, device="cuda", dtype=torch.float32)
    z_parts = []
    row = 0
    for expert, count in enumerate(counts_host):
        z_parts.append(x[row : row + count] @ w1[expert].transpose(0, 1))
        row += count
    z_compact = torch.cat(z_parts, dim=0)
    z_padded = pad_megatron_local_rows(z_compact, metadata)

    def opus_allocating_fn():
        return opus.opus_moe_backward(dout, x, z_padded, w1, w2, scores, metadata)

    opus_reference = opus_allocating_fn()
    allocating_median, allocating_p90 = event_samples(
        opus_allocating_fn, warmup=warmup, iterations=iterations
    )

    opus_workspace = allocate_opus_preallocated(x, z_padded, w1, w2, scores, metadata)

    def opus_preallocated_fn():
        return launch_opus_preallocated(
            dout,
            x,
            z_padded,
            w1,
            w2,
            scores,
            metadata,
            opus_workspace,
            consume_status=False,
        )

    def opus_safe_fn():
        return launch_opus_preallocated(
            dout,
            x,
            z_padded,
            w1,
            w2,
            scores,
            metadata,
            opus_workspace,
            consume_status=True,
        )

    opus_preallocated_fn()
    preallocated_median, preallocated_p90 = event_samples(
        opus_preallocated_fn, warmup=warmup, iterations=iterations
    )
    safe_median, safe_p90 = event_samples(
        opus_safe_fn, warmup=warmup, iterations=iterations
    )

    te_results = {}
    te_errors = {}
    te_version = None
    for layout, grouped_weight in (("per_expert", False), ("single_grouped", True)):
        te_version, te_graph, te_weight_grads = make_te_proxy_backend(
            e,
            d,
            i,
            counts,
            w1,
            w2,
            grouped_weight=grouped_weight,
        )
        te_output, te_x, te_scores, te_w1, te_w2 = te_graph(x, scores)
        te_output.backward(dout)
        te_errors[layout] = {
            name: compare(f"{layout}:{name}", actual, expected)
            for name, actual, expected in zip(
                ("dX", "dW1", "dW2", "dProb"),
                (
                    te_x.grad,
                    te_weight_grads(te_w1),
                    te_weight_grads(te_w2),
                    te_scores.grad.reshape_as(opus_reference.d_scores),
                ),
                (
                    opus_reference.d_x,
                    opus_reference.d_w1,
                    opus_reference.d_w2,
                    opus_reference.d_scores,
                ),
            )
        }
        te_results[layout] = measure_te_proxy(
            te_graph,
            x,
            scores,
            dout,
            warmup=warmup,
            iterations=iterations,
        )
        del te_graph, te_weight_grads, te_output, te_x, te_scores, te_w1, te_w2
        torch.cuda.empty_cache()

    flops = 12 * routes * d * i
    capacity = metadata.sorted_token_ids.numel()
    nbytes = 2 * (
        2 * routes * d
        + capacity * 3 * i
        + capacity * d
        + 2 * e * 2 * i * d
        + 2 * e * d * i
    )
    timing = {
        "opus_allocating": add_roofline_metrics(
            {"median_us": allocating_median, "p90_us": allocating_p90},
            flops=flops,
            nbytes=nbytes,
        ),
        "opus_preallocated": add_roofline_metrics(
            {"median_us": preallocated_median, "p90_us": preallocated_p90},
            flops=flops,
            nbytes=nbytes,
        ),
        "opus_preallocated_with_status": add_roofline_metrics(
            {"median_us": safe_median, "p90_us": safe_p90},
            flops=flops,
            nbytes=nbytes,
        ),
    }
    result = {
        "case": case_name,
        "distribution": distribution,
        "seed": seed,
        "shape": {
            "routes": routes,
            "D": d,
            "I": i,
            "E_local": e,
            "K_local": 1,
            "counts": counts_host,
        },
        "timing": timing,
        "te_ck_local_expert_proxy": te_results,
        "te_ck_local_expert_proxy_vs_opus_errors": te_errors,
        "versions": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformer_engine": te_version,
            "gpu": torch.cuda.get_device_name(),
        },
        "contract": (
            "backward-only local expert segment after dispatch; metadata and FC1 "
            "forward are excluded; no EP collectives, main_grad, or optimizer"
        ),
    }
    print("RESULT_JSON", json.dumps(result), flush=True)
    return result


def table_row(result):
    row = {
        "case": result["case"],
        "distribution": result["distribution"],
        "seed": result["seed"],
    }
    for name, values in result["timing"].items():
        row[f"{name} us"] = values["median_us"]
        row[f"{name} TFLOPS"] = values["equivalent_tflops"]
        row[f"{name} TB/s"] = values["traffic_tb_s"]
    for layout, values in result["te_ck_local_expert_proxy"].items():
        row[f"te_{layout}_bwd us"] = values["backward_median_us"]
        row[f"te_{layout}_fwd_bwd us"] = values["forward_backward_median_us"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument(
        "--distribution",
        choices=("balanced", "skewed", "empty", "hot"),
        action="append",
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    cases = args.case or list(CASES)
    distributions = args.distribution or ["balanced", "skewed", "empty", "hot"]
    seeds = args.seed or [20261012, 20261013, 20261014]
    rows = []
    for case in cases:
        for distribution in distributions:
            for seed in seeds:
                result = run(
                    case,
                    distribution,
                    seed,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                if result is not None:
                    rows.append(table_row(result))
                torch.cuda.empty_cache()
    print("SUMMARY_MARKDOWN")
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
