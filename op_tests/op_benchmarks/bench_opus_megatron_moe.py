# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MI355X local-expert backward benchmark for Megatron MoE shapes.

This measures the dispatch-after/all-to-all-before local expert segment.  It
does not claim full-network throughput.  Metadata and forward-cache production
are reported separately from the steady Opus backward.
"""

import argparse
import time

import torch

from aiter.ops.opus import moe_backward as opus
from aiter.ops.opus.megatron_moe import (
    build_megatron_local_metadata,
    pad_megatron_local_rows,
)
from op_tests.op_benchmarks.opus_megatron_cases import CASES


def event_us(fn, warmup=10, iterations=50):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--routes", type=int)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--peak-tflops", type=float, default=2500.0)
    parser.add_argument("--hbm-tb-s", type=float, default=8.0)
    parser.add_argument("--tune-route", action="store_true")
    parser.add_argument("--tune-weights", action="store_true")
    args = parser.parse_args()
    shape = CASES[args.case]
    routes = args.routes or shape.routes
    e, d, i = shape.local_experts, shape.model_dim, shape.expert_inter_dim
    device = "cuda"
    torch.manual_seed(20261004)

    counts = torch.full((e,), routes // e, device=device, dtype=torch.int32)
    counts[: routes % e] += 1
    t0 = time.perf_counter()
    md = build_megatron_local_metadata(counts, routes)
    torch.cuda.synchronize()
    metadata_us = (time.perf_counter() - t0) * 1e6
    metadata_median, metadata_p90 = event_us(
        lambda: build_megatron_local_metadata(counts, routes),
        warmup=3,
        iterations=min(args.iterations, 20),
    )
    capacity = md.sorted_token_ids.numel()
    live = md.reverse_sorted.long()

    def rand(*dims):
        return torch.randn(*dims, device=device, dtype=torch.bfloat16)

    x, dout = rand(routes, d), rand(routes, d)
    w1, w2 = rand(e, 2 * i, d), rand(e, d, i)
    w1.mul_(d**-0.5)
    w2.mul_(i**-0.5)
    scores = torch.rand(routes, 1, device=device, dtype=torch.float32)
    compact_z = rand(routes, 2 * i)
    compact_a = rand(routes, i)

    def pad_rows(compact):
        return pad_megatron_local_rows(compact, md)

    z = pad_rows(compact_z)
    saved_x = pad_rows(x)
    prep_components = {
        "metadata": metadata_median,
        "pad_z": event_us(
            lambda: pad_rows(compact_z), warmup=3, iterations=args.iterations
        )[0],
        "pad_a": event_us(
            lambda: pad_rows(compact_a), warmup=3, iterations=args.iterations
        )[0],
        "pad_x": event_us(lambda: pad_rows(x), warmup=3, iterations=args.iterations)[0],
    }

    def plain():
        return opus.opus_moe_backward(dout, x, z, w1, w2, scores, md)

    def saved_x_plain():
        return opus.opus_moe_backward(
            dout,
            x,
            z,
            w1,
            w2,
            scores,
            md,
            saved_x_sorted=saved_x,
            dw1_kernel_id=18,
        )

    down_state = opus.opus_moe_down_backward(
        dout,
        z,
        w2,
        scores,
        md.sorted_token_ids,
        md.sorted_expert_ids,
        md.num_valid_ids,
        md.expert_padded_offsets,
        block_m=32,
    )
    component_fns = {
        "k1_down": lambda: opus.opus_moe_down_backward(
            dout,
            z,
            w2,
            scores,
            md.sorted_token_ids,
            md.sorted_expert_ids,
            md.num_valid_ids,
            md.expert_padded_offsets,
            block_m=32,
        ),
        "k2_k3_route": lambda: opus.opus_moe_route_backward(
            down_state.d_z_sorted,
            w1,
            md.sorted_token_ids,
            md.sorted_expert_ids,
            md.num_valid_ids,
            md.expert_padded_offsets,
            md.reverse_sorted,
            topk=1,
            block_m=32,
        ),
        "k4_dw1": lambda: opus.opus_moe_dw1_backward(
            x,
            down_state.d_z_sorted,
            md.sorted_token_ids,
            md.num_valid_ids,
            md.expert_padded_offsets,
            topk=1,
            block_m=32,
        ),
        "k5_dw2": lambda: opus.opus_moe_dw2_backward(
            dout,
            down_state.a_scaled,
            md.sorted_token_ids,
            md.num_valid_ids,
            md.expert_padded_offsets,
            topk=1,
            block_m=32,
        ),
    }
    components = {
        name: event_us(fn, iterations=args.iterations)[0]
        for name, fn in component_fns.items()
    }
    if args.tune_route:
        expected_dx = component_fns["k2_k3_route"]().d_x
        for route_kid, reduce_kid in (
            (5, 0),
            (7, 0),
            (9, 0),
            (13, 0),
            (11, 1),
            (14, 1),
            (15, 1),
            (16, 1),
            (17, 1),
            (18, 1),
            (19, 1),
        ):

            def candidate(route_kid=route_kid, reduce_kid=reduce_kid):
                return opus.opus_moe_route_backward(
                    down_state.d_z_sorted,
                    w1,
                    md.sorted_token_ids,
                    md.sorted_expert_ids,
                    md.num_valid_ids,
                    md.expert_padded_offsets,
                    md.reverse_sorted,
                    topk=1,
                    block_m=32,
                    route_dx_kernel_id=route_kid,
                    route_reduce_kernel_id=reduce_kid,
                )

            result = candidate()
            torch.testing.assert_close(result.d_x, expected_dx, rtol=0, atol=0)
            candidate_us, candidate_p90 = event_us(
                candidate, warmup=5, iterations=args.iterations
            )
            print(
                f"route_candidate kid={route_kid}/{reduce_kid} "
                f"median_us={candidate_us:.3f} p90_us={candidate_p90:.3f}"
            )
    if args.tune_weights:
        expected_dw1 = component_fns["k4_dw1"]()
        for kid in (5, 8, 9, 11, 13):

            def candidate(kid=kid):
                return opus.opus_moe_dw1_backward(
                    x,
                    down_state.d_z_sorted,
                    md.sorted_token_ids,
                    md.num_valid_ids,
                    md.expert_padded_offsets,
                    topk=1,
                    block_m=32,
                    kernel_id=kid,
                )

            actual = candidate()
            torch.testing.assert_close(actual, expected_dw1, rtol=0, atol=0)
            candidate_us, candidate_p90 = event_us(
                candidate, warmup=5, iterations=args.iterations
            )
            print(
                f"dw1_candidate kid={kid} median_us={candidate_us:.3f} "
                f"p90_us={candidate_p90:.3f}"
            )
        saved_x = torch.zeros(capacity, d, device=device, dtype=torch.bfloat16)
        saved_x[live] = x
        for kid in (14, 15, 18):

            def candidate(kid=kid):
                return opus.opus_moe_dw1_backward(
                    x,
                    down_state.d_z_sorted,
                    md.sorted_token_ids,
                    md.num_valid_ids,
                    md.expert_padded_offsets,
                    topk=1,
                    block_m=32,
                    kernel_id=kid,
                    saved_x_sorted=saved_x,
                )

            actual = candidate()
            torch.testing.assert_close(actual, expected_dw1, rtol=0, atol=0)
            candidate_us, candidate_p90 = event_us(
                candidate, warmup=5, iterations=args.iterations
            )
            print(
                f"dw1_saved_x_candidate kid={kid} median_us={candidate_us:.3f} "
                f"p90_us={candidate_p90:.3f}"
            )
        expected_dw2 = component_fns["k5_dw2"]()
        for kid in (3, 5, 6, 7, 9, 10, 11, 12, 13, 16, 18):

            def candidate(kid=kid):
                return opus.opus_moe_dw2_backward(
                    dout,
                    down_state.a_scaled,
                    md.sorted_token_ids,
                    md.num_valid_ids,
                    md.expert_padded_offsets,
                    topk=1,
                    block_m=32,
                    kernel_id=kid,
                )

            try:
                actual = candidate()
            except RuntimeError as error:
                print(f"dw2_candidate kid={kid} rejected={error}")
                continue
            torch.testing.assert_close(actual, expected_dw2, rtol=0, atol=0)
            candidate_us, candidate_p90 = event_us(
                candidate, warmup=5, iterations=args.iterations
            )
            print(
                f"dw2_candidate kid={kid} median_us={candidate_us:.3f} "
                f"p90_us={candidate_p90:.3f}"
            )
    median, p90 = event_us(plain, iterations=args.iterations)
    saved_x_timing = (
        event_us(saved_x_plain, iterations=args.iterations) if e >= 2 else None
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _held = plain()
    graph_median, graph_p90 = event_us(graph.replay, iterations=args.iterations)
    # The four GEMMs contribute 12*R*D*I FLOPs.  This is a transparent roofline
    # proxy; activation/reduction operations make the actual lower bound higher.
    flops = 12 * routes * d * i
    compute_floor_us = flops / (args.peak_tflops * 1e12) * 1e6
    # Mandatory input/output traffic, assuming each tensor is read/written once.
    bytes_floor = 2 * (
        2 * routes * d
        + capacity * 2 * i
        + e * 2 * i * d
        + e * d * i
        + e * 2 * i * d
        + e * d * i
    )
    bandwidth_floor_us = bytes_floor / (args.hbm_tb_s * 1e12) * 1e6
    floor_us = max(compute_floor_us, bandwidth_floor_us)
    print(f"case={args.case} routes={routes} E_local={e} D={d} I={i}")
    print(
        f"metadata_first_us={metadata_us:.3f} median_us={metadata_median:.3f} "
        f"p90_us={metadata_p90:.3f} capacity={capacity}"
    )
    print(
        "forward_prep_median_us="
        + ",".join(f"{name}:{value:.3f}" for name, value in prep_components.items())
    )
    print(f"opus_backward_median_us={median:.3f} p90_us={p90:.3f}")
    if saved_x_timing is not None:
        saved_x_median, saved_x_p90 = saved_x_timing
        print(
            f"saved_x_backward_median_us={saved_x_median:.3f} "
            f"p90_us={saved_x_p90:.3f} "
            f"with_pad_x_us={saved_x_median + prep_components['pad_x']:.3f}"
        )
    print(
        "component_median_us="
        + ",".join(f"{name}:{value:.3f}" for name, value in components.items())
        + f",sum:{sum(components.values()):.3f}"
    )
    print(f"opus_backward_graph_median_us={graph_median:.3f} p90_us={graph_p90:.3f}")
    print(
        f"equivalent_gemm_tflops_direct={flops / median / 1e6:.3f} "
        f"graph={flops / graph_median / 1e6:.3f}"
    )
    print(
        f"roofline_proxy_us={floor_us:.3f} compute={compute_floor_us:.3f} "
        f"bandwidth={bandwidth_floor_us:.3f} "
        f"direct_gap={median / floor_us:.2f}x "
        f"graph_gap={graph_median / floor_us:.2f}x"
    )


if __name__ == "__main__":
    main()
