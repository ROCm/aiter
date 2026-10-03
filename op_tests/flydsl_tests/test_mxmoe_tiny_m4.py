# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 MarloweAI Contributors
"""Native/candidate GPU fixtures with changing routes, zero and graph reuse.

Every packed expert row has a constant nibble/scale, so native weight shuffles
leave these independently constructed values invariant. These fixtures are
additional state/layout checks; captured real-weight qualification is mandatory.
"""

import argparse


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time synthetic native/candidate graph replays after correctness checks",
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")
    return args


def benchmark(native, candidate, args):
    """Synthetic full-operator graph timing, not captured-model table reproduction."""
    import random
    import statistics
    import torch

    graphs, outputs = {}, {}
    for name, call in (("native", native), ("candidate", candidate)):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = call()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        outputs["candidate"], outputs["native"], atol=0.02, rtol=0.02
    )
    for _ in range(args.warmups):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in graphs
    }
    samples = {name: [] for name in graphs}
    rng = random.Random(355052)
    for _ in range(args.samples):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            start, end = events[name]
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    before, after = (
        statistics.median(samples[name]) for name in ("native", "candidate")
    )
    print(
        f"SYNTHETIC graph replay: native={before:.3f} us candidate={after:.3f} us "
        f"saved={before-after:.3f} us reduction={100*(before-after)/before:.3f}%"
    )
    print(
        f"{args.samples} randomized paired samples; {args.warmups} warmups; "
        "not the captured 480-case performance table"
    )


def main():
    args = arguments()
    import torch

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.kernels.mxmoe_tiny_m4 import make_operator

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        print("SKIP: exact M4 fixtures require gfx950")
        return
    device = torch.device("cuda")
    w1 = torch.full((257, 512, 3072), 0x22, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    w2_bytes = torch.empty((257, 6144, 128), dtype=torch.uint8, device=device)
    for expert in range(257):
        nibble = (2, 4, 6)[expert % 3]
        w2_bytes[expert].fill_(nibble | (nibble << 4))
    w2 = w2_bytes.view(torch.float4_e2m1fn_x2)
    w1.is_shuffled = w2.is_shuffled = True
    weights = {
        "w1": w1,
        "w2": w2,
        "w1_scale": torch.full((257, 512, 192), 120, dtype=torch.uint8, device=device),
        "w2_scale": torch.full((257, 6144, 8), 120, dtype=torch.uint8, device=device),
    }
    frozen = {key: value.view(torch.uint8).clone() for key, value in weights.items()}
    first = make_operator(weights=weights, rows=4)
    second = make_operator(weights=weights, rows=4)
    x = torch.full((4, 6144), 1 / 128, dtype=torch.bfloat16, device=device)
    route_weights = torch.full((4, 9), 1 / 16, dtype=torch.float32, device=device)
    route_weights[:, 8] = 1
    ids = torch.cat(
        (
            torch.arange(8, dtype=torch.int32, device=device).expand(4, 8),
            torch.full((4, 1), 256, dtype=torch.int32, device=device),
        ),
        1,
    ).contiguous()
    initial_x, initial_ids = x.clone(), ids.clone()

    def native():
        return fused_moe(
            x,
            w1,
            w2,
            route_weights,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=weights["w1_scale"],
            w2_scale=weights["w2_scale"],
        )

    def check(operator):
        expected, actual = native().clone(), operator.run(x, ids, route_weights)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        if not bool(x.count_nonzero()):
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        assert operator.state_check()["passed"]

    # Concentrated repeated cross-token routes, then completely changed IDs.
    check(first)
    ids[:, :8].copy_(torch.arange(32, dtype=torch.int32, device=device).reshape(4, 8))
    check(first)
    check(second)
    x.zero_()
    check(first)
    check(second)
    x.copy_(initial_x)
    ids.copy_(initial_ids)
    check(first)
    # All factories compile/warm up before capture. Graphs own separate scratch.
    torch.cuda.synchronize()
    for operator in (first, second):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = operator.run(x, ids, route_weights)
        for value in (0.0, 1 / 128, 1 / 64, 0.0, 1 / 128):
            x.fill_(value)
            ids[:, :8].copy_((initial_ids[:, :8] + int(value * 128) * 8) % 256)
            graph.replay()
            torch.testing.assert_close(result, native(), atol=0.02, rtol=0.02)
    for key, before in frozen.items():
        torch.testing.assert_close(
            weights[key].view(torch.uint8), before, atol=0, rtol=0
        )
    print("PASS: native M4 route/reset/two-workspace graph fixtures")
    if args.benchmark:
        x.copy_(initial_x)
        ids.copy_(initial_ids)
        benchmark(native, lambda: first.run(x, ids, route_weights), args)


if __name__ == "__main__":
    main()
