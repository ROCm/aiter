# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Identical-input, randomized paired graph benchmark for the decode pipeline."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from op_tests import test_flydsl_pa_mqa_logits_fp4 as helpers
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
    compile_pa_mqa_logits_fp4,
    compute_varctx_schedule,
)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_decode import (
    compile_pa_mqa_logits_fp4_decode,
)


CASES = [
    (2, 1, 512, 64), (3, 1, 1024, 64), (4, 1, 2048, 64),
    (16, 1, 4352, 64), (8, 1, 8192, 64), (1, 1, 32768, 64),
    (1, 1, 65536, 64), (1, 1, 131072, 64), (2, 2, 512, 64),
    (2, 1, 768, 128),
]


def inputs(batch, next_n, ctx, heads, dim, page, seed=42, full=False):
    helpers.setup_seed(seed)
    lengths = [ctx] * batch if full else helpers._make_varctx(batch, ctx, page)
    contexts = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    padded = (ctx + 255) // 256 * 256
    q = torch.randn(batch, next_n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, padded, dim, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(batch * next_n, heads, device="cuda") * 0.1).bfloat16()
    qp, qe = helpers.fp4_quant_e2m1_with_e8m0(q.reshape(-1, dim))
    qp = qp.reshape(batch, next_n, heads, dim // 2)
    qe = qe.reshape(batch, next_n, heads, dim // 32)
    bt = torch.arange(batch * padded // page, device="cuda", dtype=torch.int32).reshape(batch, -1)
    cache, scales, kd, ks = helpers.create_paged_preshuffle_kv_fp4(
        kv, page, batch * padded // page, bt,
    )
    reference = helpers.ref_mqa_logits_mixed(
        qp, qe, kd, ks, w, contexts, next_n=next_n, weight_scale=1.5,
    )
    m = heads // 16
    shuffled = qe.reshape(batch, next_n, m, 16, dim // 128, 4).permute(0, 1, 4, 5, 3, 2).contiguous()
    shuffled = torch.nn.functional.pad(shuffled, (0, (m + 3) // 4 * 4 - m)).contiguous()
    return qp, shuffled, cache, scales, bt, w, contexts, reference, padded, lengths


def capture(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            fn()
    return graph


def measure(graph, repeats):
    graph.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (10 * repeats)


def run_case(case, args):
    b, nn, ctx, h = case
    q, qs, kv, kvs, bt, w, context, reference, padded, lengths = inputs(
        b, nn, ctx, h, args.dim, args.page, full=args.full,
    )
    configurations = {
        "old256": (256, 4, 0, 0, 16, False),
        "old64": (64, 1, 0, 0, 16, False),
        "lds64w1h1": (64, 1, 1, 2, 32, False),
        "lds128w2h1": (128, 2, 1, 2, 32, False),
        "lds128w4h2": (128, 4, 2, 2, 32, False),
        "lds64w2h2": (64, 2, 2, 2, 32, False),
        "lds64w2h2sw": (64, 2, 2, 2, 32, True),
        "lds256w4h2": (256, 4, 2, 2, 32, False),
        "lds256w4h1": (256, 4, 1, 2, 32, False),
        "lds64w4h4": (64, 4, 4, 2, 32, False),
        "lds64w4h4sw": (64, 4, 4, 2, 32, True),
        "lds64w4h4m16": (64, 4, 4, 2, 16, False),
        "lds64w2h2m16": (64, 2, 2, 2, 16, False),
        "lds128w4h2m16": (128, 4, 2, 2, 16, False),
        "lds64w2h2s1": (64, 2, 2, 1, 32, False),
        "lds128w2h2": (128, 2, 2, 2, 32, False),
    }
    functions, outputs, schedules, resources = {}, {}, {}, {}
    for name in args.variants.split(","):
        direct = name.endswith("d")
        block, waves, hw, stages, mfma_m, scalar_weights = configurations[
            name[:-1] if direct else name
        ]
        if block < args.page and not hw:
            continue
        direct_chunks = 0
        if direct:
            direct_chunks = (padded + block - 1) // block
            if args.new_ctas:
                direct_chunks = min(direct_chunks, max(1, args.new_ctas // (b * nn)))
            schedule, total = context, b * nn * direct_chunks
        else:
            target = args.new_ctas if hw and args.new_ctas else args.ctas
            _, schedule, total = compute_varctx_schedule(
                context, block, target, padded, next_n=nn,
            )
        schedules[name] = schedule
        out = torch.full((b * nn, padded), float("-inf"), device="cuda", dtype=torch.float32)
        outputs[name] = out
        if hw:
            launcher, _ = compile_pa_mqa_logits_fp4_decode(
                heads=h, head_dim=args.dim, next_n=nn, kv_block_size=args.page,
                kv_page_stride=kv.stride(0), kv_scale_page_stride=kvs.stride(0),
                block_table_stride=bt.stride(0), block_k=block, num_warps=waves,
                head_waves=hw, stages=stages, mfma_m=mfma_m,
                scalar_weights=scalar_weights,
                direct_chunks=direct_chunks, max_seq_len=padded,
            )
        else:
            launcher, _ = compile_pa_mqa_logits_fp4(
                block_k=block, kv_block_size=args.page, max_blocks_per_seq=bt.shape[1],
                num_warps=waves, next_n=nn, heads=h, head_dim=args.dim,
            )
        def launch(launcher=launcher, out=out, schedule=schedule, total=total):
            launcher(out, q, qs, kv, kvs, bt, w, schedule,
                     out.stride(0), 1.5, total, torch.cuda.current_stream())
        print(f"compile {case} {name}", flush=True)
        launch()
        torch.cuda.synchronize()
        assert torch.equal(torch.isneginf(out), torch.isneginf(reference)), name
        mask = torch.isfinite(reference)
        actual, expected = out[mask], reference[mask]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4)
        cosine = torch.nn.functional.cosine_similarity(actual.double(), expected.double(), dim=0).item()
        print(f"correct {name} cosine={cosine:.12f} max_abs={(actual-expected).abs().max().item():.8g}", flush=True)
        resources[name] = {
            "block_k": block,
            "num_warps": waves,
            "head_waves": hw,
            "scalar_weights": scalar_weights,
            "grid": total,
        }
        functions[name] = launch
    graphs = {name: capture(fn, args.repeats) for name, fn in functions.items()}
    samples = {name: [] for name in graphs}
    rng = random.Random(20260921)
    for _ in range(args.rounds):
        names = list(graphs)
        rng.shuffle(names)
        for name in names:
            samples[name].append(measure(graphs[name], args.repeats))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {
        "shape": case, "head_dim": args.dim, "page": args.page, "context_lens": lengths,
        "medians_us": medians, "samples_us": samples, "configurations": resources,
        "speedups": {name: medians["old256"] / us for name, us in medians.items()},
    }
    print("RESULT " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--ctas", type=int, default=None)
    parser.add_argument("--new-ctas", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--variants", default="old256,old64,lds128w4h2,lds64w2h2,lds256w4h2,lds256w4h1")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name()} args={args}", flush=True)
    cases = CASES if args.case < 0 else [CASES[args.case]]
    results = [run_case(case, args) for case in cases]
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
