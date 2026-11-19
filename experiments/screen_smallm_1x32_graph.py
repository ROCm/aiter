"""Screen the dedicated gfx1250 16x32 MXFP8 kernel under graph replay."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from aiter import dtypes
from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    MX32_SMALLM_1X32_NAME_PREFIX,
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
)
from aiter.ops.gemm_op_a8w8 import _mxfp8_32_fallback_kernel_name
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight

DEFAULT_FAMILIES = (
    (1536, 6144),
    (2304, 6144),
    (2560, 6144),
    (6144, 768),
    (6144, 2048),
    (6144, 3072),
    (6144, 6144),
)
DEFAULT_MS = (64, 128, 256, 512)


def error_ratio(output, reference):
    return 1.0 - float(
        torch.isclose(output.float(), reference.float(), atol=0.01, rtol=0.01)
        .float()
        .mean()
    )


def reference(data):
    x = data["x"].float() * torch.exp2(
        data["x_scale_u8"].float() - 127
    ).repeat_interleave(32, 1)
    weight = data["weight"].float() * torch.exp2(
        data["w_scale_u8"].float() - 127
    ).repeat_interleave(32, 1)
    return (x @ weight.T).to(torch.bfloat16)


def make_data(m, n, k):
    torch.manual_seed(113000 + m + n + k)
    x = (torch.randn((m, k), device="cuda") * 2).to(dtypes.fp8)
    weight = (torch.randn((n, k), device="cuda") * 2).to(dtypes.fp8)
    x_scale_u8 = torch.randint(124, 131, (m, k // 32), dtype=torch.uint8, device="cuda")
    w_scale_u8 = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
    return {
        "x": x,
        "weight": weight,
        "x_scale_u8": x_scale_u8,
        "w_scale_u8": w_scale_u8,
        "x_scale": x_scale_u8.view(dtypes.fp8_e8m0),
        "w_scale": shuffle_mxfp8fp4_scale(w_scale_u8.view(dtypes.fp8_e8m0)),
        "shuffled_weight": shuffle_weight(weight, layout=(16, 16)),
        "out": torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    }


def runner(data, kernel_name):
    if kernel_name.startswith("flydsl_bmm_mxfp8_32_wmma"):

        def run():
            return run_bmm_a8w8_mxfp8(
                data["x"].view(data["x"].shape[0], 1, data["x"].shape[1]),
                data["shuffled_weight"].view(
                    1,
                    data["shuffled_weight"].shape[0],
                    data["shuffled_weight"].shape[1],
                ),
                data["x_scale"].view(
                    data["x_scale"].shape[0], 1, data["x_scale"].shape[1]
                ),
                data["w_scale"].view(
                    1, data["w_scale"].shape[0], data["w_scale"].shape[1]
                ),
                data["out"].view(data["out"].shape[0], 1, data["out"].shape[1]),
                kernel_name,
            )

        return run

    def run():
        return run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
            data["x"],
            data["shuffled_weight"],
            data["x_scale"],
            data["w_scale"],
            data["out"],
            kernel_name,
        )

    return run


def config_rows(path):
    if path is None:
        return {}
    with open(path, newline="") as handle:
        return {
            (int(row["M"]), int(row["N"]), int(row["K"])): row
            for row in csv.DictReader(handle)
            if row["gfx"] == "gfx1250" and row["w_scale_block"] == "1x32"
        }


def parse_pairs(value):
    return tuple(tuple(map(int, item.split(","))) for item in value.split(";"))


def parse_ints(value):
    return tuple(map(int, value.split(",")))


def sample_graph(graph, replays):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replays


def percentile(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument(
        "--families",
        default=";".join(f"{n},{k}" for n, k in DEFAULT_FAMILIES),
    )
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument("--coarse-rounds", type=int, default=3)
    parser.add_argument("--coarse-replays", type=int, default=30)
    parser.add_argument("--stable-rounds", type=int, default=11)
    parser.add_argument("--stable-replays", type=int, default=100)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    families = parse_pairs(args.families)
    ms = parse_ints(args.ms)
    rows = config_rows(args.config)
    results = []
    for n, k in families:
        for m in ms:
            data = make_data(m, n, k)
            baseline_name = (
                rows[(m, n, k)]["kernelName"]
                if rows
                else _mxfp8_32_fallback_kernel_name(m, n, k)
            )
            names = [baseline_name]
            names.extend(
                f"{MX32_SMALLM_1X32_NAME_PREFIX}_w{waves}_sk{split_k}"
                for waves in (1, 2)
                for split_k in (1, 2, 4, 8)
                if k % (128 * split_k) == 0
            )
            names = list(dict.fromkeys(names))
            eager_reference = reference(data)
            graphs = {}
            errors = {}
            failures = {}
            for name in names:
                try:
                    run = runner(data, name)
                    run()
                    torch.cuda.synchronize()
                    eager_err = error_ratio(data["out"], eager_reference)
                    if eager_err > 0.05:
                        failures[name] = f"eager errRatio {eager_err:.6f}"
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    graphs[name] = graph
                    errors[name] = {"eager": eager_err}
                except Exception as exc:  # noqa: BLE001 - a screen records failures
                    failures[name] = f"{type(exc).__name__}: {exc}"

            # A and A-scale keep their captured addresses but get fresh values.
            data["x"].copy_(
                (torch.randn(data["x"].shape, device="cuda") * 1.5).to(dtypes.fp8)
            )
            data["x_scale_u8"].copy_(
                torch.randint(
                    123,
                    130,
                    data["x_scale_u8"].shape,
                    dtype=torch.uint8,
                    device="cuda",
                )
            )
            changed_reference = reference(data)
            for name, graph in list(graphs.items()):
                data["out"].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                changed_err = error_ratio(data["out"], changed_reference)
                errors[name]["changed_graph"] = changed_err
                if changed_err > 0.05:
                    failures[name] = f"changed graph errRatio {changed_err:.6f}"
                    del graphs[name]

            if baseline_name not in graphs:
                raise RuntimeError(
                    f"baseline failed for {(m, n, k)}: "
                    f"{failures.get(baseline_name, 'unknown')}"
                )
            for graph in graphs.values():
                for _ in range(10):
                    graph.replay()
            torch.cuda.synchronize()

            coarse = {}
            for name, graph in graphs.items():
                coarse[name] = [
                    sample_graph(graph, args.coarse_replays)
                    for _ in range(args.coarse_rounds)
                ]
            ranked = sorted(
                graphs,
                key=lambda name: sorted(coarse[name])[len(coarse[name]) // 2],
            )
            finalists = list(dict.fromkeys([baseline_name, *ranked[: args.finalists]]))
            samples = {name: [] for name in finalists}
            for round_index in range(args.stable_rounds):
                order = (
                    finalists[round_index % len(finalists) :]
                    + finalists[: round_index % len(finalists)]
                )
                if round_index % 2:
                    order.reverse()
                for name in order:
                    samples[name].append(
                        sample_graph(graphs[name], args.stable_replays)
                    )

            measured = []
            for name, values in samples.items():
                ordered = sorted(values)
                measured.append(
                    {
                        "kernelName": name,
                        "median_us": ordered[len(ordered) // 2],
                        "min_us": ordered[0],
                        "p90_us": percentile(ordered, 0.9),
                        "samples_us": values,
                        "err_ratio": errors[name],
                    }
                )
            measured.sort(key=lambda row: row["median_us"])
            baseline = next(
                row for row in measured if row["kernelName"] == baseline_name
            )
            winner = measured[0]
            speedup = baseline["median_us"] / winner["median_us"]
            retained = winner["kernelName"] != baseline_name and speedup >= 1.02
            result = {
                "M": m,
                "N": n,
                "K": k,
                "baseline": baseline,
                "winner": winner,
                "speedup": speedup,
                "retained": retained,
                "candidate_count": len(names),
                "finalists": measured,
                "coarse_us": coarse,
                "errors": errors,
                "failures": failures,
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "M",
                            "N",
                            "K",
                            "speedup",
                            "retained",
                            "winner",
                        )
                    }
                ),
                flush=True,
            )
            del data, eager_reference, changed_reference, graphs
            torch.cuda.empty_cache()

    payload = {
        "baseline": args.config or "current fallback heuristic",
        "families": families,
        "ms": ms,
        "timing": {
            "mode": "CUDA Graph replay",
            "coarse_rounds": args.coarse_rounds,
            "coarse_replays": args.coarse_replays,
            "stable_rounds": args.stable_rounds,
            "stable_replays": args.stable_replays,
            "retention_threshold": 1.02,
        },
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
