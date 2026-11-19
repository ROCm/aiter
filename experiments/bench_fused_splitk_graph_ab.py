"""Paired CUDA-graph A/B for the gfx1250 MXFP8 fused split-K path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
)
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight


def _error_ratio(output: torch.Tensor, reference: torch.Tensor) -> float:
    return 1.0 - float(
        torch.isclose(
            output.float(), reference.float(), atol=0.01, rtol=0.01
        ).float().mean()
    )


def _dequant_reference(data: dict[str, torch.Tensor]) -> torch.Tensor:
    m, k = data["x"].shape
    n = data["w"].shape[0]
    a = data["x"].float() * torch.exp2(
        data["xs_u8"].float() - 127
    ).repeat_interleave(32, 1)
    b = data["w"].float() * torch.exp2(
        data["ws_u8"].float() - 127
    ).repeat_interleave(32, 1)
    return F.linear(a.view(m, k), b.view(n, k))


def _make_data(m: int, n: int, k: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    x = (torch.randn((m, k), device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn((n, k), device="cuda") * 2).to(dtypes.fp8)
    xs_u8 = torch.randint(124, 131, (m, k // 32), dtype=torch.uint8, device="cuda")
    ws_u8 = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
    return {
        "x": x,
        "w": w,
        "xs_u8": xs_u8,
        "ws_u8": ws_u8,
        "wq": shuffle_weight(w, layout=(16, 16)),
        "ws": shuffle_mxfp8fp4_scale(ws_u8.view(dtypes.fp8_e8m0)),
    }


def _runner(data: dict[str, torch.Tensor], name: str):
    out = torch.empty(
        data["x"].shape[0],
        data["w"].shape[0],
        dtype=torch.bfloat16,
        device="cuda",
    )

    def run():
        return run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
            data["x"],
            data["wq"],
            data["xs_u8"].view(dtypes.fp8_e8m0),
            data["ws"],
            out,
            name,
        )

    return run, out


def _sample(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replays


def _percentile(values: list[float], q: float) -> float:
    return sorted(values)[math.ceil(q * len(values)) - 1]


def _measure_case(
    m: int,
    n: int,
    k: int,
    names: list[str],
    *,
    rounds: int,
    replays: int,
) -> dict:
    data = _make_data(m, n, k, 113000 + m + n + k)
    reference = _dequant_reference(data)
    runs, outputs, graphs = {}, {}, {}
    eager_errors = {}
    for name in names:
        run, out = _runner(data, name)
        runs[name], outputs[name] = run, out
        eager_errors[name] = _error_ratio(run(), reference)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs[name] = graph
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for round_index in range(rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        if round_index % 2:
            order.reverse()
        for name in order:
            samples[name].append(_sample(graphs[name], replays))

    # Replays must be deterministic, then must consume changed graph inputs.
    deterministic, changed_errors = {}, {}
    for name in names:
        graphs[name].replay()
        torch.cuda.synchronize()
        first = outputs[name].clone()
        graphs[name].replay()
        torch.cuda.synchronize()
        deterministic[name] = bool(torch.equal(first, outputs[name]))

    torch.manual_seed(991000 + m + n + k)
    data["x"].copy_((torch.randn_like(data["x"].float()) * 2).to(dtypes.fp8))
    data["xs_u8"].random_(124, 131)
    changed_reference = _dequant_reference(data)
    for name in names:
        outputs[name].fill_(float("nan"))
        graphs[name].replay()
        torch.cuda.synchronize()
        changed_errors[name] = _error_ratio(outputs[name], changed_reference)

    candidates = []
    for name in names:
        values = sorted(samples[name])
        candidates.append(
            {
                "kernelName": name,
                "median_us": values[len(values) // 2],
                "min_us": values[0],
                "p90_us": _percentile(values, 0.9),
                "samples_us": values,
                "eager_err_ratio": eager_errors[name],
                "changed_input_err_ratio": changed_errors[name],
                "deterministic_replay": deterministic[name],
            }
        )
    candidates.sort(key=lambda row: row["median_us"])
    baseline = next(row for row in candidates if row["kernelName"] == names[0])
    return {
        "M": m,
        "N": n,
        "K": k,
        "baseline": baseline,
        "winner": candidates[0],
        "speedup": baseline["median_us"] / candidates[0]["median_us"],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        required=True,
        help="semicolon-separated M,N,K triples",
    )
    parser.add_argument("--names", required=True, help="semicolon-separated names")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = [tuple(map(int, item.split(","))) for item in args.cases.split(";")]
    names = args.names.split(";")
    results = []
    for case in cases:
        result = _measure_case(
            *case, names, rounds=args.rounds, replays=args.replays
        )
        results.append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("M", "N", "K", "speedup", "winner")
                }
            ),
            flush=True,
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
