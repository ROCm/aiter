import argparse
import hashlib
import json
import random
import statistics
from itertools import pairwise
from pathlib import Path

import torch

import aiter
from aiter.ops.triton.attention.pa_ps_metadata import plan_pa_ps_metadata


def make_lengths(batch, pattern):
    if pattern == "mixed":
        return [200003] + [257] * (batch - 1)
    lengths = [1921 + sequence % 128 for sequence in range(batch)]
    if pattern in ("prefix", "shuffled"):
        for sequence in range(min(batch, 1024)):
            lengths[sequence] = 1 + sequence % 2048
    if pattern == "shuffled":
        random.Random(42).shuffle(lengths)
    return lengths


def summarize(metadata, lengths, page_offsets, num_heads_k, gqa, block_size, overhead):
    _, work_indptr, work_info, reduce_indptr, _, _ = metadata
    work_ptr = work_indptr.cpu().tolist()
    records = work_info[:work_ptr[-1]].cpu().tolist()
    reduce_ptr = reduce_indptr.cpu().tolist()
    assert work_ptr[0] == reduce_ptr[0] == 0
    assert work_ptr == sorted(work_ptr) and reduce_ptr == sorted(reduce_ptr)
    assert work_ptr[-1] <= work_info.shape[0]
    assert reduce_ptr[-1] <= metadata[-1].numel()
    coverage = [[[] for _ in lengths] for _ in range(num_heads_k)]
    work_counts, tile_counts, costs = [], [], []
    for first, last in pairwise(work_ptr):
        tiles = 0
        for sequence, _, query_start, query_end, begin, end, offset, head_range in records[first:last]:
            head = (head_range & 0xFFFF) // gqa
            assert head_range == ((head + 1) * gqa << 16) | (head * gqa)
            assert 0 <= head < num_heads_k and 0 <= sequence < len(lengths)
            assert query_start == sequence and query_end == sequence + 1 and offset == 0
            assert page_offsets[sequence] <= begin < end <= page_offsets[sequence + 1]
            coverage[head][sequence].append((begin, end))
            tiles += (min((end - begin) * block_size, lengths[sequence] - (begin - page_offsets[sequence]) * block_size) + 255) // 256
        work_counts.append(last - first)
        tile_counts.append(tiles)
        costs.append(tiles + (last - first) * overhead)
    for head in range(num_heads_k):
        for sequence, intervals in enumerate(coverage[head]):
            position = page_offsets[sequence]
            for begin, end in sorted(intervals):
                assert begin == position
                position = end
            assert position == page_offsets[sequence + 1]
    return {
        "total_work": len(records),
        "reduce_groups": sum(last > first for first, last in pairwise(reduce_ptr)),
        "partial_count": reduce_ptr[-1],
        "max_work_per_tg": max(work_counts),
        "max_tiles_per_tg": max(tile_counts),
        "max_estimated_cost": max(costs),
        "max_mean_cost_ratio": max(costs) / statistics.mean(costs),
        "work_per_tg": work_counts,
        "tiles_per_tg": tile_counts,
        "estimated_cost_per_tg": costs,
    }


def measure(calls, rounds, repeat):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, samples = {}, {name: [] for name in calls}
    with torch.cuda.stream(stream):
        for name, call in calls.items():
            call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(repeat):
                    call()
            graphs[name] = graph
        for round_index in range(rounds):
            order = list(graphs)
            if round_index % 2:
                order.reverse()
            for name in order:
                graph = graphs[name]
                graph.replay()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / repeat)
        for graph in graphs.values():
            graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
    return {
        name: {"median_us": statistics.median(values), "min_us": min(values),
               "max_us": max(values), "samples_us": values}
        for name, values in samples.items()
    }


def run_case(batch, pattern, args):
    lengths = make_lengths(batch, pattern)
    context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=context.device)
    kv_indptr = torch.zeros_like(qo_indptr)
    kv_indptr[1:] = ((context + args.block_size - 1) // args.block_size).cumsum(0)
    legacy = [
        torch.empty(shape, dtype=dtype, device=context.device)
        for shape, dtype in aiter.get_pa_metadata_info_v1(batch, args.kv_heads)
    ]
    plan = plan_pa_ps_metadata(
        qo_indptr, kv_indptr, context, args.gqa, args.kv_heads,
        max_qlen=1, block_size=args.block_size,
    )

    def legacy_call():
        aiter.get_pa_metadata_v1(
            qo_indptr, kv_indptr, context, args.gqa, args.kv_heads, False, *legacy,
            kv_granularity=args.block_size, block_size=args.block_size,
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True, max_split_per_batch=-1,
        )

    def tile_call():
        plan_pa_ps_metadata(
            qo_indptr, kv_indptr, context, args.gqa, args.kv_heads,
            max_qlen=1, block_size=args.block_size, plan=plan,
        )

    legacy_call()
    tile = [getattr(plan, name) for name in (
        "work_metadata_ptrs", "work_indptr", "work_info", "reduce_indptr",
        "reduce_final_map", "reduce_partial_map",
    )]
    offsets = kv_indptr.cpu().tolist()
    summaries = {
        name: summarize(metadata, lengths, offsets, args.kv_heads, args.gqa,
                        args.block_size, plan.work_overhead)
        for name, metadata in (("legacy", legacy), ("tile", tile))
    }
    timing = measure({"legacy": legacy_call, "tile": tile_call}, args.rounds, args.repeat)
    return {"batch": batch, "pattern": pattern, "summary": summaries, "timing": timing}


def main():
    parser = argparse.ArgumentParser(description="Compare GPU PA_PS metadata planners without changing ASM.")
    parser.add_argument("--batch", nargs="+", type=int, default=[8, 200, 32768])
    parser.add_argument("--pattern", nargs="+", choices=["uniform", "mixed", "prefix", "shuffled"],
                        default=["uniform", "mixed", "prefix", "shuffled"])
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--gqa", type=int, choices=[8, 16], default=16)
    parser.add_argument("--block-size", type=int, choices=[16], default=16)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rounds < 1 or args.repeat < 1:
        raise ValueError("use a new output path and positive rounds/repeat")
    root = Path(aiter.__file__).resolve().parents[1]
    protected = [root / "aiter/ops/attention.py", root / "csrc/py_itfs_cu/asm_pa.cu",
                 root / "csrc/py_itfs_cu/asm_pa_ps_reduce.cu", *sorted((root / "hsa/gfx950/pa").glob("*.co"))]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    result = {
        "status": "running", "scope": "metadata_only", "args": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "cu_count": torch.cuda.get_device_properties(context_device := torch.cuda.current_device()).multi_processor_count,
        "device_ordinal": context_device, "protected_sha256": hashes, "results": [],
    }
    try:
        for batch in args.batch:
            for pattern in args.pattern:
                row = run_case(batch, pattern, args)
                result["results"].append(row)
                aiter.logger.info("B=%d pattern=%s legacy=%.3f us tile=%.3f us",
                                  batch, pattern, row["timing"]["legacy"]["median_us"],
                                  row["timing"]["tile"]["median_us"])
        assert hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        result["protected_unchanged"] = True
        result["status"] = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output_file:
            json.dump(result, output_file, indent=2)


if __name__ == "__main__":
    main()