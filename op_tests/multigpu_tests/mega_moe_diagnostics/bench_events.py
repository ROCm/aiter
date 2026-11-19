"""Time native Mega-MoE graph kernels with explicit HIP event nodes.

The first timing pass uses the unmodified graph. The second inserts HIP event
nodes around GEMM nodes, preserving their incoming and outgoing dependencies.
Unlike captured torch Event.record(), explicit nodes update event timestamps.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path.cwd() / "op_tests/multigpu_tests"))
import bench_mega_moe as benchmark

_HIP = ct.CDLL("/opt/rocm/lib/libamdhip64.so")
_PTR = ct.c_void_p
_ORIGINAL_CAPTURE = benchmark.DeviceMoEPipeline.capture
_ORIGINAL_BENCH = benchmark.DeviceMoEPipeline.bench


class Dim3(ct.Structure):
    _fields_ = [("x", ct.c_uint), ("y", ct.c_uint), ("z", ct.c_uint)]


class KernelParams(ct.Structure):
    _fields_ = [
        ("block", Dim3),
        ("extra", _PTR),
        ("func", _PTR),
        ("grid", Dim3),
        ("args", _PTR),
        ("smem", ct.c_uint),
    ]


def hip_call(name, types, *args):
    """Invoke a HIP API and fail on every non-success result."""
    function = getattr(_HIP, name)
    function.argtypes = types
    code = function(*args)
    if code:
        raise RuntimeError(f"{name}: HIP error {code}")


def node_list(api, handle):
    """Return graph nodes or a node's dependency list."""
    count = ct.c_size_t()
    hip_call(api, [_PTR, _PTR, _PTR], handle, None, ct.byref(count))
    nodes = (_PTR * count.value)()
    hip_call(api, [_PTR, _PTR, _PTR], handle, nodes, ct.byref(count))
    return list(nodes)


def edges(api, graph, sources, targets):
    """Change graph edges in one batch."""
    if sources:
        hip_call(
            api,
            [_PTR, _PTR, _PTR, ct.c_size_t],
            graph,
            (_PTR * len(sources))(*sources),
            (_PTR * len(targets))(*targets),
            len(sources),
        )


def instrument(graph, records):
    """Add explicit timing nodes, retaining original dependency ordering."""
    _HIP.hipKernelNameRef.argtypes = [_PTR]
    _HIP.hipKernelNameRef.restype = ct.c_char_p
    _HIP.hipKernelNameRefByPtr.argtypes = [_PTR, _PTR]
    _HIP.hipKernelNameRefByPtr.restype = ct.c_char_p
    for node in node_list("hipGraphGetNodes", graph):
        kind = ct.c_int()
        hip_call("hipGraphNodeGetType", [_PTR, _PTR], node, ct.byref(kind))
        if kind.value == 4:
            child = _PTR()
            hip_call(
                "hipGraphChildGraphNodeGetGraph", [_PTR, _PTR], node, ct.byref(child)
            )
            instrument(child, records)
        if kind.value != 0:
            continue
        params = KernelParams()
        hip_call("hipGraphKernelNodeGetParams", [_PTR, _PTR], node, ct.byref(params))
        name_bytes = _HIP.hipKernelNameRefByPtr(params.func, None)
        if name_bytes is None:
            name_bytes = _HIP.hipKernelNameRef(params.func)
        if name_bytes is None:
            raise RuntimeError("Missing HIP kernel symbol")
        name = name_bytes.decode()
        if "K5120" in name:
            stage = "gemm1"
        elif "K2304" in name:
            stage = "gemm2"
        elif "s1f_pull_ready" in name:
            stage = "ready"
        elif "tdm_compact_plan" in name:
            stage = "plan"
        else:
            continue
        parents = node_list("hipGraphNodeGetDependencies", node)
        children = node_list("hipGraphNodeGetDependentNodes", node)
        start, end, start_node, end_node = _PTR(), _PTR(), _PTR(), _PTR()
        hip_call("hipEventCreate", [_PTR], ct.byref(start))
        hip_call("hipEventCreate", [_PTR], ct.byref(end))
        hip_call(
            "hipGraphAddEventRecordNode",
            [_PTR, _PTR, _PTR, ct.c_size_t, _PTR],
            ct.byref(start_node),
            graph,
            (_PTR * len(parents))(*parents),
            len(parents),
            start,
        )
        edges("hipGraphRemoveDependencies", graph, parents, [node] * len(parents))
        edges("hipGraphAddDependencies", graph, [start_node.value], [node])
        hip_call(
            "hipGraphAddEventRecordNode",
            [_PTR, _PTR, _PTR, ct.c_size_t, _PTR],
            ct.byref(end_node),
            graph,
            (_PTR * 1)(node),
            1,
            end,
        )
        edges("hipGraphRemoveDependencies", graph, [node] * len(children), children)
        edges(
            "hipGraphAddDependencies", graph, [end_node.value] * len(children), children
        )
        records.append(
            {
                "stage": stage,
                "symbol": name,
                "grid": [params.grid.x, params.grid.y, params.grid.z],
                "block": [params.block.x, params.block.y, params.block.z],
                "dynamic_shared_bytes": params.smem,
                "start": start,
                "end": end,
                "samples_us": [],
            }
        )


def capture(self, x0):
    """Retain the original graph so it can be instrumented after native timing."""
    original_constructor = benchmark.torch.cuda.CUDAGraph
    benchmark.torch.cuda.CUDAGraph = lambda: original_constructor(keep_graph=True)
    try:
        _ORIGINAL_CAPTURE(self, x0)
    finally:
        benchmark.torch.cuda.CUDAGraph = original_constructor


def bench(self, warmup=5, iters=32):
    """Return unmodified E2E timing and write instrumented GEMM intervals."""
    native = _ORIGINAL_BENCH(self, warmup=warmup, iters=iters)
    output = Path(os.environ["PROFILE_OUT"])
    with (output / f"native_rank{self.dist_ctx.rank}.json").open("w") as handle:
        json.dump({"native_e2e_us": native}, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Rank {self.dist_ctx.rank} native E2E: {native}", flush=True)
    if os.environ.get("PROFILE_STAGE") == "native":
        return native
    records = []
    instrument(_PTR(self.graph.raw_cuda_graph()), records)
    counts = {
        stage: sum(r["stage"] == stage for r in records)
        for stage in ("gemm1", "gemm2", "plan", "ready")
    }
    expected_ready = self.n_layers if os.environ["BENCH_VARIANT"] == "current" else 0
    assert all(counts[s] == self.n_layers for s in ("gemm1", "gemm2", "plan")), counts
    assert counts["ready"] == expected_ready, counts
    expected_prefix = "s1f_pull_" if expected_ready else "s1f_stage1_"
    assert all(
        r["symbol"].startswith(expected_prefix)
        for r in records
        if r["stage"] == "gemm1"
    ), "S1F fallback or wrong implementation"
    self.graph.instantiate()
    for _ in range(warmup):
        self.graph.replay()
    benchmark.torch.cuda.synchronize()
    self.comm.barrier()
    for _ in range(iters):
        self.graph.replay()
        benchmark.torch.cuda.synchronize()
        for row in records:
            duration = ct.c_float()
            hip_call(
                "hipEventElapsedTime",
                [_PTR, _PTR, _PTR],
                ct.byref(duration),
                row["start"],
                row["end"],
            )
            row["samples_us"].append(duration.value * 1000)
    self.comm.barrier()
    for row in records:
        del row["start"], row["end"]
    details = {
        "method": "explicit HIP graph event nodes; includes event boundary overhead",
        "rank": self.dist_ctx.rank,
        "tokens_per_rank": self.ct,
        "benchmark_source": benchmark.__file__,
        "variant": os.environ["BENCH_VARIANT"],
        "native_e2e_us": native,
        "kernels": records,
    }
    output = Path(os.environ["PROFILE_OUT"])
    with (output / f"events_rank{self.dist_ctx.rank}.json").open("w") as handle:
        json.dump(details, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    summaries = {}
    for stage in counts:
        if counts[stage] == 0:
            continue
        values = [
            v for row in records if row["stage"] == stage for v in row["samples_us"]
        ]
        summaries[stage] = {
            "mean_us": statistics.mean(values),
            "min_us": min(values),
            "max_us": max(values),
            "median_us": statistics.median(values),
            "calls": len(values),
            "nonpositive_samples": sum(v <= 0 for v in values),
        }
    print(
        json.dumps({"explicit_graph_events": summaries, "rank": self.dist_ctx.rank}),
        flush=True,
    )
    return native


benchmark.DeviceMoEPipeline.capture = capture
benchmark.DeviceMoEPipeline.bench = bench
if __name__ == "__main__":
    benchmark.main()
