# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Profiler regression test for HIP graph replay.

Detects rocprofiler-sdk interception overhead that degrades HIP graph replay
performance. When PyTorch kineto is linked against librocprofiler-sdk.so
instead of libroctracer64.so, every hipGraphLaunch call incurs ~270us of
overhead from profiler interception hooks, even when the profiler is not
actively collecting traces. This drops GPU occupancy from ~97% to ~75%.

This test serves as a regression guard to catch this issue early in any
new ROCm container or PyTorch build.

Reference: ROCm/rocm-systems#4401, ROCm/pytorch#2579, ROCm/pytorch#3056
"""

import ctypes
import json
import os
import subprocess
import tempfile

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
GPU_OCCUPANCY_PASS = 0.90  # >90% = healthy graph replay
GPU_OCCUPANCY_FAIL = 0.80  # <80% = regression detected
HIPGRAPHLAUNCH_MAX_US = 150  # healthy ~50us, regressed ~316us
GAP_100US_MAX_COUNT = 5  # healthy = 0, regressed = 9+
KERNEL_COUNT_TOLERANCE = 0.20  # allow 20% variance from expected

# Workload config: small transformer to amplify dispatch overhead
NUM_LAYERS = 4
BATCH = 8
SEQ_LEN = 1
HIDDEN = 2048
HEADS = 16
GRAPH_ITERS = 100


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=HEADS):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn_up = nn.Linear(hidden, 4 * hidden, bias=False)
        self.ffn_down = nn.Linear(4 * hidden, hidden, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        B, S, H = x.shape
        head_dim = H // HEADS
        h = self.ln1(x)
        qkv = self.attn_qkv(h).reshape(B, S, 3, HEADS, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, H)
        x = x + self.attn_out(attn)
        h = self.ln2(x)
        x = x + self.ffn_down(self.act(self.ffn_up(h)))
        return x


def _build_model(device="cuda:0"):
    model = (
        nn.Sequential(*[_TransformerBlock() for _ in range(NUM_LAYERS)])
        .to(device)
        .half()
        .eval()
    )
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_libtorch_profiler_backend():
    """Check which profiler backend libtorch_cpu.so links against."""
    import torch as _torch

    torch_lib_dir = os.path.dirname(_torch.__file__) + "/lib"
    libtorch_cpu = os.path.join(torch_lib_dir, "libtorch_cpu.so")
    if not os.path.exists(libtorch_cpu):
        return "unknown"
    try:
        result = subprocess.run(
            ["ldd", libtorch_cpu],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "libroctracer64" in line:
                return "roctracer"
            if "librocprofiler-sdk" in line:
                return "rocprofiler-sdk"
    except Exception:
        pass
    return "unknown"


def _capture_graph(model, device="cuda:0"):
    """Capture a CUDA/HIP graph of the model forward pass."""
    static_input = torch.randn(
        BATCH, SEQ_LEN, HIDDEN, device=device, dtype=torch.float16
    )
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_output = model(static_input)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            static_output = model(static_input)
    torch.cuda.synchronize()
    return g, static_input, static_output


def _profile_graph_replay(model, g, static_input, num_iters=GRAPH_ITERS):
    """Profile graph replay with torch.profiler and return trace dict."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iters):
            static_input.copy_(torch.randn_like(static_input))
            g.replay()
        torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)

    with open(trace_path) as f:
        trace = json.load(f)
    os.unlink(trace_path)
    return trace, prof


def _analyze_trace(trace):
    """Analyze chrome trace for GPU gaps and occupancy."""
    events = trace.get("traceEvents", [])

    gpu_events = []
    hipgraph_launches = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)

        if "kernel" in cat.lower() or "gpu_memcpy" in cat.lower():
            gpu_events.append((ts, ts + dur))
        if "hipGraphLaunch" in name:
            hipgraph_launches.append(dur)

    if not gpu_events:
        return {
            "gpu_occupancy": 0.0,
            "gap_count_gt100us": 0,
            "gap_max_us": 0.0,
            "hipgraphlaunch_avg_us": 0.0,
            "kernel_count": 0,
        }

    gpu_events.sort()
    total_kernel = sum(e - s for s, e in gpu_events)
    span = gpu_events[-1][1] - gpu_events[0][0]
    occupancy = total_kernel / span if span > 0 else 0.0

    gaps = sorted(
        [
            gpu_events[i][0] - gpu_events[i - 1][1]
            for i in range(1, len(gpu_events))
            if gpu_events[i][0] > gpu_events[i - 1][1]
        ]
    )

    gap_gt100 = sum(1 for g in gaps if g > 100)  # trace is in us
    gap_max = gaps[-1] if gaps else 0.0

    hipgraph_avg = (
        sum(hipgraph_launches) / len(hipgraph_launches) if hipgraph_launches else 0.0
    )

    return {
        "gpu_occupancy": occupancy,
        "gap_count_gt100us": gap_gt100,
        "gap_max_us": gap_max,
        "hipgraphlaunch_avg_us": hipgraph_avg,
        "kernel_count": len(gpu_events),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestProfilerRegression:
    """Regression tests for profiler overhead on HIP graph replay."""

    def test_kineto_backend_is_roctracer(self):
        """Verify PyTorch kineto links roctracer, not rocprofiler-sdk."""
        backend = _get_libtorch_profiler_backend()
        if backend == "rocprofiler-sdk":
            pytest.fail(
                f"libtorch_cpu.so links librocprofiler-sdk.so instead of "
                f"libroctracer64.so. This causes ~270us overhead per "
                f"hipGraphLaunch. See ROCm/pytorch#2579 for the fix."
            )
        assert backend in (
            "roctracer",
            "unknown",
        ), f"Unexpected profiler backend: {backend}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_hipgraph_replay_occupancy(self):
        """HIP graph replay should have >90% GPU occupancy under profiler."""
        device = "cuda:0"
        torch.cuda.set_device(device)

        model = _build_model(device)

        # Warmup
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device, dtype=torch.float16)
        for _ in range(20):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()

        # Capture and warmup graph
        g, static_input, _ = _capture_graph(model, device)
        for _ in range(20):
            static_input.copy_(torch.randn_like(static_input))
            g.replay()
        torch.cuda.synchronize()

        # Profile
        trace, prof = _profile_graph_replay(model, g, static_input)
        stats = _analyze_trace(trace)

        # Report
        print(f"\n{'=' * 60}")
        print("HIP Graph Profiler Regression Test Results")
        print(f"{'=' * 60}")
        print(f"  Profiler backend:     {_get_libtorch_profiler_backend()}")
        print(f"  GPU occupancy:        {stats['gpu_occupancy']:.1%}")
        print(f"  Gaps > 100us:         {stats['gap_count_gt100us']}")
        print(f"  Max gap:              {stats['gap_max_us']:.1f}us")
        print(f"  hipGraphLaunch avg:   {stats['hipgraphlaunch_avg_us']:.1f}us")
        print(f"  Kernel count:         {stats['kernel_count']}")
        print(f"{'=' * 60}")

        # Assertions
        assert stats["gpu_occupancy"] > GPU_OCCUPANCY_FAIL, (
            f"GPU occupancy {stats['gpu_occupancy']:.1%} < {GPU_OCCUPANCY_FAIL:.0%}. "
            f"Likely rocprofiler-sdk interception overhead. "
            f"Check: ldd libtorch_cpu.so | grep roctracer"
        )

        if stats["gpu_occupancy"] < GPU_OCCUPANCY_PASS:
            import warnings

            warnings.warn(
                f"GPU occupancy {stats['gpu_occupancy']:.1%} is below "
                f"optimal ({GPU_OCCUPANCY_PASS:.0%}). Monitor for regression."
            )

        assert stats["gap_count_gt100us"] <= GAP_100US_MAX_COUNT, (
            f"{stats['gap_count_gt100us']} gaps > 100us detected "
            f"(threshold: {GAP_100US_MAX_COUNT}). "
            f"Profiler interception may be adding latency."
        )

        assert stats["hipgraphlaunch_avg_us"] < HIPGRAPHLAUNCH_MAX_US, (
            f"hipGraphLaunch avg {stats['hipgraphlaunch_avg_us']:.0f}us "
            f"> {HIPGRAPHLAUNCH_MAX_US}us threshold. "
            f"Expected ~50us with roctracer, ~316us with rocprofiler-sdk."
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Profiler backend: {_get_libtorch_profiler_backend()}")

    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
        exit(0)

    device = "cuda:0"
    torch.cuda.set_device(device)
    model = _build_model(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(
                torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device, dtype=torch.float16)
            )
    torch.cuda.synchronize()

    g, static_input, _ = _capture_graph(model, device)
    for _ in range(20):
        static_input.copy_(torch.randn_like(static_input))
        g.replay()
    torch.cuda.synchronize()

    trace, prof = _profile_graph_replay(model, g, static_input)
    stats = _analyze_trace(trace)

    print(f"\n{'=' * 60}")
    print("HIP Graph Profiler Regression Test")
    print(f"{'=' * 60}")
    print(f"  Profiler backend:     {_get_libtorch_profiler_backend()}")
    print(f"  GPU occupancy:        {stats['gpu_occupancy']:.1%}")
    print(f"  Gaps > 100us:         {stats['gap_count_gt100us']}")
    print(f"  Max gap:              {stats['gap_max_us']:.1f}us")
    print(f"  hipGraphLaunch avg:   {stats['hipgraphlaunch_avg_us']:.1f}us")
    print(f"  Kernel count:         {stats['kernel_count']}")
    print(f"{'=' * 60}")

    failed = False
    if stats["gpu_occupancy"] < GPU_OCCUPANCY_FAIL:
        print(
            f"FAIL: GPU occupancy {stats['gpu_occupancy']:.1%} < {GPU_OCCUPANCY_FAIL:.0%}"
        )
        failed = True
    if stats["gap_count_gt100us"] > GAP_100US_MAX_COUNT:
        print(
            f"FAIL: {stats['gap_count_gt100us']} gaps > 100us (max {GAP_100US_MAX_COUNT})"
        )
        failed = True
    if stats["hipgraphlaunch_avg_us"] > HIPGRAPHLAUNCH_MAX_US:
        print(
            f"FAIL: hipGraphLaunch {stats['hipgraphlaunch_avg_us']:.0f}us > {HIPGRAPHLAUNCH_MAX_US}us"
        )
        failed = True

    if failed:
        print("\nREGRESSION DETECTED. Check profiler backend linkage.")
        exit(1)
    else:
        print("\nPASS")
        exit(0)
