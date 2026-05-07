# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-component microbench for the SplitK zero-init fusion mechanism.

Where ``bench_zero_init_splitk_demo.py`` measures the producer + GEMM
sequence end-to-end, this script breaks the same sequence into its
three timed components so we can answer questions about *where* the
savings come from:

    prod_us   : producer (per_group_quant_hip) alone.  In splitk_fused
                mode the producer ALSO writes M*N zeros into the GEMM
                output buffer, so prod_us captures the extra producer
                work induced by the fusion.

    gemm_us   : GEMM alone, properly set up for the mode:
                  - none         : k_batch = 1, no in-kernel Y.zero_().
                  - splitk       : k_batch > 1 (read from the tuned CSV)
                                   and the kernel does its own internal
                                   Y.zero_() before the atomic-add.  The
                                   fill kernel launch is therefore
                                   *inside* this measurement.
                  - splitk_fused : k_batch > 1 and y_is_zeroed=True; the
                                   caller (this bench) zeros Y outside
                                   the timed region so the kernel skips
                                   its internal Y.zero_().

    total_us  : producer + GEMM in one timed region (matches what the
                ATOM serving path actually pays).

    fill_us_inferred = total_us - prod_us - gemm_us  (gives the cost
    of inter-kernel scheduling slack; should be small).

Optional ``--profile`` captures a torch.profiler chrome trace per
(shape, M, mode), then parses it to count launches by category:

    - producer (dynamic_per_group_scaled_quant)
    - fill / memset (Memset / aten::zero_ / vectorized_elementwise_kernel)
    - gemm (a8w8_blockscale_cktile / blockwise / kBatch)

The fill count must be 0 in ``splitk_fused`` for every shape if the
fusion is firing correctly across the whole call-site set (Q4).

Usage (one mode per process; remember to delete
``module_gemm_a8w8_blockscale_bpreshuffle_cktile.so`` when switching
between CSVs because the kernel manifest is built at compile time):

    python op_tests/bench_zero_init_components.py \\
        --mode splitk_fused \\
        --tuned-csv .../splitk_yz.csv \\
        --shapes-csv .../qwen3_next_per1x128.csv \\
        --m-values 1,4,8 --iters 100 --warmup 30 \\
        --out per_comp_splitk_fused.csv \\
        --profile --trace-dir traces_splitk_fused
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import torch

from aiter import dtypes
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle,
    get_CKGEMM_config,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant


VALID_MODES = ("none", "splitk", "splitk_fused")
VALID_PRODUCERS = ("p1", "p2", "p3")
PRODUCER_DESC = {
    "p1": "dynamic_per_token_scaled_quant (HIP) - MoE/o_proj path",
    "p2": "fused_qk_rmsnorm_group_quant (HIP) - qkv_proj/in_proj_qkvz path",
    "p3": "gated_rmsnorm_fp8_group_quant (HIP) - GDN out_proj path",
}


# -- shape / config helpers -----------------------------------------------


def _shape_splitk(M: int, N: int, K: int, tuned_file: str) -> int:
    try:
        cfg = get_CKGEMM_config(M, N, K, tuned_file=tuned_file)
    except Exception:  # noqa: BLE001
        return 0
    if cfg is None:
        return 0
    try:
        return int(cfg.get("splitK", 0))
    except (TypeError, ValueError):
        return 0


def _read_shapes_csv(path: str) -> list[tuple[int, int]]:
    """Return a list of unique (N, K) tuples from the shapes CSV.

    The shapes CSV may include an M column (for the existing demo) but
    we ignore it here -- this microbench iterates M independently via
    ``--m-values``.
    """
    nk: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_k = (int(row["N"]), int(row["K"]))
            if n_k not in seen:
                seen.add(n_k)
                nk.append(n_k)
    return nk


def _gen_inputs(
    M: int,
    N: int,
    K: int,
    producer: str,
    seed: int = 12345,
    device: str = "cuda",
):
    """Generate weights/inputs for a given (M, N, K) and producer family.

    All three producers feed an FP8 blockscale GEMM whose K dim is the
    flattened producer output width.  We always allocate the standard
    ``(M, K)`` quantized buffer and ``(K/128, M)`` scale buffer (transposed)
    even though the underlying kernel reads its inputs differently -- this
    keeps the GEMM-side machinery uniform across producers.
    """
    torch.manual_seed(seed)
    block_n, block_k = 128, 128
    scale_n = (N + block_n - 1) // block_n
    scale_k = (K + block_k - 1) // block_k

    weight = (torch.rand((N, K), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)

    inputs: dict = {"weight_shuffle": weight_shuffle, "w_scale": w_scale}
    if producer == "p1":
        inputs["x_bf16"] = torch.rand((M, K), dtype=dtypes.bf16, device=device) / 10
    elif producer == "p2":
        # Layernorm + quant: 2D input [M, K], 1D weight [K].  We mirror the
        # ATOM input_layernorm path (gemma_norm=True, transpose_scale=True).
        inputs["x_bf16"] = torch.rand((M, K), dtype=dtypes.bf16, device=device) / 10
        inputs["q_weight"] = torch.rand((K,), dtype=dtypes.bf16, device=device) / 10
    elif producer == "p3":
        # Gated RMSNorm: kernel requires head_dim=128.  num_heads = K // 128.
        if K % 128 != 0:
            raise ValueError(f"P3 requires K divisible by 128 (got K={K})")
        num_heads = K // 128
        inputs["x_3d"] = torch.rand(
            (M, num_heads, 128), dtype=dtypes.bf16, device=device
        ) / 10
        inputs["z_3d"] = torch.rand(
            (M, num_heads, 128), dtype=dtypes.bf16, device=device
        ) / 10
        inputs["weight_1d"] = (
            torch.rand((128,), dtype=dtypes.bf16, device=device) / 10
        )
        inputs["num_heads"] = num_heads
    else:
        raise ValueError(f"Unknown producer: {producer}")
    return inputs


# -- per-iteration call helpers (no timing) -------------------------------


def _alloc_xq_xs(M: int, K: int, device: str = "cuda"):
    """Pre-allocate the FP8 quantized buffer and (transposed) scale buffer
    that the GEMM consumes.  Producers P2/P3 take these as out-tensors.
    """
    num_groups = K // 128
    x_q = torch.empty((M, K), dtype=dtypes.fp8, device=device)
    # transpose_scale=True: stored as [num_groups, M] viewed as [M, num_groups].
    x_scale = torch.empty(
        (num_groups, M), dtype=torch.float32, device=device
    ).view(M, num_groups)
    return x_q, x_scale


def _call_producer(
    *,
    producer: str,
    inputs: dict,
    M: int,
    K: int,
    out_buf: torch.Tensor | None,
    do_fused_zero_init: bool,
    x_q: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
):
    """Run the producer kernel with optional fused zero-init.

    For P1 the wrapper allocates and returns (x_q, x_scale).  For P2/P3
    the caller pre-allocates them via ``_alloc_xq_xs`` and we write
    in-place; the call returns the same buffers for symmetry.
    """
    Y = out_buf if do_fused_zero_init else None
    if producer == "p1":
        return per_group_quant_hip(
            inputs["x_bf16"],
            quant_dtype=dtypes.fp8,
            group_size=128,
            transpose_scale=True,
            gemm_out_zero_init=Y,
        )
    if producer == "p2":
        if x_q is None or x_scale is None:
            x_q, x_scale = _alloc_xq_xs(M, K)
        fused_qk_rmsnorm_group_quant(
            x_q,
            x_scale,
            inputs["x_bf16"],
            inputs["q_weight"],
            1e-6,
            group_size=128,
            transpose_scale=True,
            gemma_norm=True,
            gemm_out_zero_init=Y,
        )
        return x_q, x_scale
    if producer == "p3":
        if x_q is None or x_scale is None:
            x_q, x_scale = _alloc_xq_xs(M, K)
        gated_rmsnorm_fp8_group_quant(
            x_q,
            x_scale,
            inputs["x_3d"],
            inputs["z_3d"],
            inputs["weight_1d"],
            1e-6,
            128,
            True,
            gemm_out_zero_init=Y,
        )
        return x_q, x_scale
    raise ValueError(f"Unknown producer: {producer}")


def _call_gemm(
    *,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight_shuffle: torch.Tensor,
    w_scale: torch.Tensor,
    out_buf: torch.Tensor | None,
    tuned_file: str,
    y_is_zeroed: bool,
):
    return gemm_a8w8_blockscale_bpreshuffle(
        x_q,
        weight_shuffle,
        x_scale,
        w_scale,
        dtype=dtypes.bf16,
        out=out_buf,
        y_is_zeroed=y_is_zeroed,
        tuned_file=tuned_file,
    )


# -- timing primitives ----------------------------------------------------


def _time_graph(closure, *, iters: int, warmup: int) -> dict:
    """Time ``closure`` wrapped in a ``torch.cuda.CUDAGraph``.

    The closure is called inside the capture context exactly once and its
    side effects (kernel launches) are recorded; subsequent timed iterations
    just call ``graph.replay()``. This mirrors what ATOM gets via vLLM's
    full-decode-graph capture: the launch-overhead cost of small kernels
    (incl. any standalone Y.zero_() fill) is amortized to a single
    graph-replay launch, leaving only the in-graph GPU work.

    Caller is responsible for ensuring the closure's tensors are statically
    addressed (use the same out_buf/x_q/x_scale buffers each iteration).
    """
    # Eager warmup so the closure's lazy paths (JIT compile, kernel
    # symbol lookup, tuned-CSV read, autotune) all run *before* capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            closure()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # graph_pool_handle() lets multiple captures (e.g. across modes) share
    # the caching allocator pool, but we only capture one graph per call,
    # so default pool is fine.
    with torch.cuda.graph(g, stream=s):
        closure()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    samples_us.sort()
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p10_us": samples_us[max(0, int(0.10 * len(samples_us)) - 1)],
        "p90_us": samples_us[min(len(samples_us) - 1, int(0.90 * len(samples_us)))],
        "stdev_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
    }


def _time_phase(fn, *, iters: int, warmup: int, pre_iter=None) -> dict:
    """Time ``fn`` with cudaEvent.

    ``pre_iter`` runs *before* each timed iteration outside the event
    boundaries -- e.g. zeroing out_buf for gemm_us measurement in modes
    where the kernel doesn't zero it itself.
    """
    for _ in range(warmup):
        if pre_iter is not None:
            pre_iter()
        fn()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(iters):
        if pre_iter is not None:
            pre_iter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    samples_us.sort()
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p10_us": samples_us[max(0, int(0.10 * len(samples_us)) - 1)],
        "p90_us": samples_us[min(len(samples_us) - 1, int(0.90 * len(samples_us)))],
        "stdev_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
    }


def _bench_components(
    *,
    M: int,
    N: int,
    K: int,
    mode: str,
    producer: str,
    tuned_file: str,
    iters: int,
    warmup: int,
    seed: int,
    graph: bool = False,
) -> dict:
    inputs = _gen_inputs(M, N, K, producer=producer, seed=seed)
    weight_shuffle = inputs["weight_shuffle"]
    w_scale = inputs["w_scale"]

    splitk_csv = _shape_splitk(M, N, K, tuned_file)
    do_fused = (mode == "splitk_fused" and splitk_csv > 0)

    # Output buffer: required for splitk and splitk_fused so we exercise
    # the production code path where the caller owns Y.  None mode lets
    # the wrapper allocate.
    out_buf = (
        torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
        if mode != "none"
        else None
    )

    # P2/P3 want pre-allocated x_q / x_scale buffers (out-of-place quant
    # producers).  P1 returns its own each call so we re-allocate via the
    # wrapper on every call too -- _alloc_xq_xs is a no-op for P1.
    x_q_pre, x_scale_pre = _alloc_xq_xs(M, K) if producer in ("p2", "p3") else (None, None)

    # We need a one-shot producer call to materialize x_q and x_scale
    # for the gemm-only timing.  These are reused across all gemm-only
    # iterations (their content doesn't matter for timing).
    x_q_ref, x_scale_ref = _call_producer(
        producer=producer,
        inputs=inputs,
        M=M,
        K=K,
        out_buf=out_buf,
        do_fused_zero_init=do_fused,
        x_q=x_q_pre,
        x_scale=x_scale_pre,
    )
    torch.cuda.synchronize()

    # ----- prod_us (producer alone) ------------------------------------
    def prod_call():
        _call_producer(
            producer=producer,
            inputs=inputs,
            M=M,
            K=K,
            out_buf=out_buf,
            do_fused_zero_init=do_fused,
            x_q=x_q_pre,
            x_scale=x_scale_pre,
        )

    prod_stats = _time_phase(prod_call, iters=iters, warmup=warmup)

    # ----- gemm_us (gemm alone) ----------------------------------------
    # In none mode (splitK=0) the GEMM fully writes Y, no zero needed.
    # In splitk mode the kernel internally Y.zero_()'s before atomic_add,
    # so we explicitly do NOT pre-zero (and we don't pass y_is_zeroed).
    # In splitk_fused mode the caller (== this bench) must pre-zero out
    # Y outside the timed region to mimic the real producer-fused path.
    pre_iter = None
    y_is_zeroed_flag = False
    if mode == "splitk_fused":
        y_is_zeroed_flag = do_fused
        if do_fused:
            def pre_iter():
                out_buf.zero_()

    def gemm_call():
        _call_gemm(
            x_q=x_q_ref,
            x_scale=x_scale_ref,
            weight_shuffle=weight_shuffle,
            w_scale=w_scale,
            out_buf=out_buf,
            tuned_file=tuned_file,
            y_is_zeroed=y_is_zeroed_flag,
        )

    gemm_stats = _time_phase(gemm_call, iters=iters, warmup=warmup, pre_iter=pre_iter)

    # ----- total_us (producer + gemm in one timed region) --------------
    def total_call():
        x_q, x_scale = _call_producer(
            producer=producer,
            inputs=inputs,
            M=M,
            K=K,
            out_buf=out_buf,
            do_fused_zero_init=do_fused,
            x_q=x_q_pre,
            x_scale=x_scale_pre,
        )
        _call_gemm(
            x_q=x_q,
            x_scale=x_scale,
            weight_shuffle=weight_shuffle,
            w_scale=w_scale,
            out_buf=out_buf,
            tuned_file=tuned_file,
            y_is_zeroed=do_fused,
        )

    total_stats = _time_phase(total_call, iters=iters, warmup=warmup)

    fill_inferred = (
        total_stats["median_us"] - prod_stats["median_us"] - gemm_stats["median_us"]
    )

    result = {
        "splitK_csv": splitk_csv,
        "do_fused": int(do_fused),
        "prod_us": prod_stats["median_us"],
        "prod_stdev": prod_stats["stdev_us"],
        "gemm_us": gemm_stats["median_us"],
        "gemm_stdev": gemm_stats["stdev_us"],
        "total_us": total_stats["median_us"],
        "total_stdev": total_stats["stdev_us"],
        "fill_us_inferred": fill_inferred,
    }

    # ----- total_us under HIP/CUDA graph capture ------------------------
    # Same producer + GEMM as the eager total above, but wrapped in a
    # CUDA graph so launch overhead is amortized to one
    # cudaGraphLaunch per replay. This is the regime ATOM actually
    # runs in (vLLM captures the whole decode forward as a graph), so
    # ``total_us_graph`` is the apples-to-apples "expected impact on
    # ATOM TPOT" number. The savings of the standalone Y.zero_() fill
    # mostly *come from* its launch overhead, so we expect the
    # splitk vs splitk_fused gap to shrink (often substantially) here.
    if graph:
        # x_q / x_scale must be statically-allocated for graph replay.
        # P1 returns fresh tensors each call (its wrapper internally
        # allocates xq+scale). To keep the graph's input/output set
        # static we re-run a single eager producer call to get a pair
        # we can reuse, then bind it into the graph closure via
        # the in-place graph-friendly producer wrappers.
        #
        # For P1: per_group_quant_hip allocates a new xq each call;
        # under torch.cuda.graph() the caching allocator routes those
        # through the graph's private memory pool, so replay reuses
        # the same pointers automatically. We simply call the public
        # wrapper inside the closure and trust the pool.
        # For P2/P3: x_q_pre / x_scale_pre are persistent and the
        # producers write in-place. Already graph-safe.

        out_buf_graph = (
            torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
            if mode != "none"
            else torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
            # In none mode the GEMM allocates its own out by default;
            # for graph capture we need a stable buffer to write into.
        )

        def graph_total_call():
            # In splitk_fused mode the producer writes zeros into Y as
            # a side effect; in splitk mode the GEMM kernel internally
            # zeros Y before atomic_add (y_is_zeroed=False). Both are
            # captured as part of the graph here -- the captured
            # kernel sequence will include the standalone fill in
            # splitk mode and skip it in splitk_fused mode.
            x_q_g, x_scale_g = _call_producer(
                producer=producer,
                inputs=inputs,
                M=M,
                K=K,
                out_buf=out_buf_graph,
                do_fused_zero_init=do_fused,
                x_q=x_q_pre,
                x_scale=x_scale_pre,
            )
            _call_gemm(
                x_q=x_q_g,
                x_scale=x_scale_g,
                weight_shuffle=weight_shuffle,
                w_scale=w_scale,
                out_buf=out_buf_graph,
                tuned_file=tuned_file,
                y_is_zeroed=do_fused,
            )

        graph_stats = _time_graph(
            graph_total_call, iters=iters, warmup=warmup,
        )
        result["total_graph_us"] = graph_stats["median_us"]
        result["total_graph_stdev"] = graph_stats["stdev_us"]

    return result


# -- profiler-driven kernel-launch counting -------------------------------


_FILL_PATTERNS = (
    "memset",                  # hipMemsetAsync, raw HSA memset
    "Memset",                  # cuda::Memset / hipMemset friendly names
    "aten::zero_",             # high-level torch op
    "vectorized_elementwise_kernel",  # torch zero_() lowering
    "fillkernel",              # kernel-internal Y.zero_() compute kernel
)
_PROD_PATTERNS = (
    "dynamic_per_group_scaled_quant",     # P1
    "dynamic_per_token_scaled_quant",      # P1 friendly name
    "fused_qk_rmsnorm_group_quant_kernel", # P2
    "gated_rmsnorm_fp8_group_quant_kernel",# P3
)
_GEMM_PATTERNS = (
    "a8w8_blockscale_cktile",
    "blockwise_gemm",
    "kBatch",
    "QuantGemmKernel",   # mangled CKTile entry symbol
    "ck_tile6kentry",    # mangled CKTile launcher
)


def _classify(name: str) -> str | None:
    n = name or ""
    nl = n.lower()
    if any(p.lower() in nl for p in _FILL_PATTERNS):
        return "fill"
    if any(p.lower() in nl for p in _PROD_PATTERNS):
        return "prod"
    if any(p.lower() in nl for p in _GEMM_PATTERNS):
        return "gemm"
    return None


def _count_kernels_in_trace(trace_path: str) -> dict:
    """Read a chrome trace JSON and count GPU kernel events by category."""
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace.get("traceEvents") or []
    counts = {"prod": 0, "fill": 0, "gemm": 0, "other": 0}
    other_names: dict[str, int] = {}
    for ev in events:
        # We only care about device-side kernels.  In torch's chrome
        # trace those land on the "kernel" / "gpu_op" categories with
        # a "cat" field of "kernel" or category prefix "kernel".  Be
        # permissive: anything with ph='X', cat starts with 'kernel'
        # OR the standard Kineto cat 'kernel' / 'gpu_memcpy'.
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat") or ""
        if "kernel" not in cat and "gpu" not in cat:
            continue
        name = ev.get("name", "")
        cls = _classify(name)
        if cls is None:
            counts["other"] += 1
            other_names[name] = other_names.get(name, 0) + 1
        else:
            counts[cls] += 1
    counts["_top_other"] = sorted(other_names.items(), key=lambda kv: -kv[1])[:5]
    return counts


def _capture_trace(
    *,
    M: int,
    N: int,
    K: int,
    mode: str,
    producer: str,
    tuned_file: str,
    trace_dir: str,
    trace_iters: int,
    seed: int,
    graph: bool = False,
) -> tuple[str, dict]:
    os.makedirs(trace_dir, exist_ok=True)
    inputs = _gen_inputs(M, N, K, producer=producer, seed=seed)
    weight_shuffle = inputs["weight_shuffle"]
    w_scale = inputs["w_scale"]
    splitk_csv = _shape_splitk(M, N, K, tuned_file)
    do_fused = (mode == "splitk_fused" and splitk_csv > 0)
    out_buf = (
        torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
        if mode != "none"
        else None
    )
    x_q_pre, x_scale_pre = _alloc_xq_xs(M, K) if producer in ("p2", "p3") else (None, None)

    # warm
    for _ in range(5):
        x_q, x_scale = _call_producer(
            producer=producer, inputs=inputs, M=M, K=K,
            out_buf=out_buf, do_fused_zero_init=do_fused,
            x_q=x_q_pre, x_scale=x_scale_pre,
        )
        _call_gemm(
            x_q=x_q, x_scale=x_scale,
            weight_shuffle=weight_shuffle, w_scale=w_scale,
            out_buf=out_buf, tuned_file=tuned_file,
            y_is_zeroed=do_fused,
        )
    torch.cuda.synchronize()

    suffix = "_graph" if graph else ""
    out_file = os.path.join(
        trace_dir, f"trace_{mode}_{producer}_M{M}_N{N}_K{K}{suffix}.json"
    )

    if graph:
        # Capture once, then replay under the profiler. The same kernels
        # show up in the chrome trace (replay re-issues them) so the
        # fill-count classifier still works -- we simply observe whether
        # a standalone fill kernel made it into the captured graph.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                x_q, x_scale = _call_producer(
                    producer=producer, inputs=inputs, M=M, K=K,
                    out_buf=out_buf, do_fused_zero_init=do_fused,
                    x_q=x_q_pre, x_scale=x_scale_pre,
                )
                _call_gemm(
                    x_q=x_q, x_scale=x_scale,
                    weight_shuffle=weight_shuffle, w_scale=w_scale,
                    out_buf=out_buf, tuned_file=tuned_file,
                    y_is_zeroed=do_fused,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            x_q, x_scale = _call_producer(
                producer=producer, inputs=inputs, M=M, K=K,
                out_buf=out_buf, do_fused_zero_init=do_fused,
                x_q=x_q_pre, x_scale=x_scale_pre,
            )
            _call_gemm(
                x_q=x_q, x_scale=x_scale,
                weight_shuffle=weight_shuffle, w_scale=w_scale,
                out_buf=out_buf, tuned_file=tuned_file,
                y_is_zeroed=do_fused,
            )
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        ) as prof:
            for _ in range(trace_iters):
                g.replay()
            torch.cuda.synchronize()
        prof.export_chrome_trace(out_file)
    else:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        ) as prof:
            for _ in range(trace_iters):
                x_q, x_scale = _call_producer(
                    producer=producer, inputs=inputs, M=M, K=K,
                    out_buf=out_buf, do_fused_zero_init=do_fused,
                    x_q=x_q_pre, x_scale=x_scale_pre,
                )
                _call_gemm(
                    x_q=x_q, x_scale=x_scale,
                    weight_shuffle=weight_shuffle, w_scale=w_scale,
                    out_buf=out_buf, tuned_file=tuned_file,
                    y_is_zeroed=do_fused,
                )
            torch.cuda.synchronize()
        prof.export_chrome_trace(out_file)

    counts = _count_kernels_in_trace(out_file)
    counts["per_iter_prod"] = counts["prod"] / max(trace_iters, 1)
    counts["per_iter_fill"] = counts["fill"] / max(trace_iters, 1)
    counts["per_iter_gemm"] = counts["gemm"] / max(trace_iters, 1)
    return out_file, counts


# -- CLI ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument(
        "--producers",
        default="p1",
        help="Comma-separated producers to sweep. Choose from: "
        + ",".join(VALID_PRODUCERS) + ". Default: p1.",
    )
    p.add_argument("--tuned-csv", required=True)
    p.add_argument("--shapes-csv", required=True)
    p.add_argument(
        "--m-values",
        default="1,4,8",
        help="Comma-separated M values to sweep per (N,K). Default: 1,4,8.",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None, help="Output CSV path (also printed).")
    p.add_argument("--profile", action="store_true",
                   help="Capture a torch.profiler trace per (shape, M) and "
                        "report kernel-launch counts by category.")
    p.add_argument("--trace-dir", default=None,
                   help="Directory for chrome traces (only with --profile).")
    p.add_argument("--trace-iters", type=int, default=10)
    p.add_argument("--graph", action="store_true",
                   help="Also time the producer+GEMM sequence wrapped in a "
                        "torch.cuda.CUDAGraph (HIP graph on AMD). This "
                        "amortizes kernel launch overhead -- it is the "
                        "regime ATOM actually runs in via vLLM full-decode "
                        "graph capture, so total_graph_us is the right "
                        "predictor of e2e TPOT impact. When combined with "
                        "--profile, the captured trace replays the graph "
                        "instead of issuing the kernels eagerly.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    for path in (args.tuned_csv, args.shapes_csv):
        if not os.path.exists(path):
            print(f"ERROR: required CSV not found: {path}", file=sys.stderr)
            return 1
    if args.profile and args.trace_dir is None:
        print("ERROR: --profile requires --trace-dir", file=sys.stderr)
        return 1

    m_values = [int(x) for x in args.m_values.split(",")]
    producers = [p.strip() for p in args.producers.split(",") if p.strip()]
    for p in producers:
        if p not in VALID_PRODUCERS:
            print(f"ERROR: unknown producer '{p}'. Valid: {VALID_PRODUCERS}",
                  file=sys.stderr)
            return 1
    nk_pairs = _read_shapes_csv(args.shapes_csv)

    print(
        f"# mode={args.mode} producers={producers} tuned_csv={args.tuned_csv} "
        f"iters={args.iters} warmup={args.warmup} m_values={m_values} "
        f"device={torch.cuda.get_device_name(0)}"
    )
    cols = (
        "mode", "producer", "M", "N", "K", "splitK_csv", "do_fused",
        "prod_us", "prod_stdev",
        "gemm_us", "gemm_stdev",
        "total_us", "total_stdev",
        "fill_us_inferred",
    )
    if args.graph:
        cols = cols + ("total_graph_us", "total_graph_stdev")
    if args.profile:
        cols = cols + ("trace_prod", "trace_fill", "trace_gemm",
                       "trace_other_top")

    print("\t".join(cols))

    rows: list[dict] = []
    for producer in producers:
        for (N, K) in nk_pairs:
            for M in m_values:
                # P3 only supports K divisible by 128; skip incompatible
                # shapes rather than raising.
                if producer == "p3" and K % 128 != 0:
                    continue
                try:
                    stats = _bench_components(
                        M=M, N=N, K=K,
                        mode=args.mode, producer=producer,
                        tuned_file=args.tuned_csv,
                        iters=args.iters, warmup=args.warmup, seed=args.seed,
                        graph=args.graph,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"# {args.mode} producer={producer} M={M} N={N} K={K}: ERROR {e}",
                        file=sys.stderr,
                    )
                    continue

                row = {
                    "mode": args.mode,
                    "producer": producer,
                    "M": M, "N": N, "K": K,
                    **stats,
                }

                if args.profile:
                    try:
                        trace_path, counts = _capture_trace(
                            M=M, N=N, K=K,
                            mode=args.mode, producer=producer,
                            tuned_file=args.tuned_csv,
                            trace_dir=args.trace_dir,
                            trace_iters=args.trace_iters,
                            seed=args.seed,
                            graph=args.graph,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"# trace fail {args.mode} producer={producer} "
                            f"M={M} N={N} K={K}: {e}", file=sys.stderr,
                        )
                        counts = {"per_iter_prod": float("nan"),
                                  "per_iter_fill": float("nan"),
                                  "per_iter_gemm": float("nan"),
                                  "_top_other": []}
                    row["trace_prod"] = counts.get("per_iter_prod", float("nan"))
                    row["trace_fill"] = counts.get("per_iter_fill", float("nan"))
                    row["trace_gemm"] = counts.get("per_iter_gemm", float("nan"))
                    row["trace_other_top"] = ";".join(
                        f"{name}={c}" for name, c in counts.get("_top_other", [])
                    )

                rows.append(row)

                line = (
                    f"{args.mode}\t{producer}\t{M}\t{N}\t{K}\t"
                    f"{stats['splitK_csv']}\t{stats['do_fused']}\t"
                    f"{stats['prod_us']:.3f}\t{stats['prod_stdev']:.3f}\t"
                    f"{stats['gemm_us']:.3f}\t{stats['gemm_stdev']:.3f}\t"
                    f"{stats['total_us']:.3f}\t{stats['total_stdev']:.3f}\t"
                    f"{stats['fill_us_inferred']:.3f}"
                )
                if args.graph:
                    line += (
                        f"\t{stats.get('total_graph_us', float('nan')):.3f}"
                        f"\t{stats.get('total_graph_stdev', float('nan')):.3f}"
                    )
                if args.profile:
                    line += (
                        f"\t{row['trace_prod']:.2f}"
                        f"\t{row['trace_fill']:.2f}"
                        f"\t{row['trace_gemm']:.2f}"
                        f"\t{row['trace_other_top']}"
                    )
                print(line, flush=True)

    if args.out is not None:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols))
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"# results -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
