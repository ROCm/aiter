#!/usr/bin/env python3
"""Benchmark gemm_afp8wfp8_preshuffle (gluon/triton) against the FlyDSL kernel.

Calls aiter.ops.triton.gemm.basic.gemm_afp8wfp8.gemm_afp8wfp8_preshuffle with
preshuffled weights and e8m0 scales, matching the test harness.

Unlike the a8w8 blockscale GEMM, both scale operands ride inside wmma_scaled
(gluon) / dot_scaled (triton), so scales are e8m0 bytes rather than fp32:
  * activations: one byte per --x-scale-group K elements (128 = blockscale,
    what ATOM's per_1x128 quant emits; 32 = MX)
  * weights:     one byte per 128(N) x 128(K) block

MEASUREMENT NOTES (these kernels are ~7 us; the harness has to be careful):
  * Python dispatch costs 50-90 us per call here -- an order of magnitude more
    than the kernel. Any non-graph wall-clock timing (plain triton do_bench,
    time.perf_counter) measures Python, not the GPU. Only cudagraph replay or
    profiler device-time is meaningful.
  * The flydsl path pays *more* host time than gluon (torch custom-op dispatch
    + a pandas tuned-CSV lookup per call), so host-bound methods are biased
    against it by ~35 us of pure Python.
  * triton's do_bench_cudagraph `rep` is a MILLISECOND BUDGET, not a replay
    count. It sizes the graph as n_repeat = rep / eager_estimate, and the eager
    estimate is host-bound -- so a small rep yields a short graph and leaves
    ~40 us of graph-launch overhead in the per-iter number (and leaves *more*
    of it in the flydsl number, since its bigger host cost shrinks n_repeat).
    Use --rep-ms 20 (the triton default) or larger.
  * Backends must be compared in ONE process, interleaved, min-of-trials.
    Separate sequential runs drift by up to 2x on this part.

Example:
    python bench_gemm_afp8wfp8_preshuffle.py --m 64 --n 2048 --k 7168
    python bench_gemm_afp8wfp8_preshuffle.py --m 64 --backend gluon,flydsl \
        --transpose-x-scale --method all
    python bench_gemm_afp8wfp8_preshuffle.py --m 64 --sweep tp4 --backend gluon,flydsl
"""

import argparse
import copy
import gc

import torch
import triton
import torch.profiler as tpf
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    gemm_afp8wfp8_preshuffle,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

_, e4m3_type = get_fp8_dtypes()

W_SCALE_N_GROUP = 128
W_SCALE_K_GROUP = 128
SHUFFLE_LAYOUT = (16, 16)

BACKENDS = ("gluon", "triton", "flydsl", "flydsl_raw")
METHODS = ("cudagraph", "perftest", "event")

# bf16 linears from the DeepSeek-V4-Pro fp8 path (see the EP4/TP4 traces).
DSV4_SHAPES = {
    "tp4": [
        ("attn.wqkv_a", 2048, 7168),
        ("attn.wq_b", 16384, 1536),
        ("attn.indexer.wq_b", 8192, 1536),
        ("attn.wo_b", 7168, 4096),
        ("shared_experts.gate_up_proj", 1536, 7168),
        ("shared_experts.w2", 7168, 768),
    ],
    "ep4": [
        ("attn.wqkv_a", 2048, 7168),
        ("attn.wq_b", 65536, 1536),
        ("attn.indexer.wq_b", 8192, 1536),
        ("attn.wo_b", 7168, 16384),
        ("shared_experts.gate_up_proj", 6144, 7168),
        ("shared_experts.w2", 7168, 3072),
    ],
}


def generate_inputs(M, N, K, dtype, output, x_scale_group, transpose_x_scale):
    assert N % 16 == 0, "N must be a multiple of 16 for preshuffle"
    assert K % 32 == 0, "K must be a multiple of 32 for preshuffle"
    assert K % x_scale_group == 0, f"K must be a multiple of {x_scale_group}"

    scale_n = (N + W_SCALE_N_GROUP - 1) // W_SCALE_N_GROUP
    scale_k = (K + W_SCALE_K_GROUP - 1) // W_SCALE_K_GROUP

    x = (torch.rand(M, K, dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    weight = (torch.rand(N, K, dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    # e8m0 bytes around 127 (== 2^0) so the dequant stays unit-ish. uint8 is the
    # layout the kernels are validated against; it is bit-identical to
    # fp8_e8m0fnu, so ATOM's quant output can be passed through a .view().
    x_scale = torch.randint(
        125, 130, (M, K // x_scale_group), dtype=torch.uint8, device="cuda"
    )
    w_scale = torch.randint(
        125, 130, (scale_n, scale_k), dtype=torch.uint8, device="cuda"
    )

    # shuffle_weight keeps the (N, K) shape; the wrapper does the (N//16, K*16)
    # view internally.
    w_shuffled = shuffle_weight(weight.view(torch.uint8), SHUFFLE_LAYOUT)

    if transpose_x_scale:
        # Same bytes laid out (Kg, M) row-major, reinterpreted as (M, Kg) --
        # what per_group_quant_hip(transpose_scale=True) hands back.
        x_scale = x_scale.T.contiguous().reshape(M, K // x_scale_group)

    y = None
    if output:
        y = torch.empty(M, N, dtype=dtype, device="cuda")

    return x, w_shuffled, x_scale, w_scale, y


def _pick_flydsl_kernel(M, N, K):
    """Resolve the kernelName the aiter dispatch would pick for this shape."""
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        kernel_fits_shape,
        kernels_list,
    )

    config = get_CKGEMM_config(
        M, N, K, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    if config is not None and config.get("libtype") == "flydsl":
        return str(config["kernelName"]), "tuned-csv"
    fits = [ki for ki in kernels_list.values() if kernel_fits_shape(ki, M, N, K)]
    if not fits:
        raise RuntimeError(f"no flydsl kernel fits M={M} N={N} K={K}")
    want_tm = min(256, max(16, 1 << (M - 1).bit_length()))
    ki = min(fits, key=lambda x: (abs(x.tile_m - want_tm), -x.tile_n, -x.tile_k))
    return ki.name, "heuristic"


def _make_flydsl_launch(x, w_shuffled, x_scale, w_scale, y, dtype, args, raw):
    """Launch the FlyDSL mxfp8_128 bpreshuffle kernel on the same inputs.

    This is the kernel ATOM actually runs today: linear.py's per_1x128 path calls
    gemm_a8w8_blockscale_bpreshuffle, which on gfx1250 with fp8_e8m0 scales
    dispatches to gemm_a8w8_mxfp8_128_bpreshuffle_flydsl and returns before any
    triton/gluon branch. Benchmarking it here gives a head-to-head on identical
    operands rather than across two harnesses.

    Only the scale dtype differs from the gluon call: the dispatch is gated on
    both scales being dtypes.fp8_e8m0, while our wrapper wants uint8. The bytes
    are identical, so a .view() is the whole conversion -- which is also all
    ATOM would need to feed our kernel.

    raw=True skips the aiter dispatch (torch custom op + tuned-CSV lookup) and
    calls the kernel entry point directly. Same GPU work, ~35 us less host time
    per call -- the two agree under cudagraph replay and diverge under any
    host-bound timing, which is the cleanest way to see whether a number is
    measuring the GPU or Python.
    """
    assert args.x_scale_group == W_SCALE_K_GROUP, (
        f"flydsl is the mxfp8_128 path: it needs 1x{W_SCALE_K_GROUP} activation "
        f"scales, got 1x{args.x_scale_group}"
    )
    assert args.transpose_x_scale, (
        "flydsl expects column-major x_scale (what ATOM's per_1x128 quant emits "
        "with transpose_scale=True); pass --transpose-x-scale"
    )

    xs = x_scale.view(dtypes.fp8_e8m0)
    ws = w_scale.view(dtypes.fp8_e8m0)
    # WQ stays (N, K): the dispatch reads n = WQ.shape[0]. That is the shuffled
    # weight as-is, not the (N//16, K*16) view the triton/gluon wrapper takes.
    wq = w_shuffled
    M, K = x.shape
    N = wq.shape[0]

    if raw:
        from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
            run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250 as _run,
        )

        assert y is not None, "flydsl_raw needs a pre-allocated output (--output)"
        kernel_name, _src = _pick_flydsl_kernel(M, N, K)

        def _call(x_, wq_, xs_, ws_, y_):
            _run(x_, wq_, xs_, ws_, y_, kernel_name)

        return _call, (x, wq, xs, ws, y)

    from aiter import gemm_a8w8_blockscale_bpreshuffle

    # This build's op takes out= and writes into it, so hand it the same y the
    # gluon path reuses -- otherwise flydsl eats a fresh torch.empty per iter
    # (and, under graph capture, a fresh graph-pool buffer per unrolled call,
    # which changes its cache behaviour relative to gluon).
    def _call(x_, wq_, xs_, ws_, y_):
        gemm_a8w8_blockscale_bpreshuffle(x_, wq_, xs_, ws_, dtype, y_)

    return _call, (x, wq, xs, ws, y)


def make_call(backend, x, w_shuffled, x_scale, w_scale, y, dtype, args):
    """Return ``(fn, tensor_args)`` rather than a zero-arg closure.

    aiter's ``run_perftest`` defeats L2 by deep-copying the ARGUMENTS it is
    handed and cycling through the copies. A zero-arg closure has nothing to
    copy, so passing one in disables rotation *silently* -- the run still
    completes and reports a number, just a hot-cache one. Keeping the tensors
    out here as real arguments is what makes --method perftest match what
    aiter's own op_tests measure. Everything the kernel needs that is NOT an
    operand (dtype, group size, kernel_type) stays captured, so rotation copies
    exactly the buffers and nothing else.
    """
    if backend in ("flydsl", "flydsl_raw"):
        return _make_flydsl_launch(
            x, w_shuffled, x_scale, w_scale, y, dtype, args, raw=backend.endswith("raw")
        )

    def _call(x_, w_, xs_, ws_, y_):
        gemm_afp8wfp8_preshuffle(
            x_,
            w_,
            xs_,
            ws_,
            dtype,
            y_,
            x_scale_group_size=args.x_scale_group,
            is_x_scale_transposed=args.transpose_x_scale,
            backend=backend,
            kernel_type=args.kernel_type,
        )

    return _call, (x, w_shuffled, x_scale, w_scale, y)


def make_launch(backend, x, w_shuffled, x_scale, w_scale, y, dtype, args):
    """Zero-arg closure, for the graph-based methods that cannot use rotation."""
    fn, fargs = make_call(backend, x, w_shuffled, x_scale, w_scale, y, dtype, args)
    return lambda: fn(*fargs)


# --------------------------------------------------------------------------
# timing methods -- all return us/iter
# --------------------------------------------------------------------------
def time_cudagraph(fn, rep_ms):
    """triton do_bench_cudagraph. rep_ms is a time budget, not a replay count."""
    return triton.testing.do_bench_cudagraph(fn, rep=rep_ms) * 1e3


def rotate_sets(fn, fargs, cap):
    """Deep-copied operand sets, sized exactly the way run_perftest sizes them.

    Reuses aiter's own device_memory_profiling so --method event and --method
    perftest rotate over the same number of copies: the target working set is
    ``L2_cache_size * 64 * 128`` (32 GiB on gfx1250, which is aiming past the
    4 MB L2 at the far larger last-level cache), clamped to 90% of what is free
    after one iteration's scratch, then divided by the operand footprint. On
    m512/n7168/k16384 one set is 127 MiB, so the cap is what binds and we get
    ``cap`` sets.

    The list ends with the caller's own tensors so set 0 is the live operand
    set, not a copy -- same convention as run_perftest, and it means `del` on
    the returned list frees only the copies.
    """
    from aiter.test_common import device_memory_profiling

    gpu_id = torch.cuda.current_device()
    iter_used_memory, input_size, _, _ = device_memory_profiling(fn, *fargs)
    properties = torch.cuda.get_device_properties(gpu_id)
    free_memory = torch.cuda.mem_get_info(gpu_id)[0]
    cache_size = min(
        getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
        (free_memory - iter_used_memory + input_size) * 0.9,
    )
    n = int((max(cache_size, 0) + input_size - 1) // input_size)
    n = max(1, min(n, cap))
    return [copy.deepcopy(fargs) for _ in range(n - 1)] + [fargs]


def time_event(fn, fargs, n_replays=200, rotate=True):
    """Hand-rolled: capture n_replays into one graph, event-time the replay.

    Independent of triton entirely -- if this and time_cudagraph disagree, the
    do_bench_cudagraph number is the suspect one (its graph is sized off a
    host-bound eager estimate).

    rotate=True cycles the unrolled calls through rotate_sets() copies so the
    weights are never cache-resident, which is what makes this number
    comparable to --method perftest instead of being a hot-cache lower bound.
    Rotation is the right cold-cache mechanism *here* specifically because a
    graph is one timed window: a flush kernel interleaved between the calls
    (what triton's do_bench does eagerly) would land inside that window, and on
    these shapes zeroing triton's 256 MB flush buffer costs ~3x the GEMM -- you
    would be timing the flush and subtracting a second graph to get back to a
    number smaller than the noise on either. Rotation adds no device work at
    all; it only changes which addresses the same kernels read.

    The copies are allocated BEFORE capture on purpose. Anything allocated
    during capture comes out of the graph's private pool, so deep-copying
    inside the `with` block would both distort the pool and capture the copy
    kernels into the replay.

    --no-rotate keeps the old hot-cache behaviour, which is what you want when
    comparing against time_cudagraph (triton's do_bench_cudagraph does not
    flush either).
    """
    sets = rotate_sets(fn, fargs, n_replays) if rotate else [fargs]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(5):
            fn(*sets[i % len(sets)])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_replays):
            fn(*sets[i % len(sets)])
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / n_replays)
    del g, sets
    reclaim()
    return best


def reclaim():
    """Drop dead CUDA graphs and their private memory pools.

    Every do_bench_cudagraph call captures a fresh graph and never frees the
    pool until the graph is collected. Left alone across a sweep that is
    hundreds of live pools; the resulting fragmentation moves where the
    operands land and shifts memory-bound shapes by 2-3x between runs. This
    is the difference between a sweep that reproduces and one that doesn't.
    """
    gc.collect()
    torch.cuda.empty_cache()


_PROFILER_OK = None


def profiler_sees_kernels():
    """Does torch's profiler actually report GPU kernel activity on this box?

    On this gfx1250 stack it does not. roctracer's kernel-dispatch records come
    back with timestamps in a clock domain ~26.17 s ahead of CLOCK_MONOTONIC
    (the HIP *API* records and the SDMA copy records are in the right domain --
    only KernelExecution is skewed). kineto compares every GPU record against
    the CPU-side capture window, finds all of them past the end, and drops them:

        Processed 1203 GPU records
        Record counts: Out-of-range = 201, ...          # == every kernel

    The profile then contains cuda_runtime rows and nothing else, aiter's
    get_trace_perf logs "no valida data after post process!" and returns 0.0,
    and the harness divides by zero. Nothing is wrong with the kernel -- the
    graph-based methods time the same launch fine -- so probe once and fall
    back rather than reporting a zero.

    rocprofv3 gets the same kernels right (rocprofiler-sdk applies a correction
    the roctracer shim does not), so `rocprofv3 --kernel-trace` is the way to
    get a per-kernel breakdown until the stack is fixed.
    """
    global _PROFILER_OK
    if _PROFILER_OK is None:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            a @ a
        torch.cuda.synchronize()
        with tpf.profile(
            activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(5):
                a @ a
            torch.cuda.synchronize()
        _PROFILER_OK = any(
            str(e.device_type) == "DeviceType.CUDA" for e in prof.events()
        )
    return _PROFILER_OK


def time_perftest(fn, fargs, num_iters=100, rotate=True, timer="auto"):
    """aiter's own ``run_perftest``, not a local reimplementation.

    This is what every aiter op_test and mp_tuner reports, so a number from here
    is directly comparable to e.g.
    ``op_tests/test_gemm_a8w8_blockscale.py --flydsl``.

    The measurement itself is torch-profiler CUDA self-time, meaned over
    ``num_iters`` (minus one warm iter) with outlier rows dropped -- host
    dispatch is excluded without needing a graph. What a hand-rolled copy gets
    wrong is everything AROUND that: run_perftest auto-sizes ``num_rotate_args``
    from the L2 size and cycles the call through that many deep-copied operand
    sets, so the weights are never cache-resident. On gfx1250 the target is
    ``L2_cache_size * 64 * 128`` (32 GiB), capped at ``num_iters`` -- ~100 copies
    for these shapes. Measured on m512/n2048/k7168 flydsl that rotation is worth
    +2.3 us (12.8 -> 15.1), which is most of the gap between this harness and
    the op_test.

    rotate=False pins num_rotate_args=1 to reproduce the old hot-cache
    behaviour, for when you specifically want to compare against the graph
    methods rather than against aiter.

    timer="event" switches run_perftest to its use_cuda_event path, for stacks
    where the torch profiler reports no kernels (see profiler_sees_kernels).
    That is NOT the same measurement. run_perftest's event path brackets each
    *eager* call in cuda events, ignores rotate_args entirely (so it is
    hot-cache regardless of --rotate) and calls empty_cache() per iteration --
    it is a launch-bound wall-clock number, not device time. On m512/n7168/
    k16384 gluon it reads 92.8 us against 30.7 us from --method event and 30.0
    us min from rocprofv3, i.e. ~3x the actual kernel. It exists so the harness
    reports something honest instead of 0.00/inf; it is not an aiter op_test
    number. "auto" (the default) probes once and uses the profiler when it
    works.
    """
    from aiter.test_common import run_perftest

    kwargs = {"num_iters": num_iters, "num_warmup": 2}
    if not rotate:
        kwargs["num_rotate_args"] = 1
    if timer == "event" or (timer == "auto" and not profiler_sees_kernels()):
        kwargs["use_cuda_event"] = True
    _, us = run_perftest(fn, *fargs, **kwargs)
    return us


def per_kernel_split(fn, num_iters=50):
    """Per-kernel device-time attribution, for the breakdown table only.

    Separate from time_perftest because run_perftest returns a single scalar.
    This is a plain non-rotated profile: use it to see WHERE the time goes (e.g.
    the split-K reduce kernel next to the main GEMM), not as a headline number --
    its total will read low next to the rotated figure, for the reason above.

    Returns {} on a stack where the profiler drops every GPU record -- there is
    no event-based substitute for a per-kernel split, use rocprofv3.
    """
    if not profiler_sees_kernels():
        return {}
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(num_iters):
            fn()
        torch.cuda.synchronize()

    per_kernel = {}
    for e in prof.key_averages():
        if str(e.device_type) == "DeviceType.CUDA" and e.self_device_time_total > 0:
            per_kernel[e.key[:58]] = e.self_device_time_total / num_iters
    return per_kernel


def bench_one(M, N, K, args, dtype, label=""):
    """Time every requested backend on this shape, interleaved across trials."""
    launches, calls, kernels = {}, {}, {}
    for backend in args.backend:
        x, w_shuffled, x_scale, w_scale, y = generate_inputs(
            M, N, K, dtype, args.output, args.x_scale_group, args.transpose_x_scale
        )
        # Both forms of the same call: `calls` keeps the operands as arguments so
        # run_perftest can rotate them, `launches` is the zero-arg closure the
        # graph-capture methods need.
        calls[backend] = make_call(
            backend, x, w_shuffled, x_scale, w_scale, y, dtype, args
        )
        fn, fargs = calls[backend]
        launches[backend] = lambda fn=fn, fargs=fargs: fn(*fargs)
        launches[backend]()  # compile / autotune before any timing
    torch.cuda.synchronize()

    # Interleave: one trial of every backend, then repeat. Sequential blocks (or
    # separate processes) let clock drift land entirely on one backend -- that
    # alone moved flydsl between 7 and 15 us on this part.
    samples = {b: {m: [] for m in args.method} for b in args.backend}
    for _ in range(args.trials):
        for backend in args.backend:
            fn = launches[backend]
            if "cudagraph" in args.method:
                samples[backend]["cudagraph"].append(time_cudagraph(fn, args.rep_ms))
                reclaim()
            if "event" in args.method:
                cfn, cargs = calls[backend]
                samples[backend]["event"].append(
                    time_event(cfn, cargs, args.n_replays, rotate=args.rotate)
                )
            if "perftest" in args.method:
                cfn, cargs = calls[backend]
                samples[backend]["perftest"].append(
                    time_perftest(
                        cfn,
                        cargs,
                        args.perftest_iters,
                        rotate=args.rotate,
                        timer=args.perftest_timer,
                    )
                )
                reclaim()  # rotation allocates ~100 operand copies; free them
                kernels[backend] = per_kernel_split(fn)

    total_flops = 2 * M * N * K
    scale_n = (N + W_SCALE_N_GROUP - 1) // W_SCALE_N_GROUP
    scale_k = (K + W_SCALE_K_GROUP - 1) // W_SCALE_K_GROUP
    out_bytes = 2  # bf16 / fp16
    bytes_rw = (
        (M * K + N * K)  # fp8 operands
        + (M * (K // args.x_scale_group) + scale_n * scale_k)  # e8m0 scales
        + M * N * out_bytes
    )

    rows = []
    for backend in args.backend:
        by_method = {m: min(v) for m, v in samples[backend].items() if v}
        spread = {
            m: (max(v) - min(v)) / min(v) * 100 for m, v in samples[backend].items() if v
        }
        # Headline number: the primary method (first one requested).
        us = by_method[args.method[0]]
        per_iter_s = us * 1e-6
        rows.append(
            {
                "label": label,
                "backend": backend,
                "tag": f"afp8wfp8_preshuffle_M{M}_N{N}_K{K}_{args.dtype}_{backend}",
                "M": M,
                "N": N,
                "K": K,
                "us": us,
                "by_method": by_method,
                "spread": spread,
                "kernels": kernels.get(backend, {}),
                "tflops": total_flops / per_iter_s / 1e12,
                "gbs": bytes_rw / per_iter_s / 1e9,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m", type=int, default=64, help="M dimension")
    ap.add_argument(
        "--n", type=int, default=2048, help="N dimension (must be multiple of 16)"
    )
    ap.add_argument(
        "--k", type=int, default=7168, help="K dimension (must be multiple of 32)"
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument(
        "--output", action="store_true", default=True, help="pre-allocate output tensor"
    )
    ap.add_argument("--no-output", dest="output", action="store_false")
    ap.add_argument(
        "--x-scale-group",
        type=int,
        choices=[32, 128],
        default=128,
        help="K elements per activation scale (128=blockscale, 32=MX)",
    )
    ap.add_argument(
        "--transpose-x-scale",
        action="store_true",
        default=False,
        help="x_scale bytes are column-major (ATOM's per_1x128 layout)",
    )
    ap.add_argument(
        "--kernel-type", choices=["bandwidth_bound"], default="bandwidth_bound"
    )
    ap.add_argument(
        "--backend",
        default="gluon",
        help="comma-separated list of "
        + "/".join(BACKENDS)
        + "; gluon/triton run gemm_afp8wfp8_preshuffle, flydsl runs the "
        "gemm_a8w8_blockscale_bpreshuffle op ATOM dispatches to today, "
        "flydsl_raw calls that kernel without the aiter dispatch layer "
        "(flydsl* require --transpose-x-scale and --x-scale-group 128)",
    )
    ap.add_argument(
        "--method",
        default="cudagraph",
        help="comma-separated list of "
        + "/".join(METHODS)
        + " (or 'all'); the first is the headline number. cudagraph=triton "
        "do_bench_cudagraph (always hot-cache), event=hand-rolled graph "
        "capture + cuda events, cold-cache via operand rotation unless "
        "--no-rotate, perftest=torch-profiler device time (what aiter's own "
        "harness uses), also rotated",
    )
    ap.add_argument(
        "--rep-ms",
        type=float,
        default=20.0,
        help="do_bench_cudagraph time budget in MILLISECONDS (triton's `rep`). "
        "Below ~10 the graph gets too short and graph-launch overhead leaks "
        "into the per-iter number",
    )
    ap.add_argument(
        "--n-replays",
        type=int,
        default=200,
        help="calls unrolled into the graph for --method event",
    )
    ap.add_argument(
        "--perftest-iters",
        type=int,
        default=100,
        help="num_iters for --method perftest (aiter run_perftest). 100 matches "
        "TEST_NUM_ITERS in aiter's own gemm op_tests",
    )
    ap.add_argument(
        "--perftest-timer",
        choices=["auto", "profiler", "event"],
        default="auto",
        help="--method perftest only: how run_perftest measures. profiler=torch "
        "profiler device time (aiter's default). event=run_perftest's "
        "use_cuda_event path, for stacks where the profiler reports no kernels "
        "-- hot-cache and not op_test-comparable. auto probes the profiler once "
        "and falls back to event if it sees no GPU records",
    )
    ap.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--method perftest and --method event: rotate through cache-sized "
        "deep copies of the operands, as aiter's op_tests do (default). "
        "--no-rotate reuses one operand set, which is the hot-cache regime "
        "--method cudagraph measures (triton's do_bench_cudagraph does not "
        "flush or rotate either), and reads faster on these shapes",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=3,
        help="interleaved repeats per backend; the minimum is reported",
    )
    ap.add_argument(
        "--sweep",
        choices=sorted(DSV4_SHAPES),
        default=None,
        help="benchmark the DeepSeek-V4-Pro fp8 linear shapes for this config",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.backend = [b.strip() for b in args.backend.split(",") if b.strip()]
    for b in args.backend:
        if b not in BACKENDS:
            ap.error(f"unknown backend {b!r}; pick from {'/'.join(BACKENDS)}")
    args.method = (
        list(METHODS)
        if args.method == "all"
        else [m.strip() for m in args.method.split(",") if m.strip()]
    )
    for m in args.method:
        if m not in METHODS:
            ap.error(f"unknown method {m!r}; pick from {'/'.join(METHODS)}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_device("cuda")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(f"backends       : {', '.join(args.backend)}")
    print(f"triton variant : {args.kernel_type}")
    print(f"methods        : {', '.join(args.method)}  (headline: {args.method[0]})")
    print(f"trials         : {args.trials} interleaved, min reported")
    if "cudagraph" in args.method:
        print(f"rep budget     : {args.rep_ms} ms")
    print(f"dtype          : {args.dtype}")
    print("input dtype    : fp8 (e4m3)")
    print(
        f"x scale group  : 1x{args.x_scale_group} e8m0 (transposed={args.transpose_x_scale})"
    )
    print(f"w scale block  : ({W_SCALE_N_GROUP}, {W_SCALE_K_GROUP}) e8m0")
    print(f"pre-alloc out  : {args.output}")
    for b in args.backend:
        if b.startswith("flydsl"):
            name, src = _pick_flydsl_kernel(args.m, args.n, args.k)
            if not args.sweep:
                print(f"flydsl kernel  : {name}  ({src})")
            break
    print()

    shapes = (
        DSV4_SHAPES[args.sweep]
        if args.sweep
        else [("", args.n, args.k)]
    )
    rows = []
    for name, n, k in shapes:
        rows += bench_one(args.m, n, k, args, dtype, label=name)

    # First --backend is the baseline everything else is scored against. flydsl
    # is the incumbent, so `--backend flydsl,gluon` reads the way the work does:
    # vs>1 means gluon is ahead, and `gap us` is what gluon still owes.
    baseline = args.backend[0]
    base_us = {
        (r["label"], r["N"], r["K"]): r["us"] for r in rows if r["backend"] == baseline
    }

    lw = max([len(r["label"]) for r in rows] + [6])
    bw = max(len(b) for b in args.backend)
    vw = max(len(baseline) + 3, 8)
    extra = [m for m in args.method[1:]]
    head = (
        f"{'linear':<{lw}}  {'backend':<{bw}} {'N':>6} {'K':>6} "
        f"{'us':>9} {'+/-%':>6} {'vs ' + baseline:>{vw}} {'gap us':>8} "
        f"{'TFLOPS':>9} {'GB/s':>9}"
    )
    head += "".join(f" {m:>10}" for m in extra)
    if "perftest" in args.method and not profiler_sees_kernels():
        print(
            "WARNING: the torch profiler reports no GPU kernels on this stack, so\n"
            "  --method perftest is NOT device time. roctracer's kernel-dispatch\n"
            "  timestamps land ~26 s past kineto's capture window and every record\n"
            "  is dropped as out-of-range; aiter's get_trace_perf would return 0.0.\n"
            "  perftest fell back to run_perftest(use_cuda_event=True), which wraps\n"
            "  each eager call in cuda events and empty_cache()s per iteration, so\n"
            "  it is launch-bound and reads ~3x high on these shapes.\n"
            "  Trust --method event / --method cudagraph for the headline, and use\n"
            "  `rocprofv3 --kernel-trace` for a per-kernel device-time split.\n"
        )
    print(head)
    print("-" * len(head))
    for r in rows:
        ref = base_us.get((r["label"], r["N"], r["K"]))
        if r["backend"] == baseline or ref is None:
            vs, gap = f"{'--':>{vw}}", f"{'--':>8}"
        else:
            vs, gap = f"{ref / r['us']:>{vw}.3f}", f"{r['us'] - ref:>+8.2f}"
        line = (
            f"{r['label']:<{lw}}  {r['backend']:<{bw}} {r['N']:>6} {r['K']:>6} "
            f"{r['us']:>9.2f} {r['spread'][args.method[0]]:>6.1f} {vs} {gap} "
            f"{r['tflops']:>9.2f} {r['gbs']:>9.1f}"
        )
        line += "".join(f" {r['by_method'][m]:>10.2f}" for m in extra)
        print(line)
    print("-" * len(head))

    if args.sweep:
        totals = {
            b: sum(r["us"] for r in rows if r["backend"] == b) for b in args.backend
        }
        for b, tot in totals.items():
            tail = (
                ""
                if b == baseline
                else f"   {totals[baseline] / tot:.3f}x vs {baseline}"
                f"  ({tot - totals[baseline]:+.2f} us)"
            )
            print(f"total us ({b:<{bw}}): {tot:.2f}  (M={args.m}){tail}")

    if "perftest" in args.method and any(r["kernels"] for r in rows):
        print("\nper-kernel device us/iter (torch profiler):")
        seen = set()
        for r in rows:
            key = (r["label"], r["backend"])
            if not r["kernels"] or key in seen:
                continue
            seen.add(key)
            tag = f"{r['label']}/{r['backend']}" if r["label"] else r["backend"]
            print(f"  {tag}:")
            for k, v in sorted(r["kernels"].items(), key=lambda z: -z[1]):
                print(f"    {v:8.2f}  {k}")

    if len(args.backend) > 1 and not args.sweep:
        base = rows[0]
        print()
        for r in rows[1:]:
            print(
                f"{r['backend']} vs {base['backend']}: "
                f"{base['us'] / r['us']:.3f}x "
                f"({r['us']:.2f} vs {base['us']:.2f} us, "
                f"{r['us'] - base['us']:+.2f} us)"
            )
    print("_" * 80)


if __name__ == "__main__":
    main()
