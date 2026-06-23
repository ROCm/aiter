"""Standalone correctness + bench harness for the FlyDSL MLA reduce kernel.

Reuses the input builder / torch reference shape from bench_mla_reduce_standalone.py.
Runs the compiled FlyDSL kernel directly (bypasses aiter's flydsl __init__ version gate).
"""

import argparse
import torch
import aiter

from aiter.ops.flydsl.kernels.mla_reduce import compile_mla_reduce, select_tier
from aiter.test_common import run_perftest


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    num_partial_rows = num_tiles * num_splits
    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(num_partial_rows, H, dtype=torch.float32, device=device, generator=g)
        * 2.0
    )
    reduce_indptr = torch.arange(
        0, num_partial_rows + 1, num_splits, dtype=torch.int32, device=device
    )
    reduce_partial_map = torch.arange(
        num_partial_rows, dtype=torch.int32, device=device
    )
    reduce_final_map = torch.stack(
        [
            torch.arange(num_tiles, dtype=torch.int32, device=device),
            torch.arange(1, num_tiles + 1, dtype=torch.int32, device=device),
        ],
        dim=1,
    ).contiguous()
    final_output = torch.empty(num_tiles, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.empty(num_tiles, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def torch_ref(partial_output, partial_lse, num_tiles, num_splits, H, Dv, out_dtype):
    po = partial_output.view(num_tiles, num_splits, H, Dv).double()
    pl = partial_lse.view(num_tiles, num_splits, H).double()
    max_lse = pl.max(dim=1, keepdim=True).values
    w = torch.exp(pl - max_lse)
    denom = w.sum(dim=1)
    num = (w.unsqueeze(-1) * po).sum(dim=1)
    out = (num / denom.unsqueeze(-1)).to(out_dtype)
    lse = (max_lse.squeeze(1) + torch.log(denom)).float()
    return out, lse


def hip_ref(po, pl, indptr, fmap, pmap, num_tiles, H, Dv, out_dtype):
    """Reference output from the HIP kn_mla_reduce_v1 kernel (the kernel this
    FlyDSL port replaces). Same input buffers as the FlyDSL launch."""
    ref_out = torch.empty(num_tiles, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.empty(num_tiles, H, dtype=torch.float32, device=po.device)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, 1, 0, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, out_dtype_str, output_lse):
    """Precompile + bind args; return a zero-overhead closure for the timed loop."""
    num_tiles = fmap.shape[0]
    max_splits = torch.cuda.get_device_properties(0).multi_processor_count
    splits = pmap.numel() // num_tiles
    tier = select_tier(splits)
    kernel = compile_mla_reduce(
        H=H, Dv=Dv, out_dtype=out_dtype_str, tier=tier,
        persistent=False, output_lse=output_lse, use_reduce_final_map=True,
    )
    head = (
        po, pl, indptr, pmap, fmap, fout, flse,
        int(fout.stride(0)), int(fout.stride(1)),
        int(max_splits), int(num_tiles),
    )

    def run():
        # Read the stream at call time so CUDA-graph capture (side stream) sees
        # the launch — binding it once would pin the kernel to the capture stream.
        kernel(*head, torch.cuda.current_stream())

    return run


def run_flydsl(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, out_dtype_str, output_lse):
    make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, out_dtype_str, output_lse)()


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    """CUDA-graph + cuda.Event cross-check: capture num_iters launches into one
    graph and time a single replay, so per-call Python launch overhead is removed.
    Returns ms/iter. Modeled on _opus_run_perftest in opus_gemm_tune.py."""
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(num_iters):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / num_iters  # ms/iter


def check_one(H, Dv, T_, S, dtype_str, output_lse, atol=None, ltol=1e-3):
    """Run one config, compare vs the HIP kn_mla_reduce_v1 kernel, return
    (ok, out_err, lse_err). Both kernels emit the same output dtype, so the
    output may differ by one ULP of that dtype from independent rounding of
    near-equal fp32 results — tolerate that (bf16 ULP ≈ 6.3e-2, fp16 ≈ 1e-3)."""
    out_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    if atol is None:
        atol = 6.3e-2 if dtype_str == "bf16" else 2e-3
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(T_, S, H, Dv, out_dtype)
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dtype_str, output_lse)
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref(po, pl, indptr, fmap, pmap, T_, H, Dv, out_dtype)
    out_err = (fout.float() - ref_out.float()).abs().max().item()
    lse_err = (flse - ref_lse).abs().max().item() if output_lse else 0.0
    ok = (out_err <= atol) and (lse_err <= ltol)
    return ok, out_err, lse_err


def run_matrix():
    shapes = [(128, 512), (16, 512), (128, 128)]
    dtypes = ["bf16", "fp16"]
    splits = [2, 3, 4, 8, 16, 64, 256, 300]
    n_pass = n_fail = 0
    for H, Dv in shapes:
        for dt in dtypes:
            for S in splits:
                ok, oe, le = check_one(H, Dv, 4, S, dt, output_lse=True)
                tag = "PASS" if ok else "FAIL"
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                print(
                    f"[{tag}] H={H:3d} Dv={Dv} dt={dt} splits={S:3d} "
                    f"out_err={oe:.2e} lse_err={le:.2e}"
                )
    print(f"\n=== {n_pass} passed, {n_fail} failed ===")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--lse", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--matrix", action="store_true", help="run full correctness sweep")
    args = ap.parse_args()

    if args.matrix:
        ok = run_matrix()
        raise SystemExit(0 if ok else 1)

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    H, Dv, T_, S = args.H, args.Dv, args.tiles, args.splits

    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(T_, S, H, Dv, out_dtype)
    fout.zero_()
    flse.zero_()

    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse
    )

    run()
    torch.cuda.synchronize()

    ref_out, ref_lse = hip_ref(po, pl, indptr, fmap, pmap, T_, H, Dv, out_dtype)
    od = (fout.float() - ref_out.float()).abs()
    print(
        f"[check] H={H} Dv={Dv} tiles={T_} splits={S} dtype={args.dtype} "
        f"vs HIP out max_abs_err={od.max().item():.3e} mean={od.mean().item():.3e}"
    )
    if args.lse:
        ld = (flse - ref_lse).abs()
        print(f"[check] lse max_abs_err={ld.max().item():.3e}")

    if args.bench:
        # Primary: device-only time from aiter's profiler-based run_perftest
        # (excludes the ~230us per-call Python host overhead). Returns us/iter.
        _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
        # Cross-check: CUDA-graph replay (per-iter launch overhead amortized).
        graph_us = bench_cudagraph(run) * 1e3
        bytes_partial_o = T_ * S * H * Dv * 4
        bytes_partial_lse = T_ * S * H * 4
        bytes_final_o = T_ * H * Dv * out_dtype.itemsize
        bytes_final_lse = T_ * H * 4
        total = bytes_partial_o + bytes_partial_lse + bytes_final_o + bytes_final_lse
        gbps = total / (kernel_us * 1e-6) / 1e9
        path = "massive" if S >= 4 else "simple"
        print(
            f"[bench] H={H} Dv={Dv} tiles={T_} splits={S} path={path} "
            f"kernel={kernel_us:.1f}us graph={graph_us:.1f}us BW={gbps:.0f}GB/s "
            f"({gbps/5300*100:.0f}% of 5.3TB/s)"
        )


if __name__ == "__main__":
    main()
