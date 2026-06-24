"""Standalone benchmark + correctness check for kn_mla_reduce_v1 (csrc/kernels/mla/reduce.cu).

Builds a uniform synthetic workload directly (full control over num_splits) so we can
isolate the reduce epilogue and measure latency / achieved HBM bandwidth / roofline.

Work decomposition (matches the kernel): a "work item" = (head, q-pos-group, reduce-tile).
Here NTG=1 (decode, max_seqlen_q=1), so total WGs = H * num_reduce_tile.
Each work item online-softmax-combines `num_splits` partials of one head into one Dv row.
"""

import argparse
import torch
import aiter
from aiter.test_common import run_perftest


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    num_partial_rows = num_tiles * num_splits

    # partial buffers (stage-1 outputs) — must be fp32
    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    # LSEs in a realistic range so exp() doesn't blow up
    partial_lse = (
        torch.randn(num_partial_rows, H, dtype=torch.float32, device=device, generator=g)
        * 2.0
    )

    # CSR over splits: tile t owns rows [t*num_splits, (t+1)*num_splits)
    reduce_indptr = torch.arange(
        0, num_partial_rows + 1, num_splits, dtype=torch.int32, device=device
    )
    reduce_partial_map = torch.arange(
        num_partial_rows, dtype=torch.int32, device=device
    )
    # explicit final map {q_start,q_end} = {t, t+1}  (one seq position per tile)
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
    # vectorized reference: rows reshape [tiles, splits, H, Dv]
    po = partial_output.view(num_tiles, num_splits, H, Dv).double()
    pl = partial_lse.view(num_tiles, num_splits, H).double()
    max_lse = pl.max(dim=1, keepdim=True).values  # [tiles,1,H]
    w = torch.exp(pl - max_lse)  # [tiles,splits,H]
    denom = w.sum(dim=1)  # [tiles,H]
    num = (w.unsqueeze(-1) * po).sum(dim=1)  # [tiles,H,Dv]
    out = (num / denom.unsqueeze(-1)).to(out_dtype)
    lse = (max_lse.squeeze(1) + torch.log(denom)).float()
    return out, lse


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--tiles", type=int, default=256, help="num_reduce_tile (≈ batch)")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp8", "fp8_e5m2"],
        default="bf16",
        help="final_output dtype",
    )
    ap.add_argument(
        "--partial-dtype",
        choices=["fp32", "fp8", "fp8_e5m2"],
        default="fp32",
        help="partial_output/partial_lse dtype (kernel requires fp32)",
    )
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    _dt = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fnuz,
        "fp8_e5m2": torch.float8_e5m2fnuz,
        "fp32": torch.float32,
    }
    out_dtype = _dt[args.dtype]
    H, Dv, T, S = args.H, args.Dv, args.tiles, args.splits

    (po, pl, indptr, fmap, pmap, fout, flse) = build_inputs(T, S, H, Dv, out_dtype)

    if args.partial_dtype != "fp32":
        po = po.to(_dt[args.partial_dtype])
        pl = pl.to(_dt[args.partial_dtype])

    def run():
        aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, 1, 0, fout, flse)

    run()
    torch.cuda.synchronize()

    if args.check:
        ref_out, ref_lse = torch_ref(po, pl, T, S, H, Dv, out_dtype)
        od = (fout.float() - ref_out.float()).abs()
        ld = (flse - ref_lse).abs()
        print(
            f"[check] out max_abs_err={od.max().item():.3e} mean={od.mean().item():.3e} | "
            f"lse max_abs_err={ld.max().item():.3e}"
        )

    # Primary: device-only time from aiter's profiler-based run_perftest
    # (excludes per-call Python host overhead). Returns us/iter.
    _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
    # Cross-check: CUDA-graph replay (per-iter launch overhead amortized).
    graph_us = bench_cudagraph(run) * 1e3

    # traffic model (the byte floor for this reduction)
    bytes_partial_o = T * S * H * Dv * 4
    bytes_partial_lse = T * S * H * 4
    bytes_final_o = T * H * Dv * out_dtype.itemsize
    bytes_final_lse = T * H * 4
    # gather map staged to LDS once per tile per (loaded by kernel from gmem)
    total_bytes = bytes_partial_o + bytes_partial_lse + bytes_final_o + bytes_final_lse
    gbps = total_bytes / (kernel_us * 1e-6) / 1e9

    path = "massive" if S >= 4 else "simple"
    print(
        f"H={H} Dv={Dv} tiles={T} splits={S} dtype={args.dtype} path={path} "
        f"work_items={H*T}"
    )
    print(
        f"  kernel = {kernel_us:.2f} us (graph {graph_us:.2f} us) | "
        f"traffic = {total_bytes/1e6:.1f} MB "
        f"({bytes_partial_o/1e6:.0f} MB partial_O read) | "
        f"achieved BW = {gbps:.0f} GB/s"
    )
    # MI300X HBM3 peak ~5.3 TB/s
    print(f"  HBM utilization vs 5300 GB/s peak = {gbps/5300*100:.1f}%")


if __name__ == "__main__":
    main()
