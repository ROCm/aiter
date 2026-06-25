"""Shared helpers for FlyDSL MLA reduce test + benchmark harnesses."""

import torch
import aiter

from aiter.ops.flydsl.kernels.mla_reduce import compile_mla_reduce, select_tier


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    """Build reduce inputs for `num_tiles` tiles x `num_splits` splits, with M
    q-positions per token group (M = max_seqlen_q). For M > 1 each split owns M
    contiguous partial rows (one per q-position), and each tile's final q-range
    spans [tile*M, tile*M + M) — mirroring the get_mla_metadata_v1 layout where
    partial rows are reduce_partial_map.size(0) * max_seqlen_q (aiter/mla.py)."""
    g = torch.Generator(device=device).manual_seed(seed)
    num_partial_rows = num_tiles * num_splits * M
    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )
    reduce_indptr = torch.arange(
        0, num_tiles * num_splits + 1, num_splits, dtype=torch.int32, device=device
    )
    reduce_partial_map = (
        torch.arange(num_tiles * num_splits, dtype=torch.int32, device=device) * M
    )
    reduce_final_map = torch.stack(
        [
            torch.arange(num_tiles, dtype=torch.int32, device=device) * M,
            torch.arange(num_tiles, dtype=torch.int32, device=device) * M + M,
        ],
        dim=1,
    ).contiguous()
    final_output = torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.empty(num_tiles * M, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_degenerate_inputs(num_tiles, H, Dv, out_dtype, device="cuda", seed=0):
    """Degenerate n_splits=0 metadata for empty-tile guard regression."""
    g = torch.Generator(device=device).manual_seed(seed)
    partial_output = torch.randn(
        1, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = torch.randn(1, H, dtype=torch.float32, device=device, generator=g)
    reduce_indptr = torch.zeros(num_tiles + 1, dtype=torch.int32, device=device)
    reduce_partial_map = torch.zeros(1, dtype=torch.int32, device=device)
    reduce_final_map = torch.randint(
        1 << 20, 1 << 24, (num_tiles, 2), dtype=torch.int32, device=device, generator=g
    )
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


def torch_ref(partial_output, partial_lse, num_tiles, num_splits, H, Dv, out_dtype, M=1):
    """Vectorized online-softmax reduce reference (any max_seqlen_q M)."""
    po = partial_output.view(num_tiles, num_splits, M, H, Dv).double()
    pl = partial_lse.view(num_tiles, num_splits, M, H).double()
    max_lse = pl.max(dim=1, keepdim=True).values
    w = torch.exp(pl - max_lse)
    denom = w.sum(dim=1)
    num = (w.unsqueeze(-1) * po).sum(dim=1)
    out = (num / denom.unsqueeze(-1)).to(out_dtype)
    lse = (max_lse.squeeze(1) + torch.log(denom)).float()
    return out.reshape(num_tiles * M, H, Dv), lse.reshape(num_tiles * M, H)


def hip_ref(po, pl, indptr, fmap, pmap, num_tiles, H, Dv, out_dtype, M=1):
    """Reference output from HIP kn_mla_reduce_v1."""
    ref_out = torch.empty(num_tiles * M, H, Dv, dtype=out_dtype, device=po.device)
    ref_lse = torch.empty(num_tiles * M, H, dtype=torch.float32, device=po.device)
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, 0, ref_out, ref_lse)
    torch.cuda.synchronize()
    return ref_out, ref_lse


def make_runner(
    po, pl, indptr, pmap, fmap, fout, flse, H, Dv, out_dtype_str, output_lse, M=1
):
    """Precompile + bind args; return a zero-overhead closure for the timed loop."""
    num_tiles = fmap.shape[0]
    max_splits = torch.cuda.get_device_properties(0).multi_processor_count
    splits = int(indptr[1].item() - indptr[0].item())
    tier = select_tier(splits)
    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=tier,
        persistent=False,
        output_lse=output_lse,
        use_reduce_final_map=True,
    )
    head = (
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        int(fout.stride(0)),
        int(fout.stride(1)),
        int(max_splits),
        int(num_tiles),
        int(M),
    )

    def run():
        kernel(*head, torch.cuda.current_stream())

    return run


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    """CUDA-graph replay timing; returns ms/iter."""
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
    return start.elapsed_time(end) / num_iters
