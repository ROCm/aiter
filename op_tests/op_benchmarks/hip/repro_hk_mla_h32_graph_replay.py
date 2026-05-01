import argparse
import math

import torch

import aiter
from aiter import dtypes
from op_tests.op_benchmarks.hip.compare_hk_mla_h32_reference import (
    NHEAD,
    NHEAD_KV,
    PAGE_SIZE,
    QK_HEAD_DIM,
    V_HEAD_DIM,
    Shape,
    error_metrics,
    make_metadata,
    reference_mla,
)


def alloc_outputs(shape: Shape, reduce_partial_map: torch.Tensor):
    split = torch.empty(
        (reduce_partial_map.size(0) * shape.qlen, 1, NHEAD, V_HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    lse = torch.empty(
        (reduce_partial_map.size(0) * shape.qlen, 1, NHEAD, 1),
        dtype=torch.float32,
        device="cuda",
    )
    out = torch.empty(
        (shape.total_q, NHEAD, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    return split, lse, out


def fill_metadata(shape: Shape, tensors):
    (
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        kv_indices,
        work_meta,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = make_metadata(shape)
    tensors["qo_indptr"].copy_(qo_indptr)
    tensors["kv_indptr"].copy_(kv_indptr)
    tensors["kv_last_page_lens"].copy_(kv_last_page_lens)
    tensors["kv_indices"][: kv_indices.numel()].copy_(kv_indices)
    tensors["work_meta"].copy_(work_meta)
    tensors["work_indptr"].copy_(work_indptr)
    tensors["work_info"].copy_(work_info)
    tensors["reduce_indptr"].copy_(reduce_indptr)
    tensors["reduce_final_map"].copy_(reduce_final_map)
    tensors["reduce_partial_map"].copy_(reduce_partial_map)


def make_metadata_from_lens(lens: torch.Tensor, qlen: int):
    batch = int(lens.numel())
    cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    max_split_per_batch = min((cu_num + batch - 1) // batch, 32)
    max_ctx = int(lens.max().item())
    sizes = aiter.get_mla_metadata_info_v1(
        batch,
        qlen,
        NHEAD,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )
    ((wms, wmt), (wis, wit), (wss, wst), (ris, rit), (rfms, rfmt), (rpms, rpmt)) = sizes
    work_meta = torch.empty(wms, dtype=wmt, device="cuda")
    work_indptr = torch.empty(wis, dtype=wit, device="cuda")
    work_info = torch.empty(wss, dtype=wst, device="cuda")
    reduce_indptr = torch.empty(ris, dtype=rit, device="cuda")
    reduce_final_map = torch.empty(rfms, dtype=rfmt, device="cuda")
    reduce_partial_map = torch.empty(rpms, dtype=rpmt, device="cuda")
    qo_indptr = torch.arange(0, batch * qlen + 1, qlen, dtype=torch.int32, device="cuda")
    kv_indptr = torch.empty((batch + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(lens.to(torch.int32), dim=0)
    kv_last_page_lens = torch.ones((batch,), dtype=torch.int32, device="cuda")
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        NHEAD,
        NHEAD_KV,
        False,
        work_meta,
        work_info,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
    )
    return (
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        torch.arange(int(lens.sum().item()), dtype=torch.int32, device="cuda"),
        work_meta,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_ctx,
    )


def fill_metadata_from_lens(lens: torch.Tensor, qlen: int, tensors):
    meta = make_metadata_from_lens(lens, qlen)
    tensors["qo_indptr"].copy_(meta[0])
    tensors["kv_indptr"].copy_(meta[1])
    tensors["kv_last_page_lens"].copy_(meta[2])
    tensors["kv_indices"][: meta[3].numel()].copy_(meta[3])
    tensors["work_meta"].copy_(meta[4])
    tensors["work_indptr"].copy_(meta[5])
    tensors["work_info"].copy_(meta[6])
    tensors["reduce_indptr"].copy_(meta[7])
    tensors["reduce_final_map"].copy_(meta[8])
    tensors["reduce_partial_map"].copy_(meta[9])


def run_hk(q, kv, shape: Shape, tensors, use_long_kernel: bool):
    split, lse, out = alloc_outputs(shape, tensors["reduce_partial_map"])
    hk_decode = aiter.hk_mla_decode_fwd_long if use_long_kernel else aiter.hk_mla_decode_fwd
    hk_decode(
        q,
        kv,
        tensors["qo_indptr"],
        tensors["kv_indptr"],
        tensors["kv_indices"],
        tensors["kv_last_page_lens"],
        tensors["work_indptr"],
        tensors["work_info"],
        shape.qlen,
        1.0 / math.sqrt(QK_HEAD_DIM),
        split,
        lse,
        out,
    )
    aiter.mla_reduce_v1(
        split,
        lse,
        tensors["reduce_indptr"],
        tensors["reduce_final_map"],
        tensors["reduce_partial_map"],
        shape.qlen,
        out,
        None,
    )
    return out


def run_asm(q, kv, shape: Shape, tensors):
    split, lse, out = alloc_outputs(shape, tensors["reduce_partial_map"])
    one = torch.ones((1,), dtype=torch.float32, device="cuda")
    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv,
        tensors["qo_indptr"],
        tensors["kv_indptr"],
        tensors["kv_indices"],
        tensors["kv_last_page_lens"],
        None,
        tensors["work_meta"],
        tensors["work_indptr"],
        tensors["work_info"],
        shape.qlen,
        PAGE_SIZE,
        NHEAD_KV,
        1.0 / math.sqrt(QK_HEAD_DIM),
        split,
        lse,
        out,
        None,
        one,
        one,
    )
    aiter.mla_reduce_v1(
        split,
        lse,
        tensors["reduce_indptr"],
        tensors["reduce_final_map"],
        tensors["reduce_partial_map"],
        shape.qlen,
        out,
        None,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--raw-batch", type=int, default=0)
    parser.add_argument("--capture-ctx", type=int, default=8)
    parser.add_argument("--replay-ctx", type=int, default=2048)
    parser.add_argument("--pad-ctx", type=int, default=4)
    parser.add_argument("--hk-long-kernel", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_default_device("cuda")

    capture = Shape(args.batch, args.capture_ctx, 4)
    replay = Shape(args.batch, args.replay_ctx, 4)
    replay_meta = make_metadata(replay)
    tensors = {
        "qo_indptr": replay_meta[0].clone(),
        "kv_indptr": replay_meta[1].clone(),
        "kv_last_page_lens": replay_meta[2].clone(),
        "kv_indices": replay_meta[3].clone(),
        "work_meta": replay_meta[4].clone(),
        "work_indptr": replay_meta[5].clone(),
        "work_info": replay_meta[6].clone(),
        "reduce_indptr": replay_meta[7].clone(),
        "reduce_final_map": replay_meta[8].clone(),
        "reduce_partial_map": replay_meta[9].clone(),
    }

    q = torch.randn(
        (replay.total_q, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).to(dtypes.fp8)
    kv = torch.randn(
        (replay.batch * replay.ctx, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtypes.fp8)

    graph_split, graph_lse, graph_out = alloc_outputs(replay, tensors["reduce_partial_map"])

    def graph_hk_total():
        hk_decode = aiter.hk_mla_decode_fwd_long if args.hk_long_kernel else aiter.hk_mla_decode_fwd
        hk_decode(
            q,
            kv,
            tensors["qo_indptr"],
            tensors["kv_indptr"],
            tensors["kv_indices"],
            tensors["kv_last_page_lens"],
            tensors["work_indptr"],
            tensors["work_info"],
            replay.qlen,
            1.0 / math.sqrt(QK_HEAD_DIM),
            graph_split,
            graph_lse,
            graph_out,
        )
        aiter.mla_reduce_v1(
            graph_split,
            graph_lse,
            tensors["reduce_indptr"],
            tensors["reduce_final_map"],
            tensors["reduce_partial_map"],
            replay.qlen,
            graph_out,
            None,
        )

    fill_metadata(capture, tensors)
    for _ in range(3):
        graph_hk_total()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hk_total()

    if args.raw_batch > 0 and args.raw_batch < args.batch:
        replay_lens = torch.full(
            (args.batch,), args.pad_ctx, dtype=torch.int32, device="cuda"
        )
        replay_lens[: args.raw_batch] = args.replay_ctx
        fill_metadata_from_lens(replay_lens, replay.qlen, tensors)
        compare_tokens = args.raw_batch * replay.qlen
    else:
        fill_metadata(replay, tensors)
        compare_tokens = replay.total_q
    graph.replay()
    torch.cuda.synchronize()

    eager_hk = run_hk(q, kv, replay, tensors, args.hk_long_kernel)
    eager_asm = run_asm(q, kv, replay, tensors)
    ref = reference_mla(q, kv, replay)
    graph_cmp = graph_out[:compare_tokens]
    eager_hk_cmp = eager_hk[:compare_tokens]
    eager_asm_cmp = eager_asm[:compare_tokens]
    ref_cmp = ref[:compare_tokens]
    torch.cuda.synchronize()

    for name, out in (
        ("graph_hk/ref", graph_cmp),
        ("eager_hk/ref", eager_hk_cmp),
        ("asm/ref", eager_asm_cmp),
        ("graph_hk/eager_hk", graph_cmp),
        ("graph_hk/asm", graph_cmp),
    ):
        other = ref_cmp
        if name == "graph_hk/eager_hk":
            other = eager_hk_cmp
        elif name == "graph_hk/asm":
            other = eager_asm_cmp
        metrics = error_metrics(other, out)
        print(
            f"{name}: cos={metrics['cos']:.6f} rmse={metrics['rmse']:.6f} "
            f"max_abs={metrics['max_abs']:.6f} mean_abs={metrics['mean_abs']:.6f}"
        )


if __name__ == "__main__":
    main()
