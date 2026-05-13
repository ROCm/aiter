# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import math
import os
from dataclasses import dataclass
from statistics import median
from typing import Callable

os.environ.setdefault("AITER_ENABLE_EXPERIMENTAL", "1")

import torch

import aiter
from aiter import dtypes


NHEAD = 128
NHEAD_KV = 1
KV_LORA = 512
QK_ROPE = 64
QK_HEAD_DIM = KV_LORA + QK_ROPE
V_HEAD_DIM = KV_LORA


@dataclass
class MlaShape:
    batch_size: int
    ctx_len: int
    decode_qlen: int
    page_size: int

    @property
    def total_q(self) -> int:
        return self.batch_size * self.decode_qlen

    @property
    def total_kv_tokens(self) -> int:
        return self.batch_size * self.ctx_len

    @property
    def num_pages_per_seq(self) -> int:
        return math.ceil(self.ctx_len / self.page_size)

    @property
    def num_pages(self) -> int:
        return self.batch_size * self.num_pages_per_seq


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def make_indptr(shape: MlaShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_indptr = torch.arange(
        0,
        shape.total_q + 1,
        shape.decode_qlen,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.arange(
        0,
        shape.num_pages + 1,
        shape.num_pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    last_page_len = shape.ctx_len % shape.page_size
    if last_page_len == 0:
        last_page_len = shape.page_size
    kv_last_page_lens = torch.full(
        (shape.batch_size,), last_page_len, dtype=torch.int32, device="cuda"
    )
    return qo_indptr, kv_indptr, kv_last_page_lens


def make_metadata(
    shape: MlaShape,
    max_split_per_batch: int,
    force_hk_native_metadata: bool = False,
    cap_split_to_cu_per_batch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_force_native = os.environ.get("AITER_HK_MLA_FORCE_NATIVE_METADATA")
    if force_hk_native_metadata:
        os.environ["AITER_HK_MLA_FORCE_NATIVE_METADATA"] = "1"
    else:
        os.environ.pop("AITER_HK_MLA_FORCE_NATIVE_METADATA", None)

    cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if cap_split_to_cu_per_batch:
        max_split_per_batch = min(
            (cu_num + shape.batch_size - 1) // shape.batch_size,
            max_split_per_batch,
        )

    try:
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = aiter.get_mla_metadata_info_v1(
            shape.batch_size,
            shape.decode_qlen,
            NHEAD,
            dtypes.fp8,
            dtypes.fp8,
            is_sparse=False,
            fast_mode=True,
            num_kv_splits=max_split_per_batch,
            intra_batch_mode=False,
        )

        work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device="cuda"
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        qo_indptr, kv_indptr, kv_last_page_lens = make_indptr(shape)
        aiter.get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            NHEAD // NHEAD_KV,
            NHEAD_KV,
            False,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=shape.page_size,
            kv_granularity=max(shape.page_size, 16),
            max_seqlen_qo=shape.decode_qlen,
            uni_seqlen_qo=shape.decode_qlen,
            fast_mode=True,
            max_split_per_batch=max_split_per_batch,
            dtype_q=dtypes.fp8,
            dtype_kv=dtypes.fp8,
        )
    finally:
        if old_force_native is None:
            os.environ.pop("AITER_HK_MLA_FORCE_NATIVE_METADATA", None)
        else:
            os.environ["AITER_HK_MLA_FORCE_NATIVE_METADATA"] = old_force_native

    return (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )


def make_inputs(
    shape: MlaShape,
    shuffle_pages: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    qo_indptr, kv_indptr, kv_last_page_lens = make_indptr(shape)
    if shuffle_pages:
        kv_indices = torch.randperm(shape.num_pages, dtype=torch.int32, device="cuda")
    else:
        kv_indices = torch.arange(shape.num_pages, dtype=torch.int32, device="cuda")

    q = torch.randn(
        (shape.total_q, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).to(dtypes.fp8)
    kv_buffer = torch.randn(
        (shape.num_pages, shape.page_size, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtypes.fp8)
    return q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens


def alloc_outputs(
    shape: MlaShape,
    reduce_partial_map: torch.Tensor,
    nhead: int,
    output_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if output_tokens is None:
        output_tokens = shape.total_q
    split_output = torch.empty(
        (reduce_partial_map.size(0) * shape.decode_qlen, 1, nhead, V_HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    split_lse = torch.empty(
        (reduce_partial_map.size(0) * shape.decode_qlen, 1, nhead, 1),
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty(
        (output_tokens, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    return split_output, split_lse, output


def fold_q_for_gfx950_asm(q: torch.Tensor, shape: MlaShape) -> torch.Tensor:
    if NHEAD == 32:
        return q
    fold_factor = NHEAD // 16
    return (
        q.reshape(
            shape.batch_size,
            shape.decode_qlen,
            fold_factor,
            16,
            QK_HEAD_DIM,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(shape.total_q * fold_factor, 16, QK_HEAD_DIM)
    )


def unfold_o_from_gfx950_asm(o: torch.Tensor, shape: MlaShape) -> torch.Tensor:
    if NHEAD == 32:
        return o
    fold_factor = NHEAD // 16
    return (
        o.reshape(shape.batch_size, fold_factor, shape.decode_qlen, 16, V_HEAD_DIM)
        .permute(0, 2, 1, 3, 4)
        .reshape(shape.total_q, NHEAD, V_HEAD_DIM)
        .contiguous()
    )


def time_cuda_us(fn: Callable[[], None], warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return float(sum(samples) / len(samples)), float(median(samples))


def cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x_f = x.float().flatten()
    y_f = y.float().flatten()
    denom = torch.clamp((x_f * x_f + y_f * y_f).sum(), min=1e-12)
    return float(1 - 2 * (x_f * y_f).sum() / denom)


def metadata_stats(
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    reduce_indptr: torch.Tensor,
) -> dict[str, float | int]:
    work_indptr_cpu = work_indptr.cpu()
    work_info_cpu = work_info_set.cpu()
    reduce_indptr_cpu = reduce_indptr.cpu()

    work_count = int(work_indptr_cpu[-1].item())
    works_per_cu = work_indptr_cpu[1:] - work_indptr_cpu[:-1]
    if work_count > 0:
        live_work_info = work_info_cpu[:work_count]
        kv_spans = live_work_info[:, 5] - live_work_info[:, 4]
        qo_spans = live_work_info[:, 3] - live_work_info[:, 2]
    else:
        kv_spans = torch.zeros((1,), dtype=torch.int32)
        qo_spans = torch.zeros((1,), dtype=torch.int32)

    reduce_counts = reduce_indptr_cpu[1:] - reduce_indptr_cpu[:-1]
    partial_tiles = reduce_counts[reduce_counts > 0]
    return {
        "work_count": work_count,
        "active_cu": int((works_per_cu > 0).sum().item()),
        "max_works_per_cu": int(works_per_cu.max().item()),
        "avg_works_per_cu": float(works_per_cu.float().mean().item()),
        "max_kv_span": int(kv_spans.max().item()),
        "avg_kv_span": float(kv_spans.float().mean().item()),
        "max_qo_span": int(qo_spans.max().item()),
        "partial_tile_count": int(partial_tiles.numel()),
        "max_partials_per_tile": int(partial_tiles.max().item()) if partial_tiles.numel() else 0,
        "avg_partials_per_tile": (
            float(partial_tiles.float().mean().item()) if partial_tiles.numel() else 0.0
        ),
    }


def benchmark_shape(args: argparse.Namespace, shape: MlaShape) -> dict[str, object]:
    q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens = make_inputs(
        shape, args.shuffle_pages
    )
    (
        asm_work_meta_data,
        asm_work_indptr,
        asm_work_info_set,
        asm_reduce_indptr,
        asm_reduce_final_map,
        asm_reduce_partial_map,
    ) = make_metadata(
        shape,
        args.max_split_per_batch,
        cap_split_to_cu_per_batch=not args.no_split_cap,
    )

    sm_scale = 1.0 / math.sqrt(QK_HEAD_DIM)
    q_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    kv_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

    q_asm = fold_q_for_gfx950_asm(q, shape)
    asm_nhead = 16 if NHEAD == 128 else NHEAD
    asm_split, asm_lse, asm_out = alloc_outputs(
        shape, asm_reduce_partial_map, asm_nhead, output_tokens=q_asm.size(0)
    )

    def asm_stage() -> None:
        aiter.mla_decode_stage1_asm_fwd(
            q_asm,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            None,
            asm_work_meta_data,
            asm_work_indptr,
            asm_work_info_set,
            shape.decode_qlen,
            shape.page_size,
            NHEAD_KV,
            sm_scale,
            asm_split,
            asm_lse,
            asm_out,
            None,
            q_scale,
            kv_scale,
        )

    def asm_reduce() -> None:
        aiter.mla_reduce_v1(
            asm_split,
            asm_lse,
            asm_reduce_indptr,
            asm_reduce_final_map,
            asm_reduce_partial_map,
            shape.decode_qlen,
            asm_out,
            None,
        )

    def asm_total() -> None:
        asm_stage()
        asm_reduce()

    asm_stage()
    asm_reduce()
    torch.cuda.synchronize()
    asm_decode_avg, asm_decode_p50 = time_cuda_us(asm_stage, args.warmup, args.iters)
    asm_stage()
    asm_reduce_avg, asm_reduce_p50 = time_cuda_us(asm_reduce, args.warmup, args.iters)
    asm_total_avg, asm_total_p50 = time_cuda_us(asm_total, args.warmup, args.iters)

    row: dict[str, object] = {
        "batch": shape.batch_size,
        "ctx": shape.ctx_len,
        "decode_qlen": shape.decode_qlen,
        "page_size": shape.page_size,
        "total_kv": shape.total_kv_tokens,
        "asm_decode_avg_us": asm_decode_avg,
        "asm_decode_p50_us": asm_decode_p50,
        "asm_reduce_avg_us": asm_reduce_avg,
        "asm_reduce_p50_us": asm_reduce_p50,
        "asm_total_avg_us": asm_total_avg,
        "asm_total_p50_us": asm_total_p50,
        "hk_supported": shape.page_size == 1,
        "hk_decode_avg_us": None,
        "hk_decode_p50_us": None,
        "hk_reduce_avg_us": None,
        "hk_reduce_p50_us": None,
        "hk_total_avg_us": None,
        "hk_total_p50_us": None,
        "hk_speedup_p50": None,
        "cos_diff": None,
        "max_abs_diff": None,
    }
    if args.metadata_stats:
        row.update(
            {
                f"meta_{key}": value
                for key, value in metadata_stats(
                    asm_work_indptr, asm_work_info_set, asm_reduce_indptr
                ).items()
            }
        )

    (
        _hk_work_meta_data,
        hk_work_indptr,
        hk_work_info_set,
        hk_reduce_indptr,
        hk_reduce_final_map,
        hk_reduce_partial_map,
    ) = make_metadata(
        shape,
        args.max_split_per_batch,
        force_hk_native_metadata=True,
        cap_split_to_cu_per_batch=not args.no_split_cap,
    )
    hk_split, hk_lse, hk_out = alloc_outputs(shape, hk_reduce_partial_map, NHEAD)

    def hk_stage() -> None:
        hk_decode = (
            aiter.hk_mla_decode_fwd_long
            if args.hk_long_kernel
            else aiter.hk_mla_decode_fwd
        )
        hk_decode(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            hk_work_indptr,
            hk_work_info_set,
            shape.decode_qlen,
            sm_scale,
            hk_split,
            hk_lse,
            hk_out,
        )

    def hk_reduce() -> None:
        aiter.mla_reduce_v1(
            hk_split,
            hk_lse,
            hk_reduce_indptr,
            hk_reduce_final_map,
            hk_reduce_partial_map,
            shape.decode_qlen,
            hk_out,
            None,
        )

    def hk_total() -> None:
        hk_stage()
        hk_reduce()

    hk_stage()
    hk_reduce()
    torch.cuda.synchronize()
    hk_decode_avg, hk_decode_p50 = time_cuda_us(hk_stage, args.warmup, args.iters)
    hk_stage()
    hk_reduce_avg, hk_reduce_p50 = time_cuda_us(hk_reduce, args.warmup, args.iters)
    hk_total_avg, hk_total_p50 = time_cuda_us(hk_total, args.warmup, args.iters)

    asm_total()
    hk_total()
    torch.cuda.synchronize()
    asm_out_unfolded = unfold_o_from_gfx950_asm(asm_out, shape)
    row.update(
        {
            "hk_decode_avg_us": hk_decode_avg,
            "hk_decode_p50_us": hk_decode_p50,
            "hk_reduce_avg_us": hk_reduce_avg,
            "hk_reduce_p50_us": hk_reduce_p50,
            "hk_total_avg_us": hk_total_avg,
            "hk_total_p50_us": hk_total_p50,
            "hk_speedup_p50": asm_total_p50 / hk_total_p50,
            "cos_diff": cosine_diff(asm_out_unfolded, hk_out),
            "max_abs_diff": float(
                (asm_out_unfolded.float() - hk_out.float()).abs().max()
            ),
        }
    )
    return row


def print_row(row: dict[str, object]) -> None:
    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    print(
        "batch={batch} ctx={ctx} q={decode_qlen} page={page_size} "
        "total_kv={total_kv} | "
        "asm_total_p50={asm_total_p50_us}us hk_total_p50={hk_total_p50_us}us "
        "speedup={hk_speedup_p50} cos_diff={cos_diff} max_abs={max_abs_diff}".format(
            **{k: fmt(v) for k, v in row.items()}
        ),
        flush=True,
    )
    if "meta_work_count" in row:
        print(
            "  metadata: works={meta_work_count} active_cu={meta_active_cu} "
            "max_work_per_cu={meta_max_works_per_cu} avg_kv_span={meta_avg_kv_span} "
            "max_kv_span={meta_max_kv_span} partial_tiles={meta_partial_tile_count} "
            "max_partials={meta_max_partials_per_tile}".format(
                **{k: fmt(v) for k, v in row.items()}
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HipKitten and AITER ASM persistent MLA decode for "
            "DeepSeek-R1 FP8 shapes on gfx950."
        )
    )
    parser.add_argument("--batch-sizes", default="2,4", type=parse_int_list)
    parser.add_argument(
        "--ctx-lens",
        default="2048,4096,8192,16384,32768,49152,65536,70000",
        type=parse_int_list,
    )
    parser.add_argument("--decode-qlen", default=4, type=int)
    parser.add_argument(
        "--nhead",
        default=128,
        type=int,
        choices=(32, 128),
        help="Query heads per rank. Use 32 for TP4 DeepSeek-R1, 128 for TP1.",
    )
    parser.add_argument("--page-sizes", default="1,16,32", type=parse_int_list)
    parser.add_argument("--max-split-per-batch", default=32, type=int)
    parser.add_argument(
        "--no-split-cap",
        action="store_true",
        help="Do not cap max_split_per_batch by ceil(CU/batch). The AITER metadata kernel still caps total splits by CU count.",
    )
    parser.add_argument(
        "--hk-long-kernel",
        action="store_true",
        help="Benchmark the separate experimental HK H32 long-context kernel.",
    )
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--shuffle-pages", action="store_true")
    parser.add_argument(
        "--metadata-stats",
        action="store_true",
        help="Include work split and reduce metadata statistics in CSV/log output.",
    )
    parser.add_argument("--csv", default="", help="Optional path to write CSV results.")
    return parser.parse_args()


def main() -> None:
    global NHEAD
    args = parse_args()
    NHEAD = args.nhead
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    fields = [
        "batch",
        "ctx",
        "decode_qlen",
        "page_size",
        "total_kv",
        "asm_decode_avg_us",
        "asm_decode_p50_us",
        "asm_reduce_avg_us",
        "asm_reduce_p50_us",
        "asm_total_avg_us",
        "asm_total_p50_us",
        "hk_supported",
        "hk_decode_avg_us",
        "hk_decode_p50_us",
        "hk_reduce_avg_us",
        "hk_reduce_p50_us",
        "hk_total_avg_us",
        "hk_total_p50_us",
        "hk_speedup_p50",
        "cos_diff",
        "max_abs_diff",
    ]
    if args.metadata_stats:
        fields.extend(
            [
                "meta_work_count",
                "meta_active_cu",
                "meta_max_works_per_cu",
                "meta_avg_works_per_cu",
                "meta_max_kv_span",
                "meta_avg_kv_span",
                "meta_max_qo_span",
                "meta_partial_tile_count",
                "meta_max_partials_per_tile",
                "meta_avg_partials_per_tile",
            ]
        )
    rows = []
    for batch_size in args.batch_sizes:
        for page_size in args.page_sizes:
            for ctx_len in args.ctx_lens:
                shape = MlaShape(batch_size, ctx_len, args.decode_qlen, page_size)
                row = benchmark_shape(args, shape)
                print_row(row)
                rows.append(row)

    wins = [
        row
        for row in rows
        if row["hk_speedup_p50"] is not None and float(row["hk_speedup_p50"]) >= 1.0
    ]
    if wins:
        max_winning_total_kv = max(int(row["total_kv"]) for row in wins)
        print(
            "Suggested AITER_HK_MLA_MAX_TOTAL_KV="
            f"{max_winning_total_kv} for this sweep (p50 speedup >= 1.0).",
            flush=True,
        )
    else:
        print("HK did not beat ASM on any supported page_size=1 row.", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
