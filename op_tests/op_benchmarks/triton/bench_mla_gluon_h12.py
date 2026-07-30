# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark split-KV geometry for low-head gfx950 MLA decode.

The default shape matches Kimi K3 TP8 (12 local heads, batch 1, 100K context).
The oversized KV pool intentionally forces mla_gluon's 64-bit global-load path,
matching a production vLLM cache whose allocation is larger than 2 GiB.
"""

import argparse
import math
from collections.abc import Callable

import torch
import triton

from aiter.ops.triton.gluon.mla_gluon import mla_gluon


def _parse_split(value: str) -> int | None:
    if value == "auto":
        return None
    split = int(value)
    if split < 1:
        raise argparse.ArgumentTypeError("splits must be positive integers or 'auto'")
    return split


def _make_page_metadata(
    batch_size: int,
    context_length: int,
    pool_tokens: int,
    layout: str,
    page_order: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Vary flattened lengths by up to three tokens while preserving
    # context_length as the true minimum used by the split heuristic.
    lengths = torch.full(
        (batch_size,), context_length, dtype=torch.int32, device=device
    )
    if layout == "varlen":
        lengths += torch.arange(batch_size, dtype=torch.int32, device=device) % 4
    max_length = int(lengths.max().item())

    if page_order == "sequential":
        rows = torch.arange(max_length, dtype=torch.int32, device=device).expand(
            batch_size, -1
        )
    else:
        rows = torch.randint(
            0,
            pool_tokens,
            (batch_size, max_length),
            dtype=torch.int32,
            device=device,
        )

    if layout == "2d":
        return rows, lengths, rows.flatten().unique()

    row_chunks = [rows[i, : int(lengths[i].item())] for i in range(batch_size)]
    page_table = torch.cat(row_chunks)
    seq_info = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            lengths.cumsum(0, dtype=torch.int32),
        )
    )
    return page_table, seq_info, page_table.unique()


def _measure(
    function: Callable[[], None], warmup: int, rep: int
) -> tuple[float, float, float]:
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        function,
        warmup=warmup,
        rep=rep,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median_ms * 1000, p20_ms * 1000, p80_ms * 1000


def _output_diffs(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float, bool]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    delta = actual_fp32 - reference_fp32
    max_abs = delta.abs().max().item()
    rmse = delta.square().mean().sqrt().item()
    denominator = max(
        (actual_fp32.square() + reference_fp32.square()).sum().item(), 1e-12
    )
    cos_diff = 1.0 - 2.0 * (actual_fp32 * reference_fp32).sum().item() / denominator
    finite = bool(torch.isfinite(actual_fp32).all().item())
    return max_abs, rmse, cos_diff, finite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-length",
        "--context-lengths",
        dest="context_lengths",
        type=int,
        nargs="+",
        default=[100_000],
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[12])
    parser.add_argument(
        "--layouts", choices=("2d", "varlen"), nargs="+", default=["2d"]
    )
    parser.add_argument(
        "--page-orders",
        choices=("sequential", "random"),
        nargs="+",
        default=["sequential"],
    )
    parser.add_argument("--pool-tokens", type=int, default=2_097_152)
    parser.add_argument(
        "--splits",
        type=_parse_split,
        nargs="+",
        default=[None, 32, 64, 96, 112, 128, 160, 192, 224, 256],
    )
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if min(args.context_lengths) < 1:
        raise ValueError("context lengths must be positive")
    if min(args.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if min(args.num_heads) < 1 or max(args.num_heads) > 16:
        raise ValueError("this benchmark supports 1 through 16 heads")
    if args.pool_tokens < max(args.context_lengths) + 3:
        raise ValueError("pool-tokens must cover the longest varlen sequence")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    kv = torch.empty((args.pool_tokens, 576), dtype=torch.bfloat16, device=device)
    sm_scale = 1.0 / math.sqrt(576)

    print(
        "context,batch,heads,layout,page_order,splits,median_us,p20_us,p80_us,"
        "speedup_vs_legacy,tb_per_s,max_abs_diff,rmse,cos_diff,lse_max_abs,finite"
    )
    for context_length in args.context_lengths:
        for batch_size in args.batch_sizes:
            for num_heads in args.num_heads:
                for layout in args.layouts:
                    for page_order in args.page_orders:
                        page_table, seq_info, referenced_pages = _make_page_metadata(
                            batch_size,
                            context_length,
                            args.pool_tokens,
                            layout,
                            page_order,
                            device,
                        )
                        kv[referenced_pages.long()] = torch.randn(
                            (referenced_pages.numel(), 576),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        q_nope = torch.randn(
                            (batch_size, num_heads, 512),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        q_pe = torch.randn(
                            (batch_size, num_heads, 64),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        use_2d_view = layout == "2d"
                        legacy_splits = max(
                            1,
                            min(
                                256 // batch_size,
                                triton.cdiv(context_length, 64),
                            ),
                        )
                        reference = torch.empty_like(q_nope)

                        def run_reference() -> None:
                            mla_gluon(
                                q_nope,
                                q_pe,
                                kv,
                                reference,
                                page_table,
                                seq_info,
                                sm_scale,
                                use_2d_view=use_2d_view,
                                min_kv_seq_len=context_length,
                                return_lse=args.return_lse,
                                num_kv_splits=legacy_splits,
                            )

                        legacy_us, _, _ = _measure(
                            run_reference, args.warmup, args.rep
                        )
                        reference, reference_lse = mla_gluon(
                            q_nope,
                            q_pe,
                            kv,
                            reference,
                            page_table,
                            seq_info,
                            sm_scale,
                            use_2d_view=use_2d_view,
                            min_kv_seq_len=context_length,
                            return_lse=args.return_lse,
                            num_kv_splits=legacy_splits,
                        )
                        torch.cuda.synchronize()

                        for splits in args.splits:
                            if splits is not None and splits > context_length:
                                continue
                            output = torch.empty_like(reference)
                            output_lse = None

                            def run() -> None:
                                nonlocal output, output_lse
                                output, output_lse = mla_gluon(
                                    q_nope,
                                    q_pe,
                                    kv,
                                    output,
                                    page_table,
                                    seq_info,
                                    sm_scale,
                                    use_2d_view=use_2d_view,
                                    min_kv_seq_len=context_length,
                                    return_lse=args.return_lse,
                                    num_kv_splits=splits,
                                )

                            median_us, p20_us, p80_us = _measure(
                                run, args.warmup, args.rep
                            )
                            run()
                            torch.cuda.synchronize()
                            max_abs, rmse, cos_diff, finite = _output_diffs(
                                output, reference
                            )
                            if output_lse is None or reference_lse is None:
                                lse_max_abs = 0.0
                            else:
                                lse_max_abs = (
                                    output_lse - reference_lse
                                ).abs().max().item()
                                finite = finite and bool(
                                    torch.isfinite(output_lse).all().item()
                                )
                            bytes_read = (
                                batch_size
                                * context_length
                                * 576
                                * kv.element_size()
                            )
                            tb_per_s = bytes_read / median_us / 1e6
                            split_label = "auto" if splits is None else str(splits)
                            print(
                                f"{context_length},{batch_size},{num_heads},"
                                f"{layout},{page_order},{split_label},"
                                f"{median_us:.3f},{p20_us:.3f},{p80_us:.3f},"
                                f"{legacy_us / median_us:.4f},{tb_per_s:.4f},"
                                f"{max_abs:.6f},{rmse:.6f},{cos_diff:.8f},"
                                f"{lse_max_abs:.6f},{finite}"
                            )


if __name__ == "__main__":
    main()
