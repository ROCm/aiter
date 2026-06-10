# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Standalone test for the MLA reduce kernel `mla_reduce_v1`
(csrc/kernels/mla/reduce.cu).

The kernel merges the per-split partial attention outputs/LSEs produced by the
split-KV decode/prefill stage into a single final output via an online
(flash-attention style) log-sum-exp reduction.

This test does NOT run any attention stage. It synthesizes the reduce metadata
and the partial buffers directly, runs the GPU kernel, and compares against the
pure-torch reference `torch_mla_reduce_v1` (reused from test_mla_persistent.py).

Tensor layout (see reduce.cu / mla.py):
  partial_output    : [P, h, dv]  float32   partial buffer, indexed by
                                            reduce_partial_map[split] + local_seq
  partial_lse       : [P, h]      float32
  reduce_indptr     : [T + 1]     int32     per-tile [start, end) into partial_map
  reduce_final_map  : [T, 2]      int32     (q_start, q_end) output rows of tile
  reduce_partial_map: [N]         int32     partial buffer base location per split
  final_output      : [bs, h, dv] out dtype (bf16 / fp16)
  final_lse         : [bs, h]     float32
"""

import argparse
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.manual_seed(0)


def torch_mla_reduce_v1(
    partial_output: torch.Tensor,  # [P, h, dv] float32
    partial_lse: torch.Tensor,  # [P, h]    float32
    reduce_indptr: torch.Tensor,  # [T + 1]
    reduce_final_map: torch.Tensor,  # [T, 2] or None
    reduce_partial_map: torch.Tensor,  # [N]
    max_seqlen_q: int,
    final_output: torch.Tensor,  # [bs, h, dv]
    final_lse: torch.Tensor,  # [bs, h] or None
) -> None:
    """Pure-torch reference for the `mla_reduce_v1` HIP kernel.

    Online (flash-attention style) log-sum-exp combine of per-split partial
    outputs into the final attention output. Mirrors the numerically-stable
    update in mla_reduce_v1_impl_simple in csrc/kernels/mla/reduce.cu.
    """
    device = partial_output.device
    dtype = partial_output.dtype

    assert partial_output.dtype == torch.float32, "partial_output must be float32"
    assert partial_lse.dtype == torch.float32, "partial_lse must be float32"

    num_reduce_tile = reduce_indptr.shape[0] - 1
    num_heads = partial_output.shape[1]
    head_dim = final_output.shape[2]

    for tile_idx in range(num_reduce_tile):
        reduce_tile_start = reduce_indptr[tile_idx].item()
        reduce_tile_end = reduce_indptr[tile_idx + 1].item()
        if reduce_tile_start == reduce_tile_end:
            continue
        num_splits = reduce_tile_end - reduce_tile_start
        tile_reduce_partial_map = reduce_partial_map[reduce_tile_start:reduce_tile_end]

        if reduce_final_map is not None:
            q_start = reduce_final_map[tile_idx, 0].item()
            q_end = reduce_final_map[tile_idx, 1].item()
        else:
            if num_splits >= 2:
                rpm0 = tile_reduce_partial_map[0].item()
                rpm1 = tile_reduce_partial_map[1].item()
                qo_len = rpm1 - rpm0
                q_start = tile_idx * qo_len
                q_end = (tile_idx + 1) * qo_len
            else:
                q_start = tile_idx * max_seqlen_q
                q_end = (tile_idx + 1) * max_seqlen_q

        for seq_idx in range(q_start, q_end):
            for head_idx in range(num_heads):
                local_seq_idx = seq_idx - q_start
                lses = []
                outs = []
                for split_idx in range(num_splits):
                    partial_qo_loc = tile_reduce_partial_map[split_idx].item()
                    buf_idx = partial_qo_loc + local_seq_idx

                    if buf_idx < partial_lse.shape[0] and head_idx < partial_lse.shape[1]:
                        lse_val = partial_lse[buf_idx, head_idx].item()
                        if math.isnan(lse_val):
                            lse_val = float("-inf")
                    else:
                        lse_val = float("-inf")

                    if (
                        buf_idx < partial_output.shape[0]
                        and head_idx < partial_output.shape[1]
                    ):
                        out_vals = partial_output[buf_idx, head_idx, :].clone()
                        out_vals = torch.where(
                            torch.isnan(out_vals), torch.zeros_like(out_vals), out_vals
                        )
                    else:
                        out_vals = torch.zeros(head_dim, dtype=dtype, device=device)

                    lses.append(lse_val)
                    outs.append(out_vals)

                max_lse = lses[0]
                reg_out = outs[0].clone()
                sum_e_lse = 1.0
                for split_idx in range(1, num_splits):
                    lse = lses[split_idx]
                    oaccu = outs[split_idx]
                    new_max_lse = max(max_lse, lse)
                    old_scale = math.exp(max_lse - new_max_lse)
                    new_scale = math.exp(lse - new_max_lse)
                    reg_out = old_scale * reg_out + new_scale * oaccu
                    max_lse = new_max_lse
                    sum_e_lse = sum_e_lse * old_scale + new_scale

                if sum_e_lse > 0 and not math.isnan(sum_e_lse):
                    reg_out = reg_out / sum_e_lse
                else:
                    reg_out = torch.zeros_like(reg_out)

                final_output[seq_idx, head_idx, :] = reg_out.to(final_output.dtype)

                if final_lse is not None:
                    if sum_e_lse > 0 and not math.isnan(sum_e_lse):
                        final_lse_val = max_lse + math.log(sum_e_lse)
                    else:
                        final_lse_val = float("inf")
                    final_lse[seq_idx, head_idx] = final_lse_val


def build_reduce_problem(
    splits_per_tile,   # list[int]: number of splits for each reduce tile
    num_heads,
    head_dim,
    out_dtype,
    qo_len=1,          # output rows produced per reduce tile
    device="cuda",
):
    """Build a synthetic reduce problem.

    Each reduce tile t produces `qo_len` output rows and merges
    `splits_per_tile[t]` partials. Every (tile, split, local_seq) triple gets a
    unique row in the partial buffer, filled with random data.
    """
    num_tiles = len(splits_per_tile)

    # reduce_indptr: cumulative split counts.
    indptr = [0]
    for s in splits_per_tile:
        indptr.append(indptr[-1] + s)
    total_splits = indptr[-1]

    # Assign every partial a unique base location in the partial buffer. The
    # kernel reads partial row = base + local_seq, so reserve qo_len rows per
    # partial to avoid collisions.
    partial_map = []
    next_base = 0
    final_map = []
    out_row = 0
    for s in splits_per_tile:
        for _ in range(s):
            partial_map.append(next_base)
            next_base += qo_len
        final_map.append([out_row, out_row + qo_len])
        out_row += qo_len

    num_partial_rows = next_base
    num_out_rows = out_row

    reduce_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    reduce_partial_map = torch.tensor(partial_map, dtype=torch.int32, device=device)
    reduce_final_map = torch.tensor(final_map, dtype=torch.int32, device=device)

    partial_output = torch.randn(
        num_partial_rows, num_heads, head_dim, dtype=torch.float32, device=device
    )
    # LSEs in a realistic-ish range; flash combine is invariant to the scale but
    # spread them out so scales differ across splits.
    partial_lse = (
        torch.randn(num_partial_rows, num_heads, dtype=torch.float32, device=device)
        * 4.0
    )

    final_output = torch.empty(
        num_out_rows, num_heads, head_dim, dtype=out_dtype, device=device
    )
    final_lse = torch.empty(
        num_out_rows, num_heads, dtype=torch.float32, device=device
    )

    return dict(
        partial_output=partial_output,
        partial_lse=partial_lse,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        final_output=final_output,
        final_lse=final_lse,
        max_seqlen_q=qo_len,
        num_out_rows=num_out_rows,
    )


def run_case(splits_per_tile, num_heads, head_dim, out_dtype, qo_len=1):
    p = build_reduce_problem(
        splits_per_tile, num_heads, head_dim, out_dtype, qo_len=qo_len
    )

    # ---- Reference (pure torch) ----
    ref_out = torch.empty_like(p["final_output"])
    ref_lse = torch.empty_like(p["final_lse"])
    torch_mla_reduce_v1(
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        ref_out,
        ref_lse,
    )

    # ---- GPU kernel (timed) ----
    gpu_out = p["final_output"]
    gpu_lse = p["final_lse"]
    _, us = run_perftest(
        aiter.mla_reduce_v1,
        p["partial_output"],
        p["partial_lse"],
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        p["max_seqlen_q"],
        gpu_out,
        gpu_lse,
    )

    tag = (
        f"splits={splits_per_tile if len(splits_per_tile) <= 4 else f'{len(splits_per_tile)}x[{splits_per_tile[0]}]'}"
        f" h={num_heads} dv={head_dim} {str(out_dtype).split('.')[-1]}"
    )

    err_out = checkAllclose(
        ref_out.float(),
        gpu_out.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] output {us:>8.2f} us ",
    )
    err_lse = checkAllclose(
        ref_lse,
        gpu_lse,
        rtol=2e-2,
        atol=2e-2,
        msg=f"[{tag}] lse    ",
    )
    ok = (err_out == 0) and (err_lse == 0)
    print(f"{'PASS' if ok else 'FAIL'}  {tag}  {us:>8.2f} us")

    # Bytes moved by the reduce: read all partial outputs + partial lses +
    # metadata, write final outputs + final lses. (dominant traffic is the f32
    # partials; metadata is small but counted for completeness)
    total_splits = sum(splits_per_tile)
    num_tiles = len(splits_per_tile)
    read_bytes = (
        total_splits * qo_len * num_heads * head_dim * 4  # partial_output f32
        + total_splits * qo_len * num_heads * 4  # partial_lse f32
        + (num_tiles + 1) * 4  # reduce_indptr int32
        + num_tiles * 2 * 4  # reduce_final_map int32 [T,2]
        + total_splits * 4  # reduce_partial_map int32 [N]
    )
    write_bytes = (
        p["num_out_rows"] * num_heads * head_dim * (torch.finfo(out_dtype).bits // 8)
        + p["num_out_rows"] * num_heads * 4  # final_lse f32
    )
    bytes_total = read_bytes + write_bytes

    ret = {
        "splits": (
            str(splits_per_tile)
            if len(splits_per_tile) <= 4
            else f"{len(splits_per_tile)}x[{splits_per_tile[0]}]"
        ),
        "nhead": num_heads,
        "dv": head_dim,
        "dtype": str(out_dtype).split(".")[-1],
        "qo_len": qo_len,
        "us": round(us, 2),
        "GB/s": round(bytes_total / us / 1e3, 1),
        "pass": ok,
    }
    return ok, ret


def main():
    parser = argparse.ArgumentParser(description="Test mla_reduce_v1 kernel")
    parser.add_argument("-d", "--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--head_dim",
        type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=[128, 512],
        help="comma-separated head dims, e.g. 128 or 128,512 (default: 128,512)",
    )
    args = parser.parse_args()

    out_dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    head_dims = args.head_dim

    # The kernel sizes its LDS scratch to `max_splits = multiProcessorCount`
    # (see reduce.cu: params.max_splits = dev_prop.multiProcessorCount and the
    # lds_size computation in dispatch_mla_reduce_v1). A reduce tile with more
    # splits than the device CU count overflows LDS -> GPU memory fault. In real
    # usage the metadata kernel guarantees num_splits per tile <= num_CU, so we
    # mirror that invariant here and cap each tile's split count.
    cu_count = torch.cuda.get_device_properties(0).multi_processor_count

    # (head_dim -> supported head counts from MLA_REDUCE_ROUTER in reduce.cu)
    heads_for_dim = {
        128: [1, 16, 128],
        512: [8, 16, 128],
    }

    # Cover both kernel paths:
    #   num_splits in [2,3]  -> simple impl
    #   num_splits >= 4      -> massive impl (<=64 / <=256 buckets)
    big = min(cu_count, 256)  # largest single-tile split count we can safely run
    split_configs = [
        [2],                 # simple, single tile
        [3, 2],              # simple, multi tile
        [4],                 # massive, exactly threshold
        [8, 5, 7],           # massive <=64, ragged
        [33],                # massive <=64
        [big],               # massive: <=256 bucket if cu_count>64, else <=64
        [2, 4, 16, 64],      # mixed: simple + massive buckets
    ] + [[6] * 40]           # many tiles -> exercises persistent-grid path

    # Drop any config whose per-tile split count exceeds the LDS capacity.
    dropped = [c for c in split_configs if max(c) > cu_count]
    if dropped:
        print(f"NOTE: cu_count={cu_count}; skipping configs exceeding it: {dropped}")
    split_configs = [c for c in split_configs if max(c) <= cu_count]

    all_ok = True
    df = []
    for head_dim in head_dims:
        for num_heads in heads_for_dim[head_dim]:
            for splits in split_configs:
                ok, ret = run_case(splits, num_heads, head_dim, out_dtype)
                all_ok &= ok
                df.append(ret)

    # multi-row tiles (qo_len > 1), decode/prefill-style
    for head_dim in head_dims:
        ok, ret = run_case([8, 4], 16, head_dim, out_dtype, qo_len=3)
        all_ok &= ok
        df.append(ret)

    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_reduce summary (markdown):\n%s", df_md)

    print("\n" + ("ALL PASSED" if all_ok else "SOME FAILED"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
