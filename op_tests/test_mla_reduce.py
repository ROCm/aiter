# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# The uniform-decode fast path in mla_reduce_v1 is compiled/validated for these
# archs (generic HIP + opus buffer intrinsics on gfx9).
SUPPORTED_GFX = ["gfx942", "gfx950"]

# Fast-path shape constraints (dispatch_mla_reduce_v1): 16 q-heads, 512 v-dim,
# bf16 output. Anything else falls back to the general kernel.
H = 16
DV = 512


def build_uniform_decode_meta(num_tiles, nsplits, device):
    """Synthesize the metadata a uniform decode (max_seqlen_q==1) produces, i.e.
    the exact layout the fast path assumes:
      - reduce_indptr: prefix sums, nsplits partials per tile -> [0, ns, 2*ns, ...]
      - reduce_final_map[t] = [t, t+1]      (output row == tile == q_start)
      - reduce_partial_map  = arange(P)     (pool rows are the identity range)
    These are invariants (A) and (B) documented in csrc/kernels/mla/reduce.cu.
    """
    P = num_tiles * nsplits
    reduce_indptr = torch.arange(
        0, (num_tiles + 1) * nsplits, nsplits, dtype=dtypes.i32, device=device
    )
    rows = torch.arange(num_tiles, dtype=dtypes.i32, device=device)
    reduce_final_map = torch.stack([rows, rows + 1], dim=-1).contiguous()
    reduce_partial_map = torch.arange(P, dtype=dtypes.i32, device=device)
    return P, reduce_indptr, reduce_final_map, reduce_partial_map


def run_torch(partial_output, partial_lse, reduce_indptr, seed_out):
    """Reference log-sum-exp reduce over each tile's partial splits. Reference
    only: fp32 math, cast back — not timed, not in the table. Tiles with < 2
    splits are left at their seed value, mirroring the kernel which skips them
    (stage1 already wrote those outputs)."""
    num_tiles = reduce_indptr.numel() - 1
    out = seed_out.to(dtypes.fp32).clone()
    for t in range(num_tiles):
        base = int(reduce_indptr[t])
        end = int(reduce_indptr[t + 1])
        if end - base < 2:
            continue
        lse = partial_lse[base:end]  # [n, H]
        m = lse.max(dim=0).values  # [H]
        w = torch.exp(lse - m)  # [n, H]
        den = w.sum(dim=0)  # [H]
        acc = (w.unsqueeze(-1) * partial_output[base:end]).sum(dim=0)  # [H, DV]
        out[t] = acc / den.unsqueeze(-1)
    return out.to(seed_out.dtype)


@benchmark()  # (num_tiles, nsplits, dtype) become the table's left-hand columns
def test_mla_reduce(num_tiles, nsplits, dtype):
    device = "cuda"
    P, reduce_indptr, reduce_final_map, reduce_partial_map = build_uniform_decode_meta(
        num_tiles, nsplits, device
    )

    torch.manual_seed(0)
    partial_output = torch.randn(P, H, DV, dtype=dtypes.fp32, device=device)
    partial_lse = torch.randn(P, H, dtype=dtypes.fp32, device=device)
    # Pre-seed final_output: nosplit tiles (nsplits < 2) must be left untouched,
    # so the seed IS the expected output there.
    seed_out = torch.randn(num_tiles, H, DV, dtype=dtype, device=device)
    ref = run_torch(partial_output, partial_lse, reduce_indptr, seed_out)

    def fast_call():
        # final_lse=None + bf16 [num_tiles,16,512] out + max_seqlen_q=1 -> fast path.
        out = seed_out.clone()
        aiter.mla_reduce_v1(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            1,  # max_seqlen_q
            nsplits,  # num_kv_splits
            out,
            None,
        )
        return out

    candidates = {"fast": fast_call}

    if nsplits >= 2:
        # Requesting an LSE output makes fast_eligible False, forcing the general
        # kernel, which reduces the SAME rows via the reduce_final_map /
        # reduce_partial_map indirections. Its final_output must match the fast
        # path — this is the fast==slow parity check. Skipped for the nosplit
        # config, where the general kernel does not share the fast path's
        # "leave it to stage1" skip semantics.
        final_lse = torch.empty(num_tiles, H, dtype=dtypes.fp32, device=device)

        def slow_call():
            out = seed_out.clone()
            aiter.mla_reduce_v1(
                partial_output,
                partial_lse,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                1,
                nsplits,
                out,
                final_lse,
            )
            return out

        candidates["slow"] = slow_call

    # Memory-bound reduce roofline: read all partials + lse, write the reduced
    # tiles; ~2 flops (mul+add) per accumulated element.
    nbytes = (
        P * H * DV * partial_output.element_size()
        + P * H * partial_lse.element_size()
        + num_tiles * H * DV * seed_out.element_size()
    )
    flops = P * H * DV * 2

    ret = {"gfx": get_gfx(), "P": P}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: mla_reduce_v1 [{num_tiles=} {nsplits=}]",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mla_reduce_v1 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="output dtype. Fast path is bf16-only. Default: bf16.",
    )
    parser.add_argument(
        "-t",
        "--tiles",
        type=int,
        nargs="*",
        default=[1, 8, 64, 128],
        help="number of reduce tiles (== decode tokens). Default: 1 8 64 128.",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        nargs="*",
        # 1 exercises the nosplit skip; 2/4/8/16 sweep the fast-path chunk widths;
        # 3/17/33 straddle those boundaries and the MAX_SPLITS chunk loop.
        default=[1, 2, 3, 4, 8, 16, 17, 33],
        help="kv splits per tile. Default: 1 2 3 4 8 16 17 33.",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for num_tiles, nsplits in itertools.product(args.tiles, args.nsplits):
            df.append(test_mla_reduce(num_tiles, nsplits, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce_v1 summary (%s, markdown):\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
