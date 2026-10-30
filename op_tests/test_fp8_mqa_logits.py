# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf sweep for the prefill FP8 MQA indexer logits kernels.

Candidates are the Triton/Gluon kernel (`aiter.ops.triton.attention.fp8_mqa_logits`)
and the hand-written HIP one (`aiter.ops.fp8_mqa_logits`), graded against the same
fp32 torch reference on identical inputs.

The HIP kernel only covers nh=32/hd=128 on gfx950, so it is added per case rather
than unconditionally; its cells stay NaN elsewhere.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    calc_diff,
    e4m3_type,
    generate_cp_test_data,
    per_custom_dims_cast_to_fp8,
    ref_fp8_mqa_logits,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

try:
    from aiter.ops.fp8_mqa_logits import fp8_mqa_logits as hip_logits
    from aiter.ops.fp8_mqa_logits import is_supported as hip_supported
except ImportError:
    hip_logits = None

    def hip_supported(num_heads, head_dim):
        return False


def _num_requests(s_q, s_k):
    """How many requests `multi_req` packs into a [s_q, s_k] chunk.

    Four where the shape allows it, fewer on the small shapes so every request
    still gets at least two query rows and two KV columns.
    """
    return max(1, min(4, s_q // 2, s_k // 2))


def make_windows(s_q, s_k, window):
    """The [ks, ke) row windows for one case.

    `causal` and `cp` are what the model produces. The other three are legal
    inputs the indexer really sees at a chunk boundary, and they are where the
    masking and the -inf fill are easiest to get wrong:

      misaligned  windows that start off a tile boundary
      empty       rows with no window at all (cu_ends below zero, or below
                  cu_starts), interleaved with normal ones so one block's union
                  window mixes the two -- a causal mask yields this whenever
                  s_kv < s_q
      past_end    cu_starts/cu_ends beyond seq_len_kv. Legal, and simply empty
                  over the part that is out of range -- but a kernel that trusts
                  the bounds unclamped will run off the end of the row.
    """
    rows = torch.arange(s_q, device="cuda")
    if window == "causal":
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
        return ks, ke
    if window == "cp":
        return generate_cp_test_data(s_q, s_k)
    if window == "misaligned":
        ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(torch.int32)
        ke = torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k))
        return ks, ke.to(torch.int32)
    if window == "empty":
        ks = torch.where(rows % 3 == 1, min(100, s_k), 0)
        ke = torch.where(
            rows % 3 == 0,
            -1 - (rows % 7),
            torch.where(rows % 3 == 1, 0, torch.minimum(rows + 1, ks + s_k)),
        )
        return ks.to(torch.int32), ke.to(torch.int32)
    if window == "past_end":
        ks = torch.where(rows % 2 == 0, s_k + 17, 0)
        ke = torch.where(rows % 2 == 0, s_k + 64, torch.minimum(rows + 1, ks + s_k))
        return ks.to(torch.int32), ke.to(torch.int32)
    if window == "multi_req":
        # Several requests packed into one [s_q, s_k] chunk, which is what the
        # prefill indexer is actually handed: each request owns a contiguous run of
        # query rows and a contiguous slice of the gathered K buffer, so cu_starts
        # jumps at every request boundary and cu_ends is that request's own causal
        # diagonal. A block whose rows straddle a boundary has to cover the union of
        # their ranges and mask each row back to its own -- the one path the
        # single-request modes above never reach, since they all share cu_starts.
        nreq = _num_requests(s_q, s_k)
        rows_per_req = (s_q + nreq - 1) // nreq
        kv_per_req = s_k // nreq
        req = torch.clamp(rows // rows_per_req, max=nreq - 1)
        ks = req * kv_per_req
        local = rows - req * rows_per_req
        # Causal *within* the request: its last query row attends to all of its
        # KV slice, so cu_ends runs to the end of the slice rather than to the
        # row index. Anything else models a request whose KV is only as long as
        # its query chunk, which is not how prefill is chunked -- and it would
        # make every window a few hundred columns wide, so the perf half of the
        # sweep would measure launch overhead instead of the kernel.
        ke = ks + (kv_per_req - rows_per_req) + local + 1
        ke = torch.clamp(ke, min=0, max=None)
        ke = torch.minimum(ke, ks + kv_per_req)
        return ks.to(torch.int32), torch.maximum(ke, ks).to(torch.int32)
    raise ValueError(f"unknown window mode: {window}")


# Peak fp32 bytes the reference is allowed for its [heads, chunk, s_k] score
# tensor. It builds that tensor whole, so at the long-context shapes an unchunked
# call wants tens of GiB (heads=32, s_q=1024, s_k=131072 is 17 GiB).
_REF_SCORE_BUDGET = 512 << 20


def run_torch(q_fp8, kv_fp8, scales, weights, ks, ke):
    """Reference only: fp32 math over exactly what the kernels consume.

    Calls `ref_fp8_mqa_logits` -- the same reference the Triton lane's test uses --
    over query-row chunks, so the result is identical to one unchunked call (rows
    are independent) while the score tensor stays inside the budget above.
    """
    q = q_fp8.to(dtypes.fp32)
    kv = kv_fp8.to(dtypes.fp32) * scales.reshape(-1, 1)
    s_q, num_heads, _ = q.shape
    s_k = kv.shape[0]

    chunk = max(1, _REF_SCORE_BUDGET // max(1, num_heads * s_k * 4))
    if chunk >= s_q:
        return ref_fp8_mqa_logits(
            q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
        )

    logits = torch.empty((s_q, s_k), dtype=dtypes.fp32, device=q.device)
    cost = torch.zeros((), dtype=torch.int64, device=q.device)
    for m0 in range(0, s_q, chunk):
        m1 = min(m0 + chunk, s_q)
        part, part_cost = ref_fp8_mqa_logits(
            q=q[m0:m1],
            kv=kv,
            weights=weights[m0:m1],
            cu_seqlen_ks=ks[m0:m1],
            cu_seqlen_ke=ke[m0:m1],
        )
        logits[m0:m1] = part
        cost += part_cost
        del part
    return logits, cost


def _rehydrate(out, ks, ke, s_q, s_k):
    """Re-apply the -inf mask a clean_logits=False launcher did not write.

    Clamp into [0, s_k] first: cu_ends may be negative or below cu_starts (both
    mean "no window"), and a raw negative bound would select from the end of the
    row instead of nothing.
    """
    col = torch.arange(s_k, device=out.device)[None, :]
    keep = (col >= ks.clamp(0, s_k)[:, None]) & (col < ke.clamp(0, s_k)[:, None])
    return torch.where(keep, out, torch.full_like(out, float("-inf")))


@benchmark()
def test_fp8_mqa_logits(s_q, s_k, num_heads, head_dim, clean_logits, window):
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, dtype=dtypes.bf16)
    kv = torch.randn(s_k, head_dim, dtype=dtypes.bf16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(dtypes.fp32) * scales.reshape(-1, 1)).to(dtypes.bf16)
    weights = torch.randn(s_q, num_heads, dtype=dtypes.fp32)
    ks, ke = make_windows(s_q, s_k, window)

    q_fp8 = q.to(e4m3_type)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref, cost = run_torch(q_fp8, kv_fp8, scales, weights, ks, ke)
    ref_mask = ref == float("-inf")

    # (timed call, graded call) per candidate.
    #
    # They differ because grading needs a poisoned output buffer and timing must
    # not pay for it. The timed call is the one a caller makes -- each launcher
    # allocates its own output, which is the whole point of `clean_logits`: the
    # Triton path pre-fills all s_q*s_k elements with torch.full and overwrites
    # the valid ones, the HIP kernel writes the -inf itself. Poisoning inside the
    # timed call would instead charge one candidate a full-output memset per
    # iteration -- 537 MB at s_k=131072, which swamps the kernel.
    def _triton():
        return triton_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

    candidates = {"triton": (_triton, _triton)}

    # gfx950-only, nh=32/hd=128-only; NaN cells elsewhere rather than a wrong row.
    if hip_logits is not None and hip_supported(num_heads, head_dim):

        def _hip(out=None):
            return hip_logits(
                q_fp8,
                kv_fp8,
                scales,
                weights,
                ks,
                ke,
                clean_logits=clean_logits,
                out=out,
            )

        # Any position the kernel fails to write stays NaN and the -inf mask check
        # below catches it. Without that the check is close to vacuous: the caching
        # allocator hands back a block a previous case already left holding the
        # correct -inf, so an under-fill would pass.
        def _hip_graded():
            return _hip(torch.full((s_q, s_k), float("nan"), dtype=dtypes.fp32))

        candidates["hip"] = (_hip, _hip_graded)

    flops = cost.item() * num_heads * head_dim * 2
    nbytes = (
        (s_q * num_heads * head_dim + s_k * head_dim) * q_fp8.element_size()
        + (s_k + s_q * num_heads) * 4  # scales + weights
        + 2 * s_q * 4  # ks + ke
        + s_q * s_k * 4  # output
    )

    ret = {"gfx": get_gfx()}
    for name, (time_fn, grade_fn) in candidates.items():
        _, us = run_perftest(time_fn)
        out = grade_fn()
        if not clean_logits:
            out = _rehydrate(out, ks, ke, s_q, s_k)

        out_mask = out == float("-inf")
        assert torch.equal(out_mask, ref_mask), (
            f"{name}: -inf mask mismatch, "
            f"{int((out_mask != ref_mask).sum())} of {ref.numel()} positions"
        )
        err = checkAllclose(
            ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
            out.masked_fill(out_mask, 0).to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: fp8_mqa_logits",
            printLog=False,
        )
        if int((~ref_mask).sum()) > 1:
            # calc_diff is an aggregate similarity; over a single finite element it
            # degenerates and one borderline ReLU term moves it by percent, so
            # assert it only where it is meaningful. checkAllclose bounds the rest.
            diff = calc_diff(
                out.masked_fill(out_mask, 0).to(dtypes.fp32),
                ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
            )
            assert diff < 1e-3, f"{name}: calc_diff={diff.item():.3e}"

        ret[f"{name} us"] = us
        # NaN, not 0, when the windows select nothing: a 0 in a TFLOPS column reads
        # as a broken measurement rather than as "there was no work to do". The
        # kernel still writes the -inf fill, so TB/s stays meaningful.
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if flops > 0 else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fp8_mqa_logits unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1),
            (1, 113),
            (17, 76),
            (61, 1024),
            (128, 1024),
            (1024, 1024),
            (1024, 1560),
            # Small-M / long-KV: the row grid alone cannot fill the device, so the
            # launcher splits each row's KV window hard across grid.y.
            (64, 2048),
            (64, 8192),
            # s_kv < s_q, which puts cu_ends below zero on the leading rows.
            (128, 64),
            (1024, 1000),
            # The real prefill chunk shapes. A request arrives split so that
            # M*N*4 stays under VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (512 MB), so a
            # 32K request is (4096, 32768) and a 128K one is (1024, 131072).
            (4096, 32768),
            (2048, 65536),
            (1024, 131072),
        ],
        help="(s_q, s_k) pairs",
    )
    parser.add_argument("--num-heads", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
    parser.add_argument(
        "--clean-logits", type=int, nargs="*", default=[0, 1], choices=[0, 1]
    )
    parser.add_argument(
        "-w",
        "--window",
        type=str,
        nargs="*",
        default=["causal", "cp", "misaligned", "empty", "past_end", "multi_req"],
        choices=["causal", "cp", "misaligned", "empty", "past_end", "multi_req"],
    )
    args = parser.parse_args()

    df = []
    skipped = []
    for (s_q, s_k), nh, hd, cl, win in itertools.product(
        args.shapes, args.num_heads, args.head_dim, args.clean_logits, args.window
    ):
        # generate_cp_test_data asserts these divisibility conditions.
        if win == "cp" and (s_k % s_q != 0 or s_q % 2 != 0):
            continue
        # multi_req degenerates to causal below two requests; skip rather than
        # duplicate the causal row.
        if win == "multi_req" and _num_requests(s_q, s_k) < 2:
            continue
        try:
            df.append(test_fp8_mqa_logits(s_q, s_k, nh, hd, bool(cl), win))
        except torch.OutOfMemoryError:
            # The long-context shapes need ~0.5 GiB per [s_q, s_k] fp32 tensor and
            # the case holds several. Record what was dropped rather than letting a
            # short table read as full coverage.
            skipped.append((s_q, s_k, nh, hd, bool(cl), win))
            torch.cuda.empty_cache()
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fp8_mqa_logits summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if skipped:
        aiter.logger.warning(
            "fp8_mqa_logits: %d of %d cases skipped, out of memory:\n%s",
            len(skipped),
            len(skipped) + len(df),
            "\n".join(
                f"  s_q={a} s_k={b} num_heads={c} head_dim={d} clean_logits={e} "
                f"window={f}"
                for a, b, c, d, e, f in skipped
            ),
        )


if __name__ == "__main__":
    main()
