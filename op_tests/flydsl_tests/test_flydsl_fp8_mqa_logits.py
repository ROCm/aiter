# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

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
# `e4m3_type` is arch-dependent (get_fp8_dtypes): FNUZ on gfx942, FN on gfx950.
# So on gfx950 both keys resolve to float8_e4m3fn and the two cases coincide --
# only gfx942 has a genuine FN/FNUZ split.
DTYPE_MAP = {"fnuz": e4m3_type, "fn": torch.float8_e4m3fn}

# Default operand-dtype sweep, per arch.
#
# gfx942's native MFMA operand format is FNUZ, so the kernel takes FN operands
# by patching them (see `convert_q_fn`/`convert_kv_fn`). Both fnuz/fnuz and the
# live DeepSeek-V4 indexer combo fn/fnuz are therefore real, distinct paths.
#
# gfx950's native format is FN and its CDNA4 scaled atoms reject FNUZ outright,
# so the kernel never converts there and fn/fn is the only combination that can
# occur.
_DEFAULT_Q_DTYPES = ["fn"] if get_gfx() == "gfx950" else ["fnuz", "fn"]
_DEFAULT_KV_DTYPES = ["fn"] if get_gfx() == "gfx950" else ["fnuz"]

try:
    from aiter.ops.flydsl import flydsl_fp8_mqa_logits
except ImportError:
    flydsl_fp8_mqa_logits = None


def _make_windows(s_q, s_k, mode):
    if mode == "causal":
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
        return ks, ke
    if mode == "cp":
        return generate_cp_test_data(s_q, s_k)
    if mode == "misaligned":
        rows = torch.arange(s_q, device="cuda")
        ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(torch.int32)
        ke = torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k)).to(
            torch.int32
        )
        return ks, ke
    if mode == "empty":
        # Rows with no window at all, interleaved with normal ones so a single
        # block's union window mixes the two. Both spellings of "empty" appear:
        #
        #   r%3==0  cu_ends < 0        -- what a causal mask yields whenever
        #                                 s_kv < s_q, so this is ordinary input,
        #                                 not a synthetic edge case
        #   r%3==1  cu_ends <= cu_starts
        #   r%3==2  an ordinary non-empty window
        #
        # Both leave the union tile_end below tile_start, which the kernel must
        # collapse to zero width before the (unsigned) grid.y split arithmetic.
        rows = torch.arange(s_q, device="cuda")
        ks = torch.where(rows % 3 == 1, min(100, s_k), 0)
        ke = torch.where(
            rows % 3 == 0,
            -1 - (rows % 7),
            torch.where(rows % 3 == 1, 0, torch.minimum(rows + 1, ks + s_k)),
        )
        return ks.to(torch.int32), ke.to(torch.int32)
    raise ValueError(f"unknown window mode: {mode}")


def _rehydrate(out, ks, ke, s_q, s_k, clean_logits):
    if clean_logits:
        return out
    full = torch.full((s_q, s_k), float("-inf"), device="cuda")
    # Clamp into [0, s_k] before slicing: cu_ends is allowed to be negative or
    # to sit below cu_starts (both mean "this row has no window"), and a raw
    # negative bound would wrap into a from-the-end slice and copy most of the
    # row instead of none of it.
    lo = ks.clamp(0, s_k)
    hi = ke.clamp(0, s_k)
    for i in range(s_q):
        a, b = int(lo[i]), int(hi[i])
        if a < b:
            full[i, a:b] = out[i, a:b]
    return full


def _kv_in_dtype(kv_fp8_fnuz, kv_dtype):
    if kv_dtype == e4m3_type:
        return kv_fp8_fnuz
    return kv_fp8_fnuz.to(torch.float32).to(kv_dtype)


@benchmark()
def test_fp8_mqa_logits(
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window
):
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, dtype=torch.float32)

    ks, ke = _make_windows(s_q, s_k, window)

    q_fp8 = q.to(DTYPE_MAP[q_dtype])
    # Grade against what the kernels actually consume: kv is already fake-
    # quantized above, so round-trip q through fp8 too. Otherwise q's
    # quantization error is charged to the kernel, which dominates the measured
    # diff and says nothing about kernel correctness.
    q = q_fp8.to(torch.float32).to(torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    # A no-op when the request is already the arch-native format.
    kv_fp8 = _kv_in_dtype(kv_fp8, DTYPE_MAP[kv_dtype])

    with torch.inference_mode():
        ref, cost = ref_fp8_mqa_logits(
            q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
        )
    ref_mask = ref == float("-inf")

    flops = cost.item() * num_heads * head_dim * 2
    nbytes = (
        s_q * num_heads * head_dim
        + s_k * head_dim  # Q + KV (fp8)
        + (s_k + s_q * num_heads) * 4  # scales + weights
        + 2 * s_q * 4  # ks + ke
        + s_q * s_k * 4  # output
    )

    candidates = {
        "flydsl": lambda: flydsl_fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits
        ),
    }
    if DTYPE_MAP[kv_dtype] == e4m3_type:
        candidates["triton"] = lambda: triton_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits
        )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        with torch.inference_mode():
            out, us = run_perftest(fn)
        out = _rehydrate(out, ks, ke, s_q, s_k, clean_logits)

        out_mask = out == float("-inf")
        assert torch.equal(out_mask, ref_mask), f"{name}: -inf mask mismatch"

        err = 0.0
        if not ref_mask.all():
            diff = calc_diff(out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0))
            # calc_diff is 1 - 2xy/(x^2+y^2), an aggregate similarity. Over a
            # single finite element it degenerates to (a-b)^2/(a^2+b^2), where
            # one borderline ReLU term (a dot product near zero flipping sign
            # between fp32 accumulation orders) moves it by percent. Both the
            # FlyDSL and Triton kernels land on the same value there and differ
            # from the fp32 reference identically, so assert the aggregate only
            # where it is meaningful; checkAllclose below still bounds the
            # magnitude in every case.
            if int((~ref_mask).sum()) > 1:
                assert diff < 1e-3, f"{name} calc_diff={diff}"
            err = diff.item()
            checkAllclose(
                ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
                out.masked_fill(out_mask, 0).to(dtypes.fp32),
                rtol=1e-2,
                atol=5.0,
                msg=f"{name}: fp8_mqa_logits",
                printLog=False,
            )

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0
        ret[f"{name} err"] = err

    return ret


def _cp_eligible(s_q, s_k):
    return s_k % s_q == 0 and s_q % 2 == 0


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fp8_mqa_logits unsupported on %s; skipping", get_gfx())
        return
    if flydsl_fp8_mqa_logits is None:
        aiter.logger.warning("flydsl package not installed; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fp8_mqa_logits correctness + perf sweep",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1),
            (1, 16),
            (1, 113),
            (17, 76),
            (61, 113),
            (61, 1024),
            (128, 1024),
            (1024, 1024),
            (1024, 1560),
            # Small-M / long-KV. The row grid alone is far too small to fill the
            # device here (64 rows is a 16-block grid on the gfx950 default
            # variant), so the launcher splits each row's KV window hard across
            # grid.y -- these are the only shapes reaching a split count high
            # enough that most blocks end up owning an empty column range.
            (64, 2048),
            (64, 8192),
            # s_kv < s_q. A causal mask then puts cu_ends below zero on the
            # leading rows, so these cover the negative-window path end to end.
            (128, 64),
            (1024, 1000),
        ],
    )
    parser.add_argument("--num-heads", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
    parser.add_argument(
        "--q-dtype",
        type=str,
        nargs="*",
        default=_DEFAULT_Q_DTYPES,
        choices=["fnuz", "fn"],
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        nargs="*",
        default=_DEFAULT_KV_DTYPES,
        choices=["fnuz", "fn"],
    )
    parser.add_argument(
        "--clean-logits",
        type=int,
        nargs="*",
        default=[0, 1],
        choices=[0, 1],
    )
    parser.add_argument(
        "-w",
        "--window",
        type=str,
        nargs="*",
        default=["causal", "cp", "misaligned", "empty"],
        choices=["causal", "cp", "misaligned", "empty"],
    )
    args = parser.parse_args()

    df = []
    for (s_q, s_k), nh, hd, qd, kvd, cl, win in itertools.product(
        args.shapes,
        args.num_heads,
        args.head_dim,
        args.q_dtype,
        args.kv_dtype,
        args.clean_logits,
        args.window,
    ):
        if win == "cp" and not _cp_eligible(s_q, s_k):
            continue
        df.append(
            test_fp8_mqa_logits(
                s_q,
                s_k,
                nh,
                hd,
                qd,
                kvd,
                bool(cl),
                win,
            )
        )

    df = pd.DataFrame(df)
    aiter.logger.info(
        "fp8_mqa_logits summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
