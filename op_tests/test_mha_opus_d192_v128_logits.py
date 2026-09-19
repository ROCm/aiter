# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fmha_fwd_bf16_opus_fwd sweep: bf16, BSHD, D_QK=192 / D_V=128.

Dense MHA prefill through the OPUS gfx950 kernel (same path as
test_flash_attn_func_opus_d192_v128 in op_tests/test_mha.py). There is no
decode phase. Shape defaults follow the M x N serving axes in ROCm/aiter#5434:

  seqlen_q  1024, 2048, 4096, 8192, 16384
  seqlen_k  4096, 16384, 65664  (skips seqlen_k < seqlen_q)
  nheads    32, 64
  batch     1, 16, 64, 128

Shapes that OOM'd on MI355X (sk=131072; b=128/h=64/sk=65664; b=256) are
not in the default product.

Layout is BSHD; Q/K last dim 192, V last dim 128. nheads_k = nheads. No dropout,
bias, alibi, or local window. Forward only. gfx950 only.

The full [B, H, Sq, Sk] fp32 score tensor does not fit at the large shapes, so
output accuracy is checked on a query-row sample. LSE uses opus_ref_lse, which
already chunks over query rows.

Examples:
    python op_tests/test_mha_opus_d192_v128_logits.py
    python op_tests/test_mha_opus_d192_v128_logits.py -n 32 -b 1 -q 1024 -k 4096
    python op_tests/test_mha_opus_d192_v128_logits.py --ref
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import fmha_fwd_bf16_opus_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.test_mha_common import attention_ref, opus_check_lse, opus_ref_lse

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
D_QK = 192
D_V = 128
_REF_SCORE_BYTES = 1 << 30
_REF_MAX_ROWS = 64


def _ref_rows(seqlen_q, batch, nheads, seqlen_k):
    budget = max(1, _REF_SCORE_BYTES // (max(batch, 1) * nheads * seqlen_k * 4))
    want = max(1, min(seqlen_q, _REF_MAX_ROWS, budget))
    spread = torch.linspace(0, seqlen_q - 1, steps=want).round().long().tolist()
    rows = []
    for r in [
        0,
        1,
        seqlen_q // 2,
        seqlen_q // 2 + 1,
        seqlen_q - 2,
        seqlen_q - 1,
    ] + spread:
        if 0 <= r < seqlen_q and r not in rows:
            rows.append(r)
        if len(rows) >= want:
            break
    return torch.tensor(sorted(rows), dtype=torch.long)


def _causal_bias(rows, seqlen_q, seqlen_k, device):
    q_pos = rows.to(device)
    k_pos = torch.arange(seqlen_k, device=device)
    visible = k_pos[None, :] <= (seqlen_k - seqlen_q + q_pos)[:, None]
    bias = torch.zeros(1, 1, rows.numel(), seqlen_k, device=device)
    bias.masked_fill_(~visible[None, None], float("-inf"))
    return bias


def run_torch(q, k, v, causal=True, upcast=True, reorder_ops=False, attn_bias=None):
    out, _, softmax_lse = attention_ref(
        q,
        k,
        v,
        attn_bias=attn_bias,
        causal=causal if attn_bias is None else False,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )
    return out, softmax_lse


@benchmark()
def test_mha_opus_d192_v128(
    batch_size, nheads, seqlen_q, seqlen_k, causal, check_ref=False
):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    dtype = dtypes.bf16

    q = torch.randn(batch_size, seqlen_q, nheads, D_QK, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, D_QK, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, D_V, dtype=dtype)

    (out, lse), us = run_perftest(
        fmha_fwd_bf16_opus_fwd,
        q,
        k,
        v,
        D_QK**-0.5,
        causal,
        return_lse=True,
        num_rotate_args=1,
    )
    aiter.logger.info(
        "opus d192/v128 perf b=%s h=%s sq=%s sk=%s  %.2f us",
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        us,
    )

    err = None
    ref_n = 0
    if check_ref:
        rows = _ref_rows(seqlen_q, batch_size, nheads, seqlen_k)
        ref_n = int(rows.numel())
        q_s = q[:, rows]
        attn_bias = _causal_bias(rows, seqlen_q, seqlen_k, q.device) if causal else None
        out_ref, _ = run_torch(q_s, k, v, causal=causal, attn_bias=attn_bias)
        out_pt, _ = run_torch(
            q_s,
            k,
            v,
            causal=causal,
            attn_bias=attn_bias,
            upcast=False,
            reorder_ops=True,
        )

        got = out[:, rows]
        out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
        err = checkAllclose(
            out_ref,
            got,
            atol=out_tol,
            rtol=0.0,
            msg=f"opus d192/v128 BSHD b={batch_size} h={nheads} "
            f"sq={seqlen_q} sk={seqlen_k} causal={causal} {us:.2f} us",
        )

        lse_ref = opus_ref_lse(q, k, causal)
        assert tuple(lse.shape) == (
            batch_size,
            nheads,
            seqlen_q,
        ), f"lse {tuple(lse.shape)}"
        opus_check_lse("opus-d192", lse, lse_ref)
        dead = torch.isneginf(lse_ref)
        if dead.any():
            dead_o = dead.permute(0, 2, 1).unsqueeze(-1).expand_as(out)
            assert (out[dead_o] == 0).all(), "fully-masked rows must produce O=0"

    flops = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * D_QK * 2 + seqlen_q * seqlen_k * D_V * 2)
    )
    if causal:
        flops = flops / 2
    nbytes = (
        batch_size
        * nheads
        * 2
        * (seqlen_q * D_QK + seqlen_k * D_QK + seqlen_k * D_V + seqlen_q * D_V)
    )

    return {
        "gfx": get_gfx(),
        "ref_rows": ref_n,
        "fwd us": us,
        "fwd TFLOPS": flops / us / 1e6,
        "fwd TB/s": nbytes / us / 1e6,
        "fwd err": err,
    }


def _summarize(name, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    keep = [
        c
        for c in (
            "batch_size",
            "nheads",
            "seqlen_q",
            "seqlen_k",
            "causal",
            "gfx",
            "ref_rows",
            "fwd us",
            "fwd TFLOPS",
            "fwd TB/s",
            "fwd err",
        )
        if c in df.columns
    ]
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, df[keep].to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mha_opus_d192_v128_logits unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128],
        help="""Prefill batch size.
    e.g.: -b 1 16""",
    )
    parser.add_argument(
        "-n",
        "--nheads",
        type=int,
        nargs="*",
        default=[32, 64],
        help="""Q/K/V heads (MHA, nheads_k = nheads).
    e.g.: -n 32""",
    )
    parser.add_argument(
        "-q",
        "--seqlen_q",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 16384],
        help="""Query length.
    e.g.: -q 1024 4096""",
    )
    parser.add_argument(
        "-k",
        "--seqlen_k",
        type=int,
        nargs="*",
        default=[4096, 16384, 65664],
        help="""KV length. Skips seqlen_k < seqlen_q.
    e.g.: -k 4096 16384""",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="causal mask. Default: True.",
    )
    parser.add_argument(
        "--ref",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare against torch golden. Default: False. Pass --ref to enable.",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Alias for --no-ref (kernel timing only).",
    )
    args = parser.parse_args()
    check_ref = bool(args.ref) and not args.perf

    rows = []
    for batch_size, nheads, seqlen_q, seqlen_k in itertools.product(
        args.batch_size, args.nheads, args.seqlen_q, args.seqlen_k
    ):
        if seqlen_k < seqlen_q:
            continue
        try:
            rows.append(
                test_mha_opus_d192_v128(
                    batch_size,
                    nheads,
                    seqlen_q,
                    seqlen_k,
                    args.causal,
                    check_ref=check_ref,
                )
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            aiter.logger.warning(
                "OOM skip b=%s h=%s sq=%s sk=%s",
                batch_size,
                nheads,
                seqlen_q,
                seqlen_k,
            )
        torch.cuda.empty_cache()
    _summarize("mha_opus_d192_v128_logits", rows)


if __name__ == "__main__":
    main()
