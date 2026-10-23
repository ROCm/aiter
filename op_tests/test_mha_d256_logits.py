# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""flash_attn_func sweep: bf16, BSHD, Q/K/V head_dim=256.

Dense MHA prefill (same as op_tests/test_mha.py / flash_attn_func). There is no
decode phase. Shape defaults follow the M x N serving axes in ROCm/aiter#5434:

  seqlen_q  1024, 2048, 4096, 8192, 16384
  seqlen_k  4096, 16384, 65664  (skips seqlen_k < seqlen_q)
  nheads    32, 64
  batch     1, 16

Shapes that hit HIP illegal access on MI355X (b>=64, and b=16/h=64/sk=131072)
are not in the default product.

Layout is BSHD [B, S, H, D]; D is fixed at 256. No dropout, bias, alibi, or
local window. Forward only.

The full [B, H, Sq, Sk] fp32 score tensor does not fit at the large shapes, so
accuracy is checked on a query-row sample (grid edges first).

Examples:
    python op_tests/test_mha_d256_logits.py
    python op_tests/test_mha_d256_logits.py -n 32 -b 1 -q 1024 -k 4096
    python op_tests/test_mha_d256_logits.py --ref
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.test_mha_common import attention_ref

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEAD_DIM = 256
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
    """Bottom-right causal visibility for sampled query rows [R, Sk]."""
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


def run_flash(q, k, v, causal=True):
    ret, us = run_perftest(
        aiter.flash_attn_func,
        q,
        k,
        v,
        0.0,
        None,
        causal,
        (-1, -1, 0),
        None,
        None,
        False,
        return_lse=True,
        return_attn_probs=False,
        how_v3_bf16_cvt=2,
        num_rotate_args=1,
    )
    if not isinstance(ret, (tuple, list)):
        return ret, None, us
    out = ret[0]
    softmax_lse = ret[1] if len(ret) > 1 else None
    return out, softmax_lse, us


@benchmark()
def test_flash_attn_bshd_d256(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    causal,
    check_ref=False,
):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    dtype = dtypes.bf16
    d = HEAD_DIM

    q = torch.randn(batch_size, seqlen_q, nheads, d, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, dtype=dtype)

    out, softmax_lse, us = run_flash(q, k, v, causal=causal)

    err = None
    ref_n = 0
    if check_ref:
        rows = _ref_rows(seqlen_q, batch_size, nheads, seqlen_k)
        ref_n = int(rows.numel())
        q_s = q[:, rows]
        attn_bias = _causal_bias(rows, seqlen_q, seqlen_k, q.device) if causal else None
        out_ref, lse_ref = run_torch(q_s, k, v, causal=causal, attn_bias=attn_bias)
        out_pt, lse_pt = run_torch(
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
            msg=f"flash_attn BSHD d={d} b={batch_size} h={nheads} "
            f"sq={seqlen_q} sk={seqlen_k} causal={causal}",
        )
        lse_got = softmax_lse[:, :, rows] if softmax_lse is not None else None
        if lse_got is not None:
            checkAllclose(
                lse_ref.float(),
                lse_got.float(),
                atol=max(
                    2 * (lse_pt.float() - lse_ref.float()).abs().max().item(), 0.01
                ),
                rtol=0.0,
                msg=f"flash_attn lse b={batch_size} h={nheads} sq={seqlen_q} sk={seqlen_k}",
            )
    else:
        aiter.logger.info(
            "flash_attn BSHD d=%s b=%s h=%s sq=%s sk=%s [no-ref] %.2f us",
            d,
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            us,
        )

    flops = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d * 2)
    )
    if causal:
        flops = flops / 2
    nbytes = (
        batch_size
        * nheads
        * 2
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d + seqlen_q * d)
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
        aiter.logger.warning("mha_d256_logits unsupported on %s; skipping", get_gfx())
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
        default=[1, 16],
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
    args = parser.parse_args()

    rows = []
    for batch_size, nheads, seqlen_q, seqlen_k in itertools.product(
        args.batch_size, args.nheads, args.seqlen_q, args.seqlen_k
    ):
        if seqlen_k < seqlen_q:
            continue
        try:
            rows.append(
                test_flash_attn_bshd_d256(
                    batch_size,
                    nheads,
                    seqlen_q,
                    seqlen_k,
                    args.causal,
                    check_ref=args.ref,
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
    _summarize("mha_d256_logits", rows)


if __name__ == "__main__":
    main()
