# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA decode must never return wrong attention when kv_buffer spans more than 4 GiB.

For each decode config the same requests are placed at page 0 and just past the
4 GiB mark of one kv_buffer (576-element rows, page_size 1). Each call must either
match the torch reference at both placements, or refuse with an error at both.
A kernel that forms KV offsets in 32 bits returns the right answer low and the
wrong answer high, which is what this test catches.

Examples:
  python op_tests/test_mla_kv_4gib.py
  python op_tests/test_mla_kv_4gib.py -d fp8 -kvd fp8 -n 32 --qlen 1 --mode ps
"""

from __future__ import annotations

import argparse
import itertools
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

KV_LORA, ROPE = 512, 64
D = KV_LORA + ROPE
FOUR_GIB = 1 << 32
TOL = 0.05


def _ref(q, rows, bs, qlen, ctx, sm_scale, causal):
    qf = q.float().view(bs, qlen, q.shape[1], D)
    out = []
    for b in range(bs):
        r = rows[b * ctx : (b + 1) * ctx]
        for t in range(qlen):
            n = ctx - (qlen - 1) + t if causal else ctx
            s = torch.einsum("hd,nd->hn", qf[b, t], r[:n]) * sm_scale
            out.append(torch.softmax(s, -1) @ r[:n, :KV_LORA])
    return torch.stack(out)


def _decode(q, kv, idx, bs, qlen, ctx, persistent, sm_scale):
    dev = q.device
    q_dtype, kv_dtype, nhead = q.dtype, kv.dtype, q.shape[1]
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * qlen
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * ctx
    last = torch.ones(bs, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    o = torch.empty(bs * qlen, nhead, KV_LORA, dtype=dtypes.bf16, device=dev)
    kw = {}
    if q_dtype == dtypes.fp8:
        kw.update(q_scale=one, kv_scale=one)
    elif kv_dtype == dtypes.fp8:
        kw.update(kv_scale=one)
    if persistent:
        info = aiter.get_mla_metadata_info_v1(
            bs, qlen, nhead, q_dtype, kv_dtype, is_sparse=False, fast_mode=True
        )
        wmd, wip, wis, rip, rfm, rpm = [
            torch.empty(size, dtype=dtype, device=dev) for size, dtype in info
        ]
        aiter.get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            last,
            nhead,
            1,
            True,
            wmd,
            wis,
            wip,
            rip,
            rfm,
            rpm,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=qlen,
            uni_seqlen_qo=qlen,
            fast_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )
        kw.update(
            work_meta_data=wmd,
            work_indptr=wip,
            work_info_set=wis,
            reduce_indptr=rip,
            reduce_final_map=rfm,
            reduce_partial_map=rpm,
        )
    aiter.mla.mla_decode_fwd(
        q,
        kv.view(-1, 1, 1, D),
        o,
        qo_indptr,
        kv_indptr,
        idx,
        last,
        qlen,
        page_size=1,
        nhead_kv=1,
        sm_scale=sm_scale,
        **kw,
    )
    torch.cuda.synchronize()
    return o.float()


def test_mla_kv_4gib(kv, q_dtype, nhead, qlen, persistent, bs=4, ctx=256):
    """Returns 'ok', 'refused', 'unsupported', or 'WRONG'."""
    dev = kv.device
    torch.manual_seed(0)
    need = bs * ctx
    high = FOUR_GIB // (D * kv.element_size()) + 4096
    src = torch.randn(need, D, device=dev).to(kv.dtype)
    q = (torch.randn(bs * qlen, nhead, D, device=dev) * 0.5).to(q_dtype)
    sm_scale = 1.0 / (D**0.5)
    # Multi-token non-persistent kernels attend to the whole context (no causal
    # mask); judge each call against whichever reference it matches at page 0.
    refs = [_ref(q, src.float(), bs, qlen, ctx, sm_scale, c) for c in (True, False)]
    scale = refs[0].abs().max().item()
    results = []
    for base in (0, high):
        kv[base : base + need] = src
        idx = torch.arange(base, base + need, dtype=torch.int32, device=dev)
        try:
            out = _decode(q, kv, idx, bs, qlen, ctx, persistent, sm_scale)
            results.append([(out - r).abs().max().item() / scale for r in refs])
        except Exception as e:  # noqa: BLE001
            results.append("32-bit offsets" in str(e) or str(e)[:200])
        finally:
            kv[base : base + need] = 0
    low, hi = results
    if isinstance(low, list) and isinstance(hi, list):
        pick = 0 if low[0] <= low[1] else 1
        low, hi = low[pick], hi[pick]
    if isinstance(low, float) and isinstance(hi, float):
        if low > TOL:
            return "unsupported"  # config is wrong at page 0 too; out of scope here
        return "ok" if hi <= TOL else "WRONG"
    if low is True and hi is True:
        return "refused"
    if isinstance(low, str) and isinstance(hi, str) and low == hi:
        return "unsupported"
    return f"WRONG (low={low}, high={hi})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-d", "--dtype", nargs="*", default=["bf16", "fp8"])
    parser.add_argument("-kvd", "--kv_dtype", nargs="*", default=["bf16", "fp8"])
    parser.add_argument(
        "-n", "--nhead", type=int, nargs="*", default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--qlen", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument(
        "--mode", nargs="*", default=["ps", "nps"], choices=["ps", "nps"]
    )
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "test_mla_kv_4gib targets gfx950; skipping on %s", get_gfx()
        )
        return 0

    failures = 0
    for kvd in args.kv_dtype:
        kv_dtype = dtypes.d_dtypes[kvd]
        rows = FOUR_GIB // (D * kv_dtype.itemsize) + 4096 + 4 * 256 + 16
        kv = torch.zeros(rows, D, dtype=kv_dtype, device="cuda")
        for qd, nhead, qlen, mode in itertools.product(
            args.dtype, args.nhead, args.qlen, args.mode
        ):
            if qd == "fp8" and kvd != "fp8":
                continue
            if (
                mode == "ps"
                and nhead < 16
                and kvd == "fp8"
                and not (qd == "fp8" and qlen == 4)
            ):
                continue  # get_mla_metadata_v1 aborts the process for these head counts
            verdict = test_mla_kv_4gib(
                kv, dtypes.d_dtypes[qd], nhead, qlen, mode == "ps"
            )
            failures += verdict.startswith("WRONG")
            aiter.logger.info(
                f"mla_kv_4gib q={qd} kv={kvd} nhead={nhead} qlen={qlen} {mode}: {verdict}"
            )
        del kv
        torch.cuda.empty_cache()
    aiter.logger.info(f"mla_kv_4gib: {failures} config(s) read wrong KV past 4 GiB")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
