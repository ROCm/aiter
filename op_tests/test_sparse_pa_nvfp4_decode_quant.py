# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged decode with the MXFP8 o_proj input emitted by the attention
(``sparse_pa_nvfp4_decode(quant_out=...)``, FlyDSL, gfx950).

The codes and E8M0 scales must be byte-identical to quantizing the bf16 output of the same kernel
(group 32, E8M0 = ceil(amax / 448) exponent, amax floored at 1e-10, E4M3 round-to-nearest-even,
clamp +-448: the rule of ATOM's ``quantize_fp8``), for every split count, with and without the
in-kernel index selection, at the decode row counts 8..256, with 16 and with 8 query heads per kv
head (the two group sizes the kernel accepts; the group sets which lanes of a wave are live in
the head and scale stores).

It ends with a fused vs unfused timing table (``test_quant_perf``): the quantizing epilogue
against the attention writing bf16 followed by the aiter HIP MXFP8 activation quantize.

Run (exits nonzero if any check fails):
    python op_tests/test_sparse_pa_nvfp4_decode_quant.py
"""

import argparse
import itertools
import sys

import pandas as pd
import torch
from test_flydsl_sparse_pa_nvfp4 import (
    PAGE_B,
    SM_SCALE,
    make_case,
    perf_decode_inputs,
    perf_decode_topk,
)

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import sparse_pa_nvfp4_decode
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.test_common import benchmark, checkAllclose, run_perftest


def reference_quant(x):
    """[rows, N] bf16 -> (e4m3 codes [rows, N], e8m0 bytes [rows, N // 32])."""
    rows, n = x.shape
    v = x.float().view(rows, n // 32, 32)
    amax = v.abs().amax(-1).clamp_min(1e-10)
    bits = (
        amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    ).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    rcp = (((254 - code).clamp_min(0)) << 23).to(torch.int32).view(torch.float32)
    q = (v * rcp[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(rows, n), code.to(torch.uint8)


def run_case(name, seq_lens, ql, splits, select, scale_up=1.0, hkv=1, group=16):
    c = make_case(seq_lens, [ql] * len(seq_lens), hkv, group, seed=len(name) + splits)
    rows = c["q"].shape[0]
    q = c["q"]
    # scale_up multiplies the attention output (through the post-cache v_scale), so the quantizer sees
    # tiny / large magnitudes; scaling q would change the softmax instead
    v_scale = c["v_scale"] * scale_up
    h = q.shape[1]
    kw = {"query_len": ql, "splits": splits}
    topk = c["topk"]
    if (
        select
    ):  # the in-kernel top-16 selection (index_score=): same kernel, quant epilogue behind it
        width = max(seq_lens) // 128 + 8
        scores = torch.randn(hkv, rows, width, device=q.device, dtype=torch.float32)
        kw.update(
            index_score=scores,
            index_max_seq_len=max(seq_lens),
            init_blocks=1,
            local_blocks=2,
        )
        topk = torch.full((hkv, rows, 16), -1, dtype=torch.int32, device=q.device)
    out = torch.zeros_like(q)
    sparse_pa_nvfp4_decode(
        q,
        c["cache"],
        topk,
        c["bt"],
        c["seq_lens"],
        SM_SCALE,
        out,
        c["k_scale"],
        v_scale,
        **kw,
    )
    codes = torch.full((rows, h * 128), 0x55, dtype=torch.uint8, device=q.device)
    scales = torch.full((rows, h * 4), 0xAA, dtype=torch.uint8, device=q.device)
    r = sparse_pa_nvfp4_decode(
        q, c["cache"], topk, c["bt"], c["seq_lens"], SM_SCALE, None, c["k_scale"], v_scale,
        quant_out=(codes, scales), **kw,
    )  # fmt: skip
    torch.cuda.synchronize()
    ref_c, ref_s = reference_quant(out.view(rows, -1))
    ok_c = torch.equal(
        codes.view(torch.float8_e4m3fn).view(torch.uint8), ref_c.view(torch.uint8)
    )
    ok_s = torch.equal(scales, ref_s)
    ok = r is None and ok_c and ok_s
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:26s} rows={rows:3d} group={group} splits={splits} "
        f"codes={'ok' if ok_c else 'MISMATCH'} scales={'ok' if ok_s else 'MISMATCH'}"
    )
    if not ok_c:
        bad = (codes.view(torch.uint8) != ref_c.view(torch.uint8)).sum().item()
        print("   mismatching code bytes:", bad)
    return ok


@benchmark()
def test_quant_perf(rows, query_len, ctx, hkv, group):
    """One decode step of ``rows`` rows at ``ctx`` keys with random top-16 ids: the attention
    writing the MXFP8 o_proj input itself (``fused``, ``quant_out``) against the attention writing
    bf16 followed by the aiter HIP MXFP8 quantize (``unfused``: ``per_1x32_mx_quant_hip``, e8m0
    scales). The reference is the torch rule on the bf16 output; the codes and e8m0 bytes of both
    candidates must match it (compared as fp32 values of the codes and of the scale bytes).
    """
    pool, _, _, bt, seq, kv, sb, q, nblk, g = perf_decode_inputs(
        rows, query_len, hkv, group, ctx
    )
    topk = perf_decode_topk(hkv, rows, nblk, sb, g)
    h = hkv * group
    ks = vs = torch.ones(1, dtype=torch.float32, device="cuda")
    out = torch.empty_like(q)
    codes = torch.empty(rows, h * 128, dtype=torch.uint8, device="cuda")
    scales = torch.empty(rows, h * 4, dtype=torch.uint8, device="cuda")

    def attend(o):
        sparse_pa_nvfp4_decode(
            q, pool, topk, bt, seq, SM_SCALE, o, ks, vs, query_len=query_len
        )

    def fused():
        sparse_pa_nvfp4_decode(
            q, pool, topk, bt, seq, SM_SCALE, None, ks, vs, query_len=query_len,
            quant_out=(codes, scales),
        )  # fmt: skip
        return codes.view(torch.float8_e4m3fn), scales

    def unfused():
        attend(out)
        c, sc = per_1x32_mx_quant_hip(
            out.view(rows, h * 128), quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        return c.view(torch.float8_e4m3fn), sc.view(torch.uint8)

    attend(out)
    ref_c, ref_s = reference_quant(out.view(rows, -1))
    # roofline numerators: keys read through the ids (the tail block is partial), the pages, the
    # bf16 q read and the MXFP8 output (codes + one scale byte per 32 values) written
    keys = int(((topk >= 0).sum() * 128 - ((sb + 1) * 128 - kv).sum() * hkv).item())
    nbytes = (
        int((topk >= 0).sum().item()) * 2 * PAGE_B
        + q.numel() * q.element_size()
        + codes.numel()
        + scales.numel()
    )
    flops = 4 * group * 128 * keys
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in {"fused": fused, "unfused": unfused}.items():
        (c, sc), us = run_perftest(fn, num_iters=20, num_warmup=3)
        err = checkAllclose(
            ref_c.to(torch.float32), c.reshape(rows, -1).to(torch.float32),
            rtol=0, atol=0, msg=f"{name}: e4m3 codes",
        )  # fmt: skip
        err_s = checkAllclose(
            ref_s.to(torch.float32), sc.reshape(rows, -1).to(torch.float32),
            rtol=0, atol=0, msg=f"{name}: e8m0 scales",
        )  # fmt: skip
        assert err == 0 and err_s == 0, f"{name}: codes {err:.2%} scales {err_s:.2%}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err + err_s
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="MXFP8 o_proj output (quant_out) tests"
    )
    parser.add_argument("-r", "--rows", type=int, nargs="*", default=[8, 64, 256])
    parser.add_argument("-c", "--ctx", type=int, nargs="*", default=[32768, 175000])
    parser.add_argument(
        "-g", "--heads", type=dtypes.str2tuple, nargs="*", default=[(1, 16), (2, 8)],
        help="(kv heads, q heads per kv head)",
    )  # fmt: skip
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        print("SKIP: gfx950 only")
        return 0
    ok = True
    for splits in (1, 2, 4):
        for name, lens, ql in (
            ("g1_k7", [175000], 8),
            ("g2_k7", [175000, 90001], 8),
            ("g7_k7", [175000, 90001, 4097, 32768, 20000, 100000, 60000], 8),
            ("g16_k7", [100000 + 777 * i for i in range(16)], 8),
            ("g32_k7", [60000 + 1111 * i for i in range(32)], 8),
            ("short", [1, 100, 1500, 2047, 2048, 2049, 129, 130], 1),
        ):
            ok &= run_case(f"{name}", lens, ql, splits, False)
    for splits in (1, 4):
        ok &= run_case(
            "g7_k7/select",
            [175000, 90001, 4097, 32768, 20000, 100000, 60000],
            8,
            splits,
            True,
        )
        ok &= run_case(
            "g16_k7/select", [100000 + 777 * i for i in range(16)], 8, splits, True
        )
    for hkv in (2, 4):  # more than one kv head per rank (head = kv head * 16 + lane)
        ok &= run_case(f"g3_k7/hkv{hkv}", [175000, 90001, 4097], 8, 4, False, hkv=hkv)
    for sc in (1e-3, 30.0):  # tiny / large magnitudes: scales across the e8m0 range
        ok &= run_case(
            f"g7_k7/scale{sc}",
            [175000, 90001, 4097, 32768, 20000, 100000, 60000],
            8,
            4,
            False,
            sc,
        )
    # group 8 (8 query heads per kv head): only half of a wave's lanes hold a head of the group,
    # so the masked head stores and the scale stores are on a different lane map than group 16
    for hkv in (1, 2, 4):
        for splits in (1, 4):
            ok &= run_case(
                f"g7_k7/group8/hkv{hkv}",
                [175000, 90001, 4097, 32768, 20000, 100000, 60000],
                8,
                splits,
                False,
                hkv=hkv,
                group=8,
            )
    ok &= run_case("short/group8/hkv2", [1, 100, 1500, 2047, 2048, 2049, 129, 130], 1, 1, False, hkv=2, group=8)  # fmt: skip
    for splits in (1, 4):
        ok &= run_case(
            "g7_k7/group8/select",
            [175000, 90001, 4097, 32768, 20000, 100000, 60000],
            8,
            splits,
            True,
            hkv=2,
            group=8,
        )
    ok &= run_case("g7_k7/group8/scale30", [175000, 90001, 4097, 32768, 20000, 100000, 60000], 8, 4, False, 30.0, hkv=2, group=8)  # fmt: skip
    perf = [
        test_quant_perf(rows, 8, ctx, hkv, group)
        for rows, ctx, (hkv, group) in itertools.product(
            args.rows, args.ctx, args.heads
        )
    ]
    df = pd.DataFrame(perf)
    aiter.logger.info(
        "fused quant_out vs attention + MXFP8 quantize (markdown):\n%s",
        df.to_markdown(index=False),
    )
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
