# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged decode with the MXFP8 o_proj input emitted by the attention
(``sparse_pa_nvfp4_decode(quant_out=...)``, FlyDSL, gfx950).

The codes and E8M0 scales must be byte-identical to quantizing the bf16 output of the same kernel
(group 32, E8M0 = ceil(amax / 448) exponent, amax floored at 1e-10, E4M3 round-to-nearest-even,
clamp +-448: the rule of ATOM's ``quantize_fp8``), for every split count, with and without the
in-kernel index selection, at the decode row counts 8..256.

Run (exits nonzero if any check fails):
    python op_tests/test_sparse_pa_nvfp4_decode_quant.py
"""

import sys

import torch
from test_flydsl_sparse_pa_nvfp4 import SM_SCALE, make_case

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import sparse_pa_nvfp4_decode


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


def run_case(name, seq_lens, ql, splits, select, scale_up=1.0, hkv=1):
    group = 16
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
        f"{'PASS' if ok else 'FAIL'}  {name:26s} rows={rows:3d} splits={splits} "
        f"codes={'ok' if ok_c else 'MISMATCH'} scales={'ok' if ok_s else 'MISMATCH'}"
    )
    if not ok_c:
        bad = (codes.view(torch.uint8) != ref_c.view(torch.uint8)).sum().item()
        print("   mismatching code bytes:", bad)
    return ok


def main():
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
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
