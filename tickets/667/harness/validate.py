#!/usr/bin/env python3
"""D7-lite numerical cross-check of the CK warp-decode kernels.

The CK bench (built from ``ck_bench_warp_decode.cpp``) run with ``CK_WD_VALIDATE=1``
dumps, for the FP8/BF16 kernels, the *exact* quantized inputs it fed the GPU plus
the GPU output.  This script reloads those identical bytes, rebuilds the torch
reference (the same math the FlyDSL op-test reference uses), and reports cosine
similarity + max abs delta per kernel.

Scope (D7-lite): gate_bf16_d2, gate_fp8_d2, down_h2_d2.  FP4 is skipped -- it needs
the e8m0 (1,32) MXFP4 pack (tracked in full D7 / B1).  Scales are a single uniform
constant per operand, so this validates the matmul + fp8 dequant + silu + router
math independent of CK's internal Block2D scale-index layout (that layout check is
also deferred to full D7).

Usage:
    python validate.py [--dir DIR] [--cos 0.99]

Exit code 0 iff every dumped kernel passes.
"""

import argparse
import json
import os
import sys

import torch


def _load(path, torch_dtype):
    raw = torch.frombuffer(bytearray(open(path, "rb").read()), dtype=torch.uint8)
    return raw.view(torch_dtype)


def _fp8(path):
    return _load(path, torch.float8_e4m3fn).float()


def _bf16(path):
    return _load(path, torch.bfloat16).float()


def _i32(path):
    return _load(path, torch.int32)


def _f32(path):
    return _load(path, torch.float32)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1), b.reshape(-1), dim=0
    ).item()


def _report(name, ref, got, cos_thresh):
    ref, got = ref.float(), got.float()
    cos = _cos(ref, got)
    max_delta = (ref - got).abs().max().item()
    denom = ref.abs().max().item() + 1e-6
    ok = cos >= cos_thresh
    print(
        f"  {name:<14} cos={cos:.6f} (thresh {cos_thresh})  "
        f"max_delta={max_delta:.4f} ({100 * max_delta / denom:.2f}% of max)  "
        f"--> {'PASS' if ok else 'FAIL'}"
    )
    print(f"      ref  sample: {ref.reshape(-1)[:5].tolist()}")
    print(f"      ck   sample: {got.reshape(-1)[:5].tolist()}")
    return ok


def _ref_gate_up(x, wg, wu, rids, B, H, I, K, E, wscale, xscale):  # noqa: E741
    """silu(gate)*up with uniform weight+act scales. x already dequantized (float).

    xscale is folded here because for the fp8-act kernel x carries its own scale;
    for bf16-act pass xscale=1.0.
    """
    x = x.reshape(B, H) * xscale
    wg = (wg.reshape(E, I, H)) * wscale
    wu = (wu.reshape(E, I, H)) * wscale
    rids = rids.reshape(B, K)
    out = torch.empty(B, K, I)
    for b in range(B):
        for k in range(K):
            e = int(rids[b, k])
            gate = x[b] @ wg[e].T
            up = x[b] @ wu[e].T
            silu = gate / (1.0 + torch.exp(-gate))
            out[b, k] = silu * up
    return out


def _ref_down(inter, wd, rids, rwts, B, H, I, K, E, wscale):  # noqa: E741
    inter = inter.reshape(B, K, I)
    wd = wd.reshape(E, H, I) * wscale
    rids = rids.reshape(B, K)
    rwts = rwts.reshape(B, K)
    y = torch.zeros(B, H)
    for b in range(B):
        for k in range(K):
            e = int(rids[b, k])
            y[b] += float(rwts[b, k]) * (inter[b, k] @ wd[e].T)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", default=os.path.join(os.path.dirname(__file__), "ck_validate_dump")
    )
    ap.add_argument("--cos", type=float, default=0.99)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.dir, "manifest.json")))
    B, H, I, K, E = (man[k] for k in ("B", "H", "I", "K", "E"))  # noqa: E741
    wscale, xscale = man["wscale"], man["xscale"]
    kernels = man["kernels"]
    print(f"CK D7-lite validation  dir={args.dir}")
    print(f"  shape B={B} H={H} I={I} K={K} E={E}  wscale={wscale} xscale={xscale}")
    print(f"  kernels dumped: {kernels}\n")
    if not kernels:
        print("No kernels dumped (all unsupported at the validation shape).")
        return 1

    def p(name):
        return os.path.join(args.dir, name)

    all_ok = True

    if "gate_bf16_d2" in kernels:
        ref = _ref_gate_up(
            _bf16(p("gate_bf16_d2.x.bin")),
            _fp8(p("gate_bf16_d2.wg.bin")),
            _fp8(p("gate_bf16_d2.wu.bin")),
            _i32(p("gate_bf16_d2.rids.bin")),
            B,
            H,
            I,
            K,
            E,
            wscale,
            xscale=1.0,  # bf16 act: no x scale
        )
        got = _bf16(p("gate_bf16_d2.out.bin")).reshape(B, K, I)
        all_ok &= _report("gate_bf16_d2", ref, got, args.cos)

    if "gate_fp8_d2" in kernels:
        ref = _ref_gate_up(
            _fp8(p("gate_fp8_d2.x.bin")),
            _fp8(p("gate_fp8_d2.wg.bin")),
            _fp8(p("gate_fp8_d2.wu.bin")),
            _i32(p("gate_fp8_d2.rids.bin")),
            B,
            H,
            I,
            K,
            E,
            wscale,
            xscale=xscale,
        )
        got = _bf16(p("gate_fp8_d2.out.bin")).reshape(B, K, I)
        all_ok &= _report("gate_fp8_d2", ref, got, args.cos)

    if "down_h2_d2" in kernels:
        ref = _ref_down(
            _bf16(p("down_h2_d2.inter.bin")),
            _fp8(p("down_h2_d2.wd.bin")),
            _i32(p("down_h2_d2.rids.bin")),
            _f32(p("down_h2_d2.rwts.bin")),
            B,
            H,
            I,
            K,
            E,
            wscale,
        )
        got = _bf16(p("down_h2_d2.out.bin")).reshape(B, H)
        all_ok &= _report("down_h2_d2", ref, got, args.cos)

    print(f"\n{'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
