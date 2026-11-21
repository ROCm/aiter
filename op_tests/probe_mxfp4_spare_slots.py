# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Probe 4: mxfp4 (a8w4, per_1x32) with spare expert slots + the is_shuffled trap.

Probe 3 found no "spare slots zero the output" behaviour in bf16 or fp8
blockscale.  mxfp4 is the one production path it could not reach, and it is
what DSv4 actually runs on gfx950.  Two questions here:

  Q1  Does declaring experts that no token routes to break the mxfp4 path?
      Same sweep as probe 3, compared against a compacted reference.

  Q2  ``fused_moe.py:758`` reads the shuffle flag off the tensor:
          isShuffled = getattr(w1, "is_shuffled", False) or getattr(w2, ...)
      ``shuffle_weight`` sets it (``ops/shuffle.py:189,205``).  A weight tensor
      re-wrapped over foreign memory -- exactly what the VMM ``[E+B]`` pool
      does via ``__cuda_array_interface__`` -- does NOT carry the attribute.
      Does that silently change the numbers, and does setattr restore them?
      (We have been bitten by the slicing version of this already:
      bug_aiter_is_shuffled_attribute_lost_on_slice -- no error, wrong values.)

Single GPU.

Run:
    python3 op_tests/probe_mxfp4_spare_slots.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Force the fp8 (a8w4) kernel regardless of token count: below the default
# AITER_BF16_FP8_MOE_BOUND (256) the picker selects bf16/a16w4, which for Silu
# at ksplit<=1 has no kernel and dispatch-crashes. Mirrors test_moe_ep.py.
os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"

import torch  # noqa: E402

import aiter  # noqa: E402
from aiter import ActivationType, QuantType, dtypes, get_gfx  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4  # noqa: E402


def mxfp4_quant(w):
    q, s = aiter.get_torch_quant(QuantType.per_1x32)(w, quant_dtype=dtypes.fp4x2)
    return q.view(w.shape[0], w.shape[1], w.shape[2] // 2), s


def prep(w1, w2, n_experts):
    """mxfp4 quant + the CK a16w4 interleaved shuffle used by a8w4 on gfx950."""
    w1_qt, w1_sc = mxfp4_quant(w1)
    w2_qt, w2_sc = mxfp4_quant(w2)
    return (
        shuffle_weight_a16w4(w1_qt, 16, True),
        shuffle_weight_a16w4(w2_qt, 16, False),
        shuffle_scale_a16w4(w1_sc, n_experts, True),
        shuffle_scale_a16w4(w2_sc, n_experts, False),
    )


def call(x, w1, w2, ids, tw, s1, s2, mask=None):
    return fused_moe(
        x, w1, w2, tw, ids, mask,
        activation=ActivationType.Silu,
        gate_mode=GateMode.INTERLEAVE.value,
        quant_type=QuantType.per_1x32,
        w1_scale=s1, w2_scale=s2,
    )


class Rewrap:
    """Adopt another tensor's storage -- what the VMM pool does, minus the VMM."""

    def __init__(self, t: torch.Tensor):
        self.__cuda_array_interface__ = {
            "data": (t.data_ptr(), False),
            "shape": (t.numel() * t.element_size(),),
            "typestr": "<u1",
            "strides": None,
            "version": 3,
        }


def rewrap(t: torch.Tensor) -> torch.Tensor:
    return (
        torch.as_tensor(Rewrap(t), device="cuda").view(t.dtype).view(*t.shape)
    )


def verdict(ref, got):
    if got is None:
        return "ERROR", float("nan")
    if bool((got == 0).all()):
        return "ZERO", float("nan")
    d = (ref.float() - got.float()).abs()
    rel = d.max().item() / max(ref.float().abs().max().item(), 1e-9)
    return ("OK" if rel < 5e-2 else "WRONG"), rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()

    if get_gfx() != "gfx950":
        print(f"skip: mxfp4 a8w4 needs gfx950, got {get_gfx()}")
        return 0

    dev = "cuda:0"
    torch.cuda.set_device(0)
    H, I, n = args.hidden, args.inter, args.tokens
    g = torch.Generator(device=dev).manual_seed(4242)
    x = (torch.randn(n, H, generator=g, device=dev) * 0.1).to(dtypes.bf16)

    cases = [
        ("all 16 routed", 16, list(range(16))),
        ("18 decl, 16 routed", 18, list(range(16))),  # the MoonEP E+B shape
        ("16 decl, 8 trailing", 16, list(range(8))),
        ("16 decl, 8 even", 16, list(range(0, 16, 2))),
        ("48 decl, 6 scattered", 48, [0, 7, 13, 22, 35, 44]),
    ]

    print(f"{'=' * 78}\nmxfp4 a8w4 (per_1x32)  H={H} I={I} tokens={n}\n{'=' * 78}")
    for cname, E_decl, routed in cases:
        k = len(routed)
        gw = torch.Generator(device=dev).manual_seed(7)
        w1 = (torch.randn(E_decl, 2 * I, H, generator=gw, device=dev) * 0.05).to(
            dtypes.bf16
        )
        w2 = (torch.randn(E_decl, H, I, generator=gw, device=dev) * 0.05).to(
            dtypes.bf16
        )
        ids_g = torch.tensor(
            [routed[i % k] for i in range(n)], dtype=torch.int32, device=dev
        ).view(n, 1)
        remap = {e: i for i, e in enumerate(routed)}
        ids_l = torch.tensor(
            [remap[int(v)] for v in ids_g.flatten()], dtype=torch.int32, device=dev
        ).view(n, 1)
        tw = torch.ones(n, 1, dtype=torch.float32, device=dev)

        print(f"\n  [{cname}]  E_decl={E_decl} routed={k} spare={E_decl - k}")
        try:
            c1, c2, cs1, cs2 = prep(w1[routed].contiguous(), w2[routed].contiguous(), k)
            ref = call(x, c1, c2, ids_l, tw, cs1, cs2)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"    reference ERROR {type(e).__name__}: {e}")
            continue
        if bool((ref == 0).all()):
            print("    !! reference itself all-zero, case unusable")
            continue

        f1, f2, fs1, fs2 = prep(w1, w2, E_decl)
        print(f"    is_shuffled on shuffled w1: {getattr(f1, 'is_shuffled', False)}")

        # Q1: spare slots
        try:
            got = call(x, f1, f2, ids_g, tw, fs1, fs2)
            torch.cuda.synchronize()
            v, rel = verdict(ref, got)
        except Exception as e:  # noqa: BLE001
            v, rel = f"ERROR {type(e).__name__}", float("nan")
            got = None
        print(f"    Q1 full/global (spare slots)        {v:<8s} rel={rel:.3e}")

        # Q2: the same weights re-wrapped over their own storage, as the VMM
        # pool would present them -- with and without the attribute restored.
        r1, r2 = rewrap(f1), rewrap(f2)
        print(f"    is_shuffled after rewrap:  {getattr(r1, 'is_shuffled', False)}")
        for label, mark in (("Q2a rewrapped, attr LOST", False),
                            ("Q2b rewrapped, attr set", True)):
            a, b = rewrap(f1), rewrap(f2)
            if mark:
                a.is_shuffled = True
                b.is_shuffled = True
            try:
                out = call(x, a, b, ids_g, tw, fs1, fs2)
                torch.cuda.synchronize()
                vv, rr = verdict(got if got is not None else ref, out)
            except Exception as e:  # noqa: BLE001
                vv, rr = f"ERROR {type(e).__name__}", float("nan")
            print(f"    {label:<35s} {vv:<8s} rel={rr:.3e}  (vs Q1)")

    print(
        "\nlegend: OK matches reference; ZERO all-zero; WRONG ran but wrong numbers.\n"
        "Q2 compares against Q1, i.e. the same weights reached the kernel by a "
        "different route."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
