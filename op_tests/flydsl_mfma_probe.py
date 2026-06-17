#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Empirically reverse-engineer the 32x32x64 fp8 MFMA fragment layouts.

Strategy: feed A/B fragments in *lane-major* raw order (lane L's 32 fp8
= bytes [L*32, L*32+32)) with KNOWN small integer values, run the MFMA,
dump the raw per-lane C fragment, then in Python search for the (lane,v)
-> (row,col) maps of A, B, C that make C == A @ B self-consistent.

We parametrize candidate maps from the cute TV layout strides and verify
against the hardware output.
"""

import os
import sys

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import range_constexpr

_HERE = os.path.dirname(os.path.abspath(__file__))
_AITER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

from aiter.ops.flydsl.rocdl_mfma_fp8 import Mfma32x32x64  # noqa: E402

FP8 = torch.float8_e4m3fn
M, N, K = 32, 32, 64
NL = 64


def build():
    @flyc.kernel
    def kernel(a_in: fx.Tensor, b_in: fx.Tensor, c_out: fx.Tensor):
        mfma = Mfma32x32x64()
        f8_t = fx.Float8E4M3FN.ir_type
        lane = fx.thread_idx.x % 64
        gA = fx.rocdl.make_buffer_tensor(a_in, max_size=False)
        gB = fx.rocdl.make_buffer_tensor(b_in, max_size=False)
        gC = fx.rocdl.make_buffer_tensor(c_out, max_size=False)

        def f8view(buf):
            it = fx.get_iter(buf)
            pty = fx.PointerType.get(
                elem_ty=f8_t,
                address_space=TargetAddressSpace.BufferDesc,
                alignment=fx.PointerType(it.type).alignment,
            )
            return fx.Tensor(fx.make_view(fx.recast_iter(pty, it), fx.get_layout(buf)))

        a_div = fx.logical_divide(f8view(gA), fx.make_layout(1, 1))
        b_div = fx.logical_divide(f8view(gB), fx.make_layout(1, 1))
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        cp = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
        st = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        def load(div, base):
            r0 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FN)
            r1 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FN)
            fx.copy_atom_call(cp, fx.slice(div, (None, base)), r0)
            fx.copy_atom_call(cp, fx.slice(div, (None, base + fx.Int32(16))), r1)
            v0 = fx.memref_load_vec(r0).bitcast(fx.Int32)
            v1 = fx.memref_load_vec(r1).bitcast(fx.Int32)
            return v0.shuffle(v1, list(range(8)))

        a = load(a_div, lane * fx.Int32(32))
        b = load(b_div, lane * fx.Int32(32))
        c = mfma.call(a, b, mfma.zero_value)
        cb = lane * fx.Int32(16)
        for ch in range_constexpr(4):
            rc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            sub = fx.Vector(c).shuffle(fx.Vector(c), [ch * 4 + i for i in range(4)])
            fx.memref_store_vec(sub, rc)
            fx.copy_atom_call(st, rc, fx.slice(c_div, (None, cb + fx.Int32(ch * 4))))

    @flyc.jit
    def launch(a_in: fx.Tensor, b_in: fx.Tensor, c_out: fx.Tensor, stream: fx.Stream):
        kernel(a_in, b_in, c_out).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


# ---- candidate coordinate maps (from cute TV layout, col-major idx) ----
def a_map(lane, v):  # col-major decode
    return lane % 32, (lane // 32) * 32 + v


def b_map(lane, v):
    return lane % 32 + (v % 2) * 32, (lane // 32) * 16 + v // 2


def c_map(lane, v):
    return (lane // 32) * 4 + (v % 4) + 8 * (v // 4), lane % 32


def main():
    torch.manual_seed(1)
    dev = "cuda"
    # Random small fp8 A,B as logical tiles.
    A = (torch.randn(M, K, device=dev) * 0.4).clamp(-3, 3).to(FP8)
    B = (torch.randn(K, N, device=dev) * 0.4).clamp(-3, 3).to(FP8)
    Ai = A.view(torch.int8).cpu().numpy()
    Bi = B.view(torch.int8).cpu().numpy()

    # Pack lane-major using candidate maps.
    af = np.empty((NL, 32), np.int8)
    bf = np.empty((NL, 32), np.int8)
    for L in range(NL):
        for v in range(32):
            r, c = a_map(L, v)
            af[L, v] = Ai[r, c]
            r, c = b_map(L, v)
            bf[L, v] = Bi[r, c]

    a_t = torch.from_numpy(af).to(dev).contiguous().view(-1)
    b_t = torch.from_numpy(bf).to(dev).contiguous().view(-1)
    c_t = torch.zeros(NL * 16, dtype=torch.float32, device=dev)
    launch = build()
    args = (a_t, b_t, c_t, torch.cuda.current_stream())
    flyc.compile(launch, *args)(*args)
    torch.cuda.synchronize()
    cf = c_t.cpu().numpy().reshape(NL, 16)

    Ar = A.to(torch.float32).cpu().numpy()
    Br = B.to(torch.float32).cpu().numpy()
    Cref = Ar @ Br  # (32,32)

    # Build kernel C tile from candidate c_map and compare.
    Ck = np.full((M, N), np.nan, np.float32)
    for L in range(NL):
        for v in range(16):
            r, c = c_map(L, v)
            Ck[r, c] = cf[L, v]
    err = np.nanmax(np.abs(Ck - Cref))
    cos = (Ck.reshape(-1) @ Cref.reshape(-1)) / (
        np.linalg.norm(Ck) * np.linalg.norm(Cref) + 1e-12
    )
    print(f"candidate maps: cos={cos:.4f} maxerr={err:.4f}")

    # If wrong, brute-force the C map: for each (L,v) find (r,c) where
    # cf[L,v] best matches Cref. This tells us the true C layout given
    # A,B maps are correct.  We test it by checking each lane-value's
    # value against all 1024 Cref cells.
    if cos < 0.99:
        print("\nBrute-forcing C map (assuming A,B maps correct)...")
        # Use distinct Cref values: regenerate with full-rank random so
        # collisions are unlikely.
        matches = {}
        flat = Cref.reshape(-1)
        for L in range(NL):
            for v in range(16):
                val = cf[L, v]
                j = int(np.argmin(np.abs(flat - val)))
                matches[(L, v)] = (j // N, j % N, abs(flat[j] - val))
        # Print map for lanes 0,1,31,32,33,63
        for L in [0, 1, 2, 31, 32, 33, 63]:
            row = " ".join(
                f"v{v}->(r{matches[(L, v)][0]},c{matches[(L, v)][1]})"
                for v in range(16)
            )
            print(f"lane {L:2d}: {row}")
        worst = max(m[2] for m in matches.values())
        print(f"worst match residual = {worst:.4f}")


if __name__ == "__main__":
    main()
