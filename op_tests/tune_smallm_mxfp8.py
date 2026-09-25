# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Autotune the decode small-M MX-FP8 dense GEMV: for each (K,N,M) sweep the
MFMA (n_sub, k_splits) config, time HIP vs the Triton dot_scaled fallback, and
emit a CSV (best config + whether HIP wins). The wrapper loads this CSV to
refine its envelope-based dispatch (falling back to the hand-tuned _MFMA_CFG).

  python op_tests/tune_smallm_mxfp8.py            # writes aiter/configs/smallm_mxfp8_tuned.csv
"""

import csv
import os

import torch
import triton
import triton.language as tl

import aiter.ops.smallm_gemm_mxfp8 as M

# raw kernels (not mxfp8_gemv -- that reads the CSV we are generating)
from aiter.ops.smallm_gemm_mxfp8 import _run_gemv, smallm_mxfp8_mfma

DEVICE = "cuda"
FP8 = 448.0
SHAPES = [
    # TP4 per-GPU shapes
    (6144, 2304),  # qkv
    (2048, 6144),  # o_proj
    (6144, 1536),  # gate_up
    (1536, 6144),  # down
    # TP8 per-GPU shapes (col-parallel halves N, row-parallel halves K)
    (6144, 1152),  # qkv        @TP8
    (6144, 768),  # gate_up    @TP8
    (1024, 6144),  # o_proj     @TP8
    (768, 6144),  # down       @TP8
    # TP1 per-GPU shapes (full, unsharded; gate_up == down == 6144x6144)
    (6144, 9216),  # qkv          @TP1
    (6144, 6144),  # gate_up/down @TP1
    (8192, 6144),  # o_proj       @TP1
    # TP2 per-GPU shapes
    (6144, 4608),  # qkv        @TP2
    (6144, 3072),  # gate_up    @TP2
    (4096, 6144),  # o_proj     @TP2
    (3072, 6144),  # down       @TP2
]
MS = [1, 2, 4, 8, 16, 32, 64]
MFMA_M = {8, 16, 32, 64}
N_SUBS = (1, 2, 4)
K_SPLITS = (1, 2, 4, 8)


def quant(x):
    K = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], K // 32, 32)
    a = xb.abs().amax(-1, keepdim=True).clamp(min=1e-20)
    e = torch.ceil(torch.log2(a / FP8)).clamp(-127, 127)
    q = (xb / torch.exp2(e)).clamp(-FP8, FP8).to(torch.float8_e4m3fn)
    s = (e + 127).to(torch.uint8)
    return (
        q.reshape(*x.shape[:-1], K).contiguous(),
        s.squeeze(-1).reshape(*x.shape[:-1], K // 32).contiguous(),
    )


def dq(q, s):
    K = q.shape[-1]
    qb = q.float().reshape(*q.shape[:-1], K // 32, 32)
    return (qb * torch.exp2(s.reshape(*s.shape, 1).float() - 127)).reshape(
        *q.shape[:-1], K
    )


@triton.jit
def _tl(
    xp,
    xsp,
    wp,
    wsp,
    op,
    Mn,
    N,
    K,
    sxm,
    sxk,
    sxsm,
    sxsk,
    swn,
    swk,
    swsn,
    swsk,
    som,
    son,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    osk = tl.arange(0, BK // 32)
    mm = om < Mn
    nm = on < N
    xpt = xp + om[:, None] * sxm + ok[None, :] * sxk
    xspt = xsp + om[:, None] * sxsm + osk[None, :] * sxsk
    wpt = wp + on[:, None] * swn + ok[None, :] * swk
    wspt = wsp + on[:, None] * swsn + osk[None, :] * swsk
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        x = tl.load(xpt, mask=mm[:, None], other=0.0)
        w = tl.load(wpt, mask=nm[:, None], other=0.0)
        xs = tl.load(xspt, mask=mm[:, None], other=0)
        ws = tl.load(wspt, mask=nm[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        xpt += BK * sxk
        wpt += BK * swk
        xspt += (BK // 32) * sxsk
        wspt += (BK // 32) * swsk
    opt = op + om[:, None] * som + on[None, :] * son
    tl.store(opt, acc.to(op.dtype.element_ty), mask=mm[:, None] & nm[None, :])


def triton_dense(xq, xs, wq, ws):
    """Vanilla Triton dot_scaled GEMM -- the unmodified upstream fallback
    (stock BLOCK 64/128/128, num_warps=8), the baseline the HIP kernel
    replaces."""
    Mn, K = xq.shape
    N = wq.shape[0]
    out = torch.empty((Mn, N), dtype=torch.bfloat16, device=xq.device)
    BM, BN, BK = 64, 128, 128
    g = (triton.cdiv(Mn, BM), triton.cdiv(N, BN))
    _tl[g](
        xq,
        xs,
        wq,
        ws,
        out,
        Mn,
        N,
        K,
        xq.stride(0),
        xq.stride(1),
        xs.stride(0),
        xs.stride(1),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        out.stride(0),
        out.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
        num_warps=8,
    )
    return out


# Engage HIP only when it beats the vanilla Triton fallback by this margin -- at a
# near-tie, prefer Triton (simpler path, no run-to-run flapping).
USE_HIP_MARGIN = 0.03


def bench(fn, it=300, wu=40):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1e3


def u8(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    rows, mistuned = [], 0
    print(f"# autotune  ({torch.cuda.get_device_properties(0).gcnArchName})")
    print(
        f"{'K':>5} {'N':>5} {'M':>4} {'kern':>5} {'best cfg':>10} {'HIP us':>7} "
        f"{'Tri us':>7} {'best':>6} {'shipped':>8} {'note':>10}"
    )
    for K, N in SHAPES:
        w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 0.1
        wq, ws = quant(w)
        kiters = K // 128
        for m in MS:
            x = torch.randn(m, K, device=DEVICE, dtype=torch.bfloat16) * 0.5
            xq, xs = quant(x)
            ref = torch.nn.functional.linear(dq(xq, xs), dq(wq, ws))
            tri = bench(lambda: triton_dense(xq, xs, wq, ws))
            if m in MFMA_M:
                best = None  # (us, ns, ks)
                for ns in N_SUBS:
                    for ks in K_SPLITS:
                        if kiters % ks:
                            continue
                        try:
                            o = smallm_mxfp8_mfma(
                                u8(xq), xs, u8(wq), ws, torch.bfloat16, ns, ks
                            )
                            if relerr(o, ref) > 5e-2:
                                continue
                            t = bench(
                                lambda: smallm_mxfp8_mfma(
                                    u8(xq), xs, u8(wq), ws, torch.bfloat16, ns, ks
                                )
                            )
                            if best is None or t < best[0]:
                                best = (t, ns, ks)
                        except Exception:
                            pass
                if best is None:
                    continue
                hip, ns, ks = best
                uh = tri / hip > 1 + USE_HIP_MARGIN
                ship = M._MFMA_CFG.get((K, N), {}).get(m)  # (k_splits, n_sub)
                if not uh:
                    note = "->Triton"
                elif ship and (ks, ns) != ship:
                    note = "MISTUNED"
                    mistuned += 1
                else:
                    note = ""
                rows.append([K, N, m, "mfma", ns, ks, int(uh)])
                print(
                    f"{K:>5} {N:>5} {m:>4} {'mfma':>5} {f'ns{ns},ks{ks}':>10} {hip:>7.2f} "
                    f"{tri:>7.2f} {tri/hip:>5.2f}x {str(ship):>8} {note:>10}"
                )
            else:
                try:
                    o = _run_gemv(xq, xs, wq, ws, torch.bfloat16, m, K)
                except Exception:
                    o = None
                if o is None:
                    continue
                hip = bench(lambda: _run_gemv(xq, xs, wq, ws, torch.bfloat16, m, K))
                uh = tri / hip > 1 + USE_HIP_MARGIN
                rows.append([K, N, m, "gemv", 0, 0, int(uh)])
                print(
                    f"{K:>5} {N:>5} {m:>4} {'gemv':>5} {'-':>10} {hip:>7.2f} "
                    f"{tri:>7.2f} {tri/hip:>5.2f}x {'-':>8} {('' if uh else '->Triton'):>10}"
                )

    out_dir = os.path.join(os.path.dirname(M.__file__), "..", "configs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(out_dir, "smallm_mxfp8_tuned.csv"))
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["K", "N", "M", "kernel", "n_sub", "k_splits", "use_hip"])
        wr.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {path}")
    print(f"hand table MISTUNED cells (best != shipped and HIP wins): {mistuned}")


def _skip_in_ci() -> bool:
    # CI runs every op_tests/*.py via `python3 <file>` (60-min timeout). This is
    # a developer autotuning sweep (HIP-vs-Triton over many configs) that writes
    # a CSV, not a correctness test, so make it a no-op in CI (AITER_TEST set)
    # and on non-gfx95x devices.
    if os.environ.get("AITER_TEST"):
        return True
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950"
    except Exception:
        return True


if __name__ == "__main__":
    if _skip_in_ci():
        print("skip tune_smallm_mxfp8: dev autotune sweep (gfx95x only, not under CI)")
    else:
        main()
