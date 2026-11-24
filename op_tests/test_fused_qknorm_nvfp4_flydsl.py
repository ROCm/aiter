# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL NVFP4 insert of ``fused_qknorm_idxrqknorm`` (gfx950, MiniMax-M3 TP4 head set) against the HIP kernel.

Both write NVFP4 K / V / index-K pages, ``q_out`` and ``index_q_out`` from the same ``qkv``; the FlyDSL kernel must
leave every byte of every buffer equal to the HIP kernel's, pages that already hold data included (a V octet shares
nibbles and scales between tokens, a partially filled page keeps its other tokens). ``--perf`` also prints the
device time of both.

    python op_tests/test_fused_qknorm_nvfp4_flydsl.py [--perf] [-n N ...]
"""

import argparse
import sys

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import fused_qknorm_idxrqknorm as op
from aiter.ops.flydsl.kernels import fused_qknorm_idxr_nvfp4_insert as fl

NQ, NKV, NIQ, ROT, D = 16, 1, 1, 64, 128
ROW = (NQ + 2 * NKV + NIQ + 1) * D
PAGE = fl.PAGE_B
EPS = 1e-6


def make(n, pages, slots_kind, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    qkv = (torch.randn(n, ROW, generator=g, device=dev) * 2).bfloat16()
    if seed % 3 == 1:  # heavy rows
        qkv = (
            qkv.float() * torch.exp(torch.randn(n, 1, generator=g, device=dev) * 2)
        ).bfloat16()
    w = [(torch.randn(D, generator=g, device=dev) * 0.2).bfloat16() for _ in range(4)]
    ang = torch.rand(max(2 * n, 4096), ROT // 2, generator=g, device=dev) * 6.28
    cs = torch.cat([ang.cos(), ang.sin()], -1).bfloat16().contiguous()
    pos = torch.randint(0, cs.shape[0], (n,), generator=g, device=dev)
    nb = (n + 127) // 128 + 3
    if slots_kind == "contig":  # one request's chunk starting mid page
        start = int(torch.randint(0, 128, (1,), generator=g, device=dev))
        slot = torch.arange(start, start + n, device=dev)
        nb = (start + n + 127) // 128 + 1
    elif slots_kind == "multi":
        # several requests, each a run of consecutive slots from a random offset, pages apart
        k = min(5, n - 1)
        cuts = (
            sorted(
                set(torch.randint(1, max(n, 2), (k,), generator=g, device=dev).tolist())
            )
            if n > 1
            else []
        )
        lens = [b - a for a, b in zip([0] + cuts, cuts + [n])]
        slot, base = [], 0
        for ln in lens:
            off = int(torch.randint(0, 128, (1,), generator=g, device=dev))
            slot += list(range(base * 128 + off, base * 128 + off + ln))
            base += (off + ln + 127) // 128
        nb = base + 1
        slot = torch.tensor(slot, device=dev)
    elif slots_kind == "pad":  # a run, then padding tokens (-1)
        start = int(torch.randint(0, 128, (1,), generator=g, device=dev))
        slot = torch.arange(start, start + n, device=dev)
        nb = (start + n + 127) // 128 + 1
        slot[n - n // 4 :] = -1
    else:  # "sparse": every token alone in its V octet (8 tokens), pages hold other data: the octet's rest must stay
        nb = (n + 15) // 16 + 1
        oct_ids = torch.randperm(nb * 16, generator=g, device=dev)[:n]
        slot = oct_ids * 8 + torch.randint(0, 8, (n,), generator=g, device=dev)
    pages_kv = torch.randint(
        0, 255, (nb, 2 * NKV * PAGE), generator=g, dtype=torch.uint8, device=dev
    )
    pages_idx = torch.randint(
        0, 255, (nb, PAGE), generator=g, dtype=torch.uint8, device=dev
    )
    ks = torch.tensor([0.05 + 0.01 * (seed % 5)], device=dev)
    vs = torch.tensor([0.03 + 0.01 * (seed % 7)], device=dev)
    return {
        "qkv": qkv,
        "w": w,
        "cs": cs,
        "pos": pos,
        "slot": slot.to(torch.int64),
        "kv": pages_kv,
        "idx": pages_idx,
        "ks": ks,
        "vs": vs,
    }


def run(c, fn, n):
    kv, idx = c["kv"].clone(), c["idx"].clone()
    q = torch.zeros(n, NQ * D, dtype=torch.bfloat16, device="cuda")
    iq = torch.zeros(n, NIQ * D, dtype=torch.float8_e4m3fn, device="cuda")
    fn(c, kv, idx, q, iq)
    torch.cuda.synchronize()
    return kv, idx, q, iq.view(torch.uint8)


def hip(c, kv, idx, q, iq):
    op._fused_qknorm_idxrqknorm_hip(
        c["qkv"], c["w"][0], c["w"][1], c["cs"], c["pos"], NQ, NKV, ROT, EPS, c["w"][2], c["w"][3], NIQ,
        c["slot"], kv, kv, idx, 128, q, iq, c["slot"], "nvfp4", "nvfp4", c["ks"], c["vs"], False, False,
    )  # fmt: skip


def flydsl(c, kv, idx, q, iq):
    fl.fused_qknorm_idxr_nvfp4_insert(
        c["qkv"], c["w"][0], c["w"][1], c["cs"], c["pos"], EPS, c["w"][2], c["w"][3], c["slot"], kv, idx, q, iq,
        c["slot"], c["ks"], c["vs"],
    )  # fmt: skip


def public(c, kv, idx, q, iq):
    op.fused_qknorm_idxrqknorm(
        c["qkv"], c["w"][0], c["w"][1], c["cs"], c["pos"], NQ, NKV, ROT, EPS, c["w"][2], c["w"][3], NIQ,
        c["slot"], kv, kv, idx, 128, q, iq, c["slot"], kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4",
        k_scale=c["ks"], v_scale=c["vs"], asm_layout=False, skip_index_branch=False,
    )  # fmt: skip


def timeit(fn, c, n, copies=6):
    cs = [make(n, 0, "contig", 100 + i) for i in range(copies)]
    kvs = [(d["kv"].clone(), d["idx"].clone()) for d in cs]
    q = torch.zeros(n, NQ * D, dtype=torch.bfloat16, device="cuda")
    iq = torch.zeros(n, NIQ * D, dtype=torch.float8_e4m3fn, device="cuda")
    calls = [lambda d=d, k=k: fn(d, k[0], k[1], q, iq) for d, k in zip(cs, kvs)]
    for f in calls:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    ts = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / copies)
    return sorted(ts)[len(ts) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n",
        type=int,
        nargs="*",
        default=[
            1,
            7,
            8,
            64,
            300,
            488,
            489,
            512,
            1024,
            1536,
            1537,
            2048,
            5248,
            8192,
            16384,
        ],
    )
    ap.add_argument("--perf", action="store_true")
    ap.add_argument(
        "--large",
        action="store_true",
        help="also write the last pages of a 5.5 GB K/V + 2.8 GB index cache (byte offsets past 2^32)",
    )
    a = ap.parse_args(argv)
    if get_gfx_runtime() != "gfx950":
        print("SKIP: gfx950 only")
        return 0
    bad = 0
    for n in a.n:
        for kind in ("contig", "multi", "pad", "sparse"):
            for seed in (0, 1, 2):
                c = make(n, 0, kind, seed)
                ref, got = run(c, hip, n), run(c, flydsl, n)
                ok = all(torch.equal(r, o) for r, o in zip(ref, got))
                if not ok:
                    bad += 1
                    names = ("kv pages", "index pages", "q_out", "index_q_out")
                    print(
                        f"FAIL n={n} {kind} seed={seed}:",
                        [
                            nm
                            for nm, r, o in zip(names, ref, got)
                            if not torch.equal(r, o)
                        ],
                    )
        # the public op takes the FlyDSL path for this call and writes the same bytes
        c = make(n, 0, "multi", 5)
        if not all(
            torch.equal(r, o) for r, o in zip(run(c, hip, n), run(c, public, n))
        ):
            bad += 1
            print(f"FAIL public n={n}")
    if a.large:
        n, nb = 2048, 300_000
        c = make(n, 0, "contig", 11)
        start = (nb - 40) * 128 + 5
        c["slot"] = torch.arange(start, start + n, device="cuda")
        c["kv"] = torch.randint(
            0, 255, (nb, 2 * NKV * PAGE), dtype=torch.uint8, device="cuda"
        )
        c["idx"] = torch.randint(0, 255, (nb, PAGE), dtype=torch.uint8, device="cuda")
        ref, got = run(c, hip, n), run(c, flydsl, n)
        if not all(torch.equal(r, o) for r, o in zip(ref, got)):
            bad += 1
            print("FAIL large cache (page offsets past 2^32 bytes)")
        del c, ref, got
    # calls the kernel does not take stay on the HIP path and still work
    c = make(64, 0, "multi", 9)
    c32 = {**c, "pos": c["pos"].to(torch.int32)}
    assert fl.nvfp4_insert_blocker(c32["qkv"], c["w"][0], c["w"][1], c["cs"], c32["pos"], NQ, NKV, ROT, EPS, c["w"][2], c["w"][3], NIQ, c["slot"], c["kv"], c["kv"], c["idx"], 128, torch.empty(64, NQ * D, dtype=torch.bfloat16, device="cuda"), torch.empty(64, D, dtype=torch.float8_e4m3fn, device="cuda"), c["slot"], c["ks"], c["vs"]) == "positions"  # fmt: skip
    print("PASS" if not bad else f"FAILED ({bad})")
    if a.perf:
        print("n, hip us, flydsl us")
        for n in a.n:
            print(n, round(timeit(hip, None, n), 2), round(timeit(flydsl, None, n), 2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
