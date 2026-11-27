# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The NVFP4 prefill insert of fused_qknorm_idxrqknorm (MiniMax-M3 TP4 head set, gfx950) against the generic kernel.

The prefill path takes a call of at least AITER_FUSED_QKNORM_NVFP4_PREFILL_MIN_TOKENS tokens (default 2,
kPfDefaultMinTokens); any slot
pattern with distinct slots must give the bytes of the generic kernel in every buffer (pages that already hold data
included: a V octet shares nibbles and scales between tokens). The parent runs this file twice, once with the
prefill path forced on from 2 tokens and once with it off, and compares the SHA-256 of every output buffer.

    python op_tests/test_fused_qknorm_nvfp4_prefill.py [-n N ...] [--kinds ...] [--seeds S]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import torch

NQ, NKV, NIQ, ROT, D = 16, 1, 1, 64, 128
ROW = (NQ + 2 * NKV + NIQ + 1) * D
PAGE = 9216
EPS = 1e-6
dev = "cuda"


class NoSlots(Exception):
    """make() cannot build this slot pattern for this n (the only reason a case is skipped)."""


def distinct_from(g, total, k):
    return torch.randperm(total, generator=g, device="cpu")[:k]


def make_slots(kind, n, g):
    """returns (slot int64 [n] cpu, islot, nb)"""
    gi = lambda hi, k=1: torch.randint(0, hi, (k,), generator=g).tolist()
    if kind == "contig":
        st = gi(128)[0]
        slot = torch.arange(st, st + n)
        nb = (st + n + 127) // 128 + 1
    elif kind == "aligned":
        slot = torch.arange(0, n)
        nb = (n + 127) // 128 + 1
    elif kind == "multi":
        k = min(5, n - 1)
        cuts = (
            sorted(set(torch.randint(1, max(n, 2), (k,), generator=g).tolist()))
            if n > 1
            else []
        )
        lens = [b - a for a, b in zip([0] + cuts, cuts + [n])]
        s, base = [], 0
        for ln in lens:
            off = gi(128)[0]
            s += list(range(base * 128 + off, base * 128 + off + ln))
            base += (off + ln + 127) // 128
        slot = torch.tensor(s)
        nb = base + 1
    elif kind == "pad":
        st = gi(128)[0]
        slot = torch.arange(st, st + n)
        nb = (st + n + 127) // 128 + 1
        slot[n - n // 4 :] = -1
    elif kind == "padmid":
        st = gi(128)[0]
        slot = torch.arange(st, st + n)
        nb = (st + n + 127) // 128 + 1
        m = torch.rand(n, generator=g) < 0.1
        slot[m] = -1
    elif kind == "sparse":
        nb = (n + 15) // 16 + 1
        oi = distinct_from(g, nb * 16, n)
        slot = oi * 8 + torch.randint(0, 8, (n,), generator=g)
    elif kind == "scatter":
        nb = (n + 63) // 64 + 2
        slot = distinct_from(g, nb * 128, n)
    elif (
        kind == "mtp"
    ):  # requests of 1..4 tokens, each in its own 16-slot stretch at a random offset
        lens, tot = [], 0
        while tot < n:
            k = int(torch.randint(1, 5, (1,), generator=g))
            k = min(k, n - tot)
            lens.append(k)
            tot += k
        order = torch.randperm(len(lens), generator=g).tolist()
        s = []
        for r, k in zip(order, lens):
            off = int(torch.randint(0, 16 - k + 1, (1,), generator=g))
            s += list(range(16 * r + off, 16 * r + off + k))
        slot = torch.tensor(s)
        nb = (16 * len(lens)) // 128 + 2
    elif (
        kind == "shared"
    ):  # octets split between runs that sit far apart in token order
        no = (n + 3) // 3 + 4
        runs = []
        for o in range(no):
            a = int(torch.randint(1, 8, (1,), generator=g))
            b = int(torch.randint(a, 9, (1,), generator=g))
            for lo, hi in ((0, a), (a, b), (b, 8)):
                if hi > lo:
                    runs.append(list(range(o * 8 + lo, o * 8 + hi)))
        perm = torch.randperm(len(runs), generator=g).tolist()
        s = []
        for i in perm:
            s += runs[i]
            if len(s) >= n:
                break
        slot = torch.tensor(s[:n])
        nb = no // 16 + 2
        if slot.numel() < n:
            raise NoSlots("shared: too few slots")
    elif kind == "rev":
        st = gi(128)[0]
        slot = torch.arange(st + n - 1, st - 1, -1)
        nb = (st + n + 127) // 128 + 1
    elif kind == "gap2":
        st = gi(128)[0]
        slot = st + 2 * torch.arange(n)
        nb = (st + 2 * n + 127) // 128 + 1
    elif (
        kind == "chunks"
    ):  # runs of random length 1..40 with random jumps, ascending slots
        s, cur = [], gi(60)[0]
        while len(s) < n:
            ln = int(torch.randint(1, 41, (1,), generator=g))
            s += list(range(cur, cur + ln))
            cur += ln + int(torch.randint(1, 30, (1,), generator=g))
        slot = torch.tensor(s[:n])
        nb = (cur + 127) // 128 + 1
    else:
        raise ValueError(kind)
    nb = max(nb, int(slot.max()) // 128 + 2)
    islot = slot.clone()
    if kind in ("multi", "chunks"):
        islot[torch.rand(n, generator=g) < 0.2] = -1
    return slot.to(torch.int64), islot.to(torch.int64), nb


KINDS = [
    "contig",
    "aligned",
    "multi",
    "pad",
    "padmid",
    "sparse",
    "scatter",
    "mtp",
    "shared",
    "rev",
    "gap2",
    "chunks",
]


def make(n, kind, seed, misaligned_w=False):
    g = torch.Generator().manual_seed(seed * 1000 + n)
    gd = torch.Generator(device=dev).manual_seed(seed * 1000 + n)
    qkv = (torch.randn(n, ROW, generator=gd, device=dev) * 2).bfloat16()
    if seed % 3 == 1:
        qkv = (
            qkv.float() * torch.exp(torch.randn(n, 1, generator=gd, device=dev) * 2)
        ).bfloat16()
    w = [(torch.randn(D, generator=gd, device=dev) * 0.2).bfloat16() for _ in range(4)]
    if misaligned_w:
        # A contiguous bf16 view sliced at storage offset 1 (2 B) keeps shape/dtype/contiguity
        # but is no longer 8 B aligned: the prefill wave kernel's uint2 loads of the norm weight
        # need that alignment (launchNvfp4Prefill's pfPtrAligned8 check), so this must fall back
        # to the generic kernel instead of reaching those loads.
        padded = torch.zeros(D + 1, dtype=w[0].dtype, device=dev)
        padded[1:] = w[0]
        w[0] = padded[1:]
        assert (
            w[0].data_ptr() % 8 != 0
        ), "test setup: weight view is unexpectedly aligned"
    ang = torch.rand(max(2 * n, 4096), ROT // 2, generator=gd, device=dev) * 6.28
    cs = torch.cat([ang.cos(), ang.sin()], -1).bfloat16().contiguous()
    pos = torch.randint(0, cs.shape[0], (n,), generator=gd, device=dev)
    slot, islot, nb = make_slots(kind, n, g)
    kv = torch.randint(
        0, 255, (nb, 2 * NKV * PAGE), generator=gd, dtype=torch.uint8, device=dev
    )
    idx = torch.randint(0, 255, (nb, PAGE), generator=gd, dtype=torch.uint8, device=dev)
    ks = torch.tensor([0.05 + 0.01 * (seed % 5)], device=dev)
    vs = torch.tensor([0.03 + 0.01 * (seed % 7)], device=dev)
    return {
        "qkv": qkv,
        "w": w,
        "cs": cs,
        "pos": pos,
        "slot": slot.to(dev),
        "islot": islot.to(dev),
        "kv": kv,
        "idx": idx,
        "ks": ks,
        "vs": vs,
    }


def run(c, n, marshalled=False):
    """marshalled=True calls the generic aiter_tensor_t entry directly, bypassing the raw-pointer front."""
    from aiter.ops import fused_qknorm_idxrqknorm as op

    kv, idx = c["kv"].clone(), c["idx"].clone()
    q = torch.zeros(n, NQ * D, dtype=torch.bfloat16, device=dev)
    iq = torch.zeros(n, NIQ * D, dtype=torch.float8_e4m3fn, device=dev)
    fn = op._fused_qknorm_idxrqknorm_hip if marshalled else op.fused_qknorm_idxrqknorm
    fn(
        c["qkv"], c["w"][0], c["w"][1], c["cs"], c["pos"], NQ, NKV, ROT, EPS, c["w"][2], c["w"][3], NIQ,
        c["slot"], kv, kv, idx, 128, q, iq, c["islot"], kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4",
        k_scale=c["ks"], v_scale=c["vs"], asm_layout=False, skip_index_branch=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    return {"kv": kv, "idx": idx, "q": q, "iq": iq.view(torch.uint8)}


def dig(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def prefill_launched(n):
    """Whether one raw-entry call at n tokens launches the prefill kernel (kernel-name trace)."""
    from torch.profiler import ProfilerActivity, profile

    c = make(n, "contig", 0)
    run(c, n)  # build and warm up outside the trace
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run(c, n)
    return any(
        "fusedNvfp4InsertPrefillKernel" in e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )


# the dispatch receipt is taken at one fixed, supported size, whatever correctness sizes -n names
# (1 and above 419430 tokens legitimately select the generic kernel)
DISPATCH_PROBE_N = 2

# the prefill kernel reads qkv through a 32-bit byte extent: 2**31 // (ROW * 2) = 419430 tokens is
# its largest call, the generic kernel takes the one above it
BIG_N = (419430, 419431)

# Regression (junhaha666, #8472): int64 slots whose magnitude is >= 2**32 have a nonzero low 32-bit
# word (e.g. -4294967295 = 0xFFFFFFFF00000001 -> low word 1), so reading the slot as a 32-bit int
# treats the padding slot as a valid (positive) slot and writes the KV / index cache. The generic
# kernel keys off the full int64 (slot_mapping[token_idx] < 0) and skips it; the prefill insert must
# skip it too. -1 (low word also -1) is included to keep the common padding value covered.
BIGNEG_SLOTS = [-1, -4294967295, -4294967294, -(2**40), -(2**32) - 123]


def bigneg_make(n, mode):
    """A valid baseline case (real qkv / positions / weights, aligned slots) whose slot_mapping and
    index_slot_mapping carry large-magnitude negative int64 slots. mode 'all': every token is
    padding (both kernels write nothing); 'mix': even tokens padding, odd tokens keep their valid
    slot; 'junhaha': exactly the two-token pattern [-4294967295, -4294967294] from the report.
    """
    c = make(n, "aligned", 0)
    slot = c["slot"].clone()
    neg = torch.tensor(BIGNEG_SLOTS, dtype=torch.int64, device=slot.device)
    if mode == "junhaha":
        slot = torch.tensor(
            [-4294967295, -4294967294], dtype=torch.int64, device=slot.device
        )
    else:
        idx = (
            torch.arange(n, device=slot.device)
            if mode == "all"
            else torch.arange(0, n, 2, device=slot.device)
        )
        slot[idx] = neg[torch.arange(idx.numel(), device=slot.device) % neg.numel()]
    c["slot"], c["islot"] = slot, slot.clone()
    return c


def bigneg_child(path):
    """Runs the raw-pointer entry on the large-magnitude-negative slot patterns and records, per
    case, the input KV / index digests and the digests of every output buffer."""
    res = {}
    specs = [(f"{n}/{m}", n, m) for n in (2, 3, 8, 17, 64) for m in ("all", "mix")]
    specs.append(("2/junhaha", 2, "junhaha"))
    for name, n, mode in specs:
        c = bigneg_make(n, mode)
        res[name] = {
            "in": {"kv": dig(c["kv"]), "idx": dig(c["idx"])},
            "out": {k: dig(v) for k, v in run(c, n).items()},
        }
        del c
    res["_prefill_launched"] = prefill_launched(DISPATCH_PROBE_N)
    with open(path, "w") as f:
        json.dump(res, f)


def test_bigneg_slots_skipped():
    """The prefill insert must skip large-magnitude negative int64 slots exactly as the generic
    kernel does (junhaha666, #8472). Forces the prefill path on in one child and off in another and
    asserts every output buffer is byte-identical between the two, and that the all-padding patterns
    leave the KV and index caches byte-for-byte unchanged (nothing written) in both kernels.
    """
    import pytest

    pytest.importorskip("aiter")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("the NVFP4 prefill insert is gfx950 only")

    out = {}
    for tag, floor in (("generic", "999999999"), ("prefill", "2")):
        path = os.path.join(tempfile.mkdtemp(), tag + ".json")
        env = dict(os.environ, AITER_FUSED_QKNORM_NVFP4_PREFILL_MIN_TOKENS=floor)
        subprocess.run(
            [sys.executable, __file__, "--bigneg-child", path], env=env, check=True
        )
        with open(path) as f:
            out[tag] = json.load(f)
    # dispatch sanity: only the forced child may launch the prefill kernel, or the comparison below
    # proves nothing about the prefill path
    assert out["prefill"].pop("_prefill_launched") is True
    assert out["generic"].pop("_prefill_launched") is False
    # the prefill insert leaves every buffer byte-identical to the generic kernel
    for name, g in out["generic"].items():
        for buf, h in g["out"].items():
            assert out["prefill"][name]["out"][buf] == h, (
                f"{name}/{buf}: prefill insert differs from the generic kernel on "
                "large-magnitude negative slots"
            )
    # every slot of the 'all' and 'junhaha' cases is padding: both kernels write nothing, so the KV
    # and index caches come back equal to the input
    for tag, cases in out.items():
        for name, case in cases.items():
            if name.endswith(("/all", "/junhaha")):
                assert (
                    case["out"]["kv"] == case["in"]["kv"]
                ), f"[{tag}] {name}: KV changed"
                assert (
                    case["out"]["idx"] == case["in"]["idx"]
                ), f"[{tag}] {name}: index cache changed"


def child(args):
    res = {}
    cases = [(n, k, s) for n in args.n for k in args.kinds for s in range(args.seeds)]
    cases += [(n, "aligned", 0) for n in BIG_N]
    for n, kind, seed in cases:
        try:
            c = make(n, kind, seed)
        except NoSlots:  # the pattern needs more slots than it can make for this n
            continue
        res[f"{n}/{kind}/{seed}"] = {k: dig(v) for k, v in run(c, n).items()}
        res[f"{n}/{kind}/{seed}/marshalled"] = {
            k: dig(v) for k, v in run(c, n, marshalled=True).items()
        }
        del c
    # q_norm_weight sliced at an odd storage offset (misaligned_w=True): must give the generic
    # kernel's bytes under both floors (below and at/above the default pfMinTokens()=512), proving
    # the prefill wave declines instead of reaching its uint2 loads misaligned.
    for n in (DISPATCH_PROBE_N, 511, 512, 2048):
        try:
            c = make(n, "aligned", 0, misaligned_w=True)
        except NoSlots:
            continue
        res[f"{n}/misaligned_w"] = {k: dig(v) for k, v in run(c, n).items()}
        res[f"{n}/misaligned_w/marshalled"] = {
            k: dig(v) for k, v in run(c, n, marshalled=True).items()
        }
        del c
    res["_prefill_launched"] = prefill_launched(DISPATCH_PROBE_N)
    with open(args.child, "w") as f:
        json.dump(res, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--child", default=None, help="(internal) write the digests to this file"
    )
    ap.add_argument(
        "--bigneg-child",
        default=None,
        help="(internal) write the large-magnitude-negative-slot digests to this file",
    )
    ap.add_argument(
        "-n",
        type=int,
        nargs="*",
        default=[
            1,
            2,
            4,
            7,
            8,
            9,
            16,
            64,
            256,
            300,
            489,
            512,
            1023,
            1025,
            1537,
            2048,
            6143,
            6145,
            8192,
            24577,
            32768,
        ],
    )
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    if args.bigneg_child:
        bigneg_child(args.bigneg_child)
        return
    if args.child:
        child(args)
        return
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        print("the NVFP4 prefill insert is gfx950 only: skipping")
        return
    out = {}
    for tag, floor in (("generic", "999999999"), ("prefill", "2")):
        path = os.path.join(tempfile.mkdtemp(), tag + ".json")
        env = dict(os.environ, AITER_FUSED_QKNORM_NVFP4_PREFILL_MIN_TOKENS=floor)
        cmd = [
            sys.executable,
            __file__,
            "--child",
            path,
            "-n",
            *map(str, args.n),
            "--kinds",
            *args.kinds,
            "--seeds",
            str(args.seeds),
        ]
        subprocess.run(cmd, env=env, check=True)
        with open(path) as f:
            out[tag] = json.load(f)
        # the dispatch itself: a forced-prefill child must launch the prefill kernel and the
        # generic child must not, or the digest comparison below proves nothing
        launched = out[tag].pop("_prefill_launched")
        assert launched == (tag == "prefill"), (
            f"[{tag}] prefill kernel launched={launched}: the dispatch does not follow "
            "AITER_FUSED_QKNORM_NVFP4_PREFILL_MIN_TOKENS"
        )
        assert out[tag], f"[{tag}] no case ran"
    # the raw-pointer entry (every token count) against the generic marshalled entry, per floor
    for tag, digests in out.items():
        rawbad = [
            k
            for k in digests
            if not k.endswith("/marshalled")
            and digests[k] != digests[k + "/marshalled"]
        ]
        print(f"[{tag}] raw entry vs marshalled entry: {len(rawbad)} differ")
        assert (
            not rawbad
        ), f"the raw entry differs from the marshalled entry: {rawbad[:5]}"
    assert (
        out["generic"].keys() == out["prefill"].keys()
    ), "the two children ran different cases"
    out = {
        tag: {k: v for k, v in digests.items() if not k.endswith("/marshalled")}
        for tag, digests in out.items()
    }
    bad = [k for k in out["generic"] if out["generic"][k] != out["prefill"][k]]
    for k in bad:
        print(
            "DIFF",
            k,
            {
                b: out["generic"][k][b] == out["prefill"][k][b]
                for b in out["generic"][k]
            },
        )
    print(f"compared {len(out['generic'])} cases, {len(bad)} differ")
    assert not bad, "the prefill insert differs from the generic kernel"


if __name__ == "__main__":
    main()
