"""Does a hand-off semaphore need a store drain on gfx950? Producers write data then
bump a counter (no s_waitcnt vmcnt before it); consumers spin on the counter, then
read every word and count stale ones (a stale word holds the previous launch's value).

  python3 tests/probes/sem_probe.py [--words W] [--producers P] [--consumers C] [--iters N]

MODE 0  per-CTA: barrier, one thread atomic-adds 1 (no drain)
MODE 1  per-wave: every wave's lane 0 atomic-adds 1 right after its own stores (no drain)
MODE 2  per-CTA: barrier, one thread plain-stores a flag (no drain), consumer polls P flags
MODE 3  per-CTA with drain: s_waitcnt vmcnt(0) + barrier, then atomic-add (the safe form)
MODE 4  tagged pairs (value | tag), consumer polls every pair (the current scheme)
"""
import argparse

import torch
import triton
import triton.language as tl

NT = tl.constexpr(512)


@triton.jit
def _rt(x):
    return tl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v,~{memory}", [x], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _atomic_add(addr, pred):
    # global_atomic_add without return: a pure device-scope (sc1) atomic, no fence, no wait
    tl.inline_asm_elementwise(
        "v_cmp_ne_u32 vcc, 0, $2\ns_and_saveexec_b64 $0, vcc\ns_cbranch_execz 1f\n"
        "global_atomic_add $1, $3, off sc1\n1:\ns_or_b64 exec, exec, $0",
        "=&s,v,v,v,~{vcc},~{memory}", [addr.to(tl.int64, bitcast=True), pred.to(tl.int32), (pred * 0 + 1).to(tl.int32)],
        dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _store_flag(addr, val, pred):
    tl.inline_asm_elementwise(
        "v_cmp_ne_u32 vcc, 0, $2\ns_and_saveexec_b64 $0, vcc\ns_cbranch_execz 1f\n"
        "global_store_dword $1, $3, off sc0 sc1\n1:\ns_or_b64 exec, exec, $0",
        "=&s,v,v,v,~{vcc},~{memory}", [addr.to(tl.int64, bitcast=True), pred.to(tl.int32), val.to(tl.int32)],
        dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _fence_asm(x, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(ASM, "=v,0,~{memory}", [x], dtype=x.dtype, is_pure=False, pack=1)


@triton.jit(do_not_specialize=["epoch"])
def probe(data, pairs, cnt, flags, err, stamps, epoch, P: tl.constexpr, W: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.arange(0, NT)
    PER: tl.constexpr = W // NT
    if pid < P:
        t0 = _rt(tid)
        tl.store(stamps + pid * 2 + tid * 0, t0, mask=tid == 0)
        for j in tl.static_range(PER):
            i = j * NT + tid
            pv = epoch * 1000003 + pid * W + i
            if MODE == 4:
                pw = (pv.to(tl.int64) & 0xFFFFFFFF) | (epoch.to(tl.int64) << 32)
                tl.store(pairs + pid * W + i, pw, cache_modifier=".wt")
            else:
                tl.store(data + pid * W + i, pv, cache_modifier=".wt")
        if MODE == 0:
            tb = _fence_asm(tid, "s_barrier")
            _atomic_add(cnt + tb * 0, tb == 0)
        elif MODE == 1:
            _atomic_add(cnt + tid * 0, tid % 64 == 0)
        elif MODE == 2:
            tb = _fence_asm(tid, "s_barrier")
            _store_flag(flags + pid + tb * 0, epoch + tb * 0, tb == 0)
        elif MODE == 3:
            tb = _fence_asm(tid, "s_waitcnt vmcnt(0)\ns_barrier")
            _atomic_add(cnt + tb * 0, tb == 0)
    else:
        o = tl.arange(0, P * W)
        expect = epoch * 1000003 + o
        if MODE == 4:
            w = tl.load(pairs + o, volatile=True, cache_modifier=".cv")
            pend = tl.max(((w >> 32).to(tl.int32) != epoch).to(tl.int32), axis=0)
            while pend != 0:
                w = tl.load(pairs + o, volatile=True, cache_modifier=".cv")
                pend = tl.max(((w >> 32).to(tl.int32) != epoch).to(tl.int32), axis=0)
            v = (w & 0xFFFFFFFF).to(tl.int32)
        else:
            if MODE == 2:
                fm = epoch - 1
                while fm < epoch:
                    fv = tl.load(flags + tid * 0, volatile=True, cache_modifier=".cv")
                    for q in tl.static_range(1, P):
                        fv = tl.minimum(fv, tl.load(flags + q + tid * 0, volatile=True, cache_modifier=".cv"))
                    fm = tl.min(fv, axis=0)
                o = _fence_asm(o + fm * 0, "s_nop 0")
            else:
                target = epoch * (P * (NT // 64) if MODE == 1 else P)
                c = tl.load(cnt + tid * 0, volatile=True, cache_modifier=".cv")
                cm = tl.min(c, axis=0)
                while cm < target:
                    c = tl.load(cnt + tid * 0, volatile=True, cache_modifier=".cv")
                    cm = tl.min(c, axis=0)
                o = _fence_asm(o + cm * 0, "s_nop 0")
            v = tl.load(data + o, volatile=True, cache_modifier=".cv")
        bad = (v != expect).to(tl.int32)
        nb = tl.sum(bad, axis=0)
        t1 = _rt(tid + nb)
        tl.store(stamps + pid * 2 + 1 + tid * 0, t1, mask=tid == 0)
        tl.atomic_add(err + tid * 0, nb, mask=tid == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=2048)
    ap.add_argument("--producers", type=int, default=8)
    ap.add_argument("--consumers", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--modes", default="0,1,2,3,4")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    P, W, C = a.producers, a.words, a.consumers
    for mode in [int(m) for m in a.modes.split(",")]:
        data = torch.zeros(P * W, dtype=torch.int32, device=dev)
        pairs = torch.zeros(P * W, dtype=torch.int64, device=dev)
        cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        flags = torch.zeros(512, dtype=torch.int32, device=dev)
        err = torch.zeros(1, dtype=torch.int32, device=dev)
        stamps = torch.zeros((P + C) * 2, dtype=torch.int64, device=dev)
        lat = []
        for e in range(1, a.iters + 1):
            k = probe[(P + C,)](data, pairs, cnt, flags, err, stamps, e, P=P, W=W, MODE=mode, num_warps=8)
            if e == 1:
                open(f"tests/probes/sem_mode{mode}.s", "w").write(k.asm["amdgcn"])
            if e % 50 == 0 or e == a.iters:
                torch.cuda.synchronize()
                s = stamps.view(-1, 2).cpu()
                t0 = s[:P, 0].min()
                d = (s[P:, 1] - t0).double() / 100.0
                lat.append((d.max().item(), d.median().item()))
        torch.cuda.synchronize()
        lat.sort()
        mx = sorted(l[0] for l in lat)[len(lat) // 2]
        md = sorted(l[1] for l in lat)[len(lat) // 2]
        print(f"MODE {mode}: P={P} C={C} W={W} iters={a.iters}: stale words {err.item()}  "
              f"producer start -> consumer done: median {md:.2f} us, slowest {mx:.2f} us", flush=True)


if __name__ == "__main__":
    main()
