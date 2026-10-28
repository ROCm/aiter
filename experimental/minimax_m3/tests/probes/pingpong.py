"""Hand-off latency between two CTAs of one GPU: CTA 0 and CTA X alternate tagged-pair
writes N times (one lane each); time / (2 N) = one hop (store -> visible -> poll sees it).
Store / load cache bits set by inline asm.   python3 pingpong.py"""
import torch
import triton
import triton.language as tl

STORES = {"sc0sc1": "sc0 sc1", "sc1": "sc1", "nt_sc1": "sc1 nt"}
LOADS = {"sc0sc1": "sc0 sc1", "sc1": "sc1"}


@triton.jit
def pp(buf, big, NBIG, N, BUSY, OTHER: tl.constexpr, ST: tl.constexpr, LD: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0 or pid == OTHER:
        me = (pid != 0).to(tl.int32)
        a_mine = buf.to(tl.int64, bitcast=True) + me * 256  # separate 256 B lines
        a_peer = buf.to(tl.int64, bitcast=True) + (1 - me) * 256
        i = 0
        while i < N:
            if me == 0:
                tl.inline_asm_elementwise(
                    "global_store_dword $1, $2, off " + ST + "\ns_mov_b32 $0, 0", "=s,v,v,~{memory}",
                    [a_mine, i + 1], dtype=tl.int32, is_pure=False, pack=1)
            # wait for peer's tag == i + 1
            tl.inline_asm_elementwise(
                "1:\nglobal_load_dword $0, $1, off " + LD + "\ns_waitcnt vmcnt(0)\nv_cmp_ne_u32 vcc, $0, $2\n"
                "s_cbranch_vccnz 1b", "=&v,v,v,~{vcc},~{memory}", [a_peer, i + 1], dtype=tl.int32, is_pure=False, pack=1)
            if me == 1:
                tl.inline_asm_elementwise(
                    "global_store_dword $1, $2, off " + ST + "\ns_mov_b32 $0, 0", "=s,v,v,~{memory}",
                    [a_mine, i + 1], dtype=tl.int32, is_pure=False, pack=1)
            i += 1
        tl.atomic_xchg(buf + 128 + pid, 1)  # done: the busy CTAs stop
    else:
        # keep the GPU's clocks up while the pair runs
        # stream HBM (as the MoE weight loads do) while the pair runs
        offs = tl.arange(0, 1024)
        acc = tl.zeros([1024], dtype=tl.int32)
        j = 0
        while j < BUSY:
            base = ((pid * 7919 + j * 256) % NBIG) * 1024
            acc += tl.load(big + base + offs, cache_modifier=".cg")
            j += 1
        tl.store(buf + 192 + (pid % 32), tl.sum(acc))


import os
if os.environ.get("DUMP"):
    c = pp.warmup(torch.zeros(256, dtype=torch.int32, device="cuda"), torch.zeros(1024, dtype=torch.int32, device="cuda"), 1, 10, 10, 9, "sc1", "sc1", grid=(256,), num_warps=1)
    print(c.asm["amdgcn"])
    raise SystemExit
N = 2000
BUSY = int(os.environ.get('BUSY', '0'))
NBIG = 1 << 18  # 1 GB of int32 rows of 4 KB
big = torch.ones(NBIG * 1024, dtype=torch.int32, device='cuda')
for other in (8, 1, 128):  # CTA placement: neighbour / other XCD guesses
    for sn, st in STORES.items():
        for ln, ld in LOADS.items():
            buf = torch.zeros(256, dtype=torch.int32, device="cuda")
            pp[(256,)](buf, big, NBIG, 10, 10, other, st, ld, num_warps=1)
            torch.cuda.synchronize()
            buf.zero_()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            pp[(256,)](buf, big, NBIG, N, BUSY, other, st, ld, num_warps=1)
            b.record()
            torch.cuda.synchronize()
            print(f"other CTA {other:3d}  store {sn:7s} load {ln:7s}: {a.elapsed_time(b) * 1000 / (2 * N):6.3f} us/hop", flush=True)
