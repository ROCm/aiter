"""Time the Gluon selection pieces (rank_entries / place) alone: 32 CTAs, each ranking E x 512
random keys (n live), stamps (s_memrealtime, 100 MHz) after each piece. One GPU.
    python3 tests/probes/rank_probe.py [--n 321] [--e 2] [--floor 1]"""
import argparse
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import vllm.models.minimax_m3.amd.mono.gluon.kernel as K
from vllm.models.minimax_m3.amd.mono.gluon.prims import T1, THREADS


@gluon.jit
def clk():
    return gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s", [], dtype=gl.int64,
                                     is_pure=False, pack=1)


@gluon.jit
def probe(keys_ptr, out_ptr, stamps, n, E: gl.constexpr, FLOOR: gl.constexpr, REPS: gl.constexpr):
    bid = gl.program_id(0)
    t = K.ar(THREADS, T1)
    lcand = gl.allocate_shared_memory(gl.int64, [2 * THREADS], K.FLAT1)
    lctr = gl.allocate_shared_memory(gl.int32, [THREADS], K.FLAT1)
    k0 = gl.load(keys_ptr + bid * 1024 + t)
    k1 = gl.load(keys_ptr + bid * 1024 + 512 + t)
    lv0 = t < n
    lv1 = 512 + t < n
    gl.barrier()
    r0 = t * 0
    r1 = t * 0
    c0 = clk()
    for r in range(REPS):
        lctr.store(t * 0)
        gl.barrier()
        if E == 2:
            r0, r1 = K.rank_entries((k0, k1), (t, t + 512), (lv0, lv1), lcand, lctr, FLOOR)
        else:
            (r0,) = K.rank_entries((k0,), (t,), (lv0,), lcand, lctr, FLOOR, None, 0, 0 if FLOOR else 128)
            r1 = r0
        gl.barrier()
    c1 = clk()
    gl.store(out_ptr + bid * 1024 + t, r0)
    gl.store(out_ptr + bid * 1024 + 512 + t, r1)
    gl.store(stamps + bid + t * 0, (c1 - c0) + t * 0, mask=t == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=321)
    ap.add_argument("--e", type=int, default=2)
    ap.add_argument("--floor", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    keys = (torch.randn(32, 1024, generator=g, device=dev) * 1e4).to(torch.int32)
    out = torch.zeros(32, 1024, dtype=torch.int32, device=dev)
    st = torch.zeros(32, dtype=torch.int64, device=dev)
    for _ in range(2):
        probe[(32,)](keys, out, st, a.n, E=a.e, FLOOR=bool(a.floor), REPS=a.reps, num_warps=8)
    torch.cuda.synchronize()
    us = st.float().median().item() / 100.0 / a.reps
    # check: ranks of the live entries vs torch (key desc, id desc)
    n = a.n
    ok = True
    for b in range(4):
        kk = keys[b, :n].long() * (1 << 32) + torch.arange(n, device=dev)
        ref = (kk[None, :] > kk[:, None]).sum(1)
        got = out[b, :n].long()
        top = ref < 16
        ok &= bool((got[top] == ref[top]).all())
    print(f"n {n} E {a.e} floor {a.floor}: {us:.2f} us per rank_entries, top-16 exact {ok}")




@gluon.jit
def probe_parts(keys_ptr, out_ptr, stamps, n, PART: gl.constexpr, REPS: gl.constexpr):
    """PART 1: floor only; 2: + compaction; 3: + count (= rank_entries, E = 1)."""
    bid = gl.program_id(0)
    t = K.ar(THREADS, T1)
    lcand = gl.allocate_shared_memory(gl.int64, [2 * THREADS], K.FLAT1)
    lctr = gl.allocate_shared_memory(gl.int32, [THREADS], K.FLAT1)
    key = gl.load(keys_ptr + bid * 1024 + t)
    live = t < n
    acc = t * 0
    gl.barrier()
    c0 = clk()
    for r in range(REPS):
        lctr.store(t * 0)
        gl.barrier()
        least = t * 0 - 2147483647 - 1
        mine = gl.where(live, K.key64(key, t), K.key64(least, t * 0))
        top = gl.where(live, key, least)
        gmax = gl.max(gl.convert_layout(gl.reshape(top, [32, 16]), K.G16), axis=0)
        floor = gl.min(gmax, axis=0)
        acc = acc + floor
        if PART >= 2:
            ce = live & (key >= floor)
            slot = lctr.atomic_scatter_add(t * 0 + 1, t * 0, 0, mask=ce)
            lcand.scatter(mine, gl.where(ce & (slot < 256), slot, 2 * THREADS - 1), 0)
            gl.barrier()
            m = gl.max(lctr.load(T1), axis=0)
            acc = acc + m
            if PART >= 3:
                acc = acc + K.count_beaten(mine, lcand, m, 16)
        gl.barrier()
    c1 = clk()
    gl.store(out_ptr + bid * 1024 + t, acc)
    gl.store(stamps + bid + t * 0, (c1 - c0) + t * 0, mask=t == 0)


def parts():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    keys = (torch.randn(32, 1024, generator=g, device=dev) * 1e4).to(torch.int32)
    out = torch.zeros(32, 1024, dtype=torch.int32, device=dev)
    st = torch.zeros(32, dtype=torch.int64, device=dev)
    for part in (1, 2, 3):
        for _ in range(2):
            probe_parts[(32,)](keys, out, st, 321, PART=part, REPS=20, num_warps=8)
        torch.cuda.synchronize()
        print(f"part {part}: {st.float().median().item() / 100.0 / 20:.2f} us")


if __name__ == "__main__":
    import sys
    parts() if "--parts" in sys.argv else main()
