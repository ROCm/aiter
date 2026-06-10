"""Correctness: jagged_dense_bmm_persist_dev (Design B-1, on-device CUM mapping)
vs torch eager, all 4 headline shapes. Tests BOTH a UNIFORM Mi (regression check
vs the static-grid baseline) and a SKEWED Mi distribution with empty / single-row
/ full-envelope groups (where the persistent scheduler should help, and where the
in-kernel CUM search must skip zero-width intervals correctly). cos > 0.999."""
import sys
import torch
import importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist_dev")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def skewed_mi(B, msl, seed=1234):
    """Power-law-ish skew with empty (M_b=0), single-row, and full-envelope groups."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(B, generator=g)
    mi = (msl * (u ** 4)).floor().to(torch.int64)  # heavy tail toward 0
    mi[: max(1, B // 5)] = 0          # force a block of empties
    mi[-1] = msl                       # full envelope
    if B > 1:
        mi[-2] = int(0.9 * msl)
    if B > 2:
        mi[B // 2] = 1                 # a single-row group
    if B > 3:
        mi[B // 2 + 1] = 0             # an empty group between occupied ones
    return [int(x) for x in mi.tolist()]


def run(B, D, Kout, mi_list, msl):
    N, K = Kout, D
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + mi_list[i]
    L = int(so[-1].item())
    torch.manual_seed(0)
    jag = torch.randn(max(L, 1), K, dtype=torch.bfloat16).cuda()
    dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
    sod = so.cuda()
    ref = torch.zeros(max(L, 1), N, dtype=torch.bfloat16).cuda()
    for b in range(B):
        s, e = int(so[b]), int(so[b + 1])
        if e > s:
            ref[s:e] = (jag[s:e].float() @ dense[b].float() + bias[b].float()).to(torch.bfloat16)
    dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bf = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    st = torch.cuda.current_stream()
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st)
    torch.cuda.synchronize()
    if L == 0:
        return 1.0
    return cos(out[:L], ref[:L])


SHAPES = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]
MSL = 7680

allok = True
print("=== UNIFORM Mi=%d (regression check) ===" % MSL)
for (B, D, Kout) in SHAPES:
    mi = [MSL] * B
    c = run(B, D, Kout, mi, MSL)
    ok = c > 0.999
    allok &= ok
    print(f"  B={B} D={D} Kout={Kout}: cos={c:.6f}  {'OK' if ok else 'FAIL'}")

print("\n=== SKEWED Mi (power-law; empty/single-row/full-envelope) ===")
for (B, D, Kout) in SHAPES:
    mi = skewed_mi(B, MSL)
    nz = [x for x in mi if x > 0]
    pct_empty = 100.0 * (len(mi) - len(nz)) / len(mi)
    meanmi = sum(mi) / len(mi)
    c = run(B, D, Kout, mi, MSL)
    ok = c > 0.999
    allok &= ok
    print(f"  B={B} D={D} Kout={Kout}: cos={c:.6f} meanMi={meanmi:.0f} maxMi={max(mi)} "
          f"empty={pct_empty:.0f}%  {'OK' if ok else 'FAIL'}")

print("\nALL PASS" if allok else "\nSOME FAILED")
sys.exit(0 if allok else 1)
