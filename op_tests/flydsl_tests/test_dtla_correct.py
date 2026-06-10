"""Correctness: jagged_dense_bmm_dtla (DirectToLDS A load) vs torch eager across
all 4 headline shapes, each over a small C grid. cos > 0.999 required. Mi small
(512) for speed, plus a real-Mi=7680 spot-check at the best-guess C per shape."""
import torch
import importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_dtla")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run(B, D, Kout, Mi, W, C):
    N, K, msl = Kout, D, Mi
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + Mi
    L = B * Mi
    torch.manual_seed(0)
    jag = torch.randn(L, K, dtype=torch.bfloat16).cuda()
    dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
    sod = so.cuda()
    ref = torch.empty(L, N, dtype=torch.bfloat16).cuda()
    for b in range(B):
        s, e = int(so[b]), int(so[b + 1])
        ref[s:e] = (jag[s:e].float() @ dense[b].float() + bias[b].float()).to(torch.bfloat16)
    dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bf = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    st = torch.cuda.current_stream()
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, xcd_c=C, xcd_w=W)
    torch.cuda.synchronize()
    return cos(out[:L], ref)


# (B, D, Kout)
SHAPES = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]
# C grid spans round-robin (small) -> whole-group (large). W fixed at 8.
CGRID = [16, 32, 60, 120, 240]
W = 8

allok = True
for (B, D, Kout) in SHAPES:
    print(f"=== B={B} D={D} Kout={Kout} (Mi=512) ===")
    for C in CGRID:
        c = run(B, D, Kout, 512, W, C)
        ok = c > 0.999
        allok &= ok
        print(f"  W={W} C={C:>3}: cos={c:.6f}  {'OK' if ok else 'FAIL'}")

print("\n=== real Mi=7680 spot-check (C=120) ===")
for (B, D, Kout) in SHAPES:
    c = run(B, D, Kout, 7680, W, 120)
    ok = c > 0.999
    allok &= ok
    print(f"  B={B} D={D} Kout={Kout}: cos={c:.6f}  {'OK' if ok else 'FAIL'}")

print("\nALL PASS" if allok else "\nSOME FAILED")
