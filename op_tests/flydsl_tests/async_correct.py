"""Correctness for jagged_dense_bmm_async across BLOCK_M sweep + optional atom.
Usage: python async_correct.py <block_m>
cos > 0.999 required vs torch eager. Checks Mi=512 (fast) and Mi=7680 spot."""
import sys
import torch
import importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_async")
BM = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run(B, D, Kout, Mi, W, C, bm):
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
    out = torch.zeros(L + 256, N, dtype=torch.bfloat16).cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    st = torch.cuda.current_stream()
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, xcd_c=C, xcd_w=W, block_m=bm)
    torch.cuda.synchronize()
    return cos(out[:L], ref)


SHAPES = [(120, 512, 512), (1024, 512, 512)]
W = 8
allok = True
print(f"=== BLOCK_M={BM} ===")
for (B, D, Kout) in SHAPES:
    c512 = run(B, D, Kout, 512, W, 60, BM)
    ok = c512 > 0.999
    allok &= ok
    print(f"  B={B} D={D} Mi=512 : cos={c512:.6f} {'OK' if ok else 'FAIL'}")
for (B, D, Kout) in SHAPES:
    c = run(B, D, Kout, 7680, W, 60, BM)
    ok = c > 0.999
    allok &= ok
    print(f"  B={B} D={D} Mi=7680: cos={c:.6f} {'OK' if ok else 'FAIL'}")
print("ALL PASS" if allok else "SOME FAILED")
