"""Correctness: jagged_dense_bmm_xcd vs torch eager. Sweeps the (W,C) configs
and checks cosine similarity > 0.999 for both a small Mi and the real Mi=7680."""
import sys, torch, importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_xcd")


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run(B, D, Kout, Mi, W, C):
    N = Kout
    K = D
    msl = Mi
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + Mi
    L = B * Mi
    torch.manual_seed(0)
    jag = torch.randn(L, K, dtype=torch.bfloat16).cuda()
    dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
    sod = so.cuda()

    # torch reference: per group b, out[s:e] = jag[s:e] @ dense[b] + bias[b]
    ref = torch.empty(L, N, dtype=torch.bfloat16).cuda()
    for b in range(B):
        s = int(so[b])
        e = int(so[b + 1])
        ref[s:e] = (jag[s:e].float() @ dense[b].float() + bias[b].float()).to(torch.bfloat16)

    dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bf = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    st = torch.cuda.current_stream()
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, xcd_c=C, xcd_w=W)
    torch.cuda.synchronize()
    got = out[:L]
    c = cos(got, ref)
    return c


CONFIGS = [(4, 16), (8, 16), (8, 32), (5, 25), (8, 64), (8, 120), (8, 240)]

print("=== small: B=1024 D=512 Kout=512 Mi=512 ===")
for W, C in CONFIGS:
    c = run(1024, 512, 512, 512, W, C)
    print(f"  W={W:>3} C={C:>3}: cos={c:.6f}  {'OK' if c > 0.999 else 'FAIL'}")

print("=== real: B=1024 D=512 Kout=512 Mi=7680 ===")
for W, C in [(8, 16), (8, 64), (8, 240)]:
    c = run(1024, 512, 512, 7680, W, C)
    print(f"  W={W:>3} C={C:>3}: cos={c:.6f}  {'OK' if c > 0.999 else 'FAIL'}")
