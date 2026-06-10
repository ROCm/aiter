"""Correctness: jagged_dense_bmm_warps vs torch eager across the D512 + D256
headline shapes, sweeping the MMA warp layout (n_warps) and tile_n (block_n).
cos > 0.999 required."""
import torch
import importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_warps")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run(B, D, Kout, Mi, n_warps, block_n):
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
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st,
                       n_warps=n_warps, block_n=block_n)
    torch.cuda.synchronize()
    return cos(out[:L], ref)


# (B, D, Kout)
D512 = [(120, 512, 512), (1024, 512, 512)]
D256 = [(120, 256, 256), (1024, 256, 256)]
# (n_warps, block_n)
CONFIGS = [(4, 128), (8, 128), (8, 256), (16, 256), (16, 512)]

allok = True
print("=== D512 shapes (Mi=512) — warp-layout sweep ===")
for (B, D, Kout) in D512:
    print(f"--- B={B} D={D} Kout={Kout} ---")
    for (nw, bn) in CONFIGS:
        if bn > Kout:
            print(f"  n_warps={nw:>2} block_n={bn:>3}: skip (block_n>N)")
            continue
        try:
            c = run(B, D, Kout, 512, nw, bn)
            ok = c > 0.999
            allok &= ok
            print(f"  n_warps={nw:>2} block_n={bn:>3}: cos={c:.6f}  {'OK' if ok else 'FAIL'}")
        except Exception as ex:
            print(f"  n_warps={nw:>2} block_n={bn:>3}: NOT EXPRESSIBLE / ERROR: {type(ex).__name__}: {str(ex)[:160]}")

print("\n=== D256 shapes (Mi=512) — re-check best D512 configs ===")
for (B, D, Kout) in D256:
    print(f"--- B={B} D={D} Kout={Kout} ---")
    for (nw, bn) in [(4, 128), (8, 128), (8, 256), (16, 256)]:
        if bn > Kout:
            print(f"  n_warps={nw:>2} block_n={bn:>3}: skip (block_n>N)")
            continue
        try:
            c = run(B, D, Kout, 512, nw, bn)
            ok = c > 0.999
            allok &= ok
            print(f"  n_warps={nw:>2} block_n={bn:>3}: cos={c:.6f}  {'OK' if ok else 'FAIL'}")
        except Exception as ex:
            print(f"  n_warps={nw:>2} block_n={bn:>3}: NOT EXPRESSIBLE / ERROR: {type(ex).__name__}: {str(ex)[:160]}")

print("\nALL PASS" if allok else "\nSOME FAILED")
