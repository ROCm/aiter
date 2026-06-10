"""Correctness: jagged_dense_bmm_wpe vs torch eager across all 4 headline shapes
at Mi=7680, for each waves_per_eu in {0,1,2,3,4}. cos > 0.999 required."""
import torch
import importlib
import flydsl.compiler as flyc

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_wpe")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run(B, D, Kout, Mi, wpe):
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
    m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, waves_per_eu=wpe)
    torch.cuda.synchronize()
    return cos(out[:L], ref)


SHAPES = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]
WPES = [0, 1, 2, 3, 4]
Mi = 7680

allok = True
print(f"=== correctness Mi={Mi} ===")
hdr = "shape".ljust(20) + "".join(f"wpe={w}".rjust(12) for w in WPES)
print(hdr)
for (B, D, Kout) in SHAPES:
    row = f"B{B}_D{D}_K{Kout}".ljust(20)
    for w in WPES:
        c = run(B, D, Kout, Mi, w)
        ok = c > 0.999
        allok &= ok
        row += f"{c:.5f}{'' if ok else '!'}".rjust(12)
    print(row)
print("\nALL PASS" if allok else "\nSOME FAILED (cos<=0.999 marked !)")
