import sys
import importlib
import torch
import torch.nn.functional as F
import flydsl.compiler as flyc

# args: <B> <D> <Kout> <Mi> <S>   (D=reduction K, Kout=output N, S=epi_strips)
B = int(sys.argv[1])
D = int(sys.argv[2])
Kout = int(sys.argv[3])
Mi = int(sys.argv[4])
S = int(sys.argv[5])
N = Kout
K = D
msl = Mi

torch.manual_seed(0)
so = torch.zeros(B + 1, dtype=torch.int32)
for i in range(B):
    so[i + 1] = so[i] + Mi
L = B * Mi
jag = torch.randn(L, K, dtype=torch.bfloat16).cuda()
dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
sod = so.cuda()

m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_cstrip")
dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
bf = bias.reshape(B * N).contiguous()
out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
st = torch.cuda.current_stream()
m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, epi_strips=S)
torch.cuda.synchronize()

# torch eager reference
ref = torch.zeros(L, N, dtype=torch.float32).cuda()
for b in range(B):
    s, e = int(so[b]), int(so[b + 1])
    ref[s:e] = (jag[s:e].float() @ dense[b].float()) + bias[b].float()
got = out[:L].float()
cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
maxabs = (got - ref).abs().max().item()
print(f"B={B} D={D} Kout={Kout} Mi={Mi} S={S}  cos={cos:.6f}  maxabs={maxabs:.4f}")
