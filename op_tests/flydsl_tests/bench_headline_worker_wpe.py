import sys, torch, importlib
import flydsl.compiler as flyc
# args: <B> <D> <Kout> <Mi> <WPE>   (D=reduction K, Kout=output N, WPE=waves_per_eu)
B = int(sys.argv[1]); D = int(sys.argv[2]); Kout = int(sys.argv[3]); Mi = int(sys.argv[4])
WPE = int(sys.argv[5]) if len(sys.argv) > 5 else 0
N = Kout; K = D; msl = Mi
so = torch.zeros(B + 1, dtype=torch.int32)
for i in range(B):
    so[i + 1] = so[i] + Mi
L = B * Mi
jag = torch.randn(L, K, dtype=torch.bfloat16).cuda()
dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
sod = so.cuda()
m = importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_wpe")
dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
bf = bias.reshape(B * N).contiguous()
out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
st = torch.cuda.current_stream()
def launch(): m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, waves_per_eu=WPE)
for _ in range(5): launch()
torch.cuda.synchronize()
for _ in range(30): launch()
torch.cuda.synchronize()
