"""Confirm waves_per_eu hint lands in lowered ISA. Compiles one shape at the
given waves_per_eu and dumps IR/ISA via FLYDSL_DUMP_IR. Run twice (wpe=0 vs
wpe=4) into different dump dirs, then grep the .s for VGPR / amdhsa metadata."""
import sys
import torch
import importlib
import flydsl.compiler as flyc

WPE = int(sys.argv[1])
B, D, Kout, Mi = 120, 256, 256, 512
N, K, msl = Kout, D, Mi
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
m.jagged_dense_bmm(tC, tA, dt, bf, sod, B, msl, stream=st, waves_per_eu=WPE)
torch.cuda.synchronize()
print(f"compiled+ran wpe={WPE}")
