"""Single fixed launch of kn_mla_reduce_v1 for rocprofv3 counter collection."""
import os, torch, aiter

T = int(os.environ.get("PROF_TILES", 256))
S = int(os.environ.get("PROF_SPLITS", 8))
H, Dv = 128, 512
g = torch.Generator(device="cuda").manual_seed(0)
N = T * S
po = torch.randn(N, H, Dv, dtype=torch.float32, device="cuda", generator=g)
pl = torch.randn(N, H, dtype=torch.float32, device="cuda", generator=g) * 2
indptr = torch.arange(0, N + 1, S, dtype=torch.int32, device="cuda")
pmap = torch.arange(N, dtype=torch.int32, device="cuda")
fmap = torch.stack([torch.arange(T, dtype=torch.int32, device="cuda"),
                    torch.arange(1, T + 1, dtype=torch.int32, device="cuda")], 1).contiguous()
fout = torch.empty(T, H, Dv, dtype=torch.bfloat16, device="cuda")
flse = torch.empty(T, H, dtype=torch.float32, device="cuda")

for _ in range(10):  # warmup + JIT
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, 1, 0, fout, flse)
torch.cuda.synchronize()
for _ in range(int(os.environ.get("PROF_ITERS", 20))):
    aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, 1, 0, fout, flse)
torch.cuda.synchronize()
print("done")
