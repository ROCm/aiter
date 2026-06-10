import sys, torch, importlib, time
import flydsl.compiler as flyc
# args: <B> <D> <Kout> <max_seq_len> <variant> [seed] [dist]
#   variant: persist_dev | persist_dev_kernel | baseline_gen | persist_host
#   dist (default skew): skew | uniform
#     - persist_dev:        END-TO-END wall (CUM built on-device EACH call) <- KEY METRIC
#     - persist_dev_kernel: kernel-only-ish wall (CUM prebuilt once, reused)
#     - baseline_gen:       baseline static-grid _gen wall
#     - persist_host:       OLD Design-A persist with seq_offsets.cpu() each call
B=int(sys.argv[1]); D=int(sys.argv[2]); Kout=int(sys.argv[3]); MSL=int(sys.argv[4])
variant=sys.argv[5]
seed=int(sys.argv[6]) if len(sys.argv)>6 else 1234
dist=sys.argv[7] if len(sys.argv)>7 else "skew"
N=Kout; K=D

if dist=="uniform":
    mi=[MSL]*B
else:
    g=torch.Generator().manual_seed(seed)
    u=torch.rand(B,generator=g)
    mi=(MSL*(u**4)).floor().to(torch.int64)
    mi[:max(1,B//5)]=0
    mi[-1]=MSL
    if B>1: mi[-2]=int(0.9*MSL)
    mi=[int(x) for x in mi.tolist()]

so=torch.zeros(B+1,dtype=torch.int32)
for i in range(B): so[i+1]=so[i]+mi[i]
L=int(so[-1])
torch.manual_seed(0)
jag=torch.randn(max(L,1),K,dtype=torch.bfloat16).cuda()
dense=torch.randn(B,K,N,dtype=torch.bfloat16).cuda()
bias=torch.randn(B,N,dtype=torch.bfloat16).cuda()
sod=so.cuda()
dt=dense.transpose(1,2).reshape(B*N,K).contiguous()
bf=bias.reshape(B*N).contiguous()
out=torch.zeros(L+128,N,dtype=torch.bfloat16).cuda()
tA=flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1,divisibility=8)
tC=flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1,divisibility=8)
st=torch.cuda.current_stream()

if variant=="baseline_gen":
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_gen")
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st)
elif variant=="persist_dev":
    # END-TO-END: CUM is built on-device EACH call (the public default path).
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist_dev")
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st)
elif variant=="persist_dev_kernel":
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist_dev")
    cum=m.build_cum(sod,B,N)
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st,cum=cum)
elif variant=="persist_host":
    # OLD Design A: seq_offsets.cpu() + host work-list build + upload EACH call.
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist")
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st)
else:
    raise SystemExit("bad variant")

for _ in range(5): launch()
torch.cuda.synchronize()
# Wall-clock (covers host work for persist_host and on-device CUM for persist_dev).
t0=time.perf_counter()
for _ in range(30): launch()
torch.cuda.synchronize()
t1=time.perf_counter()
nz=[x for x in mi if x>0]
print(f"WALL variant={variant} dist={dist} B={B} D={D} meanMi={sum(mi)/len(mi):.0f} maxMi={max(mi)} "
      f"empty={100.0*(len(mi)-len(nz))/len(mi):.0f}% wall_us/iter={1e6*(t1-t0)/30:.1f}")
