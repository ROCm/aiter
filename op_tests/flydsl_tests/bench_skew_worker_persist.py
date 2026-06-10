import sys, torch, importlib, time
import flydsl.compiler as flyc
# args: <B> <D> <Kout> <max_seq_len> <variant> [seed]
#   variant: persist_kernel | persist_host | baseline_gen
B=int(sys.argv[1]); D=int(sys.argv[2]); Kout=int(sys.argv[3]); MSL=int(sys.argv[4])
variant=sys.argv[5]
seed=int(sys.argv[6]) if len(sys.argv)>6 else 1234
N=Kout; K=D

# Power-law skew (matches test_persist_correct.skewed_mi): a few groups near MSL,
# many short, ~27-28% empty.
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
elif variant=="persist_kernel":
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist")
    wi_cpu,nt=m.build_work_items(so,B,MSL,N)
    wi=wi_cpu.cuda()
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st,work_items=wi,num_tiles=nt)
elif variant=="persist_host":
    # Includes the seq_offsets.cpu() sync + host work-list build + upload each call.
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_persist")
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,MSL,stream=st)
else:
    raise SystemExit("bad variant")

for _ in range(5): launch()
torch.cuda.synchronize()
# Wall-clock (covers host work for persist_host).
t0=time.perf_counter()
for _ in range(30): launch()
torch.cuda.synchronize()
t1=time.perf_counter()
nz=[x for x in mi if x>0]
print(f"WALL variant={variant} B={B} D={D} meanMi={sum(mi)/len(mi):.0f} maxMi={max(mi)} "
      f"empty={100.0*(len(mi)-len(nz))/len(mi):.0f}% wall_us/iter={1e6*(t1-t0)/30:.1f}")
