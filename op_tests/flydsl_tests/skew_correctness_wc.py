import sys, torch, importlib
import flydsl.compiler as flyc

# args: <B> <D> <Kout> <max_seq_len> <W> <C> [seed] [mode]
# mode: "rand" (default) skewed M_i ~ 0.95*Uniform(1,msl); "edge" forces empty + single-row groups.
B=int(sys.argv[1]); D=int(sys.argv[2]); Kout=int(sys.argv[3]); MSL=int(sys.argv[4])
W=int(sys.argv[5]); C=int(sys.argv[6])
seed=int(sys.argv[7]) if len(sys.argv)>7 else 1234
mode=sys.argv[8] if len(sys.argv)>8 else "rand"
N=Kout; K=D
torch.manual_seed(seed)

# Build skewed M_i.
Mi=[]
for b in range(B):
    m=max(1,int(0.95*torch.rand(1).item()*MSL))
    Mi.append(m)
if mode=="edge":
    # force an empty group, a single-row group, and a full-envelope group.
    Mi[0]=0
    Mi[1]=1
    if B>2: Mi[2]=MSL
actual_msl=max(Mi+[1])  # max over drawn M_i (>=1 so the grid is non-empty)
so=torch.zeros(B+1,dtype=torch.int32)
for i in range(B): so[i+1]=so[i]+Mi[i]
L=int(so[-1].item())
print(f"B={B} D={D} N={N} MSL_env={MSL} actual_max_Mi={actual_msl} L={L} mode={mode} empty={sum(1 for m in Mi if m==0)} single={sum(1 for m in Mi if m==1)}")

jag=torch.randn(max(L,1),K,dtype=torch.bfloat16).cuda()
dense=torch.randn(B,K,N,dtype=torch.bfloat16).cuda()
bias=torch.randn(B,N,dtype=torch.bfloat16).cuda()
sod=so.cuda()

m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_wc")
dt=dense.transpose(1,2).reshape(B*N,K).contiguous()
bf=bias.reshape(B*N).contiguous()
out=torch.zeros(L+128,N,dtype=torch.bfloat16).cuda()
tA=flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1,divisibility=8)
tC=flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1,divisibility=8)
st=torch.cuda.current_stream()
# pass max_seq_len = actual max Mi (grid envelope); kernel early-exits per group.
m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,actual_msl,stream=st,xcd_c=C,xcd_w=W)
torch.cuda.synchronize()

# torch eager reference (per-group).
ref=torch.zeros(L,N,dtype=torch.float32).cuda()
for b in range(B):
    s=int(so[b].item()); e=int(so[b+1].item())
    if e>s:
        ref[s:e]=(jag[s:e].float()@dense[b].float())+bias[b].float()
got=out[:L].float()
if L==0:
    print("L==0 (all empty) cos=N/A trivially OK"); sys.exit(0)
cos=torch.nn.functional.cosine_similarity(got.flatten(),ref.flatten(),dim=0).item()
maxabs=(got-ref).abs().max().item()
print(f"cos={cos:.6f} max_abs_err={maxabs:.4f} {'PASS' if cos>0.999 else 'FAIL'}")
