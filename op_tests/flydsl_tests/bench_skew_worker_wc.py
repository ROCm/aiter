import sys, torch, importlib
import flydsl.compiler as flyc
# args: <B> <D> <Kout> <max_seq_len> <W> <C> [seed]
B=int(sys.argv[1]); D=int(sys.argv[2]); Kout=int(sys.argv[3]); MSL=int(sys.argv[4])
W=int(sys.argv[5]); C=int(sys.argv[6])
seed=int(sys.argv[7]) if len(sys.argv)>7 else 1234
N=Kout; K=D
torch.manual_seed(seed)
Mi=[max(1,int(0.95*torch.rand(1).item()*MSL)) for _ in range(B)]
amsl=max(Mi)
so=torch.zeros(B+1,dtype=torch.int32)
for i in range(B): so[i+1]=so[i]+Mi[i]
L=int(so[-1].item())
jag=torch.randn(L,K,dtype=torch.bfloat16).cuda()
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
def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,amsl,stream=st,xcd_c=C,xcd_w=W)
for _ in range(5): launch()
torch.cuda.synchronize()
for _ in range(30): launch()
torch.cuda.synchronize()
