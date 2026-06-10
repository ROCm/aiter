import sys, torch, importlib
import flydsl.compiler as flyc
# args: <B> <D> <Kout> <Mi> <W> <C> <BLOCK_M>
B=int(sys.argv[1]); D=int(sys.argv[2]); Kout=int(sys.argv[3]); Mi=int(sys.argv[4])
W=int(sys.argv[5]); C=int(sys.argv[6]); BM=int(sys.argv[7])
N=Kout; K=D; msl=Mi
so=torch.zeros(B+1,dtype=torch.int32)
for i in range(B): so[i+1]=so[i]+Mi
L=B*Mi
jag=torch.randn(L,K,dtype=torch.bfloat16).cuda()
dense=torch.randn(B,K,N,dtype=torch.bfloat16).cuda()
bias=torch.randn(B,N,dtype=torch.bfloat16).cuda()
sod=so.cuda()
m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_async")
dt=dense.transpose(1,2).reshape(B*N,K).contiguous()
bf=bias.reshape(B*N).contiguous()
out=torch.zeros(L+256,N,dtype=torch.bfloat16).cuda()
tA=flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1,divisibility=8)
tC=flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1,divisibility=8)
st=torch.cuda.current_stream()
def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,msl,stream=st,xcd_c=C,xcd_w=W,block_m=BM)
for _ in range(5): launch()
torch.cuda.synchronize()
for _ in range(30): launch()
torch.cuda.synchronize()
