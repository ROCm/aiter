import sys, torch, importlib
import flydsl.compiler as flyc
# args: which(flydsl|triton) B D K_out Mi [n_warps block_n]
# (D=reduction K, K_out=output N)
which=sys.argv[1]; B=int(sys.argv[2]); D=int(sys.argv[3]); Kout=int(sys.argv[4]); Mi=int(sys.argv[5])
n_warps=int(sys.argv[6]) if len(sys.argv)>6 else 4
block_n=int(sys.argv[7]) if len(sys.argv)>7 else 128
# GEMM dims: reduction K=D, output N=Kout
N=Kout; K=D; msl=Mi
so=torch.zeros(B+1,dtype=torch.int32)
for i in range(B): so[i+1]=so[i]+Mi
L=B*Mi
jag=torch.randn(L,K,dtype=torch.bfloat16).cuda()
dense=torch.randn(B,K,N,dtype=torch.bfloat16).cuda()
bias=torch.randn(B,N,dtype=torch.bfloat16).cuda()
sod=so.cuda()
if which=="flydsl":
    m=importlib.import_module("aiter.ops.flydsl.kernels.jagged_dense_bmm_warps")
    dt=dense.transpose(1,2).reshape(B*N,K).contiguous()
    bf=bias.reshape(B*N).contiguous()
    out=torch.zeros(L+128,N,dtype=torch.bfloat16).cuda()
    tA=flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1,divisibility=8)
    tC=flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1,divisibility=8)
    st=torch.cuda.current_stream()
    def launch(): m.jagged_dense_bmm(tC,tA,dt,bf,sod,B,msl,stream=st,n_warps=n_warps,block_n=block_n)
else:
    from generative_recommenders.ops.triton.triton_jagged import triton_jagged_dense_bmm_add_fwd
    so64=sod.to(torch.int64)
    def launch(): return triton_jagged_dense_bmm_add_fwd(msl,so64,jag,dense,bias)
for _ in range(5): launch()
torch.cuda.synchronize()
for _ in range(30): launch()
torch.cuda.synchronize()
