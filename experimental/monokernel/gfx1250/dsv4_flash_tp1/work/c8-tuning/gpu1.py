from common import *
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
from qkv_full_batch import qkv_batch
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
n=NativeQKV();f=n.f;m=8
torch.manual_seed(1776);x=f['live'].cuda().reshape(1,4096).repeat(m,1)+torch.randn((m,4096),device='cuda',dtype=torch.bfloat16)*.01
w=f['W_bytes'].cuda().view(torch.float8_e4m3fn);ws=f['WS'].cuda().view(torch.uint8);nw=f['NW'].cuda();qb=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);qbs=f['QBS'].cuda().view(torch.uint8)
xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16);q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
cos=n.cos[1024].reshape(1,32).repeat(m,1);sin=n.sin[1024].reshape(1,32).repeat(m,1);counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
qbr=torch.empty((m,32768),device='cuda');qo=torch.empty((m,32768),device='cuda',dtype=torch.bfloat16);pos=torch.full((m,),1024,device='cuda',dtype=torch.int64)
def fused():
 qkv_batch[(256,)](x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,qb,qbs,qbr,qo,m,PREFETCH_STYLE=1,**kw)
 return qo
def native():
 a=n.qa(x);rq,rs=n.norm(a[:,:1024]);raw=n.qb(rq,x_scale=rs)
 return qk_norm_rope_maybe_quant(raw,a[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6).q_sa
rr=native();vv=fused();torch.cuda.synchronize();bench('qkv',{'native':native,'fused':fused},err(vv,rr))
from attention_defs import batch,scratch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
for t in [128,144,384,640]:
 splits=64;qatt=qo.view(8,64,512);cache=torch.randn((8,t,512),device='cuda',dtype=torch.bfloat16)*.1;cache[:,-1]=kv
 idx=torch.arange(t,device='cuda',dtype=torch.int32).repeat(8,1);ptr=torch.tensor([[0,t]]*8,device='cuda',dtype=torch.int32);sink=torch.randn(64,device='cuda');aa=scratch(8,t,splits)
 ni=torch.arange(8*t,device='cuda',dtype=torch.int32);np=torch.arange(0,9*t,t,device='cuda',dtype=torch.int32)
 def fa():
  batch[(256,)](qatt,kv,cache,idx,ptr,sink,*aa,8,t,splits,**kw)
  return aa[3]
 def na():return sparse_attn_v4_paged_decode(qatt,cache.reshape(-1,512),ni,np,sink,512**-.5)
 rr=na();vv=fa();torch.cuda.synchronize();bench('attention_t'+str(t),{'native':na,'fused':fa},err(vv,rr))
