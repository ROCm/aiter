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
rr=native();vv=fused();torch.cuda.synchronize()
from attention_defs import scratch
from attention_paged import attention_paged
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
for n in [128,136,144,384,512,640]:
 splits=8;qatt=qo.view(8,64,512);pages=8*640
 torch.manual_seed(42)
 cache=torch.randn((pages,512),device='cuda',dtype=torch.bfloat16)*.1
 ni=torch.randperm(pages,device='cuda',dtype=torch.int32)[:8*n].contiguous();ptr=torch.arange(0,9*n,n,device='cuda',dtype=torch.int32)
 dest=ni.reshape(8,n)[:,-1].contiguous();cache[dest.long()]=kv
 sink=torch.randn(64,device='cuda');aa=scratch(8,640,splits)
 def fa():
  obj=attention_paged[(256,)](qatt,kv,cache,ni,ptr,sink,*aa,dest,pages,8,8,**kw)
  return aa[3]
 def na():return sparse_attn_v4_paged_decode(qatt,cache,ni,ptr,sink,512**-.5)
 rr=na();vv=fa();torch.cuda.synchronize();bench('attention_paged_'+str(n),{'native':na,'fused':fa},err(vv,rr))
