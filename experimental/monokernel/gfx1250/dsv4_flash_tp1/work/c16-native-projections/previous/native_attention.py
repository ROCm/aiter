from previous.common import *
from previous.attention_paged import attention_paged
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
torch.manual_seed(42)
q=torch.randn((M,64,512),device='cuda',dtype=torch.bfloat16)*.1;kv=torch.randn((M,512),device='cuda',dtype=torch.bfloat16)*.1
sink=torch.randn(64,device='cuda')
SPLITS=int(os.getenv("SPLITS","8"))
for n in [128,384,136]:
 pages=M*640;cache=torch.randn((pages,512),device='cuda',dtype=torch.bfloat16)*.1
 idx=torch.randperm(pages,device='cuda',dtype=torch.int32)[:M*n].contiguous();ptr=torch.arange(0,(M+1)*n,n,device='cuda',dtype=torch.int32)
 dest=idx.reshape(M,n)[:,-1].contiguous();cache[dest.long()]=kv
 e=lambda sh,dt=torch.float32:torch.empty(sh,device='cuda',dtype=dt)
 aa=[e((M,SPLITS,64)),e((M,SPLITS,64)),e((M,SPLITS,64,512)),e((M,64,512),torch.bfloat16),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
 def fa():
  attention_paged[(256,)](q,kv,cache,idx,ptr,sink,*aa,dest,pages,M,SPLITS,**kw)
  return aa[3]
 def na():return sparse_attn_v4_paged_decode(q,cache,idx,ptr,sink,512**-.5)
 rr=na();fa();torch.cuda.synchronize();assert int(aa[-1])==0
 bench('attention_'+str(n)+('_s'+str(SPLITS) if SPLITS!=8 else ''),{'native':na,'fused':fa},err(aa[3],rr))
