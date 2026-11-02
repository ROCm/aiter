import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native/prefix']
from attention_batch import attention_body
from attention_matched import attention_kernel
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def batch(Q,KV,C,I,P,S,MAX,SUM,A,O,COUNT,COUNTER,H,M:gl.constexpr,T:gl.constexpr,SPLITS:gl.constexpr):
 attention_body(Q,KV,C,I,P,S,MAX,SUM,A,O,COUNT,COUNTER,H,M,T,SPLITS)
def scratch(m,t,s):
 return [torch.empty((m,s,64),device='cuda'),torch.empty((m,s,64),device='cuda'),torch.empty((m,s,64,512),device='cuda'),torch.empty((m,64,512),device='cuda',dtype=torch.bfloat16),torch.zeros(4,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
torch.manual_seed(20);results=[]
for m in [1,2,4,8]:
 for t in [128,144,640]:
  splits=64
  q=torch.randn((m,64,512),device='cuda',dtype=torch.bfloat16)*.1;kv=torch.randn((m,512),device='cuda',dtype=torch.bfloat16)*.1
  cache=torch.randn((m,t,512),device='cuda',dtype=torch.bfloat16)*.1
  idx=torch.arange(t,device='cuda',dtype=torch.int32).repeat(m,1);ptr=torch.zeros((m,2),device='cuda',dtype=torch.int32)
  for row in range(m):
   n=t-row*7;idx[row,n-1]=t-1;idx[row,n:]=-1;ptr[row,1]=n
  sink=torch.randn(64,device='cuda');a=scratch(m,t,splits)
  obj=batch[(256,)](q,kv,cache,idx,ptr,sink,*a,m,t,splits,num_warps=4,enable_fp_fusion=False)
  refs=[]
  for row in range(m):
   b=scratch(1,t,splits)
   attention_kernel[(256,)](q[row],kv[row],cache[row],idx[row],ptr[row],sink,*b,T=t,PAGES=t,WRITE_SLOT=t-1,SPLITS=splits,GRID=256,BH=16,BK=16,VECTOR=False,DIRECT=True,num_warps=4,enable_fp_fusion=False);refs.append(b)
  torch.cuda.synchronize();checks={str(k):torch.equal(a[k],torch.cat([r[k] for r in refs])) for k in range(4)}
  result=dict(m=m,t=t,checks=checks,health=int(a[-1]),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills);print(result,flush=True);results.append(result)
  assert all(checks.values()) and result['health']==0
open('/work/batch-full/attention-checks.json','w').write(json.dumps(results,indent=2))
