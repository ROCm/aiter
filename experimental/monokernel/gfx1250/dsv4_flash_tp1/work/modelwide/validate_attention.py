import sys,json,os
sys.path[:0]=['/work/modelwide','/work/full-attention-native/prefix','/app/ATOM','/app/aiter']
import torch
if os.getenv("MATCHED") == "1":
 from attention_matched import attention_kernel
elif os.getenv("NATIVE_PROB") == "1":
 from attention_nativeprob import attention_kernel
else:
 from attention_extended import attention_kernel
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
splits=int(os.getenv("SPLITS","4"));bk=int(os.getenv("BK","32"))
checks=[]
for n in [1,31,32,33,65,127,128,129,136,144,257,384,513,640]:
 torch.manual_seed(n)
 q=torch.randn((64,512),device='cuda',dtype=torch.bfloat16)*.1
 cache=torch.randn((640,512),device='cuda',dtype=torch.bfloat16)*.1
 kv=torch.randn((512,),device='cuda',dtype=torch.bfloat16)*.1
 idx=torch.arange(n,device='cuda',dtype=torch.int32);idx[-1]=639
 idx=torch.cat([idx,torch.full((16384-n,),-1,device='cuda',dtype=torch.int32)])
 ptr=torch.tensor([0,n],device='cuda',dtype=torch.int32)
 sink=torch.randn(64,device='cuda');m=torch.empty((splits,64),device='cuda');l=torch.empty_like(m);acc=torch.empty((splits,64,512),device='cuda');out=torch.empty_like(q)
 count=torch.zeros(4,device='cuda',dtype=torch.int32);counter=torch.zeros(1,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32)
 cache[639]=kv
 ref=sparse_attn_v4_paged_decode(q[None],cache,idx,ptr,sink,512**-.5).view_as(q)
 for _ in range(2):attention_kernel[(256,)](q,kv,cache,idx,ptr,sink,m,l,acc,out,count,counter,health,T=n,PAGES=640,WRITE_SLOT=639,SPLITS=splits,GRID=256,BH=16,BK=bk,VECTOR=False,DIRECT=True,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize()
 diff=(out.float()-ref.float());checks.append(dict(length=n,rms=(diff.square().mean()/ref.float().square().mean()).sqrt().item(),finite=torch.isfinite(out).all().item(),health=health.item(),counts_zero=not count.any().item()))
 print(checks[-1],flush=True)
open('/work/modelwide/attention-validation'+os.getenv('TEST_TAG','')+('-nativeprob' if os.getenv('NATIVE_PROB')=='1' else '')+'.json','w').write(json.dumps(checks,indent=2))
