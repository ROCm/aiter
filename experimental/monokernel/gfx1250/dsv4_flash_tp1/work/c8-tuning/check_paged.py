from common import *
from attention_paged import attention_paged
from attention_defs import scratch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
res=[]
for seed in range(8):
 torch.manual_seed(seed);m=8;pages=8192
 lengths=[0,1,16,128,136,384,512,640] if seed%2==0 else [640,136,128,0,0,144,384,1]
 q=torch.randn((m,64,512),device='cuda',dtype=torch.bfloat16)*.1;kv=torch.randn((m,512),device='cuda',dtype=torch.bfloat16)*.1
 cache=torch.randn((pages,512),device='cuda',dtype=torch.bfloat16);refcache=cache.clone();sink=torch.randn(64,device='cuda')
 ptr=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32);indices=torch.randperm(pages,device='cuda',dtype=torch.int32)[:sum(lengths)].contiguous();dest=torch.full((m,),-1,device='cuda',dtype=torch.int32)
 for row,n in enumerate(lengths):
  if n:dest[row]=indices[int(ptr[row])+n-1];refcache[dest[row].long()]=kv[row]
 a=scratch(m,640,8)
 # Poison the inactive-split scratch: inactive segments must be masked, not zeroed.
 for t in a[:3]:t.fill_(float('nan'))
 ref=sparse_attn_v4_paged_decode(q,refcache,indices,ptr,sink,512**-.5)
 for _ in range(3):attention_paged[(256,)](q,kv,cache,indices,ptr,sink,*a,dest,pages,m,8,**kw)
 torch.cuda.synchronize();check=err(a[3],ref);ref8=pa_decode_sparse(q,refcache,indices,ptr,sink,512**-.5,kv_splits=8,has_invalid=False);torch.cuda.synchronize()
 check.update(seed=seed,health=int(a[-1]),native_same_splits=err(a[3],ref8));res.append(check)
 assert check['finite'] and check['native_same_splits']['relative_rms']<.001 and check['health']==0,check
(ROOT/'paged-validation.json').write_text(json.dumps(res,indent=2));print('PASS',res,flush=True)
