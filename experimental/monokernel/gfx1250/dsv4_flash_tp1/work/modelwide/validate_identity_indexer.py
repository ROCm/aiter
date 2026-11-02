import sys,json
sys.path[:0]=['/app/ATOM','/app/aiter','/work/modelwide']
import torch,aiter
from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack
from identity_indexer import identity_csa
checks=[]
for length in [0,1,2,255,256,257,511,512]:
 logits=torch.randn((1,1024),device='cuda')
 lens=torch.tensor([length],device='cuda',dtype=torch.int32)
 top=torch.empty((1,512),device='cuda',dtype=torch.int32);out=torch.empty_like(top)
 aiter.top_k_per_row_decode(logits,1,lens,top,1,1024,1,k=512)
 pos=torch.tensor([max(length*4-1,0)],device='cuda',dtype=torch.int64)
 skip=min(pos.item()+1,128);ptr=torch.tensor([0,length+skip],device='cuda',dtype=torch.int32)
 blocks=torch.randperm(16,device='cuda',dtype=torch.int32).view(1,16)
 idx=torch.full((640,),-7,device='cuda',dtype=torch.int32);idx2=idx.clone();bid=torch.zeros(1,device='cuda',dtype=torch.int32);health=bid.clone()
 csa_translate_pack(top,blocks,pos,ptr,bid,None,idx,envelope_rows=12345,csa_block_capacity=64,window_size=128)
 identity_csa[(4,)](pos,ptr,blocks,idx2,out,health,12345,num_warps=4)
 checks.append(dict(length=length,topk_exact=torch.equal(top,out),indices_exact=torch.equal(idx,idx2),health=health.item()))
print(checks,flush=True)
open('/work/modelwide/identity-indexer-validation.json','w').write(json.dumps(checks,indent=2))
assert all(c['topk_exact'] and c['indices_exact'] and c['health']==0 for c in checks)
