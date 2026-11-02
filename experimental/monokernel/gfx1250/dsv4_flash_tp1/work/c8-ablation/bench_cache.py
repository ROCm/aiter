from common import *
from cache_stage import stage
from scratch_batch import BatchState
for t,n in [(128,128),(144,136),(640,384),(640,512),(640,640)]:
 s=BatchState(8,t);a=tuple(s.a);pos=torch.full((8,),1024,device='cuda',dtype=torch.int32);cs=torch.randn((2048,32),device='cuda',dtype=torch.bfloat16);sn=torch.randn_like(cs)
 cache=torch.randn((8*t,512),device='cuda',dtype=torch.bfloat16);idx=torch.arange(8*n,device='cuda',dtype=torch.int32);ptr=torch.arange(0,9*n,n,device='cuda',dtype=torch.int32)
 def fused():return stage[(256,)](a,pos,cs,sn,cache,idx,ptr,8,t,**kw)
 bench('cache_stage_'+str(t)+'_'+str(n),{'fused':fused},dict(note='extra monokernel compact-cache copy, index and rotary preparation; native attention reads paged cache directly'))
