import sys,json,torch
sys.path[:0]=['/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix','/work/atom-actual-microbench/qkv']
from qkv_batch import qkv_batch
from native import NativeQKV
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import qk_norm_rope_maybe_quant
f=torch.load('/work/modelwide/qkv-layer38-pos8.pt',weights_only=True)
n=NativeQKV('/work/concurrency-rollout/qa-native-fixture.pt')
results=[]
for m in [1,2,4,8]:
 def e(shape,dt=torch.float32):return torch.empty(shape,device='cuda',dtype=dt)
 x=(f['x'].cuda().reshape(1,4096).repeat(m,1).float()*torch.randn((m,4096),device='cuda')*.3).bfloat16();w=f['qa_weight'].cuda();ws=f['qa_scale'].cuda();nw=f['norm'].cuda();qb=f['qb_weight'].cuda();qbs=f['qb_scale'].cuda()
 xq=e((m,4096),torch.float8_e4m3fn);xs=e((m,32),torch.uint8);p=e((16,m,1536));raw=e((m,1536),torch.bfloat16);qr=e((m,1024),torch.float8_e4m3fn);qs=e((m,8),torch.uint8);kv=e((m,512),torch.bfloat16);qbr=e((m,32768),torch.bfloat16);qo=e((m,64,512),torch.bfloat16)
 pos=torch.full((m,),8,device='cuda',dtype=torch.int64);cos=n.cos[8].reshape(1,32).repeat(m,1);sin=n.sin[8].reshape(1,32).repeat(m,1)
 counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
 cache=torch.full((64,512),-17,device='cuda',dtype=torch.bfloat16);dest=torch.arange(m,device='cuda',dtype=torch.int32)+10;bid=torch.arange(m,device='cuda',dtype=torch.int32)
 def call():return qkv_batch[(256,)](x,xq,xs,w,ws,p,nw,raw,qr,qs,cos,sin,kv,counter,health,qb,qbs,qbr,qo,m,Cache=cache,Dest=dest,BatchId=bid,CacheRows=64,Positions=pos,num_warps=4,enable_fp_fusion=False)
 ref=n.qa(x);rq,rs=n.norm(ref[:,:1024]);rb=n.qb(rq,x_scale=rs);out=qk_norm_rope_maybe_quant(rb,ref[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6)
 obj=call();torch.cuda.synchronize()
 qerr=float(((qo.float()-out.q_sa.float()).square().mean()/out.q_sa.float().square().mean()).sqrt());kerr=float(((kv.float()-out.kv.float()).square().mean()/out.kv.float().square().mean()).sqrt())
 assert torch.equal(cache[dest.long()],kv)
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):call()
 cases=[('valid',8,0,10),('negative_position',-1,0,10),('inactive',8,-1,10),('negative_destination',8,0,-1),('past_end',8,0,64)]
 for name,position,batch,destination in cases:
  pos.fill_(position);bid.fill_(batch);dest.copy_(torch.arange(m,device='cuda',dtype=torch.int32)+destination if name=='valid' else torch.full((m,),destination,device='cuda',dtype=torch.int32));cache.fill_(-17);g.replay();torch.cuda.synchronize()
  if name=='valid':assert torch.equal(cache[dest.long()],kv)
  else:assert bool((cache==-17).all()),name
 assert int(health[0])==0
 results.append(dict(m=m,q_relative_rms=qerr,kv_relative_rms=kerr,qa_equal=bool(torch.equal(raw,ref)),qr_equal=bool(torch.equal(qr.view(torch.uint8),rq.view(torch.uint8))),qb_equal=bool(torch.equal(qbr,rb)),cache_contract_cases=[c[0] for c in cases],health=int(health[0]),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills));print(results[-1],flush=True)
open('/work/resume-prep/random-row-contract-checks.json','w').write(json.dumps(results,indent=2)+'\n')
