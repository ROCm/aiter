import sys,json
sys.path[:0]=['/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix','/work/atom-actual-microbench/qkv']
import torch
from native import NativeQKV
from qa_batch import qa_batch
n=NativeQKV();f=n.f
reports=[]
for m in [2,4,8]:
 x=n.x.repeat(m,1).contiguous()
 x=(x.float()+torch.arange(m,device='cuda')[:,None]*.0001).bfloat16()
 w=f['W'].cuda() if 'W' in f else None
 # Canonical raw checkpoint layout from the retained paired-forward fixture.
 fixture=torch.load('/work/modelwide/qkv-layer38-pos8.pt',weights_only=True)
 # Use the fixture's own weights and input to compare torch projection/norm
 # separately from serving-module weights; this initial check is compilation only.
 x=fixture['x'].cuda().reshape(1,4096).repeat(m,1)
 w=fixture['qa_weight'].cuda();ws=fixture['qa_scale'].cuda();nw=fixture['norm'].cuda()
 xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
 p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda');q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
 cos=fixture['cos'].cuda().repeat(m,1);sin=fixture['sin'].cuda().repeat(m,1)
 counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
 obj=qa_batch[(256,)](x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,m,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize()
 expected=fixture['our_qquant'].cuda().flatten();expectedkv=fixture['native_kv'].cuda().flatten()
 report=dict(m=m,finite=bool(torch.isfinite(y).all()),health=int(health[0]),quant_neq=int((q.float()-expected.float()).count_nonzero()),kv_rms=float(((kv.float()-expectedkv.float()).square().mean()/expectedkv.float().square().mean()).sqrt()),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 reports.append(report);print(report,flush=True)
 open('/work/concurrency-rollout/qa-batch-check.json','w').write(json.dumps(reports,indent=2))
