import sys,json,torch
sys.path[:0]=['/work/concurrency-rollout','/work/modelwide','/work/full-attention-native/prefix','/work/atom-actual-microbench/qkv']
from native import NativeQKV
from qa_batch import qa_batch
f=torch.load('/work/modelwide/qkv-layer38-pos8.pt',weights_only=True)
fixture=dict(W_bytes=f['qa_weight'].view(torch.uint8),WS=f['qa_scale'],QB_bytes=f['qb_weight'].view(torch.uint8),QBS=f['qb_scale'],NW=f['norm'],live=f['x'])
torch.save(fixture,'/work/concurrency-rollout/qa-native-fixture.pt')
n=NativeQKV('/work/concurrency-rollout/qa-native-fixture.pt')
from triton.testing import do_bench_cudagraph
reports=[]
for m in [2,4,8]:
 x=f['x'].cuda().reshape(1,4096).repeat(m,1);w=f['qa_weight'].cuda();ws=f['qa_scale'].cuda();nw=f['norm'].cuda();qb=f['qb_weight'].cuda()
 xq=torch.empty((m,4096),device='cuda',dtype=torch.float8_e4m3fn);xs=torch.empty((m,32),device='cuda',dtype=torch.uint8)
 p=torch.empty((16,m,1536),device='cuda');y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16);q=torch.empty((m,1024),device='cuda',dtype=torch.float8_e4m3fn);s=torch.empty((m,8),device='cuda',dtype=torch.uint8);kv=torch.empty((m,512),device='cuda',dtype=torch.bfloat16)
 cos=f['cos'].cuda().repeat(m,1);sin=f['sin'].cuda().repeat(m,1);counter=torch.zeros(1,device='cuda',dtype=torch.int32);health=torch.zeros_like(counter)
 def fused():return qa_batch[(256,)](x,xq,xs,w,ws,p,nw,y,q,s,cos,sin,kv,counter,health,m,QBW=qb,num_warps=4,enable_fp_fusion=False)
 def native():
  a=n.qa(x);return a,n.norm(a[:,:1024])
 checks=[]
 gen=torch.Generator(device='cuda').manual_seed(1776)
 for seed in range(32):
  if seed:x.copy_((f['x'].cuda().reshape(1,4096)+torch.randn((m,4096),device='cuda',generator=gen)*.01).bfloat16())
  ref,(rq,rs)=native();obj=fused();torch.cuda.synchronize()
  checks.append(dict(qa_neq=int((y!=ref).sum()),quant_neq=int((q.float()!=rq.float()).sum()),scale_neq=int((s!=rs.view(torch.uint8)).sum()),health=int(health[0])))
 timings=[]
 for rnd in range(2):timings.append({k:do_bench_cudagraph(fn,rep=100)*1000 for k,fn in [('native_qa_norm_only',native),('fused_qa_norm_plus_kv_rope',fused)]})
 row=dict(m=m,checks=checks,timings=timings,vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 reports.append(row);print(m,'mismatches',sum(c['quant_neq'] for c in checks),'qa',sum(c['qa_neq'] for c in checks),timings[-1],flush=True)
 open('/work/concurrency-rollout/qa-batch-native-results.json','w').write(json.dumps(reports,indent=2))
