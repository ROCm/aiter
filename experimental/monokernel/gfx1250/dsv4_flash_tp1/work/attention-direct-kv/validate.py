from common import *
rows=[]
for t in [127,128,256,1024]:
 cache=makecache(t);idx=torch.arange(t,device='cuda',dtype=torch.int32);sink=Sink.clone()
 q=Q.clone();kv=KV.clone()
 states={s:AttentionRegion(cache.clone(),idx,sink,splits=s) for s in [1,2,4]}
 graphs={}
 for s,obj in states.items():
  obj(q,kv);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):obj(q,kv)
  graphs[s]=g
 for case in range(32):
  gen=torch.Generator(device='cuda').manual_seed(9000+case)
  q.copy_((torch.randn(q.shape,device='cuda',generator=gen)*[.001,.2,1.,4.][case%4]).bfloat16())
  kv.copy_((torch.randn(kv.shape,device='cuda',generator=gen)*[.1,1.,2.][case%3]).bfloat16())
  base=(torch.randn(cache.shape,device='cuda',generator=gen)*[.01,.2,1.,4.][case%4]).bfloat16();base[-1].copy_(kv)
  ii=torch.randperm(t,device='cuda',generator=gen).int()
  if case%4==1:ii[::3]=-1
  if case%4==2:ii[:32]=-1
  if case%4==3:ii.fill_(-1)
  idx.copy_(ii);sink.copy_(torch.randn(64,device='cuda',generator=gen)*[1.,10.,100.][case%3])
  ref=reference(q,base,idx,sink)
  for s,obj in states.items():
   obj.cache.copy_(base)
   for rep in range(3):
    graphs[s].replay();e=error(obj.out,ref)
    row=dict(t=t,case=case,split=s,replay=rep,error=e,passed=e<.001 and torch.isfinite(obj.out).all().item() and obj.check());rows.append(row)
    assert row['passed'],row
 print('validated length',t,'max error',max(r['error'] for r in rows),flush=True)
 (root/'validation.json').write_text(json.dumps(rows,indent=2))
# Integrated QKV+attention, against validated QKV control and FP64 attention.
chain=[];idx=torch.arange(128,device='cuda',dtype=torch.int32)
for fused in [False,True]:
 obj=QKVAttentionRegion(W,WS,NW,BW,BS,makecache(),idx,Sink,fused=fused)
 obj(X,Cos,Sin);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):obj(X,Cos,Sin)
 for case in range(32):
  gen=torch.Generator(device='cuda').manual_seed(12000+case)
  x=f['inputs'][case%len(f['inputs'])].cuda() if case<12 else (torch.randn(4096,device='cuda',generator=gen)*2.**(case%9-4)).bfloat16()
  X.copy_(x);q,kv=qregion(X,Cos,Sin);g.replay();torch.cuda.synchronize()
  ref=reference(q,obj.attn.cache,idx,Sink)
  row=dict(fused=fused,case=case,q_error=error(obj.q,q),kv_error=error(obj.kv,kv),attention_error=error(obj.attn.out,ref),health=obj.check());chain.append(row)
  assert max(row['q_error'],row['kv_error'],row['attention_error'])<.001 and row['health'],row
 print('chain validated',fused,flush=True)
(root/'chain-validation.json').write_text(json.dumps(chain,indent=2))
