from common import *
results=[]
qs=[];ks=[];cs=[];ids=[];sinks=[];refs=[]
for i in range(32):
 gen=torch.Generator(device='cuda').manual_seed(20100+i)
 q=(torch.randn((64,512),device='cuda',generator=gen)*[.1,1.,3.][i%3]).bfloat16();kv=torch.randn(512,device='cuda',generator=gen).bfloat16();c=(torch.randn((128,512),device='cuda',generator=gen)*.5).bfloat16();c[-1].copy_(kv)
 idx=torch.randperm(128,device='cuda',generator=gen).int()
 if i%3==0:idx[::2]=-1
 if i==0:idx.fill_(-1)
 sn=torch.randn(64,device='cuda',generator=gen)*3
 qs.append(q);ks.append(kv);cs.append(c);ids.append(idx);sinks.append(sn);refs.append(reference(q,c,idx,sn))
ref=torch.stack(refs).float()
for c in cs:c[-1].fill_(123) # Deliberately stale cache: direct KV must supply the new row.
for split in [1,2,4]:
 q=qs[0].clone();kv=ks[0].clone();idx=ids[0].clone();sn=sinks[0].clone()
 obj=AttentionRegion(cs[0].clone(),idx,sn,splits=split);obj(q,kv);torch.cuda.synchronize()
 outputs=torch.empty_like(ref,dtype=torch.bfloat16);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for i in range(32):
   q.copy_(qs[i]);kv.copy_(ks[i]);obj.cache.copy_(cs[i]);idx.copy_(ids[i]);sn.copy_(sinks[i]);obj(q,kv);outputs[i].copy_(obj.out)
 maxerr=0
 for rep in range(320):
  g.replay();errs=((outputs.float()-ref).square().mean((1,2))/ref.square().mean((1,2)).clamp_min(1e-30)).sqrt();v=errs.max().item();maxerr=max(maxerr,v)
  assert torch.isfinite(errs).all() and v<.001 and obj.check(),(split,rep,v)
 row=dict(kind='attention',split=split,checked_invocations=10240,max_error=maxerr,passed=True);results.append(row);print(row,flush=True)
 (root/'stress.json').write_text(json.dumps(results,indent=2))
# Chain reference uses the validated QKV kernel, FP64 attention, changing X/cache.
xs=[];caches=[];chain_refs=[]
idx=torch.arange(128,device='cuda',dtype=torch.int32)
for i in range(32):
 gen=torch.Generator(device='cuda').manual_seed(30100+i)
 x=(torch.randn(4096,device='cuda',generator=gen)*2.**(i%9-4)).bfloat16();c=(torch.randn((128,512),device='cuda',generator=gen)*.2).bfloat16()
 q,kv=qregion(x,Cos,Sin);c[-1].copy_(kv);chain_refs.append(reference(q,c,idx,Sink));xs.append(x);caches.append(c)
r=torch.stack(chain_refs).float()
for c in caches:c[-1].fill_(123)
for fused,split in [(False,1),(True,1),(True,4)]:
 obj=QKVAttentionRegion(W,WS,NW,BW,BS,caches[0].clone(),idx,Sink,splits=split,fused=fused);obj(X,Cos,Sin);torch.cuda.synchronize()
 outputs=torch.empty_like(r,dtype=torch.bfloat16);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for i in range(32):
   X.copy_(xs[i]);obj.attn.cache.copy_(caches[i]);obj(X,Cos,Sin);outputs[i].copy_(obj.attn.out)
 maxerr=0
 for rep in range(320):
  g.replay();errs=((outputs.float()-r).square().mean((1,2))/r.square().mean((1,2)).clamp_min(1e-30)).sqrt();v=errs.max().item();maxerr=max(maxerr,v)
  assert torch.isfinite(errs).all() and v<.001 and obj.check(),(fused,split,rep,v)
 row=dict(kind='chain',fused=fused,split=split,checked_invocations=10240,max_error=maxerr,passed=True);results.append(row);print(row,flush=True)
 (root/'stress.json').write_text(json.dumps(results,indent=2))
