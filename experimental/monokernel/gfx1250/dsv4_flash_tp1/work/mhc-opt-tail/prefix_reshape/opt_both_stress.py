from setup import *
a=State();b=State();xs=[x.cuda().contiguous() for x in f2['inputs']]
for i in range(23):
 gen=torch.Generator(device='cuda').manual_seed(81200+i);xs.append((torch.randn((4,4096),device='cuda',generator=gen)*2**(i%15-7)).bfloat16())
keys=['mix','raw','x','q','kv'];refs={k:[] for k in keys+['out','cache']}
for r in xs:
 a.run(r,'fused')
 for k in keys:refs[k].append(getattr(a,k).clone())
 refs['out'].append(a.attn.out.clone());refs['cache'].append(a.attn.cache.clone())
refs={k:torch.stack(v) for k,v in refs.items()};outs={k:torch.empty_like(v) for k,v in refs.items()};r=R.clone();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
 for i,x in enumerate(xs):
  r.copy_(x);b.mix.fill_(float('nan'));b.raw.fill_(float('nan'));b.x.fill_(float('nan'));b.attn.cache[b.attn.write_slot].fill_(123)
  b.run(r,'opt_both')
  for k in keys:outs[k][i].copy_(getattr(b,k))
  outs['out'][i].copy_(b.attn.out);outs['cache'][i].copy_(b.attn.cache)
mx={k:0 for k in refs}
for j in range(320):
 g.replay()
 for k in refs:
  x=outs[k].float().reshape(32,-1);y=refs[k].float().reshape(32,-1)
  e=((x-y).square().mean(1)/y.square().mean(1).clamp_min(1e-30)).sqrt().max().item();mx[k]=max(mx[k],e);assert e<.001,(j,k,e)
 assert b.check() and torch.equal(outs['cache'],refs['cache'])
 if j%80==0:print(j,mx,flush=True)
(root/'opt-both-stress.json').write_text(json.dumps({'checks':10240,'max_errors':mx,'health':b.health.item(),'cache_exact':True},indent=2));print('passed',mx,flush=True)
