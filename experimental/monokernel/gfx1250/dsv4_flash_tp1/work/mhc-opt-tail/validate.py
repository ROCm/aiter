from setup import *
rows=[]
for t in [128,1024]:
 a=State(t);b=State(t);c=State(t)
 inputs=[x.cuda().contiguous() for x in f2['inputs']]
 for i in range(23):
  gen=torch.Generator(device='cuda').manual_seed(50100+i)
  inputs.append((torch.randn((4,4096),device='cuda',generator=gen)*2.**(i%7-3)).bfloat16())
 for i,r in enumerate(inputs):
  a.run(r,'unfused');ref=a.attn.out.clone();b.run(r,'fused');c.run(r,'three_regions');torch.cuda.synchronize()
  errors={name:{k:err(getattr(s,k),getattr(a,k)) for k in ['mix','raw','x','y','qp','q','kv']} for name,s in [('fused',b),('three_regions',c)]}
  for name,s in [('fused',b),('three_regions',c)]:
   errors[name]['out']=err(s.attn.out,ref);assert max(errors[name].values())<.001 and s.check(),(t,i,name,errors[name])
   assert torch.equal(s.attn.cache,a.attn.cache)
  independent=err(b.attn.out,attn_ref(b.q,b.attn.cache,b.attn.indices,Sink));assert independent<.001
  # Independent mHC learned projection before Sinkhorn; recover raw coefficients
  # with standalone WMMA producer/reducer, compare to FP64 matvec.
  mhc_stage[(256,)](r,HAF,HAS,HAB,AN,a.mix,a.temp,a.raw,a.x,STAGE=0,**kw)
  mhc_stage[(24,)](r,HAF,HAS,HAB,AN,a.mix,a.temp,a.raw,a.x,STAGE=1,**kw)
  rd=r.double().flatten();z=(HAF.double().reshape(24,16384)@rd)*torch.rsqrt(rd.square().mean()+1e-6)
  scales=torch.cat([HAS[0].expand(4),HAS[1].expand(4),HAS[2].expand(16)]).double();mref=z*scales+HAB.double().flatten()
  mixerr=err(a.mix,mref);assert mixerr<.001,(i,mixerr)
  rows.append(dict(tokens=t,case=i,errors=errors,attention_reference_error=independent,mhc_projection_reference_error=mixerr))
 print('validated',t,flush=True)
(root/'validation.json').write_text(json.dumps(rows,indent=2))
# Changing-input graphs with stale cache rows; compare full prefix against split.
a=State();b=State();xs=[x.cuda().contiguous() for x in f2['inputs']]
refs=[]
for x in xs:refs.append(a.run(x,'unfused').clone())
refs=torch.stack(refs).float();r=R.clone();outputs=torch.empty_like(refs,dtype=torch.bfloat16);g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
 for i,x in enumerate(xs):
  r.copy_(x);b.attn.cache[-1].fill_(123);b.run(r,'fused');outputs[i].copy_(b.attn.out)
mx=0
for i in range(1024):
 g.replay();v=((outputs.float()-refs).square().mean((1,2))/refs.square().mean((1,2)).clamp_min(1e-30)).sqrt().max().item();mx=max(mx,v)
 assert v<.001 and b.check(),(i,v)
(root/'stress.json').write_text(json.dumps(dict(checked_invocations=1024*len(xs),max_error=mx,passed=True),indent=2));print('stress passed',mx,flush=True)
