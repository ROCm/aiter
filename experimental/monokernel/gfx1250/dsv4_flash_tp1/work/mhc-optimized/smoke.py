from setup import *
a=State();b=State();c=State()
a.run(R,'unfused');torch.cuda.synchronize();print('unfused done',flush=True)
b.run(R,'fused');torch.cuda.synchronize();print('fused done',flush=True)
c.run(R,'three_regions');torch.cuda.synchronize()
for n,s in [('fused',b),('three_regions',c)]:
 errors={k:err(getattr(s,k),getattr(a,k)) for k in ['mix','raw','x','y','qp','q','kv']};errors['out']=err(s.attn.out,a.attn.out)
 print(n,errors,s.check(),flush=True);assert max(errors.values())<.001 and s.check()
ref=attn_ref(b.q,b.attn.cache,b.attn.indices,Sink);print('attn ref',err(b.attn.out,ref),flush=True)
k=b.obj;print('resources',k.n_regs,k.n_spills,k.metadata.shared,flush=True)
(root/'full_prefix.amdgcn').write_text(k.asm['amdgcn'])
