from setup import *
a=State();b=State();rows=[]
inputs=[x.cuda().contiguous() for x in f2['inputs']]
for i in range(512):
 gen=torch.Generator(device='cuda').manual_seed(71100+i)
 inputs.append((torch.randn((4,4096),device='cuda',generator=gen)*2.**(i%15-7)).bfloat16())
for i,r in enumerate(inputs):
 a.run(r,'fused');b.run(r,'opt');torch.cuda.synchronize()
 e={k:err(getattr(b,k),getattr(a,k)) for k in ['mix','raw','x','y','qp','q','kv']};e['out']=err(b.attn.out,a.attn.out)
 assert max(e.values())<.001 and b.check(),(i,e)
 rows.append(dict(case=i,errors=e));
 if i%100==0:print('validated',i,flush=True)
(root/'prefix-validation.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows),flush=True)
