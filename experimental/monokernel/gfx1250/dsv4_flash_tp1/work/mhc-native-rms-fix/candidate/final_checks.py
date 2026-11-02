from setup import *
from tdm_tail import candidate
rows=[]
# Longer-context acceptance for the selected no-extra-QA-prefetch variant.
a=State(1024);b=State(1024)
for i in range(64):
 gen=torch.Generator(device='cuda').manual_seed(91100+i)
 r=(torch.randn((4,4096),device='cuda',generator=gen)*2.**(i%15-7)).bfloat16()
 a.run(r,'fused');b.run(r,'opt');torch.cuda.synchronize()
 e={k:err(getattr(b,k),getattr(a,k)) for k in ['mix','raw','x','y','qp','q','kv']};e['out']=err(b.attn.out,a.attn.out)
 assert max(e.values())<.001 and b.check();rows.append(dict(case=i,errors=e))
(root/'long-context-validation.json').write_text(json.dumps(rows,indent=2))
traces=[]
for mode,expected in [('unfused',17),('fused',1),('opt',1),('opt_both',1)]:
 s=State();s.run(R,mode);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):s.run(R,mode)
 for _ in range(5):g.replay()
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  g.replay();torch.cuda.synchronize()
 dest=root/f'final-{mode}-trace.json';prof.export_chrome_trace(str(dest));tr=json.loads(dest.read_text());events=[e for e in tr['traceEvents'] if e.get('cat')=='kernel'];assert len(events)==expected
 traces.append(dict(mode=mode,count=len(events),names=[e['name'] for e in events]))
(root/'final-trace-verification.json').write_text(json.dumps(traces,indent=2));print(traces,flush=True)
isa={}
for mode in ['fused','opt','opt_both']:
 s=(root/f't128_{mode}.amdgcn').read_text();isa[mode]=dict(atomic_add=s.count('global_atomic_add_u32'),tdm='tensor_load_to_lds' in s,wmma='v_wmma' in s,prefetch='global_prefetch_b8' in s,rcp='v_rcp_f32' in s)
 assert all(isa[mode][k] for k in ['tdm','wmma','prefetch'])
assert isa['fused']['atomic_add']==isa['opt']['atomic_add']+3
(root/'final-isa-verification.json').write_text(json.dumps(isa,indent=2));print(isa,flush=True)
