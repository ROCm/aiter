exec(open(__file__.replace('ordered_bench.py','tune_projection.py')).read().split('entries=[]')[0])
from ordered import final_region
Wh=W.bfloat16();Wp=torch.cat([Wh,(W-Wh.float()).bfloat16()],0).contiguous()
variants=[('original',False,0),('prefetch',False,1),('localprefetch',False,2),('packed',True,0),('packedprefetch',True,1)]
entries=[]
for rotate in [1,256]:
 weights=[W.clone() for _ in range(rotate)];packed=[Wp.clone() for _ in range(rotate)]
 for name,pk,pf in variants:
  st=state();ws=packed if pk else weights
  def fn(w=ws[0],st=st,pk=pk,pf=pf):return final_region[(256,)](R,w,S,B,AN,*st,PACKED=pk,PREFETCH=pf,**kw)
  obj=fn();torch.cuda.synchronize()
  for field in [0,2,3]:
   err=((st[field].float()-ref[field].float()).square().mean()/ref[field].float().square().mean()).sqrt().item();assert err<.001,(name,field,err)
  (root/f'ordered-{name}.amdgcn').write_text(obj.asm['amdgcn'])
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for i in range(256):fn(ws[i%rotate])
  entries.append((f'{name}_{rotate}',st,g,dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared),ws))
rows=[]
for rnd in [1,2]:
 samples={n:[] for n,*_ in entries}
 for it in range(40):
  for n,st,g,res,ws in entries[::1 if it%2==0 else -1]:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)*1000/256)
 row=dict(round=rnd,results={n:dict(us=statistics.median(samples[n]),resources=res,health=st[-1].item(),samples=samples[n]) for n,st,g,res,ws in entries});rows.append(row)
 (root/'ordered-timings.json').write_text(json.dumps(rows,indent=2));print({n:r['us'] for n,r in row['results'].items()},flush=True)
