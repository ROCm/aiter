import harness as h
from harness import *
from variants import CONFIGS,run,State
out=root/'scratch-benchmark';out.mkdir(exist_ok=True)
configs=[(n,0) for n in ['two_scratch_fp32', 'two_scratch_fp64', 'grid_scratch_fp32', 'grid_scratch_fp64', 'qready_scratch_fp32', 'qready_scratch_fp64', 'grid_scratch_rt_fp32']]
resources={}
for mode,bn in configs:
 s=State();ret=run(s,mode)
 qa,o=ret if isinstance(ret,tuple) else (ret,ret)
 resources[mode]={}
 for key,obj in [('qa',qa),('qb',o)]:
  resources[mode][key]=dict(vgpr=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,private=re.findall(r'\.amdhsa_private_segment_fixed_size (\d+)',obj.asm['amdgcn']))
  (out/f'{mode}-{key}.amdgcn').write_text(obj.asm['amdgcn']);(out/f'{mode}-{key}.llir').write_text(obj.asm['llir'])
(out/'resources.json').write_text(json.dumps(resources,indent=2))
results=[]
for copies in [1,16]:
 wa=W.view(torch.uint8)[None].repeat(copies,1,1).view(W.dtype);wb=BW.view(torch.uint8)[None].repeat(copies,1,1).view(BW.dtype)
 vs=[(wa[i],wb[i]) for i in range(copies)]
 graphs=[]
 for mode,bn in configs:
  s=State();h.W,h.BW=vs[0];run(s,mode);torch.cuda.synchronize()
  cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):
   for i in range(128):h.W,h.BW=vs[i%copies];run(s,mode)
  for _ in range(5):cg.replay()
  graphs.append((mode,s,cg))
 torch.cuda.synchronize()
 for rnd in [1,2]:
  samples={m:[] for m,_,_ in graphs}
  for sample in range(40):
   for m,s,cg in (graphs if sample%2==0 else reversed(graphs)):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();cg.replay();b.record();b.synchronize();samples[m].append(a.elapsed_time(b)*1000/128)
  row=dict(copies=copies,weight_bytes=(wa.numel()+wb.numel()),round=rnd,results={m:dict(median_us=statistics.median(samples[m]),samples_us=samples[m],state=s.check_state()) for m,s,cg in graphs})
  results.append(row);(out/'timings.json').write_text(json.dumps(results,indent=2));print(copies,rnd,{m:v['median_us'] for m,v in row['results'].items()},flush=True)
 del graphs,vs,wa,wb
h.W=W;h.BW=BW
(out/'environment.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),properties=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,triton=__import__('triton').__version__),indent=2))
