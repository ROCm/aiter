from setup import *
entries=[]
for group in ['mhc','qkv','attn']:
 for mode in ['split','fused']:
  s=State();s.run(R,'unfused');fn=getattr(s,group+'_'+mode);args=(R,) if group=='mhc' else ()
  fn(*args);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn(*args)
  for _ in range(5):g.replay()
  entries.append((f'{group}_{mode}',s,g))
rows=[]
for rnd in [1,2]:
 results={}
 for name,s,g in entries:
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   for _ in range(30):g.replay()
   torch.cuda.synchronize()
  path=root/f'groups-r{rnd}-{name}.json';prof.export_chrome_trace(str(path));tr=json.loads(path.read_text());events=sorted([e for e in tr['traceEvents'] if e.get('cat')=='kernel'],key=lambda e:e['ts'])
  nk={'mhc_split':5,'qkv_split':9,'attn_split':3}.get(name,1);assert len(events)==30*nk
  durations=[sum(e['dur'] for e in events[i*nk:(i+1)*nk]) for i in range(30)]
  results[name]=dict(median_us=statistics.median(durations),samples_us=durations,kernels_per_invocation=nk,names=[e['name'] for e in events[:nk]],health=s.check())
 row=dict(round=rnd,results=results);rows.append(row);(root/'group-profile-timings.json').write_text(json.dumps(rows,indent=2));print(rnd,{n:r['median_us'] for n,r in results.items()},flush=True)
