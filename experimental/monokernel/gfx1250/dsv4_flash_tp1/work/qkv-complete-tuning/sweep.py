from harness import *
out=root/'sweep';out.mkdir(exist_ok=True)
r=reference(X);runs=[];checks=[]
for au in [1,2]:
 for bn,waves,u in [(64,4,1),(64,2,1),(64,4,2),(64,2,2),(64,4,4),(64,2,4),(64,4,8),(64,2,8),(128,4,2),(128,4,4),(128,4,8)]:
  s=State();name=f'qa_u{au}_qb_n{bn}_w{waves}_u{u}'
  def run(s=s,au=au,bn=bn,waves=waves,u=u):
   a=project_fused[(192,)](X,s.xs,W,WS,s.p,NW,s.acc,s.count,s.y,s.lp,s.ls,Cos,Sin,s.kv,INLINE=True,TAIL=True,**dict(cbase,UNROLL=au),**kw)
   b=qb_kernel[(32768//bn,)](s.lp,s.ls,BW,BS,s.qp,s.qc,Cos,Sin,s.q,TAIL=True,BN=bn,WAVES=waves,U=u,num_warps=waves,enable_fp_fusion=False)
   return a,b
  try:
   a,b=run();torch.cuda.synchronize()
  except Exception as e:
   print('COMPILE_FAILURE',name,str(e),flush=True);continue
  metrics={k:metric(getattr(s,k),r[k]) for k in ['q','kv']}
  assert all(m['relative_rms']<.001 for m in metrics.values()),(name,metrics)
  res={k:dict(vgpr=o.n_regs,spills=o.n_spills,lds=o.metadata.shared) for k,o in [('qa',a),('qb',b)]}
  for k,o in [('qa',a),('qb',b)]:(out/(name+'-'+k+'.amdgcn')).write_text(o.asm['amdgcn'])
  checks.append(dict(name=name,metrics=metrics,resources=res))
  cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):
   for _ in range(128):run()
  for _ in range(3):cg.replay()
  torch.cuda.synchronize();runs.append((name,s,cg));print('READY',name,res,flush=True)
(out/'checks.json').write_text(json.dumps(checks,indent=2))
results=[]
for rnd in [1,2]:
 vals={n:[] for n,_,_ in runs}
 for sample in range(30):
  for n,s,cg in (runs if sample%2==0 else reversed(runs)):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();cg.replay();b.record();b.synchronize();vals[n].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={n:dict(median_us=statistics.median(v),samples_us=v) for n,v in vals.items()});results.append(row)
 print('ROUND',rnd,sorted((v['median_us'],n) for n,v in row['results'].items()),flush=True)
 (out/'timings.json').write_text(json.dumps(results,indent=2))
