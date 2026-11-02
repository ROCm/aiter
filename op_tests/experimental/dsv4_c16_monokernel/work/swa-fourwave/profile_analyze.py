"""Analyze per-wave intervals without summing overlapping CTA waits."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent/os.getenv('PROFILE_DIR','profiles')
names={1:'mHC pre GEMM',2:'mHC reduce/apply/norm',3:'input FP8 quant',4:'QA/KV',5:'Q norm/quant',6:'QB',7:'QK norm/RoPE/cache',8:'split attention',9:'attention reduce',10:'inverse RoPE/quant',11:'OA',12:'OB input quant',13:'OB',14:'post/pre mHC GEMM',15:'FFN mHC reduce/apply',16:'shared input quant',17:'shared GU',18:'shared SiLU',19:'shared hidden quant',20:'shared GD',21:'router',22:'HERD clear',23:'topk',24:'keep6/renorm',25:'routing metadata',26:'routed input quant',27:'routed GU',28:'routed GD',29:'route reduction',30:'output add'}
rows=[];runs=[]
for path in sorted(ROOT.glob('swa4_profile_*.json')):
 meta=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'));x=z['traces']/float(z['ticks_per_us']);events=list(z['events']);runs.append({'file':path.name,**meta})
 def col(label):return x[:,:,:,events.index(label)]
 for stage in meta['stages']:
  start=col(f'{stage}:stage_start');end=col(f'{stage}:stage_end');active=start>0
  wait=np.zeros_like(start);acquire=wait.copy();publish=wait.copy();local=wait.copy()
  for label in events:
   if not label.startswith(f'{stage}:') or not label.endswith(':begin'):continue
   a=col(label);b=col(label.replace(':begin',':end'));mask=a>0
   if ':get' in label:
    r=col(label.replace(':begin',':ready'));wait+=np.where(mask,r-a,0);acquire+=np.where(mask,b-r,0)
   elif ':publish' in label:publish+=np.where(mask,b-a,0)
   elif ':local' in label:local+=np.where(mask,b-a,0)
  compute=np.where(active,end-start-wait-acquire-publish-local,0)
  assert compute[active].min()>-.01,(stage,compute[active].min())
  # Select the wave with the longest compute interval in each CTA. Use that
  # SAME wave for every duration, so inactive-wave convergence cannot be
  # mistaken for publication cost or double-counted with active computation.
  # Publication includes outstanding-operation drain and CTA convergence, NOT
  # just the flag store. Idle waves can wait here for computing waves.
  valid=active.any(axis=2)
  owner=compute.argmax(axis=2)
  select=lambda a:np.take_along_axis(a,owner[:,:,None],axis=2)[:,:,0]
  row={'stage':stage,'name':names[stage],'active_ctas':int(valid[0].sum()),'completion_tail_us':float(np.median(end.max(axis=(1,2))))}
  for label,a in [('compute',compute),('dependency_wait',wait),('acquire',acquire),('publish_drain',publish),('local_drain',local),('elapsed',end-start)]:
   vals=select(a)[valid];row[label+'_median_us']=float(np.median(vals));row[label+'_p95_us']=float(np.percentile(vals,95));row[label+'_max_median_us']=float(np.median(np.where(valid,select(a),0).max(axis=1)))
  if stage in (2,15) and int(valid[0].sum())==32:
   row['roles']={}
   for role,lo,hi in [('apply_norm',128,144),('post_sinkhorn',144,160)]:
    mask=valid.copy();mask[:,:lo]=False;mask[:,hi:]=False
    rr={'ctas':int(mask[0].sum())}
    for label,a in [('compute',compute),('dependency_wait',wait),('acquire',acquire),('publish_drain',publish),('local_drain',local),('elapsed',end-start)]:
     vals=select(a)[mask];rr[label+'_median_us']=float(np.median(vals));rr[label+'_p95_us']=float(np.percentile(vals,95))
    row['roles'][role]=rr
  row['all_wave_publish_p95_us']=float(np.percentile(publish[active],95))
  # Per-sample last finishing wave: a local critical endpoint, not a reconstructed
  # full-layer critical path. Never sum this row over independent branches.
  last=end.reshape(len(x),-1).argmax(axis=1)
  for label,a in [('compute',compute),('wait',wait),('acquire',acquire),('publish',publish),('local',local)]:
   row['last_finisher_'+label+'_us']=float(np.median(a.reshape(len(x),-1)[np.arange(len(x)),last]))
  rows.append(row)
rows.sort(key=lambda r:r['stage'])
(ROOT/'SUMMARY.json').write_text(json.dumps({'runs':runs,'stages':rows},indent=2))
lines=['Per-stage durations for the longest-compute wave in each CTA, microseconds; median over samples/participating CTAs.',
       'compute excludes measured wait/acquire/publication/local drains. Publication includes producer completion and CTA convergence.',
       'Stage tails are timestamps relative to earliest CTA entry, measured in separate profiling batches; NOT additive.',
       'stage operation                 CTAs compute p95comp wait   acq   pub   local  tail']
for r in rows:
 for role,v in r.get('roles',{'':{}}).items():
  q=dict(r,**v)
  if role:q.update(name=('mHC ' if r['stage']==2 else 'FFN mHC ')+role,active_ctas=v['ctas'])
  lines.append(f"{q['stage']:2} {q['name']:26} {q['active_ctas']:3} {q['compute_median_us']:6.2f} {q['compute_p95_us']:7.2f} {q['dependency_wait_median_us']:6.2f} {q['acquire_median_us']:5.2f} {q['publish_drain_median_us']:5.2f} {q['local_drain_median_us']:6.2f} {q['completion_tail_us']:6.2f}")
lines+=['','Instrumentation overhead, interleaved graph16 medians:']
for r in runs:
 a=r['median_us'];lines.append(f"{r['stages']}: {a['baseline']:.2f} -> {a['instrumented']:.2f} us ({(a['instrumented']/a['baseline']-1)*100:.1f}%), active kernel span {r['profile_kernel_span_us']:.2f} us")
(ROOT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
