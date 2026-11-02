import json,collections,statistics
from pathlib import Path
r=Path('/work/rollout-c1248') if Path('/work/rollout-c1248').exists() else Path('/home/vigokhal/dsv4-monokernel/rollout-c1248')
events=collections.defaultdict(list)
for p in r.glob('prefill-*.jsonl'):
 for line in p.read_text().splitlines():
  d=json.loads(line);events[d['tag']].append(d)
rows=[]
for name in ['native','previous','selected','replica']:
 for kind in ['', 'prefill-only-']:
  for c in [1,2,4,8]:
   tag=f'{kind}{name}-c{c}-round2';p=r/(tag+'.json')
   if not p.exists() or tag not in events:continue
   d=json.loads(p.read_text());ev=sorted(events[tag],key=lambda e:e['wall_start'])[1:] # exclude benchmark's single test request
   weighted=[];cpu=[];shapes=collections.Counter()
   for e in ev:
    n=sum(t==1024 for t in e.get('num_scheduled_tokens',[1024]*e['scheduled_bs']))
    weighted.extend([e['gpu_ms']]*n);cpu.extend([e['cpu_enqueue_ms']]*n);shapes[(e['scheduled_bs'],e['tokens'])]+=1
   if not weighted:continue
   rows.append(dict(configuration=name,c=c,kind='1k/1' if kind else '1k/1k',ttft_median_ms=d['median_ttft_ms'],ttft_mean_ms=d['mean_ttft_ms'],prefill_gpu_median_ms=statistics.median(weighted),prefill_gpu_mean_ms=statistics.mean(weighted),prefill_cpu_enqueue_median_ms=statistics.median(cpu),mean_ttft_minus_mean_prefill_ms=d['mean_ttft_ms']-statistics.mean(weighted),prefill_request_count=len(weighted),completed=d['completed'],shapes={str(k):v for k,v in shapes.items()},monokernel_calls=sum(e['monokernel_calls_during_prefill'] for e in ev)))
(r/'TTFT_COMPONENTS.json').write_text(json.dumps(rows,indent=2))
traces={}
for name in ['native','previous','selected','replica']:
 for index,p in enumerate(sorted(r.glob(f'prefill-{name}-*-prefill-trace.json'))):
  data=json.loads(p.read_text());kernels=[e for e in data['traceEvents'] if e.get('cat')=='kernel'];kernels.sort(key=lambda e:e['ts'])
  if not kernels:continue
  sums=collections.defaultdict(lambda:dict(count=0,us=0.))
  for e in kernels:sums[e['name']]['count']+=1;sums[e['name']]['us']+=e['dur']
  span=max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)
  traces[name,index]=dict(file=p.name,kernels=len(kernels),kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,kernel_span_ms=span/1000,sequence=[e['name'] for e in kernels],by_name=dict(sums))
comparisons=[]
for (name,index),d in traces.items():
 ref=traces.get(('native',index));row={k:v for k,v in d.items() if k not in ['sequence','by_name']};row.update(configuration=name,profile_index=index)
 if ref:
  row['same_sequence_as_native']=d['sequence']==ref['sequence'];row['same_kernel_multiset_as_native']=collections.Counter(d['sequence'])==collections.Counter(ref['sequence'])
  delta=[]
  for k in set(ref['by_name'])|set(d['by_name']):
   a=ref['by_name'].get(k,dict(count=0,us=0.));b=d['by_name'].get(k,dict(count=0,us=0.));delta.append(dict(kernel=k,native_count=a['count'],current_count=b['count'],native_ms=a['us']/1000,current_ms=b['us']/1000,delta_ms=(b['us']-a['us'])/1000))
  row['largest_differences']=sorted(delta,key=lambda x:abs(x['delta_ms']),reverse=True)[:15]
 comparisons.append(row)
(r/'PREFILL_TRACE_COMPARISON.json').write_text(json.dumps(comparisons,indent=2))
print(json.dumps(rows,indent=2));print('trace summaries',[(x['configuration'],x['profile_index'],x['kernels'],x.get('same_sequence_as_native')) for x in comparisons])
