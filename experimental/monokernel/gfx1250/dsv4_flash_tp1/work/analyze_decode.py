#!/usr/bin/env python3
"""Average every kernel in the final pure-decode suffix of an ATOM trace."""
import argparse, bisect, collections, csv, gzip, json, pathlib, re, statistics
p=argparse.ArgumentParser(); p.add_argument('trace'); p.add_argument('--out',required=True); a=p.parse_args()
with (gzip.open if a.trace.endswith('.gz') else open)(a.trace,'rt') as f: events=json.load(f)['traceEvents']
def end(e): return e['ts']+e.get('dur',0)
labels=sorted((e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_user_annotation' and re.match(r'^(prefill|decode|eager_decode)\[',e.get('name',''))),key=lambda e:e['ts'])
merged={}
for e in labels:
    key=(e.get('args',{}).get('External id',e['ts']),e['name'])
    if key not in merged: merged[key]=dict(e)
    else:
        old=merged[key]; hi=max(end(old),end(e)); old['ts']=min(old['ts'],e['ts']);old['dur']=hi-old['ts']
labels=sorted(merged.values(),key=lambda e:e['ts'])
if not labels: raise RuntimeError('Missing GPU forward labels; cannot infer decode from CPU timestamps.')
# Walk backwards from trace end to the last prefill, as requested.
suffix=[]; boundary=None
for e in reversed(labels):
    if e['name'].startswith('prefill[') or re.search(r'\bp=[1-9]',e['name']): boundary=e; break
    suffix.append(e)
suffix.reverse()
if not suffix: raise RuntimeError('No pure decode suffix.')
starts=[e['ts'] for e in suffix]; groups=collections.defaultdict(list); perstep=collections.defaultdict(list)
for e in events:
    if e.get('cat')!='kernel' or e.get('ph')!='X': continue
    i=bisect.bisect_right(starts,e['ts'])-1
    if i>=0 and end(e)<=end(suffix[i])+0.01:
        groups[(e['name'],suffix[i]['name'])].append(e); perstep[i].append(e)
if not groups: raise RuntimeError('No kernels contained in decode GPU labels.')
out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
rows=[]
for (name,bucket),ev in groups.items():
    ds=[x['dur'] for x in ev]
    rows.append(dict(kernel=name,bucket=bucket,count=len(ds),mean_us=statistics.mean(ds),median_us=statistics.median(ds),sum_us=sum(ds),min_us=min(ds),max_us=max(ds),streams=','.join(map(str,sorted({x.get('args',{}).get('stream','?') for x in ev},key=str)))))
rows.sort(key=lambda r:r['sum_us'],reverse=True)
with (out/'kernels.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
steps=[]
for i,es in sorted(perstep.items()):
    es.sort(key=lambda e:e['ts']); union=0; lo=hi=None
    for e in es:
        if lo is None: lo,hi=e['ts'],end(e)
        elif e['ts']>hi: union+=hi-lo; lo,hi=e['ts'],end(e)
        else: hi=max(hi,end(e))
    union+=hi-lo
    summed=sum(e['dur'] for e in es); wall=max(map(end,es))-es[0]['ts']
    steps.append(dict(label=suffix[i]['name'],kernels=len(es),sum_us=summed,busy_union_us=union,wall_us=wall,overlap_us=summed-union,idle_us=wall-union))
summary=dict(trace=a.trace,last_prefill=boundary,decode_steps=len(steps),kernel_count=sum(r['count'] for r in rows),steps=steps)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='steps'},indent=2))
for r in rows[:25]:print(f"{r['count']:7d} {r['mean_us']:9.3f} us {r['sum_us']/1000:10.3f} ms {r['kernel'][:130]}")
