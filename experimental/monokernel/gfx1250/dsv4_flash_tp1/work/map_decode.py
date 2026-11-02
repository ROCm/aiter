#!/usr/bin/env python3
"""Map all pure-decode suffix kernels to ATOM capture-trace operators."""
import argparse, collections, csv, json, pathlib, re, statistics
import parse_trace as pt
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--out',required=True);a=p.parse_args()
events=pt.load_events(a.trace)
labels=sorted((e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_user_annotation' and re.match(r'^(prefill|decode)\[',e.get('name',''))),key=lambda e:e['ts'])
selected=[];seen=set()
for e in reversed(labels):
    if e['name'].startswith('prefill[') or re.search(r'\bp=[1-9]',e['name']):break
    key=e.get('args',{}).get('External id',e['ts'])
    if key not in seen: selected.append(e);seen.add(key)
selected.reverse();cache={};groups=collections.defaultdict(list);unmatched=total=0
for step,e in enumerate(selected):
    bs,gb=pt.decode_batch_sizes(e);q=pt.decode_query_len(e,bs);key=(gb,q)
    if key not in cache:
        cap=pt.load_events(pt.resolve_capture_trace(a.trace,gb,q));g=pt.find_capture_graph_for_bs(cap,gb)
        lo,hi=pt.warmup_window_for_graph(cap,g);cache[key]=pt.build_warmup_mapping(cap,lo,hi)
    lo,hi=pt.decode_gpu_window(events,e)
    rows=pt.match_replay_to_warmup(pt.replay_kernels_in_window(events,lo,hi),cache[key])
    for r in rows:
        total+=1; unmatched+=r['cpu_module']=='<unmatched>'
        groups[(gb,r['cpu_module'],r['kernel_name'])].append(r['duration_us'])
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
with (out/'operators.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['graph_batch','operator','kernel','count','mean_us','median_us','total_us'])
    for key,ds in sorted(groups.items()):w.writerow([*key,len(ds),statistics.mean(ds),statistics.median(ds),sum(ds)])
print(json.dumps(dict(trace=a.trace,steps=len(selected),kernels=total,unmatched=unmatched,buckets=list(cache)),indent=2))
