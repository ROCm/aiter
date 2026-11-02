import gzip,json,statistics,collections
from pathlib import Path
root=Path(__file__).resolve().parent
path=sorted((root/'traces/rank_0').glob('*.trace.json.gz'))[-1]
e=json.load(gzip.open(path))['traceEvents'];ks=sorted((x for x in e if x.get('cat')=='kernel'),key=lambda x:x['ts'])
rows=[]
for i,k in enumerate(ks):
 if k['name']!='mhc_res_unshuffle_h4096_hc4_ks64_b2':continue
 seq=ks[i:i+10]
 if len(seq)!=10 or not seq[-1]['name'].startswith('_pa_decode_sparse_reduce'):continue
 rows.append({'start_ts':k['ts'],'kernels':[{'name':x['name'],'us':x['dur']} for x in seq],'group_sum_us':{'mhc':sum(x['dur'] for x in seq[:3]),'qkv_cache':sum(x['dur'] for x in seq[3:8]),'attention':sum(x['dur'] for x in seq[8:])},'prefix_sum_us':sum(x['dur'] for x in seq),'prefix_wall_us':seq[-1]['ts']+seq[-1]['dur']-k['ts']})
result={'trace':str(path.relative_to(root)),'forward_markers':collections.Counter(x['name'] for x in e if x.get('name','').startswith('ATOM::forward')),'complete_layer0_samples':len(rows),'rows':rows,'mean_group_sum_us':{k:statistics.mean(r['group_sum_us'][k] for r in rows) for k in ['mhc','qkv_cache','attention']},'mean_prefix_kernel_sum_us':statistics.mean(r['prefix_sum_us'] for r in rows),'mean_prefix_wall_us':statistics.mean(r['prefix_wall_us'] for r in rows)}
(root/'live-decode-verification.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'})
