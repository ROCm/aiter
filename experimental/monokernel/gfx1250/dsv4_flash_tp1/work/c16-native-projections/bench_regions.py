import torch,triton,statistics,json,os
from pathlib import Path
NAMES=['attention_mhc_pre_and_input_layout','compressor_and_index','input_quant_qkva_norm_rope_qb','attention_splitkv_and_reduce','inverse_rope_quant_oa_ob','attention_mhc_post_ffn_mhc_pre','routing_shared_routed_moe_sum_final_mhc_post_layout']
def benchmark_regions(region,batch,args,kw,s,m,saved):
 results=[]
 selected_tile=kw["TILE_MASK"];selected_overlap=kw["OVERLAP"]
 for group,name in enumerate(NAMES):
  if group==1 and kw['RATIO']==0:continue
  funcs={};graphs={};outs={};resources={}
  for use in [0,1]:
   k=dict(kw,M=m,BLOCK_STRIDE=32,OPT_PRE=True,EDGE_MASK=0,Edges=s.edges if use else None,Tiles=s.tiles if use else None,TILE_MASK=selected_tile if use else 0,OVERLAP=selected_overlap if use else 0)
   for x,backup in saved:x.copy_(backup)
   batch[(256,)](*args,**k);torch.cuda.synchronize()
   def fn(k=k):return region[(256,)](*args,**k,GROUP=group)
   obj=fn();torch.cuda.synchronize()
   assert not int(s.a[30]) and not int(s.f[-1]),(group,use,'health')
   check={0:[s.a[10],s.a[7]],1:[args[6],args[7]],2:[s.a[28],s.a[23]],3:[s.a[38]],4:[s.a[51]],5:[s.f[0],s.f[9]],6:[s.output]}[group]
   outs[use]=[t.clone() for t in check]
   assert all(torch.isfinite(t).all() for t in check)
   resources[use]=dict(vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills,warps=obj.metadata.num_warps)
   print("REGION_RESOURCES",name,m,use,resources[use],flush=True)
   for _ in range(3):fn()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
    for _ in range(32):fn()
   graphs[use]=graph
  exact=all(torch.equal(a,b) for a,b in zip(outs[0],outs[1]))
  assert exact,(name,m,kw['RATIO'],'A/B mismatch')
  rounds=[]
  for rnd in [1,2]:
   samples={0:[],1:[]}
   for _ in range(5):
    for graph in graphs.values():graph.replay()
   for it in range(40):
    for use in ([0,1] if it%2 else [1,0]):
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();graphs[use].replay();b.record();b.synchronize()
     samples[use].append(a.elapsed_time(b)*1000/32)
   rounds.append(dict(round=rnd,median_us={u:statistics.median(v) for u,v in samples.items()},samples_us=samples))
  assert not int(s.a[30]) and not int(s.f[-1]),(group,'replay health')
  row=dict(group=group,name=name,m=m,ratio=kw['RATIO'],rounds=rounds,resources=resources,exact=exact)
  results.append(row);print('REGION',name,m,kw['RATIO'],rounds[-1]['median_us'],flush=True)
  Path(f'/work/c16-native-projections/groups-{m}-{kw["RATIO"]}.json').write_text(json.dumps(results,indent=2))
 return results
