from harness import *
from region import QKVRegion
r=reference(X);rows=[]
for mode,pf,tdm in [('two','none',False)]+[('single',pf,t) for pf in ['none','speculative','non_speculative'] for t in [False,True]]:
 s=QKVRegion(W,WS,NW,BW,BS,mode,prefetch=pf,stage_first_tile=tdm);s(X,Cos,Sin);torch.cuda.synchronize()
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):s(X,Cos,Sin)
 for _ in range(100):cg.replay()
 q,k=s(X,Cos,Sin);torch.cuda.synchronize()
 row=dict(mode=mode,prefetch=pf,tdm=tdm,q=metric(q.flatten(),r['q']),kv=metric(k,r['kv']),health=s.health.item())
 assert row['q']['relative_rms']<.001 and row['kv']['relative_rms']<.001 and not row['health']
 rows.append(row)
(root/'api-validation.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
