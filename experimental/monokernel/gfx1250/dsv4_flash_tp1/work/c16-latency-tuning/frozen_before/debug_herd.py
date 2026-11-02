from frozen_before.native_routing import *
from aiter.ops.triton.moe.moe_routing.topk import topk
v,ii,_=topk(logits,7,apply_softmax=False,score_mode='sqrtsoftplus',renorm=False,routed_scaling_factor=1.,HIST_BLOCK_M=32)
pop=torch.bincount(ii.long().flatten(),minlength=256)
refids=rd.topk_ids.long()
scores=torch.sqrt(torch.nn.functional.softplus(logits.float())).bfloat16()
records=[]
for row in range(M):
 ids=ii[row].tolist();vals=v[row].tolist();pp=[int(pop[k]) for k in ids]
 drop=min(range(7),key=lambda j:(pp[j],vals[j],j))
 ours=torch.where(ch[row]!=0)[0].tolist();expected=refids[row].tolist()
 records.append(dict(row=row,top7=ids,values=vals,popularity=pp,drop=ids[drop],python_keep=sorted(ids[:drop]+ids[drop+1:]),native=expected,ours=ours))
print(json.dumps(records,indent=2))
print('hists',h.tolist(),rd.expt_data.hist.tolist())
(ROOT/'herd-debug.json').write_text(json.dumps(records,indent=2))
